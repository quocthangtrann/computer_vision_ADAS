"""Khung camera -> tiền xử lý -> mạng -> hậu xử lý -> gói kết quả perception_v0, ghi thời gian từng khâu.

Dùng (trên board hoặc Colab, từ thư mục gốc của mã):
  python -m runtime.pipeline --config config/pipeline_b.json --root THƯ_MỤC_CHỨA_GÓI --images 'anh/*.jpg' [--backend onnx]
         [--mode standard|fast|half|quarter] [--resize pil_bilinear|cv2_linear|cv2_area] [--overlap] [--out goi.bin] [--n 20]
Chạy gối đầu (--overlap, Pipeline.run_stream): luồng phụ tiền xử lý khung t+1 trong lúc luồng chính chạy mạng, hậu xử lý, tầng 2 của khung t.
  Thông lượng tăng, độ trễ mỗi khung tăng thêm khoảng một khung. Chỉ dùng threading (cv2 và NumPy nhả GIL ở các phép trên mảng lớn).
Đường dẫn trong cấu hình: "preprocess" tính từ thư mục của tệp cấu hình; "pack_dir" tính từ --root
(mặc định biến môi trường PERCEPTION_ROOT, rồi CAPSTONE_ROOT, rồi thư mục hiện tại).
"""
if __name__ == '__main__' and not __package__:      # chạy kiểu "python runtime/pipeline.py": tránh runtime/types.py che mô-đun types chuẩn
    import os as _os, sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path[:] = [p for p in _sys.path if _os.path.abspath(p or '.') != _here]
    _sys.path.insert(0, _os.path.dirname(_here))

import argparse
import glob
import json
import os
import queue
import threading
import time

import numpy as np

from runtime import types as T
from runtime.infer import Backend, make_backend
from runtime.postprocess import Postprocessor, logit_grid_to_frame
from runtime.preprocess import Preprocessor, load_json, resolve


def default_root():
    return os.environ.get('PERCEPTION_ROOT') or os.environ.get('CAPSTONE_ROOT') or os.getcwd()


class Pipeline:
    def __init__(self, config, root=None, backend=None, mode=None, resize=None):
        """config: đường dẫn tệp JSON hoặc dict. backend (tên hoặc một đối tượng Backend), mode, resize: ghi đè mục tương ứng trong cấu hình."""
        cdir = '.'
        if isinstance(config, str):
            cdir = os.path.dirname(os.path.abspath(config))
            config = load_json(config)
        self.cfg = config
        self.root = root or default_root()
        pcfg = config['preprocess']
        if isinstance(pcfg, str):
            pcfg = load_json(resolve(pcfg, cdir))
        self.pack_dir = resolve(config.get('pack_dir', ''), self.root)
        if resize:
            pcfg = dict(pcfg, resize=resize)
        self.pre = Preprocessor(pcfg, self.root)          # "int8_meta" trong cấu hình tiền xử lý tính từ root
        if isinstance(backend, Backend):
            self.backend = backend
        else:
            bcfg = config['backend']
            bcfg = {'name': bcfg} if isinstance(bcfg, str) else dict(bcfg)
            if backend:
                bcfg['name'] = backend
            self.backend = make_backend(bcfg, self.pack_dir, self.pre.input_scale)
        post = dict(config.get('postprocess', {}))
        if mode:
            post['mode'] = mode
        self.post = Postprocessor(post)
        self.packet_cfg = config.get('packet', {})
        self.model = config.get('model_name', '')
        self.seq = 0
        self.net_hw = (self.pre.out_wh[1], self.pre.out_wh[0])

    def process(self, frame, t_capture_ns=None, color=None, keep=False):
        """Một khung -> PerceptionResult. keep=True giữ tầng 0 và tầng 1 trong kết quả (để đóng gói hoặc kiểm)."""
        t_cap = int(t_capture_ns if t_capture_ns is not None else time.monotonic_ns())
        t0 = time.perf_counter_ns()
        p = self.pre(frame, color)
        return self.process_prepared(p, t_cap, keep, t0)

    def process_prepared(self, p, t_capture_ns=None, keep=False, t0=None, t_pre_us=None):
        """Phần sau tiền xử lý (dùng lại một Prepared cho nhiều backend khi đánh giá). t0: lúc bắt đầu tiền xử lý (perf_counter_ns).
        t_pre_us: thời gian tiền xử lý đã đo ở nơi khác (chạy gối đầu); khi đó thời gian đổi kiểu đầu vào cho backend float không tính vào "pre"."""
        t_cap = int(t_capture_ns if t_capture_ns is not None else time.monotonic_ns())
        tm = {}
        t0 = t0 if t0 is not None else time.perf_counter_ns()
        x = p.x_int8 if self.backend.input_kind == 'int8' else self.pre.to_float(p.chw)
        t1 = time.perf_counter_ns()
        lg = self.backend.run(x)
        t2 = time.perf_counter_ns()
        m, g2f = self.post.mask(lg, self.net_hw, p.grid_to_frame)
        t3 = time.perf_counter_ns()
        t2r = self.post.tier2(m, g2f)
        t4 = time.perf_counter_ns()
        tm.update(pre=(t1 - t0) // 1000 if t_pre_us is None else int(t_pre_us), infer=(t2 - t1) // 1000, post=(t3 - t2) // 1000, tier2=(t4 - t3) // 1000)
        self.seq += 1
        keep = keep or self.packet_cfg.get('include_logits') or self.packet_cfg.get('include_mask')
        return T.PerceptionResult(
            seq=self.seq, t_capture_ns=t_cap, frame_wh=p.frame_wh, grid_wh=(m.shape[1], m.shape[0]), tier2=t2r,
            tier3=T.Tier3.empty(t2r.n_cols, t2r.n_rows), model=self.model,
            logits=lg if keep else None, mask=T.Mask(m, g2f) if keep else None, timings_us=tm,
            drivable_class1=self.post.t2.drivable_include_class1, fast=self.post.mode != 'standard',
            logits_grid_to_frame=logit_grid_to_frame(p.grid_to_frame, self.net_hw, lg.shape[-2:]))

    def run_stream(self, frames, keep=False, overlap=True, depth=1):
        """Chạy một dòng khung, trả về (generator) từng PerceptionResult theo đúng thứ tự.
        frames: iterable của khung (h, w, 3) uint8, hoặc của cặp (khung, t_capture_ns). overlap=True: luồng phụ lấy khung và tiền xử lý
        khung t+1 trong lúc luồng chính chạy mạng, hậu xử lý, tầng 2 của khung t (hàng đợi giới hạn `depth` khung đã tiền xử lý).
        Kết quả y hệt chạy tuần tự process(); chỉ đổi thời gian. t_capture_ns mặc định là lúc luồng phụ nhận khung.
        Lỗi ở luồng phụ (kể cả trong iterable frames) được ném lại ở nơi gọi. Bỏ dở vòng lặp thì luồng phụ dừng."""
        def split(it):
            return (it[0], it[1]) if isinstance(it, tuple) else (it, None)
        if not overlap:
            for it in frames:
                f, tc = split(it)
                yield self.process(f, tc, keep=keep)
            return
        q = queue.Queue(maxsize=max(1, depth))
        stop = threading.Event()
        END = object()

        def put(x):
            while not stop.is_set():
                try:
                    q.put(x, timeout=0.05)
                    return True
                except queue.Full:
                    pass
            return False

        def work():
            try:
                for it in frames:
                    if stop.is_set():
                        return
                    f, tc = split(it)
                    tc = int(tc if tc is not None else time.monotonic_ns())
                    t0 = time.perf_counter_ns()
                    p = self.pre(f)
                    if not put((p, tc, (time.perf_counter_ns() - t0) // 1000)):
                        return
                put(END)
            except BaseException as e:      # noqa: BLE001: chuyển nguyên lỗi sang luồng chính
                put(e)

        th = threading.Thread(target=work, name='perception-pre', daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is END:
                    return
                if isinstance(item, BaseException):
                    raise item
                p, tc, t_pre = item
                yield self.process_prepared(p, tc, keep, t_pre_us=t_pre)
        finally:
            stop.set()
            th.join(timeout=5)

    def packet(self, r):
        return T.pack(r, include_logits=bool(self.packet_cfg.get('include_logits')), include_mask=bool(self.packet_cfg.get('include_mask')),
                      t_publish_ns=time.monotonic_ns())


def read_image(path):
    """Đọc ảnh thành RGB uint8 (OpenCV nếu có, không thì PIL)."""
    try:
        import cv2
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(path)
        return np.ascontiguousarray(im[:, :, ::-1])
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path).convert('RGB'))


def summarize(rows):
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], np.float64) / 1000.0
        out[k] = dict(mean_ms=float(v.mean()), p50_ms=float(np.percentile(v, 50)), p95_ms=float(np.percentile(v, 95)), max_ms=float(v.max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/pipeline_b.json')
    ap.add_argument('--root', default=None)
    ap.add_argument('--images', required=True, help="mẫu glob, ví dụ 'anh/*.jpg' (ảnh khung camera, cỡ theo config preprocess)")
    ap.add_argument('--backend', default=None, help='ghi đè backend: onnx | int8ref | vart | ip')
    ap.add_argument('--mode', default=None, help='ghi đè hậu xử lý: standard | fast | half (lưới 1/2) | quarter (lưới 1/4)')
    ap.add_argument('--resize', default=None, help='ghi đè cách resize của tiền xử lý: pil_bilinear | cv2_linear | cv2_area')
    ap.add_argument('--overlap', action='store_true', help='chạy gối đầu hai luồng (Pipeline.run_stream); in thông lượng')
    ap.add_argument('--n', type=int, default=0, help='chỉ N ảnh đầu')
    ap.add_argument('--out', default=None, help='ghi các gói nối tiếp nhau vào tệp này')
    a = ap.parse_args()
    files = sorted(glob.glob(a.images))[:a.n or None]
    if not files:
        raise SystemExit(f'không có ảnh nào khớp {a.images}')
    pl = Pipeline(a.config, a.root, a.backend, a.mode, a.resize)
    print(f'backend {pl.backend.name} | tiền xử lý {pl.pre.method} | hậu xử lý {pl.post.mode} | lưới vào {pl.net_hw[1]}x{pl.net_hw[0]} | '
          f'{"gối đầu" if a.overlap else "tuần tự"} | {len(files)} ảnh', flush=True)
    rows, fh = [], open(a.out, 'wb') if a.out else None
    results = pl.run_stream((read_image(f) for f in files), overlap=a.overlap)
    tw = time.perf_counter()
    for f, r in zip(files, results):
        t0 = time.perf_counter_ns()
        buf = pl.packet(r)
        tm = dict(r.timings_us, pack=(time.perf_counter_ns() - t0) // 1000)
        tm['total'] = sum(tm.values())
        rows.append(tm)
        if fh:
            fh.write(buf)
        t2 = r.tier2
        print(f'{os.path.basename(f)}: {len(buf)} byte | cột hợp lệ {int((t2.col_flags & T.COL_VALID > 0).sum())}/{t2.n_cols} | '
              f'tâm làn hợp lệ {int((t2.row_flags & T.ROW_CENTER > 0).sum())}/{t2.n_rows} | ' + ' '.join(f'{k} {v / 1000:.1f}' for k, v in tm.items()) + ' ms',
              flush=True)
    if fh:
        fh.close()
    print(f'tổng thời gian {1000 * (time.perf_counter() - tw):.0f} ms cho {len(files)} khung (gồm đọc ảnh, đóng gói, in): {1000 * (time.perf_counter() - tw) / len(files):.1f} ms mỗi khung')
    print('thời gian (ms):')
    for k, v in summarize(rows).items():
        print(f'  {k:6s} trung bình {v["mean_ms"]:8.1f} | p50 {v["p50_ms"]:8.1f} | p95 {v["p95_ms"]:8.1f} | lớn nhất {v["max_ms"]:8.1f}')


if __name__ == '__main__':
    main()
