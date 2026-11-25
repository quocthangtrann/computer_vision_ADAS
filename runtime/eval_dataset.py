"""Đánh giá runtime trên 408 ảnh test có nhãn (chạy trên Colab: cần ảnh gốc trong data/bdd100k_seg.tar và gói INT8 thật).

Với mỗi ảnh test (thứ tự dòng split_v2 == test của data/v2/pilot_drv_manifest.csv, như int8_sim.py):
  JPEG 1280x720 trong tar -> giải mã PIL (như lúc dựng cache) -> runtime.pipeline (tiền xử lý NumPy, mạng, hậu xử lý NumPy).
Kiểm:
  1. Ảnh sau tiền xử lý trùng từng byte với cache pilot_drv_x_{W}x{H}.npy (cùng quy tắc huấn luyện).
  2. IoU ở lưới 320x160 (mặt nạ thu nhỏ nearest, so pilot_drv_y_320x160.npy), cùng cách tính iou_320 của int8_sim.py:
     int8ref so với "pct" (hoặc phương pháp của gói), onnx so với "fp32" trong int8_sim_summary.json của gói. Lệch quá --tol thì KHÔNG ĐẠT.
  3. Mặt nạ NumPy trùng mặt nạ torch (F.interpolate bilinear + argmax, như int8_sim.pred_at) nếu có torch.
  4. Logit int8ref trùng từng bit test_int8_logits_quarter.npy của gói trên ảnh data/v2/hd_test_{W}x{H}/ (ảnh int8_pack.py đã dùng).
  5. Lệch tâm làn ở hàng gần xe so với nhãn (định nghĩa của runtime, xem cuối tệp); thời gian từng khâu.
  Mục in 5. So sánh các chế độ hậu xử lý (standard, fast, half = lưới 1/2, quarter = lưới 1/4): IoU, lệch tâm làn, thời gian (--modes).
     Tiêu chí đặt TRƯỚC khi chạy: chế độ làm tỉ lệ ảnh lệch tâm > 100 px (tính cả ảnh dự đoán thiếu tâm) xấu hơn standard quá --max_gt100_worse thì KHÔNG DÙNG.
  Mục in 6. So sánh các cách resize của tiền xử lý (--resizes; mặc định pil_bilinear và cv2_linear) trên backend --cmp_backend (mặc định onnx):
     pixel lệch so với cache, IoU, thời gian. Tiêu chí đặt TRƯỚC: IoU (4 lớp hoặc IoU đường) giảm quá --max_iou_drop so với resize mặc định thì GIỮ resize mặc định.
  Mục in 7 và 8. Chạy gối đầu hai luồng (Pipeline.run_stream) so với tuần tự trên --overlap_n khung đầu: thông lượng và kết quả có y hệt không; bảng pre + post + tier2 so với --target_ms.
Dùng (thư mục gốc của mã):  python -m runtime.eval_dataset --root /content/drive/MyDrive/capstone_ai [--n 408] [--backends int8ref,onnx]
"""
import argparse
import csv
import io
import json
import os
import sys
import time

import numpy as np

from runtime import types as T
from runtime.pipeline import Pipeline, summarize
from runtime.postprocess import MODES, Postprocessor, compute_tier2, mask_standard, nearest_resize, tier2_deviation
from runtime.preprocess import Preprocessor, load_json

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


def iou_from_conf(c):
    """Như int8_sim.iou_from_conf: IoU 4 lớp và IoU đường (lớp 1..3 gộp)."""
    c = np.asarray(c, float).reshape(4, 4)
    iou = [c[k, k] / (c[k].sum() + c[:, k].sum() - c[k, k]) for k in range(4)]
    road = c[1:, 1:].sum() / (c[1:, 1:].sum() + c[0, 1:].sum() + c[1:, 0].sum())
    return [float(v) for v in iou], float(road)


def tar_reader(root, seg_tar, scratch):
    """Trả về hàm đọc bytes JPEG theo (split, id), dùng chỉ mục tar của repo/make_hd_pilot.py (bản chép cục bộ nếu có)."""
    sys.path.insert(0, os.path.join(root, 'repo'))
    import make_hd_pilot as mhp
    src = os.path.join(root, seg_tar)
    loc = os.path.join(scratch, os.path.basename(src)) if scratch else None
    path = loc if loc and os.path.exists(loc) and os.path.getsize(loc) == os.path.getsize(src) else src
    print('đọc ảnh gốc từ', path, flush=True)
    imgs, _ = mhp.index_tar(path)
    fh = open(path, 'rb')

    def read(key):
        off, size = imgs[key]
        fh.seek(off)
        return fh.read(size)
    return read, fh


def decode(b, how):
    if how == 'cv2':
        import cv2
        return np.ascontiguousarray(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1])
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(b)).convert('RGB'))


def fmt(v, d=1):
    return 'n/a' if v is None else f'{v:.{d}f}'


def lane_gt(y_hd, g2f, cfg):
    """Tầng 2 tính trên nhãn (255 coi như không phải làn)."""
    y = np.where(y_hd == 255, 0, y_hd).astype(np.uint8)
    return compute_tier2(y, g2f, cfg)


def lane_add(L, t2, gt):
    """Cộng một ảnh vào thống kê lệch tâm làn ở hàng gần xe (hàng 0). gt: tầng 2 tính trên nhãn (hoặc None)."""
    if gt is None:
        return
    a_ok, g_ok = bool(t2.row_flags[0] & T.ROW_CENTER), bool(gt.row_flags[0] & T.ROW_CENTER)
    if a_ok and g_ok:
        L['both'] += 1
        L['err'].append(float(abs(t2.row_center[0] - gt.row_center[0])))
    elif g_ok:
        L['gt_only'] += 1
    elif a_ok:
        L['pred_only'] += 1


def lane_summary(L):
    """frac_gt100: tỉ lệ > 100 px chỉ tính ảnh cả hai có tâm (định nghĩa cũ, có thể đẹp giả nếu chế độ làm mất tâm).
    frac_gt100_strict: ảnh nhãn có tâm mà dự đoán không có tâm tính là lỗi > 100 px (dùng cho tiêu chí chế độ hậu xử lý)."""
    e = np.array(L['err'])
    n_gt = L['both'] + L['gt_only']
    if not n_gt:
        return None
    return dict(both=L['both'], gt_only=L['gt_only'], pred_only=L['pred_only'], mean_px=float(e.mean()) if len(e) else None,
                p90_px=float(np.percentile(e, 90)) if len(e) else None, frac_gt100=float((e > 100).mean()) if len(e) else None,
                frac_gt100_strict=float(((e > 100).sum() + L['gt_only']) / n_gt))


def new_lane():
    return dict(err=[], both=0, gt_only=0, pred_only=0)


def iou_drop(ref, cur):
    """Mức giảm lớn nhất (dương = cur kém hơn ref) trong 4 IoU lớp và IoU đường."""
    (ri, rr), (ci, cr) = ref, cur
    return max(max(r - c for r, c in zip(ri, ci)), rr - cr)


def mode_verdict(lc, base_lc, max_worse, min_images):
    """Tiêu chí đặt trước cho chế độ hậu xử lý: tỉ lệ ảnh lệch tâm > 100 px (tính cả ảnh thiếu tâm) xấu hơn standard quá max_worse thì KHÔNG DÙNG.
    Dưới min_images ảnh có tâm hợp lệ ở một trong hai bên thì không kết luận (mô hình không nhận ra làn thì con số vô nghĩa)."""
    if not (lc and base_lc and min(lc['both'], base_lc['both']) >= min_images):
        return f'chưa kết luận (thiếu nhãn HD hoặc dưới {min_images} ảnh có tâm làn)'
    worse = lc['frac_gt100_strict'] - base_lc['frac_gt100_strict']
    return f'KHÔNG DÙNG (xấu hơn {worse:+.3f})' if worse > max_worse else f'DÙNG ĐƯỢC (chênh {worse:+.3f})'


def resize_verdict(drop, max_drop, ref):
    """Tiêu chí đặt trước cho resize: IoU giảm quá max_drop so với mốc thì GIỮ mốc."""
    return f'GIỮ {ref} (IoU giảm {drop:.4f})' if drop > max_drop else f'ĐẠT tiêu chí IoU (giảm {drop:.4f}): có thể làm mặc định'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('CAPSTONE_ROOT', '/content/drive/MyDrive/capstone_ai'))
    ap.add_argument('--config', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'pipeline_b.json'))
    ap.add_argument('--backends', default='int8ref,onnx')
    ap.add_argument('--n', type=int, default=0, help='0 = cả 408 ảnh test; ít hơn thì không so được với int8_sim')
    ap.add_argument('--tol', type=float, default=0.001)
    ap.add_argument('--seg_tar', default='data/bdd100k_seg.tar')
    ap.add_argument('--scratch', default=os.environ.get('HD_SCRATCH', '/content/hd_scratch'))
    ap.add_argument('--decoder', default='pil', help='pil (như lúc dựng cache) hoặc cv2')
    ap.add_argument('--n_gold', type=int, default=20, help='số ảnh của gói dùng để so từng bit logit (tối đa 100)')
    ap.add_argument('--out', default='eval_runtime')
    ap.add_argument('--modes', default='standard,fast,half,quarter', help='các chế độ hậu xử lý để so sánh (standard luôn có, là mốc)')
    ap.add_argument('--resizes', default='pil_bilinear,cv2_linear', help='các cách resize của tiền xử lý để so sánh (mốc là cách trong cấu hình)')
    ap.add_argument('--cmp_backend', default='onnx', help='backend cho phần so sánh resize và gối đầu (int8ref rất chậm)')
    ap.add_argument('--max_iou_drop', type=float, default=0.001, help='resize làm IoU giảm quá mức này thì giữ resize mặc định')
    ap.add_argument('--max_gt100_worse', type=float, default=0.01, help='chế độ hậu xử lý làm tỉ lệ > 100 px xấu hơn standard quá mức này thì không dùng')
    ap.add_argument('--min_lane_images', type=int, default=30, help='số ảnh tối thiểu có tâm làn hợp lệ ở cả hai bên mới kết luận chế độ hậu xử lý')
    ap.add_argument('--target_ms', type=float, default=10.0, help='mục tiêu pre + post + tier2 (ms)')
    ap.add_argument('--overlap_n', type=int, default=60, help='số khung đo chạy gối đầu (0 = bỏ qua)')
    a = ap.parse_args()
    R = a.root
    bes = [b for b in a.backends.split(',') if b]
    pls = {b: Pipeline(a.config, R, backend=b) for b in bes}
    pl0 = pls[bes[0]]
    H, W = pl0.net_hw
    pack = pl0.pack_dir
    meta = load_json(os.path.join(pack, 'int8_meta.json'))
    summ = load_json(os.path.join(pack, 'int8_sim_summary.json'))
    method = meta.get('calibration', 'pct').split()[0]
    ref_key = {'int8ref': method, 'onnx': 'fp32'}
    t2cfg = pl0.post.t2
    print(f'gói {pack} | lưới vào {W}x{H} | backend {bes} | hậu xử lý {pl0.post.mode} | thang vào {pl0.pre.input_scale:.6g}', flush=True)

    modes = ['standard'] + [m for m in a.modes.split(',') if m and m != 'standard']
    for m in modes:
        assert m in MODES, f'chế độ hậu xử lý không biết: {m}'
    posts = {m: Postprocessor({**pl0.cfg.get('postprocess', {}), 'mode': m}) for m in modes}
    cb = a.cmp_backend if a.cmp_backend in bes else bes[0]
    ref_rz = pl0.pre.method
    alt = {v: Preprocessor(dict(pl0.pre.cfg, resize=v), R) for v in a.resizes.split(',') if v and v != ref_rz}

    with open(os.path.join(R, 'data/v2/pilot_drv_manifest.csv'), newline='', encoding='utf-8') as f:
        man = list(csv.DictReader(f))
    te = [i for i, r in enumerate(man) if r['split_v2'] == 'test']
    n = min(a.n or len(te), len(te))
    V2 = os.path.join(R, 'data/v2')
    X = np.load(f'{V2}/pilot_drv_x_{W}x{H}.npy', mmap_mode='r') if os.path.exists(f'{V2}/pilot_drv_x_{W}x{H}.npy') else None
    Y320 = np.load(f'{V2}/pilot_drv_y_320x160.npy', mmap_mode='r')
    YHD = np.load(f'{V2}/pilot_drv_y_{W}x{H}.npy', mmap_mode='r') if os.path.exists(f'{V2}/pilot_drv_y_{W}x{H}.npy') else None
    read, fh = tar_reader(R, a.seg_tar, a.scratch)
    print(f'{len(te)} ảnh test trong manifest, đánh giá {n}', flush=True)

    conf = {b: np.zeros(16) for b in bes}
    torch_agree = {b: [0, 0] for b in bes}
    lane = {b: new_lane() for b in bes}
    # chế độ hậu xử lý: mỗi (backend, chế độ) một bộ thống kê
    MS = {b: {m: dict(conf=np.zeros(16), agree=[], dc=[], dy=[], vmis=0, post=[], t2=[], lane=new_lane(), shape=None) for m in modes} for b in bes}
    # cách resize: mốc (cấu hình) lấy từ vòng chính của backend cb, các cách khác chạy thêm trên cb
    RZ = {v: dict(pre=[], pix_bad=0, pix_n=0, absdiff=0.0, conf=np.zeros(16), lane=new_lane()) for v in alt}
    pre_ms = []
    bench = []
    times = {b: [] for b in bes}
    pix_bad = pix_n = 0
    dec_diff = []
    masks_cmp = [0, 0]
    T0 = time.time()
    for j in range(n):
        i = te[j]
        r = man[i]
        b = read((r['v1_split'], r['id']))
        frame = decode(b, a.decoder)
        if j < 10 and a.decoder == 'pil':
            try:
                dec_diff.append(int((decode(b, 'cv2') != frame).any(-1).sum()))
            except Exception:
                pass
        t0 = time.perf_counter_ns()
        p = pl0.pre(frame)
        t_pre = time.perf_counter_ns() - t0
        pre_ms.append(t_pre / 1e6)
        if a.overlap_n and j < a.overlap_n:
            bench.append(frame)
        if X is not None:
            pix_bad += int((p.image != np.asarray(X[i])).any(-1).sum())
            pix_n += H * W
        y3 = np.asarray(Y320[i])
        v = y3 < 4
        gt = lane_gt(np.asarray(YHD[i]), p.grid_to_frame, t2cfg) if YHD is not None else None
        mk = {}
        for be in bes:
            t1 = time.perf_counter_ns()
            res = pls[be].process_prepared(p, keep=True, t0=t1)
            res.timings_us['pre'] = t_pre // 1000
            times[be].append(dict(res.timings_us))
            m = res.mask.m
            mk[be] = m
            P3 = nearest_resize(m, 160, 320)
            conf[be] += np.bincount(y3[v].astype(np.int64) * 4 + P3[v], minlength=16)
            if torch is not None and m.shape == (H, W):
                lg = res.logits.as_float()
                ref = F.interpolate(torch.from_numpy(lg[None]), size=(H, W), mode='bilinear', align_corners=False).argmax(1)[0].numpy()
                torch_agree[be][0] += int((ref == m).sum())
                torch_agree[be][1] += m.size
            lane_add(lane[be], res.tier2, gt)
            std_P3 = std_t2 = None
            for md in modes:         # cùng logit, từng chế độ hậu xử lý (standard đi đầu, là mốc)
                ps, S = posts[md], MS[be][md]
                tq = time.perf_counter_ns()
                m2, g2 = ps.mask(res.logits, (H, W), p.grid_to_frame)
                tr = time.perf_counter_ns()
                t2r = ps.tier2(m2, g2)
                te_ = time.perf_counter_ns()
                S['post'].append((tr - tq) / 1e6)
                S['t2'].append((te_ - tr) / 1e6)
                S['shape'] = m2.shape
                P3m = nearest_resize(m2, 160, 320) if m2.shape != (160, 320) else m2
                S['conf'] += np.bincount(y3[v].astype(np.int64) * 4 + P3m[v], minlength=16)
                if md == 'standard':
                    std_P3, std_t2 = P3m, t2r
                S['agree'].append(float((P3m == std_P3).mean()))
                d = tier2_deviation(t2r, std_t2)
                S['dc'] += d['row_center_absdiff_px']
                S['dy'] += d['col_y_absdiff_px']
                S['vmis'] += d['col_valid_mismatch'] + d['row_center_valid_mismatch']
                lane_add(S['lane'], t2r, gt)
        for rv, pv in alt.items():       # cách resize khác, chạy trên backend cb với hậu xử lý standard
            Z = RZ[rv]
            tq = time.perf_counter_ns()
            pr = pv(frame)
            Z['pre'].append((time.perf_counter_ns() - tq) / 1e6)
            if X is not None:
                dd = np.abs(pr.image.astype(np.int16) - np.asarray(X[i]).astype(np.int16))
                Z['pix_bad'] += int((dd > 0).any(-1).sum())
                Z['pix_n'] += H * W
                Z['absdiff'] += float(dd.mean())
            xin = pr.x_int8 if pls[cb].backend.input_kind == 'int8' else pv.to_float(pr.chw)
            mv = mask_standard(pls[cb].backend.run(xin), H, W)
            Pv = nearest_resize(mv, 160, 320)
            Z['conf'] += np.bincount(y3[v].astype(np.int64) * 4 + Pv[v], minlength=16)
            lane_add(Z['lane'], compute_tier2(mv, pr.grid_to_frame, t2cfg), gt)
        if len(bes) > 1:
            masks_cmp[0] += int((mk[bes[0]] == mk[bes[1]]).sum())
            masks_cmp[1] += mk[bes[0]].size
        if (j + 1) % 50 == 0 or j + 1 == n:
            print(f'  {j + 1}/{n} ảnh | {time.time() - T0:.0f}s', flush=True)
    fh.close()

    out = dict(pack=pack, n=n, n_test=len(te), backends={}, net_wh=[W, H], mode=pl0.post.mode)
    print('\n=== 1) ảnh sau tiền xử lý so với cache', f'pilot_drv_x_{W}x{H}.npy:',
          f'{pix_bad} / {pix_n} pixel lệch' if X is not None else 'KHÔNG CÓ cache, bỏ qua')
    out['preprocess_pixel_mismatch'] = pix_bad if X is not None else None
    if dec_diff:
        print(f'    (tham khảo) giải mã JPEG bằng cv2 khác PIL ở {dec_diff} pixel trên 10 ảnh đầu: board nhận khung thô từ camera nên không ảnh hưởng')
    print('=== 2) IoU ở lưới 320x160 so với int8_sim_summary.json (thứ tự lớp: ' + ', '.join(T.CLASSES) + ')')
    full = n == len(te)
    all_ok = pix_bad == 0 or X is None
    for be in bes:
        iou, road = iou_from_conf(conf[be])
        k = ref_key.get(be)
        ref = summ.get(k) if k else None
        o = dict(iou=iou, road=road, ref_key=k)
        line = f'  {be:8s} {[round(x, 4) for x in iou]} | IoU đường {road:.4f}'
        if ref is not None:
            dev = max(max(abs(x - y) for x, y in zip(iou, ref['iou'])), abs(road - ref['road']))
            o.update(ref_iou=ref['iou'], ref_road=ref['road'], max_dev=dev)
            ok = full and dev <= a.tol
            all_ok &= ok
            o['pass'] = ok
            line += f' | int8_sim {k}: {[round(x, 4) for x in ref["iou"]]} đường {ref["road"]:.4f} | lệch lớn nhất {dev:.5f} -> ' + \
                    ('ĐẠT' if ok else ('KHÔNG ĐẠT' if full else f'chưa so được (mới {n}/{len(te)} ảnh)'))
        print(line)
        if torch is not None:
            o['torch_mask_agree'] = torch_agree[be][0] / max(torch_agree[be][1], 1)
            print(f'           mặt nạ NumPy trùng mặt nạ torch (int8_sim.pred_at): {o["torch_mask_agree"]:.6f}')
        lc = lane_summary(lane[be])
        if lc:
            o['lane_center_near'] = lc
            print(f'           lệch tâm làn ở hàng gần xe (pixel khung, so với nhãn): {lc["both"]} ảnh cả hai có tâm | trung bình {fmt(lc["mean_px"])} | '
                  f'p90 {fmt(lc["p90_px"])} | tỉ lệ > 100 px: {fmt(lc["frac_gt100"], 3)} (tính cả thiếu tâm: {fmt(lc["frac_gt100_strict"], 3)}) | '
                  f'nhãn có mà dự đoán không: {lc["gt_only"]} | ngược lại: {lc["pred_only"]}')
        o['timing'] = summarize(times[be])
        out['backends'][be] = o
    if len(bes) > 1:
        out['mask_agree_between_backends'] = masks_cmp[0] / max(masks_cmp[1], 1)
        print(f'  trùng pixel {bes[0]} so với {bes[1]}: {out["mask_agree_between_backends"]:.4f}')

    print('=== 3) logit so với tệp tham chiếu của gói (ảnh data/v2/hd_test_*)')
    tdir = os.path.join(V2, f'hd_test_{W}x{H}') if (W, H) != (320, 160) else os.path.join(R, 'handoff_s3a_v2/test')
    gq, gf = os.path.join(pack, 'test_int8_logits_quarter.npy'), os.path.join(pack, 'test_fp32_logits_quarter.npy')
    if os.path.isdir(tdir) and os.path.exists(gq):
        from PIL import Image
        lq, lf = np.load(gq), (np.load(gf) if os.path.exists(gf) else None)
        ng = min(a.n_gold, len(lq))
        for be in bes:
            bad, mx = 0, 0.0
            for k in range(ng):
                img = np.asarray(Image.open(f'{tdir}/test_{k:03d}.png').convert('RGB'))
                xq = pls[be].pre.quantize(img)
                x = xq if pls[be].backend.input_kind == 'int8' else pls[be].pre.to_float(img)
                lg = pls[be].backend.run(x)
                if lg.q is not None:
                    bad += int((lg.q != lq[k]).sum())
                elif lf is not None:
                    mx = max(mx, float(np.abs(lg.f - lf[k]).max()))
            if be == 'int8ref':
                print(f'  {be}: {ng} ảnh, {bad} giá trị logit INT8 lệch -> ' + ('ĐẠT (trùng từng bit)' if bad == 0 else 'KHÔNG ĐẠT'))
                all_ok &= bad == 0
                out['backends'][be]['gold_mismatch'] = bad
            else:
                print(f'  {be}: {ng} ảnh, lệch logit FP32 lớn nhất {mx:.2e}')
                out['backends'][be]['gold_fp32_maxabs'] = mx
    else:
        print(f'  bỏ qua: không có {tdir} hoặc {gq}')

    print('=== 4) thời gian từng khâu (ms, CPU của máy đang chạy; KHÔNG phải số đo trên board)')
    for be in bes:
        print(f'  {be}:')
        for k, v in out['backends'][be]['timing'].items():
            print(f'    {k:6s} trung bình {v["mean_ms"]:8.1f} | p50 {v["p50_ms"]:8.1f} | p95 {v["p95_ms"]:8.1f}')
    mean = lambda v: float(np.mean(v)) if len(v) else None
    p95 = lambda v: float(np.percentile(v, 95)) if len(v) else None

    print(f'\n=== 5) chế độ hậu xử lý, cùng logit (IoU ở lưới 320x160; lệch tâm làn ở hàng gần xe so với nhãn, pixel khung; ms trung bình trên CPU đang chạy)')
    print(f'    Tiêu chí đặt trước: tỉ lệ > 100 px (tính cả ảnh thiếu tâm) xấu hơn standard quá {a.max_gt100_worse} -> KHÔNG DÙNG')
    out['modes'] = {}
    for be in bes:
        base = MS[be]['standard']
        base_iou, base_lc = iou_from_conf(base['conf']), lane_summary(base['lane'])
        print(f'  {be}:')
        print(f'    {"chế độ":8s} {"lưới":8s} {"IoU 4 lớp":28s} {"đường":6s} {"ΔIoU*":7s} {"trùng px":8s} {"p95|Δtâm|":9s} {"tb lệch":7s} {"p90":6s} {">100px":6s} {"chặt":6s} {"post+t2 ms":10s} kết luận')
        out['modes'][be] = {}
        for md in modes:
            S = MS[be][md]
            iou, road = iou_from_conf(S['conf'])
            lc = lane_summary(S['lane'])
            drop = iou_drop(base_iou, (iou, road))
            ms_ = (mean(S['post']) or 0) + (mean(S['t2']) or 0)
            verdict = 'mốc' if md == 'standard' else mode_verdict(lc, base_lc, a.max_gt100_worse, a.min_lane_images)
            out['modes'][be][md] = dict(grid_wh=[S['shape'][1], S['shape'][0]], iou=iou, road=road, iou_drop_vs_standard=drop, pixel_agree_vs_standard=mean(S['agree']),
                                        center_dev_p95_px=p95(S['dc']), col_y_dev_p95_px=p95(S['dy']), valid_mismatch=S['vmis'], lane_center_near=lc,
                                        post_ms=mean(S['post']), tier2_ms=mean(S['t2']), post_tier2_ms=ms_, verdict=verdict)
            print(f'    {md:8s} {S["shape"][1]}x{S["shape"][0]:<4d} {str([round(x, 4) for x in iou]):28s} {road:.4f} {drop:+.4f} {mean(S["agree"]):.4f}   '
                  f'{fmt(p95(S["dc"])):>9s} {fmt(lc and lc["mean_px"]):>7s} {fmt(lc and lc["p90_px"]):>6s} {fmt(lc and lc["frac_gt100"], 3):>6s} {fmt(lc and lc["frac_gt100_strict"], 3):>6s} '
                  f'{ms_:10.2f} {verdict}')
    print('    * ΔIoU = mức giảm lớn nhất (4 lớp và IoU đường) so với standard, dương = kém hơn. "trùng px" so với standard ở lưới 320x160. '
          '">100px" chỉ tính ảnh cả hai có tâm, "chặt" tính cả ảnh thiếu tâm là lỗi.')

    print(f'\n=== 6) cách resize của tiền xử lý (backend {cb}, hậu xử lý standard; mốc = {ref_rz} trong cấu hình)')
    print(f'    Tiêu chí đặt trước: IoU (4 lớp hoặc IoU đường) giảm quá {a.max_iou_drop} so với mốc -> GIỮ mốc làm mặc định')
    ref_iou = iou_from_conf(MS[cb]['standard']['conf'])
    ref_lc = lane_summary(MS[cb]['standard']['lane'])
    out['resize'] = {}
    print(f'    {"resize":12s} {"pixel lệch cache":22s} {"|Δ| tb":7s} {"pre ms (tb / p50 / p95)":26s} {"IoU 4 lớp":28s} {"đường":6s} {"ΔIoU*":7s} {"chặt":6s} kết luận')
    for rv in [ref_rz] + list(alt):
        Z = RZ.get(rv)
        if Z is None:
            pre, iou, road, lc = pre_ms, ref_iou[0], ref_iou[1], ref_lc
            pb, pn, ad = pix_bad, pix_n, 0.0
        else:
            pre, (iou, road), lc = Z['pre'], iou_from_conf(Z['conf']), lane_summary(Z['lane'])
            pb, pn, ad = Z['pix_bad'], Z['pix_n'], (Z['absdiff'] / n if X is not None else None)
        drop = 0.0 if Z is None else iou_drop(ref_iou, (iou, road))
        verdict = 'mốc' if Z is None else resize_verdict(drop, a.max_iou_drop, ref_rz)
        out['resize'][rv] = dict(pixel_mismatch=pb if X is not None else None, pixel_total=pn, mean_absdiff=ad, pre_ms=dict(mean=mean(pre), p50=float(np.percentile(pre, 50)), p95=p95(pre)),
                                 iou=iou, road=road, iou_drop_vs_ref=drop, lane_center_near=lc, verdict=verdict)
        pxs = f'{pb} / {pn} ({100 * pb / max(pn, 1):.1f}%)' if X is not None else 'không có cache'
        print(f'    {rv:12s} {pxs:22s} {fmt(ad, 2):>7s} {mean(pre):7.2f} /{np.percentile(pre, 50):6.2f} /{p95(pre):6.2f}    {str([round(x, 4) for x in iou]):28s} {road:.4f} {drop:+.4f} '
              f'{fmt(lc and lc["frac_gt100_strict"], 3):>6s} {verdict}')
    print(f'    * Đo trên đường {cb}: ' + ('ảnh chuẩn hóa float vào mạng FP32, KHÔNG qua lượng tử INT8' if pls[cb].backend.input_kind == 'float' else 'ảnh lượng tử INT8') +
          '. OpenCV khác phiên bản hoặc nền tảng (board) có thể cho ảnh hơi khác; bản NumPy của PIL thì xác định, trùng từng byte.')

    ovl = None
    if bench and a.overlap_n:
        pb_ = pls[cb]
        pb_.process(bench[0])                           # làm nóng (onnxruntime chạy lần đầu chậm)
        t0 = time.perf_counter()
        seq_r = [pb_.process(f) for f in bench]
        t_seq = (time.perf_counter() - t0) / len(bench)
        t0 = time.perf_counter()
        ovl_r = list(pb_.run_stream(bench))
        t_ovl = (time.perf_counter() - t0) / len(bench)
        same = all(np.array_equal(x.tier2.col_y, y.tier2.col_y, equal_nan=True) and np.array_equal(x.tier2.row_center, y.tier2.row_center, equal_nan=True) and
                   np.array_equal(x.tier2.row_flags, y.tier2.row_flags) and np.array_equal(x.tier2.col_flags, y.tier2.col_flags) for x, y in zip(seq_r, ovl_r))
        st = summarize([r.timings_us for r in seq_r])
        ideal = max(st['pre']['mean_ms'], st['infer']['mean_ms'] + st['post']['mean_ms'] + st['tier2']['mean_ms'])
        ovl = dict(backend=cb, n=len(bench), seq_ms_per_frame=1000 * t_seq, overlap_ms_per_frame=1000 * t_ovl, speedup=t_seq / t_ovl, identical=bool(same),
                   stage_mean_ms={k: v['mean_ms'] for k, v in st.items()}, ideal_ms_per_frame=ideal)
        out['overlap'] = ovl
        print(f'\n=== 7) chạy gối đầu hai luồng (backend {cb}, {len(bench)} khung, tiền xử lý {pb_.pre.method}, hậu xử lý {pb_.post.mode})')
        print(f'    tuần tự {1000 * t_seq:.1f} ms/khung | gối đầu {1000 * t_ovl:.1f} ms/khung | nhanh hơn {100 * (t_seq / t_ovl - 1):+.1f}% | kết quả y hệt tuần tự: ' + ('CÓ' if same else 'KHÔNG'))
        print('    các khâu (tuần tự, ms): ' + ' | '.join(f'{k} {v["mean_ms"]:.1f}' for k, v in st.items()) +
              f' | trần lý thuyết max(pre, infer+post+tier2) = {ideal:.1f} ms/khung')
        print('    Trên board mạng chạy ở PL nên CPU rảnh trong lúc infer: lợi ích gối đầu phụ thuộc thời gian infer của IP (chưa đo). Độ trễ mỗi khung tăng thêm khoảng một khung.')

    print(f'\n=== 8) pre + post + tier2 (ms trung bình, backend {cb}; mục tiêu <= {a.target_ms:g} ms; CPU đang chạy, A53 chậm hơn)')
    pres = {rv: mean(RZ[rv]['pre'] if rv in RZ else pre_ms) for rv in [ref_rz] + list(alt)}
    print(f'    {"chế độ":8s} ' + ' '.join(f'pre={rv} ({pres[rv]:.1f})'.rjust(30) for rv in pres))
    out['target'] = dict(target_ms=a.target_ms, rows={})
    for md in modes:
        S = MS[cb][md]
        pt = (mean(S['post']) or 0) + (mean(S['t2']) or 0)
        out['target']['rows'][md] = {rv: pres[rv] + pt for rv in pres}
        print(f'    {md:8s} ' + ' '.join((f'{pres[rv] + pt:.1f} ms ' + ('ĐẠT' if pres[rv] + pt <= a.target_ms else 'chưa đạt')).rjust(30) for rv in pres))

    out['pass'] = bool(all_ok and full)
    os.makedirs(os.path.join(R, a.out), exist_ok=True)
    with open(os.path.join(R, a.out, 'runtime_eval_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('\nKẾT QUẢ:', 'ĐẠT' if out['pass'] else ('KHÔNG ĐẠT' if full else f'chưa kết luận (mới {n}/{len(te)} ảnh)'),
          '| đã ghi', os.path.join(a.out, 'runtime_eval_summary.json'))
    print('Ghi chú: "lệch tâm làn ở hàng gần xe" = |tâm làn đang đi dự đoán - tâm tính trên nhãn| ở hàng mẫu gần xe nhất (row_bottom_frac),\n'
          '  đo bằng pixel khung 1280 rộng, chỉ ở ảnh mà cả hai có tâm hợp lệ. Định nghĩa của runtime, có thể khác cách tính con số 16% trước đây.')
    sys.exit(0 if out['pass'] or not full else 1)


if __name__ == '__main__':
    main()
