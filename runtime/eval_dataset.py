"""Đánh giá runtime trên 408 ảnh test có nhãn (chạy trên Colab: cần ảnh gốc trong data/bdd100k_seg.tar và gói INT8 thật).

Với mỗi ảnh test (thứ tự dòng split_v2 == test của data/v2/pilot_drv_manifest.csv, như int8_sim.py):
  JPEG 1280x720 trong tar -> giải mã PIL (như lúc dựng cache) -> runtime.pipeline (tiền xử lý NumPy, mạng, hậu xử lý NumPy).
Kiểm:
  1. Ảnh sau tiền xử lý trùng từng byte với cache pilot_drv_x_{W}x{H}.npy (cùng quy tắc huấn luyện).
  2. IoU ở lưới 320x160 (mặt nạ thu nhỏ nearest, so pilot_drv_y_320x160.npy), cùng cách tính iou_320 của int8_sim.py:
     int8ref so với "pct" (hoặc phương pháp của gói), onnx so với "fp32" trong int8_sim_summary.json của gói. Lệch quá --tol thì KHÔNG ĐẠT.
  3. Mặt nạ NumPy trùng mặt nạ torch (F.interpolate bilinear + argmax, như int8_sim.pred_at) nếu có torch.
  4. Logit int8ref trùng từng bit test_int8_logits_quarter.npy của gói trên ảnh data/v2/hd_test_{W}x{H}/ (ảnh int8_pack.py đã dùng).
  5. Bản nhanh so với bản chuẩn; lệch tâm làn ở hàng gần xe so với nhãn (định nghĩa của runtime, xem cuối tệp); thời gian từng khâu.
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
from runtime.postprocess import compute_tier2, fast_deviation, nearest_resize
from runtime.preprocess import load_json

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
    fast = {b: dict(agree=[], dy=[], dc=[], vmis=0) for b in bes}
    lane = {b: dict(err=[], both=0, gt_only=0, pred_only=0) for b in bes}
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
            if torch is not None:
                lg = res.logits.as_float()
                ref = F.interpolate(torch.from_numpy(lg[None]), size=(H, W), mode='bilinear', align_corners=False).argmax(1)[0].numpy()
                torch_agree[be][0] += int((ref == m).sum())
                torch_agree[be][1] += m.size
            d = fast_deviation(res.logits, (H, W), p.grid_to_frame, t2cfg)
            fast[be]['agree'].append(d['pixel_agree'])
            fast[be]['dy'] += d['col_y_absdiff_px']
            fast[be]['dc'] += d['row_center_absdiff_px']
            fast[be]['vmis'] += d['col_valid_mismatch'] + d['row_center_valid_mismatch']
            if gt is not None:
                a_ok, g_ok = bool(res.tier2.row_flags[0] & T.ROW_CENTER), bool(gt.row_flags[0] & T.ROW_CENTER)
                L = lane[be]
                if a_ok and g_ok:
                    L['both'] += 1
                    L['err'].append(float(abs(res.tier2.row_center[0] - gt.row_center[0])))
                elif g_ok:
                    L['gt_only'] += 1
                elif a_ok:
                    L['pred_only'] += 1
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
        fa = fast[be]
        o['fast'] = dict(pixel_agree=float(np.mean(fa['agree'])), col_y_p95_px=float(np.percentile(fa['dy'], 95)) if fa['dy'] else None,
                         center_p95_px=float(np.percentile(fa['dc'], 95)) if fa['dc'] else None, valid_mismatch=fa['vmis'])
        print(f'           bản nhanh (argmax ở lưới logit) so với bản chuẩn: trùng pixel {o["fast"]["pixel_agree"]:.4f} | '
              f'|Δ col_y| p95 {fmt(o["fast"]["col_y_p95_px"])} px | |Δ tâm| p95 {fmt(o["fast"]["center_p95_px"])} px | cờ hợp lệ khác {fa["vmis"]} lần')
        L = lane[be]
        if L['both'] or L['gt_only']:
            e = np.array(L['err'])
            o['lane_center_near'] = dict(both=L['both'], gt_only=L['gt_only'], pred_only=L['pred_only'], mean_px=float(e.mean()) if len(e) else None,
                                         p90_px=float(np.percentile(e, 90)) if len(e) else None, frac_gt100=float((e > 100).mean()) if len(e) else None)
            lc = o['lane_center_near']
            print(f'           lệch tâm làn ở hàng gần xe (pixel khung, so với nhãn): {L["both"]} ảnh cả hai có tâm | trung bình {fmt(lc["mean_px"])} | '
                  f'p90 {fmt(lc["p90_px"])} | tỉ lệ > 100 px: {fmt(lc["frac_gt100"], 3)} | nhãn có mà dự đoán không: {L["gt_only"]} | ngược lại: {L["pred_only"]}')
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
