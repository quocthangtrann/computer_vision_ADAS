"""Dựng ảnh hiệu chuẩn và ảnh test INT8 ở cỡ tuỳ ý (ví dụ 640x320), từ data/bdd100k_seg.tar.

Ra (tương đối so với ROOT = $CAPSTONE_ROOT, mặc định /content/drive/MyDrive/capstone_ai):
  data/v2/hd_calib_{w}x{h}/calib_{k:03d}.png        200 ảnh RGB đã crop + resize, lấy từ ảnh TRAIN của bộ thí điểm (split_v2 == train)
  data/v2/hd_calib_{w}x{h}/calib_list.csv           k, id, bright, road_frac, tầng
  data/v2/hd_test_{w}x{h}/test_{k:03d}.png          100 ảnh RGB đã crop + resize, lấy từ 408 ảnh TEST (split_v2 == test)
  data/v2/hd_test_{w}x{h}/test_{k:03d}_gt.png       nhãn 4 lớp cùng cỡ (0..3, 255 bỏ qua)
  data/v2/hd_test_{w}x{h}/test_list.csv

Chọn ảnh: phân tầng 3 x 3 theo độ sáng (cột bright) x độ phủ đường (cột road_frac) của data/v2/pilot_drv_manifest.csv,
mỗi biến chia theo tam phân vị trong chính nhóm ảnh đang chọn; mỗi ô lấy đều nhau, ô thiếu thì bù ngẫu nhiên (hạt giống cố định).
Cách dựng ảnh và nhãn dùng lại ĐÚNG hàm của make_hd_pilot.py (crop (0, 0, 1280, 600), ảnh PIL BILINEAR, mask NEAREST).
Nếu có sẵn cache pilot_drv_{x,y}_{w}x{h}.npy thì so từng pixel với cache; lệch thì mã thoát 1.

Chưa so với handoff_s3a_v2/calib_list.csv cũ (không biết cột của nó): tập 200 ảnh ở đây là tập mới, không phải đúng tập ảnh 320x160.
Dùng:  python repo/make_hd_calib.py --w 640 --h 320
"""
import os, sys, time, shutil, argparse
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_hd_pilot as mhp

ROOT = os.environ.get("CAPSTONE_ROOT", "/content/drive/MyDrive/capstone_ai")


def stratified(df, n, seed):
    """Chọn n hàng, phân tầng 3 x 3 theo bright x road_frac (tam phân vị trong df). Trả về (chỉ số hàng đã chọn, nhãn tầng)."""
    assert len(df) >= n, f"chỉ có {len(df)} ảnh, cần {n}"
    rng = np.random.default_rng(seed)
    b = pd.qcut(df.bright.rank(method="first"), 3, labels=False).values
    r = pd.qcut(df.road_frac.rank(method="first"), 3, labels=False).values
    cell = b * 3 + r
    quota = np.full(9, n // 9); quota[:n % 9] += 1
    chosen = []
    for c in range(9):
        idx = np.where(cell == c)[0]
        chosen += list(rng.choice(idx, size=min(quota[c], len(idx)), replace=False))
    short = n - len(chosen)
    if short:
        rest = np.setdiff1d(np.arange(len(df)), chosen)
        chosen += list(rng.choice(rest, size=short, replace=False))
    chosen = np.array(sorted(chosen))
    return chosen, np.array([f"b{x // 3}r{x % 3}" for x in cell])[chosen]


def build(rows, jobs_src, w, h, seg, drv, imgs, labs, drvs, cache_x, cache_y, out_dir, prefix, with_gt):
    os.makedirs(out_dir, exist_ok=True)
    mhp._init(seg, drv)
    diff_x = diff_y = 0
    for k, (i, r) in enumerate(zip(jobs_src, rows.itertuples())):
        key = (r.v1_split, r.id)
        _, x, y = mhp.one((i, imgs[key], labs[key], drvs[r.id], w, h))
        Image.fromarray(x).save(f"{out_dir}/{prefix}_{k:03d}.png")
        if with_gt:
            Image.fromarray(y).save(f"{out_dir}/{prefix}_{k:03d}_gt.png")
        if cache_x is not None:
            diff_x += int((np.asarray(cache_x[i]) != x).sum()); diff_y += int((np.asarray(cache_y[i]) != y).sum())
    return diff_x, diff_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--seg_tar", default="data/bdd100k_seg.tar")
    ap.add_argument("--drv_tar", default="data/v2/drv_masks_3274.tar")
    ap.add_argument("--scratch", default=os.environ.get("HD_SCRATCH", "/content/hd_scratch"), help="thư mục đệm trên đĩa cục bộ (hoặc biến môi trường HD_SCRATCH)")
    ap.add_argument("--no_copy", action="store_true", help="đọc tar thẳng từ Drive thay vì chép về đĩa cục bộ")
    ap.add_argument("--force", action="store_true", help="dựng lại dù hai thư mục đã đủ ảnh")
    a = ap.parse_args()
    w, h = a.w, a.h
    V2 = os.path.join(a.root, "data/v2")
    cdir, tdir = f"{V2}/hd_calib_{w}x{h}", f"{V2}/hd_test_{w}x{h}"
    done = (os.path.exists(f"{cdir}/calib_list.csv") and os.path.exists(f"{tdir}/test_list.csv")
            and len([f for f in os.listdir(cdir) if f.endswith(".png")]) == a.n_calib
            and len([f for f in os.listdir(tdir) if f.endswith(".png") and not f.endswith("_gt.png")]) == a.n_test)
    if done and not a.force:
        print(f"đã có {a.n_calib} ảnh hiệu chuẩn ở {cdir} và {a.n_test} ảnh test ở {tdir}; dùng --force để dựng lại")
        return
    t0 = time.time()
    man = pd.read_csv(f"{V2}/pilot_drv_manifest.csv")
    for c in ("bright", "road_frac", "split_v2", "v1_split", "id"):
        assert c in man.columns, f"pilot_drv_manifest.csv thiếu cột {c}"
    tr = man[man.split_v2 == "train"]; te = man[man.split_v2 == "test"]
    ic, sc = stratified(tr, a.n_calib, a.seed)
    it, st = stratified(te, a.n_test, a.seed + 1)
    rows_c, rows_t = tr.iloc[ic], te.iloc[it]
    assert set(rows_c.id).isdisjoint(rows_t.id)

    os.makedirs(a.scratch, exist_ok=True)
    seg = mhp.local_copy(os.path.join(a.root, a.seg_tar), a.scratch, not a.no_copy)
    drv = mhp.local_copy(os.path.join(a.root, a.drv_tar), a.scratch, not a.no_copy)
    imgs, labs = mhp.index_tar(seg); drvs = mhp.index_drv(drv)
    miss = [r.id for r in pd.concat([rows_c, rows_t]).itertuples() if (r.v1_split, r.id) not in imgs or (r.v1_split, r.id) not in labs or r.id not in drvs]
    if miss:
        raise SystemExit(f"thiếu thành viên trong tar cho {len(miss)} ảnh, ví dụ {miss[:3]}")

    cx = cy = None
    px, py = f"{V2}/pilot_drv_x_{w}x{h}.npy", f"{V2}/pilot_drv_y_{w}x{h}.npy"
    if os.path.exists(px) and os.path.exists(py):
        cx, cy = np.load(px, mmap_mode="r"), np.load(py, mmap_mode="r")
        assert len(cx) == len(man) == len(cy), (len(cx), len(man), len(cy))
    else:
        print(f"CHƯA có cache {w}x{h} để đối chiếu: bỏ qua bước so từng pixel")

    # dựng vào thư mục tạm cạnh đích rồi đổi tên, để lần chạy hỏng không để lại thư mục nửa vời
    out = {}
    for name, rows, ids, d, prefix, gt in (("calib", rows_c, tr.index[ic], cdir, "calib", False), ("test", rows_t, te.index[it], tdir, "test", True)):
        tmp = d + ".tmp"; shutil.rmtree(tmp, ignore_errors=True)
        dx, dy = build(rows, list(ids), w, h, seg, drv, imgs, labs, drvs, cx, cy, tmp, prefix, gt)
        out[name] = (dx, dy, tmp, d)
        print(f"{name}: {len(rows)} ảnh dựng xong sau {time.time() - t0:.0f}s" + (f" | so với cache: pixel ảnh lệch {dx}, pixel nhãn lệch {dy}" if cx is not None else ""), flush=True)
    if cx is not None and any(v[0] or v[1] for v in out.values()):
        for v in out.values(): shutil.rmtree(v[2], ignore_errors=True)
        print("KẾT QUẢ: KHÔNG ĐẠT (ảnh dựng lại khác cache)")
        sys.exit(1)
    for name, rows, sl, d in (("calib", rows_c, sc, cdir), ("test", rows_t, st, tdir)):
        pd.DataFrame(dict(k=range(len(rows)), id=rows.id.values, bright=rows.bright.values, road_frac=rows.road_frac.values, tầng=sl)).to_csv(f"{out[name][2]}/{name}_list.csv", index=False)
        shutil.rmtree(d, ignore_errors=True); os.replace(out[name][2], d)
        print(f"đã ghi {d}")
    print("tầng (độ sáng b0..b2 x độ phủ đường r0..r2) của ảnh hiệu chuẩn:", dict(pd.Series(sc).value_counts().sort_index()))
    print("KẾT QUẢ:", "ĐẠT (khớp cache từng pixel)" if cx is not None else "xong (chưa đối chiếu cache)")
    print(f"XONG sau {(time.time() - t0) / 60:.1f} phút")


if __name__ == "__main__":
    main()
