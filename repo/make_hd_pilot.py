"""Dựng cache bộ thí điểm 4 lớp ở cỡ tuỳ ý (ví dụ 640x320), đúng thứ tự dòng của data/v2/pilot_drv_manifest.csv.

Nguồn (tương đối so với ROOT = $CAPSTONE_ROOT, mặc định /content/drive/MyDrive/capstone_ai):
  data/bdd100k_seg.tar        thành viên .../images_10k/10k/{train|val}/shard_*/{id}.jpg
                              và .../seg_maps/labels/**/{id}_train_id.png (0 = đường, 255 = bỏ qua)
  data/v2/drv_masks_3274.tar  thành viên {id}.png (0 nền, 1 làn đang đi, 2 làn kề)
Ra: data/v2/pilot_drv_x_{w}x{h}.npy (N,h,w,3 uint8) và pilot_drv_y_{w}x{h}.npy (N,h,w uint8).

Quy tắc nhãn (giữ y hệt cache 320x160): cắt (0, 0, 1280, 600); ảnh resize PIL BILINEAR, mọi mask resize NEAREST;
  road = (s == 0), ign = (s == 255); lab = road; lab[d == 1] = 2; lab[d == 2] = 3;
  lab[((d == 1) | (d == 2)) & ~road & ~ign] = 255; lab[ign] = 255.

Chế độ kiểm chứng: --w 320 --h 160 dựng lại vào thư mục tạm, so từng pixel với pilot_drv_*_320x160.npy đã có
(KHÔNG ghi đè tệp đó) và chỉ in "ĐẠT" khi số pixel lệch bằng 0; mã thoát 1 nếu không đạt.
Dùng:  python repo/make_hd_pilot.py --w 320 --h 160      # kiểm chứng
       python repo/make_hd_pilot.py --w 640 --h 320      # dựng thật
"""
import os, io, sys, time, shutil, tarfile, argparse
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool

CROP = (0, 0, 1280, 600)                  # (trái, trên, phải, dưới) trên ảnh 1280x720
ROOT = os.environ.get("CAPSTONE_ROOT", "/content/drive/MyDrive/capstone_ai")
_F = {}                                    # tay cầm tệp tar mở riêng cho từng tiến trình con


def index_tar(path):
    """Duyệt tiêu đề tar một lần: trả về (ảnh, nhãn đường) -> {(split, id): (offset_dữ_liệu, kích_thước)}."""
    imgs, labs = {}, {}
    with tarfile.open(path, "r:") as tf:      # 'r:' = không nén; tar nén thì báo lỗi ngay thay vì chạy rất chậm
        for m in tf:
            if not m.isfile():
                continue
            d, b = "/" + os.path.dirname(m.name) + "/", os.path.basename(m.name)
            sp = "train" if "/train" in d else ("val" if "/val" in d else None)
            if "/images_10k/" in d and b.lower().endswith(".jpg") and sp:
                imgs[(sp, b[:-4])] = (m.offset_data, m.size)
            elif "/seg_maps/labels/" in d and b.endswith("_train_id.png") and sp:
                labs[(sp, b[:-len("_train_id.png")])] = (m.offset_data, m.size)
    return imgs, labs


def index_drv(path):
    """Mask drivable: {id: (offset_dữ_liệu, kích_thước)} từ các thành viên {id}.png."""
    drvs = {}
    with tarfile.open(path, "r:") as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith(".png"):
                drvs[os.path.basename(m.name)[:-4]] = (m.offset_data, m.size)
    return drvs


def _init(seg_path, drv_path):
    _F["seg"], _F["drv"] = open(seg_path, "rb"), open(drv_path, "rb")


def _read(fh, off, size):
    fh.seek(off)
    return fh.read(size)


def make_label(s, d):
    """s: mask đường (0 = đường, 255 = bỏ qua) đã crop+resize; d: mask drivable (0 nền, 1 đang đi, 2 kề). Trả về nhãn 4 lớp."""
    road = (s == 0)
    ign = (s == 255)
    lab = road.astype(np.uint8)
    lab[d == 1] = 2
    lab[d == 2] = 3
    bad = ((d == 1) | (d == 2)) & ~road & ~ign
    lab[bad] = 255
    lab[ign] = 255
    return lab


def one(job):
    i, ji, jl, jd, w, h = job
    im = Image.open(io.BytesIO(_read(_F["seg"], *ji))).convert("RGB")
    sl = Image.open(io.BytesIO(_read(_F["seg"], *jl)))
    dm = Image.open(io.BytesIO(_read(_F["drv"], *jd)))
    for nm, o in (("ảnh", im), ("nhãn đường", sl), ("mask drivable", dm)):
        if o.size != (1280, 720):
            raise RuntimeError(f"dòng {i}: {nm} có kích thước {o.size}, cần (1280, 720)")
    s = np.asarray(sl.crop(CROP).resize((w, h), Image.NEAREST))
    d = np.asarray(dm.crop(CROP).resize((w, h), Image.NEAREST))
    if s.ndim != 2 or d.ndim != 2:
        raise RuntimeError(f"dòng {i}: mask không phải 1 kênh (đường {s.shape}, drivable {d.shape})")
    return i, np.asarray(im.crop(CROP).resize((w, h), Image.BILINEAR)), make_label(s, d)


def local_copy(src, scratch, copy):
    if not copy:
        return src
    dst = os.path.join(scratch, os.path.basename(src))
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        print(f"chép {src} -> {dst} ({os.path.getsize(src) / 1e9:.2f} GB)", flush=True)
        shutil.copyfile(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    return dst


def differ(a, b, axis_tail, chunk=256):
    """Số pixel (h, w) có ít nhất một giá trị khác nhau, tính theo khối để không tốn RAM."""
    n = 0
    for k in range(0, len(a), chunk):
        x, y = np.asarray(a[k:k + chunk]), np.asarray(b[k:k + chunk])
        diff = x != y
        n += int(diff.any(axis=-1).sum()) if axis_tail else int(diff.sum())
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--seg_tar", default="data/bdd100k_seg.tar")
    ap.add_argument("--drv_tar", default="data/v2/drv_masks_3274.tar")
    ap.add_argument("--scratch", default="/content/hd_scratch", help="thư mục đệm trên đĩa cục bộ")
    ap.add_argument("--no_copy", action="store_true", help="đọc tar thẳng từ Drive (chậm hơn) thay vì chép về đĩa cục bộ")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="chỉ N dòng đầu (thử nhanh); không ghi lên Drive")
    a = ap.parse_args()
    w, h = a.w, a.h
    verify = (w, h) == (320, 160)
    V2 = os.path.join(a.root, "data/v2")
    man = pd.read_csv(os.path.join(V2, "pilot_drv_manifest.csv"))
    rows = man.iloc[:a.limit] if a.limit else man
    N = len(rows)
    os.makedirs(a.scratch, exist_ok=True)
    t0 = time.time()
    seg = local_copy(os.path.join(a.root, a.seg_tar), a.scratch, not a.no_copy)
    drv = local_copy(os.path.join(a.root, a.drv_tar), a.scratch, not a.no_copy)
    imgs, labs = index_tar(seg)
    drvs = index_drv(drv)
    jobs, miss = [], []
    for i, r in enumerate(rows.itertuples()):
        k = (r.v1_split, r.id)
        if k in imgs and k in labs and r.id in drvs:
            jobs.append((i, imgs[k], labs[k], drvs[r.id], w, h))
        else:
            miss.append((r.id, r.v1_split, k in imgs, k in labs, r.id in drvs))
    if miss:
        raise SystemExit(f"thiếu thành viên trong tar cho {len(miss)}/{N} dòng, ví dụ (id, split, ảnh, nhãn, drivable): {miss[:3]}")
    print(f"{N} dòng manifest | ảnh {len(imgs)}, nhãn đường {len(labs)}, drivable {len(drvs)} thành viên | chỉ mục xong sau {time.time() - t0:.0f}s", flush=True)

    out_x = os.path.join(a.scratch, f"pilot_drv_x_{w}x{h}{'_verify' if verify or a.limit else ''}.npy")
    out_y = os.path.join(a.scratch, f"pilot_drv_y_{w}x{h}{'_verify' if verify or a.limit else ''}.npy")
    X = np.lib.format.open_memmap(out_x, "w+", np.uint8, (N, h, w, 3))
    Y = np.lib.format.open_memmap(out_y, "w+", np.uint8, (N, h, w))
    seen = np.zeros(N, bool)
    with Pool(a.procs, initializer=_init, initargs=(seg, drv)) as pool:
        for k, (i, x, y) in enumerate(pool.imap(one, jobs, chunksize=16), 1):
            X[i], Y[i], seen[i] = x, y, True
            if k % 500 == 0:
                print(f"  dựng {k}/{N}  {time.time() - t0:.0f}s", flush=True)
    assert seen.all(), "có dòng chưa được dựng"
    X.flush(); Y.flush()
    cnt = np.bincount(np.asarray(Y).ravel(), minlength=256)
    print("phân bố nhãn:", {int(c): f"{cnt[c] / cnt.sum():.1%}" for c in (0, 1, 2, 3, 255)}, flush=True)

    if verify:
        rx = np.load(os.path.join(V2, "pilot_drv_x_320x160.npy"), mmap_mode="r")
        ry = np.load(os.path.join(V2, "pilot_drv_y_320x160.npy"), mmap_mode="r")
        assert rx.shape[0] >= N and ry.shape[0] >= N, f"cache có sẵn có {rx.shape[0]} dòng, cần ít nhất {N}"
        dx, dy = differ(X, rx[:N], True), differ(Y, ry[:N], False)
        print(f"KIỂM CHỨNG 320x160 trên {N} dòng: pixel ảnh lệch {dx}, pixel nhãn lệch {dy}")
        print("KẾT QUẢ:", "ĐẠT" if dx == 0 and dy == 0 else "KHÔNG ĐẠT", flush=True)
        for f in (out_x, out_y):
            os.remove(f)
        sys.exit(0 if dx == 0 and dy == 0 else 1)

    if w % 320 == 0 and h % 160 == 0 and os.path.exists(os.path.join(V2, "pilot_drv_y_320x160.npy")):   # chỉ để tham khảo
        ry = np.load(os.path.join(V2, "pilot_drv_y_320x160.npy"), mmap_mode="r")[:N]
        sub = np.asarray(Y)[:, ::h // 160, ::w // 320]
        v = np.asarray(ry) != 255
        print(f"tham khảo: nhãn {w}x{h} thu nhỏ nearest về 320x160 khớp nhãn 320x160 ở {(sub == np.asarray(ry))[v].mean():.2%} pixel hợp lệ", flush=True)
    if a.limit:
        print(f"--limit: chỉ thử, KHÔNG ghi lên Drive. Tệp tạm: {out_x}")
        return
    for f in (out_x, out_y):
        dst = os.path.join(V2, os.path.basename(f))
        shutil.copyfile(f, dst + ".tmp"); os.replace(dst + ".tmp", dst); os.remove(f)
        print("đã ghi", dst, flush=True)
    print(f"XONG sau {(time.time() - t0) / 60:.1f} phút")


if __name__ == "__main__":
    main()
