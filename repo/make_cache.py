"""Dung cache BDD100K 2 lop (san / khong-san) o nhieu do phan giai tu anh goc 1280x720.

Giu DUNG tien xu ly cua cache_v1 (cache_spec.json): crop (0,0,1280,600), anh PIL BILINEAR,
mask NEAREST, nhan 0->1, 1..18->0, 255->255, dung thu tu mau trong train_ids.csv / val_ids.csv.
--check: dung lai 320x160 va 320x320 cho vai chuc mau dau roi so voi cache cu (--ref_dir).
Che do dung that (khong co --check) cung chay buoc nay truoc va dung neu khong khop.
"""
import argparse, csv, os, sys, time
import numpy as np
from PIL import Image
from multiprocessing import Pool

CROP = (0, 0, 1280, 600)          # (trai, tren, phai, duoi) tren anh 1280x720
KEY = 17                           # id dang 'xxxxxxxx-xxxxxxxx'


def index_files(root):
    imgs = {"train": {}, "val": {}}
    labs = {"train": {}, "val": {}}
    for dp, _, fs in os.walk(root):
        d = "/" + os.path.relpath(dp, root).replace("\\", "/") + "/"
        sp = "train" if "/train" in d else ("val" if "/val" in d else None)
        if sp is None:
            continue
        if "/images_10k/" in d:
            for f in fs:
                if f.lower().endswith(".jpg"):
                    imgs[sp][f[:KEY]] = os.path.join(dp, f)
        elif "/seg_maps/labels/" in d:
            for f in fs:
                if f.lower().endswith(".png"):
                    labs[sp][f[:KEY]] = os.path.join(dp, f)
    return imgs, labs


def read_ids(path):
    with open(path) as f:
        return [r["id"] for r in csv.DictReader(f)]


def load_one(a):
    ip, lp, sizes = a
    im = Image.open(ip).convert("RGB")
    lb = Image.open(lp)
    if im.size != (1280, 720) or lb.size != (1280, 720):
        raise RuntimeError(f"kich thuoc la: {ip} {im.size} / {lp} {lb.size}")
    m = np.array(lb)
    if m.ndim != 2:
        raise RuntimeError(f"nhan khong phai 1 kenh: {lp} shape {m.shape} mode {lb.mode}")
    m = m[CROP[1]:CROP[3], CROP[0]:CROP[2]]
    m = np.where(m == 255, 255, np.where(m == 0, 1, 0)).astype(np.uint8)
    imc = im.crop(CROP)
    mi = Image.fromarray(m)
    out = []
    for (w, h) in sizes:
        out.append((np.asarray(imc.resize((w, h), Image.BILINEAR)),
                    np.asarray(mi.resize((w, h), Image.NEAREST))))
    return out


def parse_sizes(s):
    return [tuple(int(v) for v in t.lower().split("x")) for t in s.split(",") if t]


def build(root, ids_dir, split, sizes, out_dir=None, limit=0, procs=0):
    imgs, labs = index_files(root)
    ids = read_ids(os.path.join(ids_dir, f"{split}_ids.csv"))
    if limit:
        ids = ids[:limit]
    miss = [i for i in ids if i not in imgs[split] or i not in labs[split]]
    if miss:
        raise RuntimeError(f"{split}: thieu {len(miss)} mau trong tar da giai nen, vi du {miss[:3]}")
    jobs = [(imgs[split][i], labs[split][i], sizes) for i in ids]
    n = len(ids)
    xs, ys, parts = [], [], []
    for (w, h) in sizes:
        if out_dir is None:
            xs.append(np.zeros((n, h, w, 3), np.uint8)); ys.append(np.zeros((n, h, w), np.uint8))
        else:
            os.makedirs(out_dir, exist_ok=True)
            px = os.path.join(out_dir, f"{split}_x_{w}x{h}.npy.part")
            py = os.path.join(out_dir, f"{split}_y_{w}x{h}.npy.part")
            xs.append(np.lib.format.open_memmap(px, mode="w+", dtype=np.uint8, shape=(n, h, w, 3)))
            ys.append(np.lib.format.open_memmap(py, mode="w+", dtype=np.uint8, shape=(n, h, w)))
            parts.append((px, py, w, h))
    t0 = time.time()
    procs = procs or max(1, (os.cpu_count() or 2))
    with Pool(procs) as pool:
        for k, res in enumerate(pool.imap(load_one, jobs, chunksize=8)):
            for j, (xi, yi) in enumerate(res):
                xs[j][k] = xi; ys[j][k] = yi
            if (k + 1) % 500 == 0 or k + 1 == n:
                print(f"  {split}: {k+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    if out_dir is not None:
        for a_, b_ in zip(xs, ys):
            a_.flush(); b_.flush()
        del xs, ys
        for px, py, w, h in parts:
            os.replace(px, os.path.join(out_dir, f"{split}_x_{w}x{h}.npy"))
            os.replace(py, os.path.join(out_dir, f"{split}_y_{w}x{h}.npy"))
        return None
    return xs, ys


def check(root, ids_dir, ref_dir, n_train=64, n_val=32, procs=0):
    ok = True
    sizes = [(320, 160), (320, 320)]
    for split, n in (("train", n_train), ("val", n_val)):
        xs, ys = build(root, ids_dir, split, sizes, None, limit=n, procs=procs)
        for (w, h), x, y in zip(sizes, xs, ys):
            rx = np.load(os.path.join(ref_dir, f"{split}_x_{w}x{h}.npy"), mmap_mode="r")[:n]
            ry = np.load(os.path.join(ref_dir, f"{split}_y_{w}x{h}.npy"), mmap_mode="r")[:n]
            dx = np.abs(x.astype(int) - np.asarray(rx).astype(int))
            my = float((y != np.asarray(ry)).mean())
            exact = float((dx == 0).mean())
            good = (dx.max() <= 3) and (dx.mean() < 0.1) and (my < 1e-3)
            ok &= bool(good)
            print(f"  [{split} {w}x{h}] anh: lech max {dx.max()}, lech TB {dx.mean():.4f}, "
                  f"bang nhau {exact*100:.1f}% | mask khac {my*100:.4f}%  -> {'KHOP' if good else 'KHONG KHOP'}")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="thu muc da giai nen tar (chua bdd100k_seg/)")
    p.add_argument("--ids_dir", required=True)
    p.add_argument("--ref_dir", required=True, help="cache_v1 cu de doi chieu")
    p.add_argument("--out", default="/content/cache_v1")
    p.add_argument("--sizes", default="1280x608,960x448,640x288,480x224")
    p.add_argument("--check", action="store_true")
    p.add_argument("--no_check", action="store_true")
    p.add_argument("--procs", type=int, default=0)
    a = p.parse_args()
    if not a.no_check:
        print("== doi chieu voi cache_v1 (320x160, 320x320) ==")
        if not check(a.root, a.ids_dir, a.ref_dir, procs=a.procs):
            print("KHONG KHOP voi cache cu -> dung, chua dung cache nao. Gui output nay cho Claude.")
            sys.exit(1)
        print("KHOP -> tien xu ly giong cache cu.")
    if a.check:
        sys.exit(0)
    sizes = parse_sizes(a.sizes)
    for split in ("val", "train"):
        print(f"== dung {split} {sizes} -> {a.out} ==")
        build(a.root, a.ids_dir, split, sizes, a.out, procs=a.procs)
    print("xong. file:")
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".npy"):
            print(f"  {f}  {os.path.getsize(os.path.join(a.out, f))/2**30:.2f} GiB")
