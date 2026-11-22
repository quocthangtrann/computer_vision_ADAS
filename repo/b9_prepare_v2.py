"""B9a: theo từng shard, tải ảnh + mask cùng shard, dựng cache (mặc định 320x160), lưu lên Drive (chạy lại thì bỏ qua shard đã xong).

--w --h: cỡ cache. Khác 320x160 thì ghi vào data/v2/b9_chunks_{w}x{h}/ và cần có sẵn pilot_drv_y_{w}x{h}.npy
(dựng bằng repo/make_hd_pilot.py) để ảnh thí điểm giữ nhãn đủ 4 lớp ở cùng cỡ.
Ghi thẳng vào memmap cục bộ (không gom cả shard trong RAM): 1 shard 640x320 khoảng 4,5 GB ảnh + 1,5 GB nhãn.
"""
import os, time, glob, shutil, argparse, numpy as np, pandas as pd
from PIL import Image
from multiprocessing import Pool
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import snapshot_download, HfApi
ap = argparse.ArgumentParser(); ap.add_argument('--shards', type=int, default=0, help='0 = tất cả')
ap.add_argument('--w', type=int, default=320); ap.add_argument('--h', type=int, default=160); a = ap.parse_args()
W, H = a.w, a.h; TAG = f'{W}x{H}'
DRV = os.environ.get('CAPSTONE_ROOT', '/content/drive/MyDrive/capstone_ai'); V2 = f'{DRV}/data/v2'
CH = f'{V2}/b9_chunks' if (W, H) == (320, 160) else f'{V2}/b9_chunks_{TAG}'; os.makedirs(CH, exist_ok=True)
REPO = 'dronefreak/BDD100K-Drivable-Area-Segmentation'; TOK = os.environ.get('HF_TOKEN') or None
LR = os.environ.get('B9_LOCAL', '/content')                 # đĩa cục bộ cho tệp tạm
LD, LOC = f'{LR}/b9_lab_dl', f'{LR}/b9_img'
T0 = time.time()
def log(*m): print(f'[{(time.time() - T0) / 60:6.1f} phút]', *m, flush=True)
def dl(pat, local):
    for k in range(20):
        try: snapshot_download(REPO, repo_type='dataset', local_dir=local, max_workers=16, token=TOK, allow_patterns=pat); return
        except Exception as ex: log('lỗi tải, thử lại sau 60 s:', repr(ex)[:150]); time.sleep(60)
    raise RuntimeError('tải thất bại')
log('token:', 'có' if TOK else 'KHÔNG', '| cỡ cache', TAG, '->', CH)

man = pd.read_csv(f'{V2}/manifest_v2_in100k.csv')
bad = set(man[man.split_v2.isin(['val', 'test', 'dropped_overlap'])].prefix)
pm = pd.read_csv(f'{V2}/pilot_drv_manifest.csv')
if not os.path.exists(f'{V2}/pilot_drv_y_{TAG}.npy'): raise SystemExit(f'chưa có {V2}/pilot_drv_y_{TAG}.npy: dựng bằng repo/make_hd_pilot.py --w {W} --h {H}')
PY = np.load(f'{V2}/pilot_drv_y_{TAG}.npy', mmap_mode='r')
assert len(PY) == len(pm), f'pilot_drv_y_{TAG}.npy có {len(PY)} dòng, manifest có {len(pm)}'
full = {r.id: k for k, r in enumerate(pm.itertuples()) if r.split_v2 == 'train'}
shards = sorted({p.split('/')[4] for p in HfApi().list_repo_files(REPO, repo_type='dataset', token=TOK)
                 if p.startswith('data/images_100k/100k/train/shard_')})
if a.shards: shards = shards[:a.shards]
todo = [s for s in shards if not os.path.exists(f'{CH}/{s}_ids.csv')]
log(f'chọn {len(shards)} shard, đã có trên Drive {len(shards) - len(todo)}, còn {len(todo)}')

CROP, SZ = (0, 0, 1280, 600), (W, H)
def one(x):
    i, ip, lp = x
    try:
        im = Image.open(ip).convert('RGB'); dm = Image.open(lp)
        if im.size != (1280, 720) or dm.size != (1280, 720): return None
        d = np.asarray(dm.crop(CROP).resize(SZ, Image.NEAREST))
        y = np.full(d.shape, 254, np.uint8); y[d == 1] = 2; y[d == 2] = 3
        if i in full: y = np.asarray(PY[full[i]])
        return i, np.asarray(im.crop(CROP).resize(SZ, Image.BILINEAR)), y
    except Exception: return None

times = []
for n, s in enumerate(todo, 1):
    t0 = time.time(); shutil.rmtree(LOC, ignore_errors=True)
    log(f'shard {n}/{len(todo)} ({s}): tải ảnh (Terminal: find {LOC} -name "*.jpg" | wc -l)')
    dl([f'data/images_100k/100k/train/{s}/*'], LOC)
    log(f'  tải mask cùng shard (Terminal: find {LD} -name "*.png" | wc -l)')
    dl([f'data/drivable_maps/labels/train/{s}/*'], LD)
    lab = {os.path.basename(p)[:-16]: p for p in glob.glob(f'{LD}/**/*_drivable_id.png', recursive=True)}
    imgs = [(os.path.basename(p)[:-4], p) for p in glob.glob(f'{LOC}/data/images_100k/100k/train/{s}/*.jpg')]
    miss = sum(i not in lab for i, _ in imgs)
    jobs = [(i, p, lab[i]) for i, p in sorted(imgs) if i in lab and i.split('-')[0] not in bad]
    log(f'  tải xong sau {(time.time() - t0) / 60:.1f} phút | ảnh {len(imgs)}, thiếu mask {miss}, dùng {len(jobs)}')
    xf, yf = f'{LR}/{s}_x.npy', f'{LR}/{s}_y.npy'
    X = np.lib.format.open_memmap(xf, 'w+', np.uint8, (len(jobs), H, W, 3))
    Y = np.lib.format.open_memmap(yf, 'w+', np.uint8, (len(jobs), H, W))
    ids, k_ok = [], 0
    with Pool(os.cpu_count()) as pool:
        for k, r in enumerate(pool.imap(one, jobs, chunksize=64), 1):
            if r is not None: ids.append(r[0]); X[k_ok] = r[1]; Y[k_ok] = r[2]; k_ok += 1
            if k % 2000 == 0: log(f'  dựng cache {k}/{len(jobs)}')
    X.flush(); Y.flush()
    if k_ok < len(jobs):       # có ảnh lỗi/sai cỡ: ghi lại chỉ k_ok hàng đầu
        for f, arr in ((xf, X), (yf, Y)):
            np.save(f + '.trim.npy', arr[:k_ok])
        del X, Y
        for f in (xf, yf): os.replace(f + '.trim.npy', f)
    else:
        del X, Y
    log(f'  chép lên Drive ({(os.path.getsize(xf) + os.path.getsize(yf)) / 1e9:.1f} GB)')
    for nm, f in [('x', xf), ('y', yf)]:
        shutil.copyfile(f, f'{CH}/{s}_{nm}.npy.tmp'); os.replace(f'{CH}/{s}_{nm}.npy.tmp', f'{CH}/{s}_{nm}.npy')
        os.remove(f)
    pd.DataFrame(dict(id=ids, full_label=[i in full for i in ids])).to_csv(f'{CH}/{s}_ids.csv', index=False)
    shutil.rmtree(LOC, ignore_errors=True)
    times.append(time.time() - t0)
    log(f'  XONG shard {n}/{len(todo)}: {len(ids)} ảnh, {times[-1] / 60:.1f} phút | ước còn lại {(len(todo) - n) * np.mean(times) / 60:.0f} phút')
log('TẤT CẢ XONG:', len(glob.glob(f'{CH}/*_ids.csv')), 'shard trên Drive')
