
import os, time, numpy as np, pandas as pd, shutil
from PIL import Image
from multiprocessing import Pool
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import snapshot_download
DRV = '/content/drive/MyDrive/capstone_ai'; V2 = f'{DRV}/data/v2'
L, OUT = '/content/bdd100k_drv', '/content/b9'; os.makedirs(OUT, exist_ok=True)
REPO = 'dronefreak/BDD100K-Drivable-Area-Segmentation'
t0 = time.time()
for k in range(20):
    try:
        snapshot_download(REPO, repo_type='dataset', local_dir=L, max_workers=16, token=os.environ.get('HF_TOKEN') or None,
                          allow_patterns=['data/images_100k/100k/train/*', 'data/drivable_maps/labels/train/*']); break
    except Exception as ex:
        print('lỗi tải, thử lại sau 60 s:', repr(ex)[:200], flush=True); time.sleep(60)
print('tải xong sau', round(time.time() - t0), 's', flush=True)
img, lab = {}, {}
for dp, _, fs in os.walk(f'{L}/data'):
    for f in fs:
        if f.endswith('.jpg') and '/images_100k/' in dp: img[f[:-4]] = os.path.join(dp, f)
        elif f.endswith('_drivable_id.png'): lab[f[:-16]] = os.path.join(dp, f)
ids = sorted(set(img) & set(lab)); print('ảnh', len(img), '| mask', len(lab), '| cặp', len(ids), flush=True)
man = pd.read_csv(f'{V2}/manifest_v2_in100k.csv')
bad = set(man[man.split_v2.isin(['val', 'test', 'dropped_overlap'])].prefix)
keep = [i for i in ids if i.split('-')[0] not in bad]
print('loại vì cùng video với val/test v2:', len(ids) - len(keep), flush=True)
pm = pd.read_csv(f'{V2}/pilot_drv_manifest.csv'); PY = np.load(f'{V2}/pilot_drv_y_320x160.npy', mmap_mode='r')
full = {r.id: k for k, r in enumerate(pm.itertuples()) if r.split_v2 == 'train'}
CROP, SZ = (0, 0, 1280, 600), (320, 160)
def one(i):
    try:
        im = Image.open(img[i]).convert('RGB'); dm = Image.open(lab[i])
        if im.size != (1280, 720) or dm.size != (1280, 720): return None
        d = np.asarray(dm.crop(CROP).resize(SZ, Image.NEAREST))
        y = np.full(d.shape, 254, np.uint8); y[d == 1] = 2; y[d == 2] = 3
        return np.asarray(im.crop(CROP).resize(SZ, Image.BILINEAR)), y
    except Exception: return None
N = len(keep)
X = np.lib.format.open_memmap(f'{OUT}/b9_x_320x160.npy', 'w+', np.uint8, (N, 160, 320, 3))
Y = np.lib.format.open_memmap(f'{OUT}/b9_y_320x160.npy', 'w+', np.uint8, (N, 160, 320))
ok = np.zeros(N, bool); t0 = time.time()
with Pool(os.cpu_count()) as pool:
    for k, r in enumerate(pool.imap(one, keep, chunksize=64)):
        if r is not None:
            X[k] = r[0]; Y[k] = PY[full[keep[k]]] if keep[k] in full else r[1]; ok[k] = True
        if (k + 1) % 5000 == 0: print(f'  dựng cache {k + 1}/{N}  {time.time() - t0:.0f}s', flush=True)
X.flush(); Y.flush()
pd.DataFrame(dict(id=keep, ok=ok, full_label=[i in full for i in keep])).to_csv(f'{OUT}/b9_manifest.csv', index=False)
print('ảnh dùng được', int(ok.sum()), '/', N, '| có nhãn đủ 4 lớp:', sum(i in full for i in keep), flush=True)
for f in ['b9_manifest.csv', 'b9_x_320x160.npy', 'b9_y_320x160.npy']:
    shutil.copyfile(f'{OUT}/{f}', f'{V2}/{f}.tmp'); os.replace(f'{V2}/{f}.tmp', f'{V2}/{f}')
print('đã chép lên Drive. XONG', flush=True)
