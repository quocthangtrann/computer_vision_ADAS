import json, os
os.makedirs('notebooks', exist_ok=True)

DEPLOY = r"""import os, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')
CODE, URL = '/content/code', 'github.com/quocthangtrann/computer_vision_ADAS.git'
env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
if not os.path.exists(CODE):
    subprocess.run(f'git clone https://{URL} {CODE}', shell=True, check=True, env=env)
else:
    subprocess.run(f'git -C {CODE} pull', shell=True, check=True, env=env)
subprocess.run(f'rsync -a --exclude .git {CODE}/repo/ /content/drive/MyDrive/capstone_ai/repo/', shell=True, check=True)
try: import segmentation_models_pytorch
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch'], check=True)
os.chdir('/content/drive/MyDrive/capstone_ai')
print('mã mới nhất đã ở capstone_ai/repo/ | thư mục làm việc:', os.getcwd())
"""

DATA = r"""import os
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
N_SHARDS = 1   # đổi thành 2, 3, ... để tải thêm; shard đã có trên Drive được bỏ qua
!python -u repo/b9_prepare_v2.py --shards {N_SHARDS} 2>&1 | tee -a runs/b9_prepare_v2.log
"""

TRAIN = r"""import subprocess, sys, time, os
import torch; assert torch.cuda.is_available(), 'Chưa có GPU: đổi runtime sang GPU'
B9, TAG, SEEDS, EPOCHS = 'shard_000', 'b9s1', range(5), 60    # 2 shard: B9='shard_000,shard_001', TAG='b9s2'
T = time.time()
for s in SEEDS:
    n = f'{TAG}_c4aug_s{s}'
    if os.path.exists(f'runs/{n}/final_metrics.json'): print(n, 'đã xong, bỏ qua'); continue
    print(f'\n>>> {n}  [{(time.time() - T) / 60:.1f} phút]', flush=True)
    pr = subprocess.Popen([sys.executable, '-u', 'repo/train_v3.py', '--name', n, '--seed', str(s), '--b9', B9, '--epochs', str(EPOCHS)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(f'runs/{n}.log', 'a') as lg:
        for ln in pr.stdout:
            lg.write(ln)
            if not ln.startswith('ep ') or ln.split()[1].split('/')[0] in ('1', '10', '20', '30', '40', '50', '60'): print(ln, end='', flush=True)
    rc = pr.wait(); print('mã thoát:', rc, flush=True)
    if rc != 0: print('LỖI: dừng lại, dán đoạn trên cho tôi'); break
else: print(f'\nXONG sau {(time.time() - T) / 60:.1f} phút')
"""

EVAL_A = r"""from google.colab import drive
drive.mount('/content/drive')
import os; os.chdir('/content/drive/MyDrive/capstone_ai'); print(os.getcwd())
"""

EVAL_B = r"""import json, glob, numpy as np, pandas as pd
TAG = 'b9s1'
def load(pat):
    r = [json.load(open(p)) for p in sorted(glob.glob(f'runs/{pat}/final_metrics.json'))]
    return pd.DataFrame([dict(seed=f['seed'], road=f['road_iou'], ff=f['false_floor'], notours=f['iou'][1], ego=f['iou'][2], adj=f['iou'][3]) for f in r])
A, B = load('pilot_c4aug_s*'), load(f'{TAG}_c4aug_s*')
print('n:', len(A), 'cơ sở (2.826 ảnh) |', len(B), 'có thêm dữ liệu b9')
for c in ['road', 'ff', 'notours', 'ego', 'adj']:
    d = B[c].mean() - A[c].mean(); se = np.sqrt(A[c].var() / len(A) + B[c].var() / len(B))
    print(f'{c:8s} cơ sở {A[c].mean():.4f} ± {A[c].std():.4f} | thêm b9 {B[c].mean():.4f} ± {B[c].std():.4f} | hiệu {d:+.4f} (t = {d / se:+.1f})')
"""

INT8 = r"""RUN, OUT = 'runs/b9s4_c4aug_s0', 'handoff_s3b'   # đổi RUN khi lượng tử mô hình khác
!python -u repo/int8_sim.py --run {RUN} --out {OUT} 2>&1 | tee -a runs/int8_sim.log
!python -u repo/int8_pack.py --run {RUN} --out {OUT} 2>&1 | tee -a runs/int8_pack.log
"""

Q_CACHE = r"""import os, subprocess, sys
# Chạy được trên runtime CPU. Bắt buộc kiểm chứng trước: dựng lại 320x160 và so từng pixel với cache có sẵn, chỉ dựng HD khi ĐẠT.
if os.path.exists('data/v2/pilot_drv_x_640x320.npy') and os.path.exists('data/v2/pilot_drv_y_640x320.npy'):
    print('đã có cache 640x320, bỏ qua (xóa hai tệp pilot_drv_*_640x320.npy trên Drive nếu muốn dựng lại)')
else:
    for w, h in [(320, 160), (640, 320)]:
        r = subprocess.run([sys.executable, '-u', 'repo/make_hd_pilot.py', '--w', str(w), '--h', str(h)])
        if r.returncode != 0: raise SystemExit(f'make_hd_pilot {w}x{h} lỗi (mã {r.returncode}): dừng, dán đoạn trên cho tôi')
"""

Q_MAC = r"""import sys
sys.path.insert(0, 'repo')
import models_custom as mc
bad_all = {}
def table(names, h, w):
    for n in names:
        net = mc.build_fsnet(n, num_classes=4).eval()
        rows = mc.layer_table(net, h, w)
        bad = mc.contract_violations(net)
        bad_all[(n, w, h)] = bad
        print(f'{n:14s} {w}x{h}  {sum(r["mac"] for r in rows) / 1e9:7.4f} GMAC | {sum(p.numel() for p in net.parameters()) / 1e6:.3f} M tham số | vi phạm hợp đồng: {bad or "không"}')
table(['fsnet_m', 'fsnet_m_deep', 'fsnet_m_ctx', 'fsnet_l'], 160, 320)
table(['fsnet_m', 'fsnet_m_p1', 'fsnet_m_p1s5'], 320, 640)
assert not any(bad_all.values()), 'có mô hình vi phạm hợp đồng toán tử'
"""

Q_RUN = r"""import subprocess, sys, time, os
import torch; assert torch.cuda.is_available(), 'Chưa có GPU: đổi runtime sang GPU'
ONLY = None    # None = chạy cả hàng đợi; hoặc chuỗi nhóm, ví dụ 'a', 'bc', 'd'
EPOCHS = 60
S4 = 'shard_000,shard_001,shard_002,shard_003'
# (nhóm, tag, tham số thêm cho train_v3.py, các seed). Tên run = {tag}_c4aug_s{seed}
JOBS = [
    ('a', 'hd640_p1',   ['--arch', 'fsnet_m_p1',   '--w', '640', '--h', '320'], [0, 1, 2]),
    ('a', 'hd640_p1s5', ['--arch', 'fsnet_m_p1s5', '--w', '640', '--h', '320'], [0, 1, 2]),
    ('a', 'hd640_m',    ['--arch', 'fsnet_m',      '--w', '640', '--h', '320'], [0, 1, 2]),
    ('b', 'b9s1_deep',  ['--arch', 'fsnet_m_deep', '--b9', 'shard_000'], [0, 1, 2]),
    ('b', 'b9s1_ctx',   ['--arch', 'fsnet_m_ctx',  '--b9', 'shard_000'], [0, 1, 2]),
    ('c', 'b9s4_fw4',   ['--arch', 'fsnet_m', '--b9', S4, '--full_w', '4'], [0, 1, 2]),
    ('d', 'b9s4_wide',  ['--arch', 'fsnet_l', '--b9', S4], [0]),
]
def run_queue():
    T = time.time()
    for grp, tag, extra, seeds in JOBS:
        if ONLY and grp not in ONLY: continue
        for s in seeds:
            n = f'{tag}_c4aug_s{s}'
            if os.path.exists(f'runs/{n}/final_metrics.json'): print(n, 'đã xong, bỏ qua'); continue
            print(f'\n>>> [{grp}] {n}  [{(time.time() - T) / 60:.1f} phút]', flush=True)
            cmd = [sys.executable, '-u', 'repo/train_v3.py', '--name', n, '--seed', str(s), '--epochs', str(EPOCHS)] + extra
            pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with open(f'runs/{n}.log', 'a') as lg:
                for ln in pr.stdout:
                    lg.write(ln)
                    if not ln.startswith('ep ') or int(ln.split()[1].split('/')[0]) % 20 == 0: print(ln, end='', flush=True)
            rc = pr.wait(); print('mã thoát:', rc, flush=True)
            if rc != 0: print(f'LỖI ở {n}: dừng hàng đợi, dán đoạn log trên cho tôi'); return False
    print(f'\nXONG sau {(time.time() - T) / 60:.1f} phút'); return True
run_queue()
"""

Q_CMP = r"""import json, glob, numpy as np
def load(pat): return [json.load(open(p)) for p in sorted(glob.glob(f'runs/{pat}/final_metrics.json'))]
def metrics(f, g320):
    k = '_320' if g320 and 'iou_320' in f else ''
    i = f['iou' + k]
    return dict(ego=i[2], adj=i[3], notours=i[1], road=f['road_iou' + k], ff=f['false_floor' + k])
def col(fs, c, g320): return np.array([metrics(f, g320)[c] for f in fs])
def welch(a, b):
    d = b.mean() - a.mean()
    if len(a) < 2 or len(b) < 2: return d, float('nan')
    return d, d / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
def report(title, base_pat, pat, g320=False):
    A, B = load(base_pat), load(pat)
    if not B: print(f'== {title}: chưa có run nào ({pat})\n'); return
    print(f'== {title}: n = {len(B)} so với mốc {base_pat} (n = {len(A)})')
    for c in ['ego', 'adj', 'notours', 'road', 'ff']:
        a, b = col(A, c, g320), col(B, c, g320)
        d, t = welch(a, b)
        sb = f'{b.std(ddof=1):.4f}' if len(b) > 1 else '  n=1 '
        z = f' | n=1: không có t, hiệu / độ lệch chuẩn mốc = {d / a.std(ddof=1):+.1f}' if len(b) < 2 else ''
        ts = 'n/a' if t != t else f'{t:+.1f}'
        print(f'  {c:8s} mốc {a.mean():.4f} ± {a.std(ddof=1):.4f} | nhóm {b.mean():.4f} ± {sb} | hiệu {d:+.4f} (t = {ts}){z}')
    print()
# HD: mốc là pilot_c4aug_s* (cùng dữ liệu thí điểm, cùng aug mạnh). iou ở lưới 640x320 dùng nhãn riêng nên KHÔNG so thẳng; iou_320 so được.
for tag in ['hd640_p1', 'hd640_p1s5', 'hd640_m']:
    report(f'{tag} [iou ở lưới 640x320 của chính nó, chỉ để tham khảo]', 'pilot_c4aug_s*', f'{tag}_c4aug_s*', False)
    report(f'{tag} [iou_320: argmax HD thu nhỏ nearest về lưới 320x160]', 'pilot_c4aug_s*', f'{tag}_c4aug_s*', True)
for tag in ['b9s1_deep', 'b9s1_ctx']: report(tag, 'b9s1_c4aug_s*', f'{tag}_c4aug_s*')
for tag in ['b9s4_fw4', 'b9s4_wide']: report(tag, 'b9s4_c4aug_s*', f'{tag}_c4aug_s*')
"""

def nb(cells):
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": c.splitlines(keepends=True)} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}

for name, cells in [('10_data', [DEPLOY, DATA]), ('20_train', [DEPLOY, TRAIN]), ('30_eval', [EVAL_A, EVAL_B]), ('40_int8', [DEPLOY, INT8]),
                    ('60_queue', [DEPLOY, Q_CACHE, Q_MAC, Q_RUN, Q_CMP])]:
    json.dump(nb(cells), open(f'notebooks/{name}.ipynb', 'w'), ensure_ascii=False, indent=1)
    print('đã tạo notebooks/' + name + '.ipynb')
    
    