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
B9, TAG, SEEDS = 'shard_000', 'b9s1', range(5)   # 2 shard: B9='shard_000,shard_001', TAG='b9s2'
T = time.time()
for s in SEEDS:
    n = f'{TAG}_c4aug_s{s}'
    if os.path.exists(f'runs/{n}/final_metrics.json'): print(n, 'đã xong, bỏ qua'); continue
    print(f'\n>>> {n}  [{(time.time() - T) / 60:.1f} phút]', flush=True)
    pr = subprocess.Popen([sys.executable, '-u', 'repo/train_v3.py', '--name', n, '--seed', str(s), '--b9', B9],
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

def nb(cells):
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": c.splitlines(keepends=True)} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}

for name, cells in [('10_data', [DEPLOY, DATA]), ('20_train', [DEPLOY, TRAIN]), ('30_eval', [EVAL_A, EVAL_B])]:
    json.dump(nb(cells), open(f'notebooks/{name}.ipynb', 'w'), ensure_ascii=False, indent=1)
    print('đã tạo notebooks/' + name + '.ipynb')