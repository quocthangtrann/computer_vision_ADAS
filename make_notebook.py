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

INT8 = r"""import os, subprocess, sys
os.makedirs('runs', exist_ok=True)
# CẤU HÌNH. A (mặc định): FSNet-M 320x160, run b9s4_c4aug_s0.  B (HD): W, H, ARCH = 640, 320, 'fsnet_m_p1s5', RUN = 'runs/hd640s1_p1s5_kdT_c4aug_s0'
W, H, ARCH = 320, 160, 'fsnet_m'
RUN = 'runs/b9s4_c4aug_s0'      # last.pt["model"] là trọng số triển khai (bản EMA nếu run bật EMA); trùng với tệp trong models_final/
OUT = 'handoff_s3b' if (W, H) == (320, 160) else f'handoff_s3b_{ARCH}_{W}x{H}'   # gói HD ghi ra thư mục khác, không đè gói 320x160
SZ = ['--arch', ARCH, '--w', str(W), '--h', str(H)]

def run(script, *args):
    cmd = [sys.executable, '-u', f'repo/{script}', *args]
    print('>>>', ' '.join(cmd), flush=True)
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(f'runs/{script[:-3]}.log', 'a') as lg:
        for ln in pr.stdout:
            lg.write(ln); print(ln, end='', flush=True)
    if pr.wait() != 0: raise SystemExit(f'{script} dừng (mã thoát {pr.returncode}): dán đoạn log trên cho tôi, KHÔNG chạy bước sau')

# Chạy được trên runtime CPU. Bước FP32 của int8_sim so ONNX với final_metrics.json của run: lệch quá 0,002 thì dừng.
if (W, H) != (320, 160): run('make_hd_calib.py', '--w', str(W), '--h', str(H))   # 200 ảnh hiệu chuẩn + 100 ảnh test HD; có rồi thì bỏ qua
run('int8_sim.py', '--run', RUN, '--out', OUT, *SZ)
run('int8_pack.py', '--run', RUN, '--out', OUT, *SZ)
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

H_CFG = r"""# CẤU HÌNH: chạy cell 0 và cell này ở MỖI phiên Colab (CPU hay GPU đều vậy)
W, H = 640, 320     # 1280, 576 chỉ nên dùng với NS = 0 (xem hướng dẫn: shard 1280x576 nặng khoảng 22 GB mỗi shard)
NS = 1              # số shard b9 (0 = chỉ bộ thí điểm; 1 so với b9s1_c4aug_s*; 4 so với b9s4_c4aug_s*)
TAG = f'hd{W}s{NS}'
SHARDS = ','.join(f'shard_{i:03d}' for i in range(NS))
BASE = {0: 'pilot_c4aug_s*', 1: 'b9s1_c4aug_s*', 4: 'b9s4_c4aug_s*'}.get(NS)   # mốc 320x160 cùng dữ liệu và cùng aug
print(f'cỡ {W}x{H} | {NS} shard ({SHARDS or "không"}) | tag {TAG} | mốc so sánh: {BASE}')
"""

H_CACHE_P = r"""import os, subprocess, sys
# Chạy được trên runtime CPU. Kiểm chứng trước: dựng lại 320x160 và so từng pixel với cache có sẵn, chỉ dựng cỡ HD khi ĐẠT.
if os.path.exists(f'data/v2/pilot_drv_x_{W}x{H}.npy') and os.path.exists(f'data/v2/pilot_drv_y_{W}x{H}.npy'):
    print(f'đã có cache thí điểm {W}x{H}, bỏ qua (xóa hai tệp pilot_drv_*_{W}x{H}.npy trên Drive nếu muốn dựng lại)')
else:
    for w, h in [(320, 160), (W, H)]:
        r = subprocess.run([sys.executable, '-u', 'repo/make_hd_pilot.py', '--w', str(w), '--h', str(h)])
        if r.returncode != 0: raise SystemExit(f'make_hd_pilot {w}x{h} lỗi (mã {r.returncode}): dừng, dán đoạn trên cho tôi')
"""

H_CACHE_S = r"""import os, subprocess, sys
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
# Cần cache thí điểm cùng cỡ ở cell trên. Shard đã có trên Drive được bỏ qua; có thể chạy lại sau khi bị ngắt.
if NS == 0: print('NS = 0: không cần shard')
else:
    os.makedirs('runs', exist_ok=True)
    with open(f'runs/b9_prepare_{W}x{H}.log', 'a') as lg:
        pr = subprocess.Popen([sys.executable, '-u', 'repo/b9_prepare_v2.py', '--shards', str(NS), '--w', str(W), '--h', str(H)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for ln in pr.stdout: lg.write(ln); print(ln, end='', flush=True)
    if pr.wait() != 0: raise SystemExit('b9_prepare_v2 lỗi: dán đoạn log trên cho tôi')
"""

H_MAC = r"""import sys
sys.path.insert(0, 'repo')
import models_custom as mc
import segmentation_models_pytorch as smp
bad_all = {}
for n in ['fsnet_m_p1', 'fsnet_m_p1s5', 'fsnet_l_p1', 'fsnet_l_p1s5', 'fsnet_m', 'fsnet_l']:
    pc = mc.PRESETS[n]; need = 2 ** (4 + pc.get('pre', 0) + (1 if pc.get('e5') else 0))
    if W % need or H % need: print(f'{n:14s} {W}x{H}: bỏ qua (cần w, h chia hết cho {need})'); continue
    net = mc.build_fsnet(n, num_classes=4).eval(); rows = mc.layer_table(net, H, W)
    ho, wo = mc.fuse_for_deploy(net)(__import__('torch').zeros(1, 3, H, W)).shape[2:]
    bad_all[n] = mc.contract_violations(net)
    print(f'{n:14s} {W}x{H}  {sum(r["mac"] for r in rows) / 1e9:7.3f} GMAC | {sum(p.numel() for p in net.parameters()) / 1e6:.3f} M tham số | lưới logit {wo}x{ho} | vi phạm hợp đồng: {bad_all[n] or "không"}')
assert not any(bad_all.values()), 'có mô hình vi phạm hợp đồng toán tử'
for e in ['mit_b2', 'resnet34']:
    m = smp.Unet(e, encoder_weights=None, classes=4)
    print(f'giáo viên smp:Unet:{e}: encoder {sum(p.numel() for p in m.encoder.parameters()) / 1e6:.2f} M | tổng {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M tham số')
"""

H_RUN = r"""import subprocess, sys, time, os
import torch; assert torch.cuda.is_available(), 'Chưa có GPU: đổi runtime sang GPU'
ONLY = None       # None = cả hàng đợi; hoặc chuỗi nhóm: 't' giáo viên, 's' học trò thường, 'k' học trò chưng cất, 'l' học trò lớn
EPOCHS_T, EPOCHS_S = 40, 60
BIG = W >= 1280   # cỡ 1280x576: lô nhỏ hơn, cộng dồn gradient nhiều hơn (chưa đo bộ nhớ: nếu hết bộ nhớ thì giảm --bs, tăng --accum)
SZ = ['--w', str(W), '--h', str(H)]
B9 = ['--b9', SHARDS] if NS else []
T_BS = ['--bs', '2', '--accum', '8'] if BIG else ['--bs', '8', '--accum', '2']
S_BS = ['--bs', '8', '--accum', '4'] if BIG else []
T_OPT = ['--lr', '6e-4', '--lr_enc_mult', '0.1', '--wd', '0.01', '--wd_nobn', '1', '--ema', '0.999']   # encoder tiền huấn luyện: lr 6e-5
REG = ['--ema', '0.999', '--wd', '0.01', '--wd_nobn', '1', '--ls', '0.05', '--cutout', '2']              # gói chống quá khớp cho học trò
KD = ['--kd', '1.0', '--kd_T', '2.0']                                                                     # trọng số KD 1,0: CHƯA chỉnh
TM, TC = f'{TAG}_mitb2_c4aug_s0', f'{TAG}_unetr34_c4aug_s0'
# (nhóm, tag, tham số thêm cho train_v3.py, các seed, số epoch, giáo viên cần có). Tên run = {tag}_c4aug_s{seed}
JOBS = [
    ('t', f'{TAG}_mitb2',    ['--arch', 'smp:Unet:mit_b2'] + T_BS + T_OPT, [0], EPOCHS_T, None),
    ('t', f'{TAG}_unetr34', ['--arch', 'smp:Unet:resnet34'] + T_BS + T_OPT, [0], EPOCHS_T, None),
    ('s', f'{TAG}_p1s5',     ['--arch', 'fsnet_m_p1s5'] + S_BS, [0, 1, 2], EPOCHS_S, None),
    ('s', f'{TAG}_p1s5_reg', ['--arch', 'fsnet_m_p1s5'] + S_BS + REG, [0, 1, 2], EPOCHS_S, None),
    ('k', f'{TAG}_p1s5_kdT', ['--arch', 'fsnet_m_p1s5'] + S_BS + REG + KD + ['--teacher', TM], [0, 1, 2], EPOCHS_S, TM),
    ('k', f'{TAG}_p1s5_kdC', ['--arch', 'fsnet_m_p1s5'] + S_BS + REG + KD + ['--teacher', TC], [0, 1, 2], EPOCHS_S, TC),
    ('l', f'{TAG}_lp1s5_kdT', ['--arch', 'fsnet_l_p1s5'] + S_BS + REG + KD + ['--teacher', TM], [0, 1, 2], EPOCHS_S, TM),
]
def run_queue():
    T = time.time()
    for grp, tag, extra, seeds, epochs, teacher in JOBS:
        if ONLY and grp not in ONLY: continue
        for s in seeds:
            n = f'{tag}_c4aug_s{s}'
            if os.path.exists(f'runs/{n}/final_metrics.json'): print(n, 'đã xong, bỏ qua'); continue
            if teacher and not os.path.exists(f'runs/{teacher}/final_metrics.json'):
                print(f'DỪNG: {n} cần giáo viên {teacher} đã huấn luyện xong (chạy nhóm t trước)'); return False
            print(f'\n>>> [{grp}] {n}  [{(time.time() - T) / 60:.1f} phút]', flush=True)
            cmd = [sys.executable, '-u', 'repo/train_v3.py', '--name', n, '--seed', str(s), '--epochs', str(epochs)] + SZ + B9 + extra
            pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with open(f'runs/{n}.log', 'a') as lg:
                for ln in pr.stdout:
                    lg.write(ln)
                    if not ln.startswith('ep ') or int(ln.split()[1].split('/')[0]) in (1, ) or int(ln.split()[1].split('/')[0]) % 10 == 0: print(ln, end='', flush=True)
            rc = pr.wait(); print('mã thoát:', rc, flush=True)
            if rc != 0: print(f'LỖI ở {n}: dừng hàng đợi, dán đoạn log trên cho tôi'); return False
    print(f'\nXONG sau {(time.time() - T) / 60:.1f} phút'); return True
run_queue()
"""

H_CMP = r"""import json, glob, numpy as np
def load(pat): return [json.load(open(p)) for p in sorted(glob.glob(f'runs/{pat}/final_metrics.json'))]
def metrics(f):   # run HD: dùng iou_320 (quy về lưới 320x160) để so với mốc; run 320x160 không có thì dùng iou
    k = '_320' if 'iou_320' in f else ''
    i = f['iou' + k]
    return dict(ego=i[2], adj=i[3], notours=i[1], road=f['road_iou' + k], ff=f['false_floor' + k])
def col(fs, c): return np.array([metrics(f)[c] for f in fs])
def welch(a, b):
    d = b.mean() - a.mean()
    if len(a) < 2 or len(b) < 2: return d, float('nan')
    return d, d / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
def report(title, base_pat, pat):
    A, B = load(base_pat), load(pat)
    if not B or not A: print(f'== {title}: chưa đủ run (nhóm {len(B)}, mốc {len(A)})\n'); return
    print(f'== {title}: n = {len(B)} so với {base_pat} (n = {len(A)})')
    for c in ['ego', 'adj', 'notours', 'road', 'ff']:
        a, b = col(A, c), col(B, c)
        d, t = welch(a, b)
        sa = f'{a.std(ddof=1):.4f}' if len(a) > 1 else ' n=1  '
        sb = f'{b.std(ddof=1):.4f}' if len(b) > 1 else ' n=1  '
        ts = 'n/a' if t != t else f'{t:+.1f}'
        z = f' | không có t, hiệu / độ lệch chuẩn mốc = {d / a.std(ddof=1):+.1f}' if (len(b) < 2 and len(a) > 1) else ''
        print(f'  {c:8s} mốc {a.mean():.4f} ± {sa} | nhóm {b.mean():.4f} ± {sb} | hiệu {d:+.4f} (t = {ts}){z}')
    print()
R = lambda tag: f'{TAG}_{tag}_c4aug_s*'
if BASE is None: print('NS không có mốc 320x160 tương ứng: chỉ so các nhóm với nhau')
print('######## 1) GIÁO VIÊN = trần (so với mốc FSNet 320x160 cùng dữ liệu). Nếu làn đang đi hơn mốc dưới 0,01 thì chưng cất khó giúp.')
for t in ['mitb2', 'unetr34']:
    if BASE: report(f'giáo viên {t}', BASE, R(t))
print('######## 2) HỌC TRÒ HD so với mốc 320x160')
if BASE:
    for t in ['p1s5', 'p1s5_reg', 'p1s5_kdT', 'p1s5_kdC', 'lp1s5_kdT']: report(f'{t} so với mốc', BASE, R(t))
print('######## 3) TÁCH TỪNG YẾU TỐ (cùng dữ liệu, cùng cỡ)')
report('chống quá khớp (reg) so với không reg', R('p1s5'), R('p1s5_reg'))
report('KD từ transformer so với reg không KD', R('p1s5_reg'), R('p1s5_kdT'))
report('KD từ conv so với reg không KD', R('p1s5_reg'), R('p1s5_kdC'))
report('KD transformer so với KD conv (đối chứng)', R('p1s5_kdC'), R('p1s5_kdT'))
report('học trò lớn (fsnet_l_p1s5) so với nhỏ, cùng KD transformer', R('p1s5_kdT'), R('lp1s5_kdT'))
print('######## 4) KHOẢNG CÁCH QUÁ KHỚP: làn đang đi, ảnh train nhãn đủ (không aug) trừ test, cùng lưới của run')
for t in ['p1s5', 'p1s5_reg', 'p1s5_kdT', 'p1s5_kdC', 'lp1s5_kdT', 'mitb2', 'unetr34']:
    fs = load(R(t))
    if fs:
        tr = np.array([f['iou_trainsub'][2] for f in fs]); te = np.array([f['iou'][2] for f in fs])
        print(f'  {t:12s} train {tr.mean():.4f} - test {te.mean():.4f} = {tr.mean() - te.mean():+.4f} (n = {len(fs)})')
"""

def nb(cells):
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": c.splitlines(keepends=True)} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}

for name, cells in [('10_data', [DEPLOY, DATA]), ('20_train', [DEPLOY, TRAIN]), ('30_eval', [EVAL_A, EVAL_B]), ('40_int8', [DEPLOY, INT8]),
                    ('60_queue', [DEPLOY, Q_CACHE, Q_MAC, Q_RUN, Q_CMP]),
                    ('70_hd', [DEPLOY, H_CFG, H_CACHE_P, H_CACHE_S, H_MAC, H_RUN, H_CMP])]:
    json.dump(nb(cells), open(f'notebooks/{name}.ipynb', 'w'), ensure_ascii=False, indent=1)
    print('đã tạo notebooks/' + name + '.ipynb')
    
    