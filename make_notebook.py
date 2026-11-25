"""Sinh 4 notebook Colab cho mô hình B (fsnet_m_p1s5, 640x320): 10_data, 20_train, 30_eval, 40_int8.
Sửa mã ở đây (và trong repo/, runtime/), commit, push, rồi chạy cell đầu (DEPLOY) của notebook trên Colab.
Chạy:  python make_notebook.py
"""
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
print(subprocess.run(f'git -C {CODE} log --oneline -1', shell=True, capture_output=True, text=True).stdout.strip())
print('mã mới nhất đã ở capstone_ai/repo/ (runtime/, config/, tests/ chạy thẳng từ /content/code) | thư mục làm việc:', os.getcwd())
"""

READY = r"""def ready(W=640, H=320):
    # shard HD đã dựng xong (có đủ _x.npy, _y.npy, _ids.csv) trong data/v2/b9_chunks_{W}x{H}/
    out = []
    for p in glob.glob(f'data/v2/b9_chunks*{W}x{H}*/shard_*_ids.csv'):
        s, d = os.path.basename(p)[:-8], os.path.dirname(p)
        if os.path.exists(f'{d}/{s}_x.npy') and os.path.exists(f'{d}/{s}_y.npy'): out.append(s)
    return sorted(out)
"""

# ---------------------------------------------------------------- 10_data
D_STATUS = r"""import os, glob, subprocess
os.chdir('/content/drive/MyDrive/capstone_ai')
W, H = 640, 320
""" + READY + r"""
for f in ['data/bdd100k_seg.tar', 'data/v2/drv_masks_3274.tar', 'data/v2/pilot_drv_manifest.csv', 'data/v2/manifest_v2_in100k.csv',
          'data/v2/pilot_drv_x_320x160.npy', 'data/v2/pilot_drv_y_320x160.npy', f'data/v2/pilot_drv_x_{W}x{H}.npy', f'data/v2/pilot_drv_y_{W}x{H}.npy']:
    print(f'{f:45s}', f'{os.path.getsize(f) / 1e6:9.0f} MB' if os.path.exists(f) else 'KHÔNG CÓ')
print(f'shard {W}x{H} đã xong:', ready(W, H) or 'chưa có')
print('tiến trình tải shard đang chạy:', subprocess.run('pgrep -af "[b]9_prepare_v2.py"', shell=True, capture_output=True, text=True).stdout.strip() or 'không')
print(subprocess.run('df -h /content /content/drive | tail -2', shell=True, capture_output=True, text=True).stdout)
"""

D_CACHE = r"""import os, shutil, subprocess, sys, numpy as np
os.chdir('/content/drive/MyDrive/capstone_ai')
W, H = 640, 320
# Chạy được trên runtime CPU. Cache thí điểm ở 640x320 (cần cho huấn luyện, cho shard HD và cho int8/eval).
# 1) sao lưu cache 320x160 một lần (để kiểm nó không bị đụng tới)
os.makedirs('data/v2/backup_pilot_320', exist_ok=True)
for f in ['pilot_drv_x_320x160.npy', 'pilot_drv_y_320x160.npy', 'pilot_drv_manifest.csv']:
    dst = f'data/v2/backup_pilot_320/{f}'
    if not os.path.exists(dst): shutil.copyfile(f'data/v2/{f}', dst)
# 2) kiểm chứng trước: dựng lại 320x160 và so từng pixel với cache có sẵn; chỉ dựng 640x320 khi ĐẠT
if os.path.exists(f'data/v2/pilot_drv_x_{W}x{H}.npy') and os.path.exists(f'data/v2/pilot_drv_y_{W}x{H}.npy'):
    print(f'đã có cache thí điểm {W}x{H}, bỏ qua (xóa hai tệp pilot_drv_*_{W}x{H}.npy trên Drive nếu muốn dựng lại)')
else:
    for w, h in [(320, 160), (W, H)]:
        r = subprocess.run([sys.executable, '-u', 'repo/make_hd_pilot.py', '--w', str(w), '--h', str(h)])
        if r.returncode != 0: raise SystemExit(f'make_hd_pilot {w}x{h} lỗi (mã {r.returncode}): dừng, dán đoạn trên cho tôi')
# 3) kiểm sau khi dựng
for f in ['pilot_drv_x_320x160.npy', 'pilot_drv_y_320x160.npy']:
    a = np.load(f'data/v2/{f}', mmap_mode='r'); b = np.load(f'data/v2/backup_pilot_320/{f}', mmap_mode='r')
    print(f, 'cache 320x160 còn nguyên:', a.shape == b.shape and all(np.array_equal(a[i:i + 256], b[i:i + 256]) for i in range(0, len(a), 256)))
for f in [f'pilot_drv_x_{W}x{H}.npy', f'pilot_drv_y_{W}x{H}.npy']:
    a = np.load(f'data/v2/{f}', mmap_mode='r'); print(f, a.shape, a.dtype, f'{os.path.getsize(f"data/v2/{f}") / 1e6:.0f} MB')
v = np.unique(np.asarray(np.load(f'data/v2/pilot_drv_y_{W}x{H}.npy', mmap_mode='r')[:200]))
print('giá trị nhãn trong 200 ảnh đầu:', v.tolist(), '(mong đợi: tập con của 0, 1, 2, 3, 255)')
"""

D_SPOT = r"""# (tùy chọn) kiểm độc lập 24 ảnh: dựng lại từ tar bằng mã viết riêng ở đây, so với cache 640x320
import os, io, shutil, tarfile, numpy as np, pandas as pd
from PIL import Image
os.chdir('/content/drive/MyDrive/capstone_ai')
W, H = 640, 320
os.makedirs('/content/raw', exist_ok=True)
LOC = '/content/raw/bdd100k_seg.tar'
if not os.path.exists(LOC): shutil.copy('data/bdd100k_seg.tar', LOC)
tf = tarfile.open(LOC); mem = {m.name: m for m in tf.getmembers() if m.isfile()}
J = {os.path.basename(n)[:-4]: n for n in mem if n.endswith('.jpg') and '/images_10k/' in n}
S = {os.path.basename(n)[:-13]: n for n in mem if n.endswith('_train_id.png')}
dt = tarfile.open('data/v2/drv_masks_3274.tar'); D = {os.path.basename(m.name)[:-4]: m for m in dt.getmembers() if m.isfile()}
man = pd.read_csv('data/v2/pilot_drv_manifest.csv')
X = np.load(f'data/v2/pilot_drv_x_{W}x{H}.npy', mmap_mode='r'); Y = np.load(f'data/v2/pilot_drv_y_{W}x{H}.npy', mmap_mode='r')
assert len(man) == len(X) == len(Y), (len(man), len(X), len(Y))
CROP = (0, 0, 1280, 600)
idx = np.sort(np.random.default_rng(0).choice(len(man), 24, replace=False)); bx = by = 0
for k in idx:
    i = man.id[k]
    im = Image.open(io.BytesIO(tf.extractfile(mem[J[i]]).read())).convert('RGB')
    s = np.asarray(Image.open(io.BytesIO(tf.extractfile(mem[S[i]]).read())).crop(CROP).resize((W, H), Image.NEAREST))
    d = np.asarray(Image.open(io.BytesIO(dt.extractfile(D[i]).read())).crop(CROP).resize((W, H), Image.NEAREST))
    road, ign = s == 0, s == 255
    lab = road.astype(np.uint8); lab[d == 1] = 2; lab[d == 2] = 3
    lab[((d == 1) | (d == 2)) & ~road & ~ign] = 255; lab[ign] = 255
    bx += int((np.asarray(im.crop(CROP).resize((W, H), Image.BILINEAR)) != np.asarray(X[k])).sum()); by += int((lab != np.asarray(Y[k])).sum())
print(f'{len(idx)} ảnh | pixel ảnh lệch: {bx} | pixel nhãn lệch: {by}')
print('ĐẠT' if bx == 0 and by == 0 else 'LỆCH: dừng, dán kết quả cho tôi')
"""

D_SHARDS = r"""import os, sys, glob, subprocess
from google.colab import userdata
os.chdir('/content/drive/MyDrive/capstone_ai'); os.makedirs('runs', exist_ok=True)
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
W, H = 640, 320
N_SHARDS = 4          # số shard 640x320 cần có (mỗi shard khoảng 6 GB trên Drive); shard đã xong được bỏ qua
FOREGROUND = False    # False: chạy nền (cell trả về ngay, theo dõi bằng cell dưới); True: chạy và in log tại đây
""" + READY + r"""
LOG = f'runs/b9_prepare_{W}x{H}.log'
assert os.path.exists(f'data/v2/pilot_drv_y_{W}x{H}.npy'), 'cần cache thí điểm 640x320 (cell trên) trước khi dựng shard HD'
running = subprocess.run('pgrep -af "[b]9_prepare_v2.py"', shell=True, capture_output=True, text=True).stdout.strip()
cmd = [sys.executable, '-u', 'repo/b9_prepare_v2.py', '--shards', str(N_SHARDS), '--w', str(W), '--h', str(H)]
if len(ready(W, H)) >= N_SHARDS: print(f'đã đủ {N_SHARDS} shard:', ready(W, H))
elif running: print('đang có tiến trình tải chạy nền, không chạy thêm:', running)
elif FOREGROUND:
    with open(LOG, 'a') as lg:
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for ln in pr.stdout: lg.write(ln); print(ln, end='', flush=True)
    if pr.wait() != 0: raise SystemExit(f'b9_prepare_v2 lỗi: dán đoạn log trên cho tôi')
else:
    DL = subprocess.Popen(cmd, stdout=open(LOG, 'a'), stderr=subprocess.STDOUT)
    print(f'đã chạy nền tải và dựng shard {W}x{H} (pid {DL.pid}, log {LOG}). Phiên Colab ngắt thì chạy lại cell này: shard đã xong được bỏ qua.')
"""

D_WATCH = r"""import os, glob, subprocess
os.chdir('/content/drive/MyDrive/capstone_ai')
W, H = 640, 320
""" + READY + r"""
print('tiến trình:', subprocess.run('pgrep -af "[b]9_prepare_v2.py"', shell=True, capture_output=True, text=True).stdout.strip() or 'không chạy')
print(f'shard {W}x{H} đã xong:', ready(W, H) or 'chưa có')
print(subprocess.run(f'tail -5 runs/b9_prepare_{W}x{H}.log', shell=True, capture_output=True, text=True).stdout)
print(subprocess.run('df -h /content /content/drive | tail -2', shell=True, capture_output=True, text=True).stdout)
"""

# ---------------------------------------------------------------- 20_train
T_CLEAN = r"""import os, subprocess, torch
os.chdir('/content/drive/MyDrive/capstone_ai')
print('tiến trình train đang chạy:', subprocess.run('pgrep -af "[t]rain_v3.py"', shell=True, capture_output=True, text=True).stdout.strip() or 'không')
subprocess.run('pkill -f "[t]rain_v3.py"', shell=True)
print('đã dừng tiến trình thừa (nếu có)')
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ (đổi runtime sang GPU)')
print(subprocess.run('df -h /content | tail -1', shell=True, capture_output=True, text=True).stdout.strip())
"""

T_CFG = r"""import os, sys, glob, json
os.chdir('/content/drive/MyDrive/capstone_ai')
# CẤU HÌNH: chỉ huấn luyện B (fsnet_m_p1s5, 640x320). Đặt tiêu chí quyết định TRƯỚC khi chạy (xem cell so sánh).
N_SHARDS = 4             # số shard 640x320 (cần đã dựng ở 10_data); 0 = chỉ bộ thí điểm
SEEDS = [0, 1, 2]
KD = True                # True: chưng cất từ giáo viên TEACHER (tag _kdT); False: chỉ gói chống quá khớp (tag _reg)
EPOCHS = 60
BUDGET_MIN = None        # số phút: không bắt đầu run mới sau mức này (phiên Colab có giới hạn); None = không giới hạn
W, H, ARCH = 640, 320, 'fsnet_m_p1s5'
TEACHER = 'hd640s0_mitb2_c4aug_s0'
BASE = 'hd640s1_p1s5_kdT_c4aug_s*'       # B hiện tại (3 seed), mốc so sánh
TAG = f'hd{W}s{N_SHARDS}_p1s5_' + ('kdT' if KD else 'reg')
REG = ['--ema', '0.999', '--wd', '0.01', '--wd_nobn', '1', '--ls', '0.05', '--cutout', '2']   # gói chống quá khớp, như B
KDA = ['--kd', '1.0', '--kd_T', '2.0', '--teacher', TEACHER]
""" + READY + r"""
sh = ready(W, H)
need = [f'shard_{i:03d}' for i in range(N_SHARDS)]
miss = [s for s in need if s not in sh]
if miss: raise SystemExit(f'thiếu shard {W}x{H}: {miss} (đang có {sh}). Chạy 10_data trước, hoặc giảm N_SHARDS.')
if KD and not (os.path.exists(f'runs/{TEACHER}/final_metrics.json') and os.path.exists(f'runs/{TEACHER}/last.pt')):
    raise SystemExit(f'KD = True cần giáo viên runs/{TEACHER}/ đã huấn luyện xong (last.pt, final_metrics.json)')
sys.path.insert(0, 'repo'); import models_custom as mc
net = mc.build_fsnet(ARCH, num_classes=4); bad = mc.contract_violations(net)
assert not bad, f'{ARCH} vi phạm hợp đồng toán tử: {bad}'
print(f'{ARCH} {W}x{H}: {sum(r["mac"] for r in mc.layer_table(net, H, W)) / 1e9:.3f} GMAC | hợp đồng toán tử: đạt')
EXTRA = ['--arch', ARCH, '--w', str(W), '--h', str(H)] + (['--b9', ','.join(need)] if N_SHARDS else []) + REG + (KDA if KD else [])
print('run:', [f'{TAG}_c4aug_s{s}' for s in SEEDS], '| mốc so sánh:', BASE)
print('tham số train_v3.py:', ' '.join(EXTRA))
"""

T_RUN = r"""import os, re, sys, time, subprocess
import torch; assert torch.cuda.is_available(), 'Chưa có GPU: đổi runtime sang GPU'
os.chdir('/content/drive/MyDrive/capstone_ai'); os.makedirs('runs', exist_ok=True)
T0 = time.time()
for s in SEEDS:
    n = f'{TAG}_c4aug_s{s}'
    if os.path.exists(f'runs/{n}/final_metrics.json'): print(n, 'đã xong, bỏ qua'); continue
    if BUDGET_MIN and (time.time() - T0) / 60 > BUDGET_MIN: print(f'quá {BUDGET_MIN} phút: không bắt đầu {n}, chạy lại cell sau'); break
    resume = os.path.exists(f'runs/{n}/last.pt')     # train_v3.py tự tiếp tục từ last.pt (cùng tên run)
    print(f'\n>>> {n}' + (' (tiếp tục từ last.pt)' if resume else '') + f'  [{(time.time() - T0) / 60:.0f} phút]', flush=True)
    cmd = [sys.executable, '-u', 'repo/train_v3.py', '--name', n, '--seed', str(s), '--epochs', str(EPOCHS)] + EXTRA
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(f'runs/{n}.log', 'a') as lg:
        for ln in pr.stdout:
            lg.write(ln); m = re.match(r'ep (\d+)/', ln)
            if not m or int(m.group(1)) in (1, 2) or int(m.group(1)) % 10 == 0: print(ln, end='', flush=True)
    rc = pr.wait(); print('mã thoát:', rc, flush=True)
    if rc != 0: print(f'LỖI ở {n}: dừng, dán đoạn log trên cho tôi'); break
    if not os.path.exists(f'runs/{n}/final_metrics.json'): print(f'LỖI ở {n}: mã thoát 0 nhưng không có final_metrics.json, dán log cho tôi'); break
else: print(f'\nXONG sau {(time.time() - T0) / 60:.0f} phút')
"""

T_CMP = r"""import os, json, glob, numpy as np
os.chdir('/content/drive/MyDrive/capstone_ai')
# Tiêu chí đặt trước (CLAUDE.md): nhận mô hình mới thay B chỉ khi làn đang đi tăng >= 0,01 so với B với t >= 3 (Welch),
# IoU đường không giảm, false_floor không tăng (so trung bình), và TV2 xác nhận ngân sách. Số ở epoch cuối, lưới 320x160 (iou_320).
def load(pat): return [json.load(open(p)) for p in sorted(glob.glob(f'runs/{pat}/final_metrics.json'))]
def met(f):
    k = '_320' if 'iou_320' in f else ''
    i = f['iou' + k]
    return dict(ego=i[2], adj=i[3], notours=i[1], road=f['road_iou' + k], ff=f['false_floor' + k])
def welch(a, b):
    d = b.mean() - a.mean()
    return d, (d / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 1 and len(b) > 1 else float('nan'))
NEW = f'{TAG}_c4aug_s*'
A, B = load(BASE), load(NEW)
print(f'mốc B hiện tại {BASE}: n = {len(A)} | nhóm mới {NEW}: n = {len(B)}')
if NEW == BASE: print('nhóm mới trùng mốc (N_SHARDS = 1, KD = True): không có gì để so')
elif not A or not B: print('chưa đủ run để so')
else:
    r = {}
    for c, nm in [('ego', 'làn đang đi'), ('adj', 'làn kề'), ('notours', 'không thuộc ta'), ('road', 'IoU đường'), ('ff', 'false_floor (thấp là tốt)')]:
        a, b = np.array([met(f)[c] for f in A]), np.array([met(f)[c] for f in B])
        d, t = welch(a, b); r[c] = (d, t)
        sa = f'± {a.std(ddof=1):.4f}' if len(a) > 1 else '(n=1)'; sb = f'± {b.std(ddof=1):.4f}' if len(b) > 1 else '(n=1)'
        print(f'  {nm:26s} B {a.mean():.4f} {sa} | mới {b.mean():.4f} {sb} | hiệu {d:+.4f} (t = {t:+.1f})')
    ok_ego = r['ego'][0] >= 0.01 and r['ego'][1] >= 3
    ok_road, ok_ff = r['road'][0] >= 0, r['ff'][0] <= 0
    print(f'\ntiêu chí đặt trước: làn đang đi +0,01 và t >= 3: {"ĐẠT" if ok_ego else "CHƯA"} | IoU đường không giảm: {"ĐẠT" if ok_road else "CHƯA"} | '
          f'false_floor không tăng: {"ĐẠT" if ok_ff else "CHƯA"} | TV2 xác nhận ngân sách: (kiểm tay)')
    print('=> ' + ('ĐẠT phần đo được: đóng băng (cell dưới), lượng tử (40_int8) rồi đánh giá runtime (30_eval) trước khi thay B'
                   if ok_ego and ok_road and ok_ff else 'giữ B hiện tại'))
print('\nkhoảng cách train - test ở làn đang đi (ảnh train nhãn đủ, không aug; lưới của run):')
for nm, fs in [('B', A), ('mới', B)]:
    if fs and all('iou_trainsub' in f for f in fs):
        tr = np.array([f['iou_trainsub'][2] for f in fs]); te = np.array([f['iou'][2] for f in fs])
        print(f'  {nm:4s} train {tr.mean():.4f} - test {te.mean():.4f} = {tr.mean() - te.mean():+.4f} (n = {len(fs)})')
"""

T_FREEZE = r"""import os, json, glob, shutil, hashlib, zipfile, numpy as np, torch
os.chdir('/content/drive/MyDrive/capstone_ai')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
os.makedirs('models_final', exist_ok=True); os.makedirs('results', exist_ok=True)
# 1) đóng băng: chỉ lấy last.pt["model"] (trọng số đánh giá và triển khai, bản EMA nếu có). Tên đã có thì KHÔNG ghi đè.
FREEZE = {'B2_hd640s4_p1s5_kdT_s0': 'runs/hd640s4_p1s5_kdT_c4aug_s0'}     # s0 chọn trước, không chọn seed tốt nhất; đã có thì không ghi đè; {} = chỉ làm mới bảng và context_pack
info = json.load(open('models_final/MODELS.json')) if os.path.exists('models_final/MODELS.json') else {}
for name, d in FREEZE.items():
    out = f'models_final/{name}_weights.pt'
    if name in info or os.path.exists(out): print(f'{name}: đã có, không ghi đè (đổi tên nếu muốn đóng băng bản khác)'); continue
    if not os.path.exists(f'{d}/final_metrics.json'): print(f'{name}: {d} chưa huấn luyện xong (thiếu final_metrics.json), bỏ qua'); continue
    ck = torch.load(f'{d}/last.pt', map_location='cpu', weights_only=False)
    torch.save(ck['model'], out)
    for f in ['config.json', 'final_metrics.json']: shutil.copy(f'{d}/{f}', f'models_final/{name}_{f}')
    info[name] = dict(run=d, epoch=int(ck.get('epoch', -1)), bytes=os.path.getsize(out), sha256=sha(out))
    print(f'{name}: epoch {info[name]["epoch"]} | {info[name]["bytes"] / 1e6:.1f} MB | sha256 {info[name]["sha256"][:16]}...')
json.dump(info, open('models_final/MODELS.json', 'w'), indent=1)
print('models_final/MODELS.json:', list(info))

# 2) bảng kết quả (epoch cuối, lưới 320x160, 408 ảnh test)
def L(pat):
    r = []
    for p in sorted(glob.glob(f'runs/{pat}/final_metrics.json')):
        f = json.load(open(p)); i = f.get('iou_320', f['iou']); r.append((i[2], i[3], i[1], f.get('road_iou_320', f['road_iou'])))
    return np.array(r)
G = [('Mốc thí điểm 2.826 ảnh, aug mạnh, 320x160', 'pilot_c4aug_s*'), ('+1 shard, 320x160', 'b9s1_c4aug_s*'),
     ('A: +4 shard, 320x160 (dự phòng, ngừng phát triển)', 'b9s4_c4aug_s*'), ('HD 640x320 p1s5_reg, thí điểm', 'hd640s0_p1s5_reg_c4aug_s*'),
     ('Teacher MiT-B2, HD, thí điểm', 'hd640s0_mitb2_c4aug_s0'), ('HD 640x320 p1s5_reg, +1 shard', 'hd640s1_p1s5_reg_c4aug_s*'),
     ('B: HD 640x320 p1s5 + chưng cất, +1 shard (mô hình chính)', 'hd640s1_p1s5_kdT_c4aug_s*')]
for p in sorted({os.path.basename(x)[:-len('_c4aug_s0')] for x in glob.glob('runs/hd640s*_p1s5_*_c4aug_s*') if os.path.basename(x)[-1].isdigit()}):
    n = p.split('_')[0][len('hd640s'):]
    if p not in ('hd640s0_p1s5_reg', 'hd640s1_p1s5_reg', 'hd640s1_p1s5_kdT') and n.isdigit():
        G.append((f'HD 640x320 {p[len("hd640s") + len(n) + 1:]}, +{n} shard', f'{p}_c4aug_s*'))
lines, res = ['| Nhóm | n | Làn đang đi | Làn kề | Không thuộc ta | IoU đường |', '|---|---|---|---|---|---|'], {}
for title, pat in G:
    a = L(pat)
    if not len(a): continue
    cells = [f'{a[:, j].mean():.4f}' + (f' ± {a[:, j].std(ddof=1):.4f}' if len(a) > 1 else '') for j in range(4)]
    lines.append(f'| {title} | {len(a)} | ' + ' | '.join(cells) + ' |'); res[title] = dict(n=len(a), mean=a.mean(0).tolist(), pattern=pat)
md = '# Kết quả huấn luyện (epoch cuối, lưới 320x160, 408 ảnh test, GPU L4)\n\n' + '\n'.join(lines) + '\n'
open('results/final_results.md', 'w', encoding='utf-8').write(md); json.dump(res, open('results/final_results.json', 'w'), indent=1, ensure_ascii=False)
print('\n' + md)

# 3) làm mới gói ngữ cảnh cho Claude Code (chỉ tệp chữ nhỏ)
HB = 'handoff_s3b_fsnet_m_p1s5_640x320'
HB2 = HB + '_s4'
pats = ['runs/*/config.json', 'runs/*/final_metrics.json', 'runs/*/diag_last.json', 'runs/*/metrics.csv', 'runs/summary.csv', 'eval_report/*',
        'data_spec_v1.json', 'data/cache_v1/cache_spec.json', 'data/v2/manifest_v2.csv', 'data/v2/split_report.json', 'data/v2/pilot_drv_manifest.csv',
        'handoff_s3a_v2/README.md', 'handoff_s3a_v2/preprocess_spec.json', 'handoff_s3a_v2/layer_table.csv', 'handoff_s3b/README.md',
        'handoff_s3b/int8_meta.json', 'handoff_s3b/int8_sim_summary.json', 'handoff_s3b/export_info.json', 'handoff_s3b/layer_table.csv',
        'handoff_s3b/int8_layers_pct.csv'] + [f'{HB}/{f}' for f in ['README.md', 'int8_meta.json', 'int8_sim_summary.json', 'export_info.json',
        'layer_table.csv', 'int8_layers_pct.csv', 'int8_layers_max.csv', 'preprocess_spec.json', 'MANIFEST.json']] + \
       [f'{HB2}/{f}' for f in ['README.md', 'int8_meta.json', 'int8_sim_summary.json', 'export_info.json',
        'layer_table.csv', 'int8_layers_pct.csv', 'int8_layers_max.csv', 'preprocess_spec.json', 'MANIFEST.json']] + \
       ['eval_runtime/*.json', 'results/*', 'models_final/MODELS.json', 'models_final/*_final_metrics.json', 'models_final/*_config.json']
shutil.rmtree('context_pack', ignore_errors=True); os.makedirs('context_pack'); n = 0
for pat in pats:
    for p in glob.glob(pat):
        if os.path.isfile(p) and os.path.getsize(p) < 3_000_000:
            dst = os.path.join('context_pack', p); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(p, dst); n += 1
with zipfile.ZipFile('context_pack.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for d, _, fs in os.walk('context_pack'):
        for f in fs: z.write(os.path.join(d, f), os.path.relpath(os.path.join(d, f), 'context_pack'))
print(n, 'tệp | context_pack.zip', round(os.path.getsize('context_pack.zip') / 1e6, 1), 'MB')
"""

# ---------------------------------------------------------------- 30_eval
E_CFG = r"""import os, sys, subprocess
os.chdir('/content/drive/MyDrive/capstone_ai')
ROOT, CODE = '/content/drive/MyDrive/capstone_ai', '/content/code'
PACK = 'handoff_s3b_fsnet_m_p1s5_640x320_s4'  # gói INT8 của B2 (tạo bằng 40_int8). Đánh giá B cũ: PACK = '..._640x320', CFG = 'pipeline_b.json'
CFG = 'pipeline_b2.json'
N = 0                                           # 0 = cả 408 ảnh test (int8ref khoảng 1 đến 3 s mỗi ảnh trên CPU Colab: chưa đo)
BACKENDS = 'int8ref,onnx'
for f in [f'{PACK}/int8_meta.json', f'{PACK}/int8_model.npz', f'{PACK}/int8_ref.py', f'{PACK}/int8_sim_summary.json', f'{PACK}/fsnet_m_p1s5_c4_640x320.onnx',
          'data/bdd100k_seg.tar', 'data/v2/pilot_drv_x_640x320.npy', 'data/v2/pilot_drv_y_640x320.npy', 'data/v2/pilot_drv_y_320x160.npy', 'data/v2/hd_test_640x320/test_list.csv']:
    print(f'{f:55s}', 'có' if os.path.exists(f) else 'THIẾU')
try: import onnxruntime
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnxruntime'], check=True)
ENV = {**os.environ, 'PYTHONPATH': CODE, 'CAPSTONE_ROOT': ROOT, 'PERCEPTION_PACK': f'{ROOT}/{PACK}', 'PERCEPTION_TEST_DIR': f'{ROOT}/data/v2/hd_test_640x320'}
def run(args, log):
    print('>>>', ' '.join(args), flush=True)
    pr = subprocess.Popen([sys.executable, '-u'] + args, cwd=CODE, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(f'{ROOT}/runs/{log}.log', 'a') as lg:
        for ln in pr.stdout: lg.write(ln); print(ln, end='', flush=True)
    return pr.wait()
os.makedirs('runs', exist_ok=True)
"""

E_TESTS = r"""# 1) kiểm đơn vị của runtime (tầng 2 trên mặt nạ dựng tay, gói nhị phân, ma trận tọa độ, tiền xử lý so với PIL, bilinear so với torch)
#    và với gói B thật: int8ref trùng golden từng bit, lượng tử ảnh test trùng golden, pipeline chạy trọn hai backend.
rc = run(['-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'runtime_tests')
print('KẾT QUẢ kiểm đơn vị:', 'ĐẠT' if rc == 0 else f'KHÔNG ĐẠT (mã {rc}): dán đoạn trên cho tôi')
"""

E_RUN = r"""# 2) runtime trên 408 ảnh test: ảnh gốc 1280x720 trong tar -> runtime.pipeline -> mặt nạ, IoU ở lưới 320x160.
#    Phải khớp int8_sim trong +-0,001 (int8ref với "pct", onnx với "fp32" trong int8_sim_summary.json của gói); in thời gian từng khâu.
#    Thời gian ở đây là CPU Colab, KHÔNG phải A53 của board.
args = ['-m', 'runtime.eval_dataset', '--root', ROOT, '--config', f'{CODE}/config/{CFG}', '--backends', BACKENDS, '--tol', '0.001']
if N: args += ['--n', str(N)]
rc = run(args, 'runtime_eval')
if rc != 0: print(f'KHÔNG ĐẠT (mã {rc}): dán đoạn trên cho tôi')
"""

E_VIS = r"""# 3) (tùy chọn) vẽ kết quả lên 4 ảnh test để kiểm bằng mắt: màu lớp, chấm đỏ = ranh giới khoảng trống, chữ thập = mép và tâm làn
import io, csv, numpy as np
sys.path.insert(0, CODE); sys.path.insert(0, f'{ROOT}/repo')
from PIL import Image
from IPython.display import display
from runtime.pipeline import Pipeline
from runtime.viz import overlay
import make_hd_pilot as mhp
pl = Pipeline(f'{CODE}/config/{CFG}', ROOT, backend='onnx')
man = list(csv.DictReader(open('data/v2/pilot_drv_manifest.csv'))); te = [r for r in man if r['split_v2'] == 'test']
tar = '/content/hd_scratch/bdd100k_seg.tar' if os.path.exists('/content/hd_scratch/bdd100k_seg.tar') else 'data/bdd100k_seg.tar'
imgs, _ = mhp.index_tar(tar); fh = open(tar, 'rb'); os.makedirs('eval_runtime', exist_ok=True)
for k in [0, 100, 200, 300]:
    off, size = imgs[(te[k]['v1_split'], te[k]['id'])]; fh.seek(off)
    frame = np.asarray(Image.open(io.BytesIO(fh.read(size))).convert('RGB'))
    r = pl.process(frame, keep=True)
    im = Image.fromarray(overlay(frame, r)); im.save(f'eval_runtime/vis_test_{k:03d}.png'); display(im.resize((640, 360)))
"""

# ---------------------------------------------------------------- 40_int8
INT8 = r"""import os, subprocess, sys
os.chdir('/content/drive/MyDrive/capstone_ai'); os.makedirs('runs', exist_ok=True)
# CẤU HÌNH. Mặc định B2 (ứng viên thay B): fsnet_m_p1s5 640x320, 4 shard -> handoff_s3b_fsnet_m_p1s5_640x320_s4.
# B (đang là mô hình chính, giữ làm dự phòng): RUN = 'runs/hd640s1_p1s5_kdT_c4aug_s0', OUT = 'handoff_s3b_fsnet_m_p1s5_640x320'. KHÔNG đặt OUT trùng gói B.
W, H, ARCH = 640, 320, 'fsnet_m_p1s5'
RUN = 'runs/hd640s4_p1s5_kdT_c4aug_s0'     # last.pt["model"] là trọng số triển khai (bản EMA nếu run bật EMA); trùng tệp trong models_final/
OUT = 'handoff_s3b' if (W, H) == (320, 160) else f'handoff_s3b_{ARCH}_{W}x{H}_s4'   # gói HD ghi ra thư mục khác, không đè gói B
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
print(f'XONG: gói {OUT}/ (và {OUT}_v1.zip). Bước sau: 30_eval (runtime trên ảnh test với gói này).')
"""


def nb(cells):
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": c.splitlines(keepends=True)} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


NOTEBOOKS = [('10_data', [DEPLOY, D_STATUS, D_CACHE, D_SPOT, D_SHARDS, D_WATCH]),
             ('20_train', [DEPLOY, T_CLEAN, T_CFG, T_RUN, T_CMP, T_FREEZE]),
             ('30_eval', [DEPLOY, E_CFG, E_TESTS, E_RUN, E_VIS]),
             ('40_int8', [DEPLOY, INT8])]

if __name__ == '__main__':
    for name, cells in NOTEBOOKS:
        json.dump(nb(cells), open(f'notebooks/{name}.ipynb', 'w'), ensure_ascii=False, indent=1)
        print('đã tạo notebooks/' + name + '.ipynb')
