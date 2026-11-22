"""4 lớp: dữ liệu thí điểm + shard b9 (nhãn 254 = không đường HOẶC đường không thuộc ta). Đánh giá trên 408 ảnh test thí điểm.

Mô hình: FSNet (--arch fsnet_*) hoặc giáo viên smp (--arch smp:Unet:mit_b2, smp:Unet:mobilenet_v2; cần w, h chia hết cho 32).
Mọi tính năng thêm đều MẶC ĐỊNH TẮT, nên run cũ chạy lại cho cùng hành vi:
  --w --h          cỡ cache (pilot_drv_*_{w}x{h}.npy; shard b9 ở b9_chunks_{w}x{h} khi cỡ khác 320x160)
  --full_w         lấy mẫu có trọng số cho ảnh nhãn đủ
  --teacher --kd   chưng cất từ giáo viên (cùng w x h); KD tính cả trên pixel nhãn 254
  --ema --wd --wd_nobn --ls --cutout   chống quá khớp (xem README các cờ dưới đây)
  --lr_enc_mult    lr của encoder = lr * hệ số (dùng khi tinh chỉnh encoder tiền huấn luyện)
  --accum          cộng dồn gradient (cỡ lô hiệu dụng = bs * accum)
  --stream         đọc lô từ đĩa thay vì nạp hết lên GPU (tự bật khi dữ liệu > 6 GB)
last.pt["model"] luôn là trọng số dùng để đánh giá/triển khai (bản EMA nếu bật --ema); bản gốc để tiếp tục huấn luyện nằm ở ["raw"].
"""
import argparse, json, os, time, math, shutil, sys, copy, threading, queue
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
DRV = os.environ.get("CAPSTONE_ROOT", "/content/drive/MyDrive/capstone_ai"); V2 = f"{DRV}/data/v2"
LOC = os.environ.get("CAPSTONE_LOC", "/content/v3_cache"); dev = os.environ.get("CAPSTONE_DEV", "cuda"); AMP = dev == "cuda"
torch.backends.cudnn.benchmark = True
p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--arch", default="fsnet_m")
p.add_argument("--epochs", type=int, default=60); p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--dice", type=float, default=0.5)
p.add_argument("--seed", type=int, default=0); p.add_argument("--aug", default="strong")
p.add_argument("--b9", default="", help="ví dụ shard_000,shard_001")
p.add_argument("--w", type=int, default=320); p.add_argument("--h", type=int, default=160)
p.add_argument("--full_w", type=float, default=1.0,
               help="khác 1: lấy mẫu có trọng số 1 + (full_w - 1) * is_full (is_full = ảnh có nhãn đủ 4 lớp)")
p.add_argument("--pretrained", type=int, default=1, help="arch smp:*: nạp trọng số ImageNet cho encoder")
p.add_argument("--teacher", default="", help="tên run giáo viên (runs/<tên>/last.pt, cùng w x h)")
p.add_argument("--kd", type=float, default=0.0); p.add_argument("--kd_T", type=float, default=2.0)
p.add_argument("--ema", type=float, default=0.0, help="hệ số EMA của trọng số (0 = tắt), ví dụ 0.999")
p.add_argument("--wd", type=float, default=1e-4, help="weight decay của AdamW")
p.add_argument("--wd_nobn", type=int, default=0, help="1: không phân rã cho bias và BN (chỉ phân rã trọng số conv)")
p.add_argument("--lr_enc_mult", type=float, default=1.0)
p.add_argument("--ls", type=float, default=0.0, help="label smoothing của CE")
p.add_argument("--cutout", type=int, default=0, help="số ô che mỗi ảnh (nhãn giữ nguyên, xác suất 0,5 mỗi ô)")
p.add_argument("--accum", type=int, default=1)
p.add_argument("--stream", type=int, default=-1, help="1 = đọc lô từ đĩa; 0 = nạp hết lên GPU; -1 = tự chọn")
p.add_argument("--gap_n", type=int, default=408, help="số ảnh train nhãn đủ (không aug) để đo khoảng cách train - test")
a = p.parse_args(); C = 4
sys.path.insert(0, f"{DRV}/repo"); import models_custom, aug_strong
SMP = a.arch.startswith("smp:")
if SMP: _need = 32
else:
    _pc = models_custom.PRESETS[a.arch]; _need = 2 ** (4 + _pc.get("pre", 0) + (1 if _pc.get("e5") else 0))
assert a.w % _need == 0 and a.h % _need == 0, f"{a.arch} cần w, h chia hết cho {_need}, nhận {a.w}x{a.h}"
assert a.kd == 0 or a.teacher, "--kd > 0 cần --teacher"
SZ = f"{a.w}x{a.h}"; HD = (a.w, a.h) != (320, 160)
EB = max(1, min(64, int(3.3e6 // (a.w * a.h))))     # cỡ lô khi đánh giá (320x160 -> 64)
RUN = f"{DRV}/runs/{a.name}"; os.makedirs(RUN, exist_ok=True)
for f_ in (__file__, f"{DRV}/repo/models_custom.py"):
    d_ = f"{RUN}/code_{os.path.basename(f_)}"
    if not os.path.exists(d_): shutil.copyfile(f_, d_)
if not os.path.exists(f"{RUN}/config.json"): json.dump(vars(a), open(f"{RUN}/config.json", "w"), indent=2)
os.makedirs(LOC, exist_ok=True)
def cp(src):
    dst = f"{LOC}/{os.path.relpath(src, DRV).replace(os.sep, '__')}"      # giữ tên thư mục để b9_chunks và b9_chunks_640x320 không đè nhau
    if not os.path.exists(dst): shutil.copyfile(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    return dst

# ---------------- dữ liệu: danh sách nguồn (x, y, hàng được dùng) + chỉ số toàn cục ----------------
man = pd.read_csv(cp(f"{V2}/pilot_drv_manifest.csv"))
for _f in (f"{V2}/pilot_drv_x_{SZ}.npy", f"{V2}/pilot_drv_y_{SZ}.npy"):
    if not os.path.exists(_f): raise SystemExit(f"chưa có {_f}: dựng bằng repo/make_hd_pilot.py --w {a.w} --h {a.h}")
PX = np.load(cp(f"{V2}/pilot_drv_x_{SZ}.npy"), mmap_mode="r"); PY = np.load(cp(f"{V2}/pilot_drv_y_{SZ}.npy"), mmap_mode="r")
assert len(PX) == len(PY) == len(man), f"cache {SZ} lệch manifest: {len(PX)}, {len(PY)}, {len(man)}"
tr = np.where(man.split_v2.values == "train")[0]; te = np.where(man.split_v2.values == "test")[0]
CH = f"{V2}/b9_chunks" if not HD else f"{V2}/b9_chunks_{SZ}"
src, isf, b9ids = [], [], set()
for s in [t for t in a.b9.split(",") if t]:
    if not os.path.exists(f"{CH}/{s}_ids.csv"): raise SystemExit(f"chưa có {CH}/{s}_*: dựng bằng repo/b9_prepare_v2.py --w {a.w} --h {a.h}")
    d_ = pd.read_csv(f"{CH}/{s}_ids.csv"); b9ids |= set(d_.id)
    xa = np.load(cp(f"{CH}/{s}_x.npy"), mmap_mode="r"); ya = np.load(cp(f"{CH}/{s}_y.npy"), mmap_mode="r")
    assert xa.shape[1:3] == (a.h, a.w) and len(d_) == len(xa) == len(ya), f"{s}: cache b9 không khớp cỡ {SZ} hoặc ids"
    src.append((xa, ya, None)); isf.append(d_.full_label.values.astype(bool))
assert not (b9ids & set(man.id.values[te])), "ảnh test lọt vào dữ liệu huấn luyện"
tr = np.array([i for i in tr if man.id.values[i] not in b9ids])      # ảnh thí điểm đã có trong b9 (đã mang nhãn đủ) thì không lặp
src.insert(0, (PX, PY, tr)); isf.insert(0, np.ones(len(tr), bool))   # ảnh thí điểm: nhãn đủ
sizes = [len(r) if r is not None else len(x) for x, _, r in src]; offs = np.concatenate([[0], np.cumsum(sizes)])
N = int(offs[-1]); IsFull = np.concatenate(isf); n_full = int(IsFull.sum())
stream = a.stream == 1 or (a.stream < 0 and N * a.h * a.w * 4 > 6e9)

def fetch(idx):
    """idx: mảng chỉ số toàn cục -> (x uint8 (n,h,w,3), y uint8 (n,h,w)) trên CPU, đúng thứ tự idx."""
    idx = np.asarray(idx); order = np.argsort(idx, kind="stable"); sidx = idx[order]
    which = np.searchsorted(offs, sidx, side="right") - 1
    xb = np.empty((len(idx), a.h, a.w, 3), np.uint8); yb = np.empty((len(idx), a.h, a.w), np.uint8)
    for k in np.unique(which):
        m = which == k; loc = sidx[m] - offs[k]; x_, y_, rows = src[k]
        rr = rows[loc] if rows is not None else loc
        xb[order[m]] = x_[rr]; yb[order[m]] = y_[rr]
    return torch.from_numpy(xb), torch.from_numpy(yb)

class Prefetch:
    """Luồng nền đọc trước các lô từ đĩa; ngoại lệ trong luồng được ném lại ở luồng chính."""
    def __init__(s, batches, depth=3):
        s.q = queue.Queue(depth); s.pin = torch.cuda.is_available() and dev == "cuda"
        s.t = threading.Thread(target=s.run, args=(batches,), daemon=True); s.t.start()
    def run(s, batches):
        try:
            for ix in batches:
                x, y = fetch(ix)
                s.q.put((x.pin_memory(), y.pin_memory()) if s.pin else (x, y))
            s.q.put(None)
        except BaseException as e:
            s.q.put(e)
    def __iter__(s):
        while True:
            it = s.q.get()
            if it is None: return
            if isinstance(it, BaseException): raise it
            yield it

if stream:
    Xt = Yt = None
else:
    Xt = torch.cat([torch.from_numpy(np.ascontiguousarray(x if r is None else x[r])) for x, _, r in src]).to(dev)
    Yt = torch.cat([torch.from_numpy(np.ascontiguousarray(y if r is None else y[r])) for _, y, r in src]).to(dev)
Xe = torch.from_numpy(np.ascontiguousarray(PX[te])).to(dev); Ye = torch.from_numpy(np.ascontiguousarray(PY[te])).to(dev)
if HD:   # nhãn 320x160 của đúng các ảnh test (cùng thứ tự manifest) để quy IoU về lưới chung
    PY320 = np.load(cp(f"{V2}/pilot_drv_y_320x160.npy"), mmap_mode="r"); assert len(PY320) == len(man)
    Ye320 = torch.from_numpy(np.ascontiguousarray(PY320[te])).to(dev)
else:
    Ye320 = None
_rng = np.random.default_rng(0)       # tập con train nhãn đủ, cố định, không aug: để đo khoảng cách train - test
_full_idx = np.where(IsFull)[0]; _sub = np.sort(_rng.choice(_full_idx, min(a.gap_n, len(_full_idx)), replace=False))
Xg, Yg = (t.to(dev) for t in fetch(_sub))
wts = torch.from_numpy(1.0 + (a.full_w - 1.0) * IsFull.astype(np.float32))
wts_dev = wts.to(dev); share = float((wts * torch.from_numpy(IsFull.astype(np.float32))).sum() / wts.sum())
_s254 = (fetch(_rng.choice(N, min(256, N), replace=False))[1] == 254).float().mean().item()
print(f"train {N} ảnh (thí điểm {len(tr)} + b9 {N - len(tr)}) | cỡ {SZ} | test {len(Xe)} | pixel nhãn 254 (ước lượng): {_s254:.1%} | "
      f"{'đọc từ đĩa (stream)' if stream else 'nạp hết lên GPU'}", flush=True)
print(f"ảnh nhãn đủ: {n_full}/{N} ({n_full / N:.1%}) | full_w {a.full_w}: tỉ lệ mẫu nhãn đủ mỗi epoch {share:.1%} | tập đo khoảng cách train: {len(_sub)} ảnh", flush=True)

mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
def augment_base(x, y, d_):
    n = x.shape[0]
    f = torch.rand(n, device=dev) < 0.5
    x = torch.where(f.view(-1, 1, 1, 1), x.flip(3), x); y = torch.where(f.view(-1, 1, 1), y.flip(2), y)
    s = torch.empty(n, device=dev).uniform_(0.7, 1.0)
    tx = (torch.rand(n, device=dev) * 2 - 1) * (1 - s); ty = (torch.rand(n, device=dev) * 2 - 1) * (1 - s)
    th = torch.zeros(n, 2, 3, device=dev); th[:, 0, 0] = s; th[:, 1, 1] = s; th[:, 0, 2] = tx; th[:, 1, 2] = ty
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=False)
    y = F.grid_sample(y.float().unsqueeze(1), g, mode="nearest", padding_mode="border", align_corners=False).squeeze(1).long()
    b = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.6, 1.4); c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    m = x.mean((1, 2, 3), keepdim=True); x = ((x - m) * c + m) * b
    return x.clamp(0, 255), y
augment = aug_strong.augment if a.aug == "strong" else augment_base

def cutout(x, k):
    """Che k ô chữ nhật (10-30% mỗi chiều) bằng màu trung bình ImageNet, nhãn giữ nguyên: buộc mạng dùng ngữ cảnh."""
    n, _, H, W = x.shape
    rows = torch.arange(H, device=dev).view(1, H, 1); cols = torch.arange(W, device=dev).view(1, 1, W)
    for _ in range(k):
        ph = (torch.empty(n, device=dev).uniform_(0.1, 0.3) * H).long().clamp(min=1); pw = (torch.empty(n, device=dev).uniform_(0.1, 0.3) * W).long().clamp(min=1)
        y0 = (torch.rand(n, device=dev) * (H - ph)).long(); x0 = (torch.rand(n, device=dev) * (W - pw)).long()
        m = ((rows >= y0.view(-1, 1, 1)) & (rows < (y0 + ph).view(-1, 1, 1)) & (cols >= x0.view(-1, 1, 1)) & (cols < (x0 + pw).view(-1, 1, 1))
             & (torch.rand(n, device=dev) < 0.5).view(-1, 1, 1))
        x = torch.where(m.unsqueeze(1), mean.expand(n, 3, 1, 1).expand(n, 3, H, W), x)
    return x

def build(arch, pretrained):
    if arch.startswith("smp:"):
        _, dec, enc = arch.split(":")
        return getattr(smp, dec)(enc, encoder_weights="imagenet" if pretrained else None, classes=C)
    return models_custom.build_fsnet(arch, num_classes=C)

class EMA:
    """Trung bình trượt của trọng số (và bộ đệm BN). Khởi động nhanh: d = min(decay, (1 + n) / (10 + n))."""
    def __init__(s, model, decay):
        s.m = copy.deepcopy(model).eval(); s.d = decay; s.n = 0
        for q in s.m.parameters(): q.requires_grad_(False)
    @torch.no_grad()
    def update(s, model):
        d = min(s.d, (1 + s.n) / (10 + s.n)); s.n += 1
        for e, v in zip(s.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point: e.mul_(d).add_(v.detach(), alpha=1 - d)
            else: e.copy_(v)

model = build(a.arch, a.pretrained).to(dev)
print(f"mô hình {a.arch}: {sum(q.numel() for q in model.parameters()) / 1e6:.2f} M tham số", flush=True)
if a.lr_enc_mult == 1.0 and not a.wd_nobn:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)           # đúng như các run cũ khi --wd mặc định
else:
    enc_ids = {id(q) for q in model.encoder.parameters()} if hasattr(model, "encoder") else set()
    pick = lambda is_enc, nd: [q for q in model.parameters() if (id(q) in enc_ids) == is_enc and (nd is None or (q.ndim > 1) == nd)]
    groups = []
    for is_enc in (True, False):
        lr_ = a.lr * (a.lr_enc_mult if is_enc else 1.0)
        if a.wd_nobn:        # chỉ phân rã trọng số conv/linear (ndim > 1); bias và BN không phân rã
            for nd, wd_ in ((True, a.wd), (False, 0.0)):
                ps = pick(is_enc, nd)
                if ps: groups.append(dict(params=ps, lr=lr_, weight_decay=wd_))
        else:
            ps = pick(is_enc, None)
            if ps: groups.append(dict(params=ps, lr=lr_, weight_decay=a.wd))
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters())), "nhóm tham số bị sót hoặc trùng"
    opt = torch.optim.AdamW(groups, lr=a.lr)
    print("nhóm tham số (lr, weight_decay, số tensor):", [(g["lr"], g["weight_decay"], len(g["params"])) for g in groups], flush=True)
ema = EMA(model, a.ema) if a.ema > 0 else None
spe = N // (a.bs * a.accum); total = spe * a.epochs
assert spe >= 1, f"dữ liệu ({N} ảnh) ít hơn cỡ lô hiệu dụng {a.bs * a.accum} (bs {a.bs} x accum {a.accum})"
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / spe) * (0.99 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)) + 0.01))
_GS = getattr(torch.amp, "GradScaler", None)
scaler = _GS("cuda", enabled=AMP) if _GS else torch.cuda.amp.GradScaler(enabled=AMP)
ac = lambda: torch.autocast(dev, dtype=torch.float16, enabled=AMP)
dice = smp.losses.DiceLoss("multiclass", ignore_index=255)
def lossf(o, y):
    m254 = y == 254
    yy = torch.where(m254, torch.full_like(y, 255), y)
    l = F.cross_entropy(o, yy, ignore_index=255, reduction="sum", label_smoothing=a.ls)
    if m254.any():                                    # pixel 254: chỉ cần p(lớp 0) + p(lớp 1) lớn
        l = l + (torch.logsumexp(o, 1) - torch.logsumexp(o[:, :2], 1))[m254].sum()
    return l / ((yy != 255) | m254).sum().clamp(min=1) + a.dice * dice(o, yy)

teacher = None
if a.kd > 0:
    tcfg = json.load(open(f"{DRV}/runs/{a.teacher}/config.json"))
    assert (tcfg.get("w", 320), tcfg.get("h", 160)) == (a.w, a.h), f"giáo viên {a.teacher} huấn luyện ở cỡ khác {a.w}x{a.h}"
    teacher = build(tcfg["arch"], False).to(dev)
    teacher.load_state_dict(torch.load(f"{DRV}/runs/{a.teacher}/last.pt", map_location=dev, weights_only=False)["model"])
    teacher.eval()
    for q in teacher.parameters(): q.requires_grad_(False)
    print(f"giáo viên: {a.teacher} ({tcfg['arch']}) | kd {a.kd} T {a.kd_T}", flush=True)
def kd_loss(o, x, y):
    with torch.no_grad(), ac(): t = teacher(x).float()
    T = a.kd_T; v = (y != 255).unsqueeze(1).float()            # gồm cả pixel 254: giáo viên cho nhãn mềm ở đó
    kl = F.kl_div(F.log_softmax(o / T, 1), F.softmax(t / T, 1), reduction="none").sum(1, keepdim=True)
    return a.kd * T * T * (kl * v).sum() / v.sum().clamp(min=1)

def summarize(conf):
    c = conf.cpu().numpy().astype(float).reshape(C, C)
    iou = [c[k, k] / (c[k].sum() + c[:, k].sum() - c[k, k] + 1e-9) for k in range(C)]
    tp, fp, fn, tn = c[1:, 1:].sum(), c[0, 1:].sum(), c[1:, 0].sum(), c[0, 0]
    return dict(miou=float(np.mean(iou)), iou=[float(v) for v in iou], road_iou=float(tp / (tp + fp + fn)),
                nonroad_iou=float(tn / (tn + fp + fn)), false_floor=float(fp / (fp + tn)), conf=c)

@torch.no_grad()
def evaluate(net, X, Y, Y320=None):
    was = net.training; net.eval()
    conf = torch.zeros(C * C, dtype=torch.long, device=dev); conf320 = torch.zeros_like(conf)
    for i in range(0, len(X), EB):
        x = (X[i:i + EB].permute(0, 3, 1, 2).float() - mean) / std; y = Y[i:i + EB].long()
        with ac(): pr = net(x).argmax(1)
        m = y < C; conf += torch.bincount(y[m] * C + pr[m], minlength=C * C)
        if Y320 is not None:   # argmax ở lưới HD, thu nhỏ nearest về 160x320, so với nhãn 320x160 của đúng ảnh test
            p3 = F.interpolate(pr.unsqueeze(1).float(), size=(160, 320), mode="nearest").squeeze(1).long()
            y3 = Y320[i:i + EB].long(); m3 = y3 < C
            conf320 += torch.bincount(y3[m3] * C + p3[m3], minlength=C * C)
    net.train(was)
    out = summarize(conf)
    if Y320 is not None: out["at320"] = summarize(conf320)
    return out

def atomic_save(obj, path): torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)
net_eval = ema.m if ema is not None else model
start = 0
if os.path.exists(f"{RUN}/last.pt"):
    ck = torch.load(f"{RUN}/last.pt", map_location=dev, weights_only=False)
    if "raw" in ck: model.load_state_dict(ck["raw"]); ema.m.load_state_dict(ck["model"])     # run có EMA: "model" = bản EMA
    else:
        model.load_state_dict(ck["model"])
        if ema is not None: ema.m.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"]); start = ck["epoch"]; print("tiếp tục từ epoch", start)
    if ema is not None: ema.n = start * spe
for ep in range(start, a.epochs):
    t0 = time.time(); torch.manual_seed(a.seed + ep); model.train()
    nb = spe * a.accum
    if stream:
        perm = torch.multinomial(wts, nb * a.bs, replacement=True) if a.full_w != 1.0 else torch.randperm(N)
        pf = iter(Prefetch([perm[k * a.bs:(k + 1) * a.bs].numpy() for k in range(nb)]))
    else:
        perm = (torch.multinomial(wts_dev, nb * a.bs, replacement=True) if a.full_w != 1.0
                else torch.randperm(N, device=dev))
    tl = torch.zeros((), device=dev)
    for i in range(spe):
        opt.zero_grad(set_to_none=True)
        for k in range(a.accum):
            if stream:
                xb, yb = next(pf); xb = xb.to(dev, non_blocking=True); yb = yb.to(dev, non_blocking=True)
            else:
                idx = perm[(i * a.accum + k) * a.bs:(i * a.accum + k + 1) * a.bs]; xb, yb = Xt[idx], Yt[idx]
            x, y = augment(xb.permute(0, 3, 1, 2).float(), yb.long(), dev)
            if a.cutout: x = cutout(x, a.cutout)
            x = (x - mean) / std
            with ac(): o = model(x)
            o = o.float(); loss = lossf(o, y)
            if teacher is not None: loss = loss + kd_loss(o, x, y)
            scaler.scale(loss / a.accum).backward(); tl += loss.detach() / a.accum
        scaler.step(opt); scaler.update(); sched.step()
        if ema is not None: ema.update(model)
    m = evaluate(net_eval, Xe, Ye, Ye320)
    ck = {"model": net_eval.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": ep + 1}
    if ema is not None: ck["raw"] = model.state_dict()
    atomic_save(ck, f"{RUN}/last.pt")
    extra = f" | IoU320 {[round(v, 3) for v in m['at320']['iou']]} road320 {m['at320']['road_iou']:.4f}" if HD else ""
    if (ep + 1) % 10 == 0 or ep + 1 == a.epochs:    # khoảng cách train - test ở làn đang đi (cùng lưới, ảnh nhãn đủ)
        g = evaluate(net_eval, Xg, Yg); extra += f" | train {g['iou'][2]:.3f} - test {m['iou'][2]:.3f} (làn đang đi)"
    print(f"ep {ep + 1}/{a.epochs} loss {tl.item() / spe:.4f} IoU {[round(v, 3) for v in m['iou']]} road {m['road_iou']:.4f}{extra} {time.time() - t0:.0f}s", flush=True)
m = evaluate(net_eval, Xe, Ye, Ye320); g = evaluate(net_eval, Xg, Yg)
res = dict(name=a.name, classes=C, arch=a.arch, w=a.w, h=a.h, full_w=a.full_w, n_full=n_full, epochs=a.epochs, flip=1, seed=a.seed, aug=a.aug,
           b9=a.b9, n_train=N, n_test=len(Xe), teacher=a.teacher, kd=a.kd, ema=a.ema, wd=a.wd, wd_nobn=a.wd_nobn, ls=a.ls, cutout=a.cutout,
           bs=a.bs, accum=a.accum, lr=a.lr, stream=bool(stream),
           miou=m["miou"], iou=m["iou"], road_iou=m["road_iou"], nonroad_iou=m["nonroad_iou"], false_floor=m["false_floor"],
           conf=m["conf"].tolist(), reported="epoch cuối",         # iou: ở lưới của chính nó (a.w x a.h)
           n_trainsub=len(_sub), iou_trainsub=g["iou"], road_iou_trainsub=g["road_iou"])
if HD:                                                           # iou_320: argmax HD -> nearest về lưới 320x160
    t3 = m["at320"]
    res.update(iou_320=t3["iou"], miou_320=t3["miou"], road_iou_320=t3["road_iou"], nonroad_iou_320=t3["nonroad_iou"],
               false_floor_320=t3["false_floor"], conf_320=t3["conf"].tolist())
json.dump(res, open(f"{RUN}/final_metrics.json", "w"), indent=1)
print("xong")
