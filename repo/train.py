import argparse, json, os, shutil, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

DRV = "/content/drive/MyDrive/capstone_ai"
LOC = "/content/cache_v1"
dev = "cuda"
torch.backends.cudnn.benchmark = True

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--arch", default="Unet")
p.add_argument("--encoder", default="mobilenet_v2")
p.add_argument("--w", type=int, default=320)
p.add_argument("--h", type=int, default=160)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dice", type=float, default=0.5)
p.add_argument("--dec", default="")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no_aug", action="store_true")
p.add_argument("--max_train", type=int, default=0)
p.add_argument("--aug", default="base")
p.add_argument("--teacher", default="")
p.add_argument("--kd", type=float, default=0.0)
p.add_argument("--kd_T", type=float, default=2.0)
p.add_argument("--bw", type=float, default=0.0)
p.add_argument("--bk", type=int, default=3)
p.add_argument("--micro", type=int, default=0)        # 0 = khong chia nho batch
p.add_argument("--data_dev", default="gpu")           # gpu | cpu (cache lon de o RAM)
a = p.parse_args()

assert os.path.ismount("/content/drive"), "Drive CHUA mount: chay O 0 truoc"
RUN = f"{DRV}/runs/{a.name}"
os.makedirs(RUN, exist_ok=True)
for _f in (__file__, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.py")):
    _d = f"{RUN}/code_{os.path.basename(_f)}"
    if os.path.exists(_f) and not os.path.exists(_d):
        shutil.copyfile(_f, _d)
CFG = f"{RUN}/config.json"
if os.path.exists(CFG):
    cfg = json.load(open(CFG)); print("dung config da luu:", cfg)
else:
    cfg = vars(a); json.dump(cfg, open(CFG, "w"), indent=2)

# ---- du lieu: chep cache ve dia cuc bo, dua len GPU ----
tag = f"{cfg['w']}x{cfg['h']}"
os.makedirs(LOC, exist_ok=True)
for f in os.listdir(f"{DRV}/data/cache_v1"):
    if (tag in f or f.endswith(".csv")) and not os.path.exists(f"{LOC}/{f}"):
        shutil.copyfile(f"{DRV}/data/cache_v1/{f}", f"{LOC}/{f}.tmp")
        os.replace(f"{LOC}/{f}.tmp", f"{LOC}/{f}")

def ld(split):
    dd = dev if cfg.get("data_dev", "gpu") == "gpu" else "cpu"
    out = []
    for k_ in ("x", "y"):
        a_ = np.load(f"{LOC}/{split}_{k_}_{tag}.npy", mmap_mode="r")
        t_ = torch.empty(a_.shape, dtype=torch.uint8, device=dd)
        for s_ in range(0, len(a_), 128):
            t_[s_:s_ + 128] = torch.from_numpy(np.ascontiguousarray(a_[s_:s_ + 128]))
        out.append(t_)
    return out[0], out[1]
Xt, Yt = ld("train"); Xv, Yv = ld("val")
if cfg["max_train"]:
    _i = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(0))[:cfg["max_train"]].to(Xt.device)
    Xt, Yt = Xt[_i], Yt[_i]
print("train", tuple(Xt.shape), "val", tuple(Xv.shape))

mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
std  = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255

def augment(x, y):                      # x float N,3,H,W (0..255); y long N,H,W
    n = x.shape[0]
    f = torch.rand(n, device=dev) < 0.5
    x = torch.where(f.view(-1, 1, 1, 1), x.flip(3), x)
    y = torch.where(f.view(-1, 1, 1), y.flip(2), y)
    s = torch.empty(n, device=dev).uniform_(0.7, 1.0)
    tx = (torch.rand(n, device=dev) * 2 - 1) * (1 - s)
    ty = (torch.rand(n, device=dev) * 2 - 1) * (1 - s)
    th = torch.zeros(n, 2, 3, device=dev)
    th[:, 0, 0] = s; th[:, 1, 1] = s; th[:, 0, 2] = tx; th[:, 1, 2] = ty
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=False)
    y = F.grid_sample(y.float().unsqueeze(1), g, mode="nearest", padding_mode="border",
                      align_corners=False).squeeze(1).long()
    b = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * c + m) * b
    return x.clamp(0, 255), y

# ---- mo hinh / toi uu / lich hoc ----
kw = {"decoder_channels": tuple(int(v) for v in cfg["dec"].split(","))} if cfg.get("dec") else {}
import sys; sys.path.insert(0, f"{DRV}/repo"); import models
import models_custom, aug_strong, boundary_loss
model = (models_custom.build_fsnet(cfg["arch"]) if cfg["arch"] in models_custom.PRESETS else (models.build(cfg["arch"]) if cfg["arch"] in models.REGISTRY else getattr(smp, cfg["arch"])(cfg["encoder"], encoder_weights="imagenet", classes=2, **kw))).to(dev)
teacher = None
if cfg.get("teacher") and cfg.get("kd", 0) > 0:
    tcfg = json.load(open(f"{DRV}/runs/{cfg['teacher']}/config.json"))
    tkw = {"decoder_channels": tuple(int(v) for v in tcfg["dec"].split(","))} if tcfg.get("dec") else {}
    teacher = getattr(smp, tcfg["arch"])(tcfg["encoder"], encoder_weights=None, classes=2, **tkw).to(dev)
    teacher.load_state_dict(torch.load(f"{DRV}/runs/{cfg['teacher']}/best.pt", map_location=dev))
    teacher.eval()
    for _p in teacher.parameters():
        _p.requires_grad_(False)
    print("teacher:", cfg["teacher"], "| kd", cfg["kd"], "T", cfg["kd_T"])
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
spe = len(Xt) // cfg["bs"]; total = spe * cfg["epochs"]
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / spe) * (0.99 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)) + 0.01))
scaler = torch.amp.GradScaler("cuda")
ce = nn.CrossEntropyLoss(ignore_index=255)
dice = smp.losses.DiceLoss("multiclass", ignore_index=255)
lossf = lambda o, y: ce(o, y) + cfg["dice"] * dice(o, y)

EB = max(1, min(64, int(3.3e6 // (cfg["w"] * cfg["h"]))))   # 320x160 -> 64 (nhu cu)
@torch.no_grad()
def evaluate():
    model.eval(); conf = torch.zeros(4, dtype=torch.long, device=dev)
    for i in range(0, len(Xv), EB):
        x = (Xv[i:i+EB].to(dev).permute(0, 3, 1, 2).float() - mean) / std
        y = Yv[i:i+EB].to(dev).long()
        with torch.autocast("cuda", dtype=torch.float16):
            pr = model(x).argmax(1)
        m = y != 255
        conf += torch.bincount(y[m] * 2 + pr[m], minlength=4)
    model.train()
    c = conf.cpu().numpy().astype(float).reshape(2, 2)      # [that, dudoan]
    iou = [c[k, k] / (c[k].sum() + c[:, k].sum() - c[k, k]) for k in (0, 1)]
    return dict(miou=float(np.mean(iou)), iou0=iou[0], iou1=iou[1],
                floor_recall=c[1, 1] / c[1].sum(), false_floor=c[0, 1] / c[0].sum())

def atomic_save(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)

# ---- tiep tuc neu co ----
start, best = 0, -1.0
if os.path.exists(f"{RUN}/last.pt"):
    ck = torch.load(f"{RUN}/last.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
    start, best = ck["epoch"], float(ck["best"])
    print("tiep tuc tu epoch", start)

LOG = f"{RUN}/metrics.csv"
if not os.path.exists(LOG):
    open(LOG, "w").write("epoch,loss,miou,iou_notfloor,iou_floor,floor_recall,false_floor,sec\n")

for ep in range(start, cfg["epochs"]):
    t0 = time.time(); torch.manual_seed(cfg["seed"] + ep); model.train()
    perm = torch.randperm(len(Xt), device=Xt.device); tl = torch.zeros((), device=dev)
    for i in range(spe):
        idx = perm[i * cfg["bs"]:(i + 1) * cfg["bs"]]
        x = Xt[idx].to(dev).permute(0, 3, 1, 2).float(); y = Yt[idx].to(dev).long()
        if not cfg["no_aug"]:
            x, y = (aug_strong.augment(x, y, dev) if cfg.get("aug", "base") == "strong" else augment(x, y))
        x = (x - mean) / std
        MB = cfg.get("micro", 0) or cfg["bs"]
        assert cfg["bs"] % MB == 0, "bs phai chia het cho micro"
        nmb = cfg["bs"] // MB
        opt.zero_grad(set_to_none=True)
        for xs, ys in zip(x.split(MB), y.split(MB)):
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(xs)
            loss = (boundary_loss.loss(o.float(), ys, dice, cfg["dice"], cfg["bw"], cfg["bk"])
                    if cfg.get("bw", 0) > 0 else lossf(o.float(), ys))
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    t = teacher(xs).float()
                T = cfg["kd_T"]; v = (ys != 255).unsqueeze(1).float()
                kl = F.kl_div(F.log_softmax(o.float() / T, 1), F.softmax(t / T, 1), reduction="none").sum(1, keepdim=True)
                loss = loss + cfg["kd"] * T * T * (kl * v).sum() / v.sum().clamp(min=1)
            scaler.scale(loss / nmb).backward()
            tl += loss.detach() / nmb
        scaler.step(opt); scaler.update(); sched.step()
    m = evaluate(); sec = time.time() - t0
    if m["miou"] > best:
        best = m["miou"]; atomic_save(model.state_dict(), f"{RUN}/best.pt")
    atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": ep + 1, "best": best}, f"{RUN}/last.pt")
    open(LOG, "a").write(f"{ep+1},{tl.item()/spe:.4f},{m['miou']:.4f},{m['iou0']:.4f},{m['iou1']:.4f},"
                         f"{m['floor_recall']:.4f},{m['false_floor']:.4f},{sec:.0f}\n")
    print(f"ep {ep+1}/{cfg['epochs']} loss {tl.item()/spe:.4f} mIoU {m['miou']:.4f} "
          f"(0:{m['iou0']:.3f} 1:{m['iou1']:.3f}) false_floor {m['false_floor']:.4f} best {best:.4f} {sec:.0f}s")
print("xong | dinh bo nho GPU %.1f GB" % (torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0))
