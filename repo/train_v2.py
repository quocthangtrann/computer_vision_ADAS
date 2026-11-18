
import argparse, json, os, time, math, shutil, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
DRV = "/content/drive/MyDrive/capstone_ai"; LOC = "/content/pilot_cache"; dev = "cuda"
torch.backends.cudnn.benchmark = True
p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--arch", default="fsnet_m")
p.add_argument("--classes", type=int, default=4); p.add_argument("--epochs", type=int, default=60)
p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dice", type=float, default=0.5); p.add_argument("--seed", type=int, default=0)
p.add_argument("--flip", type=int, default=1); p.add_argument("--aug", default="base")
a = p.parse_args(); C = a.classes; assert C in (2, 3, 4)
RUN = f"{DRV}/runs/{a.name}"; os.makedirs(RUN, exist_ok=True)
for f_ in (__file__, f"{DRV}/repo/models_custom.py"):
    d_ = f"{RUN}/code_{os.path.basename(f_)}"
    if not os.path.exists(d_): shutil.copyfile(f_, d_)
if not os.path.exists(f"{RUN}/config.json"): json.dump(vars(a), open(f"{RUN}/config.json", "w"), indent=2)

os.makedirs(LOC, exist_ok=True)
for f_ in ("pilot_drv_x_320x160.npy", "pilot_drv_y_320x160.npy", "pilot_drv_manifest.csv"):
    if not os.path.exists(f"{LOC}/{f_}"):
        shutil.copyfile(f"{DRV}/data/v2/{f_}", f"{LOC}/{f_}.tmp"); os.replace(f"{LOC}/{f_}.tmp", f"{LOC}/{f_}")
man = pd.read_csv(f"{LOC}/pilot_drv_manifest.csv")
X = np.load(f"{LOC}/pilot_drv_x_320x160.npy", mmap_mode="r"); Y = np.load(f"{LOC}/pilot_drv_y_320x160.npy", mmap_mode="r")
assert len(man) == len(X) == len(Y)
def ld(split):
    ii = np.where(man.split_v2.values == split)[0]
    x = torch.from_numpy(np.ascontiguousarray(X[ii])).to(dev); y = torch.from_numpy(np.ascontiguousarray(Y[ii])).to(dev)
    if C == 2: y = torch.where(y == 255, y, (y >= 1).to(y.dtype))
    elif C == 3: y = torch.where(y == 255, y, (y.to(torch.int16) - 1).clamp(min=0).to(y.dtype))
    return x, y
Xt, Yt = ld("train"); Xe, Ye = ld("test")
print("train", tuple(Xt.shape), "test", tuple(Xe.shape), "| số lớp:", C, flush=True)

mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
def augment(x, y):
    n = x.shape[0]
    if a.flip:
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

sys.path.insert(0, f"{DRV}/repo"); import models_custom, aug_strong
model = models_custom.build_fsnet(a.arch, num_classes=C).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
spe = len(Xt) // a.bs; total = spe * a.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / spe) * (0.99 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)) + 0.01))
scaler = torch.amp.GradScaler("cuda")
ce = nn.CrossEntropyLoss(ignore_index=255); dice = smp.losses.DiceLoss("multiclass", ignore_index=255)
lossf = lambda o, y: ce(o, y) + a.dice * dice(o, y)

@torch.no_grad()
def evaluate():
    model.eval(); conf = torch.zeros(C * C, dtype=torch.long, device=dev)
    for i in range(0, len(Xe), 64):
        x = (Xe[i:i + 64].permute(0, 3, 1, 2).float() - mean) / std; y = Ye[i:i + 64].long()
        with torch.autocast("cuda", dtype=torch.float16): pr = model(x).argmax(1)
        m = y != 255; conf += torch.bincount(y[m] * C + pr[m], minlength=C * C)
    model.train()
    c = conf.cpu().numpy().astype(float).reshape(C, C)          # [thật, dự đoán]
    iou = [c[k, k] / (c[k].sum() + c[:, k].sum() - c[k, k] + 1e-9) for k in range(C)]
    tp, fp, fn, tn = c[1:, 1:].sum(), c[0, 1:].sum(), c[1:, 0].sum(), c[0, 0]
    ri, ni = tp / (tp + fp + fn), tn / (tn + fp + fn)
    return dict(miou=float(np.mean(iou)), iou=[float(v) for v in iou], road_iou=float(ri), nonroad_iou=float(ni),
                false_floor=float(fp / (fp + tn)), conf=c)

def atomic_save(obj, path): torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)
start = 0
if os.path.exists(f"{RUN}/last.pt"):
    ck = torch.load(f"{RUN}/last.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"]); start = ck["epoch"]; print("tiếp tục từ epoch", start)
for ep in range(start, a.epochs):
    t0 = time.time(); torch.manual_seed(a.seed + ep); model.train()
    perm = torch.randperm(len(Xt), device=dev); tl = torch.zeros((), device=dev)
    for i in range(spe):
        idx = perm[i * a.bs:(i + 1) * a.bs]
        x, y = (aug_strong.augment if a.aug == "strong" else (lambda x_, y_, d_: augment(x_, y_)))(Xt[idx].permute(0, 3, 1, 2).float(), Yt[idx].long(), dev); x = (x - mean) / std
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16): o = model(x)
        loss = lossf(o.float(), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl += loss.detach()
    m = evaluate()
    atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": ep + 1}, f"{RUN}/last.pt")
    print(f"ep {ep + 1}/{a.epochs} loss {tl.item() / spe:.4f} mIoU {m['miou']:.4f} IoU {[round(v, 3) for v in m['iou']]} "
          f"road {m['road_iou']:.4f} false_floor {m['false_floor']:.4f} {time.time() - t0:.0f}s", flush=True)
m = evaluate()
json.dump(dict(name=a.name, classes=C, epochs=a.epochs, flip=a.flip, seed=a.seed, n_train=len(Xt), n_test=len(Xe),
               miou=m["miou"], iou=m["iou"], road_iou=m["road_iou"], nonroad_iou=m["nonroad_iou"], false_floor=m["false_floor"],
               conf=m["conf"].tolist(), reported="epoch cuối, không chọn epoch tốt nhất"),
          open(f"{RUN}/final_metrics.json", "w"), indent=1)
print("xong | đỉnh bộ nhớ GPU %.1f GB" % (torch.cuda.max_memory_allocated() / 2 ** 30))
