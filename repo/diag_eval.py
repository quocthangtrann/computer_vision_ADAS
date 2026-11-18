"""Chẩn đoán một run FSNet: train vs val, lỗi gần biên, theo dải hàng ảnh, mô phỏng hậu xử lý, quét ngưỡng.

Chạy (Colab, sau khi mount Drive):
    !python /content/drive/MyDrive/capstone_ai/repo/diag_eval.py --name i_fsnet_m_320x160 --ckpt last
Ghi kết quả vào runs/<name>/diag_<ckpt>.json.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

DRV = os.environ.get("DRV", "/content/drive/MyDrive/capstone_ai")
p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--ckpt", default="last", choices=["last", "best"])
p.add_argument("--n_train", type=int, default=985, help="số ảnh train lấy ngẫu nhiên để đo (không augmentation)")
p.add_argument("--band", type=int, default=4, help="độ rộng dải gần biên (pixel ở cỡ ảnh vào)")
a = p.parse_args()

sys.path.insert(0, f"{DRV}/repo")
import models_custom

RUN = f"{DRV}/runs/{a.name}"
cfg = json.load(open(f"{RUN}/config.json"))
assert cfg["arch"] in models_custom.PRESETS, "script này chỉ dùng cho các mô hình fsnet_*"
dev = "cuda" if torch.cuda.is_available() else "cpu"
tag = f"{cfg['w']}x{cfg['h']}"

mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255


def load(split, n=None):
    x = np.load(f"{DRV}/data/cache_v1/{split}_x_{tag}.npy", mmap_mode="r")
    y = np.load(f"{DRV}/data/cache_v1/{split}_y_{tag}.npy", mmap_mode="r")
    idx = np.arange(len(x))
    if n is not None and n < len(x):
        idx = np.sort(np.random.default_rng(0).choice(len(x), n, replace=False))
    x = torch.from_numpy(np.ascontiguousarray(x[idx])).to(dev)
    y = torch.from_numpy(np.ascontiguousarray(y[idx])).to(dev)
    return x, y


net = models_custom.build_fsnet(cfg["arch"]).to(dev)
obj = torch.load(f"{RUN}/{a.ckpt}.pt", map_location=dev, weights_only=False)
sd = obj
if isinstance(obj, dict):
    for k in ("model", "state_dict", "net"):
        if k in obj:
            sd = obj[k]
            break
net.load_state_dict(sd)
net.eval()


@torch.no_grad()
def forward_all(x, deploy=False):
    """Trả về hiệu logit (sàn - không sàn) ở cỡ ảnh (deploy=False) hoặc ở 1/4 (deploy=True), float32."""
    net.deploy = deploy
    out = []
    for i in range(0, len(x), 64):
        xb = (x[i:i + 64].permute(0, 3, 1, 2).float() - mean) / std
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            o = net(xb)
        o = o.float()
        out.append(o[:, 1] - o[:, 0])
    net.deploy = False
    return torch.cat(out)


def stats(pred, y):
    """pred: bool (N,H,W) = dự đoán là sàn; y: 0/1/255."""
    v = y != 255
    t = (y == 1) & v
    f = (y == 0) & v
    tp = (pred & t).sum().item(); fn = (~pred & t).sum().item()
    fp = (pred & f).sum().item(); tn = (~pred & f).sum().item()
    iou1 = tp / max(tp + fn + fp, 1); iou0 = tn / max(tn + fn + fp, 1)
    return dict(miou=(iou0 + iou1) / 2, iou_floor=iou1, iou_non=iou0,
                floor_recall=tp / max(tp + fn, 1), false_floor=fp / max(fp + tn, 1))


def fmt(s):
    return f"mIoU {s['miou']:.4f} | IoU sàn {s['iou_floor']:.4f} | recall sàn {s['floor_recall']:.4f} | false_floor {s['false_floor']:.4f}"


res = {"run": a.name, "ckpt": a.ckpt}
print(f"== {a.name} ({a.ckpt}.pt), thiết bị {dev}")

# 1) train vs val
Xt, Yt = load("train", a.n_train)
Xv, Yv = load("val")
dt = forward_all(Xt); dv = forward_all(Xv)
st = stats(dt > 0, Yt.long()); sv = stats(dv > 0, Yv.long())
res["train_subset"] = st; res["val"] = sv
print(f"[1] train ({len(Xt)} ảnh, không augment): {fmt(st)}")
print(f"    val   ({len(Xv)} ảnh)               : {fmt(sv)}")
print(f"    chênh lệch train - val: mIoU {st['miou'] - sv['miou']:+.4f}, false_floor {st['false_floor'] - sv['false_floor']:+.4f}")
del Xt, Yt, dt

# 2) lỗi gần biên sàn / không-sàn (trên val)
y = Yv.long()
valid = y != 255
pred = dv > 0
err = (pred != (y == 1)) & valid
k = a.band
fl = (y == 1).float().unsqueeze(1); nf = (y == 0).float().unsqueeze(1)
band = ((F.max_pool2d(fl, 2 * k + 1, 1, k) > 0) & (F.max_pool2d(nf, 2 * k + 1, 1, k) > 0)).squeeze(1) & valid
n_err = max(err.sum().item(), 1)
res["boundary"] = dict(band_px=k, band_fraction=band.sum().item() / valid.sum().item(),
                       error_share_in_band=(err & band).sum().item() / n_err,
                       acc_in_band=1 - (err & band).sum().item() / max(band.sum().item(), 1))
b = res["boundary"]
print(f"[2] dải ±{k}px quanh biên chiếm {b['band_fraction']:.1%} pixel nhưng chứa {b['error_share_in_band']:.1%} số lỗi "
      f"(độ chính xác trong dải {b['acc_in_band']:.3f})")

# 3) theo dải hàng ảnh (trên xuống: xa -> gần xe)
H = y.shape[1]; rows = []
for i in range(4):
    s = slice(i * H // 4, (i + 1) * H // 4)
    r = stats(pred[:, s], y[:, s]); rows.append(r)
    print(f"[3] dải hàng {i + 1}/4 ({'xa' if i == 0 else 'gần xe' if i == 3 else 'giữa'}): recall sàn {r['floor_recall']:.4f} | false_floor {r['false_floor']:.4f} | IoU sàn {r['iou_floor']:.4f}")
res["row_bands"] = rows

# 4) mô phỏng hậu xử lý trên A53 (logits 1/4 -> mask cỡ ảnh)
dq = forward_all(Xv, deploy=True)
bil = F.interpolate(dq.unsqueeze(1), size=y.shape[-2:], mode="bilinear", align_corners=False).squeeze(1) > 0
near = F.interpolate((dq > 0).float().unsqueeze(1), size=y.shape[-2:], mode="nearest").squeeze(1) > 0
sb = stats(bil, y); sn = stats(near, y)
res["post_bilinear_logits"] = sb; res["post_nearest_mask"] = sn
print(f"[4] A53 phóng logits bilinear rồi so với 0: {fmt(sb)}")
print(f"    A53 phóng mask nearest từ 1/4         : {fmt(sn)}")

# 5) quét ngưỡng (lệch về an toàn: chỉ báo 'sàn' khi hiệu logit > bias)
sweep = []
for bias in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    s = stats(dv > bias, y); s["bias"] = bias; sweep.append(s)
    print(f"[5] bias {bias:>3}: {fmt(s)}")
res["bias_sweep"] = sweep

out = f"{RUN}/diag_{a.ckpt}.json"
json.dump(res, open(out + ".tmp", "w"), indent=1)
os.replace(out + ".tmp", out)
print("đã ghi", out)
