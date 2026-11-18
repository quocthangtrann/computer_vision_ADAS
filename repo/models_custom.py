"""FSNet: CNN thuần theo hợp đồng toán tử v1 của IP tăng tốc (KV260).

Chỉ dùng: Conv 3x3 (stride 1, groups 1, padding same) + BN + ReLU, MaxPool 2x2,
Upsample nearest x2, Concat theo kênh, Conv 1x1 ở đầu ra. Kênh là bội số của 16.

forward():
  deploy=False (mặc định, dùng khi train/eval): trả logits (N, 2, H, W) đã phóng bilinear
      lên đúng cỡ ảnh vào, để dùng chung loss/metric với các mô hình trước.
  deploy=True: trả logits ở 1/4 độ phân giải (N, 2, H/4, W/4), đúng phần chạy trên IP.
      Phần phóng lên + argmax là hậu xử lý trên A53.
"""
import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    # e0..e4: kênh encoder; n1: số conv ở 1/2; n4: số conv ở 1/16; d1, d2: kênh decoder (1/8, 1/4)
    "fsnet_m": dict(e0=16, e1=32, e2=48, e3=96, e4=128, n1=1, n4=3, d1=48, d2=32),
    "fsnet_s": dict(e0=16, e1=16, e2=32, e3=48, e4=64, n1=1, n4=2, d1=32, d2=32),
    # d3: kênh ra của tầng decoder thêm ở 1/2 (0 = không có, đầu ra ở 1/4)
    "fsnet_m_o2": dict(e0=16, e1=32, e2=48, e3=96, e4=128, n1=1, n4=3, d1=48, d2=32, d3=16),
    # pre: so lan MaxPool 2x2 them ngay sau lop dau (ha mau som, tong stride 2^(4+pre), dau ra 1/8 khi pre=1);
    # e5, d0: them tang 1/32 (2 conv) + 1 conv decoder ve 1/16 (chieu cao/rong anh phai chia het cho 64 khi pre=1)
    "fsnet_m_p1": dict(e0=16, e1=32, e2=48, e3=96, e4=128, n1=1, n4=3, d1=48, d2=32, pre=1),
    "fsnet_m_p1s5": dict(e0=16, e1=32, e2=48, e3=96, e4=128, n1=1, n4=3, d1=48, d2=32, pre=1, e5=128, d0=128),
}


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=1, padding=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=False)
        self.fused = False

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)

    @torch.no_grad()
    def fuse(self):
        """Gộp BN vào conv (đúng việc hợp đồng v1 yêu cầu khi xuất)."""
        if self.fused:
            return
        bn = self.bn
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        self.conv.weight.mul_(scale.reshape(-1, 1, 1, 1))
        bias = bn.bias - bn.running_mean * scale
        self.conv.bias = nn.Parameter(bias.clone())
        self.bn = None
        self.fused = True


class FSNet(nn.Module):
    def __init__(self, e0, e1, e2, e3, e4, n1, n4, d1, d2, d3=0, pre=0, e5=0, d0=0, num_classes=2):
        super().__init__()
        self.deploy = False
        self.e0 = ConvBNReLU(3, e0)
        self.pool = nn.MaxPool2d(2, 2)
        c, convs = e0, []
        for _ in range(n1):
            convs.append(ConvBNReLU(c, e1)); c = e1
        self.s1 = nn.Sequential(*convs)
        self.s2 = nn.Sequential(ConvBNReLU(e1, e2), ConvBNReLU(e2, e2))
        self.s3 = nn.Sequential(ConvBNReLU(e2, e3), ConvBNReLU(e3, e3))
        c, convs = e3, []
        for _ in range(n4):
            convs.append(ConvBNReLU(c, e4)); c = e4
        self.s4 = nn.Sequential(*convs)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.pre = pre
        if e5:   # tang them o 1/32, decoder ve 1/16
            self.s5 = nn.Sequential(ConvBNReLU(e4, e5), ConvBNReLU(e5, e5))
            self.dz = ConvBNReLU(e5 + e4, d0)
        else:
            self.s5 = self.dz = None
        self.d1 = nn.Sequential(ConvBNReLU((d0 if e5 else e4) + e3, d1), ConvBNReLU(d1, d1))
        self.d2 = nn.Sequential(ConvBNReLU(d1 + e2, d2), ConvBNReLU(d2, d2))
        self.d3 = ConvBNReLU(d2 + e1, d3) if d3 else None   # 1/2, skip từ s1
        self.head = nn.Conv2d(d3 or d2, num_classes, 1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.e0(x)
        for _ in range(self.pre):
            x = self.pool(x)           # ha mau som
        x = self.s1(self.pool(x)); k1 = x  # 1/2  (skip, chỉ dùng khi có d3)
        x = self.s2(self.pool(x)); k2 = x  # 1/4  (skip)
        x = self.s3(self.pool(x)); k3 = x  # 1/8  (skip)
        x = self.s4(self.pool(x))          # 1/16
        if self.s5 is not None:
            k4 = x
            x = self.dz(torch.cat([self.up(self.s5(self.pool(x))), k4], dim=1))   # 1/32 -> 1/16
        x = self.d1(torch.cat([self.up(x), k3], dim=1))   # 1/8
        x = self.d2(torch.cat([self.up(x), k2], dim=1))   # 1/4
        if self.d3 is not None:
            x = self.d3(torch.cat([self.up(x), k1], dim=1))   # 1/2
        x = self.head(x)
        if self.deploy:
            return x
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


def build_fsnet(name="fsnet_m", num_classes=2):
    return FSNet(num_classes=num_classes, **PRESETS[name])


def fuse_for_deploy(model):
    """Bản sao đã gộp BN, forward trả logits 1/4. Dùng để xuất ONNX và kiểm tra hợp đồng."""
    m = copy.deepcopy(model).eval()
    for mod in m.modules():
        if isinstance(mod, ConvBNReLU):
            mod.fuse()
    m.deploy = True
    return m


@torch.no_grad()
def layer_table(model, h=160, w=320):
    """Bảng từng lớp conv (đã gộp BN): vào, ra, kernel, MAC, tham số, line buffer."""
    m = fuse_for_deploy(model)
    rows, hooks = [], []

    def hook(mod, inp, out):
        _, cin, hi, wi = inp[0].shape
        _, cout, ho, wo = out.shape
        k = mod.kernel_size[0]
        rows.append(dict(cin=cin, cout=cout, hin=hi, win=wi, hout=ho, wout=wo, k=k,
                         mac=ho * wo * cout * cin * k * k,
                         par=cout * cin * k * k + cout,
                         lb=(k - 1) * wi * cin))

    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(hook))
    m(torch.zeros(1, 3, h, w))
    for hk in hooks:
        hk.remove()
    return rows


def contract_violations(model):
    """Liệt kê mọi module ngoài hợp đồng v1. Danh sách rỗng là đạt."""
    bad = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            ok = (mod.kernel_size in ((3, 3), (1, 1)) and mod.stride == (1, 1)
                  and mod.groups == 1 and mod.dilation == (1, 1)
                  and (mod.padding == (1, 1) if mod.kernel_size == (3, 3) else mod.padding == (0, 0)))
            if not ok:
                bad.append((name, "Conv2d ngoài hợp đồng", mod))
        elif isinstance(mod, nn.MaxPool2d):
            if not (mod.kernel_size == 2 and mod.stride == 2):
                bad.append((name, "MaxPool khác 2x2", mod))
        elif isinstance(mod, nn.Upsample):
            if not (mod.mode == "nearest" and mod.scale_factor == 2):
                bad.append((name, "Upsample khác nearest x2", mod))
        elif isinstance(mod, (nn.BatchNorm2d, nn.ReLU, nn.Identity, nn.Sequential, ConvBNReLU, FSNet)):
            pass
        elif len(list(mod.children())) == 0:
            bad.append((name, "module lạ", mod))
    return bad


def export_onnx(model, path, h=160, w=320, opset=13):
    m = fuse_for_deploy(model)
    x = torch.zeros(1, 3, h, w)
    torch.onnx.export(m, x, path, opset_version=opset, input_names=["image"],
                      output_names=["logits_quarter"], dynamo=False)
    import onnx
    g = onnx.load(path).graph
    return sorted({n.op_type for n in g.node})


def _selftest(names=("fsnet_m", "fsnet_s", "fsnet_m_o2"), h=160, w=320, onnx_dir="/tmp"):
    # Constant, Identity chỉ là hằng số/bí danh của tham số (không phải phép tính): bỏ qua khi kiểm op
    allowed = {"Conv", "Relu", "MaxPool", "Resize", "Concat", "Constant", "Identity"}
    for name in names:
        torch.manual_seed(0)
        net = build_fsnet(name).eval()
        rows = layer_table(net, h, w)
        tm = sum(r["mac"] for r in rows)
        tp = sum(r["par"] for r in rows)
        lb = max(r["lb"] for r in rows)
        print(f"== {name}: {len(rows)} conv | {tm / 1e6:.1f} MMAC | {tp / 1e3:.0f} K tham số INT8 "
              f"| line buffer lớn nhất {lb / 1024:.1f} KB")
        print("   vi phạm hợp đồng:", contract_violations(net) or "không")
        bad_ch = [r["cin"] for r in rows if r["cin"] % 16] + [r["cout"] for r in rows if r["cout"] % 16 and r["cout"] != 2]
        print("   kênh không chia hết 16:", bad_ch or "không")
        x = torch.randn(2, 3, h, w)
        with torch.no_grad():
            y = net(x)
            fused = fuse_for_deploy(net)
            y4 = fused(x)
            # so FP32 gốc với bản gộp BN: phóng bản gộp giống phần train rồi so
            y_up = F.interpolate(y4, size=(h, w), mode="bilinear", align_corners=False)
        print(f"   logits train {tuple(y.shape)} | logits deploy {tuple(y4.shape)} "
              f"| lệch sau gộp BN {float((y - y_up).abs().max()):.2e}")
        ops = export_onnx(net, f"{onnx_dir}/{name}_{h}x{w}.onnx", h, w)
        extra = [o for o in ops if o not in allowed]
        print("   op trong ONNX:", ops, "| ngoài danh sách:", extra or "không")


def _print_table(name="fsnet_m", h=160, w=320):
    rows = layer_table(build_fsnet(name), h, w)
    print(f"{'#':>2} {'vào':>14} {'ra':>14} k {'MAC(M)':>8} {'par(K)':>7} {'LB(KB)':>6}")
    for i, r in enumerate(rows):
        print(f"{i:>2} {r['hin']:>3}x{r['win']:<3}x{r['cin']:<4}  {r['hout']:>3}x{r['wout']:<3}x{r['cout']:<4} "
              f"{r['k']} {r['mac'] / 1e6:8.2f} {r['par'] / 1e3:7.1f} {r['lb'] / 1024:6.1f}")


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 320
    _selftest(h=h, w=w)
    if "--table" in sys.argv:
        _print_table("fsnet_m", h, w)
