import torch.nn as nn
import torch.nn.functional as F


def cbr(i, o, k=3, s=1, g=1, act=True):
    layers = [nn.Conv2d(i, o, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(o)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class DSConv(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.dw = cbr(i, i, 3, s, g=i)
        self.pw = cbr(i, o, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class Bottleneck(nn.Module):
    def __init__(self, i, o, s, t=6):
        super().__init__()
        m = i * t
        self.res = (s == 1 and i == o)
        self.b = nn.Sequential(cbr(i, m, 1), cbr(m, m, 3, s, g=m), cbr(m, o, 1, act=False))

    def forward(self, x):
        y = self.b(x)
        return x + y if self.res else y


def stage(i, o, n, s):
    return nn.Sequential(Bottleneck(i, o, s), *[Bottleneck(o, o, 1) for _ in range(n - 1)])


class FastSCNNLite(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.ds = nn.Sequential(cbr(3, 32, 3, 2), DSConv(32, 48, 2), DSConv(48, 64, 2))
        self.gfe = nn.Sequential(stage(64, 64, 3, 2), stage(64, 96, 3, 2), stage(96, 128, 3, 1),
                                 DSConv(128, 128))
        self.hi = cbr(64, 128, 1, act=False)
        self.lo = nn.Sequential(cbr(128, 128, 3, 1, g=128), cbr(128, 128, 1, act=False))
        self.act = nn.ReLU6(inplace=True)
        self.cls = nn.Sequential(DSConv(128, 128), DSConv(128, 128), nn.Conv2d(128, nc, 1))

    def forward(self, x):
        h = self.ds(x)
        g = F.interpolate(self.gfe(h), scale_factor=4, mode="nearest")
        y = self.cls(self.act(self.hi(h) + self.lo(g)))
        return F.interpolate(y, scale_factor=8, mode="bilinear", align_corners=False)


REGISTRY = {"fast_scnn_lite": lambda: FastSCNNLite(2)}


def build(name):
    return REGISTRY[name]()


from models_bisenet import BiSeNetV2Lite
REGISTRY['bisenetv2_lite'] = lambda: BiSeNetV2Lite(2, 1.0)
REGISTRY['bisenetv2_lite_w05'] = lambda: BiSeNetV2Lite(2, 0.5)
