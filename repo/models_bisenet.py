import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(i, o, k=3, s=1, g=1, act=True):
    layers = [nn.Conv2d(i, o, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(o)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class StemBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = cbr(i, o, 3, 2)
        self.left = nn.Sequential(cbr(o, o // 2, 1), cbr(o // 2, o, 3, 2))
        self.right = nn.MaxPool2d(3, 2, 1)
        self.fuse = cbr(2 * o, o, 3, 1)

    def forward(self, x):
        x = self.conv(x)
        return self.fuse(torch.cat([self.left(x), self.right(x)], 1))


class GELayer(nn.Module):
    def __init__(self, i, o, s, e=6):
        super().__init__()
        m = i * e
        layers = [cbr(i, i, 3, 1), cbr(i, m, 3, s, g=i, act=False)]
        if s == 2:
            layers.append(cbr(m, m, 3, 1, g=m, act=False))
        layers.append(cbr(m, o, 1, act=False))
        self.f = nn.Sequential(*layers)
        self.short = None if (s == 1 and i == o) else nn.Sequential(cbr(i, i, 3, s, g=i, act=False),
                                                                    cbr(i, o, 1, act=False))
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.f(x) + (x if self.short is None else self.short(x)))


class BGA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.d1 = nn.Sequential(cbr(c, c, 3, 1, g=c, act=False), cbr(c, c, 1, act=False))
        self.d2 = nn.Sequential(cbr(c, c, 3, 2, act=False), nn.AvgPool2d(3, 2, 1))
        self.s1 = cbr(c, c, 3, 1, act=False)
        self.s2 = nn.Sequential(cbr(c, c, 3, 1, g=c, act=False), cbr(c, c, 1, act=False))
        self.up = nn.Upsample(scale_factor=4, mode="nearest")
        self.sig = nn.Hardsigmoid()
        self.out = cbr(c, c, 3, 1)

    def forward(self, d, s):
        left = self.d1(d) * self.sig(self.up(self.s1(s)))
        right = self.up(self.d2(d) * self.sig(self.s2(s)))
        return self.out(left + right)


class BiSeNetV2Lite(nn.Module):
    def __init__(self, nc=2, w=1.0):
        super().__init__()
        r = lambda v: max(16, int(round(v * w / 16)) * 16)
        c0, c1, c2, c3 = r(16), r(32), r(64), r(128)
        d1, d2 = r(64), r(128)
        assert d2 == c3
        self.detail = nn.Sequential(cbr(3, d1, 3, 2), cbr(d1, d1, 3, 1),
                                    cbr(d1, d1, 3, 2), cbr(d1, d1, 3, 1), cbr(d1, d1, 3, 1),
                                    cbr(d1, d2, 3, 2), cbr(d2, d2, 3, 1), cbr(d2, d2, 3, 1))
        self.stem = StemBlock(3, c0)
        self.s3 = nn.Sequential(GELayer(c0, c1, 2), GELayer(c1, c1, 1))
        self.s4 = nn.Sequential(GELayer(c1, c2, 2), GELayer(c2, c2, 1))
        self.s5 = nn.Sequential(GELayer(c2, c3, 2), GELayer(c3, c3, 1), GELayer(c3, c3, 1), GELayer(c3, c3, 1))
        self.ce = cbr(c3, c3, 3, 1)
        self.bga = BGA(c3)
        self.head = nn.Sequential(cbr(c3, c3, 3, 1), nn.Conv2d(c3, nc, 1))

    def forward(self, x):
        d = self.detail(x)
        s = self.ce(self.s5(self.s4(self.s3(self.stem(x)))))
        y = self.head(self.bga(d, s))
        return F.interpolate(y, scale_factor=8, mode="bilinear", align_corners=False)
