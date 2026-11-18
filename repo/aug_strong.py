"""Augmentation mạnh hơn cho train.py (bật bằng --aug strong). Chạy trên GPU, cùng giao diện với augment() gốc.

x: float N,3,H,W (0..255); y: long N,H,W (0, 1, 255). Trả về (x đã clamp 0..255, y).

Khác bản gốc:
  - Tỉ lệ 0,6..1,2 (có cả thu nhỏ, trước đây chỉ phóng to 0,7..1,0), lệch tỉ lệ khung ±10%, xoay ±5 độ.
    Vùng ngoài ảnh khi thu nhỏ được gán nhãn 255 (bị loss bỏ qua).
  - Cân bằng màu theo kênh, độ bão hòa, gamma, làm mờ, nhiễu Gauss (mô phỏng webcam khác miền).
"""
import math

import torch
import torch.nn.functional as F


def _gauss_blur(x, sigma, k=7):
    r = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    w = torch.exp(-(r ** 2) / (2 * sigma ** 2))
    w = w / w.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def augment(x, y, dev):
    n, _, H, W = x.shape
    # --- lật ngang
    f = torch.rand(n, device=dev) < 0.5
    x = torch.where(f.view(-1, 1, 1, 1), x.flip(3), x)
    y = torch.where(f.view(-1, 1, 1), y.flip(2), y)
    # --- hình học: tỉ lệ, lệch tỉ lệ, xoay, dịch
    s = torch.empty(n, device=dev).uniform_(0.6, 1.2)          # <1 phóng to, >1 thu nhỏ
    asp = torch.empty(n, device=dev).uniform_(0.9, 1.1)
    rot = torch.empty(n, device=dev).uniform_(-5.0, 5.0) * math.pi / 180
    sx, sy = s * asp.sqrt(), s / asp.sqrt()
    m = (1 - s).clamp(min=0.1)
    tx = (torch.rand(n, device=dev) * 2 - 1) * m
    ty = (torch.rand(n, device=dev) * 2 - 1) * m
    th = torch.zeros(n, 2, 3, device=dev)
    th[:, 0, 0] = sx * torch.cos(rot); th[:, 0, 1] = -sx * torch.sin(rot) * (H / W); th[:, 0, 2] = tx
    th[:, 1, 0] = sy * torch.sin(rot) * (W / H); th[:, 1, 1] = sy * torch.cos(rot); th[:, 1, 2] = ty
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, mode="bilinear", padding_mode="reflection", align_corners=False)
    ys = F.grid_sample((y.float() + 1).unsqueeze(1), g, mode="nearest", padding_mode="zeros",
                       align_corners=False).squeeze(1)
    y = torch.where(ys < 0.5, torch.full_like(ys, 255.0), ys - 1).long()   # ngoài ảnh -> bỏ qua
    # --- quang học
    b = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    mean = x.mean((1, 2, 3), keepdim=True)
    x = ((x - mean) * c + mean) * b
    x = x * torch.empty(n, 3, 1, 1, device=dev).uniform_(0.9, 1.1)               # lệch màu theo kênh
    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)  # bão hòa
    x = x.clamp(0, 255)
    gam = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.25)
    x = 255.0 * (x / 255.0).pow(gam)
    # --- làm mờ (30% ảnh, sigma chung cho cả lô) và nhiễu Gauss (50% ảnh)
    bl = torch.rand(n, device=dev) < 0.3
    if bl.any():
        sigma = float(torch.empty(1).uniform_(0.4, 1.5))
        x = torch.where(bl.view(-1, 1, 1, 1), _gauss_blur(x, sigma), x)
    nz = (torch.rand(n, device=dev) < 0.5).float().view(-1, 1, 1, 1) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0, 6)
    x = x + torch.randn_like(x) * nz
    return x.clamp(0, 255), y
