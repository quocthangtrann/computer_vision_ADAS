"""Loss ưu tiên biên cho segmentation sàn / không-sàn (bật bằng --bw trong train.py).

CE được nhân trọng số 1 + beta cho các pixel nằm trong dải ±k px quanh biên sàn/không-sàn
(dải = pixel có cả sàn lẫn không-sàn trong cửa sổ (2k+1)x(2k+1) quanh nó, giống cách diag_eval đo
"dải gần biên"). Pixel nhãn 255 bị bỏ qua. Phần Dice giữ nguyên. Không tốn gì khi chạy trên IP.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def boundary_band(y, k=3):
    """y: (N,H,W) nhãn 0/1/255 -> (N,H,W) bool, True ở pixel hợp lệ trong dải ±k px quanh biên."""
    fl = (y == 1).float().unsqueeze(1)
    nf = (y == 0).float().unsqueeze(1)
    near_f = F.max_pool2d(fl, 2 * k + 1, 1, k) > 0
    near_n = F.max_pool2d(nf, 2 * k + 1, 1, k) > 0
    return (near_f & near_n).squeeze(1) & (y != 255)


def loss(o, y, dice, dice_w, beta=2.0, k=3):
    """o: (N,2,H,W) logits float32; y: (N,H,W) long 0/1/255; dice: hàm Dice của train.py."""
    ce = F.cross_entropy(o, y, ignore_index=255, reduction="none")      # 0 ở pixel bị bỏ qua
    w = (1.0 + beta * boundary_band(y, k).float()) * (y != 255).float()
    return (ce * w).sum() / w.sum().clamp(min=1.0) + dice_w * dice(o, y)
