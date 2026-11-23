"""Hậu xử lý phía board: logit (INT8 kèm thang, hoặc số thực) -> mặt nạ 4 lớp -> tầng 2 (tọa độ pixel khung camera).

Bản chuẩn (mode "standard"), khớp cách train_v3.py / int8_sim.py đánh giá:
  logit float32 = q * scale -> phóng bilinear align_corners=False lên lưới ảnh vào của mạng (B: 320 x 640) -> argmax theo lớp.
  upsample_bilinear() viết lại F.interpolate(mode='bilinear', align_corners=False) bằng NumPy float32, cùng thứ tự phép tính
  (ngang trước: t = x0*w0 + x1*w1; rồi dọc: t0*h0 + t1*h1). Mức trùng với torch đo ở tests/test_postprocess.py.
Bản nhanh (mode "fast"): argmax thẳng trên lưới logit (B: 40 x 80) bằng số nguyên, không phóng bilinear; mặt nạ lặp pixel
  (nearest) lên lưới ảnh vào để tầng 2 lấy mẫu đúng các hàng, cột như bản chuẩn.
  fast_deviation() báo mức lệch (pixel mặt nạ và tọa độ tầng 2) so với bản chuẩn.

Tầng 2 (compute_tier2):
  - N cột cách đều trên lưới (tâm cột i ở x = (i + 0.5) * W / N). Mỗi cột: đếm từ hàng đáy lên các pixel "đi được" LIÊN TỤC
    (lớp 2 hoặc 3; tùy chọn tính cả lớp 1). col_y = mép trên của đoạn đó. Pixel đáy không đi được -> cờ hợp lệ 0.
  - M hàng mẫu cách đều giữa row_bottom_frac và row_top_frac của chiều cao lưới (k = 0 là hàng gần xe nhất).
    Mỗi hàng: các đoạn lớp 2 (làn đang đi). Hai đoạn kẹp giữa một khoảng TOÀN lớp 0 (vật cản trong làn) và khoảng đó
    không rộng quá merge_max_gap_frac * W thì gộp lại, đặt cờ ROW_OBSTACLE. Chọn đoạn chứa (hoặc gần nhất) vị trí tham chiếu:
    hàng đầu dùng trục xe ego_x_frac * W, các hàng sau dùng tâm của hàng ngay dưới (bám làn từ gần ra xa).
    Mép trái/phải là ranh giới ngoài của đoạn; mép chạm biên ảnh thì không hợp lệ; tâm chỉ hợp lệ khi cả hai mép hợp lệ.
Mọi tham số mặc định CHƯA chỉnh trên ảnh thật (cần hiệu chuẩn webcam S4).
"""
from dataclasses import dataclass

import numpy as np

from runtime import types as T
from runtime.preprocess import apply


def _lin(in_size, out_size):
    """Chỉ số và trọng số nội suy tuyến tính như torch (align_corners=False), tính bằng float32."""
    scale = np.float32(in_size) / np.float32(out_size)
    d = np.arange(out_size, dtype=np.float32)
    src = scale * (d + np.float32(0.5)) - np.float32(0.5)
    src = np.maximum(src, np.float32(0))
    i0 = src.astype(np.int64)                      # src >= 0 nên cắt phần lẻ = làm tròn xuống
    l1 = np.clip(src - i0.astype(np.float32), np.float32(0), np.float32(1)).astype(np.float32)
    l0 = (np.float32(1) - l1).astype(np.float32)
    i1 = np.where(i0 < in_size - 1, i0 + 1, i0)
    return i0, i1, l0, l1


_LIN_CACHE = {}


def upsample_bilinear(x, out_h, out_w):
    """x: (C, h, w) float32 -> (C, out_h, out_w) float32, như F.interpolate(bilinear, align_corners=False)."""
    C, h, w = x.shape
    key = (h, w, out_h, out_w)
    if key not in _LIN_CACHE:
        _LIN_CACHE[key] = (_lin(h, out_h), _lin(w, out_w))
    (h0, h1, a0, a1), (w0, w1, b0, b1) = _LIN_CACHE[key]
    x = np.asarray(x, np.float32)
    t = x[:, :, w0] * b0 + x[:, :, w1] * b1                 # (C, h, out_w)
    return t[:, h0, :] * a0[None, :, None] + t[:, h1, :] * a1[None, :, None]


def argmax_classes(x):
    """(C, H, W) -> (H, W) uint8; hòa thì lấy lớp nhỏ nhất (như torch.argmax và np.argmax)."""
    return np.argmax(x, axis=0).astype(np.uint8)


def mask_standard(logits: T.Logits, out_h, out_w):
    lg = logits.as_float()
    lg = lg[0] if lg.ndim == 4 else lg
    return argmax_classes(upsample_bilinear(lg, out_h, out_w))


def mask_fast(logits: T.Logits):
    """argmax trên lưới logit. Với INT8: so thẳng q (thang chung mọi lớp nên cùng kết quả với q * scale)."""
    a = logits.q if logits.q is not None else logits.f
    a = a[0] if a.ndim == 4 else a
    return argmax_classes(a)


def nearest_resize(m, out_h, out_w):
    """Thu nhỏ/phóng nearest như F.interpolate(mode='nearest'): nguồn = floor(đích * vào / ra)."""
    h, w = m.shape
    iy = np.minimum(np.floor(np.arange(out_h) * (h / out_h)).astype(np.int64), h - 1)
    ix = np.minimum(np.floor(np.arange(out_w) * (w / out_w)).astype(np.int64), w - 1)
    return m[iy][:, ix]


@dataclass
class Tier2Config:
    n_cols: int = 32
    n_rows: int = 8
    drivable_include_class1: bool = False
    row_top_frac: float = 0.6
    row_bottom_frac: float = 0.98
    ego_x_frac: float = 0.5
    merge_max_gap_frac: float = 0.5

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def sample_rows(H, cfg: Tier2Config):
    """Chỉ số hàng mẫu trên lưới cao H, phần tử 0 là hàng gần xe nhất (dưới cùng)."""
    yb, yt = cfg.row_bottom_frac * (H - 1), cfg.row_top_frac * (H - 1)
    if cfg.n_rows == 1:
        return np.array([int(round(yb))])
    return np.clip(np.round(np.linspace(yb, yt, cfg.n_rows)).astype(np.int64), 0, H - 1)


def sample_cols(W, cfg: Tier2Config):
    return np.minimum(np.floor((np.arange(cfg.n_cols) + 0.5) * W / cfg.n_cols).astype(np.int64), W - 1)


def _runs(b):
    """Các đoạn True liên tục của mảng bool 1 chiều: danh sách (đầu, cuối) tính cả hai đầu."""
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def lane_row(row, ref_x, max_gap):
    """Một hàng mặt nạ -> (trái, phải, cờ) theo tọa độ lưới (ranh giới ngoài của đoạn), hoặc None nếu không có lớp 2."""
    runs = _runs(row == T.EGO_LANE)
    if not runs:
        return None
    merged = []
    for a, b in runs:
        if merged:
            pa, pb, ob = merged[-1]
            gap = row[pb + 1:a]
            if len(gap) <= max_gap and np.all(gap == T.NOT_ROAD):
                merged[-1] = (pa, b, True)
                continue
        merged.append((a, b, False))
    dist = [0.0 if a - 0.5 <= ref_x <= b + 0.5 else min(abs(a - 0.5 - ref_x), abs(b + 0.5 - ref_x)) for a, b, _ in merged]
    a, b, ob = merged[int(np.argmin(dist))]
    W = len(row)
    fl = T.ROW_FOUND | (T.ROW_OBSTACLE if ob else 0)
    if a > 0:
        fl |= T.ROW_LEFT
    if b < W - 1:
        fl |= T.ROW_RIGHT
    if (fl & T.ROW_LEFT) and (fl & T.ROW_RIGHT):
        fl |= T.ROW_CENTER
    return a - 0.5, b + 0.5, fl


def compute_tier2(mask, g2f, cfg: Tier2Config = None) -> T.Tier2:
    """mask (H, W) uint8 0..3 ở một lưới; g2f: ma trận 3x3 lưới -> khung. Trả về Tier2 theo pixel khung."""
    cfg = cfg or Tier2Config()
    H, W = mask.shape
    classes = (1, 2, 3) if cfg.drivable_include_class1 else (2, 3)
    drv = np.isin(mask, classes)
    xs = sample_cols(W, cfg)
    col = drv[::-1, xs]                                   # từ đáy lên
    bad = ~col
    n_free = np.where(bad.any(0), bad.argmax(0), H)        # số hàng đi được liên tục từ đáy
    cf = np.where(n_free > 0, T.COL_VALID, 0) | np.where(n_free == H, T.COL_REACHES_TOP, 0)
    yg = (H - n_free).astype(np.float64) - 0.5             # mép trên của pixel đi được cao nhất
    cx, cy = apply(g2f, xs.astype(np.float64), yg)
    cy = np.where(n_free > 0, cy, np.nan)

    ys = sample_rows(H, cfg)
    M = len(ys)
    rl, rr, rc = (np.full(M, np.nan) for _ in range(3))
    rf = np.zeros(M, np.uint8)
    ref = cfg.ego_x_frac * W - 0.5
    max_gap = cfg.merge_max_gap_frac * W
    for k, y in enumerate(ys):
        r = lane_row(mask[y], ref, max_gap)
        if r is None:
            continue
        a, b, fl = r
        rf[k] = fl
        rl[k] = a if fl & T.ROW_LEFT else np.nan
        rr[k] = b if fl & T.ROW_RIGHT else np.nan
        rc[k] = (a + b) / 2 if fl & T.ROW_CENTER else np.nan
        ref = (a + b) / 2
    yy = ys.astype(np.float64)
    _, ry = apply(g2f, np.zeros(M), yy)
    f = lambda v: apply(g2f, v, yy)[0]
    return T.Tier2(cx.astype(np.float32), cy.astype(np.float32), cf.astype(np.uint8),
                   ry.astype(np.float32), f(rl).astype(np.float32), f(rr).astype(np.float32), f(rc).astype(np.float32), rf)


class Postprocessor:
    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        self.mode = cfg.pop('mode', 'standard')
        assert self.mode in ('standard', 'fast'), self.mode
        self.t2 = Tier2Config.from_dict(cfg)

    def mask(self, logits, net_hw, g2f_net):
        """Trả về (mặt nạ ở lưới ảnh vào của mạng, ma trận lưới -> khung). Bản nhanh: argmax ở lưới logit rồi phóng nearest
        (lặp pixel), nên tầng 2 lấy mẫu đúng các hàng, cột như bản chuẩn."""
        H, W = net_hw
        if self.mode == 'standard':
            return mask_standard(logits, H, W), g2f_net
        return nearest_resize(mask_fast(logits), H, W), g2f_net

    def tier2(self, mask, g2f):
        return compute_tier2(mask, g2f, self.t2)


def logit_grid_to_frame(g2f_net, net_hw, logit_hw):
    """Ma trận lưới logit -> khung, từ ma trận lưới ảnh vào -> khung (lưới logit phủ đúng cùng vùng)."""
    H, W = net_hw
    h, w = logit_hw
    s_x, s_y = W / w, H / h
    # x_net = (x_l + 0.5) * s - 0.5
    A = np.array([[s_x, 0.0, 0.5 * s_x - 0.5], [0.0, s_y, 0.5 * s_y - 0.5], [0.0, 0.0, 1.0]])
    return g2f_net @ A


def fast_deviation(logits, net_hw, g2f_net, cfg: Tier2Config = None):
    """So bản nhanh với bản chuẩn trên một khung: tỉ lệ pixel trùng (ở lưới ảnh vào, bản nhanh phóng nearest) và
    độ lệch tầng 2 (pixel khung) ở những phần tử hợp lệ cả hai bên."""
    cfg = cfg or Tier2Config()
    H, W = net_hw
    ms = mask_standard(logits, H, W)
    mf = nearest_resize(mask_fast(logits), H, W)
    agree = float((mf == ms).mean())
    a = compute_tier2(ms, g2f_net, cfg)
    b = compute_tier2(mf, g2f_net, cfg)
    out = dict(pixel_agree=agree)
    v = (a.col_flags & T.COL_VALID) & (b.col_flags & T.COL_VALID) > 0
    out['col_y_absdiff_px'] = np.abs(a.col_y - b.col_y)[v].tolist()
    out['col_valid_mismatch'] = int(((a.col_flags & T.COL_VALID) != (b.col_flags & T.COL_VALID)).sum())
    v = (a.row_flags & T.ROW_CENTER) & (b.row_flags & T.ROW_CENTER) > 0
    out['row_center_absdiff_px'] = np.abs(a.row_center - b.row_center)[v].tolist()
    out['row_center_valid_mismatch'] = int(((a.row_flags & T.ROW_CENTER) != (b.row_flags & T.ROW_CENTER)).sum())
    return out
