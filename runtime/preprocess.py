"""Tiền xử lý phía board: khung camera (cỡ bất kỳ) -> tensor INT8 (1, 3, H, W) cho mạng, cùng ma trận đổi tọa độ lưới -> khung.

Cấu hình: config/preprocess_b.json. Mặc định tái tạo ĐÚNG quy tắc huấn luyện (make_hd_pilot.py): khung 1280x720,
cắt (0, 0, 1280, 600), PIL BILINEAR (có khử răng cưa) ra 640x320, chuẩn hóa (RGB - mean) / std, lượng tử
q = clip(floor(x / scale + 0.5), -127, 127) với scale lấy từ int8_meta.json của gói (giống int8_sim.forward).

PIL không có trên board: resize "pil_bilinear" ở đây là bản NumPy viết lại thuật toán Resample.c của Pillow
(hệ số tam giác giãn theo tỉ lệ thu nhỏ, số nguyên điểm cố định 22 bit, ngang trước dọc sau), trùng từng byte với PIL
(kiểm trong tests/test_preprocess.py khi có PIL). "cv2_area" và "cv2_linear" nhanh hơn nhưng KHÁC ảnh lúc huấn luyện.

Quy ước tọa độ: tâm pixel ở số nguyên (như OpenCV), pixel i phủ đoạn [i - 0.5, i + 0.5].
grid_to_frame(gw, gh) đổi (x, y, 1) ở một lưới gw x gh phủ vùng cắt sang pixel khung: x_khung = x0 + (x + 0.5) * cw / gw - 0.5.
"""
import json
import math
import os
from dataclasses import dataclass

import numpy as np

try:
    import cv2 as _cv2
except ImportError:          # không có OpenCV: dùng NumPy (chậm hơn, cùng kết quả)
    _cv2 = None

PRECISION_BITS = 32 - 8 - 2        # như Pillow (Resample.c)


def _coeffs(in_size, in0, in1, out_size):
    """Hệ số của Pillow cho bộ lọc BILINEAR (support 1). Trả về (xmin (out,), kk int32 (out, ksize))."""
    scale = (in1 - in0) / out_size
    filterscale = scale if scale > 1.0 else 1.0
    support = 1.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    ss = 1.0 / filterscale
    xmins = np.zeros(out_size, np.int64)
    kk = np.zeros((out_size, ksize), np.float64)
    for xx in range(out_size):
        center = in0 + (xx + 0.5) * scale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        ww = 0.0
        for x in range(xmax):
            t = (x + xmin - center + 0.5) * ss
            if t < 0.0:
                t = -t
            w = 1.0 - t if t < 1.0 else 0.0
            kk[xx, x] = w
            ww += w
        if ww != 0.0:
            for x in range(xmax):
                kk[xx, x] /= ww
        xmins[xx] = xmin
    # normalize_coeffs_8bpc: (int)(0.5 + k * 2^22) với k >= 0 (bộ lọc tam giác không có hệ số âm)
    kq = np.where(kk < 0, np.trunc(-0.5 + kk * (1 << PRECISION_BITS)), np.trunc(0.5 + kk * (1 << PRECISION_BITS))).astype(np.int32)
    return xmins, kq


class PilBilinear:
    """Resize uint8 (H, W, C) giống Image.resize((ow, oh), Image.BILINEAR) của Pillow, hệ số tính sẵn một lần cho mỗi cỡ."""

    def __init__(self, in_w, in_h, out_w, out_h):
        self.in_wh, self.out_wh = (in_w, in_h), (out_w, out_h)
        self.need_h = out_w != in_w
        self.need_v = out_h != in_h
        self.h = self._prep(in_w, out_w) if self.need_h else None
        self.v = self._prep(in_h, out_h) if self.need_v else None

    @staticmethod
    def _prep(n_in, n_out):
        xmin, kk = _coeffs(n_in, 0.0, float(n_in), n_out)
        idx = np.minimum(xmin[:, None] + np.arange(kk.shape[1])[None, :], n_in - 1)    # vị trí ngoài xmax có hệ số 0
        taps = [k for k in range(kk.shape[1]) if kk[:, k].any()]                       # bỏ cột hệ số toàn 0
        return [(PilBilinear._plan(idx[:, k]), np.ascontiguousarray(kk[:, k])) for k in taps]

    @staticmethod
    def _plan(idx):
        """Chia dãy chỉ số nguồn thành các đoạn: đoạn giữa là cấp số cộng (đọc bằng cắt theo bước, nhanh), hai đầu đọc bằng np.take.
        Mỗi phần tử: (đầu, cuối, None, mảng chỉ số) hoặc (đầu, cuối, bắt_đầu, bước)."""
        n = len(idx)
        whole = [(0, n, None, np.ascontiguousarray(idx))]
        if n < 8:
            return whole
        d = np.diff(idx)
        vals, cnt = np.unique(d, return_counts=True)
        s = int(vals[cnt.argmax()])
        if s <= 0:
            return whole
        g = np.concatenate([[0], (d == s).astype(np.int8), [0]])
        st, en = np.where(np.diff(g) == 1)[0], np.where(np.diff(g) == -1)[0]
        j = int(np.argmax(en - st))
        lo, hi = int(st[j]), int(en[j]) + 1             # idx[lo:hi] là cấp số cộng bước s
        if hi - lo < n // 2:
            return whole
        out = [(0, lo, None, idx[:lo])] if lo > 0 else []
        out.append((lo, hi, int(idx[lo]), s))
        if hi < n:
            out.append((hi, n, None, idx[hi:]))
        return out

    @staticmethod
    def _pass(x, taps, axis):
        """Một lượt lọc trên ảnh (C, H, W) theo trục axis (2 = ngang, 1 = dọc) bằng số nguyên điểm cố định,
        như ImagingResample{Horizontal,Vertical}_8bpc. Bố cục kênh-mặt phẳng để vòng lặp trong của NumPy chạy theo hàng dài."""
        shape = list(x.shape)
        shape[axis] = taps[0][0][-1][1]
        acc = np.full(shape, 1 << (PRECISION_BITS - 1), np.int32)
        tmp = np.empty(shape, np.int32)
        cut = (lambda a, sl: a[:, :, sl]) if axis == 2 else (lambda a, sl: a[:, sl, :])
        for plan, k in taps:
            for lo, hi, start, step in plan:
                w = k[lo:hi]
                w = w[None, None, :] if axis == 2 else w[None, :, None]
                if start is None:
                    src = np.take(x, step, axis=axis)
                else:
                    src = cut(x, slice(start, start + step * (hi - lo - 1) + 1, step))
                np.multiply(src, w, out=cut(tmp, slice(lo, hi)))
            acc += tmp
        np.right_shift(acc, PRECISION_BITS, out=acc)        # clip8: >> 22 rồi kẹp [0, 255]
        np.clip(acc, 0, 255, out=acc)
        return acc.astype(np.uint8)

    def chw(self, x):
        """(C, H, W) uint8 -> (C, oh, ow) uint8."""
        assert x.dtype == np.uint8 and x.ndim == 3 and (x.shape[2], x.shape[1]) == self.in_wh, (x.shape, self.in_wh)
        if self.need_h:      # ngang trước (như Pillow), kết quả trung gian làm tròn về uint8
            x = self._pass(x, self.h, 2)
        if self.need_v:
            x = self._pass(x, self.v, 1)
        return np.ascontiguousarray(x)

    def __call__(self, img):
        """(H, W, C) uint8 -> (oh, ow, C) uint8."""
        return np.ascontiguousarray(self.chw(np.ascontiguousarray(img.transpose(2, 0, 1))).transpose(1, 2, 0))


def grid_to_frame(crop_ltrb, gw, gh):
    """Ma trận 3x3: (x, y, 1) ở lưới gw x gh (phủ đúng vùng cắt) -> pixel khung, quy ước tâm pixel ở số nguyên."""
    l, t, r, b = [float(v) for v in crop_ltrb]
    sx, sy = (r - l) / gw, (b - t) / gh
    return np.array([[sx, 0.0, l + 0.5 * sx - 0.5],
                     [0.0, sy, t + 0.5 * sy - 0.5],
                     [0.0, 0.0, 1.0]])


def apply(M, x, y):
    """Áp ma trận 3x3 (chỉ phần affine) cho tọa độ x, y (số hoặc mảng)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]


@dataclass
class Prepared:
    x_int8: np.ndarray          # (1, 3, H, W) int8, C-contiguous
    chw: np.ndarray             # (3, H, W) uint8 RGB sau cắt + resize
    grid_to_frame: np.ndarray   # 3x3, lưới ảnh vào của mạng (W x H) -> pixel khung
    crop_ltrb: tuple
    frame_wh: tuple
    scaled_crop: bool           # khung khác cỡ cấu hình nên vùng cắt đã được co giãn theo

    @property
    def image(self):
        """(H, W, 3) uint8 RGB (để kiểm, vẽ)."""
        return self.chw.transpose(1, 2, 0)


def resolve(path, root):
    return path if os.path.isabs(path) else os.path.join(root, path)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


class Preprocessor:
    def __init__(self, cfg, root='.'):
        if isinstance(cfg, str):
            cfg = load_json(cfg)
        self.cfg = cfg
        self.frame_wh = tuple(cfg['frame_wh'])
        self.crop = tuple(cfg['crop_ltrb'])
        self.out_wh = tuple(cfg['out_wh'])
        self.method = cfg.get('resize', 'pil_bilinear')
        self.color = cfg.get('frame_color', 'RGB').upper()
        self.on_mismatch = cfg.get('on_size_mismatch', 'scale_crop')
        assert self.method in ('pil_bilinear', 'cv2_area', 'cv2_linear'), self.method
        assert self.color in ('RGB', 'BGR'), self.color
        assert self.on_mismatch in ('scale_crop', 'error'), self.on_mismatch
        self.mean = np.array(cfg['mean'], np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(cfg['std'], np.float32).reshape(1, 3, 1, 1)
        self.meta = None
        scale = cfg.get('input_scale')
        if cfg.get('int8_meta'):
            mp = resolve(cfg['int8_meta'], root)
            if os.path.exists(mp):
                self.meta = load_json(mp)
                ish = tuple(self.meta['input']['shape'])
                if ish != (1, 3, self.out_wh[1], self.out_wh[0]):
                    raise ValueError(f'{mp}: đầu vào mạng {ish}, cấu hình ra {self.out_wh} (rộng x cao)')
                if scale is None:
                    scale = float(self.meta['input']['scale'])
            elif scale is None:
                raise FileNotFoundError(f'không thấy {mp} để đọc thang lượng tử đầu vào (hoặc đặt "input_scale" trong cấu hình)')
        if scale is None:
            raise ValueError('cấu hình thiếu "int8_meta" hoặc "input_scale"')
        self.input_scale = float(scale)
        self.lut = self._lut()
        self._lut_u8 = [np.ascontiguousarray(self.lut[c].view(np.uint8)) for c in range(3)]
        self._resizers = {}

    def _lut(self):
        """Bảng tra (3, 256) int8: giá trị RGB -> q, đúng phép tính của int8_sim (float32, floor(x / scale + 0.5), kẹp +-127)."""
        v = np.broadcast_to(np.arange(256, dtype=np.float32).reshape(1, 1, 1, 256), (1, 3, 1, 256))
        xf = (v - self.mean) / self.std
        q = np.clip(np.floor(xf / self.input_scale + 0.5), -127, 127)
        return q.reshape(3, 256).astype(np.int8)

    def crop_for(self, w, h):
        """Vùng cắt (trái, trên, phải, dưới) nguyên cho khung w x h."""
        if (w, h) == self.frame_wh:
            return tuple(int(v) for v in self.crop), False
        if self.on_mismatch == 'error':
            raise ValueError(f'khung {w}x{h}, cấu hình cần {self.frame_wh[0]}x{self.frame_wh[1]}')
        fx, fy = w / self.frame_wh[0], h / self.frame_wh[1]
        l, t, r, b = self.crop
        return (int(round(l * fx)), int(round(t * fy)), int(round(r * fx)), int(round(b * fy))), True

    def resize_chw(self, x):
        """(3, h, w) uint8 -> (3, H, W) uint8 theo phương pháp trong cấu hình."""
        ow, oh = self.out_wh
        if (x.shape[2], x.shape[1]) == (ow, oh):
            return np.ascontiguousarray(x)
        if self.method == 'pil_bilinear':
            key = (x.shape[2], x.shape[1])
            if key not in self._resizers:
                self._resizers[key] = PilBilinear(key[0], key[1], ow, oh)
            return self._resizers[key].chw(x)
        import cv2
        y = cv2.resize(np.ascontiguousarray(x.transpose(1, 2, 0)), (ow, oh), interpolation=cv2.INTER_AREA if self.method == 'cv2_area' else cv2.INTER_LINEAR)
        return np.ascontiguousarray(y.transpose(2, 0, 1))

    def quantize_chw(self, x):
        """(3, H, W) uint8 RGB ở cỡ ảnh vào của mạng -> (1, 3, H, W) int8 qua bảng tra."""
        q = np.empty((1,) + x.shape, np.int8)
        if _cv2 is not None:      # cv2.LUT nhanh hơn khoảng 5 lần; cùng bảng tra nên cùng kết quả (kiểm ở tests/test_preprocess.py)
            for c in range(3):
                q[0, c] = _cv2.LUT(x[c], self._lut_u8[c]).view(np.int8)
        else:
            for c in range(3):
                q[0, c] = self.lut[c][x[c]]
        return q

    def quantize(self, img):
        """(H, W, 3) uint8 RGB -> (1, 3, H, W) int8."""
        return self.quantize_chw(np.ascontiguousarray(img.transpose(2, 0, 1)))

    def to_float(self, x):
        """(3, H, W) hoặc (H, W, 3) uint8 RGB -> (1, 3, H, W) float32 đã chuẩn hóa, đúng như int8_sim.preprocess (cho ONNX số thực)."""
        if x.shape[0] != 3:
            x = x.transpose(2, 0, 1)
        return np.ascontiguousarray((x.astype(np.float32)[None] - self.mean) / self.std)

    def __call__(self, frame, color=None):
        """frame: (h, w, 3) uint8, thứ tự kênh theo cấu hình "frame_color" (hoặc tham số color)."""
        assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3, (frame.dtype, frame.shape)
        color = (color or self.color).upper()
        h, w = frame.shape[:2]
        crop, scaled = self.crop_for(w, h)
        l, t, r, b = crop
        if not (0 <= l < r <= w and 0 <= t < b <= h):
            raise ValueError(f'vùng cắt {crop} nằm ngoài khung {w}x{h}')
        img = frame[t:b, l:r]
        if self.method != 'pil_bilinear' and _cv2 is not None and (r - l, b - t) != self.out_wh:
            # Đường nhanh của OpenCV: resize thẳng trên khung HWC, đổi bố cục (HWC -> CHW, lật kênh nếu BGR) chỉ một lần ở ảnh ĐÃ NHỎ.
            # Đường chung bên dưới chép ảnh cắt cỡ đầy đủ hai lần (HWC -> CHW -> HWC) trước khi resize. Cùng kết quả (kiểm ở tests/test_preprocess.py).
            y = _cv2.resize(img if img.flags.c_contiguous else np.ascontiguousarray(img), self.out_wh,
                            interpolation=_cv2.INTER_AREA if self.method == 'cv2_area' else _cv2.INTER_LINEAR)
            x = np.ascontiguousarray((y[:, :, ::-1] if color == 'BGR' else y).transpose(2, 0, 1))
        else:
            x = np.ascontiguousarray((img[:, :, ::-1] if color == 'BGR' else img).transpose(2, 0, 1))     # (3, h, w) RGB
            x = self.resize_chw(x)
        return Prepared(self.quantize_chw(x), x, grid_to_frame(crop, *self.out_wh), crop, (w, h), scaled)
