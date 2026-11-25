"""Kết quả 4 tầng của tầng nhận thức và gói nhị phân perception_v0 (mô tả đầy đủ: docs/interfaces/perception_v0.md).

Tầng 0  Logits      logit INT8 ở lưới đầu ra của mạng (1/8 ảnh vào với B: 4 x 40 x 80) kèm thang (logit = q * scale).
Tầng 1  Mask        mặt nạ uint8 ở lưới ảnh vào của mạng (B: 320 x 640): 0 không đường, 1 đường không thuộc ta, 2 làn đang đi, 3 làn kề.
Tầng 2  Tier2       tọa độ pixel KHUNG CAMERA (gốc ở góc trên trái, x sang phải, y xuống dưới; tâm pixel ở số nguyên, như OpenCV):
                    N cột: hàng xa nhất còn đi được tính liên tục từ đáy lên; M hàng mẫu: mép trái, mép phải, tâm làn đang đi.
Tầng 3  Tier3       hệ xe, đơn vị cm. CHƯA có hiệu chuẩn webcam (S4): mọi giá trị là NaN, mọi cờ hợp lệ bằng 0.

Mọi giá trị không hợp lệ là NaN và có cờ bằng 0: luôn đọc cờ trước khi dùng giá trị.
Chỉ dùng thư viện chuẩn và NumPy.
"""
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

CLASSES = ('không đường', 'không thuộc ta', 'làn đang đi', 'làn kề')
NOT_ROAD, ROAD_NOT_OURS, EGO_LANE, ADJ_LANE = 0, 1, 2, 3

# cờ của từng cột (tầng 2 và 3)
COL_VALID = 1          # pixel đáy của cột đi được, giá trị y có nghĩa
COL_REACHES_TOP = 2    # đi được tới tận hàng trên cùng của lưới: khoảng trống thật có thể xa hơn (y là cận)
# cờ của từng hàng mẫu (tầng 2 và 3)
ROW_LEFT = 1           # thấy mép trái của làn đang đi (không chạm biên trái ảnh)
ROW_RIGHT = 2          # thấy mép phải (không chạm biên phải ảnh)
ROW_CENTER = 4         # tâm hợp lệ (cả hai mép đều thấy)
ROW_OBSTACLE = 8       # trong làn có vùng "không đường" bị kẹp giữa hai đoạn làn đang đi (vật cản trong làn)
ROW_FOUND = 16         # hàng có pixel làn đang đi (dù mép có chạm biên)

# cờ của gói (trường flags trong tiêu đề)
PKT_HAS_LOGITS = 1
PKT_HAS_MASK = 2
PKT_TIER3_CALIBRATED = 4
PKT_DRIVABLE_CLASS1 = 8    # "đi được" của tầng 2 có tính cả lớp 1 (đường không thuộc ta)
PKT_FAST_POST = 16         # hậu xử lý không phải bản chuẩn (fast, half hoặc quarter): tầng 2 tính trên lưới thưa hơn lưới ảnh vào

MAGIC = b'PERC'
VERSION = 0
#        magic ver hsz seq  t_cap t_pub flags fw  fh  gw  gh  ncol nrow model plsz t_pre t_inf t_post t_t2
HEADER = struct.Struct('<4sHHIQQIHHHHHH16sIIIII')
STAGES = ('pre', 'infer', 'post', 'tier2')     # thời gian từng khâu (micro giây) ghi trong tiêu đề
LOGITS_HDR = struct.Struct('<fHHHH6f')        # scale, c, h, w, dự phòng, affine 2x3 (lưới logit -> khung)
MASK_HDR = struct.Struct('<HH6f')             # h, w, affine 2x3 (lưới mặt nạ -> khung)
CRC = struct.Struct('<I')


@dataclass
class Logits:
    """Tầng 0. q: int8 (C, h, w) kèm scale; hoặc f: float32 (C, h, w) khi backend trả số thực (ONNX)."""
    q: Optional[np.ndarray] = None
    scale: Optional[float] = None
    f: Optional[np.ndarray] = None

    def as_float(self) -> np.ndarray:
        """Đổi về số thực đúng như int8_sim.logits_of: q.astype(float32) * float32(scale)."""
        if self.f is not None:
            return self.f
        return self.q.astype(np.float32) * np.float32(self.scale)

    @property
    def shape(self):
        return (self.q if self.q is not None else self.f).shape


@dataclass
class Mask:
    """Tầng 1. m: uint8 (h, w), giá trị 0..3. grid_to_frame: ma trận 3x3 đổi (x, y, 1) ở lưới này sang pixel khung."""
    m: np.ndarray
    grid_to_frame: np.ndarray


@dataclass
class Tier2:
    """Tầng 2, đơn vị pixel khung camera (float32). Cột i: col_x[i] là x của cột, col_y[i] là mép trên của đoạn đi được
    liên tục tính từ đáy lưới (ranh giới giữa pixel đi được cuối cùng và pixel không đi được phía trên nó).
    Hàng k (k = 0 là hàng gần xe nhất): row_y[k], mép trái row_left[k], mép phải row_right[k], tâm row_center[k] của làn đang đi."""
    col_x: np.ndarray
    col_y: np.ndarray
    col_flags: np.ndarray
    row_y: np.ndarray
    row_left: np.ndarray
    row_right: np.ndarray
    row_center: np.ndarray
    row_flags: np.ndarray

    @property
    def n_cols(self):
        return len(self.col_x)

    @property
    def n_rows(self):
        return len(self.row_y)


@dataclass
class Tier3:
    """Tầng 3, hệ xe, cm (float32). Trục: fwd về phía trước từ cản trước, lat sang trái dương. CHƯA hiệu chuẩn: NaN, cờ 0."""
    col_lat_cm: np.ndarray
    col_fwd_cm: np.ndarray
    col_flags: np.ndarray
    row_fwd_cm: np.ndarray
    row_left_cm: np.ndarray
    row_right_cm: np.ndarray
    row_center_cm: np.ndarray
    row_flags: np.ndarray

    @classmethod
    def empty(cls, n_cols, n_rows):
        nan = lambda n: np.full(n, np.nan, np.float32)
        return cls(nan(n_cols), nan(n_cols), np.zeros(n_cols, np.uint8),
                   nan(n_rows), nan(n_rows), nan(n_rows), nan(n_rows), np.zeros(n_rows, np.uint8))


@dataclass
class PerceptionResult:
    seq: int
    t_capture_ns: int
    frame_wh: Tuple[int, int]
    grid_wh: Tuple[int, int]          # lưới dùng để tính tầng 2 (lưới ảnh vào của mạng, hoặc lưới logit ở bản nhanh)
    tier2: Tier2
    tier3: Tier3
    model: str = ''
    logits: Optional[Logits] = None
    mask: Optional[Mask] = None
    timings_us: Dict[str, int] = field(default_factory=dict)
    drivable_class1: bool = False
    fast: bool = False
    tier3_calibrated: bool = False
    t_publish_ns: int = 0
    logits_grid_to_frame: Optional[np.ndarray] = None


def _pad4(n):
    return (-n) % 4


def _affine6(M):
    M = np.asarray(M, np.float64)
    return [float(v) for v in M[:2, :].ravel()]


def _from6(v):
    M = np.eye(3)
    M[:2, :] = np.asarray(v, np.float64).reshape(2, 3)
    return M


def _f32(a, n):
    a = np.ascontiguousarray(a, dtype='<f4')
    assert a.shape == (n,), (a.shape, n)
    return a.tobytes()


def _u8(a, n):
    a = np.ascontiguousarray(a, dtype=np.uint8)
    assert a.shape == (n,), (a.shape, n)
    return a.tobytes()


def payload_size(n_cols, n_rows, logits_chw=None, mask_hw=None):
    """Số byte phần thân (không gồm tiêu đề và CRC)."""
    n = 2 * 4 * (2 * n_cols + 4 * n_rows)             # tầng 2 và tầng 3: số thực
    fl = 2 * (n_cols + n_rows)
    n += fl + _pad4(fl)                                  # cờ uint8, đệm cho chia hết 4
    if logits_chw is not None:
        c, h, w = logits_chw
        n += LOGITS_HDR.size + c * h * w + _pad4(c * h * w)
    if mask_hw is not None:
        h, w = mask_hw
        n += MASK_HDR.size + h * w + _pad4(h * w)
    return n


def packet_size(n_cols, n_rows, logits_chw=None, mask_hw=None):
    return HEADER.size + payload_size(n_cols, n_rows, logits_chw, mask_hw) + CRC.size


def pack(r: PerceptionResult, include_logits=False, include_mask=False, t_publish_ns=None) -> bytes:
    """PerceptionResult -> bytes (little-endian). Tầng 0 và 1 chỉ đóng khi được yêu cầu và có trong r."""
    t2, t3 = r.tier2, r.tier3
    N, M = t2.n_cols, t2.n_rows
    assert t3.col_flags.shape == (N,) and t3.row_flags.shape == (M,), 'tầng 3 phải cùng số cột, số hàng với tầng 2'
    inc_l = include_logits and r.logits is not None and r.logits.q is not None
    inc_m = include_mask and r.mask is not None
    flags = ((PKT_HAS_LOGITS if inc_l else 0) | (PKT_HAS_MASK if inc_m else 0) | (PKT_TIER3_CALIBRATED if r.tier3_calibrated else 0)
             | (PKT_DRIVABLE_CLASS1 if r.drivable_class1 else 0) | (PKT_FAST_POST if r.fast else 0))
    body = [_f32(t2.col_x, N), _f32(t2.col_y, N), _f32(t2.row_y, M), _f32(t2.row_left, M), _f32(t2.row_right, M), _f32(t2.row_center, M),
            _f32(t3.col_lat_cm, N), _f32(t3.col_fwd_cm, N), _f32(t3.row_fwd_cm, M), _f32(t3.row_left_cm, M), _f32(t3.row_right_cm, M),
            _f32(t3.row_center_cm, M),
            _u8(t2.col_flags, N), _u8(t2.row_flags, M), _u8(t3.col_flags, N), _u8(t3.row_flags, M), b'\0' * _pad4(2 * (N + M))]
    lchw = mhw = None
    if inc_l:
        q = np.ascontiguousarray(r.logits.q, np.int8)
        q = q[0] if q.ndim == 4 else q
        lchw = q.shape
        A = r.logits_grid_to_frame if r.logits_grid_to_frame is not None else np.eye(3)
        body += [LOGITS_HDR.pack(float(r.logits.scale), *lchw, 0, *_affine6(A)), q.tobytes(), b'\0' * _pad4(q.size)]
    if inc_m:
        m = np.ascontiguousarray(r.mask.m, np.uint8)
        mhw = m.shape
        body += [MASK_HDR.pack(*mhw, *_affine6(r.mask.grid_to_frame)), m.tobytes(), b'\0' * _pad4(m.size)]
    body = b''.join(body)
    assert len(body) == payload_size(N, M, lchw, mhw)
    tp = int(t_publish_ns if t_publish_ns is not None else r.t_publish_ns)
    model = r.model.encode('ascii', 'replace')[:16].ljust(16, b'\0')
    tu = [min(int(r.timings_us.get(s, 0)), 0xFFFFFFFF) for s in STAGES]
    hdr = HEADER.pack(MAGIC, VERSION, HEADER.size, r.seq & 0xFFFFFFFF, int(r.t_capture_ns), tp, flags, *r.frame_wh, *r.grid_wh,
                      N, M, model, len(body), *tu)
    buf = hdr + body
    return buf + CRC.pack(zlib.crc32(buf) & 0xFFFFFFFF)


class PacketError(ValueError):
    pass


def unpack(buf: bytes) -> PerceptionResult:
    """bytes -> PerceptionResult. Kiểm magic, phiên bản, độ dài và CRC32; sai thì báo PacketError."""
    if len(buf) < HEADER.size + CRC.size:
        raise PacketError(f'gói quá ngắn: {len(buf)} byte')
    (magic, ver, hsz, seq, tc, tp, flags, fw, fh, gw, gh, N, M, model, plsz, *tu) = HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise PacketError(f'magic sai: {magic!r}')
    if ver != VERSION:
        raise PacketError(f'phiên bản {ver}, bộ giải gói này là {VERSION}')
    if hsz != HEADER.size or len(buf) != hsz + plsz + CRC.size:
        raise PacketError(f'độ dài sai: tiêu đề {hsz}, thân {plsz}, gói {len(buf)}')
    (crc,) = CRC.unpack_from(buf, len(buf) - CRC.size)
    if zlib.crc32(buf[:-CRC.size]) & 0xFFFFFFFF != crc:
        raise PacketError('CRC32 sai')
    o = hsz

    def f32(n):
        nonlocal o
        a = np.frombuffer(buf, '<f4', n, o).astype(np.float32)
        o += 4 * n
        return a

    def u8(n):
        nonlocal o
        a = np.frombuffer(buf, np.uint8, n, o).copy()
        o += n
        return a

    cx, cy, ry, rl, rr, rc = f32(N), f32(N), f32(M), f32(M), f32(M), f32(M)
    a3 = [f32(N), f32(N), f32(M), f32(M), f32(M), f32(M)]
    cf2, rf2, cf3, rf3 = u8(N), u8(M), u8(N), u8(M)
    o += _pad4(2 * (N + M))
    logits = mask = A_l = None
    if flags & PKT_HAS_LOGITS:
        scale, c, h, w, _, *aff = LOGITS_HDR.unpack_from(buf, o)
        o += LOGITS_HDR.size
        q = np.frombuffer(buf, np.int8, c * h * w, o).reshape(c, h, w).copy()
        o += c * h * w + _pad4(c * h * w)
        logits, A_l = Logits(q=q, scale=float(scale)), _from6(aff)
    if flags & PKT_HAS_MASK:
        h, w, *aff = MASK_HDR.unpack_from(buf, o)
        o += MASK_HDR.size
        mask = Mask(np.frombuffer(buf, np.uint8, h * w, o).reshape(h, w).copy(), _from6(aff))
        o += h * w + _pad4(h * w)
    if o != hsz + plsz:
        raise PacketError(f'thân gói dài {plsz} byte nhưng đọc được {o - hsz}')
    return PerceptionResult(
        seq=seq, t_capture_ns=tc, frame_wh=(fw, fh), grid_wh=(gw, gh),
        tier2=Tier2(cx, cy, cf2, ry, rl, rr, rc, rf2),
        tier3=Tier3(a3[0], a3[1], cf3, a3[2], a3[3], a3[4], a3[5], rf3),
        model=model.rstrip(b'\0').decode('ascii', 'replace'), logits=logits, mask=mask,
        timings_us=dict(zip(STAGES, tu)), drivable_class1=bool(flags & PKT_DRIVABLE_CLASS1), fast=bool(flags & PKT_FAST_POST),
        tier3_calibrated=bool(flags & PKT_TIER3_CALIBRATED), t_publish_ns=tp, logits_grid_to_frame=A_l)
