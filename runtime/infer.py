"""Phần chạy mạng. Mọi backend có cùng giao diện:  run(x) -> types.Logits

  x:  (1, 3, H, W) int8, C-contiguous, NCHW, kênh RGB, q = clip(floor(((RGB - mean) / std) / s_in + 0.5), -127, 127)
      (s_in = int8_meta.json["input"]["scale"]). Riêng OnnxBackend(input_kind="float") nhận (1, 3, H, W) float32 đã chuẩn hóa.
  ra: Logits(q = (4, h, w) int8, scale = int8_meta.json["output"]["scale"])  hoặc  Logits(f = (4, h, w) float32) với ONNX.
      B: H x W = 320 x 640, h x w = 40 x 80, lớp theo thứ tự types.CLASSES.

Backend:
  onnx     onnxruntime trên tệp ONNX FP32 của gói (BN đã gộp). Dùng để đối chiếu, không phải đường INT8.
  int8ref  bản tham chiếu số nguyên thuần NumPy (int8_ref.py của gói, giống repo/int8_ref.py) với int8_model.npz, int8_meta.json.
           Chậm (vài giây mỗi khung trên A53, chưa đo) nhưng trùng từng bit với int8_sim.py: dùng để kiểm IP.
  vart     DPU qua VART (TV2, bản đối chứng): CHƯA làm.
  ip       IP CNN tự thiết kế trên PL (TV2): CHƯA làm.
"""
import importlib.util
import json
import os

import numpy as np

from runtime import types as T
from runtime.preprocess import load_json


class Backend:
    input_kind = 'int8'        # 'int8' hoặc 'float'
    name = ''

    def run(self, x) -> T.Logits:
        raise NotImplementedError


class OnnxBackend(Backend):
    name = 'onnx'

    def __init__(self, onnx_path, input_scale=None, input_kind='float', threads=0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name
        self.input_kind = input_kind
        self.input_scale = input_scale
        assert input_kind in ('float', 'int8'), input_kind
        assert input_kind == 'float' or input_scale is not None, 'input_kind="int8" cần input_scale để giải lượng tử đầu vào'

    def run(self, x):
        if self.input_kind == 'int8':      # giải lượng tử đầu vào: mạng FP32 thấy đúng ảnh mà IP thấy
            assert x.dtype == np.int8, x.dtype
            x = x.astype(np.float32) * np.float32(self.input_scale)
        out = self.sess.run(None, {self.inp: np.ascontiguousarray(x, np.float32)})[0]
        return T.Logits(f=out[0].astype(np.float32))


def _load_module(path):
    spec = importlib.util.spec_from_file_location('int8_ref_runtime', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Int8RefBackend(Backend):
    name = 'int8ref'

    def __init__(self, pack_dir, int8_ref_py=None):
        py = int8_ref_py or os.path.join(pack_dir, 'int8_ref.py')
        if not os.path.exists(py):
            raise FileNotFoundError(f'không thấy {py} (gói INT8 phải có int8_ref.py, hoặc chỉ "int8_ref_py" tới repo/int8_ref.py)')
        self.ref = _load_module(py)
        z = np.load(os.path.join(pack_dir, 'int8_model.npz'))
        self.model = {k: z[k] for k in z.files}          # giải nén một lần (NpzFile giải nén lại mỗi lần truy cập)
        self.meta = load_json(os.path.join(pack_dir, 'int8_meta.json'))
        self.in_shape = tuple(self.meta['input']['shape'])
        self.out_name = self.meta['output']['name']
        self.out_scale = float(self.meta['output']['scale'])

    def run(self, x):
        assert x.dtype == np.int8 and x.shape == self.in_shape, (x.dtype, x.shape, self.in_shape)
        Tm = self.ref.run(self.model, self.meta, x)
        return T.Logits(q=Tm[self.out_name][0].astype(np.int8), scale=self.out_scale)


class VartBackend(Backend):
    """DPU (Vitis AI, VART) trên KV260: bản đối chứng của TV2. CHƯA làm.

    Cần: tệp .xmodel biên dịch từ cùng ONNX (Vitis AI quantizer có thang riêng, KHÔNG dùng int8_model.npz).
    Đầu vào DPU thường là NHWC int8 với thang lũy thừa 2 (fix_point): q_dpu = round(x * 2^fix_point), khác s_in của gói.
    Đầu ra: NHWC int8 (1, 40, 80, 4) với fix_point riêng -> chuyển về (4, 40, 80) và scale = 2^-fix_point.
    Bộ nhớ: dùng bộ đệm của runner (get_inputs()/get_outputs()), không cấp phát mỗi khung.
    """
    name = 'vart'

    def __init__(self, *a, **k):
        raise NotImplementedError('backend vart chưa làm: chờ TV2 (xmodel, fix_point vào/ra)')


class IpBackend(Backend):
    """IP CNN tự thiết kế trên PL (TV2). CHƯA làm: chờ hợp đồng bộ nhớ (E3, Y6).

    Giả định bố cục (CHƯA xác nhận với TV2):
      vào  DDR, int8, NCHW liền khối, (1, 3, 320, 640) = 614.400 byte; kênh RGB; thang s_in của int8_meta.json.
           IP có thể cần đệm kênh 3 -> 16 (câu hỏi 5 trong README của gói) và địa chỉ căn 64 byte.
      ra   DDR, int8, NCHW, (1, 4, 40, 80) = 12.800 byte; logit = q * s_out (int8_meta.json["output"]["scale"]).
      điều khiển: ghi địa chỉ vào/ra vào thanh ghi AXI-Lite, bật start, chờ ngắt hoặc thăm dò cờ done (driver UIO hoặc XRT).
    Kiểm IP: so từng bit đầu ra với Int8RefBackend trên golden_int8.npz của gói.
    """
    name = 'ip'

    def __init__(self, *a, **k):
        raise NotImplementedError('backend ip chưa làm: chờ TV2 (driver, địa chỉ, bố cục bộ nhớ)')


def make_backend(cfg, pack_dir, input_scale):
    """cfg: mục "backend" của config/pipeline_*.json (chuỗi tên hoặc dict có khóa "name")."""
    if isinstance(cfg, str):
        cfg = {'name': cfg}
    name = cfg['name']
    if name == 'onnx':
        return OnnxBackend(os.path.join(pack_dir, cfg['onnx_file']), input_scale, cfg.get('onnx_input', 'float'), cfg.get('threads', 0))
    if name == 'int8ref':
        return Int8RefBackend(pack_dir, cfg.get('int8_ref_py'))
    if name == 'vart':
        return VartBackend(cfg)
    if name == 'ip':
        return IpBackend(cfg)
    raise ValueError(f'backend không biết: {name}')
