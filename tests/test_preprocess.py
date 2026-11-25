"""Tiền xử lý: resize NumPy trùng từng byte với PIL BILINEAR (khi có PIL), bảng tra lượng tử trùng công thức của int8_sim."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.preprocess import PilBilinear, Preprocessor    # noqa: E402

try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import cv2
except ImportError:
    cv2 = None

MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]
CFG = dict(frame_wh=[1280, 720], crop_ltrb=[0, 0, 1280, 600], out_wh=[640, 320], resize='pil_bilinear', mean=MEAN, std=STD,
           input_scale=0.020787402400820273, frame_color='RGB', on_size_mismatch='scale_crop')


def frame(rng, w=1280, h=720):
    """Ảnh giả có cả vùng trơn và cạnh sắc (nhiễu thuần không thử được hết các nhánh làm tròn)."""
    x = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    x[: h // 3] = (np.arange(w)[None, :, None] * 255 // max(w - 1, 1)).astype(np.uint8)
    x[h // 2: h // 2 + 7, :, 0] = 255
    return x


class PreprocessTest(unittest.TestCase):
    @unittest.skipIf(Image is None, 'không có PIL')
    def test_resize_matches_pil(self):
        rng = np.random.default_rng(0)
        for (iw, ih), (ow, oh) in [((1280, 600), (640, 320)), ((1280, 600), (320, 160)), ((1280, 576), (1280, 576)), ((640, 480), (640, 320)),
                                   ((100, 50), (333, 177)), ((37, 91), (16, 200)), ((1280, 600), (1280, 576))]:
            x = frame(rng, iw, ih)
            ref = np.asarray(Image.fromarray(x).resize((ow, oh), Image.BILINEAR))
            got = PilBilinear(iw, ih, ow, oh)(x)
            self.assertEqual(got.shape, ref.shape)
            self.assertEqual(int((got != ref).sum()), 0, f'{iw}x{ih} -> {ow}x{oh}')

    @unittest.skipIf(Image is None, 'không có PIL')
    def test_full_preprocess_matches_training_rule(self):
        """Khung 1280x720 -> giống make_hd_pilot.one: crop (0, 0, 1280, 600) rồi PIL BILINEAR 640x320."""
        rng = np.random.default_rng(1)
        f = frame(rng)
        p = Preprocessor(CFG)(f)
        ref = np.asarray(Image.fromarray(f).crop((0, 0, 1280, 600)).resize((640, 320), Image.BILINEAR))
        np.testing.assert_array_equal(p.image, ref)
        self.assertEqual(p.x_int8.shape, (1, 3, 320, 640))
        self.assertEqual(p.x_int8.dtype, np.int8)
        self.assertTrue(p.x_int8.flags['C_CONTIGUOUS'])
        self.assertFalse(p.scaled_crop)

    def test_quantize_matches_int8_sim(self):
        pre = Preprocessor(CFG)
        img = np.random.default_rng(2).integers(0, 256, (320, 640, 3)).astype(np.uint8)
        img[0, :256, :] = np.arange(256)[:, None]           # đủ 256 giá trị ở mỗi kênh
        xf = (img.astype(np.float32).transpose(2, 0, 1)[None] - np.array(MEAN, np.float32).reshape(1, 3, 1, 1)) / np.array(STD, np.float32).reshape(1, 3, 1, 1)
        ref = np.clip(np.floor(xf / CFG['input_scale'] + 0.5), -127, 127).astype(np.int64)       # int8_sim.forward
        np.testing.assert_array_equal(pre.quantize(img).astype(np.int64), ref)
        import runtime.preprocess as rp
        saved, rp._cv2 = rp._cv2, None          # đường NumPy (board không có OpenCV) phải cho cùng kết quả
        try:
            np.testing.assert_array_equal(pre.quantize(img).astype(np.int64), ref)
        finally:
            rp._cv2 = saved
        np.testing.assert_array_equal(pre.to_float(img), xf)

    @unittest.skipIf(cv2 is None, 'không có OpenCV')
    def test_cv2_fast_path_equals_direct_resize(self):
        f = frame(np.random.default_rng(7))
        for method, flag in (('cv2_linear', cv2.INTER_LINEAR), ('cv2_area', cv2.INTER_AREA)):
            pre = Preprocessor(dict(CFG, resize=method))
            p = pre(f)
            ref = cv2.resize(np.ascontiguousarray(f[:600, :1280]), (640, 320), interpolation=flag)       # RGB trong, RGB ra
            np.testing.assert_array_equal(p.image, ref, method)
            np.testing.assert_array_equal(p.x_int8, pre.quantize(ref), method)
            b = pre(np.ascontiguousarray(f[:, :, ::-1]), color='BGR')                                    # BGR vào cho cùng kết quả
            np.testing.assert_array_equal(b.x_int8, p.x_int8, method)
            self.assertEqual(p.chw.shape, (3, 320, 640))
            self.assertTrue(p.chw.flags.c_contiguous and p.x_int8.flags.c_contiguous)
            np.testing.assert_allclose(p.grid_to_frame, Preprocessor(CFG)(f).grid_to_frame)           # cùng ma trận tọa độ với bản PIL
        small = Preprocessor(dict(CFG, resize='cv2_linear'))(frame(np.random.default_rng(8), 640, 360))   # khung khác cỡ: vùng cắt co theo
        self.assertEqual(small.crop_ltrb, (0, 0, 640, 300))
        self.assertEqual(small.x_int8.shape, (1, 3, 320, 640))

    @unittest.skipIf(cv2 is None or Image is None, 'cần OpenCV và PIL')
    def test_cv2_linear_differs_from_training_resize(self):
        """cv2_linear KHÔNG khử răng cưa như PIL: ảnh khác ảnh lúc huấn luyện. Test này khóa sự thật đó (không phải lỗi)."""
        f = frame(np.random.default_rng(9))
        a, b = Preprocessor(CFG)(f).image, Preprocessor(dict(CFG, resize='cv2_linear'))(f).image
        self.assertGreater(int((a != b).any(-1).sum()), 0)

    def test_bgr_and_size_mismatch(self):
        pre = Preprocessor(dict(CFG, resize='pil_bilinear'))
        f = frame(np.random.default_rng(3))
        a = pre(f)
        b = pre(np.ascontiguousarray(f[:, :, ::-1]), color='BGR')
        np.testing.assert_array_equal(a.x_int8, b.x_int8)
        small = pre(frame(np.random.default_rng(4), 640, 360))
        self.assertEqual(small.crop_ltrb, (0, 0, 640, 300))
        self.assertTrue(small.scaled_crop)
        self.assertEqual(small.x_int8.shape, (1, 3, 320, 640))
        with self.assertRaises(ValueError):
            Preprocessor(dict(CFG, on_size_mismatch='error'))(frame(np.random.default_rng(5), 640, 360))

    def test_meta_shape_check(self):
        import json
        import tempfile
        def put(d, shape):
            with open(os.path.join(d, 'm.json'), 'w') as f:
                json.dump({'input': {'scale': 0.03, 'shape': shape}}, f)
        with tempfile.TemporaryDirectory() as d:
            put(d, [1, 3, 160, 320])
            with self.assertRaises(ValueError):
                Preprocessor(dict(CFG, input_scale=None, int8_meta='m.json'), d)
            put(d, [1, 3, 320, 640])
            self.assertEqual(Preprocessor(dict(CFG, input_scale=None, int8_meta='m.json'), d).input_scale, 0.03)


if __name__ == '__main__':
    unittest.main()
