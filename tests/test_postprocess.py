"""Hậu xử lý: phóng bilinear NumPy so với F.interpolate của torch (khi có torch), bản nhanh, phóng nearest."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import types as T                                              # noqa: E402
from runtime.postprocess import (fast_deviation, mask_fast, mask_standard,   # noqa: E402
                                 nearest_resize, upsample_bilinear)
from runtime.preprocess import grid_to_frame                                 # noqa: E402

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


def logits(rng, h=40, w=80, smooth=True):
    q = rng.integers(-127, 128, (4, h, w))
    if smooth:      # logit thật trơn theo không gian: lấy trung bình khối để có vùng lớp liền
        q = np.round((q + np.roll(q, 1, 1) + np.roll(q, 1, 2) + np.roll(q, (1, 1), (1, 2))) / 4)
    return T.Logits(q=q.astype(np.int8), scale=0.25281683854230746)


class PostprocessTest(unittest.TestCase):
    def test_constant_and_ramp(self):
        x = np.full((2, 5, 7), 3.25, np.float32)
        np.testing.assert_array_equal(upsample_bilinear(x, 40, 56), 3.25)
        r = np.arange(8, dtype=np.float32)[None, None, :].repeat(2, 1)        # (1, 2, 8): dốc theo chiều ngang
        up = upsample_bilinear(r, 2, 64)[0, 0]
        exp = np.clip((np.arange(64) + 0.5) / 8 - 0.5, 0, 7)                   # align_corners=False, kẹp ở biên
        np.testing.assert_allclose(up, exp, atol=1e-6)

    @unittest.skipIf(torch is None, 'không có torch')
    def test_matches_torch(self):
        rng = np.random.default_rng(0)
        for (h, w), (H, W) in [((40, 80), (320, 640)), ((40, 80), (160, 320)), ((7, 13), (50, 31))]:
            lg = logits(rng, h, w)
            x = lg.as_float()
            ref = F.interpolate(torch.from_numpy(x[None]), size=(H, W), mode='bilinear', align_corners=False)[0].numpy()
            got = upsample_bilinear(x, H, W)
            self.assertLess(float(np.abs(got - ref).max()), 1e-5, f'{h}x{w} -> {H}x{W}')
            m_ref = ref.argmax(0).astype(np.uint8)
            agree = float((mask_standard(lg, H, W) == m_ref).mean())
            print(f'\n  bilinear {h}x{w}->{H}x{W}: trùng bit {float((got == ref).mean()):.6f}, trùng nhãn {agree:.6f}')
            self.assertGreaterEqual(agree, 0.9999)

    @unittest.skipIf(torch is None, 'không có torch')
    def test_nearest_matches_torch(self):
        m = np.random.default_rng(1).integers(0, 4, (320, 640)).astype(np.uint8)
        for H, W in [(160, 320), (40, 80), (640, 1280), (100, 333)]:
            ref = F.interpolate(torch.from_numpy(m)[None, None].float(), size=(H, W), mode='nearest')[0, 0].numpy().astype(np.uint8)
            np.testing.assert_array_equal(nearest_resize(m, H, W), ref)

    def test_fast_mask(self):
        lg = logits(np.random.default_rng(2))
        np.testing.assert_array_equal(mask_fast(lg), lg.as_float().argmax(0))
        d = fast_deviation(lg, (320, 640), grid_to_frame((0, 0, 1280, 600), 640, 320))
        self.assertGreater(d['pixel_agree'], 0.5)
        self.assertIn('row_center_absdiff_px', d)


if __name__ == '__main__':
    unittest.main()
