"""Hậu xử lý: phóng bilinear NumPy so với F.interpolate của torch (khi có torch), bản nhanh, phóng nearest."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import types as T                                              # noqa: E402
from runtime.postprocess import (GRID_DIV, MODES, Postprocessor, fast_deviation, mask_fast,   # noqa: E402
                                 mask_standard, nearest_resize, tier2_deviation, upsample_bilinear)
from runtime.preprocess import apply, grid_to_frame                          # noqa: E402

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


def assert_labels_match_up_to_ties(tc, got_mask, ref_logits, tol=1e-5, msg=''):
    """Nhãn NumPy so với logit torch (C, H, W): lớp NumPy chọn phải nằm trong 2 * tol của logit lớn nhất theo torch.
    Float32 trên máy khác (CPU, NumPy, torch khác) lệch nhau bậc làm tròn, nên nhãn khác nhau CHỈ được ở pixel hai lớp đầu gần như hòa
    (logit lượng tử làm tròn trung bình có nhiều hòa chính xác, nội suy giữ nguyên hòa). Ngưỡng tỉ lệ pixel trùng thì không bền: lưới 80x160 chỉ có 12.800 pixel."""
    chosen = np.take_along_axis(ref_logits, got_mask[None].astype(np.int64), 0)[0]
    shortfall = ref_logits.max(0) - chosen
    n_diff = int((got_mask != ref_logits.argmax(0)).sum())
    tc.assertLessEqual(float(shortfall.max()), 2 * tol, f'{msg}: có pixel chọn lớp kém hơn lớp lớn nhất {shortfall.max():.2e} (> {2 * tol:.0e}); {n_diff} pixel khác nhãn torch')
    return n_diff


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
            m = mask_standard(lg, H, W)
            n_diff = assert_labels_match_up_to_ties(self, m, ref, msg=f'{h}x{w}->{H}x{W}')
            print(f'\n  bilinear {h}x{w}->{H}x{W}: trùng bit {float((got == ref).mean()):.6f}, {n_diff} / {m.size} pixel khác nhãn torch (đều là hòa)')

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


def lane_logits():
    """Logit trơn (float) có một làn lớp 2 ở giữa, bờ liên tục: nghiệm của f2 = 0 là cùng một đường cong ở mọi lưới phóng tới."""
    y, x = np.mgrid[0:40, 0:80].astype(np.float32)
    f = np.zeros((4, 40, 80), np.float32)
    f[2] = np.minimum(3 - 0.5 * np.abs(x - 40), 0.4 * (y - 10))
    return T.Logits(f=f)


class GridModesTest(unittest.TestCase):
    G2F = grid_to_frame((0, 0, 1280, 600), 640, 320)
    HW = (320, 640)

    def test_modes_registered(self):
        self.assertEqual(set(MODES), {'standard', 'fast', 'half', 'quarter'})
        with self.assertRaises(AssertionError):
            Postprocessor({'mode': 'nua'})

    def test_grid_matrix_is_that_of_the_coarse_grid(self):
        lg = logits(np.random.default_rng(5))
        for mode, d in GRID_DIV.items():
            m, g2f = Postprocessor({'mode': mode}).mask(lg, self.HW, self.G2F)
            self.assertEqual(m.shape, (320 // d, 640 // d))
            np.testing.assert_allclose(g2f, grid_to_frame((0, 0, 1280, 600), 640 // d, 320 // d), atol=1e-9, err_msg=mode)
            # tâm pixel (0, 0) của lưới thưa là tâm khối d x d của lưới ảnh vào
            np.testing.assert_allclose(apply(g2f, 0, 0), apply(self.G2F, (d - 1) / 2, (d - 1) / 2), atol=1e-9)

    @unittest.skipIf(torch is None, 'không có torch')
    def test_coarse_masks_match_torch(self):
        lg = logits(np.random.default_rng(6))
        x = lg.as_float()
        for mode, d in GRID_DIV.items():
            m, _ = Postprocessor({'mode': mode}).mask(lg, self.HW, self.G2F)
            ref = F.interpolate(torch.from_numpy(x[None]), size=(320 // d, 640 // d), mode='bilinear', align_corners=False)[0].numpy()
            self.assertEqual(m.shape, ref.shape[1:])
            assert_labels_match_up_to_ties(self, m, ref, msg=mode)

    def test_tier2_stays_close_to_standard(self):
        lg = lane_logits()
        res = {}
        for mode in ('standard', 'half', 'quarter'):
            ps = Postprocessor({'mode': mode})
            m, g2f = ps.mask(lg, self.HW, self.G2F)
            res[mode] = ps.tier2(m, g2f)
        a = res['standard']
        self.assertTrue((a.row_flags & T.ROW_CENTER > 0).sum() >= 5, 'mặt nạ thử phải có làn')
        for mode, tol_x, tol_y in (('half', 12, 12), ('quarter', 20, 20)):     # pixel khung: 1 pixel lưới thưa = 4 hoặc 8 ngang, 3,75 hoặc 7,5 dọc
            b = res[mode]
            d = tier2_deviation(a, b)
            self.assertEqual(d['row_center_valid_mismatch'], 0, mode)
            self.assertEqual(d['col_valid_mismatch'], 0, mode)
            self.assertLess(max(d['row_center_absdiff_px']), tol_x, mode)
            self.assertLess(max(d['col_y_absdiff_px']), tol_y, mode)
            np.testing.assert_array_equal(a.row_flags, b.row_flags)

    def test_tier2_deviation_zero_for_same(self):
        a = Postprocessor({}).tier2(*Postprocessor({}).mask(lane_logits(), self.HW, self.G2F))
        d = tier2_deviation(a, a)
        self.assertEqual(max(d['row_center_absdiff_px']), 0)
        self.assertEqual(d['col_valid_mismatch'] + d['row_center_valid_mismatch'], 0)


if __name__ == '__main__':
    unittest.main()
