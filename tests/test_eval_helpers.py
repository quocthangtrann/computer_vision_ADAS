"""Hàm thống kê của runtime.eval_dataset (không cần dữ liệu): tiêu chí đặt trước phải tính đúng chiều."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import types as T                                                  # noqa: E402
from runtime.eval_dataset import (iou_drop, lane_add, lane_summary, mode_verdict, new_lane,   # noqa: E402
                                  resize_verdict)


def t2(center0):
    """Tầng 2 chỉ để lane_add đọc hàng 0; center0 None = hàng 0 không có tâm."""
    n, m = 4, 2
    t = T.Tier2.__new__(T.Tier2)
    t.row_flags = np.array([T.ROW_CENTER if center0 is not None else 0, 0], np.uint8)
    t.row_center = np.array([np.nan if center0 is None else center0, np.nan])
    return t


class EvalHelpersTest(unittest.TestCase):
    def test_lane_strict_counts_missing_center_as_error(self):
        L = new_lane()
        gt = t2(500.0)
        lane_add(L, t2(520.0), gt)      # lệch 20: tốt
        lane_add(L, t2(700.0), gt)      # lệch 200: lỗi
        lane_add(L, t2(None), gt)       # nhãn có tâm, dự đoán không: lỗi (chặt)
        lane_add(L, t2(480.0), t2(None))    # nhãn không có tâm: dự đoán thừa, không thuộc mẫu số
        lane_add(L, t2(None), None)         # không có nhãn HD: bỏ qua
        s = lane_summary(L)
        self.assertEqual((s['both'], s['gt_only'], s['pred_only']), (2, 1, 1))
        self.assertAlmostEqual(s['frac_gt100'], 0.5)                  # chỉ tính ảnh cả hai có tâm: 1 / 2
        self.assertAlmostEqual(s['frac_gt100_strict'], 2 / 3)         # (1 lỗi + 1 thiếu) / 3 ảnh nhãn có tâm
        self.assertAlmostEqual(s['mean_px'], 110.0)
        self.assertIsNone(lane_summary(new_lane()))

    def test_iou_drop_sign_and_road(self):
        ref = ([0.9, 0.5, 0.8, 0.6], 0.90)
        self.assertAlmostEqual(iou_drop(ref, ([0.9, 0.5, 0.79, 0.6], 0.90)), 0.01)         # kém hơn: dương
        self.assertAlmostEqual(iou_drop(ref, ([0.91, 0.51, 0.81, 0.61], 0.91)), -0.01)     # tốt hơn: âm
        self.assertAlmostEqual(iou_drop(ref, ([0.9, 0.5, 0.8, 0.6], 0.8985)), 0.0015)      # IoU đường cũng tính
        self.assertAlmostEqual(iou_drop(ref, ref), 0.0)

    def test_mode_verdict_threshold_and_minimum(self):
        base = dict(both=100, frac_gt100_strict=0.16)
        ok = dict(both=100, frac_gt100_strict=0.17)         # xấu hơn đúng 0,01: chưa vượt ngưỡng
        bad = dict(both=100, frac_gt100_strict=0.171)       # xấu hơn 0,011: vượt
        self.assertTrue(mode_verdict(ok, base, 0.01 + 1e-9, 30).startswith('DÙNG ĐƯỢC'))
        self.assertTrue(mode_verdict(bad, base, 0.01, 30).startswith('KHÔNG DÙNG'))
        self.assertTrue(mode_verdict(dict(both=100, frac_gt100_strict=0.10), base, 0.01, 30).startswith('DÙNG ĐƯỢC'))   # tốt hơn
        self.assertTrue(mode_verdict(ok, dict(both=5, frac_gt100_strict=1.0), 0.01, 30).startswith('chưa kết luận'))   # mốc gần như không thấy làn
        self.assertTrue(mode_verdict(dict(both=5, frac_gt100_strict=1.0), base, 0.01, 30).startswith('chưa kết luận'))
        self.assertTrue(mode_verdict(None, base, 0.01, 30).startswith('chưa kết luận'))

    def test_resize_verdict_threshold(self):
        self.assertTrue(resize_verdict(0.0011, 0.001, 'pil_bilinear').startswith('GIỮ pil_bilinear'))
        self.assertTrue(resize_verdict(0.0010, 0.001, 'pil_bilinear').startswith('ĐẠT'))
        self.assertTrue(resize_verdict(-0.002, 0.001, 'pil_bilinear').startswith('ĐẠT'))        # tốt hơn mốc


if __name__ == '__main__':
    unittest.main()
