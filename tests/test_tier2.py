"""Tầng 2 trên mặt nạ dựng tay. Lưới 128 x 64, vùng cắt trùng lưới nên tọa độ khung = tọa độ lưới (kiểm phép đổi riêng ở test_mapping)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import types as T                                     # noqa: E402
from runtime.postprocess import Tier2Config, compute_tier2, sample_cols, sample_rows   # noqa: E402
from runtime.preprocess import grid_to_frame                       # noqa: E402

W, H = 128, 64
G = grid_to_frame((0, 0, W, H), W, H)          # đơn vị
CFG = Tier2Config(n_cols=16, n_rows=4, row_top_frac=0.6, row_bottom_frac=0.98, ego_x_frac=0.5, merge_max_gap_frac=0.5)
XS = sample_cols(W, CFG)                       # 4, 12, ..., 124
YS = sample_rows(H, CFG)                       # 62, 54, 46, 38
FULL = T.ROW_FOUND | T.ROW_LEFT | T.ROW_RIGHT | T.ROW_CENTER


def road(ego, adj=None, top=32, shift=lambda y: 0):
    """Mặt nạ: hàng top..H-1 có làn đang đi ở [ego[0], ego[1]] (tính cả hai đầu), làn kề ở adj; phía trên là không đường."""
    m = np.zeros((H, W), np.uint8)
    for y in range(top, H):
        s = shift(y)
        m[y, ego[0] + s:ego[1] + s + 1] = T.EGO_LANE
        if adj:
            m[y, adj[0] + s:adj[1] + s + 1] = T.ADJ_LANE
    return m


class Tier2Test(unittest.TestCase):
    def check_rows(self, t2, left, right, flags):
        for k in range(len(YS)):
            self.assertEqual(int(t2.row_flags[k]), flags[k] if isinstance(flags, list) else flags, f'hàng {k}')
            for got, exp in ((t2.row_left[k], left[k]), (t2.row_right[k], right[k]), (t2.row_center[k], (left[k] + right[k]) / 2)):
                if np.isnan(exp) or exp is None:
                    self.assertTrue(np.isnan(got))
            if t2.row_flags[k] & T.ROW_LEFT:
                self.assertAlmostEqual(float(t2.row_left[k]), left[k])
            if t2.row_flags[k] & T.ROW_RIGHT:
                self.assertAlmostEqual(float(t2.row_right[k]), right[k])
            if t2.row_flags[k] & T.ROW_CENTER:
                self.assertAlmostEqual(float(t2.row_center[k]), (left[k] + right[k]) / 2)
        np.testing.assert_allclose(t2.row_y, YS.astype(np.float32))

    def test_sample_positions(self):
        self.assertEqual(XS.tolist(), list(range(4, 128, 8)))
        self.assertEqual(YS.tolist(), [62, 54, 46, 38])

    def test_straight_lane(self):
        t2 = compute_tier2(road((48, 79), (80, 111)), G, CFG)
        inside = (XS >= 48) & (XS <= 111)
        np.testing.assert_array_equal(t2.col_flags, np.where(inside, T.COL_VALID, 0))
        np.testing.assert_allclose(t2.col_y[inside], 31.5)
        self.assertTrue(np.isnan(t2.col_y[~inside]).all())
        np.testing.assert_allclose(t2.col_x, XS.astype(np.float32))
        self.check_rows(t2, [47.5] * 4, [79.5] * 4, FULL)

    def test_curved_lane(self):
        sh = lambda y: (63 - y) // 2                 # làn cong sang phải khi đi xa
        t2 = compute_tier2(road((48, 79), (80, 111), shift=sh), G, CFG)
        s = [sh(int(y)) for y in YS]                 # 0, 4, 8, 12
        self.assertEqual(s, [0, 4, 8, 12])
        self.check_rows(t2, [47.5 + v for v in s], [79.5 + v for v in s], FULL)
        self.assertTrue(np.all(np.diff(t2.row_center) > 0), 'tâm phải dịch dần sang phải')

    def test_obstacle_in_lane(self):
        m = road((48, 79), (80, 111))
        m[44:51, 58:70] = T.NOT_ROAD                  # vật cản ở hàng 44..50, cột 58..69
        t2 = compute_tier2(m, G, CFG)
        blocked = (XS >= 58) & (XS <= 69)            # cột 60, 68
        self.assertEqual(XS[blocked].tolist(), [60, 68])
        np.testing.assert_allclose(t2.col_y[blocked], 50.5)      # đi được từ đáy tới hàng 51
        free = (XS >= 48) & (XS <= 111) & ~blocked
        np.testing.assert_allclose(t2.col_y[free], 31.5)
        # hàng 46 cắt qua vật cản: hai đoạn làn được gộp, mép làn giữ nguyên, có cờ vật cản
        self.check_rows(t2, [47.5] * 4, [79.5] * 4, [FULL, FULL, FULL | T.ROW_OBSTACLE, FULL])

    def test_offset_left(self):
        """Xe lệch trái so với tâm làn: làn hiện ra lệch sang phải ảnh."""
        t2 = compute_tier2(road((72, 103), (104, 127)), G, CFG)
        self.check_rows(t2, [71.5] * 4, [103.5] * 4, FULL)
        self.assertGreater(float(t2.row_center[0]), CFG.ego_x_frac * W - 0.5)

    def test_offset_right(self):
        t2 = compute_tier2(road((24, 55), (56, 87)), G, CFG)
        self.check_rows(t2, [23.5] * 4, [55.5] * 4, FULL)
        self.assertLess(float(t2.row_center[0]), CFG.ego_x_frac * W - 0.5)

    def test_no_lane(self):
        m = np.zeros((H, W), np.uint8)
        m[32:, 20:108] = T.ROAD_NOT_OURS
        t2 = compute_tier2(m, G, CFG)
        self.assertTrue((t2.row_flags == 0).all())
        self.assertTrue(np.isnan(t2.row_center).all() and np.isnan(t2.row_left).all() and np.isnan(t2.row_right).all())
        self.assertTrue((t2.col_flags == 0).all(), 'mặc định lớp 1 không tính là đi được')
        c1 = compute_tier2(m, G, Tier2Config(**dict(CFG.__dict__, drivable_include_class1=True)))
        inside = (XS >= 20) & (XS < 108)
        np.testing.assert_array_equal(c1.col_flags, np.where(inside, T.COL_VALID, 0))
        np.testing.assert_allclose(c1.col_y[inside], 31.5)

    def test_lane_touches_border(self):
        t2 = compute_tier2(road((0, 40), (41, 80)), G, CFG)
        fl = T.ROW_FOUND | T.ROW_RIGHT
        self.check_rows(t2, [np.nan] * 4, [40.5] * 4, fl)
        self.assertTrue(np.isnan(t2.row_center).all(), 'mép trái chạm biên: tâm không hợp lệ')

    def test_free_to_top(self):
        m = road((48, 79), (80, 111), top=0)
        t2 = compute_tier2(m, G, CFG)
        inside = (XS >= 48) & (XS <= 111)
        np.testing.assert_array_equal(t2.col_flags, np.where(inside, T.COL_VALID | T.COL_REACHES_TOP, 0))
        np.testing.assert_allclose(t2.col_y[inside], -0.5)

    def test_noise_blob_not_merged(self):
        """Một mảng lớp 2 lẻ ngăn cách bằng làn kề (không phải vật cản) không được gộp vào làn đang đi."""
        m = road((48, 79), (80, 111))
        m[32:, 100:111] = T.EGO_LANE
        t2 = compute_tier2(m, G, CFG)
        self.check_rows(t2, [47.5] * 4, [79.5] * 4, FULL)

    def test_scaled_mapping_b(self):
        """Lưới 640 x 320 của B phủ vùng cắt (0, 0, 1280, 600): mép làn đổi sang pixel khung đúng tỉ lệ."""
        g = grid_to_frame((0, 0, 1280, 600), 640, 320)
        m = np.zeros((320, 640), np.uint8)
        m[160:, 240:400] = T.EGO_LANE
        t2 = compute_tier2(m, g, Tier2Config())
        ok = t2.row_flags & T.ROW_CENTER > 0
        self.assertTrue(ok.all())
        np.testing.assert_allclose(t2.row_left, 239.5 * 2 + 0.5)    # ranh giới lưới 239,5 -> khung 479,5
        np.testing.assert_allclose(t2.row_right, 399.5 * 2 + 0.5)
        np.testing.assert_allclose(t2.row_center, 319.5 * 2 + 0.5)


if __name__ == '__main__':
    unittest.main()
