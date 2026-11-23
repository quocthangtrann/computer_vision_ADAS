"""Ma trận đổi tọa độ lưới -> khung với các điểm biết trước (quy ước tâm pixel ở số nguyên)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.postprocess import logit_grid_to_frame    # noqa: E402
from runtime.preprocess import apply, grid_to_frame    # noqa: E402

CROP_B = (0, 0, 1280, 600)


class MappingTest(unittest.TestCase):
    def pt(self, M, x, y, ex, ey):
        gx, gy = apply(M, x, y)
        self.assertAlmostEqual(float(gx), ex, places=9)
        self.assertAlmostEqual(float(gy), ey, places=9)

    def test_b_input_grid(self):
        M = grid_to_frame(CROP_B, 640, 320)             # sx = 2, sy = 1,875
        self.pt(M, 0, 0, 0.5, 0.4375)                   # tâm pixel lưới (0, 0) = tâm của khối 2 x 1,875 pixel khung đầu tiên
        self.pt(M, -0.5, -0.5, -0.5, -0.5)              # góc trên trái của vùng cắt
        self.pt(M, 639.5, 319.5, 1279.5, 599.5)         # góc dưới phải của vùng cắt (hàng 600 trở xuống không vào mạng)
        self.pt(M, 319.5, 159.5, 639.5, 299.5)          # tâm vùng cắt

    def test_b_logit_grid(self):
        net = grid_to_frame(CROP_B, 640, 320)
        L = logit_grid_to_frame(net, (320, 640), (40, 80))
        np.testing.assert_allclose(L, grid_to_frame(CROP_B, 80, 40), atol=1e-12)
        self.pt(L, 0, 0, 7.5, 7.0)                      # ô logit 16 x 15 pixel khung
        self.pt(L, 79, 39, 1271.5, 592.0)

    def test_offset_crop(self):
        M = grid_to_frame((100, 50, 1100, 550), 500, 250)    # sx = sy = 2
        self.pt(M, 0, 0, 100.5, 50.5)
        self.pt(M, 499, 249, 1098.5, 548.5)

    def test_inverse_roundtrip(self):
        M = grid_to_frame(CROP_B, 640, 320)
        rng = np.random.default_rng(0)
        x, y = rng.uniform(-1, 641, 50), rng.uniform(-1, 321, 50)
        fx, fy = apply(M, x, y)
        bx, by = apply(np.linalg.inv(M), fx, fy)
        np.testing.assert_allclose(bx, x, atol=1e-9)
        np.testing.assert_allclose(by, y, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
