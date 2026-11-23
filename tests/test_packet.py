"""Đóng gói rồi giải gói perception_v0 phải ra đúng giá trị."""
import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import types as T     # noqa: E402


def sample(rng, N=32, M=8, with_extra=False):
    f = lambda n: np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 500, n)).astype(np.float32)
    t2 = T.Tier2(f(N), f(N), rng.integers(0, 4, N).astype(np.uint8), f(M), f(M), f(M), f(M), rng.integers(0, 32, M).astype(np.uint8))
    t3 = T.Tier3.empty(N, M)
    r = T.PerceptionResult(seq=123456, t_capture_ns=1_700_000_000_123_456_789, frame_wh=(1280, 720), grid_wh=(640, 320), tier2=t2, tier3=t3,
                           model='fsnet_m_p1s5', timings_us=dict(pre=1234, infer=56789, post=4321, tier2=99), drivable_class1=True)
    if with_extra:
        r.logits = T.Logits(q=rng.integers(-127, 128, (4, 40, 80)).astype(np.int8), scale=0.2528)
        r.logits_grid_to_frame = np.array([[16, 0, 7.5], [0, 15, 7.0], [0, 0, 1.0]])
        r.mask = T.Mask(rng.integers(0, 4, (320, 640)).astype(np.uint8), np.array([[2, 0, 0.5], [0, 1.875, 0.4375], [0, 0, 1.0]]))
    return r


class PacketTest(unittest.TestCase):
    def assert_same(self, a, b):
        for k in ('col_x', 'col_y', 'col_flags', 'row_y', 'row_left', 'row_right', 'row_center', 'row_flags'):
            np.testing.assert_array_equal(getattr(a.tier2, k), getattr(b.tier2, k), err_msg=k)
        for k in ('col_lat_cm', 'col_fwd_cm', 'col_flags', 'row_fwd_cm', 'row_left_cm', 'row_right_cm', 'row_center_cm', 'row_flags'):
            np.testing.assert_array_equal(getattr(a.tier3, k), getattr(b.tier3, k), err_msg=k)
        for k in ('seq', 't_capture_ns', 'frame_wh', 'grid_wh', 'model', 'timings_us', 'drivable_class1', 'fast', 'tier3_calibrated'):
            self.assertEqual(getattr(a, k), getattr(b, k), k)

    def test_header_size(self):
        self.assertEqual(T.HEADER.size, 80)

    def test_roundtrip_basic(self):
        r = sample(np.random.default_rng(0))
        buf = T.pack(r, t_publish_ns=42)
        self.assertEqual(len(buf), T.packet_size(32, 8))
        self.assertEqual(len(buf), 932)
        self.assertEqual(buf[:4], b'PERC')
        u = T.unpack(buf)
        self.assert_same(r, u)
        self.assertEqual(u.t_publish_ns, 42)
        self.assertIsNone(u.logits)
        self.assertIsNone(u.mask)
        self.assertTrue(np.isnan(u.tier3.col_fwd_cm).all() and (u.tier3.row_flags == 0).all())

    def test_roundtrip_with_logits_and_mask(self):
        r = sample(np.random.default_rng(1), with_extra=True)
        buf = T.pack(r, include_logits=True, include_mask=True)
        self.assertEqual(len(buf), T.packet_size(32, 8, (4, 40, 80), (320, 640)))
        u = T.unpack(buf)
        self.assert_same(r, u)
        np.testing.assert_array_equal(u.logits.q, r.logits.q)
        self.assertAlmostEqual(u.logits.scale, 0.2528, places=6)
        np.testing.assert_allclose(u.logits_grid_to_frame, r.logits_grid_to_frame, rtol=1e-7)
        np.testing.assert_array_equal(u.mask.m, r.mask.m)
        np.testing.assert_allclose(u.mask.grid_to_frame, r.mask.grid_to_frame, rtol=1e-7)

    def test_other_sizes(self):
        for N, M in ((1, 1), (7, 3), (64, 16)):
            r = sample(np.random.default_rng(N), N, M)
            buf = T.pack(r)
            self.assertEqual(len(buf) % 4, 0)
            self.assert_same(r, T.unpack(buf))

    def test_corruption_detected(self):
        buf = bytearray(T.pack(sample(np.random.default_rng(2))))
        buf[100] ^= 0x01
        with self.assertRaises(T.PacketError):
            T.unpack(bytes(buf))
        good = T.pack(sample(np.random.default_rng(2)))
        with self.assertRaises(T.PacketError):
            T.unpack(good[:-1])
        with self.assertRaises(T.PacketError):
            T.unpack(b'XXXX' + good[4:])

    def test_header_fields_little_endian(self):
        buf = T.pack(sample(np.random.default_rng(3)), t_publish_ns=7)
        magic, ver, hsz, seq, tc, tp = struct.unpack_from('<4sHHIQQ', buf, 0)
        self.assertEqual((magic, ver, hsz, seq, tc, tp), (b'PERC', T.VERSION, 80, 123456, 1_700_000_000_123_456_789, 7))


if __name__ == '__main__':
    unittest.main()
