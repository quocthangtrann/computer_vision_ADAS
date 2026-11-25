"""Pipeline.run_stream (chạy gối đầu hai luồng): kết quả y hệt tuần tự, đúng thứ tự, lỗi truyền lên, bỏ dở không treo luồng.
Dùng backend giả (không cần gói INT8), chế độ hậu xử lý half để kiểm luôn cờ và lưới."""
import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import types as T                    # noqa: E402
from runtime.infer import Backend                 # noqa: E402
from runtime.pipeline import Pipeline             # noqa: E402

PRE = dict(frame_wh=[1280, 720], crop_ltrb=[0, 0, 1280, 600], out_wh=[640, 320], resize='pil_bilinear',
           mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], input_scale=0.0208, frame_color='RGB')


class FakeBackend(Backend):
    """Logit xác định từ ảnh vào: cùng khung cho cùng kết quả."""
    name = 'fake'
    input_kind = 'int8'

    def __init__(self, fail_at=None):
        self.n, self.fail_at = 0, fail_at

    def run(self, x):
        self.n += 1
        if self.fail_at is not None and self.n == self.fail_at:
            raise RuntimeError('backend hỏng')
        r = x[0, 0].astype(np.int32)[::8, ::8][:40, :80]
        q = np.stack([-r, r // 2, (r * 3) // 4, np.roll(r, 7, 1)]).astype(np.int8)
        return T.Logits(q=q, scale=0.25)


def frames(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        f = rng.integers(0, 256, (720, 1280, 3)).astype(np.uint8)
        f[200:600, 300:900] = rng.integers(60, 200)         # có vùng phẳng để tầng 2 có làn
        out.append(f)
    return out


def pl(mode='half', **kw):
    return Pipeline(dict(preprocess=PRE, postprocess=dict(mode=mode)), backend=FakeBackend(**kw))


def same(a, b):
    ta, tb = a.tier2, b.tier2
    return all(np.array_equal(getattr(ta, k), getattr(tb, k), equal_nan=True)
               for k in ('col_x', 'col_y', 'col_flags', 'row_y', 'row_left', 'row_right', 'row_center', 'row_flags'))


class StreamTest(unittest.TestCase):
    def test_equals_sequential_and_ordered(self):
        fr = frames(7)
        seq = [pl().process(f, t_capture_ns=i + 1, keep=True) for i, f in enumerate(fr)]
        p = pl()
        ovl = list(p.run_stream([(f, i + 1) for i, f in enumerate(fr)], keep=True))
        self.assertEqual([r.seq for r in ovl], list(range(1, 8)))
        self.assertEqual([r.t_capture_ns for r in ovl], list(range(1, 8)))
        for a, b in zip(seq, ovl):
            self.assertTrue(same(a, b))
            np.testing.assert_array_equal(a.mask.m, b.mask.m)
        self.assertEqual(ovl[0].grid_wh, (320, 160))
        self.assertTrue(ovl[0].fast)            # không phải bản chuẩn
        self.assertEqual(set(ovl[0].timings_us), {'pre', 'infer', 'post', 'tier2'})
        self.assertTrue(all(r.timings_us['pre'] > 0 for r in ovl))

    def test_no_overlap_option_and_standard_flag(self):
        fr = frames(3, 1)
        a = list(pl('standard').run_stream(fr, overlap=False))
        b = list(pl('standard').run_stream(fr, overlap=True))
        self.assertTrue(all(same(x, y) for x, y in zip(a, b)))
        self.assertFalse(a[0].fast)
        self.assertEqual(a[0].grid_wh, (640, 320))

    def test_packet_of_coarse_grid_roundtrips(self):
        r = pl('quarter').process(frames(1)[0], t_capture_ns=9, keep=True)
        self.assertEqual(r.grid_wh, (160, 80))
        u = T.unpack(T.pack(r, include_logits=False, include_mask=True))
        np.testing.assert_array_equal(u.mask.m, r.mask.m)
        np.testing.assert_allclose(u.mask.grid_to_frame, r.mask.grid_to_frame, atol=1e-6)
        self.assertEqual(u.grid_wh, (160, 80))

    def test_source_error_propagates(self):
        fr = frames(2)

        def src():
            yield fr[0]
            raise ValueError('camera hỏng')
        with self.assertRaises(ValueError):
            list(pl().run_stream(src()))

    def test_backend_error_propagates_and_thread_stops(self):
        before = threading.active_count()
        with self.assertRaises(RuntimeError):
            list(pl(fail_at=2).run_stream(frames(6)))
        self.assertEqual(threading.active_count(), before)

    def test_early_close_does_not_hang(self):
        before = threading.active_count()
        g = pl().run_stream(frames(10))
        next(g)
        g.close()
        self.assertEqual(threading.active_count(), before)


if __name__ == '__main__':
    unittest.main()
