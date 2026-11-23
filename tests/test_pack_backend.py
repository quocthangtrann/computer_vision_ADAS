"""Kiểm với gói INT8 thật (chỉ chạy khi có gói): đặt PERCEPTION_PACK=thư_mục_gói (ví dụ handoff_s3b_fsnet_m_p1s5_640x320).

- Int8RefBackend trên golden_int8.npz phải ra đúng test_int8_logits_quarter.npy (từng bit).
- Nếu đặt thêm PERCEPTION_TEST_DIR=thư_mục ảnh test của gói (data/v2/hd_test_640x320): Preprocessor.quantize(ảnh) == golden x.
- Pipeline chạy trọn một khung giả 1280x720 với hai backend, gói giải ra đúng.
"""
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from runtime import types as T               # noqa: E402
from runtime.infer import Int8RefBackend     # noqa: E402
from runtime.pipeline import Pipeline        # noqa: E402
from runtime.preprocess import load_json     # noqa: E402

PACK = os.environ.get('PERCEPTION_PACK')


@unittest.skipUnless(PACK and os.path.isdir(PACK), 'chưa đặt PERCEPTION_PACK')
class PackBackendTest(unittest.TestCase):
    def test_int8ref_matches_golden(self):
        be = Int8RefBackend(PACK)
        gold = np.load(os.path.join(PACK, 'golden_int8.npz'))
        lg = np.load(os.path.join(PACK, 'test_int8_logits_quarter.npy'))
        ks = sorted(int(k[1:]) for k in gold.files if k.startswith('x'))
        for k in ks[:2]:
            x = gold[f'x{k}'].reshape(be.in_shape).astype(np.int8)
            out = be.run(x)
            self.assertEqual(int((out.q != lg[k]).sum()), 0, f'ảnh golden {k}')

    @unittest.skipUnless(os.environ.get('PERCEPTION_TEST_DIR'), 'chưa đặt PERCEPTION_TEST_DIR')
    def test_quantize_matches_golden(self):
        from runtime.preprocess import Preprocessor
        from PIL import Image
        meta = load_json(os.path.join(PACK, 'int8_meta.json'))
        cfg = load_json(os.path.join(ROOT, 'config', 'preprocess_b.json'))
        pre = Preprocessor(dict(cfg, int8_meta=None, input_scale=meta['input']['scale']))
        gold = np.load(os.path.join(PACK, 'golden_int8.npz'))
        for k in (0, 1):
            img = np.asarray(Image.open(os.path.join(os.environ['PERCEPTION_TEST_DIR'], f'test_{k:03d}.png')).convert('RGB'))
            np.testing.assert_array_equal(pre.quantize(img).reshape(gold[f'x{k}'].shape), gold[f'x{k}'])

    def test_pipeline_both_backends(self):
        cfg = load_json(os.path.join(ROOT, 'config', 'pipeline_b.json'))
        cfg['pack_dir'] = os.path.abspath(PACK)
        pcfg = load_json(os.path.join(ROOT, 'config', 'preprocess_b.json'))
        cfg['preprocess'] = dict(pcfg, int8_meta=os.path.join(os.path.abspath(PACK), 'int8_meta.json'))
        frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3)).astype(np.uint8)
        masks = {}
        for be in ('int8ref', 'onnx'):
            pl = Pipeline(cfg, root=ROOT, backend=be)
            r = pl.process(frame, t_capture_ns=5, keep=True)
            self.assertEqual(r.mask.m.shape, (320, 640))
            u = T.unpack(T.pack(r, include_logits=be == 'int8ref', include_mask=True))
            np.testing.assert_array_equal(u.mask.m, r.mask.m)
            np.testing.assert_array_equal(u.tier2.col_y, r.tier2.col_y)
            masks[be] = r.mask.m
            print(f'\n  {be}: ' + ', '.join(f'{k} {v / 1000:.1f} ms' for k, v in r.timings_us.items()))
        print(f'  trùng nhãn int8ref so với onnx (khung nhiễu): {float((masks["int8ref"] == masks["onnx"]).mean()):.4f}')


if __name__ == '__main__':
    unittest.main()
