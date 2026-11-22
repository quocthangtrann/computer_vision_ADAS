# Tham chieu INT8 thuan so nguyen, chi dung NumPy. LUU Y: dua tren gia dinh Y6 (xem int8_meta.json), chua xac nhan voi FPGA.
import json, os, numpy as np
from numpy.lib.stride_tricks import sliding_window_view
HERE = os.path.dirname(os.path.abspath(__file__))


def conv_int(x, Wq, k):
    # x (N,C,H,W) int64; Wq (O,C,k,k); padding same, stride 1. Tinh bang float64 nhung CHINH XAC vi |acc| << 2^53.
    p = k // 2
    win = sliding_window_view(np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))), (k, k), axis=(2, 3))
    N, C, H, W = x.shape
    O = Wq.shape[0]
    A = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, C * k * k).astype(np.float64)
    out = np.rint(A @ Wq.reshape(O, C * k * k).astype(np.float64).T).astype(np.int64)
    return out.reshape(N, H, W, O).transpose(0, 3, 1, 2)


def requant(acc, bias, M0, sh, relu):
    acc = acc + bias[None, :, None, None]
    y = (acc * M0[None, :, None, None] + np.left_shift(np.int64(1), sh - 1)[None, :, None, None]) >> sh[None, :, None, None]
    return np.clip(y, 0 if relu else -127, 127)


def run(model, meta, x_int8):
    T = {'image': x_int8.astype(np.int64)}
    for op in meta['ops']:
        if op['op'] == 'conv':
            i = op['idx']
            acc = conv_int(T[op['in']], model[f'W{i}'], op['k'])
            T[op['out']] = requant(acc, model[f'B{i}'].astype(np.int64), model[f'M0_{i}'].astype(np.int64), model[f'SH{i}'].astype(np.int64), op['relu'])
        elif op['op'] == 'maxpool':
            x = T[op['in']]
            N, C, H, W = x.shape
            T[op['out']] = x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))
        elif op['op'] == 'resize':
            T[op['out']] = np.repeat(np.repeat(T[op['in']], 2, axis=2), 2, axis=3)
        elif op['op'] == 'concat':
            T[op['out']] = np.concatenate([T[i] for i in op['ins']], axis=1)
    return T


if __name__ == '__main__':
    model = np.load(os.path.join(HERE, 'int8_model.npz'))
    meta = json.load(open(os.path.join(HERE, 'int8_meta.json')))
    gold = np.load(os.path.join(HERE, 'golden_int8.npz'))
    lg = np.load(os.path.join(HERE, 'test_int8_logits_quarter.npy'))
    ks = sorted(int(k[1:]) for k in gold.files if k.startswith('x'))
    # co vao/ra doc tu int8_meta.json (khong viet cung 320x160); tep tham chieu phai khop voi co do (chi dung ASCII, de chay duoc moi noi)
    in_shape, out_shape = tuple(meta['input']['shape']), tuple(meta['output']['shape'])
    print(f'co vao (n,c,h,w) = {in_shape} | co ra = {out_shape}')
    assert lg.shape[1:] == out_shape[1:], f'test_int8_logits_quarter.npy co co {lg.shape[1:]}, int8_meta.json ghi {out_shape[1:]}'
    for k in ks:
        assert gold[f'x{k}'].shape == in_shape[1:] or gold[f'x{k}'].shape == in_shape, f'golden x{k} co co {gold[f"x{k}"].shape}, int8_meta.json ghi {in_shape}'
    bad = n = 0
    for k in ks:
        T = run(model, meta, gold[f'x{k}'])
        assert T[meta['output']['name']].shape[1:] == out_shape[1:], (T[meta['output']['name']].shape, out_shape)
        for op in meta['ops']:
            if op['op'] == 'conv':
                d = int((T[op['out']].astype(np.int8) != gold[f'y{k}_{op["idx"]}']).sum())
                bad += d
                n += T[op['out']].size
        bad += int((T[meta['output']['name']][0].astype(np.int8) != lg[k]).sum())
    print('anh:', len(ks), '| gia tri so sanh:', n, '| gia tri lech:', bad)
    print('KET QUA:', 'DAT (trung tung bit)' if bad == 0 else 'LECH')