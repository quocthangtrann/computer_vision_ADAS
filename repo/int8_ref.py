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
    bad = n = 0
    for k in ks:
        T = run(model, meta, gold[f'x{k}'])
        for op in meta['ops']:
            if op['op'] == 'conv':
                d = int((T[op['out']].astype(np.int8) != gold[f'y{k}_{op["idx"]}']).sum())
                bad += d
                n += T[op['out']].size
        bad += int((T[meta['output']['name']][0].astype(np.int8) != lg[k]).sum())
    print('anh:', len(ks), '| gia tri so sanh:', n, '| gia tri lech:', bad)
    print('KET QUA:', 'DAT (trung tung bit)' if bad == 0 else 'LECH')