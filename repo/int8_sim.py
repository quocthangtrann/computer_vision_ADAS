"""Mô phỏng INT8 bằng số nguyên thật cho FSNet 4 lớp.
Giả định Y6 mặc định (CHƯA xác nhận với FPGA): activation đối xứng, zero-point 0, sau ReLU trong [0, 127];
trọng số đối xứng theo kênh; bias INT32; tái lượng tử (acc*M0 + 2^(shift-1)) >> shift với M0 trong [2^30, 2^31]; tích lũy không giới hạn.
Dùng:  python repo/int8_sim.py --run runs/NAME [--arch fsnet_m] [--out handoff_s3b] [--calib handoff_s3a_v2/calib]
"""
import os, sys, json, time, argparse, hashlib, subprocess
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
try:
    import onnx, onnxruntime as ort
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx', 'onnxruntime'], check=True)
    import onnx, onnxruntime as ort
from onnx import numpy_helper, helper, TensorProto

ROOT = os.environ.get('CAPSTONE_ROOT', '/content/drive/MyDrive/capstone_ai')
V2 = 'data/v2'
MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(1, 3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], np.float32).reshape(1, 3, 1, 1)
CLASSES = ['không đường', 'không thuộc ta', 'làn đang đi', 'làn kề']


class Net:
    """Đồ thị ONNX: Conv/Relu/MaxPool/Resize/Concat. Nhóm các tensor phải chung thang đo (qua ReLU, MaxPool, Resize, Concat)."""
    def __init__(self, path):
        self.path = path
        g = onnx.load(path).graph
        self.init = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in g.initializer}
        self.nodes = [n for n in g.node if n.op_type != 'Constant']
        self.cons = {}
        for n in self.nodes:
            for i in n.input:
                self.cons.setdefault(i, []).append(n)
        self.pre_relu = set()
        for n in self.nodes:
            if n.op_type == 'Relu':
                assert len(self.cons[n.input[0]]) == 1, 'đầu ra conv đi nhiều nơi trước ReLU'
                self.pre_relu.add(n.input[0])
        self.parent = {}
        for n in self.nodes:
            o = n.output[0]
            if n.op_type in ('Relu', 'MaxPool', 'Resize'):
                self.union(n.input[0], o)
            elif n.op_type == 'Concat':
                for i in n.input:
                    self.union(i, o)

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def preprocess(path):
    return (np.array(Image.open(path).convert('RGB'), np.float32).transpose(2, 0, 1)[None] - MEAN) / STD


def calibrate(net, calib_dir):
    """Biên độ từng tensor trên ảnh hiệu chuẩn: 'max' (lớn nhất) và 'pct' (phân vị 99,99)."""
    mdl = onnx.load(net.path)
    gg = mdl.graph
    have = {o.name for o in gg.output}
    for n in gg.node:
        if n.op_type == 'Constant':
            continue
        for o in n.output:
            if o not in have:
                gg.output.append(helper.make_tensor_value_info(o, TensorProto.FLOAT, None))
    sess = ort.InferenceSession(mdl.SerializeToString(), providers=['CPUExecutionProvider'])
    names = [o.name for o in sess.get_outputs()]
    files = sorted(f for f in os.listdir(calib_dir) if f.endswith('.png'))
    amax = {'max': {}, 'pct': {}}
    for f in files:
        x = preprocess(f'{calib_dir}/{f}')
        outs = dict(zip(names, sess.run(None, {'image': np.ascontiguousarray(x)})))
        outs['image'] = x
        for nm, a in outs.items():
            if nm in net.pre_relu:
                continue
            v = np.abs(a).ravel()
            amax['max'][nm] = max(amax['max'].get(nm, 0.0), float(v.max()))
            amax['pct'][nm] = max(amax['pct'].get(nm, 0.0), float(np.percentile(v, 99.99)))
    print(len(files), 'ảnh hiệu chuẩn |', len(amax['max']), 'tensor', flush=True)
    return amax


def build(net, amax, method):
    gs = {}
    for nm, v in amax[method].items():
        r = net.find(nm)
        gs[r] = max(gs.get(r, 0.0), v)
    S = lambda t: max(gs[net.find(t)], 1e-8) / 127.0
    L = {}
    for n in net.nodes:
        if n.op_type != 'Conv':
            continue
        W, B = net.init[n.input[1]], net.init[n.input[2]]
        out = n.output[0]
        relu = out in net.pre_relu
        oname = net.cons[out][0].output[0] if relu else out
        s_x, s_y = S(n.input[0]), S(oname)
        s_w = np.maximum(np.abs(W).reshape(W.shape[0], -1).max(1) / 127.0, 1e-12)
        Wq = np.clip(np.floor(W / s_w[:, None, None, None] + 0.5), -127, 127)
        Bq = np.floor(B / (s_x * s_w) + 0.5).astype(np.int64)
        m = s_x * s_w / s_y
        e = np.floor(np.log2(m)).astype(np.int64)
        sh = 30 - e
        M0 = np.floor(m * np.exp2(sh.astype(np.float64)) + 0.5).astype(np.int64)
        over = M0 >= (1 << 31)
        M0[over] = 1 << 30
        sh[over] -= 1
        assert sh.min() >= 1 and sh.max() <= 60 and M0.min() >= (1 << 29)
        L[out] = dict(inp=n.input[0], out=oname, relu=relu, k=W.shape[2], cin=W.shape[1], cout=W.shape[0],
                      Wq=Wq, Bq=Bq, M0=M0, sh=sh, s_x=s_x, s_y=s_y)
    return S, L


def forward(net, S, L, xf, stats=None):
    """Suy luận bằng số nguyên. xf: ảnh đã chuẩn hóa (N,3,H,W) float32. Trả về dict mọi tensor (số nguyên)."""
    T = {'image': np.clip(np.floor(xf / S('image') + 0.5), -127, 127).astype(np.int64)}
    for n in net.nodes:
        op, o = n.op_type, n.output[0]
        if op == 'Conv':
            l = L[o]
            acc = F.conv2d(torch.from_numpy(T[n.input[0]].astype(np.float64)), torch.from_numpy(l['Wq']), padding=l['k'] // 2).numpy()
            acc = np.rint(acc).astype(np.int64) + l['Bq'][None, :, None, None]
            if stats is not None:
                stats[o] = max(stats.get(o, 0), int(np.abs(acc).max()))
            prod = acc * l['M0'][None, :, None, None]
            assert np.abs(prod).max() < (1 << 62)
            y = (prod + np.left_shift(np.int64(1), l['sh'] - 1)[None, :, None, None]) >> l['sh'][None, :, None, None]
            T[l['out']] = np.clip(y, 0 if l['relu'] else -127, 127)
        elif op == 'MaxPool':
            x = T[n.input[0]]
            N, C, H, W = x.shape
            T[o] = x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))
        elif op == 'Resize':
            T[o] = np.repeat(np.repeat(T[n.input[0]], 2, axis=2), 2, axis=3)
        elif op == 'Concat':
            T[o] = np.concatenate([T[i] for i in n.input], axis=1)
    return T


def logits_of(S, T):
    return T['logits_quarter'].astype(np.float32) * np.float32(S('logits_quarter'))


def metrics(Y, P):
    v = Y != 255
    c = np.bincount(Y[v].astype(np.int64) * 4 + P[v], minlength=16).reshape(4, 4).astype(float)
    iou = [c[k, k] / (c[k].sum() + c[:, k].sum() - c[k, k]) for k in range(4)]
    road = c[1:, 1:].sum() / (c[1:, 1:].sum() + c[0, 1:].sum() + c[1:, 0].sum())
    return iou, road


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='thư mục run, ví dụ runs/b9s4_c4aug_s0 (cần last.pt)')
    ap.add_argument('--arch', default='fsnet_m')
    ap.add_argument('--out', default='handoff_s3b')
    ap.add_argument('--calib', default='handoff_s3a_v2/calib')
    a = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'repo'))
    import models_custom as mc
    torch.set_num_threads(os.cpu_count())
    os.makedirs(a.out, exist_ok=True)
    print('=== 1) xuất ONNX', flush=True)
    m = mc.build_fsnet(a.arch, num_classes=4)
    ck = torch.load(f'{a.run}/last.pt', map_location='cpu', weights_only=False)
    m.load_state_dict(ck['model'], strict=True)
    m.eval()
    bad = mc.contract_violations(m)
    assert not bad, f'vi phạm hợp đồng toán tử v1: {bad}'
    path = f'{a.out}/{a.arch}_c4_320x160.onnx'
    print('epoch:', ck['epoch'], '| toán tử:', mc.export_onnx(m, path, h=160, w=320, opset=13))
    rows = mc.layer_table(m, h=160, w=320)
    pd.DataFrame(rows).rename_axis('conv').to_csv(f'{a.out}/layer_table.csv')
    gmac, params = sum(r['mac'] for r in rows) / 1e9, sum(p.numel() for p in m.parameters()) / 1e6
    json.dump(dict(run=a.run, arch=a.arch, epoch=int(ck['epoch']), gmac=round(gmac, 4), params_M=round(params, 3)),
              open(f'{a.out}/export_info.json', 'w'))
    print(f'{len(rows)} conv | {gmac:.4f} GMAC | {params:.3f} M tham số | sha256 onnx: {hashlib.sha256(open(path, "rb").read()).hexdigest()}', flush=True)

    print('=== 2) hiệu chuẩn', flush=True)
    net = Net(path)
    amax = calibrate(net, a.calib)

    print('=== 3) đánh giá trên 408 ảnh test (có nhãn)', flush=True)
    man = pd.read_csv(f'{V2}/pilot_drv_manifest.csv')
    te = np.where(man.split_v2.values == 'test')[0]
    X = np.load(f'{V2}/pilot_drv_x_320x160.npy', mmap_mode='r')
    Y = np.load(f'{V2}/pilot_drv_y_320x160.npy', mmap_mode='r')
    Ye = np.ascontiguousarray(Y[te])
    Xn = (np.ascontiguousarray(X[te]).astype(np.float32).transpose(0, 3, 1, 2) - MEAN) / STD
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    up = lambda lg: F.interpolate(torch.from_numpy(lg), size=(160, 320), mode='bilinear', align_corners=False).argmax(1).numpy().astype(np.uint8)
    lg_fp = np.concatenate([sess.run(None, {'image': np.ascontiguousarray(Xn[i:i + 1])})[0] for i in range(len(Xn))])
    P_fp = up(lg_fp)
    iou_fp, road_fp = metrics(Ye, P_fp)
    table = [dict(mô_hình='FP32 (ONNX)', **{n: round(v, 4) for n, v in zip(CLASSES, iou_fp)}, IoU_đường=round(road_fp, 4), trùng_pixel=1.0, lệch_logit_pct=0.0)]
    summ = {}
    for method in ['max', 'pct']:
        S, L = build(net, amax, method)
        stats, t0 = {}, time.time()
        lg = np.concatenate([logits_of(S, forward(net, S, L, Xn[i:i + 8], stats)) for i in range(0, len(Xn), 8)])
        P = up(lg)
        iou, road = metrics(Ye, P)
        agree = float((P == P_fp).mean())
        dev = 100 * float(np.abs(lg - lg_fp).mean() / np.abs(lg_fp).max())
        table.append(dict(mô_hình=f'INT8 ({method})', **{n: round(v, 4) for n, v in zip(CLASSES, iou)}, IoU_đường=round(road, 4),
                          trùng_pixel=round(agree, 4), lệch_logit_pct=round(dev, 3)))
        lay = []
        for o, l in L.items():
            lay.append(dict(conv=o, cin=l['cin'], cout=l['cout'], k=l['k'], relu=l['relu'], s_x=l['s_x'], s_y=l['s_y'],
                            M0_min=int(l['M0'].min()), M0_max=int(l['M0'].max()), shift_min=int(l['sh'].min()), shift_max=int(l['sh'].max()),
                            max_acc=stats[o], bit_tich_luy_can=int(np.ceil(np.log2(stats[o] + 1))) + 1))
        pd.DataFrame(lay).to_csv(f'{a.out}/int8_layers_{method}.csv', index=False)
        summ[method] = dict(iou=iou, road=road, agree=agree, logit_dev_pct=dev, max_acc_bits=max(r['bit_tich_luy_can'] for r in lay))
        print(f'  {method}: xong sau {time.time() - t0:.0f}s', flush=True)
    print(pd.DataFrame(table).to_string(index=False))
    print('\nINT8 trừ FP32 (âm = giảm):')
    for method, r in zip(['max', 'pct'], table[1:]):
        print(f'  {method}:', {n: round(r[n] - table[0][n], 4) for n in CLASSES + ['IoU_đường']})
    print('bit tích lũy có dấu cần thiết (lớn nhất qua các lớp):', {k: v['max_acc_bits'] for k, v in summ.items()})
    json.dump(dict(fp32=dict(iou=iou_fp, road=road_fp), **summ), open(f'{a.out}/int8_sim_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()