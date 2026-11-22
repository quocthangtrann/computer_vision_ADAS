"""Mô phỏng INT8 bằng số nguyên thật cho FSNet 4 lớp, mọi cỡ ảnh vào (mặc định 320x160).
Giả định Y6 mặc định (CHƯA xác nhận với FPGA): activation đối xứng, zero-point 0, sau ReLU trong [0, 127];
trọng số đối xứng theo kênh; bias INT32; tái lượng tử (acc*M0 + 2^(shift-1)) >> shift với M0 trong [2^30, 2^31]; tích lũy không giới hạn.
Dùng:  python repo/int8_sim.py --run runs/NAME [--arch fsnet_m] [--w 320 --h 160] [--out handoff_s3b] [--calib handoff_s3a_v2/calib]
       python repo/int8_sim.py --weights models_final/X_weights.pt --arch fsnet_m_p1s5 --w 640 --h 320 --out handoff_s3b_hd640   # HD
Cỡ khác 320x160: ảnh hiệu chuẩn lấy ở data/v2/hd_calib_{w}x{h}/ (dựng bằng repo/make_hd_calib.py); ảnh test lấy ở cache
data/v2/pilot_drv_x_{w}x{h}.npy (408 dòng split_v2 == test), nhãn luôn là pilot_drv_y_320x160.npy.
Đánh giá đúng như iou_320 của train_v3.py: logit INT8 -> bilinear lên (h, w) -> argmax -> thu nhỏ NEAREST về 160x320 -> so nhãn 320x160.
Bước FP32 so với final_metrics.json của run (sai lệch lớn hơn --tol thì DỪNG, không lượng tử hóa tiếp).
sha256 byte của ONNX phụ thuộc phiên bản torch/onnx; muốn so với một ONNX cũ cùng trọng số thì so dấu vân tay trọng số:
       python repo/int8_sim.py --fingerprint handoff_s3b/fsnet_m_c4_320x160.onnx
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


def onnx_fingerprint(path):
    """sha256 của trọng số và bias các Conv theo thứ tự đồ thị (chỉ giá trị float32, không tên, không tiêu đề tệp).
    Không phụ thuộc phiên bản torch/onnx đã xuất tệp, nên so được hai tệp ONNX cùng trọng số nhưng khác sha256 byte."""
    g = onnx.load(path).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    h = hashlib.sha256()
    for n in g.node:
        if n.op_type == 'Conv':
            for nm in n.input[1:3]:
                a = np.ascontiguousarray(init[nm], dtype=np.float32)
                h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def preprocess(path):
    return (np.array(Image.open(path).convert('RGB'), np.float32).transpose(2, 0, 1)[None] - MEAN) / STD


def calibrate(net, calib_dir, hw=None):
    """Biên độ từng tensor trên ảnh hiệu chuẩn: 'max' (lớn nhất) và 'pct' (phân vị 99,99). hw = (h, w) mong đợi của ảnh."""
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
        if hw is not None and x.shape[2:] != tuple(hw):
            raise SystemExit(f'ảnh hiệu chuẩn {calib_dir}/{f} có cỡ (h, w) = {x.shape[2:]}, mô hình cần {tuple(hw)}')
        outs = dict(zip(names, sess.run(None, {'image': np.ascontiguousarray(x)})))
        outs['image'] = x
        for nm, a in outs.items():
            if nm in net.pre_relu:
                continue
            v = np.abs(a).ravel()
            amax['max'][nm] = max(amax['max'].get(nm, 0.0), float(v.max()))
            amax['pct'][nm] = max(amax['pct'].get(nm, 0.0), float(np.percentile(v, 99.99)))
    if not files:
        raise SystemExit(f'không có ảnh .png nào trong {calib_dir}')
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


def conv_exact(x, Wq, k):
    """Conv số nguyên CHÍNH XÁC: x (N,C,H,W) số nguyên, Wq (O,C,k,k) số nguyên lưu float64, padding same, stride 1.
    Cộng k*k lát cắt, mỗi lát một phép nhân ma trận float64 (chính xác vì |acc| < 2^26 << 2^53). Không dùng F.conv2d: conv float64 của
    torch 2.2.2 (CPU) từng trả kết quả SAI (5e14, NaN) ở lớp 128 kênh, 5x10, bất kể bố cục bộ nhớ. Khác cách im2col của int8_ref.py."""
    N, C, H, W = x.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))).astype(np.float64)
    acc = np.zeros((N, Wq.shape[0], H, W), np.float64)
    for dy in range(k):
        for dx in range(k):
            acc += np.tensordot(Wq[:, :, dy, dx], xp[:, :, dy:dy + H, dx:dx + W], axes=([1], [1])).transpose(1, 0, 2, 3)
    assert np.abs(acc).max() < 2 ** 53, 'tích lũy vượt 2^53: phép cộng float64 không còn chính xác'
    return acc


def forward(net, S, L, xf, stats=None):
    """Suy luận bằng số nguyên. xf: ảnh đã chuẩn hóa (N,3,H,W) float32. Trả về dict mọi tensor (số nguyên)."""
    T = {'image': np.clip(np.floor(xf / S('image') + 0.5), -127, 127).astype(np.int64)}
    for n in net.nodes:
        op, o = n.op_type, n.output[0]
        if op == 'Conv':
            l = L[o]
            acc = conv_exact(T[n.input[0]], l['Wq'], l['k'])
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


def iou_from_conf(c):
    c = np.asarray(c, float).reshape(4, 4)
    iou = [c[k, k] / (c[k].sum() + c[:, k].sum() - c[k, k]) for k in range(4)]
    road = c[1:, 1:].sum() / (c[1:, 1:].sum() + c[0, 1:].sum() + c[1:, 0].sum())
    return iou, road


def metrics(Y, P):
    v = Y < 4          # 255 (và 254) bị bỏ qua, như train_v3.py
    return iou_from_conf(np.bincount(Y[v].astype(np.int64) * 4 + P[v], minlength=16))


def pred_at(lg, w, h):
    """Logit (n, 4, hq, wq) float32 -> lớp (n, h, w): phóng bilinear (align_corners=False) rồi argmax."""
    return F.interpolate(torch.from_numpy(lg), size=(h, w), mode='bilinear', align_corners=False).argmax(1).numpy().astype(np.uint8)


def to320(P):
    """Thu nhỏ NEAREST về 160x320, đúng như evaluate() của train_v3.py (cỡ 320x160 thì giữ nguyên)."""
    if P.shape[1:] == (160, 320):
        return P
    return F.interpolate(torch.from_numpy(P).unsqueeze(1).float(), size=(160, 320), mode='nearest').squeeze(1).numpy().astype(np.uint8)


def score(lg, ref, Y320, w, h, chunk=8):
    """IoU ở lưới 320x160 (như iou_320) và độ trùng pixel so với ref (None = so với chính nó), theo khối để không tốn RAM."""
    conf, ag, n, ag3, n3 = np.zeros(16), 0, 0, 0, 0
    for i in range(0, len(lg), chunk):
        P = pred_at(lg[i:i + chunk], w, h)
        R = P if ref is None else pred_at(ref[i:i + chunk], w, h)
        P3, R3 = to320(P), to320(R)
        y = Y320[i:i + chunk]; v = y < 4
        conf += np.bincount(y[v].astype(np.int64) * 4 + P3[v], minlength=16)
        ag += int((P == R).sum()); n += P.size; ag3 += int((P3 == R3).sum()); n3 += P3.size
    iou, road = iou_from_conf(conf)
    return dict(iou=iou, road=road, agree=ag / n, agree_320=ag3 / n3)


def default_dirs(w, h):
    """(thư mục ảnh hiệu chuẩn, thư mục ảnh test) mặc định theo cỡ."""
    if (w, h) == (320, 160):
        return 'handoff_s3a_v2/calib', 'handoff_s3a_v2/test'
    return f'{V2}/hd_calib_{w}x{h}', f'{V2}/hd_test_{w}x{h}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None, help='thư mục run, ví dụ runs/b9s4_c4aug_s0 (cần last.pt; ["model"] là trọng số triển khai)')
    ap.add_argument('--weights', default=None, help='thay cho --run: state_dict đã tách, ví dụ models_final/B_hd640_p1s5_kdT_s0_weights.pt (nạp strict)')
    ap.add_argument('--metrics', default=None, help='final_metrics.json để đối chiếu bước FP32 (mặc định: cạnh --run, hoặc *_final_metrics.json cạnh --weights)')
    ap.add_argument('--arch', default='fsnet_m')
    ap.add_argument('--w', type=int, default=320)
    ap.add_argument('--h', type=int, default=160)
    ap.add_argument('--out', default='handoff_s3b')
    ap.add_argument('--calib', default=None, help='mặc định theo cỡ: handoff_s3a_v2/calib (320x160) hoặc data/v2/hd_calib_{w}x{h}')
    ap.add_argument('--tol', type=float, default=0.002, help='sai lệch IoU tối đa của ONNX FP32 so với final_metrics.json; hơn thì dừng')
    ap.add_argument('--fingerprint', default=None, help='chỉ in sha256 byte và dấu vân tay trọng số của một tệp ONNX có sẵn rồi thoát')
    a = ap.parse_args()
    if a.fingerprint:
        print('sha256 byte        :', hashlib.sha256(open(a.fingerprint, 'rb').read()).hexdigest())
        print('dấu vân tay trọng số:', onnx_fingerprint(a.fingerprint))
        return
    if bool(a.run) == bool(a.weights):
        raise SystemExit('cần đúng một trong --run và --weights')
    w, h = a.w, a.h
    src = a.run or a.weights
    os.chdir(ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'repo'))
    import models_custom as mc
    torch.set_num_threads(os.cpu_count())
    os.makedirs(a.out, exist_ok=True)
    calib_dir = a.calib or default_dirs(w, h)[0]
    if a.metrics: mpath = a.metrics
    elif a.run: mpath = f'{a.run}/final_metrics.json'
    else: mpath = a.weights[:-len('_weights.pt')] + '_final_metrics.json' if a.weights.endswith('_weights.pt') else None
    fm = json.load(open(mpath)) if mpath and os.path.exists(mpath) else None
    if fm is not None:
        assert fm.get('arch', 'fsnet_m') == a.arch, f'{mpath}: arch {fm.get("arch")} khác --arch {a.arch}'
        assert (fm.get('w', 320), fm.get('h', 160)) == (w, h), f'{mpath}: cỡ {fm.get("w", 320)}x{fm.get("h", 160)} khác --w {w} --h {h}'

    print(f'=== 1) xuất ONNX ({a.arch}, {w}x{h})', flush=True)
    assert a.arch in mc.PRESETS, f'không có preset {a.arch}'
    pc = mc.PRESETS[a.arch]; need = 2 ** (4 + pc.get('pre', 0) + (1 if pc.get('e5') else 0))
    assert w % need == 0 and h % need == 0, f'{a.arch} cần w, h chia hết cho {need}'
    m = mc.build_fsnet(a.arch, num_classes=4)
    ck = torch.load(src if a.weights else f'{a.run}/last.pt', map_location='cpu', weights_only=False)
    sd = ck['model'] if 'model' in ck else ck
    m.load_state_dict(sd, strict=True)
    epoch = int(ck['epoch']) if 'epoch' in ck else int((fm or {}).get('epochs', -1))
    m.eval()
    bad = mc.contract_violations(m)
    assert not bad, f'vi phạm hợp đồng toán tử v1: {bad}'
    path = f'{a.out}/{a.arch}_c4_{w}x{h}.onnx'
    print('epoch:', epoch, '| toán tử:', mc.export_onnx(m, path, h=h, w=w, opset=13))
    rows = mc.layer_table(m, h=h, w=w)
    pd.DataFrame(rows).rename_axis('conv').to_csv(f'{a.out}/layer_table.csv')
    gmac, params = sum(r['mac'] for r in rows) / 1e9, sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad():
        hq, wq = mc.fuse_for_deploy(m)(torch.zeros(1, 3, h, w)).shape[2:]       # lưới logit, đo từ một lượt xuôi
    json.dump(dict(run=src, arch=a.arch, epoch=epoch, gmac=round(gmac, 4), params_M=round(params, 3), w=w, h=h, logit_hw=[int(hq), int(wq)]),
              open(f'{a.out}/export_info.json', 'w'))
    print(f'{len(rows)} conv | {gmac:.4f} GMAC | {params:.3f} M tham số | lưới logit {wq}x{hq} | sha256 onnx: {hashlib.sha256(open(path, "rb").read()).hexdigest()}', flush=True)
    print('  dấu vân tay trọng số (không phụ thuộc phiên bản torch/onnx):', onnx_fingerprint(path), flush=True)

    print('=== 2) FP32 (ONNX) trên 408 ảnh test (có nhãn), IoU ở lưới 320x160', flush=True)
    man = pd.read_csv(f'{V2}/pilot_drv_manifest.csv')
    te = np.where(man.split_v2.values == 'test')[0]
    X = np.load(f'{V2}/pilot_drv_x_{w}x{h}.npy', mmap_mode='r')
    Y320 = np.ascontiguousarray(np.load(f'{V2}/pilot_drv_y_320x160.npy', mmap_mode='r')[te])
    assert X.shape[1:3] == (h, w), f'cache ảnh có cỡ {X.shape[1:3]}, cần {(h, w)}'
    assert len(X) == len(man), f'cache ảnh có {len(X)} dòng, manifest có {len(man)}'
    xin = lambda i, j: (np.ascontiguousarray(X[te[i:j]]).astype(np.float32).transpose(0, 3, 1, 2) - MEAN) / STD
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    assert tuple(sess.get_outputs()[0].shape[1:]) == (4, int(hq), int(wq)), sess.get_outputs()[0].shape
    lg_fp = np.concatenate([sess.run(None, {'image': np.ascontiguousarray(xin(i, i + 1))})[0] for i in range(len(te))])
    sc_fp = score(lg_fp, None, Y320, w, h)
    iou_fp, road_fp = sc_fp['iou'], sc_fp['road']
    print('  ONNX FP32 :', [round(v, 4) for v in iou_fp], '| IoU đường', round(road_fp, 4))
    if fm is None:
        print(f'  CẢNH BÁO: không có final_metrics.json ({mpath}): KHÔNG đối chiếu được bước FP32')
    else:
        k = '_320' if 'iou_320' in fm else ''
        ref_iou, ref_road = fm['iou' + k], fm['road_iou' + k]
        dev = max(max(abs(x - y) for x, y in zip(iou_fp, ref_iou)), abs(road_fp - ref_road))
        print(f'  final_metrics (iou{k}):', [round(v, 4) for v in ref_iou], '| IoU đường', round(ref_road, 4), f'| lệch lớn nhất {dev:.4f} (ngưỡng {a.tol})')
        if dev > a.tol:
            raise SystemExit(f'DỪNG: ONNX FP32 lệch {dev:.4f} so với {mpath} (> {a.tol}). Kiểm trọng số, --arch, --w, --h, bộ ảnh test; chưa lượng tử hóa.')
        print('  khớp final_metrics.json: ĐẠT', flush=True)

    print('=== 3) hiệu chuẩn', flush=True)
    net = Net(path)
    amax = calibrate(net, calib_dir, hw=(h, w))

    print('=== 4) INT8 trên 408 ảnh test', flush=True)
    bs = max(1, 8 * 320 * 160 // (w * h))
    table = [dict(mô_hình='FP32 (ONNX)', **{n: round(v, 4) for n, v in zip(CLASSES, iou_fp)}, IoU_đường=round(road_fp, 4),
                  trùng_pixel=1.0, trùng_pixel_320=1.0, lệch_logit_pct=0.0)]
    summ = {}
    for method in ['max', 'pct']:
        S, L = build(net, amax, method)
        stats, t0 = {}, time.time()
        lg = np.concatenate([logits_of(S, forward(net, S, L, xin(i, min(i + bs, len(te))), stats)) for i in range(0, len(te), bs)])
        sc = score(lg, lg_fp, Y320, w, h)
        iou, road = sc['iou'], sc['road']
        dev = 100 * float(np.abs(lg - lg_fp).mean() / np.abs(lg_fp).max())
        table.append(dict(mô_hình=f'INT8 ({method})', **{n: round(v, 4) for n, v in zip(CLASSES, iou)}, IoU_đường=round(road, 4),
                          trùng_pixel=round(sc['agree'], 4), trùng_pixel_320=round(sc['agree_320'], 4), lệch_logit_pct=round(dev, 3)))
        lay = []
        for o, l in L.items():
            lay.append(dict(conv=o, cin=l['cin'], cout=l['cout'], k=l['k'], relu=l['relu'], s_x=l['s_x'], s_y=l['s_y'],
                            M0_min=int(l['M0'].min()), M0_max=int(l['M0'].max()), shift_min=int(l['sh'].min()), shift_max=int(l['sh'].max()),
                            max_acc=stats[o], bit_tich_luy_can=int(np.ceil(np.log2(stats[o] + 1))) + 1))
        pd.DataFrame(lay).to_csv(f'{a.out}/int8_layers_{method}.csv', index=False)
        summ[method] = dict(iou=iou, road=road, agree=sc['agree'], agree_320=sc['agree_320'], logit_dev_pct=dev,
                            max_acc_bits=max(r['bit_tich_luy_can'] for r in lay))
        print(f'  {method}: xong sau {time.time() - t0:.0f}s', flush=True)
    print(pd.DataFrame(table).to_string(index=False))
    print('(IoU ở lưới 320x160; trùng_pixel: ở lưới ' + f'{w}x{h}' + ', trùng_pixel_320: sau khi thu nhỏ nearest)')
    print('\nINT8 trừ FP32 (âm = giảm):')
    for method, r in zip(['max', 'pct'], table[1:]):
        print(f'  {method}:', {n: round(r[n] - table[0][n], 4) for n in CLASSES + ['IoU_đường']})
    print('bit tích lũy có dấu cần thiết (lớn nhất qua các lớp):', {k: v['max_acc_bits'] for k, v in summ.items()})
    json.dump(dict(fp32=dict(iou=iou_fp, road=road_fp), **summ, config=dict(arch=a.arch, w=w, h=h, logit_hw=[int(hq), int(wq)], calib=calib_dir, source=src)),
              open(f'{a.out}/int8_sim_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
