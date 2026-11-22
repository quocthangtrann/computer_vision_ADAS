"""Đóng gói INT8 cho FPGA. Chạy SAU int8_sim.py (cùng --run hoặc --weights, --arch, --w, --h và --out).
Dùng:  python repo/int8_pack.py --run runs/NAME --out handoff_s3b [--method pct] [--zip handoff_s3b_v1]
       python repo/int8_pack.py --weights models_final/X_weights.pt --arch fsnet_m_p1s5 --w 640 --h 320 --out handoff_s3b_hd640
Cần repo/int8_ref.py (bản tham chiếu NumPy): được chép vào gói rồi chạy thử.
Cỡ khác 320x160: ảnh hiệu chuẩn và ảnh test lấy ở data/v2/hd_calib_{w}x{h}/ và hd_test_{w}x{h}/ (dựng bằng repo/make_hd_calib.py)."""
import os, sys, json, shutil, hashlib, argparse, subprocess
import numpy as np, pandas as pd
import int8_sim as sim
from int8_sim import Net, calibrate, build, forward, ort

N_GOLD = 6


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None)
    ap.add_argument('--weights', default=None, help='thay cho --run: state_dict đã tách (models_final/*_weights.pt); phải trùng với lần chạy int8_sim.py')
    ap.add_argument('--arch', default='fsnet_m')
    ap.add_argument('--w', type=int, default=320)
    ap.add_argument('--h', type=int, default=160)
    ap.add_argument('--out', default='handoff_s3b')
    ap.add_argument('--method', default='pct')
    ap.add_argument('--calib', default=None, help='mặc định theo cỡ (xem int8_sim.default_dirs)')
    ap.add_argument('--test', default=None, help='mặc định theo cỡ (xem int8_sim.default_dirs)')
    ap.add_argument('--zip', default=None)
    a = ap.parse_args()
    if bool(a.run) == bool(a.weights):
        raise SystemExit('cần đúng một trong --run và --weights')
    w, h = a.w, a.h
    src = a.run or a.weights
    a.calib = a.calib or sim.default_dirs(w, h)[0]
    a.test = a.test or sim.default_dirs(w, h)[1]
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(sim.ROOT)
    out = a.out
    info = json.load(open(f'{out}/export_info.json'))
    sm = json.load(open(f'{out}/int8_sim_summary.json'))
    assert info['run'] == src, f'{out} được tạo từ {info["run"]}, không phải {src}: chạy int8_sim.py trước'
    assert info['arch'] == a.arch, f'{out} được tạo với --arch {info["arch"]}, không phải {a.arch}'
    assert (info.get('w', 320), info.get('h', 160)) == (w, h), f'{out} được tạo ở cỡ {info.get("w", 320)}x{info.get("h", 160)}, không phải {w}x{h}'
    path = f'{out}/{info["arch"]}_c4_{w}x{h}.onnx'
    net = Net(path)
    S, L = build(net, calibrate(net, a.calib, hw=(h, w)), a.method)
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    qshape = [int(v) for v in sess.run(None, {'image': np.zeros((1, 3, h, w), np.float32)})[0].shape]   # (1, 4, hq, wq): đo từ ONNX
    assert qshape == [1, 4] + list(info.get('logit_hw', qshape[2:])), (qshape, info.get('logit_hw'))

    ops, arrs, ci = [], {}, 0
    for n in net.nodes:
        o = n.output[0]
        if n.op_type == 'Conv':
            l = L[o]
            arrs[f'W{ci}'] = l['Wq'].astype(np.int8); arrs[f'B{ci}'] = l['Bq'].astype(np.int32)
            arrs[f'M0_{ci}'] = l['M0'].astype(np.int64); arrs[f'SH{ci}'] = l['sh'].astype(np.int16)
            ops.append({'op': 'conv', 'in': n.input[0], 'out': l['out'], 'k': int(l['k']), 'relu': bool(l['relu']), 'idx': ci,
                        'cin': int(l['cin']), 'cout': int(l['cout']), 's_in': float(l['s_x']), 's_out': float(l['s_y'])})
            ci += 1
        elif n.op_type == 'MaxPool':
            ops.append({'op': 'maxpool', 'in': n.input[0], 'out': o})
        elif n.op_type == 'Resize':
            ops.append({'op': 'resize', 'in': n.input[0], 'out': o})
        elif n.op_type == 'Concat':
            ops.append({'op': 'concat', 'ins': list(n.input), 'out': o})
    meta = dict(
        model=f'{info["arch"]} 4 lớp, {info["run"]}, epoch {info["epoch"]} (FP32), {info["gmac"]} GMAC' + ('' if (w, h) == (320, 160) else f', cỡ vào {w}x{h}'),
        calibration=f'{a.method} trên ảnh trong {a.calib}',
        input=dict(name='image', scale=float(S('image')), shape=[1, 3, h, w], dtype='int8',
                   quantize='q = clip(floor(x/scale + 0.5), -127, 127), x = (RGB - mean) / std (xem preprocess_spec.json)'),
        output=dict(name='logits_quarter', scale=float(S('logits_quarter')), shape=qshape, dtype='int8',
                    classes=sim.CLASSES, dequantize='logit = q * scale'),
        ops=ops,
        assumptions=['activation đối xứng, zero-point 0; sau ReLU nằm trong [0, 127]', 'trọng số đối xứng theo kênh trong [-127, 127]',
                     'bias INT32, thang s_in * s_w[kênh]',
                     'tái lượng tử: y = (acc * M0 + 2^(shift-1)) >> shift, M0 trong [2^30, 2^31], kẹp [0,127] (sau ReLU) hoặc [-127,127]',
                     'bộ tích lũy đủ rộng (mô phỏng bằng số nguyên 64 bit)',
                     'MaxPool, Resize nearest x2, Concat chạy trực tiếp trên số nguyên; các tensor liên quan chung thang đo',
                     'CHƯA xác nhận chi tiết Y6 với TV2'])
    np.savez_compressed(f'{out}/int8_model.npz', **arrs)
    json.dump(meta, open(f'{out}/int8_meta.json', 'w'), indent=1, ensure_ascii=False)
    if (w, h) == (320, 160):
        shutil.copy('handoff_s3a_v2/preprocess_spec.json', f'{out}/preprocess_spec.json')
    else:     # cùng tiền xử lý, chỉ đổi cỡ resize; mô tả đầu ra 4 lớp
        spec = json.load(open('handoff_s3a_v2/preprocess_spec.json'))
        spec['resize'].update(size_wh=[w, h], scale_x=1280 / w, scale_y=600 / h)
        spec['onnx_output'] = dict(name='logits_quarter', shape=qshape, classes={str(i): c for i, c in enumerate(sim.CLASSES)})
        spec['postprocess_reference'] = f'bilinear upsample align_corners=False to {h}x{w} (h x w), then argmax; IoU báo ở lưới 320x160 sau khi thu nhỏ nearest (iou_320)'
        spec['checkpoint'] = f'{src} (epoch {info["epoch"]})'
        json.dump(spec, open(f'{out}/preprocess_spec.json', 'w'), indent=1, ensure_ascii=False)
    print(ci, 'lớp conv |', len(ops), 'bước | thang vào', meta['input']['scale'], '| thang ra', meta['output']['scale'], flush=True)

    g, lgq, fp = {}, [], []
    for k in range(100):
        x = sim.preprocess(f'{a.test}/test_{k:03d}.png')
        T = forward(net, S, L, x)
        lgq.append(T['logits_quarter'][0].astype(np.int8))
        fp.append(sess.run(None, {'image': np.ascontiguousarray(x)})[0][0])
        if k < N_GOLD:
            g[f'x{k}'] = T['image'].astype(np.int8)
            for op in ops:
                if op['op'] == 'conv':
                    g[f'y{k}_{op["idx"]}'] = T[op['out']].astype(np.int8)
    np.save(f'{out}/test_int8_logits_quarter.npy', np.stack(lgq))
    np.save(f'{out}/test_fp32_logits_quarter.npy', np.stack(fp).astype(np.float32))
    np.savez_compressed(f'{out}/golden_int8.npz', **g)
    shutil.copy(os.path.join(here, 'int8_ref.py'), f'{out}/int8_ref.py')
    r = subprocess.run([sys.executable, 'int8_ref.py'], cwd=out, capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    print(r.stdout, r.stderr[-1500:])

    nm = sim.CLASSES
    rows = '\n'.join(f'| {k} | ' + ' | '.join(f'{v:.4f}' for v in sm[k]['iou']) + f' | {sm[k]["road"]:.4f} |' for k in ['fp32', 'max', 'pct'])
    lt = pd.read_csv(f'{out}/layer_table.csv')
    t_max = max(int((lt.cout * lt.hout * lt.wout).max()), 3 * h * w)          # tensor lớn nhất (INT8, byte)
    lb_max = int(lt.lb.max())
    kmax = lt.loc[(lt.cin * lt.k * lt.k).idxmax()]
    bound = int(np.ceil(np.log2(int(kmax.cin) * int(kmax.k) ** 2 * 127 * 127 + 1))) + 1   # cận trên bit có dấu, chưa tính bias
    hd = (w, h) != (320, 160)
    agree = sm[a.method]['agree']; agree3 = sm[a.method].get('agree_320', agree)
    agree_txt = f', {agree3:.4f} ở lưới 320x160 (sau khi thu nhỏ nearest)' if hd else ''
    ctx = (f'- Cỡ vào {w}x{h}: IoU trong bảng là IoU ở lưới 320x160 (logit INT8, bilinear lên {w}x{h}, argmax, thu nhỏ nearest), cùng cách tính `iou_320` của train_v3.py.\n' if hd else '')
    extra = (f'7. IP có nhận đầu vào {w}x{h} không (E3)? Tensor lớn nhất {t_max / 2**20:.2f} MiB (INT8) có nằm trên chip không, hay đi qua DDR? Line buffer lớn nhất {lb_max} byte.\n'
             f'8. Gói dùng crop (0, 0, 1280, 600) rồi resize về {w}x{h}; Y2 từng yêu cầu khung HD đầy đủ. Nếu IP bắt buộc nhận khung đầy đủ thì AI phải huấn luyện lại.\n') if hd else ''
    readme = f"""# Gói bàn giao S3b (bản dựng thử): {info['arch']} 4 lớp, INT8, AI -> FPGA

**Trạng thái: bản dựng thử để kiểm tra phần cứng, CHƯA phải mô hình cuối.** Mô phỏng INT8 dựa trên giả định mặc định cho Y6 (xem `int8_meta.json`, mục `assumptions`),
chưa xác nhận chi tiết với FPGA. Bản tham chiếu số nguyên sẽ chỉnh lại khi có câu trả lời.

## Nội dung
- `{info['arch']}_c4_{w}x{h}.onnx`, `layer_table.csv`: mô hình FP32 (BN đã gộp) và bảng lớp ({info['gmac']} GMAC; {ci} conv, lớp cuối cout=4). Nguồn: `{info['run']}`, epoch {info['epoch']}.
- `int8_model.npz`, `int8_meta.json`: trọng số INT8 theo kênh, bias INT32, M0, shift, thang đo; chương trình từng bước.
- `golden_int8.npz`, `test_int8_logits_quarter.npy`: giá trị tham chiếu số nguyên ({N_GOLD} ảnh đủ các lớp; logit cuối của 100 ảnh test, dùng ảnh trong `{a.test}`).
- `int8_ref.py`: bản tham chiếu chỉ NumPy (đọc cỡ từ `int8_meta.json`). Chạy `python int8_ref.py`, kết quả mong đợi: `KET QUA: DAT`.
- `int8_layers_{a.method}.csv`: thang, M0, shift và số bit tích lũy cần cho từng lớp (trên 408 ảnh test).
- Tiền xử lý: `preprocess_spec.json`. Ảnh hiệu chuẩn: `{a.calib}` (không nằm trong gói).

## Độ chính xác (408 ảnh test có nhãn, một mô hình, hiệu chuẩn {a.method})
| | {' | '.join(nm)} | IoU đường |
|---|---|---|---|---|---|
{rows}

{ctx}- Độ trùng pixel INT8 so với FP32 ({a.method}): {agree:.4f} ở lưới {w}x{h}{agree_txt}.

## Thang đầu vào và đầu ra
- Đầu vào: `q = clip(floor(x/{meta['input']['scale']:.6g} + 0.5), -127, 127)`, x là ảnh đã chuẩn hóa mean/std; hình dạng 1x3x{h}x{w}.
- Đầu ra: logit INT8 (1x4x{qshape[2]}x{qshape[3]}), `logit = q * {meta['output']['scale']:.6g}`. A53 nhân thang này rồi phóng bilinear lên {w}x{h} (rộng x cao) và chọn lớp.

## Câu hỏi cho FPGA (chi tiết Y6)
1. Activation có zero-point không, có dấu hay không dấu sau ReLU, kẹp [0, 127] hay [0, 255]?
2. Bộ tích lũy bao nhiêu bit, tràn thì xử lý thế nào? (Trên dữ liệu thật cần {sm[a.method]['max_acc_bits']} bit có dấu; cận trên tính tay cho lớp {int(kmax.cin)} kênh vào là {bound} bit có dấu, chưa tính bias.)
3. Bias bao nhiêu bit, theo thang nào? M0 bao nhiêu bit, shift trong khoảng nào, công thức làm tròn half-up cụ thể?
4. Concat khớp thang thế nào? MaxPool và Resize chạy trên số nguyên không đổi thang?
5. IP có đệm kênh lớp cuối (cout=4) và lớp đầu (cin=3) lên bội 16 không?
6. Đầu ra IP là INT8 hay INT32 trước bước tái lượng tử cuối?
{extra}"""
    open(f'{out}/README.md', 'w', encoding='utf-8').write(readme)
    shutil.rmtree(f'{out}/__pycache__', ignore_errors=True)
    man = {}
    for d, _, fs in os.walk(out):
        for f in sorted(fs):
            if f != 'MANIFEST.json':
                p = os.path.join(d, f)
                man[os.path.relpath(p, out)] = {'bytes': os.path.getsize(p), 'sha256': sha(p)}
    json.dump(man, open(f'{out}/MANIFEST.json', 'w'), indent=1)
    zp = shutil.make_archive(a.zip or out + '_v1', 'zip', out)
    print(len(man), 'tập tin | zip', zp, round(os.path.getsize(zp) / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()