"""Đóng gói INT8 cho FPGA. Chạy SAU int8_sim.py (cùng --run và --out).
Dùng:  python repo/int8_pack.py --run runs/NAME --out handoff_s3b [--method pct] [--zip handoff_s3b_v1]
Cần repo/int8_ref.py (bản tham chiếu NumPy): được chép vào gói rồi chạy thử."""
import os, sys, json, shutil, hashlib, argparse, subprocess
import numpy as np
import int8_sim as sim
from int8_sim import Net, calibrate, build, forward, ort

N_GOLD = 6


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', default='handoff_s3b')
    ap.add_argument('--method', default='pct')
    ap.add_argument('--calib', default='handoff_s3a_v2/calib')
    ap.add_argument('--test', default='handoff_s3a_v2/test')
    ap.add_argument('--zip', default=None)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(sim.ROOT)
    out = a.out
    info = json.load(open(f'{out}/export_info.json'))
    sm = json.load(open(f'{out}/int8_sim_summary.json'))
    assert info['run'] == a.run, f'{out} được tạo từ {info["run"]}, không phải {a.run}: chạy int8_sim.py trước'
    path = f'{out}/{info["arch"]}_c4_320x160.onnx'
    net = Net(path)
    S, L = build(net, calibrate(net, a.calib), a.method)

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
        model=f'{info["arch"]} 4 lớp, {info["run"]}, epoch {info["epoch"]} (FP32), {info["gmac"]} GMAC',
        calibration=f'{a.method} trên ảnh trong {a.calib}',
        input=dict(name='image', scale=float(S('image')), shape=[1, 3, 160, 320], dtype='int8',
                   quantize='q = clip(floor(x/scale + 0.5), -127, 127), x = (RGB - mean) / std (xem preprocess_spec.json)'),
        output=dict(name='logits_quarter', scale=float(S('logits_quarter')), shape=[1, 4, 40, 80], dtype='int8',
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
    shutil.copy('handoff_s3a_v2/preprocess_spec.json', f'{out}/preprocess_spec.json')
    print(ci, 'lớp conv |', len(ops), 'bước | thang vào', meta['input']['scale'], '| thang ra', meta['output']['scale'], flush=True)

    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
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
    readme = f"""# Gói bàn giao S3b (bản dựng thử): {info['arch']} 4 lớp, INT8, AI -> FPGA

**Trạng thái: bản dựng thử để kiểm tra phần cứng, CHƯA phải mô hình cuối.** Mô phỏng INT8 dựa trên giả định mặc định cho Y6 (xem `int8_meta.json`, mục `assumptions`),
chưa xác nhận chi tiết với FPGA. Bản tham chiếu số nguyên sẽ chỉnh lại khi có câu trả lời.

## Nội dung
- `{info['arch']}_c4_320x160.onnx`, `layer_table.csv`: mô hình FP32 (BN đã gộp) và bảng lớp ({info['gmac']} GMAC; {ci} conv, lớp cuối cout=4). Nguồn: `{info['run']}`, epoch {info['epoch']}.
- `int8_model.npz`, `int8_meta.json`: trọng số INT8 theo kênh, bias INT32, M0, shift, thang đo; chương trình từng bước.
- `golden_int8.npz`, `test_int8_logits_quarter.npy`: giá trị tham chiếu số nguyên ({N_GOLD} ảnh đủ các lớp; logit cuối của 100 ảnh test, dùng ảnh trong `handoff_s3a_v2/test`).
- `int8_ref.py`: bản tham chiếu chỉ NumPy. Chạy `python int8_ref.py`, kết quả mong đợi: `KET QUA: DAT`.
- `int8_layers_{a.method}.csv`: thang, M0, shift và số bit tích lũy cần cho từng lớp (trên 408 ảnh test).
- Tiền xử lý và ảnh hiệu chuẩn: dùng `preprocess_spec.json` và `calib/` của `handoff_s3a_v2`.

## Độ chính xác (408 ảnh test có nhãn, một mô hình, hiệu chuẩn {a.method})
| | {' | '.join(nm)} | IoU đường |
|---|---|---|---|---|---|
{rows}

## Thang đầu vào và đầu ra
- Đầu vào: `q = clip(floor(x/{meta['input']['scale']:.6g} + 0.5), -127, 127)`, x là ảnh đã chuẩn hóa mean/std.
- Đầu ra: logit INT8 (1x4x40x80), `logit = q * {meta['output']['scale']:.6g}`. A53 nhân thang này rồi phóng bilinear và chọn lớp.

## Câu hỏi cho FPGA (chi tiết Y6)
1. Activation có zero-point không, có dấu hay không dấu sau ReLU, kẹp [0, 127] hay [0, 255]?
2. Bộ tích lũy bao nhiêu bit, tràn thì xử lý thế nào? (Trên dữ liệu thật cần {sm[a.method]['max_acc_bits']} bit có dấu; cận trên tính tay cho lớp 224 kênh vào khoảng 26-27 bit.)
3. Bias bao nhiêu bit, theo thang nào? M0 bao nhiêu bit, shift trong khoảng nào, công thức làm tròn half-up cụ thể?
4. Concat khớp thang thế nào? MaxPool và Resize chạy trên số nguyên không đổi thang?
5. IP có đệm kênh lớp cuối (cout=4) và lớp đầu (cin=3) lên bội 16 không?
6. Đầu ra IP là INT8 hay INT32 trước bước tái lượng tử cuối?
"""
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