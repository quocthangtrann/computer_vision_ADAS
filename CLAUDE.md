# Dự án: tầng AI-CV cho xe tự hành trên FPGA (KV260)

## Mục tiêu
Mô hình phân đoạn 4 lớp chạy INT8 trên IP CNN tự thiết kế (PL của KV260), ra mặt nạ để tính `PerceptionResult`
(làn xe đang đi, làn kề, vật cản, khoảng trống theo hướng) cho tầng điều khiển. Hết Phase 1: 10/12.
Phân công: TV1 = AI/CV (người dùng), TV2 = FPGA (IP, DPU baseline), TV3 = điều khiển/nhúng. Cảm biến: webcam trước,
radar trước (trên camera), 4 siêu âm. Hợp nhất cảm biến ở tầng sau mạng, không đưa vào mạng.

## Ràng buộc cứng (hợp đồng với TV2)
- Chỉ dùng: Conv 3x3 hoặc 1x1 (stride 1, groups = 1, padding same), BN gộp vào conv, ReLU, MaxPool 2x2, Upsample nearest x2, Concat.
  KHÔNG: depthwise, ReLU6, AvgPool, cộng residual, dilated, ConvTranspose, bilinear, attention. Kiểm bằng `models_custom.contract_violations`.
- INT8 (Y6): trọng số đối xứng theo kênh, activation theo tensor, tái lượng tử (M0, shift) half-up, hai nhánh Concat chung thang.
  Chi tiết (zero-point, bit tích lũy, M0) CHƯA xác nhận với TV2; mô phỏng hiện dùng giả định mặc định trong `repo/int8_sim.py`.
- Mạng xuất logit INT8 ở lưới thô (B: 1x4x40x80, tức 1/8 ảnh vào 640x320); A53 phóng bilinear (align_corners=False) lên lưới ảnh vào rồi chọn lớp.
  Không có PyTorch trên board (chỉ NumPy, OpenCV, Python 3.10).
- Ngân sách MAC/bộ nhớ của IP chưa chốt (TV2 chưa trả lời). B: 0,725 GMAC ở 640x320, tensor lớn nhất khoảng 3,1 MiB (A: 0,585 GMAC, 800 KiB).

## Dữ liệu
- Ảnh BDD100K (giấy phép phi thương mại), crop (0,0,1280,600) rồi resize (PIL BILINEAR; nhãn NEAREST). B dùng 640x320; mốc cũ 320x160.
- Nhãn 4 lớp: 0 không đường, 1 đường không thuộc ta, 2 làn đang đi, 3 làn kề, 255 bỏ qua, 254 = "không đường HOẶC không thuộc ta" (ảnh bộ 100K, loss gộp 2 lớp).
- Mask drivable (bộ 100K): id 0 = nền, 1 = làn đang đi, 2 = làn kề (README của dataset ghi sai thứ tự; đã kiểm bằng ảnh màu).
- Test: 408 ảnh theo nhóm video, không trùng train. Mỗi nhóm so sánh chỉ trên CÙNG tập này. Không có tập test độc lập thật sự.

## Quy ước làm việc
- Báo số ở epoch CUỐI, không chọn epoch tốt nhất. So sánh nhiều seed (nhiễu giữa seed khoảng 0,002 đến 0,009), dùng t kiểu Welch; coi t >= 3 là có ý nghĩa.
- Đặt tiêu chí quyết định TRƯỚC khi chạy. Nêu giả định và điều chưa đo.
- Mã trong `repo/` (huấn luyện, INT8) và `runtime/` (phía board) là nguồn duy nhất. Notebook trong `notebooks/` do `make_notebook.py` sinh ra:
  sửa mã ở đây, commit, push, rồi Colab chạy cell deploy. Không sửa tay notebook trên Colab (lần sinh sau sẽ ghi đè).
- Dữ liệu, trọng số, kết quả nằm trên Google Drive (`/content/drive/MyDrive/capstone_ai`), không bao giờ commit lên Git. Không đưa token vào mã.
- Claude Code không có GPU và không thấy Drive: viết mã và cell để người dùng chạy trên Colab.

## Kết quả hiện tại (408 ảnh test, epoch cuối, lưới 320x160 qua `iou_320`; bảng đầy đủ: `docs/results_2026-11-23.md`)
- B (`hd640s1_p1s5_kdT_c4aug_s*`, FP32, 3 seed): làn đang đi 0,8100 ± 0,0041; làn kề 0,6484 ± 0,0066; không thuộc ta 0,5045 ± 0,0021; IoU đường 0,8927 ± 0,0017.
- Sau INT8 (hiệu chuẩn pct, một mô hình mỗi bên): B (s0) 0,8125 / 0,6423 / 0,4995 / 0,8894; A (s0) 0,8042 / 0,6275 / 0,4498 / 0,8689
  (thứ tự: làn đang đi, làn kề, không thuộc ta, IoU đường). INT8 của B trùng từng bit giữa `int8_ref.py` và mô phỏng.
  So với FP32 của chính B s0 (0,8131 / 0,6475 / 0,5021 / 0,8908), INT8 làm làn kề giảm 0,0052; A chỉ giảm không quá 0,0023 mỗi lớp.
- Hàng gần xe: trước đây khoảng 16% ảnh lệch tâm làn quá 100 px (mục tiêu dưới 10%). Cách tính con số đó không ghi trong repo;
  `runtime/eval_dataset.py` có định nghĩa riêng (tâm làn ở hàng mẫu gần xe nhất, pixel khung 1280 rộng), CHƯA chạy trên dữ liệu thật.
- Đã thử, KHÔNG giúp: đầu ra 1/2, gộp 3 lớp, tắt lật ngang (còn hại). Giúp: aug mạnh, thêm dữ liệu (làn), độ phân giải 640x320, chưng cất (ở "không thuộc ta", IoU đường).
- Thêm 4 shard ở 320x160 làm "không thuộc ta" và IoU đường đi lùi (nhãn 254 chiếm 76,8% pixel). Chưa thử 4 shard ở 640x320.

## Quy ước tên run
- Tên run = `{tag}_c4aug_s{seed}` (4 lớp, aug mạnh). Run cũ KHÔNG đổi tên.
- Tag hiện dùng (notebook `20_train`): `hd640s{N}_p1s5_reg` (gói chống quá khớp) và `hd640s{N}_p1s5_kdT` (thêm chưng cất từ `hd640s0_mitb2_c4aug_s0`), N = số shard 640x320.
  Mốc so sánh: B hiện tại `hd640s1_p1s5_kdT_c4aug_s*` (3 seed), so bằng `iou_320`.
- Tag cũ (lịch sử, không chạy thêm): `pilot` (2.826 ảnh thí điểm), `b9s{n}` (thí điểm + n shard 320x160), hậu tố `_deep`, `_ctx`, `_fw4`, `_wide`;
  `hd{W}s{N}_mitb2`, `_unetr34` (giáo viên), `_p1s5`, `_p1s5_kdC`, `_lp1s5_kdT`. `pilot_c4_fsnet_m` là aug cơ bản, KHÔNG thuộc mốc `pilot_c4aug_s*`.
- `final_metrics.json`: `iou` tính ở lưới của chính run (`w` x `h`); run HD có thêm `iou_320` (argmax HD, thu nhỏ nearest về lưới 320x160, so nhãn 320x160).
  Chỉ `iou_320` so thẳng được với mốc 320x160.

## Quyết định chốt (2026-11-24)
- **Chỉ phát triển một mô hình là B**: `fsnet_m_p1s5`, ảnh vào 640x320, đầu ra logit INT8 1x4x40x80, gói `handoff_s3b_fsnet_m_p1s5_640x320`
  (run `hd640s1_p1s5_kdT_c4aug_s0`, trọng số `models_final/B_hd640_p1s5_kdT_s0_weights.pt`).
- Lý do: B hơn A ở cả bốn chỉ số sau INT8 (làn đang đi 0,8125 so với 0,8042, làn kề 0,6423 so với 0,6275, không thuộc ta 0,4995 so với 0,4498,
  IoU đường 0,8894 so với 0,8689), và đã lượng tử xong trùng từng bit.
- **A ngừng phát triển**; gói `handoff_s3b` (A, 320x160) chỉ giữ làm dự phòng.
- Ghi chú: quyết định này thay quyết định 23/11 (A chính, vì B ở FP32 chỉ hơn A +0,0068 làn đang đi, t = 2,4, chưa đạt ngưỡng +0,01 và t >= 3 đặt trước).
  So sánh sau INT8 là một mô hình mỗi bên, không có t. A và B khác nhau nhiều yếu tố cùng lúc (số shard, cỡ ảnh, mạng, chống quá khớp, chưng cất).
- Còn chờ: TV2 xác nhận ngân sách và đầu vào 640x320. Chưa đặt ngưỡng độ trễ (ví dụ 100 ms mỗi khung): chưa chốt, không dùng làm tiêu chí.
- `models_final/`, `results/`, `context_pack/`, `eval_runtime/` là kết quả và trọng số: KHÔNG commit. `docs/` chỉ chứa bảng số tổng hợp và đặc tả giao diện.

## Đang mở
- Notebook (4, do `make_notebook.py` sinh): `10_data` (cache thí điểm 640x320, kiểm độc lập, dựng shard 640x320 chạy nền, `N_SHARDS` mặc định 4),
  `20_train` (chỉ B; `N_SHARDS`, `SEEDS`, `KD`; tự bỏ qua run đã xong, `train_v3.py` tự tiếp tục từ `last.pt`; cell so sánh với B bằng Welch theo tiêu chí đặt trước;
  cell đóng băng vào `models_final/` không ghi đè tên đã có; bảng kết quả; `context_pack`), `30_eval` (kiểm đơn vị và runtime trên 408 ảnh test với gói B thật),
  `40_int8` (mặc định B, ghi `handoff_s3b_fsnet_m_p1s5_640x320`). `60_queue`, `70_hd` đã xóa (git giữ lịch sử).
- Việc kế tiếp của huấn luyện: `20_train` mặc định `hd640s4_p1s5_kdT` (4 shard 640x320, 3 seed). Tiêu chí nhận thay B, đặt trước: làn đang đi +0,01 với t >= 3 so với B,
  IoU đường không giảm, false_floor không tăng (so trung bình), TV2 xác nhận ngân sách; rồi INT8 (`40_int8`) và `30_eval` phải ĐẠT.
- Runtime phía board (`runtime/`, cấu hình `config/pipeline_b.json`, `config/preprocess_b.json`, giao diện `docs/interfaces/perception_v0.md`):
  `preprocess` (bản NumPy của PIL BILINEAR, trùng từng byte với PIL; lượng tử bằng bảng tra), `infer` (onnx, int8ref; vart và ip để `NotImplementedError`),
  `postprocess` (bilinear như torch, argmax, tầng 2: N = 32 cột khoảng trống, M = 8 hàng mép và tâm làn; bản nhanh argmax ở lưới logit), `types` (4 tầng, gói nhị phân 932 byte),
  `pipeline` (thời gian từng khâu), `viz`, `eval_dataset` (công cụ cho `30_eval`). Kiểm: `python -m unittest discover -s tests` (thêm `PERCEPTION_PACK=gói` để kiểm với gói thật).
  Đã kiểm bằng dữ liệu giả và gói B dựng từ trọng số thật trên dữ liệu giả: tiền xử lý trùng cache từng byte, IoU trùng `int8_sim` (lệch 0), mặt nạ NumPy trùng torch,
  int8ref trùng golden từng bit. CHƯA chạy với ảnh và gói thật (cần Colab), CHƯA đo trên board (A53). Tham số tầng 2 chưa chỉnh trên ảnh webcam.
- Cờ của `train_v3.py` (mặc định tắt): `--teacher --kd --kd_T` (KD tính cả trên pixel nhãn 254), chống quá khớp `--ema --wd --wd_nobn --ls --cutout`, `--lr_enc_mult`, `--accum`,
  `--stream` (đọc lô từ đĩa; tự bật khi dữ liệu > 6 GB), `--arch smp:...`, `--pretrained`. Gói `REG` của B: EMA 0,999, wd 0,01 không áp cho bias/BN, label smoothing 0,05,
  cutout 2 ô. Mọi giá trị CHƯA chỉnh. `iou_trainsub` đo khoảng cách train - test: B khoảng 0,049, `reg` 0,032, giáo viên 0,118 (làn đang đi).
  `last.pt["model"]` luôn là trọng số triển khai (bản EMA nếu bật); `int8_sim.py` đọc đúng khoá này.
- Bộ công cụ INT8 (`int8_sim.py`, `int8_pack.py`, `int8_ref.py`, `make_hd_calib.py`) nhận `--arch --w --h` (mặc định trong script vẫn là `fsnet_m`, 320, 160;
  notebook truyền tham số của B nên không đổi mặc định) và `--weights`. Bước FP32 phải khớp `final_metrics.json` trong 0,002, hơn thì dừng.
  sha256 byte của ONNX phụ thuộc phiên bản torch/onnx: so bằng `python repo/int8_sim.py --fingerprint tệp.onnx` (dấu vân tay trọng số).
- Đã chạy trên GPU L4 24 GB (Colab, 2026-11-22/23): giáo viên MiT-B2 70 đến 74 s/epoch (bộ thí điểm); học trò +1 shard 44 s/epoch, 133 s/epoch khi có KD.
  Chưa đo: 4 shard 640x320 (thời gian, dung lượng đĩa: khoảng 6 GB mỗi shard).
- Preset trong `models_custom.py`: `fsnet_m_p1s5` (B), `fsnet_l_p1s5`, `fsnet_m_deep`, `fsnet_m_ctx`, `fsnet_l`, `fsnet_l_p1` (đạt `contract_violations` rỗng).
- Chờ TV2: ngân sách IP và bộ nhớ (B: tensor lớn nhất 3,1 MiB), đầu vào 640x320 (E3), đệm kênh, Y6 chi tiết, hàm resize, driver và bố cục bộ nhớ cho backend `ip`.
- Chưa làm: hiệu chuẩn webcam (S4) và tầng 3 (cm); thống nhất `perception_v0` với TV3 (S0: kênh truyền, tần số, đồng hồ chung); lọc theo thời gian và hợp nhất radar,
  siêu âm (S5 mới có tầng 2 từng khung).
