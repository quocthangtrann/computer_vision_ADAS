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
- **B2** (`hd640s4_p1s5_kdT_c4aug_s*`, 4 shard 640x320, FP32, 3 seed, MÔ HÌNH CHÍNH): làn đang đi 0,8265 ± 0,0013; làn kề 0,6781 ± 0,0069; không thuộc ta 0,5278 ± 0,0029;
  IoU đường 0,9003 ± 0,0007; false_floor 0,0120 ± 0,0003. So với B (Welch, 3 seed mỗi bên): +0,0165 (t = 6,6); +0,0297 (5,4); +0,0233 (11,1); +0,0076 (7,1); false_floor -0,0009 (t = -2,9, chưa tới 3).
  Khoảng cách train - test ở làn đang đi 0,0465 (B 0,0492). Đường cong test phẳng từ epoch 40. B và B2 chỉ khác số shard (1 so với 4), nên "thêm dữ liệu giúp" có cơ sở.
- B (`hd640s1_p1s5_kdT_c4aug_s*`, FP32, 3 seed, DỰ PHÒNG): làn đang đi 0,8100 ± 0,0041; làn kề 0,6484 ± 0,0066; không thuộc ta 0,5045 ± 0,0021; IoU đường 0,8927 ± 0,0017.
- Sau INT8 (hiệu chuẩn pct, một mô hình mỗi bên, s0; thứ tự: làn đang đi, làn kề, không thuộc ta, IoU đường): B2 0,8223 / 0,6702 / 0,5273 / 0,8985; B 0,8125 / 0,6423 / 0,4995 / 0,8894;
  A 0,8042 / 0,6275 / 0,4498 / 0,8689. INT8 của B và B2 trùng từng bit giữa `int8_ref.py` và mô phỏng. Bit tích lũy cần: B2 19, A 20.
  INT8 trừ FP32 (s0): B2 -0,0043 / -0,0079 / -0,0025 / -0,0010 (FP32 0,8266 / 0,6781 / 0,5298 / 0,8995); B làn kề -0,0052, làn đang đi khoảng -0,0006; A không quá 0,0023 mỗi lớp.
  B2 mất nhiều hơn B khi lượng tử hóa, nhưng sau INT8 vẫn hơn B ở cả bốn chỉ số (một mô hình mỗi bên, không có t). Chưa đặt ngưỡng riêng cho mức giảm INT8.
- `30_eval` trên B2 (gói `handoff_s3b_fsnet_m_p1s5_640x320_s4`, theo kết quả người dùng dán): ĐẠT; int8ref [0,9671, 0,5273, 0,8223, 0,6702] đường 0,8985 và onnx [0,9674, 0,5298, 0,8266, 0,6781] đường 0,8995
  khớp `int8_sim`; logit trùng từng bit. Thời gian CPU Colab (onnx): pre 17,8 ms, post 8,3 ms, tier2 1,5 ms (chưa phải A53).
- Hàng gần xe: trước đây khoảng 16% ảnh lệch tâm làn quá 100 px (mục tiêu dưới 10%). Cách tính con số đó không ghi trong repo;
  `runtime/eval_dataset.py` có định nghĩa riêng (tâm làn ở hàng mẫu gần xe nhất, pixel khung 1280 rộng; có bản "chặt" tính ảnh thiếu tâm là lỗi). Số của B2 chưa có trong tài liệu này.
- Đã thử, KHÔNG giúp: đầu ra 1/2, gộp 3 lớp, tắt lật ngang (còn hại). Giúp: aug mạnh, thêm dữ liệu (làn), độ phân giải 640x320, chưng cất (ở "không thuộc ta", IoU đường).
- Thêm 4 shard ở 320x160 làm "không thuộc ta" và IoU đường đi lùi (nhãn 254 chiếm 76,8% pixel). Ở 640x320 với chưng cất, 4 shard (B2) tốt hơn 1 shard (B) ở cả bốn chỉ số.

## Quy ước tên run
- Tên run = `{tag}_c4aug_s{seed}` (4 lớp, aug mạnh). Run cũ KHÔNG đổi tên.
- Tag hiện dùng (notebook `20_train`): `hd640s{N}_p1s5_reg` (gói chống quá khớp) và `hd640s{N}_p1s5_kdT` (thêm chưng cất từ `hd640s0_mitb2_c4aug_s0`), N = số shard 640x320.
  Mốc so sánh từ nay: B2 `hd640s4_p1s5_kdT_c4aug_s*` (3 seed), so bằng `iou_320`. B `hd640s1_p1s5_kdT_c4aug_s*` là mốc của lần so B2 với B (`BASE` trong `20_train` vẫn trỏ B; đổi sang B2 trước thí nghiệm kế).
- Tag cũ (lịch sử, không chạy thêm): `pilot` (2.826 ảnh thí điểm), `b9s{n}` (thí điểm + n shard 320x160), hậu tố `_deep`, `_ctx`, `_fw4`, `_wide`;
  `hd{W}s{N}_mitb2`, `_unetr34` (giáo viên), `_p1s5`, `_p1s5_kdC`, `_lp1s5_kdT`. `pilot_c4_fsnet_m` là aug cơ bản, KHÔNG thuộc mốc `pilot_c4aug_s*`.
- `final_metrics.json`: `iou` tính ở lưới của chính run (`w` x `h`); run HD có thêm `iou_320` (argmax HD, thu nhỏ nearest về lưới 320x160, so nhãn 320x160).
  Chỉ `iou_320` so thẳng được với mốc 320x160.

## Quyết định chốt (2026-11-25)
- **Mô hình chính là B2**: `fsnet_m_p1s5`, ảnh vào 640x320, đầu ra logit INT8 1x4x40x80, gói `handoff_s3b_fsnet_m_p1s5_640x320_s4`
  (run `hd640s4_p1s5_kdT_c4aug_s0`, trọng số `models_final/B2_hd640s4_p1s5_kdT_s0_weights.pt`). s0 được chọn trước, không chọn seed tốt nhất.
- Tiêu chí đặt trước (làn đang đi +0,01 với t >= 3, IoU đường không giảm, false_floor không tăng, INT8 và `30_eval` ĐẠT) đều ĐẠT, trừ ngân sách TV2 (chưa xác nhận).
  B2 cùng kiến trúc với B nên IP không đổi, chỉ đổi trọng số: 0,725 GMAC, tensor lớn nhất khoảng 3,1 MiB.
- **B dự phòng**: gói `handoff_s3b_fsnet_m_p1s5_640x320`, cấu hình `config/pipeline_b_fallback.json`. **A ngừng phát triển**; gói `handoff_s3b` (A, 320x160) chỉ giữ làm dự phòng cũ.
- Lịch sử: 23/11 chọn A (B ở FP32 chỉ hơn A +0,0068, t = 2,4); 24/11 chọn B (hơn A ở bốn chỉ số sau INT8, một mô hình mỗi bên); 25/11 chọn B2.
- Còn chờ: TV2 xác nhận ngân sách và đầu vào 640x320. Chưa đặt ngưỡng độ trễ của cả chuỗi (ví dụ 100 ms mỗi khung): chưa chốt, không dùng làm tiêu chí.
  Mục tiêu nội bộ của phần ARM (không phải hợp đồng): pre + post + tier2 <= 10 ms trên CPU Colab, vì A53 chậm hơn nhiều (chưa đo trên A53).
- `models_final/`, `results/`, `context_pack/`, `eval_runtime/`, `eval_runtime_cmp/` là kết quả và trọng số: KHÔNG commit. `docs/` chỉ chứa bảng số tổng hợp và đặc tả giao diện.

## Đang mở
- Notebook (4, do `make_notebook.py` sinh): `10_data` (cache thí điểm 640x320, kiểm độc lập, dựng shard 640x320 chạy nền, `N_SHARDS` mặc định 4),
  `20_train` (chỉ B; `N_SHARDS`, `SEEDS`, `KD`; tự bỏ qua run đã xong, `train_v3.py` tự tiếp tục từ `last.pt`; cell so sánh với B bằng Welch theo tiêu chí đặt trước;
  cell đóng băng vào `models_final/` không ghi đè tên đã có (mặc định đóng băng B2 s0); bảng kết quả; `context_pack`), `30_eval` (kiểm đơn vị, runtime trên 408 ảnh test với gói lấy từ CHÍNH
  `config/pipeline_b.json`, cell 4 so sánh các chế độ, cell vẽ), `40_int8` (mặc định B2, ghi `..._s4`, KHÔNG đặt `OUT` trùng gói đang dùng). `60_queue`, `70_hd` đã xóa (git giữ lịch sử).
  Colab đã lưu `40_int8.ipynb` kèm output (commit "Được tạo bằng Colab"); sinh lại notebook đó bằng `make_notebook.py` sẽ xóa output, chỉ khác dòng chú thích.
- Việc kế tiếp của huấn luyện: `BASE` của `20_train` đổi sang B2 trước thí nghiệm kế. Hướng chưa thử (nêu để chọn, chưa cam kết): tắt KD ở 4 shard (`reg`, đo KD còn giúp không),
  huấn luyện lại giáo viên MiT-B2 trên 4 shard (hiện giáo viên chỉ thấy bộ thí điểm và kém học trò ở làn đang đi). Mỗi seed 4 shard khoảng 7 giờ (425 s/epoch, 60 epoch, có KD).
- Runtime phía board (`runtime/`, cấu hình `config/pipeline_b.json` + `preprocess_b.json` trỏ gói B2; `*_fallback.json` trỏ gói B; giao diện `docs/interfaces/perception_v0.md`):
  `preprocess` (bản NumPy của PIL BILINEAR, trùng từng byte với PIL; lượng tử bằng bảng tra; tùy chọn `cv2_linear`, `cv2_area` có đường nhanh không chép ảnh cắt cỡ đầy đủ), `infer` (onnx, int8ref; vart và ip để `NotImplementedError`),
  `postprocess` (bilinear như torch, argmax, tầng 2: N = 32 cột khoảng trống, M = 8 hàng mép và tâm làn; chế độ `standard`, `fast` (argmax ở lưới logit), `half` (lưới 160x320), `quarter` (80x160);
  chế độ khác `standard` đặt cờ `fast` của gói), `types` (4 tầng, gói nhị phân 932 byte), `pipeline` (thời gian từng khâu; `run_stream` chạy gối đầu hai luồng: tiền xử lý khung t+1 lúc xử lý khung t, kết quả y hệt tuần tự),
  `viz`, `eval_dataset` (công cụ cho `30_eval`). Kiểm: `python -m unittest discover -s tests` (50 test; thêm `PERCEPTION_PACK=gói` để kiểm với gói thật).
  Máy Mac: Python hệ thống có numpy 2 làm hỏng torch và cv2 (2 test torch lỗi); chạy trong venv numpy < 2.
  Đã kiểm bằng dữ liệu giả và gói B dựng từ trọng số thật trên dữ liệu giả: tiền xử lý trùng cache từng byte, IoU trùng `int8_sim` (lệch 0), mặt nạ NumPy trùng torch,
  int8ref trùng golden từng bit; các bảng so sánh chế độ chạy được (số IoU và lệch tâm trên dữ liệu giả vô nghĩa). Tham số tầng 2 chưa chỉnh trên ảnh webcam.
- Đang chờ kết quả Colab (`30_eval` cell 4, onnx, 408 ảnh), tiêu chí ĐẶT TRƯỚC: (1) resize `cv2_linear` chỉ được làm mặc định nếu IoU (bốn lớp và IoU đường) giảm không quá 0,001 so với PIL NumPy,
  nếu không giữ PIL NumPy (`cv2_linear` không khử răng cưa nên ảnh khác lúc huấn luyện; đo trên FP32, chưa qua lượng tử INT8); (2) chế độ hậu xử lý `half`, `quarter`, `fast` chỉ dùng nếu tỉ lệ ảnh lệch tâm làn > 100 px
  (tính cả ảnh thiếu tâm) không xấu hơn `standard` quá 0,01, và chỉ kết luận khi >= 30 ảnh có tâm làn; (3) mục tiêu pre + post + tier2 <= 10 ms trên CPU Colab. Chưa có số thật nên chưa đổi mặc định nào.
  Đã thử trên Mac, không nhanh hơn nên không dùng: bilinear tính tại chỗ và argmax bằng so sánh thủ công (kết quả y hệt bit, thời gian không giảm).
- Cờ của `train_v3.py` (mặc định tắt): `--teacher --kd --kd_T` (KD tính cả trên pixel nhãn 254), chống quá khớp `--ema --wd --wd_nobn --ls --cutout`, `--lr_enc_mult`, `--accum`,
  `--stream` (đọc lô từ đĩa; tự bật khi dữ liệu > 6 GB), `--arch smp:...`, `--pretrained`. Gói `REG` của B và B2: EMA 0,999, wd 0,01 không áp cho bias/BN, label smoothing 0,05,
  cutout 2 ô. Mọi giá trị CHƯA chỉnh. `iou_trainsub` đo khoảng cách train - test (làn đang đi): B2 0,0465, B khoảng 0,049, `reg` 0,032, giáo viên 0,118.
  `last.pt["model"]` luôn là trọng số triển khai (bản EMA nếu bật); `int8_sim.py` đọc đúng khoá này.
- Bộ công cụ INT8 (`int8_sim.py`, `int8_pack.py`, `int8_ref.py`, `make_hd_calib.py`) nhận `--arch --w --h` (mặc định trong script vẫn là `fsnet_m`, 320, 160;
  notebook truyền tham số của B nên không đổi mặc định) và `--weights`. Bước FP32 phải khớp `final_metrics.json` trong 0,002, hơn thì dừng.
  sha256 byte của ONNX phụ thuộc phiên bản torch/onnx: so bằng `python repo/int8_sim.py --fingerprint tệp.onnx` (dấu vân tay trọng số).
- Đã chạy trên GPU L4 24 GB (Colab, 2026-11-22/23): giáo viên MiT-B2 70 đến 74 s/epoch (bộ thí điểm); học trò +1 shard 44 s/epoch, 133 s/epoch khi có KD.
  4 shard 640x320 có KD: 425 s/epoch (60 epoch khoảng 7 giờ mỗi seed, 3 seed 1.278 phút).
- Preset trong `models_custom.py`: `fsnet_m_p1s5` (B, B2), `fsnet_l_p1s5`, `fsnet_m_deep`, `fsnet_m_ctx`, `fsnet_l`, `fsnet_l_p1` (đạt `contract_violations` rỗng).
- Chờ TV2: ngân sách IP và bộ nhớ (B2 và B: tensor lớn nhất 3,1 MiB, bit tích lũy 19), đầu vào 640x320 (E3), đệm kênh, Y6 chi tiết, hàm resize, driver và bố cục bộ nhớ cho backend `ip`.
- Chưa làm: hiệu chuẩn webcam (S4) và tầng 3 (cm); thống nhất `perception_v0` với TV3 (S0: kênh truyền, tần số, đồng hồ chung); lọc theo thời gian và hợp nhất radar,
  siêu âm (S5 mới có tầng 2 từng khung).
