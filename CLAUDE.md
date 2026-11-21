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
- Mạng xuất logit ở 1/4 độ phân giải; A53 phóng bilinear (align_corners=False) rồi chọn lớp. Không có PyTorch trên board (chỉ NumPy, OpenCV).
- Ngân sách MAC/bộ nhớ của IP chưa chốt (TV2 chưa trả lời). FSNet-M hiện 0,585 GMAC ở 320x160.

## Dữ liệu
- Ảnh BDD100K (giấy phép phi thương mại), crop (0,0,1280,600) rồi resize về 320x160 (PIL BILINEAR; nhãn NEAREST).
- Nhãn 4 lớp: 0 không đường, 1 đường không thuộc ta, 2 làn đang đi, 3 làn kề, 255 bỏ qua, 254 = "không đường HOẶC không thuộc ta" (ảnh bộ 100K, loss gộp 2 lớp).
- Mask drivable (bộ 100K): id 0 = nền, 1 = làn đang đi, 2 = làn kề (README của dataset ghi sai thứ tự; đã kiểm bằng ảnh màu).
- Test: 408 ảnh theo nhóm video, không trùng train. Mỗi nhóm so sánh chỉ trên CÙNG tập này. Không có tập test độc lập thật sự.

## Quy ước làm việc
- Báo số ở epoch CUỐI, không chọn epoch tốt nhất. So sánh nhiều seed (nhiễu giữa seed khoảng 0,002 đến 0,009), dùng t kiểu Welch; coi t >= 3 là có ý nghĩa.
- Đặt tiêu chí quyết định TRƯỚC khi chạy. Nêu giả định và điều chưa đo.
- Mã trong `repo/` là nguồn duy nhất. Notebook trong `notebooks/` do `make_notebook.py` sinh ra: sửa mã ở đây, commit, push, rồi Colab chạy cell deploy.
- Dữ liệu, trọng số, kết quả nằm trên Google Drive (`/content/drive/MyDrive/capstone_ai`), không bao giờ commit lên Git. Không đưa token vào mã.
- Claude Code không có GPU và không thấy Drive: viết mã và cell để người dùng chạy trên Colab.

## Kết quả hiện tại (408 ảnh test, FSNet-M 4 lớp, 4 shard bộ 100K, aug mạnh, epoch cuối, 3 seed)
- Làn đang đi 0,803 ± 0,003; làn kề 0,624 ± 0,004; không thuộc ta 0,456 ± 0,004; IoU đường 0,870.
- INT8 (hiệu chuẩn pct): mỗi lớp giảm không quá 0,0023 so với FP32; bit tích lũy cần 20.
- Hàng gần xe: khoảng 16% ảnh lệch tâm làn quá 100 px (mục tiêu dưới 10%).
- Đã thử, KHÔNG giúp: đầu ra 1/2, gộp 3 lớp, tắt lật ngang (còn hại). Giúp: aug mạnh, thêm dữ liệu (làn).
- Thêm 4 shard làm "không thuộc ta" và IoU đường đi lùi (nhãn 254 chiếm 76,8% pixel).

## Quy ước tên run
- Tên run = `{tag}_c4aug_s{seed}` (4 lớp, aug mạnh). Tag: `pilot` (2.826 ảnh thí điểm), `b9s{n}` (thí điểm + n shard b9), `hd640_*` (thí điểm ở 640x320).
  Hậu tố biến thể: `_deep`, `_ctx`, `_fw4`, `_wide`. Ví dụ `b9s1_deep_c4aug_s0`, `hd640_p1s5_c4aug_s2`, `b9s4_fw4_c4aug_s1`.
- Run cũ KHÔNG đổi tên (`pilot_c4_*`, `pilot_c4aug_*`, `b9s1_c4aug_*`, `b9s4_c4aug_*`). Mốc của từng nhóm là run cũ cùng dữ liệu và cùng aug:
  HD so với `pilot_c4aug_s*` (5 seed); `_deep`, `_ctx` so với `b9s1_c4aug_s*` (5 seed); `_fw4`, `_wide` so với `b9s4_c4aug_s*` (3 seed).
  `pilot_c4_fsnet_m` là aug cơ bản, KHÔNG thuộc mốc `pilot_c4aug_s*`.
- `final_metrics.json`: `iou` tính ở lưới của chính run (`w` x `h`); run HD có thêm `iou_320` (argmax HD, thu nhỏ nearest về lưới 320x160, so nhãn 320x160).
  Chỉ `iou_320` so thẳng được với mốc 320x160.

## Đang mở
- Thí nghiệm nâng cấp, chạy bằng `notebooks/60_queue.ipynb` (biến `ONLY` chọn nhóm). Preset mới trong `models_custom.py` (ở 320x160, 4 lớp):
  `fsnet_m_deep` (n4=5) 0,644 GMAC; `fsnet_m_ctx` (thêm tầng 1/32, ảnh vào chia hết 32) 0,658 GMAC; `fsnet_l` (e1..e4 = 48, 64, 128, 192; d1, d2 = 64, 48) 1,092 GMAC;
  `fsnet_m` là 0,585 GMAC. Cả ba đạt `contract_violations` rỗng.
  Hàng đợi: (a) HD thí điểm 640x320, 3 seed: `fsnet_m_p1`, `fsnet_m_p1s5`, `fsnet_m` (GMAC ở 640x320: 0,651; 0,725; 2,339);
  (b) `fsnet_m_deep` rồi `fsnet_m_ctx`, 1 shard, 3 seed; (c) `fsnet_m` 4 shard, `--full_w 4`, 3 seed; (d) `fsnet_l` 4 shard, CHỈ seed 0 (n = 1, không có t).
  Chốt trước 25/11; chỉ nhận nếu làn đang đi tăng >= 0,01 (t >= 3), IoU đường và false_floor không xấu đi, và TV2 xác nhận ngân sách.
  Nhóm (b) chạy ở 1 shard để sàng lọc: tiêu chí trên đặt cho 4 shard, nên kết quả (b) chưa đủ để nhận. Nhóm (d) cần thêm seed trước khi kết luận.
- Chưa chạy trên GPU (mới kiểm bằng dữ liệu giả trên CPU): `train_v3.py` với `--w --h --full_w`, `make_hd_pilot.py`, hàng đợi 60_queue.
  Bắt buộc trước khi dựng HD: `make_hd_pilot.py --w 320 --h 160` phải in "ĐẠT" (trùng 100% cache 320x160 có sẵn).
- `make_notebook.py` đã lệch với `10_data.ipynb` và `20_train.ipynb` (Colab đã sửa tay: `N_SHARDS = 4`, cell dọn tiến trình): sinh lại sẽ ghi đè. Cần đồng bộ trước khi sinh lại hai notebook đó.
- Chờ TV2: ngân sách IP và bộ nhớ, đệm kênh, Y6 chi tiết, hàm resize, độ phân giải đầu vào (E3).
- Chưa làm: hiệu chuẩn webcam (S4), hậu xử lý làn và `PerceptionResult` (S5), hợp đồng gói kết quả (S0).