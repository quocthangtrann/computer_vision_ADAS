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
- Tag HD (notebook `70_hd`): `hd{W}s{N}` (cỡ W, N shard; ví dụ `hd640s1`). Biến thể: `_mitb2`, `_unetr34` (giáo viên), `_p1s5`, `_p1s5_reg`, `_p1s5_kdT`, `_p1s5_kdC`, `_lp1s5_kdT` (học trò).
  Mốc của `hd640s1_*` là `b9s1_c4aug_s*`, của `hd640s4_*` là `b9s4_c4aug_s*`, của `hd640s0_*` là `pilot_c4aug_s*` (so bằng `iou_320`).
- `final_metrics.json`: `iou` tính ở lưới của chính run (`w` x `h`); run HD có thêm `iou_320` (argmax HD, thu nhỏ nearest về lưới 320x160, so nhãn 320x160).
  Chỉ `iou_320` so thẳng được với mốc 320x160.

## Đang mở
- ƯU TIÊN HIỆN TẠI: huấn luyện ở độ phân giải cao (640x320; 1280x576 chỉ cho bộ thí điểm), KHÔNG chạy thêm run 320x160 chất lượng thấp.
  Hàng đợi 320x160 trong `60_queue` (nhóm b, c, d) bị hạ ưu tiên; nhóm a của `60_queue` (HD thí điểm không giáo viên) cũng được `70_hd` thay thế.
- Notebook `notebooks/70_hd.ipynb` (biến `W, H, NS` ở cell cấu hình; `ONLY` chọn nhóm). Trình tự: dựng cache thí điểm HD (kiểm chứng 320x160 phải ĐẠT) -> dựng shard HD
  (`b9_prepare_v2.py --w --h`, ghi vào `data/v2/b9_chunks_{W}x{H}/`) -> giáo viên (nhóm t) -> học trò (s), chưng cất (k), học trò lớn (l) -> cell so sánh.
  Giáo viên: `smp:Unet:mit_b2` (transformer, 27,5 M tham số) và đối chứng `smp:Unet:resnet34` (conv, 24,4 M). Học trò: `fsnet_m_p1s5` (0,725 GMAC ở 640x320),
  `fsnet_l_p1s5` (1,324 GMAC ở 640x320; mới), lưới logit 80x40 (p1 giữ lưới logit bằng 320x160; muốn lưới mịn hơn dùng `fsnet_m`/`fsnet_l` không `pre`, 160x80).
  Chỉ học trò là mạng chạy trên IP; giáo viên chỉ dùng lúc huấn luyện, không bị ràng buộc hợp đồng. Attention KHÔNG đưa vào mạng chạy trên IP.
- Cờ mới của `train_v3.py` (mặc định tắt, run cũ chạy lại cho cùng kết quả: đã kiểm bằng dữ liệu giả): `--teacher --kd --kd_T` (KD tính cả trên pixel nhãn 254),
  chống quá khớp `--ema --wd --wd_nobn --ls --cutout`, `--lr_enc_mult`, `--accum`, `--stream` (đọc lô từ đĩa; tự bật khi dữ liệu > 6 GB), `--arch smp:...`, `--pretrained`.
  Gói chống quá khớp của học trò (`REG`): EMA 0,999, wd 0,01 không áp cho bias/BN, label smoothing 0,05, cutout 2 ô (nhãn giữ nguyên). Mọi giá trị CHƯA chỉnh, cần đo từng yếu tố.
  Mỗi run ghi `iou_trainsub` (ảnh train nhãn đủ, không aug) để đo khoảng cách train - test thay vì giả định có quá khớp: FSNet-M 2 lớp trước đây KHÔNG quá khớp (train ~ val).
  `last.pt["model"]` luôn là trọng số triển khai (bản EMA nếu bật); `int8_sim.py` đọc đúng khoá này.
- Quyết định đặt trước: giáo viên phải hơn mốc FSNet cùng dữ liệu >= 0,01 ở làn đang đi thì mới chưng cất; nhận học trò mới chỉ khi làn đang đi tăng >= 0,01 so với mốc (t >= 3),
  IoU đường và false_floor không xấu đi, và TV2 xác nhận ngân sách. KD từ transformer chỉ được coi là "transformer giúp" nếu hơn KD từ conv (`kdT` so với `kdC`).
- Preset trong `models_custom.py`: `fsnet_m_deep`, `fsnet_m_ctx`, `fsnet_l`, `fsnet_l_p1`, `fsnet_l_p1s5` (đạt `contract_violations` rỗng).
- CHƯA chạy trên GPU (mới kiểm bằng dữ liệu giả trên CPU): toàn bộ `train_v3.py` mới, `make_hd_pilot.py`, `b9_prepare_v2.py` ở cỡ HD, giáo viên MiT-B2, các hàng đợi `60_queue` và `70_hd`.
  Chưa kiểm: tải trọng số ImageNet cho `smp` (cần mạng trên Colab), bộ nhớ GPU của giáo viên ở 640x320 và 1280x576, thời gian mỗi epoch, dung lượng đĩa Colab (shard 640x320 khoảng 6 GB mỗi shard, chép về đĩa cục bộ).
  Shard 1280x576 khoảng 22 GB mỗi shard: chưa dùng.
- `make_notebook.py` đã lệch với `10_data.ipynb` và `20_train.ipynb` (Colab đã sửa tay: `N_SHARDS = 4`, cell dọn tiến trình): sinh lại sẽ ghi đè. Cần đồng bộ trước khi sinh lại hai notebook đó.
- Chờ TV2: ngân sách IP và bộ nhớ, đệm kênh, Y6 chi tiết, hàm resize, độ phân giải đầu vào (E3).
- Chưa làm: hiệu chuẩn webcam (S4), hậu xử lý làn và `PerceptionResult` (S5), hợp đồng gói kết quả (S0).
