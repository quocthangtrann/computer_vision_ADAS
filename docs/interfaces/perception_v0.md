# Giao diện kết quả nhận thức `perception_v0`

Đọc giả: TV3 (điều khiển, nhúng) và TV2 (FPGA). Mã nguồn: `runtime/types.py` (`pack`, `unpack`), kiểm ở `tests/test_packet.py`.
Trạng thái: **bản nháp v0**, chưa thống nhất với TV3 (S0). Đổi bố cục thì tăng `version`.

## 1. Bốn tầng kết quả

| Tầng | Nội dung | Kiểu | Cỡ với mô hình B | Có trong gói |
|---|---|---|---|---|
| 0 | Logit đầu ra của mạng, kèm thang: `logit = q * scale` | int8 (4, 40, 80) + float32 | 12.800 byte | tùy chọn |
| 1 | Mặt nạ lớp ở lưới ảnh vào của mạng: 0 không đường, 1 đường không thuộc ta, 2 làn đang đi, 3 làn kề | uint8 (320, 640) | 204.800 byte | tùy chọn |
| 2 | Tọa độ pixel khung camera: N cột khoảng trống, M hàng mẫu của làn đang đi | float32 + cờ uint8 | 424 byte (N = 32, M = 8) | luôn có |
| 3 | Như tầng 2 nhưng ở hệ xe, đơn vị cm | float32 + cờ uint8 | 424 byte | luôn có, **hiện để trống** |

Tầng 3 cần hiệu chuẩn webcam (S4). Chưa có S4 thì mọi giá trị tầng 3 là NaN, mọi cờ bằng 0, và bit `TIER3_CALIBRATED` của gói bằng 0.

### Hệ tọa độ pixel khung (tầng 2)
- Gốc ở góc trên trái khung camera (B: 1280 x 720). x tăng sang phải, y tăng xuống dưới. Tâm pixel nằm ở số nguyên, như OpenCV: pixel i phủ đoạn [i − 0,5; i + 0,5].
- Mạng chỉ thấy vùng cắt (0, 0, 1280, 600), resize về 640 x 320. Các hàng 600 đến 719 của khung (sát mũi xe) **không được nhìn thấy**.
- Đổi từ lưới mạng sang khung: `x_khung = x0 + (x_lưới + 0,5) * cw / gw − 0,5`, tương tự cho y. Với B: `x = 2 * x_lưới + 0,5`, `y = 1,875 * y_lưới + 0,4375`.
  Ma trận 3 x 3 nằm ở `Prepared.grid_to_frame`; trong gói, các khối tầng 0 và 1 có kèm ma trận affine 2 x 3 của riêng chúng.

### Tầng 2: ý nghĩa từng trường
Cột i (i = 0..N−1, từ trái sang phải; tâm cột trên lưới ở `x = (i + 0,5) * W / N`):
- `col_x[i]`: x của cột (pixel khung).
- `col_y[i]`: mép trên của đoạn "đi được" liên tục tính từ đáy vùng cắt lên. Đây là ranh giới giữa pixel đi được cao nhất và pixel không đi được ngay trên nó.
  Mặc định "đi được" là lớp 2 hoặc 3. Bit `DRIVABLE_CLASS1` của gói bằng 1 nghĩa là có tính cả lớp 1.
- `col_flags[i]`: bit 0 `VALID` (pixel đáy của cột đi được); bit 1 `REACHES_TOP` (đi được tới hàng trên cùng của vùng cắt, nên khoảng trống thật có thể xa hơn và `col_y` chỉ là cận).

Hàng k (k = 0 là hàng gần xe nhất, đi dần ra xa). Mặc định các hàng cách đều từ 98% đến 60% chiều cao lưới (B: hàng lưới 313 đến 191, tức y khung từ 587 đến 359):
- `row_y[k]`: y của hàng (pixel khung).
- `row_left[k]`, `row_right[k]`: mép trái và mép phải của làn đang đi, lấy theo ranh giới ngoài của đoạn lớp 2. `row_center[k]` là trung điểm của hai mép.
- `row_flags[k]`:

  | Bit | Tên | Ý nghĩa |
  |---|---|---|
  | 0 | `LEFT` | mép trái hợp lệ (không chạm biên trái ảnh) |
  | 1 | `RIGHT` | mép phải hợp lệ (không chạm biên phải ảnh) |
  | 2 | `CENTER` | tâm hợp lệ (có cả hai mép) |
  | 3 | `OBSTACLE` | trong làn có vùng lớp 0 (vật cản) kẹp giữa hai đoạn lớp 2; mép làn vẫn tính theo hai đầu ngoài |
  | 4 | `FOUND` | hàng có pixel lớp 2 (kể cả khi mép chạm biên) |

- Khi có nhiều đoạn lớp 2 trên một hàng, đoạn được chọn như sau:
  - hàng đầu (k = 0) chọn đoạn chứa trục xe (`ego_x_frac`, mặc định giữa ảnh) hoặc đoạn gần trục xe nhất;
  - mỗi hàng sau chọn đoạn gần tâm của hàng ngay dưới nó nhất.

**Giá trị không hợp lệ là NaN. Luôn đọc cờ trước khi dùng giá trị.**

Tham số mặc định (32 cột, 8 hàng, 60% đến 98%, `merge_max_gap_frac` 0,5) nằm ở `config/pipeline_b.json` và **chưa chỉnh trên ảnh webcam thật**.

### Tầng 3 (dự kiến, chưa có số)
- Hệ trục: gốc ở cản trước, giữa xe. `fwd` hướng ra trước, `lat` dương sang trái, đơn vị cm.
- Các trường: `col_lat_cm`, `col_fwd_cm`, `col_flags` (cùng N cột với tầng 2); `row_fwd_cm`, `row_left_cm`, `row_right_cm`, `row_center_cm`, `row_flags` (cùng M hàng).
- Các trường lấy từ tầng 2 qua homography mặt đất khi có S4. Hợp nhất với radar và siêu âm làm ở tầng sau, không làm trong gói này.

## 2. Bố cục gói nhị phân

Tất cả theo little-endian, không có đệm căn lề trong tiêu đề. Với C, đọc bằng `memcpy` hoặc `struct __attribute__((packed))`.
Mỗi khối được đệm cho độ dài chia hết cho 4.

### Tiêu đề (80 byte)

| Offset | Cỡ | Trường | Kiểu | Ghi chú |
|---|---|---|---|---|
| 0 | 4 | magic | char[4] | `PERC` |
| 4 | 2 | version | u16 | 0 |
| 6 | 2 | header_size | u16 | 80 |
| 8 | 4 | seq | u32 | số thứ tự khung, tăng 1 mỗi khung |
| 12 | 8 | t_capture_ns | u64 | lúc chụp khung (`CLOCK_MONOTONIC`, ns) do nơi gọi truyền vào; không truyền thì là lúc bắt đầu xử lý |
| 20 | 8 | t_publish_ns | u64 | lúc đóng gói (`CLOCK_MONOTONIC`, ns) |
| 28 | 4 | flags | u32 | bit 0 có tầng 0; bit 1 có tầng 1; bit 2 tầng 3 đã hiệu chuẩn; bit 3 "đi được" tính cả lớp 1; bit 4 hậu xử lý bản nhanh |
| 32 | 2 | frame_w | u16 | khung camera, B: 1280 |
| 34 | 2 | frame_h | u16 | B: 720 |
| 36 | 2 | grid_w | u16 | lưới dùng để tính tầng 2, B: 640 |
| 38 | 2 | grid_h | u16 | B: 320 |
| 40 | 2 | n_cols | u16 | N |
| 42 | 2 | n_rows | u16 | M |
| 44 | 16 | model | char[16] | ASCII, đệm 0, ví dụ `fsnet_m_p1s5` |
| 60 | 4 | payload_size | u32 | số byte thân (không gồm tiêu đề và CRC) |
| 64 | 16 | t_pre_us, t_infer_us, t_post_us, t_tier2_us | u32 x 4 | thời gian từng khâu (µs) |

### Thân

| Thứ tự | Trường | Kiểu | Số phần tử |
|---|---|---|---|
| 1 | tầng 2: col_x, col_y | f32 | N, N |
| 2 | tầng 2: row_y, row_left, row_right, row_center | f32 | M, M, M, M |
| 3 | tầng 3: col_lat_cm, col_fwd_cm | f32 | N, N |
| 4 | tầng 3: row_fwd_cm, row_left_cm, row_right_cm, row_center_cm | f32 | M x 4 |
| 5 | cờ: col_flags tầng 2, row_flags tầng 2, col_flags tầng 3, row_flags tầng 3 | u8 | N, M, N, M (+ đệm tới bội 4) |
| 6 | (nếu flags bit 0) tầng 0: scale f32, c u16, h u16, w u16, dự phòng u16, affine 2 x 3 f32, rồi q int8 [c][h][w] | | 36 + c·h·w byte (+ đệm) |
| 7 | (nếu flags bit 1) tầng 1: h u16, w u16, affine 2 x 3 f32, rồi m u8 [h][w] | | 28 + h·w byte (+ đệm) |

### Đuôi
- CRC32 (u32), là `zlib.crc32` (đa thức 0x04C11DB7, giống Ethernet), tính trên toàn bộ tiêu đề và thân.

### Kích thước gói (N = 32, M = 8)

| Nội dung | Byte |
|---|---|
| Tầng 2 + tầng 3 (mặc định) | **932** |
| + tầng 0 (logit 4 x 40 x 80) | 13.768 |
| + tầng 1 (mặt nạ 320 x 640) | 205.760 |
| cả hai | 218.596 |

Tổng quát: `80 + 8·(2N + 4M) + 2(N + M) + đệm + 4`.

## 3. Điều chưa thống nhất hoặc chưa đo
- Kênh truyền tới TV3 (UDP, UART hay bộ nhớ chung) và tần số khung: chưa chốt. Với UART, gói 932 byte ở 921.600 baud mất khoảng 10 ms.
- Đồng hồ chung cho `t_capture_ns` giữa camera, radar và siêu âm: chưa chốt.
- Tầng 3: chờ S4 (hiệu chuẩn webcam, homography mặt đất).
- Ngưỡng độ trễ: chưa đặt. Thời gian từng khâu trên A53 chưa đo; số đo trên CPU Colab hoặc PC không đại diện cho board.
