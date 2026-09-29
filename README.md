# Phân tích năng lượng và phát hiện tiếng nói

Dự án xử lý các tệp WAV, tính năng lượng ngắn hạn (Short-Time Energy, STE) và ước lượng khoảng thời gian có tiếng nói. Kết quả của ba thuật toán được đối chiếu với nhãn thủ công trong `manual_labels.csv` bằng sai số bình phương trung bình (MSE).

## Cấu trúc thư mục

```text
SignalProcessing/
├── audio_data/             # Các tệp âm thanh WAV
├── energy.py               # Vẽ waveform và STE của một tệp
├── theta.py                # Chạy và so sánh ba thuật toán
├── manual_labels.csv       # Nhãn tham chiếu: file, theta, N1, N2
└── automatic_results.csv   # Kết quả tổng hợp được tạo bởi theta.py
```

Thư mục âm thanh hiện có các tệp WAV được đặt tên theo dạng `thanh1.wav`, `tien1.wav`, `toan1.wav`, `tuan1.wav` và `viet1.wav` (cùng các số thứ tự khác).

## Yêu cầu

Python 3 và các thư viện:

```bash
pip install librosa numpy pandas scipy matplotlib mplcursors
```

## Dữ liệu nhãn thủ công

`manual_labels.csv` cần có các cột:

| Cột | Ý nghĩa |
| --- | --- |
| `file` | Tên tệp WAV trong `audio_data/` |
| `theta` | Ngưỡng năng lượng tham chiếu |
| `N1` | Thời điểm bắt đầu tiếng nói, tính bằng giây |
| `N2` | Thời điểm kết thúc tiếng nói, tính bằng giây |

Mỗi tên trong cột `file` cần khớp với tên WAV để kết quả tự động có thể ghép với nhãn tương ứng.

## Chạy so sánh thuật toán

## Thêm dữ liệu mới

Để đưa bản ghi mới vào quá trình phân tích:

1. Chép tệp âm thanh vào thư mục `audio_data/`. Tệp cần ở định dạng WAV (`.wav`).
2. Thêm một dòng cho tệp đó vào `manual_labels.csv`, với các giá trị `file`, `theta`, `N1` và `N2`. Giữ nguyên hàng tiêu đề và thứ tự cột hiện có.
3. Điền `file` đúng tên tệp, bao gồm phần mở rộng `.wav` (ví dụ `mau_moi.wav`). `N1` và `N2` là thời điểm bắt đầu và kết thúc tiếng nói tính bằng giây; `theta` là ngưỡng năng lượng tham chiếu.
4. Chạy lại `python theta.py` để xử lý các WAV trong `audio_data/` và cập nhật `automatic_results.csv`.

Ví dụ về một dòng nhãn:

```csv
mau_moi.wav,1.25,0.32,4.80
```

Mỗi WAV cần có đúng một dòng nhãn với tên khớp trong CSV. Nếu không khớp, chương trình sẽ không ghép được WAV đó với nhãn khi tính MSE.

Từ thư mục dự án, chạy:

```bash
python theta.py
```

Chương trình đọc các WAV trong `audio_data/`, chuyển audio về mono 16 kHz, lọc dải thông 80–4000 Hz, chia tín hiệu thành khung 100 ms với bước 10 ms và tính STE. Sau đó:

1. Quét thời lượng dùng để ước lượng ngưỡng của thuật toán 1 từ 0,1 đến 1,0 giây, chọn thời lượng có tổng MSE thấp nhất so với nhãn thủ công.
2. Chạy ba thuật toán và báo MSE cho `theta`, `N1` và `N2`.
3. Ghi kết quả ghép với nhãn thủ công vào `automatic_results.csv`.

Các thuật toán trong `theta.py`:

- **Thuật toán 1 — Max Energy:** lấy giá trị STE lớn nhất trong đoạn đầu có thời lượng được chọn qua grid search làm ngưỡng.
- **Thuật toán 2 — Mean + Std:** lấy trung bình cộng với ba lần độ lệch chuẩn của STE trong 0,5 giây đầu làm ngưỡng.
- **Thuật toán 3 — Dual Threshold (STE + ZCR):** dùng ngưỡng STE cao để tìm vùng tiếng nói, sau đó mở rộng biên bằng ngưỡng STE thấp hoặc tỷ lệ qua điểm không (ZCR).

Trong file kết quả, các cột `theta_algo1`, `N1_algo1`, `N2_algo1` (tương tự cho `algo2` và `algo3`) là kết quả của từng thuật toán; các cột `theta`, `N1`, `N2` là nhãn thủ công. `file_manual` là tên tệp từ bảng nhãn sau khi ghép.

## Xem waveform và STE

Chạy:

```bash
python energy.py
```

Nhập tên một tệp trong `audio_data/`, có thể bỏ phần mở rộng `.wav`. Chương trình mở biểu đồ waveform và STE, với thông tin thời gian và giá trị hiển thị khi rê chuột lên đường biểu đồ.

## Tham số xử lý

Các hằng số chính được khai báo đầu `theta.py`: `FS_TARGET` (16 000 Hz), `FRAME_DURATION` (0,1 giây), `HOP_DURATION` (0,01 giây), `AUDIO_DIR`, `MANUAL_FILE` và `RESULT_FILE`. Trong `energy.py`, cửa sổ và bước khung hiện được đặt trực tiếp lần lượt là 100 ms và 10 ms.
