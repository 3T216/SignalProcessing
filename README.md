# Phân tích năng lượng và phát hiện tiếng nói

Dự án xử lý các tệp WAV, tính năng lượng ngắn hạn (Short-Time Energy, STE), ước lượng khoảng thời gian có tiếng nói và so sánh ba thuật toán với nhãn tham chiếu trong `manual_labels.csv`. Audio được chia theo môi trường trong các thư mục con của `audio_data/`; chương trình tính và lưu bảng MSE riêng cho từng môi trường.

## Cấu trúc

```text
SignalProcessing/
├── audio_data/
│   ├── fan sound/
│   ├── fan + noisy music sound/
│   ├── highway + wind sound/
│   ├── machines in construction site sound/
│   └── metal cut-off saw sound/
├── energy.py
├── theta.py
├── manual_labels.csv
└── results/                  # Được tạo khi chạy theta.py
```

`theta.py` tìm WAV đệ quy trong mọi thư mục con. Tên file WAV phải khớp với cột `file` trong `manual_labels.csv`; tên file hiện đang duy nhất giữa các môi trường.

## Cài đặt

Cần Python 3 và các thư viện sau:

```bash
pip install librosa numpy pandas scipy matplotlib mplcursors
```

## Nhãn tham chiếu

`manual_labels.csv` có các cột:

| Cột | Ý nghĩa |
| --- | --- |
| `file` | Tên WAV, ví dụ `thanh1.wav` |
| `theta` | Ngưỡng năng lượng tham chiếu |
| `N1` | Thời điểm bắt đầu tiếng nói, tính bằng giây |
| `N2` | Thời điểm kết thúc tiếng nói, tính bằng giây |

Mỗi WAV cần có một dòng nhãn và thời gian `N1`, `N2` được tính bằng giây.

## Chạy phân tích theo môi trường

Từ thư mục dự án, chạy:

```bash
python theta.py
```

Chương trình chuyển audio về mono 16 kHz, lọc dải 80–4000 Hz, chia thành khung 100 ms với bước 10 ms và tính STE. Với mỗi môi trường, chương trình:

1. Tối ưu thời lượng lấy mẫu đầu vào cho thuật toán 1 trong khoảng 0,1–1,0 giây dựa trên tổng MSE của `theta`, `N1`, `N2`.
2. Chạy ba thuật toán trên các WAV trong môi trường đó.
3. In MSE cho từng đại lượng và từng thuật toán, rồi ghi bảng kết quả.

Các thuật toán gồm Max Energy với thời lượng tối ưu riêng theo môi trường, ngưỡng Mean + 3 Std, và Dual Threshold kết hợp STE với ZCR.

## Các tệp kết quả

Thư mục `results/` được tạo tự động:

- `<ten_moi_truong>_results.csv`: dự đoán của cả ba thuật toán, ghép với nhãn tham chiếu cho môi trường tương ứng.
- `<ten_moi_truong>_mse.csv`: bảng MSE của `theta`, `N1`, `N2` và tổng MSE cho ba thuật toán; có thêm thời lượng tối ưu của thuật toán 1.
- `mse_by_environment.csv`: gộp các bảng MSE của tất cả môi trường.
- `automatic_results.csv`: gộp kết quả dự đoán và nhãn của các môi trường.

Các file tổng hợp cũ ở thư mục gốc không còn được cập nhật. Hãy đọc kết quả mới trong `results/`.

## Thêm dữ liệu

1. Đặt WAV vào thư mục con phù hợp trong `audio_data/` (tạo thư mục mới nếu cần).
2. Thêm một dòng tương ứng vào `manual_labels.csv`, với tên WAV trong cột `file` và các giá trị `theta`, `N1`, `N2`.
3. Chạy lại `python theta.py`.

Tên môi trường lấy từ tên thư mục cha chứa WAV. Tên WAV phải duy nhất giữa các nhóm vì nhãn được nối theo tên file.

## Xem waveform và STE

`energy.py` dùng để xem waveform và STE của một WAV. Chạy `python energy.py`, sau đó nhập tên file nếu file đó chỉ xuất hiện trong một nhóm (ví dụ `viet1`) hoặc nhập đường dẫn tương đối như `fan sound/viet1.wav`. Nếu tên file trùng giữa các nhóm, chương trình sẽ yêu cầu dùng đường dẫn tương đối.
