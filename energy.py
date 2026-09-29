import os
import librosa
import matplotlib.pyplot as plt
import mplcursors  # Thư viện hỗ trợ tương tác hover/click
import numpy as np

# Thư mục chứa dữ liệu âm thanh
AUDIO_DIR = 'audio_data'

# 1. Nhập tên tệp từ bàn phím
print("=" * 50)
filename = input("Nhập tên tệp âm thanh: ").strip()

# Tự động thêm đuôi .wav nếu người dùng quên nhập
if not filename.lower().endswith('.wav'):
  filename += '.wav'

audio_path = os.path.join(AUDIO_DIR, filename)

# Kiểm tra sự tồn tại của tệp
if not os.path.exists(audio_path):
  print(f"\nLỗi: Không tìm thấy tệp '{filename}' trong thư mục '{AUDIO_DIR}'!")
  if os.path.exists(AUDIO_DIR):
    available_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith('.wav')]
    print(f"Các tệp hiện có trong '{AUDIO_DIR}': {available_files}")
  else:
    print(f"Thư mục '{AUDIO_DIR}' không tồn tại.")
  exit()

print(f"--> Đang xử lý tệp: {audio_path}")

# 2. Tải tệp âm thanh
y, sr = librosa.load(audio_path, sr=None)

# 3. Tham số khung (100 ms frame, 10 ms shift)[cite: 1]
frame_length = int(0.100 * sr)
hop_length = int(0.010 * sr)

# 4. Tính Năng lượng thời gian ngắn (STE)
ste = np.array([
    np.sum((y[i : i + frame_length] * np.hamming(len(y[i : i + frame_length]))) ** 2)
    for i in range(0, len(y) - frame_length + 1, hop_length)
])

# 5. Trục thời gian
time_signal = np.linspace(0, len(y) / sr, num=len(y))
time_ste = np.linspace(0, len(y) / sr, num=len(ste))

# 6. Vẽ đồ thị
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.canvas.manager.set_window_title(f'Phân tích STE - {filename}')

# Tín hiệu gốc
(line1,) = ax1.plot(time_signal, y, color='b', alpha=0.6, label='Amplitude')
ax1.set_title(f'Tín hiệu âm thanh theo thời gian (Waveform) - {filename}')
ax1.set_ylabel('Biên độ')
ax1.grid(True)

# Đồ thị STE
(line2,) = ax2.plot(time_ste, ste, color='r', linewidth=1.5, label='STE')
ax2.set_title('Năng lượng thời gian ngắn (Short-Time Energy - STE)')
ax2.set_xlabel('Thời gian (giây)')
ax2.set_ylabel('Năng lượng')
ax2.grid(True)

# 7. Bật tính năng tương tác (Click/Hover để xem thông số)
cursor1 = mplcursors.cursor(line1, hover=True)


@cursor1.connect("add")
def _(sel):
  sel.annotation.set_text(
      f"Thời gian: {sel.target[0]:.3f}s\nBiên độ: {sel.target[1]:.4f}"
  )


cursor2 = mplcursors.cursor(line2, hover=True)


@cursor2.connect("add")
def _(sel):
  sel.annotation.set_text(
      f"Thời gian: {sel.target[0]:.3f}s\nNăng lượng STE: {sel.target[1]:.4f}"
  )


plt.tight_layout()
plt.show()