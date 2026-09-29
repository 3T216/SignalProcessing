import os
import re
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ============================================================
# CẤU HÌNH
# ============================================================
FS_TARGET = 16000

FRAME_DURATION = 0.1  # 100 ms
HOP_DURATION = 0.01  # 10 ms

AUDIO_DIR = 'audio_data'
MANUAL_FILE = 'manual_labels.csv'
RESULT_FILE = 'automatic_results.csv'


# ============================================================
# 1. ĐỌC AUDIO & XỬ LÝ TÍN HIỆU
# ============================================================
def load_audio(path, sr=FS_TARGET):
  audio, fs = librosa.load(path, sr=sr, mono=True)
  return audio, fs


def bandpass(audio, fs, low=80, high=4000, order=3):
  nyq = fs / 2
  b, a = butter(order, [low / nyq, high / nyq], btype='band')
  return filtfilt(b, a, audio)


def framing(audio, fs):
  frame_length = int(FRAME_DURATION * fs)
  hop_length = int(HOP_DURATION * fs)

  frames = []
  for start in range(0, len(audio) - frame_length + 1, hop_length):
    frames.append(audio[start : start + frame_length])

  return np.array(frames), frame_length, hop_length


def ste(frames):
  window = np.hamming(frames.shape[1])
  frames = frames * window
  energy = np.sum(frames**2, axis=1)
  return energy


def zcr(frames):
  return np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)


# ============================================================
# 2. CÁC THUẬT TOÁN ĐIỀU KIỆN / THUẬT TOÁN ĐỀ XUẤT
# ============================================================


# --- Thuật toán gốc 1: Max Energy 1s đầu ---
def detect_speech_algo1(energy, hop_length, fs, duration=1.0):
  n_frames = int(duration * fs / hop_length)
  n_frames = min(n_frames, len(energy))

  theta = np.max(energy[:n_frames])
  speech_frames = np.where(energy > theta)[0]

  if len(speech_frames) == 0:
    return theta, 0.0, 0.0

  N1 = speech_frames[0] * hop_length / fs
  N2 = speech_frames[-1] * hop_length / fs
  return theta, N1, N2


# --- Thuật toán đề xuất 1: Adaptive Statistical Thresholding (Mean + Std) ---
def detect_speech_algo2(energy, hop_length, fs, duration=0.5, k=3.0):
  n_frames = int(duration * fs / hop_length)
  n_frames = min(n_frames, len(energy))

  noise_segment = energy[:n_frames]
  theta = np.mean(noise_segment) + k * np.std(noise_segment)
  speech_frames = np.where(energy > theta)[0]

  if len(speech_frames) == 0:
    return theta, 0.0, 0.0

  N1 = speech_frames[0] * hop_length / fs
  N2 = speech_frames[-1] * hop_length / fs
  return theta, N1, N2


# --- Thuật toán đề xuất 2: Dual-Thresholding (STE + ZCR) ---
def detect_speech_algo3(energy, frames, hop_length, fs, duration=0.5):
  n_frames = int(duration * fs / hop_length)
  n_frames = min(n_frames, len(energy))

  zcr_values = zcr(frames)
  noise_ste = energy[:n_frames]
  theta_high = np.mean(noise_ste) + 4 * np.std(noise_ste)
  theta_low = np.mean(noise_ste) + 1.5 * np.std(noise_ste)

  noise_zcr = zcr_values[:n_frames]
  zcr_threshold = np.mean(noise_zcr) + 3 * np.std(noise_zcr)

  speech_high = np.where(energy > theta_high)[0]

  if len(speech_high) == 0:
    return theta_high, 0.0, 0.0

  N1_frame = speech_high[0]
  while N1_frame > 0 and (
      energy[N1_frame - 1] > theta_low or zcr_values[N1_frame - 1] > zcr_threshold
  ):
    N1_frame -= 1

  N2_frame = speech_high[-1]
  while N2_frame < len(energy) - 1 and (
      energy[N2_frame + 1] > theta_low or zcr_values[N2_frame + 1] > zcr_threshold
  ):
    N2_frame += 1

  N1 = N1_frame * hop_length / fs
  N2 = N2_frame * hop_length / fs
  return theta_high, N1, N2


# ============================================================
# 3. XỬ LÝ FILE AUDIO
# ============================================================
def process_file(file_path):
  audio, fs = load_audio(file_path)
  audio = bandpass(audio, fs)
  frames, frame_length, hop_length = framing(audio, fs)
  energy = ste(frames)

  th1, n1_1, n2_1 = detect_speech_algo1(energy, hop_length, fs)
  th2, n1_2, n2_2 = detect_speech_algo2(energy, hop_length, fs)
  th3, n1_3, n2_3 = detect_speech_algo3(energy, frames, hop_length, fs)

  return {
      'file': os.path.basename(file_path),  # Chỉ lưu tên file (vd: thanh1.wav)
      'theta_algo1': th1,
      'N1_algo1': n1_1,
      'N2_algo1': n2_1,
      'theta_algo2': th2,
      'N1_algo2': n1_2,
      'N2_algo2': n2_2,
      'theta_algo3': th3,
      'N1_algo3': n1_3,
      'N2_algo3': n2_3,
  }


# ============================================================
# 4. MAIN PROGRAM & MSE CALCULATION
# ============================================================
if __name__ == '__main__':
  files = [
      os.path.join(AUDIO_DIR, f)
      for f in os.listdir(AUDIO_DIR)
      if f.lower().endswith('.wav')
  ]

  def natural_sort_key(s):
    filename = os.path.basename(s)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', filename)
    ]

  files.sort(key=natural_sort_key)
  print(f'Tìm thấy {len(files)} file WAV trong thư mục.\n')

  results = []
  for file_path in files:
    results.append(process_file(file_path))

  df_auto = pd.DataFrame(results)

  # --------------------------------------------------------
  # TÍNH TOÁN MSE SO VỚI MANUAL_LABELS.CSV
  # --------------------------------------------------------
  if os.path.exists(MANUAL_FILE):
    # Đọc tệp CSV gán nhãn thủ công
    df_manual = pd.read_csv(MANUAL_FILE)

    # Chuẩn hóa cột tên file ở cả 2 dataframe về cùng định dạng chữ thường
    df_auto['file_clean'] = (
        df_auto['file'].astype(str).str.strip().str.lower()
    )
    df_manual['file_clean'] = (
        df_manual['file'].astype(str).str.strip().str.lower()
    )

    # Khớp dữ liệu tự động và manual theo tên file
    df_merged = pd.merge(
        df_auto, df_manual, on='file_clean', suffixes=('', '_manual')
    )

    if not df_merged.empty:
      print('=' * 65)
      print(f'KẾT QUẢ TÍNH MSE (So sánh trên {len(df_merged)} file trùng khớp):')
      print('=' * 65)

      # Tính MSE cho từng thuật toán với các cột theta, N1, N2 từ CSV
      for algo_id, algo_name in [
          ('1', 'Thuật toán 1 (Max Energy - Gốc)'),
          ('2', 'Thuật toán 2 (Adaptive Mean+Std - Đề xuất 1)'),
          ('3', 'Thuật toán 3 (Dual-Threshold STE+ZCR - Đề xuất 2)'),
      ]:
        mse_theta = np.mean(
            (df_merged[f'theta_algo{algo_id}'] - df_merged['theta']) ** 2
        )
        mse_n1 = np.mean(
            (df_merged[f'N1_algo{algo_id}'] - df_merged['N1']) ** 2
        )
        mse_n2 = np.mean(
            (df_merged[f'N2_algo{algo_id}'] - df_merged['N2']) ** 2
        )

        print(f'► {algo_name}:')
        print(f'   - MSE (theta) : {mse_theta:.8f}')
        print(f'   - MSE (N1)    : {mse_n1:.6f}')
        print(f'   - MSE (N2)    : {mse_n2:.6f}')
        print('-' * 65)

      # Lưu lại bảng tổng hợp
      df_merged.drop(columns=['file_clean'], inplace=True)
      df_merged.to_csv(RESULT_FILE, index=False)
      print(f'Đã xuất bảng kết quả tổng hợp vào: {RESULT_FILE}')
    else:
      print(
          'LỖI: Không tìm thấy file tên nào trùng nhau giữa thư mục và CSV!'
      )
  else:
    print(f'Không tìm thấy file {MANUAL_FILE}!')