import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# ============================================================
# CẤU HÌNH
# ============================================================
FS_TARGET = 16000

FRAME_DURATION = 0.1      # 100 ms
HOP_DURATION = 0.01       # 10 ms

AUDIO_DIR = 'audio_data'
MANUAL_FILE = 'manual_labels.csv'
RESULT_FILE = 'automatic_results.csv'


# ============================================================
# 1. ĐỌC AUDIO
# ============================================================
def load_audio(path, sr=FS_TARGET):

    audio, fs = librosa.load(
        path,
        sr=sr,
        mono=True
    )

    return audio, fs


# ============================================================
# 2. BANDPASS FILTER
# ============================================================
def bandpass(audio, fs, low=80, high=4000, order=3):

    nyq = fs / 2

    b, a = butter(
        order,
        [low / nyq, high / nyq],
        btype='band'
    )

    return filtfilt(b, a, audio)


# ============================================================
# 3. FRAMING
# ============================================================
def framing(audio, fs):

    frame_length = int(FRAME_DURATION * fs)
    hop_length = int(HOP_DURATION * fs)

    frames = []

    for start in range(
        0,
        len(audio) - frame_length + 1,
        hop_length
    ):

        frames.append(
            audio[start:start + frame_length]
        )

    return np.array(frames), frame_length, hop_length


# ============================================================
# 4. SHORT-TIME ENERGY
# ============================================================
def ste(frames):

    window = np.hamming(frames.shape[1])

    frames = frames * window

    energy = np.sum(frames ** 2, axis=1)

    return energy


# ============================================================
# 5. TỰ ĐỘNG TÌM THETA, N1, N2
# ============================================================
def detect_speech(energy, hop_length, fs, duration=1.0):

    # --------------------------------------------------------
    # Số frame nằm trong duration giây đầu
    # --------------------------------------------------------
    n_frames = int(duration * fs / hop_length)

    n_frames = min(
        n_frames,
        len(energy)
    )

    # --------------------------------------------------------
    # THETA:
    # max STE trong duration giây đầu
    # --------------------------------------------------------
    theta = np.max(
        energy[:n_frames]
    )

    # --------------------------------------------------------
    # Tìm các frame có STE > theta
    # --------------------------------------------------------
    speech_frames = np.where(
        energy > theta
    )[0]

    # Không tìm thấy frame
    if len(speech_frames) == 0:

        return theta, None, None

    # --------------------------------------------------------
    # N1 và N2
    # --------------------------------------------------------
    N1_frame = speech_frames[0]
    N2_frame = speech_frames[-1]

    # Chuyển frame index -> giây
    N1 = N1_frame * hop_length / fs
    N2 = N2_frame * hop_length / fs

    return theta, N1, N2


# ============================================================
# 6. XỬ LÝ MỘT FILE
# ============================================================
def process_file(file_path):

    file_name = os.path.basename(file_path)

    # Đọc audio
    audio, fs = load_audio(
        file_path
    )

    # Bandpass
    audio = bandpass(
        audio,
        fs
    )

    # Framing
    frames, frame_length, hop_length = framing(
        audio,
        fs
    )

    # STE
    energy = ste(
        frames
    )

    # Tự động tìm theta, N1, N2
    theta, N1, N2 = detect_speech(
        energy,
        hop_length,
        fs,
        duration=1.0
    )

    return {
        'file': file_path,
        'theta_auto': theta,
        'N1_auto': N1,
        'N2_auto': N2
    }


# ============================================================
# 7. MAIN
# ============================================================
if __name__ == '__main__':

    # --------------------------------------------------------
    # Lấy tất cả file WAV
    # --------------------------------------------------------
    files = [
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith('.wav')
    ]

    # Sắp xếp theo số trong tên file
    import re

    def natural_sort_key(s):
        filename = os.path.basename(s)
        # Tách tên file thành danh sách các chuỗi chữ và chuỗi số
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', filename)]


    files.sort(key=natural_sort_key)

    print(f'Tìm thấy {len(files)} file WAV.')
    print()

    results = []

    # --------------------------------------------------------
    # Xử lý từng file
    # --------------------------------------------------------
    for file_path in files:

        result = process_file(
            file_path
        )

        results.append(
            result
        )

        print('=' * 50)
        print(
            f"File: {os.path.basename(file_path)}"
        )

        print(
            f"Theta auto = "
            f"{result['theta_auto']:.6f}"
        )

        print(
            f"N1 auto    = "
            f"{result['N1_auto']:.4f} s"
        )

        print(
            f"N2 auto    = "
            f"{result['N2_auto']:.4f} s"
        )

    # --------------------------------------------------------
    # Lưu kết quả
    # --------------------------------------------------------
    df = pd.DataFrame(
        results
    )

    df.to_csv(
        RESULT_FILE,
        index=False
    )

    print()
    print('=' * 50)
    print(f'Đã lưu kết quả vào: {RESULT_FILE}')
    print('=' * 50)

    print()
    print(df)