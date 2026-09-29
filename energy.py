"""Plot the waveform and short-time energy of a WAV under audio_data/."""

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import mplcursors
import numpy as np

AUDIO_DIR = Path(__file__).resolve().parent / 'audio_data'


def find_audio(user_input):
    value = user_input.strip().strip('"')
    if not value.lower().endswith('.wav'):
        value += '.wav'

    # Accept paths such as "fan sound/viet1.wav" as well as a bare filename.
    candidate = (AUDIO_DIR / value).resolve()
    if candidate.is_relative_to(AUDIO_DIR.resolve()) and candidate.is_file():
        return candidate

    matches = [path for path in AUDIO_DIR.rglob('*.wav')
               if path.name.lower() == Path(value).name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print('Tên file xuất hiện ở nhiều môi trường. Hãy nhập đường dẫn tương đối:')
        for path in matches:
            print(f'  {path.relative_to(AUDIO_DIR)}')
        return None

    print(f'Không tìm thấy WAV: {value}')
    print('Các file hiện có:')
    for path in sorted(AUDIO_DIR.rglob('*.wav')):
        print(f'  {path.relative_to(AUDIO_DIR)}')
    return None


def main():
    print('=' * 50)
    filename = input('Nhập tên hoặc đường dẫn WAV trong audio_data: ')
    audio_path = find_audio(filename)
    if audio_path is None:
        return

    print(f'Đang xử lý: {audio_path.relative_to(AUDIO_DIR)}')
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(0.100 * sr)
    hop_length = int(0.010 * sr)
    if len(y) < frame_length:
        print('File ngắn hơn một khung 100 ms, không thể tính STE.')
        return

    ste = np.array([
        np.sum((y[i:i + frame_length] * np.hamming(frame_length)) ** 2)
        for i in range(0, len(y) - frame_length + 1, hop_length)
    ])
    time_signal = np.arange(len(y)) / sr
    time_ste = np.arange(len(ste)) * hop_length / sr
    display_name = str(audio_path.relative_to(AUDIO_DIR))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.canvas.manager.set_window_title(f'Phân tích STE - {display_name}')
    (line1,) = ax1.plot(time_signal, y, color='b', alpha=0.6, label='Amplitude')
    ax1.set_title(f'Tín hiệu âm thanh theo thời gian - {display_name}')
    ax1.set_ylabel('Biên độ')
    ax1.grid(True)

    (line2,) = ax2.plot(time_ste, ste, color='r', linewidth=1.5, label='STE')
    ax2.set_title('Năng lượng thời gian ngắn (Short-Time Energy - STE)')
    ax2.set_xlabel('Thời gian (giây)')
    ax2.set_ylabel('Năng lượng')
    ax2.grid(True)

    cursor1 = mplcursors.cursor(line1, hover=True)

    @cursor1.connect('add')
    def show_waveform_value(selection):
        selection.annotation.set_text(
            f'Thời gian: {selection.target[0]:.3f}s\nBiên độ: {selection.target[1]:.4f}')

    cursor2 = mplcursors.cursor(line2, hover=True)

    @cursor2.connect('add')
    def show_energy_value(selection):
        selection.annotation.set_text(
            f'Thời gian: {selection.target[0]:.3f}s\nNăng lượng STE: {selection.target[1]:.4f}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
