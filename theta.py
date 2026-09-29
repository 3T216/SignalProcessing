"""Run the three speech-boundary algorithms and evaluate them by environment."""

import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FS_TARGET = 16000
FRAME_DURATION = 0.1
HOP_DURATION = 0.01
AUDIO_DIR = Path(__file__).resolve().parent / 'audio_data'
MANUAL_FILE = Path(__file__).resolve().parent / 'manual_labels.csv'
OUTPUT_DIR = Path(__file__).resolve().parent / 'results'


def load_audio(path, sr=FS_TARGET):
    return librosa.load(path, sr=sr, mono=True)


def bandpass(audio, fs, low=80, high=4000, order=3):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, audio)


def framing(audio, fs):
    frame_length = int(FRAME_DURATION * fs)
    hop_length = int(HOP_DURATION * fs)
    frames = [audio[i:i + frame_length]
              for i in range(0, len(audio) - frame_length + 1, hop_length)]
    return np.asarray(frames), hop_length


def ste(frames):
    return np.sum((frames * np.hamming(frames.shape[1])) ** 2, axis=1)


def zcr(frames):
    return np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)


def bounds(frames, hop_length, fs):
    if len(frames) == 0:
        return 0.0, 0.0
    return frames[0] * hop_length / fs, frames[-1] * hop_length / fs


def detect_algo1(energy, hop, fs, duration):
    n = min(max(1, int(duration * fs / hop)), len(energy))
    threshold = np.max(energy[:n])
    active = np.flatnonzero(energy >= threshold)
    n1, n2 = bounds(active, hop, fs)
    return threshold, n1, n2


def detect_algo2(energy, hop, fs, duration=0.5, k=3.0):
    n = min(max(1, int(duration * fs / hop)), len(energy))
    noise = energy[:n]
    threshold = np.mean(noise) + k * np.std(noise)
    n1, n2 = bounds(np.flatnonzero(energy > threshold), hop, fs)
    return threshold, n1, n2


def detect_algo3(energy, frames, hop, fs, duration=0.5):
    n = min(max(1, int(duration * fs / hop)), len(energy))
    crossings = zcr(frames)
    noise_energy, noise_zcr = energy[:n], crossings[:n]
    mean_energy, std_energy = np.mean(noise_energy), np.std(noise_energy)
    high = mean_energy + 4 * std_energy
    low = mean_energy + 1.5 * std_energy
    zcr_threshold = np.mean(noise_zcr) + 3 * np.std(noise_zcr)
    active = np.flatnonzero(energy > high)
    if len(active) == 0:
        return high, 0.0, 0.0
    start, end = int(active[0]), int(active[-1])
    while start > 0 and (energy[start - 1] > low or crossings[start - 1] > zcr_threshold):
        start -= 1
    while end < len(energy) - 1 and (energy[end + 1] > low or crossings[end + 1] > zcr_threshold):
        end += 1
    n1, n2 = bounds([start, end], hop, fs)
    return high, n1, n2


def process_file(path):
    audio, fs = load_audio(path)
    audio = bandpass(audio, fs)
    frames, hop = framing(audio, fs)
    if len(frames) == 0:
        raise ValueError(f'Audio is shorter than one frame: {path}')
    return {'file': path.name, 'environment': path.parent.name,
            'energy': ste(frames), 'frames': frames, 'hop': hop, 'fs': fs}


def mse_table(predictions, manual):
    merged = predictions.merge(manual, on='file_clean', suffixes=('', '_manual'))
    if merged.empty:
        raise ValueError('No audio filenames match rows in manual_labels.csv')
    rows = []
    for algo in (1, 2, 3):
        values = {}
        for field in ('theta', 'N1', 'N2'):
            values[f'mse_{field}'] = float(np.mean(
                (merged[f'{field}_algo{algo}'] - merged[field]) ** 2))
        values.update(algorithm=f'algo{algo}', total_mse=sum(values.values()),
                      matched_files=len(merged))
        rows.append(values)
    return merged, pd.DataFrame(rows)


def safe_name(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_') or 'environment'


def main():
    files = sorted(AUDIO_DIR.rglob('*.wav'), key=lambda p: (p.parent.name.lower(),
                   [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', p.name)]))
    if not files:
        raise SystemExit(f'No WAV files found under {AUDIO_DIR}')
    if not MANUAL_FILE.exists():
        raise SystemExit(f'Missing manual labels: {MANUAL_FILE}')

    manual = pd.read_csv(MANUAL_FILE)
    required = {'file', 'theta', 'N1', 'N2'}
    if not required.issubset(manual.columns):
        raise SystemExit(f'manual_labels.csv must contain: {", ".join(sorted(required))}')
    manual['file_clean'] = manual['file'].astype(str).str.strip().str.lower()
    data = [process_file(path) for path in files]
    groups = {}
    for item in data:
        groups.setdefault(item['environment'], []).append(item)

    OUTPUT_DIR.mkdir(exist_ok=True)
    all_rows, all_mse = [], []
    durations = np.round(np.linspace(0.1, 1.0, 10), 2)
    for environment, group in sorted(groups.items()):
        records = []
        for item in group:
            records.append({'file': item['file'], 'file_clean': item['file'].lower(),
                            'environment': environment, 'data': item})
        group_manual = manual[manual['file_clean'].isin(
            [r['file_clean'] for r in records])]
        available = {r['file_clean'] for r in records} & set(group_manual['file_clean'])
        if not available:
            print(f'Skip {environment}: no matching manual labels')
            continue

        best_duration, best_mse = None, float('inf')
        for duration in durations:
            pred = []
            for record in records:
                item = record['data']
                th, n1, n2 = detect_algo1(item['energy'], item['hop'], item['fs'], duration)
                pred.append({'file_clean': record['file_clean'], 'theta_algo1': th,
                             'N1_algo1': n1, 'N2_algo1': n2})
            joined = pd.DataFrame(pred).merge(group_manual, on='file_clean')
            score = sum(np.mean((joined[f'{field}_algo1'] - joined[field]) ** 2)
                        for field in ('theta', 'N1', 'N2'))
            if score < best_mse:
                best_duration, best_mse = float(duration), float(score)

        predictions = []
        for record in records:
            item = record['data']
            th1, n1_1, n2_1 = detect_algo1(item['energy'], item['hop'], item['fs'], best_duration)
            th2, n1_2, n2_2 = detect_algo2(item['energy'], item['hop'], item['fs'])
            th3, n1_3, n2_3 = detect_algo3(item['energy'], item['frames'], item['hop'], item['fs'])
            predictions.append({'file': item['file'], 'environment': environment,
                'theta_algo1': th1, 'N1_algo1': n1_1, 'N2_algo1': n2_1,
                'theta_algo2': th2, 'N1_algo2': n1_2, 'N2_algo2': n2_2,
                'theta_algo3': th3, 'N1_algo3': n1_3, 'N2_algo3': n2_3})
        pred_df = pd.DataFrame(predictions)
        pred_df['file_clean'] = pred_df['file'].str.lower()
        merged, summary = mse_table(pred_df, group_manual)
        summary.insert(0, 'environment', environment)
        summary['algo1_best_duration_s'] = best_duration
        merged.drop(columns='file_clean').to_csv(OUTPUT_DIR / f'{safe_name(environment)}_results.csv', index=False)
        summary.to_csv(OUTPUT_DIR / f'{safe_name(environment)}_mse.csv', index=False)
        all_rows.append(merged)
        all_mse.append(summary)
        print(f'{environment}: {len(merged)} matched files; algo1 duration={best_duration:.1f}s')
        print(summary[['algorithm', 'mse_theta', 'mse_N1', 'mse_N2', 'total_mse']].to_string(index=False))

    if all_rows:
        pd.concat(all_rows, ignore_index=True).drop(columns='file_clean').to_csv(
            OUTPUT_DIR / 'automatic_results.csv', index=False)
        pd.concat(all_mse, ignore_index=True).to_csv(OUTPUT_DIR / 'mse_by_environment.csv', index=False)
        print(f'\nSaved per-environment tables and combined output in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
