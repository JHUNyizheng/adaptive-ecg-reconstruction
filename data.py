"""
PTB-ECG Dataset Preprocessing Module
1. Read data
2. Resample to 500Hz
3. Filter (bandpass filter + 50Hz notch)
4. Z-normalize filtered data (per-channel independently)
5. Sliding window segmentation (1 second window, 500 samples)
6. Save per-patient data with statistics
"""

import os
import numpy as np
import json
import random
import csv
from scipy.signal import resample, butter, filtfilt, iirnotch
import wfdb

RANDOM_SEED = 20251226
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

results_dir = "data_results"
patients_dir = os.path.join(results_dir, "patients")
os.makedirs(patients_dir, exist_ok=True)

NORMALIZATION_MODE = "per_patient"


def efficient_sliding_window(data, window_size=500, step=500):
    if len(data) < window_size:
        return np.array([]).reshape(0, data.shape[1], window_size)

    num_samples, num_channels = data.shape
    rows = (num_samples - window_size) // step + 1

    if rows <= 0:
        return np.array([]).reshape(0, num_channels, window_size)

    shape = (rows, window_size, num_channels)
    strides = (step * data.strides[0], data.strides[0], data.strides[1])
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    return np.transpose(windows, (0, 2, 1))


def design_ecg_filters(fs):
    nyquist = fs / 2.0
    low = 1 / nyquist
    high = 100.0 / nyquist
    b_band, a_band = butter(4, [low, high], btype='band')

    notch_freq = 50.0
    Q = 30.0
    b_notch, a_notch = iirnotch(notch_freq, Q, fs)

    return (b_band, a_band), (b_notch, a_notch)


def apply_ecg_filters(signal, fs):
    band_filter, notch_filter = design_ecg_filters(fs)
    b_band, a_band = band_filter
    b_notch, a_notch = notch_filter

    filtered = filtfilt(b_band, a_band, signal)
    filtered = filtfilt(b_notch, a_notch, filtered)
    return filtered


def process_single_patient_filter(patient_path, target_fs=500):
    try:
        record = wfdb.rdrecord(patient_path)
        ecg_data = record.p_signal
        original_fs = record.fs

        ecg_12lead_indices = list(range(12))
        vcg_lead_indices = [12, 13, 14]

        if original_fs != target_fs:
            new_len = int(len(ecg_data) * (target_fs / original_fs))
            ecg_data = resample(ecg_data, new_len, axis=0)

        ecg_filtered = np.zeros_like(ecg_data)
        for ch in range(ecg_data.shape[1]):
            ecg_filtered[:, ch] = apply_ecg_filters(ecg_data[:, ch], target_fs)

        n_samples = ecg_filtered.shape[0]
        channel_sums = np.sum(ecg_filtered, axis=0)
        channel_sums_sq = np.sum(ecg_filtered ** 2, axis=0)

        ecg_12lead_full = ecg_filtered[:, ecg_12lead_indices]
        vcg_full = ecg_filtered[:, vcg_lead_indices]

        return ecg_12lead_full, vcg_full, channel_sums, channel_sums_sq, n_samples

    except Exception as e:
        print(f"Error processing patient {os.path.basename(patient_path)}: {e}")
        return None, None, None, None, 0


def normalize_split_and_window(patient_data_list, global_means, global_stds, window_size=500, step=500):
    results = []
    for patient_id, ecg_12lead_full, vcg_full in patient_data_list:
        ecg_12lead_norm = np.zeros_like(ecg_12lead_full)
        vcg_norm = np.zeros_like(vcg_full)

        for ch in range(12):
            if global_stds[ch] < 1e-8:
                ecg_12lead_norm[:, ch] = ecg_12lead_full[:, ch] - global_means[ch]
            else:
                ecg_12lead_norm[:, ch] = (ecg_12lead_full[:, ch] - global_means[ch]) / global_stds[ch]

        for ch in range(3):
            std_val = global_stds[12 + ch]
            if std_val < 1e-8:
                vcg_norm[:, ch] = vcg_full[:, ch] - global_means[12 + ch]
            else:
                vcg_norm[:, ch] = (vcg_full[:, ch] - global_means[12 + ch]) / std_val

        ecg_12lead_windows = efficient_sliding_window(ecg_12lead_norm, window_size, step)
        vcg_windows = efficient_sliding_window(vcg_norm, window_size, step)

        min_windows = min(ecg_12lead_windows.shape[0], vcg_windows.shape[0])
        if min_windows == 0:
            print(f"Warning: Patient {patient_id} has insufficient windows, skipping")
            continue

        results.append((patient_id, ecg_12lead_windows[:min_windows], vcg_windows[:min_windows]))

    return results


def process_single_patient_full(patient_path, target_fs=500, window_size=500, step=500):
    try:
        record = wfdb.rdrecord(patient_path)
        ecg_data = record.p_signal
        original_fs = record.fs

        ecg_12lead_indices = list(range(12))
        vcg_lead_indices = [12, 13, 14]

        if original_fs != target_fs:
            new_len = int(len(ecg_data) * (target_fs / original_fs))
            ecg_data = resample(ecg_data, new_len, axis=0)

        ecg_filtered = np.zeros_like(ecg_data)
        for ch in range(ecg_data.shape[1]):
            ecg_filtered[:, ch] = apply_ecg_filters(ecg_data[:, ch], target_fs)

        n_samples = ecg_filtered.shape[0]
        patient_means = np.mean(ecg_filtered, axis=0)
        patient_stds = np.std(ecg_filtered, axis=0)

        ecg_normalized = np.zeros_like(ecg_filtered)
        for ch in range(ecg_filtered.shape[1]):
            if patient_stds[ch] < 1e-8:
                ecg_normalized[:, ch] = ecg_filtered[:, ch] - patient_means[ch]
            else:
                ecg_normalized[:, ch] = (ecg_filtered[:, ch] - patient_means[ch]) / patient_stds[ch]

        ecg_12lead = ecg_normalized[:, ecg_12lead_indices]
        vcg_leads = ecg_normalized[:, vcg_lead_indices]

        ecg_12lead_windows = efficient_sliding_window(ecg_12lead, window_size, step)
        vcg_windows = efficient_sliding_window(vcg_leads, window_size, step)

        min_windows = min(ecg_12lead_windows.shape[0], vcg_windows.shape[0])
        if min_windows == 0:
            print(f"    Warning: Patient {os.path.basename(patient_path)} has insufficient windows")
            return None, None, None, None, 0

        ecg_12lead_windows = ecg_12lead_windows[:min_windows]
        vcg_windows = vcg_windows[:min_windows]

        return ecg_12lead_windows, vcg_windows, patient_means, patient_stds, min_windows

    except Exception as e:
        print(f"Error processing patient {os.path.basename(patient_path)}: {e}")
        return None, None, None, None, 0


def process_single_patient_no_norm(patient_path, target_fs=500, window_size=500, step=500):
    try:
        record = wfdb.rdrecord(patient_path)
        ecg_data = record.p_signal
        original_fs = record.fs

        ecg_12lead_indices = list(range(12))
        vcg_lead_indices = [12, 13, 14]

        if original_fs != target_fs:
            new_len = int(len(ecg_data) * (target_fs / original_fs))
            ecg_data = resample(ecg_data, new_len, axis=0)

        ecg_filtered = np.zeros_like(ecg_data)
        for ch in range(ecg_data.shape[1]):
            ecg_filtered[:, ch] = apply_ecg_filters(ecg_data[:, ch], target_fs)

        ecg_12lead = ecg_filtered[:, ecg_12lead_indices]
        vcg_leads = ecg_filtered[:, vcg_lead_indices]

        ecg_12lead_windows = efficient_sliding_window(ecg_12lead, window_size, step)
        vcg_windows = efficient_sliding_window(vcg_leads, window_size, step)

        min_windows = min(ecg_12lead_windows.shape[0], vcg_windows.shape[0])
        if min_windows == 0:
            print(f"    Warning: Patient {os.path.basename(patient_path)} has insufficient windows")
            return None, None, 0

        ecg_12lead_windows = ecg_12lead_windows[:min_windows]
        vcg_windows = vcg_windows[:min_windows]

        return ecg_12lead_windows, vcg_windows, min_windows

    except Exception as e:
        print(f"Error processing patient {os.path.basename(patient_path)}: {e}")
        return None, None, 0


def save_patient_data_no_norm(patient_id, ecg_12lead, vcg, all_lead_names, save_dir):
    patient_dir = os.path.join(save_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    np.save(os.path.join(patient_dir, "ecg_12lead.npy"), ecg_12lead)
    np.save(os.path.join(patient_dir, "vcg.npy"), vcg)

    patient_info = {
        "patient_id": patient_id,
        "ecg_12lead_shape": ecg_12lead.shape,
        "vcg_shape": vcg.shape,
        "window_count": ecg_12lead.shape[0],
        "sample_rate": 500,
        "window_size": 500,
        "step": 500,
        "normalization": "None",
        "stats_file": "None"
    }

    with open(os.path.join(patient_dir, "patient_info.json"), "w") as f:
        json.dump(patient_info, f, indent=4)

    return patient_dir


def save_patient_data_windows(patient_id, ecg_12lead, vcg, global_means, global_stds,
                               all_lead_names, save_dir):
    patient_dir = os.path.join(save_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    np.save(os.path.join(patient_dir, "ecg_12lead.npy"), ecg_12lead)
    np.save(os.path.join(patient_dir, "vcg.npy"), vcg)

    stats_file = os.path.join(patient_dir, "lead_stats.csv")
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lead', 'mean', 'std'])
        for name, m, s in zip(all_lead_names, global_means, global_stds):
            writer.writerow([name, f"{m:.8f}", f"{s:.8f}"])

    patient_info = {
        "patient_id": patient_id,
        "ecg_12lead_shape": ecg_12lead.shape,
        "vcg_shape": vcg.shape,
        "window_count": ecg_12lead.shape[0],
        "sample_rate": 500,
        "window_size": 500,
        "step": 500,
        "normalization": "Global",
        "stats_file": "lead_stats.csv"
    }

    with open(os.path.join(patient_dir, "patient_info.json"), "w") as f:
        json.dump(patient_info, f, indent=4)

    return patient_dir


def save_patient_data_per_patient(patient_id, ecg_12lead, vcg, patient_means, patient_stds,
                                   all_lead_names, save_dir):
    patient_dir = os.path.join(save_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    np.save(os.path.join(patient_dir, "ecg_12lead.npy"), ecg_12lead)
    np.save(os.path.join(patient_dir, "vcg.npy"), vcg)

    stats_file = os.path.join(patient_dir, "lead_stats.csv")
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lead', 'mean', 'std'])
        for name, m, s in zip(all_lead_names, patient_means, patient_stds):
            writer.writerow([name, f"{m:.8f}", f"{s:.8f}"])

    patient_info = {
        "patient_id": patient_id,
        "ecg_12lead_shape": ecg_12lead.shape,
        "vcg_shape": vcg.shape,
        "window_count": ecg_12lead.shape[0],
        "sample_rate": 500,
        "window_size": 500,
        "step": 500,
        "normalization": "Per-patient",
        "stats_file": "lead_stats.csv"
    }

    with open(os.path.join(patient_dir, "patient_info.json"), "w") as f:
        json.dump(patient_info, f, indent=4)

    return patient_dir


def save_processing_info(processing_stats, results_dir):
    processing_info = {
        "total_patients": processing_stats.get("total_patients", 0),
        "processed_patients": processing_stats.get("processed_patients", 0),
        "failed_patients": processing_stats.get("failed_patients", 0),
        "total_windows": processing_stats.get("total_windows", 0),
        "avg_windows_per_patient": processing_stats.get("avg_windows_per_patient", 0),
        "sample_rate": 500,
        "window_size": 500,
        "step": 500,
        "normalization": NORMALIZATION_MODE,
        "random_seed": RANDOM_SEED,
        "results_dir": results_dir
    }

    with open(os.path.join(results_dir, "processing_info.json"), "w") as f:
        json.dump(processing_info, f, indent=4)


def save_summary_report(processing_stats, results_dir):
    report = f"""
PTB-ECG Dataset Processing Summary
==================================

Basic Information:
------------------
Total patients: {processing_stats['total_patients']}
Successfully processed: {processing_stats['processed_patients']}
Failed: {processing_stats['failed_patients']}
Total windows: {processing_stats['total_windows']}
Average windows per patient: {processing_stats['avg_windows_per_patient']:.1f}

Processing Parameters:
----------------------
Sample rate: 500Hz
Window size: 500 samples (1 second)
Sliding step: 500 samples (no overlap)
Normalization: {NORMALIZATION_MODE}
Random seed: {RANDOM_SEED}

Output Directory:
-----------------
{results_dir}
"""

    with open(os.path.join(results_dir, "processing_summary.txt"), "w", encoding='utf-8') as f:
        f.write(report)


def process_ptb_dataset(data_dir):
    ecg_12lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    vcg_names = ['X', 'Y', 'Z']
    all_lead_names = ecg_12lead_names + vcg_names

    patient_ids = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(patient_ids)} patients")

    processing_stats = {
        "total_patients": len(patient_ids),
        "processed_patients": 0,
        "failed_patients": 0,
        "total_windows": 0,
        "avg_windows_per_patient": 0,
        "processed_patient_ids": [],
        "patient_windows": {}
    }

    if NORMALIZATION_MODE == "global":
        patient_filtered_data = []
        total_sums = np.zeros(15)
        total_sums_sq = np.zeros(15)
        total_samples = 0

        for i, patient_id in enumerate(patient_ids):
            patient_path = os.path.join(data_dir, patient_id)
            record_files = [f for f in os.listdir(patient_path) if f.endswith('.dat')]

            if len(record_files) == 0:
                print(f"  Patient {patient_id}: No .dat files found, skipping")
                processing_stats["failed_patients"] += 1
                continue

            record_file = record_files[0]
            record_path = os.path.join(patient_path, os.path.splitext(record_file)[0])

            ecg_12lead_full, vcg_full, channel_sums, channel_sums_sq, n_samples = \
                process_single_patient_filter(record_path)

            if ecg_12lead_full is None:
                processing_stats["failed_patients"] += 1
                continue

            patient_filtered_data.append((patient_id, ecg_12lead_full, vcg_full))
            total_sums += channel_sums
            total_sums_sq += channel_sums_sq
            total_samples += n_samples
            processing_stats["processed_patients"] += 1
            processing_stats["processed_patient_ids"].append(patient_id)

            print(f"  Patient {i + 1}/{len(patient_ids)}: {patient_id} ({n_samples} samples)")

        if total_samples > 0:
            global_means = total_sums / total_samples
            global_variances = (total_sums_sq / total_samples) - (global_means ** 2)
            global_stds = np.sqrt(np.maximum(global_variances, 1e-10))
        else:
            global_means = np.zeros(15)
            global_stds = np.ones(15)

        global_stats_file = os.path.join(results_dir, "global_stats.csv")
        with open(global_stats_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['lead', 'mean', 'std'])
            for name, m, s in zip(all_lead_names, global_means, global_stds):
                writer.writerow([name, f"{m:.8f}", f"{s:.8f}"])

        patient_window_data = normalize_split_and_window(patient_filtered_data, global_means, global_stds)

        for patient_id, ecg_12lead_windows, vcg_windows in patient_window_data:
            save_patient_data_windows(patient_id, ecg_12lead_windows, vcg_windows,
                                      global_means, global_stds, all_lead_names, patients_dir)
            processing_stats["total_windows"] += ecg_12lead_windows.shape[0]
            processing_stats["patient_windows"][patient_id] = ecg_12lead_windows.shape[0]

    elif NORMALIZATION_MODE == "per_patient":
        for i, patient_id in enumerate(patient_ids):
            patient_path = os.path.join(data_dir, patient_id)
            record_files = [f for f in os.listdir(patient_path) if f.endswith('.dat')]

            if len(record_files) == 0:
                print(f"  Patient {patient_id}: No .dat files found, skipping")
                processing_stats["failed_patients"] += 1
                continue

            record_file = record_files[0]
            record_path = os.path.join(patient_path, os.path.splitext(record_file)[0])

            ecg_12lead_windows, vcg_windows, patient_means, patient_stds, min_windows = \
                process_single_patient_full(record_path)

            if ecg_12lead_windows is None:
                processing_stats["failed_patients"] += 1
                continue

            save_patient_data_per_patient(patient_id, ecg_12lead_windows, vcg_windows,
                                          patient_means, patient_stds, all_lead_names, patients_dir)

            processing_stats["processed_patients"] += 1
            processing_stats["processed_patient_ids"].append(patient_id)
            processing_stats["total_windows"] += min_windows
            processing_stats["patient_windows"][patient_id] = min_windows

            print(f"  Patient {i + 1}/{len(patient_ids)}: {patient_id} ({min_windows} windows)")

    else:
        for i, patient_id in enumerate(patient_ids):
            patient_path = os.path.join(data_dir, patient_id)
            record_files = [f for f in os.listdir(patient_path) if f.endswith('.dat')]

            if len(record_files) == 0:
                print(f"  Patient {patient_id}: No .dat files found, skipping")
                processing_stats["failed_patients"] += 1
                continue

            record_file = record_files[0]
            record_path = os.path.join(patient_path, os.path.splitext(record_file)[0])

            ecg_12lead_windows, vcg_windows, min_windows = process_single_patient_no_norm(record_path)

            if ecg_12lead_windows is None:
                processing_stats["failed_patients"] += 1
                continue

            save_patient_data_no_norm(patient_id, ecg_12lead_windows, vcg_windows, all_lead_names, patients_dir)

            processing_stats["processed_patients"] += 1
            processing_stats["processed_patient_ids"].append(patient_id)
            processing_stats["total_windows"] += min_windows
            processing_stats["patient_windows"][patient_id] = min_windows

            print(f"  Patient {i + 1}/{len(patient_ids)}: {patient_id} ({min_windows} windows)")

    if processing_stats["processed_patients"] > 0:
        processing_stats["avg_windows_per_patient"] = (
                processing_stats["total_windows"] / processing_stats["processed_patients"]
        )
    else:
        processing_stats["avg_windows_per_patient"] = 0

    print(f"\nProcessing completed!")
    print(f"  Successfully processed: {processing_stats['processed_patients']} patients")
    print(f"  Failed: {processing_stats['failed_patients']} patients")
    print(f"  Total windows: {processing_stats['total_windows']}")
    print(f"  Average windows per patient: {processing_stats['avg_windows_per_patient']:.1f}")

    return processing_stats


def main():
    print("=" * 80)
    print("PTB-ECG Dataset Preprocessing")
    print("=" * 80)

    data_dir = "./health"

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory does not exist: {data_dir}")
        print("Please place the health dataset in this directory or modify data_dir variable")
        return

    print(f"Dataset: {data_dir}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Sample rate: 500Hz")
    print(f"Window size: 500 samples (1s)")
    print(f"Step: 500 samples (no overlap)")
    print(f"Normalization: {NORMALIZATION_MODE}")
    print(f"Output: {results_dir}")
    print("=" * 80)

    processing_stats = process_ptb_dataset(data_dir)

    if processing_stats is None:
        print("Data processing failed")
        return

    save_processing_info(processing_stats, results_dir)
    save_summary_report(processing_stats, results_dir)

    print("\n" + "=" * 80)
    print("Data processing completed!")
    print("=" * 80)
    print(f"Results saved to: {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    main()