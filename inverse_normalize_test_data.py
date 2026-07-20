"""
Filename: inverse_normalize_test_data.py
Function: Perform inverse Z-normalization on PTB-ECG test set data, restore original scale
     1. Read normalized test set data from split_data/test directory
     2. Read global mean and standard deviation for each patient from data_results
     3. Perform inverse normalization for 12-lead ECG and 3-lead VCG
     4. Save inverse-normalized .npy data per patient ID
     5. Generate continuous signal CSV tables for first 13 patients (each lead's samples)
     6. Generate statistics table for inverse-normalized data (CSV format) for visualization

Input:
    - data_results/patients: Contains lead_stats.csv (mean/std) for each patient
    - split_data/test: Normalized test set data

Output:
    - inverse_normalized_data/test: Inverse-normalized data saved per patient ID
        - Each patient directory contains:
            - ecg_12lead_inverse.npy    (Inverse-normalized ECG window data)
            - vcg_inverse.npy            (Inverse-normalized VCG window data)
            - (optional) ecg_12lead_inverse.csv (ECG continuous signal table for first 13 patients)
    - inverse_normalized_data/patient_level_inverse_stats.csv  (Patient-level statistics)
    - inverse_normalized_data/lead_level_inverse_stats.csv     (Lead-level detailed statistics)
    - inverse_normalized_data/inverse_normalize_summary.json   (Summary information)
"""


import os
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

from metrics_calculator import calculate_detailed_metrics


DATA_RESULTS_DIR = "data_results"
TEST_DATA_DIR = "split_data/test"
OUTPUT_DIR = "inverse_normalized_data"
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "test")
NUM_PATIENTS_FOR_TABLE = 55

MLP_TEST_RESULTS_DIR = "12-3-Ridge-2-mlp_results/results_mlp"


ECG_12LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
VCG_LEAD_NAMES = ['X', 'Y', 'Z']
ALL_LEAD_NAMES = ECG_12LEAD_NAMES + VCG_LEAD_NAMES
INDEPENDENT_NAMES = ['I', 'II',  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
INDEPENDENT_INDICES = [0,1,2,3,4,5,6,7]

TRAIN_DATA_DIR = "split_data/train"


def load_patient_stats(patient_id):
    """
    Load normalization statistics (mean and std) for a single patient
    Args:
        patient_id: Patient ID
    Returns:
        means: Per-lead mean array (15,)
        stds: Per-lead std array (15,)
    """
    stats_path = Path(DATA_RESULTS_DIR) / patient_id / "lead_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file for patient {patient_id} not found: {stats_path}")

    df = pd.read_csv(stats_path, encoding='utf-8')
    if len(df) != 15:
        raise ValueError(f"Abnormal lead statistics count for patient {patient_id}, expected 15, got {len(df)}")

    means = df['mean'].astype(np.float32).values
    stds = df['std'].astype(np.float32).values
    return means, stds


def compute_train_set_avg_stats(data_results_dir=DATA_RESULTS_DIR, train_data_dir=TRAIN_DATA_DIR):
    """
    Compute average normalization parameters across all training set patients
    (for data-leakage-free inverse normalization)
    
    Args:
        data_results_dir: Directory containing statistics for all patients
        train_data_dir: Training set patient directory (for obtaining training patient ID list)
    
    Returns:
        avg_means: Average mean for 15 leads (15,)
        avg_stds: Average std for 15 leads (15,)
        n_train_patients: Number of training patients included in computation
    """
    print(f"\nComputing average normalization parameters for training set...")
    
    train_patient_ids = [d.name for d in Path(train_data_dir).iterdir() if d.is_dir()]
    if not train_patient_ids:
        raise ValueError("No patient data found in training set directory")
    
    all_means = []
    all_stds = []
    failed_patients = []
    
    for patient_id in train_patient_ids:
        try:
            means, stds = load_patient_stats(patient_id)
            all_means.append(means)
            all_stds.append(stds)
        except Exception as e:
            print(f"  Skipping patient {patient_id}: {e}")
            failed_patients.append(patient_id)
    
    if not all_means:
        raise ValueError("Cannot load statistics for any training set patient")
    
    all_means = np.array(all_means)
    all_stds = np.array(all_stds)
    
    avg_means = np.mean(all_means, axis=0)
    avg_stds = np.mean(all_stds, axis=0)
    
    print(f"  Training set patients: {len(train_patient_ids)}")
    print(f"  Successfully loaded: {len(all_means)}, Failed: {len(failed_patients)}")
    print(f"  Average parameters computation completed")
    
    return avg_means, avg_stds, len(all_means)


def inverse_z_normalize(data, mean, std):
    """
    Perform inverse Z-normalization on data
    Formula: Original data = Normalized data * Standard deviation + Mean
    Args:
        data: Normalized data (num_windows, num_leads, num_samples)
        mean: Mean for corresponding leads (num_leads,)
        std: Standard deviation for corresponding leads (num_leads,)
    Returns:
        Inverse-normalized data at original scale (num_windows, num_leads, num_samples)
    """
    inverse_data = data.copy().astype(np.float32)
    for ch in range(inverse_data.shape[1]):
        if std[ch] < 1e-8:
            inverse_data[:, ch, :] = inverse_data[:, ch, :] + mean[ch]
        else:
            inverse_data[:, ch, :] = (inverse_data[:, ch, :] * std[ch]) + mean[ch]
    return inverse_data


def concatenate_windows_to_continuous(windows, channel_axis=1, time_axis=2):
    """
    Concatenate windowed data into continuous signal (no window overlap).
    Args:
        windows: Array of shape (N, C, L)
        channel_axis: Channel axis
        time_axis: Time axis (samples within window)
    Returns:
        Continuous signal, shape (total_samples, C)
    """
    continuous = windows.transpose(channel_axis, 0, time_axis).reshape(windows.shape[channel_axis], -1).T
    return continuous


def save_continuous_csv(data, output_dir, lead_names, patient_id):
    """
    Concatenate inverse-normalized windowed data into continuous signal and save as CSV.
    Each column is a lead, each row is a sample.
    Args:
        data: Inverse-normalized windowed data (N, C, L)
        output_dir: Patient output directory
        lead_names: Lead name list, length C
        patient_id: Patient ID (for printing only)
    """
    continuous = concatenate_windows_to_continuous(data)
    df = pd.DataFrame(continuous, columns=lead_names)
    csv_path = Path(output_dir) / "ecg_12lead_inverse.csv"
    df.to_csv(csv_path, index=False, float_format='%.8f')
    print(f"    Continuous signal table generated: {csv_path} (total {continuous.shape[0]} samples)")


def compute_derived_leads_from_I_II(I_mv, II_mv):
    """
    Derive remaining 4 limb leads from I and II using Einthoven's equations in mV space.
    Args:
        I_mv:  Lead I data (num_windows, num_samples)
        II_mv: Lead II data (num_windows, num_samples)
    Returns:
        III_mv, aVR_mv, aVL_mv, aVF_mv each with shape (num_windows, num_samples)
    """
    III_mv = II_mv - I_mv
    aVR_mv = -(I_mv + II_mv) / 2.0
    aVL_mv = (2.0 * I_mv - II_mv) / 2.0
    aVF_mv = (II_mv + III_mv) / 2.0
    return III_mv, aVR_mv, aVL_mv, aVF_mv


def process_8lead_with_einthoven(pred_8lead_norm, means, stds, ecg_label_inverse):
    """
    Inverse-normalize 8-lead normalized predictions, derive 4 limb leads, assemble 12 leads, compute metrics.
    Args:
        pred_8lead_norm: Model-predicted 8-lead normalized data (n_windows, 8, 500)
                         Channel order: I=0, II=1, V1=2, V2=3, V3=4, V4=5, V5=6, V6=7
        means: Mean for 15 leads (from lead_stats.csv)
        stds:  Standard deviation for 15 leads
        ecg_label_inverse: Inverse-normalized true 12-lead ECG (n_windows, 12, 500)
                           Channel order: I,II,III,aVR,aVL,aVF,V1-V6
    
    Returns:
        pred_12lead_mv: Predicted 12-lead ECG (mV space) (n_windows, 12, 500)
        metrics: calculate_detailed_metrics result dictionary
    """
    n_windows = pred_8lead_norm.shape[0]
    
    independent_lead_indices = [0, 1, 6, 7, 8, 9, 10, 11]
    
    pred_8lead_mv = np.zeros_like(pred_8lead_norm)
    for i, lead_idx in enumerate(independent_lead_indices):
        mu = means[:12][lead_idx]
        sigma = stds[:12][lead_idx]
        if sigma < 1e-8:
            pred_8lead_mv[:, i, :] = pred_8lead_norm[:, i, :] + mu
        else:
            pred_8lead_mv[:, i, :] = pred_8lead_norm[:, i, :] * sigma + mu
    
    I_mv  = pred_8lead_mv[:, 0, :]
    II_mv = pred_8lead_mv[:, 1, :]
    
    III_mv, aVR_mv, aVL_mv, aVF_mv = compute_derived_leads_from_I_II(I_mv, II_mv)
    
    pred_12lead_mv = np.zeros((n_windows, 12, 500), dtype=np.float32)
    pred_12lead_mv[:, 0, :] = I_mv
    pred_12lead_mv[:, 1, :] = II_mv
    pred_12lead_mv[:, 2, :] = III_mv
    pred_12lead_mv[:, 3, :] = aVR_mv
    pred_12lead_mv[:, 4, :] = aVL_mv
    pred_12lead_mv[:, 5, :] = aVF_mv
    pred_12lead_mv[:, 6, :] = pred_8lead_mv[:, 2, :]
    pred_12lead_mv[:, 7, :] = pred_8lead_mv[:, 3, :]
    pred_12lead_mv[:, 8, :] = pred_8lead_mv[:, 4, :]
    pred_12lead_mv[:, 9, :] = pred_8lead_mv[:, 5, :]
    pred_12lead_mv[:, 10, :] = pred_8lead_mv[:, 6, :]
    pred_12lead_mv[:, 11, :] = pred_8lead_mv[:, 7, :]
    
    metrics = calculate_detailed_metrics(
        ecg_label_inverse, pred_12lead_mv,
        channel_names=ECG_12LEAD_NAMES
    )
    
    return pred_12lead_mv, metrics


def run_einthoven_pipeline(mlp_test_dir, test_output_dir, data_results_dir, 
                          avg_means, avg_stds):
    """
    Read pred_ecg_8.npy saved by 12-3-Ridge-2-mlp.py,
    inverse-normalize -> derive 4 limb leads via Einthoven -> assemble 12 leads -> save pred_12lead_einthoven.npy
    and compute metrics (compare with inverse-normalized true labels ecg_12true_inversed.npy).
    Args:
        mlp_test_dir:      MLP test results directory (contains patient_X/pred_ecg_8.npy)
        test_output_dir:   Global inverse-normalized label directory (contains patient_X/ecg_12lead_inverse_global.npy)
    """
    if not Path(mlp_test_dir).exists():
        print(f"Error: MLP test results directory does not exist: {mlp_test_dir}")
        return None

    patient_dirs = sorted([
        d for d in os.listdir(mlp_test_dir)
        if d.startswith('patient_') and os.path.isdir(os.path.join(mlp_test_dir, d))
    ])
    patient_ids = [d.replace('patient_', '') for d in patient_dirs]

    if not patient_ids:
        print(f"No patient prediction data found in {mlp_test_dir}")
        return None
    print(f"\nFound {len(patient_ids)} patients with predicted 8-lead data")

    all_patient_metrics = []
    all_patient_per_channel = []
    failed_patients = []

    for idx, patient_id in enumerate(patient_ids):
        print(f"\n[{idx+1}/{len(patient_ids)}] Patient: {patient_id}")

        try:
            pred_8_path = os.path.join(mlp_test_dir, f"patient_{patient_id}", "pred_ecg_8.npy")
            if not os.path.exists(pred_8_path):
                print(f"  Warning: pred_ecg_8.npy not found, skipping")
                failed_patients.append(patient_id)
                continue
            pred_8lead_norm = np.load(pred_8_path)

            means = avg_means
            stds = avg_stds

            label_inv_path = os.path.join(test_output_dir, patient_id, "ecg_12true_inversed_global.npy")
            if not os.path.exists(label_inv_path):
                print(f"  Warning: ecg_12true_inversed_global.npy not found, please run inverse normalization main flow first")
                failed_patients.append(patient_id)
                continue
            ecg_label_inverse = np.load(label_inv_path)

            n_w = pred_8lead_norm.shape[0]
            if ecg_label_inverse.shape[0] != n_w:
                n_w = min(pred_8lead_norm.shape[0], ecg_label_inverse.shape[0])
                pred_8lead_norm = pred_8lead_norm[:n_w]
                ecg_label_inverse = ecg_label_inverse[:n_w]

            pred_12lead_mv_global, metrics_global = process_8lead_with_einthoven(
                pred_8lead_norm, means, stds, ecg_label_inverse
            )

            patient_out_dir = os.path.join(mlp_test_dir, f"patient_{patient_id}")
            np.save(os.path.join(patient_out_dir, "ecg_12pred_inversed_einthoven_global.npy"), pred_12lead_mv_global)

            rows = []
            for ch_idx, ch_name in enumerate(ECG_12LEAD_NAMES):
                row = {
                    'Channel': ch_name,
                    'MSE': metrics_global['per_channel']['MSE'][ch_idx],
                    'MAE': metrics_global['per_channel']['MAE'][ch_idx],
                    'RMSE': metrics_global['per_channel']['RMSE'][ch_idx],
                    'R2': metrics_global['per_channel']['R2'][ch_idx],
                    'PCC': metrics_global['per_channel']['PCC'][ch_idx]
                }
                rows.append(row)
            
            avg_row = {'Channel': 'Average'}
            for key in ['MSE', 'MAE', 'RMSE', 'R2', 'PCC']:
                avg_row[f'{key}'] = metrics_global['average'][key]
            rows.append(avg_row)
            
            overall_row = {'Channel': 'Overall'}
            for key in ['MSE', 'MAE', 'RMSE', 'R2', 'PCC']:
                overall_row[f'{key}'] = metrics_global['overall'][key]
            rows.append(overall_row)

            pd.DataFrame(rows).to_csv(
                os.path.join(patient_out_dir, "per_einthoven_metrics_global.csv"),
                index=False, encoding='utf-8-sig'
            )

            per_ch = {}
            for ch_idx, ch_name in enumerate(ECG_12LEAD_NAMES):
                per_ch[f'{ch_name}__MSE'] = metrics_global['per_channel']['MSE'][ch_idx]
                per_ch[f'{ch_name}__MAE'] = metrics_global['per_channel']['MAE'][ch_idx]
                per_ch[f'{ch_name}__RMSE'] = metrics_global['per_channel']['RMSE'][ch_idx]
                per_ch[f'{ch_name}__R2'] = metrics_global['per_channel']['R2'][ch_idx]
                per_ch[f'{ch_name}__PCC'] = metrics_global['per_channel']['PCC'][ch_idx]
            all_patient_per_channel.append(per_ch)
            
            patient_metric = {
                'patient_id': patient_id,
                'MSE_avg': metrics_global['average']['MSE'],
                'MAE_avg': metrics_global['average']['MAE'],
                'RMSE_avg': metrics_global['average']['RMSE'],
                'R2_avg': metrics_global['average']['R2'],
                'PCC_avg': metrics_global['average']['PCC']
            }
            all_patient_metrics.append(patient_metric)

            print(f"  Saved pred_12lead_einthoven_global.npy, PCC: {metrics_global['average']['PCC']:.4f}")

        except Exception as e:
            print(f"  Processing failed: {e}")
            import traceback
            traceback.print_exc()
            failed_patients.append(patient_id)

    if all_patient_metrics:
        summary_df = pd.DataFrame(all_patient_metrics)
        summary_csv = os.path.join(mlp_test_dir, "all_patients_einthoven_summary_global.csv")
        summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
        print(f"\nTotal summary saved: {summary_csv}")
        
        if all_patient_per_channel:
            df_per_ch = pd.DataFrame(all_patient_per_channel)
            channel_agg_rows = []
            for ch_name in ECG_12LEAD_NAMES:
                row = {
                    'Channel': ch_name,
                    'MSE_avg': df_per_ch[f'{ch_name}_PS_MSE'].mean(),
                    'MSE_var': df_per_ch[f'{ch_name}_PS_MSE'].var(),
                    'MAE_avg': df_per_ch[f'{ch_name}_PS_MAE'].mean(),
                    'RMSE_avg': df_per_ch[f'{ch_name}_PS_RMSE'].mean(),
                    'R2_avg': df_per_ch[f'{ch_name}_PS_R2'].mean(),
                    'PCC_avg': df_per_ch[f'{ch_name}_PS_PCC'].mean(),
                    'PCC_var': df_per_ch[f'{ch_name}_PS_PCC'].var()
                }

                channel_agg_rows.append(row)
            agg_df = pd.DataFrame(channel_agg_rows)
            agg_csv = os.path.join(mlp_test_dir, "channel_metrics_aggregated_global.csv")
            agg_df.to_csv(agg_csv, index=False, encoding='utf-8-sig')
            print(f"Channel-level cross-patient summary saved: {agg_csv}")

    print(f"\nSuccess: {len(all_patient_metrics)}, Failed: {len(failed_patients)}")
    if failed_patients:
        print(f"Failed patients: {failed_patients}")

    return all_patient_metrics


def run_einthoven_pipeline_per_patient(mlp_test_dir, test_output_dir, test_data_dir):
    """
    Per-patient inverse normalization + Einthoven derivation + metric calculation
    (using each patient's own normalization parameters)
    Args:
        mlp_test_dir:      MLP test results directory (contains patient_X/pred_ecg_8.npy)
        test_output_dir:   Inverse-normalized label directory (contains patient_X/ecg_12true_inversed_patient.npy)
        test_data_dir:     Test set data directory (contains patient_X/lead_stats.csv)
    """
    if not Path(mlp_test_dir).exists():
        print(f"Error: MLP test results directory does not exist: {mlp_test_dir}")
        return None

    patient_dirs = sorted([
        d for d in os.listdir(mlp_test_dir)
        if d.startswith('patient_') and os.path.isdir(os.path.join(mlp_test_dir, d))
    ])
    patient_ids = [d.replace('patient_', '') for d in patient_dirs]

    if not patient_ids:
        print(f"No patient prediction data found in {mlp_test_dir}")
        return None
    print(f"\nFound {len(patient_ids)} patients with predicted 8-lead data")

    all_patient_metrics = []
    all_patient_per_channel = []
    failed_patients = []

    for idx, patient_id in enumerate(patient_ids):
        print(f"\n[{idx+1}/{len(patient_ids)}] Patient: {patient_id}")

        try:
            stats_path = os.path.join(DATA_RESULTS_DIR, "patients", patient_id, "lead_stats.csv")
            if not os.path.exists(stats_path):
                print(f"  Warning: lead_stats.csv not found, skipping")
                failed_patients.append(patient_id)
                continue
            df_stats = pd.read_csv(stats_path)
            means = df_stats['mean'].values
            stds = df_stats['std'].values

            pred_8_path = os.path.join(mlp_test_dir, f"patient_{patient_id}", "pred_ecg_8.npy")
            if not os.path.exists(pred_8_path):
                print(f"  Warning: pred_ecg_8.npy not found, skipping")
                failed_patients.append(patient_id)
                continue
            pred_8lead_norm = np.load(pred_8_path)

            label_inv_path = os.path.join(test_output_dir, patient_id, "ecg_12true_inversed_patient.npy")
            if not os.path.exists(label_inv_path):
                print(f"  Warning: ecg_12true_inversed_patient.npy not found, please run inverse normalization main flow first")
                failed_patients.append(patient_id)
                continue
            ecg_label_inverse = np.load(label_inv_path)

            n_w = pred_8lead_norm.shape[0]
            if ecg_label_inverse.shape[0] != n_w:
                n_w = min(pred_8lead_norm.shape[0], ecg_label_inverse.shape[0])
                pred_8lead_norm = pred_8lead_norm[:n_w]
                ecg_label_inverse = ecg_label_inverse[:n_w]

            pred_12lead_mv, metrics = process_8lead_with_einthoven(
                pred_8lead_norm, means, stds, ecg_label_inverse
            )

            patient_out_dir = os.path.join(mlp_test_dir, f"patient_{patient_id}")
            np.save(os.path.join(patient_out_dir, "ecg_12pred_inversed_einthoven_patient.npy"), pred_12lead_mv)

            rows = []
            for ch_idx, ch_name in enumerate(ECG_12LEAD_NAMES):
                row = {
                    'Channel': ch_name,
                    'MSE': metrics['per_channel']['MSE'][ch_idx],
                    'MAE': metrics['per_channel']['MAE'][ch_idx],
                    'RMSE': metrics['per_channel']['RMSE'][ch_idx],
                    'R2': metrics['per_channel']['R2'][ch_idx],
                    'PCC': metrics['per_channel']['PCC'][ch_idx]
                }
                rows.append(row)
            
            avg_row = {'Channel': 'Average'}
            for key in ['MSE', 'MAE', 'RMSE', 'R2', 'PCC']:
                avg_row[f'{key}'] = metrics['average'][key]
            rows.append(avg_row)
            
            overall_row = {'Channel': 'Overall'}
            for key in ['MSE', 'MAE', 'RMSE', 'R2', 'PCC']:
                overall_row[f'{key}'] = metrics['overall'][key]
            rows.append(overall_row)

            pd.DataFrame(rows).to_csv(
                os.path.join(patient_out_dir, "per_einthoven_metrics_patient.csv"),
                index=False, encoding='utf-8-sig'
            )

            per_ch = {}
            for ch_idx, ch_name in enumerate(ECG_12LEAD_NAMES):
                per_ch[f'{ch_name}__MSE'] = metrics['per_channel']['MSE'][ch_idx]
                per_ch[f'{ch_name}__MAE'] = metrics['per_channel']['MAE'][ch_idx]
                per_ch[f'{ch_name}__RMSE'] = metrics['per_channel']['RMSE'][ch_idx]
                per_ch[f'{ch_name}__R2'] = metrics['per_channel']['R2'][ch_idx]
                per_ch[f'{ch_name}__PCC'] = metrics['per_channel']['PCC'][ch_idx]
            all_patient_per_channel.append(per_ch)
            
            patient_metric = {
                'patient_id': patient_id,
                'MSE_avg': metrics['average']['MSE'],
                'MAE_avg': metrics['average']['MAE'],
                'RMSE_avg': metrics['average']['RMSE'],
                'R2_avg': metrics['average']['R2'],
                'PCC_avg': metrics['average']['PCC']
            }
            all_patient_metrics.append(patient_metric)

            print(f"  Saved pred_12lead_einthoven_patient.npy, PCC: {metrics['average']['PCC']:.4f}")

        except Exception as e:
            print(f"  Processing failed: {e}")
            import traceback
            traceback.print_exc()
            failed_patients.append(patient_id)

    if all_patient_metrics:
        summary_df = pd.DataFrame(all_patient_metrics)
        summary_csv = os.path.join(mlp_test_dir, "all_patients_einthoven_summary_patient.csv")
        summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
        print(f"\nTotal summary saved: {summary_csv}")
        
        if all_patient_per_channel:
            df_per_ch = pd.DataFrame(all_patient_per_channel)
            channel_agg_rows = []
            for ch_name in ECG_12LEAD_NAMES:
                row = {
                    'Channel': ch_name,
                    'MSE_avg': df_per_ch[f'{ch_name}__MSE'].mean(),
                    'MSE_var': df_per_ch[f'{ch_name}__MSE'].var(),
                    'MAE_avg': df_per_ch[f'{ch_name}__MAE'].mean(),
                    'RMSE_avg': df_per_ch[f'{ch_name}__RMSE'].mean(),
                    'R2_avg': df_per_ch[f'{ch_name}__R2'].mean(),
                    'PCC_avg': df_per_ch[f'{ch_name}__PCC'].mean(),
                    'PCC_var': df_per_ch[f'{ch_name}__PCC'].var()
                }

                channel_agg_rows.append(row)
            agg_df = pd.DataFrame(channel_agg_rows)
            agg_csv = os.path.join(mlp_test_dir, "channel_metrics_aggregated_patient.csv")
            agg_df.to_csv(agg_csv, index=False, encoding='utf-8-sig')
            print(f"Channel-level cross-patient summary saved: {agg_csv}")

    print(f"\nSuccess: {len(all_patient_metrics)}, Failed: {len(failed_patients)}")
    if failed_patients:
        print(f"Failed patients: {failed_patients}")

    return all_patient_metrics


def process_single_test_patient(patient_id, patient_index, total_patients, 
                               avg_means, avg_stds, suffix=None):
    """
    Process inverse normalization for single patient's 12-lead ECG (inverse normalization only, no statistics)
    Args:
        patient_id: Patient ID
        patient_index: Current patient index in list (starting from 1)
        total_patients: Total patients (for progress display)
        avg_means: Normalization parameters (15,), can be global or patient-specific
        avg_stds: Normalization parameters (15,), can be global or patient-specific
        suffix: Output file suffix, "global" or "patient"
    """
    print(f"[{patient_index}/{total_patients}] Processing patient: {patient_id}")

    test_patient_dir = Path(TEST_DATA_DIR) / patient_id
    ecg_norm_path = test_patient_dir / "ecg_12lead.npy"
    if not ecg_norm_path.exists():
        raise FileNotFoundError(f"Test set ECG data file missing for patient {patient_id}")
    ecg_norm = np.load(ecg_norm_path)

    ecg_means = avg_means[:12]
    ecg_stds = avg_stds[:12]
    ecg_inversed = inverse_z_normalize(ecg_norm, ecg_means, ecg_stds)
    
    output_patient_dir = Path(TEST_OUTPUT_DIR) / patient_id
    output_patient_dir.mkdir(parents=True, exist_ok=True)
    
    ecg_inverse_trueecg_path = output_patient_dir / f"ecg_12true_inversed_{suffix}.npy"
    np.save(ecg_inverse_trueecg_path, ecg_inversed)

    print(f"    Saved 12-lead ECG inverse-normalized with {suffix} parameters")


def generate_summary_tables(all_patient_stats, all_lead_stats):
    """
    Generate statistics tables for inverse-normalized data (CSV format)
    Args:
        all_patient_stats: List of statistics for all patients
        all_lead_stats: List of detailed statistics for all leads
    """
    patient_df = pd.DataFrame(all_patient_stats)
    patient_table_path = Path(OUTPUT_DIR) / "patient_level_inverse_stats.csv"
    patient_df.to_csv(patient_table_path, index=False, encoding='utf-8', float_format='%.8f')
    print(f"Patient-level statistics table saved: {patient_table_path}")


    lead_df = pd.DataFrame(all_lead_stats)
    lead_table_path = Path(OUTPUT_DIR) / "lead_level_inverse_stats.csv"
    lead_df.to_csv(lead_table_path, index=False, encoding='utf-8', float_format='%.8f')
    print(f"Lead-level detailed statistics table saved: {lead_table_path}")


    summary_info = {
        "total_patients_processed": len(all_patient_stats),
        "total_windows": sum([p["window_count"] for p in all_patient_stats]),
        "ecg_global_stats": {
            "min": min([p["ecg_inverse_min"] for p in all_patient_stats]),
            "max": max([p["ecg_inverse_max"] for p in all_patient_stats]),
            "mean": np.mean([p["ecg_inverse_mean"] for p in all_patient_stats]).item(),
            "std": np.mean([p["ecg_inverse_std"] for p in all_patient_stats]).item()
        },
        "vcg_global_stats": {
            "min": min([p["vcg_inverse_min"] for p in all_patient_stats]),
            "max": max([p["vcg_inverse_max"] for p in all_patient_stats]),
            "mean": np.mean([p["vcg_inverse_mean"] for p in all_patient_stats]).item(),
            "std": np.mean([p["vcg_inverse_std"] for p in all_patient_stats]).item()
        },
        "output_directory": str(Path(OUTPUT_DIR).absolute()),
        "test_data_directory": str(Path(TEST_OUTPUT_DIR).absolute()),
        "patient_table_path": str(patient_table_path),
        "lead_table_path": str(lead_table_path),
        "num_patients_with_csv": min(NUM_PATIENTS_FOR_TABLE, len(all_patient_stats))
    }


    summary_path = Path(OUTPUT_DIR) / "inverse_normalize_summary.json"
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump(summary_info, f, indent=4, ensure_ascii=False)
    print(f"Summary information saved: {summary_path}")


def main():
    print("=" * 80)
    print("PTB-ECG Test Set Data Inverse Normalization")
    print("=" * 80)


    if not Path(DATA_RESULTS_DIR).exists():
        raise FileNotFoundError(f"Original statistics directory does not exist: {DATA_RESULTS_DIR}")
    if not Path(TEST_DATA_DIR).exists():
        raise FileNotFoundError(f"Test set data directory does not exist: {TEST_DATA_DIR}")


    Path(TEST_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Inverse normalization results will be saved to: {Path(OUTPUT_DIR).absolute()}")

    test_patient_ids = [d.name for d in Path(TEST_DATA_DIR).iterdir() if d.is_dir()]
    if not test_patient_ids:
        raise ValueError("No patient data found in test set directory")
    test_patient_ids.sort()
    total_patients = len(test_patient_ids)
    print(f"\nFound test set patients: {total_patients}")
    print(f"Will generate detailed CSV tables for first {NUM_PATIENTS_FOR_TABLE} patients only")
    print(f"First 13 patient IDs: {test_patient_ids[:13]}")

    NORMALIZATION_MODE = "per_patient"

    if NORMALIZATION_MODE == "global":
        global_normalization_stats = pd.read_csv(Path(DATA_RESULTS_DIR) / "global_normalization_stats.csv")
        print(f"Global inverse normalization parameters: {DATA_RESULTS_DIR}/global_normalization_stats.csv")
        ecg_12lead_means = global_normalization_stats[global_normalization_stats['lead'].isin(ALL_LEAD_NAMES)]['mean'].values
        ecg_12lead_stds = global_normalization_stats[global_normalization_stats['lead'].isin(ALL_LEAD_NAMES)]['std'].values

        success_count = 0
        failed_patients = []
        for idx, pid in enumerate(test_patient_ids, start=1):
            try:
                process_single_test_patient(pid, idx, total_patients, ecg_12lead_means, ecg_12lead_stds)
                success_count += 1
            except Exception as e:
                print(f"Failed processing patient {pid}: {str(e)}")
                failed_patients.append(pid)

        print("\n" + "=" * 80)
        print(f"Inverse-normalized labels: {TEST_OUTPUT_DIR}/patient_X/ecg_12true_inversed_global.npy")
        print("Inverse normalization completed!")
        print("=" * 80)

        print("\n\n" + "=" * 80)
        print("8-lead Inverse Normalization + Einthoven Derivation + Metric Calculation")
        print("=" * 80)
        print(f"Reading {MLP_TEST_RESULTS_DIR}/patient_X/pred_ecg_8.npy")
        run_einthoven_pipeline(MLP_TEST_RESULTS_DIR, TEST_OUTPUT_DIR, DATA_RESULTS_DIR, ecg_12lead_means, ecg_12lead_stds)

    elif NORMALIZATION_MODE == "per_patient":
        print(f"Per-patient inverse normalization mode, parameters from: {DATA_RESULTS_DIR}/patients/patient_id/lead_stats.csv")

        success_count = 0
        failed_patients = []
        for idx, pid in enumerate(test_patient_ids, start=1):
            try:
                patient_stats_path = Path(DATA_RESULTS_DIR) / "patients" / pid / "lead_stats.csv"
                if not patient_stats_path.exists():
                    raise FileNotFoundError(f"Normalization parameters file not found for patient {pid}: {patient_stats_path}")
                df_stats = pd.read_csv(patient_stats_path, encoding='utf-8')
                patient_means = df_stats['mean'].astype(np.float32).values
                patient_stds = df_stats['std'].astype(np.float32).values

                process_single_test_patient(pid, idx, total_patients, patient_means, patient_stds, suffix="patient")
                success_count += 1
            except Exception as e:
                print(f"Failed processing patient {pid}: {str(e)}")
                failed_patients.append(pid)

        print("\n" + "=" * 80)
        print(f"Per-patient inverse-normalized labels: {TEST_OUTPUT_DIR}/patient_X/ecg_12true_inversed_patient.npy")
        print(f"Success: {success_count}, Failed: {len(failed_patients)}")
        print("=" * 80)

        print("\n\n" + "=" * 80)
        print("8-lead Inverse Normalization + Einthoven Derivation + Metric Calculation (per-patient parameters)")
        print("=" * 80)
        run_einthoven_pipeline_per_patient(MLP_TEST_RESULTS_DIR, TEST_OUTPUT_DIR, TEST_DATA_DIR)

    else:
        print('Normalization mode not supported')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nProgram execution error: {str(e)}")
        exit(1)