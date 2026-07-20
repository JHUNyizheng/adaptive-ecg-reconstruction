"""
Filename: 12-3-Ridge-2.py
Function: Batch analysis of ridge regression model orthogonality metrics
     1. Read training set MSE/RMSE metrics for all 220 combinations
     2. Read corresponding training, validation, and test predicted VCG signals
     3. Calculate temporal orthogonality stability:
        - For each segment (3, 500), slide 50-time-point window to compute 3×3 covariance matrix
        - Compute absolute mean of off-diagonal elements for each window
        - Compute standard deviation across all windows as stability metric
"""

import os
import numpy as np
import pandas as pd
import warnings
import json
import time
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

MATRIX_TYPE = "covariance"

script_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = f"{script_name}_results"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(results_dir)}")


def load_all_combinations_data(base_results_dir):
    """Read evaluation metrics and prediction data for all combinations"""
    print("Starting to read data for all combinations...")

    model_index_path = os.path.join(base_results_dir, "model_index.csv")
    if not os.path.exists(model_index_path):
        print(f"Error: Model index file does not exist: {model_index_path}")
        return None

    model_index_df = pd.read_csv(model_index_path)
    print(f"Found evaluation metrics for {len(model_index_df)} combinations")

    predictions_dir = os.path.join(base_results_dir, "all_combo_predictions")
    if not os.path.exists(predictions_dir):
        print(f"Error: Prediction results directory does not exist: {predictions_dir}")
        return None

    combo_folders = [f for f in os.listdir(predictions_dir)
                     if os.path.isdir(os.path.join(predictions_dir, f))]
    print(f"Found prediction data folders for {len(combo_folders)} combinations")

    use_orthogonalization = False
    
    if use_orthogonalization:
        true_signals_dir = os.path.join(base_results_dir, "true_signals_orthogonalized")
        print(f"Reading orthogonalized true VCG signals from {true_signals_dir}...")
        
        Y_train = np.load(os.path.join(true_signals_dir, "Y_train_ortho.npy"))
        Y_val = np.load(os.path.join(true_signals_dir, "Y_val_ortho.npy"))
        Y_test = np.load(os.path.join(true_signals_dir, "Y_test_ortho.npy"))
    else:
        true_signals_dir = os.path.join(base_results_dir, "true_signals")
        print(f"Reading original true VCG signals from {true_signals_dir}...")
        
        Y_train = np.load(os.path.join(true_signals_dir, "Y_train.npy"))
        Y_val = np.load(os.path.join(true_signals_dir, "Y_val.npy"))
        Y_test = np.load(os.path.join(true_signals_dir, "Y_test.npy"))

    all_combinations_data = []

    for _, row in model_index_df.iterrows():
        combo_str = row['Combination']
        combo_folder = f"combo_{combo_str}"

        if combo_folder not in combo_folders:
            print(f"Warning: Folder for combination {combo_str} does not exist, skipping")
            continue

        combo_dir = os.path.join(predictions_dir, combo_folder)

        try:
            train_pred_path = os.path.join(combo_dir, "Y_train_LR.npy")
            Y_train_pred = np.load(train_pred_path)

            val_pred_path = os.path.join(combo_dir, "Y_val_LR.npy")
            Y_val_pred = np.load(val_pred_path)

            test_pred_path = os.path.join(combo_dir, "Y_test_LR.npy")
            Y_test_pred = np.load(test_pred_path) if os.path.exists(test_pred_path) else None

            metrics_path = os.path.join(combo_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = None

            combination_data = {
                'combo_name': combo_str,
                'combo_index': row['Combination Index'],
                'combo_names': row['Combination Names'],
                'train_mse': row['Train Avg MSE'],
                'train_rmse': row['Train Avg RMSE'],
                'train_mae': row['Train Avg MAE'],
                'train_pcc': row['Train Avg PCC'],
                'train_r2': row['Train Avg R2'],
                'val_mse': row['Val Avg MSE'],
                'val_rmse': row['Val Avg RMSE'],
                'val_mae': row['Val Avg MAE'],
                'val_pcc': row['Val Avg PCC'],
                'val_r2': row['Val Avg R2'],
                'test_mse': row['Test Avg MSE'] if 'Test Avg MSE' in row else None,
                'test_rmse': row['Test Avg RMSE'] if 'Test Avg RMSE' in row else None,
                'test_mae': row['Test Avg MAE'] if 'Test Avg MAE' in row else None,
                'test_pcc': row['Test Avg PCC'] if 'Test Avg PCC' in row else None,
                'test_r2': row['Test Avg R2'] if 'Test Avg R2' in row else None,
                'Y_train_pred': Y_train_pred,
                'Y_val_pred': Y_val_pred,
                'Y_test_pred': Y_test_pred,
                'Y_train': Y_train,
                'Y_val': Y_val,
                'Y_test': Y_test,
                'metrics': metrics
            }

            all_combinations_data.append(combination_data)

        except Exception as e:
            print(f"Warning: Error reading data for combination {combo_str}: {e}")
            continue

    print(f"Successfully read complete data for {len(all_combinations_data)} combinations")
    return all_combinations_data


WINDOW_SIZE = 500


def calculate_temporal_stability(y_pred):
    """Calculate temporal orthogonality stability:
    1. For each segment (3, 500), slide 50-time-point window to compute 3×3 covariance matrix
    2. Compute absolute mean of off-diagonal elements for each window
    3. Compute determinant of covariance matrix for each window
    4. Compute condition number of covariance matrix for each window
    5. Compute mean across all windows

    Args:
        y_pred: Predicted signal, shape (n_windows, 3, 500)

    Returns:
        cov_offdiag_mean: Mean of absolute off-diagonal means across all windows
        cov_det_mean: Mean of determinants across all windows
        cov_cond_mean: Mean of condition numbers across all windows
        cov_offdiag_metrics: List of off-diagonal absolute means for all windows
        cov_det_metrics: List of determinants for all windows
        cov_cond_metrics: List of condition numbers for all windows
    """
    n_windows, n_channels, n_samples = y_pred.shape
    window_size = WINDOW_SIZE
    step = window_size
    cov_offdiag_metrics = []
    cov_det_metrics = []
    cov_cond_metrics = []

    for window_idx in range(n_windows):
        signal_segment = y_pred[window_idx]

        for start in range(0, n_samples - window_size + 1, step):
            end = start + window_size
            window_data = signal_segment[:, start:end]

            if MATRIX_TYPE == "covariance":
                matrix = np.cov(window_data.T, rowvar=False)
            else:
                matrix = np.corrcoef(window_data.T, rowvar=False)


            off_diag_indices = np.where(~np.eye(matrix.shape[0], dtype=bool))
            off_diag_values = matrix[off_diag_indices]
            cov_offdiag_metric = np.mean(np.abs(off_diag_values))
            cov_offdiag_metrics.append(cov_offdiag_metric)

            cov_det = np.linalg.det(matrix)
            cov_det_metrics.append(cov_det)

            eigvals = np.linalg.eigvals(matrix)
            lambda_max = np.max(np.abs(eigvals))
            lambda_min = np.min(np.abs(eigvals))
            cov_cond = lambda_max / lambda_min if lambda_min > 0 else np.inf
            cov_cond_metrics.append(cov_cond)

    if len(cov_offdiag_metrics) > 0:
        cov_offdiag_mean = np.mean(cov_offdiag_metrics)
        cov_det_mean = np.mean(cov_det_metrics)
        cov_cond_mean = np.mean(cov_cond_metrics)
    else:
        cov_offdiag_mean = 0
        cov_det_mean = 0
        cov_cond_mean = 0

    return cov_offdiag_mean, cov_det_mean, cov_cond_mean, cov_offdiag_metrics, cov_det_metrics, cov_cond_metrics


def calculate_all_orthogonality_metrics(y_pred):
    """Calculate temporal orthogonality stability metrics"""
    cov_offdiag_mean, cov_det_mean, cov_cond_mean, _, _, _ = calculate_temporal_stability(y_pred)

    metrics = {
        'offdiag_mean': cov_offdiag_mean,
        'det_mean': cov_det_mean,
        'cond_mean': cov_cond_mean
    }

    return metrics


def calculate_orthogonality_for_all_combinations(all_combinations_data):
    """Calculate orthogonality metrics for all combinations"""
    print("\nStarting orthogonality metric calculation for all combinations...")

    all_results = []

    print("  Calculating orthogonality metrics for all combinations...")
    
    if all_combinations_data:
        first_combo = all_combinations_data[0]
        print("  Calculating orthogonality metrics for true signals once...")
        train_true_metrics = calculate_all_orthogonality_metrics(first_combo['Y_train'])
        val_true_metrics = calculate_all_orthogonality_metrics(first_combo['Y_val'])
        test_true_metrics = calculate_all_orthogonality_metrics(first_combo['Y_test'])

    for i, combo_data in enumerate(all_combinations_data):
        combo_name = combo_data['combo_name']

        if (i + 1) % 1 == 0 or i == 0 or i == len(all_combinations_data) - 1:
            print(f"  Progress: {i + 1}/{len(all_combinations_data)} - Combination: {combo_name}")

        try:
            train_metrics = calculate_all_orthogonality_metrics(combo_data['Y_train_pred'])
            val_metrics = calculate_all_orthogonality_metrics(combo_data['Y_val_pred'])
            test_metrics = calculate_all_orthogonality_metrics(combo_data['Y_test_pred'])

            result = {
                'combo_name': combo_name,
                'combo_index': combo_data['combo_index'],
                'combo_names': combo_data['combo_names'],
                'train_mse': combo_data['train_mse'],
                'train_rmse': combo_data['train_rmse'],
                'train_mae': combo_data['train_mae'],
                'train_pcc': combo_data['train_pcc'],
                'train_r2': combo_data['train_r2'],
                'val_mse': combo_data['val_mse'],
                'val_rmse': combo_data['val_rmse'],
                'val_mae': combo_data['val_mae'],
                'val_pcc': combo_data['val_pcc'],
                'val_r2': combo_data['val_r2'],
                'test_mse': combo_data['test_mse'],
                'test_rmse': combo_data['test_rmse'],
                'test_mae': combo_data['test_mae'],
                'test_pcc': combo_data['test_pcc'],
                'test_r2': combo_data['test_r2'],

                'train_offdiag_mean': train_metrics['offdiag_mean'],
                'train_det_mean': train_metrics['det_mean'],
                'train_cond_mean': train_metrics['cond_mean'],
                'val_offdiag_mean': val_metrics['offdiag_mean'],
                'val_det_mean': val_metrics['det_mean'],
                'val_cond_mean': val_metrics['cond_mean'],
                'test_offdiag_mean': test_metrics['offdiag_mean'],
                'test_det_mean': test_metrics['det_mean'],
                'test_cond_mean': test_metrics['cond_mean'],

                'train_true_offdiag_mean': train_true_metrics['offdiag_mean'],
                'train_true_det_mean': train_true_metrics['det_mean'],
                'train_true_cond_mean': train_true_metrics['cond_mean'],
                'val_true_offdiag_mean': val_true_metrics['offdiag_mean'],
                'val_true_det_mean': val_true_metrics['det_mean'],
                'val_true_cond_mean': val_true_metrics['cond_mean'],
                'test_true_offdiag_mean': test_true_metrics['offdiag_mean'],
                'test_true_det_mean': test_true_metrics['det_mean'],
                'test_true_cond_mean': test_true_metrics['cond_mean']
            }

            all_results.append(result)

        except Exception as e:
            print(f"Warning: Error calculating orthogonality metrics for combination {combo_name}: {e}")
            continue

    print(f"Successfully calculated orthogonality metrics for {len(all_results)} combinations")
    return all_results


def main():
    """Main function"""

    base_results_dir = "12-3-Ridge-1_results"
    if not os.path.exists(base_results_dir):
        print(f"Error: Base results directory does not exist: {os.path.abspath(base_results_dir)}")
        print("Please run 12-3-Ridge-1.py script first to generate data")
        return

    print(f"Base results directory: {os.path.abspath(base_results_dir)}")

    print("\n[Step 1] Reading data for all combinations...")
    all_combinations_data = load_all_combinations_data(base_results_dir)

    if all_combinations_data is None or len(all_combinations_data) == 0:
        print("Error: Failed to read valid combination data")
        return

    print(f"Successfully read data for {len(all_combinations_data)} combinations")

    print("\n[Step 2] Calculating orthogonality metrics for all combinations...")
    all_results = calculate_orthogonality_for_all_combinations(all_combinations_data)

    if len(all_results) == 0:
        print("Error: Failed to calculate orthogonality metrics for any combination")
        return

    print("\n[Step 3] Saving orthogonality metric results...")

    detailed_results = []
    for result in all_results:
        detailed_result = {
            'Combination': result['combo_name'],
            'Combo Index': result['combo_index'],
            'Combo Leads': result['combo_names'],

            'Train Covariance Off-diagonal Mean': result['train_offdiag_mean'],
            'Train Covariance Determinant': result['train_det_mean'],
            'Train Covariance Condition Number': result['train_cond_mean'],
            'Train True Covariance Off-diagonal Mean': result['train_true_offdiag_mean'],
            'Train True Covariance Determinant': result['train_true_det_mean'],
            'Train True Covariance Condition Number': result['train_true_cond_mean'],
            'Train MSE': result['train_mse'],
            'Train RMSE': result['train_rmse'],
            'Train MAE': result['train_mae'],
            'Train PCC': result['train_pcc'],
            'Train R2': result['train_r2'],

            'Val Covariance Off-diagonal Mean': result['val_offdiag_mean'],
            'Val Covariance Determinant': result['val_det_mean'],
            'Val Covariance Condition Number': result['val_cond_mean'],
            'Val True Covariance Off-diagonal Mean': result['val_true_offdiag_mean'],
            'Val True Covariance Determinant': result['val_true_det_mean'],
            'Val True Covariance Condition Number': result['val_true_cond_mean'],
            'Val MSE': result['val_mse'],
            'Val RMSE': result['val_rmse'],
            'Val MAE': result['val_mae'],
            'Val PCC': result['val_pcc'],
            'Val R2': result['val_r2'],

            'Test Covariance Off-diagonal Mean': result['test_offdiag_mean'],
            'Test Covariance Determinant': result['test_det_mean'],
            'Test Covariance Condition Number': result['test_cond_mean'],
            'Test True Covariance Off-diagonal Mean': result['test_true_offdiag_mean'],
            'Test True Covariance Determinant': result['test_true_det_mean'],
            'Test True Covariance Condition Number': result['test_true_cond_mean'],
            'Test MSE': result['test_mse'],
            'Test RMSE': result['test_rmse'],
            'Test MAE': result['test_mae'],
            'Test PCC': result['test_pcc'],
            'Test R2': result['test_r2']
        }


        detailed_results.append(detailed_result)

    detailed_df = pd.DataFrame(detailed_results)

    detailed_csv = os.path.join(results_dir, f"detailed_orthogonality_results_size{WINDOW_SIZE}.csv")
    if os.path.exists(detailed_csv):
        base, ext = os.path.splitext(detailed_csv)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        detailed_csv = f"{base}_{counter}{ext}"

    detailed_df.to_csv(detailed_csv, index=False, encoding='utf-8-sig')
    print(f"Detailed orthogonality metric table saved: {detailed_csv}")

    print("\n" + "=" * 80)
    print("Orthogonality metric calculation completed!")
    print("=" * 80)
    print(f"\nResults saved to: {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    main()