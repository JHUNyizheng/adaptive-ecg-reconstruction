"""
Filename: 12-3-Ridge-1.py
Function: Ridge regression model training and evaluation
     Select 3 leads from 12-lead ECG, fit VCG via ridge regression
     Dataset split: by patient, ratio 7:1.5:1.5
     Save models and evaluation metrics for all 220 combinations
     Create independent folder for each combination to save predictions
"""

import os
import numpy as np
import pandas as pd
import warnings
import json
import random
import time
from itertools import combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from orthogonalization import gram_schmidt_orthogonalize
import joblib

RANDOM_SEED = 20251226
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

warnings.filterwarnings('ignore')

script_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = f"{script_name}_results"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(results_dir)}")

predictions_dir = os.path.join(results_dir, "all_combo_predictions")
os.makedirs(predictions_dir, exist_ok=True)
print(f"Combination predictions will be saved to: {os.path.abspath(predictions_dir)}")

from metrics_calculator import calculate_detailed_metrics


def create_detailed_table(metrics_dict, combo_str, dataset_type):
    """Create detailed evaluation table"""
    channel_names = ['x_ch', 'y_ch', 'z_ch']
    n_channels = len(channel_names)

    rows = []

    for i in range(n_channels):
        rows.append({
            'Combination': combo_str,
            'Dataset': dataset_type,
            'Channel': channel_names[i],
            'MSE': metrics_dict['per_channel']['MSE'][i],
            'RMSE': metrics_dict['per_channel']['RMSE'][i],
            'MAE': metrics_dict['per_channel']['MAE'][i],
            'PCC': metrics_dict['per_channel']['PCC'][i],
            'R2': metrics_dict['per_channel']['R2'][i]
        })

    rows.append({
        'Combination': combo_str,
        'Dataset': dataset_type,
        'Channel': 'Average',
        'MSE': metrics_dict['average']['MSE'],
        'RMSE': metrics_dict['average']['RMSE'],
        'MAE': metrics_dict['average']['MAE'],
        'PCC': metrics_dict['average']['PCC'],
        'R2': metrics_dict['average']['R2']
    })

    rows.append({
        'Combination': combo_str,
        'Dataset': dataset_type,
        'Channel': 'Variance',
        'MSE': metrics_dict['variance']['MSE'],
        'RMSE': metrics_dict['variance']['RMSE'],
        'MAE': metrics_dict['variance']['MAE'],
        'PCC': metrics_dict['variance']['PCC'],
        'R2': metrics_dict['variance']['R2']
    })

    rows.append({
        'Combination': combo_str,
        'Dataset': dataset_type,
        'Channel': 'Overall',
        'MSE': metrics_dict['overall']['MSE'],
        'RMSE': metrics_dict['overall']['RMSE'],
        'MAE': metrics_dict['overall']['MAE'],
        'PCC': metrics_dict['overall']['PCC'],
        'R2': metrics_dict['overall']['R2']
    })

    return pd.DataFrame(rows)


def load_and_split_data(split_data_dir="split_data"):
    """Load pre-split datasets from split_data.py"""
    print("Loading pre-split datasets from split_data directory...")

    if not os.path.exists(split_data_dir):
        print(f"Error: split_data directory does not exist: {split_data_dir}")
        print("Please run split_data.py first to generate split datasets")
        return None

    split_info_path = os.path.join(split_data_dir, "split_info.json")
    if not os.path.exists(split_info_path):
        print(f"Error: split_info.json does not exist: {split_info_path}")
        return None

    with open(split_info_path, 'r', encoding='utf-8') as f:
        split_info = json.load(f)

    def load_split_dataset(split_name):
        """Load single dataset (train/val/test)"""
        split_path = os.path.join(split_data_dir, split_name)
        if not os.path.exists(split_path):
            print(f"Error: {split_name} directory does not exist: {split_path}")
            return None, None

        record_dirs = [d for d in os.listdir(split_path)
                       if os.path.isdir(os.path.join(split_path, d))]

        if len(record_dirs) == 0:
            print(f"Warning: {split_name} directory is empty")
            return None, None

        all_X = []
        all_Y = []

        for record_dir in record_dirs:
            record_path = os.path.join(split_path, record_dir)
            try:
                X_record = np.load(os.path.join(record_path, "ecg_12lead.npy"))
                Y_record = np.load(os.path.join(record_path, "vcg.npy"))
                all_X.append(X_record)
                all_Y.append(Y_record)
            except Exception as e:
                print(f"Warning: Error loading record {record_dir}: {e}")
                continue

        if len(all_X) == 0:
            print(f"Warning: No data successfully loaded for {split_name}")
            return None, None

        X_concat = np.concatenate(all_X, axis=0)
        Y_concat = np.concatenate(all_Y, axis=0)

        return X_concat, Y_concat

    X_train, Y_train = load_split_dataset("train")
    X_val, Y_val = load_split_dataset("val")
    X_test, Y_test = load_split_dataset("test")

    if X_train is None or X_val is None or X_test is None:
        print("Error: One or more datasets failed to load")
        return None

    print(f"\nData loading completed:")
    print(f"  Train set: {X_train.shape[0]} windows")
    print(f"  Validation set: {X_val.shape[0]} windows")
    print(f"  Test set: {X_test.shape[0]} windows")

    return {
        'train': (X_train, Y_train),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test)
    }


def save_split_data(X_train, Y_train, X_val, Y_val, X_test, Y_test, results_dir):
    """Save split data to results_dir folder"""
    print("\nSaving split datasets...")

    split_data_dir = os.path.join(results_dir, "split_data")
    os.makedirs(split_data_dir, exist_ok=True)

    np.save(os.path.join(split_data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(split_data_dir, "Y_train.npy"), Y_train)

    np.save(os.path.join(split_data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(split_data_dir, "Y_val.npy"), Y_val)

    np.save(os.path.join(split_data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(split_data_dir, "Y_test.npy"), Y_test)

    data_info = {
        'X_train_shape': X_train.shape,
        'Y_train_shape': Y_train.shape,
        'X_val_shape': X_val.shape,
        'Y_val_shape': Y_val.shape,
        'X_test_shape': X_test.shape,
        'Y_test_shape': Y_test.shape,
        'total_samples': {
            'train': X_train.shape[0],
            'val': X_val.shape[0],
            'test': X_test.shape[0]
        },
        'saved_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(split_data_dir, "split_data_info.json"), 'w') as f:
        json.dump(data_info, f, indent=2)

    print(f"Split data saved to: {split_data_dir}")
    print(f"  Train: X_train.npy ({X_train.shape}), Y_train.npy ({Y_train.shape})")
    print(f"  Validation: X_val.npy ({X_val.shape}), Y_val.npy ({Y_val.shape})")
    print(f"  Test: X_test.npy ({X_test.shape}), Y_test.npy ({Y_test.shape})")

    return split_data_dir


def prepare_data_for_model(X_data, Y_data, combo):
    """Prepare data for model"""
    X_selected = X_data[:, combo, :]
    n_windows, n_selected, n_samples = X_selected.shape

    X_flat = X_selected.transpose(0, 2, 1).reshape(-1, n_selected)
    Y_flat = Y_data.transpose(0, 2, 1).reshape(-1, Y_data.shape[1])

    return X_flat, Y_flat, X_selected


class SerializableRidgeModel:
    """Serializable ridge regression model class (includes global normalization parameters)"""

    def __init__(self, weights, intercept, global_mean=None, global_std=None, best_alpha=None):
        self.weights = weights
        self.intercept = intercept
        self.global_mean = global_mean
        self.global_std = global_std
        self.best_alpha = best_alpha

    def predict(self, X):
        if self.global_mean is not None and self.global_std is not None:
            X = (X - self.global_mean) / self.global_std
        return np.dot(X, self.weights) + self.intercept

    def get_model_info(self):
        """Get model information"""
        return {
            'weights_shape': self.weights.shape,
            'intercept_shape': self.intercept.shape,
            'weights': self.weights.tolist(),
            'intercept': self.intercept.tolist(),
            'best_alpha': self.best_alpha,
            'global_mean': float(self.global_mean) if self.global_mean is not None else None,
            'global_std': float(self.global_std) if self.global_std is not None else None
        }


def train_all_ridge_combinations(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                                 ecg_12lead_names, vcg_names, results_dir, predictions_dir, alpha=1e4):
    """Train all 220 ridge regression combinations and save all models and evaluation metrics"""
    print("\n" + "=" * 80)
    print("Ridge Regression Training Phase")
    print("Fit VCG from any 3 leads selected from 12-lead ECG")
    print(f"Evaluating all {len(list(combinations(range(12), 3)))} three-lead combinations...")
    print("=" * 80)

    all_combinations = list(combinations(range(12), 3))
    print(f"Starting evaluation of {len(all_combinations)} combinations...")

    all_models = {}
    all_train_metrics = []
    all_val_metrics = []
    all_test_metrics = []

    best_val_avg_mse = float('inf')
    best_combo = None
    best_combo_str = ""
    best_model = None

    start_time = time.time()

    for i, combo in enumerate(all_combinations):
        combo = list(combo)
        combo_names = [ecg_12lead_names[idx] for idx in combo]
        combo_str = '-'.join(combo_names)

        if (i + 1) % 1 == 0 or i == 0 or i == len(all_combinations) - 1:
            elapsed_time = time.time() - start_time
            print(f"  Progress: {i + 1}/{len(all_combinations)} - Combination: {combo_str} - Time: {elapsed_time:.1f}s")

        X_train_flat, Y_train_flat, X_train_selected = prepare_data_for_model(X_train, Y_train, combo)
        X_val_flat, Y_val_flat, X_val_selected = prepare_data_for_model(X_val, Y_val, combo)
        X_test_flat, Y_test_flat, X_test_selected = prepare_data_for_model(X_test, Y_test, combo)
    
        has_not_normalized = False
        if has_not_normalized:
            global_mean = np.mean(X_train_flat)
            global_std = np.std(X_train_flat)
            X_train_flat = (X_train_flat - global_mean) / global_std
            Y_train_flat = (Y_train_flat - global_mean) / global_std
        
        n_samples, n_features = X_train_flat.shape
        X_T = X_train_flat.T
        XTX = np.dot(X_T, X_train_flat)

        if i == 0:
            print(f"\nXTX matrix shape: {XTX.shape}")
            print(f"XTX matrix mean: {np.mean(XTX):.2f}")
            print(f"XTX matrix max: {np.max(XTX):.2f}")
            print(f"XTX matrix min: {np.min(XTX):.2f}")

            
        if True:
            from sklearn.linear_model import RidgeCV

            xtx_mean = np.mean(XTX)
            alpha_range = np.logspace(np.log10(xtx_mean * 0.0001), np.log10(xtx_mean * 10), 10)

            ridge_cv = RidgeCV(alphas=alpha_range, cv=5, scoring='neg_mean_squared_error')
            ridge_cv.fit(X_train_flat, Y_train_flat)

            best_alpha = ridge_cv.alpha_
            print(f"\nBest alpha (CV): {best_alpha:.6f}")
        else:
            best_alpha = alpha
            if i == 0:
                print(f"\nUsing fixed alpha: {best_alpha:.6f}")

        eye_matrix = np.eye(n_features)
        ridge_matrix = XTX + best_alpha * eye_matrix
        ridge_matrix_inv = np.linalg.inv(ridge_matrix)
        weights = np.dot(np.dot(ridge_matrix_inv, X_T), Y_train_flat)

        X_mean = X_train_flat.mean(axis=0)
        Y_mean = Y_train_flat.mean(axis=0)
        intercept = Y_mean - np.dot(X_mean, weights)

        if has_not_normalized:
            model = SerializableRidgeModel(weights, intercept, global_mean, global_std, best_alpha)
        else:
            model = SerializableRidgeModel(weights, intercept, best_alpha)

        n_windows_train, n_selected, n_samples_train = X_train_selected.shape
        n_windows_val, n_selected, n_samples_val = X_val_selected.shape
        n_windows_test, n_selected, n_samples_test = X_test_selected.shape

        Y_train_pred = np.zeros_like(Y_train)
        Y_val_pred = np.zeros_like(Y_val)
        Y_test_pred = np.zeros_like(Y_test)

        for i in range(n_windows_train):
            X_window = X_train_selected[i].T
            Y_train_pred[i] = model.predict(X_window).T

        for i in range(n_windows_val):
            X_window = X_val_selected[i].T
            Y_val_pred[i] = model.predict(X_window).T

        for i in range(n_windows_test):
            X_window = X_test_selected[i].T
            Y_test_pred[i] = model.predict(X_window).T
        
        if has_not_normalized:
            Y_train_pred = Y_train_pred * global_std + global_mean
            Y_val_pred = Y_val_pred * global_std + global_mean
            Y_test_pred = Y_test_pred * global_std + global_mean
        
        combo_dir = os.path.join(predictions_dir, f"combo_{combo_str}")
        os.makedirs(combo_dir, exist_ok=True)
        np.save(os.path.join(combo_dir, 'Y_train_LR.npy'), Y_train_pred)
        np.save(os.path.join(combo_dir, 'Y_val_LR.npy'), Y_val_pred)
        np.save(os.path.join(combo_dir, 'Y_test_LR.npy'), Y_test_pred)
        
        if has_not_normalized:
            norm_params_path = os.path.join(combo_dir, "global_norm_params.csv")
            with open(norm_params_path, 'w') as f:
                f.write("parameter,value\n")
                f.write(f"global_mean,{global_mean}\n")
                f.write(f"global_std,{global_std}\n")
        
        train_metrics = calculate_detailed_metrics(Y_train, Y_train_pred, combo_names)
        val_metrics = calculate_detailed_metrics(Y_val, Y_val_pred, combo_names)
        test_metrics = calculate_detailed_metrics(Y_test, Y_test_pred, combo_names)

        train_df = create_detailed_table(train_metrics, combo_str, "train")
        val_df = create_detailed_table(val_metrics, combo_str, "val")
        test_df = create_detailed_table(test_metrics, combo_str, "test")

        all_train_metrics.append(train_df)
        all_val_metrics.append(val_df)
        all_test_metrics.append(test_df)

        all_models[combo_str] = {
            'model': model,
            'combo': combo,
            'combo_names': combo_names,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        save_combo_predictions(combo_str, model, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                               predictions_dir, train_metrics, val_metrics, test_metrics)

        if val_metrics['average']['MSE'] < best_val_avg_mse:
            best_val_avg_mse = val_metrics['average']['MSE']
            best_combo = combo
            best_combo_str = combo_str
            best_model = model

    total_time = time.time() - start_time

    print(f"\nAll combinations trained!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per combination: {total_time / len(all_combinations):.2f} seconds")

    print(f"\nBest combination: {best_combo_str}")
    print(f"Best combination validation average MSE: {best_val_avg_mse:.6f}")

    return all_models, all_train_metrics, all_val_metrics, all_test_metrics, best_combo_str, best_model


def save_combo_predictions(combo_str, model, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                           predictions_dir, train_metrics, val_metrics, test_metrics):
    """Save predictions and evaluation metrics for single combination"""
    combo_dir = os.path.join(predictions_dir, f"combo_{combo_str}")
    os.makedirs(combo_dir, exist_ok=True)

    train_rmse_result = np.column_stack([np.array(train_metrics['per_channel']['RMSE']),
                                         np.array(train_metrics['per_channel']['RMSE_variance'])])
    train_mae_result = np.column_stack([np.array(train_metrics['per_channel']['MAE']),
                                         np.array(train_metrics['per_channel']['MAE_variance'])])
    train_pcc_result = np.column_stack([np.array(train_metrics['per_channel']['PCC']),
                                         np.array(train_metrics['per_channel']['PCC_variance'])])
    train_r2_result = np.column_stack([np.array(train_metrics['per_channel']['R2']),
                                       np.array(train_metrics['per_channel']['R2_variance'])])

    np.save(os.path.join(combo_dir, "train_rmse_mean_var.npy"), train_rmse_result)
    np.save(os.path.join(combo_dir, "train_mae_mean_var.npy"), train_mae_result)
    np.save(os.path.join(combo_dir, "train_pcc_mean_var.npy"), train_pcc_result)
    np.save(os.path.join(combo_dir, "train_r2_mean_var.npy"), train_r2_result)

    val_rmse_result = np.column_stack([np.array(val_metrics['per_channel']['RMSE']),
                                       np.array(val_metrics['per_channel']['RMSE_variance'])])
    val_mae_result = np.column_stack([np.array(val_metrics['per_channel']['MAE']),
                                      np.array(val_metrics['per_channel']['MAE_variance'])])
    val_pcc_result = np.column_stack([np.array(val_metrics['per_channel']['PCC']),
                                      np.array(val_metrics['per_channel']['PCC_variance'])])
    val_r2_result = np.column_stack([np.array(val_metrics['per_channel']['R2']),
                                     np.array(val_metrics['per_channel']['R2_variance'])])

    np.save(os.path.join(combo_dir, "val_rmse_mean_var.npy"), val_rmse_result)
    np.save(os.path.join(combo_dir, "val_mae_mean_var.npy"), val_mae_result)
    np.save(os.path.join(combo_dir, "val_pcc_mean_var.npy"), val_pcc_result)
    np.save(os.path.join(combo_dir, "val_r2_mean_var.npy"), val_r2_result)

    test_rmse_result = np.column_stack([np.array(test_metrics['per_channel']['RMSE']),
                                        np.array(test_metrics['per_channel']['RMSE_variance'])])
    test_mae_result = np.column_stack([np.array(test_metrics['per_channel']['MAE']),
                                       np.array(test_metrics['per_channel']['MAE_variance'])])
    test_pcc_result = np.column_stack([np.array(test_metrics['per_channel']['PCC']),
                                       np.array(test_metrics['per_channel']['PCC_variance'])])
    test_r2_result = np.column_stack([np.array(test_metrics['per_channel']['R2']),
                                      np.array(test_metrics['per_channel']['R2_variance'])])

    np.save(os.path.join(combo_dir, "test_rmse_mean_var.npy"), test_rmse_result)
    np.save(os.path.join(combo_dir, "test_mae_mean_var.npy"), test_mae_result)
    np.save(os.path.join(combo_dir, "test_pcc_mean_var.npy"), test_pcc_result)
    np.save(os.path.join(combo_dir, "test_r2_mean_var.npy"), test_r2_result)

    model_path = os.path.join(combo_dir, "ridge_model.joblib")
    joblib.dump(model, model_path)

    metrics_dict = {
        'combo_name': combo_str,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }

    metrics_path = os.path.join(combo_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

    summary_rows = []

    summary_rows.append({
        'Dataset': 'Train',
        'Channel': 'Overall',
        'MSE': train_metrics['overall']['MSE'],
        'RMSE': train_metrics['overall']['RMSE'],
        'MAE': train_metrics['overall']['MAE'],
        'R2': train_metrics['overall']['R2']
    })

    summary_rows.append({
        'Dataset': 'Val',
        'Channel': 'Overall',
        'MSE': val_metrics['overall']['MSE'],
        'RMSE': val_metrics['overall']['RMSE'],
        'MAE': val_metrics['overall']['MAE'],
        'R2': val_metrics['overall']['R2']
    })

    summary_rows.append({
        'Dataset': 'Test',
        'Channel': 'Overall',
        'MSE': test_metrics['overall']['MSE'],
        'RMSE': test_metrics['overall']['RMSE'],
        'MAE': test_metrics['overall']['MAE'],
        'R2': test_metrics['overall']['R2']
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(combo_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    return combo_dir


def save_all_results(all_models, all_train_metrics, all_val_metrics, all_test_metrics,
                     best_combo_str, best_model, results_dir, predictions_dir):
    """Save all results"""
    print("\nSaving all results...")

    print("Merging evaluation tables...")
    train_df_all = pd.concat(all_train_metrics, ignore_index=True)
    val_df_all = pd.concat(all_val_metrics, ignore_index=True)
    test_df_all = pd.concat(all_test_metrics, ignore_index=True)

    train_csv_path = os.path.join(results_dir, "ridge_train_metrics.csv")
    val_csv_path = os.path.join(results_dir, "ridge_val_metrics.csv")
    test_csv_path = os.path.join(results_dir, "ridge_test_metrics.csv")

    train_df_all.to_csv(train_csv_path, index=False, encoding='utf-8-sig')
    val_df_all.to_csv(val_csv_path, index=False, encoding='utf-8-sig')
    test_df_all.to_csv(test_csv_path, index=False, encoding='utf-8-sig')

    print(f"Train set evaluation table saved: {train_csv_path}")
    print(f"Validation set evaluation table saved: {val_csv_path}")
    print(f"Test set evaluation table saved: {test_csv_path}")

    print("Saving all models...")
    models_path = os.path.join(results_dir, "all_ridge_models.joblib")
    joblib.dump(all_models, models_path)
    print(f"All models saved: {models_path}")

    print("Saving best model...")
    best_model_path = os.path.join(results_dir, f"best_ridge_model_{best_combo_str}.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved: {best_model_path}")

    print("Saving model index file...")
    model_index = []
    for combo_str, model_info in all_models.items():
        model_index.append({
            'Combination': combo_str,
            'Combination Index': model_info['combo'],
            'Combination Names': model_info['combo_names'],
            'Train Avg MSE': model_info['train_metrics']['average']['MSE'],
            'Train Avg RMSE': model_info['train_metrics']['average']['RMSE'],
            'Train Avg MAE': model_info['train_metrics']['average']['MAE'],
            'Train Avg PCC': model_info['train_metrics']['average']['PCC'],
            'Train Avg R2': model_info['train_metrics']['average']['R2'],
            'Train MSE Variance': model_info['train_metrics']['variance']['MSE'],
            'Train RMSE Variance': model_info['train_metrics']['variance']['RMSE'],
            'Train MAE Variance': model_info['train_metrics']['variance']['MAE'],
            'Train PCC Variance': model_info['train_metrics']['variance']['PCC'],
            'Train R2 Variance': model_info['train_metrics']['variance']['R2'],
            'Val Avg MSE': model_info['val_metrics']['average']['MSE'],
            'Val Avg RMSE': model_info['val_metrics']['average']['RMSE'],
            'Val Avg MAE': model_info['val_metrics']['average']['MAE'],
            'Val Avg PCC': model_info['val_metrics']['average']['PCC'],
            'Val Avg R2': model_info['val_metrics']['average']['R2'],
            'Val MSE Variance': model_info['val_metrics']['variance']['MSE'],
            'Val RMSE Variance': model_info['val_metrics']['variance']['RMSE'],
            'Val MAE Variance': model_info['val_metrics']['variance']['MAE'],
            'Val PCC Variance': model_info['val_metrics']['variance']['PCC'],
            'Val R2 Variance': model_info['val_metrics']['variance']['R2'],
            'Test Avg MSE': model_info['test_metrics']['average']['MSE'],
            'Test Avg RMSE': model_info['test_metrics']['average']['RMSE'],
            'Test Avg MAE': model_info['test_metrics']['average']['MAE'],
            'Test Avg PCC': model_info['test_metrics']['average']['PCC'],
            'Test Avg R2': model_info['test_metrics']['average']['R2'],
            'Test MSE Variance': model_info['test_metrics']['variance']['MSE'],
            'Test RMSE Variance': model_info['test_metrics']['variance']['RMSE'],
            'Test MAE Variance': model_info['test_metrics']['variance']['MAE'],
            'Test PCC Variance': model_info['test_metrics']['variance']['PCC'],
            'Test R2 Variance': model_info['test_metrics']['variance']['R2']
        })

    model_index_df = pd.DataFrame(model_index)
    model_index_csv = os.path.join(results_dir, "model_index.csv")
    model_index_df.to_csv(model_index_csv, index=False, encoding='utf-8-sig')
    print(f"Model index file saved: {model_index_csv}")

    print("Saving best combination detailed results...")
    best_model_info = all_models[best_combo_str]

    best_train_df = pd.concat([df for df in all_train_metrics if df.iloc[0]['Combination'] == best_combo_str],
                              ignore_index=True)
    best_val_df = pd.concat([df for df in all_val_metrics if df.iloc[0]['Combination'] == best_combo_str], ignore_index=True)
    best_test_df = pd.concat([df for df in all_test_metrics if df.iloc[0]['Combination'] == best_combo_str], ignore_index=True)

    best_train_csv = os.path.join(results_dir, f"best_combo_train_{best_combo_str}.csv")
    best_val_csv = os.path.join(results_dir, f"best_combo_val_{best_combo_str}.csv")
    best_test_csv = os.path.join(results_dir, f"best_combo_test_{best_combo_str}.csv")

    best_train_df.to_csv(best_train_csv, index=False, encoding='utf-8-sig')
    best_val_df.to_csv(best_val_csv, index=False, encoding='utf-8-sig')
    best_test_df.to_csv(best_test_csv, index=False, encoding='utf-8-sig')

    print(f"Best combination train detailed results saved: {best_train_csv}")
    print(f"Best combination validation detailed results saved: {best_val_csv}")
    print(f"Best combination test detailed results saved: {best_test_csv}")

    print("Saving summary report...")
    summary_path = os.path.join(results_dir, "summary.txt")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Ridge Regression Model Training Summary Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total combinations: {len(all_models)}\n")
        f.write(f"Best combination: {best_combo_str}\n")
        f.write(f"Best combination index: {best_model_info['combo']}\n")
        f.write(f"Best combination validation avg MSE: {best_model_info['val_metrics']['average']['MSE']:.6f}\n")
        f.write(f"Best combination validation avg RMSE: {best_model_info['val_metrics']['average']['RMSE']:.6f}\n")
        f.write(f"Best combination validation avg MAE: {best_model_info['val_metrics']['average']['MAE']:.6f}\n")
        f.write(f"Best combination validation avg PCC: {best_model_info['val_metrics']['average']['PCC']:.6f}\n")
        f.write(f"Best combination validation avg R2: {best_model_info['val_metrics']['average']['R2']:.6f}\n")
        f.write(f"Best combination test avg MSE: {best_model_info['test_metrics']['average']['MSE']:.6f}\n")
        f.write(f"Best combination test avg RMSE: {best_model_info['test_metrics']['average']['RMSE']:.6f}\n")
        f.write(f"Best combination test avg MAE: {best_model_info['test_metrics']['average']['MAE']:.6f}\n")
        f.write(f"Best combination test avg PCC: {best_model_info['test_metrics']['average']['PCC']:.6f}\n")
        f.write(f"Best combination test avg R2: {best_model_info['test_metrics']['average']['R2']:.6f}\n")
        f.write(f"\nPrediction results saved in: {predictions_dir}\n")
        f.write(f"Directory structure:\n")
        f.write(f"  predictions/\n")
        f.write(f"    combo_I-II-III/  # Folder named after combination\n")
        f.write(f"      Y_train_LR.npy  # Train set predictions\n")
        f.write(f"      Y_val_LR.npy    # Validation set predictions\n")
        f.write(f"      Y_test_LR.npy   # Test set predictions\n")
        f.write(f"      Y_train_true.npy  # Train set true labels\n")
        f.write(f"      Y_val_true.npy    # Validation set true labels\n")
        f.write(f"      Y_test_true.npy   # Test set true labels\n")
        f.write(f"      ridge_model.joblib  # Ridge regression model for this combination\n")
        f.write(f"      metrics.json      # Detailed evaluation metrics\n")
        f.write(f"      metrics_summary.csv  # Summary metrics table\n")
        f.write(f"\nSaved files:\n")
        f.write(f"  1. All models: all_ridge_models.joblib\n")
        f.write(f"  2. Best model: best_ridge_model_*.joblib\n")
        f.write(f"  3. Train set evaluation table: ridge_train_metrics.csv\n")
        f.write(f"  4. Validation set evaluation table: ridge_val_metrics.csv\n")
        f.write(f"  5. Test set evaluation table: ridge_test_metrics.csv\n")
        f.write(f"  6. Model index: model_index.csv\n")
        f.write(f"  7. Best combination detailed results: best_combo_*_*.csv\n")
        f.write(f"  8. Summary report: summary.txt\n")
        f.write(f"  9. All combination predictions: {predictions_dir} ({len(all_models)} folders)\n")

    print(f"Summary report saved: {summary_path}")

    return best_combo_str


def main():
    """Main function"""
    print("=" * 80)
    print("Ridge Regression Model Training and Evaluation")
    print("Goal: Fit VCG from any 3 leads selected from 12-lead ECG")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 80)

    global ecg_12lead_names
    ecg_12lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    vcg_names = ['X', 'Y', 'Z']

    print("\n[Step 1] Data preparation...")
    data = load_and_split_data(split_data_dir="split_data")

    if data is None:
        print("Error: Data loading failed")
        return

    X_train, Y_train = data['train']
    X_val, Y_val = data['val']
    X_test, Y_test = data['test']

    print(f"\nData shapes:")
    print(f"  Train X: {X_train.shape}, Y: {Y_train.shape}")
    print(f"  Validation X: {X_val.shape}, Y: {Y_val.shape}")
    print(f"  Test X: {X_test.shape}, Y: {Y_test.shape}")
    
    split_data_dir = save_split_data(X_train, Y_train, X_val, Y_val, X_test, Y_test, results_dir)

    print("\n[Step 2] Training all ridge regression combinations...")
    all_models, all_train_metrics, all_val_metrics, all_test_metrics, best_combo_str, best_model = train_all_ridge_combinations(
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        ecg_12lead_names, vcg_names, results_dir, predictions_dir, alpha=1e4
    )


    use_orthogonalization = False
    if use_orthogonalization:
        print("\nPerforming Gram-Schmidt orthogonalization on true signals...")
        Y_train = gram_schmidt_orthogonalize(Y_train)
        Y_val = gram_schmidt_orthogonalize(Y_val)
        Y_test = gram_schmidt_orthogonalize(Y_test)
        
        true_signals_ortho_dir = os.path.join(results_dir, "true_signals_orthogonalized")
        os.makedirs(true_signals_ortho_dir, exist_ok=True)
        np.save(os.path.join(true_signals_ortho_dir, 'Y_train_ortho.npy'), Y_train)
        np.save(os.path.join(true_signals_ortho_dir, 'Y_val_ortho.npy'), Y_val)
        np.save(os.path.join(true_signals_ortho_dir, 'Y_test_ortho.npy'), Y_test)
        print(f"Orthogonalized true signals saved to: {os.path.abspath(true_signals_ortho_dir)}")
    else:
        print("\nSkipping Gram-Schmidt orthogonalization...")
        
        true_signals_dir = os.path.join(results_dir, "true_signals")
        os.makedirs(true_signals_dir, exist_ok=True)
        np.save(os.path.join(true_signals_dir, 'Y_train.npy'), Y_train)
        np.save(os.path.join(true_signals_dir, 'Y_val.npy'), Y_val)
        np.save(os.path.join(true_signals_dir, 'Y_test.npy'), Y_test)
        print(f"Original true signals saved to: {os.path.abspath(true_signals_dir)}")


    print("\n[Step 3] Saving all results...")
    best_combo_str = save_all_results(
        all_models, all_train_metrics, all_val_metrics, all_test_metrics,
        best_combo_str, best_model, results_dir, predictions_dir
    )

    print("\n[Step 4] Best combination detailed information:")
    best_model_info = all_models[best_combo_str]

    print(f"\nBest combination: {best_combo_str}")
    print(f"Combination index: {best_model_info['combo']}")

    print("\nTrain set metrics:")
    print(f"  Overall MSE: {best_model_info['train_metrics']['overall']['MSE']:.6f}")
    print(f"  Overall RMSE: {best_model_info['train_metrics']['overall']['RMSE']:.6f}")
    print(f"  Overall MAE: {best_model_info['train_metrics']['overall']['MAE']:.6f}")
    print(f"  Overall R2: {best_model_info['train_metrics']['overall']['R2']:.6f}")
    print(f"  Average MSE: {best_model_info['train_metrics']['average']['MSE']:.6f}")
    print(f"  Average RMSE: {best_model_info['train_metrics']['average']['RMSE']:.6f}")
    print(f"  Average MAE: {best_model_info['train_metrics']['average']['MAE']:.6f}")
    print(f"  Average PCC: {best_model_info['train_metrics']['average']['PCC']:.6f}")
    print(f"  Average R2: {best_model_info['train_metrics']['average']['R2']:.6f}")

    print("\nValidation set metrics:")
    print(f"  Overall MSE: {best_model_info['val_metrics']['overall']['MSE']:.6f}")
    print(f"  Overall RMSE: {best_model_info['val_metrics']['overall']['RMSE']:.6f}")
    print(f"  Overall MAE: {best_model_info['val_metrics']['overall']['MAE']:.6f}")
    print(f"  Overall R2: {best_model_info['val_metrics']['overall']['R2']:.6f}")
    print(f"  Average MSE: {best_model_info['val_metrics']['average']['MSE']:.6f}")
    print(f"  Average RMSE: {best_model_info['val_metrics']['average']['RMSE']:.6f}")
    print(f"  Average MAE: {best_model_info['val_metrics']['average']['MAE']:.6f}")
    print(f"  Average PCC: {best_model_info['val_metrics']['average']['PCC']:.6f}")
    print(f"  Average R2: {best_model_info['val_metrics']['average']['R2']:.6f}")

    print("\nTest set metrics:")
    print(f"  Overall MSE: {best_model_info['test_metrics']['overall']['MSE']:.6f}")
    print(f"  Overall RMSE: {best_model_info['test_metrics']['overall']['RMSE']:.6f}")
    print(f"  Overall MAE: {best_model_info['test_metrics']['overall']['MAE']:.6f}")
    print(f"  Overall R2: {best_model_info['test_metrics']['overall']['R2']:.6f}")
    print(f"  Average MSE: {best_model_info['test_metrics']['average']['MSE']:.6f}")
    print(f"  Average RMSE: {best_model_info['test_metrics']['average']['RMSE']:.6f}")
    print(f"  Average MAE: {best_model_info['test_metrics']['average']['MAE']:.6f}")
    print(f"  Average PCC: {best_model_info['test_metrics']['average']['PCC']:.6f}")
    print(f"  Average R2: {best_model_info['test_metrics']['average']['R2']:.6f}")

    print("\n[Step 5] Best combination per-channel detailed metrics (test set):")
    for i, channel in enumerate(vcg_names):
        print(f"\n  Channel {channel}:")
        print(f"    MSE: {best_model_info['test_metrics']['per_channel']['MSE'][i]:.6f}")
        print(f"    RMSE: {best_model_info['test_metrics']['per_channel']['RMSE'][i]:.6f}")
        print(f"    MAE: {best_model_info['test_metrics']['per_channel']['MAE'][i]:.6f}")
        print(f"    PCC: {best_model_info['test_metrics']['per_channel']['PCC'][i]:.6f}")
        print(f"    R2: {best_model_info['test_metrics']['per_channel']['R2'][i]:.6f}")

    print("\n" + "=" * 80)
    print("Ridge Regression Model Training and Evaluation Completed!")
    print("=" * 80)

    print(f"\nAll results saved to: {os.path.abspath(results_dir)}")
    print(f"Best combination: {best_combo_str}")
    print(f"Total combinations: {len(all_models)}")

    print(f"\nPrediction results directory structure:")
    print(f"  {predictions_dir}/")
    print(f"    combo_I-II-III/")
    print(f"      Y_train_LR.npy    # Predicted VCG for train set (X, Y, Z)")
    print(f"      Y_val_LR.npy      # Predicted VCG for validation set")
    print(f"      Y_test_LR.npy     # Predicted VCG for test set")
    print(f"      Y_train_true.npy  # True VCG for train set")
    print(f"      Y_val_true.npy    # True VCG for validation set")
    print(f"      Y_test_true.npy   # True VCG for test set")
    print(f"      ridge_model.joblib # Ridge regression model for this combination")
    print(f"      metrics.json      # Detailed evaluation metrics")
    print(f"      metrics_summary.csv # Summary metrics table")

    print("\nGenerated files:")
    files = os.listdir(results_dir)
    for file in sorted(files):
        if file.endswith('.csv') or file.endswith('.joblib') or file.endswith('.txt'):
            print(f"  - {file}")

    combo_folders = [f for f in os.listdir(predictions_dir) if os.path.isdir(os.path.join(predictions_dir, f))]
    print(f"\nCombination folders created: {len(combo_folders)}")

    if len(combo_folders) > 0:
        print(f"First 5 combination folders:")
        for i, folder in enumerate(sorted(combo_folders)[:5]):
            print(f"  {i + 1}. {folder}")
        if len(combo_folders) > 5:
            print(f"  ... and {len(combo_folders) - 5} more folders")


if __name__ == "__main__":
    main()