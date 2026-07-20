"""
Evaluate trained MLP model on test set
Load ridge regression predictions with fixed combination II-V3-V4
Batch inference, split by patient, save pred_ecg_8.npy
"""

import os
import numpy as np
import pandas as pd
import json
import time
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

warnings.filterwarnings('ignore')

RANDOM_SEED = 20251226
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

script_name = os.path.splitext(os.path.basename(__file__))[0]
base_results_dir = f"{script_name}_results"
os.makedirs(base_results_dir, exist_ok=True)

RIDGE_RESULTS_DIR = "12-3-Ridge-1_results"
RIDGE_PRED_ROOT = os.path.join(RIDGE_RESULTS_DIR, "all_combo_predictions")
MLP_TRAIN_DIR = "12-3-Ridge-1-mlp_results"
TEST_DATA_DIR = "split_data/test"

BEST_COMBO = "II-V3-V4"
WINDOW_SIZE = 50
MATRIX_TYPE = "covariance"
DEVICE_TYPE = 'cpu'
N_SAMPLE_PLOTS = 5

INDEPENDENT_NAMES = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
INDEPENDENT_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]
ECG_12LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class MLP(nn.Module):
    def __init__(self, input_size=3, hidden1_size=6, hidden2_size=9, output_size=12, dropout_rate=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1).reshape(-1, channels)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = x.reshape(batch_size, seq_len, -1)
        x = x.permute(0, 2, 1)
        return x

    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


class TCN(nn.Module):
    def __init__(self, input_size=3, output_size=8, num_channels=[64, 64, 128, 128], 
                 kernel_size=3, dropout_rate=0.3):
        super(TCN, self).__init__()
        self.input_conv = nn.Conv1d(input_size, num_channels[0], kernel_size=1)
        
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_channels[i-1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]
            
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                             dilation=dilation_size, padding=dilation_size)
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        self.tcn_layers = nn.Sequential(*layers)
        self.output_conv = nn.Conv1d(num_channels[-1], output_size, kernel_size=1)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.tcn_layers(x)
        x = self.output_conv(x)
        return x

    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=8, output_size=8, num_layers=1, dropout=0.3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0 if num_layers == 1 else dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.reshape(-1, self.lstm.hidden_size)
        x = self.fc(lstm_out)
        x = x.reshape(batch_size, seq_len, -1)
        x = x.permute(0, 2, 1)
        return x

    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def load_test_data():
    combo_dir = os.path.join(RIDGE_PRED_ROOT, f"combo_{BEST_COMBO}")
    Y_test_LR = np.load(os.path.join(combo_dir, "Y_test_LR.npy"))
    
    split_data_dir = os.path.join(RIDGE_RESULTS_DIR, "split_data")
    X_test = np.load(os.path.join(split_data_dir, "X_test.npy"))
    
    return {
        'Y_test_LR': Y_test_LR,
        'X_test': X_test,
        'n_windows': X_test.shape[0],
        'n_samples_per_window': X_test.shape[2]
    }


def get_patient_window_counts():
    patient_ids = sorted([
        d for d in os.listdir(TEST_DATA_DIR)
        if os.path.isdir(os.path.join(TEST_DATA_DIR, d))
        and os.path.exists(os.path.join(TEST_DATA_DIR, d, "ecg_12lead.npy"))
    ])
    window_counts = []
    for pid in patient_ids:
        ecg = np.load(os.path.join(TEST_DATA_DIR, pid, "ecg_12lead.npy"))
        window_counts.append(ecg.shape[0])
    return patient_ids, window_counts


def load_model(model_type):
    models_dir = os.path.join(MLP_TRAIN_DIR, f"models_{model_type.lower()}")
    model_path = os.path.join(models_dir, f"best_{model_type.lower()}_model.pth")
    
    if model_type == "MLP":
        model = MLP()
    elif model_type == "TCN":
        model = TCN(input_size=3, output_size=8)
    else:
        model = LSTM(input_size=3, hidden_size=64, output_size=8, num_layers=1)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def calculate_window_metrics(y_true_window, y_pred_window):
    n_channels = y_pred_window.shape[1]
    mse_list, mae_list, rmse_list, r2_list, pcc_list = [], [], [], [], []
    for ch in range(n_channels):
        yt, yp = y_true_window[:, ch], y_pred_window[:, ch]
        mse = mean_squared_error(yt, yp)
        mse_list.append(mse)
        mae_list.append(mean_absolute_error(yt, yp))
        rmse_list.append(np.sqrt(mse))
        r2_list.append(r2_score(yt, yp))
        if np.std(yt) > 0 and np.std(yp) > 0:
            pcc_list.append(pearsonr(yt, yp)[0])
        else:
            pcc_list.append(0.0)
    return np.array([mse_list, mae_list, rmse_list, r2_list, pcc_list])


def compute_derived_leads_from_I_II(I_mv, II_mv):
    III_mv = II_mv - I_mv
    aVR_mv = -(I_mv + II_mv) / 2.0
    aVL_mv = (2.0 * I_mv - II_mv) / 2.0
    aVF_mv = (2.0 * II_mv - I_mv) / 2.0
    return III_mv, aVR_mv, aVL_mv, aVF_mv


def calculate_covariance_metrics(y_pred):
    n_windows, n_channels, n_samples = y_pred.shape
    step = WINDOW_SIZE
    
    cov_offdiag_metrics = []
    cov_det_metrics = []
    cov_cond_metrics = []
    
    for window_idx in range(n_windows):
        signal_segment = y_pred[window_idx]
        for start in range(0, n_samples - WINDOW_SIZE + 1, step):
            window_data = signal_segment[:, start:start+WINDOW_SIZE]
            
            if MATRIX_TYPE == "covariance":
                matrix = np.cov(window_data.T, rowvar=False)
            else:
                matrix = np.corrcoef(window_data.T, rowvar=False)
            
            off_diag_values = matrix[~np.eye(matrix.shape[0], dtype=bool)]
            cov_offdiag_metrics.append(np.mean(np.abs(off_diag_values)))
            cov_det_metrics.append(np.linalg.det(matrix))
            
            eigvals = np.linalg.eigvals(matrix)
            lambda_max, lambda_min = np.max(np.abs(eigvals)), np.min(np.abs(eigvals))
            cov_cond_metrics.append(lambda_max / lambda_min if lambda_min > 0 else np.inf)
    
    return {
        'cov_offdiag_mean': np.mean(cov_offdiag_metrics) if cov_offdiag_metrics else 0,
        'cov_det_mean': np.mean(cov_det_metrics) if cov_det_metrics else 0,
        'cov_cond_mean': np.mean(cov_cond_metrics) if cov_cond_metrics else 0
    }


def evaluate_and_save_by_patient(model, device, test_data, results_dir):
    X_test_ecg = test_data['X_test']
    n_windows = test_data['n_windows']
    
    model.eval()
    model.to(device)
    
    patient_ids, window_counts = get_patient_window_counts()
    
    process = None
    initial_memory_mb = 0
    peak_memory_mb = 0
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / (1024 ** 2)
        peak_memory_mb = initial_memory_mb
    
    gpu_info = {}
    if device.type == 'cuda':
        gpu_info = {
            'gpu_name': torch.cuda.get_device_name(device),
            'gpu_memory_total_mb': torch.cuda.get_device_properties(device).total_memory / (1024 ** 2),
            'gpu_memory_allocated_mb': 0,
            'gpu_memory_cached_mb': 0
        }
    
    lr_inference_time = 0.0
    Y_test_LR = None
    ridge_models_path = os.path.join(RIDGE_RESULTS_DIR, "all_ridge_models.joblib")
    
    if os.path.exists(ridge_models_path):
        try:
            import joblib
            lr_models = joblib.load(ridge_models_path)
            
            if BEST_COMBO in lr_models:
                lr_model = lr_models[BEST_COMBO]['model']
                
                combo_parts = BEST_COMBO.split('-')
                lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                lead_indices = [lead_names.index(p) for p in combo_parts]
                X_3lead = X_test_ecg[:, lead_indices, :]
                
                lr_start_time = time.time()
                Y_test_LR = np.zeros((n_windows, 3, test_data['n_samples_per_window']))
                for w in range(n_windows):
                    Y_test_LR[w] = lr_model.predict(X_3lead[w].T).T
                lr_inference_time = time.time() - lr_start_time
            else:
                Y_test_LR = test_data['Y_test_LR']
        except Exception:
            Y_test_LR = test_data['Y_test_LR']
    else:
        Y_test_LR = test_data['Y_test_LR']
    
    odi_calc_time = time.time()
    cov_metrics = calculate_covariance_metrics(Y_test_LR)
    odi_calc_time = time.time() - odi_calc_time
    
    all_pred_12lead = []
    all_window_metrics = []
    mlp_inference_time = 0.0
    
    with torch.no_grad():
        for w in range(n_windows):
            x = torch.FloatTensor(Y_test_LR[w]).unsqueeze(0).to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_start = time.time()
            y_pred = model(x).squeeze(0).cpu().numpy()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            mlp_inference_time += time.time() - infer_start
            
            all_pred_12lead.append(y_pred)
            
            yt = X_test_ecg[w].transpose(1, 0)
            yp = y_pred.transpose(1, 0)
            all_window_metrics.append(calculate_window_metrics(yt, yp))
    
    pred_ecg_8_flat = np.array(all_pred_12lead)
    window_metrics_array = np.array(all_window_metrics)
    
    avg_metrics = np.mean(window_metrics_array, axis=0)
    var_metrics = np.var(window_metrics_array, axis=0)
    metric_names = ['MSE', 'MAE', 'RMSE', 'R2', 'PCC']
    avg_channel = {mn: avg_metrics[i].tolist() for i, mn in enumerate(metric_names)}
    var_channel = {mn: var_metrics[i].tolist() for i, mn in enumerate(metric_names)}
    
    start_idx = 0
    for pid, nw in zip(patient_ids, window_counts):
        end_idx = start_idx + nw
        patient_out_dir = os.path.join(results_dir, f"patient_{pid}")
        os.makedirs(patient_out_dir, exist_ok=True)
        np.save(os.path.join(patient_out_dir, "pred_ecg_8.npy"), pred_ecg_8_flat[start_idx:end_idx])
        start_idx = end_idx
    
    end_to_end_time = lr_inference_time + odi_calc_time + mlp_inference_time
    
    mlp_total_params = sum(p.numel() for p in model.parameters())
    mlp_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlp_param_size_mb = mlp_total_params * 4 / (1024 ** 2)
    
    lr_model_info = {
        'total_combinations': 0,
        'total_coefficients': 0,
        'file_size_mb': 0,
        'memory_usage_mb': 0,
        'dtype': 'float64',
        'bytes_per_element': 8,
        'components': {}
    }
    
    if os.path.exists(ridge_models_path):
        try:
            import joblib
            lr_models = joblib.load(ridge_models_path)
            lr_model_info['total_combinations'] = len(lr_models)
            
            weights_count = 0
            intercept_count = 0
            for model_info in lr_models.values():
                lr_model = model_info['model']
                if hasattr(lr_model, 'weights'):
                    weights_count += lr_model.weights.size
                if hasattr(lr_model, 'intercept'):
                    intercept_count += lr_model.intercept.size
            
            lr_model_info['total_coefficients'] = weights_count + intercept_count
            lr_model_info['file_size_mb'] = os.path.getsize(ridge_models_path) / (1024 ** 2)
            lr_model_info['memory_usage_mb'] = lr_model_info['total_coefficients'] * lr_model_info['bytes_per_element'] / (1024 ** 2)
        except Exception:
            pass
    
    total_params = mlp_total_params + lr_model_info['total_coefficients']
    total_model_size_mb = mlp_param_size_mb + lr_model_info['memory_usage_mb']
    
    performance_report = {
        'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': "MLP",
        'device': str(device),
        'device_type': DEVICE_TYPE,
        'best_combo': BEST_COMBO,
        
        'dataset_info': {
            'n_patients': len(patient_ids),
            'total_windows': n_windows,
            'n_samples_per_window': 500
        },
        
        'time_metrics': {
            'total_end_to_end_time_sec': end_to_end_time,
            'stage1_lr_inference_sec': lr_inference_time,
            'stage2_odi_calculation_sec': odi_calc_time,
            'stage3_mlp_inference_sec': mlp_inference_time,
            'avg_lr_inference_per_window_ms': (lr_inference_time / n_windows * 1000) if n_windows > 0 else 0,
            'avg_odi_calculation_per_window_ms': (odi_calc_time / n_windows * 1000) if n_windows > 0 else 0,
            'avg_mlp_inference_per_window_ms': (mlp_inference_time / n_windows * 1000) if n_windows > 0 else 0,
        },
        
        'memory_metrics': {
            'initial_memory_mb': initial_memory_mb,
            'peak_memory_mb': peak_memory_mb,
            'memory_increase_mb': peak_memory_mb - initial_memory_mb
        },
        
        'gpu_info': gpu_info,
        
        'model_info': {
            'total_parameters': total_params,
            'mlp_parameters': {
                'total': mlp_total_params,
                'trainable': mlp_trainable_params,
                'size_mb': mlp_param_size_mb
            },
            'lr_models': lr_model_info,
            'complete_model_size_mb': total_model_size_mb
        },
        
        'results_dir': results_dir
    }
    
    with open(os.path.join(results_dir, "performance_report.json"), 'w') as f:
        json.dump(performance_report, f, indent=2)
    
    return {
        'window_metrics_array': window_metrics_array,
        'avg_channel_metrics': avg_channel,
        'var_channel_metrics': var_channel,
        'total_windows': n_windows,
        'evaluation_time': None,
        'total_params': total_params,
        'trainable_params': mlp_trainable_params,
        'param_size_mb': mlp_param_size_mb,
        'model_size_mb': total_model_size_mb,
        'avg_inference_time_ms': None
    }


def plot_window_comparison(window_idx, true_signals, pred_signals, channel_names,
                           n_samples_per_window, results_dir):
    n_channels = len(channel_names)
    n_rows = (n_channels + 3) // 4
    n_cols = min(n_channels, 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_channels == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    time_axis = np.arange(n_samples_per_window)
    for ch in range(n_channels):
        ax = axes[ch]
        ax.plot(time_axis, true_signals[:, ch], 'b-', linewidth=1.5, label='Ground Truth', alpha=0.8)
        ax.plot(time_axis, pred_signals[:, ch], 'r-', linewidth=1.0, label='Predicted', alpha=0.7)
        ax.set_title(f'{channel_names[ch]} Lead', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle(f'Window {window_idx} - ECG Signal Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_dir = os.path.join(results_dir, "sample_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"window_{window_idx}_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def create_sample_plots(patient_ids, results_dir, n_samples=5):
    patient_id = np.random.choice(patient_ids)
    
    pred_path = os.path.join(results_dir, f"patient_{patient_id}", "pred_ecg_8.npy")
    ecg_path = os.path.join(TEST_DATA_DIR, patient_id, "ecg_12lead.npy")
    if not os.path.exists(pred_path) or not os.path.exists(ecg_path):
        return []
    
    pred_ecg_8 = np.load(pred_path)
    ecg_true = np.load(ecg_path)
    n_windows = pred_ecg_8.shape[0]
    
    selected_indices = list(range(min(n_windows, n_samples)))
    
    plot_paths = []
    for window_idx in selected_indices:
        pred_signal = pred_ecg_8[window_idx].transpose(1, 0)
        true_signal = ecg_true[window_idx][INDEPENDENT_INDICES].transpose(1, 0)
        plot_path = plot_window_comparison(
            window_idx=f"{patient_id}_{window_idx}",
            true_signals=true_signal,
            pred_signals=pred_signal,
            channel_names=INDEPENDENT_NAMES,
            n_samples_per_window=500,
            results_dir=results_dir
        )
        plot_paths.append(plot_path)
    return plot_paths


def save_evaluation_results(evaluation_results, model_checkpoint, results_dir, patient_ids):
    np.save(os.path.join(results_dir, "window_metrics_array.npy"),
            evaluation_results['window_metrics_array'])
    
    avg_channel = evaluation_results['avg_channel_metrics']
    var_channel = evaluation_results['var_channel_metrics']
    
    rows = []
    for i, ch_name in enumerate(INDEPENDENT_NAMES):
        rows.append({
            'Channel': ch_name,
            'MSE_avg': avg_channel['MSE'][i], 'MSE_var': var_channel['MSE'][i],
            'MAE_avg': avg_channel['MAE'][i], 'MAE_var': var_channel['MAE'][i],
            'RMSE_avg': avg_channel['RMSE'][i], 'RMSE_var': var_channel['RMSE'][i],
            'R2_avg': avg_channel['R2'][i], 'R2_var': var_channel['R2'][i],
            'PCC_avg': avg_channel['PCC'][i], 'PCC_var': var_channel['PCC'][i]
        })
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, "channel_metrics.csv"),
                              index=False, encoding='utf-8-sig')
    
    summary = {
        'test_info': {
            'best_combo': BEST_COMBO,
            'n_patients': len(patient_ids),
            'n_total_windows': evaluation_results['total_windows'],
            'n_samples_per_window': 500,
            'independent_channel_names': ECG_12LEAD_NAMES,
        },
        'model_info': {
            'best_epoch': model_checkpoint['epoch'],
            'best_val_loss': float(model_checkpoint['val_loss']),
            'best_train_loss': float(model_checkpoint['train_loss']),
            'evaluation_time': evaluation_results['evaluation_time']
        },
        'avg_channel_metrics': avg_channel,
        'var_channel_metrics': var_channel
    }
    with open(os.path.join(results_dir, "evaluation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)


def main():
    print("=" * 80)
    print("Model Test Set Evaluation")
    print(f"Fixed combo: {BEST_COMBO} | Seed: {RANDOM_SEED}")
    print("=" * 80)
    
    start_time = time.time()
    
    test_data = load_test_data()
    patient_ids, _ = get_patient_window_counts()
    
    model_type = "MLP"
    results_dir = os.path.join(base_results_dir, f"results_{model_type.lower()}")
    os.makedirs(results_dir, exist_ok=True)
    
    model, model_checkpoint = load_model(model_type)
    
    if DEVICE_TYPE == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif DEVICE_TYPE == 'cpu':
        device = torch.device("cpu")
    elif DEVICE_TYPE == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluation_results = evaluate_and_save_by_patient(model, device, test_data, results_dir)
    
    save_evaluation_results(evaluation_results, model_checkpoint, results_dir, patient_ids)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"{model_type} Evaluation Complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Fixed combo: {BEST_COMBO}")
    print(f"Test patients: {len(patient_ids)}")
    print(f"Results: {os.path.abspath(results_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()