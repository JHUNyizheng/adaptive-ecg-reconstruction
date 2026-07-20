import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def calculate_detailed_metrics(y_true, y_pred, channel_names):
    """Calculate detailed evaluation metrics: including overall metrics and per-channel metrics.
    Supports 3D array input (num_windows x num_channels x sequence_length).
    For each window and each channel, compute RMSE, MAE, PCC, and R2, then average and compute variance across windows.
    Uses vectorized computation instead of per-window loops for significant performance improvement.
    """
    if len(y_true.shape) == 3:
        n_windows, n_channels, n_samples = y_true.shape
        
        mse_per_channel = []
        mse_var_per_channel = []
        rmse_per_channel = []
        rmse_var_per_channel = []
        mae_per_channel = []
        mae_var_per_channel = []
        pcc_per_channel = []
        pcc_var_per_channel = []
        r2_per_channel = []
        r2_var_per_channel = []
        
        for ch in range(n_channels):
            y_true_ch = y_true[:, ch, :]
            y_pred_ch = y_pred[:, ch, :]
            
            diff = y_true_ch - y_pred_ch
            mse_ch = np.mean(diff ** 2, axis=1)
            rmse_ch = np.sqrt(mse_ch)
            mae_ch = np.mean(np.abs(diff), axis=1)
            
            y_true_centered = y_true_ch - y_true_ch.mean(axis=1, keepdims=True)
            y_pred_centered = y_pred_ch - y_pred_ch.mean(axis=1, keepdims=True)
            numerator = np.sum(y_true_centered * y_pred_centered, axis=1)
            denominator = np.sqrt(np.sum(y_true_centered ** 2, axis=1) * np.sum(y_pred_centered ** 2, axis=1))
            pcc_ch = np.where(denominator > 0, numerator / denominator, 0.0)
            
            ss_res = np.sum(diff ** 2, axis=1)
            y_true_mean = y_true_ch.mean(axis=1, keepdims=True)
            ss_tot = np.sum((y_true_ch - y_true_mean) ** 2, axis=1)
            r2_ch = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
            
            mse_per_channel.append(float(np.mean(mse_ch)))
            mse_var_per_channel.append(float(np.var(mse_ch)))
            rmse_per_channel.append(float(np.mean(rmse_ch)))
            rmse_var_per_channel.append(float(np.var(rmse_ch)))
            mae_per_channel.append(float(np.mean(mae_ch)))
            mae_var_per_channel.append(float(np.var(mae_ch)))
            pcc_per_channel.append(float(np.mean(pcc_ch)))
            pcc_var_per_channel.append(float(np.var(pcc_ch)))
            r2_per_channel.append(float(np.mean(r2_ch)))
            r2_var_per_channel.append(float(np.var(r2_ch)))
        
        y_true_flat = y_true.transpose(0, 2, 1).reshape(-1, n_channels)
        y_pred_flat = y_pred.transpose(0, 2, 1).reshape(-1, n_channels)
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        y_true_flat_c = y_true_flat - y_true_flat.mean(axis=0, keepdims=True)
        y_pred_flat_c = y_pred_flat - y_pred_flat.mean(axis=0, keepdims=True)
        num = np.sum(y_true_flat_c * y_pred_flat_c, axis=0)
        den = np.sqrt(np.sum(y_true_flat_c ** 2, axis=0) * np.sum(y_pred_flat_c ** 2, axis=0))
        pcc = [float(p) if d > 0 else 0.0 for p, d in zip(num / np.where(den > 0, den, 1.0), den)]
        avg_pcc = np.mean(pcc)
        
        detailed_metrics = {
            'overall': {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'PCC': avg_pcc,
                'R2': r2
            },
            'per_channel': {
                'MSE': mse_per_channel,
                'RMSE': rmse_per_channel,
                'MAE': mae_per_channel,
                'PCC': pcc_per_channel,
                'R2': r2_per_channel,
                'MSE_variance': mse_var_per_channel,
                'RMSE_variance': rmse_var_per_channel,
                'MAE_variance': mae_var_per_channel,
                'PCC_variance': pcc_var_per_channel,
                'R2_variance': r2_var_per_channel
            },
            'average': {
                'MSE': np.mean(mse_per_channel),
                'RMSE': np.mean(rmse_per_channel),
                'MAE': np.mean(mae_per_channel),
                'PCC': np.mean(pcc_per_channel),
                'R2': np.mean(r2_per_channel)
            },
            'variance': {
                'MSE': np.var(mse_per_channel),
                'RMSE': np.var(rmse_per_channel),
                'MAE': np.var(mae_per_channel),
                'PCC': np.var(pcc_per_channel),
                'R2': np.var(r2_per_channel)
            },
            'per_channel_mean_var': {
                'RMSE': [[rmse, rmse_var] for rmse, rmse_var in zip(rmse_per_channel, rmse_var_per_channel)],
                'MAE': [[mae, mae_var] for mae, mae_var in zip(mae_per_channel, mae_var_per_channel)],
                'PCC': [[pcc, pcc_var] for pcc, pcc_var in zip(pcc_per_channel, pcc_var_per_channel)],
                'R2': [[r2, r2_var] for r2, r2_var in zip(r2_per_channel, r2_var_per_channel)]
            }
        }
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        n_channels = y_true.shape[1]
        mse_per_channel = []
        rmse_per_channel = []
        mae_per_channel = []
        pcc_per_channel = []
        r2_per_channel = []

        for i in range(n_channels):
            y_true_ch = y_true[:, i]
            y_pred_ch = y_pred[:, i]

            mse_ch = mean_squared_error(y_true_ch, y_pred_ch)
            rmse_ch = np.sqrt(mse_ch)
            mse_per_channel.append(mse_ch)
            rmse_per_channel.append(rmse_ch)

            mae_ch = mean_absolute_error(y_true_ch, y_pred_ch)
            mae_per_channel.append(mae_ch)

            if np.std(y_true_ch) > 0 and np.std(y_pred_ch) > 0:
                pcc_ch, _ = pearsonr(y_true_ch, y_pred_ch)
            else:
                pcc_ch = 0.0
            pcc_per_channel.append(pcc_ch)

            r2_ch = r2_score(y_true_ch, y_pred_ch)
            r2_per_channel.append(r2_ch)

        avg_mse = np.mean(mse_per_channel)
        avg_rmse = np.mean(rmse_per_channel)
        avg_mae = np.mean(mae_per_channel)
        avg_pcc = np.mean(pcc_per_channel)
        avg_r2 = np.mean(r2_per_channel)
        
        var_mse = np.var(mse_per_channel)
        var_rmse = np.var(rmse_per_channel)
        var_mae = np.var(mae_per_channel)
        var_pcc = np.var(pcc_per_channel)
        var_r2 = np.var(r2_per_channel)

        detailed_metrics = {
            'overall': {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            },
            'per_channel': {
                'MSE': mse_per_channel,
                'RMSE': rmse_per_channel,
                'MAE': mae_per_channel,
                'PCC': pcc_per_channel,
                'R2': r2_per_channel
            },
            'average': {
                'MSE': avg_mse,
                'RMSE': avg_rmse,
                'MAE': avg_mae,
                'PCC': avg_pcc,
                'R2': avg_r2
            },
            'variance': {
                'MSE': var_mse,
                'RMSE': var_rmse,
                'MAE': var_mae,
                'PCC': var_pcc,
                'R2': var_r2
            }
        }

    detailed_metrics['channel_names'] = channel_names

    return detailed_metrics