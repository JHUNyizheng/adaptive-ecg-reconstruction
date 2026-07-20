"""
Train MLP model using ridge regression best combination results
Map predicted VCG (X,Y,Z) back to 12-lead ECG
"""

import os
import copy as copy
import numpy as np
import pandas as pd
import json
import time
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

RANDOM_SEED = 20251226
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

INDEPENDENT_NAMES = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
INDEPENDENT_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]

script_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = f"{script_name}_results"
os.makedirs(results_dir, exist_ok=True)


def load_ridge_results():
    best_combo = "II-V3-V6"
    combo_indices = [1, 8, 11]
    ridge_results_dir = "12-3-Ridge-1_results"
    combo_dir = os.path.join(ridge_results_dir, "all_combo_predictions", f"combo_{best_combo}")
    
    Y_train_LR = np.load(os.path.join(combo_dir, "Y_train_LR.npy"))
    Y_val_LR = np.load(os.path.join(combo_dir, "Y_val_LR.npy"))
    
    return {
        'best_combo': best_combo,
        'combo_indices': combo_indices,
        'combo_names': best_combo,
        'Y_train_LR': Y_train_LR,
        'Y_val_LR': Y_val_LR,
    }


def load_split_ecg_data():
    ridge_results_dir = "12-3-Ridge-1_results"
    split_data_dir = os.path.join(ridge_results_dir, "split_data")
    
    X_train = np.load(os.path.join(split_data_dir, "X_train.npy"))
    X_val = np.load(os.path.join(split_data_dir, "X_val.npy"))
    
    ecg_12lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    return {
        'ecg_12lead_names': ecg_12lead_names,
        'X_train': X_train,
        'X_val': X_val,
    }


def prepare_mlp_data(ridge_data, ecg_data):
    Y_train_LR = ridge_data['Y_train_LR']
    Y_val_LR = ridge_data['Y_val_LR']
    
    X_train_ecg = ecg_data['X_train']
    X_val_ecg = ecg_data['X_val']
    
    X_train_mlp = copy.copy(Y_train_LR)
    X_val_mlp = copy.copy(Y_val_LR)
    
    Y_train_mlp = copy.copy(X_train_ecg)
    Y_val_mlp = copy.copy(X_val_ecg)
    
    return {
        'X_train': X_train_mlp,
        'Y_train': Y_train_mlp,
        'X_val': X_val_mlp,
        'Y_val': Y_val_mlp,
        'n_train': X_train_mlp.shape[0],
        'n_val': X_val_mlp.shape[0]
    }


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
        num_levels = len(num_channels)
        
        for i in range(num_levels):
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


def train_model(model, train_loader, val_loader, device, results_dir, max_epochs=100, patience=20):
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.MSELoss(reduction='none')
    
    hard_channel_indices = [2]
    phase1_epochs = 0
    phase2_hard_weight = 1.0
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    model_type = model.__class__.__name__.lower()
    models_dir = os.path.join(results_dir, f"models_{model_type}")
    os.makedirs(models_dir, exist_ok=True)
    
    start_time = time.time()
    epochs_no_improve = 0
    
    for epoch in range(1, max_epochs + 1):
        is_phase1 = epoch <= phase1_epochs
        
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            target_12 = target
            
            if is_phase1:
                loss = criterion(output[:, hard_channel_indices, :],
                               target_12[:, hard_channel_indices, :]).mean()
            else:
                channel_loss = criterion(output, target_12)
                weights = torch.ones(1, 12, 1, device=device)
                for ch in hard_channel_indices:
                    weights[0, ch, 0] = phase2_hard_weight
                loss = (channel_loss * weights).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        train_loss = total_loss / n_batches
        train_losses.append(train_loss)
        
        model.eval()
        val_total_loss = 0
        val_n_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                target_12 = target
                
                if is_phase1:
                    loss = criterion(output[:, hard_channel_indices, :],
                                     target_12[:, hard_channel_indices, :]).mean()
                else:
                    channel_loss = criterion(output, target_12)
                    weights = torch.ones(1, 12, 1, device=device)
                    for ch in hard_channel_indices:
                        weights[0, ch, 0] = phase2_hard_weight
                    loss = (channel_loss * weights).mean()

                val_total_loss += loss.item()
                val_n_batches += 1
        
        val_loss = val_total_loss / val_n_batches
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch:3d}/{max_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            
            model_path = os.path.join(models_dir, f"best_{model_type}_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, model_path)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break
    
    training_time = time.time() - start_time
    model.load_state_dict(best_model_state)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'patience': patience
    }
    
    history_path = os.path.join(models_dir, f"{model_type}_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    loss_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_df.to_csv(os.path.join(models_dir, f"{model_type}_training_losses.csv"), index=False)
    
    return {
        'training_time': training_time,
        'total_epochs': epoch,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'patience': patience
    }


def calculate_window_metrics(y_true_window, y_pred_window):
    n_channels = y_true_window.shape[1]
    
    mse_per_channel = []
    mae_per_channel = []
    rmse_per_channel = []
    r2_per_channel = []
    pcc_per_channel = []

    for ch in range(n_channels):
        y_true_ch = y_true_window[:, ch]
        y_pred_ch = y_pred_window[:, ch]

        mse_ch = mean_squared_error(y_true_ch, y_pred_ch)
        mse_per_channel.append(mse_ch)
        mae_per_channel.append(mean_absolute_error(y_true_ch, y_pred_ch))
        rmse_per_channel.append(np.sqrt(mse_ch))
        r2_per_channel.append(r2_score(y_true_ch, y_pred_ch))
        
        if np.std(y_true_ch) > 0 and np.std(y_pred_ch) > 0:
            pcc_ch, _ = pearsonr(y_true_ch, y_pred_ch)
        else:
            pcc_ch = 0.0
        pcc_per_channel.append(pcc_ch)

    return np.array([mse_per_channel, mae_per_channel, rmse_per_channel, r2_per_channel, pcc_per_channel])


def evaluate_model_on_dataset(model, dataset_name, input_data, target_data, channel_names, device):
    model.eval()
    model.to(device)
    
    window_metrics_list = []
    n_windows = input_data.shape[0]
    
    with torch.no_grad():
        for window_idx in range(n_windows):
            window_input = input_data[window_idx]
            window_input_tensor = torch.FloatTensor(window_input).unsqueeze(0).to(device)
            
            window_pred = model(window_input_tensor)
            window_pred_np = window_pred.squeeze(0).cpu().numpy().transpose(1, 0)
            
            window_true = target_data[window_idx].transpose(1, 0)[:, INDEPENDENT_INDICES]
            
            window_metrics_list.append(calculate_window_metrics(window_true, window_pred_np))
    
    window_metrics_array = np.array(window_metrics_list)
    avg_metrics = np.mean(window_metrics_array, axis=0)
    var_metrics = np.var(window_metrics_array, axis=0)
    
    metric_names = ['MSE', 'MAE', 'RMSE', 'R2', 'PCC']
    avg_channel_metrics = {mn: avg_metrics[i].tolist() for i, mn in enumerate(metric_names)}
    var_channel_metrics = {mn: var_metrics[i].tolist() for i, mn in enumerate(metric_names)}
    
    return {
        'window_metrics_array': window_metrics_array,
        'avg_channel_metrics': avg_channel_metrics,
        'var_channel_metrics': var_channel_metrics,
        'evaluation_time': None
    }


def save_evaluation_results(dataset_name, evaluation_results, channel_names, results_dir, model_checkpoint=None):
    window_metrics_array = evaluation_results['window_metrics_array']
    np.save(os.path.join(results_dir, f"window_metrics_array_{dataset_name.lower()}.npy"), window_metrics_array)
    
    avg_channel_metrics = evaluation_results['avg_channel_metrics']
    var_channel_metrics = evaluation_results['var_channel_metrics']
    
    channel_metrics_list = []
    for i, channel_name in enumerate(channel_names):
        channel_metrics_list.append({
            'Channel': channel_name,
            'MSE_avg': avg_channel_metrics['MSE'][i], 'MSE_var': var_channel_metrics['MSE'][i],
            'MAE_avg': avg_channel_metrics['MAE'][i], 'MAE_var': var_channel_metrics['MAE'][i],
            'RMSE_avg': avg_channel_metrics['RMSE'][i], 'RMSE_var': var_channel_metrics['RMSE'][i],
            'R2_avg': avg_channel_metrics['R2'][i], 'R2_var': var_channel_metrics['R2'][i],
            'PCC_avg': avg_channel_metrics['PCC'][i], 'PCC_var': var_channel_metrics['PCC'][i]
        })
    
    pd.DataFrame(channel_metrics_list).to_csv(
        os.path.join(results_dir, f"channel_metrics_{dataset_name.lower()}.csv"),
        index=False, encoding='utf-8-sig'
    )

    if model_checkpoint is not None:
        summary_results = {
            f'{dataset_name.lower()}_info': {
                'n_windows': window_metrics_array.shape[0],
                'channel_names': channel_names
            },
            'model_info': {
                'best_epoch': model_checkpoint['epoch'],
                'best_val_loss': float(model_checkpoint['val_loss']),
                'best_train_loss': float(model_checkpoint['train_loss']),
                'evaluation_time': evaluation_results['evaluation_time']
            },
            'avg_channel_metrics': evaluation_results['avg_channel_metrics'],
            'var_channel_metrics': evaluation_results['var_channel_metrics']
        }
        
        with open(os.path.join(results_dir, f"evaluation_summary_{dataset_name.lower()}.json"), 'w') as f:
            json.dump(summary_results, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)


def save_summary_results(ridge_info, ecg_info, mlp_data, model_info, training_info, results_dir, model_type='mlp'):
    models_dir = os.path.join(results_dir, f"models_{model_type}")
    
    dataset_info = {
        'ridge_combo': ridge_info['best_combo'],
        'ridge_indices': ridge_info['combo_indices'],
        'ridge_names': ridge_info['combo_names'],
        'ecg_channels': ecg_info['ecg_12lead_names'],
        'train_samples': mlp_data['n_train'],
        'val_samples': mlp_data['n_val'],
        'input_dim': 3,
        'output_dim': 12
    }
    
    with open(os.path.join(models_dir, f"{model_type}_dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    model_summary = {
        'model_name': model_type.upper(),
        'input_size': 3,
        'total_params': model_info['total_params'],
        'trainable_params': model_info['trainable_params'],
        'loss_function': 'MSELoss',
        'optimizer': 'Adam',
        'learning_rate': 0.0001
    }
    
    with open(os.path.join(models_dir, f"{model_type}_model_summary.json"), 'w') as f:
        json.dump(model_summary, f, indent=2)
    
    training_summary = {
        'start_time': training_info['start_time'],
        'training_time': training_info['training_time'],
        'total_epochs': training_info['total_epochs'],
        'best_epoch': training_info['best_epoch'],
        'best_val_loss': training_info['best_val_loss'],
        'patience': training_info['patience'],
        'random_seed': RANDOM_SEED
    }
    
    with open(os.path.join(models_dir, f"{model_type}_training_summary.json"), 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    if model_type == 'mlp':
        arch = "3->6->9->8 (independent leads)"
    elif model_type == 'cnn1d':
        arch = "Unet-like Conv1D"
    else:
        arch = "LSTM -> FC"
    
    summary_text = f"""{model_type.upper()} Training Summary
============================

Dataset:
  Ridge combo: {ridge_info['best_combo']}
  Train samples: {mlp_data['n_train']:,}
  Val samples: {mlp_data['n_val']:,}

Model:
  Architecture: {arch}
  Total params: {model_info['total_params']:,}
  Trainable params: {model_info['trainable_params']:,}

Training:
  Time: {training_info['training_time']:.2f}s
  Epochs: {training_info['total_epochs']}
  Best val loss: {training_info['best_val_loss']:.6f} (epoch {training_info['best_epoch']})
"""
    
    with open(os.path.join(models_dir, f"{model_type}_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(summary_text)


def main():
    print("=" * 80)
    print("MLP Model Training")
    print(f"Ridge combo: II-V3-V6 | Seed: {RANDOM_SEED}")
    print("=" * 80)
    
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_start_time = time.time()
    
    print("\n[1/7] Loading ridge regression results...")
    ridge_data = load_ridge_results()
    
    print("[2/7] Loading split ECG data...")
    ecg_data = load_split_ecg_data()
    
    print("[3/7] Preparing MLP data...")
    mlp_data = prepare_mlp_data(ridge_data, ecg_data)
    
    print("[4/7] Creating data loaders...")
    X_train = torch.FloatTensor(mlp_data['X_train'])
    Y_train = torch.FloatTensor(mlp_data['Y_train'])
    X_val = torch.FloatTensor(mlp_data['X_val'])
    Y_val = torch.FloatTensor(mlp_data['Y_val'])
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=32, shuffle=False)
    
    print("[5/7] Creating and training model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model_type = "MLP"
    if model_type == "MLP":
        model = MLP()
        model_name = "mlp"
    elif model_type == "TCN":
        model = TCN()
        model_name = "tcn"
    else:
        model = LSTM()
        model_name = "lstm"
    
    model.to(device)
    total_params, trainable_params = model.get_parameter_count()
    print(f"Params: {total_params:,} (trainable: {trainable_params:,})")
    
    training_result = train_model(model, train_loader, val_loader, device, results_dir, max_epochs=200, patience=100)
    
    print("\n[6/7] Evaluating model...")
    models_dir = os.path.join(results_dir, f"models_{model_name}")
    
    train_eval_results = evaluate_model_on_dataset(
        model=model,
        dataset_name="train",
        input_data=mlp_data['X_train'],
        target_data=mlp_data['Y_train'],
        channel_names=ecg_data['ecg_12lead_names'],
        device=device
    )
    
    val_eval_results = evaluate_model_on_dataset(
        model=model,
        dataset_name="val",
        input_data=mlp_data['X_val'],
        target_data=mlp_data['Y_val'],
        channel_names=ecg_data['ecg_12lead_names'],
        device=device
    )
    
    best_model_checkpoint = torch.load(os.path.join(models_dir, f"best_{model_name}_model.pth"))
    
    save_evaluation_results("Training", train_eval_results, INDEPENDENT_NAMES, models_dir, best_model_checkpoint)
    save_evaluation_results("Validation", val_eval_results, INDEPENDENT_NAMES, models_dir, best_model_checkpoint)
    
    print("[7/7] Saving results...")
    save_summary_results(ridge_data, ecg_data, mlp_data,
                         {'total_params': total_params, 'trainable_params': trainable_params},
                         {'start_time': start_time_str, **training_result},
                         results_dir, model_name)
    
    print("\n" + "=" * 80)
    print(f"{model_type} Training Complete!")
    print(f"Total time: {time.time() - total_start_time:.2f}s")
    print(f"Results: {os.path.abspath(results_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()