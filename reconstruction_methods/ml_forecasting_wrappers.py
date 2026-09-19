import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Forecasting methods will not work.")

from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

def get_device() -> 'torch.device':
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class ForecastingCNN(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dims: list = [64, 128, 64], 
                 kernel_size: int = 5):
        super(ForecastingCNN, self).__init__()

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv1d(in_dim, hidden_dim, kernel_size, 
                                   padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class ForecastingLSTM(nn.Module):
  
  
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(ForecastingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
      
        lstm_out, _ = self.lstm(x)
      
        return self.fc(lstm_out[:, -1, :])

def find_clean_segments(signal: np.ndarray) -> List[Tuple[int, int]]:
    isnan = np.isnan(signal)
    segments = []
    start = None
    
    for i, is_missing in enumerate(isnan):
        if not is_missing and start is None:
            start = i
        elif is_missing and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(signal)))

    return segments

def create_training_sequences(signal: np.ndarray, window_size: int, 
                               segments: Optional[List[Tuple[int, int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if segments is None:
        segments = find_clean_segments(signal)
    X_list = []
    y_list = []
    for start, end in segments:
        segment_len = end - start
        if segment_len < window_size + 1:
            continue
        segment = signal[start:end]
        for i in range(segment_len - window_size):
            X_list.append(segment[i:i + window_size])
            y_list.append(segment[i + window_size])
    if len(X_list) == 0:
        return np.array([]).reshape(0, window_size, 1), np.array([])
    
    X = np.array(X_list).reshape(-1, window_size, 1)
    y = np.array(y_list)
    
    return X, y

def train_forecasting_model(model_class, model_kwargs, X, y, epochs, 
                             batch_size, learning_rate, device, verbose=False):
    X_mean = X.mean()
    X_std = X.std() + 1e-8
    X_norm = (X - X_mean) / X_std
    y_norm = (y - X_mean) / X_std
    X_t = torch.FloatTensor(X_norm).to(device)
    y_t = torch.FloatTensor(y_norm).unsqueeze(-1).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model_class(**model_kwargs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
      
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")
    model.X_mean = X_mean
    model.X_std = X_std
    
    return model

def predict_gap(model, seed_window: np.ndarray, gap_length: int, device) -> np.ndarray:
    model.eval()
    window_size = len(seed_window)
    predictions = []
    window = (seed_window - model.X_mean) / model.X_std
    
    with torch.no_grad():
        for _ in range(gap_length):
            X = torch.FloatTensor(window).reshape(1, window_size, 1).to(device)
            pred_norm = model(X).cpu().numpy().flatten()[0]
            pred = pred_norm * model.X_std + model.X_mean
            predictions.append(pred)
            window = np.append(window[1:], (pred - model.X_mean) / model.X_std)
    return np.array(predictions)

def forecasting_cnn_imputation(
    df: pd.DataFrame,
    col: str = 'acc_x',
    window_size: int = 100,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_dims: list = [64, 128, 64],
    device: Optional[str] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    if device is None:
        dev = get_device()
    else:
        dev = torch.device(device)
    if verbose:
        print(f"Forecasting CNN imputation using device: {dev}")
    
    df_copy = df.copy()
    signal = df_copy[col].values.copy()
    missing_mask = np.isnan(signal)
    
    if not missing_mask.any():
        return df_copy, {'method': 'forecasting_cnn', 'n_gaps': 0}
    
    segments = find_clean_segments(signal)
    total_clean = sum(end - start for start, end in segments)
    if verbose:
        print(f"Found {len(segments)} clean segments, {total_clean} clean samples")
    if total_clean < window_size + 10:
        warnings.warn("Insufficient clean data for training. Using mean imputation.")
        df_copy.loc[missing_mask, col] = np.nanmean(signal)
        return df_copy, {'method': 'forecasting_cnn', 'n_gaps': int(missing_mask.sum()), 
                         'fallback': 'mean'}
    X, y = create_training_sequences(signal, window_size, segments)
    if len(X) == 0:
        df_copy.loc[missing_mask, col] = np.nanmean(signal)
        return df_copy, {'method': 'forecasting_cnn', 'fallback': 'mean'}
    if verbose:
        print(f"Created {len(X)} training sequences")
    model = train_forecasting_model(
        ForecastingCNN,
        {'input_dim': 1, 'hidden_dims': hidden_dims},
        X, y, epochs, batch_size, learning_rate, dev, verbose
    )
    gaps = []
    i = 0
    while i < len(signal):
        if missing_mask[i]:
            gap_start = i
            while i < len(signal) and missing_mask[i]:
                i += 1
            gap_end = i
            gaps.append((gap_start, gap_end))
        else:
            i += 1
    if verbose:
        print(f"Found {len(gaps)} gaps to fill")
    filled_count = 0
    for gap_start, gap_end in gaps:
        gap_length = gap_end - gap_start
        if gap_start >= window_size:
            seed_start = gap_start - window_size
            seed_window = signal[seed_start:gap_start]
            if not np.isnan(seed_window).any():
                predictions = predict_gap(model, seed_window, gap_length, dev)
                df_copy.iloc[gap_start:gap_end, df_copy.columns.get_loc(col)] = predictions
                filled_count += gap_length
                continue
        mean_val = np.nanmean(signal)
        df_copy.iloc[gap_start:gap_end, df_copy.columns.get_loc(col)] = mean_val
    metadata = {
        'method': 'forecasting_cnn',
        'n_gaps': len(gaps),
        'filled_samples': filled_count,
        'clean_samples': total_clean,
        'window_size': window_size,
        'epochs': epochs,
        'n_training_sequences': len(X)
    }
    return df_copy, metadata

def forecasting_lstm_imputation(
    df: pd.DataFrame,
    col: str = 'acc_x',
    window_size: int = 100,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_dim: int = 128,
    num_layers: int = 2,
    device: Optional[str] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    if device is None:
        dev = get_device()
    else:
        dev = torch.device(device)
    if verbose:
        print(f"Forecasting LSTM imputation using device: {dev}")
    df_copy = df.copy()
    signal = df_copy[col].values.copy()
    missing_mask = np.isnan(signal)
    if not missing_mask.any():
        return df_copy, {'method': 'forecasting_lstm', 'n_gaps': 0}
    segments = find_clean_segments(signal)
    total_clean = sum(end - start for start, end in segments)
    if total_clean < window_size + 10:
        warnings.warn("Insufficient clean data for training.")
        df_copy.loc[missing_mask, col] = np.nanmean(signal)
        return df_copy, {'method': 'forecasting_lstm', 'fallback': 'mean'}
    X, y = create_training_sequences(signal, window_size, segments)
    if len(X) == 0:
        df_copy.loc[missing_mask, col] = np.nanmean(signal)
        return df_copy, {'method': 'forecasting_lstm', 'fallback': 'mean'}
    model = train_forecasting_model(
        ForecastingLSTM,
        {'input_dim': 1, 'hidden_dim': hidden_dim, 'num_layers': num_layers},
        X, y, epochs, batch_size, learning_rate, dev, verbose
    )
    gaps = []
    i = 0
    while i < len(signal):
        if missing_mask[i]:
            gap_start = i
            while i < len(signal) and missing_mask[i]:
                i += 1
            gap_end = i
            gaps.append((gap_start, gap_end))
        else:
            i += 1
    filled_count = 0
    for gap_start, gap_end in gaps:
        gap_length = gap_end - gap_start
        if gap_start >= window_size:
            seed_window = signal[gap_start - window_size:gap_start]
            if not np.isnan(seed_window).any():
                predictions = predict_gap(model, seed_window, gap_length, dev)
                df_copy.iloc[gap_start:gap_end, df_copy.columns.get_loc(col)] = predictions
                filled_count += gap_length
                continue
        df_copy.iloc[gap_start:gap_end, df_copy.columns.get_loc(col)] = np.nanmean(signal)
    metadata = {
        'method': 'forecasting_lstm',
        'n_gaps': len(gaps),
        'filled_samples': filled_count,
        'clean_samples': total_clean,
        'window_size': window_size,
        'epochs': epochs,
        'n_training_sequences': len(X)
    }
    return df_copy, metadata

def find_dominant_frequencies(signal: np.ndarray, fs: float, n_freqs: int = 3) -> np.ndarray:
    from scipy.signal import find_peaks
    
    fft_vals = rfft(signal)
    fft_freqs = rfftfreq(len(signal), 1/fs)
    fft_power = np.abs(fft_vals)**2
    min_height = np.percentile(fft_power[fft_freqs > 0.1], 90)
    peaks, _ = find_peaks(fft_power[fft_freqs > 0.1], height=min_height, distance=5)
    peak_freqs = fft_freqs[fft_freqs > 0.1][peaks]
    peak_powers = fft_power[fft_freqs > 0.1][peaks]
    if len(peak_powers) > 0:
        sorted_idx = np.argsort(peak_powers)[::-1][:n_freqs]
        return peak_freqs[sorted_idx]
    return np.array([1.0])

def isolate_frequency_band(signal: np.ndarray, fs: float, center_freq: float, 
                           bandwidth: float = 0.3) -> np.ndarray:
    nyquist = fs / 2
    low = max(0.01, center_freq - bandwidth/2)
    high = min(center_freq + bandwidth/2, nyquist - 0.01)
    if low >= high:
        low = max(0.01, center_freq - 0.1)
        high = min(center_freq + 0.1, nyquist - 0.01)
    b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
    return filtfilt(b, a, signal)

def forecasting_fft_cnn_imputation(
    df: pd.DataFrame,
    col: str = 'acc_x',
    window_size: int = 100,
    n_freqs: int = 3,
    bandwidth: float = 0.3,
    epochs: int = 50,
    fs: float = 35.15,
    device: Optional[str] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    if device is None:
        dev = get_device()
    else:
        dev = torch.device(device)
    
    df_copy = df.copy()
    signal = df_copy[col].values.copy()
    missing_mask = np.isnan(signal)
    if not missing_mask.any():
        return df_copy, {'method': 'forecasting_fft_cnn', 'n_gaps': 0}
    signal_filled = signal.copy()
    signal_filled[missing_mask] = np.nanmean(signal)
    dominant_freqs = find_dominant_frequencies(signal_filled, fs, n_freqs)
    if verbose:
        print(f"Dominant frequencies: {dominant_freqs}")
    segments = find_clean_segments(signal)
    total_clean = sum(end - start for start, end in segments)
    if total_clean < window_size + 10:
        df_copy.loc[missing_mask, col] = np.nanmean(signal)
        return df_copy, {'method': 'forecasting_fft_cnn', 'fallback': 'mean'}
    models = []
    for freq in dominant_freqs:
        signal_band = isolate_frequency_band(signal_filled, fs, freq, bandwidth)
        X, y = create_training_sequences(signal_band, window_size, segments)
        if len(X) == 0:
            continue
        model = train_forecasting_model(
            ForecastingCNN,
            {'input_dim': 1, 'hidden_dims': [32, 64, 32]},
            X, y, epochs, batch_size=16, learning_rate=0.001, device=dev, verbose=False
        )
        models.append((freq, model))
    residual = signal_filled.copy()
    for freq in dominant_freqs:
        residual = residual - isolate_frequency_band(signal_filled, fs, freq, bandwidth)
    X_res, y_res = create_training_sequences(residual, window_size, segments)
    if len(X_res) > 0:
        model_res = train_forecasting_model(
            ForecastingCNN,
            {'input_dim': 1, 'hidden_dims': [32, 64, 32]},
            X_res, y_res, epochs, batch_size=16, learning_rate=0.001, device=dev, verbose=False
        )
        models.append((0, model_res))  
    gaps = []
    i = 0
    while i < len(signal):
        if missing_mask[i]:
            gap_start = i
            while i < len(signal) and missing_mask[i]:
                i += 1
            gap_end = i
            gaps.append((gap_start, gap_end))
        else:
            i += 1
    filled_count = 0
    for gap_start, gap_end in gaps:
        gap_length = gap_end - gap_start
        if gap_start >= window_size:
            seed_window = signal[gap_start - window_size:gap_start]
            if not np.isnan(seed_window).any():
                pred_sum = np.zeros(gap_length)
                for freq, model in models:
                    if freq == 0:
                        seed_band = seed_window
                    else:
                        seed_band = isolate_frequency_band(seed_window, fs, freq, bandwidth)
                    pred = predict_gap(model, seed_band, gap_length, dev)
                    pred_sum += pred
                df_copy.iloc[gap_start:gap_end, df_copy.columns.get_loc(col)] = pred_sum
                filled_count += gap_length
                continue
        df_copy.iloc[gap_start:gap_end, df_copy.columns.get_loc(col)] = np.nanmean(signal)
    
    metadata = {
        'method': 'forecasting_fft_cnn',
        'n_gaps': len(gaps),
        'filled_samples': filled_count,
        'dominant_freqs': list(dominant_freqs),
        'n_models': len(models),
        'window_size': window_size,
        'epochs': epochs
    }
  
    return df_copy, metadata

if __name__ == '__main__':
    print("ML Forecasting Wrappers Module - Ready for import")
    print("Available methods:")
    print("  - forecasting_cnn_imputation: CNN forecasting on clean data")
    print("  - forecasting_lstm_imputation: LSTM forecasting on clean data")
    print("  - forecasting_fft_cnn_imputation: FFT + CNN forecasting approach")