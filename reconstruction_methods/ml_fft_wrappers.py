import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from restoration.deep_learning import cnn_imputation, lstm_imputation, TORCH_AVAILABLE

def find_dominant_frequencies(
    signal: np.ndarray,
    fs: float,
    n_freqs: int = 5,
    min_freq: float = 0.1,
    max_freq: Optional[float] = None
) -> np.ndarray:
    
    from scipy.signal import find_peaks

    if max_freq is None:
        max_freq = fs / 2
    fft_vals = rfft(signal)
    fft_freqs = rfftfreq(len(signal), 1/fs)
    fft_power = np.abs(fft_vals)**2
    freq_mask = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
    min_height = np.percentile(fft_power[freq_mask], 90)
    peaks, _ = find_peaks(fft_power[freq_mask], height=min_height, distance=5)
    peak_freqs = fft_freqs[freq_mask][peaks]
    peak_powers = fft_power[freq_mask][peaks]
    sorted_idx = np.argsort(peak_powers)[::-1][:n_freqs]

    return peak_freqs[sorted_idx]

def isolate_frequency_band(
    signal: np.ndarray,
    fs: float,
    center_freq: float,
    bandwidth: float = 0.5,
    filter_order: int = 4
) -> np.ndarray:
    nyquist = fs / 2
    low = max(0.01, center_freq - bandwidth/2)
    high = min(center_freq + bandwidth/2, nyquist - 0.01)
    if low >= high:
      
        low = max(0.01, center_freq - 0.1)
        high = min(center_freq + 0.1, nyquist - 0.01)
  
    b, a = butter(filter_order, [low/nyquist, high/nyquist], btype='band')
    return filtfilt(b, a, signal)

def fill_missing_for_training(
    signal: np.ndarray,
    missing_mask: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    signal_filled = signal.copy()
    if method == 'linear':
        valid_idx = np.where(~missing_mask)[0]
        missing_idx = np.where(missing_mask)[0]
        if len(valid_idx) > 1:
            signal_filled[missing_idx] = np.interp(missing_idx, valid_idx, signal[valid_idx])
    elif method == 'forward':
        mean_val = np.nanmean(signal)
        signal_filled[missing_mask] = mean_val

    return signal_filled

def fft_cnn_imputation(
    df: pd.DataFrame,
    col: str = 'acc_x',
    n_freqs: int = 3,
    bandwidth: float = 0.3,
    epochs: int = 50,
    window_size: int = 50,
    fs: float = 35.15,
    device: Optional[str] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for CNN imputation")
    df_copy = df.copy()
    signal = df_copy[col].values.copy()
    missing_mask = np.isnan(signal)
    if not missing_mask.any():
        return df_copy, {'method': 'fft_cnn', 'frequencies': []}
    signal_filled = fill_missing_for_training(signal, missing_mask)
    dominant_freqs = find_dominant_frequencies(signal_filled, fs, n_freqs=n_freqs)
    if verbose:
        print(f"FFT-CNN: Found {len(dominant_freqs)} dominant frequencies")
        for i, f in enumerate(dominant_freqs):
            print(f"  {i+1}. {f:.3f} Hz")
    reconstructed = np.zeros_like(signal)
    frequencies_used = []

    for freq in dominant_freqs:
        isolated = isolate_frequency_band(signal_filled, fs, freq, bandwidth)
        df_iso = pd.DataFrame({col: isolated})
        df_iso.loc[missing_mask, col] = np.nan
        df_imputed = cnn_imputation(
            df_iso, col=col,
            window_size=window_size,
            epochs=epochs,
            device=device,
            verbose=False
        )
        reconstructed += df_imputed[col].values
        frequencies_used.append(freq)
    residual_signal = signal_filled - np.sum([
        isolate_frequency_band(signal_filled, fs, f, bandwidth) 
        for f in dominant_freqs
    ], axis=0)
    df_residual = pd.DataFrame({col: residual_signal})
    df_residual.loc[missing_mask, col] = np.nan
    df_residual_imputed = cnn_imputation(
        df_residual, col=col,
        window_size=window_size,
        epochs=epochs,
        device=device,
        verbose=False
    )
    reconstructed += df_residual_imputed[col].values
    df_copy.loc[missing_mask, col] = reconstructed[missing_mask]

    metadata = {
        'method': 'fft_cnn',
        'frequencies': frequencies_used,
        'bandwidth': bandwidth,
        'n_freqs': len(frequencies_used),
        'epochs': epochs,
        'window_size': window_size
    }

    return df_copy, metadata

def fft_lstm_imputation(
    df: pd.DataFrame,
    col: str = 'acc_x',
    n_freqs: int = 3,
    bandwidth: float = 0.3,
    epochs: int = 50,
    window_size: int = 50,
    hidden_dim: int = 128,
    fs: float = 35.15,
    device: Optional[str] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for LSTM imputation")
    df_copy = df.copy()
    signal = df_copy[col].values.copy()
    missing_mask = np.isnan(signal)
    if not missing_mask.any():
        return df_copy, {'method': 'fft_lstm', 'frequencies': []}
    signal_filled = fill_missing_for_training(signal, missing_mask)
    dominant_freqs = find_dominant_frequencies(signal_filled, fs, n_freqs=n_freqs)
    if verbose:
        print(f"FFT-LSTM: Found {len(dominant_freqs)} dominant frequencies")
        for i, f in enumerate(dominant_freqs):
            print(f"  {i+1}. {f:.3f} Hz")
    reconstructed = np.zeros_like(signal)
    frequencies_used = []
    for freq in dominant_freqs:
        isolated = isolate_frequency_band(signal_filled, fs, freq, bandwidth)
        df_iso = pd.DataFrame({col: isolated})
        df_iso.loc[missing_mask, col] = np.nan
        df_imputed = lstm_imputation(
            df_iso, col=col,
            window_size=window_size,
            epochs=epochs,
            hidden_dim=hidden_dim,
            device=device,
            verbose=False
        )

        reconstructed += df_imputed[col].values
        frequencies_used.append(freq)
    residual_signal = signal_filled - np.sum([
        isolate_frequency_band(signal_filled, fs, f, bandwidth) 
        for f in dominant_freqs
    ], axis=0)
    df_residual = pd.DataFrame({col: residual_signal})
    df_residual.loc[missing_mask, col] = np.nan
    df_residual_imputed = lstm_imputation(
        df_residual, col=col,
        window_size=window_size,
        epochs=epochs,
        hidden_dim=hidden_dim,
        device=device,
        verbose=False
    )
    reconstructed += df_residual_imputed[col].values
    df_copy.loc[missing_mask, col] = reconstructed[missing_mask]
    
    metadata = {
        'method': 'fft_lstm',
        'frequencies': frequencies_used,
        'bandwidth': bandwidth,
        'n_freqs': len(frequencies_used),
        'epochs': epochs,
        'window_size': window_size,
        'hidden_dim': hidden_dim
    }
    
    return df_copy, metadata

def cnn_imputation_simple(
    df: pd.DataFrame,
    col: str = 'acc_x',
    epochs: int = 50,
    window_size: int = 50,
    device: Optional[str] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    
    df_result = cnn_imputation(
        df, col=col,
        window_size=window_size,
        epochs=epochs,
        device=device,
        verbose=verbose
    )
    metadata = {
        'method': 'cnn_simple',
        'epochs': epochs,
        'window_size': window_size
    }
    
    return df_result, metadata

def lstm_imputation_simple(
    df: pd.DataFrame,
    col: str = 'acc_x',
    epochs: int = 50,
    window_size: int = 50,
    hidden_dim: int = 128,
    device: Optional[str] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    
    df_result = lstm_imputation(
        df, col=col,
        window_size=window_size,
        epochs=epochs,
        hidden_dim=hidden_dim,
        device=device,
        verbose=verbose
    )
    metadata = {
        'method': 'lstm_simple',
        'epochs': epochs,
        'window_size': window_size,
        'hidden_dim': hidden_dim
    }
    
    return df_result, metadata

if __name__ == '__main__':
    print("ML FFT Wrappers Module - Ready for import")
    print("Available methods:")
    print("  - fft_cnn_imputation: CNN with FFT preprocessing")
    print("  - fft_lstm_imputation: LSTM with FFT preprocessing")
    print("  - cnn_imputation_simple: CNN without FFT (baseline)")
    print("  - lstm_imputation_simple: LSTM without FFT (baseline)")