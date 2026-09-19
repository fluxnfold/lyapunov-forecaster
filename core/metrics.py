import numpy as np
from typing import Optional, Tuple, Dict
from scipy import stats
from scipy.signal import hilbert, coherence

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan
    std = np.std(y_true)
    if std == 0:
        return np.nan
    rmse = compute_rmse(y_true, y_pred)
    return rmse / std

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred))

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)

def compute_mi_rmse(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    n_bins: int = 20
) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan
    sq_errors = (y_true - y_pred) ** 2
    hist, bin_edges = np.histogram(y_true, bins=n_bins)
    bin_indices = np.digitize(y_true, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    probs = hist / len(y_true)
    probs = np.maximum(probs, 1e-10)
    weights = -np.log(probs[bin_indices])
    weights = weights / np.sum(weights)
  
    return np.sqrt(np.sum(weights * sq_errors))

def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

def compute_envelope_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if len(y_true) < 4:
        return np.nan
    env_true = np.abs(hilbert(y_true))
    env_pred = np.abs(hilbert(y_pred))
    corr = np.corrcoef(env_true, env_pred)[0, 1]
    return float(corr)

def compute_spectral_coherence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fs: float,
    mask: Optional[np.ndarray] = None,
    nperseg: Optional[int] = None
) -> float:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if nperseg is None:
        nperseg = min(int(fs), len(y_true) // 2)
    nperseg = max(nperseg, 8)
    if len(y_true) < nperseg:
        return np.nan
    freqs, coh = coherence(y_true, y_pred, fs=fs, nperseg=nperseg)
    return float(np.mean(coh))

def compute_dtw_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    radius: int = 10
) -> float:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan
    distance, _ = fastdtw(
        y_true.reshape(-1, 1),
        y_pred.reshape(-1, 1),
        dist=euclidean,
        radius=radius
    )
    return float(distance / len(y_true))

def compute_all_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
    include_phase_invariant: bool = True
) -> Dict[str, float]:
    result = {
        'rmse': compute_rmse(y_true, y_pred, mask),
        'nrmse': compute_nrmse(y_true, y_pred, mask),
        'mae': compute_mae(y_true, y_pred, mask),
        'r2': compute_r2(y_true, y_pred, mask),
        'mi_rmse': compute_mi_rmse(y_true, y_pred, mask),
    }
    if include_phase_invariant:
        result['env_corr'] = compute_envelope_correlation(y_true, y_pred, mask)
        if fs is not None:
            result['spec_coh'] = compute_spectral_coherence(y_true, y_pred, fs, mask)
        else:
            result['spec_coh'] = np.nan
        result['dtw_dist'] = compute_dtw_distance(y_true, y_pred, mask)
    
    return result
