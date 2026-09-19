import numpy as np
import pandas as pd
from scipy import signal, interpolate, optimize
from scipy.fft import fft, ifft, fftfreq
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

def spectral_reconstruction(
    observed_signal: np.ndarray,
    gap_start: int,
    gap_end: int,
    fs: float = 35.0,
    n_harmonics: int = 50
) -> np.ndarray:
    gap_length = gap_end - gap_start
    pre_signal = observed_signal[:gap_start]
    post_signal = observed_signal[gap_end:]
    if len(pre_signal) < 100 or len(post_signal) < 100:
        return _mean_reversion_fill(observed_signal, gap_start, gap_end)
    observed = np.concatenate([pre_signal[-500:], post_signal[:500]])
    n_fft = min(1024, len(observed))
    freqs, psd = signal.welch(observed, fs=fs, nperseg=n_fft)
    n_gen = gap_length + 100
    freq_gen = fftfreq(n_gen, 1/fs)
    psd_interp = np.interp(np.abs(freq_gen), freqs, psd)
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2*np.pi, n_gen)
    amplitudes = np.sqrt(psd_interp * n_gen)
    spectrum = amplitudes * np.exp(1j * phases)
    if n_gen % 2 == 0:
        spectrum[n_gen//2+1:] = np.conj(spectrum[1:n_gen//2][::-1])
        spectrum[0] = np.real(spectrum[0])
        spectrum[n_gen//2] = np.real(spectrum[n_gen//2])
    else:
        spectrum[(n_gen+1)//2:] = np.conj(spectrum[1:(n_gen+1)//2][::-1])
        spectrum[0] = np.real(spectrum[0])
    synthetic = np.real(ifft(spectrum))
    synthetic = (synthetic - np.mean(synthetic)) / (np.std(synthetic) + 1e-8)
    synthetic = synthetic * np.std(observed) + np.mean(observed)
    reconstruction = synthetic[50:50+gap_length]
    reconstruction = _boundary_blend(
        reconstruction,
        pre_signal[-1] if len(pre_signal) > 0 else reconstruction[0],
        post_signal[0] if len(post_signal) > 0 else reconstruction[-1],
        blend_samples=min(50, gap_length // 4)
    )
    return reconstruction

def surrogate_reconstruction(
    observed_signal: np.ndarray,
    gap_start: int,
    gap_end: int,
    n_candidates: int = 20,
    segment_length: int = None
) -> np.ndarray:
    gap_length = gap_end - gap_start
    if segment_length is None:
        segment_length = gap_length
    left_boundary = observed_signal[gap_start - 1] if gap_start > 0 else 0
    right_boundary = observed_signal[gap_end] if gap_end < len(observed_signal) else 0
    if gap_start > 1:
        left_slope = observed_signal[gap_start - 1] - observed_signal[gap_start - 2]
    else:
        left_slope = 0
    if gap_end < len(observed_signal) - 1:
        right_slope = observed_signal[gap_end + 1] - observed_signal[gap_end]
    else:
        right_slope = 0
    pre_signal = observed_signal[:gap_start]
    post_signal = observed_signal[gap_end:]
    candidates = []
    for i in range(len(pre_signal) - segment_length - 1):
        segment = pre_signal[i:i + segment_length]
        if len(segment) == segment_length:
            start_match = abs(segment[0] - left_boundary)
            end_match = abs(segment[-1] - right_boundary)
            if i > 0:
                seg_left_slope = segment[0] - pre_signal[i-1]
            else:
                seg_left_slope = 0
            slope_match = abs(seg_left_slope - left_slope)
            score = start_match + end_match + 0.5 * slope_match
            candidates.append((score, segment.copy()))
    for i in range(len(post_signal) - segment_length - 1):
        segment = post_signal[i:i + segment_length]
        if len(segment) == segment_length:
            start_match = abs(segment[0] - left_boundary)
            end_match = abs(segment[-1] - right_boundary)
            if i > 0:
                seg_left_slope = segment[0] - post_signal[i-1]
            else:
                seg_left_slope = 0
            slope_match = abs(seg_left_slope - left_slope)
          
            score = start_match + end_match + 0.5 * slope_match
            candidates.append((score, segment.copy()))
    if len(candidates) == 0:
        return _mean_reversion_fill(observed_signal, gap_start, gap_end)
    candidates.sort(key=lambda x: x[0])
    best_candidates = candidates[:min(n_candidates, len(candidates))]
    weights = np.array([1.0 / (c[0] + 0.1) for c in best_candidates])
    weights /= weights.sum()
    reconstruction = np.zeros(segment_length)
    for w, (_, segment) in zip(weights, best_candidates):
        reconstruction += w * segment
    reconstruction = _adjust_boundaries(
        reconstruction, left_boundary, right_boundary
    )
    if len(reconstruction) != gap_length:
        x_old = np.linspace(0, 1, len(reconstruction))
        x_new = np.linspace(0, 1, gap_length)
        reconstruction = np.interp(x_new, x_old, reconstruction)
    return reconstruction

def constrained_optimization_reconstruction(
    observed_signal: np.ndarray,
    gap_start: int,
    gap_end: int,
    smoothness_weight: float = 0.1,
    variance_weight: float = 1.0
) -> np.ndarray:
    gap_length = gap_end - gap_start
    observed = np.concatenate([
        observed_signal[:gap_start],
        observed_signal[gap_end:]
    ])
    obs_mean = np.mean(observed)
    obs_std = np.std(observed)
    left_val = observed_signal[gap_start - 1] if gap_start > 0 else obs_mean
    right_val = observed_signal[gap_end] if gap_end < len(observed_signal) else obs_mean
    x0 = np.linspace(left_val, right_val, gap_length)

    def objective(x):
        mean_term = np.sum((x - obs_mean)**2)
        if len(x) > 2:
            d2x = np.diff(x, n=2)
            smooth_term = np.sum(d2x**2)
        else:
            smooth_term = 0
        return mean_term + smoothness_weight * smooth_term * gap_length
  
    def boundary_constraint_left(x):
        return x[0] - left_val
  
    def boundary_constraint_right(x):
        return x[-1] - right_val
  
    def variance_constraint(x):
        return (np.var(x) - obs_std**2 * variance_weight)**2
    
    constraints = [
        {'type': 'eq', 'fun': boundary_constraint_left},
        {'type': 'eq', 'fun': boundary_constraint_right},
    ]
    bounds = [(obs_mean - 4*obs_std, obs_mean + 4*obs_std)] * gap_length

    try:
        result = optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 100}
        )
        reconstruction = result.x
    except Exception:
        reconstruction = x0
    return reconstruction

def _mean_reversion_fill(
    observed_signal: np.ndarray,
    gap_start: int,
    gap_end: int
) -> np.ndarray:
    gap_length = gap_end - gap_start
    observed = np.concatenate([
        observed_signal[:gap_start],
        observed_signal[gap_end:]
    ])
    obs_mean = np.mean(observed)
    left_val = observed_signal[gap_start - 1] if gap_start > 0 else obs_mean
    right_val = observed_signal[gap_end] if gap_end < len(observed_signal) else obs_mean
    t = np.linspace(0, np.pi, gap_length)
    mid_point = gap_length // 2
    reconstruction = np.zeros(gap_length)
    t1 = np.linspace(0, np.pi, mid_point)
    reconstruction[:mid_point] = left_val + (obs_mean - left_val) * (1 - np.cos(t1)) / 2
    t2 = np.linspace(0, np.pi, gap_length - mid_point)
    reconstruction[mid_point:] = obs_mean + (right_val - obs_mean) * (1 - np.cos(t2)) / 2
    return reconstruction

def _boundary_blend(
    reconstruction: np.ndarray,
    left_boundary: float,
    right_boundary: float,
    blend_samples: int = 20
) -> np.ndarray:
    
    n = len(reconstruction)
    blend_samples = min(blend_samples, n // 4)
    if blend_samples < 2:
        return reconstruction
    result = reconstruction.copy()
    left_diff = left_boundary - reconstruction[0]
    t = np.linspace(0, np.pi/2, blend_samples)
    left_blend = np.cos(t)
    result[:blend_samples] += left_diff * left_blend
    right_diff = right_boundary - reconstruction[-1]
    t = np.linspace(np.pi/2, 0, blend_samples)
    right_blend = np.cos(t)
    result[-blend_samples:] += right_diff * right_blend
    return result

def _adjust_boundaries(
    reconstruction: np.ndarray,
    left_target: float,
    right_target: float
) -> np.ndarray:
    n = len(reconstruction)
    left_current = reconstruction[0]
    right_current = reconstruction[-1]
    t = np.linspace(0, 1, n)
    left_adjust = (left_target - left_current) * (1 - t)
    right_adjust = (right_target - right_current) * t
    return reconstruction + left_adjust + right_adjust

def statistical_reconstruction(
    df: pd.DataFrame,
    col: str = 'acc_x',
    method: str = 'surrogate'
) -> pd.DataFrame:
    df_result = df.copy()
    signal = df_result[col].values.copy()
    is_nan = np.isnan(signal)
    if not np.any(is_nan):
        return df_result
    gap_starts = np.where(np.diff(np.concatenate([[False], is_nan])) == 1)[0]
    gap_ends = np.where(np.diff(np.concatenate([is_nan, [False]])) == -1)[0] + 1

    for gap_start, gap_end in zip(gap_starts, gap_ends):
        if method == 'spectral':
            reconstruction = spectral_reconstruction(signal, gap_start, gap_end)
        elif method == 'surrogate':
            reconstruction = surrogate_reconstruction(signal, gap_start, gap_end)
        elif method == 'optimized':
            reconstruction = constrained_optimization_reconstruction(signal, gap_start, gap_end)
        else:
            reconstruction = _mean_reversion_fill(signal, gap_start, gap_end)
        signal[gap_start:gap_end] = reconstruction

    df_result[col] = signal
    return df_result

if __name__ == '__main__':
    import json
    from sklearn.metrics import mean_squared_error, r2_score

    print("Testing Statistical Reconstruction Methods")
    print("=" * 70)
    with open('data/session_2026-01-23_11-52-39_1_long_pendulum.json', 'r') as f:
        data = json.load(f)

    readings = data['readings']
    acc_x = np.array([r['accel']['x'] for r in readings])
    n = len(acc_x)
    print(f"Loaded {n} samples")
    gap_frac = 0.10
    gap_size = int(n * gap_frac)
    gap_start = 2000
    gap_end = gap_start + gap_size
    true_values = acc_x[gap_start:gap_end].copy()
    print(f"\nTesting with {gap_frac*100:.0f}% gap ({gap_size} samples)")
    print("-" * 70)
    methods = ['spectral', 'surrogate', 'optimized', 'mean_reversion']
    for method in methods:
        try:
            if method == 'spectral':
                pred = spectral_reconstruction(acc_x, gap_start, gap_end)
            elif method == 'surrogate':
                pred = surrogate_reconstruction(acc_x, gap_start, gap_end)
            elif method == 'optimized':
                pred = constrained_optimization_reconstruction(acc_x, gap_start, gap_end)
            else:
                pred = _mean_reversion_fill(acc_x, gap_start, gap_end)
            rmse = np.sqrt(mean_squared_error(true_values, pred))
            r2 = r2_score(true_values, pred)
            true_mean, true_std = np.mean(true_values), np.std(true_values)
            pred_mean, pred_std = np.mean(pred), np.std(pred)
            print(f"{method:15s}: RMSE={rmse:.4f}, R2={r2:.4f}")
            print(f"                 True: mean={true_mean:.2f}, std={true_std:.2f}")
            print(f"                 Pred: mean={pred_mean:.2f}, std={pred_std:.2f}")

        except Exception as e:
            print(f"{method:15s}: Failed - {e}")
  
    print("\n" + "=" * 70)
    print("Testing complete!")