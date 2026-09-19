import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel
from typing import Tuple, Optional

def cubic_spline_interpolation(
    df: pd.DataFrame,
    col: str = 'acc_x',
    adaptive: bool = True,
    bc_type: str = 'natural'
) -> pd.DataFrame:
    df_copy = df.copy()
    valid_mask = ~df_copy[col].isna()
    x_valid = df_copy.index[valid_mask].values.astype(float)
    y_valid = df_copy.loc[valid_mask, col].values
    if len(x_valid) < 2:
        return df_copy
    if adaptive:
        knots = _adaptive_knot_selection(x_valid, y_valid)
    else:
        knots = x_valid
    knot_indices = [np.argmin(np.abs(x_valid - k)) for k in knots]
    x_knots = x_valid[knot_indices]
    y_knots = y_valid[knot_indices]
    cs = CubicSpline(x_knots, y_knots, bc_type=bc_type)
    x_all = df_copy.index.values.astype(float)
    y_interp = cs(x_all)
    df_copy.loc[~valid_mask, col] = y_interp[~valid_mask]
    return df_copy

def _adaptive_knot_selection(
    x: np.ndarray,
    y: np.ndarray,
    max_knots: int = 100
) -> np.ndarray:
    if len(x) <= max_knots:
        return x
    if len(x) < 3:
        return x
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    density = np.abs(d2y) ** (1/3)
    density = density / np.sum(density)
    cum_density = np.cumsum(density)
    target_positions = np.linspace(0, 1, max_knots)
    knot_indices = np.searchsorted(cum_density, target_positions)
    knot_indices = np.clip(knot_indices, 0, len(x) - 1)
    knot_indices = np.unique(np.concatenate([[0], knot_indices, [len(x) - 1]]))
    return x[knot_indices]

def _insert_gap_knots(
    x: np.ndarray,
    gap_start: int,
    gap_end: int,
    gap_size_threshold: int = 5
) -> np.ndarray:
    gap_size = gap_end - gap_start
    if gap_size <= 1:
        return x
    if gap_size > gap_size_threshold:
        n_gap_knots = min(gap_size // 2, 10)
        i = np.arange(n_gap_knots)
        gap_knots = (gap_start + gap_end) / 2 + \
                    (gap_end - gap_start) / 2 * \
                    np.cos((2*i + 1) * np.pi / (2 * (n_gap_knots + 1)))
    else:
        n_gap_knots = gap_size // 2
        gap_knots = np.linspace(gap_start, gap_end, n_gap_knots + 2)[1:-1]
  
    return np.sort(np.concatenate([x, gap_knots]))

def gaussian_process_regression(
    df: pd.DataFrame,
    col: str = 'acc_x',
    kernel_type: str = 'combined',
    optimize_hyperparams: bool = True,
    max_train_samples: int = 1000
) -> pd.DataFrame:
    df_copy = df.copy()
    valid_mask = ~df_copy[col].isna()
    X_train = df_copy.index[valid_mask].values.reshape(-1, 1).astype(float)
    y_train = df_copy.loc[valid_mask, col].values
  
    if len(X_train) < 2:
        return df_copy
  
    if len(X_train) > max_train_samples:
        indices = np.linspace(0, len(X_train) - 1, max_train_samples, dtype=int)
        indices = np.unique(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
  
    nan_mask = ~np.isnan(y_train).flatten()
    X_train = X_train[nan_mask]
    y_train = y_train[nan_mask]
  
    if len(X_train) < 2:
        return df_copy
  
    if kernel_type == 'rbf':
        kernel = ConstantKernel(1.0) * RBF(length_scale=10.0)

    elif kernel_type == 'periodic': 
        kernel = ConstantKernel(1.0) * ExpSineSquared(
            length_scale=1.0,
            periodicity=10.0
        )
  
    else:
        kernel = (
            ConstantKernel(1.0) * RBF(length_scale=10.0) +
            ConstantKernel(0.1) * ExpSineSquared(         
                length_scale=1.0,
                periodicity=10.0
            ) +
            WhiteKernel(noise_level=0.01)                  
        )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10 if optimize_hyperparams else 0,
        alpha=1e-6,
        normalize_y=True
    )
    gp.fit(X_train, y_train)
    X_all = df_copy.index.values.reshape(-1, 1).astype(float)
    y_pred, y_std = gp.predict(X_all, return_std=True)
    df_copy.loc[~valid_mask, col] = y_pred[~valid_mask]
    df_copy[f'{col}_std'] = y_std
  
    return df_copy

def penalized_spline_reconstruction(
    df: pd.DataFrame,
    col: str = 'acc_x',
    lambda_smooth: float = 0.1
) -> pd.DataFrame:
    df_copy = df.copy()
    valid_mask = ~df_copy[col].isna()
    x_valid = df_copy.index[valid_mask].values.astype(float)
    y_valid = df_copy.loc[valid_mask, col].values
    if len(x_valid) < 2:
        return df_copy
    cs = CubicSpline(x_valid, y_valid, bc_type='natural')
    x_all = df_copy.index.values.astype(float)
    y_interp = cs(x_all)

    if lambda_smooth > 0:
        from scipy.signal import savgol_filter
        window_length = min(51, len(y_interp) if len(y_interp) % 2 == 1 else len(y_interp) - 1)
        if window_length >= 3:
            y_interp = savgol_filter(y_interp, window_length, polyorder=3)
  
    df_copy.loc[~valid_mask, col] = y_interp[~valid_mask]
    return df_copy
