import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy.linalg import block_diag

class ExtendedKalmanFilter:
    def __init__(
        self,
        n_axes: int = 3,
        process_noise_scale: float = 0.01,
        measurement_noise_scale: float = 0.01,
        verbose: bool = False
    ):
        self.n_axes = n_axes
        self.n_state = 2 * n_axes
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.verbose = verbose
        self.is_fitted = False
        self.x_hat = None
        self.P = None
        self.F = None
        self.H = None
        self.Q = None
        self.R = None
        self.dt = 1.0

    def _init_system_model(self, y_init: np.ndarray):
        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, 1)
        self.F = np.eye(self.n_state, dtype=np.float64)
        for i in range(self.n_axes):
            self.F[self.n_axes + i, i] = 0.1 * self.dt
        self.H = np.zeros((self.n_axes, self.n_state), dtype=np.float64)
        for i in range(self.n_axes):
            self.H[i, i] = 1.0
        q_accel = np.var(y_init) * self.process_noise_scale
        q_vel = q_accel * 0.1
      
        Q_diag = np.concatenate([
            np.full(self.n_axes, q_accel),
            np.full(self.n_axes, q_vel)   
        ])
        self.Q = np.diag(Q_diag)
      
      
        r = np.var(y_init) * self.measurement_noise_scale
        self.R = r * np.eye(self.n_axes)
      
        if self.verbose:
            print(f"EKF initialized: state_dim={self.n_state}, "
                  f"Q_accel={q_accel:.6f}, R={r:.6f}")
  
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        times: Optional[np.ndarray] = None
    ):
      
        if y.ndim == 1:
            y = y.reshape(-1, 1)
      
      
        init_samples = min(100, len(y))
        y_init = y[:init_samples]
      
      
        self._init_system_model(y_init)
      
      
        self.x_hat = np.zeros(self.n_state)
        self.x_hat[:self.n_axes] = y[0]
      
      
        self.P = np.eye(self.n_state) * np.var(y_init)
      
        self.is_fitted = True
      
        if self.verbose:
            print(f"EKF fit complete: initialized on {init_samples} clean samples")
  
    def predict(
        self,
        X_indices: np.ndarray,
        y_observed: np.ndarray,
        missing_mask: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("EKF must be fit before prediction")
        if y_observed.ndim == 1:
            y_observed = y_observed.reshape(-1, 1)
        n_samples = len(X_indices)
        y_reconstructed = np.zeros_like(y_observed)
        y_std = np.zeros((n_samples, self.n_axes))
        x_hat = self.x_hat.copy()
        P = self.P.copy()
      
        for t in range(n_samples):
            x_hat_pred = self.F @ x_hat
            P_pred = self.F @ P @ self.F.T + self.Q
            if not missing_mask[t]:
                z = y_observed[t]
                S = self.H @ P_pred @ self.H.T + self.R
                K = P_pred @ self.H.T @ np.linalg.inv(S)
                y_innovation = z - (self.H @ x_hat_pred)
                x_hat = x_hat_pred + K @ y_innovation
                P = (np.eye(self.n_state) - K @ self.H) @ P_pred
            else:
                x_hat = x_hat_pred
                P = P_pred
            y_reconstructed[t] = x_hat[:self.n_axes]
            y_std[t] = np.sqrt(np.diag(P)[:self.n_axes])
        return y_reconstructed, y_std

def extended_kalman_filter(
    df: pd.DataFrame,
    col: str = 'acc_x',
    process_noise_scale: float = 0.01,
    measurement_noise_scale: float = 0.01,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:
  
    df_copy = df.copy()
  
  
    if isinstance(col, str):
        col = [col]
  
  
    for column in col:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
      
      
        valid_mask = ~df_copy[column].isna()
        X = df_copy.index.values.astype(float)
        y = df_copy[column].values
        missing_mask = ~valid_mask
      
        if valid_mask.sum() < 3:
            raise ValueError(f"Insufficient training data for {column}")
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        ekf = ExtendedKalmanFilter(
            n_axes=1,
            process_noise_scale=process_noise_scale,
            measurement_noise_scale=measurement_noise_scale,
            verbose=verbose
        )
        ekf.fit(X_clean, y_clean.reshape(-1, 1))
        y_recon, y_std = ekf.predict(
            X,
            y.reshape(-1, 1),
            missing_mask
        )
        df_copy[column] = y_recon.flatten()
        df_copy[f'{column}_std'] = y_std.flatten()
      
        if verbose:
            print(f"EKF reconstruction complete for {column}: "
                  f"{missing_mask.sum()} values filled")
  
    metadata = {
        'method': 'extended_kalman_filter',
        'columns': col,
        'process_noise_scale': process_noise_scale,
        'measurement_noise_scale': measurement_noise_scale
    }
  
    return df_copy, metadata

def tune_ekf_parameters(
    df: pd.DataFrame,
    col: str = 'acc_x',
    process_noise_scales: Optional[list] = None,
    measurement_noise_scales: Optional[list] = None,
    verbose: bool = True
) -> pd.DataFrame:
  
    if process_noise_scales is None:
        process_noise_scales = [0.001, 0.01, 0.1]
    if measurement_noise_scales is None:
        measurement_noise_scales = [0.001, 0.01, 0.1]
  
    results = []
  
    for q_scale in process_noise_scales:
        for r_scale in measurement_noise_scales:
            try:
                _, metadata = extended_kalman_filter(
                    df, col,
                    process_noise_scale=q_scale,
                    measurement_noise_scale=r_scale,
                    verbose=False
                )
              
                results.append({
                    'Process Noise': q_scale,
                    'Measurement Noise': r_scale,
                    'Status': 'Success'
                })
            except Exception as e:
                results.append({
                    'Process Noise': q_scale,
                    'Measurement Noise': r_scale,
                    'Status': f'Failed: {str(e)}'
                })
  
    return pd.DataFrame(results)

if __name__ == '__main__':
    print("Extended Kalman Filter Module - Ready for import")
    print("Use: from reconstruction_methods.extended_kalman_filter import extended_kalman_filter")
