import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ExpSineSquared, ConstantKernel
)
from typing import Tuple, Dict, Optional, Callable
import warnings

class EMAlgorithm:
    def __init__(
        self,
        model_type: str = 'gpr_fixed',
        max_iterations: int = 15,
        convergence_threshold: float = 0.001,
        verbose: bool = False
    ):
        self.model_type = model_type
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        self.model = None
        self.iterations_run = 0
        self.convergence_history = []

    def _init_missing_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        missing_mask: np.ndarray
    ) -> np.ndarray:
        
        y_imputed = y.copy()
        for i in np.where(missing_mask)[0]:
            valid_indices = np.where(~missing_mask)[0]
            if len(valid_indices) == 0:
                y_imputed[i] = np.nanmean(y[~missing_mask])
            else:
                distances = np.abs(valid_indices - i)
                closest_idx = valid_indices[np.argsort(distances)[:3]]
                y_imputed[i] = np.mean(y[closest_idx])
      
        return y_imputed
  
    def _fit_gpr_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> GaussianProcessRegressor:
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
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_train, y_train)
        return gp
  
    def fit_and_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        missing_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_reshaped = X.reshape(-1, 1).astype(float)
        n_samples = len(X)
        y_imputed = self._init_missing_values(X, y, missing_mask)
        if self.verbose:
            print(f"Starting EM algorithm with {missing_mask.sum()} missing values")
        self.iterations_run = 0
        self.convergence_history = []
        r2_prev = -np.inf
        for iteration in range(self.max_iterations):
            if iteration == 0:
                X_train = X_reshaped
                y_train = y_imputed
            else:
                X_train = X_reshaped
                y_train = y_imputed
            self.model = self._fit_gpr_model(X_train, y_train)
            y_pred, y_std = self.model.predict(X_reshaped, return_std=True)
            y_imputed[missing_mask] = y_pred[missing_mask]
            if iteration > 0:
                r2_current = self.model.score(X_train, y_imputed)
                delta_r2 = abs(r2_current - r2_prev)
                self.convergence_history.append({
                    'iteration': iteration,
                    'r2': r2_current,
                    'delta_r2': delta_r2
                })
                if self.verbose:
                    print(f"  Iteration {iteration}: R² = {r2_current:.6f}, "
                          f"Δ R² = {delta_r2:.6f}")
                if delta_r2 < self.convergence_threshold:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break
                r2_prev = r2_current
            else:
                r2_prev = self.model.score(X_train, y_imputed)
                if self.verbose:
                    print(f"  Iteration {iteration}: R² = {r2_prev:.6f}")

            self.iterations_run = iteration + 1
        return y_imputed, y_std

def em_algorithm(
    df: pd.DataFrame,
    col: str = 'acc_x',
    max_iterations: int = 15,
    convergence_threshold: float = 0.001,
    model_type: str = 'gpr_fixed',
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
        em = EMAlgorithm(
            model_type=model_type,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            verbose=verbose
        )

        y_imputed, y_std = em.fit_and_predict(X, y, missing_mask)
        df_copy[column] = y_imputed
        df_copy[f'{column}_std'] = y_std
        if verbose:
            print(f"EM reconstruction complete for {column}: "
                  f"{em.iterations_run} iterations, "
                  f"{missing_mask.sum()} values filled")
    metadata = {
        'method': 'em_algorithm',
        'columns': col,
        'max_iterations': max_iterations,
        'convergence_threshold': convergence_threshold,
        'model_type': model_type
    }
  
    return df_copy, metadata

def compare_em_configurations(
    df: pd.DataFrame,
    col: str = 'acc_x',
    max_iterations_list: Optional[list] = None,
    verbose: bool = True
) -> pd.DataFrame:
    if max_iterations_list is None:
        max_iterations_list = [5, 10, 15, 20]
    results = []
    for max_iter in max_iterations_list:
        try:
            _, metadata = em_algorithm(
                df, col,
                max_iterations=max_iter,
                verbose=False
            )
          
            results.append({
                'Max Iterations': max_iter,
                'Status': 'Success'
            })
        except Exception as e:
            results.append({
                'Max Iterations': max_iter,
                'Status': f'Failed: {str(e)}'
            })
    return pd.DataFrame(results)

if __name__ == '__main__':
    print("EM Algorithm Module - Ready for import")
    print("Use: from reconstruction_methods.em_algorithm import em_algorithm")
