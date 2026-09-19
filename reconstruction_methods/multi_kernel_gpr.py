import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ExpSineSquared, ConstantKernel, Matern
)
from typing import Tuple, Optional, Dict

def multi_kernel_gpr(
    df: pd.DataFrame,
    col: str = 'acc_x',
    kernel_combination: str = 'matern',
    optimize_hyperparams: bool = True,
    max_train_samples: int = 1000,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    df_copy = df.copy()
    valid_mask = ~df_copy[col].isna()
    X_train = df_copy.index[valid_mask].values.reshape(-1, 1).astype(float)
    y_train = df_copy.loc[valid_mask, col].values
    if len(X_train) < 3:
        raise ValueError(f"Insufficient training data: {len(X_train)} < 3 points")
    if len(X_train) > max_train_samples:
        indices = np.linspace(0, len(X_train) - 1, max_train_samples, dtype=int)
        indices = np.unique(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
    nan_mask = ~np.isnan(y_train).flatten()
    X_train = X_train[nan_mask]
    y_train = y_train[nan_mask]
    if len(X_train) < 3:
        raise ValueError(f"Insufficient valid training data after cleanup")
    kernels = _get_kernel_combinations()
    if kernel_combination == 'all':
        kernel_names = list(kernels.keys())
    elif kernel_combination in kernels:
        kernel_names = [kernel_combination]
    else:
        kernel_names = ['matern']
    if verbose:
        print(f"Evaluating {len(kernel_names)} kernel combination(s): {kernel_names}")
    best_gp = None
    best_kernel_name = None
    best_log_marginal_likelihood = -np.inf
    gp_results = {}
    for kernel_name in kernel_names:
        kernel = kernels[kernel_name]
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10 if optimize_hyperparams else 0,
            alpha=1e-6,
            normalize_y=True,
            n_targets=1
        )
        try:
            gp.fit(X_train, y_train)
            log_marginal_likelihood = gp.log_marginal_likelihood_value_
            gp_results[kernel_name] = {
                'gp': gp,
                'log_marginal_likelihood': log_marginal_likelihood,
                'kernel': kernel_name
            }
            if verbose:
                print(f"  {kernel_name}: LML = {log_marginal_likelihood:.4f}")
            if log_marginal_likelihood > best_log_marginal_likelihood:
                best_log_marginal_likelihood = log_marginal_likelihood
                best_gp = gp
                best_kernel_name = kernel_name
        except Exception as e:
            if verbose:
                print(f"  {kernel_name}: Failed to fit - {str(e)}")
            continue
    if best_gp is None:
        raise RuntimeError("Failed to fit any kernel combination")
    if verbose:
        print(f"Selected kernel: {best_kernel_name}")
    X_all = df_copy.index.values.reshape(-1, 1).astype(float)
    y_pred, y_std = best_gp.predict(X_all, return_std=True)
    df_copy.loc[~valid_mask, col] = y_pred[~valid_mask]
    df_copy[f'{col}_std'] = y_std

    metadata = {
        'method': 'multi_kernel_gpr',
        'kernel_combination': best_kernel_name,
        'log_marginal_likelihood': best_log_marginal_likelihood,
        'all_results': gp_results,
        'n_training_samples': len(X_train),
        'n_missing_samples': (~valid_mask).sum()
    }
    return df_copy, metadata

def _get_kernel_combinations() -> Dict:
    baseline = (
        ConstantKernel(1.0) * RBF(length_scale=10.0) +
        ConstantKernel(0.1) * ExpSineSquared(          
            length_scale=1.0,
            periodicity=10.0
        ) +
        WhiteKernel(noise_level=0.01)                  
    )
    matern = (
        ConstantKernel(1.0) * RBF(length_scale=10.0) +         
        ConstantKernel(0.1) * ExpSineSquared(                   
            length_scale=1.0,
            periodicity=10.0
        ) +
        ConstantKernel(0.05) * Matern(length_scale=5.0, nu=2.5)
        + WhiteKernel(noise_level=0.01)                         
    )
    squared_exp = (
        ConstantKernel(1.0) * RBF(length_scale=10.0) +         
        ConstantKernel(0.1) * ExpSineSquared(                   
            length_scale=1.0,
            periodicity=10.0
        ) +
        ConstantKernel(0.05) * RBF(length_scale=5.0)           
        + WhiteKernel(noise_level=0.01)                         
    )
    return {
        'baseline': baseline,
        'matern': matern,
        'squared_exp': squared_exp
    }

def compare_kernel_combinations(
    df: pd.DataFrame,
    col: str = 'acc_x',
    max_train_samples: int = 1000,
    verbose: bool = True
) -> pd.DataFrame:
    df_result, metadata = multi_kernel_gpr(
        df, col, kernel_combination='all',
        max_train_samples=max_train_samples,
        verbose=verbose
    )
    results_list = []
    for kernel_name, result_data in metadata['all_results'].items():
        results_list.append({
            'Kernel': kernel_name,
            'Log Marginal Likelihood': result_data['log_marginal_likelihood'],
            'Status': 'Selected' if kernel_name == metadata['kernel_combination'] else 'Evaluated'
        })
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.sort_values('Log Marginal Likelihood', ascending=False)

    return comparison_df

if __name__ == '__main__':
    print("Multi-Kernel GPR Module - Ready for import")
    print("Use: from reconstruction_methods.multi_kernel_gpr import multi_kernel_gpr")
