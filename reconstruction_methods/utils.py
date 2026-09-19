import numpy as np
import pandas as pd
from scipy.signal import hilbert
from typing import Tuple, Dict, Optional, Callable

def inject_missing_data(
    df: pd.DataFrame,
    col: str,
    missing_fraction: float = 0.1,
    gap_size: int = 50,
    n_gaps: int = 10,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df_degraded = df.copy()
    n_samples = len(df_degraded)
    actual_gap_size = int(np.floor(n_samples * missing_fraction))
    missing_mask = np.zeros(n_samples, dtype=bool)

    if actual_gap_size <= 0:
        return df_degraded, missing_mask
    
    if actual_gap_size >= n_samples:
        df_degraded[col] = np.nan
        missing_mask[:] = True
        return df_degraded, missing_mask
    
    max_start = n_samples - actual_gap_size
    start = rng.integers(0, max_start + 1)
    end = start + actual_gap_size
    missing_mask[start:end] = True
    df_degraded.iloc[start:end, df_degraded.columns.get_loc(col)] = np.nan

    return df_degraded, missing_mask

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    mask_eval: Optional[np.ndarray] = None
) -> Dict[str, float]:
    if mask_eval is None:
        mask_eval = np.ones(len(y_true), dtype=bool)
    y_true_eval = y_true[mask_eval]
    y_pred_eval = y_pred[mask_eval]
    
    if len(y_true_eval) == 0:
        raise ValueError("No evaluation samples provided")
    
    rmse = np.sqrt(np.mean((y_true_eval - y_pred_eval)**2))
    mae = np.mean(np.abs(y_true_eval - y_pred_eval))
    ss_res = np.sum((y_true_eval - y_pred_eval)**2)
    ss_tot = np.sum((y_true_eval - np.mean(y_true_eval))**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    plv = compute_plv(y_true_eval, y_pred_eval)
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'plv': plv,
        'n_samples': len(y_true_eval)
    }

    if y_std is not None:
        y_std_eval = y_std[mask_eval]
      
        residuals = np.abs(y_true_eval - y_pred_eval)
      
        within_1std = np.mean(residuals <= y_std_eval)
        within_2std = np.mean(residuals <= 2 * y_std_eval)
      
        metrics['within_1std'] = within_1std
        metrics['within_2std'] = within_2std
        metrics['mean_std'] = np.mean(y_std_eval)
    return metrics

def compute_plv(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    z_true = hilbert(y_true - np.mean(y_true))
    z_pred = hilbert(y_pred - np.mean(y_pred))
    phase_true = np.angle(z_true)
    phase_pred = np.angle(z_pred)
    delta_phase = phase_true - phase_pred
    plv = np.abs(np.mean(np.exp(1j * delta_phase)))

    return float(plv)

def evaluate_method(
    reconstruction_func: Callable,
    df_original: pd.DataFrame,
    col: str,
    missing_fractions: list = [0.05, 0.10, 0.15, 0.20],
    gap_size: int = 50,
    n_replicates: int = 10,
    method_name: str = "Method",
    verbose: bool = True,
    **method_kwargs
) -> pd.DataFrame:
    results = []
    for missing_frac in missing_fractions:
        for replicate in range(n_replicates):
            try:
                n_gaps = max(1, int(len(df_original) * missing_frac / gap_size))
                df_degraded, missing_mask = inject_missing_data(
                    df_original, col,
                    missing_fraction=missing_frac,
                    gap_size=gap_size,
                    n_gaps=n_gaps,
                    seed=replicate
                )
                df_recon, metadata = reconstruction_func(df_degraded, col, **method_kwargs)
                y_true = df_original[col].values
                y_pred = df_recon[col].values
                y_std = df_recon.get(f'{col}_std', None)
                metrics = compute_metrics(y_true, y_pred, y_std, missing_mask)
                result = {
                    'Method': method_name,
                    'Missing_Fraction': missing_frac,
                    'Replicate': replicate,
                    'N_Missing': missing_mask.sum(),
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'R2': metrics['r2'],
                    'PLV': metrics['plv'],
                    'Status': 'Success'
                }
                if 'within_1std' in metrics:
                    result['Within_1Std'] = metrics['within_1std']
                    result['Within_2Std'] = metrics['within_2std']
                results.append(result)
                if verbose and (replicate == 0 or replicate == n_replicates - 1):
                    print(f"  {method_name} @ {missing_frac*100:.0f}% "
                          f"(replicate {replicate}): R²={metrics['r2']:.4f}")
            except Exception as e:
                result = {
                    'Method': method_name,
                    'Missing_Fraction': missing_frac,
                    'Replicate': replicate,
                    'Status': f'Failed: {str(e)}'
                }
                results.append(result)
                if verbose:
                    print(f"  {method_name} @ {missing_frac*100:.0f}% "
                          f"(replicate {replicate}): Failed - {str(e)}")
  
    return pd.DataFrame(results)

def summarize_evaluation_results(
    results_df: pd.DataFrame,
    metrics: list = ['R2', 'RMSE', 'MAE', 'PLV']
) -> pd.DataFrame:
    summary_rows = []
    for method in results_df['Method'].unique():
        method_df = results_df[results_df['Method'] == method]
        for missing_frac in sorted(method_df['Missing_Fraction'].unique()):
            cond_df = method_df[method_df['Missing_Fraction'] == missing_frac]
            cond_df = cond_df[cond_df['Status'] == 'Success']
            if len(cond_df) == 0:
                continue
            summary_row = {
                'Method': method,
                'Missing_Fraction': missing_frac,
                'N_Replicates': len(cond_df)
            }
            for metric in metrics:
                if metric in cond_df.columns:
                    summary_row[f'{metric}_Mean'] = cond_df[metric].mean()
                    summary_row[f'{metric}_Std'] = cond_df[metric].std()
          
            summary_rows.append(summary_row)
    return pd.DataFrame(summary_rows)

if __name__ == '__main__':
    print("Utilities Module - Ready for import")
    print("Functions: inject_missing_data, compute_metrics, evaluate_method, summarize_evaluation_results")
