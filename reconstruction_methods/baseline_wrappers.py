import pandas as pd
from typing import Tuple, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from restoration.interpolation import gaussian_process_regression, cubic_spline_interpolation

def baseline_gpr_wrapper(
    df: pd.DataFrame,
    col: str = 'acc_x',
    verbose: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    
    
    df_result = gaussian_process_regression(df, col, **kwargs)
    
    metadata = {
        'method': 'baseline_gpr',
        'kernel_type': kwargs.get('kernel_type', 'combined'),
        'max_train_samples': kwargs.get('max_train_samples', 1000)
    }
    
    return df_result, metadata

def cubic_spline_wrapper(
    df: pd.DataFrame,
    col: str = 'acc_x',
    verbose: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    df_result = cubic_spline_interpolation(df, col, **kwargs)
    
    metadata = {
        'method': 'cubic_spline',
        'adaptive': kwargs.get('adaptive', True),
        'bc_type': kwargs.get('bc_type', 'natural')
    }
    
    return df_result, metadata

if __name__ == '__main__':
    print("Baseline Wrappers Module - Ready for import")
    print("Use: from reconstruction_methods.baseline_wrappers import baseline_gpr_wrapper")
