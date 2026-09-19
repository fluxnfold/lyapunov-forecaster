from .multi_kernel_gpr import multi_kernel_gpr, compare_kernel_combinations
from .extended_kalman_filter import extended_kalman_filter, tune_ekf_parameters
from .particle_filter import particle_filter, tune_particle_count
from .em_algorithm import em_algorithm, compare_em_configurations
from .utils import (
    inject_missing_data,
    compute_metrics,
    evaluate_method,
    summarize_evaluation_results
)
from .baseline_wrappers import baseline_gpr_wrapper, cubic_spline_wrapper

__all__ = [
    
    'multi_kernel_gpr',
    'compare_kernel_combinations',
    
    
    'extended_kalman_filter',
    'tune_ekf_parameters',
    
    
    'particle_filter',
    'tune_particle_count',
    
    
    'em_algorithm',
    'compare_em_configurations',
    
    
    'inject_missing_data',
    'compute_metrics',
    'evaluate_method',
    'summarize_evaluation_results',
    
    
    'baseline_gpr_wrapper',
    'cubic_spline_wrapper'
]

__version__ = '0.1.0'
__author__ = 'Marius Scheffel'
