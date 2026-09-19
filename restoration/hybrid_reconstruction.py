import numpy as np
import pandas as pd
from scipy import interpolate, signal
from scipy.spatial import KDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.linear_model import Ridge
from typing import Tuple, Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')

def estimate_embedding_params(signal: np.ndarray, fs: float = 35.0) -> Tuple[int, int]:
    n = len(signal)
    centered = signal - np.mean(signal)
    autocorr = np.correlate(centered[:min(500, n)], centered[:min(500, n)], mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)
    zero_crossings = np.where(autocorr < 0)[0]
    if len(zero_crossings) > 0:
        delay = max(1, zero_crossings[0])
    else:
        delay = int(fs / 4)
    delay = min(delay, 20)
    embed_dim = 6
  
    return embed_dim, delay

def create_embedding(signal: np.ndarray, dim: int, delay: int) -> np.ndarray:
  
    n = len(signal)
    n_vectors = n - (dim - 1) * delay
    if n_vectors <= 0:
        raise ValueError(f"Signal too short for embedding: {n} samples, need > {(dim-1)*delay}")
  
    embedding = np.zeros((n_vectors, dim))
    for i in range(dim):
        start = (dim - 1 - i) * delay
        end = start + n_vectors
        embedding[:, i] = signal[start:end]
  
    return embedding

def takens_local_linear_predict(
    signal: np.ndarray,
    gap_start: int,
    gap_end: int,
    n_neighbors: int = 10,
    embed_dim: int = None,
    delay: int = None
) -> np.ndarray:
    n = len(signal)
    gap_length = gap_end - gap_start
  
    if embed_dim is None or delay is None:
        embed_dim, delay = estimate_embedding_params(signal[:gap_start])
  
    min_samples = (embed_dim - 1) * delay + n_neighbors + 10

    if gap_start < min_samples or (n - gap_end) < min_samples:
        return _simple_boundary_interpolation(signal, gap_start, gap_end)
  
    pre_signal = signal[:gap_start]
    post_signal = signal[gap_end:]
    pre_embedding = create_embedding(pre_signal, embed_dim, delay)
    post_embedding = create_embedding(post_signal, embed_dim, delay)

    if len(pre_embedding) < n_neighbors or len(post_embedding) < n_neighbors:
        return _simple_boundary_interpolation(signal, gap_start, gap_end)
  
    pre_tree = KDTree(pre_embedding[:-1])
    post_tree = KDTree(post_embedding[1:])
    pre_boundary = pre_embedding[-1]
    post_boundary = post_embedding[0]
    _, pre_neighbor_idx = pre_tree.query(pre_boundary, k=n_neighbors)
    pre_neighbor_idx = np.atleast_1d(pre_neighbor_idx)
    _, post_neighbor_idx = post_tree.query(post_boundary, k=n_neighbors)
    post_neighbor_idx = np.atleast_1d(post_neighbor_idx)
    post_neighbor_idx = post_neighbor_idx + 1
    forward_preds = []
    for idx in pre_neighbor_idx:
        if idx + gap_length < len(pre_signal):
            traj = pre_signal[idx:idx + gap_length]
            if len(traj) == gap_length:
                forward_preds.append(traj)
    backward_preds = []
    for idx in post_neighbor_idx:
        start_idx = max(0, idx - gap_length)
        traj = post_signal[start_idx:idx]
        if len(traj) == gap_length:
            backward_preds.append(traj[::-1][::-1])

    if len(forward_preds) == 0 and len(backward_preds) == 0:
        return _simple_boundary_interpolation(signal, gap_start, gap_end)
  
    if len(forward_preds) > 0:
        forward_mean = np.mean(forward_preds, axis=0)
    else:
        forward_mean = None
    if len(backward_preds) > 0:
        backward_mean = np.mean(backward_preds, axis=0)
    else:
        backward_mean = None
    reconstruction = np.zeros(gap_length)
    blend_weight = np.linspace(1, 0, gap_length)
    if forward_mean is not None and backward_mean is not None:
        reconstruction = blend_weight * forward_mean + (1 - blend_weight) * backward_mean
    elif forward_mean is not None:
        reconstruction = forward_mean
    else:
        reconstruction = backward_mean
    reconstruction = _ensure_boundary_continuity(
        reconstruction, signal[gap_start-1], signal[gap_end]
    )
  
    return reconstruction

def _simple_boundary_interpolation(signal: np.ndarray, gap_start: int, gap_end: int) -> np.ndarray:
    gap_length = gap_end - gap_start
    left_val = signal[gap_start - 1] if gap_start > 0 else 0
    right_val = signal[gap_end] if gap_end < len(signal) else 0
    if gap_start > 1:
        left_slope = signal[gap_start - 1] - signal[gap_start - 2]
    else:
        left_slope = 0
  
    if gap_end < len(signal) - 1:
        right_slope = signal[gap_end + 1] - signal[gap_end]
    else:
        right_slope = 0
    t = np.linspace(0, 1, gap_length)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    reconstruction = (h00 * left_val + h10 * gap_length * left_slope + 
                     h01 * right_val + h11 * gap_length * right_slope)
    return reconstruction

def _ensure_boundary_continuity(
    reconstruction: np.ndarray,
    left_boundary: float,
    right_boundary: float,
    blend_samples: int = 10
) -> np.ndarray:
    n = len(reconstruction)
    blend_samples = min(blend_samples, n // 4)
    if blend_samples < 2:
        return reconstruction
    result = reconstruction.copy()
    left_diff = left_boundary - reconstruction[0]
    left_blend = np.linspace(1, 0, blend_samples)
    result[:blend_samples] += left_diff * left_blend
    right_diff = right_boundary - reconstruction[-1]
    right_blend = np.linspace(0, 1, blend_samples)
    result[-blend_samples:] += right_diff * right_blend
    return result

def bidirectional_gpr_reconstruct(
    signal: np.ndarray,
    gap_start: int,
    gap_end: int,
    context_samples: int = 200,
    max_train_samples: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
    gap_length = gap_end - gap_start
    pre_start = max(0, gap_start - context_samples)
    post_end = min(len(signal), gap_end + context_samples)
    pre_signal = signal[pre_start:gap_start]
    post_signal = signal[gap_end:post_end]
  
    pre_x = np.arange(len(pre_signal)).reshape(-1, 1)
    pre_y = pre_signal
    if len(pre_x) > max_train_samples:
        idx = np.linspace(0, len(pre_x)-1, max_train_samples, dtype=int)
        pre_x = pre_x[idx]
        pre_y = pre_y[idx]
  
    kernel_forward = (
        ConstantKernel(1.0, (0.1, 10.0)) * 
        Matern(length_scale=20.0, length_scale_bounds=(5.0, 100.0), nu=2.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(0.01, 1.0))
    )
    gpr_forward = GaussianProcessRegressor(
        kernel=kernel_forward,
        n_restarts_optimizer=3,
        normalize_y=True
    )
  
    try:
        gpr_forward.fit(pre_x, pre_y)
        gap_x_forward = np.arange(len(pre_signal), len(pre_signal) + gap_length).reshape(-1, 1)
        forward_pred, forward_std = gpr_forward.predict(gap_x_forward, return_std=True)
    except Exception:
        forward_pred = np.full(gap_length, np.mean(pre_signal))
        forward_std = np.ones(gap_length) * np.std(pre_signal)

    post_x = np.arange(len(post_signal)).reshape(-1, 1)
    post_y = post_signal

    if len(post_x) > max_train_samples:
        idx = np.linspace(0, len(post_x)-1, max_train_samples, dtype=int)
        post_x = post_x[idx]
        post_y = post_y[idx]
    kernel_backward = (
        ConstantKernel(1.0, (0.1, 10.0)) * 
        Matern(length_scale=20.0, length_scale_bounds=(5.0, 100.0), nu=2.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(0.01, 1.0))
    )
    gpr_backward = GaussianProcessRegressor(
        kernel=kernel_backward,
        n_restarts_optimizer=3,
        normalize_y=True
    )
    try:
        gpr_backward.fit(post_x, post_y[::-1] if len(post_y) > 0 else post_y)
        gap_x_backward = np.arange(gap_length).reshape(-1, 1)
        backward_pred_rev, backward_std = gpr_backward.predict(gap_x_backward, return_std=True)
        backward_pred = backward_pred_rev[::-1]
        backward_std = backward_std[::-1]
    except Exception:
        backward_pred = np.full(gap_length, np.mean(post_signal) if len(post_signal) > 0 else 0)
        backward_std = np.ones(gap_length) * (np.std(post_signal) if len(post_signal) > 0 else 1)
  
  
  
    forward_weight = 1.0 / (forward_std + 0.01)
    backward_weight = 1.0 / (backward_std + 0.01)
  
    total_weight = forward_weight + backward_weight
    forward_weight /= total_weight
    backward_weight /= total_weight
  
    reconstruction = forward_weight * forward_pred + backward_weight * backward_pred
    uncertainty = 1.0 / (1.0/forward_std + 1.0/backward_std)
  
  
    reconstruction = _ensure_boundary_continuity(
        reconstruction,
        signal[gap_start - 1] if gap_start > 0 else reconstruction[0],
        signal[gap_end] if gap_end < len(signal) else reconstruction[-1]
    )
    return reconstruction, uncertainty

class EchoStateNetwork:
    def __init__(
        self,
        n_reservoir: int = 300,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        input_scaling: float = 0.3,
        leak_rate: float = 0.5,
        regularization: float = 1e-4,
        random_state: int = 42
    ):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.regularization = regularization
        self.random_state = random_state
      
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.last_state = None
        self.input_mean = 0
        self.input_std = 1
      
    def _init_reservoir(self, n_inputs: int):
      
        rng = np.random.RandomState(self.random_state)
        self.W_in = rng.uniform(-1, 1, (self.n_reservoir, n_inputs)) * self.input_scaling
        W_res = rng.uniform(-1, 1, (self.n_reservoir, self.n_reservoir))
        mask = rng.random((self.n_reservoir, self.n_reservoir)) < self.sparsity
        W_res *= mask
        eigenvalues = np.linalg.eigvals(W_res)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W_res *= self.spectral_radius / current_radius
        self.W_res = W_res
      
    def _update_state(self, state: np.ndarray, input_val: np.ndarray) -> np.ndarray:
        pre_activation = np.dot(self.W_in, input_val) + np.dot(self.W_res, state)
        pre_activation = np.clip(pre_activation, -10, 10)
        new_state = (1 - self.leak_rate) * state + self.leak_rate * np.tanh(pre_activation)
        return new_state
  
    def fit(self, X: np.ndarray, y: np.ndarray, w: int = 100):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.input_mean = np.mean(X)
        self.input_std = np.std(X) + 1e-8
        X = (X - self.input_mean) / self.input_std
        y = (y - self.input_mean) / self.input_std
        n_samples, n_inputs = X.shape
        if self.W_in is None:
            self._init_reservoir(n_inputs)
        states = np.zeros((n_samples, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t in range(n_samples):
            state = self._update_state(state, X[t])
            states[t] = state
        w = min(w, n_samples // 2)
        states = states[w:]
        y_train = y[w:]
        extended_states = np.hstack([states, X[w:]])
        ridge = Ridge(alpha=self.regularization, fit_intercept=True)
        ridge.fit(extended_states, y_train)
        self.W_out = ridge.coef_
        self.bias = ridge.intercept_
        self.n_inputs = n_inputs
        self.last_state = state.copy()
        return self
  
    def predict(self, X: np.ndarray, initial_state: np.ndarray = None) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = (X - self.input_mean) / self.input_std
        n_samples = X.shape[0]
        state = initial_state if initial_state is not None else self.last_state
        if state is None:
            state = np.zeros(self.n_reservoir)
        outputs = []
        for t in range(n_samples):
            state = self._update_state(state, X[t])
            extended = np.concatenate([state, X[t]])
            output = np.dot(self.W_out, extended) + self.bias
            outputs.append(output)
        self.last_state = state.copy()
        outputs = np.array(outputs).squeeze() * self.input_std + self.input_mean
        return outputs
  
    def generate(self, n_steps: int, seed_input: np.ndarray = None, 
                 value_bounds: Tuple[float, float] = None) -> np.ndarray:
        if seed_input is not None:
            _ = self.predict(seed_input)
        state = self.last_state if self.last_state is not None else np.zeros(self.n_reservoir)
        if value_bounds is None:
            value_bounds = (self.input_mean - 4*self.input_std, 
                          self.input_mean + 4*self.input_std)
        outputs = []
        last_output_normalized = np.zeros(self.n_inputs)
      
        for _ in range(n_steps):
            state = self._update_state(state, last_output_normalized)
            extended = np.concatenate([state, last_output_normalized])
            output_normalized = np.dot(self.W_out, extended) + self.bias
            output_normalized = np.clip(output_normalized, -4, 4)
            output = np.atleast_1d(output_normalized).squeeze() * self.input_std + self.input_mean
            output = np.clip(output, value_bounds[0], value_bounds[1])
            outputs.append(output)
            last_output_normalized = (np.atleast_1d(output)[:self.n_inputs] - self.input_mean) / self.input_std
      
        self.last_state = state.copy()
        return np.array(outputs).squeeze()

def esn_reconstruct(
    signal: np.ndarray,
    gap_start: int,
    gap_end: int,
    context_samples: int = 500,
    n_reservoir: int = 500
) -> np.ndarray:
    gap_length = gap_end - gap_start
    pre_start = max(0, gap_start - context_samples)
    post_end = min(len(signal), gap_end + context_samples)
    pre_signal = signal[pre_start:gap_start]
    post_signal = signal[gap_end:post_end]
    if len(pre_signal) < 100 or len(post_signal) < 100:
        return _simple_boundary_interpolation(signal, gap_start, gap_end)
    esn_forward = EchoStateNetwork(n_reservoir=n_reservoir, random_state=42)
    X_forward = pre_signal[:-1].reshape(-1, 1)
    y_forward = pre_signal[1:]
    try:
        esn_forward.fit(X_forward, y_forward, washout=50)
      
        seed = pre_signal[-50:].reshape(-1, 1)
        _ = esn_forward.predict(seed)
        forward_pred = esn_forward.generate(gap_length)
    except Exception:
        forward_pred = np.full(gap_length, np.mean(pre_signal))
    esn_backward = EchoStateNetwork(n_reservoir=n_reservoir, random_state=43)
    post_reversed = post_signal[::-1]
    X_backward = post_reversed[:-1].reshape(-1, 1)
    y_backward = post_reversed[1:]
    try:
        esn_backward.fit(X_backward, y_backward, washout=50)
        seed = post_reversed[-50:].reshape(-1, 1)
        _ = esn_backward.predict(seed)
        backward_pred_rev = esn_backward.generate(gap_length)
        backward_pred = backward_pred_rev[::-1]
    except Exception:
        backward_pred = np.full(gap_length, np.mean(post_signal))
    blend_weight = np.linspace(1, 0, gap_length)
    reconstruction = blend_weight * forward_pred + (1 - blend_weight) * backward_pred
    reconstruction = _ensure_boundary_continuity(
        reconstruction,
        signal[gap_start - 1] if gap_start > 0 else reconstruction[0],
        signal[gap_end] if gap_end < len(signal) else reconstruction[-1]
    )
    return reconstruction

def hybrid_reconstruct(
    signal: np.ndarray,
    gap_start: int,
    gap_end: int,
    methods: List[str] = ['takens', 'gpr'],
    weights: Dict[str, float] = None,
    validate_predictions: bool = True
) -> Tuple[np.ndarray, Dict]:
    gap_length = gap_end - gap_start
    predictions = {}
    uncertainties = {}
    observed = np.concatenate([signal[:gap_start], signal[gap_end:]])
    observed = observed[~np.isnan(observed)]
    signal_mean = np.mean(observed)
    signal_std = np.std(observed)
    signal_min = np.min(observed)
    signal_max = np.max(observed)
    valid_min = signal_min - 2 * signal_std
    valid_max = signal_max + 2 * signal_std
    if weights is None:
        weights = {'takens': 0.4, 'gpr': 0.6, 'esn': 0.0}
    if 'takens' in methods and weights.get('takens', 0) > 0:
        try:
            pred = takens_local_linear_predict(signal, gap_start, gap_end)
            if validate_predictions:
              
                if np.all(np.isfinite(pred)) and np.all(pred >= valid_min) and np.all(pred <= valid_max):
                    predictions['takens'] = pred
                    uncertainties['takens'] = np.ones(gap_length) * signal_std
            else:
                predictions['takens'] = pred
                uncertainties['takens'] = np.ones(gap_length) * signal_std
        except Exception as e:
            pass
    if 'gpr' in methods and weights.get('gpr', 0) > 0:
        try:
            pred, uncert = bidirectional_gpr_reconstruct(signal, gap_start, gap_end)
            if validate_predictions:
                if np.all(np.isfinite(pred)):
                  
                    pred = np.clip(pred, valid_min, valid_max)
                    predictions['gpr'] = pred
                    uncertainties['gpr'] = uncert
            else:
                predictions['gpr'] = pred
                uncertainties['gpr'] = uncert
        except Exception as e:
            pass
    if 'esn' in methods and weights.get('esn', 0) > 0:
        try:
            pred = esn_reconstruct(signal, gap_start, gap_end)
            if validate_predictions:
                if np.all(np.isfinite(pred)) and np.all(pred >= valid_min) and np.all(pred <= valid_max):
                    predictions['esn'] = pred
                    uncertainties['esn'] = np.ones(gap_length) * signal_std * 2
            else:
                predictions['esn'] = pred
                uncertainties['esn'] = np.ones(gap_length) * signal_std * 2
        except Exception as e:
            pass
    if len(predictions) == 0:
        return _simple_boundary_interpolation(signal, gap_start, gap_end), {'error': 'All methods failed'}
    total_weight = 0
    reconstruction = np.zeros(gap_length)
    left_boundary = signal[gap_start - 1] if gap_start > 0 else signal_mean
    right_boundary = signal[gap_end] if gap_end < len(signal) else signal_mean
    effective_weights = {}
    for method, pred in predictions.items():
        w = weights.get(method, 1.0 / len(predictions))
        left_error = abs(pred[0] - left_boundary)
        right_error = abs(pred[-1] - right_boundary)
        boundary_penalty = 1.0 / (1.0 + (left_error + right_error) / signal_std)
        if method in uncertainties:
            uncertainty_factor = 1.0 / (np.mean(uncertainties[method]) / signal_std + 0.1)
        else:
            uncertainty_factor = 1.0
        effective_w = w * boundary_penalty * uncertainty_factor
        effective_weights[method] = effective_w
        reconstruction += effective_w * pred
        total_weight += effective_w
    reconstruction /= total_weight
    reconstruction = _ensure_boundary_continuity(
        reconstruction,
        left_boundary,
        right_boundary,
        blend_samples=min(30, gap_length // 5)
    )
    reconstruction = np.clip(reconstruction, valid_min, valid_max)
    metadata = {
        'predictions': predictions,
        'uncertainties': uncertainties,
        'base_weights': weights,
        'effective_weights': effective_weights,
        'methods_used': list(predictions.keys()),
        'signal_stats': {'mean': signal_mean, 'std': signal_std, 'min': signal_min, 'max': signal_max}
    }
  
    return reconstruction, metadata

def hybrid_reconstruction(
    df: pd.DataFrame,
    col: str = 'acc_x',
    methods: List[str] = ['takens', 'gpr', 'esn'],
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    df_result = df.copy()
    signal = df_result[col].values.copy()
    is_nan = np.isnan(signal)
    if not np.any(is_nan):
        return df_result
    gap_starts = np.where(np.diff(np.concatenate([[False], is_nan])) == 1)[0]
    gap_ends = np.where(np.diff(np.concatenate([is_nan, [False]])) == -1)[0] + 1
    for gap_start, gap_end in zip(gap_starts, gap_ends):
        reconstruction, _ = hybrid_reconstruct(
            signal, gap_start, gap_end, methods=methods, weights=weights
        )
        signal[gap_start:gap_end] = reconstruction
    df_result[col] = signal
    return df_result

def takens_reconstruction(df: pd.DataFrame, col: str = 'acc_x') -> pd.DataFrame:
    return hybrid_reconstruction(df, col, methods=['takens'])

def bidirectional_gpr_reconstruction(df: pd.DataFrame, col: str = 'acc_x') -> pd.DataFrame:
    return hybrid_reconstruction(df, col, methods=['gpr'])

def esn_reconstruction(df: pd.DataFrame, col: str = 'acc_x') -> pd.DataFrame:
    return hybrid_reconstruction(df, col, methods=['esn'])

if __name__ == '__main__':
    print("Testing Hybrid Reconstruction Methods")
    print("=" * 60)
    np.random.seed(42)
    t = np.linspace(0, 20, 1000)
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 1.3 * t + np.cumsum(np.random.randn(len(t)) * 0.1))
    signal += np.random.randn(len(t)) * 0.1
    gap_start, gap_end = 400, 600
    signal_with_gap = signal.copy()
    signal_with_gap[gap_start:gap_end] = np.nan
    print(f"\nSignal length: {len(signal)}")
    print(f"Gap: {gap_start} to {gap_end} ({gap_end - gap_start} samples)")
  
  
    for method in ['takens', 'gpr', 'esn']:
        print(f"\nTesting {method}...")
        try:
            df = pd.DataFrame({'acc_x': signal_with_gap})
            df_result = hybrid_reconstruction(df, methods=[method])
            reconstructed = df_result['acc_x'].values[gap_start:gap_end]
            true_values = signal[gap_start:gap_end]
            rmse = np.sqrt(np.mean((reconstructed - true_values)**2))
            corr = np.corrcoef(reconstructed, true_values)[0, 1]
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Correlation: {corr:.4f}")
        except Exception as e:
            print(f"  Failed: {e}")

    print("\nTesting hybrid ensemble...")
    df = pd.DataFrame({'acc_x': signal_with_gap})
    df_result = hybrid_reconstruction(df, methods=['takens', 'gpr', 'esn'])
    reconstructed = df_result['acc_x'].values[gap_start:gap_end]
    true_values = signal[gap_start:gap_end]
    rmse = np.sqrt(np.mean((reconstructed - true_values)**2))
    corr = np.corrcoef(reconstructed, true_values)[0, 1]
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Correlation: {corr:.4f}")
    print("\n" + "=" * 60)
    print("Testing complete!")
