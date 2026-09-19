import numpy as np
import pandas as pd
from scipy import integrate, optimize, interpolate
from scipy.signal import savgol_filter
from typing import Tuple, Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')

class DoublePendulumPhysics:
  
  
    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 0.3,
        l2: float = 0.3,
        g: float = 9.81 
    ):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
  
    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        theta1, theta2, omega1, omega2 = state
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        delta = theta1 - theta2
        den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta)**2
        den2 = (l2 / l1) * den1
        alpha1 = (m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                  m2 * g * np.sin(theta2) * np.cos(delta) +
                  m2 * l2 * omega2**2 * np.sin(delta) -
                  (m1 + m2) * g * np.sin(theta1)) / den1
        
        alpha2 = (-m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                  (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                  (m1 + m2) * l1 * omega1**2 * np.sin(delta) -
                  (m1 + m2) * g * np.sin(theta2)) / den2
        return np.array([omega1, omega2, alpha1, alpha2])
  
    def state_to_acceleration(self, state: np.ndarray) -> Tuple[float, float]:
        theta1, theta2, omega1, omega2 = state
        derivs = self.equations_of_motion(0, state)
        alpha1, alpha2 = derivs[2], derivs[3]
        a_x = (self.l1 * (alpha1 * np.cos(theta1) - omega1**2 * np.sin(theta1)) +
               self.l2 * (alpha2 * np.cos(theta2) - omega2**2 * np.sin(theta2)))
        
        a_y = (self.l1 * (alpha1 * np.sin(theta1) + omega1**2 * np.cos(theta1)) +
               self.l2 * (alpha2 * np.sin(theta2) + omega2**2 * np.cos(theta2)) - self.g)
        return a_x / self.g, a_y / self.g
  
    def compute_energy(self, state: np.ndarray) -> float:
      
        theta1, theta2, omega1, omega2 = state
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        T = (0.5 * (m1 + m2) * l1**2 * omega1**2 +
             0.5 * m2 * l2**2 * omega2**2 +
             m2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2))
        V = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
        return T + V
    
    def integrate(
        self,
        initial_state: np.ndarray,
        t_span: Tuple[float, float],
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        solution = integrate.solve_ivp(
            self.equations_of_motion,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='RK45',
            dense_output=True
        )
      
        return solution.t, solution.y.T

def build_sindy_library(X: np.ndarray, poly_order: int = 2) -> Tuple[np.ndarray, List[str]]:
    n_samples, n_features = X.shape
    library = [np.ones(n_samples)]
    names = ['1']
    for i in range(n_features):
        library.append(X[:, i])
        names.append(f'x{i}')
    if poly_order >= 2:
        for i in range(n_features):
            for j in range(i, n_features):
                library.append(X[:, i] * X[:, j])
                names.append(f'x{i}*x{j}')
    if poly_order >= 3:
        for i in range(n_features):
            for j in range(i, n_features):
                for k in range(j, n_features):
                    library.append(X[:, i] * X[:, j] * X[:, k])
                    names.append(f'x{i}*x{j}*x{k}')
    for i in range(n_features):
        x_norm = X[:, i] / (np.std(X[:, i]) + 1e-8)
        library.append(np.sin(x_norm))
        names.append(f'sin(x{i})')
        library.append(np.cos(x_norm))
        names.append(f'cos(x{i})')
    
    return np.column_stack(library), names

def sindy_reconstruct(
    signal: np.ndarray,
    gap_start: int,
    gap_end: int,
    dt: float = 1/35.0,
    threshold: float = 0.1
) -> np.ndarray:
    gap_length = gap_end - gap_start
    pre_signal = signal[:gap_start]
    post_signal = signal[gap_end:]
    if len(pre_signal) < 50 or len(post_signal) < 50:
        return _physics_fallback(signal, gap_start, gap_end)
    tau = 3
    n_embed = len(pre_signal) - 2*tau
    if n_embed < 20:
        return _physics_fallback(signal, gap_start, gap_end)
    X = np.column_stack([
        pre_signal[2*tau:],    
        pre_signal[tau:-tau],  
        pre_signal[:-2*tau]    
    ])
    dX = np.gradient(X[:, 0], dt)
    Theta, names = build_sindy_library(X[:-1], poly_order=2)
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=threshold, fit_intercept=False, max_iter=1000)
    lasso.fit(Theta, dX[:-1])
    coeffs = lasso.coef_
    active = np.abs(coeffs) > threshold/10
    if not np.any(active):
        return _physics_fallback(signal, gap_start, gap_end)
  
  
    def dynamics(t, state):
        x = state.reshape(1, -1)
        theta, _ = build_sindy_library(x, poly_order=2)
        return np.dot(theta, coeffs)
    x0 = np.array([
        pre_signal[-1],
        pre_signal[-1-tau],
        pre_signal[-1-2*tau]
    ])
    t_span = (0, gap_length * dt)
    t_eval = np.linspace(0, gap_length * dt, gap_length)
    try:
        solution = integrate.solve_ivp(
            dynamics,
            t_span,
            x0,
            t_eval=t_eval,
            method='RK45'
        )
        forward_pred = solution.y[0]
    except Exception:
        forward_pred = np.full(gap_length, np.mean(pre_signal))
    x0_post = np.array([
        post_signal[0],
        post_signal[tau] if len(post_signal) > tau else post_signal[0],
        post_signal[2*tau] if len(post_signal) > 2*tau else post_signal[0]
    ])
    try:
        solution_back = integrate.solve_ivp(
            lambda t, x: -dynamics(t, x),
            t_span,
            x0_post,
            t_eval=t_eval,
            method='RK45'
        )
        backward_pred = solution_back.y[0][::-1]
    except Exception:
        backward_pred = np.full(gap_length, np.mean(post_signal))
    blend_weight = np.linspace(1, 0, gap_length)
    reconstruction = blend_weight * forward_pred + (1 - blend_weight) * backward_pred
    reconstruction = _adjust_to_boundaries(
        reconstruction,
        pre_signal[-1],
        post_signal[0]
    )
    obs_min, obs_max = np.min(signal), np.max(signal)
    margin = (obs_max - obs_min) * 0.5
    reconstruction = np.clip(reconstruction, obs_min - margin, obs_max + margin)
    return reconstruction

def energy_conserving_reconstruct(
    signal: np.ndarray,
    gap_start: int,
    gap_end: int,
    energy_weight: float = 1.0
) -> np.ndarray:
    gap_length = gap_end - gap_start
    pre_signal = signal[:gap_start]
    post_signal = signal[gap_end:]
    observed = np.concatenate([pre_signal[-500:], post_signal[:500]])
    obs_mean = np.mean(observed)
    obs_energy = np.mean(observed**2)
    obs_std = np.std(observed)
    left_val = pre_signal[-1] if len(pre_signal) > 0 else obs_mean
    right_val = post_signal[0] if len(post_signal) > 0 else obs_mean
    t = np.linspace(0, np.pi, gap_length)
    base = left_val + (right_val - left_val) * (1 - np.cos(t)) / 2
    base_energy = np.mean(base**2)
    if base_energy < obs_energy:
        deficit = obs_energy - base_energy
        amplitude = np.sqrt(2 * deficit)

        from scipy.fft import rfft, rfftfreq
        fft_obs = np.abs(rfft(observed - np.mean(observed)))
        freqs = rfftfreq(len(observed), d=1/35.0)

        if len(fft_obs) > 1:
            dom_idx = np.argmax(fft_obs[1:]) + 1
            dom_freq = freqs[dom_idx] if dom_idx < len(freqs) else 1.0
        else:
            dom_freq = 1.0

        taper = np.sin(np.linspace(0, np.pi, gap_length))
        t_phys = np.arange(gap_length) / 35.0
        oscillation = amplitude * np.sin(2 * np.pi * dom_freq * t_phys) * taper
        reconstruction = base + energy_weight * oscillation
    else:
        reconstruction = base

    reconstruction = _adjust_to_boundaries(reconstruction, left_val, right_val)
  
    return reconstruction

def phase_space_reconstruct(
    signal: np.ndarray,
    gap_start: int,
    gap_end: int,
    embed_dim: int = 4,
    embed_delay: int = 5
) -> np.ndarray:
    gap_length = gap_end - gap_start
    pre_signal = signal[:gap_start]
    post_signal = signal[gap_end:]
    if len(pre_signal) < embed_dim * embed_delay + 50:
        return _physics_fallback(signal, gap_start, gap_end)
    n_vectors = len(pre_signal) - (embed_dim - 1) * embed_delay
    embedding = np.zeros((n_vectors, embed_dim))
    for i in range(embed_dim):
        start = (embed_dim - 1 - i) * embed_delay
        end = start + n_vectors
        embedding[:, i] = pre_signal[start:end]
    attractor_mean = np.mean(embedding, axis=0)
    attractor_cov = np.cov(embedding.T)
    attractor_cov += np.eye(embed_dim) * 0.01 * np.trace(attractor_cov) / embed_dim
    left_state = np.array([pre_signal[-1 - i*embed_delay] for i in range(embed_dim)])
    right_state = np.array([post_signal[i*embed_delay] if i*embed_delay < len(post_signal) 
                           else post_signal[-1] for i in range(embed_dim)])
    n_interp = gap_length
    interp_states = np.zeros((n_interp, embed_dim))
    for i in range(embed_dim):
        interp_states[:, i] = np.linspace(left_state[i], right_state[i], n_interp)
    inv_cov = np.linalg.inv(attractor_cov)
    reconstruction = np.zeros(n_interp)
    for i in range(n_interp):
        state = interp_states[i]
        diff = state - attractor_mean
        maha_dist = np.sqrt(np.dot(diff, np.dot(inv_cov, diff)))
        if maha_dist > 2.0:
            state = attractor_mean + diff / maha_dist * 2.0
        reconstruction[i] = state[0]
    reconstruction = _adjust_to_boundaries(
        reconstruction,
        pre_signal[-1],
        post_signal[0] if len(post_signal) > 0 else pre_signal[-1]
    )
    return reconstruction

def _physics_fallback(signal: np.ndarray, gap_start: int, gap_end: int) -> np.ndarray:
    gap_length = gap_end - gap_start
    observed = np.concatenate([signal[:gap_start], signal[gap_end:]])
    obs_mean = np.mean(observed)
    left_val = signal[gap_start - 1] if gap_start > 0 else obs_mean
    right_val = signal[gap_end] if gap_end < len(signal) else obs_mean
    t = np.linspace(0, np.pi, gap_length)
    reconstruction = left_val + (right_val - left_val) * (1 - np.cos(t)) / 2
    return reconstruction

def _adjust_to_boundaries(
    reconstruction: np.ndarray,
    left_target: float,
    right_target: float,
    blend_samples: int = 20
) -> np.ndarray:
    n = len(reconstruction)
    blend = min(blend_samples, n // 4)
    result = reconstruction.copy()
    left_diff = left_target - reconstruction[0]
    if blend > 0:
        taper = np.cos(np.linspace(0, np.pi/2, blend))
        result[:blend] += left_diff * taper
    right_diff = right_target - reconstruction[-1]
    if blend > 0:
        taper = np.cos(np.linspace(np.pi/2, 0, blend))
        result[-blend:] += right_diff * taper
  
    return result

def physics_informed_reconstruction(
    df: pd.DataFrame,
    col: str = 'acc_x',
    method: str = 'energy'
) -> pd.DataFrame:
  
    df_result = df.copy()
    signal = df_result[col].values.copy()
  
    is_nan = np.isnan(signal)
    if not np.any(is_nan):
        return df_result
  
    gap_starts = np.where(np.diff(np.concatenate([[False], is_nan])) == 1)[0]
    gap_ends = np.where(np.diff(np.concatenate([is_nan, [False]])) == -1)[0] + 1
  
    for gap_start, gap_end in zip(gap_starts, gap_ends):
        if method == 'sindy':
            reconstruction = sindy_reconstruct(signal, gap_start, gap_end)
        elif method == 'energy':
            reconstruction = energy_conserving_reconstruct(signal, gap_start, gap_end)
        elif method == 'phase_space':
            reconstruction = phase_space_reconstruct(signal, gap_start, gap_end)
        else:
            reconstruction = _physics_fallback(signal, gap_start, gap_end)
      
        signal[gap_start:gap_end] = reconstruction
  
    df_result[col] = signal
    return df_result

if __name__ == '__main__':
    import json
    from sklearn.metrics import mean_squared_error, r2_score

    print("Testing Physics-Informed Reconstruction Methods")
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
    methods = ['sindy', 'energy', 'phase_space']
    for method in methods:
        try:
            if method == 'sindy':
                pred = sindy_reconstruct(acc_x, gap_start, gap_end)
            elif method == 'energy':
                pred = energy_conserving_reconstruct(acc_x, gap_start, gap_end)
            else:
                pred = phase_space_reconstruct(acc_x, gap_start, gap_end)
          
            rmse = np.sqrt(mean_squared_error(true_values, pred))
            r2 = r2_score(true_values, pred)
          
            print(f"{method:15s}: RMSE={rmse:.4f}, R2={r2:+.4f}")
          
        except Exception as e:
            print(f"{method:15s}: Failed - {e}")
            
    print("\n" + "=" * 70)
    print("Testing complete!")
