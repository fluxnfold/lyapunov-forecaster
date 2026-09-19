import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

class ParticleFilter:
  
  
    def __init__(
        self,
        n_particles: int = 500,
        n_axes: int = 3,
        process_noise_std: float = 0.1,
        measurement_noise_std: float = 0.1,
        resampling_threshold: float = 0.5,
        verbose: bool = False
    ):

        self.n_particles = n_particles
        self.n_axes = n_axes
        self.n_state = 2 * n_axes
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.resampling_threshold = resampling_threshold
        self.verbose = verbose
        self.particles = None     
        self.weights = None       
        self.log_weights = None   
        self.is_fitted = False
    
    def _init_particles(self, y_init: np.ndarray):
        mean_y = np.mean(y_init, axis=0)
        std_y = np.std(y_init, axis=0)
        self.particles = np.zeros((self.n_particles, self.n_state))
        for i in range(self.n_axes):
            self.particles[:, i] = np.random.normal(
                mean_y[i], std_y[i], self.n_particles
            )
        for i in range(self.n_axes):
            self.particles[:, self.n_axes + i] = np.random.normal(
                0, std_y[i] * 0.1, self.n_particles
            )
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.log_weights = np.log(self.weights)

        if self.verbose:
            print(f"PF initialized: {self.n_particles} particles, "
                  f"mean_accel={mean_y}, std_accel={std_y}")
    
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
        self._init_particles(y_init)
        self.is_fitted = True
        if self.verbose:
            print(f"PF fit complete: initialized on {init_samples} clean samples")
  
    def predict(
        self,
        X_indices: np.ndarray,
        y_observed: np.ndarray,
        missing_mask: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not self.is_fitted:
            raise RuntimeError("PF must be fit before prediction")

        if y_observed.ndim == 1:
            y_observed = y_observed.reshape(-1, 1)

        n_samples = len(X_indices)
        y_reconstructed = np.zeros_like(y_observed)
        y_std = np.zeros((n_samples, self.n_axes))
        particles = self.particles.copy()
        weights = self.weights.copy()
        for t in range(n_samples):
            particles = self._propagate_particles(particles)
            if not missing_mask[t]:
                z = y_observed[t]
                measurements_pred = particles[:, :self.n_axes]
                residuals = z - measurements_pred
                log_likelihood = -0.5 * np.sum(
                    residuals**2 / (self.measurement_noise_std**2),
                    axis=1
                )
                self.log_weights = self.log_weights + log_likelihood
                max_log_weight = np.max(self.log_weights)
                self.log_weights -= max_log_weight
                weights = np.exp(self.log_weights)
                weights /= np.sum(weights)
            ess = 1.0 / np.sum(weights**2)
            if ess < self.resampling_threshold * self.n_particles:
                particles, weights = self._resample_particles(particles, weights)
                self.log_weights = np.log(weights)
            y_mean = np.average(particles[:, :self.n_axes], weights=weights, axis=0)
            y_reconstructed[t] = y_mean
            weighted_var = np.average(
                (particles[:, :self.n_axes] - y_mean)**2,
                weights=weights, axis=0
            )
            y_std[t] = np.sqrt(weighted_var)

        return y_reconstructed, y_std
  
    def _propagate_particles(self, particles: np.ndarray) -> np.ndarray:
        dt = 1.0
        particles_new = particles.copy()
        for i in range(self.n_axes):
            particles_new[:, i] = (
                particles[:, i] +
                dt * particles[:, self.n_axes + i] +
                np.random.normal(0, self.process_noise_std, self.n_particles)
            )
            particles_new[:, self.n_axes + i] = (
                particles[:, self.n_axes + i] * 0.95 +
                np.random.normal(0, self.process_noise_std * 0.1, self.n_particles)
            )

        return particles_new
  
    def _resample_particles(
        self,
        particles: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.choice(
            self.n_particles,
            size=self.n_particles,
            p=weights
        )
        particles_resampled = particles[indices]
        weights_uniform = np.ones(self.n_particles) / self.n_particles

        return particles_resampled, weights_uniform

def particle_filter(
    df: pd.DataFrame,
    col: str = 'acc_x',
    n_particles: int = 500,
    process_noise_std: float = 0.1,
    measurement_noise_std: float = 0.1,
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
        pf = ParticleFilter(
            n_particles=n_particles,
            n_axes=1,
            process_noise_std=process_noise_std,
            measurement_noise_std=measurement_noise_std,
            verbose=verbose
        )
        pf.fit(X_clean, y_clean.reshape(-1, 1))
        y_recon, y_std = pf.predict(
            X,
            y.reshape(-1, 1),
            missing_mask
        )
        df_copy[column] = y_recon.flatten()
        df_copy[f'{column}_std'] = y_std.flatten()

        if verbose:
            print(f"PF reconstruction complete for {column}: "
                  f"{missing_mask.sum()} values filled")
  
    metadata = {
        'method': 'particle_filter',
        'columns': col,
        'n_particles': n_particles,
        'process_noise_std': process_noise_std,
        'measurement_noise_std': measurement_noise_std
    }
  
    return df_copy, metadata

def tune_particle_count(
    df: pd.DataFrame,
    col: str = 'acc_x',
    particle_counts: Optional[list] = None,
    verbose: bool = True
) -> pd.DataFrame:
    if particle_counts is None:
        particle_counts = [100, 300, 500, 1000]

    results = []
    for n in particle_counts:
        try:
            _, metadata = particle_filter(
                df, col,
                n_particles=n,
                verbose=False
            )
            results.append({
                'Particle Count': n,
                'Status': 'Success'
            })
        except Exception as e:
            results.append({
                'Particle Count': n,
                'Status': f'Failed: {str(e)}'
            })
    return pd.DataFrame(results)

if __name__ == '__main__':
    print("Particle Filter Module - Ready for import")
    print("Use: from reconstruction_methods.particle_filter import particle_filter")
