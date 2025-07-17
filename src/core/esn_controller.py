#esn
import numpy as np
from scipy.linalg import pinv
from scipy import stats
from typing import Optional, Tuple, Dict, List
import warnings

class ESNController:
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 n_reservoir: int = 200,
                 spectral_radius: float = 0.9,
                 leak_rate: float = 0.05,
                 feedback_gain: float = 1.0,
                 noise_level: float = 0.01,
                 regularization: float = 1e-6):
        self.validate_params(n_reservoir, spectral_radius, leak_rate, feedback_gain)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.noise_level = noise_level
        self.regularization = regularization

        self.W_in = self._initialize_input_weights(n_reservoir, n_inputs)
        self.W_res = self._initialize_reservoir(n_reservoir, spectral_radius)
        self.W_out = np.zeros((n_outputs, n_reservoir + n_inputs))

        self.state = np.zeros(n_reservoir)
        self.rls_cov = np.eye(n_reservoir + n_inputs) * 100
        self.feedback_gain = feedback_gain
        self.state_history = []

        self.mse_history = []
        self.convergence_metrics = []
        self.performance_statistics = {}

        self.input_mean = np.zeros(n_inputs)
        self.input_std = np.ones(n_inputs)
        self.output_scale = 1.0

    def validate_params(self, n_reservoir: int, spectral_radius: float, leak_rate: float, feedback_gain: float):
        if n_reservoir <= 0:
            raise ValueError("Reservoir size must be positive")
        if not (0 < spectral_radius <= 1.5):
            raise ValueError("Spectral radius should be in (0, 1.5]")
        if not (0 < leak_rate <= 1.0):
            raise ValueError("Leak rate should be in (0, 1]")
        if not (0 < feedback_gain <= 10.0):
            raise ValueError("Feedback gain too large or non-positive")

    def _initialize_input_weights(self, n_reservoir: int, n_inputs: int) -> np.ndarray:
        scale = np.sqrt(1.0 / n_inputs)
        return np.random.uniform(-scale, scale, size=(n_reservoir, n_inputs))

    def _initialize_reservoir(self, n: int, rho: float) -> np.ndarray:
        W = np.random.randn(n, n)
        sparsity = 0.1 + 0.1 * np.exp(-n / 100)
        mask = np.random.rand(n, n) < sparsity
        W *= mask
        eigs = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigs))
        if max_eig == 0:
            raise ValueError("Reservoir matrix has zero spectral radius")
        W *= rho / max_eig
        return W

    def _normalize_input(self, u: np.ndarray) -> np.ndarray:
        self.input_mean = 0.99 * self.input_mean + 0.01 * u
        self.input_std = 0.99 * self.input_std + 0.01 * np.abs(u - self.input_mean) + 1e-6
        return (u - self.input_mean) / self.input_std

    def predict(self, extended_state: np.ndarray) -> np.ndarray:
        return self.W_out @ extended_state

    def _current_spectral_radius(self) -> float:
        try:
            eigs = np.linalg.eigvals(self.W_res)
            return float(np.max(np.abs(eigs)))
        except Exception:
            return -1.0

    def update(self, u: np.ndarray, y_feedback: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        u_norm = self._normalize_input(u)
        input_energy = np.linalg.norm(u_norm)

        new_state = self._compute_new_state(u_norm, y_feedback)
        extended_state = np.hstack([new_state, u_norm])

        diagnostics = {
            'state_norm': np.linalg.norm(new_state),
            'input_energy': input_energy,
            'feedback_gain': self.feedback_gain,
            'spectral_radius': self._current_spectral_radius()
        }
        self.state_history.append(diagnostics)

        self.state = new_state
        return extended_state, diagnostics

    def _compute_new_state(self, u_norm: np.ndarray, y_feedback: Optional[np.ndarray]) -> np.ndarray:
        input_part = self.W_in @ u_norm
        reservoir_part = self.W_res @ self.state
        new_state = (1 - self.leak_rate) * self.state + self.leak_rate * np.tanh(input_part + reservoir_part)

        if y_feedback is not None:
            self._adapt_feedback_gain(y_feedback)
            feedback = np.resize(y_feedback, new_state.shape)
            new_state += self.feedback_gain * feedback

        return new_state

    def _adapt_feedback_gain(self, error: np.ndarray):
        error_magnitude = np.linalg.norm(error)
        gain_adjustment = 0.05 * np.tanh(error_magnitude)
        self.feedback_gain = np.clip(self.feedback_gain * (1 + gain_adjustment), 0.1, 5.0)

    def train_rls(self, extended_state: np.ndarray, target: np.ndarray, lambda_: float = 0.999) -> float:
        noisy_state = extended_state + np.random.normal(0, self.noise_level, extended_state.shape)

        prediction = self.predict(noisy_state)
        error = target - prediction
        mse = np.mean(error ** 2)

        self.mse_history.append(mse)
        self._update_convergence_metrics(mse)

        denom = lambda_ + noisy_state.T @ self.rls_cov @ noisy_state
        gain = (self.rls_cov @ noisy_state) / (denom + self.regularization)

        self.W_out += np.outer(error, gain)
        outer_prod = np.outer(gain, noisy_state)
        self.rls_cov = (self.rls_cov - outer_prod @ self.rls_cov) / lambda_

        self.rls_cov = 0.5 * (self.rls_cov + self.rls_cov.T)
        eigenvals = np.linalg.eigvals(self.rls_cov)
        min_eig = np.min(eigenvals)
        if min_eig < self.regularization:
            self.rls_cov += (self.regularization - min_eig) * np.eye(self.rls_cov.shape[0])

        if mse > 1e3:
            self._handle_divergence(mse)

        return float(mse)

    def _update_convergence_metrics(self, mse: float):
        if len(self.mse_history) >= 10:
            recent_mse = self.mse_history[-10:]
            trend = np.polyfit(range(10), recent_mse, 1)[0]
            variance = np.var(recent_mse)
            self.convergence_metrics.append({
                'trend': trend,
                'variance': variance,
                'mse': mse
            })

    def _handle_divergence(self, mse: float):
        self.W_out *= 0.9
        self.rls_cov = np.eye(self.rls_cov.shape[0]) * 100
        warnings.warn(f"High MSE detected ({mse:.2f}), resetting training state")

    def reset_state(self):
        self.state = np.zeros(self.n_reservoir)
        self.state_history = []
        self.mse_history = []
        self.convergence_metrics = []

    def get_diagnostics(self) -> dict:
        if not self.state_history:
            return {}

        base_diagnostics = {
            'mean_state_norm': float(np.mean([d['state_norm'] for d in self.state_history])),
            'max_feedback_gain': float(np.max([d['feedback_gain'] for d in self.state_history])),
            'spectral_radius': float(self.state_history[-1]['spectral_radius']),
            'weights_norm': float(np.linalg.norm(self.W_out))
        }

        if self.mse_history:
            base_diagnostics.update({
                'mse_mean': float(np.mean(self.mse_history)),
                'mse_std': float(np.std(self.mse_history)),
                'mse_trend': float(np.polyfit(range(len(self.mse_history)), self.mse_history, 1)[0]) if len(self.mse_history) > 1 else 0.0,
                'convergence_stability': float(np.std(self.mse_history[-50:])) if len(self.mse_history) >= 50 else float('inf')
            })

        return base_diagnostics

    def get_statistical_summary(self) -> Dict[str, float]:
        if not self.mse_history:
            return {}

        mse_array = np.array(self.mse_history)
        return {
            'mse_mean': float(np.mean(mse_array)),
            'mse_median': float(np.median(mse_array)),
            'mse_std': float(np.std(mse_array)),
            'mse_min': float(np.min(mse_array)),
            'mse_max': float(np.max(mse_array)),
            'mse_q25': float(np.percentile(mse_array, 25)),
            'mse_q75': float(np.percentile(mse_array, 75)),
            'convergence_rate': float(-np.polyfit(range(len(mse_array)), mse_array, 1)[0]),
            'final_stability': float(np.std(mse_array[-100:])) if len(mse_array) >= 100 else float('inf')
        }

    def perform_statistical_tests(self, baseline_mse: List[float]) -> Dict[str, float]:
        if not self.mse_history or not baseline_mse:
            return {}

        t_stat, p_value = stats.ttest_ind(self.mse_history, baseline_mse)
        u_stat, u_p_value = stats.mannwhitneyu(self.mse_history, baseline_mse, alternative='two-sided')
        pooled_std = np.sqrt(((len(self.mse_history) - 1) * np.var(self.mse_history) +
                              (len(baseline_mse) - 1) * np.var(baseline_mse)) /
                             (len(self.mse_history) + len(baseline_mse) - 2))
        cohens_d = (np.mean(self.mse_history) - np.mean(baseline_mse)) / pooled_std

        return {
            't_statistic': float(t_stat),
            't_p_value': float(p_value),
            'u_statistic': float(u_stat),
            'u_p_value': float(u_p_value),
            'cohens_d': float(cohens_d),
            'significant_improvement': bool(p_value < 0.05 and np.mean(self.mse_history) < np.mean(baseline_mse))
        }

    def get_weights(self) -> dict:
        return {
            "W_in": self.W_in,
            "W_res": self.W_res,
            "W_out": self.W_out
        }

    def set_weights(self, weights: dict):
        self.W_in = weights["W_in"]
        self.W_res = weights["W_res"]
        self.W_out = weights["W_out"]

    def save(self, path: str):
        np.savez(
            path,
            W_in=self.W_in,
            W_res=self.W_res,
            W_out=self.W_out,
            rls_cov=self.rls_cov,
            input_mean=self.input_mean,
            input_std=self.input_std,
            mse_history=np.array(self.mse_history),
            config={
                'n_inputs': self.n_inputs,
                'n_outputs': self.n_outputs,
                'n_reservoir': self.n_reservoir,
                'spectral_radius': self.spectral_radius,
                'leak_rate': self.leak_rate
            }
        )

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.W_in = data['W_in']
        self.W_res = data['W_res']
        self.W_out = data['W_out']
        self.rls_cov = data['rls_cov']
        self.input_mean = data['input_mean']
        self.input_std = data['input_std']

        if 'mse_history' in data:
            self.mse_history = data['mse_history'].tolist()

        config = data['config'].item()
        self.n_inputs = config['n_inputs']
        self.n_outputs = config['n_outputs']
        self.n_reservoir = config['n_reservoir']
        self.spectral_radius = config['spectral_radius']
        self.leak_rate = config['leak_rate']