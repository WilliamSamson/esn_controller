# benchmarks/systems/arm.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.core.esn_controller import ESNController
from src.feedback.pid import PIDController
from benchmarks.metrics.statistical_tests import StatisticalValidator
import os
import json
from scipy import stats
from typing import List, Dict

os.makedirs("docs/figures", exist_ok=True)
os.makedirs("benchmarks/results", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)
np.random.seed(42)


class RoboticArmSimulator:
    def __init__(self, m1: float = 1.0, m2: float = 1.0, l1: float = 1.0, l2: float = 1.0, g: float = 9.81):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g

    def dynamics(self, state: np.ndarray, t: float, u: np.ndarray) -> np.ndarray:
        if u.shape != (2,):
            raise ValueError(f"Control input must be shape (2,), got {u.shape}")
        q1, q2, dq1, dq2 = state
        u1, u2 = u
        noise = np.random.normal(0, 0.01, size=2)

        m11 = (self.m1 + self.m2) * self.l1 ** 2 + self.m2 * self.l2 ** 2 + 2 * self.m2 * self.l1 * self.l2 * np.cos(q2)
        m12 = self.m2 * self.l2 ** 2 + self.m2 * self.l1 * self.l2 * np.cos(q2)
        m22 = self.m2 * self.l2 ** 2
        M = np.array([[m11, m12], [m12, m22]])

        c = -self.m2 * self.l1 * self.l2 * np.sin(q2) * (2 * dq1 * dq2 + dq2 ** 2)
        c2 = self.m2 * self.l1 * self.l2 * np.sin(q2) * dq1 ** 2
        C = np.array([c, c2])

        g1 = (self.m1 + self.m2) * self.g * self.l1 * np.cos(q1) + self.m2 * self.g * self.l2 * np.cos(q1 + q2)
        g2 = self.m2 * self.g * self.l2 * np.cos(q1 + q2)
        G = np.array([g1, g2])

        ddq = np.linalg.solve(M, u - C - G + noise)
        return np.array([dq1, dq2, ddq[0], ddq[1]])

    def simulate(self, u: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
        if u.shape != (2,):
            raise ValueError(f"Control input must be shape (2,), got {u.shape}")
        sol = solve_ivp(lambda t, y: self.dynamics(y, t, u), [0, dt], state, method="RK45", t_eval=[dt])
        return sol.y[:, -1]


class ArmBenchmark:
    def __init__(self, n_trials: int = 15):
        self.n_trials = n_trials
        self.simulator = RoboticArmSimulator()

    def run_single_trial(self, trial_id: int, config: Dict, use_pid_only: bool = False) -> Dict:
        np.random.seed(42 + trial_id)
        controller = None if use_pid_only else ESNController(**config)
        pid = PIDController(kp=5.0, ki=0.2, kd=1.0, dt=0.01, output_limits=(-20, 20))
        validator = StatisticalValidator()

        dt = 0.01
        T = 5.0
        n_steps = int(T / dt)
        initial_state = np.array(
            [np.random.uniform(np.pi / 6, np.pi / 3), np.random.uniform(np.pi / 6, np.pi / 3), 0.0, 0.0])
        target = np.array([0.0, 0.0])

        mse_history = []
        state_history = [initial_state]
        control_history = []
        state = initial_state

        for t in range(n_steps):
            error = target - state[:2]
            pid_output = np.array([pid.update(error[0]), pid.update(error[1])])
            if use_pid_only:
                control = np.clip(pid_output, -20, 20)
                mse = float(np.mean(error ** 2))
            else:
                u_input = state
                extended_state, diagnostics = controller.update(u_input)
                esn_output = controller.predict(extended_state)
                if esn_output.shape != (2,):
                    raise ValueError(f"ESN output must be shape (2,), got {esn_output.shape}")
                control = np.clip(esn_output + pid_output, -20, 20)
                mse = controller.train_rls(extended_state, target)

            state = self.simulator.simulate(control, state, dt)
            mse_history.append(mse)
            state_history.append(state)
            control_history.append(control)

        if trial_id % 5 == 0:
            plt.figure(figsize=(12, 8))
            plt.subplot(3, 1, 1)
            plt.plot(np.arange(n_steps + 1) * dt, [s[0] for s in state_history], label="q1")
            plt.plot(np.arange(n_steps + 1) * dt, [s[1] for s in state_history], label="q2")
            plt.plot(np.arange(n_steps + 1) * dt, [0] * (n_steps + 1), 'k--', label="Target")
            plt.xlabel("Time (s)")
            plt.ylabel("Angles (rad)")
            plt.legend()
            plt.grid(True)
            plt.subplot(3, 1, 2)
            plt.plot(np.arange(n_steps) * dt, mse_history, label="MSE")
            plt.xlabel("Time (s)")
            plt.ylabel("Mean Squared Error")
            plt.legend()
            plt.grid(True)
            plt.subplot(3, 1, 3)
            plt.plot(np.arange(n_steps) * dt, [c[0] for c in control_history], label="Torque1")
            plt.plot(np.arange(n_steps) * dt, [c[1] for c in control_history], label="Torque2")
            plt.xlabel("Time (s)")
            plt.ylabel("Control Torques (N·m)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"docs/figures/arm_trial_{'pid' if use_pid_only else 'esn'}_{trial_id}.png", dpi=300,
                        bbox_inches="tight")
            plt.close()

        stats_out = validator.validate_performance(mse_history)
        result = {
            'trial_id': trial_id,
            'final_mse': mse_history[-1],
            'mean_mse': np.mean(mse_history),
            'convergence_rate': -np.polyfit(range(len(mse_history)), mse_history, 1)[0],
            'stability': np.std(mse_history[-100:]),
            'mse_history': mse_history,
            'diagnostics': stats_out if use_pid_only else dict(diagnostics, **stats_out),
            'state_history': state_history,
            'control_history': control_history
        }
        with open(f"benchmarks/results/arm_trial_{'pid' if use_pid_only else 'esn'}_{trial_id}_mse.json", "w") as f:
            json.dump(mse_history, f, indent=2)
        if not use_pid_only:
            controller.save(f"trained_models/arm_controller_{'pid' if use_pid_only else 'esn'}_{trial_id}.npz")
        return result

    def run_benchmark(self, config: Dict, use_pid_only: bool = False) -> Dict:
        print(f"Running {'PID-only' if use_pid_only else 'ESN+PID'} benchmark with {self.n_trials} trials...")
        results = [self.run_single_trial(trial, config, use_pid_only) for trial in range(self.n_trials)]
        return self._analyze_results(results)

    def _analyze_results(self, results: List[Dict]) -> Dict:
        final_mses = [r['final_mse'] for r in results]
        convergence_rates = [r['convergence_rate'] for r in results]
        stabilities = [r['stability'] for r in results]

        def confidence_interval(data, confidence=0.95):
            n = len(data)
            mean = np.mean(data)
            se = stats.sem(data)
            h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
            return mean - h, mean + h

        return {
            'summary_statistics': {
                'final_mse': {
                    'mean': float(np.mean(final_mses)),
                    'std': float(np.std(final_mses)),
                    'median': float(np.median(final_mses)),
                    'min': float(np.min(final_mses)),
                    'max': float(np.max(final_mses)),
                    'ci_95': confidence_interval(final_mses) if np.std(final_mses) > 0 else (0, 0)
                },
                'convergence_rate': {
                    'mean': float(np.mean(convergence_rates)),
                    'std': float(np.std(convergence_rates)),
                    'ci_95': confidence_interval(convergence_rates) if np.std(convergence_rates) > 0 else (0, 0)
                },
                'stability': {
                    'mean': float(np.mean(stabilities)),
                    'std': float(np.std(stabilities)),
                    'ci_95': confidence_interval(stabilities) if np.std(stabilities) > 0 else (0, 0)
                }
            },
            'normality_tests': {
                'final_mse_shapiro': stats.shapiro(final_mses) if np.std(final_mses) > 0 else (np.nan, np.nan),
                'convergence_rate_shapiro': stats.shapiro(convergence_rates) if np.std(convergence_rates) > 0 else (
                    np.nan, np.nan)
            },
            'raw_results': results
        }

    def compare_configurations(self, config1: Dict, config2: Dict, config1_name: str = "Default",
                               config2_name: str = "Alternative") -> Dict:
        print(f"Comparing {config1_name}, {config2_name}, and PID-only")
        results1 = self.run_benchmark(config1)
        results2 = self.run_benchmark(config2)
        results_pid = self.run_benchmark({}, use_pid_only=True)

        mse1 = [r['final_mse'] for r in results1['raw_results']]
        mse2 = [r['final_mse'] for r in results2['raw_results']]
        mse_pid = [r['final_mse'] for r in results_pid['raw_results']]

        t_stat_1_2, t_p_1_2 = stats.ttest_ind(mse1, mse2) if np.std(mse1) > 0 and np.std(mse2) > 0 else (np.nan, np.nan)
        t_stat_1_pid, t_p_1_pid = stats.ttest_ind(mse1, mse_pid) if np.std(mse1) > 0 and np.std(mse_pid) > 0 else (
            np.nan, np.nan)
        t_stat_2_pid, t_p_2_pid = stats.ttest_ind(mse2, mse_pid) if np.std(mse2) > 0 and np.std(mse_pid) > 0 else (
            np.nan, np.nan)

        u_stat_1_2, u_p_1_2 = stats.mannwhitneyu(mse1, mse2, alternative='two-sided') if np.std(mse1) > 0 and np.std(
            mse2) > 0 else (np.nan, np.nan)
        u_stat_1_pid, u_p_1_pid = stats.mannwhitneyu(mse1, mse_pid, alternative='two-sided') if np.std(
            mse1) > 0 and np.std(mse_pid) > 0 else (np.nan, np.nan)
        u_stat_2_pid, u_p_2_pid = stats.mannwhitneyu(mse2, mse_pid, alternative='two-sided') if np.std(
            mse2) > 0 and np.std(mse_pid) > 0 else (np.nan, np.nan)

        pooled_std_1_2 = np.sqrt(
            ((len(mse1) - 1) * np.var(mse1) + (len(mse2) - 1) * np.var(mse2)) / (len(mse1) + len(mse2) - 2))
        pooled_std_1_pid = np.sqrt(
            ((len(mse1) - 1) * np.var(mse1) + (len(mse_pid) - 1) * np.var(mse_pid)) / (len(mse1) + len(mse_pid) - 2))
        pooled_std_2_pid = np.sqrt(
            ((len(mse2) - 1) * np.var(mse2) + (len(mse_pid) - 1) * np.var(mse_pid)) / (len(mse2) + len(mse_pid) - 2))

        cohens_d_1_2 = (np.mean(mse1) - np.mean(mse2)) / pooled_std_1_2 if pooled_std_1_2 > 0 else np.nan
        cohens_d_1_pid = (np.mean(mse1) - np.mean(mse_pid)) / pooled_std_1_pid if pooled_std_1_pid > 0 else np.nan
        cohens_d_2_pid = (np.mean(mse2) - np.mean(mse_pid)) / pooled_std_2_pid if pooled_std_2_pid > 0 else np.nan

        return {
            config1_name: results1,
            config2_name: results2,
            'PID': results_pid,
            'statistical_tests': {
                f'{config1_name}_vs_{config2_name}': {
                    't_test': {'statistic': float(t_stat_1_2), 'p_value': float(t_p_1_2)},
                    'mann_whitney': {'statistic': float(u_stat_1_2), 'p_value': float(u_p_1_2)},
                    'effect_size': {'cohens_d': float(cohens_d_1_2)},
                    'significant_difference': bool(t_p_1_2 < 0.05) if not np.isnan(t_p_1_2) else False
                },
                f'{config1_name}_vs_PID': {
                    't_test': {'statistic': float(t_stat_1_pid), 'p_value': float(t_p_1_pid)},
                    'mann_whitney': {'statistic': float(u_stat_1_pid), 'p_value': float(u_p_1_pid)},
                    'effect_size': {'cohens_d': float(cohens_d_1_pid)},
                    'significant_difference': bool(t_p_1_pid < 0.05) if not np.isnan(t_p_1_pid) else False
                },
                f'{config2_name}_vs_PID': {
                    't_test': {'statistic': float(t_stat_2_pid), 'p_value': float(t_p_2_pid)},
                    'mann_whitney': {'statistic': float(u_stat_2_pid), 'p_value': float(u_p_2_pid)},
                    'effect_size': {'cohens_d': float(cohens_d_2_pid)},
                    'significant_difference': bool(t_p_2_pid < 0.05) if not np.isnan(t_p_2_pid) else False
                }
            }
        }


def run_statistical_benchmark():
    benchmark = ArmBenchmark(n_trials=15)
    default_config = {
        'n_inputs': 4,
        'n_outputs': 2,
        'n_reservoir': 500,
        'spectral_radius': 0.9,
        'leak_rate': 0.05,
        'feedback_gain': 0.5,
        'noise_level': 0.01
    }
    alternative_config = default_config.copy()
    alternative_config['n_reservoir'] = 200
    alternative_config['spectral_radius'] = 0.85
    alternative_config['leak_rate'] = 0.1

    comparison = benchmark.compare_configurations(default_config, alternative_config, "Default", "Alternative")
    with open("benchmarks/results/arm_statistical_benchmark.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print("Statistical benchmark completed")
    print(
        f"Default vs Alternative significant difference: {comparison['statistical_tests']['Default_vs_Alternative']['significant_difference']}")
    print(
        f"Default vs PID significant difference: {comparison['statistical_tests']['Default_vs_PID']['significant_difference']}")
    print(
        f"Alternative vs PID significant difference: {comparison['statistical_tests']['Alternative_vs_PID']['significant_difference']}")


if __name__ == "__main__":
    run_statistical_benchmark()