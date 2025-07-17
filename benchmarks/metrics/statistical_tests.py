# benchmarks/metrics/statistical_tests.py
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import warnings

class StatisticalValidator:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results_cache = {}

    def validate_performance(self, mse_history: List[float], baseline_mse: Optional[List[float]] = None) -> Dict:
        if not mse_history:
            return {'error': 'Empty MSE history'}

        mse_array = np.array(mse_history)
        basic_stats = self._calculate_basic_statistics(mse_array)
        normality_results = self._test_normality(mse_array)
        convergence_results = self._analyze_convergence(mse_array)
        stability_results = self._analyze_stability(mse_array)

        results = {
            'basic_statistics': basic_stats,
            'normality_tests': normality_results,
            'convergence_analysis': convergence_results,
            'stability_analysis': stability_results
        }

        if baseline_mse:
            comparison_results = self._compare_with_baseline(mse_array, np.array(baseline_mse))
            results.update({'baseline_comparison': comparison_results})

        return results

    def _calculate_basic_statistics(self, data: np.ndarray) -> Dict:
        return {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'var': float(np.var(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'range': float(np.ptp(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data))
        }

    def _test_normality(self, data: np.ndarray) -> Dict:
        shapiro_stat, shapiro_p = stats.shapiro(data)
        dagostino_stat, dagostino_p = stats.normaltest(data)
        return {
            'shapiro_wilk': {'stat': float(shapiro_stat), 'p_value': float(shapiro_p)},
            'dagostino_k2': {'stat': float(dagostino_stat), 'p_value': float(dagostino_p)},
            'normal': bool(shapiro_p > self.alpha and dagostino_p > self.alpha)
        }

    def _analyze_convergence(self, data: np.ndarray) -> Dict:
        trend = float(np.polyfit(range(len(data)), data, 1)[0])
        rate = -trend
        return {
            'convergence_rate': rate,
            'trend_slope': trend
        }

    def _analyze_stability(self, data: np.ndarray) -> Dict:
        if len(data) < 100:
            tail = data[-len(data)//2:]
        else:
            tail = data[-100:]
        return {
            'tail_std': float(np.std(tail)),
            'tail_mean': float(np.mean(tail)),
            'stable': bool(np.std(tail) < 0.01)
        }

    def _compare_with_baseline(self, test_data: np.ndarray, baseline: np.ndarray) -> Dict:
        t_stat, t_p = stats.ttest_ind(test_data, baseline)
        u_stat, u_p = stats.mannwhitneyu(test_data, baseline, alternative='two-sided')
        pooled_std = np.sqrt(((len(test_data) - 1) * np.var(test_data) +
                             (len(baseline) - 1) * np.var(baseline)) /
                             (len(test_data) + len(baseline) - 2))
        cohens_d = (np.mean(test_data) - np.mean(baseline)) / pooled_std
        return {
            't_test': {'statistic': float(t_stat), 'p_value': float(t_p)},
            'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p)},
            'cohens_d': float(cohens_d),
            'significant': bool(t_p < self.alpha and np.mean(test_data) < np.mean(baseline))
        }