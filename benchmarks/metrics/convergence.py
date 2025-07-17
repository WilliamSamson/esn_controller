# benchmarks/metrics/convergence.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def compute_convergence_rate(mse_series: List[float], window: int = 50) -> Dict[str, float]:
    if len(mse_series) < window:
        window_data = mse_series
    else:
        window_data = mse_series[-window:]

    x = np.arange(len(window_data))
    slope = float(np.polyfit(x, window_data, 1)[0])
    std_dev = float(np.std(window_data))
    min_mse = float(np.min(mse_series))

    return {
        "convergence_slope": -slope,
        "stability": std_dev,
        "min_mse": min_mse
    }

def plot_convergence(mse_series: List[float], save_path: str = None, config_label: str = ""):
    plt.figure(figsize=(8, 4))
    plt.plot(mse_series, label=f"Convergence {config_label}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Convergence Curve")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()