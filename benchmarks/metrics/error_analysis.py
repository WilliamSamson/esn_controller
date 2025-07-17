# benchmarks/metrics/error_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def compute_error_statistics(errors: List[float]) -> Dict[str, float]:
    arr = np.array(errors)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25))
    }

def plot_error_distribution(errors: List[float], save_path: str = None, config_label: str = ""):
    arr = np.array(errors)
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    plt.figure(figsize=(6, 4))
    plt.hist(arr, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    plt.axvline(mean, color='red', linestyle='--', label=f"Mean = {mean:.4f}")
    plt.axvline(mean + std, color='green', linestyle=':', label=f"+1 STD = {mean+std:.4f}")
    plt.axvline(mean - std, color='green', linestyle=':', label=f"-1 STD = {mean-std:.4f}")

    plt.title(f"Prediction Error Distribution {config_label}")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()