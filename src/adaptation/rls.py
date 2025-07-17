# src/adaptation/rls.py
import numpy as np

class RecursiveLeastSquares:
    def __init__(self, n_features: int, lambda_: float = 0.999, delta: float = 100.0, regularization: float = 1e-6):
        self.n_features = n_features
        self.lambda_ = lambda_
        self.regularization = regularization

        self.weights = np.zeros(n_features)
        self.P = np.eye(n_features) * delta

    def update(self, x: np.ndarray, target: float) -> float:
        x = x.reshape(-1)
        prediction = np.dot(self.weights, x)
        error = target - prediction

        denom = self.lambda_ + x.T @ self.P @ x
        gain = (self.P @ x) / (denom + self.regularization)

        self.weights += gain * error
        outer = np.outer(gain, x)
        self.P = (self.P - outer @ self.P) / self.lambda_

        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(self.n_features) * self.regularization

        return error ** 2

    def predict(self, x: np.ndarray) -> float:
        x = x.reshape(-1)
        return float(np.dot(self.weights, x))

    def reset(self):
        self.weights = np.zeros(self.n_features)
        self.P = np.eye(self.n_features) * 100.0