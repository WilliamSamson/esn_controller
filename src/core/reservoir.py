# src/core/reservoir.py
import numpy as np

class Reservoir:
    def __init__(self, n_reservoir: int, spectral_radius: float = 0.9, input_scaling: float = 0.1, sparsity: float = 0.1):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.sparsity = sparsity

        self.state = np.zeros(n_reservoir)
        self.W_res = self._initialize_reservoir()
        self.W_in = None

    def _initialize_reservoir(self) -> np.ndarray:
        W = np.random.randn(self.n_reservoir, self.n_reservoir)
        mask = np.random.rand(self.n_reservoir, self.n_reservoir) < self.sparsity
        W *= mask
        eigs = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigs))
        if max_eig == 0:
            raise ValueError("Reservoir matrix has zero spectral radius")
        W *= self.spectral_radius / max_eig
        return W

    def initialize_input_weights(self, n_inputs: int):
        self.W_in = np.random.uniform(-self.input_scaling, self.input_scaling, size=(self.n_reservoir, n_inputs))

    def update(self, input_vector: np.ndarray, leak_rate: float) -> np.ndarray:
        if self.W_in is None:
            raise ValueError("Input weights not initialized")
        input_part = self.W_in @ input_vector
        reservoir_part = self.W_res @ self.state
        self.state = (1 - leak_rate) * self.state + leak_rate * np.tanh(input_part + reservoir_part)
        return self.state

    def reset(self):
        self.state = np.zeros(self.n_reservoir)