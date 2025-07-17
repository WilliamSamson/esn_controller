# src/feedback/pid.py
import numpy as np
from typing import Tuple

class PIDController:
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, dt: float = 0.01, output_limits: Tuple[float, float] = (-np.inf, np.inf)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limits = output_limits

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float) -> float:
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        self.prev_error = error
        return output