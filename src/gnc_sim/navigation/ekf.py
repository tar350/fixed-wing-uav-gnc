"""Extended Kalman Filter for longitudinal aircraft state estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnc_sim.models.uav_longitudinal import LongitudinalAircraft, LongitudinalState
from gnc_sim.sensors.measurements import measurement_vector_from_state


@dataclass(frozen=True)
class EKFConfig:
    """EKF tuning parameters."""

    initial_covariance_diag: tuple[float, float, float, float, float] = (
        1.0,
        1.0,
        np.radians(2.0) ** 2,
        np.radians(2.0) ** 2,
        4.0,
    )
    process_noise_diag: tuple[float, float, float, float, float] = (
        0.10,
        0.10,
        np.radians(0.50) ** 2,
        np.radians(0.25) ** 2,
        0.10,
    )
    finite_difference_steps: tuple[float, float, float, float, float] = (
        1.0e-3,
        1.0e-3,
        1.0e-5,
        1.0e-5,
        1.0e-2,
    )

    @classmethod
    def from_config(cls, config: dict) -> "EKFConfig":
        """Build EKF config from optional project config values."""
        nav_cfg = config.get("navigation", {})
        ekf_cfg = nav_cfg.get("ekf", {})

        return cls(
            initial_covariance_diag=tuple(
                ekf_cfg.get(
                    "initial_covariance_diag",
                    [1.0, 1.0, np.radians(2.0) ** 2, np.radians(2.0) ** 2, 4.0],
                )
            ),
            process_noise_diag=tuple(
                ekf_cfg.get(
                    "process_noise_diag",
                    [
                        0.10,
                        0.10,
                        np.radians(0.50) ** 2,
                        np.radians(0.25) ** 2,
                        0.10,
                    ],
                )
            ),
            finite_difference_steps=tuple(
                ekf_cfg.get(
                    "finite_difference_steps",
                    [1.0e-3, 1.0e-3, 1.0e-5, 1.0e-5, 1.0e-2],
                )
            ),
        )


def _wrap_angle_rad(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def _rk4_step(function, x: np.ndarray, dt: float) -> np.ndarray:
    """Generic RK4 step."""
    k1 = function(x)
    k2 = function(x + 0.5 * dt * k1)
    k3 = function(x + 0.5 * dt * k2)
    k4 = function(x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class LongitudinalEKF:
    """Extended Kalman Filter for longitudinal UAV state estimation."""

    def __init__(
        self,
        aircraft: LongitudinalAircraft,
        x0: np.ndarray,
        measurement_covariance: np.ndarray,
        config: EKFConfig | None = None,
    ) -> None:
        """Initialize EKF."""
        self.aircraft = aircraft
        self.config = config or EKFConfig()

        self.x = np.asarray(x0, dtype=float)
        self.P = np.diag(np.asarray(self.config.initial_covariance_diag, dtype=float))
        self.Q_c = np.diag(np.asarray(self.config.process_noise_diag, dtype=float))
        self.R = np.asarray(measurement_covariance, dtype=float)

        self.finite_difference_steps = np.asarray(
            self.config.finite_difference_steps,
            dtype=float,
        )

    @property
    def state(self) -> LongitudinalState:
        """Return current estimated state."""
        return LongitudinalState.from_array(self.x)

    def dynamics_vector(
        self,
        x: np.ndarray,
        elevator_rad: float,
        throttle: float,
        vertical_gust_m_s: float = 0.0,
    ) -> np.ndarray:
        """Evaluate nonlinear dynamics at a vector state."""
        state = LongitudinalState.from_array(x)

        return self.aircraft.dynamics(
            state=state,
            elevator_rad=elevator_rad,
            throttle=throttle,
            vertical_gust_m_s=vertical_gust_m_s,
        )

    def measurement_model(self, x: np.ndarray) -> np.ndarray:
        """Evaluate measurement model at a vector state."""
        state = LongitudinalState.from_array(x)
        return measurement_vector_from_state(state)

    def dynamics_jacobian(
        self,
        x: np.ndarray,
        elevator_rad: float,
        throttle: float,
        vertical_gust_m_s: float = 0.0,
    ) -> np.ndarray:
        """Finite-difference dynamics Jacobian F = df/dx."""
        n = x.size
        F = np.zeros((n, n), dtype=float)

        for idx in range(n):
            dx = np.zeros(n, dtype=float)
            dx[idx] = self.finite_difference_steps[idx]

            f_plus = self.dynamics_vector(
                x + dx,
                elevator_rad=elevator_rad,
                throttle=throttle,
                vertical_gust_m_s=vertical_gust_m_s,
            )
            f_minus = self.dynamics_vector(
                x - dx,
                elevator_rad=elevator_rad,
                throttle=throttle,
                vertical_gust_m_s=vertical_gust_m_s,
            )

            F[:, idx] = (f_plus - f_minus) / (2.0 * dx[idx])

        return F

    def measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Finite-difference measurement Jacobian H = dh/dx."""
        n_states = x.size
        z0 = self.measurement_model(x)
        n_measurements = z0.size

        H = np.zeros((n_measurements, n_states), dtype=float)

        for idx in range(n_states):
            dx = np.zeros(n_states, dtype=float)
            dx[idx] = self.finite_difference_steps[idx]

            h_plus = self.measurement_model(x + dx)
            h_minus = self.measurement_model(x - dx)

            H[:, idx] = (h_plus - h_minus) / (2.0 * dx[idx])

        return H

    def predict(
        self,
        elevator_rad: float,
        throttle: float,
        dt_s: float,
        vertical_gust_m_s: float = 0.0,
    ) -> None:
        """EKF prediction step."""
        F_c = self.dynamics_jacobian(
            x=self.x,
            elevator_rad=elevator_rad,
            throttle=throttle,
            vertical_gust_m_s=vertical_gust_m_s,
        )

        Phi = np.eye(self.x.size) + F_c * dt_s

        def f(x_vector: np.ndarray) -> np.ndarray:
            return self.dynamics_vector(
                x_vector,
                elevator_rad=elevator_rad,
                throttle=throttle,
                vertical_gust_m_s=vertical_gust_m_s,
            )

        self.x = _rk4_step(f, self.x, dt_s)
        self.P = Phi @ self.P @ Phi.T + self.Q_c * dt_s

        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z: np.ndarray) -> None:
        """EKF measurement update step."""
        z = np.asarray(z, dtype=float)

        z_pred = self.measurement_model(self.x)
        H = self.measurement_jacobian(self.x)

        residual = z - z_pred

        # Measurement vector:
        # [airspeed, alpha, q, theta, altitude]
        residual[1] = _wrap_angle_rad(residual[1])
        residual[3] = _wrap_angle_rad(residual[3])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ residual

        I = np.eye(self.P.shape[0])

        # Joseph stabilized covariance update.
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)