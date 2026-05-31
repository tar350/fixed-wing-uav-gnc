"""Longitudinal sensor measurement models.

The first navigation version uses a simplified but realistic longitudinal sensor
suite:

    z = [airspeed, angle_of_attack, pitch_rate, pitch_angle, altitude]

This is enough to estimate the longitudinal state:

    x = [u, w, q, theta, h]

where:
    u     body-axis forward velocity [m/s]
    w     body-axis vertical velocity [m/s]
    q     pitch rate [rad/s]
    theta pitch angle [rad]
    h     altitude [m]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnc_sim.models.uav_longitudinal import LongitudinalState


@dataclass(frozen=True)
class SensorNoiseConfig:
    """Standard deviations for longitudinal sensors."""

    airspeed_sigma_m_s: float = 0.30
    alpha_sigma_rad: float = np.radians(0.50)
    gyro_q_sigma_rad_s: float = np.radians(0.20)
    pitch_sigma_rad: float = np.radians(0.50)
    altitude_sigma_m: float = 1.00

    @classmethod
    def from_config(cls, config: dict) -> "SensorNoiseConfig":
        """Build noise config from optional project config values."""
        nav_cfg = config.get("navigation", {})
        sensor_cfg = nav_cfg.get("sensor_noise", {})

        return cls(
            airspeed_sigma_m_s=float(sensor_cfg.get("airspeed_sigma_m_s", 0.30)),
            alpha_sigma_rad=float(
                np.radians(sensor_cfg.get("alpha_sigma_deg", 0.50))
            ),
            gyro_q_sigma_rad_s=float(
                np.radians(sensor_cfg.get("gyro_q_sigma_deg_s", 0.20))
            ),
            pitch_sigma_rad=float(
                np.radians(sensor_cfg.get("pitch_sigma_deg", 0.50))
            ),
            altitude_sigma_m=float(sensor_cfg.get("altitude_sigma_m", 1.00)),
        )

    def covariance(self) -> np.ndarray:
        """Return measurement noise covariance R."""
        return np.diag(
            [
                self.airspeed_sigma_m_s**2,
                self.alpha_sigma_rad**2,
                self.gyro_q_sigma_rad_s**2,
                self.pitch_sigma_rad**2,
                self.altitude_sigma_m**2,
            ]
        )


@dataclass(frozen=True)
class SensorMeasurement:
    """One noisy longitudinal sensor measurement packet."""

    airspeed_m_s: float
    alpha_rad: float
    q_rad_s: float
    theta_rad: float
    altitude_m: float

    def as_array(self) -> np.ndarray:
        """Return measurement vector."""
        return np.array(
            [
                self.airspeed_m_s,
                self.alpha_rad,
                self.q_rad_s,
                self.theta_rad,
                self.altitude_m,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, z: np.ndarray) -> "SensorMeasurement":
        """Create measurement packet from vector."""
        return cls(
            airspeed_m_s=float(z[0]),
            alpha_rad=float(z[1]),
            q_rad_s=float(z[2]),
            theta_rad=float(z[3]),
            altitude_m=float(z[4]),
        )


def measurement_vector_from_state(state: LongitudinalState) -> np.ndarray:
    """Ideal measurement vector from a true or estimated state."""
    return np.array(
        [
            state.airspeed_m_s,
            state.alpha_rad,
            state.q_rad_s,
            state.theta_rad,
            state.h_m,
        ],
        dtype=float,
    )


def generate_longitudinal_measurement(
    state: LongitudinalState,
    noise_config: SensorNoiseConfig,
    rng: np.random.Generator,
) -> SensorMeasurement:
    """Generate a noisy longitudinal measurement from true state."""
    z_true = measurement_vector_from_state(state)

    noise = rng.normal(
        loc=0.0,
        scale=np.array(
            [
                noise_config.airspeed_sigma_m_s,
                noise_config.alpha_sigma_rad,
                noise_config.gyro_q_sigma_rad_s,
                noise_config.pitch_sigma_rad,
                noise_config.altitude_sigma_m,
            ],
            dtype=float,
        ),
    )

    return SensorMeasurement.from_array(z_true + noise)