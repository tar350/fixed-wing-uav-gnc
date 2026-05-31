"""Tests for longitudinal EKF navigation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnc_sim.models.uav_longitudinal import LongitudinalAircraft, LongitudinalState
from gnc_sim.navigation import LongitudinalEKF
from gnc_sim.sensors import SensorNoiseConfig, measurement_vector_from_state
from gnc_sim.simulation.ekf_navigation import run_ekf_navigation_simulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_default_config() -> dict:
    """Load default project configuration."""
    with (PROJECT_ROOT / "configs" / "uav_longitudinal.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_measurement_vector_shape() -> None:
    """Longitudinal measurement vector should have expected dimension."""
    state = LongitudinalState(
        u_m_s=22.0,
        w_m_s=2.0,
        q_rad_s=0.01,
        theta_rad=0.10,
        h_m=100.0,
    )

    z = measurement_vector_from_state(state)

    assert z.shape == (5,)
    assert np.all(np.isfinite(z))


def test_ekf_predict_update_runs() -> None:
    """EKF predict and update should run without numerical issues."""
    config = load_default_config()
    aircraft = LongitudinalAircraft(config["aircraft"])

    state = LongitudinalState(
        u_m_s=22.0,
        w_m_s=2.0,
        q_rad_s=0.01,
        theta_rad=0.10,
        h_m=100.0,
    )

    sensor_noise = SensorNoiseConfig()

    ekf = LongitudinalEKF(
        aircraft=aircraft,
        x0=state.as_array(),
        measurement_covariance=sensor_noise.covariance(),
    )

    ekf.predict(
        elevator_rad=0.0,
        throttle=0.3,
        dt_s=0.01,
    )

    z = measurement_vector_from_state(state)

    ekf.update(z)

    assert ekf.x.shape == (5,)
    assert ekf.P.shape == (5, 5)
    assert np.all(np.isfinite(ekf.x))
    assert np.all(np.isfinite(ekf.P))


def test_ekf_navigation_simulation_runs() -> None:
    """Full EKF navigation simulation should return expected columns."""
    config = load_default_config()

    config["simulation"]["t_final_s"] = 1.0
    config["simulation"]["wind"]["enabled"] = False

    results = run_ekf_navigation_simulation(config)

    expected_columns = {
        "time_s",
        "true_altitude_m",
        "est_altitude_m",
        "meas_altitude_m",
        "true_airspeed_m_s",
        "est_airspeed_m_s",
        "meas_airspeed_m_s",
        "error_altitude_m",
        "error_airspeed_m_s",
    }

    assert not results.empty
    assert expected_columns.issubset(set(results.columns))

    numeric = results[
        [
            "est_u_m_s",
            "est_w_m_s",
            "est_q_rad_s",
            "est_theta_rad",
            "est_altitude_m",
        ]
    ].to_numpy()

    assert np.all(np.isfinite(numeric))