"""Tests for estimator-in-the-loop GNC simulation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnc_sim.simulation.estimator_in_loop import simulate_estimator_in_loop_longitudinal


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_default_config() -> dict:
    """Load default project configuration."""
    with (PROJECT_ROOT / "configs" / "uav_longitudinal.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_estimator_in_loop_simulation_runs() -> None:
    """Estimator-in-the-loop simulation should run and return expected columns."""
    config = load_default_config()

    config["simulation"]["t_final_s"] = 1.0
    config["simulation"]["wind"]["enabled"] = False

    results = simulate_estimator_in_loop_longitudinal(config)

    expected_columns = {
        "time_s",
        "true_altitude_m",
        "est_altitude_m",
        "meas_altitude_m",
        "true_airspeed_m_s",
        "est_airspeed_m_s",
        "meas_airspeed_m_s",
        "elevator_deg",
        "throttle",
        "altitude_error_m",
        "airspeed_error_m_s",
        "error_altitude_m",
        "error_airspeed_m_s",
    }

    assert not results.empty
    assert expected_columns.issubset(set(results.columns))


def test_estimator_in_loop_values_are_finite() -> None:
    """Estimator-in-loop truth, estimate, and control values should stay finite."""
    config = load_default_config()

    config["simulation"]["t_final_s"] = 1.0
    config["simulation"]["wind"]["enabled"] = False

    results = simulate_estimator_in_loop_longitudinal(config)

    numeric = results[
        [
            "true_u_m_s",
            "true_w_m_s",
            "true_q_rad_s",
            "true_theta_rad",
            "true_altitude_m",
            "est_u_m_s",
            "est_w_m_s",
            "est_q_rad_s",
            "est_theta_rad",
            "est_altitude_m",
            "elevator_deg",
            "throttle",
        ]
    ].to_numpy()

    assert np.all(np.isfinite(numeric))


def test_controller_uses_estimator_columns() -> None:
    """Estimator-in-loop result should include measurement and estimation errors."""
    config = load_default_config()

    config["simulation"]["t_final_s"] = 0.5
    config["simulation"]["wind"]["enabled"] = False

    results = simulate_estimator_in_loop_longitudinal(config)

    assert "meas_theta_deg" in results.columns
    assert "est_theta_deg" in results.columns
    assert "error_theta_deg" in results.columns
    assert "error_alpha_deg" in results.columns