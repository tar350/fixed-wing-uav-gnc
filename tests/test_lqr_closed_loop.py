"""Tests for nonlinear closed-loop LQR simulation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnc_sim.simulation.lqr_closed_loop import simulate_lqr_longitudinal


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_default_config() -> dict:
    """Load default project configuration."""
    with (PROJECT_ROOT / "configs" / "uav_longitudinal.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_lqr_closed_loop_simulation_runs() -> None:
    """The nonlinear LQR simulation should run and return expected columns."""
    config = load_default_config()

    # Keep the automated test fast.
    config["simulation"]["t_final_s"] = 2.0

    results = simulate_lqr_longitudinal(config)

    expected_columns = {
        "time_s",
        "u_m_s",
        "w_m_s",
        "airspeed_m_s",
        "theta_deg",
        "altitude_m",
        "altitude_cmd_m",
        "airspeed_cmd_m_s",
        "elevator_deg",
        "throttle",
        "altitude_error_m",
        "airspeed_error_m_s",
    }

    assert not results.empty
    assert expected_columns.issubset(set(results.columns))


def test_lqr_closed_loop_outputs_are_finite() -> None:
    """Closed-loop simulation should not generate NaN or infinite state values."""
    config = load_default_config()

    # Keep the automated test fast and near trim.
    config["simulation"]["t_final_s"] = 2.0
    config["simulation"]["commands"]["altitude_m"] = 105.0
    config["simulation"]["commands"]["airspeed_m_s"] = 22.0
    config["simulation"]["initial"]["airspeed_m_s"] = 22.0
    config["simulation"]["initial"]["alpha_rad"] = 0.12
    config["simulation"]["initial"]["theta_rad"] = 0.12
    config["simulation"]["initial"]["altitude_m"] = 100.0
    config["simulation"]["wind"]["enabled"] = False

    results = simulate_lqr_longitudinal(config)

    numeric_values = results[
        [
            "u_m_s",
            "w_m_s",
            "airspeed_m_s",
            "theta_deg",
            "altitude_m",
            "elevator_deg",
            "throttle",
        ]
    ].to_numpy()

    assert np.all(np.isfinite(numeric_values))