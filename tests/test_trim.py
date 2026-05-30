"""Tests for longitudinal trim solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnc_sim.models.uav_longitudinal import LongitudinalAircraft
from gnc_sim.trim.longitudinal_trim import solve_longitudinal_trim


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_default_config() -> dict:
    """Load default project configuration."""
    with (PROJECT_ROOT / "configs" / "uav_longitudinal.json").open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def test_longitudinal_trim_converges_for_level_flight() -> None:
    """Trim should converge for the default level-flight case."""
    config = load_default_config()
    aircraft = LongitudinalAircraft(config["aircraft"])
    trim = solve_longitudinal_trim(
        aircraft=aircraft,
        target_airspeed_m_s=22.0,
        target_altitude_m=100.0,
        elevator_bounds_rad=tuple(config["aircraft"]["limits"]["elevator_rad"]),
    )

    assert trim.success
    assert trim.residual_norm < 1.0e-6
    assert 0.0 <= trim.throttle <= 1.0
    assert np.radians(-25.0) <= trim.elevator_rad <= np.radians(25.0)


def test_trim_state_has_requested_airspeed() -> None:
    """Returned trim state should match the requested airspeed."""
    config = load_default_config()
    aircraft = LongitudinalAircraft(config["aircraft"])
    trim = solve_longitudinal_trim(
        aircraft=aircraft,
        target_airspeed_m_s=22.0,
        target_altitude_m=100.0,
        elevator_bounds_rad=tuple(config["aircraft"]["limits"]["elevator_rad"]),
    )

    assert np.isclose(trim.state().airspeed_m_s, 22.0)
    assert np.isclose(trim.state().h_m, 100.0)
