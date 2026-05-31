"""Tests for LQR controller design."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnc_sim.controllers import design_lqr
from gnc_sim.linearization import linearize_longitudinal
from gnc_sim.models.uav_longitudinal import LongitudinalAircraft
from gnc_sim.trim.longitudinal_trim import solve_longitudinal_trim


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_default_config() -> dict:
    """Load default project configuration."""
    with (PROJECT_ROOT / "configs" / "uav_longitudinal.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_lqr_gain_shape() -> None:
    """LQR gain matrix should have shape inputs x states."""
    config = load_default_config()
    aircraft = LongitudinalAircraft(config["aircraft"])

    trim = solve_longitudinal_trim(
        aircraft=aircraft,
        target_airspeed_m_s=22.0,
        target_altitude_m=100.0,
        elevator_bounds_rad=tuple(config["aircraft"]["limits"]["elevator_rad"]),
    )

    assert trim.success

    lin = linearize_longitudinal(
        aircraft=aircraft,
        trim_state=trim.state(),
        trim_elevator_rad=trim.elevator_rad,
        trim_throttle=trim.throttle,
    )

    Q = np.diag([1.0, 1.0, 20.0, 40.0, 0.20])
    R = np.diag([5.0, 2.0])

    result = design_lqr(
        A=lin.A,
        B=lin.B,
        Q=Q,
        R=R,
        state_names=lin.state_names,
        input_names=lin.input_names,
    )

    assert result.K.shape == (2, 5)
    assert result.closed_loop_A.shape == (5, 5)
    assert result.closed_loop_eigenvalues.shape == (5,)


def test_lqr_closed_loop_has_no_unstable_modes() -> None:
    """Closed-loop LQR eigenvalues should not have positive real parts."""
    config = load_default_config()
    aircraft = LongitudinalAircraft(config["aircraft"])

    trim = solve_longitudinal_trim(
        aircraft=aircraft,
        target_airspeed_m_s=22.0,
        target_altitude_m=100.0,
        elevator_bounds_rad=tuple(config["aircraft"]["limits"]["elevator_rad"]),
    )

    assert trim.success

    lin = linearize_longitudinal(
        aircraft=aircraft,
        trim_state=trim.state(),
        trim_elevator_rad=trim.elevator_rad,
        trim_throttle=trim.throttle,
    )

    Q = np.diag([1.0, 1.0, 20.0, 40.0, 0.20])
    R = np.diag([5.0, 2.0])

    result = design_lqr(
        A=lin.A,
        B=lin.B,
        Q=Q,
        R=R,
        state_names=lin.state_names,
        input_names=lin.input_names,
    )

    assert np.max(result.closed_loop_eigenvalues.real) < 1.0e-6