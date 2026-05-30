"""Tests for numerical longitudinal linearization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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


def get_default_trim_and_aircraft():
    """Return the default aircraft and its level-flight trim result."""
    config = load_default_config()
    aircraft = LongitudinalAircraft(config["aircraft"])

    trim = solve_longitudinal_trim(
        aircraft=aircraft,
        target_airspeed_m_s=22.0,
        target_altitude_m=100.0,
        elevator_bounds_rad=tuple(config["aircraft"]["limits"]["elevator_rad"]),
    )

    assert trim.success

    return aircraft, trim


def test_linearization_matrix_shapes() -> None:
    """A and B should match the expected longitudinal state/input dimensions."""
    aircraft, trim = get_default_trim_and_aircraft()

    lin = linearize_longitudinal(
        aircraft=aircraft,
        trim_state=trim.state(),
        trim_elevator_rad=trim.elevator_rad,
        trim_throttle=trim.throttle,
    )

    assert lin.A.shape == (5, 5)
    assert lin.B.shape == (5, 2)
    assert lin.eigenvalues.shape == (5,)


def test_linear_model_matches_small_nonlinear_perturbation() -> None:
    """Linearized dynamics should approximate nearby nonlinear dynamics."""
    aircraft, trim = get_default_trim_and_aircraft()

    lin = linearize_longitudinal(
        aircraft=aircraft,
        trim_state=trim.state(),
        trim_elevator_rad=trim.elevator_rad,
        trim_throttle=trim.throttle,
    )

    x0 = trim.state().as_array()
    u0 = np.array([trim.elevator_rad, trim.throttle], dtype=float)
    f0 = aircraft.dynamics(trim.state(), trim.elevator_rad, trim.throttle)

    dx = np.array([0.02, -0.01, 0.0005, 0.0004, 0.1], dtype=float)
    du = np.array([0.0005, -0.001], dtype=float)

    x_perturbed = x0 + dx
    u_perturbed = u0 + du

    f_nonlinear = aircraft.dynamics(
        state=trim.state().from_array(x_perturbed),
        elevator_rad=float(u_perturbed[0]),
        throttle=float(u_perturbed[1]),
    )

    f_linear = f0 + lin.A @ dx + lin.B @ du

    assert np.allclose(f_nonlinear, f_linear, atol=5.0e-3)