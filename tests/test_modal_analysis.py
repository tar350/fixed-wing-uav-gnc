"""Tests for longitudinal modal analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnc_sim.analysis import analyze_modes
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


def test_modal_analysis_detects_expected_mode_count() -> None:
    """The 5-state longitudinal model should produce five modal entries."""
    eigenvalues = np.array(
        [
            0.0 + 0.0j,
            -1.2 + 3.1j,
            -1.2 - 3.1j,
            -0.01 + 0.6j,
            -0.01 - 0.6j,
        ],
        dtype=complex,
    )

    result = analyze_modes(eigenvalues)

    assert result.num_modes == 5
    assert len(result.modes) == 5
    assert result.has_near_integrator
    assert result.num_unstable_modes == 0
    assert result.is_open_loop_stable
    assert not result.is_asymptotically_stable


def test_modal_analysis_classifies_short_period_and_phugoid_candidates() -> None:
    """Oscillatory modes should be classified by approximate frequency range."""
    eigenvalues = np.array(
        [
            -1.2 + 3.1j,
            -1.2 - 3.1j,
            -0.01 + 0.6j,
            -0.01 - 0.6j,
        ],
        dtype=complex,
    )

    result = analyze_modes(eigenvalues)

    mode_types = [mode.mode_type for mode in result.modes]

    assert "oscillatory_short_period_candidate" in mode_types
    assert "oscillatory_phugoid_candidate" in mode_types


def test_modal_analysis_with_project_linearization() -> None:
    """Modal analysis should run on the project aircraft trim/linearization."""
    config = load_default_config()
    aircraft = LongitudinalAircraft(config["aircraft"])

    trim = solve_longitudinal_trim(
        aircraft=aircraft,
        target_airspeed_m_s=22.0,
        target_altitude_m=100.0,
        elevator_bounds_rad=tuple(config["aircraft"]["limits"]["elevator_rad"]),
    )

    assert trim.success

    linearization = linearize_longitudinal(
        aircraft=aircraft,
        trim_state=trim.state(),
        trim_elevator_rad=trim.elevator_rad,
        trim_throttle=trim.throttle,
    )

    modal_analysis = analyze_modes(linearization.eigenvalues)

    assert modal_analysis.num_modes == 5
    assert modal_analysis.num_unstable_modes == 0
    assert modal_analysis.is_open_loop_stable