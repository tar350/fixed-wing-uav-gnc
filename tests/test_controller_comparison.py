"""Tests for PID vs LQR controller comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnc_sim.analysis import compute_controller_metrics, run_pid_lqr_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_default_config() -> dict:
    """Load default project configuration."""
    with (PROJECT_ROOT / "configs" / "uav_longitudinal.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_controller_comparison_runs() -> None:
    """PID vs LQR comparison should run and return both result sets."""
    config = load_default_config()

    # Keep unit test fast.
    config["simulation"]["t_final_s"] = 2.0
    config["simulation"]["wind"]["enabled"] = False

    comparison = run_pid_lqr_comparison(config)

    assert not comparison.pid_results.empty
    assert not comparison.lqr_results.empty
    assert len(comparison.metrics) == 2

    controllers = {metric.controller for metric in comparison.metrics}

    assert controllers == {"PID", "LQR"}


def test_controller_metrics_are_finite() -> None:
    """Controller metrics should be finite where applicable."""
    config = load_default_config()

    config["simulation"]["t_final_s"] = 2.0
    config["simulation"]["wind"]["enabled"] = False

    comparison = run_pid_lqr_comparison(config)

    for metric in comparison.metrics:
        values = [
            metric.final_altitude_error_m,
            metric.final_airspeed_error_m_s,
            metric.max_altitude_error_m,
            metric.rms_altitude_error_m,
            metric.max_airspeed_error_m_s,
            metric.rms_airspeed_error_m_s,
            metric.rms_elevator_deg,
            metric.rms_throttle,
            metric.max_abs_elevator_deg,
            metric.max_throttle,
        ]

        assert np.all(np.isfinite(values))


def test_compute_controller_metrics_accepts_error_columns() -> None:
    """Metric computation should work with comparison result columns."""
    config = load_default_config()

    config["simulation"]["t_final_s"] = 1.0
    config["simulation"]["wind"]["enabled"] = False

    comparison = run_pid_lqr_comparison(config)

    metric = compute_controller_metrics("PID", comparison.pid_results)

    assert metric.controller == "PID"
    assert metric.max_altitude_error_m >= 0.0