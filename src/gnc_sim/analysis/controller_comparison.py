"""Controller comparison utilities for PID and LQR longitudinal control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from gnc_sim.simulation.integrator import simulate_longitudinal
from gnc_sim.simulation.lqr_closed_loop import simulate_lqr_longitudinal


@dataclass(frozen=True)
class ControllerMetrics:
    """Summary metrics for one controller run."""

    controller: str
    final_altitude_error_m: float
    final_airspeed_error_m_s: float
    max_altitude_error_m: float
    rms_altitude_error_m: float
    max_airspeed_error_m_s: float
    rms_airspeed_error_m_s: float
    rms_elevator_deg: float
    rms_throttle: float
    max_abs_elevator_deg: float
    max_throttle: float
    settling_time_altitude_s: float | None
    settling_time_airspeed_s: float | None

    def to_dict(self) -> dict:
        """Return JSON-serializable metric dictionary."""
        return {
            "controller": self.controller,
            "final_altitude_error_m": self.final_altitude_error_m,
            "final_airspeed_error_m_s": self.final_airspeed_error_m_s,
            "max_altitude_error_m": self.max_altitude_error_m,
            "rms_altitude_error_m": self.rms_altitude_error_m,
            "max_airspeed_error_m_s": self.max_airspeed_error_m_s,
            "rms_airspeed_error_m_s": self.rms_airspeed_error_m_s,
            "rms_elevator_deg": self.rms_elevator_deg,
            "rms_throttle": self.rms_throttle,
            "max_abs_elevator_deg": self.max_abs_elevator_deg,
            "max_throttle": self.max_throttle,
            "settling_time_altitude_s": self.settling_time_altitude_s,
            "settling_time_airspeed_s": self.settling_time_airspeed_s,
        }


@dataclass(frozen=True)
class ControllerComparisonResult:
    """Container for PID vs LQR comparison output."""

    pid_results: pd.DataFrame
    lqr_results: pd.DataFrame
    metrics: list[ControllerMetrics]

    def metrics_dataframe(self) -> pd.DataFrame:
        """Return controller metrics as a dataframe."""
        return pd.DataFrame([metric.to_dict() for metric in self.metrics])

    def save_metrics_json(self, output_path: str | Path) -> None:
        """Save metrics to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"metrics": [metric.to_dict() for metric in self.metrics]},
                file,
                indent=4,
            )


def _ensure_error_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Ensure altitude and airspeed error columns exist."""
    results = results.copy()

    if "altitude_error_m" not in results.columns:
        results["altitude_error_m"] = results["altitude_cmd_m"] - results["altitude_m"]

    if "airspeed_error_m_s" not in results.columns:
        results["airspeed_error_m_s"] = (
            results["airspeed_cmd_m_s"] - results["airspeed_m_s"]
        )

    return results


def _settling_time(
    time_s: np.ndarray,
    error: np.ndarray,
    tolerance: float,
) -> float | None:
    """Compute first time after which absolute error stays within tolerance."""
    abs_error = np.abs(error)

    for idx in range(len(abs_error)):
        if np.all(abs_error[idx:] <= tolerance):
            return float(time_s[idx])

    return None


def compute_controller_metrics(
    controller_name: str,
    results: pd.DataFrame,
    altitude_tolerance_m: float = 2.0,
    airspeed_tolerance_m_s: float = 0.5,
) -> ControllerMetrics:
    """Compute tracking and control effort metrics for one controller."""
    results = _ensure_error_columns(results)

    altitude_error = results["altitude_error_m"].to_numpy(dtype=float)
    airspeed_error = results["airspeed_error_m_s"].to_numpy(dtype=float)
    time_s = results["time_s"].to_numpy(dtype=float)

    elevator = results["elevator_deg"].to_numpy(dtype=float)
    throttle = results["throttle"].to_numpy(dtype=float)

    return ControllerMetrics(
        controller=controller_name,
        final_altitude_error_m=float(altitude_error[-1]),
        final_airspeed_error_m_s=float(airspeed_error[-1]),
        max_altitude_error_m=float(np.max(np.abs(altitude_error))),
        rms_altitude_error_m=float(np.sqrt(np.mean(altitude_error**2))),
        max_airspeed_error_m_s=float(np.max(np.abs(airspeed_error))),
        rms_airspeed_error_m_s=float(np.sqrt(np.mean(airspeed_error**2))),
        rms_elevator_deg=float(np.sqrt(np.mean(elevator**2))),
        rms_throttle=float(np.sqrt(np.mean(throttle**2))),
        max_abs_elevator_deg=float(np.max(np.abs(elevator))),
        max_throttle=float(np.max(throttle)),
        settling_time_altitude_s=_settling_time(
            time_s=time_s,
            error=altitude_error,
            tolerance=altitude_tolerance_m,
        ),
        settling_time_airspeed_s=_settling_time(
            time_s=time_s,
            error=airspeed_error,
            tolerance=airspeed_tolerance_m_s,
        ),
    )


def run_pid_lqr_comparison(config: dict) -> ControllerComparisonResult:
    """Run PID and LQR simulations using the same scenario config."""
    pid_results = _ensure_error_columns(simulate_longitudinal(config))
    lqr_results = _ensure_error_columns(simulate_lqr_longitudinal(config))

    metrics = [
        compute_controller_metrics("PID", pid_results),
        compute_controller_metrics("LQR", lqr_results),
    ]

    return ControllerComparisonResult(
        pid_results=pid_results,
        lqr_results=lqr_results,
        metrics=metrics,
    )