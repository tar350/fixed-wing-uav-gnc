"""Run a longitudinal trim demo and save the result.

Execute from the project root:

    python scripts/run_trim_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gnc_sim.config import load_json_config
from gnc_sim.models.uav_longitudinal import LongitudinalAircraft
from gnc_sim.trim.longitudinal_trim import solve_longitudinal_trim


def main() -> None:
    """Solve and save the default level-flight trim condition."""
    config = load_json_config(PROJECT_ROOT / "configs" / "uav_longitudinal.json")
    aircraft = LongitudinalAircraft(config["aircraft"])

    trim_cfg = config["trim"]
    target_airspeed_m_s = trim_cfg["target_airspeed_m_s"]
    target_altitude_m = trim_cfg["target_altitude_m"]
    flight_path_angle_rad = np.radians(trim_cfg["flight_path_angle_deg"])
    initial_guess = (
        trim_cfg["initial_guess"]["alpha_rad"],
        trim_cfg["initial_guess"]["elevator_rad"],
        trim_cfg["initial_guess"]["throttle"],
    )
    elevator_limits = tuple(config["aircraft"]["limits"]["elevator_rad"])

    trim_result = solve_longitudinal_trim(
        aircraft=aircraft,
        target_airspeed_m_s=target_airspeed_m_s,
        target_altitude_m=target_altitude_m,
        flight_path_angle_rad=flight_path_angle_rad,
        initial_guess=initial_guess,
        elevator_bounds_rad=elevator_limits,
        residual_tolerance=trim_cfg["residual_tolerance"],
    )

    output_path = PROJECT_ROOT / "outputs" / "logs" / "trim_result_v0_2.json"
    trim_result.save_json(output_path)

    print("Longitudinal trim result")
    print("=" * 28)
    print(f"Success: {trim_result.success}")
    print(f"Message: {trim_result.message}")
    print(f"Target airspeed: {trim_result.target_airspeed_m_s:.3f} m/s")
    print(f"Target altitude: {trim_result.target_altitude_m:.3f} m")
    print(f"Flight-path angle: {trim_result.flight_path_angle_deg:.3f} deg")
    print(f"Alpha trim: {trim_result.alpha_deg:.3f} deg")
    print(f"Theta trim: {trim_result.theta_deg:.3f} deg")
    print(f"Elevator trim: {trim_result.elevator_deg:.3f} deg")
    print(f"Throttle trim: {trim_result.throttle:.4f}")
    print("Residuals:")
    print(f"  u_dot: {trim_result.residual_u_dot_m_s2:.6e} m/s^2")
    print(f"  w_dot: {trim_result.residual_w_dot_m_s2:.6e} m/s^2")
    print(f"  q_dot: {trim_result.residual_q_dot_rad_s2:.6e} rad/s^2")
    print(f"  h_dot residual: {trim_result.residual_h_dot_m_s:.6e} m/s")
    print(f"  norm: {trim_result.residual_norm:.6e}")
    print(f"Saved JSON: {output_path}")

    if not trim_result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
