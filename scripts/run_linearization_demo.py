"""Run trim and numerical linearization about the trim point.

Execute from the project root:

    python scripts/run_linearization_demo.py
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
from gnc_sim.linearization import linearize_longitudinal
from gnc_sim.models.uav_longitudinal import LongitudinalAircraft
from gnc_sim.trim.longitudinal_trim import solve_longitudinal_trim


def _print_matrix(name: str, matrix: np.ndarray) -> None:
    """Print a matrix with clean formatting."""
    print(f"\n{name} =")
    with np.printoptions(precision=6, suppress=True):
        print(matrix)


def main() -> None:
    """Solve trim, linearize the aircraft, print and save results."""
    config = load_json_config(PROJECT_ROOT / "configs" / "uav_longitudinal.json")
    aircraft = LongitudinalAircraft(config["aircraft"])

    trim_cfg = config["trim"]

    trim_result = solve_longitudinal_trim(
        aircraft=aircraft,
        target_airspeed_m_s=trim_cfg["target_airspeed_m_s"],
        target_altitude_m=trim_cfg["target_altitude_m"],
        flight_path_angle_rad=np.radians(trim_cfg["flight_path_angle_deg"]),
        initial_guess=(
            trim_cfg["initial_guess"]["alpha_rad"],
            trim_cfg["initial_guess"]["elevator_rad"],
            trim_cfg["initial_guess"]["throttle"],
        ),
        elevator_bounds_rad=tuple(config["aircraft"]["limits"]["elevator_rad"]),
        residual_tolerance=trim_cfg["residual_tolerance"],
    )

    if not trim_result.success:
        raise RuntimeError(f"Trim failed: {trim_result.message}")

    linearization = linearize_longitudinal(
        aircraft=aircraft,
        trim_state=trim_result.state(),
        trim_elevator_rad=trim_result.elevator_rad,
        trim_throttle=trim_result.throttle,
    )

    # Save linearization artifacts for the digital thread
    output_json = PROJECT_ROOT / "outputs" / "logs" / "linearization_result_v0_3.json"
    output_npz = PROJECT_ROOT / "outputs" / "logs" / "state_space_matrices_v0_3.npz"

    output_json.parent.mkdir(parents=True, exist_ok=True)

    linearization.save_json(output_json)

    np.savez(
        output_npz,
        A=linearization.A,
        B=linearization.B,
        eigenvalues=linearization.eigenvalues,
    )

    print("Longitudinal linearization result")
    print("=" * 36)
    print(f"Trim alpha: {trim_result.alpha_deg:.3f} deg")
    print(f"Trim elevator: {trim_result.elevator_deg:.3f} deg")
    print(f"Trim throttle: {trim_result.throttle:.4f}")
    print("State vector: [u_m_s, w_m_s, q_rad_s, theta_rad, h_m]")
    print("Input vector: [elevator_rad, throttle]")

    _print_matrix("A", linearization.A)
    _print_matrix("B", linearization.B)

    print("\nEigenvalues of A:")

    for idx, eig in enumerate(linearization.eigenvalues, start=1):
        print(f"  lambda_{idx}: {eig.real:+.6f} {eig.imag:+.6f}j")

    print(f"\nSaved JSON: {output_json}")
    print(f"Saved NPZ: {output_npz}")
    print(f"JSON exists: {output_json.exists()}")
    print(f"NPZ exists: {output_npz.exists()}")


if __name__ == "__main__":
    main()