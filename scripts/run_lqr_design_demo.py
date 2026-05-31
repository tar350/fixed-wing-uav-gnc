"""Run longitudinal trim, linearization, modal analysis, and LQR design.

Execute from project root:

    python scripts/run_lqr_design_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gnc_sim.analysis import analyze_modes
from gnc_sim.config import load_json_config
from gnc_sim.controllers import design_lqr
from gnc_sim.linearization import linearize_longitudinal
from gnc_sim.models.uav_longitudinal import LongitudinalAircraft
from gnc_sim.trim.longitudinal_trim import solve_longitudinal_trim


def _save_eigenvalue_comparison_plot(
    open_loop_eigs: np.ndarray,
    closed_loop_eigs: np.ndarray,
    output_path: Path,
) -> None:
    """Save open-loop versus closed-loop eigenvalue comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.scatter(
        open_loop_eigs.real,
        open_loop_eigs.imag,
        marker="x",
        s=90,
        label="Open loop",
    )
    ax.scatter(
        closed_loop_eigs.real,
        closed_loop_eigs.imag,
        marker="o",
        s=60,
        label="Closed loop LQR",
    )

    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)

    ax.set_xlabel("Real axis [1/s]")
    ax.set_ylabel("Imaginary axis [rad/s]")
    ax.set_title("Open-Loop vs LQR Closed-Loop Eigenvalues")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _print_gain_matrix(K: np.ndarray, state_names: tuple[str, ...], input_names: tuple[str, ...]) -> None:
    """Print the LQR gain matrix with state/input labels."""
    print("\nLQR gain matrix K")
    print("-" * 72)
    print("Control law: u_delta = -K x_delta")
    print(f"States: {state_names}")
    print(f"Inputs: {input_names}")

    with np.printoptions(precision=6, suppress=True):
        print(K)


def _print_eigenvalues(title: str, eigenvalues: np.ndarray) -> None:
    """Print eigenvalues in clean format."""
    print(f"\n{title}")
    print("-" * 72)

    for idx, eig in enumerate(eigenvalues, start=1):
        print(f"lambda_{idx}: {eig.real:+.6f} {eig.imag:+.6f}j")


def main() -> None:
    """Run the v0.5 LQR design workflow."""
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

    # State vector:
    # x = [u_m_s, w_m_s, q_rad_s, theta_rad, h_m]
    #
    # Input vector:
    # u_input = [elevator_rad, throttle]
    #
    # These Q/R values are first-pass engineering weights.
    # We intentionally penalize pitch rate, pitch angle, and altitude more strongly.
    Q = np.diag([1.0, 1.0, 20.0, 40.0, 0.20])
    R = np.diag([5.0, 2.0])

    lqr_result = design_lqr(
        A=linearization.A,
        B=linearization.B,
        Q=Q,
        R=R,
        state_names=linearization.state_names,
        input_names=linearization.input_names,
    )

    open_loop_modes = analyze_modes(lqr_result.open_loop_eigenvalues)
    closed_loop_modes = analyze_modes(lqr_result.closed_loop_eigenvalues)

    output_json = PROJECT_ROOT / "outputs" / "logs" / "lqr_design_result_v0_5.json"
    output_plot = PROJECT_ROOT / "outputs" / "figures" / "lqr_open_closed_eigenvalues_v0_5.png"

    lqr_result.save_json(output_json)
    _save_eigenvalue_comparison_plot(
        open_loop_eigs=lqr_result.open_loop_eigenvalues,
        closed_loop_eigs=lqr_result.closed_loop_eigenvalues,
        output_path=output_plot,
    )

    print("Longitudinal LQR controller design")
    print("=" * 72)
    print(f"Trim airspeed: {trim_result.state().airspeed_m_s:.3f} m/s")
    print(f"Trim alpha: {trim_result.alpha_deg:.3f} deg")
    print(f"Trim elevator: {trim_result.elevator_deg:.3f} deg")
    print(f"Trim throttle: {trim_result.throttle:.4f}")

    _print_gain_matrix(
        K=lqr_result.K,
        state_names=lqr_result.state_names,
        input_names=lqr_result.input_names,
    )

    _print_eigenvalues("Open-loop eigenvalues", lqr_result.open_loop_eigenvalues)
    _print_eigenvalues("Closed-loop LQR eigenvalues", lqr_result.closed_loop_eigenvalues)

    print("\nStability comparison")
    print("-" * 72)
    print(f"Open-loop unstable modes: {open_loop_modes.num_unstable_modes}")
    print(f"Closed-loop unstable modes: {closed_loop_modes.num_unstable_modes}")
    print(f"Open-loop asymptotically stable: {open_loop_modes.is_asymptotically_stable}")
    print(f"Closed-loop asymptotically stable: {closed_loop_modes.is_asymptotically_stable}")

    print("\nSaved artifacts")
    print("-" * 72)
    print(f"JSON: {output_json}")
    print(f"Plot: {output_plot}")


if __name__ == "__main__":
    main()