"""Run trim, linearization, and modal stability analysis.

Execute from the project root:

    python scripts/run_modal_analysis_demo.py
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
from gnc_sim.linearization import linearize_longitudinal
from gnc_sim.models.uav_longitudinal import LongitudinalAircraft
from gnc_sim.trim.longitudinal_trim import solve_longitudinal_trim


def _print_mode_table(analysis_result) -> None:
    """Print a clean modal analysis table."""
    print("\nMode summary")
    print("-" * 104)
    print(
        f"{'Mode':>4} | {'Real':>11} | {'Imag':>11} | {'wn [rad/s]':>11} | "
        f"{'zeta':>9} | {'Period [s]':>11} | {'Stability':>9} | {'Type':>35}"
    )
    print("-" * 104)

    for mode in analysis_result.modes:
        zeta = "N/A" if mode.damping_ratio is None else f"{mode.damping_ratio:.4f}"
        period = "N/A" if mode.period_s is None else f"{mode.period_s:.3f}"

        print(
            f"{mode.index:>4} | "
            f"{mode.real:>+11.6f} | "
            f"{mode.imag:>+11.6f} | "
            f"{mode.natural_frequency_rad_s:>11.6f} | "
            f"{zeta:>9} | "
            f"{period:>11} | "
            f"{mode.stability:>9} | "
            f"{mode.mode_type:>35}"
        )


def _save_eigenvalue_plot(eigenvalues: np.ndarray, output_path: Path) -> None:
    """Save a simple eigenvalue map in the complex plane."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.scatter(eigenvalues.real, eigenvalues.imag, marker="x", s=80)

    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)

    ax.set_xlabel("Real axis [1/s]")
    ax.set_ylabel("Imaginary axis [rad/s]")
    ax.set_title("Longitudinal Open-Loop Eigenvalue Map")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run modal analysis workflow."""
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

    modal_analysis = analyze_modes(linearization.eigenvalues)

    output_json = PROJECT_ROOT / "outputs" / "logs" / "modal_analysis_result_v0_4.json"
    output_csv = PROJECT_ROOT / "outputs" / "logs" / "modal_analysis_result_v0_4.csv"
    output_plot = PROJECT_ROOT / "outputs" / "figures" / "eigenvalue_map_v0_4.png"

    modal_analysis.save_json(output_json)
    modal_analysis.save_csv(output_csv)
    _save_eigenvalue_plot(linearization.eigenvalues, output_plot)

    print("Longitudinal stability and mode analysis")
    print("=" * 48)
    print(f"Trim airspeed: {trim_result.state().airspeed_m_s:.3f} m/s")
    print(f"Trim alpha: {trim_result.alpha_deg:.3f} deg")
    print(f"Trim elevator: {trim_result.elevator_deg:.3f} deg")
    print(f"Trim throttle: {trim_result.throttle:.4f}")

    _print_mode_table(modal_analysis)

    print("\nOverall stability summary")
    print("-" * 48)
    print(f"Number of modes: {modal_analysis.num_modes}")
    print(f"Stable modes: {modal_analysis.num_stable_modes}")
    print(f"Neutral modes: {modal_analysis.num_neutral_modes}")
    print(f"Unstable modes: {modal_analysis.num_unstable_modes}")
    print(f"Open-loop stable: {modal_analysis.is_open_loop_stable}")
    print(f"Asymptotically stable: {modal_analysis.is_asymptotically_stable}")
    print(f"Near-integrator present: {modal_analysis.has_near_integrator}")

    if modal_analysis.notes:
        print("\nNotes")
        print("-" * 48)

        for note in modal_analysis.notes:
            print(f"- {note}")

    print("\nSaved artifacts")
    print("-" * 48)
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")
    print(f"Plot: {output_plot}")


if __name__ == "__main__":
    main()
