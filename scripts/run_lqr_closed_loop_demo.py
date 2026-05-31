"""Run nonlinear closed-loop longitudinal simulation using LQR.

Execute from project root:

    python scripts/run_lqr_closed_loop_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gnc_sim.config import load_json_config
from gnc_sim.simulation.lqr_closed_loop import simulate_lqr_longitudinal
from gnc_sim.visualization import animate_longitudinal_response


def _save_lqr_response_plots(results, output_dir: Path) -> None:
    """Save standard LQR response plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Altitude response
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["altitude_m"], label="Altitude")
    ax.plot(results["time_s"], results["altitude_cmd_m"], "--", label="Command")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("LQR Closed-Loop Altitude Response")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lqr_closed_loop_altitude_v0_6.png", dpi=200)
    plt.close(fig)

    # Airspeed response
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["airspeed_m_s"], label="Airspeed")
    ax.plot(results["time_s"], results["airspeed_cmd_m_s"], "--", label="Command")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Airspeed [m/s]")
    ax.set_title("LQR Closed-Loop Airspeed Response")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lqr_closed_loop_airspeed_v0_6.png", dpi=200)
    plt.close(fig)

    # Pitch response
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["theta_deg"], label="Pitch Angle")
    ax.plot(results["time_s"], results["theta_cmd_deg"], "--", label="Trim Pitch Reference")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pitch Angle [deg]")
    ax.set_title("LQR Closed-Loop Pitch Response")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lqr_closed_loop_pitch_v0_6.png", dpi=200)
    plt.close(fig)

    # Control response
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["elevator_deg"], label="Elevator [deg]")
    ax.plot(results["time_s"], results["throttle"], label="Throttle [-]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Control Input")
    ax.set_title("LQR Closed-Loop Control Inputs")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lqr_closed_loop_controls_v0_6.png", dpi=200)
    plt.close(fig)

    # Tracking error
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["altitude_error_m"], label="Altitude Error [m]")
    ax.plot(results["time_s"], results["airspeed_error_m_s"], label="Airspeed Error [m/s]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tracking Error")
    ax.set_title("LQR Closed-Loop Tracking Error")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lqr_closed_loop_tracking_error_v0_6.png", dpi=200)
    plt.close(fig)


def _print_summary(results) -> None:
    """Print closed-loop performance summary."""
    final = results.iloc[-1]

    max_altitude_error = results["altitude_error_m"].abs().max()
    max_airspeed_error = results["airspeed_error_m_s"].abs().max()

    rms_altitude_error = (results["altitude_error_m"] ** 2).mean() ** 0.5
    rms_airspeed_error = (results["airspeed_error_m_s"] ** 2).mean() ** 0.5

    print("Nonlinear closed-loop LQR simulation")
    print("=" * 56)
    print(f"Final altitude: {final['altitude_m']:.3f} m")
    print(f"Final airspeed: {final['airspeed_m_s']:.3f} m/s")
    print(f"Final pitch: {final['theta_deg']:.3f} deg")
    print(f"Final elevator: {final['elevator_deg']:.3f} deg")
    print(f"Final throttle: {final['throttle']:.4f}")
    print()
    print("Tracking metrics")
    print("-" * 56)
    print(f"Max altitude error: {max_altitude_error:.3f} m")
    print(f"RMS altitude error: {rms_altitude_error:.3f} m")
    print(f"Max airspeed error: {max_airspeed_error:.3f} m/s")
    print(f"RMS airspeed error: {rms_airspeed_error:.3f} m/s")


def main() -> None:
    """Run nonlinear LQR simulation and save artifacts."""
    config = load_json_config(PROJECT_ROOT / "configs" / "uav_longitudinal.json")

    results = simulate_lqr_longitudinal(config)

    output_log = PROJECT_ROOT / "outputs" / "logs" / "lqr_closed_loop_results_v0_6.csv"
    output_plot_dir = PROJECT_ROOT / "outputs" / "plots"
    output_animation = PROJECT_ROOT / "outputs" / "animations" / "lqr_closed_loop_motion_v0_6.gif"

    output_log.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_log, index=False)

    _save_lqr_response_plots(results, output_plot_dir)

    animate_longitudinal_response(
        results=results,
        output_path=output_animation,
        frame_step=5,
        interval_ms=40,
        aircraft_length_m=8.0,
    )

    _print_summary(results)

    print()
    print("Saved artifacts")
    print("-" * 56)
    print(f"CSV: {output_log}")
    print(f"Plots: {output_plot_dir}")
    print(f"Animation: {output_animation}")


if __name__ == "__main__":
    main()