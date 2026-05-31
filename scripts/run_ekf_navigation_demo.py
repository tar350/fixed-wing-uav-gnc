"""Run EKF navigation demo for longitudinal fixed-wing UAV simulation.

Execute from project root:

    python scripts/run_ekf_navigation_demo.py
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
from gnc_sim.simulation.ekf_navigation import run_ekf_navigation_simulation


def _save_estimation_plots(results, output_dir: Path) -> None:
    """Save EKF truth/measurement/estimate plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Altitude estimate
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["true_altitude_m"], label="True altitude")
    ax.plot(results["time_s"], results["est_altitude_m"], label="EKF estimate")
    ax.scatter(
        results["time_s"],
        results["meas_altitude_m"],
        s=4,
        alpha=0.35,
        label="Barometer measurement",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("EKF Altitude Estimation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ekf_altitude_estimation_v0_9.png", dpi=200)
    plt.close(fig)

    # Airspeed estimate
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["true_airspeed_m_s"], label="True airspeed")
    ax.plot(results["time_s"], results["est_airspeed_m_s"], label="EKF estimate")
    ax.scatter(
        results["time_s"],
        results["meas_airspeed_m_s"],
        s=4,
        alpha=0.35,
        label="Airspeed measurement",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Airspeed [m/s]")
    ax.set_title("EKF Airspeed Estimation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ekf_airspeed_estimation_v0_9.png", dpi=200)
    plt.close(fig)

    # Pitch estimate
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["true_theta_deg"], label="True pitch")
    ax.plot(results["time_s"], results["est_theta_deg"], label="EKF estimate")
    ax.scatter(
        results["time_s"],
        results["meas_theta_deg"],
        s=4,
        alpha=0.35,
        label="Pitch measurement",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pitch angle [deg]")
    ax.set_title("EKF Pitch Angle Estimation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ekf_pitch_estimation_v0_9.png", dpi=200)
    plt.close(fig)

    # Estimation errors
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["error_altitude_m"], label="Altitude error [m]")
    ax.plot(results["time_s"], results["error_airspeed_m_s"], label="Airspeed error [m/s]")
    ax.plot(results["time_s"], results["error_theta_deg"], label="Pitch error [deg]")
    ax.plot(results["time_s"], results["error_alpha_deg"], label="Alpha error [deg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Estimation Error")
    ax.set_title("EKF Estimation Errors")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ekf_estimation_errors_v0_9.png", dpi=200)
    plt.close(fig)


def _print_summary(results) -> None:
    """Print EKF estimation performance summary."""
    rms_altitude = (results["error_altitude_m"] ** 2).mean() ** 0.5
    rms_airspeed = (results["error_airspeed_m_s"] ** 2).mean() ** 0.5
    rms_pitch = (results["error_theta_deg"] ** 2).mean() ** 0.5
    rms_alpha = (results["error_alpha_deg"] ** 2).mean() ** 0.5
    rms_q = (results["error_q_deg_s"] ** 2).mean() ** 0.5

    final = results.iloc[-1]

    print("EKF navigation demo")
    print("=" * 56)
    print("RMS estimation errors")
    print("-" * 56)
    print(f"Altitude RMS error: {rms_altitude:.3f} m")
    print(f"Airspeed RMS error: {rms_airspeed:.3f} m/s")
    print(f"Pitch RMS error: {rms_pitch:.3f} deg")
    print(f"Alpha RMS error: {rms_alpha:.3f} deg")
    print(f"Pitch-rate RMS error: {rms_q:.3f} deg/s")
    print()
    print("Final estimation errors")
    print("-" * 56)
    print(f"Final altitude error: {final['error_altitude_m']:.3f} m")
    print(f"Final airspeed error: {final['error_airspeed_m_s']:.3f} m/s")
    print(f"Final pitch error: {final['error_theta_deg']:.3f} deg")
    print(f"Final alpha error: {final['error_alpha_deg']:.3f} deg")
    print(f"Final pitch-rate error: {final['error_q_deg_s']:.3f} deg/s")


def main() -> None:
    """Run EKF navigation workflow."""
    config = load_json_config(PROJECT_ROOT / "configs" / "uav_longitudinal.json")

    results = run_ekf_navigation_simulation(config)

    output_log = PROJECT_ROOT / "outputs" / "logs" / "ekf_navigation_results_v0_9.csv"
    output_plot_dir = PROJECT_ROOT / "outputs" / "plots"

    output_log.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_log, index=False)
    _save_estimation_plots(results, output_plot_dir)
    _print_summary(results)

    print()
    print("Saved artifacts")
    print("-" * 56)
    print(f"CSV: {output_log}")
    print(f"Plots: {output_plot_dir}")


if __name__ == "__main__":
    main()