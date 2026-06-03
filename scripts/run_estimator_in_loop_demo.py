"""Run estimator-in-the-loop longitudinal GNC simulation.

Execute from project root:

    python scripts/run_estimator_in_loop_demo.py
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
from gnc_sim.simulation.estimator_in_loop import simulate_estimator_in_loop_longitudinal
from gnc_sim.visualization import animate_longitudinal_response


def _save_estimator_loop_plots(results, output_dir: Path) -> None:
    """Save estimator-in-the-loop plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Altitude tracking and estimation.
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["true_altitude_m"], label="True altitude")
    ax.plot(results["time_s"], results["est_altitude_m"], label="EKF estimate")
    ax.plot(results["time_s"], results["altitude_cmd_m"], "--", label="Command")
    ax.scatter(
        results["time_s"],
        results["meas_altitude_m"],
        s=4,
        alpha=0.30,
        label="Barometer measurement",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Estimator-in-the-Loop Altitude Response")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "estimator_loop_altitude_v1_0.png", dpi=200)
    plt.close(fig)

    # Airspeed tracking and estimation.
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["true_airspeed_m_s"], label="True airspeed")
    ax.plot(results["time_s"], results["est_airspeed_m_s"], label="EKF estimate")
    ax.plot(results["time_s"], results["airspeed_cmd_m_s"], "--", label="Command")
    ax.scatter(
        results["time_s"],
        results["meas_airspeed_m_s"],
        s=4,
        alpha=0.30,
        label="Airspeed measurement",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Airspeed [m/s]")
    ax.set_title("Estimator-in-the-Loop Airspeed Response")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "estimator_loop_airspeed_v1_0.png", dpi=200)
    plt.close(fig)

    # Pitch estimate.
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["true_theta_deg"], label="True pitch")
    ax.plot(results["time_s"], results["est_theta_deg"], label="EKF estimate")
    ax.scatter(
        results["time_s"],
        results["meas_theta_deg"],
        s=4,
        alpha=0.30,
        label="Pitch measurement",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pitch angle [deg]")
    ax.set_title("Estimator-in-the-Loop Pitch Estimation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "estimator_loop_pitch_v1_0.png", dpi=200)
    plt.close(fig)

    # Controls.
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["elevator_deg"], label="Elevator [deg]")
    ax.plot(results["time_s"], results["throttle"], label="Throttle [-]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Control Input")
    ax.set_title("Estimator-in-the-Loop Control Inputs")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "estimator_loop_controls_v1_0.png", dpi=200)
    plt.close(fig)

    # Tracking errors.
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["altitude_error_m"], label="Altitude error [m]")
    ax.plot(results["time_s"], results["airspeed_error_m_s"], label="Airspeed error [m/s]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tracking Error")
    ax.set_title("Estimator-in-the-Loop Tracking Errors")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "estimator_loop_tracking_errors_v1_0.png", dpi=200)
    plt.close(fig)

    # Estimation errors.
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(results["time_s"], results["error_altitude_m"], label="Altitude estimate error [m]")
    ax.plot(results["time_s"], results["error_airspeed_m_s"], label="Airspeed estimate error [m/s]")
    ax.plot(results["time_s"], results["error_theta_deg"], label="Pitch estimate error [deg]")
    ax.plot(results["time_s"], results["error_alpha_deg"], label="AOA estimate error [deg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Estimation Error")
    ax.set_title("Estimator-in-the-Loop EKF Errors")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "estimator_loop_estimation_errors_v1_0.png", dpi=200)
    plt.close(fig)


def _print_summary(results) -> None:
    """Print estimator-in-the-loop performance summary."""
    final = results.iloc[-1]

    rms_altitude_tracking = (results["altitude_error_m"] ** 2).mean() ** 0.5
    rms_airspeed_tracking = (results["airspeed_error_m_s"] ** 2).mean() ** 0.5

    rms_altitude_est = (results["error_altitude_m"] ** 2).mean() ** 0.5
    rms_airspeed_est = (results["error_airspeed_m_s"] ** 2).mean() ** 0.5
    rms_pitch_est = (results["error_theta_deg"] ** 2).mean() ** 0.5
    rms_alpha_est = (results["error_alpha_deg"] ** 2).mean() ** 0.5

    print("Estimator-in-the-loop GNC simulation")
    print("=" * 64)
    print(f"Final true altitude: {final['true_altitude_m']:.3f} m")
    print(f"Final estimated altitude: {final['est_altitude_m']:.3f} m")
    print(f"Final true airspeed: {final['true_airspeed_m_s']:.3f} m/s")
    print(f"Final estimated airspeed: {final['est_airspeed_m_s']:.3f} m/s")
    print(f"Final elevator: {final['elevator_deg']:.3f} deg")
    print(f"Final throttle: {final['throttle']:.4f}")

    print()
    print("Tracking metrics")
    print("-" * 64)
    print(f"RMS altitude tracking error: {rms_altitude_tracking:.3f} m")
    print(f"RMS airspeed tracking error: {rms_airspeed_tracking:.3f} m/s")

    print()
    print("EKF estimation metrics")
    print("-" * 64)
    print(f"RMS altitude estimation error: {rms_altitude_est:.3f} m")
    print(f"RMS airspeed estimation error: {rms_airspeed_est:.3f} m/s")
    print(f"RMS pitch estimation error: {rms_pitch_est:.3f} deg")
    print(f"RMS AOA estimation error: {rms_alpha_est:.3f} deg")


def main() -> None:
    """Run estimator-in-the-loop demo."""
    config = load_json_config(PROJECT_ROOT / "configs" / "uav_longitudinal.json")

    results = simulate_estimator_in_loop_longitudinal(config)

    output_log = PROJECT_ROOT / "outputs" / "logs" / "estimator_in_loop_results_v1_0.csv"
    output_plot_dir = PROJECT_ROOT / "outputs" / "plots"
    output_animation = PROJECT_ROOT / "outputs" / "animations" / "estimator_loop_motion_v1_0.gif"

    output_log.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_log, index=False)

    _save_estimator_loop_plots(results, output_plot_dir)

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
    print("-" * 64)
    print(f"CSV: {output_log}")
    print(f"Plots: {output_plot_dir}")
    print(f"Animation: {output_animation}")


if __name__ == "__main__":
    main()