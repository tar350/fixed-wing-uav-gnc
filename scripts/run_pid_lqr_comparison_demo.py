"""Run PID vs LQR comparison for nonlinear longitudinal UAV simulation.

Execute from project root:

    python scripts/run_pid_lqr_comparison_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gnc_sim.analysis import run_pid_lqr_comparison
from gnc_sim.config import load_json_config


def _save_overlay_plots(comparison, output_dir: Path) -> None:
    """Save PID vs LQR overlay plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pid = comparison.pid_results
    lqr = comparison.lqr_results

    # Altitude
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(pid["time_s"], pid["altitude_m"], label="PID altitude")
    ax.plot(lqr["time_s"], lqr["altitude_m"], label="LQR altitude")
    ax.plot(pid["time_s"], pid["altitude_cmd_m"], "--", label="Command")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("PID vs LQR Altitude Response")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pid_vs_lqr_altitude_v0_8.png", dpi=200)
    plt.close(fig)

    # Airspeed
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(pid["time_s"], pid["airspeed_m_s"], label="PID airspeed")
    ax.plot(lqr["time_s"], lqr["airspeed_m_s"], label="LQR airspeed")
    ax.plot(pid["time_s"], pid["airspeed_cmd_m_s"], "--", label="Command")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Airspeed [m/s]")
    ax.set_title("PID vs LQR Airspeed Response")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pid_vs_lqr_airspeed_v0_8.png", dpi=200)
    plt.close(fig)

    # Tracking errors
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(pid["time_s"], pid["altitude_error_m"], label="PID altitude error")
    ax.plot(lqr["time_s"], lqr["altitude_error_m"], label="LQR altitude error")
    ax.plot(pid["time_s"], pid["airspeed_error_m_s"], label="PID airspeed error")
    ax.plot(lqr["time_s"], lqr["airspeed_error_m_s"], label="LQR airspeed error")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tracking Error")
    ax.set_title("PID vs LQR Tracking Errors")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pid_vs_lqr_tracking_errors_v0_8.png", dpi=200)
    plt.close(fig)

    # Elevator
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(pid["time_s"], pid["elevator_deg"], label="PID elevator")
    ax.plot(lqr["time_s"], lqr["elevator_deg"], label="LQR elevator")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Elevator [deg]")
    ax.set_title("PID vs LQR Elevator Command")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pid_vs_lqr_elevator_v0_8.png", dpi=200)
    plt.close(fig)

    # Throttle
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(pid["time_s"], pid["throttle"], label="PID throttle")
    ax.plot(lqr["time_s"], lqr["throttle"], label="LQR throttle")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Throttle [-]")
    ax.set_title("PID vs LQR Throttle Command")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pid_vs_lqr_throttle_v0_8.png", dpi=200)
    plt.close(fig)


def _print_metrics_table(metrics_df) -> None:
    """Print comparison metrics."""
    print("PID vs LQR Controller Comparison")
    print("=" * 72)

    selected_columns = [
        "controller",
        "final_altitude_error_m",
        "rms_altitude_error_m",
        "max_altitude_error_m",
        "final_airspeed_error_m_s",
        "rms_airspeed_error_m_s",
        "max_airspeed_error_m_s",
        "rms_elevator_deg",
        "rms_throttle",
        "settling_time_altitude_s",
        "settling_time_airspeed_s",
    ]

    print(metrics_df[selected_columns].to_string(index=False))


def main() -> None:
    """Run PID vs LQR comparison and save artifacts."""
    config = load_json_config(PROJECT_ROOT / "configs" / "uav_longitudinal.json")

    comparison = run_pid_lqr_comparison(config)

    output_log_dir = PROJECT_ROOT / "outputs" / "logs"
    output_plot_dir = PROJECT_ROOT / "outputs" / "plots"

    output_log_dir.mkdir(parents=True, exist_ok=True)

    pid_csv = output_log_dir / "pid_results_v0_8.csv"
    lqr_csv = output_log_dir / "lqr_results_v0_8.csv"
    metrics_csv = output_log_dir / "pid_vs_lqr_metrics_v0_8.csv"
    metrics_json = output_log_dir / "pid_vs_lqr_metrics_v0_8.json"

    comparison.pid_results.to_csv(pid_csv, index=False)
    comparison.lqr_results.to_csv(lqr_csv, index=False)
    comparison.metrics_dataframe().to_csv(metrics_csv, index=False)
    comparison.save_metrics_json(metrics_json)

    _save_overlay_plots(comparison, output_plot_dir)

    _print_metrics_table(comparison.metrics_dataframe())

    print()
    print("Saved artifacts")
    print("-" * 72)
    print(f"PID CSV: {pid_csv}")
    print(f"LQR CSV: {lqr_csv}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Metrics JSON: {metrics_json}")
    print(f"Plots: {output_plot_dir}")


if __name__ == "__main__":
    main()