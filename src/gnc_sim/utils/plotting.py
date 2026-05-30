"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_longitudinal_response(results: pd.DataFrame, save_path: str | Path) -> None:
    """Create and save standard longitudinal response plots."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot 1: Altitude
    fig = plt.figure(figsize=(10, 6))
    plt.plot(results["time_s"], results["altitude_m"], label="Altitude")
    plt.plot(results["time_s"], results["altitude_cmd_m"], "--", label="Command")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [m]")
    plt.title("Altitude Tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    altitude_path = save_path.with_name(save_path.stem + "_altitude.png")
    fig.savefig(altitude_path, dpi=200)
    plt.close(fig)

    # Plot 2: Airspeed
    fig = plt.figure(figsize=(10, 6))
    plt.plot(results["time_s"], results["airspeed_m_s"], label="Airspeed")
    plt.plot(results["time_s"], results["airspeed_cmd_m_s"], "--", label="Command")
    plt.xlabel("Time [s]")
    plt.ylabel("Airspeed [m/s]")
    plt.title("Airspeed Tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    airspeed_path = save_path.with_name(save_path.stem + "_airspeed.png")
    fig.savefig(airspeed_path, dpi=200)
    plt.close(fig)

    # Plot 3: Pitch response
    fig = plt.figure(figsize=(10, 6))
    plt.plot(results["time_s"], results["theta_deg"], label="Pitch Angle")
    plt.plot(results["time_s"], results["theta_cmd_deg"], "--", label="Pitch Command")
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch Angle [deg]")
    plt.title("Pitch Tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pitch_path = save_path.with_name(save_path.stem + "_pitch.png")
    fig.savefig(pitch_path, dpi=200)
    plt.close(fig)

    # Plot 4: Control history
    fig = plt.figure(figsize=(10, 6))
    plt.plot(results["time_s"], results["elevator_deg"], label="Elevator [deg]")
    plt.plot(results["time_s"], results["throttle"], label="Throttle [-]")
    plt.xlabel("Time [s]")
    plt.ylabel("Control")
    plt.title("Control Inputs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    controls_path = save_path.with_name(save_path.stem + "_controls.png")
    fig.savefig(controls_path, dpi=200)
    plt.close(fig)

    # Plot 5: Gust
    fig = plt.figure(figsize=(10, 6))
    plt.plot(results["time_s"], results["vertical_gust_m_s"], label="Vertical Gust")
    plt.xlabel("Time [s]")
    plt.ylabel("Vertical Gust [m/s]")
    plt.title("Disturbance Input")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    gust_path = save_path.with_name(save_path.stem + "_gust.png")
    fig.savefig(gust_path, dpi=200)
    plt.close(fig)
