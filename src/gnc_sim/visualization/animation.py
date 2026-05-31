"""Animation utilities for longitudinal fixed-wing simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _compute_forward_position_m(results: pd.DataFrame) -> np.ndarray:
    """Estimate inertial forward position from body velocities and pitch angle.

    The longitudinal model currently stores altitude but not horizontal position.
    This utility reconstructs an approximate forward trajectory for visualization.
    """
    time_s = results["time_s"].to_numpy(dtype=float)
    u = results["u_m_s"].to_numpy(dtype=float)
    w = results["w_m_s"].to_numpy(dtype=float)
    theta = np.radians(results["theta_deg"].to_numpy(dtype=float))

    # Approximate inertial forward speed from body-axis components.
    x_dot = u * np.cos(theta) + w * np.sin(theta)

    x = np.zeros_like(time_s)

    for idx in range(1, len(time_s)):
        dt = time_s[idx] - time_s[idx - 1]
        x[idx] = x[idx - 1] + 0.5 * (x_dot[idx] + x_dot[idx - 1]) * dt

    return x


def _aircraft_shape_body(length_m: float = 6.0) -> np.ndarray:
    """Return a simple aircraft side-view polygon in body axes."""
    return np.array(
        [
            [0.55 * length_m, 0.0],
            [-0.35 * length_m, 0.12 * length_m],
            [-0.55 * length_m, 0.0],
            [-0.35 * length_m, -0.12 * length_m],
            [0.55 * length_m, 0.0],
        ],
        dtype=float,
    )


def _rotate_translate(points: np.ndarray, theta_rad: float, x_m: float, h_m: float) -> np.ndarray:
    """Rotate body-frame aircraft shape and translate to inertial x-altitude frame."""
    rotation = np.array(
        [
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)],
        ],
        dtype=float,
    )

    return points @ rotation.T + np.array([x_m, h_m])


def animate_longitudinal_response(
    results: pd.DataFrame,
    output_path: str | Path,
    frame_step: int = 5,
    interval_ms: int = 40,
    aircraft_length_m: float = 8.0,
) -> Path:
    """Create and save a 2D longitudinal aircraft animation as a GIF.

    Parameters
    ----------
    results:
        Simulation result dataframe from ``simulate_longitudinal``.
    output_path:
        GIF output path.
    frame_step:
        Use every Nth simulation point to reduce animation file size.
    interval_ms:
        Delay between frames in milliseconds.
    aircraft_length_m:
        Display length of the aircraft symbol.

    Returns
    -------
    Path
        Path to the saved animation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_m = _compute_forward_position_m(results)
    h_m = results["altitude_m"].to_numpy(dtype=float)
    theta_rad = np.radians(results["theta_deg"].to_numpy(dtype=float))
    time_s = results["time_s"].to_numpy(dtype=float)

    frame_indices = np.arange(0, len(results), frame_step)

    aircraft_body = _aircraft_shape_body(length_m=aircraft_length_m)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))

    x_margin = 20.0
    h_margin = 20.0

    ax.set_xlim(float(np.min(x_m) - x_margin), float(np.max(x_m) + x_margin))
    ax.set_ylim(float(np.min(h_m) - h_margin), float(np.max(h_m) + h_margin))

    ax.set_xlabel("Forward distance [m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Longitudinal UAV Motion Animation")
    ax.grid(True)

    trajectory_line, = ax.plot([], [], linewidth=1.5, label="Trajectory")
    aircraft_line, = ax.plot([], [], linewidth=2.5, label="Aircraft")
    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
    )

    ax.legend(loc="lower right")

    def init():
        trajectory_line.set_data([], [])
        aircraft_line.set_data([], [])
        time_text.set_text("")
        return trajectory_line, aircraft_line, time_text

    def update(frame_number: int):
        idx = int(frame_indices[frame_number])

        aircraft_points = _rotate_translate(
            points=aircraft_body,
            theta_rad=float(theta_rad[idx]),
            x_m=float(x_m[idx]),
            h_m=float(h_m[idx]),
        )

        trajectory_line.set_data(x_m[: idx + 1], h_m[: idx + 1])
        aircraft_line.set_data(aircraft_points[:, 0], aircraft_points[:, 1])

        time_text.set_text(
            f"t = {time_s[idx]:.1f} s\n"
            f"h = {h_m[idx]:.1f} m\n"
            f"theta = {np.degrees(theta_rad[idx]):.1f} deg"
        )

        return trajectory_line, aircraft_line, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=interval_ms,
        blit=True,
    )

    anim.save(output_path, writer="pillow", fps=max(1, int(1000 / interval_ms)))

    plt.close(fig)

    return output_path