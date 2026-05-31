"""Interactive GNC dashboard for the fixed-wing UAV project.

Run from the project root:

    streamlit run apps/gnc_dashboard.py
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from gnc_sim.config import load_json_config
from gnc_sim.simulation.lqr_closed_loop import simulate_lqr_longitudinal


def compute_forward_position_m(results: pd.DataFrame) -> np.ndarray:
    """Estimate forward inertial position from body velocities and pitch."""
    time_s = results["time_s"].to_numpy(dtype=float)
    u = results["u_m_s"].to_numpy(dtype=float)
    w = results["w_m_s"].to_numpy(dtype=float)
    theta = np.radians(results["theta_deg"].to_numpy(dtype=float))

    x_dot = u * np.cos(theta) + w * np.sin(theta)

    x = np.zeros_like(time_s)

    for idx in range(1, len(time_s)):
        dt = time_s[idx] - time_s[idx - 1]
        x[idx] = x[idx - 1] + 0.5 * (x_dot[idx] + x_dot[idx - 1]) * dt

    return x


def aircraft_polygon(
    x_m: float,
    h_m: float,
    theta_rad: float,
    length_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple aircraft side-view polygon."""
    body = np.array(
        [
            [0.55 * length_m, 0.0],
            [-0.35 * length_m, 0.13 * length_m],
            [-0.55 * length_m, 0.0],
            [-0.35 * length_m, -0.13 * length_m],
            [0.55 * length_m, 0.0],
        ],
        dtype=float,
    )

    rotation = np.array(
        [
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)],
        ],
        dtype=float,
    )

    inertial = body @ rotation.T + np.array([x_m, h_m])

    return inertial[:, 0], inertial[:, 1]


def make_time_response_plot(
    results: pd.DataFrame,
    y_column: str,
    command_column: str | None,
    title: str,
    y_label: str,
) -> go.Figure:
    """Create a standard interactive time response plot."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["time_s"],
            y=results[y_column],
            mode="lines",
            name=y_column,
        )
    )

    if command_column is not None:
        fig.add_trace(
            go.Scatter(
                x=results["time_s"],
                y=results[command_column],
                mode="lines",
                name=command_column,
                line={"dash": "dash"},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time [s]",
        yaxis_title=y_label,
        hovermode="x unified",
        height=420,
    )

    return fig


def make_control_plot(results: pd.DataFrame) -> go.Figure:
    """Create elevator and throttle plot."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["time_s"],
            y=results["elevator_deg"],
            mode="lines",
            name="Elevator [deg]",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results["time_s"],
            y=results["throttle"],
            mode="lines",
            name="Throttle [-]",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Control Inputs",
        xaxis_title="Time [s]",
        yaxis={"title": "Elevator [deg]"},
        yaxis2={
            "title": "Throttle [-]",
            "overlaying": "y",
            "side": "right",
        },
        hovermode="x unified",
        height=420,
    )

    return fig


def make_error_plot(results: pd.DataFrame) -> go.Figure:
    """Create tracking error plot."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["time_s"],
            y=results["altitude_error_m"],
            mode="lines",
            name="Altitude error [m]",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results["time_s"],
            y=results["airspeed_error_m_s"],
            mode="lines",
            name="Airspeed error [m/s]",
        )
    )

    fig.update_layout(
        title="Tracking Errors",
        xaxis_title="Time [s]",
        yaxis_title="Error",
        hovermode="x unified",
        height=420,
    )

    return fig


def make_aircraft_motion_figure(results: pd.DataFrame, frame_step: int = 5) -> go.Figure:
    """Create interactive 2D aircraft motion visualization with slider."""
    x_m = compute_forward_position_m(results)
    h_m = results["altitude_m"].to_numpy(dtype=float)
    theta_rad = np.radians(results["theta_deg"].to_numpy(dtype=float))
    time_s = results["time_s"].to_numpy(dtype=float)

    frame_indices = np.arange(0, len(results), frame_step)

    initial_idx = int(frame_indices[0])
    aircraft_x, aircraft_h = aircraft_polygon(
        x_m=float(x_m[initial_idx]),
        h_m=float(h_m[initial_idx]),
        theta_rad=float(theta_rad[initial_idx]),
    )

    frames = []

    for idx in frame_indices:
        idx = int(idx)
        px, ph = aircraft_polygon(
            x_m=float(x_m[idx]),
            h_m=float(h_m[idx]),
            theta_rad=float(theta_rad[idx]),
        )

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x_m[: idx + 1],
                        y=h_m[: idx + 1],
                        mode="lines",
                        name="Trajectory",
                    ),
                    go.Scatter(
                        x=px,
                        y=ph,
                        mode="lines",
                        fill="toself",
                        name="Aircraft",
                    ),
                ],
                name=f"{time_s[idx]:.1f}",
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_m[: initial_idx + 1],
                y=h_m[: initial_idx + 1],
                mode="lines",
                name="Trajectory",
            ),
            go.Scatter(
                x=aircraft_x,
                y=aircraft_h,
                mode="lines",
                fill="toself",
                name="Aircraft",
            ),
        ],
        frames=frames,
    )

    fig.update_layout(
        title="2D Longitudinal Aircraft Motion",
        xaxis_title="Forward Distance [m]",
        yaxis_title="Altitude [m]",
        height=540,
        xaxis={
            "range": [float(np.min(x_m) - 25.0), float(np.max(x_m) + 25.0)],
        },
        yaxis={
            "range": [float(np.min(h_m) - 25.0), float(np.max(h_m) + 25.0)],
            "scaleanchor": "x",
            "scaleratio": 1.0,
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 40, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "label": frame.name,
                        "args": [
                            [frame.name],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for frame in frames
                ],
                "currentvalue": {"prefix": "Time [s]: "},
            }
        ],
    )

    return fig


def update_config_from_sidebar(base_config: dict) -> dict:
    """Create a modified simulation config using sidebar controls."""
    config = deepcopy(base_config)

    st.sidebar.header("Simulation Setup")

    t_final_s = st.sidebar.slider(
        "Simulation time [s]",
        min_value=5.0,
        max_value=120.0,
        value=float(config["simulation"]["t_final_s"]),
        step=5.0,
    )

    dt_s = st.sidebar.selectbox(
        "Time step [s]",
        options=[0.005, 0.01, 0.02, 0.05],
        index=1,
    )

    st.sidebar.header("Command Inputs")

    altitude_command_m = st.sidebar.slider(
        "Altitude command [m]",
        min_value=50.0,
        max_value=250.0,
        value=float(config["simulation"]["commands"]["altitude_m"]),
        step=5.0,
    )

    airspeed_command_m_s = st.sidebar.slider(
        "Airspeed command [m/s]",
        min_value=14.0,
        max_value=32.0,
        value=float(config["simulation"]["commands"]["airspeed_m_s"]),
        step=0.5,
    )

    st.sidebar.header("Initial Condition")

    initial_altitude_m = st.sidebar.slider(
        "Initial altitude [m]",
        min_value=50.0,
        max_value=200.0,
        value=float(config["simulation"]["initial"]["altitude_m"]),
        step=5.0,
    )

    initial_airspeed_m_s = st.sidebar.slider(
        "Initial airspeed [m/s]",
        min_value=14.0,
        max_value=32.0,
        value=float(config["simulation"]["initial"]["airspeed_m_s"]),
        step=0.5,
    )

    initial_alpha_deg = st.sidebar.slider(
        "Initial angle of attack [deg]",
        min_value=-5.0,
        max_value=15.0,
        value=float(np.degrees(config["simulation"]["initial"]["alpha_rad"])),
        step=0.5,
    )

    initial_pitch_deg = st.sidebar.slider(
        "Initial pitch angle [deg]",
        min_value=-10.0,
        max_value=20.0,
        value=float(np.degrees(config["simulation"]["initial"]["theta_rad"])),
        step=0.5,
    )

    st.sidebar.header("Wind Disturbance")

    wind_enabled = st.sidebar.checkbox(
        "Enable vertical gust",
        value=bool(config["simulation"]["wind"]["enabled"]),
    )

    gust_amplitude = st.sidebar.slider(
        "Gust amplitude [m/s]",
        min_value=0.0,
        max_value=5.0,
        value=float(config["simulation"]["wind"].get("vertical_gust_amplitude_m_s", 0.0)),
        step=0.25,
    )

    config["simulation"]["t_final_s"] = t_final_s
    config["simulation"]["dt_s"] = dt_s

    config["simulation"]["commands"]["altitude_m"] = altitude_command_m
    config["simulation"]["commands"]["airspeed_m_s"] = airspeed_command_m_s

    config["simulation"]["initial"]["altitude_m"] = initial_altitude_m
    config["simulation"]["initial"]["airspeed_m_s"] = initial_airspeed_m_s
    config["simulation"]["initial"]["alpha_rad"] = float(np.radians(initial_alpha_deg))
    config["simulation"]["initial"]["theta_rad"] = float(np.radians(initial_pitch_deg))

    config["simulation"]["wind"]["enabled"] = wind_enabled
    config["simulation"]["wind"]["vertical_gust_amplitude_m_s"] = gust_amplitude

    return config


def show_summary_metrics(results: pd.DataFrame) -> None:
    """Display key simulation metrics."""
    final = results.iloc[-1]

    max_altitude_error = results["altitude_error_m"].abs().max()
    rms_altitude_error = (results["altitude_error_m"] ** 2).mean() ** 0.5

    max_airspeed_error = results["airspeed_error_m_s"].abs().max()
    rms_airspeed_error = (results["airspeed_error_m_s"] ** 2).mean() ** 0.5

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Final altitude", f"{final['altitude_m']:.2f} m")
    col2.metric("Final airspeed", f"{final['airspeed_m_s']:.2f} m/s")
    col3.metric("Final pitch", f"{final['theta_deg']:.2f} deg")
    col4.metric("Final throttle", f"{final['throttle']:.3f}")

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Max altitude error", f"{max_altitude_error:.2f} m")
    col6.metric("RMS altitude error", f"{rms_altitude_error:.2f} m")
    col7.metric("Max airspeed error", f"{max_airspeed_error:.2f} m/s")
    col8.metric("RMS airspeed error", f"{rms_airspeed_error:.2f} m/s")


def main() -> None:
    """Run Streamlit dashboard."""
    st.set_page_config(
        page_title="Fixed-Wing UAV GNC Dashboard",
        page_icon="✈️",
        layout="wide",
    )

    st.title("Fixed-Wing UAV GNC Interactive Dashboard")
    st.caption(
        "Nonlinear longitudinal UAV simulation with trim-based LQR control, "
        "wind disturbance, tracking metrics, and 2D aircraft visualization."
    )

    base_config = load_json_config(PROJECT_ROOT / "configs" / "uav_longitudinal.json")
    config = update_config_from_sidebar(base_config)

    run_button = st.sidebar.button("Run Simulation", type="primary")

    if "simulation_results" not in st.session_state or run_button:
        with st.spinner("Running nonlinear LQR simulation..."):
            st.session_state["simulation_results"] = simulate_lqr_longitudinal(config)

    results = st.session_state["simulation_results"]

    show_summary_metrics(results)

    st.divider()

    tab_response, tab_controls, tab_motion, tab_data = st.tabs(
        [
            "Time Responses",
            "Controls & Errors",
            "Aircraft Motion",
            "Data",
        ]
    )

    with tab_response:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                make_time_response_plot(
                    results=results,
                    y_column="altitude_m",
                    command_column="altitude_cmd_m",
                    title="Altitude Response",
                    y_label="Altitude [m]",
                ),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                make_time_response_plot(
                    results=results,
                    y_column="airspeed_m_s",
                    command_column="airspeed_cmd_m_s",
                    title="Airspeed Response",
                    y_label="Airspeed [m/s]",
                ),
                use_container_width=True,
            )

        st.plotly_chart(
            make_time_response_plot(
                results=results,
                y_column="theta_deg",
                command_column="theta_cmd_deg",
                title="Pitch Response",
                y_label="Pitch angle [deg]",
            ),
            use_container_width=True,
        )

    with tab_controls:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(make_control_plot(results), use_container_width=True)

        with col2:
            st.plotly_chart(make_error_plot(results), use_container_width=True)

    with tab_motion:
        st.plotly_chart(
            make_aircraft_motion_figure(results, frame_step=5),
            use_container_width=True,
        )

    with tab_data:
        st.subheader("Simulation Data")
        st.dataframe(results, use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="gnc_lqr_simulation_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()