"""Interactive PID vs LQR comparison dashboard.

Run from project root:

    streamlit run apps/controller_comparison_dashboard.py
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gnc_sim.analysis import run_pid_lqr_comparison
from gnc_sim.config import load_json_config


def update_config_from_sidebar(base_config: dict) -> dict:
    """Update comparison scenario from sidebar inputs."""
    config = deepcopy(base_config)

    st.sidebar.header("Scenario")

    config["simulation"]["t_final_s"] = st.sidebar.slider(
        "Simulation time [s]",
        min_value=5.0,
        max_value=120.0,
        value=float(config["simulation"]["t_final_s"]),
        step=5.0,
    )

    config["simulation"]["commands"]["altitude_m"] = st.sidebar.slider(
        "Altitude command [m]",
        min_value=50.0,
        max_value=250.0,
        value=float(config["simulation"]["commands"]["altitude_m"]),
        step=5.0,
    )

    config["simulation"]["commands"]["airspeed_m_s"] = st.sidebar.slider(
        "Airspeed command [m/s]",
        min_value=14.0,
        max_value=32.0,
        value=float(config["simulation"]["commands"]["airspeed_m_s"]),
        step=0.5,
    )

    st.sidebar.header("Initial Condition")

    config["simulation"]["initial"]["altitude_m"] = st.sidebar.slider(
        "Initial altitude [m]",
        min_value=50.0,
        max_value=200.0,
        value=float(config["simulation"]["initial"]["altitude_m"]),
        step=5.0,
    )

    config["simulation"]["initial"]["airspeed_m_s"] = st.sidebar.slider(
        "Initial airspeed [m/s]",
        min_value=14.0,
        max_value=32.0,
        value=float(config["simulation"]["initial"]["airspeed_m_s"]),
        step=0.5,
    )

    st.sidebar.header("Wind")

    config["simulation"]["wind"]["enabled"] = st.sidebar.checkbox(
        "Enable vertical gust",
        value=bool(config["simulation"]["wind"]["enabled"]),
    )

    config["simulation"]["wind"]["vertical_gust_amplitude_m_s"] = st.sidebar.slider(
        "Gust amplitude [m/s]",
        min_value=0.0,
        max_value=5.0,
        value=float(
            config["simulation"]["wind"].get("vertical_gust_amplitude_m_s", 0.0)
        ),
        step=0.25,
    )

    return config


def overlay_plot(pid, lqr, y_col: str, cmd_col: str | None, title: str, y_label: str):
    """Create PID vs LQR overlay plot."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pid["time_s"],
            y=pid[y_col],
            mode="lines",
            name=f"PID {y_col}",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lqr["time_s"],
            y=lqr[y_col],
            mode="lines",
            name=f"LQR {y_col}",
        )
    )

    if cmd_col is not None:
        fig.add_trace(
            go.Scatter(
                x=pid["time_s"],
                y=pid[cmd_col],
                mode="lines",
                name="Command",
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


def main() -> None:
    """Run controller comparison dashboard."""
    st.set_page_config(
        page_title="PID vs LQR Comparison",
        page_icon="📈",
        layout="wide",
    )

    st.title("PID vs LQR Controller Comparison")
    st.caption(
        "Compare cascaded PID and trim-linearized LQR controllers on the same "
        "nonlinear longitudinal aircraft model."
    )

    base_config = load_json_config(PROJECT_ROOT / "configs" / "uav_longitudinal.json")
    config = update_config_from_sidebar(base_config)

    run_button = st.sidebar.button("Run Comparison", type="primary")

    if "comparison" not in st.session_state or run_button:
        with st.spinner("Running PID and LQR simulations..."):
            st.session_state["comparison"] = run_pid_lqr_comparison(config)

    comparison = st.session_state["comparison"]

    pid = comparison.pid_results
    lqr = comparison.lqr_results
    metrics_df = comparison.metrics_dataframe()

    st.subheader("Summary Metrics")
    st.dataframe(metrics_df, use_container_width=True)

    tab_response, tab_error, tab_controls, tab_data = st.tabs(
        ["Responses", "Tracking Errors", "Controls", "Data"]
    )

    with tab_response:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                overlay_plot(
                    pid,
                    lqr,
                    y_col="altitude_m",
                    cmd_col="altitude_cmd_m",
                    title="Altitude Response",
                    y_label="Altitude [m]",
                ),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                overlay_plot(
                    pid,
                    lqr,
                    y_col="airspeed_m_s",
                    cmd_col="airspeed_cmd_m_s",
                    title="Airspeed Response",
                    y_label="Airspeed [m/s]",
                ),
                use_container_width=True,
            )

    with tab_error:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                overlay_plot(
                    pid,
                    lqr,
                    y_col="altitude_error_m",
                    cmd_col=None,
                    title="Altitude Tracking Error",
                    y_label="Altitude error [m]",
                ),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                overlay_plot(
                    pid,
                    lqr,
                    y_col="airspeed_error_m_s",
                    cmd_col=None,
                    title="Airspeed Tracking Error",
                    y_label="Airspeed error [m/s]",
                ),
                use_container_width=True,
            )

    with tab_controls:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                overlay_plot(
                    pid,
                    lqr,
                    y_col="elevator_deg",
                    cmd_col=None,
                    title="Elevator Command",
                    y_label="Elevator [deg]",
                ),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                overlay_plot(
                    pid,
                    lqr,
                    y_col="throttle",
                    cmd_col=None,
                    title="Throttle Command",
                    y_label="Throttle [-]",
                ),
                use_container_width=True,
            )

    with tab_data:
        st.subheader("PID Results")
        st.dataframe(pid, use_container_width=True)

        st.subheader("LQR Results")
        st.dataframe(lqr, use_container_width=True)


if __name__ == "__main__":
    main()