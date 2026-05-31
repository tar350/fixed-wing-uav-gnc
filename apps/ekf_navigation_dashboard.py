"""Interactive EKF navigation dashboard.

Run from project root:

    streamlit run apps/ekf_navigation_dashboard.py
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

from gnc_sim.config import load_json_config
from gnc_sim.simulation.ekf_navigation import run_ekf_navigation_simulation


def update_config_from_sidebar(base_config: dict) -> dict:
    """Update EKF scenario and sensor noise from sidebar."""
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

    st.sidebar.header("Sensor Noise")

    if "navigation" not in config:
        config["navigation"] = {}

    if "sensor_noise" not in config["navigation"]:
        config["navigation"]["sensor_noise"] = {}

    config["navigation"]["sensor_noise"]["airspeed_sigma_m_s"] = st.sidebar.slider(
        "Airspeed noise sigma [m/s]",
        min_value=0.0,
        max_value=2.0,
        value=0.30,
        step=0.05,
    )

    config["navigation"]["sensor_noise"]["alpha_sigma_deg"] = st.sidebar.slider(
        "AOA noise sigma [deg]",
        min_value=0.0,
        max_value=3.0,
        value=0.50,
        step=0.10,
    )

    config["navigation"]["sensor_noise"]["gyro_q_sigma_deg_s"] = st.sidebar.slider(
        "Pitch gyro noise sigma [deg/s]",
        min_value=0.0,
        max_value=2.0,
        value=0.20,
        step=0.05,
    )

    config["navigation"]["sensor_noise"]["pitch_sigma_deg"] = st.sidebar.slider(
        "Pitch angle noise sigma [deg]",
        min_value=0.0,
        max_value=3.0,
        value=0.50,
        step=0.10,
    )

    config["navigation"]["sensor_noise"]["altitude_sigma_m"] = st.sidebar.slider(
        "Barometer noise sigma [m]",
        min_value=0.0,
        max_value=5.0,
        value=1.00,
        step=0.25,
    )

    config["navigation"]["random_seed"] = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )

    return config


def truth_estimate_measurement_plot(
    results,
    true_col: str,
    est_col: str,
    meas_col: str | None,
    title: str,
    y_label: str,
):
    """Create truth / estimate / measurement overlay plot."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["time_s"],
            y=results[true_col],
            mode="lines",
            name="Truth",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results["time_s"],
            y=results[est_col],
            mode="lines",
            name="EKF estimate",
        )
    )

    if meas_col is not None:
        fig.add_trace(
            go.Scatter(
                x=results["time_s"],
                y=results[meas_col],
                mode="markers",
                marker={"size": 3},
                name="Measurement",
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


def error_plot(results):
    """Create estimation error plot."""
    fig = go.Figure()

    error_columns = [
        ("error_altitude_m", "Altitude error [m]"),
        ("error_airspeed_m_s", "Airspeed error [m/s]"),
        ("error_theta_deg", "Pitch error [deg]"),
        ("error_alpha_deg", "AOA error [deg]"),
        ("error_q_deg_s", "Pitch-rate error [deg/s]"),
    ]

    for column, label in error_columns:
        fig.add_trace(
            go.Scatter(
                x=results["time_s"],
                y=results[column],
                mode="lines",
                name=label,
            )
        )

    fig.update_layout(
        title="EKF Estimation Errors",
        xaxis_title="Time [s]",
        yaxis_title="Error",
        hovermode="x unified",
        height=500,
    )

    return fig


def show_metrics(results) -> None:
    """Show EKF RMS metrics."""
    rms_altitude = (results["error_altitude_m"] ** 2).mean() ** 0.5
    rms_airspeed = (results["error_airspeed_m_s"] ** 2).mean() ** 0.5
    rms_pitch = (results["error_theta_deg"] ** 2).mean() ** 0.5
    rms_alpha = (results["error_alpha_deg"] ** 2).mean() ** 0.5

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("RMS altitude error", f"{rms_altitude:.2f} m")
    col2.metric("RMS airspeed error", f"{rms_airspeed:.2f} m/s")
    col3.metric("RMS pitch error", f"{rms_pitch:.2f} deg")
    col4.metric("RMS AOA error", f"{rms_alpha:.2f} deg")


def main() -> None:
    """Run EKF Streamlit dashboard."""
    st.set_page_config(
        page_title="EKF Navigation Dashboard",
        page_icon="🧭",
        layout="wide",
    )

    st.title("EKF Navigation Dashboard")
    st.caption(
        "Noisy air-data, gyro, attitude, and barometric measurements fused "
        "with a nonlinear longitudinal aircraft model."
    )

    base_config = load_json_config(PROJECT_ROOT / "configs" / "uav_longitudinal.json")
    config = update_config_from_sidebar(base_config)

    run_button = st.sidebar.button("Run EKF Simulation", type="primary")

    if "ekf_results" not in st.session_state or run_button:
        with st.spinner("Running truth simulation and EKF estimation..."):
            st.session_state["ekf_results"] = run_ekf_navigation_simulation(config)

    results = st.session_state["ekf_results"]

    show_metrics(results)

    st.divider()

    tab_estimates, tab_errors, tab_data = st.tabs(
        ["Truth vs Estimate", "Errors", "Data"]
    )

    with tab_estimates:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                truth_estimate_measurement_plot(
                    results,
                    true_col="true_altitude_m",
                    est_col="est_altitude_m",
                    meas_col="meas_altitude_m",
                    title="Altitude Estimation",
                    y_label="Altitude [m]",
                ),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                truth_estimate_measurement_plot(
                    results,
                    true_col="true_airspeed_m_s",
                    est_col="est_airspeed_m_s",
                    meas_col="meas_airspeed_m_s",
                    title="Airspeed Estimation",
                    y_label="Airspeed [m/s]",
                ),
                use_container_width=True,
            )

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(
                truth_estimate_measurement_plot(
                    results,
                    true_col="true_theta_deg",
                    est_col="est_theta_deg",
                    meas_col="meas_theta_deg",
                    title="Pitch Angle Estimation",
                    y_label="Pitch angle [deg]",
                ),
                use_container_width=True,
            )

        with col4:
            st.plotly_chart(
                truth_estimate_measurement_plot(
                    results,
                    true_col="true_alpha_deg",
                    est_col="est_alpha_deg",
                    meas_col="meas_alpha_deg",
                    title="Angle-of-Attack Estimation",
                    y_label="AOA [deg]",
                ),
                use_container_width=True,
            )

    with tab_errors:
        st.plotly_chart(error_plot(results), use_container_width=True)

    with tab_data:
        st.dataframe(results, use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download EKF Results CSV",
            data=csv_data,
            file_name="ekf_navigation_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()