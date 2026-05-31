"""Navigation simulation using noisy sensors and a longitudinal EKF."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnc_sim.models.uav_longitudinal import LongitudinalAircraft, LongitudinalState
from gnc_sim.navigation import EKFConfig, LongitudinalEKF
from gnc_sim.sensors import SensorNoiseConfig, generate_longitudinal_measurement
from gnc_sim.simulation.lqr_closed_loop import simulate_lqr_longitudinal


def _state_from_truth_row(row: pd.Series) -> LongitudinalState:
    """Build LongitudinalState from a logged truth dataframe row."""
    return LongitudinalState(
        u_m_s=float(row["u_m_s"]),
        w_m_s=float(row["w_m_s"]),
        q_rad_s=float(np.radians(row["q_deg_s"])),
        theta_rad=float(np.radians(row["theta_deg"])),
        h_m=float(row["altitude_m"]),
    )


def _initial_estimate_from_measurement(z: np.ndarray) -> np.ndarray:
    """Create EKF initial state estimate from first measurement packet."""
    airspeed = float(z[0])
    alpha = float(z[1])
    q = float(z[2])
    theta = float(z[3])
    altitude = float(z[4])

    u = airspeed * np.cos(alpha)
    w = airspeed * np.sin(alpha)

    return np.array([u, w, q, theta, altitude], dtype=float)


def _get_random_seed(config: dict) -> int:
    """Return deterministic sensor seed from config if available."""
    return int(config.get("navigation", {}).get("random_seed", 42))


def run_ekf_navigation_simulation(config: dict) -> pd.DataFrame:
    """Run truth simulation, noisy sensors, and EKF estimation.

    The truth trajectory is generated using the nonlinear closed-loop LQR
    simulation. The EKF then estimates the state using noisy measurements and
    known control inputs from the truth simulation.

    This version does not yet feed estimated states back into the controller.
    That will be a future estimator-in-the-loop version.
    """
    aircraft = LongitudinalAircraft(config["aircraft"])

    sensor_noise = SensorNoiseConfig.from_config(config)
    ekf_config = EKFConfig.from_config(config)

    truth = simulate_lqr_longitudinal(config)

    rng = np.random.default_rng(_get_random_seed(config))

    first_true_state = _state_from_truth_row(truth.iloc[0])
    first_measurement = generate_longitudinal_measurement(
        state=first_true_state,
        noise_config=sensor_noise,
        rng=rng,
    )

    ekf = LongitudinalEKF(
        aircraft=aircraft,
        x0=_initial_estimate_from_measurement(first_measurement.as_array()),
        measurement_covariance=sensor_noise.covariance(),
        config=ekf_config,
    )

    records: list[dict] = []

    previous_time = float(truth.iloc[0]["time_s"])
    previous_elevator_rad = float(np.radians(truth.iloc[0]["elevator_deg"]))
    previous_throttle = float(truth.iloc[0]["throttle"])

    for row_index, row in truth.iterrows():
        current_time = float(row["time_s"])
        true_state = _state_from_truth_row(row)

        if row_index > 0:
            dt_s = current_time - previous_time

            # Estimator currently assumes unknown gust is not directly measured.
            ekf.predict(
                elevator_rad=previous_elevator_rad,
                throttle=previous_throttle,
                dt_s=dt_s,
                vertical_gust_m_s=0.0,
            )

        measurement = generate_longitudinal_measurement(
            state=true_state,
            noise_config=sensor_noise,
            rng=rng,
        )

        ekf.update(measurement.as_array())

        est_state = ekf.state

        records.append(
            {
                "time_s": current_time,
                "true_u_m_s": true_state.u_m_s,
                "true_w_m_s": true_state.w_m_s,
                "true_q_rad_s": true_state.q_rad_s,
                "true_q_deg_s": np.degrees(true_state.q_rad_s),
                "true_theta_rad": true_state.theta_rad,
                "true_theta_deg": np.degrees(true_state.theta_rad),
                "true_altitude_m": true_state.h_m,
                "true_airspeed_m_s": true_state.airspeed_m_s,
                "true_alpha_rad": true_state.alpha_rad,
                "true_alpha_deg": np.degrees(true_state.alpha_rad),
                "est_u_m_s": est_state.u_m_s,
                "est_w_m_s": est_state.w_m_s,
                "est_q_rad_s": est_state.q_rad_s,
                "est_q_deg_s": np.degrees(est_state.q_rad_s),
                "est_theta_rad": est_state.theta_rad,
                "est_theta_deg": np.degrees(est_state.theta_rad),
                "est_altitude_m": est_state.h_m,
                "est_airspeed_m_s": est_state.airspeed_m_s,
                "est_alpha_rad": est_state.alpha_rad,
                "est_alpha_deg": np.degrees(est_state.alpha_rad),
                "meas_airspeed_m_s": measurement.airspeed_m_s,
                "meas_alpha_deg": np.degrees(measurement.alpha_rad),
                "meas_q_deg_s": np.degrees(measurement.q_rad_s),
                "meas_theta_deg": np.degrees(measurement.theta_rad),
                "meas_altitude_m": measurement.altitude_m,
                "error_u_m_s": est_state.u_m_s - true_state.u_m_s,
                "error_w_m_s": est_state.w_m_s - true_state.w_m_s,
                "error_q_deg_s": np.degrees(est_state.q_rad_s - true_state.q_rad_s),
                "error_theta_deg": np.degrees(est_state.theta_rad - true_state.theta_rad),
                "error_altitude_m": est_state.h_m - true_state.h_m,
                "error_airspeed_m_s": est_state.airspeed_m_s - true_state.airspeed_m_s,
                "error_alpha_deg": np.degrees(est_state.alpha_rad - true_state.alpha_rad),
                "elevator_deg": float(row["elevator_deg"]),
                "throttle": float(row["throttle"]),
            }
        )

        previous_time = current_time
        previous_elevator_rad = float(np.radians(row["elevator_deg"]))
        previous_throttle = float(row["throttle"])

    return pd.DataFrame.from_records(records)