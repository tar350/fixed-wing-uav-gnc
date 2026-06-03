"""Estimator-in-the-loop longitudinal GNC simulation.

This module closes the loop between:

    noisy sensors -> EKF navigation -> LQR controller -> nonlinear aircraft

Unlike the earlier LQR closed-loop simulation, the controller does not use the
true simulated state. It uses the EKF estimated state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnc_sim.controllers import design_lqr
from gnc_sim.environment.wind import vertical_gust_m_s
from gnc_sim.linearization import linearize_longitudinal
from gnc_sim.models.uav_longitudinal import LongitudinalAircraft, LongitudinalState
from gnc_sim.navigation import EKFConfig, LongitudinalEKF
from gnc_sim.sensors import SensorNoiseConfig, generate_longitudinal_measurement
from gnc_sim.simulation.integrator import rk4_step
from gnc_sim.trim.longitudinal_trim import solve_longitudinal_trim


def _get_lqr_weights(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return LQR Q and R matrices from config if available."""
    if "lqr" in config:
        lqr_cfg = config["lqr"]
        q_diag = lqr_cfg.get("q_diag", [1.0, 1.0, 20.0, 40.0, 0.20])
        r_diag = lqr_cfg.get("r_diag", [5.0, 2.0])
    else:
        q_diag = [1.0, 1.0, 20.0, 40.0, 0.20]
        r_diag = [5.0, 2.0]

    return np.diag(q_diag), np.diag(r_diag)


def _make_reference_state(
    trim_state: LongitudinalState,
    altitude_command_m: float,
) -> LongitudinalState:
    """Create reference state for the LQR controller."""
    return LongitudinalState(
        u_m_s=trim_state.u_m_s,
        w_m_s=trim_state.w_m_s,
        q_rad_s=0.0,
        theta_rad=trim_state.theta_rad,
        h_m=float(altitude_command_m),
    )


def _apply_control_limits(
    elevator_rad: float,
    throttle: float,
    aircraft_config: dict,
) -> tuple[float, float]:
    """Apply elevator and throttle saturation limits."""
    elevator_limits = aircraft_config["limits"]["elevator_rad"]
    throttle_limits = aircraft_config["limits"]["throttle"]

    elevator_sat = float(np.clip(elevator_rad, elevator_limits[0], elevator_limits[1]))
    throttle_sat = float(np.clip(throttle, throttle_limits[0], throttle_limits[1]))

    return elevator_sat, throttle_sat


def _initial_estimate_from_measurement(z: np.ndarray) -> np.ndarray:
    """Create initial EKF state estimate from first measurement."""
    airspeed = float(z[0])
    alpha = float(z[1])
    q = float(z[2])
    theta = float(z[3])
    altitude = float(z[4])

    u = airspeed * np.cos(alpha)
    w = airspeed * np.sin(alpha)

    return np.array([u, w, q, theta, altitude], dtype=float)


def _get_random_seed(config: dict) -> int:
    """Return deterministic random seed."""
    return int(config.get("navigation", {}).get("random_seed", 42))


def simulate_estimator_in_loop_longitudinal(config: dict) -> pd.DataFrame:
    """Run estimator-in-the-loop longitudinal GNC simulation.

    The truth aircraft is propagated using the nonlinear aircraft dynamics.
    Sensors are generated from the truth state.
    The EKF estimates the state from noisy measurements.
    The LQR controller uses the EKF estimate, not the truth state.
    """
    aircraft_cfg = config["aircraft"]
    sim_cfg = config["simulation"]
    trim_cfg = config["trim"]

    aircraft = LongitudinalAircraft(aircraft_cfg)

    dt = float(sim_cfg["dt_s"])
    t_final = float(sim_cfg["t_final_s"])

    altitude_command_m = float(sim_cfg["commands"]["altitude_m"])
    airspeed_command_m_s = float(sim_cfg["commands"]["airspeed_m_s"])

    trim_result = solve_longitudinal_trim(
        aircraft=aircraft,
        target_airspeed_m_s=airspeed_command_m_s,
        target_altitude_m=trim_cfg["target_altitude_m"],
        flight_path_angle_rad=np.radians(trim_cfg["flight_path_angle_deg"]),
        initial_guess=(
            trim_cfg["initial_guess"]["alpha_rad"],
            trim_cfg["initial_guess"]["elevator_rad"],
            trim_cfg["initial_guess"]["throttle"],
        ),
        elevator_bounds_rad=tuple(aircraft_cfg["limits"]["elevator_rad"]),
        residual_tolerance=trim_cfg["residual_tolerance"],
    )

    if not trim_result.success:
        raise RuntimeError(f"Trim failed: {trim_result.message}")

    trim_state = trim_result.state()

    linearization = linearize_longitudinal(
        aircraft=aircraft,
        trim_state=trim_state,
        trim_elevator_rad=trim_result.elevator_rad,
        trim_throttle=trim_result.throttle,
    )

    Q, R = _get_lqr_weights(config)

    lqr_result = design_lqr(
        A=linearization.A,
        B=linearization.B,
        Q=Q,
        R=R,
        state_names=linearization.state_names,
        input_names=linearization.input_names,
    )

    reference_state = _make_reference_state(
        trim_state=trim_state,
        altitude_command_m=altitude_command_m,
    )

    truth_state = aircraft.initial_state_from_config(sim_cfg)

    sensor_noise = SensorNoiseConfig.from_config(config)
    ekf_config = EKFConfig.from_config(config)
    rng = np.random.default_rng(_get_random_seed(config))

    initial_measurement = generate_longitudinal_measurement(
        state=truth_state,
        noise_config=sensor_noise,
        rng=rng,
    )

    ekf = LongitudinalEKF(
        aircraft=aircraft,
        x0=_initial_estimate_from_measurement(initial_measurement.as_array()),
        measurement_covariance=sensor_noise.covariance(),
        config=ekf_config,
    )

    times = np.arange(0.0, t_final + dt, dt)
    records: list[dict] = []

    for t in times:
        gust = vertical_gust_m_s(t, sim_cfg["wind"])

        measurement = generate_longitudinal_measurement(
            state=truth_state,
            noise_config=sensor_noise,
            rng=rng,
        )

        # Navigation update: estimate current state from current sensor packet.
        ekf.update(measurement.as_array())
        estimated_state = ekf.state

        # Control law uses estimated state, not truth state.
        x_delta_est = estimated_state.as_array() - reference_state.as_array()
        control_delta = -lqr_result.K @ x_delta_est

        elevator_unsat = trim_result.elevator_rad + float(control_delta[0])
        throttle_unsat = trim_result.throttle + float(control_delta[1])

        elevator_rad, throttle = _apply_control_limits(
            elevator_rad=elevator_unsat,
            throttle=throttle_unsat,
            aircraft_config=aircraft_cfg,
        )

        fm = aircraft.forces_and_moments(
            state=truth_state,
            elevator_rad=elevator_rad,
            throttle=throttle,
            vertical_gust_m_s=gust,
        )

        records.append(
            {
                "time_s": t,

                # Generic truth columns for plotting/animation compatibility.
                "u_m_s": truth_state.u_m_s,
                "w_m_s": truth_state.w_m_s,
                "airspeed_m_s": truth_state.airspeed_m_s,
                "alpha_deg": np.degrees(truth_state.alpha_rad),
                "q_deg_s": np.degrees(truth_state.q_rad_s),
                "theta_deg": np.degrees(truth_state.theta_rad),
                "altitude_m": truth_state.h_m,

                # Truth state.
                "true_u_m_s": truth_state.u_m_s,
                "true_w_m_s": truth_state.w_m_s,
                "true_q_rad_s": truth_state.q_rad_s,
                "true_q_deg_s": np.degrees(truth_state.q_rad_s),
                "true_theta_rad": truth_state.theta_rad,
                "true_theta_deg": np.degrees(truth_state.theta_rad),
                "true_altitude_m": truth_state.h_m,
                "true_airspeed_m_s": truth_state.airspeed_m_s,
                "true_alpha_rad": truth_state.alpha_rad,
                "true_alpha_deg": np.degrees(truth_state.alpha_rad),

                # EKF estimate.
                "est_u_m_s": estimated_state.u_m_s,
                "est_w_m_s": estimated_state.w_m_s,
                "est_q_rad_s": estimated_state.q_rad_s,
                "est_q_deg_s": np.degrees(estimated_state.q_rad_s),
                "est_theta_rad": estimated_state.theta_rad,
                "est_theta_deg": np.degrees(estimated_state.theta_rad),
                "est_altitude_m": estimated_state.h_m,
                "est_airspeed_m_s": estimated_state.airspeed_m_s,
                "est_alpha_rad": estimated_state.alpha_rad,
                "est_alpha_deg": np.degrees(estimated_state.alpha_rad),

                # Measurements.
                "meas_airspeed_m_s": measurement.airspeed_m_s,
                "meas_alpha_deg": np.degrees(measurement.alpha_rad),
                "meas_q_deg_s": np.degrees(measurement.q_rad_s),
                "meas_theta_deg": np.degrees(measurement.theta_rad),
                "meas_altitude_m": measurement.altitude_m,

                # Commands and control.
                "altitude_cmd_m": altitude_command_m,
                "airspeed_cmd_m_s": airspeed_command_m_s,
                "theta_cmd_deg": np.degrees(reference_state.theta_rad),
                "elevator_deg": np.degrees(elevator_rad),
                "throttle": throttle,
                "elevator_unsat_deg": np.degrees(elevator_unsat),
                "throttle_unsat": throttle_unsat,

                # Tracking errors.
                "altitude_error_m": altitude_command_m - truth_state.h_m,
                "airspeed_error_m_s": airspeed_command_m_s - truth_state.airspeed_m_s,

                # Estimation errors.
                "error_u_m_s": estimated_state.u_m_s - truth_state.u_m_s,
                "error_w_m_s": estimated_state.w_m_s - truth_state.w_m_s,
                "error_q_deg_s": np.degrees(
                    estimated_state.q_rad_s - truth_state.q_rad_s
                ),
                "error_theta_deg": np.degrees(
                    estimated_state.theta_rad - truth_state.theta_rad
                ),
                "error_altitude_m": estimated_state.h_m - truth_state.h_m,
                "error_airspeed_m_s": (
                    estimated_state.airspeed_m_s - truth_state.airspeed_m_s
                ),
                "error_alpha_deg": np.degrees(
                    estimated_state.alpha_rad - truth_state.alpha_rad
                ),

                # Environment and aero.
                "vertical_gust_m_s": gust,
                "trim_alpha_deg": trim_result.alpha_deg,
                "trim_theta_deg": trim_result.theta_deg,
                "trim_elevator_deg": trim_result.elevator_deg,
                "trim_throttle": trim_result.throttle,
                "lift_n": fm.lift_n,
                "drag_n": fm.drag_n,
                "cl": fm.cl,
                "cd": fm.cd,
                "cm": fm.cm,
            }
        )

        def truth_derivative(x_array: np.ndarray) -> np.ndarray:
            temp_state = LongitudinalState.from_array(x_array)

            return aircraft.dynamics(
                state=temp_state,
                elevator_rad=elevator_rad,
                throttle=throttle,
                vertical_gust_m_s=gust,
            )

        next_truth_array = rk4_step(truth_derivative, truth_state.as_array(), dt)
        truth_state = LongitudinalState.from_array(next_truth_array)

        # Prediction step propagates the estimator forward using the same control.
        # The estimator assumes gust is not directly measured.
        ekf.predict(
            elevator_rad=elevator_rad,
            throttle=throttle,
            dt_s=dt,
            vertical_gust_m_s=0.0,
        )

    return pd.DataFrame.from_records(records)