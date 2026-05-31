"""Closed-loop nonlinear longitudinal simulation using an LQR controller.

The LQR gain is designed from a linearized model about trim, then applied to the
nonlinear longitudinal aircraft model using:

    u_delta = -K x_delta

where:
    x_delta = x_current - x_reference

The final aircraft commands are:

    elevator = elevator_trim + elevator_delta
    throttle = throttle_trim + throttle_delta

This module is intentionally separate from the PID simulation so PID and LQR can
be compared cleanly in later versions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnc_sim.controllers import design_lqr
from gnc_sim.environment.wind import vertical_gust_m_s
from gnc_sim.linearization import linearize_longitudinal
from gnc_sim.models.uav_longitudinal import LongitudinalAircraft, LongitudinalState
from gnc_sim.simulation.integrator import rk4_step
from gnc_sim.trim.longitudinal_trim import solve_longitudinal_trim


def _get_lqr_weights(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return LQR Q and R matrices from config if available, otherwise defaults."""
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
    """Create the reference state used by the LQR controller.

    The current longitudinal LQR regulates the aircraft to the trim speed,
    trim angle of attack, zero pitch rate, trim pitch angle, and commanded altitude.
    """
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
    """Saturate elevator and throttle commands using aircraft limits."""
    elevator_limits = aircraft_config["limits"]["elevator_rad"]
    throttle_limits = aircraft_config["limits"]["throttle"]

    elevator_sat = float(np.clip(elevator_rad, elevator_limits[0], elevator_limits[1]))
    throttle_sat = float(np.clip(throttle, throttle_limits[0], throttle_limits[1]))

    return elevator_sat, throttle_sat


def simulate_lqr_longitudinal(config: dict) -> pd.DataFrame:
    """Run nonlinear closed-loop longitudinal simulation using LQR.

    Parameters
    ----------
    config:
        Project configuration dictionary.

    Returns
    -------
    pandas.DataFrame
        Time history of states, commands, controls, trim values, and selected
        aerodynamic quantities.
    """
    aircraft = LongitudinalAircraft(config["aircraft"])
    aircraft_cfg = config["aircraft"]
    sim_cfg = config["simulation"]
    trim_cfg = config["trim"]

    dt = sim_cfg["dt_s"]
    t_final = sim_cfg["t_final_s"]

    altitude_command_m = sim_cfg["commands"]["altitude_m"]
    airspeed_command_m_s = sim_cfg["commands"]["airspeed_m_s"]

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

    state = aircraft.initial_state_from_config(sim_cfg)

    times = np.arange(0.0, t_final + dt, dt)
    records: list[dict] = []

    for t in times:
        x_delta = state.as_array() - reference_state.as_array()

        control_delta = -lqr_result.K @ x_delta

        elevator_unsat = trim_result.elevator_rad + float(control_delta[0])
        throttle_unsat = trim_result.throttle + float(control_delta[1])

        elevator_rad, throttle = _apply_control_limits(
            elevator_rad=elevator_unsat,
            throttle=throttle_unsat,
            aircraft_config=aircraft_cfg,
        )

        gust = vertical_gust_m_s(t, sim_cfg["wind"])

        fm = aircraft.forces_and_moments(
            state=state,
            elevator_rad=elevator_rad,
            throttle=throttle,
            vertical_gust_m_s=gust,
        )

        records.append(
            {
                "time_s": t,
                "u_m_s": state.u_m_s,
                "w_m_s": state.w_m_s,
                "airspeed_m_s": state.airspeed_m_s,
                "alpha_deg": np.degrees(state.alpha_rad),
                "q_deg_s": np.degrees(state.q_rad_s),
                "theta_deg": np.degrees(state.theta_rad),
                "altitude_m": state.h_m,
                "altitude_cmd_m": altitude_command_m,
                "airspeed_cmd_m_s": airspeed_command_m_s,
                "theta_cmd_deg": np.degrees(reference_state.theta_rad),
                "elevator_deg": np.degrees(elevator_rad),
                "throttle": throttle,
                "elevator_unsat_deg": np.degrees(elevator_unsat),
                "throttle_unsat": throttle_unsat,
                "vertical_gust_m_s": gust,
                "altitude_error_m": altitude_command_m - state.h_m,
                "airspeed_error_m_s": airspeed_command_m_s - state.airspeed_m_s,
                "u_delta_m_s": x_delta[0],
                "w_delta_m_s": x_delta[1],
                "q_delta_rad_s": x_delta[2],
                "theta_delta_rad": x_delta[3],
                "h_delta_m": x_delta[4],
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

        def derivative(x_array: np.ndarray) -> np.ndarray:
            temp_state = LongitudinalState.from_array(x_array)
            return aircraft.dynamics(
                state=temp_state,
                elevator_rad=elevator_rad,
                throttle=throttle,
                vertical_gust_m_s=gust,
            )

        next_state_array = rk4_step(derivative, state.as_array(), dt)
        state = LongitudinalState.from_array(next_state_array)

    return pd.DataFrame.from_records(records)