"""Time-marching simulation integrator."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from gnc_sim.controllers.autopilot_longitudinal import (
    LongitudinalAutopilot,
    LongitudinalCommands,
)
from gnc_sim.environment.wind import vertical_gust_m_s
from gnc_sim.models.uav_longitudinal import LongitudinalAircraft, LongitudinalState


def rk4_step(derivative_func, state_array: np.ndarray, dt: float) -> np.ndarray:
    """Perform one fourth-order Runge-Kutta integration step."""
    k1 = derivative_func(state_array)
    k2 = derivative_func(state_array + 0.5 * dt * k1)
    k3 = derivative_func(state_array + 0.5 * dt * k2)
    k4 = derivative_func(state_array + dt * k3)

    return state_array + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_longitudinal(config: dict) -> pd.DataFrame:
    """Run the closed-loop longitudinal simulation.

    Returns
    -------
    pandas.DataFrame
        Time history of states, commands, controls, and selected aero values.
    """
    aircraft = LongitudinalAircraft(config["aircraft"])
    autopilot = LongitudinalAutopilot(config)

    sim_cfg = config["simulation"]
    dt = sim_cfg["dt_s"]
    t_final = sim_cfg["t_final_s"]

    commands = LongitudinalCommands(**sim_cfg["commands"])
    state = aircraft.initial_state_from_config(sim_cfg)

    times = np.arange(0.0, t_final + dt, dt)
    records: list[dict] = []

    for t in times:
        control = autopilot.update(state=state, commands=commands, dt=dt)
        gust = vertical_gust_m_s(t, sim_cfg["wind"])
        fm = aircraft.forces_and_moments(
            state=state,
            elevator_rad=control.elevator_rad,
            throttle=control.throttle,
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
                "altitude_cmd_m": commands.altitude_m,
                "airspeed_cmd_m_s": commands.airspeed_m_s,
                "theta_cmd_deg": np.degrees(control.theta_cmd_rad),
                "elevator_deg": np.degrees(control.elevator_rad),
                "throttle": control.throttle,
                "vertical_gust_m_s": gust,
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
                elevator_rad=control.elevator_rad,
                throttle=control.throttle,
                vertical_gust_m_s=gust,
            )

        next_state_array = rk4_step(derivative, state.as_array(), dt)
        state = LongitudinalState.from_array(next_state_array)

    return pd.DataFrame.from_records(records)
