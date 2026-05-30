"""Longitudinal autopilot.

Architecture:

altitude error -> pitch command -> elevator command
airspeed error -> throttle command
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnc_sim.controllers.pid import PIDController, PIDGains
from gnc_sim.models.uav_longitudinal import LongitudinalState


@dataclass
class LongitudinalCommands:
    """External commands for the longitudinal autopilot."""

    altitude_m: float
    airspeed_m_s: float


@dataclass
class LongitudinalControl:
    """Control output from the longitudinal autopilot."""

    elevator_rad: float
    throttle: float
    theta_cmd_rad: float


class LongitudinalAutopilot:
    """Cascaded PID longitudinal autopilot."""

    def __init__(self, config: dict) -> None:
        aircraft = config["aircraft"]
        ctrl_cfg = config["controller"]

        elevator_limits = tuple(aircraft["limits"]["elevator_rad"])
        throttle_limits = tuple(aircraft["limits"]["throttle"])
        self.theta_cmd_limits = tuple(aircraft["limits"]["theta_command_rad"])

        self.elevator_trim = ctrl_cfg["trim"]["elevator_rad"]
        self.throttle_trim = ctrl_cfg["trim"]["throttle"]

        self.altitude_to_pitch = PIDController(
            gains=PIDGains(**ctrl_cfg["altitude_to_pitch_pid"]),
            output_limits=self.theta_cmd_limits,
        )

        self.pitch_to_elevator = PIDController(
            gains=PIDGains(**ctrl_cfg["pitch_to_elevator_pid"]),
            output_limits=elevator_limits,
        )

        self.airspeed_to_throttle = PIDController(
            gains=PIDGains(**ctrl_cfg["airspeed_to_throttle_pid"]),
            output_limits=(
                throttle_limits[0] - self.throttle_trim,
                throttle_limits[1] - self.throttle_trim,
            ),
        )

    def update(
        self,
        state: LongitudinalState,
        commands: LongitudinalCommands,
        dt: float,
    ) -> LongitudinalControl:
        """Compute elevator and throttle commands."""
        altitude_error = commands.altitude_m - state.h_m

        theta_cmd = self.altitude_to_pitch.update(altitude_error, dt)
        theta_cmd = float(np.clip(theta_cmd, *self.theta_cmd_limits))

        theta_error = theta_cmd - state.theta_rad
        elevator = self.elevator_trim + self.pitch_to_elevator.update(theta_error, dt)

        airspeed_error = commands.airspeed_m_s - state.airspeed_m_s
        throttle = self.throttle_trim + self.airspeed_to_throttle.update(
            airspeed_error, dt
        )
        throttle = float(np.clip(throttle, 0.0, 1.0))

        return LongitudinalControl(
            elevator_rad=float(elevator),
            throttle=float(throttle),
            theta_cmd_rad=float(theta_cmd),
        )
