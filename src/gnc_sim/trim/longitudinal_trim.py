"""Longitudinal trim solver.

The trim solver finds a steady flight condition for the simplified longitudinal
fixed-wing aircraft model.  It solves for angle of attack, elevator deflection,
and throttle such that the translational and pitch accelerations are near zero.

State convention inherited from the aircraft model:
- u is body-axis forward velocity, positive forward.
- w is body-axis vertical velocity, positive down.
- q is pitch rate, positive nose-up.
- theta is pitch angle, positive nose-up.
- h is altitude, positive upward.

For a desired flight-path angle gamma, altitude rate is enforced kinematically by
setting theta = alpha + gamma.  In steady level flight, gamma = 0, so theta = alpha.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
from scipy.optimize import least_squares

from gnc_sim.models.uav_longitudinal import LongitudinalAircraft, LongitudinalState


@dataclass(frozen=True)
class LongitudinalTrimResult:
    """Container for a solved longitudinal trim condition."""

    success: bool
    message: str
    target_airspeed_m_s: float
    target_altitude_m: float
    flight_path_angle_rad: float
    alpha_rad: float
    theta_rad: float
    elevator_rad: float
    throttle: float
    u_m_s: float
    w_m_s: float
    q_rad_s: float
    residual_u_dot_m_s2: float
    residual_w_dot_m_s2: float
    residual_q_dot_rad_s2: float
    residual_h_dot_m_s: float
    residual_norm: float
    lift_n: float
    drag_n: float
    cl: float
    cd: float
    cm: float

    @property
    def alpha_deg(self) -> float:
        """Angle of attack in degrees."""
        return float(np.degrees(self.alpha_rad))

    @property
    def theta_deg(self) -> float:
        """Pitch angle in degrees."""
        return float(np.degrees(self.theta_rad))

    @property
    def elevator_deg(self) -> float:
        """Elevator deflection in degrees."""
        return float(np.degrees(self.elevator_rad))

    @property
    def flight_path_angle_deg(self) -> float:
        """Flight-path angle in degrees."""
        return float(np.degrees(self.flight_path_angle_rad))

    def state(self) -> LongitudinalState:
        """Return the trim state."""
        return LongitudinalState(
            u_m_s=self.u_m_s,
            w_m_s=self.w_m_s,
            q_rad_s=self.q_rad_s,
            theta_rad=self.theta_rad,
            h_m=self.target_altitude_m,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        data = asdict(self)
        data.update(
            {
                "alpha_deg": self.alpha_deg,
                "theta_deg": self.theta_deg,
                "elevator_deg": self.elevator_deg,
                "flight_path_angle_deg": self.flight_path_angle_deg,
            }
        )
        return data

    def save_json(self, output_path: str | Path) -> None:
        """Save the trim result to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)


def _make_state_from_trim_variables(
    airspeed_m_s: float,
    altitude_m: float,
    flight_path_angle_rad: float,
    alpha_rad: float,
) -> LongitudinalState:
    """Build a longitudinal state from trim variables."""
    theta_rad = alpha_rad + flight_path_angle_rad
    return LongitudinalState(
        u_m_s=float(airspeed_m_s * np.cos(alpha_rad)),
        w_m_s=float(airspeed_m_s * np.sin(alpha_rad)),
        q_rad_s=0.0,
        theta_rad=float(theta_rad),
        h_m=float(altitude_m),
    )


def solve_longitudinal_trim(
    aircraft: LongitudinalAircraft,
    target_airspeed_m_s: float,
    target_altitude_m: float,
    flight_path_angle_rad: float = 0.0,
    initial_guess: tuple[float, float, float] = (0.10, 0.05, 0.35),
    alpha_bounds_rad: tuple[float, float] = (-0.20, 0.30),
    elevator_bounds_rad: tuple[float, float] | None = None,
    throttle_bounds: tuple[float, float] = (0.0, 1.0),
    residual_tolerance: float = 1.0e-6,
) -> LongitudinalTrimResult:
    """Solve for a longitudinal trim point.

    Parameters
    ----------
    aircraft:
        Longitudinal aircraft model.
    target_airspeed_m_s:
        Desired steady airspeed.
    target_altitude_m:
        Desired altitude. The current simplified model uses constant density,
        but altitude is retained in the trim state for traceability.
    flight_path_angle_rad:
        Desired flight-path angle. Use zero for level flight.
    initial_guess:
        Initial guess tuple: ``(alpha_rad, elevator_rad, throttle)``.
    alpha_bounds_rad:
        Lower and upper angle-of-attack bounds.
    elevator_bounds_rad:
        Lower and upper elevator bounds. If omitted, the solver uses the aircraft
        limits from configuration when available, otherwise ±25 degrees.
    throttle_bounds:
        Lower and upper throttle bounds.
    residual_tolerance:
        Norm threshold used to mark the trim as successful.

    Returns
    -------
    LongitudinalTrimResult
        Trim solution, residuals, and selected aerodynamic quantities.
    """
    if target_airspeed_m_s <= 0.0:
        raise ValueError("target_airspeed_m_s must be positive.")

    if elevator_bounds_rad is None:
        # The v0.1 aircraft class stores most fields as attributes, but not limits.
        # Use a conservative default if limits were not passed explicitly.
        elevator_bounds_rad = (-np.radians(25.0), np.radians(25.0))

    lower_bounds = np.array(
        [alpha_bounds_rad[0], elevator_bounds_rad[0], throttle_bounds[0]], dtype=float
    )
    upper_bounds = np.array(
        [alpha_bounds_rad[1], elevator_bounds_rad[1], throttle_bounds[1]], dtype=float
    )

    def residual(trim_variables: np.ndarray) -> np.ndarray:
        alpha_rad, elevator_rad, throttle = trim_variables
        state = _make_state_from_trim_variables(
            airspeed_m_s=target_airspeed_m_s,
            altitude_m=target_altitude_m,
            flight_path_angle_rad=flight_path_angle_rad,
            alpha_rad=float(alpha_rad),
        )
        state_dot = aircraft.dynamics(
            state=state,
            elevator_rad=float(elevator_rad),
            throttle=float(throttle),
            vertical_gust_m_s=0.0,
        )
        return np.array([state_dot[0], state_dot[1], state_dot[2]], dtype=float)

    solution = least_squares(
        residual,
        x0=np.array(initial_guess, dtype=float),
        bounds=(lower_bounds, upper_bounds),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        x_scale=np.array([0.10, 0.10, 0.50], dtype=float),
        max_nfev=200,
    )

    alpha_rad, elevator_rad, throttle = [float(value) for value in solution.x]
    trim_state = _make_state_from_trim_variables(
        airspeed_m_s=target_airspeed_m_s,
        altitude_m=target_altitude_m,
        flight_path_angle_rad=flight_path_angle_rad,
        alpha_rad=alpha_rad,
    )
    state_dot = aircraft.dynamics(
        state=trim_state,
        elevator_rad=elevator_rad,
        throttle=throttle,
        vertical_gust_m_s=0.0,
    )
    fm = aircraft.forces_and_moments(
        state=trim_state,
        elevator_rad=elevator_rad,
        throttle=throttle,
        vertical_gust_m_s=0.0,
    )

    residual_norm = float(np.linalg.norm(state_dot[:3]))
    success = bool(solution.success and residual_norm <= residual_tolerance)
    message = (
        "Trim converged."
        if success
        else f"Trim did not meet tolerance. Optimizer message: {solution.message}"
    )

    return LongitudinalTrimResult(
        success=success,
        message=message,
        target_airspeed_m_s=float(target_airspeed_m_s),
        target_altitude_m=float(target_altitude_m),
        flight_path_angle_rad=float(flight_path_angle_rad),
        alpha_rad=alpha_rad,
        theta_rad=float(trim_state.theta_rad),
        elevator_rad=elevator_rad,
        throttle=throttle,
        u_m_s=float(trim_state.u_m_s),
        w_m_s=float(trim_state.w_m_s),
        q_rad_s=float(trim_state.q_rad_s),
        residual_u_dot_m_s2=float(state_dot[0]),
        residual_w_dot_m_s2=float(state_dot[1]),
        residual_q_dot_rad_s2=float(state_dot[2]),
        residual_h_dot_m_s=float(state_dot[4] - target_airspeed_m_s * np.sin(flight_path_angle_rad)),
        residual_norm=residual_norm,
        lift_n=float(fm.lift_n),
        drag_n=float(fm.drag_n),
        cl=float(fm.cl),
        cd=float(fm.cd),
        cm=float(fm.cm),
    )
