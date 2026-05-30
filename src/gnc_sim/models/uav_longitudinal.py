"""Longitudinal fixed-wing UAV dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LongitudinalState:
    """Longitudinal state representation.

    Sign convention:
    - u is body-axis forward velocity.
    - w is body-axis vertical velocity, positive down.
    - q is pitch rate, positive nose-up.
    - theta is pitch angle, positive nose-up.
    - h is altitude, positive upward.
    """

    u_m_s: float
    w_m_s: float
    q_rad_s: float
    theta_rad: float
    h_m: float

    @property
    def airspeed_m_s(self) -> float:
        """Return total airspeed magnitude."""
        return float(np.hypot(self.u_m_s, self.w_m_s))

    @property
    def alpha_rad(self) -> float:
        """Return angle of attack."""
        return float(np.arctan2(self.w_m_s, self.u_m_s))

    def as_array(self) -> np.ndarray:
        """Convert state to NumPy array."""
        return np.array(
            [self.u_m_s, self.w_m_s, self.q_rad_s, self.theta_rad, self.h_m],
            dtype=float,
        )

    @classmethod
    def from_array(cls, x: np.ndarray) -> "LongitudinalState":
        """Create state from array-like vector."""
        return cls(
            u_m_s=float(x[0]),
            w_m_s=float(x[1]),
            q_rad_s=float(x[2]),
            theta_rad=float(x[3]),
            h_m=float(x[4]),
        )


@dataclass
class AeroForcesMoments:
    """Aerodynamic and propulsive resultants."""

    x_force_n: float
    z_force_n: float
    pitch_moment_n_m: float
    lift_n: float
    drag_n: float
    cl: float
    cd: float
    cm: float
    alpha_rad: float


class LongitudinalAircraft:
    """Longitudinal aircraft model."""

    def __init__(self, aircraft_config: dict) -> None:
        self.name = aircraft_config["name"]
        self.mass_kg = aircraft_config["mass_kg"]
        self.iy_kg_m2 = aircraft_config["iy_kg_m2"]
        self.wing_area_m2 = aircraft_config["wing_area_m2"]
        self.mean_chord_m = aircraft_config["mean_chord_m"]
        self.max_thrust_n = aircraft_config["max_thrust_n"]
        self.rho_kg_m3 = aircraft_config["rho_kg_m3"]
        self.gravity_m_s2 = aircraft_config["gravity_m_s2"]
        self.aero = aircraft_config["aero"]

    def initial_state_from_config(self, simulation_config: dict) -> LongitudinalState:
        """Create initial state from simulation config."""
        initial = simulation_config["initial"]
        va = initial["airspeed_m_s"]
        alpha = initial["alpha_rad"]

        return LongitudinalState(
            u_m_s=va * np.cos(alpha),
            w_m_s=va * np.sin(alpha),
            q_rad_s=initial["q_rad_s"],
            theta_rad=initial["theta_rad"],
            h_m=initial["altitude_m"],
        )

    def forces_and_moments(
        self,
        state: LongitudinalState,
        elevator_rad: float,
        throttle: float,
        vertical_gust_m_s: float = 0.0,
    ) -> AeroForcesMoments:
        """Compute forces and moments.

        Parameters
        ----------
        state:
            Current aircraft state.
        elevator_rad:
            Elevator deflection in radians. In this simplified model, positive elevator
            creates a positive nose-up pitching moment.
        throttle:
            Throttle command from 0 to 1.
        vertical_gust_m_s:
            Simple vertical gust correction used only for aerodynamic angle of attack.
        """
        u = state.u_m_s
        w = state.w_m_s
        q = state.q_rad_s

        # This simple gust hook perturbs the relative vertical component used by aero.
        w_aero = w - vertical_gust_m_s

        va = max(float(np.hypot(u, w_aero)), 1e-3)
        alpha = float(np.arctan2(w_aero, u))
        q_hat = q * self.mean_chord_m / (2.0 * va)

        cl = (
            self.aero["cl0"]
            + self.aero["cl_alpha_per_rad"] * alpha
            + self.aero["cl_q"] * q_hat
            + self.aero["cl_delta_e_per_rad"] * elevator_rad
        )
        cd = self.aero["cd0"] + self.aero["oswald_k"] * cl**2
        cm = (
            self.aero["cm0"]
            + self.aero["cm_alpha_per_rad"] * alpha
            + self.aero["cm_q"] * q_hat
            + self.aero["cm_delta_e_per_rad"] * elevator_rad
        )

        dynamic_pressure = 0.5 * self.rho_kg_m3 * va**2
        lift = dynamic_pressure * self.wing_area_m2 * cl
        drag = dynamic_pressure * self.wing_area_m2 * cd
        thrust = float(np.clip(throttle, 0.0, 1.0)) * self.max_thrust_n

        # Body-axis aero force conversion. z body axis is positive down.
        x_aero = -drag * np.cos(alpha) + lift * np.sin(alpha)
        z_aero = -drag * np.sin(alpha) - lift * np.cos(alpha)

        x_force = x_aero + thrust
        z_force = z_aero
        pitch_moment = dynamic_pressure * self.wing_area_m2 * self.mean_chord_m * cm

        return AeroForcesMoments(
            x_force_n=float(x_force),
            z_force_n=float(z_force),
            pitch_moment_n_m=float(pitch_moment),
            lift_n=float(lift),
            drag_n=float(drag),
            cl=float(cl),
            cd=float(cd),
            cm=float(cm),
            alpha_rad=float(alpha),
        )

    def dynamics(
        self,
        state: LongitudinalState,
        elevator_rad: float,
        throttle: float,
        vertical_gust_m_s: float = 0.0,
    ) -> np.ndarray:
        """Return time derivative of state vector."""
        u = state.u_m_s
        w = state.w_m_s
        q = state.q_rad_s
        theta = state.theta_rad

        fm = self.forces_and_moments(
            state=state,
            elevator_rad=elevator_rad,
            throttle=throttle,
            vertical_gust_m_s=vertical_gust_m_s,
        )

        u_dot = fm.x_force_n / self.mass_kg - self.gravity_m_s2 * np.sin(theta) - q * w
        w_dot = fm.z_force_n / self.mass_kg + self.gravity_m_s2 * np.cos(theta) + q * u
        q_dot = fm.pitch_moment_n_m / self.iy_kg_m2
        theta_dot = q

        # Altitude rate, positive upward.
        h_dot = u * np.sin(theta) - w * np.cos(theta)

        return np.array([u_dot, w_dot, q_dot, theta_dot, h_dot], dtype=float)
