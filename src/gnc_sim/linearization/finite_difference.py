"""Finite-difference linearization utilities.

This module numerically linearizes the nonlinear longitudinal aircraft dynamics
around a solved trim condition. The resulting state-space model has the form:

    x_dot_delta = A x_delta + B u_delta

where x_delta and u_delta are perturbations away from the trim state and trim input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from gnc_sim.models.uav_longitudinal import LongitudinalAircraft, LongitudinalState


@dataclass(frozen=True)
class LinearizationResult:
    """Container for a numerical state-space linearization result."""

    A: np.ndarray
    B: np.ndarray
    trim_state: LongitudinalState
    trim_elevator_rad: float
    trim_throttle: float
    eigenvalues: np.ndarray
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    state_perturbations: np.ndarray
    input_perturbations: np.ndarray

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "state_space_form": "x_dot_delta = A x_delta + B u_delta",
            "state_names": list(self.state_names),
            "input_names": list(self.input_names),
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "trim_state": {
                "u_m_s": self.trim_state.u_m_s,
                "w_m_s": self.trim_state.w_m_s,
                "q_rad_s": self.trim_state.q_rad_s,
                "theta_rad": self.trim_state.theta_rad,
                "theta_deg": float(np.degrees(self.trim_state.theta_rad)),
                "h_m": self.trim_state.h_m,
                "airspeed_m_s": self.trim_state.airspeed_m_s,
                "alpha_rad": self.trim_state.alpha_rad,
                "alpha_deg": float(np.degrees(self.trim_state.alpha_rad)),
            },
            "trim_input": {
                "elevator_rad": self.trim_elevator_rad,
                "elevator_deg": float(np.degrees(self.trim_elevator_rad)),
                "throttle": self.trim_throttle,
            },
            "eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in self.eigenvalues
            ],
            "modal_summary": self.modal_summary(),
            "state_perturbations": self.state_perturbations.tolist(),
            "input_perturbations": self.input_perturbations.tolist(),
        }

    def modal_summary(self) -> list[dict]:
        """Return natural frequency and damping ratio estimates for each eigenvalue."""
        summary: list[dict] = []

        for value in self.eigenvalues:
            natural_frequency = float(abs(value))

            if natural_frequency < 1.0e-12:
                damping_ratio = None
            else:
                damping_ratio = float(-value.real / natural_frequency)

            summary.append(
                {
                    "real": float(value.real),
                    "imag": float(value.imag),
                    "natural_frequency_rad_s": natural_frequency,
                    "damping_ratio": damping_ratio,
                }
            )

        return summary

    def save_json(self, output_path: str | Path) -> None:
        """Save the linearization result to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)


def _state_derivative_from_vectors(
    aircraft: LongitudinalAircraft,
    x: np.ndarray,
    u: np.ndarray,
    vertical_gust_m_s: float = 0.0,
) -> np.ndarray:
    """Evaluate aircraft dynamics using vector state and vector input."""
    state = LongitudinalState.from_array(x)
    elevator_rad = float(u[0])
    throttle = float(u[1])

    return aircraft.dynamics(
        state=state,
        elevator_rad=elevator_rad,
        throttle=throttle,
        vertical_gust_m_s=vertical_gust_m_s,
    )


def linearize_longitudinal(
    aircraft: LongitudinalAircraft,
    trim_state: LongitudinalState,
    trim_elevator_rad: float,
    trim_throttle: float,
    state_perturbations: np.ndarray | None = None,
    input_perturbations: np.ndarray | None = None,
) -> LinearizationResult:
    """Linearize the longitudinal model around a trim point.

    Parameters
    ----------
    aircraft:
        Nonlinear longitudinal aircraft model.
    trim_state:
        State at the trim condition.
    trim_elevator_rad:
        Elevator command at trim.
    trim_throttle:
        Throttle command at trim.
    state_perturbations:
        Finite-difference perturbation size for each state.
    input_perturbations:
        Finite-difference perturbation size for each input.

    Returns
    -------
    LinearizationResult
        A/B matrices, eigenvalues, trim state/input, and perturbation metadata.
    """
    x0 = trim_state.as_array()
    u0 = np.array([trim_elevator_rad, trim_throttle], dtype=float)

    if state_perturbations is None:
        state_perturbations = np.array(
            [1.0e-3, 1.0e-3, 1.0e-5, 1.0e-5, 1.0e-2],
            dtype=float,
        )
    else:
        state_perturbations = np.asarray(state_perturbations, dtype=float)

    if input_perturbations is None:
        input_perturbations = np.array([1.0e-5, 1.0e-4], dtype=float)
    else:
        input_perturbations = np.asarray(input_perturbations, dtype=float)

    n_states = x0.size
    n_inputs = u0.size

    A = np.zeros((n_states, n_states), dtype=float)
    B = np.zeros((n_states, n_inputs), dtype=float)

    for state_index in range(n_states):
        dx = np.zeros(n_states, dtype=float)
        dx[state_index] = state_perturbations[state_index]

        f_plus = _state_derivative_from_vectors(aircraft, x0 + dx, u0)
        f_minus = _state_derivative_from_vectors(aircraft, x0 - dx, u0)

        A[:, state_index] = (f_plus - f_minus) / (2.0 * dx[state_index])

    for input_index in range(n_inputs):
        du = np.zeros(n_inputs, dtype=float)
        du[input_index] = input_perturbations[input_index]

        f_plus = _state_derivative_from_vectors(aircraft, x0, u0 + du)
        f_minus = _state_derivative_from_vectors(aircraft, x0, u0 - du)

        B[:, input_index] = (f_plus - f_minus) / (2.0 * du[input_index])

    eigenvalues = np.linalg.eigvals(A)

    return LinearizationResult(
        A=A,
        B=B,
        trim_state=trim_state,
        trim_elevator_rad=float(trim_elevator_rad),
        trim_throttle=float(trim_throttle),
        eigenvalues=eigenvalues,
        state_names=("u_m_s", "w_m_s", "q_rad_s", "theta_rad", "h_m"),
        input_names=("elevator_rad", "throttle"),
        state_perturbations=state_perturbations,
        input_perturbations=input_perturbations,
    )