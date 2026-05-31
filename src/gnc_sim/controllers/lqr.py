"""Linear Quadratic Regulator design utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from scipy.linalg import solve_continuous_are


@dataclass(frozen=True)
class LQRDesignResult:
    """Container for continuous-time LQR design result."""

    K: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    closed_loop_A: np.ndarray
    open_loop_eigenvalues: np.ndarray
    closed_loop_eigenvalues: np.ndarray
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "control_law": "u_delta = -K x_delta",
            "state_names": list(self.state_names),
            "input_names": list(self.input_names),
            "K": self.K.tolist(),
            "Q": self.Q.tolist(),
            "R": self.R.tolist(),
            "closed_loop_A": self.closed_loop_A.tolist(),
            "open_loop_eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in self.open_loop_eigenvalues
            ],
            "closed_loop_eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in self.closed_loop_eigenvalues
            ],
        }

    def save_json(self, output_path: str | Path) -> None:
        """Save the LQR design result to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)


def design_lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    state_names: tuple[str, ...],
    input_names: tuple[str, ...],
) -> LQRDesignResult:
    """Design a continuous-time LQR controller.

    The continuous-time LQR law is:

        u_delta = -K x_delta

    where x_delta is the perturbation from trim and u_delta is the control
    perturbation from trim.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    if B.shape[0] != A.shape[0]:
        raise ValueError("B row count must match A state dimension.")

    if Q.shape != A.shape:
        raise ValueError("Q must have the same shape as A.")

    if R.shape != (B.shape[1], B.shape[1]):
        raise ValueError("R must match the number of control inputs.")

    P = solve_continuous_are(A, B, Q, R)

    K = np.linalg.solve(R, B.T @ P)

    closed_loop_A = A - B @ K

    return LQRDesignResult(
        K=K,
        Q=Q,
        R=R,
        closed_loop_A=closed_loop_A,
        open_loop_eigenvalues=np.linalg.eigvals(A),
        closed_loop_eigenvalues=np.linalg.eigvals(closed_loop_A),
        state_names=state_names,
        input_names=input_names,
    )