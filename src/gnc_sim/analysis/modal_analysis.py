"""Modal and stability analysis for linearized longitudinal aircraft dynamics.

This module takes eigenvalues from a linearized state-space model and converts
them into engineering quantities:

- stability classification
- natural frequency
- damping ratio
- oscillation period
- exponential time constant
- approximate aircraft mode classification

The current aircraft model uses the longitudinal state vector:

    x = [u, w, q, theta, h]

Because altitude h is included as a state, one near-zero eigenvalue is expected.
This is not necessarily a defect. It usually represents the altitude/position
integrator in the open-loop longitudinal model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

import numpy as np


@dataclass(frozen=True)
class ModeSummary:
    """Engineering summary of one eigenvalue/mode."""

    index: int
    real: float
    imag: float
    natural_frequency_rad_s: float
    damping_ratio: float | None
    period_s: float | None
    time_constant_s: float | None
    stability: str
    mode_type: str

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "index": self.index,
            "real": self.real,
            "imag": self.imag,
            "natural_frequency_rad_s": self.natural_frequency_rad_s,
            "damping_ratio": self.damping_ratio,
            "period_s": self.period_s,
            "time_constant_s": self.time_constant_s,
            "stability": self.stability,
            "mode_type": self.mode_type,
        }


@dataclass(frozen=True)
class ModalAnalysisResult:
    """Container for modal analysis results."""

    modes: list[ModeSummary]
    num_modes: int
    num_unstable_modes: int
    num_stable_modes: int
    num_neutral_modes: int
    has_near_integrator: bool
    is_open_loop_stable: bool
    is_asymptotically_stable: bool
    notes: list[str]

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "num_modes": self.num_modes,
            "num_unstable_modes": self.num_unstable_modes,
            "num_stable_modes": self.num_stable_modes,
            "num_neutral_modes": self.num_neutral_modes,
            "has_near_integrator": self.has_near_integrator,
            "is_open_loop_stable": self.is_open_loop_stable,
            "is_asymptotically_stable": self.is_asymptotically_stable,
            "notes": self.notes,
            "modes": [mode.to_dict() for mode in self.modes],
        }

    def save_json(self, output_path: str | Path) -> None:
        """Save modal analysis result to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)

    def save_csv(self, output_path: str | Path) -> None:
        """Save modal analysis result to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "index",
            "real",
            "imag",
            "natural_frequency_rad_s",
            "damping_ratio",
            "period_s",
            "time_constant_s",
            "stability",
            "mode_type",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for mode in self.modes:
                writer.writerow(mode.to_dict())


def _classify_stability(real_part: float, tolerance: float) -> str:
    """Classify stability using the real part of an eigenvalue."""
    if real_part > tolerance:
        return "unstable"
    if real_part < -tolerance:
        return "stable"
    return "neutral"


def _classify_mode_type(
    real_part: float,
    imag_part: float,
    natural_frequency_rad_s: float,
    stability: str,
    zero_tolerance: float,
    short_period_threshold_rad_s: float,
) -> str:
    """Assign an approximate aircraft-mode label to an eigenvalue."""
    if natural_frequency_rad_s <= zero_tolerance:
        return "near_integrator"

    if abs(imag_part) <= zero_tolerance:
        if stability == "stable":
            return "real_subsidence"
        if stability == "unstable":
            return "real_divergence"
        return "neutral_real_mode"

    if natural_frequency_rad_s >= short_period_threshold_rad_s:
        return "oscillatory_short_period_candidate"

    return "oscillatory_phugoid_candidate"


def _analyze_single_mode(
    index: int,
    eigenvalue: complex,
    zero_tolerance: float,
    short_period_threshold_rad_s: float,
) -> ModeSummary:
    """Compute engineering quantities for one eigenvalue."""
    real_part = float(eigenvalue.real)
    imag_part = float(eigenvalue.imag)

    natural_frequency_rad_s = float(abs(eigenvalue))

    if natural_frequency_rad_s <= zero_tolerance:
        damping_ratio = None
    else:
        damping_ratio = float(-real_part / natural_frequency_rad_s)

    if abs(imag_part) <= zero_tolerance:
        period_s = None
    else:
        period_s = float(2.0 * math.pi / abs(imag_part))

    if abs(real_part) <= zero_tolerance:
        time_constant_s = None
    else:
        time_constant_s = float(1.0 / abs(real_part))

    stability = _classify_stability(real_part, zero_tolerance)

    mode_type = _classify_mode_type(
        real_part=real_part,
        imag_part=imag_part,
        natural_frequency_rad_s=natural_frequency_rad_s,
        stability=stability,
        zero_tolerance=zero_tolerance,
        short_period_threshold_rad_s=short_period_threshold_rad_s,
    )

    return ModeSummary(
        index=index,
        real=real_part,
        imag=imag_part,
        natural_frequency_rad_s=natural_frequency_rad_s,
        damping_ratio=damping_ratio,
        period_s=period_s,
        time_constant_s=time_constant_s,
        stability=stability,
        mode_type=mode_type,
    )


def analyze_modes(
    eigenvalues: np.ndarray,
    zero_tolerance: float = 1.0e-7,
    short_period_threshold_rad_s: float = 1.0,
) -> ModalAnalysisResult:
    """Analyze and classify eigenvalues from a longitudinal A matrix.

    Parameters
    ----------
    eigenvalues:
        Eigenvalues of the linearized A matrix.
    zero_tolerance:
        Numerical tolerance used to classify neutral/near-zero modes.
    short_period_threshold_rad_s:
        Approximate natural-frequency boundary used to distinguish a
        short-period candidate from a phugoid candidate.

    Returns
    -------
    ModalAnalysisResult
        Stability and modal summary.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=complex)

    modes = [
        _analyze_single_mode(
            index=idx,
            eigenvalue=eig,
            zero_tolerance=zero_tolerance,
            short_period_threshold_rad_s=short_period_threshold_rad_s,
        )
        for idx, eig in enumerate(eigenvalues, start=1)
    ]

    num_unstable = sum(mode.stability == "unstable" for mode in modes)
    num_stable = sum(mode.stability == "stable" for mode in modes)
    num_neutral = sum(mode.stability == "neutral" for mode in modes)
    has_near_integrator = any(mode.mode_type == "near_integrator" for mode in modes)

    is_open_loop_stable = num_unstable == 0
    is_asymptotically_stable = num_unstable == 0 and num_neutral == 0

    notes: list[str] = []

    if has_near_integrator:
        notes.append(
            "A near-zero eigenvalue is expected because altitude is included as a state."
        )

    if is_open_loop_stable and not is_asymptotically_stable:
        notes.append(
            "The open-loop model is stable in the sense of no positive-real eigenvalues, "
            "but not asymptotically stable because at least one neutral mode exists."
        )

    if num_unstable > 0:
        notes.append(
            "At least one eigenvalue has positive real part; the open-loop model is unstable."
        )

    return ModalAnalysisResult(
        modes=modes,
        num_modes=len(modes),
        num_unstable_modes=num_unstable,
        num_stable_modes=num_stable,
        num_neutral_modes=num_neutral,
        has_near_integrator=has_near_integrator,
        is_open_loop_stable=is_open_loop_stable,
        is_asymptotically_stable=is_asymptotically_stable,
        notes=notes,
    )