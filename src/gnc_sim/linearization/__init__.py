"""Linearization utilities for GNC models."""

from gnc_sim.linearization.finite_difference import (
    LinearizationResult,
    linearize_longitudinal,
)

__all__ = ["LinearizationResult", "linearize_longitudinal"]