"""Wind and gust models."""

from __future__ import annotations

import numpy as np


def vertical_gust_m_s(t_s: float, wind_config: dict) -> float:
    """Return a simple vertical gust velocity.

    This is intentionally simple for v0.1. It gives us a disturbance hook before
    implementing Dryden or Von Karman turbulence.
    """
    if not wind_config.get("enabled", False):
        return 0.0

    start = wind_config["vertical_gust_start_s"]
    end = wind_config["vertical_gust_end_s"]

    if t_s < start or t_s > end:
        return 0.0

    amplitude = wind_config["vertical_gust_amplitude_m_s"]
    frequency = wind_config["vertical_gust_frequency_rad_s"]

    return float(amplitude * np.sin(frequency * (t_s - start)))
