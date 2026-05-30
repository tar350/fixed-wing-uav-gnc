"""Configuration loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_config(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON configuration file.

    Parameters
    ----------
    config_path:
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
