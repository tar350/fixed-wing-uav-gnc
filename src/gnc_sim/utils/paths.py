"""Path utilities."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return project root based on this file location."""
    return Path(__file__).resolve().parents[3]


def ensure_directory(path: str | Path) -> Path:
    """Create directory if it does not exist and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
