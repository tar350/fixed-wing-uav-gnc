"""Run longitudinal simulation and create a 2D aircraft animation.

Execute from project root:

    python scripts/run_longitudinal_animation_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gnc_sim.config import load_json_config
from gnc_sim.simulation.integrator import simulate_longitudinal
from gnc_sim.visualization import animate_longitudinal_response


def main() -> None:
    """Run simulation and save animation."""
    config = load_json_config(PROJECT_ROOT / "configs" / "uav_longitudinal.json")

    results = simulate_longitudinal(config)

    output_path = PROJECT_ROOT / "outputs" / "animations" / "longitudinal_motion_v0_5.gif"

    saved_path = animate_longitudinal_response(
        results=results,
        output_path=output_path,
        frame_step=5,
        interval_ms=40,
        aircraft_length_m=8.0,
    )

    print("Longitudinal animation complete")
    print("=" * 40)
    print(f"Saved animation: {saved_path}")


if __name__ == "__main__":
    main()