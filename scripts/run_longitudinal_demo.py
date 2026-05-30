"""Run the v0.1 longitudinal autopilot demo.

Usage from project root:

    python scripts/run_longitudinal_demo.py
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
from gnc_sim.utils.paths import ensure_directory, project_root
from gnc_sim.utils.plotting import plot_longitudinal_response


def main() -> None:
    root = project_root()
    config_path = root / "configs" / "uav_longitudinal.json"

    config = load_json_config(config_path)
    results = simulate_longitudinal(config)

    output_log_dir = ensure_directory(root / "outputs" / "logs")
    output_plot_dir = ensure_directory(root / "outputs" / "plots")

    csv_path = output_log_dir / "longitudinal_demo_results.csv"
    results.to_csv(csv_path, index=False)

    plot_longitudinal_response(
        results=results,
        save_path=output_plot_dir / "longitudinal_response.png",
    )

    final = results.iloc[-1]
    print("Simulation complete.")
    print(f"Saved results to: {csv_path}")
    print(f"Saved plots to:   {output_plot_dir}")
    print("")
    print("Final state:")
    print(f"  Altitude: {final['altitude_m']:.2f} m")
    print(f"  Airspeed: {final['airspeed_m_s']:.2f} m/s")
    print(f"  Pitch:    {final['theta_deg']:.2f} deg")


if __name__ == "__main__":
    main()
