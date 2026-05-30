from pathlib import Path

from gnc_sim.config import load_json_config


def test_config_loads():
    root = Path(__file__).resolve().parents[1]
    config = load_json_config(root / "configs" / "uav_longitudinal.json")

    assert "aircraft" in config
    assert "simulation" in config
    assert "controller" in config
