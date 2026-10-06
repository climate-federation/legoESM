"""Direct test of scripts/cluster/levante/layeredft_oct/check_day1.py."""
import importlib.util
import json
import pathlib

import numpy as np

_P = (pathlib.Path(__file__).resolve().parents[2] / "scripts/cluster/levante"
      / "layeredft_oct" / "check_day1.py")
_spec = importlib.util.spec_from_file_location("check_day1", _P)
cd1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd1)

_LOG = "x LAYERED pack is built from its snow water (5 layers) y"


def _run(tmp_path, scheme="layered", ft=True, **over):
    ice = np.full((3, 5), 2.0, np.float32)
    arr = {"land_ml_snow_ice_layers": ice, "land_ml_snow_liq_layers": 0 * ice,
           "land_ml_snow_T_layers": ice + 260, "land_ml_snow_rho_layers": ice + 100,
           "land_ml_snow_depth": ice.sum(-1), "T": np.ones(4)}
    arr.update(over)
    np.savez(tmp_path / "checkpoint_day_0276.npz", **arr)
    np.savez(tmp_path / "checkpoint_day_0275.npz", land_ml_snow_depth=np.full(3, 10.0))
    (tmp_path / "experiment_config.json").write_text(json.dumps(
        {"land_snow_scheme": scheme, "land_soil_freeze_thaw": ft}))
    return tmp_path


def test_a_seeded_finite_layered_day_passes(tmp_path):
    assert cd1.day1_problems(_run(tmp_path), 275, "layered", _LOG) == []


def test_each_failure_is_reported(tmp_path):
    assert cd1.day1_problems(_run(tmp_path, T=np.array([np.nan])), 275, "layered", _LOG)
    assert cd1.day1_problems(_run(tmp_path, scheme="bulk"), 275, "layered", _LOG)
    assert cd1.day1_problems(_run(tmp_path, ft=False), 275, "layered", _LOG)
    assert cd1.day1_problems(_run(tmp_path), 275, "layered", "no warning")
    assert cd1.day1_problems(_run(tmp_path, land_ml_snow_depth=np.zeros(3)), 275, "layered", _LOG)
    assert cd1.day1_problems(_run(tmp_path), 280, "layered", _LOG)        # wrong day
    wiped = _run(tmp_path, land_ml_snow_ice_layers=np.zeros((3, 5), np.float32),
                 land_ml_snow_depth=np.zeros(3))
    assert cd1.day1_problems(wiped, 275, "layered", _LOG) == [
        "snow water 0 after day 1 vs 30 at restart"]
    # a thick pack truncated in one column, the global sum still fine
    np.savez(tmp_path / "checkpoint_day_0275.npz", land_ml_snow_depth=np.array([10.0, 10.0, 500.0]))
    ice = np.full((3, 5), 2.0, np.float32)
    ice[2] = 50.0
    cut = _run(tmp_path, land_ml_snow_ice_layers=ice, land_ml_snow_depth=ice.sum(-1))
    np.savez(tmp_path / "checkpoint_day_0275.npz", land_ml_snow_depth=np.array([10.0, 10.0, 500.0]))
    assert cd1.day1_problems(cut, 275, "layered", _LOG) == [
        "a column lost more than 200 kg/m2 of snow in a day"]
    # a bulk control needs neither layers nor the warning
    assert cd1.day1_problems(_run(tmp_path, scheme="bulk"), 275, "bulk", "") == []
