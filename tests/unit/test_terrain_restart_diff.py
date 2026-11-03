"""terrain_restart_diff.terrain_diff (#2303)."""
import importlib.util
from pathlib import Path

import numpy as np

_P = Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias/terrain_restart_diff.py"
_spec = importlib.util.spec_from_file_location("terrain_restart_diff", _P)
trd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trd)


def test_terrain_diff_reports_max_location_and_exact_count():
    r = trd.terrain_diff([1.0, 2.0, 3.0, 4.0], [1.0, 2.5, 3.0, 3.9])
    assert r["argmax"] == 1 and np.isclose(r["max"], 0.5) and r["n_diff"] == 2
    assert trd.terrain_diff([1.0], [1.0]) == {"max": 0.0, "argmax": 0, "n_diff": 0}
