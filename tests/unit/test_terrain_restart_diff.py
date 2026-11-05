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


class _Field:
    def __init__(self, data):
        self.data = np.asarray(data)


class _State:
    def __init__(self, phis):
        self.phis = _Field(phis)


def test_steps_hook_records_planted_drift_and_stops_at_n():
    """Each step adds 1e-3 to cell 1: rows must show exactly that, and the
    hook must stop after n steps with the last state kept."""
    def real_step(self, state):
        p = state.phis.data.copy()
        p[1] += 1e-3
        return _State(p)

    got = {}
    hook = trd.steps_hook(real_step, lambda: np.zeros(3), 3, got)
    s = _State(np.zeros(3))
    try:
        for _ in range(10):
            s = hook(None, s)
    except trd._Done:
        pass
    rows = got["rows"]
    assert [r[0] for r in rows] == [0, 1, 2]
    assert np.allclose([r[1] for r in rows], 1e-3)
    assert np.allclose([r[2] for r in rows], [1e-3, 2e-3, 3e-3])
    assert np.isclose(got["state"][1], 3e-3)


def test_steps_mode_requires_a_positive_step_count(tmp_path):
    import pytest
    for extra in ([], ["--n-steps", "0"], ["--n-steps", "-1"]):
        with pytest.raises(SystemExit):
            trd.main(["--run", "x", "--day", "1", "--mode", "steps",
                      "--out", str(tmp_path)] + extra)
