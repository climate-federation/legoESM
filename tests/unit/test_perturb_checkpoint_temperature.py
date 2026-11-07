import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load():
    p = Path(__file__).resolve().parents[2] / "scripts" / "experiment" / "perturb_checkpoint_temperature.py"
    spec = importlib.util.spec_from_file_location("_perturb_ckpt", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_perturb_ckpt"] = m
    spec.loader.exec_module(m)
    return m


def test_only_temperature_changes_and_the_seed_reproduces(tmp_path):
    m = _load()
    arrays = {"T": np.full((50, 4), 250.0), "trc_q_v": np.ones((50, 4)), "day": np.array(5.0)}
    src, dst = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(src, **arrays)
    m.main([str(src), str(dst), "--sigma-k", "0.01", "--seed", "7"])
    with np.load(dst) as z:
        assert np.array_equal(z["trc_q_v"], arrays["trc_q_v"]) and z["day"] == 5.0
        d = z["T"] - 250.0
    assert 0.005 < d.std() < 0.02 and np.all(d != 0)
    assert np.array_equal(m.perturb(arrays, 0.01, 7)["T"], m.perturb(arrays, 0.01, 7)["T"])
