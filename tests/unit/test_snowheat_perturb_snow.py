"""The snow-layer perturbation tool must touch only snow-covered T_snow."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "levante"
           / "snowheat_oct" / "perturb_snow.py")


def _mod():
    spec = importlib.util.spec_from_file_location("perturb_snow", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_perturbs_only_snow_columns_of_T_snow(tmp_path):
    m = _mod()
    src = {"land_ml_T_snow": np.array([260.0, 270.0, 250.0], np.float32),
           "land_ml_snow_depth": np.array([5.0, 0.0, 1.0]),
           "T": np.array([1.0, 2.0])}
    np.savez(tmp_path / "a.npz", **src)
    assert m.perturb(tmp_path / "a.npz", tmp_path / "b.npz", -2.0) == 2
    b = np.load(tmp_path / "b.npz")
    np.testing.assert_array_equal(b["land_ml_T_snow"], np.array([258.0, 270.0, 248.0], np.float32))
    assert b["land_ml_T_snow"].dtype == np.float32
    for k in ("land_ml_snow_depth", "T"):
        assert b[k].tobytes() == src[k].tobytes()


def test_refusals(tmp_path):
    m = _mod()
    np.savez(tmp_path / "off.npz", land_ml_snow_depth=np.array([1.0]), T=np.array([1.0]))
    with pytest.raises(SystemExit):
        m.perturb(tmp_path / "off.npz", tmp_path / "x.npz", -2.0)
    np.savez(tmp_path / "dry.npz", land_ml_T_snow=np.array([260.0]), land_ml_snow_depth=np.array([0.0]))
    with pytest.raises(SystemExit):
        m.perturb(tmp_path / "dry.npz", tmp_path / "y.npz", -2.0)
    np.savez(tmp_path / "ok.npz", land_ml_T_snow=np.array([260.0]), land_ml_snow_depth=np.array([1.0]))
    with pytest.raises(SystemExit):
        m.perturb(tmp_path / "ok.npz", tmp_path / "z.npz", 0.0)
    (tmp_path / "taken.npz").touch()
    with pytest.raises(SystemExit):
        m.perturb(tmp_path / "ok.npz", tmp_path / "taken.npz", -2.0)


def test_verify_requires_finite_colder_snow(tmp_path):
    m = _mod()
    swe = np.array([5.0, 0.0, 1.0])

    def save(name, T):
        np.savez(tmp_path / name, land_ml_T_snow=np.array(T), land_ml_snow_depth=swe)
        return tmp_path / name
    ctrl = save("ctrl.npz", [260.0, 270.0, 250.0])
    mean, why = m.colder_after_restart(save("cold.npz", [259.0, 270.0, 249.5]), ctrl)
    assert why is None and mean == -0.75
    mean, _ = m.colder_after_restart(save("warm.npz", [261.0, 270.0, 250.0]), ctrl)
    assert mean > 0
    assert m.colder_after_restart(save("same.npz", [260.0, 275.0, 250.0]), ctrl)[1]
    assert m.colder_after_restart(save("nan.npz", [np.nan, 270.0, 250.0]), ctrl)[1]
    np.savez(tmp_path / "off.npz", land_ml_snow_depth=swe)
    assert m.colder_after_restart(tmp_path / "off.npz", ctrl)[1]
