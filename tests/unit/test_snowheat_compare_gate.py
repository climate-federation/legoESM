"""The snow-node A/B bit-identity gate comparator must flag any difference."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "levante"
           / "snowheat_1003" / "compare_gate.py")


def _mod():
    spec = importlib.util.spec_from_file_location("compare_gate", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_identical_and_each_kind_of_difference(tmp_path):
    cmp = _mod().checkpoint_differences
    base = {"T": np.array([1.0, np.nan]), "step": np.array(5), "name": np.array("x")}
    np.savez(tmp_path / "a.npz", **base)
    np.savez(tmp_path / "b.npz", **base)
    assert cmp(tmp_path / "a.npz", tmp_path / "b.npz") == []
    np.savez(tmp_path / "c.npz", **{**base, "T": np.array([1.0 + 1e-15, np.nan])})
    assert cmp(tmp_path / "a.npz", tmp_path / "c.npz") == ["T"]
    np.savez(tmp_path / "d.npz", **{**base, "T": base["T"].astype(np.float32)})
    assert cmp(tmp_path / "a.npz", tmp_path / "d.npz") == ["T"]
    np.savez(tmp_path / "z.npz", **{**base, "T": np.array([-0.0, np.nan])})
    np.savez(tmp_path / "z0.npz", **{**base, "T": np.array([0.0, np.nan])})
    assert cmp(tmp_path / "z0.npz", tmp_path / "z.npz") == ["T"]       # signed zero
    np.savez(tmp_path / "e.npz", **{k: v for k, v in base.items() if k != "step"})
    assert cmp(tmp_path / "a.npz", tmp_path / "e.npz") == ["step"]
