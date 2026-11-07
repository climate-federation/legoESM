"""The state-parity probe's comparison must fail on a changed byte, on a
too-small comparison, and on a state that never moved."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_P = pathlib.Path(__file__).resolve().parents[2] / "scripts/validate/xla_flag_state_parity.py"
_spec = importlib.util.spec_from_file_location("xla_flag_state_parity", _P)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def _write(path, moved=1, **leaves):
    np.savez(path, __moved__=np.array(moved), **leaves)


def test_identical_states_compare_equal(tmp_path):
    x = np.random.default_rng(0).standard_normal(200_000).astype(np.float32)
    _write(tmp_path / "a.npz", T=x); _write(tmp_path / "b.npz", T=x.copy())
    assert probe._compare("lane", tmp_path / "a.npz", tmp_path / "b.npz")


def test_one_ulp_and_signed_zero_are_differences(tmp_path):
    x = np.random.default_rng(0).standard_normal(200_000).astype(np.float32)
    y = x.copy(); y[7] = np.nextafter(y[7], np.float32(np.inf))
    _write(tmp_path / "a.npz", T=x); _write(tmp_path / "b.npz", T=y)
    assert not probe._compare("lane", tmp_path / "a.npz", tmp_path / "b.npz")
    x[3] = 0.0; z = x.copy(); z[3] = -0.0
    _write(tmp_path / "c.npz", T=x); _write(tmp_path / "d.npz", T=z)
    assert not probe._compare("lane", tmp_path / "c.npz", tmp_path / "d.npz")


def test_vacuous_comparisons_are_fatal(tmp_path):
    small = np.ones(10, np.float32)
    _write(tmp_path / "a.npz", T=small); _write(tmp_path / "b.npz", T=small)
    with pytest.raises(SystemExit, match="elements compared"):
        probe._compare("lane", tmp_path / "a.npz", tmp_path / "b.npz")
    big = np.ones(200_000, np.float32)
    _write(tmp_path / "c.npz", moved=0, T=big); _write(tmp_path / "d.npz", moved=0, T=big)
    with pytest.raises(SystemExit, match="did not move"):
        probe._compare("lane", tmp_path / "c.npz", tmp_path / "d.npz")
