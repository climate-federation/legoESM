"""Unit coverage for les_record.TimeMeanAccumulator — the shared time-mean used by
the LES reference drivers so saved targets are time-means over the quasi-steady
window (GCSS protocol), not noisy final-timestep snapshots.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_LES_RECORD = (Path(__file__).resolve().parents[2]
               / "scripts" / "run" / "les_record.py")


def _load():
    spec = importlib.util.spec_from_file_location("les_record", _LES_RECORD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_elementwise_time_mean_of_arrays_and_scalars():
    acc = _load().TimeMeanAccumulator()
    acc.add({"qc": np.array([0.0, 2.0]), "cloud_cover": 0.10, "lwp": 4.0})
    acc.add({"qc": np.array([2.0, 4.0]), "cloud_cover": 0.20, "lwp": 8.0})
    m = acc.mean()
    assert acc.n == 2
    np.testing.assert_allclose(m["qc"], [1.0, 3.0])
    assert m["cloud_cover"] == pytest.approx(0.15)
    assert m["lwp"] == pytest.approx(6.0)


def test_non_numeric_leaves_ignored():
    acc = _load().TimeMeanAccumulator()
    acc.add({"cloud_cover": 0.1, "case": "bomex", "missing": None})
    m = acc.mean()
    assert set(m) == {"cloud_cover"}            # str/None dropped


def test_key_drift_raises():
    acc = _load().TimeMeanAccumulator()
    acc.add({"cloud_cover": 0.1, "lwp": 5.0})
    with pytest.raises(ValueError, match="drifted"):
        acc.add({"cloud_cover": 0.2})           # lwp missing on 2nd sample


def test_empty_window_raises():
    acc = _load().TimeMeanAccumulator()
    with pytest.raises(ValueError, match="empty"):
        acc.mean()


def test_bool_not_treated_as_numeric():
    acc = _load().TimeMeanAccumulator()
    acc.add({"cloud_cover": 0.1, "flag": True})
    assert set(acc.mean()) == {"cloud_cover"}   # bool excluded (it's an int subclass)
