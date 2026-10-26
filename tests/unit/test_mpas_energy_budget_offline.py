"""Direct test for ``scripts/validate/mpas_energy_budget_offline.py`` (#1354).

The flux side of the offline budget is a difference of two CMOR monthly-
accumulator sidecars.  Built here from synthetic sidecars in the driver's
manifest layout: a window that crosses a month boundary must sum both month
buckets, a window whose sample count is not one per day must be refused, and
a uniform field must come back as its own value under the cos-lat mean.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "mpas_energy_budget_offline.py"


def _load():
    spec = importlib.util.spec_from_file_location("_offline_budget", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_offline_budget"] = mod
    spec.loader.exec_module(mod)
    return mod


def _sidecar(path, buckets):
    """buckets: {(year, month): (daily_value, n_days)} -> every FLUX_VAR has
    sum = value * n on a (36, 72) grid."""
    mod = sys.modules["_offline_budget"]
    arrs, data, i = {}, [], 0
    for (y, m), (val, n) in buckets.items():
        for var in mod.FLUX_VARS:
            key = f"arr_{i}"
            arrs[f"monthly.{key}"] = np.full((36, 72), val * n)
            data.append([y, m, var, n, key])
            i += 1
    man = {"version": 1, "type": "SpatialMonthlyAccumulator", "data_2d": data}
    np.savez(path, **{"monthly.__manifest__": np.array(json.dumps(man))}, **arrs)


def test_window_spans_month_boundary(tmp_path):
    mod = _load()
    # Days 1..30 of January at 2.0 per day; days 31 (Jan) + 32..35 (Feb) at 5.0.
    _sidecar(tmp_path / "cmor_accum_day_0030.npz", {(0, 1): (2.0, 30)})
    jan = (2.0 * 30 + 5.0) / 31          # the Jan bucket's mean value per sample
    _sidecar(tmp_path / "cmor_accum_day_0035.npz",
             {(0, 1): (jan, 31), (0, 2): (5.0, 4)})
    out = mod.window_flux_means(tmp_path, 30, 35)
    for var in mod.FLUX_VARS:
        assert out[var] == pytest.approx(5.0, rel=1e-12)


def test_window_from_day_zero_uses_empty_accumulator(tmp_path):
    mod = _load()
    _sidecar(tmp_path / "cmor_accum_day_0005.npz", {(0, 1): (3.0, 5)})
    out = mod.window_flux_means(tmp_path, 0, 5)
    assert out["hfss"] == pytest.approx(3.0, rel=1e-12)


def test_window_refuses_wrong_sample_count(tmp_path):
    mod = _load()
    _sidecar(tmp_path / "cmor_accum_day_0010.npz", {(0, 1): (1.0, 10)})
    _sidecar(tmp_path / "cmor_accum_day_0015.npz", {(0, 1): (1.0, 14)})
    with pytest.raises(SystemExit, match="expected 5"):
        mod.window_flux_means(tmp_path, 10, 15)
