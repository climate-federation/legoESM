"""#2525: known-answer checks for three ocean-fidelity probes fixed together.

* drake_thermal_wind_vs_nemo: thermal-wind transport is relative to the
  BOTTOM, so the inner integral runs bottom-up.
* equatorial_gridscale_variance: the one-sided spectrum needs Parseval
  weights; an equal-variance 2-dx / 20-dx mix is exactly half grid-scale.
* tracer_variance_budget_nemo: the hourly state average spans only the hours
  of the daily record it is paired with.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parents[2] / "scripts/validate/ocean_fidelity"
sys.path.insert(0, str(_DIR))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_drake_double_integral_runs_from_the_bottom():
    """Density contrast only in the bottom layer of a 3-layer column: the
    shear lives there, so every layer above moves with it relative to the
    bottom and all three carry transport (3 d h^2, not d h^2)."""
    m = _load("drake_thermal_wind_vs_nemo")
    d, h = 0.2, 10.0
    D = m._double_integral(np.array([0.0, 0.0, d]), np.full(3, h),
                           np.ones(3, bool))
    assert D == pytest.approx(3 * d * h * h, rel=1e-12)


def test_gridscale_share_obeys_parseval():
    m = _load("equatorial_gridscale_variance")
    x = np.arange(120)
    mix = (-1.0) ** x + np.sqrt(2.0) * np.cos(2 * np.pi * x / 20.0)
    assert m.high_band_share(mix) == pytest.approx(0.5, abs=1e-9)
    mix_odd = np.r_[mix, mix[0]]
    assert 0.45 < m.high_band_share(mix_odd) < 0.55
    m.selftest()


class _Var:
    def __init__(self, data, **attrs):
        self._d = np.asarray(data)
        self.__dict__.update(attrs)

    def __getitem__(self, i):
        return self._d[i]


class _DS:
    def __init__(self, **v):
        self.variables = v


def test_hourly_records_are_restricted_to_the_daily_window():
    m = _load("tracer_variance_budget_nemo")
    day = 86400.0
    ds = _DS(time_counter=_Var([0.5 * day, 1.5 * day], bounds="tb"),
             tb=_Var([[0.0, day], [day, 2 * day]]))
    assert m.daily_record_bounds(ds, 0) == (0.0, day)
    hourly = np.arange(1, 49) * 3600.0 - 1800.0       # 48 hourly means
    assert m.records_in_window(hourly, (0.0, day)) == list(range(24))


def test_missing_bounds_or_empty_window_is_fatal():
    m = _load("tracer_variance_budget_nemo")
    with pytest.raises(SystemExit, match="no bounds"):
        m.daily_record_bounds(_DS(time_counter=_Var([0.0])), 0)
    with pytest.raises(SystemExit, match="no hourly record"):
        m.records_in_window(np.array([5.0e5]), (0.0, 86400.0))
