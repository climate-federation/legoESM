"""Direct tests for scripts/validate/amip_bias/window_diff.py.

Sidecars are built with the real ``SpatialMonthlyAccumulator`` so the probe
is exercised against the layout the model actually writes, not a mock.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def wd():
    return _load("window_diff")


def _sidecar(path, days, value_of_day, nlat=4, nlon=8):
    from legoesm.diagnostics.monthly_means import SpatialMonthlyAccumulator
    acc = SpatialMonthlyAccumulator(nlat, nlon)
    for d in days:
        v = value_of_day(d)
        acc.add_2d(d, 1979, {"rsut": np.full((nlat, nlon), v),
                             "rlut": np.full((nlat, nlon), 2 * v),
                             "rlutcs": np.full((nlat, nlon), 3 * v)})
    np.savez(path, **{f"monthly.{k}": a for k, a in acc.get_state().items()})


def test_window_mean_is_the_mean_of_the_new_samples_only(wd, tmp_path, monkeypatch):
    run = tmp_path / "arm"; run.mkdir()
    monkeypatch.setattr(wd.rb, "ROOT", str(tmp_path))
    # March: days 60..80 (all one bucket); the first sidecar holds days 61..80,
    # the second adds 81..85 with a different value
    f = lambda d: 1.0 if d <= 80 else 5.0
    _sidecar(run / "cmor_accum_day_0080.npz", range(61, 81), f)
    _sidecar(run / "cmor_accum_day_0085.npz", range(61, 86), f)
    out, n, dims = wd.window_mean("arm", 80, 85)
    assert n == 5 and dims == (4, 8)
    assert np.allclose(out["rsut"], 5.0)          # not the 25-sample mean 1.8
    assert np.allclose(out["cre_lw"], 15.0 - 10.0)


def test_window_refuses_month_crossing(wd, tmp_path, monkeypatch):
    run = tmp_path / "arm"; run.mkdir()
    monkeypatch.setattr(wd.rb, "ROOT", str(tmp_path))
    _sidecar(run / "cmor_accum_day_0085.npz", range(81, 86), lambda d: 1.0)
    _sidecar(run / "cmor_accum_day_0095.npz", range(81, 96), lambda d: 1.0)   # April opens a 2nd bucket
    with pytest.raises(SystemExit, match="month"):
        wd.window_mean("arm", 85, 95)


def test_cmor_grid_matches_published_axes(wd):
    lat, lon = wd.cmor_grid((36, 72))
    assert lat[0] == -87.5 and lat[-1] == 87.5 and lon[0] == 2.5 and lon[-1] == 357.5
