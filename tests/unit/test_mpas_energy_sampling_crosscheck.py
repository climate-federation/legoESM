"""Direct test for ``scripts/validate/mpas_energy_sampling_crosscheck.py``."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "mpas_energy_sampling_crosscheck.py"


def _load():
    spec = importlib.util.spec_from_file_location("_xcheck", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_xcheck"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cmor(run_dir, var, value):
    d = run_dir / "cmor" / "Amon"
    d.mkdir(parents=True, exist_ok=True)
    lat = np.array([-60.0, 0.0, 60.0])
    with nc.Dataset(d / f"{var}_Amon_x.nc", "w") as ds:
        ds.createDimension("time", 2)
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", 4)
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable(var, "f8", ("time", "lat", "lon"))[:] = value


def _run_dir(tmp_path, *, cmor, toa_sampled=10.0, nan=False, snapshot=False):
    n = 5
    ts = dict(days=np.arange(n, dtype=float),
              energy_toa_net=np.full(n, toa_sampled),
              hfss=np.full(n, 20.0), hfls=np.full(n, 60.0),
              sw_net_sfc=np.full(n, 150.0), lw_net_sfc=np.full(n, -60.0),
              energy_dE_dt=np.full(n, 5.0),
              energy_flux_interval_mean=np.ones(n))
    if nan:
        ts["hfss"][3] = np.nan
    if snapshot:
        ts["energy_flux_interval_mean"][2] = 0.0
    np.savez(tmp_path / "timeseries.npz", **ts)
    for v, x in cmor.items():
        _cmor(tmp_path, v, x)
    return tmp_path


_CMOR = dict(rsdt=340.0, rsut=100.0, rlut=235.0, hfss=20.0, hfls=60.0)


def test_area_weighted_cmor_mean_is_the_planted_constant(tmp_path):
    mod = _load()
    _run_dir(tmp_path, cmor=_CMOR)
    assert mod.gmean(tmp_path, "rsdt") == pytest.approx(340.0)


def test_substituting_cmor_toa_shifts_the_leak_by_the_toa_gap(tmp_path, capsys):
    """LEAK(CMOR toa) - LEAK(sampled toa) = sampled_toa - CMOR_toa, exactly."""
    mod = _load()
    _run_dir(tmp_path, cmor=_CMOR, toa_sampled=10.0)   # CMOR toa = 5
    assert mod.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    got = {}
    for line in out.splitlines():
        if "LEAK with sampled" in line:
            got["s"] = float(line.split(":")[1])
        if "LEAK with CMOR" in line:
            got["c"] = float(line.split(":")[1])
    assert got["c"] - got["s"] == pytest.approx(10.0 - 5.0)


def test_missing_cmor_variable_is_fatal(tmp_path):
    mod = _load()
    _run_dir(tmp_path, cmor={k: v for k, v in _CMOR.items() if k != "hfls"})
    with pytest.raises(SystemExit, match="hfls"):
        mod.main([str(tmp_path)])


def test_non_finite_sample_is_fatal_not_averaged_over(tmp_path):
    mod = _load()
    _run_dir(tmp_path, cmor=_CMOR, nan=True)
    with pytest.raises(SystemExit, match="non-finite"):
        mod.main([str(tmp_path)])


def test_snapshot_samples_are_labelled_in_the_provenance_line(tmp_path, capsys):
    mod = _load()
    _run_dir(tmp_path, cmor=_CMOR, snapshot=True)
    assert mod.main([str(tmp_path)]) == 0
    assert "1 of 3 samples are SNAPSHOTS" in capsys.readouterr().out


def test_missing_sampled_series_is_fatal_by_name(tmp_path):
    mod = _load()
    _run_dir(tmp_path, cmor=_CMOR)
    d = dict(np.load(tmp_path / "timeseries.npz"))
    del d["energy_dE_dt"]
    np.savez(tmp_path / "timeseries.npz", **d)
    with pytest.raises(SystemExit, match="energy_dE_dt"):
        mod.main([str(tmp_path)])


def test_empty_window_is_fatal_not_nan(tmp_path):
    mod = _load()
    _run_dir(tmp_path, cmor=_CMOR)
    d = dict(np.load(tmp_path / "timeseries.npz"))
    for key in d:
        d[key] = d[key][:2]                      # days 0 and 1 only
    np.savez(tmp_path / "timeseries.npz", **d)
    with pytest.raises(SystemExit, match="no samples"):
        mod.main([str(tmp_path)])
