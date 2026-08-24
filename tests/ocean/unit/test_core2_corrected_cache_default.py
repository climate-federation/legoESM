"""The CORE-II NYF default must resolve to the bias-corrected cache.

NEMO ORCA1's namelist reads the Large & Yeager ``_MOD`` fields.  Forcing with
the raw CORE-II winds instead gave ~33% weak equatorial trades and a +2.5 C
nino3 SST bias that fell to +1.49 C at matched day 20 once the corrected
fields were used, so which cache the default picks is a correctness question,
not a convenience one.  These tests pin the choice and the provenance number
that makes a silent fallback visible in a run log.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
from legoesm.ocean.forcing import core2


def _fake_root(tmp_path, monkeypatch):
    """Point core2's cache resolution at a throwaway directory tree."""
    root = tmp_path / "forcing"
    (root / "jra55_do").mkdir(parents=True)
    monkeypatch.setattr(core2, "_jra_cache_dir", lambda: root / "jra55_do")
    return root


def test_default_cache_prefers_the_corrected_fields(tmp_path, monkeypatch):
    root = _fake_root(tmp_path, monkeypatch)
    (root / "core2_nyf").mkdir()
    (root / "core2_nyf_mod").mkdir()
    assert core2.core2_nyf_cache_dir() == root / "core2_nyf_mod"


def test_missing_corrected_cache_falls_back_loudly(tmp_path, monkeypatch, caplog):
    root = _fake_root(tmp_path, monkeypatch)
    (root / "core2_nyf").mkdir()
    with caplog.at_level("WARNING"):
        resolved = core2.core2_nyf_cache_dir()
    assert resolved == root / "core2_nyf"
    # A silent fallback to the raw winds is the defect the default exists to
    # prevent, so the warning is part of the contract.
    assert any("RAW-wind" in r.getMessage() for r in caplog.records)


def test_nino3_wind_speed_is_the_area_weighted_box_mean():
    """Known answer: a uniform 7 m/s zonal wind must score exactly 7 m/s."""
    lat = np.linspace(-88.0, 88.0, 45)
    lon = np.linspace(0.0, 358.0, 180)
    shape = (4, lat.size, lon.size)
    ds = types.SimpleNamespace(
        lat=types.SimpleNamespace(values=lat),
        lon=types.SimpleNamespace(values=lon),
        u10=types.SimpleNamespace(values=np.full(shape, 7.0)),
        v10=types.SimpleNamespace(values=np.zeros(shape)),
    )
    assert core2._nino3_wind_speed(ds) == pytest.approx(7.0, rel=1e-12)


def test_nino3_wind_speed_reads_only_the_nino3_box():
    """A wind that is strong only outside 5S-5N/150W-90W must not register."""
    lat = np.linspace(-88.0, 88.0, 45)
    lon = np.linspace(0.0, 358.0, 180)
    shape = (2, lat.size, lon.size)
    u = np.full(shape, 100.0)
    jj = np.where((lat >= -5.0) & (lat <= 5.0))[0]
    ii = np.where(((lon % 360.0) >= 210.0) & ((lon % 360.0) <= 270.0))[0]
    u[np.ix_(np.arange(shape[0]), jj, ii)] = 3.0
    ds = types.SimpleNamespace(
        lat=types.SimpleNamespace(values=lat),
        lon=types.SimpleNamespace(values=lon),
        u10=types.SimpleNamespace(values=u),
        v10=types.SimpleNamespace(values=np.zeros(shape)),
    )
    assert core2._nino3_wind_speed(ds) == pytest.approx(3.0, rel=1e-12)


def test_load_prints_the_resolved_cache_and_its_wind_speed(tmp_path, capsys):
    """The provenance line must survive: a run log that does not name the
    cache is a run whose forcing cannot be established afterwards.

    Written as an end-to-end load rather than a call to the helper, because
    the failure this guards against is the print being dropped, not the
    arithmetic being wrong.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")

    nt, ny, nx = 3, 45, 180
    lat = np.linspace(-88.0, 88.0, ny)
    lon = np.linspace(0.0, 358.0, nx)
    dims = ("time", "lat", "lon")
    ds = xr.Dataset(
        {name: (dims, np.full((nt, ny, nx), val))
         for name, val in (("u10", 7.0), ("v10", 0.0), ("T_air", 280.0),
                           ("q_air", 0.005), ("sw_down", 200.0),
                           ("lw_down", 300.0), ("precip", 0.0),
                           ("runoff", 0.0))},
        coords={"lat": lat, "lon": lon,
                "time_s": ("time", np.arange(nt, dtype=float) * 86400.0)},
    )
    cache = tmp_path / "core2_nyf_mod"
    ds.to_zarr(cache / "nyf.zarr")

    core2.load_core2_nyf(cache_dir=cache, allow_synthetic=False)
    out = capsys.readouterr().out
    assert "[forcing] CORE-II NYF cache:" in out
    assert str(cache / "nyf.zarr") in out
    assert "7.000 m/s" in out
