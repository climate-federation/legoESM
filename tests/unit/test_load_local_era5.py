"""Portable test for ``scripts/data/load_local_era5.open_local_era5_dataset``.

Writes synthetic NetCDF files matching the NCAR-RDA ll025 naming convention (no real
archive needed) and checks the adapter merges the per-variable pressure-level + monthly
single-level chunks into one ``load_era5_slice``-ready dataset — aligning the monthly
``sfc`` times to the daily ``pl`` times and renaming to the ``resolve_var`` short names.
"""

from __future__ import annotations

import numpy as np
import pytest


def _write_synthetic_rda_era5(tmp_path):
    import xarray as xr

    nlat, nlon, nlev = 4, 5, 3
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.linspace(0.0, 288.0, nlon)
    lev = np.array([500.0, 850.0, 1000.0])
    t_sfc = (np.datetime64("2020-01-01T00")
             + np.arange(8) * np.timedelta64(1, "h")).astype("datetime64[ns]")
    t_pl = t_sfc[:4]                      # the day's 24h (here 4) ⊂ the monthly sfc times

    def _pl(code, var):
        data = np.full((4, nlev, nlat, nlon), 250.0, dtype="f4")
        ds = xr.Dataset(
            {var: (("time", "level", "latitude", "longitude"), data),
             "utc_date": ("time", np.arange(4))},      # metadata var the adapter drops
            coords={"time": t_pl, "level": lev, "latitude": lat, "longitude": lon})
        ds.to_netcdf(tmp_path / f"e5.oper.an.pl.{code}.ll025sc.2020010100_2020010123.nc")

    def _sfc(code, var, val):
        ds = xr.Dataset(
            {var: (("time", "latitude", "longitude"), np.full((8, nlat, nlon), val, "f4"))},
            coords={"time": t_sfc, "latitude": lat, "longitude": lon})
        ds.to_netcdf(tmp_path / f"e5.oper.an.sfc.{code}.ll025sc.2020010100_2020013123.nc")

    for code, var in (("128_130_t", "T"), ("128_131_u", "U"),
                      ("128_132_v", "V"), ("128_133_q", "Q")):
        _pl(code, var)
    _sfc("128_134_sp", "SP", 1.0e5)
    _sfc("128_034_sstk", "SSTK", 290.0)


def test_open_local_era5_dataset_merges_and_aligns(tmp_path):
    from scripts.data.load_local_era5 import open_local_era5_dataset

    _write_synthetic_rda_era5(tmp_path)
    ds = open_local_era5_dataset(str(tmp_path), "20200101")
    # Renamed to resolve_var short names; metadata var dropped.
    assert sorted(ds.data_vars) == ["q", "skt", "sp", "t", "u", "v"]
    # The monthly sfc (8 times) is aligned to the daily pl times (4) — one consistent axis.
    assert ds.sizes["time"] == 4 and ds.sizes["level"] == 3
    assert ds.sizes["latitude"] == 4 and ds.sizes["longitude"] == 5
    assert float(ds.sp.isel(time=0).mean()) == pytest.approx(1.0e5)
    assert float(ds.skt.isel(time=0).mean()) == pytest.approx(290.0)


def test_open_local_era5_dataset_loads_via_load_era5_slice(tmp_path):
    """End-to-end: the adapter output feeds ``load_era5_slice(ds=...)`` into a valid
    ERA5Slice — the same path real NCAR-RDA ERA5 uses (iter 408/409)."""
    from legoesm.training.era5_to_state import TrainingERA5Config, load_era5_slice

    from scripts.data.load_local_era5 import open_local_era5_dataset

    _write_synthetic_rda_era5(tmp_path)
    ds = open_local_era5_dataset(str(tmp_path), "20200101")
    cfg = TrainingERA5Config(
        levels=(1000, 850, 500),
        surface_variables=("surface_pressure", "skin_temperature"))
    sl = load_era5_slice(cfg, time_idx=0, ds=ds)
    assert tuple(sl.T.shape) == (4, 5, 3) and bool(np.all(np.isfinite(sl.T)))
    assert tuple(sl.p_s.shape) == (4, 5)
    assert np.all(np.diff(sl.plev_Pa) > 0)            # ascending Pa


def test_open_local_era5_dataset_missing_chunk_fails_loud(tmp_path):
    from scripts.data.load_local_era5 import open_local_era5_dataset

    _write_synthetic_rda_era5(tmp_path)
    with pytest.raises(FileNotFoundError, match="no ERA5 file matching"):
        open_local_era5_dataset(str(tmp_path), "20200202")   # no chunk for that date


def test_open_local_era5_dataset_ambiguous_chunk_fails_loud(tmp_path):
    """codex-review iter 414: TWO files matching one (kind, code, chunk) — e.g. a partial
    re-download alongside the original — must FAIL LOUD, not silently pick the first
    (a wrong/partial reference).  The ``ll025*`` glob catches both the sc + uv suffixes."""
    import shutil

    from scripts.data.load_local_era5 import open_local_era5_dataset

    _write_synthetic_rda_era5(tmp_path)
    orig = tmp_path / "e5.oper.an.pl.128_130_t.ll025sc.2020010100_2020010123.nc"
    shutil.copy(orig, tmp_path / "e5.oper.an.pl.128_130_t.ll025uv.2020010100_2020010123.nc")
    with pytest.raises(ValueError, match="files match"):
        open_local_era5_dataset(str(tmp_path), "20200101")
