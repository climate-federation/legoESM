"""Portable test for ``scripts/data/load_local_era5.open_local_era5_dataset``.

Writes synthetic NetCDF files matching the NCAR-RDA ll025 naming convention (no real
archive needed) and checks the adapter merges the per-variable pressure-level + monthly
single-level chunks into one ``load_era5_slice``-ready dataset — aligning the monthly
``sfc`` times to the daily ``pl`` times and renaming to the ``resolve_var`` short names.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._offline_era5_rda import pl_name, sfc_name


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
        ds.to_netcdf(tmp_path / pl_name(code, "20200101"))

    def _sfc(code, var, val):
        ds = xr.Dataset(
            {var: (("time", "latitude", "longitude"), np.full((8, nlat, nlon), val, "f4"))},
            coords={"time": t_sfc, "latitude": lat, "longitude": lon})
        ds.to_netcdf(tmp_path / sfc_name(code, "20200101"))

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


def _write_synthetic_rda_seaice(tmp_path, *, val=0.0):
    """Add the monthly sea-ice (ci) chunk the AMIP forcing needs (the compare-side helper
    only writes sstk/sp); CI is a [0,1] fraction with units '(0-1)' like the real archive."""
    import xarray as xr

    nlat, nlon = 4, 5
    lat = np.linspace(90.0, -90.0, nlat)         # ERA5 order: descending
    lon = np.linspace(0.0, 288.0, nlon)
    t = (np.datetime64("2020-01-01T00")
         + np.arange(8) * np.timedelta64(1, "h")).astype("datetime64[ns]")
    ds = xr.Dataset(
        {"CI": (("time", "latitude", "longitude"),
                np.full((8, nlat, nlon), val, "f4"), {"units": "(0-1)"})},
        coords={"time": t, "latitude": lat, "longitude": lon})
    ds.to_netcdf(tmp_path / sfc_name("128_031_ci", "20200101"))


def test_build_era5_amip_forcing_feeds_load_amip_forcing(tmp_path):
    """The local ERA5 SST (sstk, K) + sea-ice (ci, [0,1]) build a memory-safe subsampled
    custom AMIP forcing — the FORCING-side twin of open_local_era5_dataset that completes
    the fully-offline realistic AMIP boundary-condition path (iter 419).  Exercises the
    hour-stride subsample, the combined-file write, and descending ERA5 latitude end-to-end
    through load_amip_forcing's regrid + units guard."""
    import xarray as xr
    from legoesm.forcing.amip import load_amip_forcing
    from legoesm.grids.latlon import create_latlon_grid

    from legoesm import constants
    from scripts.data.load_local_era5 import build_era5_amip_forcing

    _write_synthetic_rda_era5(tmp_path)             # writes sstk=290 K (+ sp, pl)
    _write_synthetic_rda_seaice(tmp_path, val=0.0)
    out = tmp_path / "amip_forcing.nc"

    cfg = build_era5_amip_forcing(str(tmp_path), "20200101", str(out), hour_stride=4)
    assert out.exists()
    assert cfg.dataset == "custom"
    assert cfg.sst_var == "SSTK" and cfg.sic_var == "CI"
    assert cfg.sst_offset == 0.0 and cfg.sic_scale == 1.0   # K / fraction: no conversion
    assert cfg.path == str(out)

    with xr.open_dataset(out) as built:              # close before any re-write below
        assert sorted(built.data_vars) == ["CI", "SSTK"]
        assert built.sizes["time"] == 2              # 8 synthetic hourly steps / stride 4

    forcing = load_amip_forcing(cfg, create_latlon_grid(n_lat=8, n_lon=16))
    sst = np.asarray(forcing.sst)
    sic = np.asarray(forcing.sic)
    assert sst.shape[0] == 2                          # the subsampled boundary times
    assert np.all(sst >= constants.T_freeze_ocean)    # K, clamped to seawater freezing
    assert float(sst.mean()) == pytest.approx(290.0, abs=1.0)
    assert np.all((sic >= 0.0) & (sic <= 1.0))

    # A pass-through override reaches the config (e.g. a custom sea-ice surface temperature).
    cfg2 = build_era5_amip_forcing(
        str(tmp_path), "20200101", str(tmp_path / "amip_forcing2.nc"),
        hour_stride=4, T_ice=270.0)
    assert cfg2.T_ice == 270.0
    # hour_stride must be >= 1 (a non-positive stride is a programming error, not a no-op).
    with pytest.raises(ValueError, match="hour_stride"):
        build_era5_amip_forcing(
            str(tmp_path), "20200101", str(tmp_path / "x.nc"), hour_stride=0)
    # A stride that leaves a single step is rejected: the time-interpolated forcing needs
    # >= 2 (a 1-step file would index out of bounds downstream) (codex-review iter 419).
    with pytest.raises(ValueError, match=">= 2"):
        build_era5_amip_forcing(
            str(tmp_path), "20200101", str(tmp_path / "y.nc"), hour_stride=8)  # 8/8 -> 1


def _write_month_sst_sic(tmp_path, day, *, sst_val, sic_val=0.0):
    """Write a month's synthetic SSTK + CI chunk (8 hourly steps) for the d633006 ll025
    layout, keyed to ``day``'s month — for the multi-month concatenation test (iter 459)."""
    import xarray as xr

    nlat, nlon = 4, 5
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.linspace(0.0, 288.0, nlon)
    t = (np.datetime64(f"{day[:4]}-{day[4:6]}-01T00")
         + np.arange(8) * np.timedelta64(1, "h")).astype("datetime64[ns]")
    xr.Dataset(
        {"SSTK": (("time", "latitude", "longitude"), np.full((8, nlat, nlon), sst_val, "f4"))},
        coords={"time": t, "latitude": lat, "longitude": lon},
    ).to_netcdf(tmp_path / sfc_name("128_034_sstk", day))
    xr.Dataset(
        {"CI": (("time", "latitude", "longitude"),
                np.full((8, nlat, nlon), sic_val, "f4"), {"units": "(0-1)"})},
        coords={"time": t, "latitude": lat, "longitude": lon},
    ).to_netcdf(tmp_path / sfc_name("128_031_ci", day))


def test_build_era5_amip_forcing_multi_month_concatenates(tmp_path):
    """n_months>1 CONCATENATES consecutive monthly chunks along time (iter 459), so a
    multi-month climatology window stays within the forcing coverage (no iter-458 cyclic
    repeat). The concat preserves calendar order + each month's SST; a missing later month
    fails loud BEFORE any write; n_months<1 is rejected."""
    import xarray as xr

    from scripts.data.load_local_era5 import build_era5_amip_forcing

    _write_month_sst_sic(tmp_path, "20200101", sst_val=290.0)
    _write_month_sst_sic(tmp_path, "20200201", sst_val=292.0)
    out = tmp_path / "amip_multi.nc"

    cfg = build_era5_amip_forcing(str(tmp_path), "20200101", str(out),
                                  hour_stride=4, n_months=2)
    assert out.exists() and cfg.dataset == "custom"
    with xr.open_dataset(out) as built:
        assert built.sizes["time"] == 4              # 2 months x (8 / stride 4)
        sst = np.asarray(built["SSTK"])              # (time, lat, lon)
        assert float(sst[:2].mean()) == pytest.approx(290.0)   # month 1 first (concat order)
        assert float(sst[2:].mean()) == pytest.approx(292.0)   # month 2 last
        t = built["time"].values.astype("datetime64[ns]").astype("int64")
        assert np.all(np.diff(t) > 0)                # monotonic across the concat (searchsorted)

    with pytest.raises(ValueError, match="n_months"):            # n_months >= 1
        build_era5_amip_forcing(str(tmp_path), "20200101", str(tmp_path / "z.nc"), n_months=0)

    z2 = tmp_path / "z2.nc"
    with pytest.raises(FileNotFoundError):                       # month 3 (202003) absent
        build_era5_amip_forcing(str(tmp_path), "20200101", str(z2), n_months=3)
    assert not z2.exists()                                       # failed loud BEFORE writing


def test_consecutive_months_handles_year_rollover():
    from scripts.data.load_local_era5 import _consecutive_months

    assert _consecutive_months("201709", 3) == ["201709", "201710", "201711"]
    assert _consecutive_months("201711", 3) == ["201711", "201712", "201801"]
    assert _consecutive_months("202012", 1) == ["202012"]
    with pytest.raises(ValueError, match="n must be"):
        _consecutive_months("202001", 0)


def test_build_era5_amip_forcing_missing_seaice_fails_loud(tmp_path):
    """A missing ci (sea-ice) chunk must fail loud at build time (via _find), not later
    inside the model run with a cryptic KeyError — and before writing any output."""
    from scripts.data.load_local_era5 import build_era5_amip_forcing

    _write_synthetic_rda_era5(tmp_path)             # sstk present, ci absent
    out = tmp_path / "amip_forcing.nc"
    with pytest.raises(FileNotFoundError, match="128_031_ci"):
        build_era5_amip_forcing(str(tmp_path), "20200101", str(out))
    assert not out.exists()                          # failed loud BEFORE writing


def test_apply_amip_forcing_to_config_injects_the_forcing_fields():
    """The field map turns a built ERA5 forcing into a ModelDriver-ready ExperimentConfig:
    EVERY shared forcing field is injected (AMIPForcingConfig.path -> ExperimentConfig.
    forcing_path) — incl. the surface BCs T_ice/albedos so an override is not dropped
    (codex-review iter 421) — and UNRELATED config fields (grid, dycore, radiation) are
    untouched.  Uses a hand-constructed forcing with NON-default values so every assertion
    is non-vacuous (vs the config defaults)."""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.forcing.amip import AMIPForcingConfig

    from scripts.data.load_local_era5 import apply_amip_forcing_to_config

    fcfg = AMIPForcingConfig(
        dataset="custom", path="forcing.nc", sic_path="seaice.nc",
        sst_var="SSTK", sic_var="CI", time_var="time",
        lat_var="latitude", lon_var="longitude",
        sst_offset=0.0, sic_scale=1.0,
        T_ice=270.0, albedo_ice=0.7, albedo_ocean=0.05)   # all NON-default
    base = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=1), radiation="gray", days=2,
        dataset="analytical")           # the default — must be overridden by the injection
    out = apply_amip_forcing_to_config(base, fcfg)

    # Every forcing field injected (path -> forcing_path; separate sic_path non-vacuous).
    assert out.dataset == "custom" and out.forcing_path == "forcing.nc"
    assert out.sic_path == "seaice.nc"
    assert out.sst_var == "SSTK" and out.sic_var == "CI" and out.time_var == "time"
    assert out.lat_var == "latitude" and out.lon_var == "longitude"
    assert out.sst_offset == 0.0 and out.sic_scale == 1.0
    # Surface BCs carried (an override would have been silently dropped before iter 421).
    assert out.T_ice == 270.0 and out.albedo_ice == 0.7 and out.albedo_ocean == 0.05
    # Unrelated fields preserved; the input is not mutated (NamedTuple._replace is a copy).
    assert out.radiation == "gray" and out.days == 2 and out.grid.nlev == 5
    assert base.dataset == "analytical" and base.T_ice != 270.0


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
    orig = tmp_path / pl_name("128_130_t", "20200101")
    shutil.copy(orig, tmp_path / pl_name("128_130_t", "20200101").replace("ll025sc", "ll025uv"))
    with pytest.raises(ValueError, match="files match"):
        open_local_era5_dataset(str(tmp_path), "20200101")
