"""Tier-0 tests for the PLUMBER2 fluxtower forcing reader.

Hermetic: a synthetic PLUMBER2-format NetCDF is written to a temp dir with
xarray; no data download is required.  Exercises the full
``load_plumber2 -> plumber2_to_atm_series -> forcing_at_index`` path plus the
metadata/unit/error edge cases.

See ``docs/run_fluxnet_plan.md`` §8 (Tier-0 acceptance gate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from legoesm import constants
# coupler imported before land.* (package import-order convention)
from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.land.fluxnet_forcing import (
    Plumber2ForcingConfig,
    load_plumber2,
    plumber2_to_atm_series,
    forcing_at_index,
)

UMB_LAT, UMB_LON = 45.56, -84.71


# ---------------------------------------------------------------------------
# Synthetic PLUMBER2 NetCDF fixtures
# ---------------------------------------------------------------------------

def _met_series(n: int, *, celsius: bool = False, precip_mm_per_step: bool = False):
    """Build physically-plausible half-hourly met arrays of length ``n``.

    Air temperature swings across the freezing point so the rain/snow
    partition has both cold (snow) and warm (rain) timesteps to cover.
    """
    hour = (np.arange(n) * 0.5) % 24.0
    tair_K = 275.0 + 8.0 * np.sin(2.0 * np.pi * hour / 24.0)  # ~267-283 K
    tair = tair_K - constants.T_freeze if celsius else tair_K
    qair = np.full(n, 0.005)
    swdown = np.maximum(0.0, 600.0 * np.cos(2.0 * np.pi * (hour - 12.0) / 24.0))
    lwdown = np.full(n, 300.0)
    wind = np.full(n, 2.5)
    psurf = np.full(n, 9.8e4)
    precip = np.where(np.arange(n) % 24 < 2, 1e-4, 0.0)  # kg/m2/s, bursts
    if precip_mm_per_step:
        precip = precip * 1800.0  # mm per 30-min step
    co2 = np.full(n, 400.0)
    lai = np.full(n, 2.0)
    return dict(
        Tair=tair, Qair=qair, SWdown=swdown, LWdown=lwdown, Wind=wind,
        PSurf=psurf, Precip=precip, CO2air=co2, LAI=lai,
    )


def _write_met(
    path,
    *,
    n: int = 144,
    start: str = "2010-06-15 00:00",
    metadata_as_vars: bool = True,
    drop: tuple = (),
    celsius: bool = False,
    precip_mm_per_step: bool = False,
):
    """Write a synthetic PLUMBER2 ``*_Met.nc`` with (time, y, x) layout."""
    import xarray as xr

    time = pd.date_range(start, periods=n, freq="30min")
    series = _met_series(n, celsius=celsius, precip_mm_per_step=precip_mm_per_step)

    data_vars = {
        name: (("time", "y", "x"), arr.reshape(n, 1, 1))
        for name, arr in series.items()
        if name not in drop
    }
    attrs = {"site_id": "US-XXX", "IGBP_veg_short": "DBF"}
    meta = dict(latitude=UMB_LAT, longitude=UMB_LON,
                elevation=234.0, reference_height=46.0)
    if metadata_as_vars:
        for name, val in meta.items():
            data_vars[name] = (("y", "x"), np.array([[val]], dtype=np.float64))
    else:
        attrs.update(meta)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": time, "y": [0], "x": [0]},
        attrs=attrs,
    )
    ds.to_netcdf(path)
    return path


def _write_flux(path, *, n: int = 144, start: str = "2010-06-15 00:00"):
    """Write a synthetic PLUMBER2 ``*_Flux.nc`` with Qh/Qle/NEE."""
    import xarray as xr

    time = pd.date_range(start, periods=n, freq="30min")
    hour = (np.arange(n) * 0.5) % 24.0
    qh = np.maximum(0.0, 120.0 * np.cos(2.0 * np.pi * (hour - 12.0) / 24.0))
    qle = np.maximum(0.0, 80.0 * np.cos(2.0 * np.pi * (hour - 12.0) / 24.0))
    nee = np.full(n, -2.0)
    ds = xr.Dataset(
        data_vars={
            "Qh": (("time", "y", "x"), qh.reshape(n, 1, 1)),
            "Qle": (("time", "y", "x"), qle.reshape(n, 1, 1)),
            "NEE": (("time", "y", "x"), nee.reshape(n, 1, 1)),
        },
        coords={"time": time, "y": [0], "x": [0]},
    )
    ds.to_netcdf(path)
    return path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_load_basic_metadata_and_shapes(tmp_path):
    met = _write_met(tmp_path / "met.nc", n=144)
    data = load_plumber2(str(met))

    assert data.site.lat_deg == pytest.approx(UMB_LAT)
    assert data.site.lon_deg == pytest.approx(UMB_LON)
    assert data.site.elevation_m == pytest.approx(234.0)
    assert data.site.reference_height_m == pytest.approx(46.0)
    assert data.site.igbp_veg == "DBF"

    assert data.n_time == 144
    assert data.dt_s == pytest.approx(1800.0)        # 30-min cadence inferred
    assert data.tair.shape == (144,)
    assert data.lai is not None and data.lai.shape == (144,)
    # doy/hour derived consistently
    assert data.hour.min() >= 0.0 and data.hour.max() < 24.0
    assert np.all(np.isfinite(data.seconds))


def test_metadata_from_global_attrs(tmp_path):
    """Metadata stored as global attributes (alternate PLUMBER2 vintage)."""
    met = _write_met(tmp_path / "met_attrs.nc", metadata_as_vars=False)
    data = load_plumber2(str(met))
    assert data.site.lat_deg == pytest.approx(UMB_LAT)
    assert data.site.reference_height_m == pytest.approx(46.0)


def test_unit_knobs_celsius_and_mm(tmp_path):
    """tair_offset and precip_scale recover SI units from a Celsius/mm file."""
    met = _write_met(
        tmp_path / "met_c.nc", n=48, celsius=True, precip_mm_per_step=True,
    )
    cfg = Plumber2ForcingConfig(
        tair_offset=constants.T_freeze, precip_scale=1.0 / 1800.0,
    )
    data = load_plumber2(str(met), config=cfg)
    # Tair back in Kelvin (~267-283 range)
    assert 260.0 < float(data.tair.min()) and float(data.tair.max()) < 290.0
    # Precip back to a rate; max burst was 1e-4 kg/m2/s
    assert float(data.precip.max()) == pytest.approx(1e-4, rel=1e-6)


def test_co2_fallback_when_absent(tmp_path):
    met = _write_met(tmp_path / "met_noco2.nc", n=48, drop=("CO2air",))
    cfg = Plumber2ForcingConfig(co2_fallback_ppmv=415.0)
    data = load_plumber2(str(met), config=cfg)
    assert np.all(data.co2 == 415.0)


def test_missing_required_var_raises(tmp_path):
    met = _write_met(tmp_path / "met_notair.nc", n=48, drop=("Tair",))
    with pytest.raises(KeyError, match="Tair"):
        load_plumber2(str(met))


def test_eval_targets_loaded_from_flux(tmp_path):
    met = _write_met(tmp_path / "met.nc", n=144)
    flux = _write_flux(tmp_path / "flux.nc", n=144)
    data = load_plumber2(str(met), flux_path=str(flux))
    assert set(data.eval_targets) >= {"Qh", "Qle", "NEE"}
    assert data.eval_targets["Qh"].shape == (144,)
    # no flux file -> empty
    data2 = load_plumber2(str(met))
    assert data2.eval_targets == {}


def test_n_years_spans_calendar_boundary(tmp_path):
    """n_years counts distinct calendar years (drives spin-up cycling)."""
    met = _write_met(tmp_path / "met_xyear.nc", n=8, start="2009-12-31 22:00")
    data = load_plumber2(str(met))
    assert data.n_years == 2


# ---------------------------------------------------------------------------
# AtmToSurface builder + Tier-0 physical-range gate
# ---------------------------------------------------------------------------

def test_atm_series_physical_ranges(tmp_path):
    met = _write_met(tmp_path / "met.nc", n=144)
    data = load_plumber2(str(met))
    series = plumber2_to_atm_series(data)

    assert isinstance(series, AtmToSurface)
    # Shape (n_time, 1) on every field
    for field in series:
        assert field.shape == (144, 1)
        assert bool(jnp.all(jnp.isfinite(field))), "non-finite forcing field"

    T = np.asarray(series.T_lowest)
    assert T.min() > 200.0 and T.max() < 350.0
    rho = np.asarray(series.rho_lowest)
    assert rho.min() > 0.5 and rho.max() < 2.0
    cz = np.asarray(series.cos_zenith)
    assert cz.min() >= 0.0 and cz.max() <= 1.0
    # lowest-level pressure mirrors surface pressure (single tower level)
    assert np.allclose(np.asarray(series.p_lowest), np.asarray(series.p_surface))
    # gap-filled flags
    assert np.all(np.asarray(series.has_radiation) == 1.0)
    assert np.all(np.asarray(series.has_precipitation) == 1.0)


def test_q_passthrough_and_snow_partition(tmp_path):
    """q_lowest is exactly the stored Qair; snow is mass-conserving and
    fires only on sub-freezing timesteps."""
    met = _write_met(tmp_path / "met.nc", n=144)
    data = load_plumber2(str(met))
    series = plumber2_to_atm_series(data)

    assert np.allclose(np.asarray(series.q_lowest).ravel(), data.qair)

    snow = np.asarray(series.precip_snow).ravel()
    total = np.asarray(series.precip_total).ravel()
    assert np.all(snow <= total + 1e-15)              # mass-conserving
    cold = data.tair < constants.T_freeze
    # snow nonzero only where cold AND precipitating
    assert np.all(snow[~cold] == 0.0)
    assert np.all(snow[cold] == total[cold])


def test_forcing_at_index_roundtrip(tmp_path):
    met = _write_met(tmp_path / "met.nc", n=144)
    data = load_plumber2(str(met))
    series = plumber2_to_atm_series(data)

    i = 25
    single = forcing_at_index(series, i)
    assert isinstance(single, AtmToSurface)
    for sliced, full in zip(single, series):
        assert sliced.shape == (1,)
        assert float(sliced[0]) == pytest.approx(float(full[i, 0]))

    # spin-up cycling: index wraps with modulo, no out-of-bounds
    wrapped = forcing_at_index(series, 5 % data.n_time)
    assert wrapped.T_lowest.shape == (1,)
