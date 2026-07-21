"""Unit tests for the M2 6-hourly -> model-dt forcing disaggregation.

Key properties:
  * shortwave conserves each 6-h interval mean (energy conservation),
  * precip constant-hold conserves the interval mass,
  * instantaneous channels are linearly interpolated,
  * the diurnal SW cycle is physical (zero at night, peak near local noon).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.forcing import cru_jra

jax.config.update("jax_enable_x64", True)

_SEC_PER_HOUR = 3600.0
_STEP_S = cru_jra.CRUJRA_FREQ_HOURS * _SEC_PER_HOUR     # 6 h


def _cols(n_time=5, seed=1):
    """Synthetic forcing regridded to a few columns spanning latitudes."""
    f = cru_jra.synthetic_land_forcing(2000, n_time=n_time)
    # give FSDS a per-interval spread so conservation is a non-trivial check
    rng = np.random.default_rng(seed)
    fsds = f.fsds * (0.5 + rng.uniform(0, 1, (n_time, 1, 1)))
    f = f._replace(fsds=fsds.astype(np.float32))
    lat = np.radians(np.array([-30.0, 0.0, 45.0]))
    lon = np.radians(np.array([0.0, 120.0, 240.0]))
    w = cru_jra.build_forcing_weights(f, lat, lon)
    return f, cru_jra.regrid_forcing(f, w), lat, lon


def _aligned_times(n_int, dt_s):
    """Model times tiling the [0, n_int*6h) solar intervals exactly."""
    n_steps = int(round(n_int * _STEP_S / dt_s))
    return np.arange(n_steps, dtype=np.float64) * dt_s


@pytest.mark.parametrize("dt_s", [_SEC_PER_HOUR, 0.5 * _SEC_PER_HOUR])
def test_shortwave_conserves_interval_mean(dt_s):
    f, cols, lat, lon = _cols(n_time=5)
    n_int = cols.fsds.shape[0]
    tq = _aligned_times(n_int, dt_s)
    atm = cru_jra.disaggregate_forcing(cols, lat, lon, tq)
    sw = np.asarray(atm.sw_down)                        # (n_steps, ncol)

    sub = int(round(_STEP_S / dt_s))
    for k in range(n_int):
        seg = sw[k * sub:(k + 1) * sub]                 # substeps tiling interval k
        np.testing.assert_allclose(seg.mean(axis=0), cols.fsds[k], rtol=1e-9, atol=1e-9)


def test_precip_constant_hold_conserves_mass():
    f, cols, lat, lon = _cols(n_time=4)
    n_int = cols.prectmms.shape[0]
    dt_s = _SEC_PER_HOUR
    tq = _aligned_times(n_int, dt_s)
    atm = cru_jra.disaggregate_forcing(cols, lat, lon, tq)
    precip = np.asarray(atm.precip_total)
    sub = int(round(_STEP_S / dt_s))
    for k in range(n_int):
        seg = precip[k * sub:(k + 1) * sub]
        # constant over the interval AND equal to the interval value
        assert np.allclose(seg, cols.prectmms[k])


def test_linear_interp_of_temperature():
    f, cols, lat, lon = _cols(n_time=4)
    # query exactly at a midpoint stamp -> equals that sample
    t_stamp = float(cols.time_s[1])
    atm = cru_jra.disaggregate_forcing(cols, lat, lon, np.array([t_stamp]))
    np.testing.assert_allclose(np.asarray(atm.T_lowest)[0], cols.tbot[1], rtol=1e-9)
    # query halfway between stamps 1 and 2 -> mean of the two
    t_mid = 0.5 * (float(cols.time_s[1]) + float(cols.time_s[2]))
    atm2 = cru_jra.disaggregate_forcing(cols, lat, lon, np.array([t_mid]))
    expect = 0.5 * (cols.tbot[1] + cols.tbot[2])
    np.testing.assert_allclose(np.asarray(atm2.T_lowest)[0], expect, rtol=1e-9)


def test_diurnal_shape_sunrise_zero_and_noon_peak():
    # Dedicated equatorial column at lon 0 so local solar time == UTC.
    f = cru_jra.synthetic_land_forcing(2000, n_time=4)
    lat = np.radians(np.array([0.0]))
    lon = np.radians(np.array([0.0]))
    cols = cru_jra.regrid_forcing(f, cru_jra.build_forcing_weights(f, lat, lon))
    tq = np.arange(24, dtype=np.float64) * _SEC_PER_HOUR  # one day, 1-h steps
    atm = cru_jra.disaggregate_forcing(cols, lat, lon, tq)
    sw = np.asarray(atm.sw_down)[:, 0]
    # Sunrise hour (06 UTC) sits INSIDE the daylit interval [06,12): cos(zenith)
    # is ~0 there (cos(pi/2) is 0 to float precision), so the zenith weighting
    # zeroes it (not the fallback).
    assert sw[6] < 1e-6
    # Within the [12,18) interval, local noon (hour 12) is the brightest substep.
    assert int(np.asarray(atm.sw_down)[12:18, 0].argmax()) == 0
    assert np.all(sw >= 0.0)


def test_output_shapes_and_snow_bound():
    f, cols, lat, lon = _cols(n_time=4)
    tq = _aligned_times(cols.fsds.shape[0], _SEC_PER_HOUR)
    atm = cru_jra.disaggregate_forcing(cols, lat, lon, tq)
    assert isinstance(atm, AtmToSurface)
    assert np.asarray(atm.T_lowest).shape == (tq.size, lat.size)
    assert np.all(np.isfinite(np.asarray(atm.rho_lowest)))
    assert np.all(np.asarray(atm.precip_snow) <= np.asarray(atm.precip_total) + 1e-12)
    # scalar wind carried in u, zero in v
    assert np.allclose(np.asarray(atm.v_lowest), 0.0)


def test_stage_forcing_synthetic_scan_ready():
    # stage_forcing composes load->regrid->disaggregate; output must be a
    # stacked AtmToSurface with every leaf carrying the n_steps leading axis
    # (so it is directly lax.scan-able).
    lat = np.radians(np.array([-20.0, 0.0, 40.0]))
    lon = np.radians(np.array([0.0, 90.0, 200.0]))
    dt = _SEC_PER_HOUR
    model_times = dt * np.arange(30, dtype=np.float64)
    atm = cru_jra.stage_forcing(lat, lon, model_times, year=2000, data_dir=None)
    leaves = jax.tree_util.tree_leaves(atm)
    assert all(np.asarray(leaf).shape == (30, lat.size) for leaf in leaves)
    assert np.all(np.isfinite(np.asarray(atm.T_lowest)))


def test_stage_forcing_is_lax_scannable():
    import jax.lax as lax
    lat = np.radians(np.array([0.0, 30.0]))
    lon = np.radians(np.array([0.0, 120.0]))
    model_times = _SEC_PER_HOUR * np.arange(12, dtype=np.float64)
    atm = cru_jra.stage_forcing(lat, lon, model_times, year=2000, data_dir=None)
    # a trivial scan over the forcing proves the pytree leading-axis is uniform
    out = lax.scan(lambda c, f: (c, f.T_lowest.mean()), 0.0, atm)[1]
    assert np.asarray(out).shape == (12,)


@pytest.mark.skipif(
    not __import__("os").path.exists(
        "/Users/linnia/Desktop/clmforc.CRUJRAv2.5_0.5x0.5.TPQWL.2023.nc"
    ),
    reason="real CRU-JRA example files not present",
)
def test_stage_forcing_real_data_physical():
    # Full load->regrid->disaggregate on the real (land-only, ocean=NaN) files at
    # ~2 deg. Land cells must be finite & physical; ocean stays NaN. Guards the
    # NaN-aware-regrid requirement (plain IDW would NaN-poison every land cell).
    from legoesm.grids.latlon import create_latlon_grid
    grid = create_latlon_grid(90)
    lat = jnp.asarray(np.asarray(grid.lat2d).ravel())      # radians (as the driver uses)
    lon = jnp.asarray(np.asarray(grid.lon2d).ravel())
    mt = 196.0 * 86400.0 + _SEC_PER_HOUR * np.arange(24)   # mid-July, 1-day, 1-h
    atm = cru_jra.stage_forcing(lat, lon, mt, year=2023, data_dir="/Users/linnia/Desktop")
    T = np.asarray(atm.T_lowest)
    land = np.isfinite(T[0])
    assert land.sum() > 5000                               # real global land coverage
    Tl = T[:, land]
    assert np.all(np.isfinite(Tl))
    assert 200.0 < np.nanmin(Tl) and np.nanmax(Tl) < 340.0
    sw = np.asarray(atm.sw_down)[:, land]
    assert 0.0 <= np.nanmin(sw) and np.nanmax(sw) < 1400.0  # clear-sky bound


def test_night_window_guard_no_nan():
    # high-latitude winter column can have a fully-dark interval; ensure no NaNs.
    f = cru_jra.synthetic_land_forcing(2000, n_time=4)
    lat = np.radians(np.array([85.0]))                  # near-polar
    lon = np.radians(np.array([0.0]))
    cols = cru_jra.regrid_forcing(f, cru_jra.build_forcing_weights(f, lat, lon))
    tq = _aligned_times(cols.fsds.shape[0], _SEC_PER_HOUR)
    atm = cru_jra.disaggregate_forcing(cols, lat, lon, tq)
    assert np.all(np.isfinite(np.asarray(atm.sw_down)))
