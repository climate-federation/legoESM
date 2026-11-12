"""Lat-lon OMIP JRA55 forcing lands on the model's cell centres (not 0.5 deg east of them).

Before this fix ``_setup_jra55_forcing_state`` compared only grid SIZES: the 1-degree caches are
centred at lon 0.5..359.5, the model at 0..359, so cache column j drove model column j.  Now an
offset regular global cache is regridded conservatively onto the model centres, an aligned one
is passed through, and anything else is refused.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_run_omip_jra55_dispatch as D  # noqa: E402  (loads run_omip, cache/grid helpers)
from legoesm.grids.regridding import conservative_regrid_latlon, regrid_scalar  # noqa: E402

R = D.run_omip
LAT = np.arange(-89.5, 90.0, 1.0)
LON_M = np.arange(0.0, 360.0, 1.0)            # model cell centres
LON_C = LON_M + 0.5                           # the 1-degree caches' centres


def _field(lat, lon):
    return (np.cos(np.deg2rad(lon))[None, :] * 3.0 + np.sin(np.deg2rad(lat))[:, None] * 2.0
            + np.sin(3 * np.deg2rad(lon))[None, :] * np.cos(np.deg2rad(lat))[:, None])


def _area(lat):
    e = np.clip(np.concatenate([[lat[0] - .5], lat + .5]), -90, 90)
    return np.diff(np.sin(np.deg2rad(e)))[:, None] * np.ones((1, 360))


def test_aligned_cache_is_passed_through():
    assert R.latlon_forcing_regrid_weights(LAT, LON_M, LAT, LON_M) is None


def test_offset_cache_lands_on_model_centres_and_conserves():
    rw = R.latlon_forcing_regrid_weights(LAT, LON_C, LAT, LON_M)
    assert rw is not None and tuple(rw.target_shape) == (180, 360)
    f = _field(LAT, LON_C)
    got = np.asarray(regrid_scalar(jnp.asarray(f), rw))
    ref = conservative_regrid_latlon(f, LAT, LON_C, LAT, LON_M)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)
    # model column j is the mean of cache columns j-1/2 and j+1/2
    np.testing.assert_allclose(got, 0.5 * (np.roll(f, 1, axis=1) + f), atol=1e-12)
    a = _area(LAT)
    assert abs((got * a).sum() - (f * a).sum()) <= 1e-12 * np.abs(f * a).sum()
    # the shifted field is not accepted as-is
    assert np.abs(got - f).max() > 1e-3


@pytest.mark.parametrize("lat,lon,match", [
    (LAT, np.r_[LON_C[:-1], 400.0], "not a regular"),          # irregular longitude
    (LAT, np.linspace(0.25, 179.75, 360), "not a global"),      # half a globe
    (LAT[:-1], LON_C[:-1], "does not match model grid"),        # wrong size
])
def test_unusable_forcing_is_refused(lat, lon, match):
    with pytest.raises(ValueError, match=match):
        R.latlon_forcing_regrid_weights(lat, lon, LAT, LON_M)


def test_setup_regrids_an_offset_cache_onto_model_centres(tmp_path):
    """End to end through the startup path: the synthetic cache is centred half a cell east
    of the model (as the real 1-degree caches are), so setup must attach regrid weights and
    the loaded slice must equal the conservative remap onto the model centres."""
    cache = D._make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
    grid, *_ = D._make_tiny_latlon_setup(n_lat=4, n_lon=8)
    state = R._setup_jra55_forcing_state(D._argparse_namespace(jra55_cache=str(cache)),
                                         grid, "latlon")
    assert "regrid_weights" in state, "offset cache accepted on size alone"
    import xarray as xr
    ds = xr.open_zarr(str(cache))
    src = np.asarray(ds["tas"].isel(time=0).values)
    lat_m = np.rad2deg(np.asarray(grid.lat, np.float64))
    lon_m = np.rad2deg(np.asarray(grid.lon, np.float64))
    want = conservative_regrid_latlon(src, np.asarray(ds.lat), np.asarray(ds.lon), lat_m, lon_m)
    got = np.asarray(regrid_scalar(jnp.asarray(src), state["regrid_weights"]))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


def test_every_preload_lane_delivers_forcing_on_model_centres(tmp_path):
    """Loader-level: the block preload, the raw-record (interp) preload and the full-cache
    preload all hand the model the conservative remap of the cache, record by record."""
    cache = D._make_synthetic_cache(tmp_path, n_lat=4, n_lon=8, n_records=16)
    grid, *_ = D._make_tiny_latlon_setup(n_lat=4, n_lon=8)
    js = R._setup_jra55_forcing_state(D._argparse_namespace(jra55_cache=str(cache)),
                                      grid, "latlon")
    import xarray as xr
    ds = xr.open_zarr(str(cache), decode_times=False)
    lat_m = np.rad2deg(np.asarray(grid.lat, np.float64))
    lon_m = np.rad2deg(np.asarray(grid.lon, np.float64))

    def want(var, k):
        return conservative_regrid_latlon(np.asarray(ds[var].isel(time=k).values),
                                          np.asarray(ds.lat), np.asarray(ds.lon), lat_m, lon_m)

    dt = 3.0 * 3600.0                          # one record per step: step k reads record k
    atm, runoff = R._preload_jra55_forcing_block(0, 3, dt, js)
    for k in range(3):
        np.testing.assert_allclose(np.asarray(atm["sw_down"][k]), want("rsds", k), atol=1e-9)
        np.testing.assert_allclose(np.asarray(runoff[k]), want("friver", k), atol=1e-15)
    raw, _, meta = R._preload_jra55_raw_records(0, 2, dt, js)
    k0 = int(round(float(meta["record_days"][0]) * 8))
    np.testing.assert_allclose(np.asarray(raw["rsds"][0]), want("rsds", k0), atol=1e-9)
    full, _, _ = R._preload_jra55_full_cache(js)
    np.testing.assert_allclose(np.asarray(full["uas"][5]), want("uas", 5), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(full["vas"][5]), want("vas", 5), rtol=1e-6, atol=1e-6)
