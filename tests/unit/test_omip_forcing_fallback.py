"""OMIP forcing loaders must not SILENTLY substitute synthetic analytic data.

The CORE-II (OMIP-1) and JRA55-do (OMIP-2) loaders fall back to a deterministic
synthetic climatology when the on-disk cache is missing.  That fallback is a
footgun: a run can look like an OMIP integration while using non-protocol
forcing.  These tests pin the two safe behaviours:

* ``allow_synthetic=False`` raises ``FileNotFoundError`` (fail-loud — the
  OMIP-faithful pipelines, e.g. run_omip_core2.py, use this), and
* ``allow_synthetic=True`` (the convenience default for idealized spin-ups)
  emits a LOUD warning naming it as non-OMIP before returning synthetic data.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from legoesm.ocean.forcing.core2 import core2_nyf_path, load_core2_nyf
from legoesm.ocean.forcing.dai_trenberth import load_dai_trenberth
from legoesm.ocean.forcing.jra55_do import load_jra55_do
from legoesm.ocean.forcing.woa import load_woa_sst
from legoesm.ocean.forcing.woa_sss import load_woa_sss


def test_core2_missing_cache_fails_loud_when_disallowed(tmp_path):
    with pytest.raises(FileNotFoundError, match="CORE-II NYF cache missing"):
        load_core2_nyf(cache_dir=tmp_path, allow_synthetic=False)


def test_core2_synthetic_fallback_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING,
                         logger="legoesm.ocean.forcing.core2"):
        f = load_core2_nyf(cache_dir=tmp_path, allow_synthetic=True, n_time=12)
    assert f.u10.shape[0] == 12  # synthetic returned
    assert any("SYNTHETIC" in r.message and "OMIP-1" in r.message
               for r in caplog.records), "missing loud synthetic-fallback warning"


def test_jra55_missing_cache_fails_loud_when_disallowed(tmp_path):
    with pytest.raises(FileNotFoundError, match="JRA55-do cache missing"):
        load_jra55_do(1990, cache_dir=tmp_path, allow_synthetic=False)


def test_jra55_synthetic_fallback_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING,
                         logger="legoesm.ocean.forcing.jra55_do"):
        load_jra55_do(1990, cache_dir=tmp_path, allow_synthetic=True)
    assert any("SYNTHETIC" in r.message and "OMIP-2" in r.message
               for r in caplog.records), "missing loud synthetic-fallback warning"


def test_core2_nyf_path_is_the_path_the_loader_reads(tmp_path):
    """``core2_nyf_path`` exists so a run can RECORD which archive it used
    (codex r8: with --forcing-path unset the location comes from the
    environment/home, so two identical command lines can read DIFFERENT
    forcing while the restart fingerprint recorded only ``forcing_path=None``).

    It must therefore agree with the loader EXACTLY.  Proven, not asserted:
    the loader names the resolved path in its FileNotFoundError.
    """
    assert core2_nyf_path(tmp_path) == tmp_path / "nyf.zarr"
    with pytest.raises(FileNotFoundError) as exc:
        load_core2_nyf(cache_dir=tmp_path, allow_synthetic=False)
    assert str(core2_nyf_path(tmp_path)) in str(exc.value)

    # ...and the default (cache_dir=None) resolves somewhere else entirely,
    # which is the whole reason the resolved path has to be fingerprinted.
    assert core2_nyf_path() != core2_nyf_path(tmp_path)
    assert core2_nyf_path().name == "nyf.zarr"


@pytest.mark.parametrize("loader,logger_name", [
    (load_woa_sst, "legoesm.ocean.forcing.woa"),
    (load_woa_sss, "legoesm.ocean.forcing.woa_sss"),
    (load_dai_trenberth, "legoesm.ocean.forcing.dai_trenberth"),
])
def test_obs_synthetic_fallback_warns(tmp_path, caplog, loader, logger_name):
    with caplog.at_level(logging.WARNING, logger=logger_name):
        loader(cache_dir=tmp_path, allow_synthetic=True)
    assert any("SYNTHETIC" in r.message for r in caplog.records)


def test_woa_sss_real_file_surface_2d_and_lon_wrapped(tmp_path):
    """WOA s_an is (time, depth, lat, lon) on a -179.5..179.5 axis; the
    loader must return a 2-D surface field on [0, 360) ascending."""
    xr = pytest.importorskip("xarray")
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(-179.5, 180.0, 1.0)
    s = np.zeros((1, 3, lat.size, lon.size))
    s[0, 0] = np.broadcast_to(np.mod(lon, 360.0), (lat.size, lon.size))
    s[0, 1:] = -1.0                           # deeper levels must not leak
    xr.Dataset({"s_an": (("time", "depth", "lat", "lon"), s)},
               coords={"lat": lat, "lon": lon}).to_netcdf(
        tmp_path / "woa_sss_annual.nc")
    sss, _, lon_o = load_woa_sss(cache_dir=tmp_path, allow_synthetic=False)
    assert sss.shape == (lat.size, lon.size)
    assert np.all(np.diff(lon_o) > 0) and lon_o[0] >= 0.0 and lon_o[-1] < 360.0
    np.testing.assert_array_equal(sss[0], lon_o)


def test_woa_sss_real_file_coastal_missing_values_do_not_reach_targets(tmp_path):
    """_FillValue land points decode to NaN; a target whose bilinear stencil
    touches one must still get a finite salinity (filled from the nearest
    observed point), and an all-missing file must raise."""
    xr = pytest.importorskip("xarray")
    import jax.numpy as jnp
    from legoesm.ocean.forcing.sss_restoring import interp_woa_sss_to_grid
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    s = np.full((1, 1, lat.size, lon.size), 35.0)
    s[0, 0, 90:95, 100:105] = np.nan              # a "continent" of 5x5 cells
    xr.Dataset({"s_an": (("time", "depth", "lat", "lon"), s)},
               coords={"lat": lat, "lon": lon}).to_netcdf(
        tmp_path / "woa_sss_annual.nc")
    sss, lat_o, lon_o = load_woa_sss(cache_dir=tmp_path, allow_synthetic=False)
    assert np.isfinite(sss).all()
    # coastal ocean target at (0.25 N, 99.75 E): its stencil includes the NaN
    # corner at (0.5 N, 100.5 E)
    out = interp_woa_sss_to_grid(jnp.asarray(sss), jnp.asarray(lat_o),
                                 jnp.asarray(lon_o), jnp.array([0.25]),
                                 jnp.array([99.75]))
    np.testing.assert_allclose(np.asarray(out), 35.0, rtol=1e-12)
    s[:] = np.nan
    xr.Dataset({"s_an": (("time", "depth", "lat", "lon"), s)},
               coords={"lat": lat, "lon": lon}).to_netcdf(
        tmp_path / "woa_sss_annual.nc")
    with pytest.raises(ValueError, match="no valid values"):
        load_woa_sss(cache_dir=tmp_path, allow_synthetic=False)


@pytest.mark.parametrize("bad,match", [
    ("descending_lat", "latitude must be ascending"),
    ("nonuniform_lat", "latitude must be ascending and uniform"),
    ("regional_lon", "longitude must be global"),
])
def test_woa_sss_real_file_rejects_layouts_the_interpolator_misreads(tmp_path, bad, match):
    xr = pytest.importorskip("xarray")
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    if bad == "descending_lat":
        lat = lat[::-1]
    elif bad == "nonuniform_lat":
        lat = np.concatenate([lat[:90], lat[90:] + 0.25])  # 1.25 deg jump
    else:
        lon = np.arange(0.5, 180.0, 0.5)          # half the globe
    s = np.full((1, 1, lat.size, lon.size), 35.0)
    xr.Dataset({"s_an": (("time", "depth", "lat", "lon"), s)},
               coords={"lat": lat, "lon": lon}).to_netcdf(
        tmp_path / "woa_sss_annual.nc")
    with pytest.raises(ValueError, match=match):
        load_woa_sss(cache_dir=tmp_path, allow_synthetic=False)
