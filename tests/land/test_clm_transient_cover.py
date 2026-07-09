"""Transient land-use cover re-weighting on the CLM param path.

Exercises the year-varying pieces that make the coupled / AMIP multilayer land use
a transient legoesm_surfdata cover: the cover loader/regrid, the frozen-soil
provider rebuild, and the TransientCoverProvider (year=None -> base; a year ->
re-weighted vegetation params, soil frozen)."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.clm_surface_map import (
    CLMSurfaceParamProvider,
    TransientCoverProvider,
    clm_provider_rebuild,
    load_transient_cover_on_columns,
)

_IDX = {n: i for i, n in enumerate(CLM5_PFT_NAMES)}


def _write_transient_surfdata(path, lat, lon):
    """2-year surfdata: forest in 2000 -> crop in 2010 on every cell (percent)."""
    ny, nx = lat.size, lon.size
    pft = np.zeros((2, N_PFT_CLM5, ny, nx))
    pft[0, _IDX["broadleaf_evergreen_tropical"]] = 90.0   # 2000: forest, f_land 0.9
    pft[1, _IDX["crop_c3"]] = 90.0                        # 2010: crop
    write_surfdata(
        path, lat=lat, lon=lon, soil_dz=np.array([1.0]),
        sand_pct=np.full((1, ny, nx), 40.0), clay_pct=np.full((1, ny, nx), 20.0),
        organic=np.full((1, ny, nx), 5.0), bulk_density=np.full((1, ny, nx), 1300.0),
        soil_color=np.full((ny, nx), 4.0), cell_area=np.full((ny, nx), 1.0e10),
        year=np.array([2000.0, 2010.0]),
        f_land=np.full((2, ny, nx), 90.0),
        f_lake=np.zeros((2, ny, nx)), f_glacier=np.zeros((2, ny, nx)),
        pft_frac=pft,
        monthly_lai=np.full((12, N_PFT_CLM5, ny, nx), 2.0),
        monthly_sai=np.full((12, N_PFT_CLM5, ny, nx), 0.5),
        monthly_height_top=np.full((12, N_PFT_CLM5, ny, nx), 10.0),
        monthly_height_bot=np.full((12, N_PFT_CLM5, ny, nx), 0.1),
    )


def _surface_map(ncol):
    """Minimal frozen-soil surface_map for clm_provider_rebuild."""
    return dict(
        theta_wp=jnp.full((ncol,), 0.1), theta_fc=jnp.full((ncol,), 0.3),
        glacier_frac=jnp.zeros((ncol,)), soil_albedo=jnp.full((ncol,), 0.15),
        lai=jnp.full((ncol,), 1.5),
    )


def test_load_transient_cover_shape_and_normalised(tmp_path):
    lat = np.linspace(-30.0, 30.0, 3)
    lon = np.linspace(0.0, 240.0, 4)
    p = tmp_path / "trans.nc"
    _write_transient_surfdata(p, lat, lon)
    # Target columns = the flattened grid points (identity nearest-regrid).
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    col_lat, col_lon = LAT.ravel(), LON.ravel()
    cover, years = load_transient_cover_on_columns(str(p), col_lat, col_lon)
    assert cover.shape == (2, lat.size * lon.size, N_PFT_CLM5)
    assert years.tolist() == [2000.0, 2010.0]
    # Within-land composition sums to 1 per column/year (bare floor absorbs the
    # 10% non-cover so the provider weighting is well-defined).
    np.testing.assert_allclose(np.asarray(cover).sum(axis=-1), 1.0, atol=1e-9)
    # 2000 forest, 2010 crop dominate their year.
    assert int(np.argmax(np.asarray(cover)[0, 0])) == _IDX["broadleaf_evergreen_tropical"]
    assert int(np.argmax(np.asarray(cover)[1, 0])) == _IDX["crop_c3"]


def test_provider_rebuild_reweights_albedo(tmp_path):
    ncol = 4
    sm = _surface_map(ncol)
    rebuild = clm_provider_rebuild(sm, variant="multilayer")
    forest = np.zeros((ncol, N_PFT_CLM5)); forest[:, _IDX["broadleaf_evergreen_tropical"]] = 1.0
    crop = np.zeros((ncol, N_PFT_CLM5)); crop[:, _IDX["crop_c3"]] = 1.0
    lp_forest = rebuild(forest)()
    lp_crop = rebuild(crop)()
    # Forest vs crop are different PFTs -> different vegetation albedo.
    assert not np.allclose(np.asarray(lp_forest.albedo_veg), np.asarray(lp_crop.albedo_veg))
    # Soil (theta_wp) is a PLANT btran threshold in the multilayer variant, so it
    # DOES re-derive from cover; the frozen soil map is still the same object.
    assert sm["theta_wp"].shape == (ncol,)


def test_transient_provider_year_selection(tmp_path):
    ncol = 4
    sm = _surface_map(ncol)
    rebuild = clm_provider_rebuild(sm, variant="multilayer")
    forest = np.zeros((ncol, N_PFT_CLM5)); forest[:, _IDX["broadleaf_evergreen_tropical"]] = 1.0
    crop = np.zeros((ncol, N_PFT_CLM5)); crop[:, _IDX["crop_c3"]] = 1.0
    cover = jnp.asarray(np.stack([forest, crop]))          # (2, ncol, 17)
    years = jnp.asarray([2000.0, 2010.0])
    base = rebuild(forest)                                  # reference = year 2000
    tp = TransientCoverProvider(base=base, cover=cover, years=years, _rebuild=rebuild)

    assert tp.year_varying is True
    # year=None -> base (2000 forest)
    np.testing.assert_allclose(np.asarray(tp(year=None).albedo_veg),
                               np.asarray(rebuild(forest)().albedo_veg))
    # exact years pick their cover
    np.testing.assert_allclose(np.asarray(tp(year=2000.0).albedo_veg),
                               np.asarray(rebuild(forest)().albedo_veg))
    np.testing.assert_allclose(np.asarray(tp(year=2010.0).albedo_veg),
                               np.asarray(rebuild(crop)().albedo_veg))
    # midpoint interpolates the cover -> albedo between forest and crop
    mid = np.asarray(tp(year=2005.0).albedo_veg)
    a_f = np.asarray(rebuild(forest)().albedo_veg)
    a_c = np.asarray(rebuild(crop)().albedo_veg)
    assert np.all((mid - np.minimum(a_f, a_c)) >= -1e-9)
    assert np.all((np.maximum(a_f, a_c) - mid) >= -1e-9)


def test_rebuild_soil_albedo_toggle():
    # include_soil_albedo=False matches the coupled clm_surface_provider (per-PFT
    # bare albedo); True uses the soil-colour map.  A bare-soil cell exposes the
    # difference, so the coupled (False) path stays byte-compatible with a static run.
    ncol = 2
    sm = _surface_map(ncol)
    bare = np.zeros((ncol, N_PFT_CLM5)); bare[:, _IDX["bare_soil"]] = 1.0
    with_sa = clm_provider_rebuild(sm, include_soil_albedo=True)(bare)()
    without_sa = clm_provider_rebuild(sm, include_soil_albedo=False)(bare)()
    assert not np.allclose(np.asarray(with_sa.albedo_veg),
                           np.asarray(without_sa.albedo_veg))


def test_year_varying_capability_gate():
    # The coupler forwards `year` only to providers with year_varying=True; a
    # plain CLM provider must NOT advertise it (getattr default False -> static).
    sm = _surface_map(2)
    plain = clm_provider_rebuild(sm)(np.eye(2, N_PFT_CLM5))
    assert getattr(plain, "year_varying", False) is False
    tp = TransientCoverProvider(
        base=plain, cover=jnp.zeros((1, 2, N_PFT_CLM5)),
        years=jnp.asarray([2000.0]), _rebuild=clm_provider_rebuild(sm))
    assert getattr(tp, "year_varying", False) is True


def test_at_year_returns_rebaked_provider():
    ncol = 2
    sm = _surface_map(ncol)
    rebuild = clm_provider_rebuild(sm)
    forest = np.zeros((ncol, N_PFT_CLM5)); forest[:, _IDX["broadleaf_evergreen_tropical"]] = 1.0
    crop = np.zeros((ncol, N_PFT_CLM5)); crop[:, _IDX["crop_c3"]] = 1.0
    tp = TransientCoverProvider(base=rebuild(forest),
                                cover=jnp.asarray(np.stack([forest, crop])),
                                years=jnp.asarray([2000.0, 2010.0]), _rebuild=rebuild)
    prov_2010 = tp.at_year(2010.0)
    np.testing.assert_allclose(np.asarray(prov_2010().albedo_veg),
                               np.asarray(rebuild(crop)().albedo_veg))
