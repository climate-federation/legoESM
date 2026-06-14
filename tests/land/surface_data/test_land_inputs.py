"""Unit tests for building per-column land-model inputs from GlobalSurfaceData."""

import numpy as np
import jax.numpy as jnp

from legoesm.land.global_surface_data import GlobalSurfaceData, GlobalSurfaceDataConfig
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
from legoesm.land.surface_data.land_inputs import (
    build_canopy_params,
    build_soil_hydraulics,
    build_land_surface_params,
    surface_data_to_land_params,
    fill_land_param_gaps,
    surfdata_covered,
    dominant_pft_index,
    glacier_mask,
)
from legoesm.land.surface_params import LandSurfaceParams
from legoesm.land.canopy.config import CanopyLandParams
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig


def _gsd(ncol=2, nlayer=4):
    npft = N_PFT_CLM5
    pft = np.zeros((1, ncol, npft))
    be = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
    c4 = CLM5_PFT_NAMES.index("c4_grass")
    pft[0, 0, be] = 1.0          # col0 dominant BE-tropical
    pft[0, 1, c4] = 1.0          # col1 dominant C4 grass
    veg = lambda val: np.full((12, ncol, npft), val)
    lai = np.zeros((12, ncol, npft)); lai[:, 0, be] = 5.0; lai[:, 1, c4] = 1.5
    htop = np.zeros((12, ncol, npft)); htop[:, 0, be] = 25.0
    sand = np.full((ncol, nlayer), 0.40); clay = np.full((ncol, nlayer), 0.20)
    sand[1] = np.nan; clay[1] = np.nan         # col1: no HWSD soil -> fallback
    z = lambda: np.zeros((ncol, nlayer))
    return GlobalSurfaceData(
        sand_frac=jnp.asarray(sand), clay_frac=jnp.asarray(clay),
        organic=jnp.asarray(z()), bulk_density=jnp.asarray(z()),
        soil_color=jnp.asarray(np.array([5, 18])),
        cell_area=jnp.ones(ncol),
        years=jnp.asarray([2015.0]),
        f_land=jnp.ones((1, ncol)), f_lake=jnp.zeros((1, ncol)),
        f_glacier=jnp.zeros((1, ncol)), pft_frac=jnp.asarray(pft),
        months=jnp.arange(12.0),
        lai_monthly=jnp.asarray(lai), sai_monthly=jnp.asarray(veg(0.0)),
        htop_monthly=jnp.asarray(htop), hbot_monthly=jnp.asarray(veg(0.0)),
        config=GlobalSurfaceDataConfig(),
    )


def test_dominant_pft():
    dom = dominant_pft_index(_gsd())
    assert CLM5_PFT_NAMES[dom[0]] == "broadleaf_evergreen_tropical"
    assert CLM5_PFT_NAMES[dom[1]] == "c4_grass"


def test_canopy_params_from_dominant_pft():
    gsd = _gsd()
    cp = build_canopy_params(gsd, day_of_year=196.0, theta_top=jnp.full(2, 0.2))
    assert cp.LAI.shape == (2,)
    np.testing.assert_allclose(float(cp.LAI[0]), 5.0)        # BE-trop dominant LAI
    np.testing.assert_allclose(float(cp.hc[0]), 25.0)        # from surfdata htop
    # C4 column flagged, C4 Vcmax populated, C3 zero there
    np.testing.assert_allclose(float(cp.fC4[1]), 1.0)
    assert float(cp.Vcmax25_C4_leaf[1]) > 0 and float(cp.Vcmax25_C3_leaf[1]) == 0.0
    assert float(cp.Vcmax25_C3_leaf[0]) > 0                  # BE-trop is C3
    # soil-colour albedo present and ordered (darker class 18 < class 5)
    assert float(cp.ALB_VIS[1]) < float(cp.ALB_VIS[0])


def test_glacier_columns_get_ice_surface():
    gsd = _gsd()._replace(
        f_land=jnp.array([1.0, 0.0]),
        f_lake=jnp.zeros(2),
        f_glacier=jnp.array([0.0, 1.0]),       # column 1 is glacier-dominant
    )
    assert glacier_mask(gsd).tolist() == [False, True]
    cp = build_canopy_params(gsd, day_of_year=196.0, theta_top=jnp.full(2, 0.2))
    # glacier column: no veg, high ice albedo
    assert float(cp.LAI[1]) == 0.0
    assert np.isclose(float(cp.FNonVeg[1]), 1.0)
    assert np.isclose(float(cp.ALB_VIS[1]), 0.70)
    assert np.isclose(float(cp.ALB_NIR[1]), 0.50)
    # non-glacier column keeps its (darker) soil-colour background albedo
    assert float(cp.ALB_VIS[0]) < 0.70


def test_soil_hydraulics_percolumn_and_fallback():
    hy = build_soil_hydraulics(_gsd())
    assert hy.retention_curve == "clapp_hornberger"
    assert hy.theta_sat.shape == (2, 1)
    assert np.all(np.isfinite(np.asarray(hy.theta_sat)))     # NaN col fell back
    # col0 from 40% sand; col1 fell back to sandy default -> higher K_sat, lower b
    assert float(hy.K_sat[1, 0]) > float(hy.K_sat[0, 0])
    assert float(hy.b_ch[1, 0]) < float(hy.b_ch[0, 0])


def test_slab_seb_adapter_returns_land_surface_params():
    gsd = _gsd()
    p = build_land_surface_params(gsd, day_of_year=196.0, theta_top=jnp.full(2, 0.2))
    assert isinstance(p, LandSurfaceParams)
    # every field is per-column (ncol=2) and physical
    assert p.albedo_veg.shape == (2,) and p.z0.shape == (2,)
    assert np.all((np.asarray(p.albedo_veg) > 0) & (np.asarray(p.albedo_veg) < 1))
    assert np.all(np.asarray(p.theta_fc) > np.asarray(p.theta_wp))


def test_slab_seb_adapter_glacier_albedo():
    gsd = _gsd()._replace(f_land=jnp.array([1.0, 0.0]), f_lake=jnp.zeros(2),
                          f_glacier=jnp.array([0.0, 1.0]))
    p = build_land_surface_params(gsd, 196.0, jnp.full(2, 0.2))
    assert np.isclose(float(p.albedo_veg[1]), 0.6)        # glacier ice albedo
    assert float(p.albedo_veg[0]) < 0.6


def test_dispatch_by_surface_scheme():
    gsd = _gsd()
    seb = surface_data_to_land_params(gsd, SimpleSEBConfig(), 196.0, jnp.full(2, 0.2))
    can = surface_data_to_land_params(gsd, TwoLeafCanopyConfig(), 196.0, jnp.full(2, 0.2))
    assert isinstance(seb, LandSurfaceParams)
    assert isinstance(can, CanopyLandParams)


def test_fill_land_param_gaps_uses_bare_fallback():
    # mark column 1 as surfdata-uncovered (NaN pft_frac), like a coast/island the
    # driver land-mask calls land but the surfdata does not cover.
    gsd = _gsd()
    pft = np.asarray(gsd.pft_frac).copy(); pft[0, 1, :] = np.nan
    gsd = gsd._replace(pft_frac=jnp.asarray(pft))
    assert surfdata_covered(gsd).tolist() == [True, False]

    cp = fill_land_param_gaps(build_canopy_params(gsd, 196.0, jnp.full(2, 0.2)), gsd)
    assert np.all(np.isfinite(np.asarray(cp.LAI)))
    assert float(cp.FNonVeg[1]) == 1.0 and float(cp.LAI[1]) == 0.0   # col1 -> bare

    sp = fill_land_param_gaps(build_land_surface_params(gsd, 196.0, jnp.full(2, 0.2)), gsd)
    assert np.all(np.isfinite(np.asarray(sp.albedo_veg)))
    np.testing.assert_allclose(float(sp.albedo_veg[1]), 0.30)        # bare-soil row
