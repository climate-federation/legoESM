"""Unit tests for building per-column land-model inputs from GlobalSurfaceData."""

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.land.global_surface_data import GlobalSurfaceData, GlobalSurfaceDataConfig
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
from legoesm.land.boundary_data import (
    build_canopy_params,
    build_soil_hydraulics,
    surface_data_param_provider,
    SurfaceDataParamProvider,
    surface_data_to_land_params,
    fill_land_param_gaps,
    surfdata_covered,
    dominant_pft_index,
    glacier_mask,
)
from legoesm.land.boundary_data.gap_fill import bare_land_surface_params
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


def test_soil_hydraulics_perlayer_and_fallback():
    # _gsd() uses (ncol=2, nlayer=4); col0 has 40% sand / 20% clay at every
    # layer, col1 is all-NaN (HWSD has no soil) and falls back per-cell.
    hy = build_soil_hydraulics(_gsd())
    assert hy.retention_curve == "clapp_hornberger"
    assert hy.theta_sat.shape == (2, 4)                       # per-(col, layer)
    assert np.all(np.isfinite(np.asarray(hy.theta_sat)))     # NaN cells fell back
    # col0 layers are all 40% sand; col1 all fell back to the sandy default ->
    # higher K_sat, lower b_ch. Holds at every layer because the input is
    # constant in depth.
    assert np.all(np.asarray(hy.K_sat[1]) > np.asarray(hy.K_sat[0]))
    assert np.all(np.asarray(hy.b_ch[1]) < np.asarray(hy.b_ch[0]))


def test_slab_seb_provider_returns_land_surface_params():
    gsd = _gsd()
    p = surface_data_param_provider(gsd, day_of_year=196.0, theta_top=jnp.full(2, 0.2))()
    assert isinstance(p, LandSurfaceParams)
    # every field is per-column (ncol=2) and physical
    assert p.albedo_veg.shape == (2,) and p.z0.shape == (2,)
    assert np.all((np.asarray(p.albedo_veg) > 0) & (np.asarray(p.albedo_veg) < 1))
    assert np.all(np.asarray(p.theta_fc) > np.asarray(p.theta_wp))


def test_slab_seb_provider_glacier_albedo():
    gsd = _gsd()._replace(f_land=jnp.array([1.0, 0.0]), f_lake=jnp.zeros(2),
                          f_glacier=jnp.array([0.0, 1.0]))
    p = surface_data_param_provider(gsd, 196.0, jnp.full(2, 0.2))()
    assert np.isclose(float(p.albedo_veg[1]), 0.6)        # glacier ice albedo
    assert float(p.albedo_veg[0]) < 0.6


def test_surface_data_provider_is_differentiable_through_pft_table():
    # The provider composes the trainable PFTParamProvider: gradient of a loss on
    # the materialized params must flow into the (n_pft, n_params) raw table.
    import equinox as eqx
    prov = surface_data_param_provider(_gsd(), 196.0, jnp.full(2, 0.2))
    assert isinstance(prov, SurfaceDataParamProvider)
    grad = eqx.filter_grad(lambda pr: jnp.sum(pr().Vc_max25 ** 2))(prov)
    gt = np.asarray(grad.pft_provider.raw_table)
    assert gt.shape == (N_PFT_CLM5, gt.shape[1]) and np.any(gt != 0.0)


def test_surface_data_provider_zero_cover_gets_bare_row():
    # A column with no PFT cover (ocean/ice/desert gap) must get the valid
    # bare-soil table row (nonzero heat capacity), not an all-zero fracs@table row
    # that would divide-by-zero in the surface step.
    gsd = _gsd()
    pft = np.asarray(gsd.pft_frac).copy(); pft[0, 1, :] = 0.0     # col1: zero cover
    gsd = gsd._replace(pft_frac=jnp.asarray(pft))
    p = surface_data_param_provider(gsd, 196.0, jnp.full(2, 0.2))()
    bare = bare_land_surface_params(2)
    assert float(p.C_soil[1]) > 0.0 and float(p.W_max[1]) > 0.0
    np.testing.assert_allclose(float(p.C_soil[1]), float(bare.C_soil[1]))


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

    sp = fill_land_param_gaps(
        surface_data_param_provider(gsd, 196.0, jnp.full(2, 0.2))(), gsd)
    assert np.all(np.isfinite(np.asarray(sp.albedo_veg)))
    # uncovered col1 -> bare-soil fallback row
    np.testing.assert_allclose(
        float(sp.albedo_veg[1]), float(bare_land_surface_params(2).albedo_veg[1]))


def test_fill_land_param_gaps_pins_to_authoritative_f_land():
    # Both columns are surfdata-covered, but the driver mask calls col1 OCEAN.
    # With f_land given, surfdata is kept only on driver-land (col0); col1 ->
    # bare fallback regardless of surfdata coverage.
    gsd = _gsd()
    bare1 = float(bare_land_surface_params(2).albedo_veg[1])

    sp_full = fill_land_param_gaps(
        surface_data_param_provider(gsd, 196.0, jnp.full(2, 0.2))(), gsd)
    sp_mask = fill_land_param_gaps(
        surface_data_param_provider(gsd, 196.0, jnp.full(2, 0.2))(), gsd,
        f_land=jnp.array([1.0, 0.0]))

    assert np.all(np.isfinite(np.asarray(sp_mask.albedo_veg)))
    # col0 (driver-land, covered): surfdata kept, identical to the no-mask fill.
    np.testing.assert_allclose(float(sp_mask.albedo_veg[0]), float(sp_full.albedo_veg[0]))
    # col1 (driver-ocean): bare fallback even though surfdata covers it.
    np.testing.assert_allclose(float(sp_mask.albedo_veg[1]), bare1)
    # ... and that differs from the surfdata value the no-mask fill would keep.
    assert not np.isclose(float(sp_full.albedo_veg[1]), bare1)


def test_fill_land_param_gaps_f_land_shape_mismatch_raises():
    gsd = _gsd()
    sp = surface_data_param_provider(gsd, 196.0, jnp.full(2, 0.2))()
    with pytest.raises(ValueError, match="ravel / grid-column mismatch"):
        fill_land_param_gaps(sp, gsd, f_land=jnp.ones(5))
