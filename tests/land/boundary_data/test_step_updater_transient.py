"""Transient-cover behaviour of the per-step land-params updater.

The updater interpolates PFT cover to a traced calendar ``year`` (interp_annual),
so a multi-year surfdata drives LAI / dominant-PFT / C3-C4 as land use changes,
while a single-year surfdata stays year-invariant (backward compatible).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from legoesm.land.global_surface_data import GlobalSurfaceData, GlobalSurfaceDataConfig
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
from legoesm.land.boundary_data.step_updater import make_step_land_params_updater
from legoesm.land.surface_scheme import SimpleSEBConfig
from legoesm.land.canopy import CanopyConfig

_BE = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")   # C3 forest
_C4 = CLM5_PFT_NAMES.index("c4_grass")                        # C4 grass
_LAI_FOREST, _LAI_GRASS = 5.0, 1.5


def _gsd(years, pft):
    """A 1-column GlobalSurfaceData with the given years + (nyear,1,npft) cover."""
    nyear = len(years)
    ncol, npft = 1, N_PFT_CLM5
    # Monthly LAI per PFT (climatology, shared across years): forest vs grass.
    lai = np.zeros((12, ncol, npft))
    lai[:, 0, _BE] = _LAI_FOREST
    lai[:, 0, _C4] = _LAI_GRASS
    htop = np.zeros((12, ncol, npft)); htop[:, 0, _BE] = 25.0
    z2 = np.zeros((ncol, 4))
    return GlobalSurfaceData(
        sand_frac=jnp.asarray(np.full((ncol, 4), 0.4)),
        clay_frac=jnp.asarray(np.full((ncol, 4), 0.2)),
        organic=jnp.asarray(z2), bulk_density=jnp.asarray(z2),
        soil_color=jnp.asarray(np.array([5])),
        cell_area=jnp.ones(ncol),
        years=jnp.asarray(np.asarray(years, dtype=float)),
        f_land=jnp.ones((nyear, ncol)), f_lake=jnp.zeros((nyear, ncol)),
        f_glacier=jnp.zeros((nyear, ncol)),
        pft_frac=jnp.asarray(np.asarray(pft, dtype=float)),
        months=jnp.arange(12.0),
        lai_monthly=jnp.asarray(lai), sai_monthly=jnp.zeros((12, ncol, npft)),
        htop_monthly=jnp.asarray(htop), hbot_monthly=jnp.zeros((12, ncol, npft)),
        config=GlobalSurfaceDataConfig(),
    )


def _forest_then_grass():
    """2-year cover: pure forest in 1850, pure C4 grass in 2000."""
    pft = np.zeros((2, 1, N_PFT_CLM5))
    pft[0, 0, _BE] = 1.0
    pft[1, 0, _C4] = 1.0
    return _gsd([1850.0, 2000.0], pft)


def test_seb_lai_follows_transient_cover():
    gsd = _forest_then_grass()
    upd = make_step_land_params_updater(gsd, SimpleSEBConfig())
    theta, doy = jnp.full(1, 0.2), jnp.asarray(196.0)

    _, lai0 = upd(theta, doy, jnp.asarray(1850.0))
    _, lai1 = upd(theta, doy, jnp.asarray(2000.0))
    _, lai_mid = upd(theta, doy, jnp.asarray(1925.0))     # halfway -> 0.5/0.5 blend

    np.testing.assert_allclose(float(lai0[0]), _LAI_FOREST, rtol=1e-6)
    np.testing.assert_allclose(float(lai1[0]), _LAI_GRASS, rtol=1e-6)
    np.testing.assert_allclose(float(lai_mid[0]),
                               0.5 * (_LAI_FOREST + _LAI_GRASS), rtol=1e-6)


def test_seb_albedo_responds_to_cover():
    gsd = _forest_then_grass()
    upd = make_step_land_params_updater(gsd, SimpleSEBConfig())
    theta, doy = jnp.full(1, 0.2), jnp.asarray(196.0)
    lp0, _ = upd(theta, doy, jnp.asarray(1850.0))
    lp1, _ = upd(theta, doy, jnp.asarray(2000.0))
    # Forest vs grass have different PFT-table albedo AND different LAI -> the
    # blended surface albedo must differ between the two years.
    assert not np.isclose(float(lp0.albedo_veg[0]), float(lp1.albedo_veg[0]))


def test_canopy_dominant_pft_flips_c3_to_c4():
    gsd = _forest_then_grass()
    upd = make_step_land_params_updater(gsd, CanopyConfig(max_iters=1))
    theta, doy = jnp.full(1, 0.2), jnp.asarray(196.0)
    cp0, _ = upd(theta, doy, jnp.asarray(1850.0))         # forest = C3
    cp1, _ = upd(theta, doy, jnp.asarray(2000.0))         # grass = C4

    np.testing.assert_allclose(float(cp0.fC4[0]), 0.0)
    np.testing.assert_allclose(float(cp1.fC4[0]), 1.0)
    assert float(cp0.Vcmax25_C3_leaf[0]) > 0.0 and float(cp0.Vcmax25_C4_leaf[0]) == 0.0
    assert float(cp1.Vcmax25_C4_leaf[0]) > 0.0 and float(cp1.Vcmax25_C3_leaf[0]) == 0.0
    np.testing.assert_allclose(float(cp0.LAI[0]), _LAI_FOREST, rtol=1e-6)
    np.testing.assert_allclose(float(cp1.LAI[0]), _LAI_GRASS, rtol=1e-6)


def test_seb_fC4_follows_transient_cover():
    """SEB fC4 tracks the per-year dominant PFT (forest C3 -> grass C4), not a
    frozen time-mean flag."""
    gsd = _forest_then_grass()
    upd = make_step_land_params_updater(gsd, SimpleSEBConfig())
    theta, doy = jnp.full(1, 0.2), jnp.asarray(196.0)
    lp0, _ = upd(theta, doy, jnp.asarray(1850.0))         # forest = C3
    lp1, _ = upd(theta, doy, jnp.asarray(2000.0))         # grass = C4
    np.testing.assert_allclose(float(lp0.fC4[0]), 0.0)
    np.testing.assert_allclose(float(lp1.fC4[0]), 1.0)


def test_single_year_surfdata_is_year_invariant():
    """Backward compat: nyear==1 cover gives identical params for any year."""
    pft = np.zeros((1, 1, N_PFT_CLM5)); pft[0, 0, _BE] = 1.0
    gsd = _gsd([2015.0], pft)
    upd = make_step_land_params_updater(gsd, SimpleSEBConfig())
    theta, doy = jnp.full(1, 0.2), jnp.asarray(196.0)
    lp_a, lai_a = upd(theta, doy, jnp.asarray(1850.0))
    lp_b, lai_b = upd(theta, doy, jnp.asarray(2100.0))
    np.testing.assert_allclose(float(lai_a[0]), float(lai_b[0]))
    np.testing.assert_allclose(np.asarray(lp_a.albedo_veg), np.asarray(lp_b.albedo_veg))
    np.testing.assert_allclose(float(lai_a[0]), _LAI_FOREST, rtol=1e-6)


def test_canopy_slopes_are_medlyn_per_pft_under_medlyn():
    """stomatal_model="medlyn": the C3 slope is the per-PFT Medlyn g1 of the
    dominant PFT and the C4 slope the Medlyn C4 value; any other model keeps
    the Ball-Berry constants (9 / 4), which are not Medlyn slopes.  Both the
    traced per-step updater and the static builder agree."""
    from legoesm.land.boundary_data import surface_data_to_land_params
    from legoesm.land.boundary_data._internals import CLM5_MEDLYN_G1, M_C4_MEDLYN
    gsd = _forest_then_grass()
    theta, doy = jnp.full(1, 0.2), jnp.asarray(196.0)
    med = CanopyConfig(max_iters=1, stomatal_model="medlyn")
    cp0, _ = make_step_land_params_updater(gsd, med)(theta, doy, jnp.asarray(1850.0))
    np.testing.assert_allclose(float(cp0.m_C3[0]), CLM5_MEDLYN_G1[_BE])
    np.testing.assert_allclose(float(cp0.m_C4[0]), M_C4_MEDLYN)
    static = surface_data_to_land_params(gsd, med, 196.0, theta, year=1850.0)
    np.testing.assert_allclose(float(static.m_C3[0]), CLM5_MEDLYN_G1[_BE])
    np.testing.assert_allclose(float(static.m_C4[0]), M_C4_MEDLYN)
    bb, _ = make_step_land_params_updater(gsd, CanopyConfig(max_iters=1))(
        theta, doy, jnp.asarray(1850.0))
    np.testing.assert_allclose(float(bb.m_C3[0]), 9.0)
    np.testing.assert_allclose(float(bb.m_C4[0]), 4.0)


def test_medlyn_g1_table_follows_clm_except_c4_crop():
    """The C3 entries are the CLM-ML port's own g1_MED table (single source of
    the CLM5 values); both C4 PFTs carry Lin et al. (2015)'s C4 value, the
    crop_c4 slot departing from CLM's 5.79 (an irrigated C3 crop there)."""
    from legoesm.land.boundary_data._internals import CLM5_MEDLYN_G1, M_C4_MEDLYN
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLpftconMod
    clm = np.asarray(MLpftconMod.MLpftcon.g1_MED)
    assert len(CLM5_MEDLYN_G1) == N_PFT_CLM5
    np.testing.assert_allclose(CLM5_MEDLYN_G1[1:16], clm[1:16])
    i_c4crop = CLM5_PFT_NAMES.index("crop_c4")
    assert CLM5_MEDLYN_G1[i_c4crop] == M_C4_MEDLYN == 1.62
    assert CLM5_MEDLYN_G1[CLM5_PFT_NAMES.index("c4_grass")] == 1.62
    assert float(clm[i_c4crop]) == 5.79
