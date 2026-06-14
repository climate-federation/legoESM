"""Unit tests for building per-column land-model inputs from GlobalSurfaceData."""

import numpy as np
import jax.numpy as jnp

from legoesm.land.global_surface_data import GlobalSurfaceData, GlobalSurfaceDataConfig
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
from legoesm.land.surface_data.land_inputs import (
    build_canopy_params,
    build_soil_hydraulics,
    dominant_pft_index,
)


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


def test_soil_hydraulics_percolumn_and_fallback():
    hy = build_soil_hydraulics(_gsd())
    assert hy.retention_curve == "clapp_hornberger"
    assert hy.theta_sat.shape == (2, 1)
    assert np.all(np.isfinite(np.asarray(hy.theta_sat)))     # NaN col fell back
    # col0 from 40% sand; col1 fell back to sandy default -> higher K_sat, lower b
    assert float(hy.K_sat[1, 0]) > float(hy.K_sat[0, 0])
    assert float(hy.b_ch[1, 0]) < float(hy.b_ch[0, 0])
