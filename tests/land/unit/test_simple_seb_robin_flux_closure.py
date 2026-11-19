"""SimpleSEB semi-implicit ground-heat correction reaches the atmosphere (#2542).

SimpleSEB hands the soil-thermal solve a linearised surface conductance
lambda = -dG/dT_sfc (Robin BC), so the column receives
G - lambda*(T_top_new - T_top_old), not the explicit G.  The correction is a
change of the surface's turbulent + longwave loss; the atmosphere must see it,
or the land-atmosphere exchange does not close:

    Rn - SH_atm - LE_atm - (dS_soil/dt - Q_geo) = 0      (bulk snow)
    Rn - SH_atm - LE_atm - snow_ground_heat_applied = 0  (layered snow)

with Rn = sw_net + lw_net the scheme's net radiation.  Before the fix the
residual equals the Robin flux (tens of W/m2 on these columns).  float64.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land import multilayer_land as ml
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_thermal import compute_heat_capacity
from legoesm.land.surface_scheme import SimpleSEBConfig

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="float64 energy gates; run with JAX_ENABLE_X64=1")

_DT = 1800.0
_ROBIN_MIN = 1.0   # [W m-2] non-vacuity: the step's Robin correction must be real
_ATOL = 1e-8       # [W m-2] round-off on fluxes of O(100-1000) W m-2 in float64


def _forcing(n, T=300.0, q=0.012, sw=800.0):
    o = jnp.ones(n)
    ps = 1e5 * o
    return AtmToSurface(
        sw_down=sw * o, lw_down=380.0 * o, precip_total=0 * o, precip_snow=0 * o,
        T_lowest=T * o, q_lowest=q * o, u_lowest=5.0 * o, v_lowest=0 * o,
        p_lowest=0.99 * ps, p_surface=ps, rho_lowest=ps / (constants.R_d * T),
        cos_zenith=(0.8 if sw > 0 else 0.0) * o, co2_ppmv=412 * o, has_radiation=o,
        has_precipitation=o)


def _cfg(**kw):
    return MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0),
                                surface_scheme=SimpleSEBConfig(), **kw)


def _run(cfg, forcing):
    st = ml.init_multilayer_land_state(2, cfg, T_init=295.0)
    new, r, _c, so = jax.jit(lambda s: ml.step_multilayer_land_with_diagnostics(
        s, forcing, cfg, 1.0, _DT, lat=jnp.full(2, 0.3)))(st)
    assert int(so.n_held) == 0
    assert so.surface_conductance is not None
    robin = np.asarray(so.surface_conductance
                       * (new.T_soil[:, 0] - st.T_soil[:, 0]))
    assert np.all(np.abs(robin) > _ROBIN_MIN), robin       # non-vacuity
    return st, new, r, so


@pytest.mark.parametrize("freeze_thaw", [False, True])
@pytest.mark.parametrize("sw", [800.0, 0.0])
def test_bulk_snow_column_energy_closes_against_atmosphere_fluxes(freeze_thaw, sw):
    """Day (warming skin) and night (cooling skin); freeze/thaw on runs the final
    solve in sub-steps.  Warm column: the apparent heat capacity is sensible."""
    cfg = _cfg()
    cfg = cfg._replace(thermal=cfg.thermal._replace(enable_freeze_thaw=freeze_thaw))
    st, new, r, so = _run(cfg, _forcing(2, sw=sw))
    grid = make_soil_grid(cfg.soil_grid)
    C = compute_heat_capacity(new.theta, cfg.hydraulics, cfg.thermal) * grid.dz
    dS = np.asarray(jnp.sum(C * (new.T_soil - st.T_soil), axis=-1)) / _DT
    Rn = np.asarray(so.sw_net + so.lw_net)
    resid = Rn - np.asarray(r.shflx + r.lhflx) - (dS - cfg.thermal.Q_geothermal)
    np.testing.assert_allclose(resid, 0.0, rtol=0, atol=_ATOL)


def test_layered_snow_column_energy_closes_against_atmosphere_fluxes():
    cfg = _cfg(snow_scheme="layered")
    st, new, r, so = _run(cfg, _forcing(2))
    Rn = np.asarray(so.sw_net + so.lw_net)
    resid = Rn - np.asarray(r.shflx + r.lhflx) - np.asarray(so.snow_ground_heat_applied)
    np.testing.assert_allclose(resid, 0.0, rtol=0, atol=_ATOL)
