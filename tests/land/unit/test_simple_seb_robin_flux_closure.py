"""SimpleSEB semi-implicit ground-heat correction reaches the atmosphere (#2542).

SimpleSEB hands the soil-thermal solve a linearised surface conductance
lambda = -dG/dT_sfc (Robin BC), so the column receives
G - lambda*(T_top_new - T_top_old), not the explicit G.  The correction is a
change of the surface's turbulent + longwave loss; the atmosphere must see it,
or the land-atmosphere exchange does not close:

    Rn - SH_atm - LE_atm - (dS_soil/dt - Q_geo) = 0      (bulk snow)
    Rn - SH_atm - LE_atm - snow_ground_heat_applied = 0  (layered snow)

with Rn = sw_net + lw_net the scheme's net radiation.  Before the fix the
residual equals the Robin flux (measured 28-344 W/m2 on these columns).  float64.
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
from legoesm.land.soil_thermal import (
    compute_apparent_heat_capacity, compute_heat_capacity)
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


def _run(cfg, forcing, st=None):
    snow_free = st is None
    if snow_free:
        st = ml.init_multilayer_land_state(2, cfg, T_init=295.0)
    new, r, _c, so = jax.jit(lambda s: ml.step_multilayer_land_with_diagnostics(
        s, forcing, cfg, 1.0, _DT, lat=jnp.full(2, 0.3)))(st)
    assert int(so.n_held) == 0
    assert so.surface_conductance is not None
    # Start-of-step linearisation estimate of the correction (exact only for a
    # single solve on a snow-free column); printed next to the residual.
    robin = np.asarray(so.surface_conductance
                       * (new.T_soil[:, 0] - st.T_soil[:, 0]))
    if snow_free:
        assert np.all(np.abs(robin) > _ROBIN_MIN), robin   # non-vacuity
    return st, new, r, so, robin


def _residual(label, r, so, heat_gain, robin, new):
    """Rn - SH_atm - LE_atm - (heat the column received from the surface) [W m-2].
    Sign: Rn, heat_gain positive INTO the land; SH, LE positive UP."""
    terms = dict(sw_net=so.sw_net, lw_net=so.lw_net, SH_atm=r.shflx, LE_atm=r.lhflx,
                 G_explicit=so.G_soil, heat_gain=heat_gain, lam_dT=robin)
    resid = np.asarray(so.sw_net + so.lw_net - r.shflx - r.lhflx) - np.asarray(heat_gain)
    print(label, {k: float(np.asarray(v)[0]) for k, v in terms.items()},
          "resid", resid, "resid-lam_dT", resid - robin,
          "state_sum", repr(float(jnp.sum(new.T_soil) + jnp.sum(new.theta_soil)
                                  + jnp.sum(new.snow_depth))))
    for v in terms.values():        # every term is a plausible surface flux
        assert np.all(np.abs(np.asarray(v)) < 1500.0), terms
    return resid


@pytest.mark.parametrize("freeze_thaw", [False, True])
@pytest.mark.parametrize("sw", [800.0, 0.0])
def test_bulk_snow_column_energy_closes_against_atmosphere_fluxes(freeze_thaw, sw):
    """Day (warming skin) and night (cooling skin); freeze/thaw on runs the final
    solve in sub-steps.  The 295 K column holds no ice (asserted), so its heat
    storage is the sensible C*dz*dT at the water the final solve used."""
    cfg = _cfg()
    cfg = cfg._replace(thermal=cfg.thermal._replace(enable_freeze_thaw=freeze_thaw))
    st, new, r, so, robin = _run(cfg, _forcing(2, sw=sw))
    assert float(jnp.max(st.snow_depth)) == 0.0 and float(jnp.max(new.snow_depth)) == 0.0
    grid = make_soil_grid(cfg.soil_grid)
    C = compute_heat_capacity(new.theta_soil, cfg.hydraulics, cfg.thermal)
    for T in (st.T_soil, new.T_soil):   # no latent part in the apparent capacity
        np.testing.assert_allclose(compute_apparent_heat_capacity(
            T, new.theta_soil, cfg.hydraulics, cfg.thermal), C, rtol=1e-15)
    dS = np.asarray(jnp.sum(C * grid.dz * (new.T_soil - st.T_soil), axis=-1)) / _DT
    resid = _residual(f"bulk sw={sw} ft={freeze_thaw}", r, so,
                      dS - cfg.thermal.Q_geothermal, robin, new)
    np.testing.assert_allclose(resid, 0.0, rtol=0, atol=_ATOL)


@pytest.mark.parametrize("swe", [0.0, 15.0, 200.0])
def test_layered_snow_column_energy_closes_against_atmosphere_fluxes(swe):
    """Snow-free, partial cover and full cover.  ``snow_ground_heat_applied`` is the flux the
    pack + soil received from the surface; that it equals the column's enthalpy
    change is asserted by test_layered_snowpack::
    test_land_step_closes_pack_plus_soil_energy."""
    cfg = _cfg(snow_scheme="layered")
    if swe > 0.0:
        st = ml.init_multilayer_land_state(2, cfg, T_init=270.0)
        st = ml.seed_snow_layers(st._replace(snow_depth=jnp.full(2, swe)), cfg)
        st, new, r, so, robin = _run(cfg, _forcing(2, T=265.0, q=0.0015, sw=400.0), st)
    else:
        st, new, r, so, robin = _run(cfg, _forcing(2))
    f_snow = float(ml.snow_cover_fraction(st.snow_depth, cfg.land_albedo)[0])
    assert (f_snow == 0.0) if swe == 0.0 else (0.1 < f_snow < 0.9) if swe == 15.0 else (
        f_snow > 0.9), f_snow
    resid = _residual(f"layered swe={swe} f_snow={f_snow:.3f}", r, so,
                      so.snow_ground_heat_applied, robin, new)
    np.testing.assert_allclose(resid, 0.0, rtol=0, atol=_ATOL)
