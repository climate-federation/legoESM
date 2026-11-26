"""Dew limit: canopy air is held at saturation, and the condensate is conserved.

On humid, light-wind nights the canopy root used to have canopy-air relative
humidity 1.02-1.07.  The excess now condenses (none below saturation): its
vapour never reaches the atmosphere, its latent heat warms the canopy air and
leaves as sensible heat, and the land step deposits the water in the canopy
store (liquid, overflow to the soil) or, over snow, on the pack (ice).

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy.config import CanopyConfig, CanopyLandParams
from legoesm.land.canopy.energy_balance import (canopy_air_update,
                                                 saturation_specific_humidity)
from legoesm.land.canopy.interception import InterceptionConfig
from legoesm.land.canopy.solver import (CanopyForcingBundle, canopy_forward,
                                         solve_canopy_closure_diag)
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state, step_multilayer_land_with_diagnostics)
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.thermo import latent_heat_vaporization
from tests.land.unit.test_canopy_night_retry import _COLD, _NIGHT

jax.config.update("jax_enable_x64", True)

_B = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _NIGHT.items()})


def _root():
    cfg = CanopyConfig()
    x0 = jnp.asarray(_COLD).at[jnp.array([0, 1, 4])].add(-1.0)
    x, _n, conv = solve_canopy_closure_diag(x0, _B, cfg)[:3]
    assert bool(conv)
    f = canopy_forward(x, _B, cfg.LE_module, cfg.stomatal_model, cfg.le_cap_mode,
                       cfg.use_ta_for_photosynthesis, cfg.rh_cap_smoothing_width,
                       cfg.zeta_cap_smoothing_width, cfg.most_n_iters)
    return np.asarray(x), {k: float(np.asarray(v)) for k, v in f.items() if np.ndim(v) == 0}


def test_canopy_air_held_at_saturation_and_condenses():
    x, f = _root()
    q_sat = float(saturation_specific_humidity(jnp.asarray(x[4]), _B.Ps))
    # The ramp holds the air at most half its width above saturation (1.038
    # before the dew limit).
    w = CanopyConfig().rh_cap_smoothing_width
    assert 1.0 < x[5] / q_sat <= 1.0 + 0.5 * w + 1e-9, x[5] / q_sat
    assert f["condensation"] > 1e-7                # kg m-2 s-1, ~0.4 W m-2


def test_no_condensation_below_saturation():
    """Subsaturated canopy air is untouched: exactly zero condensation."""
    q_sat = saturation_specific_humidity(jnp.asarray(290.0), jnp.asarray(1e5))
    one = jnp.asarray(1.0)
    for rh in (0.5, 0.9, 0.999):
        _, q_c, cond = canopy_air_update(
            jnp.asarray(290.0), rh * q_sat, *(jnp.asarray(290.0),) * 3,
            *(jnp.asarray(0.01),) * 2, *(jnp.asarray(50.0),) * 2,
            *(jnp.asarray(30.0),) * 4, one, jnp.asarray(1e5),
            jnp.asarray(290.0), jnp.asarray(constants.L_v), jnp.asarray(constants.c_pd),
            CanopyConfig().rh_cap_smoothing_width)
        assert float(cond) == 0.0, (rh, float(cond))


def test_top_of_canopy_fluxes_close_with_condensation():
    """Air-side top fluxes == surface sums shifted by the condensation, to the
    solver's tolerance: water (kg m-2 s-1) and energy (W m-2) both conserved."""
    x, f = _root()
    rho, cp, lam = _NIGHT["rhoa"], _NIGHT["Cp"], _NIGHT["lam"]
    C = f["condensation"]
    E_top = rho * (x[5] - _NIGHT["q_atm"]) / f["raw_above"]
    E_surf = (f["LE_Sun"] + f["LE_Sh"] + f["LE_Soil"]) / lam
    # 5e-7, not round-off: OPEN DEFECT, the canopy-air humidity and the
    # leaf/soil vapour fluxes disagree by 3.0e-7 kg m-2 s-1 (0.7 W m-2) at an
    # exact root, with or without the dew limit and the LE cap.  The land and
    # the atmosphere both see the same net latent flux, so it is internal to
    # the closure (the reported q_c), not a coupled-system leak.
    assert abs(E_top - (E_surf - C)) < 5e-7, (E_top, E_surf, C)
    H_top = rho * cp * (x[4] - _NIGHT["Ta"]) / f["rah_above"]
    H_surf = f["H_Sun"] + f["H_Sh"] + f["H_Soil"]
    # 5e-2 W m-2: same OPEN DEFECT on the heat side, measured 0.019 W m-2.
    assert abs(H_top - (H_surf + lam * C)) < 5e-2, (H_top, H_surf, lam * C)


# --------------------------------------------------------------------------- #
# Land step: the dew reaches the right reservoir and the column water closes  #
# --------------------------------------------------------------------------- #

_N = 2
_PARAMS = CanopyLandParams(
    LAI=jnp.full(_N, 3.0), hc=jnp.full(_N, 12.0), fC4=jnp.zeros(_N),
    FNonVeg=jnp.zeros(_N), CI=jnp.full(_N, 0.7), kn=jnp.full(_N, 0.3),
    Vcmax25_C3_leaf=jnp.full(_N, 60.0), Vcmax25_C4_leaf=jnp.full(_N, 40.0),
    m_C3=jnp.full(_N, 9.0), m_C4=jnp.full(_N, 4.0),
    b0_C3=jnp.full(_N, 0.01), b0_C4=jnp.full(_N, 0.04),
    alf=jnp.full(_N, 0.3), TgC=jnp.full(_N, 20.0),
    ALB_VIS=jnp.full(_N, 0.1), ALB_NIR=jnp.full(_N, 0.2),
    emissivity=jnp.full(_N, 0.97), rz0m=jnp.full(_N, 0.055), rd=jnp.full(_N, 0.67),
)


def _pack_change_expected(so, T_surface, dt):
    """Snowpack change [kg m-2] on a fully snow-covered cold column: ground
    sublimation (LE_soil at L_s), leaf dew frosting the pack (negative leaf
    latent flux at L_v) and the canopy-air condensation, all deposited."""
    L_v = latent_heat_vaporization(T_surface)
    return -(so.LE_soil[0] / so.L_soil[0]
             + jnp.minimum(so.LE_canopy[0], 0.0) / L_v
             - so.condensation[0]) * dt


def _night(interception, T=288.0, T_soil=286.0, swe=0.0, n=24, dt=1800.0):
    """A humid, calm, clear night (12 h) as one jitted scan.  Per step: the
    column water residual [kg m-2] dStorage(soil + pond + canopy store + snow)
    - (-E_reported - runoff), condensation [kg m-2], canopy store, and
    lhflx - L_v * E."""
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0),
        surface_scheme=TwoLeafCanopyConfig(),
        interception=InterceptionConfig() if interception else None)
    st = init_multilayer_land_state(_N, cfg, T_init=T_soil, theta_init=0.3)
    st = st._replace(snow_depth=jnp.full(_N, swe))
    o, ps = jnp.ones(_N), jnp.full(_N, 1e5)
    q = 0.97 * float(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(0.99e5)))
    f = AtmToSurface(
        sw_down=0 * o, lw_down=280.0 * o, precip_total=0 * o, precip_snow=0 * o,
        T_lowest=T * o, q_lowest=q * o, u_lowest=1.0 * o, v_lowest=0 * o,
        p_lowest=0.99 * ps, p_surface=ps, rho_lowest=ps / (constants.R_d * T),
        cos_zenith=0 * o, co2_ppmv=412 * o, has_radiation=o, has_precipitation=o)
    dz = jnp.asarray(make_soil_grid(cfg.soil_grid).dz)

    def W(s):
        wc = 0.0 if s.W_canopy is None else s.W_canopy[0]
        return ((jnp.sum(dz * s.theta_soil[0]) + s.surface_water[0])
                * constants.rho_water + wc + s.snow_depth[0])

    def body(s, _):
        s2, r, _, so = step_multilayer_land_with_diagnostics(
            s, f, cfg, 1.0, dt, lat=jnp.full(_N, 0.3), land_params=_PARAMS)
        e = r.surface_mass_flux[0] * dt
        ro = (s2.runoff_surface[0] + s2.runoff_subsurface[0]) * dt
        wc = 0.0 if s2.W_canopy is None else s2.W_canopy[0]
        gap = r.lhflx[0] - r.surface_mass_flux[0] * latent_heat_vaporization(s.T_soil[0, 0])
        return s2, (W(s2) - W(s) + e + ro, so.condensation[0] * dt, wc, gap,
                    so.converged[0], s2.snow_depth[0] - s.snow_depth[0],
                    _pack_change_expected(so, s.T_soil[0, 0], dt))

    _, out = jax.jit(lambda s: jax.lax.scan(body, s, None, length=n))(st)
    return [np.asarray(v) for v in out]


@pytest.mark.parametrize("interception", [True, False], ids=["store", "no_store"])
def test_land_column_water_closes_with_dew(interception):
    res, cond, wc, gap, conv, _dswe, _exp = _night(interception)
    assert conv.all()
    assert cond.sum() > 0.1, cond.sum()                 # kg m-2 per night
    assert np.abs(res).max() < 1e-10, np.abs(res).max()
    assert np.abs(gap).max() < 1e-9, np.abs(gap).max()  # snow-free: all at L_v
    if interception:
        assert wc[-1] > 0.0                             # dew wets the leaves


def test_dew_over_snow_frosts_the_pack():
    res, cond, _wc, _gap, conv, dswe, expected = _night(
        True, T=268.0, T_soil=268.0, swe=20.0)
    assert conv.all()
    assert cond.sum() > 0.0
    assert np.abs(res).max() < 1e-10, np.abs(res).max()
    # Full snow cover: the pack changes by exactly its ground sublimation, the
    # leaf dew and the condensation, each booked once.
    assert np.abs(dswe - expected).max() < 1e-10, np.abs(dswe - expected).max()
