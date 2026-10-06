"""Snow latent energy becomes water at the latent heat the surface scheme CHARGED.

The snow stream's vapour mass is the scheme's own E = snow_latent / L_charged.
The ice really leaves the pack at L_s(T_surface), so lhflx_actual books
L_s * E, and evap_excess_energy = lhflx - lhflx_actual turns negative by
(L_s - L_charged) * E.  That term is drawn from the ground, once.  Below:
(a) E for a vaporization-charging scheme (two-leaf) at 260 K; (b) layered pack
+ soil energy closes with that negative excess; (c) CLM-ML below freezing
already charges sublimation (identity); (d) the bulk pack closes too.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.boundary_data.gap_fill import bare_canopy_params
from legoesm.land.multilayer_land import (
    MultiLayerLandConfig, init_multilayer_land_state, seed_snow_layers,
    step_multilayer_land_with_diagnostics)
from legoesm.land.snow_column import SnowColumnState, column_enthalpy
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.soil_thermal import compute_heat_capacity
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.thermo import latent_heat_sublimation, latent_heat_vaporization

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"), reason="float64 budgets; JAX_ENABLE_X64=1")

_T = 260.0


def _forcing(n, T_air=_T + 3.0, q=0.0003):
    o = jnp.ones(n)
    return AtmToSurface(
        sw_down=150.0 * o, lw_down=230.0 * o, precip_total=0.0 * o,
        precip_snow=0.0 * o, T_lowest=T_air * o, q_lowest=q * o,
        u_lowest=5.0 * o, v_lowest=0.0 * o, p_lowest=95000.0 * o,
        p_surface=1.0e5 * o, rho_lowest=1.0e5 / (constants.R_d * T_air) * o,
        cos_zenith=0.4 * o, co2_ppmv=400.0 * o, has_radiation=o,
        has_precipitation=o)


def _two_leaf_step(snow_scheme, n_steps=1):
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(),
                               snow_scheme=snow_scheme)
    st = init_multilayer_land_state(1, cfg, T_init=_T)
    st = st._replace(snow_depth=jnp.full(1, 60.0))       # deep pack: no cap binds
    if snow_scheme == "layered":
        st = seed_snow_layers(st, cfg)
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([0.3]))
    step = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, _forcing(1), cfg, 1.0, 1800.0, lat=jnp.full(1, 0.9), land_params=lp))
    out = []
    for _ in range(n_steps):
        st_new, resp, _, sfc = step(st)
        out.append((st, st_new, resp, sfc))
        st = st_new
    return cfg, out


def _pack(st):
    return SnowColumnState(st.snow_ice_layers, st.snow_liq_layers, st.snow_T_layers,
                           st.snow_rho_layers)


def test_two_leaf_snow_mass_is_the_schemes_E_at_260K():
    """(a) The two-leaf solve charges L_v(T_surface) to every latent stream, so
    the land's vapour mass is lhflx / L_v(260 K) -- about 12 % more than the
    former snow_latent / L_s conversion -- and the latent heat reported to the
    atmosphere is the sublimation cost L_s * E of that mass."""
    _, [(s0, _, resp, sfc)] = _two_leaf_step("bulk")
    T_s = s0.T_soil[0, 0]
    lh = float(sfc.lhflx[0])
    assert lh > 5.0, lh                                    # a real sublimation flux
    E = float(resp.surface_mass_flux[0])
    np.testing.assert_allclose(E, lh / float(latent_heat_vaporization(T_s)), rtol=1e-9)
    # Reported latent heat: the snow stream's mass sublimates at L_s, the soil /
    # plant stream (positive canopy latent) evaporates at the charged L_v.
    L_v, L_s = float(latent_heat_vaporization(T_s)), float(latent_heat_sublimation(T_s))
    le_soil, le_can = float(sfc.LE_soil[0]), lh - float(sfc.LE_soil[0])
    snow_lat = le_soil + min(le_can, 0.0)
    assert snow_lat > 5.0, snow_lat
    np.testing.assert_allclose(float(resp.lhflx[0]),
                               snow_lat / L_v * L_s + (lh - snow_lat), rtol=1e-9)
    # The former conversion gave the snow stream snow_lat / L_s: ~11 % less.
    assert (snow_lat / L_v) / (snow_lat / L_s) - 1.0 > 0.10


def test_simple_seb_over_snow_charges_sublimation_so_no_excess():
    """SimpleSEB charges L_s(T_surface) to snow, so its conversion is the
    identity: the latent heat reported equals the scheme's charge (no excess)
    on an uncapped deep pack.  Guards the driver's per-scheme snow charge."""
    from legoesm.land.surface_scheme import SimpleSEBConfig
    cfg = MultiLayerLandConfig(surface_scheme=SimpleSEBConfig())
    st = init_multilayer_land_state(1, cfg, T_init=_T)._replace(snow_depth=jnp.full(1, 60.0))
    _, resp, _, sfc = step_multilayer_land_with_diagnostics(
        st, _forcing(1), cfg, 1.0, 1800.0, lat=jnp.full(1, 0.9))
    assert float(sfc.lhflx[0]) > 5.0
    np.testing.assert_allclose(float(resp.lhflx[0]), float(sfc.lhflx[0]), rtol=1e-9)
    np.testing.assert_allclose(
        float(resp.surface_mass_flux[0]),
        float(sfc.lhflx[0]) / float(latent_heat_sublimation(st.T_soil[0, 0])), rtol=1e-9)


def test_layered_pack_plus_soil_energy_closes_with_negative_excess():
    """(b, d-layered) Every step: the column receives the scheme's ground flux
    plus the latent demand minus the latent heat reported (the excess, NEGATIVE
    here by (L_s - L_v) * E), and pack enthalpy + soil energy change by exactly
    that (in - out - dStorage = 0)."""
    cfg, out = _two_leaf_step("layered", n_steps=6)
    grid = make_soil_grid(cfg.soil_grid)
    hc, tc = cfg.hydraulics, cfg.thermal
    dt = 1800.0
    worst = 0.0
    for s, s_new, resp, sfc in out:
        assert int(sfc.n_held) == 0
        excess = sfc.lhflx - resp.lhflx
        np.testing.assert_allclose(sfc.snow_ground_heat_applied, sfc.G_soil + excess,
                                   rtol=0, atol=1e-6)
        Cg = compute_heat_capacity(s.theta_soil, hc, tc) * grid.dz
        dE = (column_enthalpy(_pack(s_new)) - column_enthalpy(_pack(s))
              + jnp.sum(Cg * (s_new.T_soil - s.T_soil), -1))
        src = dt * (sfc.snow_ground_heat_applied + tc.Q_geothermal) + sfc.snow_advected_heat
        np.testing.assert_allclose(dE, src, rtol=1e-9, atol=1e-3)
        worst = min(worst, float(excess[0]))
    assert worst < -1.0, worst                             # the extra cost was booked


def test_bulk_soil_energy_closes_with_negative_excess():
    """(d-bulk) No pack enthalpy on the bulk scheme: at 260 K with no melt, no
    precipitation and freeze/thaw off, the soil gains exactly
    dt * (G + lhflx_demand - lhflx_reported + geothermal)."""
    cfg, out = _two_leaf_step("bulk", n_steps=4)
    assert not cfg.thermal.enable_freeze_thaw
    grid = make_soil_grid(cfg.soil_grid)
    hc, tc = cfg.hydraulics, cfg.thermal
    dt = 1800.0
    for s, s_new, resp, sfc in out:
        excess = sfc.lhflx - resp.lhflx
        # Exactly the snow share's extra sublimation cost (deep pack, moist soil:
        # no cap binds): -snow_lat * (L_s / L_v - 1) at the pre-step skin T.
        T_s = s.T_soil[0, 0]
        L_v, L_s = float(latent_heat_vaporization(T_s)), float(latent_heat_sublimation(T_s))
        le_can = float(sfc.lhflx[0] - sfc.LE_soil[0])
        snow_lat = float(sfc.LE_soil[0]) + min(le_can, 0.0)
        assert snow_lat > 0.0
        np.testing.assert_allclose(float(excess[0]), -snow_lat * (L_s / L_v - 1.0),
                                   rtol=1e-6)
        # the bulk solve uses the post-hydrology moisture
        Cg = compute_heat_capacity(s_new.theta_soil, hc, tc) * grid.dz
        dE = jnp.sum(Cg * (s_new.T_soil - s.T_soil), -1)
        np.testing.assert_allclose(
            dE, dt * (sfc.G_soil + excess + tc.Q_geothermal), rtol=1e-9, atol=1e-3)


@pytest.mark.parametrize("T_air,L_charged", [
    (262.0, constants.L_s),    # (c) freezing air: hsub, the identity
    (275.0, constants.L_v),    # thawing air over the pack: hvap, ~13 % more mass
], ids=["below_freezing_hsub", "above_freezing_hvap"])
def test_clm_ml_snow_mass_is_its_charge(monkeypatch, T_air, L_charged):
    """CLM-ML charges ALL its water at LatVap(tref): hsub (== constants.L_s)
    under freezing air, hvap (== constants.L_v) above.  Total vapour mass =
    lhflx / L_charged; the snow share's ice is reported at L_s(T_surface)
    (below freezing that differs from the charge only by the Kirchhoff slope,
    ~3e-4 at 265 K)."""
    import legoesm.land.canopy.clm_ml_interface as clm
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.surface_scheme.base import SurfaceFluxOutput

    def _stub(*, T_soil_top, forcing, canopy_state, **_):
        o = jnp.ones_like(T_soil_top)
        return SurfaceFluxOutput(
            shflx=10.0 * o, lhflx=40.0 * o, tau_x=0.0 * o, tau_y=0.0 * o,
            sw_net=100.0 * o, lw_net=-40.0 * o, lw_up=300.0 * o, G_soil=10.0 * o,
            T_surface=T_soil_top, q_surface=0.001 * o, albedo=0.7 * o,
            emissivity=0.97 * o, z0=0.01 * o, T_canopy_air=T_soil_top,
            stomatal_ratio=o, LE_soil=25.0 * o, LE_canopy=15.0 * o), canopy_state

    monkeypatch.setattr(clm, "compute_clm_ml_canopy_fluxes", _stub)
    cfg = MultiLayerLandConfig(surface_scheme=CLMMLCanopyConfig())
    s0 = init_multilayer_land_state(1, cfg, T_init=265.0, theta_init=0.30)
    s0 = s0._replace(snow_depth=jnp.full(1, 60.0))
    _, resp, _, out = step_multilayer_land_with_diagnostics(
        s0, _forcing(1, T_air=T_air), cfg, 1.0, 1800.0, lat=jnp.array([0.9]),
        carbon_state=None, doy=20.0, land_params=None)
    np.testing.assert_allclose(float(resp.surface_mass_flux[0]), 40.0 / L_charged,
                               rtol=1e-9)
    # The 25 W/m2 ground share is the snow stream; its ice costs L_s(T_surface).
    E_snow = 25.0 / L_charged
    L_sT = float(latent_heat_sublimation(s0.T_soil[0, 0]))
    np.testing.assert_allclose(float(resp.lhflx[0]),
                               E_snow * L_sT + 15.0, rtol=1e-9)
    if L_charged == constants.L_v:      # the former / L_s conversion: ~13 % less
        assert (25.0 / constants.L_v) / (25.0 / L_sT) - 1.0 > 0.10


def test_bulk_capped_pack_splits_unmet_demand_to_sh_and_sublimation_cost_to_ground():
    """(e) Thin bulk pack under the two-leaf canopy: the sublimation demand
    exceeds the pack, so the excess X has BOTH parts.  The unmet demand
    (demand - pack) * L_v leaves as sensible heat; the sublimation cost of the
    ice that did leave, pack/dt * (L_v - L_s) < 0, stays in the ground."""
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig())
    s0 = init_multilayer_land_state(1, cfg, T_init=_T)
    swe0 = 1.0e-4                                         # [kg m-2], cap binds
    s0 = s0._replace(snow_depth=jnp.full(1, swe0))
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([0.3]))
    dt = 1800.0
    s1, resp, _, sfc = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, _forcing(1), cfg, 1.0, dt, lat=jnp.full(1, 0.9), land_params=lp))(s0)
    T_s = s0.T_soil[0, 0]
    L_v, L_s = float(latent_heat_vaporization(T_s)), float(latent_heat_sublimation(T_s))
    assert float(s1.snow_depth[0]) < 1e-12                 # the whole pack left
    cost = swe0 / dt * (L_v - L_s)                        # < 0: cools the ground
    X = float(sfc.lhflx[0] - resp.lhflx[0])
    unmet = X - cost
    assert unmet > 1.0, (X, cost)                         # non-vacuity: cap binds
    np.testing.assert_allclose(float(resp.shflx[0] - sfc.shflx[0]), unmet,
                               rtol=0, atol=1e-9)
    grid = make_soil_grid(cfg.soil_grid)
    Cg = compute_heat_capacity(s1.theta_soil, cfg.hydraulics, cfg.thermal) * grid.dz
    dE = jnp.sum(Cg * (s1.T_soil - s0.T_soil), -1)
    np.testing.assert_allclose(
        dE, dt * (sfc.G_soil + cost + cfg.thermal.Q_geothermal), rtol=1e-9, atol=1e-3)
