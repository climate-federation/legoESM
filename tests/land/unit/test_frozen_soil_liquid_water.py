"""Frozen soil water is not available to evaporation or roots (#1815).

With soil freeze/thaw on, evaporation, root water stress and the atmospheric
land-tile humidity see only the unfrozen liquid share of the soil water.
Richards, infiltration and the heat budget keep total water.  Freeze/thaw off
is unchanged (the helper returns its input).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land import carbon_diagnostics, multilayer_land
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    land_tile_beta_soil,
    liquid_soil_water,
    step_multilayer_land,
)
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import psi_from_theta
from legoesm.land.soil_thermal import SoilThermalConfig, liquid_water_content
from legoesm.land.surface_scheme import SimpleSEBConfig

jax.config.update("jax_enable_x64", True)

_ON = SoilThermalConfig(enable_freeze_thaw=True)
_GRID = SoilGridConfig(n_layers=8, total_depth=3.0)
_COLD = constants.T_freeze - 8.0
_WARM = constants.T_freeze + 8.0
_THETA = 0.30
_DT = 1800.0


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()


class _Stop(Exception):
    pass


def _cfg(scheme=None, thermal=_ON):
    kw = dict(soil_grid=_GRID, thermal=thermal)
    if scheme is not None:
        kw["surface_scheme"] = scheme
    return MultiLayerLandConfig(**kw)


def _forcing(ncol=1, T_air=constants.T_freeze + 5.0, q_air=0.001, wind=4.0):
    o = jnp.ones(ncol)
    p_s = 1.0e5 * o
    return AtmToSurface(
        sw_down=400.0 * o, lw_down=300.0 * o, precip_total=0.0 * o,
        precip_snow=0.0 * o, T_lowest=T_air * o, q_lowest=q_air * o,
        u_lowest=wind * o, v_lowest=0.0 * o, p_lowest=0.99 * p_s, p_surface=p_s,
        rho_lowest=p_s / (constants.R_d * T_air), cos_zenith=0.6 * o,
        co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)


def _state(cfg, T_soil):
    st = init_multilayer_land_state(1, cfg, T_init=_WARM, theta_init=_THETA)
    return st._replace(T_soil=jnp.broadcast_to(
        jnp.asarray(T_soil, dtype=st.T_soil.dtype), st.T_soil.shape))


def test_off_returns_total_water_unchanged():
    cfg = _cfg(thermal=SoilThermalConfig(enable_freeze_thaw=False))
    theta = jnp.full((2, 8), _THETA)
    assert liquid_soil_water(theta, jnp.full_like(theta, _COLD), cfg) is theta


def test_on_is_the_freezing_curve_liquid():
    cfg = _cfg()
    theta = jnp.full((3, 8), _THETA)
    T = jnp.array([_COLD, constants.T_freeze, _WARM])[:, None] * jnp.ones((1, 8))
    liq = liquid_soil_water(theta, T, cfg)
    np.testing.assert_array_equal(liq, liquid_water_content(T, theta, _ON)[0])
    assert float(liq[0, 0]) < 0.1 * _THETA            # frozen: residual film only
    assert float(liq[0, 0]) < float(liq[1, 0]) < float(liq[2, 0])
    assert float(liq[2, 0]) == pytest.approx(_THETA, rel=1e-6)


def test_land_tile_beta_frozen_is_dry():
    cfg = _cfg()
    n = make_soil_grid(_GRID).n_layers
    theta = jnp.full((2, n), _THETA)
    T = jnp.array([_COLD, _WARM])[:, None] * jnp.ones((1, n))
    beta = np.asarray(land_tile_beta_soil(theta, T, cfg))
    assert beta[0] == pytest.approx(cfg.beta_min, abs=1e-9)
    assert beta[1] > cfg.beta_min + 0.5


def test_land_tile_beta_gradient_finite_through_freezing():
    cfg = _cfg()
    n = make_soil_grid(_GRID).n_layers
    theta = jnp.full((1, n), 0.40)
    T = jnp.full((1, n), constants.T_freeze)
    liq = liquid_soil_water(theta, T, cfg)
    assert cfg.theta_wp < float(liq[0, 0]) < cfg.theta_fc   # stress is live here

    def f(theta, T):
        return jnp.sum(land_tile_beta_soil(theta, T, cfg))

    g_th, g_T = jax.grad(f, argnums=(0, 1))(theta, T)
    assert bool(jnp.all(jnp.isfinite(g_th))) and bool(jnp.all(jnp.isfinite(g_T)))
    assert float(jnp.sum(g_T)) > 0.0                   # thawing wets the root zone
    assert jnp.allclose(f(theta, T), jax.jit(f)(theta, T), rtol=1e-12)


def test_step_moisture_stress_reads_liquid_before_and_after(monkeypatch):
    """Both root-zone stress evaluations of a step (start-of-step, which drives
    transpiration and its root weights, and end-of-step, which feeds the
    reported humidity and carbon) are given liquid water."""
    cfg = _cfg(SimpleSEBConfig())
    seen = []
    real = multilayer_land.root_zone_beta_soil

    def spy(theta, *a, **k):
        seen.append(theta)
        return real(theta, *a, **k)

    monkeypatch.setattr(multilayer_land, "root_zone_beta_soil", spy)
    st = _state(cfg, constants.T_freeze - 0.5)
    new, _, _ = step_multilayer_land(st, _forcing(), cfg, 1.0, _DT,
                                     lat=jnp.full(1, 1.0))
    assert len(seen) == 2
    np.testing.assert_allclose(
        seen[0], liquid_soil_water(st.theta_soil, st.T_soil, cfg), rtol=1e-12)
    np.testing.assert_allclose(
        seen[1], liquid_soil_water(new.theta_soil, new.T_soil, cfg), rtol=1e-9)
    assert float(jnp.max(seen[0])) < 0.9 * _THETA       # the probe is partly frozen


@pytest.mark.parametrize("T_soil, series", [
    (_COLD, True),                         # Kelvin humidity of the liquid ~0
    (constants.T_freeze - 0.3, False),     # Kelvin ~1; top-layer saturation decides
])
def test_two_leaf_surface_inputs_read_liquid(monkeypatch, T_soil, series):
    cfg = _cfg()._replace(soil_evap_series_resistance=series)
    st = _state(cfg, T_soil)
    got = {}

    def stop(**kw):
        got.update(kw)
        raise _Stop

    monkeypatch.setattr(multilayer_land, "compute_two_leaf_canopy_fluxes", stop)
    with pytest.raises(_Stop):
        step_multilayer_land(st, _forcing(), cfg, 1.0, _DT, lat=jnp.full(1, 1.0))
    liq = liquid_soil_water(st.theta_soil, st.T_soil, cfg)
    hyd = cfg.hydraulics
    psi_liq = psi_from_theta(liq, hyd)
    h_r = jnp.exp(jnp.minimum(
        psi_liq[:, 0] * constants.g / (constants.R_v * st.T_soil[:, 0]), 0.0))
    s_top = jnp.clip((liq - hyd.theta_r) / (hyd.theta_sat - hyd.theta_r),
                     1e-6, 1.0)[:, 0]
    expect = h_r if series else h_r * s_top ** cfg.soil_evap_resistance_exp
    np.testing.assert_allclose(got["soil_surface_relsat"],
                               jnp.clip(liq / hyd.theta_sat, 1e-6, 1.0)[:, 0],
                               rtol=1e-12)
    np.testing.assert_allclose(got["w_frac_soil_evap"], expect, rtol=1e-9,
                               atol=0.0 if series else 1e-300)
    if not series:
        assert float(expect[0]) > 1e-3                  # not a zero comparison


def test_clm_ml_receives_a_liquid_theta_psi_pair(monkeypatch):
    from legoesm.land.canopy import clm_ml_interface
    from legoesm.land.canopy.config import CLMMLCanopyConfig

    cfg = _cfg(CLMMLCanopyConfig())
    st = _state(cfg, _COLD)
    got = {}

    def stop(**kw):
        got.update(kw)
        raise _Stop

    monkeypatch.setattr(clm_ml_interface, "compute_clm_ml_canopy_fluxes", stop)
    with pytest.raises(_Stop):
        step_multilayer_land(st, _forcing(), cfg, 1.0, _DT, lat=jnp.full(1, 1.0))
    liq = liquid_soil_water(st.theta_soil, st.T_soil, cfg)
    np.testing.assert_allclose(got["theta_soil"], liq, rtol=1e-12)
    np.testing.assert_allclose(got["psi_soil"], psi_from_theta(liq, cfg.hydraulics),
                               rtol=1e-12)


def test_carbon_diagnostics_read_liquid(monkeypatch):
    cfg = _cfg()
    st = _state(cfg, _COLD)
    got = []

    def stop(theta, *a, **k):
        got.append(theta)
        raise _Stop

    monkeypatch.setattr(carbon_diagnostics, "root_zone_beta_soil", stop)
    with pytest.raises(_Stop):
        carbon_diagnostics.reconstruct_carbon_diagnostics(
            st, None, None, cfg, None, cfg.theta_wp, cfg.theta_fc, cfg.beta_min,
            jnp.zeros(1), 180, _DT)
    np.testing.assert_allclose(
        got[0], liquid_soil_water(st.theta_soil, st.T_soil, cfg), rtol=1e-12)


def _water(st, cfg):
    dz = make_soil_grid(cfg.soil_grid).dz
    w = jnp.sum(st.theta_soil * dz[None, :], axis=-1) * constants.rho_water
    pond = 0.0 if st.surface_water is None else st.surface_water * constants.rho_water
    return w + pond + st.snow_depth


@pytest.mark.parametrize("T_soil, expect_dry", [(_COLD, True), (_WARM, False)])
def test_frozen_column_gives_no_soil_water_and_budgets_close(T_soil, expect_dry):
    """A fully frozen, snow-free column supplies (almost) no evaporation or
    transpiration; the same column thawed does.  The column water budget
    closes on both."""
    cfg = _cfg(SimpleSEBConfig())
    st = _state(cfg, T_soil)
    new, resp, _ = step_multilayer_land(st, _forcing(), cfg, 1.0, _DT,
                                        lat=jnp.full(1, 1.0))
    E = float(resp.surface_mass_flux[0])                      # kg m-2 s-1, up
    if expect_dry:
        assert abs(E) < 1.0e-8, E
    else:
        assert E > 1.0e-6, E
    runoff = float(new.runoff_surface[0] + new.runoff_subsurface[0])
    resid = float(_water(new, cfg)[0] - _water(st, cfg)[0]) + (E + runoff) * _DT
    assert abs(resid) < 1.0e-6, resid                          # kg m-2


def _seb_exp0():
    return _cfg(SimpleSEBConfig())._replace(soil_evap_resistance_exp=0.0)


def test_simple_seb_kelvin_humidity_reads_liquid():
    """Top layer frozen below residual liquid, deeper layers thawed and wet:
    with the saturation throttle off (exponent 0) only the liquid Kelvin
    humidity of the top layer stops evaporation; the thawed control evaporates."""
    cfg = _seb_exp0()
    n = make_soil_grid(_GRID).n_layers
    T_top_frozen = jnp.full((1, n), _WARM).at[:, 0].set(_COLD)
    E = []
    for T in (T_top_frozen, jnp.full((1, n), _WARM)):
        _, resp, _ = step_multilayer_land(_state(cfg, T), _forcing(), cfg, 1.0,
                                          _DT, lat=jnp.full(1, 1.0))
        E.append(float(resp.surface_mass_flux[0]))
    assert abs(E[0]) < 1.0e-8, E
    assert E[1] > 1.0e-6, E


def test_soil_evaporation_capped_by_liquid_water():
    """Top layer just below freezing (liquid a little above residual, Kelvin
    humidity ~1), deeper layers frozen: the column's liquid supply caps the
    step's evaporation, which binds under a strong drying demand."""
    cfg = _seb_exp0()
    n = make_soil_grid(_GRID).n_layers
    dt = 7200.0
    T = jnp.full((1, n), _COLD).at[:, 0].set(constants.T_freeze - 0.5)
    st = _state(cfg, T)
    dz = make_soil_grid(_GRID).dz
    liq = liquid_soil_water(st.theta_soil, st.T_soil, cfg)
    cap = float(jnp.sum(jnp.maximum(liq - cfg.hydraulics.theta_r, 0.0) * dz)
                * constants.rho_water)                                # kg m-2
    _, resp, _ = step_multilayer_land(
        st, _forcing(T_air=constants.T_freeze + 2.0, q_air=1.0e-4, wind=15.0),
        cfg, 1.0, dt, lat=jnp.full(1, 1.0))
    E_dt = float(resp.surface_mass_flux[0]) * dt
    assert 0.0 < cap < 1.0, cap
    assert E_dt <= cap * (1.0 + 1e-6) + 1e-9, (E_dt, cap)
    assert E_dt > 0.5 * cap, (E_dt, cap)                  # the cap is what binds
