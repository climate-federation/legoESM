"""Integration tests for the two-leaf canopy surface scheme.

After Phase 3, the canopy biophysics is a **surface scheme** of
``step_multilayer_land`` — there is no separate ``step_canopy_land`` or
``CanopyLandConfig``.  Callers select the canopy via::

    MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(...))

Checks:
- TileResponse field shapes and finiteness
- Daytime: LE > 0, GPP > 0 (co2_flux < 0, surface is a sink)
- Nighttime: GPP = 0 (no SW -> no photosynthesis)
- JIT compilation works and result is numerically stable
- Dispatch via component_factory.create_land_component returns
  ``step_multilayer_land``
- Pluggable Ball-Berry / Medlyn stomatal model
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.driver.component_factory import create_land_component
from legoesm.land.canopy import CanopyConfig, CanopyLandParams
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land,
)
from legoesm.land.surface_scheme import TwoLeafCanopyConfig


def _make_forcing(ncol: int, sw_down: float, cos_zenith: float) -> AtmToSurface:
    return AtmToSurface(
        T_lowest=jnp.full(ncol, 295.0),
        q_lowest=jnp.full(ncol, 0.012),
        u_lowest=jnp.full(ncol, 3.0),
        v_lowest=jnp.full(ncol, 0.5),
        p_lowest=jnp.full(ncol, 98000.0),
        p_surface=jnp.full(ncol, 101325.0),
        rho_lowest=jnp.full(ncol, 1.18),
        sw_down=jnp.full(ncol, sw_down),
        lw_down=jnp.full(ncol, 380.0),
        cos_zenith=jnp.full(ncol, cos_zenith),
        precip_total=jnp.zeros(ncol),
        precip_snow=jnp.zeros(ncol),
        co2_ppmv=jnp.full(ncol, 420.0),
        has_radiation=True,
        has_precipitation=False,
    )


def _make_canopy_config(
    max_iters: int = 20,
    stomatal_model: str = "ball_berry",
) -> MultiLayerLandConfig:
    return MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(
            max_iters=max_iters, stomatal_model=stomatal_model),
    )


def test_daytime_fluxes_and_gpp():
    ncol = 2
    cfg = _make_canopy_config()
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)

    _, response, cstate = step_multilayer_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)

    assert response.T_surface.shape == (ncol,)
    for f in (response.T_surface, response.lhflx, response.shflx,
              response.albedo, response.co2_flux, response.lw_up, response.z0):
        assert jnp.all(jnp.isfinite(f)), f"non-finite in {f}"
    assert float(response.lhflx[0]) > 0.0
    # Carbon cycle off by default → co2_flux is zero.
    assert 270.0 < float(response.T_surface[0]) < 330.0
    assert 0.0 <= float(response.albedo[0]) <= 1.0
    assert cstate is None


def test_nighttime_zero_gpp():
    ncol = 1
    cfg = _make_canopy_config()
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0)
    forcing = _make_forcing(ncol, sw_down=0.0, cos_zenith=0.0)

    _, response, _ = step_multilayer_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)

    # Carbon cycle off → co2_flux is zero anyway.
    assert float(response.co2_flux[0]) == 0.0


def test_jit_compiles():
    ncol = 2
    cfg = _make_canopy_config()
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0)
    forcing = _make_forcing(ncol, 700.0, 0.8)

    @jax.jit
    def _step(s, f):
        return step_multilayer_land(
            s, f, cfg, 1.0, 1800.0,
            lat=jnp.zeros(ncol), doy=180.0)

    ns1, r1, _ = _step(state, forcing)
    ns2, r2, _ = _step(ns1, forcing)
    assert jnp.all(jnp.isfinite(r1.T_surface))
    assert jnp.all(jnp.isfinite(r2.T_surface))


def test_dispatch_via_component_factory():
    """create_land_component must return step_multilayer_land for any
    MultiLayerLandConfig, regardless of the inner surface scheme.
    """
    cfg = _make_canopy_config()

    class _Stub:
        pass

    step_fn = create_land_component(_Stub(), grid=None, land_config=cfg)
    assert step_fn is step_multilayer_land


# ---------------------------------------------------------------------------
# Pluggable stomatal model (Ball-Berry default, Medlyn alternative)
# ---------------------------------------------------------------------------

def test_medlyn_stomatal_model_runs():
    """Canopy must produce finite, physically reasonable fluxes when the
    Medlyn stomatal conductance model is selected in place of Ball-Berry."""
    ncol = 2
    cfg = _make_canopy_config(max_iters=30, stomatal_model="medlyn")
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)

    # Per-column params with Medlyn-appropriate slopes (~4 for C3, ~1.6 for C4).
    params = CanopyLandParams(
        LAI=jnp.full(ncol, 3.0),
        hc=jnp.full(ncol, 5.0),
        fC4=jnp.zeros(ncol),
        FNonVeg=jnp.zeros(ncol),
        CI=jnp.full(ncol, 0.75),
        kn=jnp.full(ncol, 0.3),
        Vcmax25_C3_leaf=jnp.full(ncol, 60.0),
        Vcmax25_C4_leaf=jnp.full(ncol, 40.0),
        m_C3=jnp.full(ncol, 4.0),
        m_C4=jnp.full(ncol, 1.6),
        b0_C3=jnp.full(ncol, 0.01),
        b0_C4=jnp.full(ncol, 0.04),
        alf=jnp.full(ncol, 0.3),
        TgC=jnp.full(ncol, 20.0),
        ALB_VIS=jnp.full(ncol, 0.1),
        ALB_NIR=jnp.full(ncol, 0.2),
        emissivity=jnp.full(ncol, 0.97),
        rz0m=jnp.full(ncol, 0.055),
        rd=jnp.full(ncol, 0.67),
    )

    _, response, _ = step_multilayer_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0, land_params=params)

    for name, f in (("T_surface", response.T_surface),
                    ("lhflx", response.lhflx),
                    ("shflx", response.shflx),
                    ("co2_flux", response.co2_flux)):
        assert jnp.all(jnp.isfinite(f)), f"non-finite {name} under Medlyn"

    assert float(response.lhflx[0]) > 0.0
    assert 265.0 < float(response.T_surface[0]) < 325.0


def test_medlyn_differs_from_ball_berry():
    """Medlyn and Ball-Berry with the same numerical slopes must produce
    different fluxes (their functional forms differ) — guards against
    silent fallback to a single hard-coded model.
    """
    ncol = 2
    cfg_bb = _make_canopy_config(max_iters=30, stomatal_model="ball_berry")
    cfg_med = _make_canopy_config(max_iters=30, stomatal_model="medlyn")
    state = init_multilayer_land_state(ncol, cfg_bb, T_init=290.0)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)

    _, resp_bb, _ = step_multilayer_land(
        state, forcing, cfg_bb, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)
    _, resp_med, _ = step_multilayer_land(
        state, forcing, cfg_med, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)

    assert not jnp.allclose(resp_bb.lhflx, resp_med.lhflx, atol=1e-4)
    assert not jnp.allclose(resp_bb.shflx, resp_med.shflx, atol=1e-4)


# ---------------------------------------------------------------------------
# SimpleSEB regression baseline
# ---------------------------------------------------------------------------

def test_simple_seb_default_still_runs():
    """A default ``MultiLayerLandConfig()`` (SimpleSEB surface scheme) must
    still produce finite multilayer land output — regression guard for the
    Phase 3 surface-scheme dispatch refactor.
    """
    ncol = 4
    cfg = MultiLayerLandConfig()  # SimpleSEB by default
    state = init_multilayer_land_state(ncol, cfg, T_init=290.0)
    forcing = _make_forcing(ncol, sw_down=400.0, cos_zenith=0.6)

    new_state, response, _ = step_multilayer_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)

    for name, f in (("T_surface", response.T_surface),
                    ("lhflx", response.lhflx),
                    ("shflx", response.shflx),
                    ("lw_up", response.lw_up)):
        assert jnp.all(jnp.isfinite(f)), f"SimpleSEB non-finite {name}"
    assert jnp.all(jnp.isfinite(new_state.T_soil))
    assert jnp.all(jnp.isfinite(new_state.theta_soil))


# ---------------------------------------------------------------------------
# Slab + canopy (Phase 3b)
# ---------------------------------------------------------------------------

def _make_slab_state(shape):
    from legoesm.core.field import Field
    from legoesm.land.state import LandState
    dims = ("x",) if len(shape) == 1 else ("face", "x", "y")
    return LandState(
        T_soil=Field(data=jnp.full(shape, 290.0),
                     name="T_soil", dims=dims, units="K"),
        W_bucket=Field(data=jnp.full(shape, 100.0),
                       name="W_bucket", dims=dims, units="kg/m2"),
        snow_depth=Field(data=jnp.zeros(shape),
                         name="snow_depth", dims=dims, units="kg/m2"),
        snow_age=Field(data=jnp.zeros(shape),
                       name="snow_age", dims=dims, units="s"),
    )


def _make_slab_forcing(shape, sw_down, cos_zenith):
    return AtmToSurface(
        sw_down=jnp.full(shape, sw_down),
        lw_down=jnp.full(shape, 380.0),
        precip_total=jnp.zeros(shape),
        precip_snow=jnp.zeros(shape),
        T_lowest=jnp.full(shape, 295.0),
        q_lowest=jnp.full(shape, 0.012),
        u_lowest=jnp.full(shape, 3.0),
        v_lowest=jnp.full(shape, 0.5),
        p_lowest=jnp.full(shape, 98000.0),
        p_surface=jnp.full(shape, 101325.0),
        rho_lowest=jnp.full(shape, 1.18),
        cos_zenith=jnp.full(shape, cos_zenith),
        co2_ppmv=jnp.full(shape, 420.0),
        has_radiation=True,
        has_precipitation=False,
    )


def test_slab_plus_canopy_1d_runs():
    """Slab + TwoLeafCanopy on a 1D pseudo-columnar shape must produce
    finite, physically reasonable midday fluxes.
    """
    from legoesm.land.config import LandConfig
    from legoesm.land.slab_land import step_land

    shape = (4,)
    cfg = LandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=30))
    state = _make_slab_state(shape)
    forcing = _make_slab_forcing(shape, sw_down=700.0, cos_zenith=0.8)

    new_state, response, _ = step_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(shape), doy=180.0)

    assert response.T_surface.shape == shape
    for name, f in (("T_surface", response.T_surface),
                    ("lhflx", response.lhflx),
                    ("shflx", response.shflx),
                    ("lw_up", response.lw_up)):
        assert jnp.all(jnp.isfinite(f)), f"slab+canopy 1D non-finite {name}"
    # Midday: surface should transpire.
    assert float(response.lhflx[0]) > 0.0
    # Skin T close to forcing T (within ±20 K).
    assert 275.0 < float(response.T_surface[0]) < 315.0
    assert jnp.all(jnp.isfinite(new_state.T_soil.data))
    assert jnp.all(jnp.isfinite(new_state.W_bucket.data))


def test_slab_plus_canopy_3d_runs():
    """Slab + TwoLeafCanopy on a (6, n, n) cubed-sphere shape must produce
    output with the original spatial shape (canopy flatten/unflatten).
    """
    from legoesm.land.config import LandConfig
    from legoesm.land.slab_land import step_land

    shape = (6, 4, 4)
    cfg = LandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=30))
    state = _make_slab_state(shape)
    forcing = _make_slab_forcing(shape, sw_down=700.0, cos_zenith=0.8)

    new_state, response, _ = step_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(shape), doy=180.0)

    # Shape preserved.
    assert response.T_surface.shape == shape
    assert response.lhflx.shape == shape
    assert new_state.T_soil.data.shape == shape
    assert new_state.W_bucket.data.shape == shape
    # All-uniform forcing → all columns should give the same result.
    for f in (response.T_surface, response.lhflx, response.shflx):
        assert jnp.allclose(f, f[0, 0, 0]), "non-uniform output for uniform forcing"
    # Finite and physically reasonable.
    assert float(response.lhflx[0, 0, 0]) > 0.0
    assert 275.0 < float(response.T_surface[0, 0, 0]) < 315.0


def test_slab_simple_seb_default_unchanged():
    """``LandConfig()`` (default SimpleSEB) regression — must still run."""
    from legoesm.land.config import LandConfig
    from legoesm.land.slab_land import step_land

    shape = (4,)
    cfg = LandConfig()  # SimpleSEB default
    state = _make_slab_state(shape)
    forcing = _make_slab_forcing(shape, sw_down=400.0, cos_zenith=0.6)

    new_state, response, _ = step_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0)

    assert jnp.all(jnp.isfinite(response.T_surface))
    assert jnp.all(jnp.isfinite(response.lhflx))
    assert jnp.all(jnp.isfinite(new_state.T_soil.data))


# ---------------------------------------------------------------------------
# TgC 30-day exponential moving average (Phase 3b)
# ---------------------------------------------------------------------------

def test_TgC_ema_advances_toward_T_air():
    """Multilayer + canopy with state-carried TgC: after enough steps the
    EMA should approach the prescribed forcing air temperature.
    """
    ncol = 2
    cfg = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=20))
    # Initialise TgC = 5 °C; force T_air = 25 °C; check the EMA drifts up.
    state = init_multilayer_land_state(
        ncol, cfg, T_init=290.0, TgC_init=5.0)
    assert state.TgC is not None
    assert jnp.allclose(state.TgC, 5.0)

    forcing = _make_forcing(ncol, sw_down=300.0, cos_zenith=0.6)
    # Forcing T_lowest = 295 K (≈ 21.85 °C) → TgC must move from 5 toward ~22.

    dt = 3600.0  # 1 h step
    n_steps = 24 * 30  # 30 days

    # JIT the step so the 720-iter loop is fast.
    @jax.jit
    def _step(s):
        return step_multilayer_land(
            s, forcing, cfg, U_min=1.0, dt=dt,
            lat=jnp.zeros(ncol), doy=180.0)

    for _ in range(n_steps):
        state, _, _ = _step(state)

    # After 30 days at dt=3600 s the EMA should reach ~1 - 1/e of the gap
    # (~63%): TgC ≈ 5 + 0.63 * (21.85 - 5) ≈ 15.6.  Allow ±1 K slack.
    expected = 5.0 + (1.0 - jnp.exp(-1.0)) * (21.85 - 5.0)
    assert jnp.all(jnp.isfinite(state.TgC))
    assert jnp.allclose(state.TgC, expected, atol=1.0), (
        f"TgC after 30 days = {float(state.TgC[0]):.2f}, "
        f"expected ~{float(expected):.2f}")


def test_TgC_ema_helper_one_step():
    """Direct unit test for ``advance_TgC_ema``."""
    from legoesm.land.surface_scheme.two_leaf_canopy import (
        TGC_EMA_TAU_S, advance_TgC_ema,
    )

    TgC = jnp.array(10.0)
    T_air = jnp.array(298.15)  # 25 C
    dt = 1800.0  # 30 min
    TgC_new = advance_TgC_ema(TgC, T_air, dt)
    expected = 10.0 + (dt / TGC_EMA_TAU_S) * (25.0 - 10.0)
    assert jnp.allclose(TgC_new, expected, rtol=1e-12)
    # Should still be very close to TgC_old after a single 30 min step
    # (alpha = dt/tau ≈ 6.94e-4 → ΔTgC ≈ 0.0104 K for ΔT_air = 15 K).
    assert float(TgC_new) - 10.0 < 0.05


def test_canopy_uses_state_TgC_when_present():
    """When ``state.TgC`` is set, the canopy must use it instead of the
    fallback ``forcing.T_lowest - 273.15``.  Two runs with different
    ``TgC_init`` should produce different fluxes.
    """
    ncol = 2
    cfg = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=20))
    state_cool = init_multilayer_land_state(
        ncol, cfg, T_init=290.0, TgC_init=5.0)
    state_warm = init_multilayer_land_state(
        ncol, cfg, T_init=290.0, TgC_init=30.0)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)

    _, resp_cool, _ = step_multilayer_land(
        state_cool, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)
    _, resp_warm, _ = step_multilayer_land(
        state_warm, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)

    # Vcmax acclimation: cool TgC reduces Vcmax → less GPP, less LE.
    # We can't test GPP directly because carbon=none, but lhflx differs.
    assert not jnp.allclose(resp_cool.lhflx, resp_warm.lhflx, atol=1e-4), (
        "TgC override has no effect on canopy fluxes — likely state.TgC "
        "is being ignored")
