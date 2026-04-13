"""Integration tests for step_canopy_land — full canopy energy balance step.

Checks:
- TileResponse field shapes and finiteness
- Daytime: LE > 0, GPP > 0 (co2_flux < 0, surface is a sink)
- Nighttime: GPP = 0 (no SW -> no photosynthesis)
- JIT compilation works and result is numerically stable
- Dispatch via component_factory.create_land_component returns step_canopy_land
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.driver.component_factory import create_land_component
from legoesm.land.canopy import (
    CanopyConfig,
    CanopyLandConfig,
    init_canopy_land_state,
    step_canopy_land,
)
from legoesm.land.config import MultiLayerLandConfig


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


def _make_config(max_iters: int = 20) -> CanopyLandConfig:
    return CanopyLandConfig(
        multilayer=MultiLayerLandConfig(),
        canopy=CanopyConfig(max_iters=max_iters),
    )


def test_daytime_fluxes_and_gpp():
    ncol = 2
    cfg = _make_config()
    state = init_canopy_land_state(ncol, cfg, T_init=290.0)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)

    _, response, cstate = step_canopy_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)

    # Shape
    assert response.T_surface.shape == (ncol,)
    # Finite
    for f in (response.T_surface, response.lhflx, response.shflx,
              response.albedo, response.co2_flux, response.lw_up, response.z0):
        assert jnp.all(jnp.isfinite(f)), f"non-finite in {f}"
    # Daytime LE > 0
    assert float(response.lhflx[0]) > 0.0
    # GPP > 0 -> co2_flux < 0 (uptake)
    assert float(response.co2_flux[0]) < 0.0
    # T_surface within reasonable band
    assert 270.0 < float(response.T_surface[0]) < 330.0
    # Albedo sensible
    assert 0.0 <= float(response.albedo[0]) <= 1.0
    # carbon_state is None in Stage 1
    assert cstate is None


def test_nighttime_zero_gpp():
    ncol = 1
    cfg = _make_config()
    state = init_canopy_land_state(ncol, cfg, T_init=290.0)
    # night: sw_down = 0, cos_zenith near zero
    forcing = _make_forcing(ncol, sw_down=0.0, cos_zenith=0.0)

    _, response, _ = step_canopy_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)

    # No photosynthesis at night
    assert float(response.co2_flux[0]) == 0.0


def test_jit_compiles():
    ncol = 2
    cfg = _make_config()
    state = init_canopy_land_state(ncol, cfg, T_init=290.0)
    forcing = _make_forcing(ncol, 700.0, 0.8)

    @jax.jit
    def _step(s, f):
        return step_canopy_land(s, f, cfg, 1.0, 1800.0,
                                lat=jnp.zeros(ncol), doy=180.0)

    ns1, r1, _ = _step(state, forcing)
    # Second call reuses compiled kernel — just check it runs
    ns2, r2, _ = _step(ns1, forcing)
    assert jnp.all(jnp.isfinite(r1.T_surface))
    assert jnp.all(jnp.isfinite(r2.T_surface))


def test_dispatch_via_component_factory():
    """create_land_component must return step_canopy_land for CanopyLandConfig."""
    from legoesm.driver.config import ExperimentConfig

    cfg = _make_config()
    # ExperimentConfig is not needed internally for dispatch — pass None-like
    # object via a minimal shim. The factory only logs using `config`.
    class _Stub:
        pass
    step_fn = create_land_component(_Stub(), grid=None, land_config=cfg)
    assert step_fn is step_canopy_land


# ---------------------------------------------------------------------------
# Pluggable stomatal model (Ball-Berry default, Medlyn alternative)
# ---------------------------------------------------------------------------

def test_medlyn_stomatal_model_runs():
    """Canopy must produce finite, physically reasonable fluxes when the
    Medlyn stomatal conductance model is selected in place of Ball-Berry."""
    ncol = 2
    # Medlyn slope g1_med ≈ 4 for C3, g0 = 0.01 — use per-column m/b0 accordingly.
    cfg = CanopyLandConfig(
        multilayer=MultiLayerLandConfig(),
        canopy=CanopyConfig(stomatal_model="medlyn", max_iters=30),
    )
    state = init_canopy_land_state(ncol, cfg, T_init=290.0)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)

    # Override per-column Ball-Berry params to be Medlyn-appropriate: the
    # canopy_land fallback provides m_C3=9 which is right for Ball-Berry
    # but too large for Medlyn — use land_params to set m=4.0.
    from legoesm.land.canopy.config import CanopyLandParams
    import jax.numpy as jnp
    params = CanopyLandParams(
        LAI=jnp.full(ncol, 3.0),
        hc=jnp.full(ncol, 5.0),
        fC4=jnp.zeros(ncol),
        FNonVeg=jnp.zeros(ncol),
        CI=jnp.full(ncol, 0.75),
        kn=jnp.full(ncol, 0.3),
        Vcmax25_C3_leaf=jnp.full(ncol, 60.0),
        Vcmax25_C4_leaf=jnp.full(ncol, 40.0),
        m_C3=jnp.full(ncol, 4.0),   # Medlyn g1 [kPa^0.5]
        m_C4=jnp.full(ncol, 1.6),   # Medlyn g1 for C4
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

    _, response, _ = step_canopy_land(
        state, forcing, cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0, land_params=params)

    for name, f in (("T_surface", response.T_surface),
                    ("lhflx", response.lhflx),
                    ("shflx", response.shflx),
                    ("co2_flux", response.co2_flux)):
        assert jnp.all(jnp.isfinite(f)), f"non-finite {name} under Medlyn"

    assert float(response.lhflx[0]) > 0.0, "Medlyn LE ≤ 0 at midday"
    assert float(response.co2_flux[0]) < 0.0, "Medlyn GPP ≤ 0 at midday"
    assert 265.0 < float(response.T_surface[0]) < 325.0


def test_medlyn_differs_from_ball_berry():
    """Medlyn and Ball-Berry with the same numerical slopes must produce
    different fluxes (their functional forms differ) — guards against
    silent fallback to a single hard-coded model.
    """
    ncol = 2
    state_bb = init_canopy_land_state(
        CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                         canopy=CanopyConfig()),
        ncol=ncol, T_init=290.0) if False else None
    # Build both configs.
    cfg_bb = CanopyLandConfig(
        multilayer=MultiLayerLandConfig(),
        canopy=CanopyConfig(stomatal_model="ball_berry", max_iters=30))
    cfg_med = CanopyLandConfig(
        multilayer=MultiLayerLandConfig(),
        canopy=CanopyConfig(stomatal_model="medlyn", max_iters=30))
    state = init_canopy_land_state(ncol, cfg_bb, T_init=290.0)
    forcing = _make_forcing(ncol, sw_down=700.0, cos_zenith=0.8)

    _, resp_bb, _ = step_canopy_land(
        state, forcing, cfg_bb, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)
    _, resp_med, _ = step_canopy_land(
        state, forcing, cfg_med, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=180.0)

    # Fluxes must differ (functional form difference).
    assert not jnp.allclose(resp_bb.lhflx, resp_med.lhflx, atol=1e-4)
    assert not jnp.allclose(resp_bb.co2_flux, resp_med.co2_flux, atol=1e-10)
