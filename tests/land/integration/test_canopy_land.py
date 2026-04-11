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
