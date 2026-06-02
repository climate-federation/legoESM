"""Smoke tests for coupler/surface_exchange.py."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.coupler.config import CouplerConfig
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.coupler.surface_exchange import extract_atm_to_surface
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.vertical import create_sigma_coordinate


def _make_state(shape, nlev, *, with_tracers=False):
    spatial = (*shape, nlev)
    u = Field(jnp.full(spatial, 5.0), name="u", dims=("face", "i", "j", "lev"), units="m/s")
    v = Field(jnp.full(spatial, -2.0), name="v", dims=("face", "i", "j", "lev"), units="m/s")
    T = Field(jnp.full(spatial, 280.0), name="T", dims=("face", "i", "j", "lev"), units="K")
    p_s = Field(jnp.full(shape, 1e5), name="p_s", dims=("face", "i", "j"), units="Pa")
    phis = Field(jnp.zeros(shape), name="phis", dims=("face", "i", "j"), units="m^2/s^2")
    tracers = None
    if with_tracers:
        q_v = Field(
            jnp.full(spatial, 0.005), name="q_v",
            dims=("face", "i", "j", "lev"), units="kg/kg",
        )
        tracers = {"q_v": q_v}
    return HydrostaticState(u=u, T=T, p_s=p_s, phis=phis, v=v, tracers=tracers)


def test_extract_atm_to_surface_minimal():
    shape = (6, 4, 4)
    nlev = 6
    state = _make_state(shape, nlev)
    sigma = create_sigma_coordinate(n_levels=nlev)
    config = CouplerConfig()

    out = extract_atm_to_surface(state, sigma, config)

    assert isinstance(out, AtmToSurface)
    assert out.T_lowest.shape == shape
    assert out.q_lowest.shape == shape
    assert jnp.all(out.q_lowest == 0.0)  # no tracers → dry
    assert jnp.all(jnp.isfinite(out.rho_lowest))
    assert float(out.has_radiation) == 0.0
    assert float(out.has_precipitation) == 0.0


def test_extract_atm_to_surface_with_tracers_and_fluxes():
    shape = (6, 4, 4)
    nlev = 6
    state = _make_state(shape, nlev, with_tracers=True)
    sigma = create_sigma_coordinate(n_levels=nlev)
    config = CouplerConfig(co2_ppmv_default=420.0)

    sw_down = jnp.full(shape, 200.0)
    lw_down = jnp.full(shape, 300.0)
    precip_total = jnp.full(shape, 1e-5)

    out = extract_atm_to_surface(
        state, sigma, config,
        sw_down=sw_down, lw_down=lw_down, precip_total=precip_total,
    )

    assert jnp.allclose(out.sw_down, 200.0)
    assert jnp.allclose(out.lw_down, 300.0)
    assert jnp.allclose(out.precip_total, 1e-5)
    assert float(out.has_radiation) == 1.0
    assert float(out.has_precipitation) == 1.0
    assert jnp.allclose(out.q_lowest, 0.005)
    assert float(out.co2_ppmv) == 420.0
