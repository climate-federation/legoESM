"""Vertex-mask build-once cache (halo-census lever: the land-mask-derived
vertex mask cost an N-S exchange x2 sites per traced step; census 8474554).

Pins:
1. step() fills the cache eagerly; the traced tendencies then use the
   constant (zero in-trace compute_vertex_mask calls).
2. Threaded constant == in-graph fallback bit-exactly (same function,
   same inputs).
3. A DIFFERENT land mask on the same model REFUSES loudly (the compiled
   step baked the old mask as a closure constant — a silent refresh
   cannot reach it; codex round-2).
4. prime_step_caches is tracer-safe (no-op under trace) and idempotent.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


def _build(n_lat=12, n_lon=24, nlev=3, land=True):
    grid = create_latlon_grid(n_lat, n_lon)
    z = create_ocean_z_star(n_levels=nlev)
    model = LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat())
    mask_override = None
    if land:
        m = np.ones((n_lat, n_lon))
        m[4:6, 8:12] = 0.0
        mask_override = jnp.asarray(m)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, land_mask_override=mask_override)
    return model, state, grid


def test_step_fills_cache_and_matches_fallback():
    model, state, grid = _build()
    assert model._vertex_mask is None
    _ = model.step(state, 300.0)
    assert model._vertex_mask is not None
    ref = compute_vertex_mask(state.land_mask.data, grid=model.grid)
    np.testing.assert_array_equal(
        np.asarray(model._vertex_mask), np.asarray(ref))


def test_prime_idempotent_and_tracer_safe():
    model, state, _ = _build()
    model.prime_step_caches(state)
    first = model._vertex_mask
    assert first is not None
    model.prime_step_caches(state)          # idempotent
    assert model._vertex_mask is first
    # tracer-safe: priming inside a trace is a no-op, not an error
    fresh_model, fresh_state, _ = _build()

    @jax.jit
    def traced_prime(st):
        fresh_model.prime_step_caches(st)
        return st.T.data

    traced_prime(fresh_state)
    assert fresh_model._vertex_mask is None


def test_mask_swap_refuses_loudly():
    model, state, _ = _build()
    model.prime_step_caches(state)
    m2 = np.asarray(state.land_mask.data).copy()
    m2[0, 0] = 0.0                            # genuinely different mask
    state2 = state._replace(
        land_mask=state.land_mask.replace(data=jnp.asarray(m2)))
    with pytest.raises(ValueError, match="land mask changed"):
        model.prime_step_caches(state2)


def test_same_value_new_identity_adopted():
    model, state, _ = _build()
    model.prime_step_caches(state)
    same = jnp.asarray(np.asarray(state.land_mask.data).copy())
    state2 = state._replace(land_mask=state.land_mask.replace(data=same))
    model.prime_step_caches(state2)           # equal values: no refusal
    assert model._vertex_mask_src is same
