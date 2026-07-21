"""Rigid-lid island cache must be safe to build inside a user's jit trace.

Regression for the 2026-06-11 differentiability-audit finding: the FIRST
rigid-lid step taken inside ``jax.jit(value_and_grad(loss))`` (state a
closure constant — the standard training-loss shape) lazily built the
island decomposition with omnistaged ``jnp`` ops and cached TRACERS on the
model, raising ``UnexpectedTracerError`` downstream. The fix forces
compile-time evaluation of the build (concrete masks => concrete cache).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def rigid_lid_setup():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(12, 24)
    z_coord = create_ocean_z_star(n_levels=4, H_max=1000.0)
    # land_lat_threshold=70 -> the +-82.5 deg rows are land: the rigid lid
    # needs at least one land mass to pin the streamfunction.
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=1000.0, land_lat_threshold=70.0)
    key = jax.random.PRNGKey(1)
    ku, kv = jax.random.split(key)
    state = state._replace(
        T=state.T.replace(data=state.T.data + 0.5 * jax.random.normal(
            jax.random.PRNGKey(2), state.T.data.shape)),
        u=state.u.replace(data=(state.u.data + 0.02 * jax.random.normal(
            ku, state.u.data.shape)) * state.u_mask.data[..., None]),
        v=state.v.replace(data=(state.v.data + 0.02 * jax.random.normal(
            kv, state.v.data.shape)) * state.v_mask.data[..., None]),
    )
    config = LatLonCGridOceanConfig.from_flat(
        A_h=1000.0, K_h=100.0, A_v=1e-3, K_v=1e-4, bottom_drag_r=1e-3,
        barotropic_solver="rigid_lid", enable_runtime_checks=False)
    return grid, z_coord, state, config


def test_first_use_inside_jit_value_and_grad(rigid_lid_setup):
    """Cold model; the FIRST rigid-lid step happens inside jit+grad."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid, z_coord, state, config = rigid_lid_setup
    model = LatLonCGridOceanModel(grid, z_coord, config)  # cache COLD
    wet3 = state.land_mask.data[..., None] * jnp.ones((1, 1, 4))

    def loss(dT0):
        s = state._replace(T=state.T.replace(data=state.T.data + dT0))
        for _ in range(2):
            s = model._step_impl(s, 600.0)
        return jnp.sum((s.T.data * wet3) ** 2) + 1e4 * jnp.sum(s.u.data ** 2)

    zero = jnp.zeros_like(state.T.data)
    val, g = jax.jit(jax.value_and_grad(loss))(zero)
    assert bool(jnp.isfinite(val))
    assert int(jnp.sum(~jnp.isfinite(g))) == 0
    nz = float(jnp.mean(jnp.abs(g) > 0))
    assert nz > 0.3, f"gradient unexpectedly sparse: {nz:.0%}"
    # cache must hold CONCRETE arrays (the original bug cached tracers,
    # which exploded only on reuse) — second jitted call exercises reuse.
    val2, _ = jax.jit(jax.value_and_grad(loss))(zero + 1e-3)
    assert bool(jnp.isfinite(val2))
    rl = model.rigid_lid_data
    for field in ("inv_diag", "psin", "line_psin", "inv_H_u", "inv_H_v",
                  "solve_mask", "A_vertex", "island_vertex_masks"):
        assert not isinstance(getattr(rl, field), jax.core.Tracer), field


def test_public_step_inside_jit_value_and_grad(rigid_lid_setup):
    """Public-path (eager ``step`` shim) twin of the _step_impl AD test: the
    FIRST rigid-lid step happens inside ``jax.jit(value_and_grad(loss))`` via
    ``model.step`` with the masks as CLOSURE CONSTANTS (the standard
    training-loss shape).  The shim's ``prime_step_caches`` builds the island
    cache as CONCRETE arrays (ensure_compile_time_eval) — not cached tracers —
    so the gradient is finite and dense and reuse is safe.  Codex round-1 asked
    for public-path AD coverage (existing test uses ``_step_impl`` directly).
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid, z_coord, state, config = rigid_lid_setup
    model = LatLonCGridOceanModel(grid, z_coord, config)  # cache COLD
    wet3 = state.land_mask.data[..., None] * jnp.ones((1, 1, 4))

    def loss(dT0):
        s = state._replace(T=state.T.replace(data=state.T.data + dT0))
        for _ in range(2):
            s = model.step(s, 600.0)   # eager shim → prime_step_caches warms RL
        return jnp.sum((s.T.data * wet3) ** 2) + 1e4 * jnp.sum(s.u.data ** 2)

    zero = jnp.zeros_like(state.T.data)
    val, g = jax.jit(jax.value_and_grad(loss))(zero)
    assert bool(jnp.isfinite(val))
    assert int(jnp.sum(~jnp.isfinite(g))) == 0
    nz = float(jnp.mean(jnp.abs(g) > 0))
    assert nz > 0.3, f"gradient unexpectedly sparse: {nz:.0%}"
    rl = model.rigid_lid_data
    for field in ("inv_diag", "psin", "line_psin", "island_vertex_masks"):
        assert not isinstance(getattr(rl, field), jax.core.Tracer), field


def test_rigid_lid_cache_rejects_changed_topology(rigid_lid_setup):
    """Keyed cache (codex round-1): a reused model instance must NOT silently
    serve stale island data for a different bathymetry/land mask — it raises,
    matching the _vertex_mask_src guard."""
    import numpy as np
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid, z_coord, state, config = rigid_lid_setup
    model = LatLonCGridOceanModel(grid, z_coord, config)
    model._ensure_rigid_lid_data(state)            # build the cache

    # Same-value different-identity mask is adopted (no raise).
    same = state._replace(
        land_mask=state.land_mask.replace(
            data=jnp.asarray(np.array(state.land_mask.data))),
        H_bathy=state.H_bathy.replace(
            data=jnp.asarray(np.array(state.H_bathy.data))))
    model._ensure_rigid_lid_data(same)             # no raise

    # A genuinely different land mask must raise (stale-topology guard).
    new_mask = state.land_mask.data.at[1, :].set(0.0)   # flood a new land row
    changed = state._replace(
        land_mask=state.land_mask.replace(data=new_mask))
    with pytest.raises(ValueError, match="bathymetry/land mask changed"):
        model._ensure_rigid_lid_data(changed)


def test_eager_step_auto_warms_rigid_lid_cache(rigid_lid_setup):
    """The eager ``step`` shim must warm the rigid-lid island cache from the
    CONCRETE input state before crossing its jit boundary — so a plain
    ``model.step(state, dt)`` loop (run_dino / model.integrate / step_checked)
    works WITHOUT a manual ``_ensure_rigid_lid_data`` pre-build.

    Regression for the DINO barotropic-solver audit failure ('rigid-lid island
    decomposition must be built from CONCRETE bathymetry/masks'): the shim
    warmed only the vertex-mask cache, so the first rigid-lid step reached the
    TRACED mid-step state inside ``_step_jitted`` and raised.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid, z_coord, state, config = rigid_lid_setup
    model = LatLonCGridOceanModel(grid, z_coord, config)  # cache COLD
    assert model.rigid_lid_data is None

    # No manual warm, no _step_impl — the eager shim must pre-build the cache.
    nxt = model.step(state, 600.0)

    assert model.rigid_lid_data is not None
    rl = model.rigid_lid_data
    assert rl.nisle >= 1
    for field in ("inv_diag", "psin", "line_psin", "island_vertex_masks"):
        assert not isinstance(getattr(rl, field), jax.core.Tracer), field
    assert bool(jnp.all(jnp.isfinite(nxt.u.data)))
    assert bool(jnp.all(jnp.isfinite(nxt.v.data)))
    # Second step reuses the warmed cache (idempotent — same object).
    assert model._ensure_rigid_lid_data(nxt) is rl
    nxt2 = model.step(nxt, 600.0)
    assert bool(jnp.all(jnp.isfinite(nxt2.u.data)))


def test_warm_then_jit_matches_cold_jit(rigid_lid_setup):
    """Pre-warming the cache eagerly gives bit-identical results."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid, z_coord, state, config = rigid_lid_setup

    def run(model):
        def loss(dT0):
            s = state._replace(T=state.T.replace(data=state.T.data + dT0))
            s = model._step_impl(s, 600.0)
            return jnp.sum(s.T.data ** 2)
        return jax.jit(jax.value_and_grad(loss))(jnp.zeros_like(state.T.data))

    cold = LatLonCGridOceanModel(grid, z_coord, config)
    v_cold, g_cold = run(cold)

    warm = LatLonCGridOceanModel(grid, z_coord, config)
    warm._ensure_rigid_lid_data(state)  # eager pre-build
    v_warm, g_warm = run(warm)

    assert float(v_cold) == float(v_warm)
    assert bool(jnp.all(g_cold == g_warm))
