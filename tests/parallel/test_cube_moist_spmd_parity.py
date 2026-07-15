"""Cube-moist SPMD wiring parity: the sharded step threads the REAL Kessler
warm-rain physics through the REAL FV3 C-D cubed-sphere dycore identically to
the serial model.

This is the single-process gate for the ``--cs-spmd --physics moist`` bench
path (``run_cpu_mpi_scaling._build_cubed_sphere_spmd``): on a 1-device mesh the
sharded step must equal ``model.step_with_physics`` step-for-step, proving the
moist tracers (q_v/q_c/q_r) flow through ``make_sharded_step`` and the column
physics runs inside the jitted sharded program.  TRUE multi-device halo parity
(np>1) is the separate srun-based receipt; this isolates the physics wiring.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.parallel.mesh import create_device_mesh, shard_pytree
from legoesm.parallel.sharded_dynamics import make_sharded_step
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    CDGridPrimitiveEquationConfig,
    hydrostatic_to_fv3,
    create_cubed_sphere_cdgrid,
)
from legoesm.atmosphere.forcing.idealized.kessler_forcing import make_kessler_forcing_cube
from tests.test_cases.baroclinic_wave import baroclinic_wave_init

_RES = 12
_NLEV = 6
_DT = 200.0
_NSTEPS = 3


def _build():
    grid = create_cubed_sphere(_RES)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sigma = create_sigma_coordinate(_NLEV)
    config = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=True, fix_mass=True,
        anchor_mass_to_initial=True, zero_mean_ps_tendency=True,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    state = hydrostatic_to_fv3(
        baroclinic_wave_init(grid, sigma, perturbed=True, moist=True), cdgrid)
    return grid, model, state


def test_moist_state_has_tracers():
    """hydrostatic_to_fv3(moist init) carries q_v/q_c/q_r into the FV3 state."""
    _, _, state = _build()
    tracers = getattr(state, "tracers", None)
    assert tracers is not None, "moist FV3 state must carry a tracers dict"
    for q in ("q_v", "q_c", "q_r"):
        assert q in tracers, f"missing tracer {q}"
        assert jnp.all(jnp.isfinite(tracers[q].data))
    # q_v must be non-trivial (a dry init would leave these zero).
    assert float(jnp.max(tracers["q_v"].data)) > 0.0


def test_sharded_moist_equals_serial():
    """1-device sharded step with Kessler == serial step_with_physics, leaf for
    leaf, over several steps (the cs-spmd moist wiring is faithful).

    INDEPENDENT model instances for serial vs sharded: a model can lazily
    populate ``_target_mass`` on its first step, so sharing one model would let
    the serial warm-up seed the sharded run and mask the fresh-model behavior
    that ``_build_cubed_sphere_spmd`` actually exercises (codex)."""
    phys = make_kessler_forcing_cube(_DT)

    # Serial reference (its own model).
    _, model_ref, state_ref = _build()
    s_ref = state_ref
    for _ in range(_NSTEPS):
        s_ref = model_ref.step_with_physics(s_ref, _DT, phys)

    # 1-device sharded path with a FRESH model (exactly what
    # _build_cubed_sphere_spmd builds: a never-stepped model).
    _, model_sh, state_sh = _build()
    mesh = create_device_mesh(n_devices=1)
    sharded_step = make_sharded_step(model_sh, mesh, n=_RES, nlev=_NLEV)
    s_shard = shard_pytree(state_sh, mesh)
    for _ in range(_NSTEPS):
        s_shard = sharded_step(s_shard, _DT, physics_fn=phys)

    ref_leaves = jax.tree_util.tree_leaves(s_ref)
    shard_leaves = jax.tree_util.tree_leaves(s_shard)
    assert len(ref_leaves) == len(shard_leaves)
    max_diff = 0.0
    for a, b in zip(ref_leaves, shard_leaves):
        a = jnp.asarray(a); b = jnp.asarray(b)
        if jnp.issubdtype(a.dtype, jnp.floating):
            assert jnp.all(jnp.isfinite(a)) and jnp.all(jnp.isfinite(b))
            max_diff = max(max_diff, float(jnp.max(jnp.abs(a - b))))
    # Same device, same math -> near-exact (only jit/sharding reassociation).
    assert max_diff < 1e-9, f"sharded-moist diverged from serial: {max_diff:.2e}"


def test_dry_path_unaffected():
    """physics_level='none' equivalent: sharded dry step still matches serial
    (guard against the moist wiring perturbing the dry path)."""
    # Dry init (no tracers) — the cs-spmd dry path. INDEPENDENT models so the
    # serial step can't seed the sharded model's lazy target mass (codex).
    def _dry():
        grid, model, _ = _build()
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sigma = create_sigma_coordinate(_NLEV)
        state = hydrostatic_to_fv3(
            baroclinic_wave_init(grid, sigma, perturbed=True, moist=False),
            cdgrid)
        return model, state
    model_ref, state = _dry()
    s_ref = model_ref.step(state, _DT)
    model_sh, state_sh = _dry()
    mesh = create_device_mesh(n_devices=1)
    sharded_step = make_sharded_step(model_sh, mesh, n=_RES, nlev=_NLEV)
    s_shard = sharded_step(shard_pytree(state_sh, mesh), _DT)
    md = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(s_ref),
                    jax.tree_util.tree_leaves(s_shard)):
        a = jnp.asarray(a); b = jnp.asarray(b)
        if jnp.issubdtype(a.dtype, jnp.floating):
            md = max(md, float(jnp.max(jnp.abs(a - b))))
    assert md < 1e-9, f"sharded dry diverged: {md:.2e}"
