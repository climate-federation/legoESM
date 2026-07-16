"""Dry-air mass conservation tests over multiple steps of the plane NH dycore.

PR2b uses cell-centred centred differences for the horizontal mass
flux divergence (an A-grid simplification). Over a doubly-periodic
domain the discrete divergence of any smooth field still sums to zero
to machine epsilon, so the *spatial* operator is exactly conservative.
Drift can come from the time discretisation (SSP-RK3 outer +
forward-backward acoustic substeps), which is what these tests
quantify.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    compute_dry_mass_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


def _setup(fix_mass=False, anchor_mass_to_initial=False):
    grid = create_plane_grid(
        nx=4, ny=4, nlev=6, dx=1.0e3, dy=1.0e3, dtype=jnp.float64
    )
    height_coord = create_height_coordinate(grid.nlev, H=20.0e3)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    config = CompressibleEulerConfig(
        sponge_coeff=0.0,
        hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False,
        use_coriolis=False,
        fix_mass=fix_mass,
        anchor_mass_to_initial=anchor_mass_to_initial,
    )
    model = PlaneCompressibleEulerModel(grid, height_coord, terrain, config)
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)
    return model, state, grid, height_coord, terrain


def _seed_random_perturbation(state, seed=0, amplitude=1.0e-4):
    """Seed small ``rho_prime`` and ``theta_prime`` perturbations so the
    integration actually evolves (rest state is trivially conserved)."""
    rng = np.random.default_rng(seed)
    rho_pert = jnp.asarray(
        rng.standard_normal(state.rho_prime.data.shape) * amplitude
    )
    theta_pert = jnp.asarray(
        rng.standard_normal(state.theta_prime.data.shape) * amplitude
    )
    return state._replace(
        rho_prime=state.rho_prime.replace(data=rho_pert),
        theta_prime=state.theta_prime.replace(data=theta_pert),
    )


def test_dry_mass_drift_bound_without_fixer():
    """Without the mass fixer, dry mass should drift by less than a
    documented bound over a short integration. The PR2b cell-centred
    divergence is exactly conservative on the periodic plane, so any
    drift comes from rho' updates inside the acoustic substep (which
    do not currently feed back through a horizontal-divergence
    correction). Document the bound here; tighten if PR2c benchmarks
    surface a tighter result."""
    model, state, grid, height_coord, terrain = _setup(fix_mass=False)
    state = _seed_random_perturbation(state, amplitude=1.0e-3)
    mass_0 = compute_dry_mass_plane(state, grid, height_coord, terrain)

    for _ in range(20):
        state = model.step(state, dt=0.5)

    mass_1 = compute_dry_mass_plane(state, grid, height_coord, terrain)
    rel_drift = float(jnp.abs(mass_1 - mass_0) / jnp.abs(mass_0))
    # Empirically observed bound on this small case; tighten in PR2c.
    assert rel_drift < 1.0e-6, f"rel_drift={rel_drift}"


def test_dry_mass_anchored_to_initial_under_fixer():
    """With ``fix_mass=True, anchor_mass_to_initial=True`` the fixer
    re-applies the initial mass after every outer step, so the
    integral matches the anchor to floating-point precision
    indefinitely."""
    model, state, grid, height_coord, terrain = _setup(
        fix_mass=True, anchor_mass_to_initial=True,
    )
    state = _seed_random_perturbation(state, amplitude=1.0e-3)
    mass_0 = compute_dry_mass_plane(state, grid, height_coord, terrain)
    # Trigger the anchor target on the first call.
    state = model.step(state, dt=0.5)

    for _ in range(20):
        state = model.step(state, dt=0.5)

    mass_1 = compute_dry_mass_plane(state, grid, height_coord, terrain)
    rel_drift = float(jnp.abs(mass_1 - mass_0) / jnp.abs(mass_0))
    assert rel_drift < 1.0e-11, f"rel_drift={rel_drift}"


def test_precompute_target_mass_enables_lax_scan_with_fix_mass():
    """``precompute_target_mass(state)`` pre-populates ``_target_mass``
    with a concrete (non-traced) array, so ``model.step`` inside
    ``jax.lax.scan`` does not trigger a tracer leak.

    Without precompute, the first call inside scan captures a traced
    target_mass → UnexpectedTracerError. This test pins the API contract
    for the iter-37 fix.
    """
    import jax
    model, state, grid, height_coord, terrain = _setup(
        fix_mass=True, anchor_mass_to_initial=True,
    )
    state = _seed_random_perturbation(state, amplitude=1.0e-3)
    mass_0 = compute_dry_mass_plane(state, grid, height_coord, terrain)

    # Pre-populate target_mass with concrete (non-traced) value
    model.precompute_target_mass(state)
    assert model._target_mass is not None
    assert not isinstance(model._target_mass, jax.core.Tracer)

    @jax.jit
    def run(s):
        def body(c, _):
            return model.step(c, 0.5), None
        return jax.lax.scan(body, s, None, length=20)[0]

    out = run(state)
    jax.block_until_ready(jax.tree.leaves(out))
    mass_1 = compute_dry_mass_plane(out, grid, height_coord, terrain)
    rel_drift = float(jnp.abs(mass_1 - mass_0) / jnp.abs(mass_0))
    # fix_mass under scan should preserve mass to machine precision
    assert rel_drift < 1.0e-11, f"rel_drift={rel_drift}"
