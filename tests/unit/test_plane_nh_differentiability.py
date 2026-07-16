"""Finite-difference gradient check through one full step of the plane NH dycore.

A small (``nx=ny=4, nlev=4``) plane is integrated for one outer
SSP-RK3 step and the scalar loss ``sum(rho_prime ** 2)`` is
differentiated with respect to the initial ``rho_prime`` array.
``jax.test_util.check_grads`` compares reverse-mode AD against
second-order centred finite differences with tolerances appropriate
for ``JAX_ENABLE_X64=1``.

The test runs in two configurations:

1. ``fix_mass=False`` — pure dycore step, no mass correction.
2. ``fix_mass=True, anchor_mass_to_initial=True`` — the dycore step
   followed by a uniform-additive mass correction. The fixer's global
   reduction is differentiable in JAX, so the gradient should still
   match FD.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


def _model(fix_mass=False):
    grid = create_plane_grid(
        nx=4, ny=4, nlev=4, dx=1.0e3, dy=1.0e3, dtype=jnp.float64
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
        anchor_mass_to_initial=fix_mass,
    )
    model = PlaneCompressibleEulerModel(grid, height_coord, terrain, config)
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)
    return model, state


def _loss_factory(model, base_state):
    """Build a scalar loss that takes a ``rho_prime`` array argument
    (so ``check_grads`` perturbs a single explicit floating array
    rather than the whole NamedTuple state).
    """
    def loss(rho_prime_data):
        s = base_state._replace(
            rho_prime=base_state.rho_prime.replace(data=rho_prime_data),
        )
        s_next = model.step(s, dt=1.0)
        return jnp.sum(s_next.rho_prime.data ** 2)
    return loss


def _seed_rho_prime(state, seed=0, amplitude=1.0e-4):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal(state.rho_prime.data.shape) * amplitude,
    )


def test_grad_check_through_one_step_no_fixer():
    model, state = _model(fix_mass=False)
    rho_p = _seed_rho_prime(state, seed=0)
    loss = _loss_factory(model, state)
    jax.test_util.check_grads(
        loss, (rho_p,), order=1, modes=["rev"],
        rtol=1.0e-3, atol=1.0e-6,
    )


def test_grad_check_through_one_step_with_fixer():
    model, state = _model(fix_mass=True)
    # Pre-warm the ``_target_mass`` cache outside the gradient context
    # so the first traced call does not capture a transient tracer (the
    # cache is a Python-side attribute on the model and would leak a
    # LinearizeTracer across ``check_grads``'s FD evaluations otherwise).
    _ = model.step(state, dt=1.0)
    rho_p = _seed_rho_prime(state, seed=1)
    loss = _loss_factory(model, state)
    jax.test_util.check_grads(
        loss, (rho_p,), order=1, modes=["rev"],
        rtol=1.0e-3, atol=1.0e-6,
    )


def test_grad_is_not_all_zero_for_perturbed_state():
    """Sanity check: the gradient must depend on the seed input
    (a zero gradient would silently pass ``check_grads``)."""
    model, state = _model(fix_mass=False)
    rho_p = _seed_rho_prime(state, seed=2)
    loss = _loss_factory(model, state)
    grad = jax.grad(loss)(rho_p)
    assert float(jnp.max(jnp.abs(grad))) > 0.0
