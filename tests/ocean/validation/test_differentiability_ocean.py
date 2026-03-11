#!/usr/bin/env python
"""Test JAX differentiability of all ocean discretizations over 10 steps.

For each ocean discretization (centered, finite_volume, fc_gram, fc_gram_cgrid),
compute jax.grad through 10 time steps and verify finite, non-zero gradients.
"""

import sys
import traceback

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init import rest_state_ocean
from legoesm.ocean.state import OceanConfig
from legoesm.ocean.dynamics.ocean_model import OceanModel

N = 8
N_STEPS = 10
DT = 600.0

DISCRETIZATIONS = ["centered", "finite_volume", "fc_gram", "fc_gram_cgrid"]


@pytest.fixture(scope="module")
def ocean_setup():
    """Create grid, z_coord, state, and config once per module."""
    grid = create_cubed_sphere(N)
    z_coord = create_ocean_z_star(n_levels=5, H_max=5500.0)
    state = rest_state_ocean(grid, z_coord)
    # Add small velocity perturbation so tendencies are non-trivial
    u_pert = 0.01 * jnp.sin(grid.lon[..., jnp.newaxis] * 3)
    u_pert = jnp.broadcast_to(u_pert, state.u.data.shape).astype(jnp.float32)
    state = state._replace(u=state.u.replace(data=state.u.data + u_pert))
    config = OceanConfig(
        use_conservation_fixer=False,
        enable_runtime_checks=False,
        n_barotropic_substeps=2,
    )
    return grid, z_coord, state, config


@pytest.mark.parametrize("disc_name", DISCRETIZATIONS)
def test_differentiability(disc_name, ocean_setup):
    """Test jax.grad through N_STEPS of model.step()."""
    grid, z_coord, state, config = ocean_setup
    model = OceanModel(grid, z_coord, config, discretization=disc_name)

    def loss(T_data):
        s = state._replace(T=state.T.replace(data=T_data))
        for _ in range(N_STEPS):
            s = model.step(s, DT)
        return jnp.mean(s.T.data ** 2)

    grads = jax.grad(loss)(state.T.data)
    assert jnp.all(jnp.isfinite(grads)), f"{disc_name}: non-finite gradients"
    assert not jnp.allclose(grads, 0.0), f"{disc_name}: gradients are all zero"


if __name__ == "__main__":
    # Script-style execution for standalone use
    grid = create_cubed_sphere(N)
    z_coord = create_ocean_z_star(n_levels=5, H_max=5500.0)
    state = rest_state_ocean(grid, z_coord)
    u_pert = 0.01 * jnp.sin(grid.lon[..., jnp.newaxis] * 3)
    u_pert = jnp.broadcast_to(u_pert, state.u.data.shape).astype(jnp.float32)
    state = state._replace(u=state.u.replace(data=state.u.data + u_pert))
    config = OceanConfig(
        use_conservation_fixer=False,
        enable_runtime_checks=False,
        n_barotropic_substeps=2,
    )

    print("=" * 70)
    print(f"OCEAN MODEL  (C{N}, {N_STEPS} steps, dt={DT}s)")
    print("=" * 70)

    for disc in DISCRETIZATIONS:
        try:
            model = OceanModel(grid, z_coord, config, discretization=disc)

            def loss(T_data):
                s = state._replace(T=state.T.replace(data=T_data))
                for _ in range(N_STEPS):
                    s = model.step(s, DT)
                return jnp.mean(s.T.data ** 2)

            grads = jax.grad(loss)(state.T.data)
            is_finite = bool(jnp.all(jnp.isfinite(grads)))
            is_nonzero = bool(not jnp.allclose(grads, 0.0))
            grad_norm = float(jnp.max(jnp.abs(grads)))

            if is_finite and is_nonzero:
                print(f"  PASS  {disc}  |grad|_max={grad_norm:.4e}")
            elif is_finite:
                print(f"  WARN  {disc}  gradients are all zero")
            else:
                print(f"  FAIL  {disc}  non-finite gradients, |grad|_max={grad_norm}")
        except Exception as e:
            print(f"  ERROR {disc}  {type(e).__name__}: {e}")
            traceback.print_exc()

    try:
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel
        print()
        print("  (spectral ocean requires float64, skipping in float32 mode)")
    except ImportError:
        pass

    print()
    print("Done.")
