"""Bitwise equality of vertical-last wrappers with PR1 plane operators.

For each operator the wrapper is defined as

    op_vlast(field_yxz) := moveaxis(op_3d(moveaxis(field_yxz, -1, 0), grid), 0, -1)

so the wrapper output must equal the explicit two-moveaxis composition
exactly (this is a no-op shape rearrangement; no algebra differs).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les import plane_operators as _ops
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    curl_vlast,
    divergence_vlast,
    grad_x_vlast,
    grad_y_vlast,
    laplacian_vlast,
)
from legoesm.grids.plane import create_plane_grid


jax.config.update("jax_enable_x64", True)


def _grid(ny=6, nx=8, nlev=4):
    return create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=1.0e3, dy=2.0e3, dtype=jnp.float64
    )


def _random_yxz(grid, seed=0, vertical_size=None):
    if vertical_size is None:
        vertical_size = grid.nlev
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((grid.ny, grid.nx, vertical_size)))


@pytest.mark.parametrize(
    "wrapper, op_3d",
    [
        (grad_x_vlast, _ops.grad_x_3d),
        (grad_y_vlast, _ops.grad_y_3d),
        (laplacian_vlast, _ops.laplacian_3d),
    ],
)
def test_scalar_wrapper_matches_moveaxis(wrapper, op_3d):
    grid = _grid()
    phi_yxz = _random_yxz(grid, seed=1)
    expected = jnp.moveaxis(
        op_3d(jnp.moveaxis(phi_yxz, -1, 0), grid), 0, -1
    )
    actual = wrapper(phi_yxz, grid)
    assert jnp.array_equal(actual, expected)


@pytest.mark.parametrize(
    "wrapper, op_3d",
    [
        (divergence_vlast, _ops.divergence_3d),
        (curl_vlast, _ops.curl_3d),
    ],
)
def test_vector_wrapper_matches_moveaxis(wrapper, op_3d):
    grid = _grid()
    u_yxz = _random_yxz(grid, seed=2)
    v_yxz = _random_yxz(grid, seed=3)
    expected = jnp.moveaxis(
        op_3d(
            jnp.moveaxis(u_yxz, -1, 0),
            jnp.moveaxis(v_yxz, -1, 0),
            grid,
        ),
        0,
        -1,
    )
    actual = wrapper(u_yxz, v_yxz, grid)
    assert jnp.array_equal(actual, expected)


def test_wrappers_preserve_shape():
    """Output shape must equal the vertical-last input shape so downstream
    state-class assembly never sees unexpected layouts."""
    grid = _grid()
    phi_yxz = _random_yxz(grid)
    assert grad_x_vlast(phi_yxz, grid).shape == phi_yxz.shape
    assert grad_y_vlast(phi_yxz, grid).shape == phi_yxz.shape
    assert laplacian_vlast(phi_yxz, grid).shape == phi_yxz.shape
    assert divergence_vlast(phi_yxz, phi_yxz, grid).shape == phi_yxz.shape
    assert curl_vlast(phi_yxz, phi_yxz, grid).shape == phi_yxz.shape


def test_wrappers_under_jit_smoke():
    grid = _grid()
    phi_yxz = _random_yxz(grid)

    jitted = jax.jit(lambda f: divergence_vlast(f, f, grid).sum())
    out = jitted(phi_yxz)
    assert jnp.isfinite(out)
