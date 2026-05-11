"""FV3_3D iter 311: ``pad_halo_vector`` zero + linearity.

The vector halo padding chain is::

    (u_grid, v_grid)  --rotation->  (u_east, v_north)
    pad as scalars (cross-panel halo)
    (u_east, v_north)  --inverse rotation->  (u_grid, v_grid)

Each step is linear in (u, v):
    * forward rotation: linear (cos/sin multiplications)
    * scalar pad_halo: pinned linear at iter-310
    * inverse rotation: linear

So pad_halo_vector is LINEAR in (u, v) at fixed grid angles.

iter-310 pinned scalar pad_halo invariants; iter-311 extends
the same characterization to the vector halo path used by
every (u, v) cubed-sphere operator (Smagorinsky, divergence,
del6 damp_v, etc.).

Tests
-----

1. ``test_pad_halo_vector_zero`` — zero (u, v) → zero padded.
2. ``test_pad_halo_vector_linear`` — bit-for-bit superposition
   in (u, v).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_vector


def _setup(seed=311):
    n = 8
    grid = create_cubed_sphere(n)
    rng = np.random.default_rng(seed=seed)
    u1 = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n, n)))
    v1 = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n, n)))
    u2 = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n, n)))
    v2 = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n, n)))
    return n, grid, u1, v1, u2, v2


def _do_pad(u, v, grid):
    return pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )


def test_pad_halo_vector_zero():
    """pad_halo_vector(0, 0) → (0_padded, 0_padded) exactly.

    Linear operations preserve the zero element — no constant
    bias term in the rotation chain.
    """
    n, grid, *_ = _setup()
    z = jnp.zeros((6, n, n))
    u_pad, v_pad = _do_pad(z, z, grid)
    assert jnp.all(u_pad == 0.0), (
        "pad_halo_vector(u=0, v=0) must yield u_pad=0 exactly."
    )
    assert jnp.all(v_pad == 0.0), (
        "pad_halo_vector(u=0, v=0) must yield v_pad=0 exactly."
    )


@pytest.mark.parametrize("ab", [(1.0, 1.0), (2.0, -3.0),
                                (0.5, 0.5), (-1.0, 7.0)])
def test_pad_halo_vector_linear(ab):
    """``pad_halo_vector(α u1+β u2, α v1+β v2)`` =
    α × pad(u1, v1) + β × pad(u2, v2) bit-for-bit."""
    alpha, beta = ab
    _, grid, u1, v1, u2, v2 = _setup()

    u_lhs, v_lhs = _do_pad(alpha * u1 + beta * u2,
                           alpha * v1 + beta * v2, grid)
    u_pad1, v_pad1 = _do_pad(u1, v1, grid)
    u_pad2, v_pad2 = _do_pad(u2, v2, grid)
    u_rhs = alpha * u_pad1 + beta * u_pad2
    v_rhs = alpha * v_pad1 + beta * v_pad2

    np.testing.assert_allclose(
        np.asarray(u_lhs), np.asarray(u_rhs),
        rtol=1e-13, atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(v_lhs), np.asarray(v_rhs),
        rtol=1e-13, atol=1e-14,
    )
