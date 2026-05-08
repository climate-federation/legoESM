"""FV3_3D iter 316: ``cgrid_divergence`` zero + linearity in
(u_c, v_c).

cgrid_divergence is the exact flux-form cell-centre
divergence used by the PE div_damp path::

    flux_x = u_c * dy_edge_x      (linear in u_c)
    flux_y = v_c * dx_edge_y      (linear in v_c)
    div = (Δflux_x + Δflux_y) / area    (linear FD + scalar
                                         normalize)

Linear in (u_c, v_c) at fixed grid metrics.  Combined with
iter-303 (corner divergence linearity), every divergence
helper used in the d_con cluster is now characterized as a
linear operator.

Tests
-----

1. ``test_cgrid_divergence_zero_velocity`` — div(0, 0) = 0
   exactly.
2. ``test_cgrid_divergence_superposition`` — linearity holds
   bit-for-bit at multiple (α, β) in 2D.
3. ``test_cgrid_divergence_3d_linearity`` — same in 3D shape.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import cgrid_divergence
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _setup(seed=316):
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed=seed)
    # u_c: (6, n+1, n), v_c: (6, n, n+1)
    u1_2d = jnp.asarray(rng.uniform(-3.0, 3.0,
                                    size=(6, n + 1, n)))
    v1_2d = jnp.asarray(rng.uniform(-3.0, 3.0,
                                    size=(6, n, n + 1)))
    u2_2d = jnp.asarray(rng.uniform(-3.0, 3.0,
                                    size=(6, n + 1, n)))
    v2_2d = jnp.asarray(rng.uniform(-3.0, 3.0,
                                    size=(6, n, n + 1)))
    u1_3d = jnp.asarray(rng.uniform(-3.0, 3.0,
                                    size=(6, n + 1, n, nlev)))
    v1_3d = jnp.asarray(rng.uniform(-3.0, 3.0,
                                    size=(6, n, n + 1, nlev)))
    u2_3d = jnp.asarray(rng.uniform(-3.0, 3.0,
                                    size=(6, n + 1, n, nlev)))
    v2_3d = jnp.asarray(rng.uniform(-3.0, 3.0,
                                    size=(6, n, n + 1, nlev)))
    return (cdgrid, u1_2d, v1_2d, u2_2d, v2_2d,
            u1_3d, v1_3d, u2_3d, v2_3d)


def test_cgrid_divergence_zero_velocity():
    """div(0, 0) = 0 exactly at both 2D and 3D shapes."""
    cdgrid, *_ = _setup()
    n = cdgrid.n
    nlev = 5

    z_u_2d = jnp.zeros((6, n + 1, n))
    z_v_2d = jnp.zeros((6, n, n + 1))
    div_2d = cgrid_divergence(z_u_2d, z_v_2d, cdgrid)
    assert jnp.all(div_2d == 0.0)

    z_u_3d = jnp.zeros((6, n + 1, n, nlev))
    z_v_3d = jnp.zeros((6, n, n + 1, nlev))
    div_3d = cgrid_divergence(z_u_3d, z_v_3d, cdgrid)
    assert jnp.all(div_3d == 0.0)


@pytest.mark.parametrize("ab", [(1.0, 1.0), (2.0, -3.0),
                                (0.5, 0.5), (-1.0, 7.0)])
def test_cgrid_divergence_superposition(ab):
    """Linearity in (u, v) at 2D shape (rtol=1e-13)."""
    alpha, beta = ab
    cdgrid, u1, v1, u2, v2, *_ = _setup()

    lhs = cgrid_divergence(
        alpha * u1 + beta * u2,
        alpha * v1 + beta * v2,
        cdgrid,
    )
    rhs = (
        alpha * cgrid_divergence(u1, v1, cdgrid) +
        beta * cgrid_divergence(u2, v2, cdgrid)
    )
    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs), rtol=1e-13, atol=1e-15,
    )


@pytest.mark.parametrize("ab", [(1.0, 1.0), (2.0, -3.0)])
def test_cgrid_divergence_3d_linearity(ab):
    """Linearity in 3D shape — vmaps over levels correctly."""
    alpha, beta = ab
    cdgrid, *_, u1, v1, u2, v2 = _setup()

    lhs = cgrid_divergence(
        alpha * u1 + beta * u2,
        alpha * v1 + beta * v2,
        cdgrid,
    )
    rhs = (
        alpha * cgrid_divergence(u1, v1, cdgrid) +
        beta * cgrid_divergence(u2, v2, cdgrid)
    )
    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs), rtol=1e-13, atol=1e-15,
    )
