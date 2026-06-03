"""FV3_3D iter 309: ``fv3_del6_vorticity_damping`` linearity
in (u, v) at fixed damp / nord.

The helper chain is::

    vt(u) = u * dx_at_v        (linear in u)
    ut(v) = v * dy_at_u        (linear in v)
    wk(vt, ut) = (i, j)-vorticity FD  (linear in vt, ut)
    fx2, fy2 = del^n flux of wk    (linear in wk)
    du_d  = +fy2 / dx_edge_y       (linear in fy2)
    dv_d  = -fx2 / dy_edge_x       (linear in fx2)

Every step is a fixed-weight linear operation (FD coefficients,
metric multiplications, halo padding, area normalization), so
the operator is LINEAR in (u, v):

    helper(α u1 + β u2, α v1 + β v2, damp, nord)
        = α * helper(u1, v1, damp, nord)
          + β * helper(u2, v2, damp, nord)

iter-308 pinned linearity in ``damp``; iter-309 pins linearity
in (u, v) — orthogonal axis.  Combined, the helper is bilinear:
linear in damp at fixed (u, v) AND linear in (u, v) at fixed
damp.

Tests
-----

1. ``test_del6_damp_v_zero_uv`` — (u=0, v=0) → (du, dv) = 0
   exactly.
2. ``test_del6_damp_v_linear_in_uv`` — superposition holds
   for (α, β) at fixed damp / nord.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.fv3_del6_vt_flux import fv3_del6_vorticity_damping
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _setup(seed=309):
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed=seed)
    u1 = jnp.asarray(rng.uniform(-3.0, 3.0,
                                 size=(6, n, n + 1)))
    v1 = jnp.asarray(rng.uniform(-3.0, 3.0,
                                 size=(6, n + 1, n)))
    u2 = jnp.asarray(rng.uniform(-3.0, 3.0,
                                 size=(6, n, n + 1)))
    v2 = jnp.asarray(rng.uniform(-3.0, 3.0,
                                 size=(6, n + 1, n)))
    return cdgrid, u1, v1, u2, v2


@pytest.mark.parametrize("nord", [0, 1, 2])
def test_del6_damp_v_zero_uv(nord):
    """(u=0, v=0) → (du, dv) = 0 exactly."""
    cdgrid, u, v, _, _ = _setup()
    n = cdgrid.n
    z_u = jnp.zeros((6, n, n + 1))
    z_v = jnp.zeros((6, n + 1, n))
    du, dv = fv3_del6_vorticity_damping(
        z_u, z_v, damp=1e-12, nord=nord, cdgrid=cdgrid,
    )
    assert jnp.all(du == 0.0)
    assert jnp.all(dv == 0.0)


@pytest.mark.parametrize("nord", [0, 1, 2])
@pytest.mark.parametrize("ab", [(1.0, 1.0), (2.0, -3.0),
                                (0.5, 0.5), (-1.0, 7.0)])
def test_del6_damp_v_linear_in_uv(nord, ab):
    """``helper(α u1 + β u2, α v1 + β v2)`` = α × helper(u1, v1)
    + β × helper(u2, v2) at fixed (damp, nord)."""
    alpha, beta = ab
    cdgrid, u1, v1, u2, v2 = _setup()
    base_damp = 1e-12 if nord == 0 else 1e-30 if nord == 2 else 1e-21

    du_lhs, dv_lhs = fv3_del6_vorticity_damping(
        alpha * u1 + beta * u2,
        alpha * v1 + beta * v2,
        damp=base_damp, nord=nord, cdgrid=cdgrid,
    )
    du_1, dv_1 = fv3_del6_vorticity_damping(
        u1, v1, damp=base_damp, nord=nord, cdgrid=cdgrid,
    )
    du_2, dv_2 = fv3_del6_vorticity_damping(
        u2, v2, damp=base_damp, nord=nord, cdgrid=cdgrid,
    )
    du_rhs = alpha * du_1 + beta * du_2
    dv_rhs = alpha * dv_1 + beta * dv_2

    np.testing.assert_allclose(
        np.asarray(du_lhs), np.asarray(du_rhs),
        rtol=1e-13, atol=1e-30,
    )
    np.testing.assert_allclose(
        np.asarray(dv_lhs), np.asarray(dv_rhs),
        rtol=1e-13, atol=1e-30,
    )
