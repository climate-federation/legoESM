"""FV3_3D iter 304: ``fv3_corner_laplacian_iteration`` linearity.

The iter-18/iter-187 nord>=1 path applies this Laplacian
iteration in a Python loop ``for _ in range(nord): div =
lap(div)``.  The helper is a port of FV3 sw_core.F90:1746-1782
and consists of fixed-weight finite differences plus area
normalization — all linear in the input ``divg_d``::

    vc(i, j) = (divg_d(i+1, j) - divg_d(i, j)) * divg_u(i, j)
    uc(i, j) = (divg_d(i, j+1) - divg_d(i, j)) * divg_v(i, j)
    out(i, j) = (uc(j-1) - uc(j) + vc(i-1) - vc(i)) * rarea_c

Therefore::

    lap(0)              = 0
    lap(α f1 + β f2)    = α lap(f1) + β lap(f2)

iter-303 pinned ``fv3_divergence_corner_3d`` linearity in
(u, v); iter-304 pins the next stage in the corner-div
damping pipeline.

Tests
-----

1. ``test_corner_laplacian_zero_input`` — lap(0) = 0 exactly.
2. ``test_corner_laplacian_superposition`` — linear-
   combination identity at multiple (α, β).
3. ``test_corner_laplacian_homogeneity`` — lap(α*f) =
   α*lap(f).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core._fv3_divergence_corner import (
    fv3_corner_laplacian_iteration,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _setup(seed=304):
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed=seed)
    f1 = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, n + 1, n + 1)))
    f2 = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, n + 1, n + 1)))
    return cdgrid, f1, f2


def test_corner_laplacian_zero_input():
    """lap(0) = 0 exactly — every term vanishes when input
    is uniformly zero."""
    cdgrid, _, _ = _setup()
    n = cdgrid.n
    z = jnp.zeros((6, n + 1, n + 1))
    lap = fv3_corner_laplacian_iteration(z, cdgrid)
    assert jnp.all(lap == 0.0), (
        "fv3_corner_laplacian_iteration(0) must be exactly 0."
    )


@pytest.mark.parametrize("ab", [(1.0, 1.0), (2.0, -3.0),
                                (0.5, 0.5), (-1.0, 7.0)])
def test_corner_laplacian_superposition(ab):
    """lap(α*f1 + β*f2) = α*lap(f1) + β*lap(f2)."""
    alpha, beta = ab
    cdgrid, f1, f2 = _setup()

    lhs = fv3_corner_laplacian_iteration(
        alpha * f1 + beta * f2, cdgrid,
    )
    rhs = (
        alpha * fv3_corner_laplacian_iteration(f1, cdgrid) +
        beta * fv3_corner_laplacian_iteration(f2, cdgrid)
    )
    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs), rtol=1e-12, atol=1e-14,
        err_msg=(
            f"fv3_corner_laplacian_iteration not linear: alpha="
            f"{alpha}, beta={beta} superposition violated."
        ),
    )


@pytest.mark.parametrize("alpha", [-2.5, 0.5, 3.0, 1e-6])
def test_corner_laplacian_homogeneity(alpha):
    """lap(α*f) = α*lap(f)."""
    cdgrid, f, _ = _setup()
    lhs = fv3_corner_laplacian_iteration(alpha * f, cdgrid)
    rhs = alpha * fv3_corner_laplacian_iteration(f, cdgrid)
    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs), rtol=1e-12, atol=1e-14,
    )
