"""FV3_3D iter 302: ``a2b_ord4`` linearity / superposition.

The 4th-order A→B corner interpolant is a LINEAR operator in
the input scalar field — every step (qx, qy, qxx, qyy, qout)
is a fixed-weight linear combination plus a halo padding step
that's also linear.  Therefore::

    a2b_ord4(α*f1 + β*f2) = α*a2b_ord4(f1) + β*a2b_ord4(f2)

iter-301 pinned constant-preservation (zeroth-order); iter-302
pins linearity (first-order operator property).  Together they
imply the operator is an AFFINE OPERATOR with the unique
identity for constants — strong structural correctness.

Tests
-----

1. ``test_a2b_ord4_superposition`` — the linear-combination
   identity for several (α, β) pairs.
2. ``test_a2b_ord4_homogeneity`` — degenerate case
   ``a2b_ord4(α*f) = α*a2b_ord4(f)``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import (
    interp_center_to_corner_a2b_ord4,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _setup(n=8, seed=302):
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed=seed)
    f1 = jnp.asarray(rng.uniform(-2.0, 2.0, size=(6, n, n)))
    f2 = jnp.asarray(rng.uniform(-2.0, 2.0, size=(6, n, n)))
    return cdgrid, f1, f2


@pytest.mark.parametrize("ab", [(1.0, 1.0), (2.0, -3.0),
                                (0.5, 0.5), (-1.0, 7.0)])
def test_a2b_ord4_superposition(ab):
    """``a2b_ord4`` distributes over linear combinations."""
    alpha, beta = ab
    cdgrid, f1, f2 = _setup()

    lhs = interp_center_to_corner_a2b_ord4(
        alpha * f1 + beta * f2, cdgrid,
    )
    rhs = alpha * interp_center_to_corner_a2b_ord4(f1, cdgrid) + \
          beta * interp_center_to_corner_a2b_ord4(f2, cdgrid)

    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs), rtol=1e-14, atol=1e-14,
        err_msg=(
            f"a2b_ord4 must be linear: alpha={alpha}, beta={beta}"
            f" superposition violated."
        ),
    )


@pytest.mark.parametrize("alpha", [-2.5, 0.5, 3.0, 1e-6])
def test_a2b_ord4_homogeneity(alpha):
    """``a2b_ord4(α*f) = α*a2b_ord4(f)``."""
    cdgrid, f, _ = _setup()

    lhs = interp_center_to_corner_a2b_ord4(alpha * f, cdgrid)
    rhs = alpha * interp_center_to_corner_a2b_ord4(f, cdgrid)

    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs), rtol=1e-14, atol=1e-14,
    )
