"""FV3_3D iter 306: ``interp_center_to_corner`` (2nd-order 4-pt
average) preserves constants and is linear.

The 2nd-order interpolant ``0.25 * (NW + NE + SW + SE)`` is
used in many production paths (PE/NH zeta_corner non-iter-170
fallback, theta_corner, T_corner reciprocal interp, ah_smag
fallback, etc.).  It must satisfy the same affine-operator
contract as iter-301/302's a2b_ord4:

    * preserves constants (sum of weights = 1)
    * linear in input

iter-306 pins both for the 2nd-order helper that's still used
as the production default when ``use_fv3_a2b_zeta_corner=False``
or for non-zeta scalars.

Tests
-----

1. ``test_interp_2nd_order_preserves_constant`` — constant
   input → constant output (rtol=1e-14).
2. ``test_interp_2nd_order_superposition`` — linearity holds
   bit-for-bit.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import interp_center_to_corner
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _setup(seed=306):
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed=seed)
    f1 = jnp.asarray(rng.uniform(-2.0, 2.0, size=(6, n, n)))
    f2 = jnp.asarray(rng.uniform(-2.0, 2.0, size=(6, n, n)))
    return cdgrid, f1, f2


@pytest.mark.parametrize("c", [-3.5, 0.0, 1.0, 1e-6, 1e6])
def test_interp_2nd_order_preserves_constant(c):
    """Constant input → constant output (rtol=1e-14)."""
    cdgrid, _, _ = _setup()
    n = cdgrid.n
    field = jnp.full((6, n, n), c, dtype=jnp.float64)
    out = interp_center_to_corner(field, cdgrid)
    assert out.shape == (6, n + 1, n + 1)
    expected = jnp.full((6, n + 1, n + 1), c, dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(expected),
        rtol=1e-14, atol=1e-14,
        err_msg=(
            f"interp_center_to_corner must reproduce constant "
            f"{c} exactly (4-pt average sums to 1)."
        ),
    )


@pytest.mark.parametrize("ab", [(1.0, 1.0), (2.0, -3.0),
                                (0.5, 0.5), (-1.0, 7.0)])
def test_interp_2nd_order_superposition(ab):
    """Linearity: interp(α f1 + β f2) = α interp(f1) +
    β interp(f2)."""
    alpha, beta = ab
    cdgrid, f1, f2 = _setup()

    lhs = interp_center_to_corner(alpha * f1 + beta * f2, cdgrid)
    rhs = (
        alpha * interp_center_to_corner(f1, cdgrid) +
        beta * interp_center_to_corner(f2, cdgrid)
    )
    np.testing.assert_allclose(
        np.asarray(lhs), np.asarray(rhs),
        rtol=1e-14, atol=1e-14,
    )
