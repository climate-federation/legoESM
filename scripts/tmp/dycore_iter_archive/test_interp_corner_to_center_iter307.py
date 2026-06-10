"""FV3_3D iter 307: ``interp_corner_to_center`` (D-corner →
cell-centre 4-pt average) preserves constants and is linear.

Mirror of iter-306 in the opposite direction.  This helper is
used by every iter-22[1-6] tendency-form d_con KE→heat path:

    dT_dt_cc = -d_con * interp_corner_to_center(dKE_dt_corner)
               / c_pd

(see primitive_eq_cdgrid.py line 1015 for corner_div_damp d_con
and analogous sites for div_damp d_con and ah_d_con).  The
helper has no halo exchange (purely interior 4-point stencil
over the 4 surrounding corners), so the affine-operator
invariants hold trivially.

Tests
-----

1. ``test_interp_corner_to_center_preserves_constant`` —
   constant input → constant output (rtol=1e-14) at both 2D
   (6, n+1, n+1) and 3D (6, n+1, n+1, nlev) shapes.
2. ``test_interp_corner_to_center_superposition`` — linear
   in input.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import interp_corner_to_center
from legoesm.grids.cubed_sphere import create_cubed_sphere


def _setup(seed=307):
    n = 8
    nlev = 5
    rng = np.random.default_rng(seed=seed)
    f1_2d = jnp.asarray(rng.uniform(-2.0, 2.0,
                                    size=(6, n + 1, n + 1)))
    f2_2d = jnp.asarray(rng.uniform(-2.0, 2.0,
                                    size=(6, n + 1, n + 1)))
    f1_3d = jnp.asarray(rng.uniform(-2.0, 2.0,
                                    size=(6, n + 1, n + 1, nlev)))
    f2_3d = jnp.asarray(rng.uniform(-2.0, 2.0,
                                    size=(6, n + 1, n + 1, nlev)))
    return n, nlev, f1_2d, f2_2d, f1_3d, f2_3d


@pytest.mark.parametrize("c", [-3.5, 0.0, 1.0, 1e-6, 1e6])
def test_interp_corner_to_center_preserves_constant(c):
    """Constant input → constant output (rtol=1e-14).
    Tested at both 2D and 3D shapes."""
    n, nlev, *_ = _setup()
    # 2D
    field_2d = jnp.full((6, n + 1, n + 1), c, dtype=jnp.float64)
    out_2d = interp_corner_to_center(field_2d)
    expected_2d = jnp.full((6, n, n), c, dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(out_2d), np.asarray(expected_2d),
        rtol=1e-14, atol=1e-14,
    )
    # 3D
    field_3d = jnp.full((6, n + 1, n + 1, nlev), c, dtype=jnp.float64)
    out_3d = interp_corner_to_center(field_3d)
    expected_3d = jnp.full((6, n, n, nlev), c, dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(out_3d), np.asarray(expected_3d),
        rtol=1e-14, atol=1e-14,
    )


@pytest.mark.parametrize("ab", [(1.0, 1.0), (2.0, -3.0),
                                (0.5, 0.5), (-1.0, 7.0)])
def test_interp_corner_to_center_superposition(ab):
    """Linearity at both 2D and 3D shapes."""
    alpha, beta = ab
    _, _, f1_2d, f2_2d, f1_3d, f2_3d = _setup()

    # 2D
    lhs_2d = interp_corner_to_center(alpha * f1_2d + beta * f2_2d)
    rhs_2d = (
        alpha * interp_corner_to_center(f1_2d) +
        beta * interp_corner_to_center(f2_2d)
    )
    np.testing.assert_allclose(
        np.asarray(lhs_2d), np.asarray(rhs_2d),
        rtol=1e-14, atol=1e-14,
    )

    # 3D
    lhs_3d = interp_corner_to_center(alpha * f1_3d + beta * f2_3d)
    rhs_3d = (
        alpha * interp_corner_to_center(f1_3d) +
        beta * interp_corner_to_center(f2_3d)
    )
    np.testing.assert_allclose(
        np.asarray(lhs_3d), np.asarray(rhs_3d),
        rtol=1e-14, atol=1e-14,
    )
