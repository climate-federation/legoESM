"""FV3_3D iter 305: ``fv3_corner_laplacian_iteration`` zeros
constant fields.

Any divergence-of-gradient Laplacian operator must zero out
constants because the gradient of a constant is exactly 0::

    vc(i, j) = (c - c) * divg_u(i, j) = 0
    uc(i, j) = (c - c) * divg_v(i, j) = 0
    out(i, j) = (0 - 0 + 0 - 0) * rarea_c = 0

The cube-vertex corner-removal terms (``±uc`` at sw/se/ne/nw)
also vanish because uc=0 everywhere.

iter-304 tested ``lap(0) = 0``; iter-305 tests the strictly
stronger ``lap(c) = 0`` for ``c != 0`` — the constants-in-
kernel property of the discrete Laplacian.

Tests
-----

1. ``test_corner_laplacian_zeros_constant`` — at multiple
   non-zero constant values c, lap(c-field) = 0 exactly
   (atol=1e-14).  Tested across resolutions {C4, C8, C16}.
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


@pytest.mark.parametrize("c", [-3.5, 1.0, 1e-6, 1e6])
@pytest.mark.parametrize("n", [4, 8, 16])
def test_corner_laplacian_zeros_constant(c, n):
    """``lap(c)`` = 0 exactly for any constant c.

    Constants live in the kernel of the FV3 corner Laplacian
    by construction (gradient of a constant is 0).
    """
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    field = jnp.full((6, n + 1, n + 1), c, dtype=jnp.float64)
    lap = fv3_corner_laplacian_iteration(field, cdgrid)
    np.testing.assert_array_equal(
        np.asarray(lap), np.zeros_like(np.asarray(lap)),
        err_msg=(
            f"fv3_corner_laplacian_iteration must zero out "
            f"constant c={c} at C{n} (constants are in the "
            f"kernel of any divergence-of-gradient Laplacian)."
        ),
    )
