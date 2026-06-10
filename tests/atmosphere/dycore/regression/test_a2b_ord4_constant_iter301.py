"""FV3_3D iter 301: ``a2b_ord4`` 4th-order A→B interpolation
helper preserves constant fields exactly.

The Fortran ``a2b_ord4`` is the 4-point Lagrange interpolant
used by FV3 for cell-centre → corner stencils on quantities
like vorticity (sw_core.F90:1795 wk → vort).  Its weights
satisfy:

    b1 + b2 = 7/12 + (-1/12) = 6/12 = 0.5   (qx, qy step)
    a1 + a2 = 9/16 + (-1/16) = 8/16 = 0.5   (qxx, qyy step)

So for a constant input field c::

    qx  = 2*(b1 + b2)*c = c
    qxx = 2*(a1 + a2)*c = c
    qout = 0.5 * (qxx + qyy) = 0.5*(c + c) = c

The helper preserves constants EXACTLY (rtol=1e-14), which
is the most basic interpolation correctness invariant — and
it must hold even at panel edges where ``_pad_halo_auto_h2``
exchanges with cross-panel rotation (a constant field is
invariant under any rotation).

iter-170/187 use this helper; iter-190 deduplicates a single
shared halo-2 exchange between both sites.

Tests
-----

1. ``test_a2b_ord4_preserves_constant`` — constant input on
   cell centres → constant output on corners (rtol=1e-14)
   for several constant values.
2. ``test_a2b_ord4_zero_preserved`` — zero input → zero
   output exactly.
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


def _setup(n=8):
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    return cdgrid, n


@pytest.mark.parametrize("c", [-3.5, 0.0, 1.0, 1e-6, 1e6])
def test_a2b_ord4_preserves_constant(c):
    """``a2b_ord4`` exact for constant fields (rtol=1e-14)."""
    cdgrid, n = _setup()
    field = jnp.full((6, n, n), c, dtype=jnp.float64)
    out = interp_center_to_corner_a2b_ord4(field, cdgrid)

    assert out.shape == (6, n + 1, n + 1), (
        f"Expected shape (6, n+1, n+1) corner output but got "
        f"{out.shape}"
    )

    expected = jnp.full((6, n + 1, n + 1), c, dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(expected), rtol=1e-14, atol=1e-14,
        err_msg=(
            f"a2b_ord4 must reproduce constant {c} exactly at "
            f"corners (b1+b2 = a1+a2 = 0.5 sum-to-half "
            f"invariant in 4-pt Lagrange weights)."
        ),
    )


def test_a2b_ord4_zero_preserved():
    """Zero input → zero output (no spurious bias term)."""
    cdgrid, n = _setup()
    field = jnp.zeros((6, n, n), dtype=jnp.float64)
    out = interp_center_to_corner_a2b_ord4(field, cdgrid)
    assert jnp.all(out == 0.0), (
        "a2b_ord4(0) must be 0 exactly — no constant bias term."
    )


@pytest.mark.parametrize("n", [4, 16, 36])
def test_a2b_ord4_preserves_constant_resolutions(n):
    """Constant-preservation holds across cube resolutions."""
    cdgrid, n_actual = _setup(n=n)
    field = jnp.full((6, n_actual, n_actual), 2.5, dtype=jnp.float64)
    out = interp_center_to_corner_a2b_ord4(field, cdgrid)
    expected = jnp.full(
        (6, n_actual + 1, n_actual + 1), 2.5, dtype=jnp.float64,
    )
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(expected), rtol=1e-14, atol=1e-14,
    )
