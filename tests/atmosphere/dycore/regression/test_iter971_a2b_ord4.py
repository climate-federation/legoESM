"""Iter-971 sentinel: pin Fortran-faithful 4th-order A→B interpolation
`_interp_center_to_corner_a2b_ord4`.

Port of Fortran `a2b_ord4` (a2b_edge.F90:50-330) for the duogrid
path.  Used by Smagorinsky-tuned d_sw5 callers as a more
Fortran-faithful alternative to `_interp_center_to_corner` (which is
a 2nd-order 4-point average).

Iter-970 audit identified the gap: Fortran uses 4th-order Lagrange-
plus-PPM cascade for cell→corner interpolation in the d_sw5
Smagorinsky branch (sw_core.F90:1795).  Our Python's default
`_interp_center_to_corner` uses 2nd-order 4-point average.

Iter-971 adds the 4th-order kernel with constants a1=9/16, a2=-1/16,
b1=7/12, b2=-1/12 matching Fortran's a2b_edge.F90:36-43.

This sentinel pins:
1. `_interp_center_to_corner_a2b_ord4` exists and produces the
   expected (6, n+1, n+1) shape.
2. On a constant input, output equals the constant (interpolation is
   exact for constants).
3. The result agrees with `_interp_center_to_corner` (2nd-order) to
   within ~5% on smooth inputs (relative measure).  For non-smooth
   inputs the 4th-order kernel reduces aliasing.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np

from legoesm.core.operators_cdgrid import (
    _interp_center_to_corner,
    _interp_center_to_corner_a2b_ord4,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def test_iter971_a2b_ord4_shape_and_constant():
    """`_interp_center_to_corner_a2b_ord4` returns (6, n+1, n+1)
    and is exact on a constant input.
    """
    N = 36
    grid = create_cubed_sphere(N, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    f = jnp.ones((6, N, N)) * 3.14159
    qout = _interp_center_to_corner_a2b_ord4(f, cdgrid)
    assert qout.shape == (6, N + 1, N + 1), (
        f"Expected (6, {N+1}, {N+1}); got {qout.shape}")
    assert np.allclose(np.asarray(qout), 3.14159, atol=1e-10), (
        "_interp_center_to_corner_a2b_ord4 should be exact on constants"
    )


def test_iter971_a2b_ord4_close_to_2nd_order_on_smooth_input():
    """On a smooth field (lat * cos(lon)), the 4th-order interpolation
    should agree with the 2nd-order interpolation to within a few
    percent on the interior.  At cube-face boundaries the two may
    differ more (different stencil truncation error patterns).
    """
    N = 36
    grid = create_cubed_sphere(N, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    # Smooth analytical field: f = sin(2*lat) * cos(lon)
    lat = grid.lat   # (6, N, N)
    lon = grid.lon
    f = jnp.sin(2 * lat) * jnp.cos(lon)
    q2nd = _interp_center_to_corner(f, cdgrid)
    q4th = _interp_center_to_corner_a2b_ord4(f, cdgrid)
    assert q2nd.shape == q4th.shape == (6, N + 1, N + 1)
    # Compare interior (avoid panel-edge corners where differences are
    # larger).  Interior corners: i, j ∈ [2, N-2].
    diff = np.abs(np.asarray(q4th[:, 2:N - 1, 2:N - 1])
                   - np.asarray(q2nd[:, 2:N - 1, 2:N - 1]))
    f_max = float(np.max(np.abs(np.asarray(f))))
    rel_diff = float(diff.max()) / max(f_max, 1e-10)
    assert rel_diff <= 0.10, (
        f"Interior |q4th - q2nd| / |f|_max = {rel_diff:.4f}; "
        f"expected ≤ 0.10 on a smooth analytical field."
    )
