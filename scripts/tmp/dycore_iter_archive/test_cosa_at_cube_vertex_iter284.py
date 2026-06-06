"""FV3_3D iter 284: precise test for the ``cosa`` metric value
at the cube vertex.

At a cube vertex, 3 panels meet at 60° angles (the cubed
sphere is the gnomonic projection of the cube).  Therefore
the angle between the i-direction and j-direction tangents
at a corner cell adjacent to a cube vertex is 60°, so:

    cos(60°) = 0.5

This test pins this STRUCTURAL property: |cosa_corner| at
the 8 cube vertex CORNERS (= 4 corner cells per face × 6
faces / shared 3-way = 8 vertices) should equal exactly 0.5
within FP tolerance.

iter-266 verified |cosa| ≤ 0.6 (looser bound).  iter-284
pins the EXACT 0.5 value, which is the cube-geometry
signature.

Tests
-----

1. ``test_cosa_corner_at_cube_vertex_is_0p5`` — at the 4
   panel corner cells (i, j ∈ {0, n}), |cosa_corner| ≈ 0.5.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def test_cosa_corner_at_cube_vertex_is_0p5():
    """|cosa_corner| at cube-vertex corner cells = 0.5
    (= cos(60°)) within FP tolerance."""
    n = 16
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # cosa_corner shape: (6, n+1, n+1).  The corners at
    # (i, j) ∈ {(0, 0), (0, n), (n, 0), (n, n)} on each face
    # are the panel corners — these are where 3 panels meet
    # at cube vertices.
    cosa = cdgrid.cosa_corner
    assert cosa.shape == (6, n + 1, n + 1)

    # Extract the 4 panel corners per face.
    corners = jnp.stack([
        cosa[:, 0, 0],
        cosa[:, 0, n],
        cosa[:, n, 0],
        cosa[:, n, n],
    ], axis=0)    # (4, 6)
    abs_corners = jnp.abs(corners)

    # At the cube vertex, each panel-corner cell has
    # |cosa| = 0.5 = cos(60°).  Allow small FP slack.
    np.testing.assert_allclose(
        abs_corners, 0.5,
        rtol=1e-6, atol=1e-6,
        err_msg=(
            f"|cosa_corner| at cube vertex panel-corners must "
            f"equal 0.5 = cos(60°) for the gnomonic cubed "
            f"sphere.  Got max diff from 0.5 = "
            f"{float(jnp.max(jnp.abs(abs_corners - 0.5))):.3e}."
        ),
    )

    # Sanity: the value should be NEGATIVE on roughly half the
    # corners and POSITIVE on the other half (depending on the
    # local orientation of i, j tangents).  Verify mix.
    pos_count = int(jnp.sum(corners > 0))
    neg_count = int(jnp.sum(corners < 0))
    total = pos_count + neg_count
    assert total > 0
    # At least 1 of each sign expected.
    assert pos_count > 0 and neg_count > 0, (
        "Expected mix of +0.5 and -0.5 at cube vertices "
        "(orientation-dependent); got "
        f"pos={pos_count}, neg={neg_count}."
    )
