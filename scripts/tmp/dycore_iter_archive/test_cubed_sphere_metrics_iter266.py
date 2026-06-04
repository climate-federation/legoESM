"""FV3_3D iter 266: sanity test for cubed-sphere non-
orthogonality metrics (``cosa_corner``, ``rsin2_corner``,
``cosa_cell``, ``rsin2_cell``).

These metrics are used by FV3-faithful d_sw5 paths (and the
iter-238 metric-aware d_con form, if ported in the future).
This test pins down their structural properties:

* ``cosa`` magnitude is bounded (well below 1, since the
  cubed sphere is "approximately orthogonal" — angles between
  i and j tangents stay close to 90° everywhere).
* ``rsin2 > 0`` everywhere (sin² > 0 prevents 1/0).
* ``cosa`` is much smaller in panel interior than near cube
  vertices (vertices are where non-orthogonality concentrates).

Tests
-----

1. ``test_metrics_are_finite_and_bounded`` — basic sanity
   (finite, |cosa| < 0.5, rsin2 > 0.5).
2. ``test_cosa_concentrates_at_cube_vertices`` — corner cells
   adjacent to the 8 cube vertices have larger |cosa| than
   interior cells.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def test_metrics_are_finite_and_bounded():
    """All non-orthogonality metrics finite + bounded."""
    n = 16
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # Corner-staggered metrics.
    assert jnp.all(jnp.isfinite(cdgrid.cosa_corner))
    assert jnp.all(jnp.isfinite(cdgrid.rsin2_corner))
    # At cube vertices (3 panels meet at 60°) cos(60°) = 0.5 is
    # the structural max.  Allow up to 0.6 for numerical drift.
    assert float(jnp.max(jnp.abs(cdgrid.cosa_corner))) <= 0.6, (
        "|cosa_corner| should be ≤ 0.6 (cube vertex value is "
        "0.5 = cos(60°); interior is much smaller)."
    )
    # rsin2 = 1/sin²; sin²(90°) = 1 → rsin2 = 1.0 at orthogonal
    # cells; sin²(60°) = 0.75 → rsin2 ≈ 1.33 at vertices.
    # Allow tiny FP slack below 1.0 for numerical drift.
    assert float(jnp.min(cdgrid.rsin2_corner)) >= 0.99, (
        "rsin2_corner = 1/sin²(angle) should be ≥ 1.0 since "
        "sin²(angle) ≤ 1; with FP drift allow >= 0.99."
    )

    # Cell-centred metrics.
    assert jnp.all(jnp.isfinite(cdgrid.cosa_cell))
    assert jnp.all(jnp.isfinite(cdgrid.rsin2_cell))
    assert float(jnp.max(jnp.abs(cdgrid.cosa_cell))) <= 0.6
    assert float(jnp.min(cdgrid.rsin2_cell)) >= 0.99

    # sina_cell should be positive.
    assert jnp.all(jnp.isfinite(cdgrid.sina_cell))
    assert float(jnp.min(cdgrid.sina_cell)) > 0.0


def test_cosa_concentrates_at_cube_vertices():
    """|cosa| at cells adjacent to cube vertices is LARGER
    than |cosa| in panel interior.  Cube vertices are the 8
    points where 3 panels meet; non-orthogonality is
    structurally larger there."""
    n = 16
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # Vertex cells: i,j ∈ {0, n-1} on each face — there are 4
    # corner cells per face × 6 faces = 24 cells touching cube
    # vertices.  Use cosa_cell shape (6, n, n).
    cosa_abs = jnp.abs(cdgrid.cosa_cell)
    vertex_cells = jnp.stack([
        cosa_abs[:, 0, 0],     # (-, -)
        cosa_abs[:, 0, n - 1],   # (-, +)
        cosa_abs[:, n - 1, 0],   # (+, -)
        cosa_abs[:, n - 1, n - 1], # (+, +)
    ], axis=0).reshape(-1)

    # Interior cells: i, j ∈ [n/4, 3n/4).
    q1, q3 = n // 4, 3 * n // 4
    interior_cells = cosa_abs[:, q1:q3, q1:q3].reshape(-1)

    mean_vertex = float(jnp.mean(vertex_cells))
    mean_interior = float(jnp.mean(interior_cells))

    assert mean_vertex > mean_interior, (
        f"Cube vertex |cosa| should be > panel interior |cosa|: "
        f"vertex_mean={mean_vertex:.4e}, "
        f"interior_mean={mean_interior:.4e}.  If equal, the "
        f"grid metrics may not capture the cube vertex non-"
        f"orthogonality correctly."
    )
