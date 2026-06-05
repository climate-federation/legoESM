"""FV3_3D iter 489: localize iter-475's 3.5% halo overshoot.

iter-475 found duogrid halo for linear field gives max-abs
14.49 vs interior max 14.00 (3.5% overshoot).  Where exactly
in the halo are the overshooting cells?  Edges, corners, or
both?

Decompose the halo of pad_halo_4d output into:
* edge cells (along a face boundary, not at vertex)
* corner cells (at the 4 cube-face vertices)
* count cells exceeding interior max

If overshooting cells are ONLY at corners (4 per face × 6
faces = 24 cells globally per level), a targeted fix is
feasible.  If overshoots are spread across all edges, the
fix is harder.

Tests
-----

1. ``test_pad_halo_4d_duogrid_overshoot_localization``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_4d


def test_pad_halo_4d_duogrid_overshoot_localization(capsys):
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    i_arr = jnp.arange(n).astype(jnp.float64)
    j_arr = jnp.arange(n).astype(jnp.float64)
    ij = i_arr[None, :, None, None] + j_arr[None, None, :, None]
    field = jnp.broadcast_to(ij, (6, n, n, nlev))
    padded = pad_halo_4d(
        field, halo=1, interp_offsets=None, duogrid=grid.duogrid,
    )
    arr = np.asarray(padded)
    interior_max = float(np.max(np.abs(np.asarray(field))))

    # Halo cells: indices (i, j) where i in {0, n+1} or
    # j in {0, n+1} (the halo strip).
    # Padded shape: (6, n+2, n+2, nlev) = (6, 10, 10, 5).
    np_padded = arr  # shape (6, 10, 10, nlev)

    # Edge cells (along a face edge, NOT at corner):
    # halo row i=0 or i=n+1 but j in 1..n  (west/east face edges)
    # halo col j=0 or j=n+1 but i in 1..n  (south/north face edges)
    # Corner cells: i in {0, n+1} AND j in {0, n+1}.

    edge_max = 0.0
    corner_max = 0.0
    edge_count_over = 0
    corner_count_over = 0
    n_pad = n + 2
    for face in range(6):
        for i in range(n_pad):
            for j in range(n_pad):
                in_edge = (i == 0 or i == n_pad - 1) ^ (
                    j == 0 or j == n_pad - 1
                )
                in_corner = (
                    (i == 0 or i == n_pad - 1) and
                    (j == 0 or j == n_pad - 1)
                )
                if not (in_edge or in_corner):
                    continue
                level_vals = np_padded[face, i, j, :]
                cell_max = float(np.max(np.abs(level_vals)))
                if in_corner:
                    if cell_max > corner_max:
                        corner_max = cell_max
                    if cell_max > interior_max:
                        corner_count_over += 1
                else:
                    if cell_max > edge_max:
                        edge_max = cell_max
                    if cell_max > interior_max:
                        edge_count_over += 1

    with capsys.disabled():
        print(
            f"\n[iter-489 duogrid halo overshoot localization]"
        )
        print(f"  interior max-abs:       {interior_max:.4f}")
        print(f"  edge-cell halo max-abs: {edge_max:.4f}")
        print(f"  corner-cell halo max-abs: {corner_max:.4f}")
        print(
            f"  edge cells > interior_max:   {edge_count_over}"
        )
        print(
            f"  corner cells > interior_max: {corner_count_over}"
        )
        if corner_count_over > 0 and edge_count_over == 0:
            print(
                f"  Conclusion: overshoot CONFINED TO CORNER "
                f"cells → targeted fix feasible (24 cells/"
                f"level globally)."
            )
        elif edge_count_over > corner_count_over * 3:
            print(
                f"  Conclusion: overshoot WIDESPREAD across "
                f"edges → general halo-interpolation issue."
            )
        else:
            print(
                f"  Conclusion: overshoot distributed across "
                f"both edges and corners."
            )
    assert np.isfinite(edge_max) and np.isfinite(corner_max)
