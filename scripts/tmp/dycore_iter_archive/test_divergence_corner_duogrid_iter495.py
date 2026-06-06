"""FV3_3D iter 495: isolate ``fv3_divergence_corner_3d`` —
does this op amplify edge artifacts under duogrid?

iter-477 ruled out Laplacian amplification.  But the corner-
divergence op (``divergence_corner`` in d_sw5, used for the
corner-div damping) is a DIFFERENT discrete operator that
reads halo + computes a corner-aware divergence.  Test if
THIS op produces edge artifacts under duogrid.

Methodology:
* Build u_corner / v_corner of random noise on cubed sphere.
* Apply ``fv3_divergence_corner_3d`` with grid_off vs grid_on.
* Measure edge_var / interior_var of the output divergence.

Tests
-----

1. ``test_divergence_corner_duogrid_edge_ratio``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core._fv3_divergence_corner import (
    fv3_divergence_corner_3d,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _edge_and_interior_std(field_data):
    n_face, n_x, n_y, n_lev = field_data.shape
    edge_mask = np.zeros((n_x, n_y), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mask_b = np.broadcast_to(
        edge_mask[None, :, :, None], field_data.shape,
    )
    interior_mask = ~edge_mask_b
    arr = np.asarray(field_data)
    return (
        float(arr[edge_mask_b].std()),
        float(arr[interior_mask].std()),
    )


def test_divergence_corner_duogrid_edge_ratio(capsys):
    n = 8
    nlev = 5
    grid_on = create_cubed_sphere(n, use_duogrid=True)
    grid_off = create_cubed_sphere(n, use_duogrid=False)
    cdg_on = create_cubed_sphere_cdgrid(grid_on)
    cdg_off = create_cubed_sphere_cdgrid(grid_off)
    rng = np.random.default_rng(seed=495)
    # u, v at D-grid corners shape (6, n+1, n+1, nlev)
    u_corner = jnp.asarray(
        rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev)),
    )
    v_corner = jnp.asarray(
        rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev)),
    )
    divg_off = fv3_divergence_corner_3d(u_corner, v_corner, cdg_off)
    divg_on = fv3_divergence_corner_3d(u_corner, v_corner, cdg_on)
    e_off, i_off = _edge_and_interior_std(divg_off)
    e_on, i_on = _edge_and_interior_std(divg_on)
    with capsys.disabled():
        print(
            f"\n[iter-495 fv3_divergence_corner_3d duogrid effect]"
        )
        print(f"  no-duogrid: edge std = {e_off:.4e}, interior = {i_off:.4e}, ratio = {e_off/max(i_off, 1e-30):.4f}")
        print(f"  duogrid:    edge std = {e_on:.4e}, interior = {i_on:.4e}, ratio = {e_on/max(i_on, 1e-30):.4f}")
        edge_change = e_on / max(e_off, 1e-30)
        int_change = i_on / max(i_off, 1e-30)
        print(f"  duogrid effect: edge × {edge_change:.2f}, interior × {int_change:.2f}")
        if edge_change > 2.0 * int_change:
            print(
                f"  Conclusion: divergence_corner_3d AMPLIFIES "
                f"edge artifacts under duogrid → primary culprit."
            )
        else:
            print(
                f"  Conclusion: divergence_corner_3d edge effect "
                f"is modest; bug is elsewhere."
            )
    assert np.isfinite(e_on) and np.isfinite(e_off)
