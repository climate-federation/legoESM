"""FV3_3D iter 477: test iter-476 hypothesis — does
``laplacian_compact_3d`` with duogrid amplify edge values
when iterated (Laplacian-smoothing style)?

iter-476 narrowed the duogrid edge-std bug to:
  (a) iter-475's 3.5% halo overshoot for linear input
  (b) iterated Laplacian smoothing reading the halo

If iterating ``laplacian_compact_3d`` on a linear field with
duogrid produces growing edge values while no-duogrid stays
bounded, that confirms the hypothesis.

Methodology:
* Use the EXACT del-2 smoothing iteration from iter-457:
  ``f += cd * laplacian_compact_3d(f, grid)`` with
  ``cd = 0.20 * da_min``.
* Apply to a linear-in-index field.
* After 2 iterations (FV3 nf_ke at nord=1), measure edge std
  for duogrid vs no-duogrid grids.

Tests
-----

1. ``test_laplacian_compact_3d_iteration_edge_growth``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_3d import laplacian_compact_3d
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


def test_laplacian_compact_3d_iteration_edge_growth(capsys):
    """Apply 2 iterations of the iter-457 del-2 smoothing on
    a linear-in-index field for duogrid OFF vs ON.  Measure
    edge vs interior std change."""
    n = 8
    nlev = 5
    coeff = 0.20    # FV3 cnst_0p20 (iter-457 default)
    n_iters = 2    # FV3 nf_ke at nord=1
    grid_off = create_cubed_sphere(n, use_duogrid=False)
    grid_on = create_cubed_sphere(n, use_duogrid=True)
    cdg_off = create_cubed_sphere_cdgrid(grid_off)
    cdg_on = create_cubed_sphere_cdgrid(grid_on)
    da_min_off = float(jnp.min(cdg_off.area_corner))
    da_min_on = float(jnp.min(cdg_on.area_corner))

    # Linear-in-index field on each face.
    i_arr = jnp.arange(n).astype(jnp.float64)
    j_arr = jnp.arange(n).astype(jnp.float64)
    ij = i_arr[None, :, None, None] + j_arr[None, None, :, None]
    field0 = jnp.broadcast_to(ij, (6, n, n, nlev))

    # Initial edge/interior stds.
    e0, i0 = _edge_and_interior_std(field0)

    # Iterate del-2 smoothing on grid_off.
    f_off = field0
    for _ in range(n_iters):
        lap = laplacian_compact_3d(f_off, grid_off)
        f_off = f_off + coeff * da_min_off * lap
    e_off_after, i_off_after = _edge_and_interior_std(f_off)

    # Iterate del-2 smoothing on grid_on (duogrid).
    f_on = field0
    for _ in range(n_iters):
        lap = laplacian_compact_3d(f_on, grid_on)
        f_on = f_on + coeff * da_min_on * lap
    e_on_after, i_on_after = _edge_and_interior_std(f_on)

    with capsys.disabled():
        print(
            f"\n[iter-477 laplacian_compact_3d iteration test]"
            f"\n  initial:           edge std = {e0:.4f}, "
            f"interior std = {i0:.4f}"
        )
        print(
            f"  after {n_iters} iters (no duogrid):"
            f" edge = {e_off_after:.4f}, interior = {i_off_after:.4f}"
            f"\n    edge change: × {e_off_after / max(e0, 1e-30):.3f}"
        )
        print(
            f"  after {n_iters} iters (duogrid):"
            f"   edge = {e_on_after:.4f}, interior = {i_on_after:.4f}"
            f"\n    edge change: × {e_on_after / max(e0, 1e-30):.3f}"
        )
        if e_on_after > 2.0 * e_off_after:
            print(
                f"  Conclusion: duogrid iteration AMPLIFIES "
                f"edge std 2×+ relative to no-duogrid → "
                f"hypothesis CONFIRMED."
            )
        else:
            print(
                f"  Conclusion: duogrid edge growth not 2×+ "
                f"larger — hypothesis NOT supported in this "
                f"isolation test."
            )
    assert np.isfinite(e_on_after) and np.isfinite(e_off_after)
