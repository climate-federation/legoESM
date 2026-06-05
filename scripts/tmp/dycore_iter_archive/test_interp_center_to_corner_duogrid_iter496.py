"""FV3_3D iter 496: isolate ``_interp_center_to_corner`` —
does this halo-aware interpolation amplify edges under
duogrid?

iter-495 ruled out ``fv3_divergence_corner_3d``.  The
remaining suspect is the cell-center → corner interpolation
used by NH to lift u/v to the D-grid corner.  This op DOES
read halo (calls ``_pad_halo_auto`` internally).

Methodology:
* Build random cell-centered scalar field.
* Apply ``_interp_center_to_corner`` with grid_off vs grid_on.
* Compare edge_std / interior_std of the corner output.

Tests
-----

1. ``test_interp_center_to_corner_duogrid_edge_amplification``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import _interp_center_to_corner
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _edge_and_interior_std(field_data):
    """Field is corner-staggered: shape (6, n+1, n+1, nlev)."""
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


def test_interp_center_to_corner_duogrid_edge_amplification(capsys):
    n = 8
    nlev = 5
    grid_on = create_cubed_sphere(n, use_duogrid=True)
    grid_off = create_cubed_sphere(n, use_duogrid=False)
    cdg_on = create_cubed_sphere_cdgrid(grid_on)
    cdg_off = create_cubed_sphere_cdgrid(grid_off)
    rng = np.random.default_rng(seed=496)
    # cell-center scalar, shape (6, n, n, nlev)
    field = jnp.asarray(
        rng.normal(loc=2.0, scale=1.5, size=(6, n, n, nlev)),
    )
    corner_off = _interp_center_to_corner(field, cdg_off)
    corner_on = _interp_center_to_corner(field, cdg_on)
    e_off, i_off = _edge_and_interior_std(corner_off)
    e_on, i_on = _edge_and_interior_std(corner_on)
    with capsys.disabled():
        print(
            f"\n[iter-496 _interp_center_to_corner duogrid effect]"
        )
        print(
            f"  no-duogrid: edge std = {e_off:.4f}, "
            f"interior = {i_off:.4f}, ratio = "
            f"{e_off/max(i_off, 1e-30):.4f}"
        )
        print(
            f"  duogrid:    edge std = {e_on:.4f}, "
            f"interior = {i_on:.4f}, ratio = "
            f"{e_on/max(i_on, 1e-30):.4f}"
        )
        edge_change = e_on / max(e_off, 1e-30)
        int_change = i_on / max(i_off, 1e-30)
        print(
            f"  duogrid effect: edge × {edge_change:.3f}, "
            f"interior × {int_change:.3f}"
        )
        if edge_change > 1.5 * int_change:
            print(
                f"  Conclusion: ``_interp_center_to_corner`` "
                f"AMPLIFIES edge artifacts under duogrid → "
                f"matches expected dycore amplifier."
            )
        elif edge_change < 0.95 or edge_change > 1.05:
            print(
                f"  Conclusion: some duogrid effect on edge "
                f"but moderate magnitude."
            )
        else:
            print(
                f"  Conclusion: ``_interp_center_to_corner`` "
                f"is essentially duogrid-invariant — bug "
                f"elsewhere."
            )
    assert np.isfinite(e_on) and np.isfinite(e_off)
