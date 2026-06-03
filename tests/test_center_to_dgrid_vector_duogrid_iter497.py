"""FV3_3D iter 497: ``center_to_dgrid_vector`` (iter-328
vector-aware halo interp) — does it amplify edges more or
less than ``_interp_center_to_corner`` under duogrid?

iter-496 found ``_interp_center_to_corner`` (4-point average
on stacked u/v) amplifies edge std × 1.122 under duogrid.
iter-328 added ``center_to_dgrid_vector`` (vector-aware halo
with rotation across face boundaries) — does this version
amplify more or less?

Methodology:
* Build random u_cc, v_cc on cubed sphere cell centers.
* Apply ``center_to_dgrid_vector`` with grid_off vs grid_on.
* Compare edge std × of u_d and v_d outputs.

Tests
-----

1. ``test_center_to_dgrid_vector_duogrid_amplification``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import center_to_dgrid_vector
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _edge_std(field_data):
    n_face, n_x, n_y, n_lev = field_data.shape
    edge_mask = np.zeros((n_x, n_y), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mask_b = np.broadcast_to(
        edge_mask[None, :, :, None], field_data.shape,
    )
    arr = np.asarray(field_data)
    return float(arr[edge_mask_b].std())


def _interior_std(field_data):
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
    return float(arr[interior_mask].std())


def test_center_to_dgrid_vector_duogrid_amplification(capsys):
    n = 8
    nlev = 5
    grid_on = create_cubed_sphere(n, use_duogrid=True)
    grid_off = create_cubed_sphere(n, use_duogrid=False)
    cdg_on = create_cubed_sphere_cdgrid(grid_on)
    cdg_off = create_cubed_sphere_cdgrid(grid_off)
    rng = np.random.default_rng(seed=497)
    u_cc = jnp.asarray(rng.normal(loc=0.0, scale=1.5, size=(6, n, n, nlev)))
    v_cc = jnp.asarray(rng.normal(loc=0.0, scale=1.5, size=(6, n, n, nlev)))
    u_d_off, v_d_off = center_to_dgrid_vector(u_cc, v_cc, cdg_off)
    u_d_on, v_d_on = center_to_dgrid_vector(u_cc, v_cc, cdg_on)
    ue_off, ui_off = _edge_std(u_d_off), _interior_std(u_d_off)
    ue_on, ui_on = _edge_std(u_d_on), _interior_std(u_d_on)
    ve_off, vi_off = _edge_std(v_d_off), _interior_std(v_d_off)
    ve_on, vi_on = _edge_std(v_d_on), _interior_std(v_d_on)
    with capsys.disabled():
        print(
            f"\n[iter-497 center_to_dgrid_vector duogrid effect]"
        )
        print(
            f"  u_d no-duogrid: edge std = {ue_off:.4f}, "
            f"interior = {ui_off:.4f}, ratio = "
            f"{ue_off/max(ui_off, 1e-30):.4f}"
        )
        print(
            f"  u_d duogrid:    edge std = {ue_on:.4f}, "
            f"interior = {ui_on:.4f}, ratio = "
            f"{ue_on/max(ui_on, 1e-30):.4f}"
        )
        u_edge_change = ue_on / max(ue_off, 1e-30)
        u_int_change = ui_on / max(ui_off, 1e-30)
        v_edge_change = ve_on / max(ve_off, 1e-30)
        v_int_change = vi_on / max(vi_off, 1e-30)
        print(
            f"  u_d effect: edge × {u_edge_change:.3f}, "
            f"interior × {u_int_change:.3f}"
        )
        print(
            f"  v_d effect: edge × {v_edge_change:.3f}, "
            f"interior × {v_int_change:.3f}"
        )
        print(
            f"  Compare iter-496 (scalar interp): "
            f"edge × 1.122"
        )
        max_change = max(u_edge_change, v_edge_change)
        if max_change > 1.5:
            print(
                f"  Conclusion: vector_halo amplifies MORE "
                f"than scalar — primary dycore amplifier."
            )
        elif max_change > 1.1:
            print(
                f"  Conclusion: vector_halo amplifies "
                f"COMPARABLY to scalar — both contribute."
            )
        else:
            print(
                f"  Conclusion: vector_halo less amplifying "
                f"than scalar."
            )
    assert all(np.isfinite([ue_on, ui_on, ve_on, vi_on]))
