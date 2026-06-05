"""FV3_3D iter 499: does iter-498's vector-halo clip reduce
iter-497's 4% vector-interp residual amplification?

iter-497: center_to_dgrid_vector edge × 1.040 / 1.027
iter-498: vector halo clip cuts random overshoot 55.6%
iter-499: monkey-patch pad_halo_vector_4d with clip=True
inside center_to_dgrid_vector and re-measure.

Tests
-----

1. ``test_center_to_dgrid_vector_with_clip``.
"""
from __future__ import annotations

from unittest.mock import patch
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import center_to_dgrid_vector
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import pad_halo_vector_4d as _real_pad_halo_vec


_clipped_pad_halo_vec = functools.partial(
    _real_pad_halo_vec, monotone_clip=True,
)


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


def test_center_to_dgrid_vector_with_clip(capsys):
    n = 8
    nlev = 5
    grid_on = create_cubed_sphere(n, use_duogrid=True)
    grid_off = create_cubed_sphere(n, use_duogrid=False)
    cdg_on = create_cubed_sphere_cdgrid(grid_on)
    cdg_off = create_cubed_sphere_cdgrid(grid_off)
    rng = np.random.default_rng(seed=499)
    u_cc = jnp.asarray(rng.normal(loc=0.0, scale=1.5, size=(6, n, n, nlev)))
    v_cc = jnp.asarray(rng.normal(loc=0.0, scale=1.5, size=(6, n, n, nlev)))

    # Baseline: no clip
    u_d_off, _ = center_to_dgrid_vector(u_cc, v_cc, cdg_off)
    u_d_on, _ = center_to_dgrid_vector(u_cc, v_cc, cdg_on)
    ue_off, ui_off = _edge_std(u_d_off), _interior_std(u_d_off)
    ue_on, ui_on = _edge_std(u_d_on), _interior_std(u_d_on)
    edge_change_baseline = ue_on / max(ue_off, 1e-30)

    # Patched: pad_halo_vector_4d uses clip=True
    with patch(
        "legoesm.core.operators_cdgrid.pad_halo_vector_4d",
        _clipped_pad_halo_vec,
    ):
        u_d_off_c, _ = center_to_dgrid_vector(u_cc, v_cc, cdg_off)
        u_d_on_c, _ = center_to_dgrid_vector(u_cc, v_cc, cdg_on)
        ue_off_c, ui_off_c = _edge_std(u_d_off_c), _interior_std(u_d_off_c)
        ue_on_c, ui_on_c = _edge_std(u_d_on_c), _interior_std(u_d_on_c)
        edge_change_clipped = ue_on_c / max(ue_off_c, 1e-30)

    with capsys.disabled():
        print(
            f"\n[iter-499 center_to_dgrid_vector + clip]"
        )
        print(
            f"  No clip:   u_d edge × {edge_change_baseline:.4f}  "
            f"(iter-497 baseline 1.040)"
        )
        print(
            f"  With clip: u_d edge × {edge_change_clipped:.4f}"
        )
        if edge_change_clipped < edge_change_baseline:
            reduction = (
                (edge_change_baseline - edge_change_clipped)
                / max(edge_change_baseline - 1.0, 1e-30) * 100
            )
            print(
                f"  Clip reduced edge amplification "
                f"{reduction:.0f}% of the way to neutral."
            )
        else:
            print(f"  No improvement from clip.")
    assert np.isfinite(edge_change_baseline)
    assert np.isfinite(edge_change_clipped)
