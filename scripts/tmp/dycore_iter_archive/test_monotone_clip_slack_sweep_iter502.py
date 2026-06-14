"""FV3_3D iter 502: sweep ``monotone_clip_slack`` values to
find the slack that gets ``center_to_dgrid_vector`` edge ×
≈ 1.000 (neutral, matching iter-499 over-correction analysis).

iter-499 strict clip: edge × 0.9378 (UNDER interior)
iter-497 no clip:     edge × 1.0396 (4% over interior)

Try slack ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}.  The neutral
slack is where edge × ≈ 1.0 (no NET amplification or
suppression).

Tests
-----

1. ``test_slack_sweep_in_center_to_dgrid_vector``.
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


def test_slack_sweep_in_center_to_dgrid_vector(capsys):
    n = 8
    nlev = 5
    grid_on = create_cubed_sphere(n, use_duogrid=True)
    grid_off = create_cubed_sphere(n, use_duogrid=False)
    cdg_on = create_cubed_sphere_cdgrid(grid_on)
    cdg_off = create_cubed_sphere_cdgrid(grid_off)
    rng = np.random.default_rng(seed=502)
    u_cc = jnp.asarray(rng.normal(loc=0.0, scale=1.5, size=(6, n, n, nlev)))
    v_cc = jnp.asarray(rng.normal(loc=0.0, scale=1.5, size=(6, n, n, nlev)))

    # Baseline no-clip (for off reference)
    u_d_off, _ = center_to_dgrid_vector(u_cc, v_cc, cdg_off)
    e_off = _edge_std(u_d_off)

    slack_values = [None, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    results = []
    for slack in slack_values:
        if slack is None:
            # No clip at all (default monotone_clip=False)
            u_d_on, _ = center_to_dgrid_vector(u_cc, v_cc, cdg_on)
            label = "no clip"
        else:
            clipped = functools.partial(
                _real_pad_halo_vec,
                monotone_clip=True,
                monotone_clip_slack=slack,
            )
            with patch(
                "legoesm.core.operators_cdgrid.pad_halo_vector_4d",
                clipped,
            ):
                u_d_on, _ = center_to_dgrid_vector(u_cc, v_cc, cdg_on)
            label = f"slack={slack:.2f}"
        e_on = _edge_std(u_d_on)
        ratio = e_on / max(e_off, 1e-30)
        results.append((label, ratio))
    with capsys.disabled():
        print(
            f"\n[iter-502 center_to_dgrid_vector slack sweep]"
        )
        for label, ratio in results:
            print(f"  {label:14s}: u_d edge × {ratio:.4f}")
        # Find slack closest to neutral (× 1.000)
        finite = [(l, r) for l, r in results if np.isfinite(r)]
        if finite:
            best = min(finite, key=lambda x: abs(x[1] - 1.0))
            print(f"\n  Closest to neutral (1.000): {best[0]} → {best[1]:.4f}")
    assert all(np.isfinite(r) for _, r in results)
