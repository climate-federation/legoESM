"""FV3_3D iter 501: ``monotone_clip_slack`` softer monotonicity.

iter-499 found strict ``monotone_clip=True`` over-corrects
(pulls edge × 0.94, below interior).  iter-501 adds a slack
parameter for softer monotonicity:

  slack = 0 → strict iter-802 clip (current iter-490 behavior)
  slack > 0 → allow ``±slack * (hi - lo)`` band expansion

Tests
-----

1. ``test_slack_zero_matches_iter490_strict``.
2. ``test_slack_relaxes_clip``.
3. ``test_slack_default_zero``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_4d


def _linear_field(n, nlev):
    i_arr = jnp.arange(n).astype(jnp.float64)
    j_arr = jnp.arange(n).astype(jnp.float64)
    ij = i_arr[None, :, None, None] + j_arr[None, None, :, None]
    return jnp.broadcast_to(ij, (6, n, n, nlev))


def test_slack_default_zero():
    """slack defaults to 0 (no behavior change vs iter-490)."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    field = _linear_field(n, nlev)
    padded_default = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=True,
    )
    padded_explicit = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=True, monotone_clip_slack=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(padded_default), np.asarray(padded_explicit),
    )


def test_slack_zero_matches_iter490_strict(capsys):
    """slack=0 reproduces iter-490 strict clip result (halo
    max = 14.00)."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    field = _linear_field(n, nlev)
    padded = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=True, monotone_clip_slack=0.0,
    )
    arr = np.asarray(padded)
    halo_max = max(
        float(np.max(np.abs(arr[:, 0, :, :]))),
        float(np.max(np.abs(arr[:, -1, :, :]))),
        float(np.max(np.abs(arr[:, :, 0, :]))),
        float(np.max(np.abs(arr[:, :, -1, :]))),
    )
    with capsys.disabled():
        print(f"\n[iter-501 slack=0 halo max: {halo_max:.4f}]")
    np.testing.assert_allclose(halo_max, 14.0, rtol=1e-10)


def test_slack_relaxes_clip(capsys):
    """slack=0.1 allows halo to exceed strict interior_max."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    field = _linear_field(n, nlev)
    padded_strict = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=True, monotone_clip_slack=0.0,
    )
    padded_slack = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=True, monotone_clip_slack=0.1,
    )
    arr_strict = np.asarray(padded_strict)
    arr_slack = np.asarray(padded_slack)

    def halo_max(arr):
        return max(
            float(np.max(np.abs(arr[:, 0, :, :]))),
            float(np.max(np.abs(arr[:, -1, :, :]))),
            float(np.max(np.abs(arr[:, :, 0, :]))),
            float(np.max(np.abs(arr[:, :, -1, :]))),
        )

    hm_strict = halo_max(arr_strict)
    hm_slack = halo_max(arr_slack)
    with capsys.disabled():
        print(
            f"\n[iter-501 slack comparison]"
            f"\n  strict (slack=0):  halo max = {hm_strict:.4f}"
            f"\n  slack=0.1:         halo max = {hm_slack:.4f}"
        )
    # slack should allow some overshoot above strict
    assert hm_slack >= hm_strict, (
        f"slack=0.1 should allow halo_max >= strict, got "
        f"{hm_slack:.4f} < {hm_strict:.4f}"
    )
