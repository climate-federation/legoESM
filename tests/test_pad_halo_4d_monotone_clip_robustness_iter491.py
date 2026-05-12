"""FV3_3D iter 491: extend iter-490 monotone_clip robustness
to random + large-magnitude fields.

iter-490 verified clip works on constant + linear fields.
iter-491 tests:
* Random Gaussian field with mean ≠ 0
* Large-magnitude field (1e6 scale)
* Mixed-sign field

Tests
-----

1. ``test_monotone_clip_random_field_preserves_bound``.
2. ``test_monotone_clip_large_magnitude``.
3. ``test_monotone_clip_mixed_sign``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_4d


def _halo_max_abs(arr):
    return max(
        float(np.max(np.abs(arr[:, 0, :, :]))),
        float(np.max(np.abs(arr[:, -1, :, :]))),
        float(np.max(np.abs(arr[:, :, 0, :]))),
        float(np.max(np.abs(arr[:, :, -1, :]))),
    )


def test_monotone_clip_random_field_preserves_bound(capsys):
    """Random Gaussian field: halo max ≤ interior max under clip."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    rng = np.random.default_rng(seed=491)
    field = jnp.asarray(
        rng.normal(loc=2.0, scale=1.5, size=(6, n, n, nlev)),
    )
    # No clip
    padded_off = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=False,
    )
    # With clip
    padded_on = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=True,
    )
    interior_max = float(np.max(np.abs(np.asarray(field))))
    halo_max_off = _halo_max_abs(np.asarray(padded_off))
    halo_max_on = _halo_max_abs(np.asarray(padded_on))
    with capsys.disabled():
        print(
            f"\n[iter-491 random field, n=8 nlev=5]"
        )
        print(f"  interior max-abs:    {interior_max:.4f}")
        print(f"  halo max (no clip):  {halo_max_off:.4f}")
        print(f"  halo max (clip on):  {halo_max_on:.4f}")
    # Clip should bound halo to ≤ interior max (allowing
    # numerical roundoff for fields that include the max
    # at edge cells).
    assert halo_max_on <= interior_max + 1e-9, (
        f"clip=True failed to bound: halo_max_on={halo_max_on:.4f}"
        f" > interior_max={interior_max:.4f}."
    )


def test_monotone_clip_large_magnitude():
    """Field with values O(1e6) — clip should still work."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    rng = np.random.default_rng(seed=492)
    field = jnp.asarray(
        rng.uniform(-1e6, 1e6, size=(6, n, n, nlev)),
    )
    padded = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=True,
    )
    arr = np.asarray(padded)
    interior_max = float(np.max(np.abs(np.asarray(field))))
    halo_max = _halo_max_abs(arr)
    assert halo_max <= interior_max + 1e-3, (
        f"large-magnitude clip failed: halo={halo_max:.2e}, "
        f"interior={interior_max:.2e}."
    )


def test_monotone_clip_mixed_sign():
    """Mixed-sign field — clip should preserve both ends."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    rng = np.random.default_rng(seed=493)
    field = jnp.asarray(
        rng.uniform(-5.0, 5.0, size=(6, n, n, nlev)),
    )
    padded = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=True,
    )
    arr = np.asarray(padded)
    field_np = np.asarray(field)
    interior_min = float(np.min(field_np))
    interior_max = float(np.max(field_np))
    # Halo cells should be within [interior_min, interior_max]
    # (modulo 1e-9 roundoff)
    halo_cells = np.concatenate([
        arr[:, 0, :, :].ravel(),
        arr[:, -1, :, :].ravel(),
        arr[:, :, 0, :].ravel(),
        arr[:, :, -1, :].ravel(),
    ])
    halo_min = float(np.min(halo_cells))
    halo_max = float(np.max(halo_cells))
    assert halo_min >= interior_min - 1e-9, (
        f"halo_min {halo_min:.4f} < interior_min {interior_min:.4f}"
    )
    assert halo_max <= interior_max + 1e-9, (
        f"halo_max {halo_max:.4f} > interior_max {interior_max:.4f}"
    )
