"""FV3_3D iter 490: opt-in ``monotone_clip`` arg for
``pad_halo_4d`` propagates to ``fill_corner_region`` to fix
iter-489's confined-to-corner overshoot.

iter-489: 3.5% overshoot CONFINED to cube-vertex cells.
iter-802 already added a ``monotone_clip`` knob to
``fill_corner_region`` but it wasn't exposed via
``pad_halo_4d``.  iter-490 plumbs the arg through.

Tests
-----

1. ``test_default_monotone_clip_false`` — default off,
   matches existing iter-475 result (3.5% overshoot).
2. ``test_monotone_clip_eliminates_corner_overshoot`` —
   with clip=True, no cell > interior max.
3. ``test_monotone_clip_preserves_constant`` — clip doesn't
   break iter-474's constant test.
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


def test_default_monotone_clip_false():
    """Default monotone_clip=False reproduces iter-475 result
    (max halo > interior max for duogrid)."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    field = _linear_field(n, nlev)
    padded = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
    )
    arr = np.asarray(padded)
    interior_max = float(np.max(np.abs(np.asarray(field))))
    halo_max = max(
        float(np.max(np.abs(arr[:, 0, :, :]))),
        float(np.max(np.abs(arr[:, -1, :, :]))),
        float(np.max(np.abs(arr[:, :, 0, :]))),
        float(np.max(np.abs(arr[:, :, -1, :]))),
    )
    # Should overshoot (iter-475 found ~14.49 vs 14.0)
    assert halo_max > interior_max, (
        f"Default (clip=False) should reproduce iter-475 "
        f"overshoot; got halo_max={halo_max:.4f} ≤ "
        f"interior_max={interior_max:.4f}."
    )


def test_monotone_clip_eliminates_corner_overshoot(capsys):
    """monotone_clip=True clips corner overshoot."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    field = _linear_field(n, nlev)
    padded = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=True,
    )
    arr = np.asarray(padded)
    interior_max = float(np.max(np.abs(np.asarray(field))))
    halo_max = max(
        float(np.max(np.abs(arr[:, 0, :, :]))),
        float(np.max(np.abs(arr[:, -1, :, :]))),
        float(np.max(np.abs(arr[:, :, 0, :]))),
        float(np.max(np.abs(arr[:, :, -1, :]))),
    )
    with capsys.disabled():
        print(
            f"\n[iter-490 monotone_clip linear-field test]"
        )
        print(f"  interior max: {interior_max:.4f}")
        print(f"  halo max:     {halo_max:.4f}")
        print(f"  ratio:        {halo_max / interior_max:.4f}")
    # With clip, halo should be ≤ interior_max (modulo numerical
    # roundoff).
    assert halo_max <= interior_max + 1e-10, (
        f"monotone_clip=True did NOT prevent overshoot: "
        f"halo_max={halo_max:.4f} > interior_max="
        f"{interior_max:.4f}."
    )


def test_monotone_clip_preserves_constant():
    """monotone_clip=True still preserves constants exactly
    (iter-474 invariance)."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    field = jnp.full((6, n, n, nlev), 273.15)
    padded = pad_halo_4d(
        field, halo=1, duogrid=grid.duogrid,
        monotone_clip=True,
    )
    np.testing.assert_allclose(
        np.asarray(padded), 273.15,
        rtol=1e-12, atol=1e-12,
    )
