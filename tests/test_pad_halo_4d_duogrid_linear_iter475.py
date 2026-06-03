"""FV3_3D iter 475: continue iter-474 bisection — test
``pad_halo_4d`` with duogrid on a LINEAR-in-index field.

iter-474 ruled out scalar+constant: duogrid halo preserves
constants to 5e-13.  iter-475 tests scalar+linear: a field
``f(i,j) = i + j`` (linear in cube-face indices).  The
correct halo cells (continuous extension across face
boundaries) are well-defined but interpolation-dependent —
linear extrapolation should give clean values, while a
buggy interpolation will produce noise.

Compare duogrid ON vs duogrid OFF on the same linear input.
If duogrid halo gives MUCH larger deviation from
extrapolation than the interp_offsets path, that's the bug.

Tests
-----

1. ``test_pad_halo_4d_no_duogrid_linear_field``.
2. ``test_pad_halo_4d_duogrid_linear_field``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_4d


def _make_linear_field(n, nlev):
    """f(face, i, j, k) = i + j (independent of face / k)."""
    i_arr = jnp.arange(n).astype(jnp.float64)
    j_arr = jnp.arange(n).astype(jnp.float64)
    ij = i_arr[None, :, None, None] + j_arr[None, None, :, None]
    field = jnp.broadcast_to(ij, (6, n, n, nlev))
    return field


def test_pad_halo_4d_no_duogrid_linear_field(capsys):
    """Linear-in-index field with interp_offsets path.  Check
    max value in halo (no rigorous expected — just measure)."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=False)
    field = _make_linear_field(n, nlev)
    padded = pad_halo_4d(
        field,
        halo=1,
        interp_offsets=grid.halo_interp_offsets,
        duogrid=None,
    )
    arr = np.asarray(padded)
    # Interior should match input.
    interior = arr[:, 1:-1, 1:-1, :]
    np.testing.assert_allclose(
        np.asarray(field), interior, rtol=1e-13, atol=1e-13,
    )
    # Halo cells: measure max absolute value across all halo
    # rows + cols.
    halo_west = arr[:, 0, :, :]
    halo_east = arr[:, -1, :, :]
    halo_south = arr[:, :, 0, :]
    halo_north = arr[:, :, -1, :]
    halo_max = float(max(
        np.max(np.abs(halo_west)),
        np.max(np.abs(halo_east)),
        np.max(np.abs(halo_south)),
        np.max(np.abs(halo_north)),
    ))
    with capsys.disabled():
        print(
            f"\n[iter-475 NO duogrid linear-field halo max-abs]"
        )
        print(f"  interior max-abs: {float(np.max(np.abs(np.asarray(field)))):.4f}")
        print(f"  halo max-abs:     {halo_max:.4f}")
    assert np.isfinite(halo_max)


def test_pad_halo_4d_duogrid_linear_field(capsys):
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    field = _make_linear_field(n, nlev)
    padded = pad_halo_4d(
        field, halo=1, interp_offsets=None, duogrid=grid.duogrid,
    )
    arr = np.asarray(padded)
    interior = arr[:, 1:-1, 1:-1, :]
    np.testing.assert_allclose(
        np.asarray(field), interior, rtol=1e-13, atol=1e-13,
    )
    halo_west = arr[:, 0, :, :]
    halo_east = arr[:, -1, :, :]
    halo_south = arr[:, :, 0, :]
    halo_north = arr[:, :, -1, :]
    halo_max = float(max(
        np.max(np.abs(halo_west)),
        np.max(np.abs(halo_east)),
        np.max(np.abs(halo_south)),
        np.max(np.abs(halo_north)),
    ))
    with capsys.disabled():
        print(
            f"\n[iter-475 WITH duogrid linear-field halo max-abs]"
        )
        print(f"  interior max-abs: {float(np.max(np.abs(np.asarray(field)))):.4f}")
        print(f"  halo max-abs:     {halo_max:.4f}")
    assert np.isfinite(halo_max)
