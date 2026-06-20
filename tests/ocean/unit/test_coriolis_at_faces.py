"""Direct unit test for the shared coriolis_at_faces helper (#517 item 2).

Three lat-lon C-grid solvers (barotropic explicit/implicit + the full PE
step) reconstructed the face Coriolis with a byte-identical inline block.
This pins the factored helper:
- stored grid.f_u/f_v are returned verbatim (the shipping path),
- the reconstruct-from-grid.f fallback is bit-identical to the old inline
  formula,
- a tripolar/folded grid missing stored f_u/f_v RAISES instead of silently
  mis-reconstructing (the latent fold-sign footgun this dedup closes).
"""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.barotropic_common import coriolis_at_faces


def _reference_reconstruct(f_cell, dtype):
    """The exact inline formula the three sites used before #517."""
    f_cell = f_cell.astype(dtype)
    f_u = 0.5 * (jnp.roll(f_cell, 1, axis=1) + f_cell)
    f_u = jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)
    f_v_int = 0.5 * (f_cell[:-1] + f_cell[1:])
    f_v = jnp.concatenate([f_cell[0:1], f_v_int, f_cell[-1:]], axis=0)
    return f_u, f_v


def test_prefers_stored_face_metrics():
    """When f_u/f_v are stored, they are returned verbatim (cast only)."""
    f_u_stored = jnp.arange(6 * 9, dtype=jnp.float64).reshape(6, 9)   # (n_lat, n_lon+1)
    f_v_stored = jnp.arange(7 * 8, dtype=jnp.float64).reshape(7, 8)   # (n_lat+1, n_lon)
    grid = types.SimpleNamespace(f_u=f_u_stored, f_v=f_v_stored,
                                 f=jnp.zeros((6, 8)))
    f_u, f_v = coriolis_at_faces(grid, jnp.float32)
    assert f_u.dtype == jnp.float32 and f_v.dtype == jnp.float32
    assert np.array_equal(np.asarray(f_u), np.asarray(f_u_stored, np.float32))
    assert np.array_equal(np.asarray(f_v), np.asarray(f_v_stored, np.float32))


def test_reconstruct_is_bit_identical_to_old_inline():
    """Lean grid (only grid.f): fallback matches the pre-#517 formula exactly."""
    rng = np.random.default_rng(0)
    f_cell = jnp.asarray(rng.standard_normal((6, 8)), dtype=jnp.float64)
    grid = types.SimpleNamespace(f=f_cell)  # no f_u/f_v, no fold
    f_u, f_v = coriolis_at_faces(grid, jnp.float64)
    ref_u, ref_v = _reference_reconstruct(f_cell, jnp.float64)
    assert np.array_equal(np.asarray(f_u), np.asarray(ref_u))
    assert np.array_equal(np.asarray(f_v), np.asarray(ref_v))
    assert f_u.shape == (6, 9) and f_v.shape == (7, 8)


def test_folded_grid_without_stored_metrics_raises():
    """A folded grid that reaches the non-fold-aware fallback must raise,
    not silently produce wrong vorticity at the seam."""
    fold = types.SimpleNamespace(is_active=True)
    grid = types.SimpleNamespace(f=jnp.zeros((6, 8)), fold=fold)
    with pytest.raises(ValueError, match="fold-aware"):
        coriolis_at_faces(grid, jnp.float64)


def test_inactive_fold_still_reconstructs():
    """A grid carrying an INACTIVE fold descriptor (regular/Mercator) is not
    folded, so the fallback proceeds normally."""
    fold = types.SimpleNamespace(is_active=False)
    f_cell = jnp.ones((4, 5), dtype=jnp.float64)
    grid = types.SimpleNamespace(f=f_cell, fold=fold)
    f_u, f_v = coriolis_at_faces(grid, jnp.float64)
    ref_u, ref_v = _reference_reconstruct(f_cell, jnp.float64)
    assert np.array_equal(np.asarray(f_u), np.asarray(ref_u))
    assert np.array_equal(np.asarray(f_v), np.asarray(ref_v))
