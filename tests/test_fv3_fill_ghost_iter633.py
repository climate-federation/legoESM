"""FV3_3D iter 633: fill_ghost port.

Faithful JAX port of FV3 ``fill_ghost_r4``/``fill_ghost_r8``
(fv_grid_utils.F90:3070-3147).  Fills the 4 corner-ghost regions
outside the face corners with a constant.

Tests
-----

1. ``test_fill_ghost_shape_preserved``.
2. ``test_fill_ghost_interior_unchanged``.
3. ``test_fill_ghost_corners_set``.
4. ``test_fill_ghost_face_edges_unchanged``.
5. ``test_fill_ghost_value_zero``.
6. ``test_fill_ghost_3d_input``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import fill_ghost


def test_fill_ghost_shape_preserved():
    """Output shape matches input."""
    n = 12
    q = jnp.zeros((n, n))
    out = fill_ghost(q, ng=2, value=999.0)
    assert out.shape == (n, n)


def test_fill_ghost_interior_unchanged():
    """Interior cells (away from any corner ghost) are unchanged."""
    n = 12
    ng = 2
    rng = np.random.default_rng(seed=633)
    q = jnp.asarray(rng.normal(size=(n, n)))
    out = fill_ghost(q, ng=ng, value=999.0)
    # Interior: i ∈ [ng, n-ng-1], j ∈ [ng, n-ng-1]
    assert jnp.allclose(out[ng:-ng, ng:-ng], q[ng:-ng, ng:-ng], atol=1e-14)


def test_fill_ghost_corners_set():
    """All 4 corner-ghost regions are filled with the value."""
    n = 10
    ng = 2
    q = jnp.zeros((n, n))
    out = fill_ghost(q, ng=ng, value=42.0)
    # SW corner: (i < ng, j < ng) → (0:2, 0:2)
    assert jnp.allclose(out[:ng, :ng], 42.0)
    # SE corner: (i >= n-ng, j < ng) → (8:10, 0:2)
    assert jnp.allclose(out[-ng:, :ng], 42.0)
    # NE corner: (i >= n-ng, j >= n-ng) → (8:10, 8:10)
    assert jnp.allclose(out[-ng:, -ng:], 42.0)
    # NW corner: (i < ng, j >= n-ng) → (0:2, 8:10)
    assert jnp.allclose(out[:ng, -ng:], 42.0)


def test_fill_ghost_face_edges_unchanged():
    """Face-edge halo (only one of i/j outside interior) is NOT a corner
    ghost and stays unchanged."""
    n = 10
    ng = 2
    rng = np.random.default_rng(seed=634)
    q = jnp.asarray(rng.normal(size=(n, n)))
    out = fill_ghost(q, ng=ng, value=42.0)
    # i in halo but j in interior: (i < ng, ng <= j < n-ng)
    assert jnp.allclose(out[:ng, ng:-ng], q[:ng, ng:-ng], atol=1e-14)
    # i in interior, j in halo
    assert jnp.allclose(out[ng:-ng, :ng], q[ng:-ng, :ng], atol=1e-14)


def test_fill_ghost_value_zero():
    """value=0 is a valid (no-op-ish) fill — corners → 0."""
    n = 8
    ng = 1
    q = jnp.ones((n, n)) * 5.0
    out = fill_ghost(q, ng=ng, value=0.0)
    assert float(out[0, 0]) == 0.0
    assert float(out[-1, -1]) == 0.0


def test_fill_ghost_3d_input():
    """Leading axes (e.g., face index) are preserved."""
    n_faces = 6
    n = 8
    ng = 1
    rng = np.random.default_rng(seed=635)
    q = jnp.asarray(rng.normal(size=(n_faces, n, n)))
    out = fill_ghost(q, ng=ng, value=999.0)
    assert out.shape == (n_faces, n, n)
    # Corner check across all faces
    assert jnp.allclose(out[:, :ng, :ng], 999.0)
    assert jnp.allclose(out[:, -ng:, -ng:], 999.0)
    # Interior unchanged on all faces
    assert jnp.allclose(out[:, ng:-ng, ng:-ng], q[:, ng:-ng, ng:-ng], atol=1e-14)
