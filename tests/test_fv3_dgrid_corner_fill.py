"""Unit tests for FV3-faithful D-grid vector and A-grid scalar
cube-vertex corner fill.

Validates the JAX ports of FV3 ``fill_corners_dgrid`` and
``fill_corners_agrid`` (``../FV3/atmos_cubed_sphere-symmetryclean/
tools/fv_mp_mod.F90``).

Tests
-----

1. ``fv3_fill_corners_agrid_scalar`` overwrites EXACTLY the 4 cube-
   vertex halo cells per face, leaves all others alone.
2. The diagonal-mirror values match the FV3 indexing pattern
   (sourced from the cell at the diagonal-flipped index).
3. ``fv3_fill_corners_dgrid_vector`` overwrites EXACTLY the 4 cube-
   vertex halo cells per face for both x and y components.
4. The Fortran sign pattern is correctly applied: SW and NE
   corners flip sign, NW and SE corners do not.
5. Symmetry: applying the fill to a uniform input gives the
   uniform value back at the cube-vertex halo cells (after sign
   correction for SW/NE on the vector case).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids._fv3_dgrid_corner_fill import (
    fv3_fill_corners_agrid_scalar,
    fv3_fill_corners_cdgrid_vector,
    fv3_fill_corners_dgrid_vector,
)


def test_agrid_scalar_overwrites_only_cube_vertices():
    n = 6
    rng = np.random.default_rng(seed=42)
    q_np = rng.uniform(-1.0, 1.0, size=(6, n + 2, n + 2))
    q = jnp.asarray(q_np)
    q_filled = fv3_fill_corners_agrid_scalar(q, n)
    diff = np.asarray(q_filled) - q_np

    # Only the 4 cube-vertex halo cells per face should be modified.
    expected_modified = np.zeros((6, n + 2, n + 2), dtype=bool)
    expected_modified[:, 0, 0] = True       # SW
    expected_modified[:, 0, -1] = True      # NW
    expected_modified[:, -1, 0] = True      # SE
    expected_modified[:, -1, -1] = True     # NE

    actually_modified = diff != 0.0
    # All "should be modified" cells should differ (probabilistically
    # they should differ for random input).
    np.testing.assert_array_equal(actually_modified, expected_modified)


def test_agrid_scalar_uses_diagonal_mirror():
    n = 6
    rng = np.random.default_rng(seed=42)
    q_np = rng.uniform(-1.0, 1.0, size=(6, n + 2, n + 2))
    q = jnp.asarray(q_np)
    q_filled = fv3_fill_corners_agrid_scalar(q, n)
    out = np.asarray(q_filled)

    # SW corner (0, 0) should equal source at (0, 1).
    np.testing.assert_array_equal(out[:, 0, 0], q_np[:, 0, 1])
    # NW corner (0, -1) should equal source at (0, -2).
    np.testing.assert_array_equal(out[:, 0, -1], q_np[:, 0, -2])
    # SE corner (-1, 0) should equal source at (-1, 1).
    np.testing.assert_array_equal(out[:, -1, 0], q_np[:, -1, 1])
    # NE corner (-1, -1) should equal source at (-1, -2).
    np.testing.assert_array_equal(out[:, -1, -1], q_np[:, -1, -2])


def test_dgrid_vector_overwrites_only_cube_vertices():
    n = 6
    rng = np.random.default_rng(seed=42)
    x_np = rng.uniform(-1.0, 1.0, size=(6, n + 2, n + 3))
    y_np = rng.uniform(-1.0, 1.0, size=(6, n + 3, n + 2))
    x_filled, y_filled = fv3_fill_corners_dgrid_vector(
        jnp.asarray(x_np), jnp.asarray(y_np), n,
    )
    out_x = np.asarray(x_filled)
    out_y = np.asarray(y_filled)

    expected_modified_x = np.zeros((6, n + 2, n + 3), dtype=bool)
    expected_modified_x[:, 0, 0] = True
    expected_modified_x[:, 0, -1] = True
    expected_modified_x[:, -1, 0] = True
    expected_modified_x[:, -1, -1] = True

    expected_modified_y = np.zeros((6, n + 3, n + 2), dtype=bool)
    expected_modified_y[:, 0, 0] = True
    expected_modified_y[:, 0, -1] = True
    expected_modified_y[:, -1, 0] = True
    expected_modified_y[:, -1, -1] = True

    actually_modified_x = (out_x - x_np) != 0.0
    actually_modified_y = (out_y - y_np) != 0.0
    np.testing.assert_array_equal(actually_modified_x, expected_modified_x)
    np.testing.assert_array_equal(actually_modified_y, expected_modified_y)


def test_dgrid_vector_sign_flip_pattern():
    """SW and NE corners flip sign; NW and SE corners do not.

    Per FV3 fv_mp_mod.F90 line 1270-1273 (and 1282-1285 for the y
    component).  Use a constant input so the sign is unambiguous.
    """
    n = 6
    x = jnp.full((6, n + 2, n + 3), 1.0)
    y = jnp.full((6, n + 3, n + 2), 1.0)
    x_filled, y_filled = fv3_fill_corners_dgrid_vector(x, y, n)
    out_x = np.asarray(x_filled)
    out_y = np.asarray(y_filled)

    # SW (0, 0): x ← -y(0, 1) = -1.  y ← -x(1, 0) = -(after x update) = -(-1) = 1???
    # Actually we read the post-update x to set y.  But the x at (1, 0)
    # is NOT modified by the SW corner update (which writes only x[0,0]).
    # So x[1, 0] = 1 (untouched).  y[0, 0] ← -x[1, 0] = -1.
    np.testing.assert_array_equal(out_x[:, 0, 0], -np.ones(6))
    np.testing.assert_array_equal(out_y[:, 0, 0], -np.ones(6))

    # NW (0, -1): no sign flip.  x ← y(0, -2) = 1; y ← x(1, -1) = 1
    # (NB x[0, -1] is set first to y[0, -2]=1 before y read).
    np.testing.assert_array_equal(out_x[:, 0, -1], np.ones(6))
    np.testing.assert_array_equal(out_y[:, 0, -1], np.ones(6))

    # SE (-1, 0): no sign flip.
    np.testing.assert_array_equal(out_x[:, -1, 0], np.ones(6))
    np.testing.assert_array_equal(out_y[:, -1, 0], np.ones(6))

    # NE (-1, -1): sign flip.
    np.testing.assert_array_equal(out_x[:, -1, -1], -np.ones(6))
    np.testing.assert_array_equal(out_y[:, -1, -1], -np.ones(6))


def test_dgrid_zero_input_stays_zero():
    """Zero input → zero output, regardless of sign pattern."""
    n = 6
    x = jnp.zeros((6, n + 2, n + 3))
    y = jnp.zeros((6, n + 3, n + 2))
    x_filled, y_filled = fv3_fill_corners_dgrid_vector(x, y, n)
    np.testing.assert_array_equal(x_filled, np.zeros_like(x_filled))
    np.testing.assert_array_equal(y_filled, np.zeros_like(y_filled))


def test_agrid_uniform_constant_is_invariant():
    """A uniform scalar field is unchanged by the diagonal mirror."""
    n = 6
    q = jnp.full((6, n + 2, n + 2), 7.5)
    q_filled = fv3_fill_corners_agrid_scalar(q, n)
    np.testing.assert_array_equal(q_filled, q)


def test_cdgrid_vector_overwrites_only_cube_vertices():
    """C-D grid (n+1, n+1) variant: only the 4 cube-vertex halo cells
    per face are modified."""
    n = 6
    rng = np.random.default_rng(seed=11)
    u_np = rng.uniform(-1.0, 1.0, size=(6, n + 3, n + 3))
    v_np = rng.uniform(-1.0, 1.0, size=(6, n + 3, n + 3))
    u_filled, v_filled = fv3_fill_corners_cdgrid_vector(
        jnp.asarray(u_np), jnp.asarray(v_np), n,
    )
    out_u = np.asarray(u_filled)
    out_v = np.asarray(v_filled)

    expected = np.zeros((6, n + 3, n + 3), dtype=bool)
    expected[:, 0, 0] = True
    expected[:, 0, -1] = True
    expected[:, -1, 0] = True
    expected[:, -1, -1] = True

    np.testing.assert_array_equal((out_u - u_np) != 0.0, expected)
    np.testing.assert_array_equal((out_v - v_np) != 0.0, expected)


def test_cdgrid_vector_sign_pattern():
    """SW and NE flip sign; NW and SE do not (FV3 fv_mp_mod.F90:1270-1273
    pattern carried over to the C-D grid layout)."""
    n = 6
    u = jnp.full((6, n + 3, n + 3), 1.0)
    v = jnp.full((6, n + 3, n + 3), 2.0)
    u_filled, v_filled = fv3_fill_corners_cdgrid_vector(u, v, n)
    out_u = np.asarray(u_filled)
    out_v = np.asarray(v_filled)

    # SW (0, 0): sign flip — u ← -v_at_(1,1) = -2; v ← -u_at_(1,1) = -1.
    np.testing.assert_array_equal(out_u[:, 0, 0], -2.0 * np.ones(6))
    np.testing.assert_array_equal(out_v[:, 0, 0], -1.0 * np.ones(6))

    # NW (0, -1): no sign — u ← v_at_(1,-2) = 2; v ← u_at_(1,-2) = 1.
    np.testing.assert_array_equal(out_u[:, 0, -1], 2.0 * np.ones(6))
    np.testing.assert_array_equal(out_v[:, 0, -1], 1.0 * np.ones(6))

    # SE (-1, 0): no sign.
    np.testing.assert_array_equal(out_u[:, -1, 0], 2.0 * np.ones(6))
    np.testing.assert_array_equal(out_v[:, -1, 0], 1.0 * np.ones(6))

    # NE (-1, -1): sign flip.
    np.testing.assert_array_equal(out_u[:, -1, -1], -2.0 * np.ones(6))
    np.testing.assert_array_equal(out_v[:, -1, -1], -1.0 * np.ones(6))


def test_cdgrid_vector_with_trailing_axis():
    """Vertical-level trailing axis is passed through correctly."""
    n = 5
    nlev = 4
    rng = np.random.default_rng(seed=99)
    u_np = rng.uniform(-1.0, 1.0, size=(6, n + 3, n + 3, nlev))
    v_np = rng.uniform(-1.0, 1.0, size=(6, n + 3, n + 3, nlev))
    u_filled, v_filled = fv3_fill_corners_cdgrid_vector(
        jnp.asarray(u_np), jnp.asarray(v_np), n,
    )
    assert u_filled.shape == (6, n + 3, n + 3, nlev)
    assert v_filled.shape == (6, n + 3, n + 3, nlev)
    # Verify the cube-vertex halo SW cell across all levels: u ← -v_at_(1,1).
    np.testing.assert_array_equal(
        np.asarray(u_filled[:, 0, 0, :]), -v_np[:, 1, 1, :],
    )
