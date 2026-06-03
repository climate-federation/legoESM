"""FV3_3D iter-1076: staggered D-grid scalar halo tests.

Tests ``pad_halo_dgrid_scalar_4d`` from
``src/legoesm/grids/dgrid_halo.py``:

- Shape contract for staggered (n, n+1) and (n+1, n) inputs.
- Same-axis edges produce FV3-faithful cross-face halos
  bit-for-bit-correct (constant-per-face value → halo cells
  match the source face's constant).
- Axis-swap edges fall back to ``mode='edge'`` (replicate from
  adjacent interior cell) per iter-1076's documented limitation.
- Rejects square data (caller should use ``pad_halo_4d`` instead).

Run with::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \\
        tests/test_dgrid_halo_iter1076.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


def _build_constant_field(n_i: int, n_j: int, nlev: int = 1) -> jax.Array:
    """``data[f, i, j, k] = f + 1`` — distinct value per face."""
    data = jnp.zeros((6, n_i, n_j, nlev), dtype=jnp.float64)
    for f in range(6):
        data = data.at[f].set(float(f + 1))
    return data


class TestPadHaloDgridScalar4d:

    def test_u_d_shape_n_by_n_plus_1(self):
        """u_d shape (6, n, n+1, nlev) → padded (6, n+2, n+3, nlev)."""
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        n = 4
        u_d = _build_constant_field(n, n + 1)
        padded = pad_halo_dgrid_scalar_4d(u_d)
        assert padded.shape == (6, n + 2, n + 3, 1)

    def test_v_d_shape_n_plus_1_by_n(self):
        """v_d shape (6, n+1, n, nlev) → padded (6, n+3, n+2, nlev)."""
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        n = 4
        v_d = _build_constant_field(n + 1, n)
        padded = pad_halo_dgrid_scalar_4d(v_d)
        assert padded.shape == (6, n + 3, n + 2, 1)

    def test_rejects_square_data(self):
        """Square data is rejected — caller should use pad_halo_4d."""
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        n = 4
        data = _build_constant_field(n, n)
        with pytest.raises(ValueError) as exc:
            pad_halo_dgrid_scalar_4d(data)
        assert "NON-square" in str(exc.value)
        assert "pad_halo_4d" in str(exc.value)

    def test_rejects_3d_data(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        data = jnp.zeros((6, 4, 5))
        with pytest.raises(ValueError) as exc:
            pad_halo_dgrid_scalar_4d(data)
        assert "4D" in str(exc.value)

    def test_rejects_non_6_faces(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        data = jnp.zeros((3, 4, 5, 1))
        with pytest.raises(ValueError) as exc:
            pad_halo_dgrid_scalar_4d(data)
        assert "6 faces" in str(exc.value)

    def test_rejects_invalid_axis_swap_fill(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        data = jnp.zeros((6, 4, 5, 1))
        with pytest.raises(ValueError) as exc:
            pad_halo_dgrid_scalar_4d(data, axis_swap_fill="invalid")
        assert "axis_swap_fill" in str(exc.value)


class TestSameAxisEdgesFV3Faithful:
    """For non-axis-swap edges, halo cells contain the neighbor
    face's edge data exactly (constant-per-face → exact neighbor
    value)."""

    @pytest.fixture(params=[(4, 5), (5, 4), (3, 6), (6, 3)])
    def shape(self, request):
        return request.param

    def test_face_0_west_halo_is_face_3(self, shape):
        """Face 0 WEST → face 3 EAST (same-axis, equator-equator)."""
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        n_i, n_j = shape
        data = _build_constant_field(n_i, n_j)
        padded = pad_halo_dgrid_scalar_4d(data)
        # Halo cells at face 0's WEST = padded[0, 0, 1:n_j+1, :]
        # All these should equal face 3's value (4.0).
        np.testing.assert_array_equal(
            np.asarray(padded[0, 0, 1:n_j + 1, 0]),
            np.full(n_j, 4.0),
            err_msg="Face 0 WEST halo should mirror face 3 (=4).",
        )

    def test_face_0_east_halo_is_face_1(self, shape):
        """Face 0 EAST → face 1 WEST (same-axis, equator-equator)."""
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        n_i, n_j = shape
        data = _build_constant_field(n_i, n_j)
        padded = pad_halo_dgrid_scalar_4d(data)
        np.testing.assert_array_equal(
            np.asarray(padded[0, n_i + 1, 1:n_j + 1, 0]),
            np.full(n_j, 2.0),
            err_msg="Face 0 EAST halo should mirror face 1 (=2).",
        )

    def test_face_0_south_halo_is_face_5(self, shape):
        """Face 0 SOUTH → face 5 NORTH (same-axis)."""
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        n_i, n_j = shape
        data = _build_constant_field(n_i, n_j)
        padded = pad_halo_dgrid_scalar_4d(data)
        np.testing.assert_array_equal(
            np.asarray(padded[0, 1:n_i + 1, 0, 0]),
            np.full(n_i, 6.0),
            err_msg="Face 0 SOUTH halo should mirror face 5 (=6).",
        )

    def test_face_0_north_halo_is_face_4(self, shape):
        """Face 0 NORTH → face 4 SOUTH (same-axis)."""
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        n_i, n_j = shape
        data = _build_constant_field(n_i, n_j)
        padded = pad_halo_dgrid_scalar_4d(data)
        np.testing.assert_array_equal(
            np.asarray(padded[0, 1:n_i + 1, n_j + 1, 0]),
            np.full(n_i, 5.0),
            err_msg="Face 0 NORTH halo should mirror face 4 (=5).",
        )


class TestAxisSwapEdgesEdgeFallback:
    """At axis-swap edges (e.g., face 1 NORTH ↔ face 4 EAST), iter-1076
    falls back to ``mode='edge'`` (replicate adjacent interior cell).
    Face value is not mixed with the neighbor — caller should treat
    these halos as approximate."""

    def test_face_1_north_falls_back_to_edge(self):
        """Face 1 NORTH ↔ face 4 EAST is axis-swap; falls back to
        adjacent interior cell (face 1's own j=n-1 row)."""
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        n_i, n_j = 4, 5
        data = _build_constant_field(n_i, n_j)
        padded = pad_halo_dgrid_scalar_4d(data)
        # Face 1's NORTH halo: padded[1, 1:n_i+1, n_j+1, :]
        # With edge fallback: should equal face 1's row at j=n_j-1
        # (replicated outward), which is face 1's own value (=2).
        np.testing.assert_array_equal(
            np.asarray(padded[1, 1:n_i + 1, n_j + 1, 0]),
            np.full(n_i, 2.0),  # face 1's own value
            err_msg=(
                "Face 1 NORTH (axis-swap) should fall back to "
                "edge-replicate face 1's own value."
            ),
        )

    def test_face_4_east_falls_back_to_edge(self):
        """Face 4 EAST ↔ face 1 NORTH is axis-swap; falls back."""
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        n_i, n_j = 4, 5
        data = _build_constant_field(n_i, n_j)
        padded = pad_halo_dgrid_scalar_4d(data)
        # Face 4's EAST halo: padded[4, n_i+1, 1:n_j+1, :]
        np.testing.assert_array_equal(
            np.asarray(padded[4, n_i + 1, 1:n_j + 1, 0]),
            np.full(n_j, 5.0),  # face 4's own value
            err_msg=(
                "Face 4 EAST (axis-swap) should fall back to "
                "edge-replicate face 4's own value."
            ),
        )

    def test_axis_swap_fill_zero(self):
        """When ``axis_swap_fill='zero'``, axis-swap halos are zero."""
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_scalar_4d
        n_i, n_j = 4, 5
        data = _build_constant_field(n_i, n_j)
        padded = pad_halo_dgrid_scalar_4d(data, axis_swap_fill="zero")
        # Face 1 NORTH (axis-swap) halo should be zero.
        np.testing.assert_array_equal(
            np.asarray(padded[1, 1:n_i + 1, n_j + 1, 0]),
            np.zeros(n_i),
        )


class TestSameAxisClassification:
    """Verify the axis-swap classifier matches the documented 4 pairs."""

    def test_eight_directed_axis_swap_edges(self):
        from legoesm.grids.dgrid_halo import _is_axis_swap
        from legoesm.grids.halo import WEST, EAST, SOUTH, NORTH
        expected = {
            (1, SOUTH), (1, NORTH), (3, SOUTH), (3, NORTH),
            (4, WEST), (4, EAST), (5, WEST), (5, EAST),
        }
        actual = {
            (f, e)
            for f in range(6)
            for e in (WEST, EAST, SOUTH, NORTH)
            if _is_axis_swap(f, e)
        }
        assert actual == expected, (
            f"Axis-swap classifier off: extra={actual - expected}, "
            f"missing={expected - actual}"
        )

    def test_sixteen_directed_same_axis_edges(self):
        from legoesm.grids.dgrid_halo import _is_axis_swap
        from legoesm.grids.halo import WEST, EAST, SOUTH, NORTH
        same_axis_count = sum(
            1
            for f in range(6)
            for e in (WEST, EAST, SOUTH, NORTH)
            if not _is_axis_swap(f, e)
        )
        assert same_axis_count == 16, (
            f"Expected 16 same-axis directed edges (24 total - 8 "
            f"axis-swap); got {same_axis_count}."
        )
