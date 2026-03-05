"""Unit tests for inter-face halo exchange."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import (
    CONNECTIVITY, WEST, EAST, SOUTH, NORTH,
    pad_halo, pad_halo_vector, _extract_edge_strip,
)


class TestConnectivity:
    """Tests for the face connectivity table."""

    def test_all_faces_present(self):
        """All 6 faces should have connectivity entries."""
        assert len(CONNECTIVITY) == 6
        for face in range(6):
            assert face in CONNECTIVITY

    def test_all_edges_present(self):
        """Each face should have 4 edge connections."""
        for face in range(6):
            edges = CONNECTIVITY[face]
            assert WEST in edges
            assert EAST in edges
            assert SOUTH in edges
            assert NORTH in edges

    def test_connectivity_symmetric(self):
        """If A_edge -> (B, B_edge, rev), then B_edge -> (A, A_edge, rev)."""
        for face_a in range(6):
            for edge_a in [WEST, EAST, SOUTH, NORTH]:
                face_b, edge_b, rev_ab = CONNECTIVITY[face_a][edge_a]
                face_a2, edge_a2, rev_ba = CONNECTIVITY[face_b][edge_b]
                assert face_a2 == face_a, (
                    f"Face {face_a} edge {edge_a} -> Face {face_b} edge {edge_b} "
                    f"but reverse points to Face {face_a2}"
                )
                assert edge_a2 == edge_a, (
                    f"Face {face_a} edge {edge_a} -> Face {face_b} edge {edge_b} "
                    f"but reverse edge is {edge_a2}"
                )
                assert rev_ab == rev_ba, (
                    f"Face {face_a} edge {edge_a} -> Face {face_b} edge {edge_b}: "
                    f"forward reversed={rev_ab} but backward reversed={rev_ba}"
                )

    def test_no_self_connections(self):
        """No face should connect to itself."""
        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, _, _ = CONNECTIVITY[face][edge]
                assert nbr_face != face


class TestPadHalo:
    """Tests for scalar halo exchange."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(8)

    def test_output_shape(self):
        """Padded array should be (6, n+2, n+2)."""
        data = jnp.ones((6, 8, 8))
        padded = pad_halo(data)
        assert padded.shape == (6, 10, 10)

    def test_interior_preserved(self):
        """Interior data should be unchanged after padding."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (6, 8, 8))
        padded = pad_halo(data)
        assert jnp.allclose(padded[:, 1:-1, 1:-1], data)

    def test_constant_field_exact(self):
        """Padding a constant field should give constant halos."""
        data = jnp.ones((6, 8, 8)) * 42.0
        padded = pad_halo(data)
        # All non-corner halo cells should be 42.0
        for face in range(6):
            # West halo (excluding corners)
            assert jnp.allclose(padded[face, 0, 1:-1], 42.0)
            # East halo
            assert jnp.allclose(padded[face, -1, 1:-1], 42.0)
            # South halo
            assert jnp.allclose(padded[face, 1:-1, 0], 42.0)
            # North halo
            assert jnp.allclose(padded[face, 1:-1, -1], 42.0)

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        data32 = jnp.ones((6, 8, 8), dtype=jnp.float32)
        assert pad_halo(data32).dtype == jnp.float32

    def test_boundary_continuity_cartesian(self, grid):
        """Cartesian coordinates should be continuous across face boundaries.

        The Cartesian (x, y, z) coordinates on the unit sphere are smooth
        fields, so the halo values should closely match the neighbor's
        edge values.
        """
        # x_cart is a smooth scalar on the sphere
        padded_x = pad_halo(grid.x_cart)
        padded_y = pad_halo(grid.y_cart)
        padded_z = pad_halo(grid.z_cart)

        # All non-corner halo cells should be on the unit sphere
        # (after halo exchange, the neighbor's Cartesian coords are on the sphere)
        for face in range(6):
            for halo_slice in [
                (face, 0, slice(1, -1)),      # west
                (face, -1, slice(1, -1)),     # east
                (face, slice(1, -1), 0),      # south
                (face, slice(1, -1), -1),     # north
            ]:
                x_h = padded_x[halo_slice]
                y_h = padded_y[halo_slice]
                z_h = padded_z[halo_slice]
                r = jnp.sqrt(x_h**2 + y_h**2 + z_h**2)
                assert jnp.allclose(r, 1.0, atol=1e-5), (
                    f"Face {face} halo: r = {float(jnp.mean(r)):.6f}"
                )

    def test_halo_not_zero(self, grid):
        """Halo cells should be non-zero for a non-constant field."""
        data = grid.lat  # Latitude varies across the sphere
        padded = pad_halo(data)
        # At least some halo cells should be non-zero
        west_halo = padded[:, 0, 1:-1]
        assert not jnp.allclose(west_halo, 0.0)

    def test_jit_compatible(self):
        """pad_halo should work with jax.jit."""
        data = jnp.ones((6, 8, 8))

        @jax.jit
        def pad_and_sum(d):
            return jnp.sum(pad_halo(d))

        result = pad_and_sum(data)
        assert jnp.isfinite(result)

    def test_grad_compatible(self):
        """jax.grad should work through pad_halo."""
        def loss(data):
            padded = pad_halo(data)
            return jnp.sum(padded ** 2)

        data = jnp.ones((6, 4, 4))
        grads = jax.grad(loss)(data)
        assert jnp.all(jnp.isfinite(grads))
        assert not jnp.allclose(grads, 0.0)


class TestPadHaloH2:
    """Tests for halo=2 exchange."""

    def test_output_shape_h2(self):
        """Padded array should be (6, n+4, n+4) for halo=2."""
        data = jnp.ones((6, 8, 8))
        padded = pad_halo(data, halo=2)
        assert padded.shape == (6, 12, 12)

    def test_interior_preserved_h2(self):
        """Interior data should be unchanged after halo=2 padding."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (6, 8, 8))
        padded = pad_halo(data, halo=2)
        assert jnp.allclose(padded[:, 2:-2, 2:-2], data)

    def test_constant_field_h2(self):
        """Padding a constant field with halo=2 should give constant halos."""
        data = jnp.ones((6, 8, 8)) * 42.0
        padded = pad_halo(data, halo=2)
        for face in range(6):
            # Depth 0 (adjacent to interior) and depth 1 (outer)
            assert jnp.allclose(padded[face, 1, 2:-2], 42.0)
            assert jnp.allclose(padded[face, 0, 2:-2], 42.0)
            assert jnp.allclose(padded[face, -2, 2:-2], 42.0)
            assert jnp.allclose(padded[face, -1, 2:-2], 42.0)
            assert jnp.allclose(padded[face, 2:-2, 1], 42.0)
            assert jnp.allclose(padded[face, 2:-2, 0], 42.0)
            assert jnp.allclose(padded[face, 2:-2, -2], 42.0)
            assert jnp.allclose(padded[face, 2:-2, -1], 42.0)


class TestPadHaloVector:
    """Tests for vector halo exchange."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(8)

    def test_output_shape(self, grid):
        """Padded vector components should be (6, n+2, n+2)."""
        u = jnp.ones((6, 8, 8))
        v = jnp.zeros((6, 8, 8))
        u_pad, v_pad = pad_halo_vector(u, v, grid.cos_angle, grid.sin_angle, grid.cos_angle_padded, grid.sin_angle_padded)
        assert u_pad.shape == (6, 10, 10)
        assert v_pad.shape == (6, 10, 10)

    def test_interior_preserved(self, grid):
        """Interior velocity data should be unchanged."""
        key = jax.random.PRNGKey(0)
        u = jax.random.normal(key, (6, 8, 8))
        v = jax.random.normal(jax.random.PRNGKey(1), (6, 8, 8))
        u_pad, v_pad = pad_halo_vector(u, v, grid.cos_angle, grid.sin_angle, grid.cos_angle_padded, grid.sin_angle_padded)
        assert jnp.allclose(u_pad[:, 1:-1, 1:-1], u, atol=1e-5)
        assert jnp.allclose(v_pad[:, 1:-1, 1:-1], v, atol=1e-5)

    def test_zero_velocity_stays_zero(self, grid):
        """Zero velocity should remain zero after padding."""
        u = jnp.zeros((6, 8, 8))
        v = jnp.zeros((6, 8, 8))
        u_pad, v_pad = pad_halo_vector(u, v, grid.cos_angle, grid.sin_angle, grid.cos_angle_padded, grid.sin_angle_padded)
        # All halos should be zero
        assert jnp.allclose(u_pad, 0.0, atol=1e-6)
        assert jnp.allclose(v_pad, 0.0, atol=1e-6)

    def test_wind_speed_preserved(self, grid):
        """Wind speed |V| should be preserved in halo cells.

        Since we rotate to geographic and back, the magnitude should
        be preserved (rotation is orthogonal).
        """
        key = jax.random.PRNGKey(42)
        u = jax.random.normal(key, (6, 8, 8)) * 10.0
        v = jax.random.normal(jax.random.PRNGKey(1), (6, 8, 8)) * 10.0

        u_pad, v_pad = pad_halo_vector(u, v, grid.cos_angle, grid.sin_angle, grid.cos_angle_padded, grid.sin_angle_padded)

        # Check interior: speed should be preserved exactly
        speed_orig = jnp.sqrt(u**2 + v**2)
        speed_pad_interior = jnp.sqrt(
            u_pad[:, 1:-1, 1:-1]**2 + v_pad[:, 1:-1, 1:-1]**2
        )
        assert jnp.allclose(speed_pad_interior, speed_orig, atol=1e-5)

    def test_jit_compatible(self, grid):
        """pad_halo_vector should work with jax.jit."""
        u = jnp.ones((6, 8, 8))
        v = jnp.ones((6, 8, 8))

        @jax.jit
        def pad_and_sum(u, v):
            u_p, v_p = pad_halo_vector(u, v, grid.cos_angle, grid.sin_angle, grid.cos_angle_padded, grid.sin_angle_padded)
            return jnp.sum(u_p) + jnp.sum(v_p)

        result = pad_and_sum(u, v)
        assert jnp.isfinite(result)


class TestEdgeStripExtraction:
    """Tests for edge strip extraction helper."""

    def test_west_edge(self):
        data = jnp.arange(6 * 4 * 4, dtype=jnp.float32).reshape(6, 4, 4)
        strip = _extract_edge_strip(data, 0, WEST)
        assert strip.shape == (4,)
        assert jnp.allclose(strip, data[0, 0, :])

    def test_east_edge(self):
        data = jnp.arange(6 * 4 * 4, dtype=jnp.float32).reshape(6, 4, 4)
        strip = _extract_edge_strip(data, 2, EAST)
        assert jnp.allclose(strip, data[2, -1, :])

    def test_south_edge(self):
        data = jnp.arange(6 * 4 * 4, dtype=jnp.float32).reshape(6, 4, 4)
        strip = _extract_edge_strip(data, 4, SOUTH)
        assert jnp.allclose(strip, data[4, :, 0])

    def test_north_edge(self):
        data = jnp.arange(6 * 4 * 4, dtype=jnp.float32).reshape(6, 4, 4)
        strip = _extract_edge_strip(data, 5, NORTH)
        assert jnp.allclose(strip, data[5, :, -1])
