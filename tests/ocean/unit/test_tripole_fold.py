"""Tests for tripolar fold halo exchange.

Validates that the fold permutation in pad_ns_scalar / pad_ns_vector
correctly mirrors data across the northern boundary of a tripolar grid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    pad_ns_scalar,
    pad_ns_vector_u,
    pad_ns_vector_v,
    gradient_y_cgrid,
    divergence_cgrid,
    curl_vertex_cgrid,
    coriolis_cgrid,
    laplacian_cgrid,
)


@pytest.fixture
def tripole_grid():
    """Create a synthetic tripolar grid for testing."""
    return create_synthetic_tripole(36, 72)


@pytest.fixture
def regular_grid():
    """Create a regular lat-lon grid for comparison."""
    from legoesm.grids.latlon import create_latlon_geometry
    return create_latlon_geometry(36, 72)


# =========================================================================
# Fold round-trip tests
# =========================================================================


class TestFoldRoundTrip:
    """Verify the fold permutation is correct."""

    def test_scalar_fold_is_i_reversed(self, tripole_grid):
        """The fold at the north boundary should reverse the i-index."""
        n_lat = tripole_grid.n_lat
        n_lon = tripole_grid.n_lon

        # Create a known pattern: T[j, i] = j * 1000 + i
        interior = jnp.arange(n_lon, dtype=jnp.float32)[jnp.newaxis, :]
        interior = interior + jnp.arange(n_lat - 1, dtype=jnp.float32)[:, jnp.newaxis] * 1000

        padded = pad_ns_scalar(interior, tripole_grid)
        assert padded.shape == (n_lat + 1, n_lon)

        # South boundary should be zero (wall)
        assert jnp.all(padded[0] == 0.0)

        # North boundary should be fold-reflected: last interior row, i-reversed
        last_row = interior[-1]  # the row closest to the fold
        expected_north = last_row[::-1]  # i-reversed
        assert jnp.allclose(padded[-1], expected_north), (
            f"North fold mismatch: got {padded[-1][:5]}, "
            f"expected {expected_north[:5]}"
        )

    def test_vector_u_fold_has_sign_flip(self, tripole_grid):
        """u-component across fold should be sign-reversed and i-reversed."""
        n_lat = tripole_grid.n_lat
        n_lon = tripole_grid.n_lon

        interior = jnp.ones((n_lat - 1, n_lon), dtype=jnp.float32)
        padded = pad_ns_vector_u(interior, tripole_grid)

        # South = zero
        assert jnp.all(padded[0] == 0.0)

        # North = sign-flipped
        assert jnp.allclose(padded[-1], -1.0), (
            f"Expected -1.0 at fold, got {padded[-1, 0]}"
        )

    def test_vector_v_fold_has_sign_flip(self, tripole_grid):
        """v-component across fold should be sign-reversed and i-reversed."""
        n_lat = tripole_grid.n_lat
        n_lon = tripole_grid.n_lon

        interior = jnp.ones((n_lat - 1, n_lon), dtype=jnp.float32)
        padded = pad_ns_vector_v(interior, tripole_grid)

        # North = sign-flipped
        assert jnp.allclose(padded[-1], -1.0)

    def test_fold_permutation_is_involution(self, tripole_grid):
        """Applying the fold twice should give the identity."""
        fold = tripole_grid.fold
        perm = fold.perm_T

        # perm[perm[i]] == i for all i
        roundtrip = perm[perm]
        assert jnp.all(roundtrip == jnp.arange(tripole_grid.n_lon))

    def test_scalar_fold_3d(self, tripole_grid):
        """Fold should work on 3D fields (lat, lon, lev)."""
        n_lat = tripole_grid.n_lat
        n_lon = tripole_grid.n_lon
        nlev = 5

        interior = jax.random.normal(
            jax.random.PRNGKey(42), (n_lat - 1, n_lon, nlev)
        )
        padded = pad_ns_scalar(interior, tripole_grid)
        assert padded.shape == (n_lat + 1, n_lon, nlev)

        # South = zero
        assert jnp.all(padded[0] == 0.0)

        # North = fold-reflected last row
        fold = tripole_grid.fold
        expected = interior[-1][fold.perm_T]
        assert jnp.allclose(padded[-1], expected)


# =========================================================================
# Operator tests on tripolar grid
# =========================================================================


class TestOperatorsOnTripole:
    """Verify operators produce finite results on a synthetic tripolar grid."""

    def test_gradient_y_finite(self, tripole_grid):
        """gradient_y should produce finite results on tripolar grid."""
        f = jax.random.normal(jax.random.PRNGKey(0), (36, 72))
        result = gradient_y_cgrid(f, tripole_grid)
        assert result.shape == (37, 72)
        assert jnp.all(jnp.isfinite(result))
        # North boundary should be nonzero (fold gives data, not wall)
        assert not jnp.all(result[-1] == 0.0), (
            "North boundary of gradient_y should be nonzero with fold"
        )

    def test_divergence_finite(self, tripole_grid):
        """divergence should produce finite results on tripolar grid."""
        u = jax.random.normal(jax.random.PRNGKey(1), (36, 73))
        v = jax.random.normal(jax.random.PRNGKey(2), (37, 72))
        result = divergence_cgrid(u, v, tripole_grid)
        assert result.shape == (36, 72)
        assert jnp.all(jnp.isfinite(result))

    def test_curl_finite(self, tripole_grid):
        """curl should produce finite results on tripolar grid."""
        u = jax.random.normal(jax.random.PRNGKey(1), (36, 73))
        v = jax.random.normal(jax.random.PRNGKey(2), (37, 72))
        result = curl_vertex_cgrid(u, v, tripole_grid)
        assert result.shape == (37, 73)
        assert jnp.all(jnp.isfinite(result))
        # Vorticity at north boundary should be nonzero with fold
        assert not jnp.all(result[-1] == 0.0)

    def test_coriolis_finite(self, tripole_grid):
        """Coriolis should produce finite results on tripolar grid."""
        u = jax.random.normal(jax.random.PRNGKey(1), (36, 73))
        v = jax.random.normal(jax.random.PRNGKey(2), (37, 72))
        cor_u, cor_v = coriolis_cgrid(u, v, tripole_grid)
        assert cor_u.shape == (36, 73)
        assert cor_v.shape == (37, 72)
        assert jnp.all(jnp.isfinite(cor_u))
        assert jnp.all(jnp.isfinite(cor_v))

    def test_laplacian_finite(self, tripole_grid):
        """Laplacian should produce finite results on tripolar grid."""
        f = jax.random.normal(jax.random.PRNGKey(0), (36, 72))
        result = laplacian_cgrid(f, tripole_grid)
        assert result.shape == (36, 72)
        assert jnp.all(jnp.isfinite(result))


# =========================================================================
# Regular lat-lon comparison (backward compat)
# =========================================================================


class TestRegularLatLonBackwardCompat:
    """Ensure pad_ns_* on regular lat-lon gives same results as jnp.pad."""

    def test_pad_ns_scalar_matches_jnp_pad(self, regular_grid):
        """pad_ns_scalar with inactive fold == jnp.pad zero."""
        interior = jax.random.normal(jax.random.PRNGKey(0), (35, 72))
        result = pad_ns_scalar(interior, regular_grid)
        expected = jnp.pad(interior, ((1, 1), (0, 0)))
        assert jnp.all(result == expected)

    def test_pad_ns_vector_u_matches_jnp_pad(self, regular_grid):
        """pad_ns_vector_u with inactive fold == jnp.pad zero."""
        interior = jax.random.normal(jax.random.PRNGKey(1), (35, 72))
        result = pad_ns_vector_u(interior, regular_grid)
        expected = jnp.pad(interior, ((1, 1), (0, 0)))
        assert jnp.all(result == expected)

    def test_pad_ns_scalar_3d_matches_jnp_pad(self, regular_grid):
        """pad_ns_scalar 3D with inactive fold == jnp.pad zero."""
        interior = jax.random.normal(jax.random.PRNGKey(2), (35, 72, 5))
        result = pad_ns_scalar(interior, regular_grid)
        expected = jnp.pad(interior, ((1, 1), (0, 0), (0, 0)))
        assert jnp.all(result == expected)
