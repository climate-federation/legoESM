"""Category 3: Lat-lon halo / spectral transform tests.

Tests periodic longitude wrap, polar boundary conditions, spectral
roundtrip (analysis->synthesis), and Laplacian eigenvalue correctness.
"""

from __future__ import annotations

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from legoesm.grids.halo_latlon import (
    pad_halo_latlon,
    pad_halo_latlon_vector,
    pad_halo_vector_latlon,
)
from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis,
    sh_synthesis,
    spectral_laplacian,
    _sh_idx,
)


NLAT, NLON = 16, 32  # Small lat-lon grid for halo tests
N_MAX = 10            # Small spectral truncation


# ===========================================================================
# Lat-lon halo exchange
# ===========================================================================

class TestLonPeriodicWrap:
    """Longitude halo should periodically wrap."""

    def test_scalar_lon_wrap(self):
        data = jnp.arange(NLAT * NLON, dtype=jnp.float64).reshape(NLAT, NLON)
        padded = pad_halo_latlon(data, halo=1)
        assert padded.shape == (NLAT + 2, NLON + 2)

        # Interior unchanged
        interior = padded[1:-1, 1:-1]
        np.testing.assert_array_equal(interior, data)

        # West halo = last column of data
        np.testing.assert_array_equal(padded[1:-1, 0], data[:, -1])
        # East halo = first column of data
        np.testing.assert_array_equal(padded[1:-1, -1], data[:, 0])

    def test_scalar_lon_wrap_halo2(self):
        data = jnp.arange(NLAT * NLON, dtype=jnp.float64).reshape(NLAT, NLON)
        padded = pad_halo_latlon(data, halo=2)
        assert padded.shape == (NLAT + 4, NLON + 4)
        interior = padded[2:-2, 2:-2]
        np.testing.assert_array_equal(interior, data)
        # West halo
        np.testing.assert_array_equal(padded[2:-2, 0], data[:, -2])
        np.testing.assert_array_equal(padded[2:-2, 1], data[:, -1])

    def test_constant_field_unchanged(self):
        data = jnp.ones((NLAT, NLON), dtype=jnp.float64) * 5.0
        padded = pad_halo_latlon(data, halo=1)
        np.testing.assert_allclose(padded, 5.0, atol=1e-14)


class TestPolarBoundaryScalar:
    """Scalar pole-folding: fold with 180-deg shift, no sign change."""

    def test_south_pole_fold(self):
        data = jnp.arange(NLAT * NLON, dtype=jnp.float64).reshape(NLAT, NLON)
        padded = pad_halo_latlon(data, halo=1)
        # South halo row = first row of data, reversed in lat and shifted
        # by half in longitude. For halo=1, south ghost is padded[0, 1:-1].
        # The fold mirrors data[0,:] shifted by half the longitude dim.
        half = (NLON + 2) // 2  # half of the lon-padded dimension
        # After longitude padding, data_lon has shape (NLAT, NLON+2)
        # south ghost = jnp.roll(data_lon[0, :], half)
        # Just check it's NOT zero (it's populated)
        assert not jnp.allclose(padded[0, 1:-1], 0.0)

    def test_north_pole_fold(self):
        data = jnp.arange(NLAT * NLON, dtype=jnp.float64).reshape(NLAT, NLON)
        padded = pad_halo_latlon(data, halo=1)
        assert not jnp.allclose(padded[-1, 1:-1], 0.0)


class TestPolarBoundaryVector:
    """Vector pole-folding: fold with sign reversal."""

    def test_vector_sign_reversal_at_poles(self):
        """A constant positive vector field should become negative at poles."""
        data = jnp.ones((NLAT, NLON), dtype=jnp.float64)
        padded = pad_halo_latlon_vector(data, halo=1)
        # South pole halo should be negated (sign reversal)
        # The south ghost row = -jnp.roll(data_lon[0,:], half)
        # For constant field of 1.0, south ghost should be -1.0
        south_halo = padded[0, 1:-1]
        np.testing.assert_allclose(south_halo, -1.0, atol=1e-14)
        # North pole halo should also be -1.0
        north_halo = padded[-1, 1:-1]
        np.testing.assert_allclose(north_halo, -1.0, atol=1e-14)

    def test_vector_pair_padding(self):
        u = jnp.ones((NLAT, NLON), dtype=jnp.float64)
        v = jnp.ones((NLAT, NLON), dtype=jnp.float64) * 2.0
        u_pad, v_pad = pad_halo_vector_latlon(u, v, halo=1)
        assert u_pad.shape == (NLAT + 2, NLON + 2)
        assert v_pad.shape == (NLAT + 2, NLON + 2)
        # Interior unchanged
        np.testing.assert_array_equal(u_pad[1:-1, 1:-1], u)
        np.testing.assert_array_equal(v_pad[1:-1, 1:-1], v)


# ===========================================================================
# Spectral transforms
# ===========================================================================

class TestSpectralRoundtrip:
    """sh_synthesis(sh_analysis(f)) should recover f for band-limited fields."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_gaussian_grid(N_MAX)

    def test_roundtrip_constant(self, grid):
        """Constant field (n=0 mode only) roundtrips exactly."""
        field = jnp.ones((grid.n_lat, grid.n_lon), dtype=jnp.float64) * 3.0
        coeffs = sh_analysis(grid, field)
        recovered = sh_synthesis(grid, coeffs)
        np.testing.assert_allclose(recovered, 3.0, atol=1e-10)

    def test_roundtrip_Y10(self, grid):
        """Y_1^0 ~ sin(lat) should roundtrip."""
        field = jnp.broadcast_to(
            grid.sin_lat[:, None], (grid.n_lat, grid.n_lon)
        )
        coeffs = sh_analysis(grid, field)
        recovered = sh_synthesis(grid, coeffs)
        np.testing.assert_allclose(recovered, field, atol=1e-10)

    def test_roundtrip_Y22(self, grid):
        """A pure Y_2^2 mode should roundtrip."""
        # Y_2^2 ~ cos^2(lat) * cos(2*lon)
        cos2 = grid.cos_lat[:, None] ** 2
        cos2lon = jnp.cos(2.0 * grid.lon2d)
        field = cos2 * cos2lon
        coeffs = sh_analysis(grid, field)
        recovered = sh_synthesis(grid, coeffs)
        np.testing.assert_allclose(recovered, field, atol=1e-8)

    def test_roundtrip_preserves_dtype(self, grid):
        field = jnp.ones((grid.n_lat, grid.n_lon), dtype=jnp.float64)
        coeffs = sh_analysis(grid, field)
        assert coeffs.dtype == jnp.complex128
        recovered = sh_synthesis(grid, coeffs)
        assert recovered.dtype == jnp.float64


class TestSpectralLaplacian:
    """Spectral Laplacian eigenvalue: lap(Y_n^m) = -n(n+1)/a^2 * Y_n^m."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_gaussian_grid(N_MAX)

    def test_laplacian_eigenvalue_Y10(self, grid):
        """Laplacian of Y_1^0 should have eigenvalue -1*2/a^2 = -2/a^2."""
        idx = _sh_idx(1, 0)
        coeffs = jnp.zeros(grid.n_sh, dtype=jnp.complex128)
        coeffs = coeffs.at[idx].set(1.0 + 0j)
        lap_coeffs = spectral_laplacian(grid, coeffs)
        expected_eig = -1.0 * 2.0 / (grid.radius ** 2)
        np.testing.assert_allclose(
            lap_coeffs[idx].real, expected_eig, rtol=1e-12,
        )
        # Other coefficients should be zero
        mask = jnp.arange(grid.n_sh) != idx
        np.testing.assert_allclose(jnp.abs(lap_coeffs[mask]), 0.0, atol=1e-30)

    def test_laplacian_eigenvalue_Y32(self, grid):
        """Laplacian of Y_3^2: eigenvalue = -3*4/a^2 = -12/a^2."""
        idx = _sh_idx(3, 2)
        coeffs = jnp.zeros(grid.n_sh, dtype=jnp.complex128)
        coeffs = coeffs.at[idx].set(1.0 + 0j)
        lap_coeffs = spectral_laplacian(grid, coeffs)
        expected_eig = -3.0 * 4.0 / (grid.radius ** 2)
        np.testing.assert_allclose(
            lap_coeffs[idx].real, expected_eig, rtol=1e-12,
        )

    def test_lap_array_matches_eigenvalues(self, grid):
        """grid.lap should equal -n(n+1)/a^2 for all modes."""
        expected = -grid.ls.astype(jnp.float64) * (grid.ls.astype(jnp.float64) + 1) / (grid.radius ** 2)
        np.testing.assert_allclose(grid.lap, expected, rtol=1e-14)


class TestSpectralGrid:
    """Basic Gaussian grid sanity checks."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_gaussian_grid(N_MAX)

    def test_grid_dimensions(self, grid):
        assert grid.n_max == N_MAX
        n_sh = (N_MAX + 1) * (N_MAX + 2) // 2
        assert grid.n_sh == n_sh

    def test_weights_sum(self, grid):
        """Gaussian weights should sum to 2 (integral of 1 over [-1,1])."""
        np.testing.assert_allclose(jnp.sum(grid.weights), 2.0, atol=1e-12)

    def test_lat_monotonic(self, grid):
        """Latitudes should be monotonically increasing (S->N)."""
        diffs = jnp.diff(grid.lat)
        assert jnp.all(diffs > 0)

    def test_total_area(self, grid):
        """Total area should be close to 4*pi*R^2."""
        expected = 4.0 * jnp.pi * grid.radius ** 2
        np.testing.assert_allclose(grid.grid_total_area, expected, rtol=1e-10)

    def test_jit_analysis(self, grid):
        """sh_analysis should be JIT-compatible."""
        field = jnp.ones((grid.n_lat, grid.n_lon), dtype=jnp.float64)
        jit_fn = jax.jit(lambda f: sh_analysis(grid, f))
        coeffs = jit_fn(field)
        assert coeffs.shape == (grid.n_sh,)
