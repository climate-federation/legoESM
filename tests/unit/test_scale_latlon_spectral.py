"""Category 3: Lat-lon and spectral halo / transform portability.

Tests periodic longitude wrapping, polar boundary conditions, and
spectral transform roundtrip accuracy.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.halo_latlon import pad_halo_latlon
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis, sh_synthesis
from legoesm.grids.latlon import create_latlon_grid


# =========================================================================
# 3a) Lat-lon halo — periodic longitude
# =========================================================================

class TestLatLonPeriodic:
    """Longitude direction should wrap periodically."""

    def test_periodic_east_west(self):
        grid = create_latlon_grid(8, 16)
        lon = grid.lon  # (16,)
        data = jnp.sin(jnp.broadcast_to(lon[None, :], (8, 16)))
        padded = pad_halo_latlon(data, halo=1)

        # West ghost should equal east edge of data
        np.testing.assert_allclose(
            np.array(padded[1:-1, 0]), np.array(data[:, -1]), atol=1e-12
        )
        # East ghost should equal west edge of data
        np.testing.assert_allclose(
            np.array(padded[1:-1, -1]), np.array(data[:, 0]), atol=1e-12
        )

    def test_constant_field_unchanged(self):
        data = jnp.full((8, 16), 5.0)
        padded = pad_halo_latlon(data, halo=1)
        np.testing.assert_allclose(np.array(padded), 5.0, atol=1e-12)

    def test_output_shape(self):
        data = jnp.ones((8, 16))
        padded = pad_halo_latlon(data, halo=1)
        assert padded.shape == (10, 18)
        padded2 = pad_halo_latlon(data, halo=2)
        assert padded2.shape == (12, 20)


# =========================================================================
# 3b) Lat-lon halo — polar boundary
# =========================================================================

class TestLatLonPolar:
    """Polar boundaries should use zero-gradient (Neumann) BC."""

    def test_south_pole_neumann(self):
        data = jnp.arange(8 * 16, dtype=jnp.float64).reshape(8, 16)
        padded = pad_halo_latlon(data, halo=1)
        # South ghost row = first data row
        np.testing.assert_allclose(
            np.array(padded[0, 1:-1]), np.array(data[0, :]), atol=1e-12
        )

    def test_north_pole_neumann(self):
        data = jnp.arange(8 * 16, dtype=jnp.float64).reshape(8, 16)
        padded = pad_halo_latlon(data, halo=1)
        # North ghost row = last data row
        np.testing.assert_allclose(
            np.array(padded[-1, 1:-1]), np.array(data[-1, :]), atol=1e-12
        )


# =========================================================================
# 3c) Spectral transform roundtrip
# =========================================================================

class TestSpectralRoundtrip:
    """Spectral analysis+synthesis should be identity for resolved harmonics."""

    def test_roundtrip_smooth(self):
        grid = create_gaussian_grid(10)  # T10
        # Create Y_3^2 on Gaussian grid (use spectral eigenstructure)
        n, m = 3, 2
        # Initialize via spectral synthesis: set one coefficient
        n_sh = grid.n_sh
        coeffs = jnp.zeros(n_sh, dtype=jnp.complex128)
        # Find index for (n=3, m=2) in the spectral ordering
        ls = np.array(grid.ls)
        ms = np.array(grid.ms)
        idx = np.where((ls == n) & (ms == m))[0]
        if len(idx) > 0:
            coeffs = coeffs.at[idx[0]].set(1.0 + 0.0j)
            field = sh_synthesis(grid, coeffs)
            coeffs_back = sh_analysis(grid, field)
            field_back = sh_synthesis(grid, coeffs_back)
            np.testing.assert_allclose(np.array(field_back), np.array(field), atol=1e-10)

    def test_constant_roundtrip(self):
        grid = create_gaussian_grid(10)
        field = jnp.ones((grid.n_lat, grid.n_lon))
        coeffs = sh_analysis(grid, field)
        field_back = sh_synthesis(grid, coeffs)
        np.testing.assert_allclose(np.array(field_back), 1.0, atol=1e-10)


# =========================================================================
# 3d) Spectral Laplacian eigenvalue
# =========================================================================

class TestSpectralLaplacian:
    """Spectral Laplacian should give exact eigenvalue for spherical harmonics."""

    @pytest.mark.parametrize("n,m", [(2, 0), (3, 1), (4, 2)])
    def test_laplacian_eigenvalue(self, n, m):
        grid = create_gaussian_grid(10)
        ls = np.array(grid.ls)
        ms = np.array(grid.ms)
        idx = np.where((ls == n) & (ms == m))[0]
        if len(idx) == 0:
            pytest.skip(f"Y_{n}^{m} not in T10 grid")

        # Create single harmonic
        coeffs = jnp.zeros(grid.n_sh, dtype=jnp.complex128)
        coeffs = coeffs.at[idx[0]].set(1.0 + 0.0j)

        # Apply spectral Laplacian: multiply by eigenvalue
        lap_eigenvalue = grid.lap  # shape (n_sh,)
        coeffs_lap = coeffs * lap_eigenvalue

        # Synthesize both
        field = sh_synthesis(grid, coeffs)
        field_lap = sh_synthesis(grid, coeffs_lap)

        # Expected: lap(Y_n^m) = -n(n+1)/a^2 * Y_n^m
        from legoesm import constants
        a = constants.R_earth
        expected_eigenvalue = -n * (n + 1) / a**2
        expected_field_lap = expected_eigenvalue * field

        np.testing.assert_allclose(
            np.array(field_lap), np.array(expected_field_lap),
            rtol=1e-8, atol=1e-20,
        )


# =========================================================================
# 3f) Spectral transform dtype
# =========================================================================

class TestSpectralDtype:
    """Spectral coefficients should have correct dtype."""

    def test_complex128_in_x64_mode(self):
        grid = create_gaussian_grid(5)
        field = jnp.ones((grid.n_lat, grid.n_lon), dtype=jnp.float64)
        coeffs = sh_analysis(grid, field)
        assert coeffs.dtype == jnp.complex128
