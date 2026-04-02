"""Unit tests for the CFL module (legoesm.core.cfl)."""

import numpy as np
import pytest

from legoesm.core.cfl import (
    estimate_min_dx_cubed_sphere,
    estimate_min_dx_latlon,
    estimate_min_dx_gaussian,
    estimate_min_dx_icosahedral,
    cfl_max_dt,
    adaptive_hyperdiff_coeff,
    cfl_check_and_adjust,
)


EARTH_RADIUS = 6.371229e6


# ============================================================================
# Grid spacing estimators
# ============================================================================

class TestEstimateMinDx:
    """Tests for minimum grid spacing estimators."""

    def test_cubed_sphere_positive(self):
        dx = estimate_min_dx_cubed_sphere(48)
        assert dx > 0

    def test_cubed_sphere_decreases_with_resolution(self):
        dx16 = estimate_min_dx_cubed_sphere(16)
        dx48 = estimate_min_dx_cubed_sphere(48)
        assert dx48 < dx16

    def test_cubed_sphere_c48_order_of_magnitude(self):
        """C48 ~ 200 km nominal → min dx ~ 100-150 km."""
        dx = estimate_min_dx_cubed_sphere(48)
        assert 50e3 < dx < 250e3

    def test_latlon_positive(self):
        dx = estimate_min_dx_latlon(64)
        assert dx > 0

    def test_latlon_decreases_with_resolution(self):
        dx32 = estimate_min_dx_latlon(32)
        dx128 = estimate_min_dx_latlon(128)
        assert dx128 < dx32

    def test_gaussian_positive(self):
        dx = estimate_min_dx_gaussian(21)
        assert dx > 0

    def test_gaussian_T21_order_of_magnitude(self):
        """T21 has ~44 longitudes → dx ~ 900 km at equator."""
        dx = estimate_min_dx_gaussian(21)
        assert 500e3 < dx < 1500e3

    def test_icosahedral_positive(self):
        """estimate_min_dx_icosahedral returns positive spacing."""
        dx = estimate_min_dx_icosahedral(5)
        assert dx > 0

    def test_icosahedral_decreases_with_level(self):
        """Higher subdivision level → finer grid."""
        dx4 = estimate_min_dx_icosahedral(4)
        dx6 = estimate_min_dx_icosahedral(6)
        assert dx6 < dx4

    def test_icosahedral_level5_order_of_magnitude(self):
        """Level 5 → ~10242 cells → dx ~ 200 km average, min ~ 170 km."""
        dx = estimate_min_dx_icosahedral(5)
        n_cells = 10 * 4**5 + 2
        assert n_cells == 10242
        assert 100e3 < dx < 300e3

    def test_icosahedral_cell_count_formula(self):
        """Verify the icosahedral cell count formula: 10 * 4^L + 2."""
        for level in range(1, 8):
            expected = 10 * 4**level + 2
            # Cross-check: dx_avg = sqrt(4*pi*R^2/N)
            dx_avg = np.sqrt(4 * np.pi * EARTH_RADIUS**2 / expected)
            dx_min = estimate_min_dx_icosahedral(level)
            # Min should be ~85% of average
            assert 0.7 * dx_avg < dx_min < dx_avg


# ============================================================================
# CFL max dt
# ============================================================================

class TestCFLMaxDt:
    def test_basic_computation(self):
        dt = cfl_max_dt(1000.0, 100.0, cfl_number=1.0, ndim=1)
        assert dt == pytest.approx(10.0)

    def test_lower_cfl_gives_smaller_dt(self):
        dt1 = cfl_max_dt(1000.0, 100.0, cfl_number=1.0)
        dt08 = cfl_max_dt(1000.0, 100.0, cfl_number=0.8)
        assert dt08 < dt1

    def test_2d_smaller_than_1d(self):
        dt1d = cfl_max_dt(1000.0, 100.0, ndim=1)
        dt2d = cfl_max_dt(1000.0, 100.0, ndim=2)
        assert dt2d < dt1d


# ============================================================================
# Adaptive hyperdiffusion coefficient
# ============================================================================

class TestAdaptiveHyperdiff:
    def test_positive(self):
        nu = adaptive_hyperdiff_coeff(100e3, 600.0, order=4)
        assert nu > 0

    def test_scales_with_dx(self):
        """Coefficient scales as dx^order."""
        nu_fine = adaptive_hyperdiff_coeff(50e3, 600.0, order=4, safety=1.0)
        nu_coarse = adaptive_hyperdiff_coeff(100e3, 600.0, order=4, safety=1.0)
        # dx doubled → coefficient should increase by 2^4 = 16
        ratio = nu_coarse / nu_fine
        assert 15.0 < ratio < 17.0


# ============================================================================
# CFL check and adjust — all grid types
# ============================================================================

class TestCFLCheckAndAdjust:
    def test_cubed_sphere_safe_dt_unchanged(self):
        """Sufficiently small dt returned unchanged for cubed-sphere."""
        dt = cfl_check_and_adjust(
            dt=60.0, n=48, grid_type="cubed_sphere", verbose=False,
        )
        assert dt == 60.0

    def test_latlon_safe_dt_unchanged(self):
        """Sufficiently small dt returned unchanged for lat-lon."""
        dt = cfl_check_and_adjust(
            dt=10.0, n=64, grid_type="latlon", verbose=False,
        )
        assert dt == 10.0

    def test_gaussian_safe_dt_unchanged(self):
        dt = cfl_check_and_adjust(
            dt=600.0, n=21, grid_type="gaussian", verbose=False,
        )
        assert dt == 600.0

    def test_voronoi_safe_dt_unchanged(self):
        """Voronoi grid type dispatches to icosahedral estimator."""
        dt = cfl_check_and_adjust(
            dt=300.0, n=5, grid_type="voronoi", verbose=False,
        )
        assert dt == 300.0

    def test_voronoi_unsafe_dt_reduced(self):
        """Excessively large dt is reduced for voronoi grid."""
        dt = cfl_check_and_adjust(
            dt=5000.0, n=5, grid_type="voronoi", verbose=False,
        )
        assert dt < 5000.0
        # Should be a "nice" value
        nice = [600, 450, 300, 240, 200, 180, 150, 120, 100, 90, 60, 45, 30, 20, 15, 10]
        assert dt in nice

    def test_unsafe_dt_reduced(self):
        """Excessively large dt is reduced."""
        dt = cfl_check_and_adjust(
            dt=10000.0, n=16, grid_type="cubed_sphere", verbose=False,
        )
        assert dt < 10000.0

    def test_model_type_compressible(self):
        """Compressible model type works."""
        dt = cfl_check_and_adjust(
            dt=300.0, n=48, model_type="compressible",
            grid_type="cubed_sphere", verbose=False,
        )
        assert dt > 0
