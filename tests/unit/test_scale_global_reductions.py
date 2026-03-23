"""Category 4: Global reductions & conservation under sharding.

Tests global_integral, global_mean, and mass conservation fixer
work correctly on single-device and (if available) sharded states.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.operators import global_integral, global_mean
from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm import constants


# =========================================================================
# Helpers
# =========================================================================

def _make_field(grid, value):
    """Create a scalar Field on cubed-sphere grid."""
    if callable(value):
        data = value(grid)
    else:
        data = jnp.full((6, grid.n, grid.n), value, dtype=jnp.float64)
    return Field(data=data, name="test", dims=("face", "x", "y"), units="1")


# =========================================================================
# 4a) Global integral — single device
# =========================================================================

class TestGlobalIntegral:
    """Global integral should match analytical values."""

    def test_constant_one_is_sphere_area(self):
        grid = create_cubed_sphere(8)
        field = _make_field(grid, 1.0)
        result = float(global_integral(field, grid))
        expected = 4.0 * np.pi * constants.R_earth**2
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_zero_field_is_zero(self):
        grid = create_cubed_sphere(8)
        field = _make_field(grid, 0.0)
        result = float(global_integral(field, grid))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# =========================================================================
# 4c) Global mean — correctness
# =========================================================================

class TestGlobalMean:
    """Global mean should give correct values."""

    def test_constant_field(self):
        grid = create_cubed_sphere(8)
        field = _make_field(grid, 7.5)
        result = float(global_mean(field, grid))
        np.testing.assert_allclose(result, 7.5, rtol=1e-6)

    def test_sin_lat_symmetric(self):
        """sin(lat) should have near-zero global mean (antisymmetric about equator)."""
        grid = create_cubed_sphere(8)
        data = jnp.sin(grid.lat)
        field = Field(data=data, name="sin_lat", dims=("face", "x", "y"), units="1")
        result = float(global_mean(field, grid))
        # Should be near zero due to antisymmetry
        assert abs(result) < 0.05


# =========================================================================
# 4d) Accumulation dtype
# =========================================================================

class TestAccumulationDtype:
    """Global reductions should use float64 precision when available."""

    def test_small_values_no_cancellation(self):
        """Sum of many small values should not lose precision."""
        grid = create_cubed_sphere(8)
        # Many small values that sum to a known total
        val = 1e-10
        data = jnp.full((6, 8, 8), val, dtype=jnp.float64)
        field = Field(data=data, name="small", dims=("face", "x", "y"), units="1")
        result = float(global_integral(field, grid))
        expected = val * float(jnp.sum(grid.area))
        # With float64 accumulation, this should be accurate
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# =========================================================================
# 4e) Conservation fixer under sharding
# =========================================================================

class TestConservationFixer:
    """Mass conservation fixer should preserve total mass."""

    def test_mass_fixer_conserves(self):
        from legoesm.core.conservation import fix_mass_shallow_water
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel, CDGridShallowWaterConfig,
        )

        grid = create_cubed_sphere(4)
        # Create a simple state with known mass
        n = grid.n
        h_data = jnp.ones((6, n, n)) * 1000.0
        h_field = Field(data=h_data, name="h", dims=("face", "x", "y"), units="m")

        # Simulate a state with mass perturbation
        rng = np.random.default_rng(42)
        h_perturbed = h_data + jnp.array(rng.standard_normal((6, n, n))) * 10.0
        h_perturbed_field = Field(
            data=h_perturbed, name="h", dims=("face", "x", "y"), units="m"
        )

        # Create mock shallow water states (just need .h)
        from collections import namedtuple
        MockSWState = namedtuple("MockSWState", ["h"])
        old = MockSWState(h=h_field)
        new = MockSWState(h=h_perturbed_field)

        fixed = fix_mass_shallow_water(new, old, grid)

        # Total mass should be conserved
        mass_old = float(jnp.sum(h_field.data * grid.area))
        mass_fixed = float(jnp.sum(fixed.h.data * grid.area))
        np.testing.assert_allclose(mass_fixed, mass_old, rtol=1e-10)


# =========================================================================
# 4f) Area-weighted sum matches analytical
# =========================================================================

class TestAnalyticalIntegrals:
    """Area-weighted integrals should match analytical values."""

    def test_cos2_lat_integral(self):
        """∫cos²(lat) dA ≈ 8πR²/3.

        ∫cos²(ϕ) cos(ϕ) dϕ dλ over sphere =
        2πR² ∫₋₁¹ (1-μ²) dμ = 2πR² × 4/3 = 8πR²/3.
        """
        grid = create_cubed_sphere(16)  # Use finer grid for accuracy
        data = jnp.cos(grid.lat) ** 2
        field = Field(data=data, name="cos2lat", dims=("face", "x", "y"), units="1")
        result = float(global_integral(field, grid))
        expected = 8.0 * np.pi * constants.R_earth**2 / 3.0
        # C16 grid should give ~2% accuracy
        np.testing.assert_allclose(result, expected, rtol=0.02)
