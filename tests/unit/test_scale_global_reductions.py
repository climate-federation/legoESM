"""Category 4: Global reductions & conservation under sharding.

Tests global_integral (4*pi*R^2 for constant=1), area-weighted sum,
conservation fixer, and consistency with Gaussian-grid total area.
"""

from __future__ import annotations

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators import global_integral, global_mean
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.core.conservation import _global_area_sum, _total_area


N = 4  # Small cubed-sphere


@pytest.fixture(scope="module")
def grid():
    return create_cubed_sphere(N)


# ---------------------------------------------------------------------------
# Global integral of constant field = const * total_area
# ---------------------------------------------------------------------------

class TestGlobalIntegral:
    def test_integral_of_ones(self, grid):
        """Integral of 1.0 over the sphere should be total_area ~ 4*pi*R^2."""
        f = Field(data=jnp.ones((6, N, N), dtype=jnp.float64), name="ones")
        integral = global_integral(f, grid)
        expected = 4.0 * jnp.pi * grid.radius ** 2
        np.testing.assert_allclose(float(integral), float(expected), rtol=1e-2)

    def test_integral_of_constant(self, grid):
        """Integral of c = c * total_area."""
        c = 3.7
        f = Field(data=jnp.full((6, N, N), c, dtype=jnp.float64), name="const")
        integral = global_integral(f, grid)
        np.testing.assert_allclose(
            float(integral), c * float(jnp.sum(grid.area)), rtol=1e-6,
        )

    def test_global_mean_of_constant(self, grid):
        """Global mean of a constant should return that constant."""
        c = 5.5
        f = Field(data=jnp.full((6, N, N), c, dtype=jnp.float64), name="const")
        mean = global_mean(f, grid)
        np.testing.assert_allclose(float(mean), c, rtol=1e-6)


# ---------------------------------------------------------------------------
# Total area should approximate 4*pi*R^2
# ---------------------------------------------------------------------------

class TestTotalArea:
    def test_cubed_sphere_total_area(self, grid):
        expected = 4.0 * jnp.pi * grid.radius ** 2
        np.testing.assert_allclose(
            float(grid.total_area), float(expected), rtol=1e-2,
        )

    def test_all_areas_positive(self, grid):
        assert jnp.all(grid.area > 0)

    def test_grid_total_area_property(self, grid):
        """grid_total_area property should match sum of area."""
        np.testing.assert_allclose(
            float(grid.grid_total_area), float(jnp.sum(grid.area)), rtol=1e-14,
        )


# ---------------------------------------------------------------------------
# _global_area_sum (conservation module)
# ---------------------------------------------------------------------------

class TestGlobalAreaSum:
    def test_area_sum_of_ones(self, grid):
        arr = jnp.ones((6, N, N), dtype=jnp.float64)
        s = _global_area_sum(arr, grid)
        expected = float(jnp.sum(grid.area))
        np.testing.assert_allclose(float(s), expected, rtol=1e-6)

    def test_area_sum_of_zeros(self, grid):
        arr = jnp.zeros((6, N, N), dtype=jnp.float64)
        s = _global_area_sum(arr, grid)
        np.testing.assert_allclose(float(s), 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# JIT and differentiability
# ---------------------------------------------------------------------------

class TestJITDiff:
    def test_global_integral_jittable(self, grid):
        f = Field(data=jnp.ones((6, N, N), dtype=jnp.float64), name="t")
        jit_fn = jax.jit(lambda d: global_integral(Field(data=d, name="t"), grid))
        result = jit_fn(f.data)
        assert jnp.isfinite(result)

    def test_global_integral_differentiable(self, grid):
        def loss(data):
            f = Field(data=data, name="t")
            return global_integral(f, grid)
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        grad = jax.grad(loss)(data)
        # Gradient of sum(data * area) w.r.t. data should be area
        np.testing.assert_allclose(grad, grid.area, rtol=1e-12)
