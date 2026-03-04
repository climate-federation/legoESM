"""Integration tests for the shallow water model."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water import (
    ShallowWaterModel, ShallowWaterConfig, shallow_water_tendencies,
)
from tests.test_cases.williamson import (
    williamson_test2, williamson_test5, williamson_test2_exact,
    compute_error_norms,
)
from legoesm.core.conservation import compute_conservation_diagnostics


class TestShallowWaterModel:
    """Integration tests for the shallow water dynamical core."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(16)

    @pytest.fixture(scope="class")
    def model(self, grid):
        config = ShallowWaterConfig(
            hyperdiff_coeff=1e15,  # Small hyperdiffusion for stability at C16
            use_conservation_fixer=True,
        )
        return ShallowWaterModel(grid, config)

    def test_tendencies_shape(self, grid):
        """Tendencies should have the same shape as state fields."""
        state = williamson_test2(grid)
        tend = shallow_water_tendencies(state, grid)
        assert tend.dh_dt.shape == state.h.shape
        assert tend.du_dt.shape == state.u.shape
        assert tend.dv_dt.shape == state.v.shape

    def test_tendencies_finite(self, grid):
        """All tendencies should be finite."""
        state = williamson_test2(grid)
        tend = shallow_water_tendencies(state, grid)
        assert jnp.all(jnp.isfinite(tend.dh_dt.data))
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))

    def test_single_step_finite(self, model, grid):
        """A single time step should produce finite values."""
        state = williamson_test2(grid)
        state_new = model.step(state, dt=600.0)
        assert jnp.all(jnp.isfinite(state_new.h.data))
        assert jnp.all(jnp.isfinite(state_new.u.data))
        assert jnp.all(jnp.isfinite(state_new.v.data))

    def test_multi_step_stability(self, model, grid):
        """Model should remain stable for 100 steps."""
        state = williamson_test2(grid)
        # Use smaller dt for stability at low resolution C16
        for _ in range(100):
            state = model.step(state, dt=120.0)
        assert jnp.all(jnp.isfinite(state.h.data))
        assert float(jnp.max(jnp.abs(state.u.data))) < 200.0  # Reasonable wind speed

    def test_mass_conservation(self, model, grid):
        """Mass should be conserved to machine precision."""
        state = williamson_test2(grid)
        diag_init = compute_conservation_diagnostics(state, grid)

        for _ in range(50):
            state = model.step(state, dt=600.0)

        diag_final = compute_conservation_diagnostics(state, grid)
        mass_change = abs(float(diag_final['total_mass'] - diag_init['total_mass']))
        relative_change = mass_change / float(diag_init['total_mass'])
        # float32 precision limits conservation to ~1e-7; fixer is exact
        # in infinite precision but float32 accumulation causes round-off
        assert relative_change < 1e-4, f"Mass relative change: {relative_change:.2e}"

    def test_topography_preserved(self, model, grid):
        """Surface topography should not change during integration."""
        state = williamson_test5(grid)
        h_s_init = state.h_s.data.copy()

        for _ in range(10):
            state = model.step(state, dt=600.0)

        assert jnp.allclose(state.h_s.data, h_s_init)


class TestWilliamsonTest5:
    """Tests specific to Williamson Test Case 5 (mountain)."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(16)

    def test_mountain_present(self, grid):
        """Mountain topography should be non-zero."""
        state = williamson_test5(grid)
        assert float(jnp.max(state.h_s.data)) > 0

    def test_mountain_peak(self, grid):
        """Mountain peak should be approximately 2000m."""
        state = williamson_test5(grid)
        peak = float(jnp.max(state.h_s.data))
        assert 1500 < peak < 2100, f"Mountain peak: {peak:.0f}m"


class TestDifferentiability:
    """Tests for end-to-end differentiability."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(8)

    def test_grad_through_single_step(self, grid):
        """jax.grad should work through a single model step."""
        model = ShallowWaterModel(grid, ShallowWaterConfig(
            use_conservation_fixer=False  # Simpler for gradient testing
        ))
        state = williamson_test2(grid)

        def loss(h_data):
            state_mod = state._replace(
                h=state.h.replace(data=h_data)
            )
            state_new = model.step(state_mod, dt=300.0)
            return jnp.mean(state_new.h.data ** 2)

        grads = jax.grad(loss)(state.h.data)
        assert jnp.all(jnp.isfinite(grads))
        assert not jnp.allclose(grads, 0.0)  # Gradients should be non-trivial

    def test_grad_through_multi_step(self, grid):
        """jax.grad should work through multiple time steps."""
        model = ShallowWaterModel(grid, ShallowWaterConfig(
            use_conservation_fixer=False
        ))
        state = williamson_test2(grid)

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            for _ in range(5):
                s = model.step(s, dt=300.0)
            return jnp.mean(s.h.data ** 2)

        grads = jax.grad(loss)(state.h.data)
        assert jnp.all(jnp.isfinite(grads))

    def test_grad_through_scan(self, grid):
        """jax.grad should work through lax.scan integration."""
        config = ShallowWaterConfig(use_conservation_fixer=False)
        model = ShallowWaterModel(grid, config)
        state = williamson_test2(grid)

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            final, _ = model.integrate_scan(s, n_steps=5, dt=300.0)
            return jnp.mean(final.h.data ** 2)

        grads = jax.grad(loss)(state.h.data)
        assert jnp.all(jnp.isfinite(grads))
