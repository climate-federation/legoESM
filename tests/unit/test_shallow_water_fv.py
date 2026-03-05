"""Integration tests for the FV shallow water model."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv import (
    FVShallowWaterModel, FVShallowWaterConfig, fv_shallow_water_tendencies,
)
from tests.test_cases.williamson import (
    williamson_test2, williamson_test5,
    compute_error_norms,
)
from legoesm.core.conservation import compute_conservation_diagnostics


class TestFVShallowWaterModel:
    """Integration tests for the FV shallow water dynamical core."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(16)

    @pytest.fixture(scope="class")
    def model(self, grid):
        config = FVShallowWaterConfig(
            hyperdiff_coeff=1e15,
            use_conservation_fixer=True,
        )
        return FVShallowWaterModel(grid, config)

    def test_tendencies_shape(self, grid):
        """Tendencies should have the same shape as state fields."""
        state = williamson_test2(grid)
        tend = fv_shallow_water_tendencies(state, grid)
        assert tend.dh_dt.shape == state.h.shape
        assert tend.du_dt.shape == state.u.shape
        assert tend.dv_dt.shape == state.v.shape

    def test_tendencies_finite(self, grid):
        """All tendencies should be finite."""
        state = williamson_test2(grid)
        tend = fv_shallow_water_tendencies(state, grid)
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
        for _ in range(100):
            state = model.step(state, dt=120.0)
        assert jnp.all(jnp.isfinite(state.h.data))
        assert float(jnp.max(jnp.abs(state.u.data))) < 200.0

    def test_mass_conservation(self, model, grid):
        """Mass should be conserved to good precision."""
        state = williamson_test2(grid)
        diag_init = compute_conservation_diagnostics(state, grid)

        for _ in range(50):
            state = model.step(state, dt=600.0)

        diag_final = compute_conservation_diagnostics(state, grid)
        mass_change = abs(float(diag_final['total_mass'] - diag_init['total_mass']))
        relative_change = mass_change / float(diag_init['total_mass'])
        assert relative_change < 1e-4, f"Mass relative change: {relative_change:.2e}"

    def test_topography_preserved(self, model, grid):
        """Surface topography should not change during integration."""
        state = williamson_test5(grid)
        h_s_init = state.h_s.data.copy()

        for _ in range(10):
            state = model.step(state, dt=600.0)

        assert jnp.allclose(state.h_s.data, h_s_init)


class TestFVDifferentiability:
    """Tests for end-to-end differentiability of FV model."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(8)

    def test_grad_through_single_step(self, grid):
        """jax.grad should work through a single FV model step."""
        model = FVShallowWaterModel(grid, FVShallowWaterConfig(
            use_conservation_fixer=False,
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
        assert not jnp.allclose(grads, 0.0)

    def test_grad_through_scan(self, grid):
        """jax.grad should work through lax.scan integration."""
        config = FVShallowWaterConfig(use_conservation_fixer=False)
        model = FVShallowWaterModel(grid, config)
        state = williamson_test2(grid)

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            final, _ = model.integrate_scan(s, n_steps=3, dt=300.0)
            return jnp.mean(final.h.data ** 2)

        grads = jax.grad(loss)(state.h.data)
        assert jnp.all(jnp.isfinite(grads))


class TestFVFactoryWiring:
    """Test that FV models can be created via the factory."""

    def test_resolve_fv_shallow_water(self):
        """finite_volume + shallow_water should resolve correctly."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(
            dynamics="shallow_water", discretization="finite_volume"
        )
        assert name == "fv_shallow_water"

    def test_resolve_fv_hydrostatic(self):
        """finite_volume + hydrostatic should resolve correctly."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(
            dynamics="hydrostatic", discretization="finite_volume"
        )
        assert name == "fv_primitive_equations"

    def test_resolve_fv_nonhydrostatic(self):
        """finite_volume + nonhydrostatic should resolve correctly."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        name = resolve_solver_name(
            dynamics="nonhydrostatic", discretization="finite_volume"
        )
        assert name == "fv_compressible_euler"

    def test_create_fv_sw_model(self):
        """Factory should create FV shallow water model."""
        from legoesm.atmosphere.dynamics import create_model
        grid = create_cubed_sphere(8)
        model = create_model("fv_shallow_water", grid=grid)
        assert isinstance(model, FVShallowWaterModel)

    def test_fv_in_discretization_options(self):
        """finite_volume should be in DISCRETIZATION_OPTIONS."""
        from legoesm.atmosphere.dynamics import DISCRETIZATION_OPTIONS
        assert "finite_volume" in DISCRETIZATION_OPTIONS

    def test_fv_solvers_in_available(self):
        """FV solver names should be in AVAILABLE_SOLVERS."""
        from legoesm.atmosphere.dynamics import AVAILABLE_SOLVERS
        assert "fv_shallow_water" in AVAILABLE_SOLVERS
        assert "fv_primitive_equations" in AVAILABLE_SOLVERS
        assert "fv_compressible_euler" in AVAILABLE_SOLVERS
