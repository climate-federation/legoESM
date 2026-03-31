"""Tests for the FV shallow water model on the lat-lon grid.

Covers operator correctness, Williamson test cases, conservation,
and differentiability.
"""

import pytest
import jax
import jax.numpy as jnp

from legoesm.grids.latlon import create_latlon_grid
from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.core.operators_fv_latlon import (
    fv_flux_divergence_latlon,
    fv_gradient_lon,
    fv_gradient_lat,
)
from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
    FVShallowWaterLatLonModel,
    FVShallowWaterLatLonConfig,
    fv_shallow_water_tendencies_latlon,
)
from legoesm.core.conservation import (
    compute_conservation_diagnostics,
)


# Require float64 for these tests
pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="Requires JAX_ENABLE_X64=1",
)


def _make_dims():
    return ("lat", "lon")


def _make_state(grid, h_data, u_data, v_data, h_s_data=None):
    """Helper to create a ShallowWaterState on a lat-lon grid."""
    dims = _make_dims()
    if h_s_data is None:
        h_s_data = jnp.zeros_like(h_data)
    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=dims, units="m"),
        u=Field(data=u_data, name="u", dims=dims, units="m/s"),
        v=Field(data=v_data, name="v", dims=dims, units="m/s"),
        h_s=Field(data=h_s_data, name="h_s", dims=dims, units="m"),
    )


# ==============================================================================
# Operator tests
# ==============================================================================

class TestOperators:
    """Test PPM transport operators on lat-lon grid."""

    def test_constant_field_zero_tendency(self):
        """Flux divergence of a constant field should be zero."""
        grid = create_latlon_grid(32)
        q = jnp.ones((grid.n_lat, grid.n_lon)) * 1000.0
        u = jnp.ones((grid.n_lat, grid.n_lon)) * 10.0
        v = jnp.zeros((grid.n_lat, grid.n_lon))

        tend = fv_flux_divergence_latlon(q, u, v, grid)
        assert jnp.max(jnp.abs(tend)) < 1e-8, \
            f"Constant field should give zero tendency, got max={float(jnp.max(jnp.abs(tend))):.2e}"

    def test_zero_velocity_zero_tendency(self):
        """Zero velocity should give zero flux divergence."""
        grid = create_latlon_grid(32)
        q = jnp.sin(grid.lat2d) * jnp.cos(grid.lon2d) + 5.0
        u = jnp.zeros((grid.n_lat, grid.n_lon))
        v = jnp.zeros((grid.n_lat, grid.n_lon))

        tend = fv_flux_divergence_latlon(q, u, v, grid)
        assert jnp.max(jnp.abs(tend)) < 1e-10, \
            f"Zero velocity should give zero tendency, got max={float(jnp.max(jnp.abs(tend))):.2e}"

    def test_gradient_constant_field(self):
        """Gradient of a constant field should be zero."""
        grid = create_latlon_grid(32)
        q = jnp.ones((grid.n_lat, grid.n_lon)) * 42.0

        dq_dx = fv_gradient_lon(q, grid)
        dq_dy = fv_gradient_lat(q, grid)

        assert jnp.max(jnp.abs(dq_dx)) < 1e-10
        assert jnp.max(jnp.abs(dq_dy)) < 1e-10


# ==============================================================================
# Williamson Test Cases
# ==============================================================================

class TestWilliamsonTC2:
    """Test Williamson Test Case 2: Steady geostrophic flow."""

    def test_tc2_one_day_error(self):
        """TC2 L2 error after 1 day should be small."""
        from tests.atmosphere.shallow_water.test_cases.williamson_latlon import (
            williamson_test2_latlon,
            compute_error_norms_latlon,
        )

        grid = create_latlon_grid(48)
        config = FVShallowWaterLatLonConfig(
            use_polar_filter=True,
            polar_filter_cutoff_deg=60.0,
        )
        dt = 600.0
        model = FVShallowWaterLatLonModel(grid, config, dt=dt)

        state = williamson_test2_latlon(grid)
        ref = state

        n_steps = int(86400 / dt)  # 1 day
        for _ in range(n_steps):
            state = model.step(state, dt)

        errs = compute_error_norms_latlon(state, ref, grid)
        print(f"TC2 1-day: L1={errs['l1']:.6f}, L2={errs['l2']:.6f}, Linf={errs['linf']:.6f}")
        assert errs["l2"] < 0.05, f"TC2 1-day L2 error too large: {errs['l2']:.6f}"

    def test_tc2_five_day_stable(self):
        """TC2 should remain stable for 5 days."""
        from tests.atmosphere.shallow_water.test_cases.williamson_latlon import williamson_test2_latlon

        grid = create_latlon_grid(32)
        config = FVShallowWaterLatLonConfig(
            use_polar_filter=True,
        )
        dt = 900.0
        model = FVShallowWaterLatLonModel(grid, config, dt=dt)

        state = williamson_test2_latlon(grid)

        n_steps = int(5 * 86400 / dt)
        for _ in range(n_steps):
            state = model.step(state, dt)

        # Check that h is finite and bounded
        assert jnp.all(jnp.isfinite(state.h.data)), "h is not finite after 5 days"
        h_mean = jnp.mean(state.h.data)
        assert h_mean > 100.0, f"h collapsed: mean={float(h_mean):.2f}"


class TestWilliamsonTC5:
    """Test Williamson Test Case 5: Mountain flow."""

    def test_tc5_stable_15_days(self):
        """TC5 should remain stable for 15 days."""
        from tests.atmosphere.shallow_water.test_cases.williamson_latlon import williamson_test5_latlon

        grid = create_latlon_grid(32)
        config = FVShallowWaterLatLonConfig(
            use_polar_filter=True,
        )
        dt = 900.0
        model = FVShallowWaterLatLonModel(grid, config, dt=dt)

        state = williamson_test5_latlon(grid)

        n_steps = int(15 * 86400 / dt)
        for _ in range(n_steps):
            state = model.step(state, dt)

        assert jnp.all(jnp.isfinite(state.h.data)), "h is not finite after 15 days"
        h_mean = jnp.mean(state.h.data)
        assert h_mean > 100.0, f"h collapsed: mean={float(h_mean):.2f}"

    def test_tc5_topography_preserved(self):
        """Topography h_s should be unchanged after integration."""
        from tests.atmosphere.shallow_water.test_cases.williamson_latlon import williamson_test5_latlon

        grid = create_latlon_grid(32)
        dt = 900.0
        model = FVShallowWaterLatLonModel(grid, dt=dt)

        state = williamson_test5_latlon(grid)
        h_s_init = state.h_s.data

        for _ in range(10):
            state = model.step(state, dt)

        assert jnp.allclose(state.h_s.data, h_s_init, atol=1e-12), \
            "Topography changed during integration"


# ==============================================================================
# Conservation tests
# ==============================================================================

class TestConservation:
    """Test conservation properties."""

    def test_mass_conservation(self):
        """Mass should be conserved to high precision with the fixer."""
        from tests.atmosphere.shallow_water.test_cases.williamson_latlon import williamson_test2_latlon

        grid = create_latlon_grid(32)
        config = FVShallowWaterLatLonConfig(
            fix_mass=True,
            fix_energy=False,
            use_polar_filter=True,
        )
        dt = 900.0
        model = FVShallowWaterLatLonModel(grid, config, dt=dt)

        state = williamson_test2_latlon(grid)
        diag_init = compute_conservation_diagnostics(state, grid)
        mass_init = float(diag_init['total_mass'])

        for _ in range(50):
            state = model.step(state, dt)

        diag_final = compute_conservation_diagnostics(state, grid)
        mass_final = float(diag_final['total_mass'])

        rel_err = abs(mass_final - mass_init) / abs(mass_init)
        assert rel_err < 1e-4, f"Mass conservation failed: rel_err={rel_err:.2e}"


# ==============================================================================
# Differentiability tests
# ==============================================================================

class TestDifferentiability:
    """Test JAX differentiability."""

    def test_grad_single_step(self):
        """Single step should be differentiable via jax.grad."""
        from tests.atmosphere.shallow_water.test_cases.williamson_latlon import williamson_test2_latlon

        grid = create_latlon_grid(16)
        config = FVShallowWaterLatLonConfig(
            use_conservation_fixer=False,
            use_polar_filter=True,
        )
        dt = 1800.0
        model = FVShallowWaterLatLonModel(grid, config, dt=dt)

        state = williamson_test2_latlon(grid)

        def loss_fn(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            s_new = model.step(s, dt)
            return jnp.mean(s_new.h.data**2)

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(state.h.data)
        assert jnp.all(jnp.isfinite(grad_val)), "Gradient contains NaN/Inf"
        assert grad_val.shape == state.h.data.shape

    def test_grad_scan(self):
        """Multi-step scan integration should be differentiable."""
        from tests.atmosphere.shallow_water.test_cases.williamson_latlon import williamson_test2_latlon

        grid = create_latlon_grid(16)
        config = FVShallowWaterLatLonConfig(
            use_conservation_fixer=False,
            use_polar_filter=True,
        )
        dt = 1800.0
        model = FVShallowWaterLatLonModel(grid, config, dt=dt)

        state = williamson_test2_latlon(grid)

        def loss_fn(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            final, _ = model.integrate_scan(s, n_steps=3, dt=dt)
            return jnp.mean(final.h.data**2)

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(state.h.data)
        assert jnp.all(jnp.isfinite(grad_val)), "Gradient through scan contains NaN/Inf"
