"""Integration tests for the shallow water model (CDGrid)."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterModel, CDGridShallowWaterConfig,
    CDGridShallowWaterState, cdgrid_shallow_water_tendencies,
)
from tests.test_cases.williamson import (
    williamson_test2, williamson_test5,
)


def _sw_to_cdgrid(state, cdgrid):
    """Convert A-grid ShallowWaterState to CDGridShallowWaterState."""
    from legoesm.grids.halo import pad_halo_vector
    h = state.h.data
    u_center = state.u.data
    v_center = state.v.data
    h_s = state.h_s.data
    u_pad, v_pad = pad_halo_vector(
        u_center, v_center,
        cdgrid.base.cos_angle, cdgrid.base.sin_angle,
        cdgrid.base.cos_angle_padded, cdgrid.base.sin_angle_padded,
        interp_offsets=cdgrid.base.halo_interp_offsets,
    )
    u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1] +
                   u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
    v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1] +
                   v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
    return CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


class TestShallowWaterModel:
    """Integration tests for the CDGrid shallow water dynamical core."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(16)

    @pytest.fixture(scope="class")
    def cdgrid(self, grid):
        return create_cubed_sphere_cdgrid(grid)

    @pytest.fixture(scope="class")
    def model(self, grid):
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=1e15,
            use_conservation_fixer=True,
        )
        return CDGridShallowWaterModel(grid, config)

    def test_tendencies_shape(self, grid, cdgrid):
        """Tendencies should have correct shapes."""
        sw_state = williamson_test2(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        dh, du, dv = cdgrid_shallow_water_tendencies(state, cdgrid)
        n = grid.n
        assert dh.shape == (6, n, n)
        assert du.shape == (6, n + 1, n + 1)

    def test_tendencies_finite(self, grid, cdgrid):
        """All tendencies should be finite."""
        sw_state = williamson_test2(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        dh, du, dv = cdgrid_shallow_water_tendencies(state, cdgrid)
        assert jnp.all(jnp.isfinite(dh))
        assert jnp.all(jnp.isfinite(du))

    def test_single_step_finite(self, model, grid, cdgrid):
        """A single time step should produce finite values."""
        sw_state = williamson_test2(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        state_new = model.step(state, dt=600.0)
        assert jnp.all(jnp.isfinite(state_new.h))

    def test_multi_step_stability(self, model, grid, cdgrid):
        """Model should remain stable for 100 steps."""
        sw_state = williamson_test2(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        for _ in range(100):
            state = model.step(state, dt=120.0)
        assert jnp.all(jnp.isfinite(state.h))

    def test_mass_conservation(self, model, grid, cdgrid):
        """Mass should be conserved with fixer."""
        sw_state = williamson_test2(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        area = grid.area
        mass_init = float(jnp.sum(state.h * area))

        for _ in range(50):
            state = model.step(state, dt=600.0)

        mass_final = float(jnp.sum(state.h * area))
        # iter-164: centralized helper.
        from legoesm.diagnostics import compute_relative_drift
        rel_change = compute_relative_drift([mass_init, mass_final])
        assert rel_change < 1e-4, f"Mass relative change: {rel_change:.2e}"

    def test_topography_preserved(self, model, grid, cdgrid):
        """Surface topography should not change during integration."""
        sw_state = williamson_test5(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        h_s_init = state.h_s.copy()

        for _ in range(10):
            state = model.step(state, dt=600.0)

        assert jnp.allclose(state.h_s, h_s_init)


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

    @pytest.fixture(scope="class")
    def cdgrid(self, grid):
        return create_cubed_sphere_cdgrid(grid)

    def test_grad_through_single_step(self, grid, cdgrid):
        """jax.grad should work through a single model step."""
        model = CDGridShallowWaterModel(grid, CDGridShallowWaterConfig(
            use_conservation_fixer=False
        ))
        sw_state = williamson_test2(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)

        def loss(h_data):
            s = state._replace(h=h_data)
            s_new = model.step(s, dt=300.0)
            return jnp.mean(s_new.h ** 2)

        grads = jax.grad(loss)(state.h)
        assert jnp.all(jnp.isfinite(grads))
        assert not jnp.allclose(grads, 0.0)

    def test_grad_through_multi_step(self, grid, cdgrid):
        """jax.grad should work through multiple time steps."""
        model = CDGridShallowWaterModel(grid, CDGridShallowWaterConfig(
            use_conservation_fixer=False
        ))
        sw_state = williamson_test2(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)

        def loss(h_data):
            s = state._replace(h=h_data)
            for _ in range(5):
                s = model.step(s, dt=300.0)
            return jnp.mean(s.h ** 2)

        grads = jax.grad(loss)(state.h)
        assert jnp.all(jnp.isfinite(grads))

    def test_grad_through_scan(self, grid, cdgrid):
        """jax.grad should work through lax.scan integration."""
        config = CDGridShallowWaterConfig(use_conservation_fixer=False)
        model = CDGridShallowWaterModel(grid, config)
        sw_state = williamson_test2(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)

        def loss(h_data):
            s = state._replace(h=h_data)
            final, _ = model.integrate_scan(s, n_steps=5, dt=300.0)
            return jnp.mean(final.h ** 2)

        grads = jax.grad(loss)(state.h)
        assert jnp.all(jnp.isfinite(grads))
