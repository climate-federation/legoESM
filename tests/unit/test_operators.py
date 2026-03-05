"""Unit tests for discrete operators."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.operators import (
    gradient_x, gradient_y, gradient,
    divergence, curl_z, laplacian,
    global_integral, global_mean,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere


class TestOperators:
    """Tests for discrete differential operators."""

    def test_gradient_constant_field(self, small_grid):
        """Gradient of a constant field should be zero."""
        f = Field(data=jnp.ones((6, 8, 8)) * 5.0, name="const",
                  dims=("face", "x", "y"), units="m")
        gx, gy = gradient(f, small_grid)
        assert jnp.allclose(gx.data, 0.0, atol=1e-10)
        assert jnp.allclose(gy.data, 0.0, atol=1e-10)

    def test_divergence_zero_field(self, small_grid):
        """Divergence of zero vector field should be zero."""
        zero = Field(data=jnp.zeros((6, 8, 8)), name="zero",
                     dims=("face", "x", "y"), units="m/s")
        div = divergence(zero, zero, small_grid)
        assert jnp.allclose(div.data, 0.0, atol=1e-10)

    def test_laplacian_constant(self, small_grid):
        """Laplacian of a constant field should be zero."""
        f = Field(data=jnp.ones((6, 8, 8)) * 3.0, name="const",
                  dims=("face", "x", "y"), units="m")
        lap = laplacian(f, small_grid)
        assert jnp.allclose(lap.data, 0.0, atol=1e-6)

    def test_global_integral_ones(self, small_grid):
        """Global integral of ones should equal total area."""
        f = Field(data=jnp.ones((6, 8, 8)), name="ones",
                  dims=("face", "x", "y"), units="1")
        integral = global_integral(f, small_grid)
        assert jnp.allclose(integral, small_grid.total_area, rtol=1e-6)

    def test_global_mean_constant(self, small_grid):
        """Global mean of a constant field should be that constant."""
        val = 42.0
        f = Field(data=jnp.ones((6, 8, 8)) * val, name="const",
                  dims=("face", "x", "y"), units="1")
        mean = global_mean(f, small_grid)
        assert jnp.allclose(mean, val, rtol=1e-6)

    def test_operators_differentiable(self, small_grid):
        """All operators should be differentiable with jax.grad."""
        def loss(data):
            f = Field(data=data, name="test", dims=("face", "x", "y"), units="m")
            gx = gradient_x(f, small_grid)
            return jnp.sum(gx.data ** 2)

        data = jax.random.normal(jax.random.PRNGKey(0), (6, 8, 8))
        grad_fn = jax.grad(loss)
        grads = grad_fn(data)
        assert jnp.all(jnp.isfinite(grads))

    def test_divergence_differentiable(self, small_grid):
        """Divergence should be differentiable."""
        def loss(u_data, v_data):
            u = Field(data=u_data, name="u", dims=("face", "x", "y"), units="m/s")
            v = Field(data=v_data, name="v", dims=("face", "x", "y"), units="m/s")
            div = divergence(u, v, small_grid)
            return jnp.sum(div.data ** 2)

        key = jax.random.PRNGKey(1)
        u_data = jax.random.normal(key, (6, 8, 8))
        v_data = jax.random.normal(jax.random.split(key)[0], (6, 8, 8))
        grads = jax.grad(loss, argnums=(0, 1))(u_data, v_data)
        assert all(jnp.all(jnp.isfinite(g)) for g in grads)


class TestZeroMeanTendency:
    """Tests for the zero_mean_tendency correction."""

    def test_2d_zero_mean(self, small_grid):
        """2D tendency: correction reduces global integral by orders of magnitude."""
        from legoesm.core.conservation import zero_mean_tendency, _global_area_sum

        data = jax.random.normal(jax.random.PRNGKey(0), (6, 8, 8))
        uncorrected_sum = abs(float(_global_area_sum(data, small_grid)))
        corrected = zero_mean_tendency(data, small_grid)
        corrected_sum = abs(float(_global_area_sum(corrected, small_grid)))
        # Correction should reduce the global sum by at least 6 orders of magnitude
        assert corrected_sum < uncorrected_sum * 1e-6 or corrected_sum < 0.1

    def test_3d_zero_mean(self, small_grid):
        """3D tendency: correction reduces per-level global integrals."""
        from legoesm.core.conservation import zero_mean_tendency, _global_area_sum

        nlev = 5
        data = jax.random.normal(jax.random.PRNGKey(0), (6, 8, 8, nlev))
        corrected = zero_mean_tendency(data, small_grid)
        for k in range(nlev):
            orig = abs(float(_global_area_sum(data[..., k], small_grid)))
            fixed = abs(float(_global_area_sum(corrected[..., k], small_grid)))
            assert fixed < orig * 1e-5 or fixed < 0.1

    def test_zero_mean_preserves_pattern(self, small_grid):
        """Correction should only remove global mean, preserving spatial pattern."""
        from legoesm.core.conservation import zero_mean_tendency

        data = jax.random.normal(jax.random.PRNGKey(0), (6, 8, 8))
        corrected = zero_mean_tendency(data, small_grid)
        # Pattern should be the same up to a uniform offset
        diff = corrected - data
        assert jnp.std(diff) < 1e-6


class TestConservationFixers:
    """Tests for conservation fixer enhancements."""

    def test_fix_mass_hydrostatic_target(self, small_grid):
        """Target-anchored mass fixer restores exact target mass."""
        from legoesm.core.conservation import fix_mass_hydrostatic_target
        from legoesm.core.operators import global_integral
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState

        n = small_grid.n
        nlev = 5
        shape_2d = (6, n, n)
        shape_3d = (6, n, n, nlev)
        mk = lambda d, name, dims, u="1": Field(data=d, name=name, dims=dims, units=u)

        p_s_init = jnp.ones(shape_2d) * 1e5
        target_mass = global_integral(
            mk(p_s_init, "p_s", ("face", "x", "y"), "Pa"), small_grid,
        )

        # Perturbed p_s
        p_s_new = p_s_init + jax.random.normal(jax.random.PRNGKey(0), shape_2d) * 100.0
        state_new = HydrostaticState(
            u=mk(jnp.zeros(shape_3d), "u", ("face", "x", "y", "level"), "m/s"),
            v=mk(jnp.zeros(shape_3d), "v", ("face", "x", "y", "level"), "m/s"),
            T=mk(jnp.ones(shape_3d) * 300.0, "T", ("face", "x", "y", "level"), "K"),
            p_s=mk(p_s_new, "p_s", ("face", "x", "y"), "Pa"),
            phis=mk(jnp.zeros(shape_2d), "phis", ("face", "x", "y"), "m^2/s^2"),
        )
        fixed = fix_mass_hydrostatic_target(state_new, target_mass, small_grid)
        mass_fixed = global_integral(fixed.p_s, small_grid)
        rel_err = float(jnp.abs(mass_fixed - target_mass) / target_mass)
        assert rel_err < 1e-10, f"Relative error = {rel_err}"


class TestDifferentiability:
    """Tests that conservation code is differentiable with jax.grad."""

    def test_zero_mean_tendency_differentiable(self, small_grid):
        """zero_mean_tendency should be differentiable."""
        from legoesm.core.conservation import zero_mean_tendency

        def loss(data):
            corrected = zero_mean_tendency(data, small_grid)
            return jnp.sum(corrected ** 2)

        data = jax.random.normal(jax.random.PRNGKey(0), (6, 8, 8))
        grads = jax.grad(loss)(data)
        assert jnp.all(jnp.isfinite(grads))

    def test_zero_mean_tendency_3d_differentiable(self, small_grid):
        """3D zero_mean_tendency should be differentiable."""
        from legoesm.core.conservation import zero_mean_tendency

        def loss(data):
            corrected = zero_mean_tendency(data, small_grid)
            return jnp.sum(corrected ** 2)

        data = jax.random.normal(jax.random.PRNGKey(0), (6, 8, 8, 5))
        grads = jax.grad(loss)(data)
        assert jnp.all(jnp.isfinite(grads))

    def test_fix_mass_hydrostatic_target_differentiable(self, small_grid):
        """fix_mass_hydrostatic_target should be differentiable w.r.t. p_s."""
        from legoesm.core.conservation import fix_mass_hydrostatic_target
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState

        n = small_grid.n
        shape_2d = (6, n, n)
        shape_3d = (6, n, n, 5)
        mk = lambda d, name, dims, u="1": Field(data=d, name=name, dims=dims, units=u)
        target_mass = jnp.float32(1e5 * small_grid.total_area)

        def loss(p_s_data):
            state = HydrostaticState(
                u=mk(jnp.zeros(shape_3d), "u", ("face", "x", "y", "level"), "m/s"),
                v=mk(jnp.zeros(shape_3d), "v", ("face", "x", "y", "level"), "m/s"),
                T=mk(jnp.ones(shape_3d) * 300.0, "T", ("face", "x", "y", "level"), "K"),
                p_s=mk(p_s_data, "p_s", ("face", "x", "y"), "Pa"),
                phis=mk(jnp.zeros(shape_2d), "phis", ("face", "x", "y"), "m^2/s^2"),
            )
            fixed = fix_mass_hydrostatic_target(state, target_mass, small_grid)
            return jnp.sum(fixed.p_s.data ** 2)

        p_s_data = jnp.ones(shape_2d) * 1e5 + jax.random.normal(
            jax.random.PRNGKey(0), shape_2d,
        ) * 100.0
        grads = jax.grad(loss)(p_s_data)
        assert jnp.all(jnp.isfinite(grads))
