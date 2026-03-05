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


class TestFVFluxConservation:
    """Tests for FV flux operator conservation with symmetrized boundary fluxes."""

    @pytest.mark.skipif(
        not jax.config.jax_enable_x64,
        reason="Requires float64 for tight conservation check",
    )
    def test_fv_divergence_global_sum_zero(self):
        """Global sum of FV divergence * area should be near zero (float64)."""
        from legoesm.core.operators_fv import fv_flux_divergence

        grid = create_cubed_sphere(16)
        key = jax.random.PRNGKey(0)
        h = jax.random.uniform(key, (6, 16, 16), minval=0.5, maxval=1.5)
        u = jax.random.normal(jax.random.PRNGKey(1), (6, 16, 16))
        v = jax.random.normal(jax.random.PRNGKey(2), (6, 16, 16))
        div = fv_flux_divergence(h, u, v, grid, g=0.0)
        global_sum = float(jnp.sum(div * grid.area))
        assert abs(global_sum) < 1e-6, f"Global sum = {global_sum}"

    @pytest.mark.skipif(
        not jax.config.jax_enable_x64,
        reason="Requires float64 for tight conservation check",
    )
    def test_fv_scalar_advection_global_sum_zero(self):
        """Global sum of FV scalar advection * area should be near zero."""
        from legoesm.core.operators_fv import fv_scalar_advection

        grid = create_cubed_sphere(16)
        q = jax.random.uniform(jax.random.PRNGKey(0), (6, 16, 16), minval=0.5, maxval=1.5)
        u = jax.random.normal(jax.random.PRNGKey(1), (6, 16, 16))
        v = jax.random.normal(jax.random.PRNGKey(2), (6, 16, 16))
        adv = fv_scalar_advection(q, u, v, grid)
        global_sum = float(jnp.sum(adv * grid.area))
        assert abs(global_sum) < 1e-6, f"Global sum = {global_sum}"

    @pytest.mark.skipif(
        not jax.config.jax_enable_x64,
        reason="Requires float64 for tight conservation check",
    )
    def test_edge_flux_mismatch_diagnostic(self):
        """Edge flux mismatch diagnostic should be finite and non-negative."""
        from legoesm.core.operators_fv import (
            fv_flux_divergence, edge_flux_mismatch,
            pad_halo, pad_halo_vector, _reconstruct_x, _reconstruct_y,
            _get_halo_width, _halo_fields,
        )

        grid = create_cubed_sphere(16)
        h = jax.random.uniform(jax.random.PRNGKey(0), (6, 16, 16), minval=0.5, maxval=1.5)
        u = jax.random.normal(jax.random.PRNGKey(1), (6, 16, 16)) * 10.0
        v = jax.random.normal(jax.random.PRNGKey(2), (6, 16, 16)) * 10.0

        # Build fluxes the same way as fv_flux_divergence, BEFORE symmetrization
        halo = _get_halo_width("mc")
        interp_off, cos_ap, sin_ap = _halo_fields(grid, halo)
        u_pad, v_pad = pad_halo_vector(
            u, v, grid.cos_angle, grid.sin_angle,
            cos_ap, sin_ap, interp_offsets=interp_off, halo=halo,
        )
        h_pad = pad_halo(h, halo=halo, interp_offsets=interp_off)

        h_L_x, h_R_x = _reconstruct_x(h_pad, "mc")
        u_L_x, u_R_x = _reconstruct_x(u_pad, "mc")
        alpha_x = jnp.maximum(jnp.abs(u_L_x), jnp.abs(u_R_x))
        F_x = 0.5 * (h_L_x * u_L_x + h_R_x * u_R_x) - 0.5 * alpha_x * (h_R_x - h_L_x)
        hy_edge = 0.5 * (grid.hy_ext[:, :-1, 1:-1] + grid.hy_ext[:, 1:, 1:-1])
        Phi_x = F_x * hy_edge

        h_L_y, h_R_y = _reconstruct_y(h_pad, "mc")
        v_L_y, v_R_y = _reconstruct_y(v_pad, "mc")
        alpha_y = jnp.maximum(jnp.abs(v_L_y), jnp.abs(v_R_y))
        G_y = 0.5 * (h_L_y * v_L_y + h_R_y * v_R_y) - 0.5 * alpha_y * (h_R_y - h_L_y)
        hx_edge = 0.5 * (grid.hx_ext[:, 1:-1, :-1] + grid.hx_ext[:, 1:-1, 1:])
        Phi_y = G_y * hx_edge

        diag = edge_flux_mismatch(Phi_x, Phi_y)
        assert diag['max_mismatch'] >= 0.0
        assert diag['mean_mismatch'] >= 0.0
        assert len(diag['per_edge']) == 12


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


class TestFVMassTendency:
    """Tests for FV mass tendency global integral closure."""

    @pytest.mark.skipif(
        not jax.config.jax_enable_x64,
        reason="Requires float64 for tight conservation check",
    )
    def test_fv_divergence_with_zero_mean(self):
        """FV divergence with zero_mean_tendency has exact zero global sum."""
        from legoesm.core.operators_fv import fv_flux_divergence
        from legoesm.core.conservation import zero_mean_tendency

        grid = create_cubed_sphere(16)
        h = jax.random.uniform(jax.random.PRNGKey(0), (6, 16, 16), minval=0.5, maxval=1.5)
        u = jax.random.normal(jax.random.PRNGKey(1), (6, 16, 16)) * 10.0
        v = jax.random.normal(jax.random.PRNGKey(2), (6, 16, 16)) * 10.0
        div = fv_flux_divergence(h, u, v, grid, g=0.0)
        corrected = zero_mean_tendency(div, grid)
        global_sum = float(jnp.sum(corrected * grid.area))
        # After zero-mean correction, residual is limited by float64 precision
        # at Earth area scales (~5e14 m²)
        assert abs(global_sum) < 1e-4, f"Global sum = {global_sum}"


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
