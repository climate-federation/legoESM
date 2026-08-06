"""Unit tests for the hydrostatic primitive equation dynamical core."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.core.operators import global_integral
from legoesm.core.conservation import fix_mass_hydrostatic
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    SigmaCoordinate,
    create_sigma_coordinate,
    pressure_from_sigma,
    compute_geopotential,
    compute_sigma_dot,
    vertical_advection,
    compute_pressure_velocity,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig as PrimitiveEquationConfig,
    CDGridPrimitiveEquationModel as PrimitiveEquationModel,
    cdgrid_hydrostatic_tendencies as hydrostatic_tendencies,
)
from legoesm.core.operators_3d import (
    vorticity_3d as _vorticity_3d,
    divergence_3d as _divergence_3d,
    gradient_x_3d as _gradient_x_3d,
)
from legoesm import constants


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def grid():
    """Small cubed-sphere grid for testing."""
    return create_cubed_sphere(8)


@pytest.fixture
def cdgrid(grid):
    """CDGrid for testing."""
    return create_cubed_sphere_cdgrid(grid)


@pytest.fixture
def sigma():
    """5-level sigma coordinate for testing."""
    return create_sigma_coordinate(5)


@pytest.fixture
def sigma_20():
    """20-level sigma coordinate for testing."""
    return create_sigma_coordinate(20)


def _make_state(grid, sigma, T_val=300.0, u_val=0.0, v_val=0.0, p_s_val=1e5):
    """Helper: create a HydrostaticState with uniform fields."""
    shape_3d = (6, grid.n, grid.n, sigma.n_levels)
    shape_2d = (6, grid.n, grid.n)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return HydrostaticState(
        u=Field(data=jnp.ones(shape_3d) * u_val, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.ones(shape_3d) * v_val, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=jnp.ones(shape_3d) * T_val, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=jnp.ones(shape_2d) * p_s_val, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
    )


# ==============================================================================
# Sigma Coordinate Tests
# ==============================================================================

class TestSigmaCoordinate:
    """Tests for the sigma vertical coordinate."""

    def test_create_sigma_coordinate(self):
        """Sigma coordinate should have correct structure."""
        sigma = create_sigma_coordinate(20)
        assert sigma.n_levels == 20
        assert sigma.sigma_full.shape == (20,)
        assert sigma.sigma_half.shape == (21,)
        assert sigma.dsigma.shape == (20,)

    def test_sigma_half_boundaries(self):
        """Sigma half-levels should be sigma_top at top and 1 at surface."""
        sigma = create_sigma_coordinate(10)
        assert float(sigma.sigma_half[0]) == pytest.approx(0.01)  # sigma_top default
        assert float(sigma.sigma_half[-1]) == 1.0

    def test_dsigma_sums_to_range(self):
        """Layer thicknesses should sum to (1 - sigma_top)."""
        sigma = create_sigma_coordinate(20)
        assert jnp.allclose(jnp.sum(sigma.dsigma), 0.99, atol=1e-10)

    def test_sigma_top_zero(self):
        """Explicit sigma_top=0 should give dsigma summing to 1."""
        sigma = create_sigma_coordinate(10, sigma_top=0.0)
        assert float(sigma.sigma_half[0]) == 0.0
        assert jnp.allclose(jnp.sum(sigma.dsigma), 1.0, atol=1e-10)

    def test_sigma_full_within_bounds(self):
        """Full levels should be between 0 and 1."""
        sigma = create_sigma_coordinate(20)
        assert jnp.all(sigma.sigma_full > 0)
        assert jnp.all(sigma.sigma_full < 1)

    def test_sigma_full_are_midpoints(self):
        """Full levels should be midpoints of half levels."""
        sigma = create_sigma_coordinate(10)
        expected = 0.5 * (sigma.sigma_half[:-1] + sigma.sigma_half[1:])
        assert jnp.allclose(sigma.sigma_full, expected, atol=1e-12)


class TestPressure:
    """Tests for pressure computation."""

    def test_pressure_from_sigma(self, grid, sigma):
        """p = σ * p_s should give correct shape and values."""
        p_s = jnp.ones((6, 8, 8), dtype=jnp.float32) * 1e5
        p = pressure_from_sigma(sigma.sigma_full, p_s)
        assert p.shape == (6, 8, 8, 5)
        # At surface (σ≈1): p ≈ p_s (float32 precision)
        assert jnp.allclose(p[..., -1], sigma.sigma_full[-1] * 1e5, rtol=1e-5)

    def test_pressure_half_levels(self, grid, sigma):
        """Pressure at half levels should include p_top at top and p=p_s at surface."""
        p_s = jnp.ones((6, 8, 8), dtype=jnp.float32) * 1e5
        p_half = pressure_from_sigma(sigma.sigma_half, p_s)
        assert p_half.shape == (6, 8, 8, 6)
        # Top: p = sigma_top * p_s = 0.01 * 1e5 = 1000 Pa (float32 precision)
        assert jnp.allclose(p_half[..., 0], 0.01 * 1e5, rtol=1e-5)
        assert jnp.allclose(p_half[..., -1], 1e5, rtol=1e-5)  # surface

    def test_pressure_monotonic(self, grid, sigma):
        """Pressure should increase with level index (top to bottom)."""
        p_s = jnp.ones((6, 8, 8)) * 1e5
        p = pressure_from_sigma(sigma.sigma_full, p_s)
        # Level 0 (top) < Level 1 < ... < Level N-1 (bottom)
        for k in range(sigma.n_levels - 1):
            assert jnp.all(p[..., k] < p[..., k + 1])


class TestGeopotential:
    """Tests for geopotential computation."""

    def test_isothermal_exact(self, grid, sigma):
        """Isothermal atmosphere should have geopotential close to analytic.

        For constant T, the continuous hydrostatic equation gives:
            Φ(p) = Φ_s + R_d·T·ln(p_s/p)

        At full level k: p_k = σ_k · p_s, so:
            Φ_k = Φ_s - R_d·T·ln(σ_k)

        The Simmons-Burridge (1981) discrete integration uses
        α_k = 1 - (σ_{k-1/2}/Δσ_k)·ln(σ_{k+1/2}/σ_{k-1/2}), which
        is second-order accurate but NOT algebraically identical to the
        continuous formula.  We check against the continuous reference
        with a tolerance appropriate for the discrete approximation.
        """
        T_val = 250.0
        p_s_val = 1e5
        shape_3d = (6, 8, 8, sigma.n_levels)
        shape_2d = (6, 8, 8)

        T = jnp.ones(shape_3d) * T_val
        p_s = jnp.ones(shape_2d) * p_s_val
        phis = jnp.zeros(shape_2d)

        Phi = compute_geopotential(T, p_s, sigma, phis)

        # Continuous analytic reference
        R_d = constants.R_d
        sigma_full = sigma.sigma_full

        # Check each level (except top which may have large ln(σ))
        for k in range(1, sigma.n_levels):
            expected_k = float(-R_d * T_val * jnp.log(sigma_full[k]))
            computed_k = float(Phi[0, 4, 4, k])
            rel_err = abs(computed_k - expected_k) / abs(expected_k)
            # SB81 discrete alpha is O(Δσ²) accurate vs continuous;
            # with 5 coarse levels the error can be ~2% at upper levels.
            assert rel_err < 5e-2, (
                f"Level {k}: expected {expected_k:.2f}, got {computed_k:.2f} "
                f"(rel err {rel_err:.2e})"
            )

    def test_geopotential_decreases_with_height(self, grid, sigma):
        """Geopotential should increase with height (decrease with level index)."""
        shape_3d = (6, 8, 8, sigma.n_levels)
        shape_2d = (6, 8, 8)
        T = jnp.ones(shape_3d) * 250.0
        p_s = jnp.ones(shape_2d) * 1e5
        phis = jnp.zeros(shape_2d)

        Phi = compute_geopotential(T, p_s, sigma, phis)

        # Level 0 (top) should have highest geopotential
        for k in range(sigma.n_levels - 1):
            assert jnp.all(Phi[..., k] > Phi[..., k + 1]), (
                f"Geopotential not decreasing: level {k} vs {k+1}"
            )

    def test_geopotential_with_topography(self, grid, sigma):
        """Surface geopotential should shift all levels."""
        shape_3d = (6, 8, 8, sigma.n_levels)
        shape_2d = (6, 8, 8)
        T = jnp.ones(shape_3d) * 250.0
        p_s = jnp.ones(shape_2d) * 1e5

        phis_flat = jnp.zeros(shape_2d)
        phis_mountain = jnp.ones(shape_2d) * 1000.0  # 1000 m²/s² ≈ 100m

        Phi_flat = compute_geopotential(T, p_s, sigma, phis_flat)
        Phi_mountain = compute_geopotential(T, p_s, sigma, phis_mountain)

        # Mountain should shift geopotential at all levels
        diff = Phi_mountain - Phi_flat
        assert jnp.allclose(diff, 1000.0, rtol=1e-5)


class TestSigmaDot:
    """Tests for sigma-dot computation."""

    def test_sigma_dot_boundaries(self, sigma):
        """Sigma-dot should be zero at top and bottom."""
        shape = (6, 8, 8, sigma.n_levels)
        div_3d = jnp.ones(shape) * 1e-5  # Some divergence

        sigma_dot = compute_sigma_dot(div_3d, sigma)

        assert sigma_dot.shape == (6, 8, 8, sigma.n_levels + 1)
        assert jnp.allclose(sigma_dot[..., 0], 0.0, atol=1e-15)
        assert jnp.allclose(sigma_dot[..., -1], 0.0, atol=1e-10)

    def test_sigma_dot_zero_divergence(self, sigma):
        """Zero divergence should give zero sigma-dot everywhere."""
        shape = (6, 8, 8, sigma.n_levels)
        div_3d = jnp.zeros(shape)

        sigma_dot = compute_sigma_dot(div_3d, sigma)
        assert jnp.allclose(sigma_dot, 0.0, atol=1e-15)

    def test_sigma_dot_uniform_divergence(self, sigma):
        """Uniform divergence should give near-zero sigma-dot.

        For σ_top=0 this is exactly zero. With σ_top>0, there is a small
        residual O(σ_top · D) because the column integral of Δσ = 1-σ_top
        instead of 1. This is standard for GCMs with finite model tops.
        """
        shape = (6, 8, 8, sigma.n_levels)
        div_3d = jnp.ones(shape) * 1e-5  # Uniform divergence

        sigma_dot = compute_sigma_dot(div_3d, sigma)

        # Residual is O(sigma_top * D) = O(0.01 * 1e-5) = O(1e-7)
        assert jnp.allclose(sigma_dot, 0.0, atol=1e-6)


class TestVerticalAdvection:
    """Tests for vertical advection."""

    def test_uniform_field_zero_tendency(self, sigma):
        """Advecting a uniform field should give zero tendency everywhere.

        For uniform field C, ∂C/∂σ = 0, so -σ̇·∂C/∂σ = 0.
        """
        shape = (6, 8, 8, sigma.n_levels)
        field = jnp.ones(shape) * 300.0  # Uniform temperature

        # Non-trivial sigma_dot profile
        sigma_dot = jnp.zeros((6, 8, 8, sigma.n_levels + 1))
        sigma_dot = sigma_dot.at[..., 1:-1].set(0.01)

        tendency = vertical_advection(field, sigma_dot, sigma)
        # Advective form: -σ̇ · ∂f/∂σ = 0 for uniform field at ALL levels
        assert jnp.allclose(tendency, 0.0, atol=1e-10)

    def test_zero_sigma_dot_zero_tendency(self, sigma):
        """Zero sigma-dot should give zero vertical advection."""
        shape = (6, 8, 8, sigma.n_levels)
        field = jnp.linspace(200, 300, sigma.n_levels)[None, None, None, :]
        field = jnp.broadcast_to(field, shape)

        sigma_dot = jnp.zeros((6, 8, 8, sigma.n_levels + 1))

        tendency = vertical_advection(field, sigma_dot, sigma)
        assert jnp.allclose(tendency, 0.0, atol=1e-15)

    def test_vertical_advection_finite(self, sigma):
        """Vertical advection should produce finite values."""
        shape = (6, 8, 8, sigma.n_levels)
        field = jnp.linspace(200, 300, sigma.n_levels)[None, None, None, :]
        field = jnp.broadcast_to(field, shape)

        sigma_dot = jnp.zeros((6, 8, 8, sigma.n_levels + 1))
        sigma_dot = sigma_dot.at[..., 1:-1].set(0.001)

        tendency = vertical_advection(field, sigma_dot, sigma)
        assert jnp.all(jnp.isfinite(tendency))

    def test_linear_field_exact(self, sigma):
        """Advecting a linear field f=aσ+b should give -σ̇·a exactly."""
        shape = (6, 8, 8, sigma.n_levels)
        a, b = 100.0, 200.0  # f = 100σ + 200
        field = a * sigma.sigma_full[None, None, None, :] + b
        field = jnp.broadcast_to(field, shape)

        # Constant σ̇ at all interior interfaces
        sigma_dot = jnp.zeros((6, 8, 8, sigma.n_levels + 1))
        sigma_dot = sigma_dot.at[..., 1:-1].set(0.005)

        tendency = vertical_advection(field, sigma_dot, sigma)

        # At full levels: σ̇_full = 0.5*(σ̇_{k-1/2} + σ̇_{k+1/2})
        # For interior levels (not top/bottom): σ̇_full = 0.005
        # Expected: -σ̇ · a = -0.005 * 100 = -0.5
        # Top and bottom levels have σ̇_full = 0.5*0.005 = 0.0025
        # so expected = -0.0025 * 100 = -0.25
        interior = tendency[..., 1:-1]
        assert jnp.allclose(interior, -0.005 * a, rtol=1e-5)


# ==============================================================================
# 3D Operator Tests
# ==============================================================================

class TestOperators3D:
    """Tests for 3D operator wrappers."""

    def test_vorticity_3d_shape(self, grid, sigma):
        """3D vorticity should have correct shape."""
        shape = (6, 8, 8, sigma.n_levels)
        u = jnp.ones(shape) * 10.0
        v = jnp.ones(shape) * 5.0
        vort = _vorticity_3d(u, v, grid)
        assert vort.shape == shape

    def test_divergence_3d_shape(self, grid, sigma):
        """3D divergence should have correct shape."""
        shape = (6, 8, 8, sigma.n_levels)
        u = jnp.ones(shape) * 10.0
        v = jnp.ones(shape) * 5.0
        div = _divergence_3d(u, v, grid)
        assert div.shape == shape

    def test_gradient_3d_shape(self, grid, sigma):
        """3D gradient should have correct shape."""
        shape = (6, 8, 8, sigma.n_levels)
        f = jnp.ones(shape) * 100.0
        gx = _gradient_x_3d(f, grid)
        assert gx.shape == shape

    def test_vorticity_3d_finite(self, grid, sigma):
        """3D vorticity should be finite."""
        shape = (6, 8, 8, sigma.n_levels)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, shape)
        v = jax.random.normal(k2, shape)
        vort = _vorticity_3d(u, v, grid)
        assert jnp.all(jnp.isfinite(vort))


# ==============================================================================
# Hydrostatic Tendency Tests
# ==============================================================================

class TestHydrostaticTendencies:
    """Tests for the hydrostatic primitive equation tendency computation."""

    def test_isothermal_rest_state_small_tendencies(self, grid, cdgrid, sigma):
        """Isothermal atmosphere at rest should have near-zero tendencies.

        For uniform T, u=v=0, uniform p_s, phis=0:
        - No pressure gradients
        - No advection (v=0)
        - No vorticity (u=v=0)
        - Geopotential is horizontally uniform → ∇B = 0
        - σ̇ = 0 (no divergence)
        """
        state = _make_state(grid, sigma, T_val=250.0, u_val=0.0, v_val=0.0)
        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)

        tend = hydrostatic_tendencies(state, grid, sigma, cdgrid, config)

        # All tendencies should be near zero
        assert jnp.allclose(tend.du_dt.data, 0.0, atol=1e-8), \
            f"du_dt max: {float(jnp.max(jnp.abs(tend.du_dt.data)))}"
        assert jnp.allclose(tend.dv_dt.data, 0.0, atol=1e-8), \
            f"dv_dt max: {float(jnp.max(jnp.abs(tend.dv_dt.data)))}"
        assert jnp.allclose(tend.dT_dt.data, 0.0, atol=1e-6), \
            f"dT_dt max: {float(jnp.max(jnp.abs(tend.dT_dt.data)))}"
        assert jnp.allclose(tend.dp_s_dt.data, 0.0, atol=1e-4), \
            f"dp_s_dt max: {float(jnp.max(jnp.abs(tend.dp_s_dt.data)))}"
        assert jnp.allclose(tend.dphis_dt.data, 0.0, atol=1e-15)

    def test_tendencies_finite(self, grid, cdgrid, sigma):
        """Tendencies should be finite for any reasonable state."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)

        tend = hydrostatic_tendencies(state, grid, sigma, cdgrid, config)

        assert jnp.all(jnp.isfinite(tend.du_dt.data)), "du_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dv_dt.data)), "dv_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dT_dt.data)), "dT_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data)), "dp_s_dt not finite"

    def test_tendencies_with_hyperdiffusion(self, grid, cdgrid, sigma):
        """Tendencies with hyperdiffusion should be finite."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = PrimitiveEquationConfig(hyperdiff_coeff=1e15)

        tend = hydrostatic_tendencies(state, grid, sigma, cdgrid, config)

        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))

    def test_tendency_pytree_structure(self, grid, cdgrid, sigma):
        """Tendency should have same structure as state."""
        state = _make_state(grid, sigma)
        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)

        tend = hydrostatic_tendencies(state, grid, sigma, cdgrid, config)

        assert isinstance(tend, HydrostaticTendencies)
        assert tend.du_dt.data.shape == state.u.data.shape
        assert tend.dv_dt.data.shape == state.v.data.shape
        assert tend.dT_dt.data.shape == state.T.data.shape
        assert tend.dp_s_dt.data.shape == state.p_s.data.shape
        assert tend.dphis_dt.data.shape == state.phis.data.shape


# ==============================================================================
# Model Integration Tests
# ==============================================================================

class TestPrimitiveEquationModel:
    """Tests for the PrimitiveEquationModel class."""

    def test_single_step_finite(self, grid, sigma):
        """A single time step should produce finite results."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=5.0, v_val=2.0)
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
        )
        model = PrimitiveEquationModel(grid, sigma, config)

        state_new = model.step(state, dt=300.0)

        assert jnp.all(jnp.isfinite(state_new.u.data)), "u not finite after step"
        assert jnp.all(jnp.isfinite(state_new.v.data)), "v not finite after step"
        assert jnp.all(jnp.isfinite(state_new.T.data)), "T not finite after step"
        assert jnp.all(jnp.isfinite(state_new.p_s.data)), "p_s not finite after step"

    def test_rest_state_stays_at_rest(self, grid, sigma):
        """A rest state should remain nearly at rest after one step."""
        state = _make_state(grid, sigma, T_val=250.0, u_val=0.0, v_val=0.0)
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
        )
        model = PrimitiveEquationModel(grid, sigma, config)

        state_new = model.step(state, dt=300.0)

        # Winds should stay near zero
        assert jnp.allclose(state_new.u.data, 0.0, atol=1e-4), \
            f"u drifted: max={float(jnp.max(jnp.abs(state_new.u.data)))}"
        assert jnp.allclose(state_new.v.data, 0.0, atol=1e-4), \
            f"v drifted: max={float(jnp.max(jnp.abs(state_new.v.data)))}"

    def test_mass_conservation_with_fixer(self, grid, sigma):
        """Mass fixer should preserve global dry air mass."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
            fix_mass=True,
        )
        model = PrimitiveEquationModel(grid, sigma, config)

        mass_before = float(global_integral(state.p_s, grid))
        state_new = model.step(state, dt=300.0)
        mass_after = float(global_integral(state_new.p_s, grid))

        # float32 limits precision to ~1e-7 relative; mass fixer is exact
        # in infinite precision, but float32 accumulation rounds off
        assert abs(mass_after - mass_before) / abs(mass_before) < 1e-5, (
            f"Mass not conserved: before={mass_before:.6e}, after={mass_after:.6e}"
        )


# ==============================================================================
# Conservation Tests
# ==============================================================================

class TestMassFixerHydrostatic:
    """Tests for the hydrostatic mass fixer."""

    def test_mass_fixer_exact(self, grid, sigma):
        """Mass fixer should restore exact mass."""
        state_old = _make_state(grid, sigma, T_val=280.0, p_s_val=1e5)

        # Perturb surface pressure
        key = jax.random.PRNGKey(0)
        perturbation = jax.random.normal(key, state_old.p_s.data.shape) * 100.0
        state_new = state_old._replace(
            p_s=state_old.p_s.replace(data=state_old.p_s.data + perturbation)
        )

        state_fixed = fix_mass_hydrostatic(state_new, state_old, grid)

        mass_old = float(global_integral(state_old.p_s, grid))
        mass_fixed = float(global_integral(state_fixed.p_s, grid))
        # float32 limits precision to ~1e-7 relative
        assert abs(mass_fixed - mass_old) / abs(mass_old) < 1e-5

    def test_mass_fixer_preserves_gradients(self, grid, sigma):
        """Mass fixer should be differentiable."""
        state = _make_state(grid, sigma)

        def loss(ps_data):
            state_new = state._replace(
                p_s=state.p_s.replace(data=ps_data + 100.0)
            )
            state_fixed = fix_mass_hydrostatic(state_new, state, grid)
            return jnp.sum(state_fixed.p_s.data ** 2)

        grads = jax.grad(loss)(state.p_s.data)
        assert jnp.all(jnp.isfinite(grads))


# ==============================================================================
# Differentiability Tests
# ==============================================================================

class TestDifferentiability:
    """Tests that the PE model is differentiable."""

    def test_grad_through_tendencies(self, grid, cdgrid, sigma):
        """jax.grad should work through tendency computation."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=5.0, v_val=2.0)
        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)

        def loss(u_data):
            state_new = state._replace(u=state.u.replace(data=u_data))
            tend = hydrostatic_tendencies(state_new, grid, sigma, cdgrid, config)
            return jnp.sum(tend.du_dt.data ** 2)

        grads = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grads)), "Gradients through tendencies are not finite"

    def test_grad_through_single_step(self, grid, sigma):
        """jax.grad should work through a single model step."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=5.0, v_val=2.0)
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
        )
        model = PrimitiveEquationModel(grid, sigma, config)

        def loss(u_data):
            state_new = state._replace(u=state.u.replace(data=u_data))
            state_stepped = model.step(state_new, dt=60.0)
            return jnp.sum(state_stepped.u.data ** 2)

        grads = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grads)), "Gradients through model step are not finite"


# ==============================================================================
# Pressure Velocity Tests
# ==============================================================================

class TestPressureVelocity:
    """Tests for omega (pressure velocity) computation."""

    def test_omega_zero_for_rest(self, sigma):
        """Omega should be zero when σ̇=0 and dp_s/dt=0."""
        shape_2d = (6, 8, 8)
        sigma_dot = jnp.zeros((*shape_2d, sigma.n_levels + 1))
        p_s = jnp.ones(shape_2d) * 1e5
        dp_s_dt = jnp.zeros(shape_2d)

        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt, sigma)
        assert jnp.allclose(omega, 0.0, atol=1e-15)

    def test_omega_shape(self, sigma):
        """Omega should have shape (6,n,n,nlev)."""
        shape_2d = (6, 8, 8)
        sigma_dot = jnp.zeros((*shape_2d, sigma.n_levels + 1))
        p_s = jnp.ones(shape_2d) * 1e5
        dp_s_dt = jnp.ones(shape_2d) * 10.0

        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt, sigma)
        assert omega.shape == (*shape_2d, sigma.n_levels)


# ==============================================================================
# Continuity Closure Tests
# ==============================================================================

class TestContinuityClosure:
    """Test that the discrete continuity equation is exactly satisfied.

    The σ-coordinate continuity equation:
        ∂(ln p_s)/∂t + div(v_k) + (σ̇_{k+1/2} - σ̇_{k-1/2}) / Δσ_k = 0

    must hold at every level to machine precision when dp_s/dt and σ̇
    are derived from the same div(v) field.
    """

    def test_continuity_closure_uniform_ps(self, grid, sigma):
        """Continuity closure with uniform p_s (as in baroclinic wave IC)."""
        # Create a state with nonzero winds but uniform p_s
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        u, v = state.u.data, state.v.data
        p_s = state.p_s.data
        dsigma = sigma.dsigma
        sigma_top = float(sigma.sigma_half[0])
        sigma_range = 1.0 - sigma_top

        # Compute divergence
        div_v = _divergence_3d(u, v, grid)
        D_total = jnp.sum(div_v * dsigma[None, None, None, :], axis=-1)

        # dp_s/dt and sigma_dot from the same div_v
        dp_s_dt = -p_s * D_total / sigma_range
        dlnps_dt = dp_s_dt / p_s
        sigma_dot = compute_sigma_dot(div_v, sigma)

        # Check closure at every level
        for k in range(sigma.n_levels):
            dsigma_dot = (sigma_dot[..., k + 1] - sigma_dot[..., k]) / dsigma[k]
            residual = dlnps_dt + div_v[..., k] + dsigma_dot
            max_res = float(jnp.max(jnp.abs(residual)))
            max_div = float(jnp.max(jnp.abs(div_v[..., k])))
            # Residual should be near machine precision relative to div_v
            assert max_res < 1e-6 * max(max_div, 1e-15), (
                f"Level {k}: continuity residual {max_res:.2e} too large "
                f"relative to div_v {max_div:.2e}"
            )

    def test_continuity_closure_nonuniform_ps(self, grid, sigma):
        """Continuity closure with non-uniform p_s."""
        # Create state with spatially varying p_s
        key = jax.random.PRNGKey(123)
        shape_3d = (6, 8, 8, sigma.n_levels)
        shape_2d = (6, 8, 8)

        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, shape_3d) * 10.0
        v = jax.random.normal(k2, shape_3d) * 10.0
        p_s = jnp.ones(shape_2d) * 1e5 + jax.random.normal(key, shape_2d) * 1000.0

        dsigma = sigma.dsigma
        sigma_top = float(sigma.sigma_half[0])
        sigma_range = 1.0 - sigma_top

        div_v = _divergence_3d(u, v, grid)
        D_total = jnp.sum(div_v * dsigma[None, None, None, :], axis=-1)
        dp_s_dt = -p_s * D_total / sigma_range
        dlnps_dt = dp_s_dt / p_s
        sigma_dot = compute_sigma_dot(div_v, sigma)

        # Check closure at every level
        for k in range(sigma.n_levels):
            dsigma_dot = (sigma_dot[..., k + 1] - sigma_dot[..., k]) / dsigma[k]
            residual = dlnps_dt + div_v[..., k] + dsigma_dot
            max_res = float(jnp.max(jnp.abs(residual)))
            max_div = float(jnp.max(jnp.abs(div_v[..., k])))
            assert max_res < 1e-6 * max(max_div, 1e-15), (
                f"Level {k}: continuity residual {max_res:.2e} too large "
                f"(non-uniform p_s)"
            )


# ==============================================================================
# Temperature Stability Regression Tests
# ==============================================================================

class TestTemperatureStability:
    """Regression tests for temperature staying within physical bounds.

    After several time steps with Held-Suarez forcing or baroclinic wave
    dynamics, temperatures should remain within reasonable atmospheric
    bounds (100-400 K) everywhere.
    """

    def test_held_suarez_temperature_bounds(self, grid, sigma_20):
        """Held-Suarez forcing should keep T within bounds over 100 steps."""
        from legoesm.atmosphere.forcing.idealized.held_suarez import (
            held_suarez_forcing,
            held_suarez_init,
        )

        model = PrimitiveEquationModel(
            grid, sigma_20,
            PrimitiveEquationConfig(
                hyperdiff_coeff=1e18,
                use_conservation_fixer=True,
                fix_mass=True,
            ),
        )
        state = held_suarez_init(grid, sigma_20)

        # Run 100 steps with dt=600s (~17 hours)
        for _ in range(100):
            state = model.step_with_physics(state, 600.0, held_suarez_forcing)

        T = state.T.data
        T_min = float(jnp.min(T))
        T_max = float(jnp.max(T))

        assert T_min > 100.0, (
            f"Temperature too cold: T_min = {T_min:.1f} K (expected > 100 K)"
        )
        assert T_max < 400.0, (
            f"Temperature too hot: T_max = {T_max:.1f} K (expected < 400 K)"
        )
        assert jnp.all(jnp.isfinite(T)), "Non-finite temperatures detected"

    def test_baroclinic_wave_temperature_bounds(self, grid):
        """Baroclinic wave IC should keep T within bounds over 50 steps."""
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init

        sigma_26 = create_sigma_coordinate(26)
        model = PrimitiveEquationModel(
            grid, sigma_26,
            PrimitiveEquationConfig(
                hyperdiff_coeff=1e18,
                use_conservation_fixer=True,
                fix_mass=True,
            ),
        )
        state = baroclinic_wave_init(grid, sigma_26, perturbed=False)

        # Run 50 steps with dt=600s (~8 hours)
        for _ in range(50):
            state = model.step(state, 600.0)

        T = state.T.data
        T_min = float(jnp.min(T))
        T_max = float(jnp.max(T))

        assert T_min > 100.0, (
            f"Temperature too cold: T_min = {T_min:.1f} K (expected > 100 K)"
        )
        assert T_max < 400.0, (
            f"Temperature too hot: T_max = {T_max:.1f} K (expected < 400 K)"
        )
        assert jnp.all(jnp.isfinite(T)), "Non-finite temperatures detected"


class TestMassConservation:
    """Tests for mass conservation options in PE model."""

    def test_anchor_mass_to_initial(self, grid, sigma):
        """Anchor to initial mass prevents drift over multiple steps."""
        config = PrimitiveEquationConfig(
            anchor_mass_to_initial=True,
        )
        model = PrimitiveEquationModel(grid, sigma, config)
        state = _make_state(grid, sigma, T_val=300.0, u_val=5.0)
        from legoesm.core.operators import global_integral
        initial_mass = float(global_integral(state.p_s, grid))
        # Run a few steps
        for _ in range(3):
            state = model.step(state, 60.0)
        final_mass = float(global_integral(state.p_s, grid))
        rel_err = abs(final_mass - initial_mass) / abs(initial_mass)
        # fp32 ULP floor: the corrected p_s is quantized back to fp32
        # storage between steps, so cumulative quantization sets a 1e-6
        # ULP floor on the relative drift (was 1e-8 originally — only
        # achievable on a bit-exact fp64 storage, which we no longer
        # have under the default precision policy).
        assert rel_err < 1e-6, f"Mass drift = {rel_err}"


class TestHydrostaticToFV3VectorHalo:
    """Pin the iter-199 fix: ``hydrostatic_to_fv3`` must use
    ``pad_halo_vector`` (with cos/sin angle rotation across cube
    faces), NOT the scalar ``interp_center_to_corner`` path.  Pre-fix
    the scalar halo introduced ~70 % distortion on the round-trip and
    drove a 1009 Pa/step ps imbalance from a balanced JW jet IC.
    See scaling.md §3 for the audit history.
    """

    def test_round_trip_preserves_vector_field(self):
        """``state → hydrostatic_to_fv3 → fv3_to_hydrostatic`` round-trip
        on the JW BCW IC must preserve cell-centre winds to within ~1
        m/s (vector-aware halo); the pre-fix scalar halo distorted u
        by ~20 m/s out of a 28 m/s field at C24."""
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            hydrostatic_to_fv3, fv3_to_hydrostatic,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init

        n = 24
        cs_grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(cs_grid)
        sigma = create_sigma_coordinate(8)
        state_cc = baroclinic_wave_init(cs_grid, sigma, perturbed=False)
        state_fv3 = hydrostatic_to_fv3(state_cc, cdgrid)
        state_back = fv3_to_hydrostatic(state_fv3, cdgrid)

        u_err = float(jnp.max(jnp.abs(state_back.u.data - state_cc.u.data)))
        v_err = float(jnp.max(jnp.abs(state_back.v.data - state_cc.v.data)))
        # Vector-aware halo (post-fix): u_err ≈ 0.32 m/s.  Pre-fix:
        # ≈ 19.6 m/s (70 % of field).  Threshold of 1 m/s catches a
        # regression to the scalar-halo path while tolerating the
        # legitimate centre→corner→centre interpolation roundoff.
        assert u_err < 1.0, (
            f"u round-trip distortion {u_err:.2f} m/s — likely a "
            f"reversion to scalar pad_halo in hydrostatic_to_fv3"
        )
        assert v_err < 1.0, (
            f"v round-trip distortion {v_err:.2f} m/s — likely a "
            f"reversion to scalar pad_halo in hydrostatic_to_fv3"
        )

    def test_ic_divergence_is_balanced(self):
        """The JW jet's analytic state has near-zero horizontal
        divergence by construction.  After ``hydrostatic_to_fv3``,
        the C-grid divergence diagnosed from the projected D-grid
        winds must stay below ~1e-6 s^-1.  Pre-iter-199 the scalar
        halo produced |div_v|_max = 5.3e-5 s^-1 (146× too high) and
        drove the BCW blow-up at day 0.35.
        """
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            hydrostatic_to_fv3,
        )
        from legoesm.core.operators_cdgrid import dgrid_to_cgrid, cgrid_divergence
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init

        n = 24
        cs_grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(cs_grid)
        sigma = create_sigma_coordinate(8)
        state_cc = baroclinic_wave_init(cs_grid, sigma, perturbed=False)
        state_fv3 = hydrostatic_to_fv3(state_cc, cdgrid)
        u_c, v_c = dgrid_to_cgrid(state_fv3.u_d.data, state_fv3.v_d.data, cdgrid)
        div_v = cgrid_divergence(u_c, v_c, cdgrid)
        max_div = float(jnp.max(jnp.abs(div_v)))
        # 1e-6 catches regression cleanly: scalar halo gave 5.3e-5,
        # vector halo gives 3.8e-7; this threshold is 10× safety.
        assert max_div < 1e-6, (
            f"|div_v|_max = {max_div:.2e} 1/s on JW IC after "
            f"hydrostatic_to_fv3 — possible reversion to scalar "
            f"pad_halo (was 5.3e-5 1/s pre-iter-199)"
        )

    def test_first_step_ps_balanced(self):
        """One BCW step from the JW IC must leave ps within ~10 hPa of
        1000 hPa.  Pre-iter-199 the scalar halo distorted the IC enough
        to shift ps by 1009 Pa in step 1 (1 % of total field).  The
        post-fix step-1 swing is ~7 Pa.  Threshold of 100 Pa is the
        right gate: catches a reversion (which gives 1000+ Pa) without
        false-positives on legitimate gravity-wave adjustment.
        """
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
            hydrostatic_to_fv3, fv3_to_hydrostatic,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init

        n = 24
        cs_grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(cs_grid)
        sigma = create_sigma_coordinate(8)
        state_cc = baroclinic_wave_init(cs_grid, sigma, perturbed=False)
        state_fv3 = hydrostatic_to_fv3(state_cc, cdgrid)
        # Disable conservation fixer so we measure the raw dycore
        # imbalance, not the post-step correction.
        config = CDGridPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
            use_conservation_fixer=False, fix_mass=False,
            anchor_mass_to_initial=False,
        )
        model = CDGridPrimitiveEquationModel(cs_grid, sigma, config)
        state_fv3 = model.step(state_fv3, 300.0)
        cc = fv3_to_hydrostatic(state_fv3, cdgrid)
        ps = cc.p_s.data
        ps_init = state_cc.p_s.data
        max_swing = float(jnp.max(jnp.abs(ps - ps_init)))
        assert max_swing < 100.0, (
            f"Step-1 ps swing on JW IC: {max_swing:.1f} Pa — possible "
            f"reversion to scalar pad_halo (was ~1009 Pa pre-iter-199)"
        )


class TestSpongeDefault1028:
    """#1028: pin the cd-grid top-sponge default at the FV3 Ray_fast-like
    days scale.

    The old 1-hour default (tau=3600 s) was ~430-860x stronger than FV3's
    Ray_fast and measured as the dominant global KE sink on a balanced jet
    (-1.0/day, fp64 budget closed to 4e-16; 99% of all KE loss on an
    eddying state) — it capped the Held-Suarez jet and drained any
    ERA5-initialised/AMIP circulation.  A revert to seconds-scale tau
    silently reintroduces the #1028 dead-jet drain; this pin makes that
    loud.
    """

    def test_sponge_tau_default_is_days_scale(self):
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationConfig)
        cfg = CDGridPrimitiveEquationConfig()
        assert cfg.sponge_tau_sec == 432000.0, (
            f"cd-grid sponge_tau_sec default changed to "
            f"{cfg.sponge_tau_sec!r}; the #1028 decision is 5 d "
            f"(432000 s, FV3 Ray_fast-like). Re-litigate on the issue, "
            f"not silently."
        )
        # Profile shape/extent unchanged: quadratic below sigma=0.15.
        assert cfg.sponge_sigma == 0.15

    def test_sponge_rate_at_top_levels_is_sub_1_per_day(self):
        """With the 5-d tau, the peak per-level rate on a 20-level sigma
        grid is ~0.12/day (was 14.2/day at tau=1 h) — weaker than the
        Held-Suarez surface friction (1/day), as an upper sponge should
        be."""
        import numpy as np
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationConfig)
        from legoesm.grids.vertical import create_sigma_coordinate
        cfg = CDGridPrimitiveEquationConfig()
        sigma = create_sigma_coordinate(20)
        frac = np.clip(
            (cfg.sponge_sigma - np.asarray(sigma.sigma_full, dtype=float))
            / cfg.sponge_sigma, 0.0, 1.0)
        rate_per_day = frac ** 2 / cfg.sponge_tau_sec * 86400.0
        assert 0.0 < rate_per_day.max() < 1.0, (
            f"peak sponge rate {rate_per_day.max():.3f}/day — must be "
            f"positive (sponge active) and below the HS boundary-layer "
            f"friction scale (1/day)")
