"""Tests for sea ice dynamics, rheology, ITD, and transport.

Tests cover:
- Ice strength (Hibler 1979)
- Strain rates on cubed sphere
- Deformation invariant
- VP stress tensor
- EVP stress relaxation
- EVP subcycled momentum solver
- Free-drift velocity
- Multi-category ITD bounds and remapping
- State conversion (slab <-> dynamic)
- Tracer transport
- Full integration through step_sea_ice dispatch
- Backward compatibility with slab model
- JAX differentiability
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere

# Rheology
from legoesm.ice.rheology import (
    ice_strength,
    strain_rates,
    delta_deformation,
    vp_stress,
    evp_stress_update,
)
# Dynamics
from legoesm.ice.dynamics import (
    stress_divergence,
    evp_solver,
    free_drift_velocity,
    air_ice_stress,
    ocean_ice_stress,
)
# ITD
from legoesm.ice.itd import (
    category_bounds,
    upper_bounds,
    aggregate_state,
    distribute_to_categories,
    linear_remap,
)
# Transport
from legoesm.ice.transport import advect_ice_tracers
# Config and state
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import (
    SeaIceState,
    DynamicSeaIceState,
    init_dynamic_ice_state,
    dynamic_to_slab,
    slab_to_dynamic,
)
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.core.field import Field


# ---- Helpers ----

def _make_grid(n=8):
    return create_cubed_sphere(n)


def _make_forcing(shape=(6, 8, 8)):
    """Create minimal AtmToSurface forcing."""
    from legoesm.coupler.coupling_fields import AtmToSurface
    return AtmToSurface(
        sw_down=jnp.full(shape, 100.0),
        lw_down=jnp.full(shape, 200.0),
        precip_total=jnp.zeros(shape),
        precip_snow=jnp.zeros(shape),
        T_lowest=jnp.full(shape, 250.0),
        q_lowest=jnp.full(shape, 1e-3),
        u_lowest=jnp.full(shape, 5.0),
        v_lowest=jnp.full(shape, -3.0),
        p_lowest=jnp.full(shape, 9.5e4),
        p_surface=jnp.full(shape, 1e5),
        rho_lowest=jnp.full(shape, 1.2),
        cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.full(shape, 400.0),
        has_radiation=jnp.ones(shape),
        has_precipitation=jnp.ones(shape),
    )


def _make_slab_state(shape=(6, 8, 8), h=1.0, T=260.0, conc=0.8):
    dims = ("face", "x", "y")
    return SeaIceState(
        h_ice=Field(data=jnp.full(shape, h), name="h_ice", dims=dims, units="m"),
        T_ice=Field(data=jnp.full(shape, T), name="T_ice", dims=dims, units="K"),
        concentration=Field(data=jnp.full(shape, conc), name="conc", dims=dims, units="1"),
    )


# ==============================================================================
# Test Rheology
# ==============================================================================

class TestIceStrength:
    def test_zero_concentration(self):
        """P = 0 when A = 0 (exponential decay)."""
        P = ice_strength(jnp.array(1.0), jnp.array(0.0))
        # exp(-20 * 1) ≈ 2e-9, so P ~ 2.75e4 * 1.0 * 2e-9 ≈ 5.5e-5
        assert float(P) < 1.0

    def test_increases_with_thickness(self):
        P_thin = ice_strength(jnp.array(0.5), jnp.array(1.0))
        P_thick = ice_strength(jnp.array(2.0), jnp.array(1.0))
        assert float(P_thick) > float(P_thin)

    def test_increases_with_concentration(self):
        P_low = ice_strength(jnp.array(1.0), jnp.array(0.5))
        P_high = ice_strength(jnp.array(1.0), jnp.array(1.0))
        assert float(P_high) > float(P_low)

    def test_full_concentration_value(self):
        """At A=1, P = P* * h."""
        P = ice_strength(jnp.array(1.0), jnp.array(1.0))
        assert abs(float(P) - 2.75e4) < 1.0


class TestStrainRates:
    def test_uniform_velocity_zero_strain(self):
        """Uniform translation → near-zero strain rates.

        On cubed sphere, face boundaries introduce O(1/R) errors
        for uniform grid-aligned velocities due to coordinate rotation.
        Interior values should be machine zero.
        """
        grid = _make_grid()
        n = grid.n
        u = jnp.ones((6, n, n)) * 5.0
        v = jnp.ones((6, n, n)) * 3.0
        eps_11, eps_22, eps_12 = strain_rates(u, v, grid)
        # Interior should be near-zero; boundary cells have O(1e-5) from
        # cubed-sphere metric mismatch at face edges
        assert jnp.max(jnp.abs(eps_11)) < 1e-4
        assert jnp.max(jnp.abs(eps_22)) < 1e-4
        assert jnp.max(jnp.abs(eps_12)) < 1e-4

    def test_shapes(self):
        grid = _make_grid()
        n = grid.n
        u = jnp.ones((6, n, n))
        v = jnp.ones((6, n, n))
        eps_11, eps_22, eps_12 = strain_rates(u, v, grid)
        assert eps_11.shape == (6, n, n)
        assert eps_22.shape == (6, n, n)
        assert eps_12.shape == (6, n, n)

    def test_finite(self):
        grid = _make_grid()
        n = grid.n
        key = jax.random.PRNGKey(42)
        u = jax.random.normal(key, (6, n, n)) * 0.1
        v = jax.random.normal(jax.random.PRNGKey(43), (6, n, n)) * 0.1
        eps_11, eps_22, eps_12 = strain_rates(u, v, grid)
        assert jnp.all(jnp.isfinite(eps_11))
        assert jnp.all(jnp.isfinite(eps_22))
        assert jnp.all(jnp.isfinite(eps_12))


class TestDeformation:
    def test_positive(self):
        """Delta >= Delta_min always."""
        eps_11 = jnp.array(0.0)
        eps_22 = jnp.array(0.0)
        eps_12 = jnp.array(0.0)
        D = delta_deformation(eps_11, eps_22, eps_12)
        assert float(D) == pytest.approx(2e-9)

    def test_pure_divergence(self):
        """Pure divergence: Delta = |eps_11 + eps_22|."""
        eps_11 = jnp.array(1e-5)
        eps_22 = jnp.array(1e-5)
        eps_12 = jnp.array(0.0)
        D = delta_deformation(eps_11, eps_22, eps_12)
        assert float(D) == pytest.approx(2e-5, rel=0.01)


class TestVPStress:
    def test_zero_strain_gives_pressure_only(self):
        """With zero strain, sigma = -P/2 * I."""
        P = jnp.array(1e4)
        Delta = jnp.array(2e-9)  # Delta_min
        s11, s22, s12 = vp_stress(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
            P, Delta,
        )
        assert float(s11) == pytest.approx(-5e3, rel=0.01)
        assert float(s22) == pytest.approx(-5e3, rel=0.01)
        assert float(s12) == pytest.approx(0.0, abs=1.0)

    def test_symmetric(self):
        """sigma_12 with symmetric strain."""
        P = jnp.array(1e4)
        eps_12 = jnp.array(1e-6)
        Delta = delta_deformation(jnp.array(0.0), jnp.array(0.0), eps_12)
        _, _, s12 = vp_stress(
            jnp.array(0.0), jnp.array(0.0), eps_12, P, Delta,
        )
        assert jnp.isfinite(s12)
        assert float(s12) != 0.0


class TestEVPStressUpdate:
    def test_relaxes_toward_vp(self):
        """After many EVP subcycles, stress should approach VP solution."""
        eps_11 = jnp.array(1e-6)
        eps_22 = jnp.array(-5e-7)
        eps_12 = jnp.array(2e-7)
        P = jnp.array(1e4)
        Delta = delta_deformation(eps_11, eps_22, eps_12)
        s11_vp, s22_vp, s12_vp = vp_stress(eps_11, eps_22, eps_12, P, Delta)

        # Start from zero stress and iterate
        s11, s22, s12 = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
        for _ in range(200):
            s11, s22, s12 = evp_stress_update(
                s11, s22, s12, eps_11, eps_22, eps_12,
                P, e_yield=2.0, T_evp=0.36, dt_s=100.0,
            )

        assert float(s11) == pytest.approx(float(s11_vp), rel=0.05)
        assert float(s22) == pytest.approx(float(s22_vp), rel=0.05)
        assert float(s12) == pytest.approx(float(s12_vp), rel=0.05)


# ==============================================================================
# Test Dynamics
# ==============================================================================

class TestStressDivergence:
    def test_uniform_stress_zero_force(self):
        """Uniform stress field → zero divergence."""
        grid = _make_grid()
        n = grid.n
        s11 = jnp.ones((6, n, n)) * 1e4
        s22 = jnp.ones((6, n, n)) * 1e4
        s12 = jnp.zeros((6, n, n))
        Fx, Fy = stress_divergence(s11, s22, s12, grid)
        assert jnp.max(jnp.abs(Fx)) < 1.0  # near zero
        assert jnp.max(jnp.abs(Fy)) < 1.0

    def test_shapes(self):
        grid = _make_grid()
        n = grid.n
        s = jnp.zeros((6, n, n))
        Fx, Fy = stress_divergence(s, s, s, grid)
        assert Fx.shape == (6, n, n)
        assert Fy.shape == (6, n, n)


class TestFreeDrift:
    def test_matches_slab(self):
        """Free drift should match the original slab diagnostic velocity."""
        config = SeaIceConfig()
        ou = jnp.array(0.1)
        ov = jnp.array(-0.05)
        wu = jnp.array(5.0)
        wv = jnp.array(-3.0)

        u_fd, v_fd = free_drift_velocity(
            ou, ov, wu, wv,
            drag_ocean=config.drag_ocean,
            drag_atm=config.drag_atm,
            rho_air=config.rho_air_ref,
            rho_ice=config.rho_ice,
        )

        u_slab = (config.drag_ocean * ou
                   + config.drag_atm * (config.rho_air_ref / config.rho_ice) * wu)
        assert float(u_fd) == pytest.approx(float(u_slab), rel=1e-10)


class TestEVPSolver:
    def test_zero_strength_free_drift(self):
        """With P_star=0, EVP should give near free-drift velocity."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        u0 = jnp.zeros(shape)
        v0 = jnp.zeros(shape)
        s0 = jnp.zeros(shape)

        u_new, v_new, _, _, _ = evp_solver(
            u0, v0, s0, s0, s0,
            h_ice=jnp.ones(shape),
            concentration=jnp.ones(shape),
            wind_u=jnp.full(shape, 10.0),
            wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape),
            ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0,
            N_evp=10, P_star=0.0,
            differentiable=False,
        )
        # Should have acquired nonzero velocity from wind forcing
        assert jnp.max(jnp.abs(u_new)) > 1e-4
        assert jnp.all(jnp.isfinite(u_new))
        assert jnp.all(jnp.isfinite(v_new))

    def test_high_strength_resists_motion(self):
        """With higher P_star, ice velocity should be smaller."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        u0 = jnp.zeros(shape)
        v0 = jnp.zeros(shape)
        s0 = jnp.zeros(shape)

        # Moderate strength — enough to resist but not blow up
        u_strong, _, _, _, _ = evp_solver(
            u0, v0, s0, s0, s0,
            h_ice=jnp.full(shape, 2.0),
            concentration=jnp.ones(shape),
            wind_u=jnp.full(shape, 10.0),
            wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape),
            ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0,
            N_evp=30, P_star=2.75e4,
            differentiable=False,
        )

        u_free, _, _, _, _ = evp_solver(
            u0, v0, s0, s0, s0,
            h_ice=jnp.full(shape, 2.0),
            concentration=jnp.ones(shape),
            wind_u=jnp.full(shape, 10.0),
            wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape),
            ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0,
            N_evp=30, P_star=0.0,
            differentiable=False,
        )

        # Velocity with strength should be smaller than free drift
        assert jnp.all(jnp.isfinite(u_strong))
        assert jnp.mean(jnp.abs(u_strong)) < jnp.mean(jnp.abs(u_free))

    def test_finite_output(self):
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)
        u_new, v_new, s11, s22, s12 = evp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=jnp.ones(shape), concentration=jnp.full(shape, 0.9),
            wind_u=jnp.full(shape, 5.0), wind_v=jnp.full(shape, -2.0),
            ocean_u=jnp.full(shape, 0.1), ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0, N_evp=20,
        )
        for arr in [u_new, v_new, s11, s22, s12]:
            assert jnp.all(jnp.isfinite(arr)), f"Non-finite values found"


# ==============================================================================
# Test ITD
# ==============================================================================

class TestCategoryBounds:
    def test_5_categories(self):
        b = category_bounds(5)
        assert b.shape == (5,)
        assert float(b[0]) == 0.0
        assert float(b[-1]) == 3.6

    def test_upper_bounds(self):
        hi = upper_bounds(5)
        assert hi.shape == (5,)
        assert float(hi[0]) == 0.6
        assert float(hi[-1]) == 100.0

    def test_1_category(self):
        b = category_bounds(1)
        assert b.shape == (1,)
        assert float(b[0]) == 0.0

    def test_bounds_increasing(self):
        for nc in [1, 3, 5, 7]:
            b = category_bounds(nc)
            for i in range(len(b) - 1):
                assert float(b[i + 1]) > float(b[i])


class TestAggregation:
    def test_single_category_identity(self):
        # Shape (n_cat=1,) — last axis is categories
        h = jnp.array([1.5])
        T = jnp.array([260.0])
        a = jnp.array([0.8])
        h_agg, T_agg, a_agg = aggregate_state(h, T, a)
        assert float(h_agg.squeeze()) == pytest.approx(1.5)
        assert float(T_agg.squeeze()) == pytest.approx(260.0)
        assert float(a_agg.squeeze()) == pytest.approx(0.8)

    def test_volume_conservation(self):
        """Total volume h*a should be conserved in aggregation."""
        # 3 categories: last axis = n_cat
        h = jnp.array([1.0, 2.0, 3.0])  # (3,) = (n_cat,)
        a = jnp.array([0.2, 0.3, 0.1])
        T = jnp.array([260.0, 260.0, 260.0])
        vol_original = jnp.sum(h * a)
        h_agg, _, a_agg = aggregate_state(h, T, a)
        vol_agg = h_agg * a_agg
        assert float(vol_agg) == pytest.approx(float(vol_original), rel=1e-10)

    def test_no_ice_returns_defaults(self):
        h = jnp.array([0.0, 0.0])
        T = jnp.array([250.0, 260.0])
        a = jnp.array([0.0, 0.0])
        h_agg, T_agg, a_agg = aggregate_state(h, T, a)
        assert float(a_agg) == 0.0
        assert float(h_agg) == 0.0


class TestDistribution:
    def test_puts_ice_in_correct_category(self):
        """Ice of thickness 1.0 should go into category 1 (bounds [0.6, 1.4])."""
        h = jnp.array(1.0)
        T = jnp.array(260.0)
        a = jnp.array(0.8)
        h_mc, T_mc, a_mc = distribute_to_categories(h, T, a, 5)
        # Category 1 (0.6-1.4) should have the ice
        assert float(a_mc[..., 1]) == pytest.approx(0.8)
        assert float(h_mc[..., 1]) == pytest.approx(1.0)
        # Others should be zero
        assert float(a_mc[..., 0]) == pytest.approx(0.0)
        assert float(a_mc[..., 2]) == pytest.approx(0.0)

    def test_thin_ice_in_first_category(self):
        h_mc, _, a_mc = distribute_to_categories(
            jnp.array(0.1), jnp.array(260.0), jnp.array(0.5), 5,
        )
        assert float(a_mc[..., 0]) == pytest.approx(0.5)


class TestLinearRemap:
    def test_no_change_preserves_state(self):
        """If h doesn't change, remap should preserve state."""
        n_cat = 5
        h = jnp.array([0.3, 1.0, 2.0, 3.0, 5.0])
        a = jnp.array([0.1, 0.2, 0.15, 0.1, 0.05])
        h_remap, a_remap = linear_remap(h, a, h, a, n_cat)
        for k in range(n_cat):
            assert float(h_remap[k]) == pytest.approx(float(h[k]), rel=1e-8)
            assert float(a_remap[k]) == pytest.approx(float(a[k]), rel=1e-8)

    def test_volume_conserved(self):
        """Total volume should be preserved through remapping."""
        n_cat = 5
        h_old = jnp.array([0.3, 1.0, 2.0, 3.0, 5.0])
        a_old = jnp.array([0.1, 0.2, 0.15, 0.1, 0.05])
        # Simulate growth: all categories get thicker
        h_new = h_old + 0.3
        a_new = a_old * 1.05
        vol_before = jnp.sum(h_new * a_new)
        h_remap, a_remap = linear_remap(h_old, a_old, h_new, a_new, n_cat)
        vol_after = jnp.sum(h_remap * a_remap)
        assert float(vol_after) == pytest.approx(float(vol_before), rel=0.01)

    def test_concentration_bounds(self):
        n_cat = 5
        h = jnp.array([0.3, 1.0, 2.0, 3.0, 5.0])
        a = jnp.array([0.1, 0.2, 0.15, 0.1, 0.05])
        h_new = h + 0.5
        a_new = a
        h_remap, a_remap = linear_remap(h, a, h_new, a_new, n_cat)
        assert jnp.all(a_remap >= 0.0)
        assert jnp.all(a_remap <= 1.0)


# ==============================================================================
# Test State Conversion
# ==============================================================================

class TestStateConversion:
    def test_slab_to_dynamic_roundtrip(self):
        slab = _make_slab_state()
        dyn = slab_to_dynamic(slab)
        assert isinstance(dyn, DynamicSeaIceState)
        # Velocity should be zero
        assert jnp.all(dyn.u_ice.data == 0.0)

        slab2 = dynamic_to_slab(dyn)
        assert isinstance(slab2, SeaIceState)
        assert jnp.allclose(slab.h_ice.data, slab2.h_ice.data)
        assert jnp.allclose(slab.T_ice.data, slab2.T_ice.data)

    def test_init_dynamic_state(self):
        state = init_dynamic_ice_state((6, 8, 8))
        assert isinstance(state, DynamicSeaIceState)
        assert state.h_ice.data.shape == (6, 8, 8)
        assert jnp.all(state.u_ice.data == 0.0)


# ==============================================================================
# Test Transport
# ==============================================================================

class TestTransport:
    def test_shapes(self):
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        h = jnp.ones(shape)
        a = jnp.full(shape, 0.8)
        T = jnp.full(shape, 260.0)
        u = jnp.full(shape, 0.01)
        v = jnp.zeros(shape)
        h_new, a_new, T_new = advect_ice_tracers(h, a, T, u, v, grid, 3600.0)
        assert h_new.shape == shape
        assert a_new.shape == shape
        assert T_new.shape == shape

    def test_finite(self):
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        h = jnp.ones(shape) * 1.5
        a = jnp.full(shape, 0.8)
        T = jnp.full(shape, 260.0)
        u = jnp.full(shape, 0.05)
        v = jnp.full(shape, -0.02)
        h_new, a_new, T_new = advect_ice_tracers(h, a, T, u, v, grid, 3600.0)
        assert jnp.all(jnp.isfinite(h_new))
        assert jnp.all(jnp.isfinite(a_new))
        assert jnp.all(jnp.isfinite(T_new))

    def test_nonneg_concentration(self):
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        h = jnp.ones(shape) * 0.5
        a = jnp.full(shape, 0.3)
        T = jnp.full(shape, 260.0)
        u = jnp.full(shape, 0.1)
        v = jnp.full(shape, 0.1)
        _, a_new, _ = advect_ice_tracers(h, a, T, u, v, grid, 3600.0)
        assert jnp.all(a_new >= 0.0)
        assert jnp.all(a_new <= 1.0)


# ==============================================================================
# Test Full Integration
# ==============================================================================

class TestStepSeaIceBackwardCompat:
    def test_slab_mode(self):
        """dynamics='none', n_categories=1 uses slab model."""
        config = SeaIceConfig()
        state = _make_slab_state()
        forcing = _make_forcing()
        shape = (6, 8, 8)
        new_state, response = step_sea_ice(
            state, forcing, jnp.full(shape, 273.0),
            jnp.zeros(shape), jnp.zeros(shape),
            config, U_min=1.0, dt=3600.0,
        )
        assert isinstance(new_state, SeaIceState)
        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
        assert jnp.all(jnp.isfinite(response.T_surface))

    def test_slab_accepts_dynamic_state(self):
        """Slab mode should accept DynamicSeaIceState and convert."""
        config = SeaIceConfig()
        slab_state = _make_slab_state()
        dyn_state = slab_to_dynamic(slab_state)
        forcing = _make_forcing()
        shape = (6, 8, 8)
        new_state, response = step_sea_ice(
            dyn_state, forcing, jnp.full(shape, 273.0),
            jnp.zeros(shape), jnp.zeros(shape),
            config, U_min=1.0, dt=3600.0,
        )
        assert isinstance(new_state, SeaIceState)


class TestStepSeaIceDynamic:
    def test_free_drift_mode(self):
        config = SeaIceConfig(dynamics="free_drift")
        state = init_dynamic_ice_state((6, 8, 8))
        # Put some ice
        state = state._replace(
            h_ice=state.h_ice.replace(data=jnp.ones((6, 8, 8))),
            concentration=state.concentration.replace(data=jnp.full((6, 8, 8), 0.8)),
        )
        forcing = _make_forcing()
        shape = (6, 8, 8)
        new_state, response = step_sea_ice(
            state, forcing, jnp.full(shape, 273.0),
            jnp.full(shape, 0.1), jnp.zeros(shape),
            config, U_min=1.0, dt=3600.0,
        )
        assert isinstance(new_state, DynamicSeaIceState)
        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
        # Should have nonzero velocity from free drift
        assert jnp.max(jnp.abs(new_state.u_ice.data)) > 0

    def test_evp_mode(self):
        grid = _make_grid()
        config = SeaIceConfig(dynamics="evp", N_evp=5)
        state = init_dynamic_ice_state((6, 8, 8))
        state = state._replace(
            h_ice=state.h_ice.replace(data=jnp.ones((6, 8, 8)) * 1.5),
            concentration=state.concentration.replace(data=jnp.full((6, 8, 8), 0.9)),
        )
        forcing = _make_forcing()
        shape = (6, 8, 8)
        new_state, response = step_sea_ice(
            state, forcing, jnp.full(shape, 271.0),
            jnp.zeros(shape), jnp.zeros(shape),
            config, U_min=1.0, dt=3600.0, grid=grid,
        )
        assert isinstance(new_state, DynamicSeaIceState)
        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
        assert jnp.all(jnp.isfinite(new_state.u_ice.data))
        assert jnp.all(jnp.isfinite(new_state.sigma_11.data))

    def test_evp_with_transport(self):
        grid = _make_grid()
        config = SeaIceConfig(dynamics="evp", transport="advect", N_evp=5)
        state = init_dynamic_ice_state((6, 8, 8))
        state = state._replace(
            h_ice=state.h_ice.replace(data=jnp.ones((6, 8, 8))),
            concentration=state.concentration.replace(data=jnp.full((6, 8, 8), 0.8)),
        )
        forcing = _make_forcing()
        shape = (6, 8, 8)
        new_state, response = step_sea_ice(
            state, forcing, jnp.full(shape, 270.0),
            jnp.zeros(shape), jnp.zeros(shape),
            config, U_min=1.0, dt=3600.0, grid=grid,
        )
        assert isinstance(new_state, DynamicSeaIceState)
        assert jnp.all(jnp.isfinite(new_state.h_ice.data))

    def test_multi_step_stable(self):
        """Multiple steps should remain stable."""
        grid = _make_grid()
        config = SeaIceConfig(dynamics="evp", N_evp=5)
        state = init_dynamic_ice_state((6, 8, 8))
        state = state._replace(
            h_ice=state.h_ice.replace(data=jnp.ones((6, 8, 8))),
            concentration=state.concentration.replace(data=jnp.full((6, 8, 8), 0.8)),
        )
        forcing = _make_forcing()
        shape = (6, 8, 8)
        for _ in range(5):
            state, _ = step_sea_ice(
                state, forcing, jnp.full(shape, 271.0),
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0, grid=grid,
            )
        assert jnp.all(jnp.isfinite(state.h_ice.data))
        assert jnp.all(jnp.isfinite(state.u_ice.data))


class TestMultiCategoryIntegration:
    def test_5_cat_step(self):
        """Multi-category step produces valid output."""
        config = SeaIceConfig(dynamics="free_drift", n_categories=5)
        shape = (6, 8, 8)
        dims = ("face", "x", "y")

        # Create multi-cat state by distributing thickness
        from legoesm.ice.itd import distribute_to_categories
        h_mc, T_mc, a_mc = distribute_to_categories(
            jnp.full(shape, 1.5), jnp.full(shape, 255.0),
            jnp.full(shape, 0.7), 5,
        )
        state = DynamicSeaIceState(
            h_ice=Field(data=h_mc, name="h_ice", dims=("face", "x", "y", "cat"), units="m"),
            T_ice=Field(data=T_mc, name="T_ice", dims=("face", "x", "y", "cat"), units="K"),
            concentration=Field(data=a_mc, name="conc", dims=("face", "x", "y", "cat"), units="1"),
            u_ice=Field(data=jnp.zeros(shape), name="u_ice", dims=dims, units="m/s"),
            v_ice=Field(data=jnp.zeros(shape), name="v_ice", dims=dims, units="m/s"),
            sigma_11=Field(data=jnp.zeros(shape), name="s11", dims=dims, units="N/m"),
            sigma_22=Field(data=jnp.zeros(shape), name="s22", dims=dims, units="N/m"),
            sigma_12=Field(data=jnp.zeros(shape), name="s12", dims=dims, units="N/m"),
        )
        forcing = _make_forcing()
        new_state, response = step_sea_ice(
            state, forcing, jnp.full(shape, 271.0),
            jnp.zeros(shape), jnp.zeros(shape),
            config, U_min=1.0, dt=3600.0,
        )
        assert isinstance(new_state, DynamicSeaIceState)
        assert new_state.h_ice.data.shape == h_mc.shape
        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
        # Response should be 2D (aggregated)
        assert response.T_surface.shape == shape


# ==============================================================================
# Test Differentiability
# ==============================================================================

class TestDifferentiability:
    def test_grad_through_rheology(self):
        """ice_strength is differentiable."""
        def f(h):
            return jnp.sum(ice_strength(h, jnp.array(0.9)))
        g = jax.grad(f)(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert float(g) > 0

    def test_grad_through_strain_rates(self):
        grid = _make_grid()
        n = grid.n

        def f(u):
            eps_11, eps_22, eps_12 = strain_rates(u, jnp.zeros((6, n, n)), grid)
            return jnp.sum(eps_11 ** 2)

        g = jax.grad(f)(jnp.ones((6, n, n)))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_evp(self):
        """Full EVP solver should be differentiable in scan mode."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        def f(wind):
            u, v, _, _, _ = evp_solver(
                jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
                h_ice=jnp.ones(shape), concentration=jnp.full(shape, 0.8),
                wind_u=wind, wind_v=jnp.zeros(shape),
                ocean_u=jnp.zeros(shape), ocean_v=jnp.zeros(shape),
                grid=grid, dt=3600.0, N_evp=3,
                differentiable=True,
            )
            return jnp.sum(u ** 2 + v ** 2)

        g = jax.grad(f)(jnp.full(shape, 5.0))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_slab_step(self):
        """Slab step_sea_ice is differentiable."""
        config = SeaIceConfig()
        state = _make_slab_state()
        forcing = _make_forcing()
        shape = (6, 8, 8)

        def f(T_init):
            s = SeaIceState(
                h_ice=state.h_ice,
                T_ice=state.T_ice.replace(data=T_init),
                concentration=state.concentration,
            )
            new_s, _ = step_sea_ice(
                s, forcing, jnp.full(shape, 273.0),
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0,
            )
            return jnp.sum(new_s.h_ice.data)

        g = jax.grad(f)(jnp.full(shape, 260.0))
        assert jnp.all(jnp.isfinite(g))


# ==============================================================================
# Surface melt bookkeeping
# ==============================================================================

class TestSurfaceMelt:
    """Test that strong positive Q_sfc melts ice instead of silently
    discarding excess enthalpy at the freezing clamp."""

    def test_warm_forcing_melts_ice(self):
        """Strong SW forcing at ocean freezing should reduce h, not increase it."""
        shape = (6, 4, 4)
        config = SeaIceConfig()
        state = _make_slab_state(
            shape=shape, h=1.0, T=config.T_freeze_ocean - 0.1, conc=0.9,
        )
        # Very strong SW, warm air → large positive Q_sfc
        forcing = _make_forcing(shape=shape)
        forcing = forcing._replace(
            sw_down=jnp.full(shape, 500.0),
            lw_down=jnp.full(shape, 350.0),
            T_lowest=jnp.full(shape, 280.0),
            q_lowest=jnp.full(shape, 5e-3),
        )
        ocean_sst = jnp.full(shape, config.T_freeze_ocean + 0.5)

        new_state, _ = step_sea_ice(
            state, forcing, ocean_sst,
            jnp.zeros(shape), jnp.zeros(shape),
            config, U_min=1.0, dt=3600.0,
        )

        # With strong warming and T near freezing, ice should thin (melt)
        assert jnp.all(new_state.h_ice.data < state.h_ice.data), (
            "Strong positive Q_sfc near freezing must melt ice, not thicken it"
        )

    def test_temperature_stays_at_freezing(self):
        """After surface melt, T_ice should not exceed T_freeze_ocean."""
        shape = (6, 4, 4)
        config = SeaIceConfig()
        state = _make_slab_state(
            shape=shape, h=2.0, T=config.T_freeze_ocean - 0.5, conc=1.0,
        )
        forcing = _make_forcing(shape=shape)
        forcing = forcing._replace(
            sw_down=jnp.full(shape, 600.0),
            lw_down=jnp.full(shape, 400.0),
            T_lowest=jnp.full(shape, 290.0),
        )
        ocean_sst = jnp.full(shape, config.T_freeze_ocean)

        new_state, _ = step_sea_ice(
            state, forcing, ocean_sst,
            jnp.zeros(shape), jnp.zeros(shape),
            config, U_min=1.0, dt=3600.0,
        )

        assert jnp.all(new_state.T_ice.data <= config.T_freeze_ocean + 1e-6)
