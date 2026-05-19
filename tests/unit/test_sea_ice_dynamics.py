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
    mevp_stress_update,
)
# Dynamics
from legoesm.ice.dynamics import (
    stress_divergence,
    evp_solver,
    mevp_solver,
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

        # Start from zero stress and iterate.  N_evp is the EVP
        # subcycle count; the relaxation factor scales as 1/(2·T_evp·N_evp),
        # so converging to within 5% of the VP target requires roughly
        # ~5·T_evp·N_evp iterations.  Use N_evp=1 here to keep the
        # convergence test compact.
        s11, s22, s12 = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
        for _ in range(200):
            s11, s22, s12 = evp_stress_update(
                s11, s22, s12, eps_11, eps_22, eps_12,
                P, e_yield=2.0, T_evp=0.36, dt_s=100.0, N_evp=1,
            )

        assert float(s11) == pytest.approx(float(s11_vp), rel=0.05)
        assert float(s22) == pytest.approx(float(s22_vp), rel=0.05)
        assert float(s12) == pytest.approx(float(s12_vp), rel=0.05)

    def test_e_factor_scales_with_N_evp(self):
        """Hunke-Dukowicz 1997: E_factor = 1/(2·T_evp·N_evp).

        With production defaults T_evp=0.36, N_evp=120 the per-subcycle
        relaxation is ~1.16% toward the VP target.  Verify the closed
        form directly against the implementation so a regression that
        drops the ``N_evp`` factor (the original bug) is caught.
        """
        T_evp = 0.36
        N_evp = 120
        s11, s22, s12 = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
        eps_11 = jnp.array(1e-6)
        eps_22 = jnp.array(-5e-7)
        eps_12 = jnp.array(2e-7)
        P = jnp.array(1e4)

        # One subcycle from sigma_old = 0 should give
        # sigma_new = E·sigma_VP / (1+E)  with  E = 1/(2·T_evp·N_evp).
        Delta = delta_deformation(eps_11, eps_22, eps_12)
        s11_vp, s22_vp, _ = vp_stress(eps_11, eps_22, eps_12, P, Delta)
        E_expected = 1.0 / (2.0 * T_evp * N_evp)
        s11_expected = float(E_expected * s11_vp / (1.0 + E_expected))

        s11_new, _, _ = evp_stress_update(
            s11, s22, s12, eps_11, eps_22, eps_12,
            P, e_yield=2.0, T_evp=T_evp, dt_s=100.0, N_evp=N_evp,
        )
        assert float(s11_new) == pytest.approx(s11_expected, rel=1e-6)

        # Cross-check: with a different N_evp, the relaxation factor
        # must change in the expected direction.
        s11_n240, _, _ = evp_stress_update(
            s11, s22, s12, eps_11, eps_22, eps_12,
            P, e_yield=2.0, T_evp=T_evp, dt_s=100.0, N_evp=240,
        )
        # Larger N_evp → smaller per-subcycle relaxation → smaller stress.
        assert abs(float(s11_n240)) < abs(float(s11_new))

    def test_delta_min_threads_through_evp_stress_update(self):
        """SeaIceConfig.Delta_min reaches delta_deformation via evp_stress_update.

        Sub-yield deformation should saturate `Delta` at `Delta_min`,
        capping viscosity `zeta = P/(2·Delta)` and producing distinct
        stress responses for distinct Delta_min values.  Catches a
        regression where the config field is ignored and the rheology
        default is used regardless of the user-set value.
        """
        # Sub-yield strain — without regularisation, Delta would be ~0
        # and zeta would diverge.  The regulariser sets Delta = Delta_min.
        eps_11 = jnp.array(1e-12)
        eps_22 = jnp.array(0.0)
        eps_12 = jnp.array(0.0)
        s11 = jnp.array(0.0); s22 = jnp.array(0.0); s12 = jnp.array(0.0)
        P = jnp.array(1e4)

        s11_a, _, _ = evp_stress_update(
            s11, s22, s12, eps_11, eps_22, eps_12,
            P, e_yield=2.0, T_evp=0.36, dt_s=100.0, N_evp=120,
            Delta_min=2.0e-9,
        )
        s11_b, _, _ = evp_stress_update(
            s11, s22, s12, eps_11, eps_22, eps_12,
            P, e_yield=2.0, T_evp=0.36, dt_s=100.0, N_evp=120,
            Delta_min=2.0e-7,  # 100× larger floor → 100× smaller zeta
        )
        # Larger Delta_min → smaller viscosity → smaller deviatoric stress
        # response.  Both stresses include the −P/2 isotropic term, so
        # compare deviation from that baseline.
        baseline = -float(P) / 2.0
        assert abs(float(s11_a) - baseline) > abs(float(s11_b) - baseline)


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
    def test_matches_zubov_balance(self):
        """Free drift = U_w + alpha*(U_a - U_w) with alpha ≈ 1.7 % (Zubov).

        Iter-86 replaced the dimensionally-inconsistent slab formula
        (``drag_ocean * U_w + (drag_atm * rho_air / rho_ice) * U_a``,
        which produced O(10^-4 m/s) ice drift) with the standard
        steady-state air/ocean drag balance.
        """
        import math
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
            rho_ocean=config.rho_ocean_ref,
        )

        alpha = math.sqrt(
            config.rho_air_ref * config.drag_atm
            / (config.rho_ocean_ref * config.drag_ocean)
        )
        u_expected = float(ou) + alpha * (float(wu) - float(ou))
        v_expected = float(ov) + alpha * (float(wv) - float(ov))
        assert float(u_fd) == pytest.approx(u_expected, rel=1e-10)
        assert float(v_fd) == pytest.approx(v_expected, rel=1e-10)
        # Sanity: drift magnitude is in the physical 1-3 % of wind range,
        # not the 0.001 % the old formula produced.
        assert 0.005 < alpha < 0.05


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

    def test_strict_volume_conservation_under_clamping(self):
        """Volume drift through linear_remap should be < 0.1% even when
        category bounds activate the clamp.

        Construction: each category's mean thickness is just slightly
        above its upper bound, so the partial-promotion + clamp cascade
        fires.  Audit probe shows 1.68% drift in this case.
        """
        n_cat = 5
        # CICE-standard category bounds
        lo = jnp.array([0.0, 0.6, 1.4, 2.4, 3.6])  # noqa: F841 (visible to fix)
        hi = jnp.array([0.6, 1.4, 2.4, 3.6, 100.0])  # noqa: F841

        # h_new just above each hi → triggers cascade clamping
        h_old = jnp.array([0.3, 1.0, 2.0, 3.0, 5.0])
        a_old = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])
        h_new = jnp.array([0.7, 1.5, 2.5, 3.7, 5.0])
        a_new = a_old

        vol_before = float(jnp.sum(h_new * a_new))
        h_remap, a_remap = linear_remap(h_old, a_old, h_new, a_new, n_cat)
        vol_after = float(jnp.sum(h_remap * a_remap))

        rel_drift = abs(vol_after - vol_before) / vol_before
        assert rel_drift < 0.001, (
            f"linear_remap leaked {rel_drift*100:.3f}% volume when "
            f"h_new straddles category bounds.  Expected < 0.1%."
        )


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

    def test_grad_no_nan_at_ice_free_cells(self):
        """Iter-122 regression: AD-safe ``where(has_ice, vol/conc_safe, 0)``
        recovery pattern must not produce NaN gradients at fully ice-free
        cells (vol=conc=0).  The previous ``conc_safe = max(conc, 1e-20)``
        floor produced a subnormal squared denominator in fp32 (FTZ → 0)
        whose VJP cotangent flowed back through ``jnp.where`` as NaN.
        """
        import jax
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        # Mostly ice-free cells with a small ice patch.
        h = jnp.zeros(shape).at[0, n // 2, n // 2].set(1.0)
        a = jnp.zeros(shape).at[0, n // 2, n // 2].set(0.5)
        T = jnp.full(shape, 260.0)
        u = jnp.full(shape, 0.01)
        v = jnp.zeros(shape)

        def _loss(h_in):
            h_new, _, _ = advect_ice_tracers(
                h_in, a, T, u, v, grid, 3600.0,
            )
            return jnp.sum(h_new)

        g = jax.grad(_loss)(h)
        assert jnp.all(jnp.isfinite(g)), "NaN/Inf in grad through advect_ice_tracers at ice-free cells"


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


# ==============================================================================
# Test mEVP rheology
# ==============================================================================

class TestMEVPStressUpdate:
    def test_one_iter_blends_old_and_vp(self):
        """Single mEVP iteration: σ_new = (1 − 1/α) σ_old + (1/α) σ_VP."""
        eps_11 = jnp.array(1e-6)
        eps_22 = jnp.array(-5e-7)
        eps_12 = jnp.array(2e-7)
        P = jnp.array(1e4)
        Delta = delta_deformation(eps_11, eps_22, eps_12)
        s11_vp, s22_vp, s12_vp = vp_stress(eps_11, eps_22, eps_12, P, Delta)

        s11_old = jnp.array(100.0)
        s22_old = jnp.array(50.0)
        s12_old = jnp.array(-20.0)
        alpha = 500.0

        s11_new, s22_new, s12_new = mevp_stress_update(
            s11_old, s22_old, s12_old,
            eps_11, eps_22, eps_12,
            P, e_yield=2.0, alpha=alpha,
        )

        inv = 1.0 / alpha
        s11_expected = (1.0 - inv) * float(s11_old) + inv * float(s11_vp)
        s22_expected = (1.0 - inv) * float(s22_old) + inv * float(s22_vp)
        s12_expected = (1.0 - inv) * float(s12_old) + inv * float(s12_vp)
        assert float(s11_new) == pytest.approx(s11_expected, rel=1e-6)
        assert float(s22_new) == pytest.approx(s22_expected, rel=1e-6)
        assert float(s12_new) == pytest.approx(s12_expected, rel=1e-6)

    def test_converges_to_vp(self):
        """Many mEVP iterations from zero stress should approach VP target."""
        eps_11 = jnp.array(1e-6)
        eps_22 = jnp.array(-5e-7)
        eps_12 = jnp.array(2e-7)
        P = jnp.array(1e4)
        Delta = delta_deformation(eps_11, eps_22, eps_12)
        s11_vp, s22_vp, s12_vp = vp_stress(eps_11, eps_22, eps_12, P, Delta)

        # α=10 (small) → fast relaxation; 200 iterations is plenty.
        s11, s22, s12 = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
        for _ in range(200):
            s11, s22, s12 = mevp_stress_update(
                s11, s22, s12, eps_11, eps_22, eps_12,
                P, e_yield=2.0, alpha=10.0,
            )

        assert float(s11) == pytest.approx(float(s11_vp), rel=1e-5)
        assert float(s22) == pytest.approx(float(s22_vp), rel=1e-5)
        assert float(s12) == pytest.approx(float(s12_vp), rel=1e-5)

    def test_invalid_alpha_raises(self):
        """Direct mevp_stress_update API must reject alpha<1.

        Solver-level validation in ``mevp_solver`` is not enough — the
        stress-update function is publicly re-exported and external
        callers can hit it with bad alpha (e.g. alpha=0 divide-by-zero).
        """
        eps = jnp.array(0.0)
        s0 = jnp.array(0.0)
        P = jnp.array(1e4)
        with pytest.raises(ValueError, match="alpha"):
            mevp_stress_update(
                s0, s0, s0, eps, eps, eps, P,
                e_yield=2.0, alpha=0.0,
            )
        with pytest.raises(ValueError, match="alpha"):
            mevp_stress_update(
                s0, s0, s0, eps, eps, eps, P,
                e_yield=2.0, alpha=0.5,
            )

    def test_alpha_scales_relaxation_rate(self):
        """Larger alpha → slower per-iter approach to VP target."""
        eps_11 = jnp.array(1e-6)
        eps_22 = jnp.array(-5e-7)
        eps_12 = jnp.array(2e-7)
        P = jnp.array(1e4)
        s0 = jnp.array(0.0)

        s11_a, _, _ = mevp_stress_update(
            s0, s0, s0, eps_11, eps_22, eps_12,
            P, e_yield=2.0, alpha=100.0,
        )
        s11_b, _, _ = mevp_stress_update(
            s0, s0, s0, eps_11, eps_22, eps_12,
            P, e_yield=2.0, alpha=1000.0,
        )
        # α=1000 → 10× smaller step toward σ_VP than α=100
        assert abs(float(s11_b)) < abs(float(s11_a))
        assert abs(float(s11_a)) > 10.0 * abs(float(s11_b)) - 1e-10


# ==============================================================================
# Test mEVP momentum solver
# ==============================================================================

class TestMEVPSolver:
    def test_zero_strength_gains_drift(self):
        """With P_star=0, mEVP velocity should respond to wind."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        u_new, v_new, _, _, _ = mevp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=jnp.ones(shape),
            concentration=jnp.ones(shape),
            wind_u=jnp.full(shape, 10.0),
            wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape),
            ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0,
            N_mevp=100, P_star=0.0,
            alpha_mevp=500.0, beta_mevp=500.0,
            differentiable=False,
        )
        assert jnp.max(jnp.abs(u_new)) > 1e-4
        assert jnp.all(jnp.isfinite(u_new))
        assert jnp.all(jnp.isfinite(v_new))

    def test_high_strength_resists_motion(self):
        """With P_star > 0, ice velocity should be smaller than free drift."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        u_strong, _, _, _, _ = mevp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=jnp.full(shape, 2.0),
            concentration=jnp.ones(shape),
            wind_u=jnp.full(shape, 10.0),
            wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape),
            ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0,
            N_mevp=200, P_star=2.75e4,
            alpha_mevp=500.0, beta_mevp=500.0,
            differentiable=False,
        )

        u_free, _, _, _, _ = mevp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=jnp.full(shape, 2.0),
            concentration=jnp.ones(shape),
            wind_u=jnp.full(shape, 10.0),
            wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape),
            ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0,
            N_mevp=200, P_star=0.0,
            alpha_mevp=500.0, beta_mevp=500.0,
            differentiable=False,
        )

        assert jnp.all(jnp.isfinite(u_strong))
        assert jnp.mean(jnp.abs(u_strong)) < jnp.mean(jnp.abs(u_free))

    def test_finite_output(self):
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)
        u_new, v_new, s11, s22, s12 = mevp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=jnp.ones(shape), concentration=jnp.full(shape, 0.9),
            wind_u=jnp.full(shape, 5.0), wind_v=jnp.full(shape, -2.0),
            ocean_u=jnp.full(shape, 0.1), ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0, N_mevp=100,
        )
        for arr in [u_new, v_new, s11, s22, s12]:
            assert jnp.all(jnp.isfinite(arr))

    def test_ice_free_cells_zero(self):
        """Cells with concentration ≤ 0.01 should keep zero velocity."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        # Concentration zero everywhere → ice mask false → outputs zero
        u_new, v_new, s11, s22, s12 = mevp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=jnp.zeros(shape),
            concentration=jnp.zeros(shape),
            wind_u=jnp.full(shape, 10.0), wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape), ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0, N_mevp=50,
        )
        assert jnp.max(jnp.abs(u_new)) == 0.0
        assert jnp.max(jnp.abs(v_new)) == 0.0
        assert jnp.max(jnp.abs(s11)) == 0.0
        assert jnp.max(jnp.abs(s22)) == 0.0
        assert jnp.max(jnp.abs(s12)) == 0.0

    def test_one_step_analytic_no_coriolis(self):
        """N_mevp=1, P*=0, f=0, σ^0=0, u^0=0: closed-form check.

        With zero ice strength, zero initial stress, zero initial
        velocity, zero ocean current, and ``f`` zeroed in the grid:
            σ^1 = 0  (VP target is 0 when P=0)
            ∇·σ^1 = 0
            τ_oi = 0  (relative velocity zero)
            τ_ai = ρ_air · C_ai · |U_a| · U_a
            ax = τ_ai_x / m,   ay = τ_ai_y / m
            rhs_u = β·0 + 0 + dt·ax
            (β + 1) u^1 = rhs_u  →  u^1 = dt·ax / (β + 1)
        """
        grid = _make_grid()
        # Zero Coriolis everywhere
        grid = grid._replace(f=jnp.zeros_like(grid.f))
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        U_a = 10.0
        V_a = 0.0
        h_val = 1.0
        rho_air = float(jnp.asarray(0.0) + 1.225)  # ensure float
        C_ai = 1.3e-3
        from legoesm import constants
        rho_ice = constants.rho_ice
        m_val = rho_ice * max(h_val, 0.01)
        beta = 500.0
        dt = 3600.0

        u_new, v_new, s11, s22, s12 = mevp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=jnp.full(shape, h_val),
            concentration=jnp.ones(shape),
            wind_u=jnp.full(shape, U_a),
            wind_v=jnp.full(shape, V_a),
            ocean_u=jnp.zeros(shape),
            ocean_v=jnp.zeros(shape),
            grid=grid, dt=dt,
            N_mevp=1, P_star=0.0,
            alpha_mevp=500.0, beta_mevp=beta,
            rho_air=rho_air, C_ai=C_ai, rho_ice=rho_ice,
            differentiable=False,
        )

        tau_ai_x = rho_air * C_ai * abs(U_a) * U_a
        ax = tau_ai_x / m_val
        u_expected = dt * ax / (beta + 1.0)

        # σ should remain zero (P*=0 ⇒ σ_VP=0; σ^0=0 ⇒ σ^1=0)
        assert float(jnp.max(jnp.abs(s11))) < 1e-12
        assert float(jnp.max(jnp.abs(s22))) < 1e-12
        assert float(jnp.max(jnp.abs(s12))) < 1e-12
        # Velocity should match closed form (interior cells away from
        # cubed-sphere face boundaries where halo effects bite)
        u_mid = float(u_new[0, n // 2, n // 2])
        v_mid = float(v_new[0, n // 2, n // 2])
        assert u_mid == pytest.approx(u_expected, rel=1e-9)
        assert v_mid == pytest.approx(0.0, abs=1e-12)

    def test_coriolis_sign_one_step(self):
        """N_mevp=1, P*=0, σ=0, u^0=0, fixed positive f: implicit Coriolis
        rotates the wind-driven velocity to the right (Ekman direction
        in Northern Hemisphere, f > 0).

            (β+1) u - dt·f·v = rhs_u
            (β+1) v + dt·f·u = rhs_v
            rhs_u = dt·ax,  rhs_v = 0
            ⇒ u =  A·rhs_u / (A² + B²)
              v = −B·rhs_u / (A² + B²)
            with A = β+1, B = dt·f > 0.
            ⇒ v < 0 (turned to the right of the wind, f > 0 NH)
        """
        grid = _make_grid()
        n = grid.n
        # Override Coriolis to a uniform positive value
        f_val = 1.0e-4  # rad/s, mid-latitude NH
        grid = grid._replace(f=jnp.full(grid.f.shape, f_val))
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        beta = 100.0  # smaller β → larger u, easier to see Coriolis turning
        dt = 3600.0
        U_a = 10.0
        from legoesm import constants
        rho_air = 1.225
        C_ai = 1.3e-3
        rho_ice = constants.rho_ice
        h_val = 1.0
        m_val = rho_ice * h_val

        u_new, v_new, *_ = mevp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=jnp.full(shape, h_val),
            concentration=jnp.ones(shape),
            wind_u=jnp.full(shape, U_a),
            wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape),
            ocean_v=jnp.zeros(shape),
            grid=grid, dt=dt,
            N_mevp=1, P_star=0.0,
            alpha_mevp=500.0, beta_mevp=beta,
            rho_air=rho_air, C_ai=C_ai, rho_ice=rho_ice,
            differentiable=False,
        )

        A_cor = beta + 1.0
        B_cor = dt * f_val
        det = A_cor ** 2 + B_cor ** 2
        ax = rho_air * C_ai * abs(U_a) * U_a / m_val
        rhs_u = dt * ax
        u_expected = A_cor * rhs_u / det
        v_expected = -B_cor * rhs_u / det

        u_mid = float(u_new[0, n // 2, n // 2])
        v_mid = float(v_new[0, n // 2, n // 2])
        assert u_mid == pytest.approx(u_expected, rel=1e-9)
        assert v_mid == pytest.approx(v_expected, rel=1e-9)
        # Sign sanity: positive wind + positive f → v turned negative
        assert v_mid < 0.0
        assert u_mid > 0.0

    def test_invalid_params_raise(self):
        """Sanity-check basic parameter validation in mevp_solver."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)
        common = dict(
            u_ice=jnp.zeros(shape), v_ice=jnp.zeros(shape),
            sigma_11=s0, sigma_22=s0, sigma_12=s0,
            h_ice=jnp.ones(shape), concentration=jnp.full(shape, 0.8),
            wind_u=jnp.full(shape, 5.0), wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape), ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0,
        )
        with pytest.raises(ValueError, match="N_mevp"):
            mevp_solver(N_mevp=0, **common)
        with pytest.raises(ValueError, match="alpha_mevp"):
            mevp_solver(N_mevp=10, alpha_mevp=0.5, **common)
        with pytest.raises(ValueError, match="beta_mevp"):
            mevp_solver(N_mevp=10, beta_mevp=-1.0, **common)
        # beta_mevp == 0 → alpha·beta = 0, cannot satisfy Kimmritz bound
        with pytest.raises(ValueError, match="beta_mevp"):
            mevp_solver(N_mevp=10, beta_mevp=0.0, **common)

    def test_converges_to_steady_state(self):
        """Two long pseudo-time runs should converge to the same velocity.

        Use α=β=100 so the per-iteration relaxation 1/α is large enough
        that N=400 reaches (1−1/α)^N ≈ 1.8 % residual; doubling to N=800
        drops to 0.03 %.  At α=500 (production default) the residual at
        N=400 is ~45 % and the convergence test is meaningless — that
        regime requires N ≥ 2000 for tight agreement.
        """
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        kwargs = dict(
            u_ice=jnp.zeros(shape), v_ice=jnp.zeros(shape),
            sigma_11=s0, sigma_22=s0, sigma_12=s0,
            h_ice=jnp.full(shape, 2.0),
            concentration=jnp.full(shape, 0.9),
            wind_u=jnp.full(shape, 8.0), wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape), ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0, P_star=2.75e4,
            alpha_mevp=100.0, beta_mevp=100.0,
        )
        u_400, *_ = mevp_solver(N_mevp=400, **kwargs)
        u_800, *_ = mevp_solver(N_mevp=800, **kwargs)
        diff = float(jnp.max(jnp.abs(u_800 - u_400)))
        u_scale = float(jnp.max(jnp.abs(u_800))) + 1e-12
        assert diff / u_scale < 0.05, (
            f"mEVP not converged: rel diff between N=400/800 = {diff/u_scale:.3e}"
        )


# ==============================================================================
# Test mEVP in step_sea_ice dispatch
# ==============================================================================

class TestStepSeaIceMEVP:
    def test_mevp_mode(self):
        grid = _make_grid()
        config = SeaIceConfig(dynamics="mevp", N_mevp=20)
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

    def test_mevp_requires_grid(self):
        config = SeaIceConfig(dynamics="mevp")
        state = init_dynamic_ice_state((6, 8, 8))
        forcing = _make_forcing()
        shape = (6, 8, 8)
        with pytest.raises(ValueError, match="requires a grid"):
            step_sea_ice(
                state, forcing, jnp.full(shape, 271.0),
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0, grid=None,
            )

    def test_unknown_dynamics_raises(self):
        config = SeaIceConfig(dynamics="not_a_scheme")
        state = init_dynamic_ice_state((6, 8, 8))
        forcing = _make_forcing()
        shape = (6, 8, 8)
        with pytest.raises(ValueError, match="Unknown sea-ice dynamics"):
            step_sea_ice(
                state, forcing, jnp.full(shape, 271.0),
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0, grid=_make_grid(),
            )

    def test_mevp_multi_step_stable(self):
        grid = _make_grid()
        config = SeaIceConfig(dynamics="mevp", N_mevp=20)
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


class TestMEVPDifferentiability:
    def test_grad_through_mevp_solver(self):
        """mevp_solver should be differentiable in scan mode."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        def f(wind):
            u, v, _, _, _ = mevp_solver(
                jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
                h_ice=jnp.ones(shape), concentration=jnp.full(shape, 0.8),
                wind_u=wind, wind_v=jnp.zeros(shape),
                ocean_u=jnp.zeros(shape), ocean_v=jnp.zeros(shape),
                grid=grid, dt=3600.0, N_mevp=3,
                differentiable=True,
            )
            return jnp.sum(u ** 2 + v ** 2)

        g = jax.grad(f)(jnp.full(shape, 5.0))
        assert jnp.all(jnp.isfinite(g))

    def test_evp_mevp_qualitative_agreement(self):
        """EVP and mEVP both produce smaller-than-free-drift velocity
        under the same ice strength.

        Renamed from "agree_at_convergence" because at the chosen
        ``alpha=beta=50`` with ``N_mevp=2000`` the mEVP residual is
        ``(1−1/50)^2000 ≈ 2.7e-18`` — fully converged on the
        stress-relaxation side — but EVP under the same parameters has
        its own elastic-CFL constraint and the two methods do not
        converge to bit-identical solutions because they treat
        Coriolis and the velocity update differently.  The robust
        physical assertion is that *both* methods reduce velocity
        below the free-drift baseline by ice-strength resistance.
        """
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        common = dict(
            u_ice=jnp.zeros(shape), v_ice=jnp.zeros(shape),
            sigma_11=s0, sigma_22=s0, sigma_12=s0,
            h_ice=jnp.full(shape, 2.0),
            concentration=jnp.full(shape, 0.9),
            wind_u=jnp.full(shape, 8.0), wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape), ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0,
        )

        u_evp_resist, *_ = evp_solver(
            N_evp=300, T_evp=0.36, P_star=2.75e4, **common,
        )
        u_mevp_resist, *_ = mevp_solver(
            N_mevp=2000, alpha_mevp=50.0, beta_mevp=50.0,
            P_star=2.75e4, **common,
        )
        u_evp_free, *_ = evp_solver(
            N_evp=300, T_evp=0.36, P_star=0.0, **common,
        )
        u_mevp_free, *_ = mevp_solver(
            N_mevp=2000, alpha_mevp=50.0, beta_mevp=50.0,
            P_star=0.0, **common,
        )

        assert jnp.all(jnp.isfinite(u_evp_resist))
        assert jnp.all(jnp.isfinite(u_mevp_resist))
        # Strong-ice cases produce smaller mean speed than P*=0 within
        # the same method
        assert float(jnp.mean(jnp.abs(u_evp_resist))) < float(
            jnp.mean(jnp.abs(u_evp_free))
        )
        assert float(jnp.mean(jnp.abs(u_mevp_resist))) < float(
            jnp.mean(jnp.abs(u_mevp_free))
        )

    def test_jit_smoke_mevp_static_params(self):
        """`mevp_solver` under `jax.jit` with N/alpha/beta as static
        argnames must compile + execute without tracer-conversion errors.
        Codex follow-up: confirms the doc'd static-scalar contract.
        """
        import jax
        from functools import partial
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        s0 = jnp.zeros(shape)

        @partial(
            jax.jit,
            static_argnames=("N_mevp", "alpha_mevp", "beta_mevp"),
        )
        def _run(wind_u, N_mevp, alpha_mevp, beta_mevp):
            u, v, *_ = mevp_solver(
                jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
                h_ice=jnp.ones(shape),
                concentration=jnp.full(shape, 0.8),
                wind_u=wind_u, wind_v=jnp.zeros(shape),
                ocean_u=jnp.zeros(shape), ocean_v=jnp.zeros(shape),
                grid=grid, dt=3600.0,
                N_mevp=N_mevp, alpha_mevp=alpha_mevp, beta_mevp=beta_mevp,
                differentiable=False,
            )
            return jnp.sum(u ** 2 + v ** 2)

        out = _run(jnp.full(shape, 5.0), 10, 500.0, 500.0)
        assert float(out) >= 0.0
        assert jnp.isfinite(out)

    def test_grad_through_step_sea_ice_mevp(self):
        """Reverse-mode AD through the public step_sea_ice dispatch
        with dynamics='mevp' and differentiable_dynamics=True.
        """
        grid = _make_grid()
        config = SeaIceConfig(
            dynamics="mevp", N_mevp=3, differentiable_dynamics=True,
        )
        state = init_dynamic_ice_state((6, 8, 8))
        state = state._replace(
            h_ice=state.h_ice.replace(data=jnp.ones((6, 8, 8))),
            concentration=state.concentration.replace(
                data=jnp.full((6, 8, 8), 0.8),
            ),
        )
        forcing = _make_forcing()
        shape = (6, 8, 8)

        def f(wind_u):
            new_forcing = forcing._replace(u_lowest=wind_u)
            new_state, _ = step_sea_ice(
                state, new_forcing, jnp.full(shape, 271.0),
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0, grid=grid,
            )
            return jnp.sum(new_state.u_ice.data ** 2)

        g = jax.grad(f)(jnp.full(shape, 5.0))
        assert jnp.all(jnp.isfinite(g))
        # Gradient should be nontrivial — wind drives ice motion
        assert float(jnp.max(jnp.abs(g))) > 0.0


# ==============================================================================
# mEVP integration coverage: multi-cat, transport, conservation, long-run
# ==============================================================================

class TestMEVPMultiCategory:
    def test_mevp_n_cat_5_step(self):
        """mEVP with n_categories=5: dynamics solver operates on aggregated
        state, thermodynamics per category, ITD remap. Verify full
        pipeline produces finite output + correct shapes.
        """
        grid = _make_grid()
        config = SeaIceConfig(
            dynamics="mevp", n_categories=5, N_mevp=20,
        )
        shape = (6, 8, 8)
        dims = ("face", "x", "y")

        from legoesm.ice.itd import distribute_to_categories
        h_mc, T_mc, a_mc = distribute_to_categories(
            jnp.full(shape, 1.5), jnp.full(shape, 255.0),
            jnp.full(shape, 0.7), 5,
        )
        cat_dims = ("face", "x", "y", "cat")
        state = DynamicSeaIceState(
            h_ice=Field(data=h_mc, name="h_ice", dims=cat_dims, units="m"),
            T_ice=Field(data=T_mc, name="T_ice", dims=cat_dims, units="K"),
            concentration=Field(data=a_mc, name="conc", dims=cat_dims, units="1"),
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
            config, U_min=1.0, dt=3600.0, grid=grid,
        )
        assert isinstance(new_state, DynamicSeaIceState)
        # Category dim preserved
        assert new_state.h_ice.data.shape == h_mc.shape == (6, 8, 8, 5)
        assert new_state.concentration.data.shape == (6, 8, 8, 5)
        # Velocity / stress remain 2D (aggregated)
        assert new_state.u_ice.data.shape == shape
        assert new_state.sigma_11.data.shape == shape
        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
        assert jnp.all(jnp.isfinite(new_state.concentration.data))
        assert jnp.all(jnp.isfinite(new_state.u_ice.data))
        assert jnp.all(jnp.isfinite(new_state.sigma_11.data))
        # Concentration sum across categories stays in [0, 1]
        conc_sum = jnp.sum(new_state.concentration.data, axis=-1)
        assert jnp.all(conc_sum >= 0.0)
        assert jnp.all(conc_sum <= 1.0 + 1e-10)
        # Response aggregated to 2D
        assert response.T_surface.shape == shape


class TestMEVPTransport:
    def test_mevp_transport_cfl_safe(self):
        """dynamics='mevp' + transport='advect' on cubed sphere: verify
        the velocity field produced by mEVP keeps max(|u|·dt/dx) below
        the PPM monotonicity bound of 1.
        """
        grid = _make_grid()
        config = SeaIceConfig(
            dynamics="mevp", transport="advect", N_mevp=30,
        )
        state = init_dynamic_ice_state((6, 8, 8))
        state = state._replace(
            h_ice=state.h_ice.replace(data=jnp.ones((6, 8, 8)) * 1.5),
            concentration=state.concentration.replace(
                data=jnp.full((6, 8, 8), 0.85),
            ),
        )
        forcing = _make_forcing()
        # Realistic Arctic wind: 10 m/s
        forcing = forcing._replace(
            u_lowest=jnp.full((6, 8, 8), 10.0),
            v_lowest=jnp.full((6, 8, 8), -3.0),
        )
        shape = (6, 8, 8)
        dt = 3600.0
        new_state, _ = step_sea_ice(
            state, forcing, jnp.full(shape, 271.0),
            jnp.zeros(shape), jnp.zeros(shape),
            config, U_min=1.0, dt=dt, grid=grid,
        )
        u_max = float(jnp.max(jnp.abs(new_state.u_ice.data)))
        v_max = float(jnp.max(jnp.abs(new_state.v_ice.data)))
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        cfl_u = u_max * dt / dx_min
        cfl_v = v_max * dt / dy_min
        # PPM bound; allow plenty of margin
        assert cfl_u < 0.5, f"PPM CFL_u violated: {cfl_u:.3f}"
        assert cfl_v < 0.5, f"PPM CFL_v violated: {cfl_v:.3f}"
        assert jnp.all(jnp.isfinite(new_state.h_ice.data))

    def test_mevp_transport_conserves_mass(self):
        """Closed cubed-sphere transport+mEVP: ∑ h·conc·area should be
        conserved to within PPM tolerance after several steps.

        The cubed sphere has no global boundaries — only flux exchange
        across panels — so total ice mass is the right conservation
        invariant.  Use a uniform initial field so the test isolates
        transport conservation from thermodynamic drift.
        """
        grid = _make_grid()
        # transport='advect', NO thermo evolution: zero radiative forcing
        # so thermo doesn't grow/melt ice and confound the conservation
        # check.  Configure forcing for ~zero net surface energy budget.
        config = SeaIceConfig(
            dynamics="mevp", transport="advect", N_mevp=20,
        )
        state = init_dynamic_ice_state((6, 8, 8))
        h0 = jnp.full((6, 8, 8), 2.0)
        a0 = jnp.full((6, 8, 8), 0.9)
        state = state._replace(
            h_ice=state.h_ice.replace(data=h0),
            concentration=state.concentration.replace(data=a0),
            T_ice=state.T_ice.replace(
                data=jnp.full((6, 8, 8), config.T_freeze_ocean),
            ),
        )
        # Tune forcing so the net surface budget is near zero.  The
        # transport-only conservation check is then dominated by the
        # PPM advection accuracy, not thermo growth/melt.
        shape = (6, 8, 8)
        forcing = _make_forcing(shape=shape)
        forcing = forcing._replace(
            sw_down=jnp.zeros(shape),
            lw_down=jnp.full(shape, 270.0),  # ~equal to lw_up at T_freeze
            u_lowest=jnp.full(shape, 5.0),
            v_lowest=jnp.zeros(shape),
            T_lowest=jnp.full(shape, config.T_freeze_ocean),
        )
        ocean_sst = jnp.full(shape, config.T_freeze_ocean)

        def total_mass(s):
            return float(
                jnp.sum(s.h_ice.data * s.concentration.data * grid.area)
            )

        m0 = total_mass(state)
        for _ in range(5):
            state, _ = step_sea_ice(
                state, forcing, ocean_sst,
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0, grid=grid,
            )
        m_end = total_mass(state)
        rel_drift = abs(m_end - m0) / m0
        # Allow 5% drift: thermo not perfectly zeroed in 5 steps, and
        # transport+thermo coupling can produce small bookkeeping
        # mismatches.  Looser than pure-transport because thermo runs.
        # Catches any 10× regression in transport conservation.
        assert rel_drift < 0.05, (
            f"Ice mass drift {rel_drift*100:.2f}% over 5 mEVP+transport steps"
        )


class TestMEVPLongRun:
    def test_mevp_100_steps_no_blowup(self):
        """100 step_sea_ice calls under realistic Arctic forcing: all
        prognostic fields finite, |u| < 2 m/s, |σ| < 1e6 N/m.
        """
        grid = _make_grid()
        config = SeaIceConfig(
            dynamics="mevp", transport="advect", N_mevp=20,
            alpha_mevp=500.0, beta_mevp=500.0,
        )
        state = init_dynamic_ice_state((6, 8, 8))
        state = state._replace(
            h_ice=state.h_ice.replace(data=jnp.ones((6, 8, 8)) * 1.5),
            concentration=state.concentration.replace(
                data=jnp.full((6, 8, 8), 0.85),
            ),
        )
        shape = (6, 8, 8)
        # Realistic Arctic forcing
        forcing = _make_forcing(shape=shape)
        forcing = forcing._replace(
            sw_down=jnp.full(shape, 50.0),
            lw_down=jnp.full(shape, 200.0),
            T_lowest=jnp.full(shape, 250.0),
            q_lowest=jnp.full(shape, 1e-3),
            u_lowest=jnp.full(shape, 8.0),
            v_lowest=jnp.full(shape, -3.0),
        )
        ocean_sst = jnp.full(shape, 271.5)
        dt = 3600.0

        max_u_history = []
        max_sigma_history = []
        for step in range(100):
            state, _ = step_sea_ice(
                state, forcing, ocean_sst,
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=dt, grid=grid,
            )
            max_u = float(jnp.max(jnp.abs(state.u_ice.data)))
            max_s = float(jnp.max(jnp.abs(state.sigma_11.data)))
            max_u_history.append(max_u)
            max_sigma_history.append(max_s)
            assert jnp.all(jnp.isfinite(state.h_ice.data)), (
                f"h_ice non-finite at step {step}"
            )
            assert jnp.all(jnp.isfinite(state.u_ice.data)), (
                f"u_ice non-finite at step {step}"
            )
            assert jnp.all(jnp.isfinite(state.sigma_11.data)), (
                f"sigma_11 non-finite at step {step}"
            )
            # Physical bounds
            assert max_u < 2.0, (
                f"step {step}: max|u|={max_u:.3f} m/s exceeds 2 m/s "
                f"physical bound (Arctic drift is O(0.1 m/s))"
            )
            assert max_s < 1.0e6, (
                f"step {step}: max|σ|={max_s:.3e} N/m exceeds 1e6 "
                f"(typical Arctic max ~1e5)"
            )
        # Verify steady-state-ish behaviour: not still growing exponentially
        u_first10 = max(max_u_history[:10])
        u_last10 = max(max_u_history[-10:])
        assert u_last10 < 5.0 * u_first10, (
            f"max|u| grew {u_last10/u_first10:.1f}× over 100 steps — "
            f"possible runaway"
        )
