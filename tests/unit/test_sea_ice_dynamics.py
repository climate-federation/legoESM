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
    _grid_coriolis,
)
from legoesm import constants as _constants
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

# Production dynamics defaults (single source of truth = SeaIceConfig), used
# by the analytic / stability checks so no drag coefficient is hardcoded.
_DYN_H_MIN = SeaIceConfig().h_ice_min
_DYN_C_AI = SeaIceConfig().drag_atm
_DYN_C_OI = SeaIceConfig().drag_ocean


def _make_grid(n=8):
    return create_cubed_sphere(n)


def _make_forcing(shape=(6, 8, 8)):
    """Create minimal AtmToSurface forcing."""
    from legoesm.core.coupling_fields import AtmToSurface
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

    def test_h_ice_min_threads_into_evp_and_mevp(self):
        """The configured ice-thickness floor reaches the dynamics mass floor:
        thin ice with a larger h_ice_min is heavier (m = rho_ice*max(h, floor))
        and so accelerates less under the same wind. Guards the codex finding
        that EVP/mEVP previously ignored a custom SeaIceConfig.h_ice_min."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        z = jnp.zeros(shape)
        # Ice thinner than both floors so max(h, floor) == floor governs the mass;
        # both floors are well above h (heavy, numerically stable) and differ 10x.
        thin = jnp.full(shape, 0.02)
        kw = dict(
            h_ice=thin, concentration=jnp.ones(shape),
            wind_u=jnp.full(shape, 5.0), wind_v=z,
            ocean_u=z, ocean_v=z, grid=grid, dt=600.0,
            P_star=0.0, differentiable=False,
        )
        u_light, _, _, _, _ = evp_solver(z, z, z, z, z, N_evp=10, h_ice_min=0.1, **kw)
        u_heavy, _, _, _, _ = evp_solver(z, z, z, z, z, N_evp=10, h_ice_min=1.0, **kw)
        assert jnp.all(jnp.isfinite(u_light)) and jnp.all(jnp.isfinite(u_heavy))
        # Heavier floor -> more mass -> strictly slower acquired velocity.
        assert jnp.max(jnp.abs(u_heavy)) < jnp.max(jnp.abs(u_light))
        # mEVP accepts and uses the same kwarg.
        u_m_light, _, _, _, _ = mevp_solver(z, z, z, z, z, N_mevp=10, h_ice_min=0.1, **kw)
        u_m_heavy, _, _, _, _ = mevp_solver(z, z, z, z, z, N_mevp=10, h_ice_min=1.0, **kw)
        assert jnp.all(jnp.isfinite(u_m_heavy))
        assert jnp.max(jnp.abs(u_m_heavy)) < jnp.max(jnp.abs(u_m_light))

    def test_shear_deformation_adds_ridging_closing(self):
        """Audit: shear deformation must contribute to the ridging closing rate
        (Rothrock 1975 / CICE), not convergence alone.  The shear term is
        non-negative and additive, so cs_shear>0 gives closing >= the
        convergence-only rate everywhere, strictly greater where there is shear.
        """
        import numpy as np
        from legoesm.ice.sea_ice import _closing_rate_from_velocity
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        rng = np.arange(n, dtype=float)
        u = jnp.asarray(np.broadcast_to(np.sin(rng)[None, None, :], shape).copy())
        v = jnp.asarray(np.broadcast_to(np.cos(rng)[None, :, None], shape).copy())
        conv_only = _closing_rate_from_velocity(u, v, grid, cap=1.0, cs_shear=0.0)
        with_shear = _closing_rate_from_velocity(
            u, v, grid, cap=1.0, cs_shear=0.25, e_yield=2.0)
        assert jnp.all(with_shear >= conv_only - 1e-12)         # shear only adds
        assert float(jnp.max(with_shear - conv_only)) > 1e-9    # strictly more

    def test_sea_surface_tilt_drives_downslope_drift(self):
        """Audit: the optional sea-surface-tilt force ``-g grad(eta)`` drives
        ice DOWN the SSH slope.  With no wind / current / stress, a positive
        ssh_grad_x accelerates the ice toward lower SSH (u < 0); ssh_grad None
        applies no force (u ~ 0).  Both EVP and mEVP paths."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        z = jnp.zeros(shape)
        base = dict(
            h_ice=jnp.full(shape, 1.0), concentration=jnp.ones(shape),
            wind_u=z, wind_v=z, ocean_u=z, ocean_v=z,
            grid=grid, dt=3600.0, P_star=0.0, differentiable=False,
        )
        slope = jnp.full(shape, 1e-5)  # dη/dx > 0 -> force toward -x
        u_none, *_ = evp_solver(z, z, z, z, z, N_evp=30, **base)
        u_tilt, *_ = evp_solver(z, z, z, z, z, N_evp=30, ssh_grad_x=slope, **base)
        assert jnp.all(jnp.isfinite(u_tilt))
        assert float(jnp.mean(jnp.abs(u_none))) < 1e-9   # no forcing -> no drift
        assert float(jnp.mean(u_tilt)) < -1e-4           # down-gradient drift
        u_m, *_ = mevp_solver(z, z, z, z, z, N_mevp=30, ssh_grad_x=slope, **base)
        assert float(jnp.mean(u_m)) < -1e-4

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
# Ocean-drag implicit-solve stability (Hunke & Dukowicz 1997 / CICE stepu)
# ==============================================================================

class TestOceanDragImplicitStability:
    """Guards the semi-implicit ocean-drag fix in the EVP/mEVP velocity
    solve.  With thin ice (small mass m) and strong ocean-current shear,
    the EXPLICIT drag treatment used previously has stability bound
    dt_s < 2 m / vrel; at the production subcycle timestep this is
    violated in marginal-ice-zone cells and the velocity oscillates and
    blows up.  Treating the linear u^{n+1} drag term implicitly is
    unconditionally stable.
    """

    # Thin-ice MIZ regime: h = 0.1 m, conc = 0.5, dt = 3600 s, default
    # N_evp/N_mevp (=> dt_s = 30 s), and a strong ocean current U_w = 5
    # m/s.  Numbers: m = rho_ice*max(0.1, h_ice_min) = 91.7 kg/m^2,
    # vrel ~ rho_ocean*C_oi*5 = 28.2 kg/m^2/s, so 2 m / vrel ~ 6.5 s
    # << dt_s = 30 s: the explicit scheme is firmly unstable here.
    _H = 0.1
    _CONC = 0.5
    _DT = 3600.0
    _U_OCN = 5.0

    def _fields(self, grid):
        n = grid.n
        shape = (6, n, n)
        z = jnp.zeros(shape)
        return dict(
            h_ice=jnp.full(shape, self._H),
            concentration=jnp.full(shape, self._CONC),
            wind_u=jnp.full(shape, 10.0),
            wind_v=z,
            ocean_u=jnp.full(shape, self._U_OCN),
            ocean_v=z,
            grid=grid, dt=self._DT,
        )

    def _explicit_evp_reference(self, grid, fields, N_evp):
        """Reproduce the OLD explicit-ocean-drag EVP velocity update to
        demonstrate it diverges at this dt_s.  Mirrors the pre-fix code:
        ocean stress uses vrel*(u_ocn - u^n) with u^{n+1} nowhere on the
        LHS (drag fully explicit), Coriolis semi-implicit."""
        m_ice = _constants.rho_ice * max(self._H, _DYN_H_MIN)
        f = _grid_coriolis(grid)
        dt_s = self._DT / N_evp
        alpha = 0.5 * f * dt_s
        denom = 1.0 + alpha ** 2
        u = jnp.zeros_like(fields["wind_u"])
        v = jnp.zeros_like(u)
        ocean_u = fields["ocean_u"]
        ocean_v = fields["ocean_v"]
        wind_u = fields["wind_u"]
        C_ai = _DYN_C_AI
        C_oi = _DYN_C_OI
        for _ in range(N_evp):
            # air stress (explicit)
            dua = wind_u - u
            dva = -v
            spa = jnp.sqrt(dua ** 2 + dva ** 2 + 1e-10)
            tau_ax = _constants.rho_air * C_ai * spa * dua
            tau_ay = _constants.rho_air * C_ai * spa * dva
            # ocean stress fully EXPLICIT: vrel*(u_ocn - u^n)
            duo = ocean_u - u
            dvo = ocean_v - v
            spo = jnp.sqrt(duo ** 2 + dvo ** 2 + 1e-10)
            tau_ox = _constants.rho_ocean * C_oi * spo * duo
            tau_oy = _constants.rho_ocean * C_oi * spo * dvo
            ax = (tau_ax + tau_ox) / m_ice
            ay = (tau_ay + tau_oy) / m_ice
            rhs_u = u + dt_s * ax + alpha * v
            rhs_v = v + dt_s * ay - alpha * u
            u = (rhs_u + alpha * rhs_v) / denom
            v = (rhs_v - alpha * rhs_u) / denom
        return u, v

    def test_evp_thin_ice_strong_current_stays_bounded(self):
        """Implicit EVP stays finite and bounded; explicit reference
        blows up at the same thin-ice / dt_s regime."""
        grid = _make_grid()
        fields = self._fields(grid)
        z = jnp.zeros_like(fields["wind_u"])
        u_new, v_new, *_ = evp_solver(
            z, z, z, z, z, P_star=0.0, differentiable=False, **fields,
        )
        assert jnp.all(jnp.isfinite(u_new))
        assert jnp.all(jnp.isfinite(v_new))
        # Ice cannot outrun the faster of wind/ocean forcing by much;
        # a generous physical bound.  The explicit scheme violates this.
        speed = jnp.sqrt(u_new ** 2 + v_new ** 2)
        assert float(jnp.max(speed)) < 12.0

        # The OLD explicit scheme, same inputs, diverges.
        u_exp, v_exp = self._explicit_evp_reference(grid, fields, N_evp=120)
        exp_speed = jnp.sqrt(u_exp ** 2 + v_exp ** 2)
        exp_max = float(jnp.nan_to_num(jnp.max(exp_speed), nan=jnp.inf))
        assert (not jnp.all(jnp.isfinite(u_exp))) or exp_max > 1e3, (
            "explicit reference should diverge in this thin-ice regime; "
            f"got max speed {exp_max}"
        )

    def test_mevp_thin_ice_strong_current_stays_bounded(self):
        """Implicit mEVP stays finite and bounded in the same regime."""
        grid = _make_grid()
        fields = self._fields(grid)
        z = jnp.zeros_like(fields["wind_u"])
        u_new, v_new, *_ = mevp_solver(
            z, z, z, z, z, P_star=0.0, differentiable=False, **fields,
        )
        assert jnp.all(jnp.isfinite(u_new))
        assert jnp.all(jnp.isfinite(v_new))
        speed = jnp.sqrt(u_new ** 2 + v_new ** 2)
        assert float(jnp.max(speed)) < 12.0

    def test_evp_free_drift_matches_analytic_balance(self):
        """At convergence (P*=0, no internal stress) the implicit solve
        reproduces the exact steady free-drift momentum balance
            tau_air + vrel*(u_ocn - u) + m f (k x u) = 0
        component-wise, where the Coriolis sign convention is +f v in the
        u-equation and -f u in the v-equation (unchanged by this fix).

        EVP subcycling relaxes over a total pseudo-time equal to one dt,
        so the steady balance is reached only when dt >> m/vrel (the
        inertial/drag time).  Thin ice (m = 91.7 kg/m^2, m/vrel ~ 30-80 s)
        converges within dt = 3600 s to many e-foldings; thick ice would
        not.  Face-edge cells carry cubed-sphere metric/halo residual
        unrelated to this fix, so the balance is checked on the interior."""
        grid = _make_grid()
        n = grid.n
        shape = (6, n, n)
        z = jnp.zeros(shape)
        wind_u = jnp.full(shape, 8.0)
        wind_v = jnp.full(shape, -3.0)
        ocean_u = jnp.full(shape, 0.2)
        ocean_v = jnp.full(shape, 0.05)
        h_ice = jnp.full(shape, self._H)  # thin -> converges within one dt
        conc = jnp.ones(shape)
        u_new, v_new, *_ = evp_solver(
            z, z, z, z, z,
            h_ice=h_ice, concentration=conc,
            wind_u=wind_u, wind_v=wind_v,
            ocean_u=ocean_u, ocean_v=ocean_v,
            grid=grid, dt=self._DT, N_evp=400, P_star=0.0,
            differentiable=False,
        )
        assert jnp.all(jnp.isfinite(u_new))

        # Reconstruct the steady momentum residual (F = 0 since P* = 0).
        m_ice = _constants.rho_ice * jnp.maximum(h_ice, _DYN_H_MIN)
        f = _grid_coriolis(grid)
        tau_ax, tau_ay = air_ice_stress(
            u_new, v_new, wind_u, wind_v, _constants.rho_air, _DYN_C_AI,
        )
        tau_ox, tau_oy = ocean_ice_stress(
            u_new, v_new, ocean_u, ocean_v, _constants.rho_ocean, _DYN_C_OI,
        )
        res_x = tau_ax + tau_ox + m_ice * f * v_new
        res_y = tau_ay + tau_oy - m_ice * f * u_new
        # Normalise by the wind-stress scale; interior residual -> 0.
        scale = float(jnp.max(jnp.abs(tau_ax)) + jnp.max(jnp.abs(tau_ay)))
        assert scale > 0.0
        # Interior only (strip the 2-cell face border where cubed-sphere
        # metric/halo effects dominate — same convention as the one-step
        # analytic tests that sample the face centre).
        res = (jnp.abs(res_x) + jnp.abs(res_y))[:, 2:-2, 2:-2]
        rel = float(jnp.max(res)) / scale
        assert rel < 1e-3, f"steady free-drift balance residual too large: {rel}"


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
        assert jnp.all(jnp.isfinite(response.T_sfc))

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
        cat_dims_local = ("face", "x", "y", "cat")
        _z_cat = jnp.zeros(h_mc.shape)
        state = DynamicSeaIceState(
            h_ice=Field(data=h_mc, name="h_ice", dims=cat_dims_local, units="m"),
            T_ice=Field(data=T_mc, name="T_ice", dims=cat_dims_local, units="K"),
            concentration=Field(data=a_mc, name="conc", dims=cat_dims_local, units="1"),
            u_ice=Field(data=jnp.zeros(shape), name="u_ice", dims=dims, units="m/s"),
            v_ice=Field(data=jnp.zeros(shape), name="v_ice", dims=dims, units="m/s"),
            sigma_11=Field(data=jnp.zeros(shape), name="s11", dims=dims, units="N/m"),
            sigma_22=Field(data=jnp.zeros(shape), name="s22", dims=dims, units="N/m"),
            sigma_12=Field(data=jnp.zeros(shape), name="s12", dims=dims, units="N/m"),
            h_snow=Field(data=_z_cat, name="h_snow", dims=cat_dims_local, units="m"),
            S_ice=Field(data=_z_cat, name="S_ice", dims=cat_dims_local, units="g/kg"),
            pond_area=Field(data=_z_cat, name="pond_area", dims=cat_dims_local, units="1"),
            pond_depth=Field(data=_z_cat, name="pond_depth", dims=cat_dims_local, units="m"),
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
        assert response.T_sfc.shape == shape


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

        # Strong warming near freezing melts ice → ice VOLUME (h*conc) drops.
        # Under the volume-based V=h*A update (#28) melt retreats floe AREA at
        # ~constant thickness (lateral convention, matching _thermo_v2), so the
        # mean thickness h can stay flat while the conserved volume h*conc falls.
        vol_old = state.h_ice.data * state.concentration.data
        vol_new = new_state.h_ice.data * new_state.concentration.data
        assert jnp.all(vol_new < vol_old), (
            "Strong positive Q_sfc near freezing must melt ice (reduce volume)"
        )

    def test_temperature_stays_at_freezing(self):
        """After surface melt, T_ice should not exceed the surface melt point."""
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

        # The fresh ice/snow TOP surface melts at T_melt_surface (273.15 K),
        # not the saline basal/ocean freezing point T_freeze_ocean (271.35 K);
        # the skin temperature is clamped to the surface melt point.
        assert jnp.all(new_state.T_ice.data <= config.T_melt_surface + 1e-6)


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
            τ_ai = ρ_air · C_ai · |U_a| · U_a
            ax = τ_ai_x / m,   ay = τ_ai_y / m
            rhs_u = β·0 + 0 + dt·ax + drag·U_w   (U_w = 0)
            (β + 1 + drag) u^1 = rhs_u  →  u^1 = dt·ax / (β + 1 + drag)

        Ocean drag is now treated IMPLICITLY (Hunke & Dukowicz 1997):
        the linear u^{n+1} term folds into the diagonal as
        drag = dt·vrel/m with vrel = ρ_oc·C_oi·|U_w − u^0|.  At u^0 = 0,
        U_w = 0 the relative velocity is only the 1e-10 sqrt floor, so
        ``drag`` is tiny but nonzero and MUST appear in the closed form.
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
        # Implicit ocean drag at u^0 = 0, U_w = 0: vrel is only the eps
        # floor, drag = dt·vrel/m folds into the (beta+1) diagonal.
        C_oi = SeaIceConfig().drag_ocean
        vrel = constants.rho_ocean * C_oi * (1e-10) ** 0.5
        drag = dt * vrel / m_val
        u_expected = dt * ax / (beta + 1.0 + drag)

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

        # Implicit ocean drag folds into the diagonal A (drag = dt·vrel/m;
        # vrel is only the eps floor here since u^0 = U_w = 0).
        C_oi = SeaIceConfig().drag_ocean
        vrel = constants.rho_ocean * C_oi * (1e-10) ** 0.5
        drag = dt * vrel / m_val
        A_cor = beta + 1.0 + drag
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
        _z_cat = jnp.zeros(h_mc.shape)
        state = DynamicSeaIceState(
            h_ice=Field(data=h_mc, name="h_ice", dims=cat_dims, units="m"),
            T_ice=Field(data=T_mc, name="T_ice", dims=cat_dims, units="K"),
            concentration=Field(data=a_mc, name="conc", dims=cat_dims, units="1"),
            u_ice=Field(data=jnp.zeros(shape), name="u_ice", dims=dims, units="m/s"),
            v_ice=Field(data=jnp.zeros(shape), name="v_ice", dims=dims, units="m/s"),
            sigma_11=Field(data=jnp.zeros(shape), name="s11", dims=dims, units="N/m"),
            sigma_22=Field(data=jnp.zeros(shape), name="s22", dims=dims, units="N/m"),
            sigma_12=Field(data=jnp.zeros(shape), name="s12", dims=dims, units="N/m"),
            h_snow=Field(data=_z_cat, name="h_snow", dims=cat_dims, units="m"),
            S_ice=Field(data=_z_cat, name="S_ice", dims=cat_dims, units="g/kg"),
            pond_area=Field(data=_z_cat, name="pond_area", dims=cat_dims, units="1"),
            pond_depth=Field(data=_z_cat, name="pond_depth", dims=cat_dims, units="m"),
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
        assert response.T_sfc.shape == shape


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


# ==============================================================================
# F11: legacy dynamic (free_drift) ice->ocean exchange conservation
# ==============================================================================

class TestLegacyDynamicExchangeF11:
    """The legacy free_drift dynamic path (``_step_dynamic`` / ``_build_response``)
    must deliver the ice->ocean freshwater/heat pulse on melt retreat and at
    terminal melt-out — not zero it because the post-step concentration is 0.

    Regression for F11: ``_build_response`` weights the per-ice-area thickness
    budget by ``max(conc_old, conc)`` (the participating area), so a cell that
    melts out (conc_old > 0, conc_new == 0) still reports the melt freshwater.
    """

    def _legacy_state(self, shape=(6, 8, 8), h=0.015, T=272.99, conc=0.4):
        st = init_dynamic_ice_state(shape, n_categories=1)
        return st._replace(
            h_ice=st.h_ice.replace(data=jnp.full(shape, h)),
            T_ice=st.T_ice.replace(data=jnp.full(shape, T)),
            concentration=st.concentration.replace(data=jnp.full(shape, conc)),
        )

    def _warm_forcing(self, shape=(6, 8, 8)):
        from legoesm.core.coupling_fields import AtmToSurface
        # Dry air (q_lowest below saturation over melting ice ~3.8e-3) so the
        # latent flux is sublimation (mass loss), not deposition.  Under the
        # volume-based V=h*A update (#28) deposition adds ice and leaves a
        # sliver behind, defeating the full melt-out premise.
        return AtmToSurface(
            sw_down=jnp.full(shape, 1500.0), lw_down=jnp.full(shape, 420.0),
            precip_total=jnp.zeros(shape), precip_snow=jnp.zeros(shape),
            T_lowest=jnp.full(shape, 305.0), q_lowest=jnp.full(shape, 1e-4),
            u_lowest=jnp.full(shape, 12.0), v_lowest=jnp.full(shape, 2.0),
            p_lowest=jnp.full(shape, 9.5e4), p_surface=jnp.full(shape, 1e5),
            rho_lowest=jnp.full(shape, 1.2), cos_zenith=jnp.full(shape, 0.5),
            co2_ppmv=jnp.full(shape, 400.0),
            has_radiation=jnp.ones(shape), has_precipitation=jnp.ones(shape),
        )

    def test_meltout_retains_freshwater_pulse(self):
        shape = (6, 8, 8)
        config = SeaIceConfig(dynamics="free_drift")  # legacy path, no new physics
        state = self._legacy_state(shape)
        forcing = self._warm_forcing(shape)
        warm_sst = jnp.full(shape, config.T_freeze_ocean + 8.0)
        zeros = jnp.zeros(shape)
        new, resp = step_sea_ice(state, forcing, warm_sst, zeros, zeros,
                                 config, U_min=1.0, dt=3600.0)
        # The cell melts out within the step.
        assert jnp.all(new.concentration.data <= 1e-6), (
            f"premise: expected melt-out, max conc={float(jnp.max(new.concentration.data)):.3e}"
        )
        # Melt freshwater must still be delivered (positive into ocean), NOT
        # zeroed by the post-step conc = 0.
        assert float(jnp.max(resp.freshwater_flux)) > 0.0, (
            "legacy melt-out dropped the freshwater pulse (conc_old not used)"
        )
        assert jnp.all(jnp.isfinite(resp.freshwater_flux))
        assert jnp.all(jnp.isfinite(resp.ocean_heat_extraction))

    def test_partial_retreat_freshwater_positive(self):
        shape = (6, 8, 8)
        config = SeaIceConfig(dynamics="free_drift")
        # Thicker ice: melts (FW > 0) but does not fully ablate in one step.
        state = self._legacy_state(shape, h=0.5, T=272.5, conc=0.7)
        forcing = self._warm_forcing(shape)
        warm_sst = jnp.full(shape, config.T_freeze_ocean + 4.0)
        zeros = jnp.zeros(shape)
        new, resp = step_sea_ice(state, forcing, warm_sst, zeros, zeros,
                                 config, U_min=1.0, dt=3600.0)
        assert jnp.all(new.h_ice.data >= 0.0)
        assert float(jnp.max(resp.freshwater_flux)) > 0.0, "retreat should melt -> FW>0"
        assert jnp.all(jnp.isfinite(resp.freshwater_flux))

    def _cold_forcing(self, shape=(6, 8, 8)):
        from legoesm.core.coupling_fields import AtmToSurface
        return AtmToSurface(
            sw_down=jnp.zeros(shape), lw_down=jnp.full(shape, 150.0),
            precip_total=jnp.zeros(shape), precip_snow=jnp.zeros(shape),
            T_lowest=jnp.full(shape, 235.0), q_lowest=jnp.full(shape, 1e-4),
            u_lowest=jnp.full(shape, 6.0), v_lowest=jnp.full(shape, 2.0),
            p_lowest=jnp.full(shape, 9.5e4), p_surface=jnp.full(shape, 1e5),
            rho_lowest=jnp.full(shape, 1.2), cos_zenith=jnp.full(shape, 0.5),
            co2_ppmv=jnp.full(shape, 400.0),
            has_radiation=jnp.ones(shape), has_precipitation=jnp.ones(shape),
        )

    def test_open_water_formation_extracts_freshwater(self):
        """Pure open-water cell under strong cooling forms new ice (lead
        freeze) -> freshwater EXTRACTED from the ocean (FW < 0), finite."""
        shape = (6, 8, 8)
        config = SeaIceConfig(dynamics="free_drift")
        state = self._legacy_state(shape, h=0.0, T=271.0, conc=0.0)
        forcing = self._cold_forcing(shape)
        sst = jnp.full(shape, config.T_freeze_ocean)  # ocean at freezing
        zeros = jnp.zeros(shape)
        new, resp = step_sea_ice(state, forcing, sst, zeros, zeros,
                                 config, U_min=1.0, dt=3600.0)
        assert jnp.all(new.concentration.data >= 0.0)
        assert float(jnp.min(resp.freshwater_flux)) < 0.0, (
            "open-water lead freeze must extract ocean freshwater (FW < 0)"
        )
        assert jnp.all(jnp.isfinite(resp.freshwater_flux))
        assert jnp.all(jnp.isfinite(resp.ocean_heat_extraction))

    def test_single_cat_response_matches_per_process_budget(self):
        """The legacy single-cat FW/heat response equals the per-process
        per-cell budget from _thermo_single (the F11 override path), not the
        aggregate (h-h_old)/dt — verifies basal growth and lead freeze are
        correctly separated."""
        from legoesm.ice.sea_ice import (
            _thermo_single, _bulk_flux_dispatch, _ocean_exchange_from_diag)
        shape = (6, 8, 8)
        config = SeaIceConfig(dynamics="free_drift")
        # Partial cover + cold: existing-ice basal growth AND lead freeze.
        state = self._legacy_state(shape, h=1.0, T=250.0, conc=0.5)
        forcing = self._cold_forcing(shape)
        sst = jnp.full(shape, config.T_freeze_ocean)
        zeros = jnp.zeros(shape)
        _, resp = step_sea_ice(state, forcing, sst, zeros, zeros,
                               config, U_min=1.0, dt=3600.0)
        # Reconstruct the per-process budget the override path uses.
        h = state.h_ice.data
        T_ice = state.T_ice.data
        conc = state.concentration.data
        _, _, sh, lh = _bulk_flux_dispatch(T_ice, forcing, config, 1.0)
        _, _, _, diag = _thermo_single(
            h, T_ice, conc, forcing, sst, config, 1.0, 3600.0,
            shflx=sh, lhflx=lh, return_diagnostics=True,
        )
        exp_fw, exp_heat, exp_subl = _ocean_exchange_from_diag(diag, conc, config)
        assert jnp.allclose(resp.freshwater_flux, exp_fw, rtol=1e-7, atol=1e-12), (
            "legacy single-cat FW not the per-process budget"
        )
        assert jnp.allclose(resp.ocean_heat_extraction, exp_heat,
                            rtol=1e-7, atol=1e-12), (
            "legacy single-cat heat not the per-process budget"
        )
        # The per-cell sublimation mass is threaded to surface_mass_flux (#28).
        assert jnp.allclose(resp.surface_mass_flux, exp_subl,
                            rtol=1e-7, atol=1e-12), (
            "legacy single-cat surface_mass_flux not the per-cell sublim budget"
        )


# ==============================================================================
# F10: lat-lon spherical-metric terms in strain rate + stress divergence
# ==============================================================================

class TestF10LatLonSphericalMetric:
    """The lat-lon EVP strain rate and stress divergence must include the
    spherical-metric (Christoffel) terms tan(theta)/r, previously omitted.

    Analytic checks use uniform fields so the centered differences vanish and
    ONLY the metric term remains, giving a closed-form expected value.  Interior
    latitude rows are used so the pole-fold halo does not contaminate the
    finite differences.
    """

    def _grid(self, n_lat=16, n_lon=32):
        from legoesm.grids.latlon import create_latlon_grid
        return create_latlon_grid(n_lat=n_lat, n_lon=n_lon)

    def _interior(self, n_lat):
        # rows away from the two pole-adjacent rows (fold-free centered diffs)
        return slice(3, n_lat - 3)

    def test_uniform_zonal_flow_shear_from_metric(self):
        from legoesm.ice.rheology import _strain_rates_latlon
        grid = self._grid()
        n_lat, n_lon = grid.n_lat, grid.n_lon
        u = jnp.full((n_lat, n_lon), 0.2)
        v = jnp.zeros((n_lat, n_lon))
        e11, e22, e12 = _strain_rates_latlon(u, v, grid)
        # Uniform flow: all gradients zero -> only the metric term survives.
        #   eps_12 = 0.5 * u * tan(theta) / r ;  eps_11 = eps_22 = 0.
        expected_e12 = 0.5 * 0.2 * jnp.tan(grid.lat)[:, None] / grid.radius
        ii = self._interior(n_lat)
        assert jnp.allclose(e12[ii], expected_e12[ii], rtol=1e-5, atol=1e-12)
        assert jnp.allclose(e11[ii], 0.0, atol=1e-18)
        assert jnp.allclose(e22[ii], 0.0, atol=1e-18)
        # Non-trivial at mid-latitude (old metric-free code gave exactly 0).
        assert float(jnp.max(jnp.abs(e12[ii]))) > 0.0

    def test_uniform_meridional_flow_normal_from_metric(self):
        from legoesm.ice.rheology import _strain_rates_latlon
        grid = self._grid()
        n_lat, n_lon = grid.n_lat, grid.n_lon
        u = jnp.zeros((n_lat, n_lon))
        v = jnp.full((n_lat, n_lon), 0.2)
        e11, e22, e12 = _strain_rates_latlon(u, v, grid)
        #   eps_11 = - v * tan(theta) / r ;  eps_22 = 0 ;  eps_12 = 0.
        expected_e11 = -0.2 * jnp.tan(grid.lat)[:, None] / grid.radius
        ii = self._interior(n_lat)
        assert jnp.allclose(e11[ii], expected_e11[ii], rtol=1e-5, atol=1e-12)
        assert jnp.allclose(e22[ii], 0.0, atol=1e-18)
        assert jnp.allclose(e12[ii], 0.0, atol=1e-18)
        assert float(jnp.max(jnp.abs(e11[ii]))) > 0.0

    def test_metric_antisymmetric_about_equator(self):
        """tan(theta) is odd in latitude, so the metric strain reverses sign
        across the equator and is smallest near it."""
        from legoesm.ice.rheology import _strain_rates_latlon
        grid = self._grid()
        n_lat, n_lon = grid.n_lat, grid.n_lon
        u = jnp.full((n_lat, n_lon), 0.2)
        v = jnp.zeros((n_lat, n_lon))
        _, _, e12 = _strain_rates_latlon(u, v, grid)
        ii = self._interior(n_lat)
        lat = grid.lat[ii]
        e12_i = e12[ii][:, 0]
        # Northern (lat>0) and southern (lat<0) interior rows have opposite sign.
        north = e12_i[lat > 0.0]
        south = e12_i[lat < 0.0]
        assert float(jnp.min(north)) > 0.0
        assert float(jnp.max(south)) < 0.0

    def test_solid_body_rotation_gives_zero_strain(self):
        """DEFINITIVE physical check: rigid (solid-body) zonal rotation
        u = U0*cos(theta), v = 0 is a pure rotation -> ZERO strain rate.  The
        metric term cancels the du/dy shear EXACTLY, leaving only an
        O(dtheta^2) discretization residual that converges to zero with
        resolution.  The metric-FREE code gave a spurious eps_12 ~ U0*sin/(2R)
        (~1e3x larger) which would drive spurious polar EVP stress / grid-scale
        noise.  This is the test that ``passing norms NECESSARY != SUFFICIENT''
        (CLAUDE.md) demands for a grid-metric change."""
        from legoesm.ice.rheology import _strain_rates_latlon
        U0 = 10.0
        prev = None
        for n_lat, n_lon in [(64, 128), (128, 256)]:
            grid = self._grid(n_lat=n_lat, n_lon=n_lon)
            lat = grid.lat[:, None]
            u = U0 * jnp.cos(lat) * jnp.ones((n_lat, n_lon))
            v = jnp.zeros((n_lat, n_lon))
            e11, e22, e12 = _strain_rates_latlon(u, v, grid)
            ii = slice(4, n_lat - 4)
            # Normal strains vanish exactly (no longitude dependence, v = 0).
            assert jnp.allclose(e11[ii], 0.0, atol=1e-15)
            assert jnp.allclose(e22[ii], 0.0, atol=1e-15)
            # Shear strain is ONLY the O(dtheta^2) residual, >100x below the
            # metric-free spurious value 0.5*U0*sin/R.
            max_e12 = float(jnp.max(jnp.abs(e12[ii])))
            metric_free = float(jnp.max(jnp.abs(
                0.5 * U0 * jnp.sin(lat[ii]) / grid.radius)))
            assert max_e12 < 1e-8, f"solid-body shear not ~0: {max_e12:.2e}"
            assert max_e12 < 0.01 * metric_free, (
                f"metric term did not cancel rigid-rotation shear: "
                f"{max_e12:.2e} vs metric-free {metric_free:.2e}")
            if prev is not None:
                # ~2nd-order: doubling resolution at least halves the residual.
                assert max_e12 < 0.5 * prev
            prev = max_e12

    def test_stress_div_uniform_sigma12_metric(self):
        from legoesm.ice.dynamics import _stress_divergence_latlon
        grid = self._grid()
        n_lat, n_lon = grid.n_lat, grid.n_lon
        s12 = jnp.full((n_lat, n_lon), 1000.0)
        zero = jnp.zeros((n_lat, n_lon))
        Fx, Fy = _stress_divergence_latlon(zero, zero, s12, grid)
        #   Fx = -2 sigma_12 tan(theta)/r ;  Fy = 0.
        expected_Fx = -2.0 * 1000.0 * jnp.tan(grid.lat)[:, None] / grid.radius
        ii = self._interior(n_lat)
        assert jnp.allclose(Fx[ii], expected_Fx[ii], rtol=1e-5, atol=1e-10)
        assert jnp.allclose(Fy[ii], 0.0, atol=1e-12)
        assert float(jnp.max(jnp.abs(Fx[ii]))) > 0.0

    def test_stress_div_uniform_normal_stress_metric(self):
        from legoesm.ice.dynamics import _stress_divergence_latlon
        grid = self._grid()
        n_lat, n_lon = grid.n_lat, grid.n_lon
        s11 = jnp.full((n_lat, n_lon), 1000.0)
        zero = jnp.zeros((n_lat, n_lon))
        Fx, Fy = _stress_divergence_latlon(s11, zero, zero, grid)
        #   Fy = (sigma_11 - sigma_22) tan(theta)/r = sigma_11 tan(theta)/r ; Fx=0.
        expected_Fy = 1000.0 * jnp.tan(grid.lat)[:, None] / grid.radius
        ii = self._interior(n_lat)
        assert jnp.allclose(Fy[ii], expected_Fy[ii], rtol=1e-5, atol=1e-10)
        assert jnp.allclose(Fx[ii], 0.0, atol=1e-12)

    def test_strain_rates_differentiable(self):
        import jax
        from legoesm.ice.rheology import _strain_rates_latlon
        grid = self._grid()
        n_lat, n_lon = grid.n_lat, grid.n_lon

        def loss(scale):
            u = jnp.full((n_lat, n_lon), 0.1) * scale
            v = jnp.full((n_lat, n_lon), 0.05)
            e11, e22, e12 = _strain_rates_latlon(u, v, grid)
            return jnp.sum(e11 ** 2 + e22 ** 2 + e12 ** 2)

        g = jax.grad(loss)(1.0)
        assert jnp.isfinite(g)


class TestF10LatLonMetricNoClip:
    """The spherical metric must use the EXACT tan(theta) on every cell-center
    row (no clip) and the pole-reaching grid must be guarded (Codex F10)."""

    def test_metric_exact_at_high_latitude(self):
        """On a 1-degree grid (centers at +/-89.5deg, tan>100) the metric term
        equals the exact tan(theta)/r, not a capped value."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ice.rheology import _strain_rates_latlon
        grid = create_latlon_grid(n_lat=180, n_lon=360)
        n_lat = grid.n_lat
        u = jnp.zeros((n_lat, 360))
        v = jnp.full((n_lat, 360), 0.1)
        # Uniform u=0 -> du_dx=0 -> eps_11 = -v*tan(theta)/r exactly (the metric
        # term is algebraic, no finite difference, so pole-fold is irrelevant).
        e11, _, _ = _strain_rates_latlon(u, v, grid)
        hi = int(jnp.argmax(jnp.abs(grid.lat)))   # most poleward row
        assert abs(float(jnp.tan(grid.lat[hi]))) > 100.0, (
            "premise: poleward tan(theta) exceeds the old 1e2 clip"
        )
        expected = -0.1 * jnp.tan(grid.lat) / grid.radius
        assert float(e11[hi, 0]) == pytest.approx(float(expected[hi]), rel=1e-5)

    def test_strain_and_stress_use_same_metric(self):
        """Work-conjugacy: strain rate and stress divergence use the identical
        (uncapped) metric coefficient."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ice.rheology import _strain_rates_latlon
        from legoesm.ice.dynamics import _stress_divergence_latlon
        grid = create_latlon_grid(n_lat=180, n_lon=360)
        n_lat = grid.n_lat
        # eps_11 metric coefficient from uniform meridional flow: -v*tan/r.
        _, _, _ = _strain_rates_latlon(
            jnp.zeros((n_lat, 360)), jnp.full((n_lat, 360), 1.0), grid)
        # stress Fx metric from uniform sigma_12: -2*sigma_12*tan/r.
        Fx, _ = _stress_divergence_latlon(
            jnp.zeros((n_lat, 360)), jnp.zeros((n_lat, 360)),
            jnp.full((n_lat, 360), 1.0), grid)
        hi = int(jnp.argmax(jnp.abs(grid.lat)))
        metric_hi = float(jnp.tan(grid.lat[hi]) / grid.radius)
        # Both derive from the same metric_hi; stress Fx = -2*1*metric_hi.
        assert float(Fx[hi, 0]) == pytest.approx(-2.0 * metric_hi, rel=1e-5)

    def test_pole_reaching_latlon_evp_raises(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ice.state import init_dynamic_ice_state
        grid = create_latlon_grid(n_lat=16, n_lon=32)
        # Force the most-northern row exactly onto the pole (cos -> 0).
        bad_grid = grid._replace(lat=grid.lat.at[-1].set(jnp.pi / 2))
        config = SeaIceConfig(dynamics="evp")
        state = init_dynamic_ice_state((16, 32))
        forcing = _make_forcing(shape=(16, 32))
        sst = jnp.full((16, 32), config.T_freeze_ocean)
        z = jnp.zeros((16, 32))
        with pytest.raises(ValueError, match="spherical metric"):
            step_sea_ice(state, forcing, sst, z, z, config,
                         U_min=1.0, dt=3600.0, grid=bad_grid)

    def test_metric_finite_at_exact_pole(self):
        """A degenerate grid with a row exactly at the pole must not produce
        inf/NaN strain (the |cosθ| floor keeps the metric finite); real-grid
        rows are unaffected (exact tanθ)."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ice.rheology import _strain_rates_latlon
        grid = create_latlon_grid(n_lat=16, n_lon=32)
        bad = grid._replace(lat=grid.lat.at[-1].set(jnp.pi / 2))
        u = jnp.full((16, 32), 0.1)
        v = jnp.full((16, 32), 0.1)
        e11, e22, e12 = _strain_rates_latlon(u, v, bad)
        assert jnp.all(jnp.isfinite(e11))
        assert jnp.all(jnp.isfinite(e22))
        assert jnp.all(jnp.isfinite(e12))

    def test_pole_avoiding_latlon_evp_does_not_raise(self):
        """A normal cell-centered lat-lon grid (rows inside the poles) is
        accepted under EVP."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ice.state import init_dynamic_ice_state
        grid = create_latlon_grid(n_lat=16, n_lon=32)
        config = SeaIceConfig(dynamics="evp")
        state = init_dynamic_ice_state((16, 32))
        forcing = _make_forcing(shape=(16, 32))
        sst = jnp.full((16, 32), config.T_freeze_ocean)
        z = jnp.zeros((16, 32))
        new, resp = step_sea_ice(state, forcing, sst, z, z, config,
                                 U_min=1.0, dt=3600.0, grid=grid)
        assert jnp.all(jnp.isfinite(new.h_ice.data))


class TestF10LatLonEVPJit:
    """The F10 pole guard must not break jax.jit on the lat-lon EVP step
    (Codex: the guard must not concretize grid.lat during tracing)."""

    def test_latlon_evp_step_jittable_grid_closed_over(self):
        import jax
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ice.state import init_dynamic_ice_state
        grid = create_latlon_grid(n_lat=16, n_lon=32)
        config = SeaIceConfig(dynamics="evp")
        forcing = _make_forcing(shape=(16, 32))
        sst = jnp.full((16, 32), config.T_freeze_ocean)
        z = jnp.zeros((16, 32))

        @jax.jit
        def step(state, u, v):
            return step_sea_ice(state, forcing, sst, u, v, config,
                                U_min=1.0, dt=3600.0, grid=grid)

        state = init_dynamic_ice_state((16, 32))
        new, resp = step(state, z, z)
        assert jnp.all(jnp.isfinite(new.h_ice.data))
        assert jnp.all(jnp.isfinite(resp.freshwater_flux))

    def test_latlon_mevp_step_jittable(self):
        import jax
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ice.state import init_dynamic_ice_state
        grid = create_latlon_grid(n_lat=16, n_lon=32)
        config = SeaIceConfig(dynamics="mevp")
        forcing = _make_forcing(shape=(16, 32))
        sst = jnp.full((16, 32), config.T_freeze_ocean)
        z = jnp.zeros((16, 32))

        @jax.jit
        def step(state, u, v):
            return step_sea_ice(state, forcing, sst, u, v, config,
                                U_min=1.0, dt=3600.0, grid=grid)

        state = init_dynamic_ice_state((16, 32))
        new, _ = step(state, z, z)
        assert jnp.all(jnp.isfinite(new.h_ice.data))


class TestMulticatTracerRemapGuard:
    """Multi-category + brine/snow/pond tracers must require the tracer-aware
    Lipscomb (2001) ITD remap; the 'simple' linear remap drops the tracers
    across category transfers, breaking salt/snow/pond conservation (Codex)."""

    def test_multicat_brine_simple_remap_raises(self):
        from legoesm.ice.config import SeaIceConfig, BrineConfig
        config = SeaIceConfig(
            n_categories=2, itd_remap="simple",
            brine=BrineConfig(enabled=True),
        )
        state = _make_slab_state()   # guard is config-only, fires before dispatch
        forcing = _make_forcing()
        sst = jnp.full((6, 8, 8), config.T_freeze_ocean)
        z = jnp.zeros((6, 8, 8))
        with pytest.raises(ValueError, match="lipscomb2001"):
            step_sea_ice(state, forcing, sst, z, z, config, U_min=1.0, dt=3600.0)

    def test_multicat_no_tracers_simple_remap_ok(self):
        """Multi-category WITHOUT tracers may use the simple remap (it carries
        h/conc/T, which is all that exists) — the guard must NOT fire."""
        from legoesm.ice.config import SeaIceConfig
        config = SeaIceConfig(n_categories=2, itd_remap="simple")
        state = _make_slab_state()
        forcing = _make_forcing()
        sst = jnp.full((6, 8, 8), config.T_freeze_ocean)
        z = jnp.zeros((6, 8, 8))
        # Must not raise the itd_remap guard (a slab state on the multicat
        # path raises a later state-type error, but NOT the lipscomb2001
        # message — i.e. execution got PAST the guard).
        try:
            step_sea_ice(state, forcing, sst, z, z, config, U_min=1.0, dt=3600.0)
        except Exception as e:  # noqa: BLE001 - asserting the guard did NOT fire
            assert "lipscomb2001" not in str(e)

    def test_multicat_brine_typo_remap_rejected(self):
        """A typo'd itd_remap (!= 'simple' and != 'lipscomb2001') must NOT slip
        past the guard into the tracer-dropping linear-remap fallback — the
        scheme is validated up front (Codex dispatch-bypass finding)."""
        from legoesm.ice.config import SeaIceConfig, BrineConfig
        config = SeaIceConfig(
            n_categories=2, itd_remap="lipcomb2001",   # typo
            brine=BrineConfig(enabled=True),
        )
        state = _make_slab_state()
        forcing = _make_forcing()
        sst = jnp.full((6, 8, 8), config.T_freeze_ocean)
        z = jnp.zeros((6, 8, 8))
        with pytest.raises(ValueError, match="itd_remap"):
            step_sea_ice(state, forcing, sst, z, z, config, U_min=1.0, dt=3600.0)


class TestMulticatTransportTracerConservation:
    """Multi-category transport must advect snow/salt/pond as conserved
    INVENTORIES (V_snow=h_snow*a, salt=S*h*a, V_pond=area*depth*a), not the
    intensive fields on their own (Codex transport tracer-conservation)."""

    def _state(self, n=8, n_cat=2):
        from legoesm.ice.state import init_dynamic_ice_state
        shape = (6, n, n, n_cat)
        st = init_dynamic_ice_state(shape, n_categories=n_cat)
        # Nonuniform per-cat ice + tracers so transport actually moves them.
        h = jnp.zeros(shape).at[..., 0].set(0.6).at[..., 1].set(2.0)
        # Concentrate ice in one hemisphere so advection has a gradient.
        face0 = jnp.zeros((6, n, n)).at[0].set(0.4).at[1].set(0.2)
        conc = jnp.stack([face0, face0 * 0.5], axis=-1)
        # Nonuniform salinity so transport moves real salinity gradients
        # (a constant field could not expose a tracer-conservation regression).
        S = jnp.full(shape, 5.0).at[0, ..., 0].set(8.0).at[1, ..., 0].set(2.0)
        pa = jnp.zeros(shape).at[..., 0].set(0.2)
        pd = jnp.zeros(shape).at[..., 0].set(0.05)
        return st._replace(
            h_ice=st.h_ice.replace(data=h),
            concentration=st.concentration.replace(data=conc),
            T_ice=st.T_ice.replace(data=jnp.full(shape, 263.0)),
            S_ice=st.S_ice.replace(data=S),
            pond_area=st.pond_area.replace(data=pa),
            pond_depth=st.pond_depth.replace(data=pd),
        )

    def test_multicat_transport_brine_ponds_runs_finite(self):
        from legoesm.ice.config import SeaIceConfig, BrineConfig, MeltPondConfig
        n = 8
        grid = create_cubed_sphere(n)
        config = SeaIceConfig(
            n_categories=2, dynamics="free_drift", transport="advect",
            itd_remap="lipscomb2001",
            brine=BrineConfig(enabled=True), ponds=MeltPondConfig(enabled=True),
        )
        state = self._state(n=n)
        forcing = _make_forcing(shape=(6, n, n))   # default winds -> advection
        sst = jnp.full((6, n, n), config.T_freeze_ocean)
        z = jnp.zeros((6, n, n))
        new, resp = step_sea_ice(state, forcing, sst, z, z, config,
                                 U_min=1.0, dt=3600.0, grid=grid)
        # The new inventory-based transport + reconstruction must stay finite
        # and physical (no NaN/inf from the S=salt/h or pond area/depth recovery).
        assert jnp.all(jnp.isfinite(new.S_ice.data))
        assert jnp.all(jnp.isfinite(new.pond_area.data))
        assert jnp.all(jnp.isfinite(new.pond_depth.data))
        assert jnp.all(new.S_ice.data >= 0.0)
        assert jnp.all(new.pond_area.data >= 0.0)
        assert jnp.all(new.pond_depth.data >= 0.0)
        assert jnp.all(jnp.isfinite(resp.salt_flux))
        assert jnp.all(jnp.isfinite(resp.freshwater_flux))
        # Bulk salinity stays within the configured physical bound.
        assert jnp.all(new.S_ice.data <= config.brine.S_ice_max + 1e-6)

    def test_multicat_transport_no_concentration_overfill(self):
        """Multi-category transport must keep sum_k(conc_k) <= 1: advecting each
        category area independently can transiently overfill a cell; the
        post-transport renormalisation (uniform thickening) restores the
        aggregate-area invariant so thermo/brine/ridging never act on > 1 cell
        area (Codex aggregate-area finding)."""
        from legoesm.ice.config import SeaIceConfig, BrineConfig
        from legoesm.ice.state import init_dynamic_ice_state
        n = 8
        grid = create_cubed_sphere(n)
        shape = (6, n, n, 2)
        st = init_dynamic_ice_state(shape, n_categories=2)
        # Two complementary categories whose aggregate concentration is ~1 with
        # a spatial gradient (so independent advection would overfill).
        c0 = jnp.zeros((6, n, n)).at[0].set(0.6).at[1].set(0.4)
        conc = jnp.stack([c0, 1.0 - c0], axis=-1)  # sums to 1 where ice exists
        # restrict ice to faces 0,1 so there is a sharp edge to advect across
        mask = jnp.zeros((6, n, n)).at[0].set(1.0).at[1].set(1.0)[..., None]
        conc = conc * mask
        h = jnp.zeros(shape).at[..., 0].set(0.5).at[..., 1].set(2.0)
        state = st._replace(
            h_ice=st.h_ice.replace(data=h),
            concentration=st.concentration.replace(data=conc),
            T_ice=st.T_ice.replace(data=jnp.full(shape, 260.0)),
        )
        config = SeaIceConfig(
            n_categories=2, dynamics="free_drift", transport="advect",
            itd_remap="lipscomb2001", brine=BrineConfig(enabled=True),
        )
        forcing = _make_forcing(shape=(6, n, n))   # default winds -> advection
        sst = jnp.full((6, n, n), config.T_freeze_ocean)
        z = jnp.zeros((6, n, n))
        new, _ = step_sea_ice(state, forcing, sst, z, z, config,
                              U_min=1.0, dt=3600.0, grid=grid)
        sum_conc = jnp.sum(new.concentration.data, axis=-1)
        assert float(jnp.max(sum_conc)) <= 1.0 + 1e-6, (
            f"multicat transport overfilled the cell: max sum_k conc_k="
            f"{float(jnp.max(sum_conc)):.6f}"
        )
        assert jnp.all(jnp.isfinite(new.h_ice.data))

    def test_overfilled_restart_capped_without_transport(self):
        """An already-overfilled multicat state (e.g. a restart) must be
        capped to sum_k(conc_k) <= 1 even with transport='none' — the cap is
        unconditional, not gated on advection (Codex)."""
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import init_dynamic_ice_state
        n = 8
        grid = create_cubed_sphere(n)
        shape = (6, n, n, 2)
        st = init_dynamic_ice_state(shape, n_categories=2)
        # Per-category concentrations summing to 1.3 (overfilled restart).
        conc = (jnp.zeros(shape).at[..., 0].set(0.7).at[..., 1].set(0.6))
        h = jnp.zeros(shape).at[..., 0].set(0.5).at[..., 1].set(1.5)
        state = st._replace(
            h_ice=st.h_ice.replace(data=h),
            concentration=st.concentration.replace(data=conc),
            T_ice=st.T_ice.replace(data=jnp.full(shape, 260.0)),
        )
        forcing = _make_forcing(shape=(6, n, n))
        sst = jnp.full((6, n, n), SeaIceConfig().T_freeze_ocean)
        z = jnp.zeros((6, n, n))
        # The cap must fire ahead of the dynamics (EVP/mEVP rheology is
        # exponential in concentration) AND in the returned state, for every
        # dynamics scheme.
        for dyn in ("free_drift", "evp", "mevp"):
            config = SeaIceConfig(n_categories=2, dynamics=dyn,
                                  transport="none")
            new, resp = step_sea_ice(state, forcing, sst, z, z, config,
                                     U_min=1.0, dt=3600.0, grid=grid)
            sum_conc = jnp.sum(new.concentration.data, axis=-1)
            assert float(jnp.max(sum_conc)) <= 1.0 + 1e-6, (
                f"[{dyn}] overfilled restart not capped: max sum="
                f"{float(jnp.max(sum_conc)):.5f}"
            )
            assert jnp.all(jnp.isfinite(new.u_ice.data))
            assert jnp.all(jnp.isfinite(resp.ocean_heat_extraction))

    def test_ridging_enabled_overfill_capped_before_ridging(self):
        """With ridging enabled, an overfilled multicat state must be capped
        BEFORE the ridging kernel (which computes donor area / drainage /
        fluxes), and the returned state stays sum_k(conc_k) <= 1 (Codex)."""
        from legoesm.ice.config import (
            SeaIceConfig, BrineConfig, RidgingConfig)
        from legoesm.ice.state import init_dynamic_ice_state
        n = 8
        grid = create_cubed_sphere(n)
        shape = (6, n, n, 2)
        st = init_dynamic_ice_state(shape, n_categories=2)
        conc = jnp.zeros(shape).at[..., 0].set(0.7).at[..., 1].set(0.6)
        h = jnp.zeros(shape).at[..., 0].set(0.5).at[..., 1].set(1.5)
        state = st._replace(
            h_ice=st.h_ice.replace(data=h),
            concentration=st.concentration.replace(data=conc),
            T_ice=st.T_ice.replace(data=jnp.full(shape, 258.0)),
        )
        config = SeaIceConfig(
            n_categories=2, dynamics="evp", transport="advect",
            itd_remap="lipscomb2001", brine=BrineConfig(enabled=True),
            ridging=RidgingConfig(enabled=True),
        )
        forcing = _make_forcing(shape=(6, n, n))
        sst = jnp.full((6, n, n), config.T_freeze_ocean)
        z = jnp.zeros((6, n, n))
        new, resp = step_sea_ice(state, forcing, sst, z, z, config,
                                 U_min=1.0, dt=3600.0, grid=grid)
        sum_conc = jnp.sum(new.concentration.data, axis=-1)
        assert float(jnp.max(sum_conc)) <= 1.0 + 1e-6
        assert jnp.all(jnp.isfinite(new.h_ice.data))
        assert jnp.all(jnp.isfinite(resp.freshwater_flux))
        assert jnp.all(jnp.isfinite(resp.salt_flux))

    def test_thermo_lead_freeze_cannot_overfill_aggregate(self):
        """Strong freezing flux + long dt must NOT let lead-freeze grow cat-0
        area beyond the available lead: the new-ice area is bounded by the
        open-water fraction so sum_k(conc_k) <= 1 reaches the ITD remap, even
        from a VALID (non-overfilled) start.  Without the source bound,
        dh_dt_open*dt/h_new_ice > 1 over-spreads cat 0 (Codex high finding)."""
        from legoesm.ice.config import SeaIceConfig, BrineConfig
        from legoesm.ice.state import init_dynamic_ice_state
        n = 8
        grid = create_cubed_sphere(n)
        shape = (6, n, n, 2)
        st = init_dynamic_ice_state(shape, n_categories=2)
        # Valid start: sum_k a_k = 0.8 < 1, with a real lead fraction (0.2).
        conc = jnp.zeros(shape).at[..., 0].set(0.4).at[..., 1].set(0.4)
        h = jnp.zeros(shape).at[..., 0].set(0.1).at[..., 1].set(1.0)
        state = st._replace(
            h_ice=st.h_ice.replace(data=h),
            concentration=st.concentration.replace(data=conc),
            T_ice=st.T_ice.replace(data=jnp.full(shape, 255.0)),
        )
        sum_in = float(jnp.max(jnp.sum(conc, axis=-1)))
        assert sum_in <= 1.0  # start is valid (not overfilled)
        # free_drift dynamics: the overfill is a thermo property (lead-freeze
        # area growth), independent of the momentum solver; free_drift stays
        # stable at the multi-day step needed to make the bound bind, whereas
        # EVP subcycling is unstable at dt = 2 days (separate, known limit).
        config = SeaIceConfig(
            n_categories=2, dynamics="free_drift", transport="advect",
            itd_remap="lipscomb2001", brine=BrineConfig(enabled=True),
        )
        # Freezing surface forcing: no shortwave, reduced downwelling longwave,
        # cold air → net surface cooling → lead-freeze flux.  With h_new_ice =
        # 0.05 m even a ~2-day step drives the uncapped dconc_growth past the
        # 0.2 lead fraction (dh_dt_open*dt/h_new_ice > 1), so the bound binds.
        forcing = _make_forcing(shape=(6, n, n))._replace(
            sw_down=jnp.zeros((6, n, n)),
            lw_down=jnp.full((6, n, n), 160.0),
            T_lowest=jnp.full((6, n, n), 240.0),
            cos_zenith=jnp.zeros((6, n, n)),
        )
        sst = jnp.full((6, n, n), config.T_freeze_ocean)
        z = jnp.zeros((6, n, n))
        new, resp = step_sea_ice(state, forcing, sst, z, z, config,
                                 U_min=1.0, dt=2 * 86400.0, grid=grid)
        sum_conc = jnp.sum(new.concentration.data, axis=-1)
        assert float(jnp.max(sum_conc)) <= 1.0 + 1e-6
        assert jnp.all(jnp.isfinite(new.h_ice.data))
        assert jnp.all(jnp.isfinite(resp.freshwater_flux))
        # Lead-freeze actually fired (aggregate area grew toward 1).
        assert float(jnp.max(sum_conc)) > sum_in
        # Volume increased (net freezing adds ice mass).
        vol_in = float(jnp.sum(h * conc))
        vol_out = float(jnp.sum(new.h_ice.data * new.concentration.data))
        assert vol_out > vol_in

    def test_v2_meltout_surface_mass_flux_is_capped(self):
        """New-physics (v2) path: under a dry strong-melt over-ablation step the
        reported surface_mass_flux is the REALIZED (capped) per-cell sublimation
        mass, bounded by the ice the column actually held — not the uncapped
        bulk latent demand lhflx/L_s (#28, codex finding on the v2 path)."""
        from legoesm.ice.config import SeaIceConfig, BrineConfig
        from legoesm.ice.state import init_dynamic_ice_state
        n = 8
        grid = create_cubed_sphere(n)
        shape = (6, n, n, 2)
        st = init_dynamic_ice_state(shape, n_categories=2)
        conc = jnp.zeros(shape).at[..., 0].set(0.5).at[..., 1].set(0.3)
        h = jnp.zeros(shape).at[..., 0].set(0.02).at[..., 1].set(0.05)
        state = st._replace(
            h_ice=st.h_ice.replace(data=h),
            concentration=st.concentration.replace(data=conc),
            T_ice=st.T_ice.replace(data=jnp.full(shape, 272.0)),
        )
        config = SeaIceConfig(
            n_categories=2, dynamics="free_drift", transport="advect",
            itd_remap="lipscomb2001", brine=BrineConfig(enabled=True),
        )
        # Dry, strongly melting forcing over a warm ocean -> over-ablation.
        forcing = _make_forcing(shape=(6, n, n))._replace(
            sw_down=jnp.full((6, n, n), 600.0),
            lw_down=jnp.full((6, n, n), 400.0),
            T_lowest=jnp.full((6, n, n), 290.0),
            q_lowest=jnp.full((6, n, n), 1e-4),
        )
        sst = jnp.full((6, n, n), config.T_freeze_ocean + 6.0)
        z = jnp.zeros((6, n, n))
        new, resp = step_sea_ice(state, forcing, sst, z, z, config,
                                 U_min=1.0, dt=86400.0, grid=grid)
        assert jnp.all(jnp.isfinite(resp.surface_mass_flux))
        # Realized per-cell sublimation mass cannot exceed the per-cell ice mass
        # the column actually held (the cap bounds it; the uncapped bulk demand
        # would far exceed this on 2-5 cm ice over a day).
        max_ice_mass_rate = config.rho_ice * jnp.sum(h * conc, axis=-1) / 86400.0
        assert jnp.all(resp.surface_mass_flux <= max_ice_mass_rate + 1e-6)
        # v2 ocean heat is scaled by the realized basal-melt fraction, so it
        # cannot exceed the uncapped turbulent demand F_ocean * sum(conc_pre)
        # (over-cooling a prognostic ocean for ice that surface melt removed).
        F_ocean = config.ocean_heat_transfer_coeff * (
            float(jnp.max(sst)) - config.T_freeze_ocean)
        raw_heat = F_ocean * jnp.sum(conc, axis=-1)
        assert jnp.all(jnp.isfinite(resp.ocean_heat_extraction))
        assert jnp.all(resp.ocean_heat_extraction <= raw_heat + 1e-6)
        # v2 MULTICAT latent<->mass pairing: response.lhflx (per-ice-area) is
        # derived from the realized sublimation MASS on the sum_conc_safe basis,
        # so resp.lhflx * sum_conc_safe == L_s * surface_mass_flux exactly (it
        # must NOT be aggregated on the post-thermo conc, codex).
        from legoesm import constants
        sum_conc_safe = jnp.maximum(
            jnp.maximum(jnp.sum(conc, axis=-1),
                        jnp.sum(new.concentration.data, axis=-1)),
            1e-30,
        )
        assert jnp.allclose(resp.lhflx * sum_conc_safe,
                            constants.L_s * resp.surface_mass_flux,
                            rtol=1e-6, atol=1e-12)

    def test_v2_single_cat_latent_basis_under_lead_freeze_growth(self):
        """Single-cat v2: when lead-freeze GROWS the ice fraction, the response
        latent must use the max(pre, post) conc basis (derived from the realized
        sublimation MASS), so resp.lhflx * conc_basis == L_s * surface_mass_flux.
        Passing result['lhflx'] through (input-conc basis) would overstate by
        conc_new/conc_in once conc grows (#28, codex)."""
        from legoesm.ice.config import SeaIceConfig, BrineConfig
        from legoesm.ice.state import init_dynamic_ice_state
        from legoesm import constants
        n = 8
        grid = create_cubed_sphere(n)
        shape = (6, n, n)
        st = init_dynamic_ice_state(shape)  # single category (no cat axis)
        conc0 = 0.5
        state = st._replace(
            h_ice=st.h_ice.replace(data=jnp.full(shape, 0.5)),
            concentration=st.concentration.replace(data=jnp.full(shape, conc0)),
            T_ice=st.T_ice.replace(data=jnp.full(shape, 258.0)),
        )
        # transport='advect' so the thermo-input conc differs from the pre-step
        # conc -- exercises that the latent basis (max(pre-dynamics, post)) is
        # the SAME for single-cat and multicat (codex).
        config = SeaIceConfig(
            n_categories=1, dynamics="free_drift", transport="advect",
            brine=BrineConfig(enabled=True),
        )
        # Cold + dry: strong open-water freezing grows conc (lead freeze) while
        # some sublimation also occurs (q_sfc over cold ice > q_lowest).
        forcing = _make_forcing(shape=shape)._replace(
            sw_down=jnp.zeros(shape), lw_down=jnp.full(shape, 150.0),
            T_lowest=jnp.full(shape, 245.0), q_lowest=jnp.full(shape, 1e-4),
        )
        sst = jnp.full(shape, config.T_freeze_ocean)
        z = jnp.zeros(shape)
        new, resp = step_sea_ice(state, forcing, sst, z, z, config,
                                 U_min=1.0, dt=3600.0, grid=grid)
        # Lead freeze grew the ice fraction (the case that breaks input-conc basis).
        assert float(jnp.min(new.concentration.data)) > conc0
        conc_basis = jnp.maximum(
            jnp.maximum(state.concentration.data, new.concentration.data), 1e-30)
        assert jnp.allclose(resp.lhflx * conc_basis,
                            constants.L_s * resp.surface_mass_flux,
                            rtol=1e-6, atol=1e-12)

    def test_transport_conserves_tracer_inventories(self):
        """The transport step advects salt / snow / pond as CONSERVED
        inventories: ``advect_ice_tracers`` conserves ``first_arg * conc``, so
        salt mass (S*h*a via the enthalpy channel), snow volume (h_snow*a), and
        pond water (area*depth*a) are each globally conserved (Codex)."""
        from legoesm.ice.transport import advect_ice_tracers
        n = 8
        grid = create_cubed_sphere(n)
        dt = 3600.0
        shp = (6, n, n)
        h = jnp.zeros(shp).at[0].set(1.2).at[1].set(0.4)
        conc = jnp.zeros(shp).at[0].set(0.8).at[1].set(0.3)
        S = jnp.zeros(shp).at[0].set(6.0).at[1].set(3.0)
        snow = jnp.zeros(shp).at[0].set(0.2).at[1].set(0.05)
        pond_thick = jnp.zeros(shp).at[0].set(0.06).at[1].set(0.02)  # area*depth
        u = jnp.full(shp, 4.0)
        v = jnp.full(shp, -2.0)

        # Tolerance = the cubed-sphere PPM flux-form conservation limit on this
        # coarse (n=8) grid: ice VOLUME (h*conc) itself conserves only to
        # ~4e-4 here, and each tracer inventory conserves AS WELL AS the volume
        # (verified: same order).  The previous scalar-tracer transport
        # (advecting S / pond_area / pond_depth on their own) conserved the
        # WRONG quantity and would drift far more than this with h varying 3x.
        rtol = 1e-3
        # Salt mass via the enthalpy (T) channel with WIDE, non-binding bounds
        # (matching the source): conserves S*h*conc.
        hS, cS, S_new = advect_ice_tracers(
            h, conc, S, u, v, grid, dt,
            T_ice_min=-10.0, T_freeze_ocean=0.0, T_max=100.0)
        assert float(jnp.sum(S_new * hS * cS)) == pytest.approx(
            float(jnp.sum(S * h * conc)), rel=rtol)
        # Cap-binding case: S exactly at the physical cap (12).  The transport
        # bound is NON-binding, so the tiny monotone-PPM overshoot is NOT
        # clipped away (salt is conserved); the physical clamp is left to the
        # brine budget (which routes its residual to the ocean conservatively).
        S_cap = jnp.full(shp, 12.0)
        hC, cC, S_cap_new = advect_ice_tracers(
            h, conc, S_cap, u, v, grid, dt,
            T_ice_min=-10.0, T_freeze_ocean=0.0, T_max=100.0)
        assert float(jnp.sum(S_cap_new * hC * cC)) == pytest.approx(
            float(jnp.sum(S_cap * h * conc)), rel=rtol), "cap salt deleted"
        # Overshoot is tiny (monotone PPM): well below any large spurious flux.
        assert float(jnp.max(S_cap_new)) <= 12.0 + 0.05
        # Snow volume via the thickness (h) channel: conserves h_snow*conc.
        snow_new, cSn, _ = advect_ice_tracers(
            snow, conc, jnp.full(shp, 263.0), u, v, grid, dt)
        assert float(jnp.sum(snow_new * cSn)) == pytest.approx(
            float(jnp.sum(snow * conc)), rel=rtol)
        # Pond water via the thickness channel: conserves (area*depth)*conc.
        pond_new, cP, _ = advect_ice_tracers(
            pond_thick, conc, jnp.full(shp, 263.0), u, v, grid, dt)
        assert float(jnp.sum(pond_new * cP)) == pytest.approx(
            float(jnp.sum(pond_thick * conc)), rel=rtol)


# ==============================================================================
# Sea ice COUPLED + functional on ALL ocean grid types (goal verification)
# ==============================================================================

class TestAllOceanGridsCoupled:
    """``step_sea_ice`` (new-physics v2: EVP + advective transport + brine) must
    run end-to-end, stay finite/bounded/stable, conserve water across
    ice+ocean+atmosphere, and deliver a paired ice->ocean / ice->atmosphere
    exchange through ``blend_tiles`` on EVERY ocean grid type the ice supports:
    cubed-sphere, lat-lon C-grid, and MPAS Voronoi.  Verifies the session goal
    'coupled and functional with all ocean grid types'."""

    def _forcing(self, shape):
        from legoesm.core.coupling_fields import AtmToSurface
        z = jnp.zeros(shape)
        f = lambda v: jnp.full(shape, v)
        return AtmToSurface(
            sw_down=f(50.0), lw_down=f(200.0), precip_total=z, precip_snow=z,
            T_lowest=f(250.0), q_lowest=f(1e-3), u_lowest=f(5.0), v_lowest=f(-3.0),
            p_lowest=f(9.5e4), p_surface=f(1e5), rho_lowest=f(1.2),
            cos_zenith=f(0.3), co2_ppmv=f(400.0),
            has_radiation=jnp.ones(shape), has_precipitation=jnp.ones(shape))

    def _run_grid(self, grid, shape, dynamics):
        from legoesm.ice.config import SeaIceConfig, BrineConfig
        from legoesm.ice.state import init_dynamic_ice_state
        from legoesm.core.coupling_fields import TileResponse
        from legoesm.coupler.config import TileConfig
        from legoesm.coupler.tile_fractions import (
            compute_tile_fractions, blend_tiles)
        from legoesm import constants

        st = init_dynamic_ice_state(shape)  # single-category dynamic state
        st = st._replace(
            h_ice=st.h_ice.replace(data=jnp.full(shape, 1.0)),
            concentration=st.concentration.replace(data=jnp.full(shape, 0.8)),
            T_ice=st.T_ice.replace(data=jnp.full(shape, 260.0)))
        config = SeaIceConfig(n_categories=1, dynamics=dynamics,
                              transport="advect", brine=BrineConfig(enabled=True))
        forcing = self._forcing(shape)
        sst = jnp.full(shape, config.T_freeze_ocean)
        z = jnp.zeros(shape)

        # ---- 3-step stability + finiteness + bounds on this grid ----
        state = st
        for _ in range(3):
            state, resp = step_sea_ice(state, forcing, sst, z, z, config,
                                       U_min=1.0, dt=3600.0, grid=grid)
            assert jnp.all(jnp.isfinite(state.h_ice.data))
            assert jnp.all(jnp.isfinite(state.concentration.data))
            assert jnp.all(jnp.isfinite(state.T_ice.data))
            assert float(jnp.min(state.concentration.data)) >= -1e-9
            assert float(jnp.max(state.concentration.data)) <= 1.0 + 1e-9
            assert float(jnp.min(state.h_ice.data)) >= 0.0
            # ice->ocean / ice->atmosphere exchange channels finite + grid-shaped
            for fld in (resp.freshwater_flux, resp.ocean_heat_extraction,
                        resp.salt_flux, resp.ocean_stress_x, resp.ocean_stress_y,
                        resp.surface_mass_flux, resp.lhflx, resp.shflx):
                assert jnp.all(jnp.isfinite(fld))
                assert fld.shape == shape

        # ---- one-step PER-CELL total water closure (ice + ocean + atmosphere) ----
        # Transport redistributes ice BETWEEN cells (an internal flux divergence
        # that is NOT an ocean/atmosphere exchange), so the per-cell closure only
        # isolates the thermo exchange with transport='none'.  Then the per-cell
        # ice mass change equals exactly the freshwater sent to the ocean plus
        # the realized sublimation mass sent to the atmosphere.
        no_transport = config._replace(transport="none")
        prev = state
        new, resp = step_sea_ice(prev, forcing, sst, z, z, no_transport,
                                 U_min=1.0, dt=3600.0, grid=grid)
        dV_dt = (new.h_ice.data * new.concentration.data
                 - prev.h_ice.data * prev.concentration.data) / 3600.0
        water_resid = (config.rho_ice * dV_dt + resp.freshwater_flux
                       + resp.surface_mass_flux)
        assert float(jnp.max(jnp.abs(water_resid))) < 1e-9, (
            f"ice+ocean+atmosphere water not conserved on this grid: "
            f"{float(jnp.max(jnp.abs(water_resid))):.3e}")

        # ---- coupler blend delivers a paired exchange on this grid ----
        zt = jnp.zeros(shape)
        zero = TileResponse(
            T_sfc=zt, albedo=zt, emissivity=zt, z0=zt, q_surface=zt,
            shflx=zt, lhflx=zt, tau_x=zt, tau_y=zt, lw_up=zt, u_ocean_sfc=zt,
            v_ocean_sfc=zt, co2_flux=zt, freshwater_flux=zt,
            ocean_heat_extraction=zt, ocean_stress_x=zt, ocean_stress_y=zt,
            surface_mass_flux=zt, salt_flux=zt)
        tcfg = TileConfig(f_land=zt, f_lake=zt)  # pure-water cell
        fracs = compute_tile_fractions(tcfg, new.concentration.data)
        blended = blend_tiles(zero, resp, zero, zero, fracs)
        # Atmosphere energy <-> water pairing holds on every grid.
        assert jnp.allclose(blended.lhflx,
                            constants.L_s * blended.surface_mass_flux,
                            rtol=1e-10, atol=1e-12)
        assert jnp.all(jnp.isfinite(blended.freshwater_flux))
        assert jnp.all(jnp.isfinite(blended.ocean_heat_extraction))
        assert blended.freshwater_flux.shape == shape

    def test_cubed_sphere_coupled(self):
        grid = create_cubed_sphere(8)
        self._run_grid(grid, (6, 8, 8), "evp")

    def test_latlon_cgrid_coupled(self):
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(n_lat=16, n_lon=32)
        self._run_grid(grid, (16, 32), "evp")

    def test_mpas_voronoi_coupled(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
        self._run_grid(mesh, (mesh.nCells,), "free_drift")


class TestVoronoiTransportUpwind:
    """First-order upwind Voronoi transport must be POSITIVITY-preserving: a
    non-negative scalar advected at CFL<=1 stays >= 0.  The old centered
    0.5*(q1+q2) edge reconstruction undershoots to negative volume, which the
    downstream jnp.maximum(vol,0)/clip(conc,0,1) silently turn into a mass
    SOURCE (audit finding #1)."""

    def test_upwind_preserves_positivity(self):
        import numpy as np
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ice.transport import fv_flux_divergence_voronoi
        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
        nC = int(mesh.nCells)
        # Sharp step field in [0,1] -> strong gradients at every interface.
        q = jnp.asarray((np.arange(nC) % 2).astype(np.float64))
        u = jnp.ones((nC,)); v = jnp.zeros((nC,))   # uniform east, |u_edge|<=1
        # CFL ~ 0.5: |u|=1 m/s, dx = smallest cell spacing.
        dt = 0.5 * float(jnp.min(mesh.dcEdge))
        q_new = q + dt * fv_flux_divergence_voronoi(q, u, v, mesh)
        # Positivity is the property the audit bug violated (negative volume).
        # (No upper-bound check: uniform east flow on a sphere is not
        # divergence-free, so flux-form legitimately amplifies converging cells.)
        assert float(jnp.min(q_new)) >= -1e-9, (
            f"upwind produced negative volume: {float(jnp.min(q_new)):.3e}")


# ==============================================================================
# #28 multicat ITD bin-accuracy of the overfill cap + Lipscomb remap
# ==============================================================================

class TestMulticatITDBinAccuracy:
    """The overfill cap (_cap_multicat_concentration, uniform compaction) feeds
    the Lipscomb ITD remap.  For REALISTIC sub-CFL overfill the compacted ice is
    re-sorted strictly within its bin bounds; for pathological LARGE overfill it
    stays volume/salt/area conserving (ITD shape may degrade — accepted, the
    regime is unreachable).  Verifies #28."""

    def _remap(self, h, a, S_ice):
        from legoesm.ice.sea_ice import _cap_multicat_concentration
        from legoesm.ice.itd import lipscomb_2001_remap
        n_cat = h.shape[-1]
        ac, hc, _, _ = _cap_multicat_concentration(a, h)
        rem = lipscomb_2001_remap(
            h_old=h, a_old=a, h_new=hc, a_new=ac, n_cat=n_cat, dt=3600.0,
            T_new=jnp.full(h.shape, 258.0), S_new=S_ice,
            V_snow_new=jnp.zeros(h.shape), V_pond_new=jnp.zeros(h.shape),
            T_max=273.15)
        return ac, hc, rem

    def test_cap_realistic_overfill_stays_bin_accurate(self):
        from legoesm.ice.itd import category_bounds, upper_bounds
        n_cat = 5
        lo, hi = category_bounds(n_cat), upper_bounds(n_cat)
        # Each category at a thickness inside its bin; sum_a = 1.03 (sub-CFL
        # overfill is normally <<1% — this is already ~30x a realistic value).
        h = jnp.array([[0.3, 1.0, 1.8, 3.0, 5.0]])
        a = jnp.array([[0.206, 0.206, 0.206, 0.206, 0.206]])
        S_ice = jnp.full((1, n_cat), 5.0)
        ac, hc, rem = self._remap(h, a, S_ice)
        hr, ar = rem["h"], rem["a"]
        # Every occupied category sits strictly within its own bin bounds.
        for k in range(n_cat):
            if float(ar[0, k]) > 1e-9:
                assert float(lo[k]) - 1e-6 <= float(hr[0, k]) <= float(hi[k]) + 1e-6, (
                    f"cat {k} h={float(hr[0,k]):.4f} outside bin "
                    f"[{float(lo[k]):.3f},{float(hi[k]):.3f}]")
        # Aggregate-area capped + volume + salt conserved.
        assert float(jnp.sum(ar)) <= 1.0 + 1e-9
        assert float(jnp.sum(hr * ar)) == pytest.approx(float(jnp.sum(h * a)), rel=1e-3)
        assert float(jnp.sum(rem["S"] * hr * ar)) == pytest.approx(
            float(jnp.sum(S_ice * h * a)), rel=1e-3)

    def test_cap_large_overfill_conserves(self):
        # Pathological sum_a = 2.0 (unreachable in practice): the ITD shape may
        # degrade but volume / salt / total area MUST stay conserved + finite.
        n_cat = 5
        h = jnp.array([[0.05, 0.4, 1.0, 2.2, 4.0]])
        a = jnp.array([[0.4, 0.4, 0.4, 0.4, 0.4]])  # sum = 2.0
        S_ice = jnp.full((1, n_cat), 5.0)
        ac, hc, rem = self._remap(h, a, S_ice)
        hr, ar = rem["h"], rem["a"]
        assert jnp.all(jnp.isfinite(hr)) and jnp.all(jnp.isfinite(ar))
        assert float(jnp.sum(ar)) <= 1.0 + 1e-9
        assert float(jnp.sum(hr * ar)) == pytest.approx(float(jnp.sum(h * a)), rel=1e-3)
        assert float(jnp.sum(rem["S"] * hr * ar)) == pytest.approx(
            float(jnp.sum(S_ice * h * a)), rel=1e-3)

    def test_cap_differentiable(self):
        from legoesm.ice.sea_ice import _cap_multicat_concentration
        a = jnp.array([[0.3, 0.3, 0.3, 0.3, 0.3]])

        def loss(h_row):
            ac, hc, _, _ = _cap_multicat_concentration(a, h_row[None, :])
            return jnp.sum(hc * ac)  # conserved volume -> grad finite

        g = jax.grad(loss)(jnp.array([0.3, 1.0, 1.8, 3.0, 5.0]))
        assert jnp.all(jnp.isfinite(g))


# ==============================================================================
# #30 multicat FULL-pipeline tracer audit: brine + snow(flooding) + ponds +
#     ridging ALL enabled through transport + thermo + ITD remap + ridging
# ==============================================================================

class TestMulticatFullPipelineTracers:
    """The component tests cover apply_ridging / step_ponds in isolation; this
    exercises the WHOLE multi-category step (EVP dynamics -> advective transport
    -> v2 thermo -> Lipscomb ITD remap -> ridging) with brine, snow+flooding,
    ponds, and ridging ALL enabled together, asserting every tracer stays
    finite, physically bounded, and non-exploding over several steps (#30)."""

    def _state(self, n=8, n_cat=3):
        from legoesm.ice.state import init_dynamic_ice_state
        shape = (6, n, n, n_cat)
        st = init_dynamic_ice_state(shape, n_categories=n_cat)
        # Nonuniform per-cat ice + tracers + a hemispheric gradient so transport
        # AND ridging convergence are exercised.
        h = jnp.zeros(shape).at[..., 0].set(0.4).at[..., 1].set(1.2).at[..., 2].set(3.0)
        face = jnp.zeros((6, n, n)).at[0].set(0.5).at[1].set(0.25).at[2].set(0.1)
        conc = jnp.stack([face, face * 0.6, face * 0.3], axis=-1)  # sum < 1
        S = jnp.full(shape, 5.0).at[0, ..., 0].set(9.0).at[1, ..., 0].set(2.0)
        snow = jnp.zeros(shape).at[..., 0].set(0.15).at[..., 1].set(0.05)
        pa = jnp.zeros(shape).at[..., 0].set(0.3)
        pd = jnp.zeros(shape).at[..., 0].set(0.08)
        return st._replace(
            h_ice=st.h_ice.replace(data=h),
            concentration=st.concentration.replace(data=conc),
            T_ice=st.T_ice.replace(data=jnp.full(shape, 262.0)),
            S_ice=st.S_ice.replace(data=S),
            h_snow=st.h_snow.replace(data=snow),
            pond_area=st.pond_area.replace(data=pa),
            pond_depth=st.pond_depth.replace(data=pd),
        )

    def test_all_physics_on_multistep_bounded_and_conserving(self):
        from legoesm.ice.config import (
            SeaIceConfig, BrineConfig, SnowConfig, MeltPondConfig, RidgingConfig)
        n = 8
        grid = create_cubed_sphere(n)
        config = SeaIceConfig(
            n_categories=3, dynamics="evp", transport="advect",
            itd_remap="lipscomb2001",
            brine=BrineConfig(enabled=True),
            snow=SnowConfig(enabled=True, flooding=True),
            ponds=MeltPondConfig(enabled=True),
            ridging=RidgingConfig(enabled=True),
        )
        state = self._state(n=n)
        # Mild melt-ish forcing + winds (advection + ridging convergence).
        forcing = _make_forcing(shape=(6, n, n))
        sst = jnp.full((6, n, n), config.T_freeze_ocean + 0.5)
        z = jnp.zeros((6, n, n))
        Smax = config.brine.S_ice_max
        for _ in range(4):
            state, resp = step_sea_ice(state, forcing, sst, z, z, config,
                                       U_min=1.0, dt=3600.0, grid=grid)
            d = state
            # finiteness of every prognostic tracer
            for fld in (d.h_ice.data, d.concentration.data, d.T_ice.data,
                        d.S_ice.data, d.h_snow.data, d.pond_area.data,
                        d.pond_depth.data):
                assert jnp.all(jnp.isfinite(fld))
            # physical bounds
            assert float(jnp.min(d.h_ice.data)) >= 0.0
            assert float(jnp.min(d.h_snow.data)) >= -1e-12
            assert float(jnp.min(d.pond_depth.data)) >= -1e-12
            assert -1e-9 <= float(jnp.min(d.pond_area.data))
            assert float(jnp.max(d.pond_area.data)) <= 1.0 + 1e-6
            assert -1e-9 <= float(jnp.min(d.S_ice.data))
            assert float(jnp.max(d.S_ice.data)) <= Smax + 1e-6
            assert float(jnp.max(jnp.sum(d.concentration.data, axis=-1))) <= 1.0 + 1e-6
            assert float(jnp.min(d.T_ice.data)) >= config.T_ice_min - 1e-6
            assert float(jnp.max(d.T_ice.data)) <= config.T_melt_surface + 1e-6
            # ice->ocean / ice->atmosphere exchange finite
            for fld in (resp.freshwater_flux, resp.ocean_heat_extraction,
                        resp.salt_flux, resp.surface_mass_flux,
                        resp.ocean_stress_x, resp.ocean_stress_y):
                assert jnp.all(jnp.isfinite(fld))
        # Salt inventory stayed bounded (no tracer blow-up over the run).
        salt_mass = float(jnp.sum(state.S_ice.data * state.h_ice.data
                                  * state.concentration.data))
        assert jnp.isfinite(salt_mass) and salt_mass >= 0.0


class TestEvpSolverInputGuards:
    """PR C: evp_solver gained the fail-early input guards its mevp_solver
    sibling already had — N_evp<1 / non-finite T_evp/e_yield/Delta_min raise
    instead of silently no-op'ing or poisoning every gradient."""

    def _args(self):
        grid = _make_grid(4)
        z = jnp.zeros((6, 4, 4))
        # u,v,s11,s22,s12,h,conc,wu,wv,ou,ov, grid, dt
        return (z, z, z, z, z, z, z, z, z, z, z, grid, 3600.0)

    def test_n_evp_below_one_raises(self):
        with pytest.raises(ValueError, match="N_evp"):
            evp_solver(*self._args(), N_evp=0)

    def test_non_finite_t_evp_raises(self):
        with pytest.raises(ValueError, match="T_evp"):
            evp_solver(*self._args(), N_evp=10, T_evp=float("nan"))

    def test_jit_with_traced_e_yield_does_not_break(self):
        """``e_yield`` is a TRAINABLE param, so it can arrive as a JAX tracer
        via the training-override path under jit/grad. The input guards must NOT
        apply Python boolean control flow to it (PR C codex catch) — only the
        static iteration params (N_evp / T_evp) are validated."""
        grid = _make_grid(4)
        h = jnp.full((6, 4, 4), 0.5)
        a = jnp.full((6, 4, 4), 0.5)
        z = jnp.zeros((6, 4, 4))

        def run(e_yield):
            return evp_solver(
                z, z, z, z, z, h, a, z, z, z, z, grid, 3600.0,
                N_evp=5, e_yield=e_yield,
            )[0]

        # Traced e_yield under jit must compile + run (would raise a
        # TracerBoolConversionError with the old np.isfinite(e_yield) guard).
        out = jax.jit(run)(jnp.asarray(2.0))
        assert jnp.all(jnp.isfinite(out))
        # And it is differentiable w.r.t. the trainable e_yield.
        g = jax.grad(lambda e: jnp.sum(run(e) ** 2))(jnp.asarray(2.0))
        assert jnp.isfinite(g)
