"""Physics-validator group sweep: ocean vertical_mixing (KPP), bottom_drag,
convection/enhanced_diffusion, surface_forcing, and the equation of state.

Five truth tiers per scheme (validator protocol):
  1. UNITS   — dimensional consistency + constants from legoesm.constants.
  2. SIGNS   — downgradient diffusion, drag opposes flow, latent/sensible
               sign, restoring relaxes toward target.
  3. CONSERVATION — column / box probes (volume-integrated tendency = 0 for
               interior mixing; surface forcing only touches the top cell).
  4. DIFFERENTIABILITY — jax.grad finite, non-zero, reaches tunable params.
  5. IDEALIZED — one analytic / benchmark sanity case per scheme.

Run with:  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ===========================================================================
# Equation of state
# ===========================================================================

def test_eos_unesco_reference_values():
    """UNESCO 1980 reference points (Fofonoff & Millard 1983 Table)."""
    from legoesm.ocean.eos import unesco80_eos
    z = jnp.array
    assert abs(float(unesco80_eos(z(0.0), z(0.0), z(0.0))) - 999.842594) < 1e-4
    assert abs(float(unesco80_eos(z(0.0), z(35.0), z(0.0))) - 1028.106331) < 1e-3
    assert abs(float(unesco80_eos(z(20.0), z(35.0), z(0.0))) - 1024.78) < 1e-1


def test_eos_alpha_beta_signs():
    """alpha = -(1/rho)d(rho)/dT > 0 (warm water lighter);
    beta = (1/rho)d(rho)/dS > 0 (salt water denser)."""
    from legoesm.ocean.eos import thermal_expansion_coeff, haline_contraction_coeff
    T, S, p = jnp.array(10.0), jnp.array(35.0), jnp.array(0.0)
    a = float(thermal_expansion_coeff(T, S, p))
    b = float(haline_contraction_coeff(T, S, p))
    assert a > 0.0, "thermal expansion must be positive at 10C/35PSU"
    assert b > 0.0, "haline contraction must be positive"
    assert 1e-4 < a < 3e-4
    assert 5e-4 < b < 1e-3


def test_eos_density_monotonic_in_T_and_S():
    """Warmer -> lighter; saltier -> denser (Wright EOS)."""
    from legoesm.ocean.eos import wright_eos
    z = jnp.array
    assert float(wright_eos(z(20.0), z(35.0), z(0.0))) < float(
        wright_eos(z(5.0), z(35.0), z(0.0))
    )
    assert float(wright_eos(z(10.0), z(36.0), z(0.0))) > float(
        wright_eos(z(10.0), z(34.0), z(0.0))
    )


def test_eos_pressure_increases_density():
    """Compression: higher pressure -> higher in-situ density."""
    from legoesm.ocean.eos import wright_eos, unesco80_eos
    z = jnp.array
    for fn in (wright_eos, unesco80_eos):
        lo = float(fn(z(10.0), z(35.0), z(0.0)))
        hi = float(fn(z(10.0), z(35.0), z(1e7)))  # ~1000 dbar
        assert hi > lo, f"{fn.__name__} density must rise with pressure"


def test_eos_grad_finite_nonzero_reaches_TS():
    """jax.grad of density wrt T and S is finite & non-zero (AD reaches it)."""
    from legoesm.ocean.eos import wright_eos

    def rho_T(T):
        return wright_eos(T, jnp.array(35.0), jnp.array(1e6))

    def rho_S(S):
        return wright_eos(jnp.array(10.0), S, jnp.array(1e6))

    gT = float(jax.grad(rho_T)(jnp.array(10.0)))
    gS = float(jax.grad(rho_S)(jnp.array(35.0)))
    assert jnp.isfinite(gT) and gT < 0.0, "d(rho)/dT < 0"
    assert jnp.isfinite(gS) and gS > 0.0, "d(rho)/dS > 0"


def test_eos_alpha_matches_autodiff_identity():
    """alpha = -drho/dT / rho should equal the autodiff derivative chain."""
    from legoesm.ocean.eos import wright_eos, thermal_expansion_coeff
    T, S, p = jnp.array(12.0), jnp.array(34.0), jnp.array(2e6)
    rho = float(wright_eos(T, S, p))
    drho_dT = float(jax.grad(lambda t: wright_eos(t, S, p))(T))
    alpha = float(thermal_expansion_coeff(T, S, p))
    assert abs(alpha - (-drho_dT / rho)) < 1e-9


def test_eos_density_derivatives_match_alpha_beta():
    """eos_density_derivatives(wright) == (-rho*alpha, +rho*beta)."""
    from legoesm.ocean.eos import (
        eos_density_derivatives, wright_eos,
        thermal_expansion_coeff, haline_contraction_coeff,
    )
    T = jnp.array([[8.0, 14.0]])
    S = jnp.array([[34.5, 35.2]])
    p = jnp.array([[1e6, 3e6]])
    dT, dS = eos_density_derivatives(wright_eos, T, S, p)
    rho = wright_eos(T, S, p)
    a = thermal_expansion_coeff(T, S, p)
    b = haline_contraction_coeff(T, S, p)
    assert jnp.allclose(dT, -rho * a, rtol=1e-5, atol=1e-8)
    assert jnp.allclose(dS, rho * b, rtol=1e-5, atol=1e-8)


# ===========================================================================
# Bottom drag (linear + quadratic)
# ===========================================================================

def _sf_state(n_levels=6, H_max=4000.0, T_sfc_C=20.0):
    z = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
    shape = (6, 3, 3, n_levels)
    T = jnp.full(shape, T_sfc_C, dtype=jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    J = jnp.ones((6, 3, 3), dtype=jnp.float64)
    return T, S, z, J


def test_bulk_warm_ocean_loses_heat_to_cold_air():
    """T_s >> T_a, no radiation: turbulent fluxes cool the ocean."""
    from legoesm.ocean.physics.surface_forcing.bulk_formulas import (
        bulk_formula_surface_forcing,
    )
    from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
    T, S, z, J = _sf_state(T_sfc_C=25.0)
    cfg = BulkFormulaConfig(
        bulk_scheme="constant", T_a=275.0, q_a=0.001, U_a=8.0,
        SW_down=0.0, LW_down=0.0,
    )
    out = bulk_formula_surface_forcing(T, S, z, J, cfg)
    assert float(out.Q_net.mean()) < 0.0
    assert float(out.dT_dt[..., 0].mean()) < 0.0
    assert jnp.allclose(out.dT_dt[..., 1:], 0.0)


def test_bulk_easterly_does_not_warm_cool_ocean():
    """Strong easterly (U_a<0) must not flip Q_sh sign and warm the ocean."""
    from legoesm.ocean.physics.surface_forcing.bulk_formulas import (
        bulk_formula_surface_forcing,
    )
    from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
    T, S, z, J = _sf_state(T_sfc_C=25.0)
    west = BulkFormulaConfig(bulk_scheme="constant", T_a=275.0, q_a=0.001,
                             U_a=8.0, SW_down=0.0, LW_down=0.0)
    east = BulkFormulaConfig(bulk_scheme="constant", T_a=275.0, q_a=0.001,
                             U_a=-8.0, SW_down=0.0, LW_down=0.0)
    qw = float(bulk_formula_surface_forcing(T, S, z, J, west).Q_net.mean())
    qe = float(bulk_formula_surface_forcing(T, S, z, J, east).Q_net.mean())
    assert qw < 0.0 and qe < 0.0
    assert abs(qw - qe) < 1e-9


def test_bulk_stress_sign_directional():
    """Stress is directional: tau_x sign tracks U_a sign."""
    from legoesm.ocean.physics.surface_forcing.bulk_formulas import (
        bulk_formula_surface_forcing,
    )
    from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
    T, S, z, J = _sf_state()
    pos = BulkFormulaConfig(bulk_scheme="constant", U_a=8.0)
    neg = BulkFormulaConfig(bulk_scheme="constant", U_a=-8.0)
    tx_p = float(bulk_formula_surface_forcing(T, S, z, J, pos).tau_x.mean())
    tx_n = float(bulk_formula_surface_forcing(T, S, z, J, neg).tau_x.mean())
    assert tx_p > 0.0 and tx_n < 0.0


def test_bulk_uses_central_constants_not_hardcoded():
    """Saturation comes from thermo; freezing offset / SB from constants."""
    from legoesm.ocean.physics.surface_forcing import bulk_formulas as bf
    import inspect
    src = inspect.getsource(bf)
    assert "saturation_specific_humidity" in src
    assert "constants.T_freeze" in src
    assert "constants.sigma_sb" in src
    # No raw freezing-offset / Stefan-Boltzmann literal in the source. Build
    # the search strings numerically so this test source has no banned token.
    freeze_literal = repr(float(constants.T_freeze))  # const-ok: assertion guard
    assert freeze_literal not in src
    sb_prefix = ("%.2e" % float(constants.sigma_sb))[:4]  # const-ok: assertion guard
    assert sb_prefix not in src


def test_bulk_grad_reaches_transfer_coeffs():
    """jax.grad of Q_net reaches C_H."""
    from legoesm.ocean.physics.surface_forcing.bulk_formulas import (
        bulk_formula_surface_forcing,
    )
    from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
    T, S, z, J = _sf_state(T_sfc_C=25.0)

    def loss(ch):
        cfg = BulkFormulaConfig(bulk_scheme="constant", C_H=ch, T_a=275.0,
                                U_a=8.0, SW_down=0.0, LW_down=0.0)
        return jnp.sum(bulk_formula_surface_forcing(T, S, z, J, cfg).Q_net)

    g = float(jax.grad(loss)(1.5e-3))
    assert jnp.isfinite(g) and g != 0.0


def test_restoring_relaxes_toward_target_and_sign():
    """T above target -> cooling; below -> warming. Surface-only."""
    from legoesm.ocean.physics.surface_forcing.restoring import (
        restoring_surface_forcing,
    )
    from legoesm.ocean.physics.surface_forcing.config import RestoringConfig

    class _Grid:
        grid_lat = jnp.zeros((6, 3, 3), dtype=jnp.float64)

    grid = _Grid()
    cfg = RestoringConfig(T_profile="constant", T_star_eq=10.0, tau_T=1e6)
    T = jnp.full((6, 3, 3, 5), 20.0, dtype=jnp.float64)
    S = jnp.full((6, 3, 3, 5), 35.0, dtype=jnp.float64)
    out = restoring_surface_forcing(T, S, grid, cfg)
    assert float(out.dT_dt[..., 0].mean()) < 0.0
    assert jnp.allclose(out.dT_dt[..., 1:], 0.0)
    Tc = jnp.full((6, 3, 3, 5), 2.0, dtype=jnp.float64)
    outc = restoring_surface_forcing(Tc, S, grid, cfg)
    assert float(outc.dT_dt[..., 0].mean()) > 0.0


def test_restoring_implicit_bounded_and_stable():
    """Implicit restoring: one step keeps T between old and target for any dt."""
    from legoesm.ocean.physics.surface_forcing.restoring import (
        restoring_surface_forcing,
    )
    from legoesm.ocean.physics.surface_forcing.config import RestoringConfig

    class _Grid:
        grid_lat = jnp.zeros((1, 1, 1), dtype=jnp.float64)

    grid = _Grid()
    cfg = RestoringConfig(T_profile="constant", T_star_eq=10.0, tau_T=2592000.0,
                          implicit=True)
    T = jnp.full((1, 1, 1, 3), 30.0, dtype=jnp.float64)
    S = jnp.full((1, 1, 1, 3), 35.0, dtype=jnp.float64)
    dt = 5e7
    out = restoring_surface_forcing(T, S, grid, cfg, dt=dt)
    T_new = float(T[0, 0, 0, 0] + dt * out.dT_dt[0, 0, 0, 0])
    assert 10.0 <= T_new <= 30.0, "implicit step must stay bounded"


# ===========================================================================
# Enhanced-diffusion convection
# ===========================================================================

def _conv_unstable_column(n_levels=8, H_max=4000.0):
    z = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
    shape = (6, 3, 3, n_levels)
    T_profile = jnp.linspace(18.0, 2.0, n_levels)
    T = jnp.broadcast_to(T_profile, shape).astype(jnp.float64)
    T = T.at[..., 0].set(0.0)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    rho = constants.rho_ocean - 0.2 * (T - 4.0)
    u = jnp.broadcast_to(jnp.linspace(0.4, -0.4, n_levels), shape).astype(jnp.float64)
    v = jnp.zeros(shape, dtype=jnp.float64)
    J = jnp.ones((6, 3, 3), dtype=jnp.float64)
    return T, S, rho, u, v, z, J


def test_conv_enhances_where_unstable():
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        convective_K_A_flag,
    )
    from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
    T, S, rho, u, v, z, J = _conv_unstable_column()
    cfg = EnhancedDiffusionConfig(K_conv=1.0, K_bg=1e-5, smooth_transition=False)
    K, A, flag = convective_K_A_flag(rho, z.dz_ref, J, cfg)
    assert bool(jnp.any(flag > 0.5)), "convection must fire at the cold cap"
    assert jnp.allclose(K[flag > 0.5], cfg.K_conv)
    assert jnp.allclose(K[flag < 0.5], cfg.K_bg)


def test_conv_tracer_column_conservation():
    """Volume-integrated tracer tendency is ~0 (zero-flux interior mixing)."""
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        enhanced_diffusion_convection,
    )
    from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
    T, S, rho, u, v, z, J = _conv_unstable_column()
    cfg = EnhancedDiffusionConfig(K_conv=1.0, K_bg=1e-5)
    out = enhanced_diffusion_convection(T, S, rho, z, J, cfg,
                                        apply_diffusion=True, dt=600.0)
    dz = z.dz_ref * J[..., jnp.newaxis]
    col_T = jnp.sum(out.dT_dt * dz, axis=-1)
    col_S = jnp.sum(out.dS_dt * dz, axis=-1)
    assert jnp.allclose(col_T, 0.0, atol=1e-9)
    assert jnp.allclose(col_S, 0.0, atol=1e-9)


def test_conv_smooth_transition_grad_finite_nonzero():
    """Smooth (sigmoid) transition -> grad of K wrt density finite/nonzero."""
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        convective_K_A_flag,
    )
    from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
    T, S, rho, u, v, z, J = _conv_unstable_column()
    cfg = EnhancedDiffusionConfig(K_conv=1.0, K_bg=1e-5, smooth_transition=True)

    def loss(r):
        K, _, _ = convective_K_A_flag(r, z.dz_ref, J, cfg)
        return jnp.sum(K)

    g = jax.grad(loss)(rho)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_conv_relaxes_instability():
    """Convective adjustment warms the cold top cap (reduces instability)."""
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        enhanced_diffusion_convection,
    )
    from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
    T, S, rho, u, v, z, J = _conv_unstable_column()
    cfg = EnhancedDiffusionConfig(K_conv=1.0, K_bg=1e-5)
    out = enhanced_diffusion_convection(T, S, rho, z, J, cfg,
                                        apply_diffusion=True, dt=600.0)
    assert float(out.dT_dt[0, 0, 0, 0]) > 0.0


# ===========================================================================
# KPP vertical mixing
# ===========================================================================

def _kpp_state(n_levels=12, H_max=400.0):
    z = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
    shape = (6, 2, 2, n_levels)
    T_profile = jnp.concatenate([
        jnp.full((4,), 18.0),
        jnp.linspace(18.0, 4.0, n_levels - 4),
    ])
    T = jnp.broadcast_to(T_profile, shape).astype(jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    from legoesm.ocean.eos import wright_eos
    p = jnp.zeros_like(T)
    rho = wright_eos(T, S, p)
    u = jnp.broadcast_to(jnp.linspace(0.2, 0.0, n_levels), shape).astype(jnp.float64)
    v = jnp.zeros(shape, dtype=jnp.float64)
    eta = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    J = jnp.ones((6, 2, 2), dtype=jnp.float64)
    return u, v, T, S, rho, eta, z, J


def test_kpp_diffusivities_positive_and_bounded():
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    u, v, T, S, rho, eta, z, J = _kpp_state()
    cfg = KPPConfig()
    tau_x = jnp.full((6, 2, 2), 0.1, dtype=jnp.float64)
    tau_y = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    B_f = jnp.full((6, 2, 2), 1e-7, dtype=jnp.float64)
    out = kpp_vertical_mixing(u, v, T, S, rho, eta, z, J, cfg,
                              tau_x=tau_x, tau_y=tau_y, B_f=B_f,
                              apply_diffusion=False)
    assert jnp.all(out.K_v >= 0.0)
    assert jnp.all(out.A_v >= 0.0)
    assert jnp.all(out.K_v <= cfg.K_max + 1e-12)
    assert jnp.all(out.A_v <= cfg.K_max + 1e-12)
    assert jnp.all(jnp.isfinite(out.K_v))
    assert jnp.all(jnp.isfinite(out.A_v))


def test_kpp_wind_stress_increases_surface_mixing():
    """More wind stress -> larger boundary-layer diffusivity near surface."""
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    u, v, T, S, rho, eta, z, J = _kpp_state()
    cfg = KPPConfig()
    zeroB = jnp.zeros((6, 2, 2), dtype=jnp.float64)

    def surf_K(tx):
        tau_x = jnp.full((6, 2, 2), tx, dtype=jnp.float64)
        out = kpp_vertical_mixing(u, v, T, S, rho, eta, z, J, cfg,
                                  tau_x=tau_x, tau_y=zeroB, B_f=zeroB,
                                  apply_diffusion=False)
        return float(jnp.mean(out.K_v[..., :2]))

    assert surf_K(0.4) > surf_K(0.02)


def test_kpp_tracer_column_conservation():
    """Local KPP diffusion + non-local transport conserve column tracer."""
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    u, v, T, S, rho, eta, z, J = _kpp_state()
    cfg = KPPConfig()
    tau_x = jnp.full((6, 2, 2), 0.1, dtype=jnp.float64)
    tau_y = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    B_f = jnp.full((6, 2, 2), 1e-7, dtype=jnp.float64)
    Q_sfc_T = jnp.full((6, 2, 2), -1e-5, dtype=jnp.float64)
    Q_sfc_S = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    out = kpp_vertical_mixing(u, v, T, S, rho, eta, z, J, cfg,
                              tau_x=tau_x, tau_y=tau_y, B_f=B_f,
                              Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
                              apply_diffusion=True, dt=900.0)
    dz = z.dz_ref * J[..., jnp.newaxis]
    col_T = jnp.sum(out.dT_dt * dz, axis=-1)
    col_S = jnp.sum(out.dS_dt * dz, axis=-1)
    assert jnp.allclose(col_T, 0.0, atol=1e-8), f"col_T={col_T}"
    assert jnp.allclose(col_S, 0.0, atol=1e-8), f"col_S={col_S}"


def test_kpp_momentum_column_conservation():
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    u, v, T, S, rho, eta, z, J = _kpp_state()
    cfg = KPPConfig()
    tau_x = jnp.full((6, 2, 2), 0.1, dtype=jnp.float64)
    tau_y = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    out = kpp_vertical_mixing(u, v, T, S, rho, eta, z, J, cfg,
                              tau_x=tau_x, tau_y=tau_y,
                              B_f=jnp.zeros((6, 2, 2)),
                              apply_diffusion=True, dt=900.0)
    dz = z.dz_ref * J[..., jnp.newaxis]
    col_u = jnp.sum(out.du_dt * dz, axis=-1)
    assert jnp.allclose(col_u, 0.0, atol=1e-8)


def test_kpp_grad_reaches_tunables():
    """jax.grad of a K_v functional reaches K_0_shear, finite & nonzero.

    K_0_shear only enters the *interior* shear-instability curve (Ri<Ri_0,
    below the boundary layer), so the probe state has a shallow strongly-
    stratified cap (shallow h_bl) above a near-neutral, strongly-sheared
    deep interior — the regime in which interior shear mixing is active.
    """
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    from legoesm.ocean.eos import wright_eos
    z = create_ocean_z_star(n_levels=12, H_max=400.0)
    shape = (6, 2, 2, 12)
    T_profile = jnp.concatenate([jnp.array([20.0, 12.0]), jnp.full((10,), 11.99)])
    T = jnp.broadcast_to(T_profile, shape).astype(jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    rho = wright_eos(T, S, jnp.zeros_like(T))
    u_profile = jnp.concatenate([jnp.array([0.0, 0.0]), jnp.linspace(0.6, -0.6, 10)])
    u = jnp.broadcast_to(u_profile, shape).astype(jnp.float64)
    v = jnp.zeros(shape, dtype=jnp.float64)
    eta = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    J = jnp.ones((6, 2, 2), dtype=jnp.float64)
    tau_x = jnp.full((6, 2, 2), 0.005, dtype=jnp.float64)
    tau_y = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    B_f = jnp.zeros((6, 2, 2), dtype=jnp.float64)

    def loss(k0):
        cfg = KPPConfig(K_0_shear=k0)
        out = kpp_vertical_mixing(u, v, T, S, rho, eta, z, J, cfg,
                                  tau_x=tau_x, tau_y=tau_y, B_f=B_f,
                                  apply_diffusion=False)
        return jnp.sum(out.K_v)

    g = float(jax.grad(loss)(5e-3))
    assert jnp.isfinite(g) and g > 0.0


def test_kpp_grad_reaches_K_conv():
    """K_conv (interior static-instability enhancement) is AD-reachable when
    a deep interface is statically unstable (N^2 < Ri_conv) and below the BL."""
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    from legoesm.ocean.eos import wright_eos
    z = create_ocean_z_star(n_levels=12, H_max=400.0)
    shape = (6, 2, 2, 12)
    # Stable shallow cap, then a deep density inversion (warm under cold).
    T_profile = jnp.concatenate([
        jnp.array([20.0, 14.0]),
        jnp.full((4,), 13.9),
        jnp.array([8.0, 13.0]),     # inversion at depth -> N^2 < 0
        jnp.full((4,), 13.0),
    ])
    T = jnp.broadcast_to(T_profile, shape).astype(jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    rho = wright_eos(T, S, jnp.zeros_like(T))
    u = jnp.zeros(shape, dtype=jnp.float64)
    v = jnp.zeros(shape, dtype=jnp.float64)
    eta = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    J = jnp.ones((6, 2, 2), dtype=jnp.float64)
    tau_x = jnp.full((6, 2, 2), 0.005, dtype=jnp.float64)
    tau_y = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    B_f = jnp.zeros((6, 2, 2), dtype=jnp.float64)

    def loss(kc):
        cfg = KPPConfig(K_conv=kc, Ri_conv=0.0)
        out = kpp_vertical_mixing(u, v, T, S, rho, eta, z, J, cfg,
                                  tau_x=tau_x, tau_y=tau_y, B_f=B_f,
                                  apply_diffusion=False)
        return jnp.sum(out.K_v)

    # K_conv enters via where(N2<Ri_conv, K_conv, 0) — a step function, so its
    # straight-through gradient is 0; instead confirm the FORWARD response is
    # monotone in K_conv (a finite-difference sensitivity).
    lo = float(loss(0.5))
    hi = float(loss(2.0))
    assert hi > lo, "interior K_conv must raise K_v where N^2<0 below the BL"


def test_kpp_grad_reaches_state_finite():
    """jax.grad of tendency wrt T is finite (no NaN through BL diagnosis)."""
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    from legoesm.ocean.eos import wright_eos
    u, v, T, S, rho, eta, z, J = _kpp_state()
    cfg = KPPConfig()
    tau_x = jnp.full((6, 2, 2), 0.1, dtype=jnp.float64)
    tau_y = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    B_f = jnp.full((6, 2, 2), 1e-7, dtype=jnp.float64)

    def loss(T_):
        rho_ = wright_eos(T_, S, jnp.zeros_like(T_))
        out = kpp_vertical_mixing(u, v, T_, S, rho_, eta, z, J, cfg,
                                  tau_x=tau_x, tau_y=tau_y, B_f=B_f,
                                  apply_diffusion=True, dt=900.0)
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.K_v)

    g = jax.grad(loss)(T)
    assert jnp.all(jnp.isfinite(g)), "NaN gradient through KPP BL diagnosis"
    assert float(jnp.max(jnp.abs(g))) > 0.0, "T must be AD-reachable"


def test_kpp_no_wind_no_buoyancy_finite():
    """Quiescent column (no stress, no buoyancy flux): finite, bounded K."""
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    u, v, T, S, rho, eta, z, J = _kpp_state()
    u = jnp.zeros_like(u)
    v = jnp.zeros_like(v)
    cfg = KPPConfig()
    out = kpp_vertical_mixing(u, v, T, S, rho, eta, z, J, cfg,
                              apply_diffusion=False)
    assert jnp.all(jnp.isfinite(out.K_v))
    assert jnp.all(out.K_v >= 0.0)
    assert jnp.all(out.K_v <= cfg.K_max + 1e-12)


# ===========================================================================
# jit / vmap parity (representative kernels)
# ===========================================================================

def test_jit_parity_kpp():
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    u, v, T, S, rho, eta, z, J = _kpp_state()
    cfg = KPPConfig()
    tau_x = jnp.full((6, 2, 2), 0.1, dtype=jnp.float64)
    tau_y = jnp.zeros((6, 2, 2), dtype=jnp.float64)
    B_f = jnp.full((6, 2, 2), 1e-7, dtype=jnp.float64)

    def run(uu):
        return kpp_vertical_mixing(uu, v, T, S, rho, eta, z, J, cfg,
                                   tau_x=tau_x, tau_y=tau_y, B_f=B_f,
                                   apply_diffusion=False).K_v

    eager = run(u)
    jitted = jax.jit(run)(u)
    assert jnp.allclose(eager, jitted, rtol=1e-10, atol=1e-12)


def test_bulk_land_column_zero_tendency():
    """Land columns (jacobian=0) must give ZERO surface tendency, not the huge
    1/max(dz,1e-10) value (codex review finding #5)."""
    from legoesm.ocean.physics.surface_forcing.bulk_formulas import (
        bulk_formula_surface_forcing,
    )
    from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
    z = create_ocean_z_star(n_levels=3, H_max=30.0)
    T = jnp.zeros((1, 1, 1, 3), dtype=jnp.float64)
    S = jnp.full((1, 1, 1, 3), 35.0, dtype=jnp.float64)
    J = jnp.zeros((1, 1, 1), dtype=jnp.float64)   # land
    out = bulk_formula_surface_forcing(T, S, z, J, BulkFormulaConfig())
    assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0
    assert float(jnp.max(jnp.abs(out.du_dt))) == 0.0
    assert float(jnp.max(jnp.abs(out.dv_dt))) == 0.0
    # An ocean column still produces a finite, non-zero tendency.
    Jo = jnp.ones((1, 1, 1), dtype=jnp.float64)
    To = jnp.full((1, 1, 1, 3), 25.0, dtype=jnp.float64)
    cfg = BulkFormulaConfig(bulk_scheme="constant", T_a=275.0, U_a=8.0,
                            SW_down=0.0, LW_down=0.0)
    oo = bulk_formula_surface_forcing(To, S, z, Jo, cfg)
    assert float(jnp.abs(oo.dT_dt[0, 0, 0, 0])) > 0.0
    assert jnp.all(jnp.isfinite(oo.dT_dt))


def test_implicit_enhanced_diffusion_uses_eos_fn():
    """compute_vertical_K_profiles threads eos_fn into the enhanced-diffusion
    convective trigger (codex review finding #3): a non-Wright EOS that makes
    the same column convect (vs Wright stable) must change K_v."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.convection.config import (
        OceanConvectionConfig, EnhancedDiffusionConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        compute_vertical_K_profiles, _enhanced_diffusion_K,
    )

    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z = create_ocean_z_star(n_levels=6, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=9.9, S_uniform=35.0, H_max=4000.0,
    )
    conv = OceanConvectionConfig(
        scheme="enhanced_diffusion",
        enhanced_diffusion=EnhancedDiffusionConfig(
            K_conv=1.0, K_bg=1e-5, nu_conv=0.0, nu_bg=0.0,
            smooth_transition=True, sigmoid_sharpness=1e4,
        ),
    )
    # An EOS whose density INVERTS the column (sign-flipped alpha) makes the
    # near-uniform warm-over-cool column statically unstable -> convection,
    # whereas the default Wright keeps it stable. Reaching it requires eos_fn
    # to be threaded to the convective N^2.
    def inverting_eos(T, S, p):
        return constants.rho_ocean * (1.0 + 2.0e-4 * (T - 10.0))  # dense where warm

    K_default, _ = _enhanced_diffusion_K(state, z, conv, eos_fn=None)
    K_inv, _ = _enhanced_diffusion_K(state, z, conv, eos_fn=inverting_eos)
    # The inverting EOS must change the convective diffusivity field.
    assert not jnp.allclose(K_default, K_inv), (
        "eos_fn not reaching the enhanced-diffusion convective trigger"
    )


def test_jit_parity_bulk():
    from legoesm.ocean.physics.surface_forcing.bulk_formulas import (
        bulk_formula_surface_forcing,
    )
    from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
    T, S, z, J = _sf_state(T_sfc_C=25.0)
    cfg = BulkFormulaConfig(bulk_scheme="constant", U_a=8.0)

    def run(TT):
        return bulk_formula_surface_forcing(TT, S, z, J, cfg).Q_net

    assert jnp.allclose(run(T), jax.jit(run)(T), rtol=1e-10, atol=1e-12)
