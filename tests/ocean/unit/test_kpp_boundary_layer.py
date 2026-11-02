"""KPP boundary-layer depth: LMD94 V_t^2 forcing dependence and stability.

Run with JAX_ENABLE_X64=1.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    LinearEOSConfig,
    compute_buoyancy_frequency,
    locally_referenced_buoyancy_frequency,
    make_eos_fn,
    rho_0,
    wright_eos,
)
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.physics.vertical_mixing.kpp import (
    kpp_surface_fluxes,
    kpp_turbulent_velocity_scale,
    kpp_vertical_mixing,
)
from legoesm.ocean.vertical import create_ocean_z_star

CFG = KPPConfig()
LIN = LinearEOSConfig()
LINEAR_EOS = make_eos_fn("linear", LIN)


def _uniform_column(nlev=200, H=200.0):
    z = create_ocean_z_star(n_levels=nlev, H_max=H, dz_surface=H / nlev, dz_deep=H / nlev)
    depth = -z.z_full_ref
    return z, depth


def _linear_T_column(N, nlev=200, H=200.0):
    """T decreasing linearly with depth so that N^2 = g alpha_T Gamma (linear EOS)."""
    z, depth = _uniform_column(nlev, H)
    gamma = N**2 / (constants.g * LIN.alpha_T)
    T = 15.0 - gamma * depth
    S = jnp.full_like(T, LIN.S_ref)
    return z, T, S


def _h_bl(z, T, S, tau=0.1, B_f=0.0, eos_fn=LINEAR_EOS, u=None, v=None):
    zero = jnp.zeros_like(T)
    out = kpp_vertical_mixing(
        zero if u is None else u, zero if v is None else v, T, S,
        jnp.zeros(()), z, jnp.ones(()), CFG,
        tau_x=jnp.asarray(tau), tau_y=jnp.zeros(()),
        B_f=jnp.asarray(B_f), apply_diffusion=False, eos_fn=eos_fn,
    )
    return out.h_bl


def _h_analytic(tau, N):
    """Ri_b crossing for linear N, no shear, B_f = 0: Ri_b(d) = N d / (Cv kappa u* V_tc)."""
    u_star = np.sqrt(tau / rho_0)
    V_tc = np.sqrt(-CFG.beta_T / (CFG.c_s * CFG.epsilon_lmd)) / (CFG.Ri_crit * CFG.kappa_vk**2)
    return CFG.Ri_crit * CFG.Cv * CFG.kappa_vk * u_star * V_tc / N


@pytest.mark.parametrize("tau", [0.4, 1.6])
def test_h_bl_matches_analytic_no_shear_neutral_flux(tau):
    N = 1e-3
    z, T, S = _linear_T_column(N)
    h = float(_h_bl(z, T, S, tau=tau))
    h_ref = _h_analytic(tau, N)
    assert h_ref > 5.0  # well below / above grid scale
    assert abs(h - h_ref) < 1.5, (h, h_ref)


def test_h_bl_scales_with_u_star():
    N = 1e-3
    z, T, S = _linear_T_column(N)
    h1 = float(_h_bl(z, T, S, tau=0.4))
    h4 = float(_h_bl(z, T, S, tau=1.6))  # u* doubles
    assert h4 / h1 == pytest.approx(2.0, rel=0.1)


def test_h_bl_responds_to_buoyancy_flux():
    N = 3e-3
    z, T, S = _linear_T_column(N)
    h_heat = float(_h_bl(z, T, S, tau=0.1, B_f=-1e-7))
    h_neut = float(_h_bl(z, T, S, tau=0.1, B_f=0.0))
    h_cool = float(_h_bl(z, T, S, tau=0.1, B_f=1e-7))
    assert h_cool > h_neut > h_heat


def test_h_bl_deepens_with_resolved_shear():
    N = 3e-3
    z, T, S = _linear_T_column(N)
    depth = -z.z_full_ref
    u_slab = jnp.where(depth < 20.0, 0.2, 0.0)  # 0.2 m/s slab over 20 m
    h0 = float(_h_bl(z, T, S, tau=0.1))
    h1 = float(_h_bl(z, T, S, tau=0.1, u=u_slab))
    assert h1 > h0


def test_locally_referenced_N2_removes_compressibility():
    z, depth = _uniform_column(nlev=50, H=1000.0)
    T = jnp.full_like(depth, 5.0)
    S = jnp.full_like(depth, 35.0)
    z_half = 0.5 * (depth[:-1] + depth[1:])
    N2_local = locally_referenced_buoyancy_frequency(
        T, S, rho_0 * constants.g * z_half, z.dz_half_ref)
    rho_insitu = wright_eos(T, S, rho_0 * constants.g * depth)
    N2_insitu = compute_buoyancy_frequency(rho_insitu, z.dz_ref, jnp.ones(()))
    assert float(jnp.max(jnp.abs(N2_local))) < 1e-10
    assert float(jnp.min(N2_insitu)) > 1e-5  # the bias being removed


def test_h_bl_well_mixed_column_cooling_reaches_bottom():
    z, depth = _uniform_column(nlev=50, H=200.0)
    T = jnp.full_like(depth, 10.0)
    S = jnp.full_like(depth, 35.0)
    h = float(_h_bl(z, T, S, tau=0.05, B_f=1e-7, eos_fn=wright_eos))
    assert h > 0.8 * 200.0


def test_w_s_continuous_at_convective_switch():
    u_star = jnp.asarray(0.01)
    d = jnp.asarray(10.0)
    # X = kappa d B_f = -zeta_s u*^3 at the switch
    B_switch = -CFG.zeta_s * u_star**3 / (CFG.kappa_vk * d)
    below = kpp_turbulent_velocity_scale(d, u_star, B_switch * (1 - 1e-6), CFG)
    above = kpp_turbulent_velocity_scale(d, u_star, B_switch * (1 + 1e-6), CFG)
    assert float(above) == pytest.approx(float(below), rel=1e-3)


def test_w_s_neutral_and_free_convection_limits():
    d = jnp.asarray(10.0)
    u_star = jnp.asarray(0.01)
    w0 = kpp_turbulent_velocity_scale(d, u_star, jnp.asarray(0.0), CFG)
    assert float(w0) == pytest.approx(CFG.kappa_vk * 0.01, rel=1e-12)
    # u* -> 0: w_s -> kappa (c_s kappa d B_f)^(1/3), finite
    B_f = 1e-7
    w_free = kpp_turbulent_velocity_scale(d, jnp.asarray(0.0), jnp.asarray(B_f), CFG)
    ref = CFG.kappa_vk * (CFG.c_s * CFG.kappa_vk * 10.0 * B_f) ** (1.0 / 3.0)
    assert float(w_free) == pytest.approx(ref, rel=1e-6)
    # stable forcing suppresses w_s
    w_stab = kpp_turbulent_velocity_scale(d, u_star, jnp.asarray(-1e-7), CFG)
    assert float(w_stab) < float(w0)


def test_surface_flux_signs():
    T = jnp.asarray(15.0)
    S = jnp.asarray(35.0)
    B_cool, Q_T, _ = kpp_surface_fluxes(T, S, q_net=jnp.asarray(-100.0))
    assert float(B_cool) > 0.0 and float(Q_T) < 0.0
    B_fresh, _, Q_S = kpp_surface_fluxes(T, S, freshwater=jnp.asarray(1e-4))
    assert float(B_fresh) < 0.0 and float(Q_S) < 0.0
    B_none, Q_T0, Q_S0 = kpp_surface_fluxes(T, S)
    assert B_none is None and Q_T0 is None and Q_S0 is None


def test_h_bl_gradient_wrt_wind_stress():
    N = 1e-3
    z, T, S = _linear_T_column(N)
    dh_dtau = jax.grad(lambda t: _h_bl(z, T, S, tau=t))(jnp.asarray(0.4))
    assert bool(jnp.isfinite(dh_dtau))
    assert float(dh_dtau) > 0.0
    eps = 1e-4
    fd = (_h_bl(z, T, S, tau=0.4 + eps) - _h_bl(z, T, S, tau=0.4 - eps)) / (2 * eps)
    assert float(dh_dtau) == pytest.approx(float(fd), rel=1e-3)
