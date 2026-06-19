"""Direct unit tests for the CATKE column physics (Wagner et al. 2025).

Exercises the pure diffusivity / dissipation / surface-flux helpers in
``ocean.physics.vertical_mixing.catke`` on synthetic columns: shapes + signs,
the idealized behaviours that define CATKE (a stably stratified column gives
near-background mixing; a convective column Jb>0 & N2<0 gives strongly enhanced
mixing via the dynamic Deardorff length), and differentiability.  No grid / EOS
coupling — the prognostic TKE solve is tested separately once it is wired.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.physics.vertical_mixing.config import CATKEConfig
from legoesm.ocean.physics.vertical_mixing import catke as C

_NI = 10


def _column(N2_val, e_val, S2_val=1.0e-4, Jb_val=0.0):
    """Build a uniform synthetic interface column."""
    N2 = jnp.full((_NI,), N2_val)
    N2_above = N2  # uniform column
    S2 = jnp.full((_NI,), S2_val)
    e = jnp.full((_NI,), e_val)
    depth = jnp.linspace(5.0, 95.0, _NI)        # below surface [m]
    hab = jnp.linspace(95.0, 5.0, _NI)          # above bottom [m]
    H = 100.0
    Jb = Jb_val
    return e, N2, N2_above, S2, depth, hab, H, Jb


def test_diffusivities_shapes_signs_finite():
    cfg = CATKEConfig()
    e_val, H = 1.0e-3, 100.0
    e, N2, N2a, S2, depth, hab, _, Jb = _column(1.0e-4, e_val)
    K_u, K_c, K_e = C.catke_diffusivities(e, N2, N2a, S2, depth, hab, H, Jb, cfg)
    k_bound = H * (e_val ** 0.5) + 1e-9          # K = l*w* <= H*sqrt(e); caps=inf
    for K in (K_u, K_c, K_e):
        assert K.shape == (_NI,)
        assert jnp.all(jnp.isfinite(K))
        assert jnp.all(K >= 0.0)
        assert jnp.all(K <= k_bound)


def test_convective_column_enhances_mixing():
    """Convective column (Jb>0, N2<0) -> strongly enhanced K vs a stably
    stratified column (Deardorff dynamic convective length)."""
    cfg = CATKEConfig()
    # Convective: unstable stratification + positive surface buoyancy flux,
    # buoyancy-dominated (weak shear so the sheared-convection factor stays
    # positive — strong shear correctly suppresses the convective length).
    args_conv = _column(N2_val=-1.0e-5, e_val=1.0e-2, S2_val=1.0e-9,
                        Jb_val=1.0e-7)
    _, K_c_conv, _ = C.catke_diffusivities(*args_conv[:8], cfg)
    # Stratified: stable, quiescent.
    args_strat = _column(N2_val=1.0e-3, e_val=1.0e-9, Jb_val=0.0)
    _, K_c_strat, _ = C.catke_diffusivities(*args_strat[:8], cfg)
    assert jnp.max(K_c_conv) > 1.0                  # large (toward the cap)
    assert jnp.max(K_c_strat) < 1.0e-2              # near background
    assert jnp.mean(K_c_conv) > 1.0e3 * jnp.mean(K_c_strat)


def test_high_ri_background_mixing():
    """Strongly stratified, weak shear (high Ri) -> small background K."""
    cfg = CATKEConfig()
    e, N2, N2a, S2, depth, hab, H, Jb = _column(
        N2_val=1.0e-2, e_val=1.0e-9, S2_val=1.0e-8)
    _, K_c, _ = C.catke_diffusivities(e, N2, N2a, S2, depth, hab, H, Jb, cfg)
    assert jnp.all(K_c >= 0.0)
    assert jnp.max(K_c) < 1.0e-3


def test_dissipation_rate_signs():
    cfg = CATKEConfig()
    e, N2, N2a, S2, depth, hab, H, Jb = _column(1.0e-3, 1.0e-3)
    omega = C.catke_dissipation_rate(e, N2, N2a, S2, depth, hab, H, Jb, cfg)
    assert jnp.all(jnp.isfinite(omega))
    assert jnp.all(omega >= 0.0)
    # Negative TKE -> numerical damping rate 1/negative_tke_damping_time_s.
    e_neg = -jnp.abs(e)
    omega_neg = C.catke_dissipation_rate(
        e_neg, N2, N2a, S2, depth, hab, H, Jb, cfg)
    assert jnp.allclose(omega_neg, 1.0 / cfg.negative_tke_damping_time_s)


def test_surface_tke_flux_downward():
    cfg = CATKEConfig()
    q = C.catke_surface_tke_flux(u_star=0.01, w_convective_cubed=1.0e-6, cfg=cfg)
    assert q <= 0.0                                 # downward (into the column)
    # Zero forcing -> zero flux.
    assert C.catke_surface_tke_flux(0.0, 0.0, cfg) == 0.0


def test_stability_function_regimes():
    cfg = CATKEConfig()
    # Negative Ri -> c_un; very high Ri -> c_hi; mid -> between c_lo and c_hi.
    s_neg = C._stability_function(jnp.array(-1.0), cfg.c_un_c, cfg.c_lo_c,
                                  cfg.c_hi_c, cfg.c_ri_lower, cfg.c_ri_width)
    s_hi = C._stability_function(jnp.array(100.0), cfg.c_un_c, cfg.c_lo_c,
                                 cfg.c_hi_c, cfg.c_ri_lower, cfg.c_ri_width)
    assert jnp.allclose(s_neg, cfg.c_un_c)
    assert jnp.allclose(s_hi, cfg.c_hi_c)


def test_differentiable_wrt_e_and_N2():
    cfg = CATKEConfig()
    e, N2, N2a, S2, depth, hab, H, Jb = _column(1.0e-4, 1.0e-3, Jb_val=1.0e-8)

    def loss(e_in, N2_in):
        K_u, K_c, K_e = C.catke_diffusivities(
            e_in, N2_in, N2_in, S2, depth, hab, H, Jb, cfg)
        return jnp.sum(K_u ** 2 + K_c ** 2 + K_e ** 2)

    ge, gN2 = jax.grad(loss, argnums=(0, 1))(e, N2)
    assert ge.shape == e.shape and gN2.shape == N2.shape
    assert jnp.all(jnp.isfinite(ge)) and jnp.all(jnp.isfinite(gN2))


def test_differentiable_through_convective_branches():
    """Gradients stay finite through the convective AND entrainment branches
    (the masked where-branch denominator grad-trap fix)."""
    cfg = CATKEConfig()
    S2 = jnp.full((_NI,), 1.0e-9)
    depth = jnp.linspace(5.0, 95.0, _NI)
    hab = jnp.linspace(95.0, 5.0, _NI)
    H, Jb = 100.0, 1.0e-7
    # Mixed column: unstable at top (convecting), stable below an unstable cell
    # (entraining) — exercises both masked branches.
    N2 = jnp.linspace(-1.0e-5, 1.0e-4, _NI)
    N2_above = jnp.concatenate([N2[:1], N2[:-1]])   # shallower neighbour

    def loss(e_in, N2_in):
        N2a = jnp.concatenate([N2_in[:1], N2_in[:-1]])
        K_u, K_c, K_e = C.catke_diffusivities(
            e_in, N2_in, N2a, S2, depth, hab, H, Jb, cfg)
        omega = C.catke_dissipation_rate(
            e_in, N2_in, N2a, S2, depth, hab, H, Jb, cfg)
        return jnp.sum(K_c ** 2 + K_u ** 2 + K_e ** 2) + jnp.sum(omega ** 2)

    e = jnp.full((_NI,), 1.0e-2)
    ge, gN2 = jax.grad(loss, argnums=(0, 1))(e, N2)
    assert jnp.all(jnp.isfinite(ge)), "non-finite grad wrt e (masked-branch trap)"
    assert jnp.all(jnp.isfinite(gN2)), "non-finite grad wrt N2 (masked-branch trap)"


# ---------------------------------------------------------------------------
# Prognostic column solve (catke_vertical_mixing)
# ---------------------------------------------------------------------------


def _solver_column(rho_top, rho_bot, Jb=0.0, n=11):
    """Uniform column for catke_vertical_mixing (cell-centre fields length n;
    interface arrays length n-1). rho_top < rho_bot => stable stratification."""
    import legoesm.constants as _const
    rho = jnp.linspace(rho_top, rho_bot, n)
    u = jnp.linspace(0.0, 0.2, n)            # weak shear
    v = jnp.zeros((n,))
    T = jnp.linspace(15.0, 5.0, n)
    S = jnp.full((n,), 35.0)
    dz_half = jnp.full((n - 1,), 10.0)
    depth = jnp.linspace(5.0, 95.0, n - 1)
    hab = jnp.linspace(95.0, 5.0, n - 1)
    H = 100.0
    return dict(u_cell=u, v_cell=v, T_cell=T, S_cell=S, rho_cell=rho,
                dz_half=dz_half, depth_iface=depth,
                height_above_bottom_iface=hab, H_col=H, Jb=Jb,
                u_star=0.01, dt=3600.0, rho_0=float(_const.rho_ocean))


def test_vertical_mixing_advances_tke():
    cfg = CATKEConfig()
    col = _solver_column(1025.0, 1027.0)     # stable
    K_M, K_H, e_new = C.catke_vertical_mixing(tke_old=None, cfg=cfg, **col)
    assert K_M.shape == (10,) and K_H.shape == (10,) and e_new.shape == (10,)
    for arr in (K_M, K_H, e_new):
        assert jnp.all(jnp.isfinite(arr))
        assert jnp.all(arr >= 0.0)
    assert jnp.all(e_new >= cfg.minimum_tke - 1e-30)


def test_vertical_mixing_convective_vs_stratified():
    """Convective column (denser on top, Jb>0) -> larger TKE + viscosity than a
    stably stratified, unforced column."""
    cfg = CATKEConfig()
    # Seed a realistic TKE so w*=sqrt(e) is non-trivial (at the 1e-9 cold-start
    # floor the Deardorff length ~w*^3 is ~0 and convection can't kick in one
    # step — the chicken-and-egg of a prognostic closure).
    seed = jnp.full((10,), 1.0e-4)
    conv = _solver_column(1027.0, 1025.0, Jb=1.0e-7)   # unstable, forced
    strat = _solver_column(1025.0, 1027.0, Jb=0.0)     # stable, unforced
    K_M_conv, _, e_conv = C.catke_vertical_mixing(tke_old=seed, cfg=cfg, **conv)
    K_M_strat, _, e_strat = C.catke_vertical_mixing(tke_old=seed, cfg=cfg, **strat)
    assert jnp.max(e_conv) > jnp.max(e_strat)
    assert jnp.max(K_M_conv) > jnp.max(K_M_strat)


def test_vertical_mixing_differentiable():
    cfg = CATKEConfig()
    col = _solver_column(1027.0, 1025.0, Jb=1.0e-7)
    rho0 = col.pop("rho_cell")
    e0 = jnp.full((10,), 1.0e-3)

    def loss(rho_in, e_in):
        K_M, K_H, e_new = C.catke_vertical_mixing(
            rho_cell=rho_in, tke_old=e_in, cfg=cfg, **col)
        return jnp.sum(K_M ** 2 + K_H ** 2 + e_new ** 2)

    grho, ge = jax.grad(loss, argnums=(0, 1))(rho0, e0)
    assert jnp.all(jnp.isfinite(grho)) and jnp.all(jnp.isfinite(ge))
