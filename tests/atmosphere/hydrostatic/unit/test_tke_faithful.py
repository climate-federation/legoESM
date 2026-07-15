"""Scheme-level oracle-faithfulness tests for the prognostic-TKE turbulence closure.

This is a MY2.5-INSPIRED k-l closure, NOT literal Mellor-Yamada level 2.5: it carries the
prognostic TKE budget but REPLACES MY2.5's defining algebraic stability functions
Sm(GM, GH), Sh(GM, GH) (both depend on shear GM and buoyancy GH) with constant Ck, Pr_t.

MY-SHAPED forms (the FORM is MY2.5; the constants/numerics baked into each pin are
DEPARTURES, listed below):
  * the k-l eddy viscosity has MY's Km = l·q·Sm SHAPE — pinned as Km = Ck·l·√max(TKE,
    tke_min) vs an independent reimpl (shared Blackadar mixing_length; a normal and a
    sub-floor TKE case exercise the floor). The CONSTANT Ck and the tke_min floor are
    departures/numerics baked into this pin;
  * the dissipation has MY's ε = q³/(B1·l) SHAPE (ε = Ce·e^{3/2}/l) — isolated in a
    neutral/no-shear/uniform-TKE column (diffusion a no-op) as tke_new = tke/(1 + dt·Ce·
    √tke/l). The semi-implicit linearization is numerics;
  * the budget SIGNS — shear production Km·S² > 0 (source), buoyancy −Kh·N² (sink if
    stable, source if unstable).

DEPARTURES / NUMERICS (locked + labeled):
  * CONSTANT Ck (no Sm(GM, GH)): the PRE-update Km is INDEPENDENT of N² at fixed (e, l) —
    a stable vs an unstable column ⇒ BIT-IDENTICAL Km, where MY2.5's Sm would differ.
    Key inspired-vs-faithful discriminator. (N² still changes later Km INDIRECTLY, via
    buoyancy's effect on e.)
  * CONSTANT Pr_t (no Sh/Sm): Kh/Km = 1/Pr_t ≈ 3.03 fixed across stratifications; Pr_t=0.33
    is the scheme's own value, NOT the MY neutral Sh/Sm ≈ 1.06;
  * the tke_min floor and semi-implicit dissipation linearization are NUMERICS
    (exercised in the Km and dissipation pins above);
  * unknown turbulence scheme raises ValueError (dispatch hardening; sanity, not a pin).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    mixing_length,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.config import TKEConfig, TurbulenceConfig
from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn
from legoesm.atmosphere.physics.turbulence.tke import tke_turbulence
from legoesm.thermo import saturation_mixing_ratio

from legoesm import constants


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _grid(ncol=2, nlev=6):
    """A FIXED (p, z) grid, decoupled from the test θ profile.

    z is set directly (linear top→surface) so that changing θ (hence N²) between two
    states leaves z_full — and therefore the Blackadar length and Km — bit-identical.
    Level 0 is the top, nlev-1 the surface.
    """
    p_half = jnp.broadcast_to(
        jnp.linspace(7.0e4, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_half = jnp.broadcast_to(
        jnp.linspace(1500.0, 0.0, nlev + 1)[None, :], (ncol, nlev + 1)
    )
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    return p_full, p_half, z_full, z_half


def _state(grid, theta_top=300.0, theta_sfc=300.0, u_top=0.0, u_sfc=0.0, tke_val=0.5):
    """Build (u, v, T, q_v, tke, rho) on a fixed grid; θ linear top→surface, u linear."""
    p_full, p_half, z_full, z_half = grid
    ncol, nlev = p_full.shape
    theta = jnp.broadcast_to(
        jnp.linspace(theta_top, theta_sfc, nlev)[None, :], (ncol, nlev)
    )
    T = theta * (p_full / constants.p_ref) ** constants.kappa
    rho = p_full / (constants.R_d * T)
    u = jnp.broadcast_to(jnp.linspace(u_top, u_sfc, nlev)[None, :], (ncol, nlev))
    v = jnp.zeros((ncol, nlev))
    q_v = jnp.zeros((ncol, nlev))
    tke = jnp.full((ncol, nlev), tke_val)
    return u, v, T, q_v, tke, rho


def _run(cfg, grid=None, dt=300.0, **state_kw):
    grid = grid if grid is not None else _grid()
    p_full, p_half, z_full, z_half = grid
    u, v, T, q_v, tke, rho = _state(grid, **state_kw)
    T_sfc = jnp.full((u.shape[0],), float(np.asarray(T)[0, -1]))
    q_sfc = saturation_mixing_ratio(T_sfc, p_half[:, -1])
    out, tke_new = tke_turbulence(
        u, v, T, q_v, tke, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho, dt, cfg,
    )
    return out, tke_new, (u, v, T, q_v, tke, p_full, p_half, z_full, z_half, rho)


def _N2_half0(inp):
    """Brunt-Väisälä N² at the top half-interface (col 0), mirroring production's N²."""
    u, v, T, q_v, tke, p_full, p_half, z_full, z_half, rho = inp
    dz = np.clip(abs(np.asarray(z_full)[0, 0] - np.asarray(z_full)[0, 1]), 1.0, None)
    exner = 1.0 / np.asarray(exner_function(p_full))[0]
    thv = np.asarray(virtual_temperature(T, q_v))[0] * exner
    thv_bar = 0.5 * (thv[0] + thv[1])
    dthv_dz = (thv[0] - thv[1]) / dz
    return float(buoyancy_coefficient(jnp.clip(jnp.array(thv_bar), 1.0, None))) * dthv_dz


# ===========================================================================
# MY-SHAPED forms (shape faithful; constants/numerics are departures)
# ===========================================================================
def test_tke_config_defaults():
    """Canary: the inspired k-l constants (Ck, Ce, Pr_t) that stand in for MY2.5 forms."""
    c = TKEConfig()
    assert c.Ck == 0.1          # constant Km coefficient (replaces Sm(GM, GH))
    assert c.Ce == 0.19         # dissipation coeff; MY 2^{3/2}/B1 ≈ 0.17
    assert c.Pr_t == 0.33       # scheme's fixed Kh/Km≈3 (NOT the MY neutral Sh/Sm≈1.06)
    assert c.l_mix_max == 100.0  # fixed Blackadar asymptote [m] (vs MY's integral l0)
    assert c.tke_min == 1e-6    # numerics floor


def test_tke_Km_kl_form_pinned():
    """Km = Ck·l·√max(TKE, tke_min) pinned exactly vs an independent reimpl (shared l).

    Two cases exercise the ``max(·, tke_min)`` floor: a normal TKE (0.5, above the floor)
    and a SUB-FLOOR TKE (1e-9 < tke_min=1e-6), where Km must use √tke_min not √TKE.
    """
    cfg = TKEConfig()
    for tke_val in (0.5, 1e-9):
        out, _, inp = _run(cfg, theta_top=300.0, theta_sfc=300.0, tke_val=tke_val)
        z_full = inp[7]
        Km = np.asarray(out.Km)
        l_mix = np.asarray(mixing_length(z_full, cfg.l_mix_max))
        Km_exp = cfg.Ck * l_mix * np.sqrt(np.maximum(tke_val, cfg.tke_min))
        assert np.all(Km > 0.0)
        assert np.allclose(Km, Km_exp, rtol=1e-12, atol=0.0)
    # the sub-floor case genuinely hit the floor (√tke_min ≠ √1e-9)
    assert cfg.tke_min > 1e-9


def test_tke_prandtl_closure_nonunit_Pr():
    """Kh = Km/Pr_t with the default NON-UNIT Pr_t=0.33 (not the trivial Kh==Km)."""
    cfg = TKEConfig()
    assert cfg.Pr_t != 1.0
    out, _, _ = _run(cfg, tke_val=0.5)
    Km = np.asarray(out.Km)
    Kh = np.asarray(out.Kh)
    assert np.all(Km > 0.0)
    assert not np.allclose(Kh, Km)                       # Pr_t≠1 ⇒ Kh≠Km
    assert np.allclose(Kh, Km / cfg.Pr_t, rtol=1e-12, atol=0.0)


def test_tke_dissipation_semi_implicit_form():
    """ε = Ce·e^{3/2}/l: neutral, no-shear, uniform-TKE ⇒ tke_new = tke/(1 + dt·Ce·√tke/l).

    With θ constant (N²=0) and u constant (S²=0), production and buoyancy vanish; a UNIFORM
    TKE field makes the TKE diffusion a numerical no-op (~1e-14), so the semi-implicit
    dissipation update is isolated and pinned against the closed form (MY ε = q³/(B1·l)).
    """
    cfg = TKEConfig()
    dt = 300.0
    tke_val = 0.5
    _, tke_new, inp = _run(cfg, theta_top=300.0, theta_sfc=300.0, u_top=0.0, u_sfc=0.0,
                           tke_val=tke_val, dt=dt)
    z_full = inp[7]
    l_safe = np.clip(np.asarray(mixing_length(z_full, cfg.l_mix_max)), 1.0, None)
    diss = cfg.Ce * np.sqrt(tke_val) / l_safe
    tke_exp = tke_val / (1.0 + dt * diss)                # tke_diffused == tke (no-op)
    assert np.allclose(np.asarray(tke_new), tke_exp, rtol=1e-9, atol=0.0)


def test_tke_shear_production_raises_tke():
    """Shear production P = Km·S² > 0: a sheared neutral column ends above the no-shear one."""
    cfg = TKEConfig()
    grid = _grid()
    _, tke_shear, _ = _run(cfg, grid=grid, theta_top=300.0, theta_sfc=300.0,
                           u_top=30.0, u_sfc=0.0, tke_val=0.5)
    _, tke_calm, _ = _run(cfg, grid=grid, theta_top=300.0, theta_sfc=300.0,
                          u_top=0.0, u_sfc=0.0, tke_val=0.5)
    # interior levels gain TKE from shear production
    assert np.all(np.asarray(tke_shear)[:, 1:-1] > np.asarray(tke_calm)[:, 1:-1])


def test_tke_buoyancy_sign():
    """Buoyancy B = −Kh·N²: stable stratification is a SINK, unstable a SOURCE (vs neutral)."""
    cfg = TKEConfig()
    grid = _grid()
    # stable: θ increases upward (warm aloft) ⇒ N²>0 ⇒ B<0
    _, tke_stable, _ = _run(cfg, grid=grid, theta_top=310.0, theta_sfc=300.0, tke_val=0.5)
    _, tke_neutral, _ = _run(cfg, grid=grid, theta_top=300.0, theta_sfc=300.0, tke_val=0.5)
    # unstable: θ decreases upward (warm surface) ⇒ N²<0 ⇒ B>0
    _, tke_unstable, _ = _run(cfg, grid=grid, theta_top=298.0, theta_sfc=302.0, tke_val=0.5)
    assert np.all(np.asarray(tke_stable)[:, 1:-1] < np.asarray(tke_neutral)[:, 1:-1])
    assert np.all(np.asarray(tke_unstable)[:, 1:-1] > np.asarray(tke_neutral)[:, 1:-1])


# ===========================================================================
# DEPARTURES / NUMERICS (constant Ck/Pr_t, floor, dispatch)
# ===========================================================================
def test_departure_tke_Km_independent_of_stratification():
    """The PRE-update Km is INDEPENDENT of N² at fixed (e, l) — key MY2.5 departure (no Sm).

    On a FIXED grid (z, l byte-identical) with the SAME TKE, a strongly STABLE (N²>0) and a
    strongly UNSTABLE (N²<0) column produce BIT-IDENTICAL diagnostic ``out.Km``. Real MY2.5
    would differ: Sm(GM, GH) with GH ∝ −l²N²/q² decreases with stability, so Km would drop
    in the stable column. (This pins the PRE-update Km only — over a prognostic run N² still
    changes later Km INDIRECTLY, via buoyancy's effect on e.)
    """
    cfg = TKEConfig()
    grid = _grid()
    out_stable, _, inp_stable = _run(cfg, grid=grid, theta_top=320.0, theta_sfc=300.0, tke_val=0.5)
    out_unstable, _, inp_unstable = _run(cfg, grid=grid, theta_top=290.0, theta_sfc=310.0)
    # non-vacuity: the stratifications genuinely have OPPOSITE-sign N²
    assert _N2_half0(inp_stable) > 0.0        # warm aloft ⇒ stable
    assert _N2_half0(inp_unstable) < 0.0      # warm surface ⇒ unstable
    # yet identical (z, TKE) ⇒ BIT-IDENTICAL diagnostic Km regardless of stratification
    assert np.array_equal(np.asarray(out_stable.Km), np.asarray(out_unstable.Km))


def test_departure_tke_constant_prandtl_across_stratifications():
    """Kh/Km = 1/Pr_t is CONSTANT across stable/neutral/unstable columns.

    MY2.5's Sh(GM, GH)/Sm(GM, GH) ratio is stability-dependent; here Kh = Km/Pr_t makes
    Kh/Km structurally fixed at 1/Pr_t ≈ 3.03 regardless of the imposed stratification.
    """
    cfg = TKEConfig()
    grid = _grid()
    # (θ_top, θ_sfc, expected N²-sign): stable N²>0, neutral N²≈0, unstable N²<0
    for theta_top, theta_sfc, n2_sign in ((320.0, 300.0, 1), (300.0, 300.0, 0), (290.0, 310.0, -1)):
        out, _, inp = _run(cfg, grid=grid, theta_top=theta_top, theta_sfc=theta_sfc,
                           u_top=20.0, u_sfc=0.0, tke_val=0.5)
        n2 = _N2_half0(inp)                              # verify the imposed stratification
        if n2_sign > 0:
            assert n2 > 1e-6
        elif n2_sign < 0:
            assert n2 < -1e-6
        else:
            assert abs(n2) < 1e-9                        # neutral
        Km = np.asarray(out.Km)
        Kh = np.asarray(out.Kh)
        assert np.allclose(Kh / Km, 1.0 / cfg.Pr_t, rtol=1e-12, atol=0.0)


def test_dispatch_unknown_turbulence_raises():
    """Dispatch hardening: an unknown turbulence scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown turbulence scheme"):
        get_turbulence_fn(TurbulenceConfig(scheme="not_a_scheme"))
