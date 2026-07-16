"""Scheme-level oracle-faithfulness tests for the Smagorinsky-Lilly turbulence closure.

This is a MODERN 1-D specialization of the Smagorinsky-Lilly idea, not the literal
published scheme: Smagorinsky (1963) used the HORIZONTAL deformation + a grid length
(k_s≈0.28). Lilly (1962) left the Ri dependence and K_h/K_m as undetermined functions
in his GENERAL theory, but his equilibrium EXPERIMENT already gives THIS √(1−Ri/Pr_t)
stability factor with K_h/K_m=1 and K_m→0 for Ri>1 — all reproduced (the √ ramp is
Lilly's own form, NOT a modern replacement; only the max/double-where guard + the
S²+1e-10 floor, the AD-safe numerics, are modern). The (C_s·l)²·|S| deformation core is
faithful to Smagorinsky's STRUCTURE;
C_s=0.2 and l_mix_max=100 are modern choices.

FAITHFUL core (Smagorinsky deformation structure):
  * K_m ∝ (C_s·l)²·|S|, the length²·deformation form (constants modernized) — pinned
    directly by the C_s² scaling and shear-linearity canaries.

ASSEMBLY / BEHAVIOR canaries (pin the FULL assembled expression, which ALSO includes
the modern Blackadar length, the Lilly-type √ cutoff and the AD floor — NOT purely the
faithful core):
  * the full K_m = (C_s·l)²·|S|·√(max(0,1−Ri/Pr_t)) pinned to ~1e-11 rtol (a closed-form
    match, not f64 machine precision) at a boundary half-interface (where
    Km_full==Km_half) against an ASSEMBLY-level reimpl that reuses only the shared
    mixing-length + thermodynamic helpers and reassembles the shear, Ri and the
    Smagorinsky combination independently (a partial oracle);
  * the shut-off threshold is EXACTLY Ri ≥ Pr_t (inclusive): ON just below, OFF exactly
    AT the cutoff (Ri==Pr_t via an exact buoy_arg==0 construction — Pr_t set to a
    bit-identical mirror of production's Ri, so Ri/Pr_t==1.0 by IEEE), OFF just above;
    plus the unstable (Ri<0) enhancement;
  * jax.grad is finite EXACTLY at Ri=Pr_t (buoy_arg==0 by construction) — the
    double-where guard's finite selected VJP; a bare √(max(·,0)) would NaN there.

LILLY-EXPERIMENT-CONSISTENT (matches Lilly's 1962 equilibrium experiment, not his
general theory — the √ stability factor is Lilly's OWN form, not a modern replacement):
  * the √(1−Ri/Pr_t) stability factor, the Ri≥Pr_t shut-off, and default Pr_t=1
    (= Lilly's experimental K_h/K_m=1); the constant-Prandtl closure K_h = K_m/Pr_t
    pinned with a NON-UNIT Pr_t (not trivial Kh==Km).

DEPARTURE / RE-TUNED (locked + labeled):
  * modern deformation constant C_s=0.2 (vs k_s≈0.28) and l_mix_max=100;
  * 1-D vertical-shear PROXY |S| = √((∂u/∂z)²+(∂v/∂z)²) CHOSEN as a substitute for
    Smagorinsky's horizontal deformation (a single column carries no horizontal strain)
    — K_m is APPROXIMATELY linear in the vertical-shear magnitude (the S²+1e-10 AD floor
    makes it only ~linear; it does not distinguish the modern 3-D-LES √(2·S_ij·S_ij) form);
  * the AD-safe NUMERICS (modern): the max(0,·) clamp, the double-where guard (finite
    reverse-mode VJP at the Ri=Pr_t kink, pinned exactly — a bare √(max(·,0)) would NaN
    there), and the S²+1e-10 shear floor;
  * unknown turbulence scheme raises ValueError (dispatch hardening; sanity, not a
    faithfulness pin).
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
from legoesm.atmosphere.physics.turbulence.config import (
    SmagorinskyConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn
from legoesm.atmosphere.physics.turbulence.smagorinsky import smagorinsky_turbulence
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


def _column(theta_top=300.6, theta_sfc=300.0, u_top=40.0, u_sfc=1.0, ncol=2, nlev=4):
    """A single column; θ near-neutral (top→surface) with a linear wind shear.

    Level 0 is the top, nlev-1 the surface. Defaults give Ri≪Pr_t at the interfaces
    (mixing on). Returns the full smagorinsky_turbulence input tuple.
    """
    p_half = jnp.broadcast_to(
        jnp.linspace(7.0e4, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    theta = jnp.broadcast_to(
        jnp.linspace(theta_top, theta_sfc, nlev)[None, :], (ncol, nlev)
    )
    T = theta * (p_full / constants.p_ref) ** constants.kappa
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None)))
    z_half = jnp.concatenate(
        [jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1], jnp.zeros((ncol, 1))], axis=1
    )
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    rho = p_full / (constants.R_d * T)
    u = jnp.broadcast_to(jnp.linspace(u_top, u_sfc, nlev)[None, :], (ncol, nlev))
    v = jnp.zeros((ncol, nlev))
    q_v = jnp.zeros((ncol, nlev))
    return u, v, T, q_v, p_full, p_half, z_full, z_half, rho


def _run(cfg, **col_kw):
    u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _column(**col_kw)
    T_sfc = jnp.full((u.shape[0],), float(np.asarray(T)[0, -1]))
    q_sfc = saturation_mixing_ratio(T_sfc, p_half[:, -1])
    out = smagorinsky_turbulence(
        u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho, 300.0, cfg,
    )
    return out, (u, v, T, q_v, p_full, p_half, z_full, z_half, rho)


def _assemble_Km(cfg, inputs, i):
    """Independent Smagorinsky assembly of K_m at interface i (col 0).

    Reuses only the shared mixing-length + thermodynamic helpers (mixing_length,
    virtual_temperature, exner_function, buoyancy_coefficient — covered by their own
    tests); the shear |S|, the Richardson number Ri and the
    (C_s·l)²·|S|·√(max(0,1−Ri/Pr_t)) combination are reassembled independently here
    (a partial oracle, not a fully independent end-to-end reimplementation).
    """
    u, v, T, q_v, p_full, p_half, z_full, z_half, rho = inputs
    zhi = 0.5 * (np.asarray(z_full)[0, :-1] + np.asarray(z_full)[0, 1:])
    l_mix = float(mixing_length(jnp.asarray(zhi), cfg.l_mix_max)[i])
    dzh = np.clip(abs(np.asarray(z_full)[0, i] - np.asarray(z_full)[0, i + 1]), 1.0, None)
    dudz = (np.asarray(u)[0, i] - np.asarray(u)[0, i + 1]) / dzh
    dvdz = (np.asarray(v)[0, i] - np.asarray(v)[0, i + 1]) / dzh
    S2 = dudz ** 2 + dvdz ** 2 + 1e-10
    S = np.sqrt(S2)
    exner = 1.0 / np.asarray(exner_function(p_full))[0]
    thv = np.asarray(virtual_temperature(T, q_v))[0] * exner
    thvbar = 0.5 * (thv[i] + thv[i + 1])
    dthvdz = (thv[i] - thv[i + 1]) / dzh
    N2 = float(buoyancy_coefficient(jnp.clip(jnp.array(thvbar), 1.0, None))) * dthvdz
    Ri = N2 / S2
    f_buoy = np.sqrt(max(0.0, 1.0 - Ri / cfg.Pr_t))
    return (cfg.C_s * l_mix) ** 2 * S * f_buoy, Ri


def _production_Ri(u, v, T, q_v, p_full, z_full):
    """jnp mirror of production's Ri = N²/S² (bit-identical, op-for-op to smagorinsky.py).

    Replays the EXACT JAX operations of ``smagorinsky_turbulence`` (same shared helpers,
    same order, same 1e-10 floor) so ``float(Ri[0, 0])`` equals production's internal Ri
    at that interface to the bit. Setting ``Pr_t = Ri[0, 0]`` then makes production's
    ``buoy_arg = 1 − Ri/Pr_t`` EXACTLY 0 there (IEEE ``x/x == 1.0``), landing an input
    precisely on the Lilly-type cutoff Ri == Pr_t.
    """
    dz_half = jnp.clip(jnp.abs(z_full[:, :-1] - z_full[:, 1:]), 1.0, None)
    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2 = du_dz ** 2 + dv_dz ** 2 + 1e-10
    exner_pref = 1.0 / exner_function(p_full)
    theta_v = virtual_temperature(T, q_v) * exner_pref
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    N2 = buoyancy_coefficient(jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz
    return N2 / S2


# ===========================================================================
# FAITHFUL core + assembly / behavior / Lilly-experiment canaries
# (only the C_s²-scaling test pins the truly-faithful (C_s·l)²·|S| core)
# ===========================================================================
def test_smagorinsky_config_defaults():
    """Canary: the single-column config defaults.

    C_s=0.2 and l_mix_max=100 m are MODERN atmospheric choices (Smagorinsky 1963
    reported k_s ≈ 0.28 with a horizontal-deformation/grid-length scheme); Pr_t=1 is
    LILLY-EXPERIMENT-CONSISTENT (it reproduces Lilly 1962's experimental K_h/K_m = 1,
    not a modern invention). Pinned here as config defaults (not oracle constants).
    """
    c = SmagorinskyConfig()
    assert c.C_s == 0.2          # modern C_s (vs Smagorinsky's k_s ≈ 0.28)
    assert c.Pr_t == 1.0         # = Lilly's experimental K_h/K_m = 1 (not a modern choice)
    assert c.l_mix_max == 100.0  # Blackadar asymptotic mixing length [m] (modern)


def test_smagorinsky_Km_full_expression_assembly_canary():
    """FULL K_m = (C_s·l)²·|S|·√(max(0,1−Ri/Pr_t)) pinned to ~1e-11 at the boundary interfaces.

    An ASSEMBLY canary of the whole expression — the faithful (C_s·l)²·|S| core AND the
    modern Blackadar length, the Lilly-type √ cutoff and the AD floor together — so it
    is a behavior pin, NOT a pure faithful-core pin (the C_s² and shear-linearity
    canaries pin the core directly). Km_full equals Km_half at the first and last
    interfaces (concat), so the closed form is pinned there against an ASSEMBLY-level
    reimpl that reuses only the shared mixing-length + thermodynamic helpers (covered by
    their own tests) and reassembles the shear, Ri and the (C_s·l)²·|S|·f_buoy
    combination independently (a partial oracle, not a fully independent end-to-end
    reimpl). The K_h = K_m/Pr_t closure is pinned separately with a non-unit Pr_t below.
    """
    cfg = SmagorinskyConfig()
    o, inputs = _run(cfg)
    Km = np.asarray(o.Km)[0]
    for full_idx, iface in ((0, 0), (-1, Km.shape[0] - 2)):
        Km_exp, Ri = _assemble_Km(cfg, inputs, iface)
        assert Ri < cfg.Pr_t                          # mixing on (below cutoff)
        assert Km_exp > 0.0
        assert Km[full_idx] == pytest.approx(Km_exp, rel=1e-11)


def test_smagorinsky_prandtl_closure_nonunit_Pr():
    """K_h = K_m/Pr_t pinned with a NON-UNIT Pr_t (not the trivial Kh==Km identity).

    With the default Pr_t=1 the closure is Kh==Km, which passes even if Kh were
    wrongly hard-coded to Km. Using Pr_t=0.7 makes Kh≠Km, so this genuinely pins the
    /Pr_t division elementwise.
    """
    cfg = SmagorinskyConfig(Pr_t=0.7)
    o, _ = _run(cfg)
    Km = np.asarray(o.Km)[0]
    Kh = np.asarray(o.Kh)[0]
    assert np.any(Km > 0.0)                            # non-vacuous (mixing is on)
    assert not np.allclose(Kh, Km)                     # Pr_t≠1 ⇒ Kh≠Km
    assert np.allclose(Kh, Km / cfg.Pr_t, rtol=1e-12, atol=0.0)


def test_smagorinsky_stable_cutoff_threshold_at_Pr_t():
    """The shut-off threshold is EXACTLY Ri ≥ Pr_t (inclusive), not merely 'very stable → off'.

    Fix a stable column, read its interface-0 Ri0 from ``_production_Ri`` — a jnp mirror
    written op-for-op against production (that SOURCE-LEVEL identity, not the assertions
    here, is what makes Pr_t=Ri0 land exactly on the cutoff), then sweep Pr_t across Ri0:
      * Pr_t = Ri0·(1+δ) ⇒ Ri0 < Pr_t ⇒ mixing ON  (Km[0] > 0);
      * Pr_t = Ri0        ⇒ Ri0 == Pr_t ⇒ mixing OFF (Km[0] == 0 — inclusive equality);
      * Pr_t = Ri0·(1−δ) ⇒ Ri0 > Pr_t ⇒ mixing OFF (Km[0] == 0).
    With δ=1e-6 this BRACKETS the threshold coefficient c (cutoff at Ri ≥ c·Pr_t) to the
    one-sided interval 1/(1+δ) < c ≤ 1 — i.e. c==1 to ~1e-6, ruling out e.g. c=0.5 — not
    an exact proof that c is 1. (Km[0]==0 alone would also hold for buoy_arg<0, so it
    confirms cutoff OUTPUT, not the bracket by itself; the ON case supplies the lower bound.)
    """
    col_kw = dict(theta_top=305.0, theta_sfc=300.0, u_top=12.0, u_sfc=1.0)
    u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _column(**col_kw)
    Ri0 = float(_production_Ri(u, v, T, q_v, p_full, z_full)[0, 0])
    assert Ri0 > 0.0                                          # a genuinely stable interface

    km_on = np.asarray(_run(SmagorinskyConfig(Pr_t=Ri0 * (1.0 + 1e-6)), **col_kw)[0].Km)[0, 0]
    km_eq = np.asarray(_run(SmagorinskyConfig(Pr_t=Ri0), **col_kw)[0].Km)[0, 0]
    km_off = np.asarray(_run(SmagorinskyConfig(Pr_t=Ri0 * (1.0 - 1e-6)), **col_kw)[0].Km)[0, 0]

    assert km_on > 0.0                     # just below cutoff (Ri0 < Pr_t) ⇒ mixing ON
    assert km_eq == 0.0                     # Ri0 == Pr_t ⇒ OFF (inclusive equality)
    assert km_off == 0.0                    # just above cutoff (Ri0 > Pr_t) ⇒ mixing OFF


def test_smagorinsky_finite_grad_at_exact_cutoff():
    """jax.grad is finite EXACTLY at Ri = Pr_t — the double-where guard (a bare √(max) NaNs).

    Set Pr_t = the column's interface-0 Ri0 read from ``_production_Ri`` (an op-for-op
    jnp mirror of production; that source-level identity makes Ri[0]/Pr_t = Ri0/Ri0 =
    1.0 by IEEE, so buoy_arg[0] = 0 EXACTLY). Differentiate Σ K_m w.r.t a wind scale s at
    s=1, where interface-0 sits precisely on the kink (s·u = 1·u = u ⇒ Ri[0]==Ri0). The
    double-``where`` guard yields a finite cotangent; a bare f_buoy = √(max(1−Ri/Pr_t, 0))
    would leak 0·∞ = NaN there (√ slope diverges as buoy_arg→0⁺). This is the exact-cutoff
    discriminator the near-cutoff spanning approach cannot provide.
    """
    col_kw = dict(theta_top=305.0, theta_sfc=300.0, u_top=12.0, u_sfc=1.0)
    u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _column(**col_kw)
    Ri0 = float(_production_Ri(u, v, T, q_v, p_full, z_full)[0, 0])
    cfg = SmagorinskyConfig(Pr_t=Ri0)
    T_sfc = jnp.full((u.shape[0],), float(np.asarray(T)[0, -1]))
    q_sfc = saturation_mixing_ratio(T_sfc, p_half[:, -1])

    def total_Km(s):
        out = smagorinsky_turbulence(
            s * u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho, 300.0, cfg,
        )
        return jnp.sum(out.Km)

    # buoy_arg[0]==0 exactly by the op-for-op mirror construction (above); Km[0]==0 here
    # confirms interface-0 output IS at/below the cutoff (consistent with buoy_arg[0]==0,
    # though Km==0 alone would also hold for buoy_arg<0 — the mirror is what pins ==0).
    assert float(np.asarray(_run(cfg, **col_kw)[0].Km)[0, 0]) == 0.0
    g = float(jax.grad(total_Km)(1.0))
    assert np.isfinite(g)                    # guard keeps the exact-cutoff VJP finite


def test_smagorinsky_unstable_enhancement():
    """Unstable stratification (Ri<0) ⇒ Lilly-type factor √(1−Ri/Pr_t) > 1 (enhanced K_m)."""
    cfg = SmagorinskyConfig()
    # warm below, cold aloft ⇒ θ decreases upward ⇒ Ri<0
    o, inputs = _run(cfg, theta_top=298.0, theta_sfc=302.0, u_top=20.0, u_sfc=1.0)
    Km = np.asarray(o.Km)[0]
    Km_exp, Ri0 = _assemble_Km(cfg, inputs, 0)
    assert Ri0 < 0.0                                  # unstable
    assert Km[0] == pytest.approx(Km_exp, rel=1e-11)
    # the enhancement factor √(1−Ri/Pr_t) exceeds the neutral value 1
    assert np.sqrt(1.0 - Ri0 / cfg.Pr_t) > 1.0


def test_faithful_smagorinsky_Cs_squared_scaling():
    """K_m ∝ C_s² (the (C_s·l)² Smagorinsky structure): doubling C_s quadruples K_m."""
    o1, _ = _run(SmagorinskyConfig(C_s=0.2))
    o2, _ = _run(SmagorinskyConfig(C_s=0.4))
    km1 = np.asarray(o1.Km)[0, 0]
    km2 = np.asarray(o2.Km)[0, 0]
    assert km1 > 0.0
    assert km2 / km1 == pytest.approx(4.0, rel=1e-11)   # (0.4/0.2)² = 4


# ===========================================================================
# DEPARTURES (single-column deformation surrogate)
# ===========================================================================
def test_smagorinsky_deformation_approx_linear_in_vertical_shear():
    """K_m is APPROXIMATELY linear in the vertical wind-shear magnitude (the |S| form).

    In a neutral column (Ri=0, f_buoy=1), K_m=(C_s·l)²·|S| ∝ |S|, so scaling the wind
    profile by 2 scales K_m by ~2. Linearity is only APPROXIMATE because |S|=√(S²+1e-10)
    carries the AD-safety floor; at the strong resolved shear used here the floor is
    negligible, so the ratio matches 2 to rel=1e-6 (not exact). This pins the
    shear-(near-)linearity of |S|; it does NOT by itself distinguish the 1-D
    |S|=√((∂u/∂z)²+(∂v/∂z)²) surrogate from the modern 3-D-LES tensor √(2·S_ij·S_ij)
    (both degree-1 homogeneous in velocity) — the single-column departure is documented
    by inspection (no horizontal strain).
    """
    cfg = SmagorinskyConfig()
    # neutral: θ constant ⇒ N²=0 ⇒ Ri=0 ⇒ f_buoy=1. Strong shear so the AD-safety
    # S²+1e-10 floor (itself a departure) is negligible and the linearity is clean.
    o1, _ = _run(cfg, theta_top=300.0, theta_sfc=300.0, u_top=200.0, u_sfc=20.0)
    o2, _ = _run(cfg, theta_top=300.0, theta_sfc=300.0, u_top=400.0, u_sfc=40.0)
    km1 = np.asarray(o1.Km)[0, 0]
    km2 = np.asarray(o2.Km)[0, 0]
    assert km1 > 0.0
    assert km2 / km1 == pytest.approx(2.0, rel=1e-6)    # ~linear (floor negligible here)


def test_dispatch_unknown_turbulence_raises():
    """Dispatch hardening: an unknown turbulence scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown turbulence scheme"):
        get_turbulence_fn(TurbulenceConfig(scheme="not_a_scheme"))
