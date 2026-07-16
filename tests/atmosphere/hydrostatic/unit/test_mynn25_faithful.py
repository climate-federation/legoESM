"""Scheme-level oracle-faithfulness tests for the MYNN level-2.5 turbulence closure.

Unlike ``tke`` (a MY-INSPIRED k-l closure with CONSTANT diffusivity coefficients), this
implements the algebraic level-2.5 system of Nakanishi & Niino (2009): the stability
functions SM(G_M, G_H), SH(G_M, G_H) are genuine functions of shear and buoyancy, so ``Km``
RESPONDS to the stratification (in contrast to the tke ``Km ⊥ N²`` departure). Equation
numbers below are verified against the NN09 paper.

FAITHFUL to NN09 (forms / constants pinned or canaried here):
  * config defaults are the NN09 eq-66 constants (A1, A2, B1, B2, C1-C5; gamma1 in appendix A);
  * the length-scale constants — L_T 0.23 (eq 54), the L_B inner coefficient 5.0 (eq 55), the
    eq-53 L_S coefficients (0.2 the ζ<0 unstable-branch exponent, 2.7/3.7 the stable-branch
    constants), and the φ constants 9, 12 (eqs 34, 36) — are published NN09 literals;
  * ``Km`` is stability-DEPENDENT (SM live) — a stable and an unstable column give DIFFERENT
    diagnostic Km, and the unstable column mixes more (SM increases with instability);
  * ``Kh`` vs ``Km`` responds to stability (SH > SM in the convective G_H>0 column) — SH/SM live.

DEPARTURES / NUMERICS (locked + labeled):
  * the surface ``qke_sfc = B1^(2/3)·u*²`` is the MY82/MYNN surface production-dissipation
    balance (an inherited MY/WRF/jax_scm ground BC, NOT NN09 eq 54); the pin is an exact
    implementation-coupling check that qke_new[:,-1] is set from the returned u*;
  * ``Km``/``Kh`` FLOORED at 0: a DEFENSIVE guard on the AD-safe approximations — the exact
    NN09 level-2.5 SM/SH are analytically nonnegative (level-3 S'_M/S'_H are what can turn
    negative), so the test pins the K≥0 & finite contract plus the SH-suppression response,
    NOT clamp activation (the floor rarely bites in a static column);
  * the 1-2-1 vertical filter on L/SM/SH (jax_scm smoother, not a NN09 core term) — pinned
    against an independent reflect-pad reimpl;
  * the ``_QKE_FLOOR`` on the prognostic qke is a numerics floor;
  * unknown turbulence scheme raises ValueError (dispatch hardening; sanity, not a pin).

NOT pinned here (further NN09 departures, documented in the module docstring): the dry
θ_v / q_v buoyancy vs NN09's θ_l / q_w partial-condensation treatment (appendix B), and the
per-column lowest-height θ_v reference vs NN09's Θ_0 reference state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics._shared import exner_function, virtual_temperature
from legoesm.atmosphere.physics.turbulence.config import MYNN25Config, TurbulenceConfig
from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn
from legoesm.atmosphere.physics.turbulence.mynn25 import (
    _L_FLOOR,
    _MYNN_LB_COEFF,
    _MYNN_LS_STABLE_FLOOR,
    _MYNN_LS_STABLE_HIGH,
    _MYNN_LS_STABLE_MID,
    _MYNN_LT_COEFF,
    _MYNN_PHI_C9,
    _MYNN_PHI_C12,
    _QKE_FLOOR,
    _filter_121,
    mynn25_turbulence,
)
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

    z is set directly (linear top→surface) so changing θ (hence N²) between two states leaves
    the z_full GEOMETRY bit-identical (the master length still varies, since L_B ∝ q/N and L_S
    depend on the stratification / surface state). Level 0 is the top, nlev-1 the surface.
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


def _state(grid, theta_top=300.0, theta_sfc=300.0, u_top=0.0, u_sfc=0.0, qke_val=1.0):
    """Build (u, v, T, q_v, qke, rho) on a fixed grid; θ linear top→surface, u linear."""
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
    qke = jnp.full((ncol, nlev), qke_val)
    return u, v, T, q_v, qke, rho


def _run(cfg, grid=None, dt=300.0, **state_kw):
    grid = grid if grid is not None else _grid()
    p_full, p_half, z_full, z_half = grid
    u, v, T, q_v, qke, rho = _state(grid, **state_kw)
    T_sfc = jnp.full((u.shape[0],), float(np.asarray(T)[0, -1]))
    q_sfc = saturation_mixing_ratio(T_sfc, p_half[:, -1])
    out, qke_new = mynn25_turbulence(
        u, v, T, q_v, qke, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho, dt, cfg,
    )
    return out, qke_new, (u, v, T, q_v, qke, p_full, p_half, z_full, z_half, rho)


def _N2_half0(inp):
    """MYNN25's N² at the top half-interface (col 0): (g/th_ref)·dθv/dz, th_ref = surface θv."""
    u, v, T, q_v, qke, p_full, p_half, z_full, z_half, rho = inp
    dz = np.clip(abs(np.asarray(z_full)[0, 0] - np.asarray(z_full)[0, 1]), 1.0, None)
    exner = 1.0 / np.asarray(exner_function(p_full))[0]
    thv = np.asarray(virtual_temperature(T, q_v))[0] * exner
    th_ref = thv[-1]                                     # surface θv (MYNN25 th_ref)
    dthv_dz = (thv[0] - thv[1]) / dz
    return float(constants.g / th_ref * dthv_dz)


# ===========================================================================
# FAITHFUL to NN09 (config, length constants, live SM/SH)
# ===========================================================================
def test_mynn25_config_defaults_nn09_eq66():
    """Canary: MYNN25Config defaults are the NN09 eq-66 closure constants (gamma1 app. A)."""
    c = MYNN25Config()
    assert c.A1 == 1.18
    assert c.A2 == 0.665
    assert c.B1 == 24.0
    assert c.B2 == 15.0
    assert c.C1 == 0.137
    assert c.C2 == 0.75
    assert c.C3 == 0.352
    assert c.C4 == 0.0
    assert c.C5 == 0.2
    assert c.gamma1 == 0.235


def test_mynn25_length_scale_constants_nn09():
    """Canary: the master-length constants are the published NN09 literals.

    L_T = 0.23·∫qz dz/∫q dz (eq 54); L_B = (q/N)·(1 + 5√(q_c/(L_T N))) in the ζ<0
    (unstable surface-layer) branch (eq 55, so 5.0 is the INNER coefficient, not an outer
    factor); the eq-53 L_S coefficients — 0.2 is the ζ<0 unstable-branch EXPONENT (the
    `_MYNN_LS_STABLE_FLOOR` name is a misnomer), 2.7 and 3.7 the stable-branch (ζ≥0) constants;
    and the φ constants (9, 12) from eqs 34 and 36.
    """
    assert _MYNN_LT_COEFF == 0.23
    assert _MYNN_LB_COEFF == 5.0
    assert _MYNN_LS_STABLE_FLOOR == 0.2
    assert _MYNN_LS_STABLE_MID == 2.7
    assert _MYNN_LS_STABLE_HIGH == 3.7
    assert _MYNN_PHI_C9 == 9.0
    assert _MYNN_PHI_C12 == 12.0


# --- DEPARTURE: surface qke BC is the MY82/MYNN balance, not NN09 (placed here for locality) ---
def test_mynn25_surface_qke_reset():
    """EXACT pin (DEPARTURE): the surface cell is RESET to qke_sfc = B1^(2/3)·u*².

    This is the MY82/MYNN surface production-dissipation balance (an inherited ground BC,
    NOT NN09 eq 54, which is L_T) — a DEPARTURE from NN09, not a faithful NN09 form. The pin
    is an implementation-coupling check that the returned surface qke is computed from the
    SAME returned u* — exact by construction.
    """
    cfg = MYNN25Config()
    out, qke_new, _ = _run(cfg, theta_top=305.0, theta_sfc=300.0, u_top=20.0, u_sfc=5.0)
    ustar = np.asarray(out.ustar)
    qke_sfc_expected = cfg.B1 ** (2.0 / 3.0) * ustar ** 2
    assert np.all(ustar > 0.0)                           # non-vacuity: real friction velocity
    assert np.allclose(np.asarray(qke_new)[:, -1], qke_sfc_expected, rtol=1e-12, atol=0.0)


# --- FAITHFUL (continued): the live level-2.5 stability functions ---
def test_mynn25_Km_depends_on_stratification():
    """Whole-scheme stratification response: Km RESPONDS to N² — contrast to the tke departure.

    On a FIXED (p, z) grid with the SAME qke and winds, a stable (N²>0) and an unstable (N²<0)
    column give DIFFERENT diagnostic Km, and the unstable column mixes MORE. Here the response
    flows through BOTH the stability functions SM(G_M, G_H) AND the buoyancy/surface length
    scales L_B, L_S (so this is a whole-scheme canary, not an isolation of SM alone). tke's
    CONSTANT Ck with a θ-decoupled length would instead give BIT-IDENTICAL Km — the qualitative
    discriminator that this closure's stability functions are live.
    """
    cfg = MYNN25Config()
    grid = _grid()
    out_stable, _, inp_stable = _run(
        cfg, grid=grid, theta_top=310.0, theta_sfc=300.0, u_top=20.0, u_sfc=5.0
    )
    out_unstable, _, inp_unstable = _run(
        cfg, grid=grid, theta_top=298.0, theta_sfc=312.0, u_top=20.0, u_sfc=5.0
    )
    # non-vacuity: the stratifications genuinely have OPPOSITE-sign N²
    assert _N2_half0(inp_stable) > 0.0                   # warm aloft ⇒ stable
    assert _N2_half0(inp_unstable) < 0.0                 # warm surface ⇒ unstable
    Km_stable = np.asarray(out_stable.Km)
    Km_unstable = np.asarray(out_unstable.Km)
    # SM is live: Km is NOT stratification-independent...
    assert not np.array_equal(Km_stable, Km_unstable)
    # ...and the unstable column mixes more (SM increases with instability)
    assert Km_unstable.mean() > Km_stable.mean()


def test_mynn25_Kh_exceeds_Km_when_unstable():
    """Faithful: SH > SM in a convective column ⇒ Kh > Km (heat mixes more than momentum)."""
    cfg = MYNN25Config()
    out, _, _ = _run(cfg, theta_top=298.0, theta_sfc=312.0, u_top=20.0, u_sfc=5.0)
    Km = np.asarray(out.Km)
    Kh = np.asarray(out.Kh)
    interior = slice(1, -1)
    assert np.all(Km[:, interior] > 0.0)                 # non-vacuity: real mixing
    assert np.all(Kh[:, interior] > Km[:, interior])     # SH/SM > 1 when unstable


# ===========================================================================
# DEPARTURES / NUMERICS (K≥0 floor, 1-2-1 filter, qke floor, dispatch)
# ===========================================================================
def test_departure_mynn25_diffusivity_nonnegative_floor():
    """DEPARTURE: Km/Kh floored at 0 + AD-safe D25/disc floors keep K finite and ≥0.

    The EXACT NN09 level-2.5 SM/SH are analytically nonnegative (only the level-3 corrections
    S'_M/S'_H can turn negative), so the ``max(·, 0)`` clamp is a DEFENSIVE guard on the AD-safe
    approximations (the floored D25 / discriminant / 1−Rf / Rf2−Rf), which can yield a slightly
    negative SM25/SH25 in numerical edge cases. An extreme inversion (weak shear, huge dθ/dz) is
    the regime that stresses those floors; the clamp keeps K≥0 and finite. The stability
    functions still push heat mixing down in strong stable — the strong-stable column mixes far
    LESS heat than a matched unstable one (SH suppressed), the physical response behind K≥0.
    """
    cfg = MYNN25Config()
    grid = _grid()
    out_stable, _, inp = _run(
        cfg, grid=grid, theta_top=420.0, theta_sfc=300.0, u_top=0.5, u_sfc=0.1
    )
    out_unstable, _, _ = _run(
        cfg, grid=grid, theta_top=298.0, theta_sfc=312.0, u_top=0.5, u_sfc=0.1
    )
    assert _N2_half0(inp) > 0.0                           # strongly stable
    Km = np.asarray(out_stable.Km)
    Kh = np.asarray(out_stable.Kh)
    assert np.all(np.isfinite(Km)) and np.all(np.isfinite(Kh))
    assert np.all(Km >= 0.0) and np.all(Kh >= 0.0)       # the floor invariant
    # non-vacuity: SH is suppressed toward the floor — Kh_stable ≪ Kh_unstable
    assert np.asarray(out_stable.Kh).mean() < np.asarray(out_unstable.Kh).mean()


def test_departure_mynn25_filter_121_reflect():
    """DEPARTURE: the 1-2-1 reflect-pad smoother (jax_scm) vs an independent NumPy reimpl."""
    x = jnp.asarray([[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], [2.0, 7.0, 1.0, 8.0, 2.0, 8.0]])
    got = np.asarray(_filter_121(x))
    xp = np.pad(np.asarray(x), ((0, 0), (1, 1)), mode="reflect")
    exp = (xp[:, :-2] + 2.0 * xp[:, 1:-1] + xp[:, 2:]) * 0.25
    assert np.allclose(got, exp, rtol=1e-12, atol=0.0)
    # non-vacuity: the filter is not the identity on a non-uniform field
    assert not np.allclose(got, np.asarray(x))


def test_numerics_mynn25_qke_stays_above_floor():
    """NUMERICS: the prognostic qke never drops below _QKE_FLOOR."""
    cfg = MYNN25Config()
    _, qke_new, _ = _run(cfg, theta_top=320.0, theta_sfc=300.0, u_top=10.0, u_sfc=2.0)
    assert _QKE_FLOOR > 0.0 and _L_FLOOR > 0.0           # the floors are live positive guards
    assert np.all(np.asarray(qke_new) >= _QKE_FLOOR)


def test_dispatch_unknown_turbulence_raises():
    """Dispatch hardening: an unknown turbulence scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown turbulence scheme"):
        get_turbulence_fn(TurbulenceConfig(scheme="not_a_scheme"))
