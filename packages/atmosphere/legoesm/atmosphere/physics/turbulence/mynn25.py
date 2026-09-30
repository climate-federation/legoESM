"""Mellor-Yamada-Nakanishi-Niino level-2.5 (MYNN-2.5) turbulence closure.

Faithful port of the closure used by jax_scm (Pierzyna 2026, arXiv:2605.24544)
for jax_scm-oracle parity on the GABLS1 / Wangara / Ekman benchmarks.  Follows
Nakanishi & Niino (2009) — *Development of an Improved Turbulence Closure
Model for the Atmospheric Boundary Layer*, JMSJ 87, 895–912.

Prognostic variable
-------------------
``qke = q² = 2·TKE`` [m²/s²].  Carried in :class:`PhysicsState.tke` —
the slot is shared with the older Mellor-Yamada-1982 ``tke`` scheme to
avoid a PhysicsState schema change for every closure variant; users
must not mix schemes mid-run (which legoESM already forbids via
``make_physics(config, …)`` being config-locked).

Master length scale
-------------------
``L = (1/L_S + 1/L_T + 1/L_B)^-1`` (NN09 eq. 52), the harmonic mean of
the surface, turbulent, and buoyancy length scales (eqs. 53–55).  A
1-2-1 vertical smoother is applied to ``L`` (and to the stability
functions ``SM`` / ``SH``) to damp grid-scale oscillations that the
``L_B = q/N`` denominator can amplify above the boundary layer.

Stability functions
-------------------
Algebraic level-2.5 closure — ``SM25``, ``SH25`` — solved from the
gradient Richardson number ``Ri`` via the level-2 flux-Richardson
quadratic (NN09 appendix A), with the ``alpha_c`` rescaling (eq. 42)
that bridges low-turbulence regimes.  Eddy diffusivities are
``Km = L·q·SM``, ``Kh = L·q·SH``, ``Kq = L·q·Sq`` with ``Sq = 3·SM``
(eq. 67) for the qke transport term.

Boundary conditions
-------------------
Surface qke is the MY82/MYNN surface production-dissipation balance
``qke_sfc = B1^(2/3) · u*²`` (an inherited MY/WRF/jax_scm ground BC, not
an NN09-specific equation), imposed as a post-step surface reset AFTER the
diffusion + production + dissipation update.  Surface heat and
moisture fluxes are taken from :func:`compute_surface_fluxes` so the
prescribed-flux / prescribed-T_s SCM forcing hooks (Phase B v2) carry
through unchanged.  Top is zero-flux.

Faithfulness to NN09 / jax_scm
------------------------------
Unlike the ``tke`` scheme (a MY-inspired k-l closure with CONSTANT
diffusivity coefficients), this closure IS the algebraic level-2.5 system:
the stability functions ``SM``/``SH`` are genuine functions of the shear
``G_M`` and buoyancy ``G_H`` (so ``Km`` responds to the stratification).
FAITHFUL to NN09 (forms — equation numbers verified against the NN09 paper):
  * prognostic ``qke = q² = 2·TKE`` with dissipation ``ε = qke^{3/2}/(B1·L)`` (eq. 12),
    entering the ``q²`` budget as ``−2ε`` (eq. 5) — the leading ``2`` is the continuum
    ``q² = 2·TKE`` coefficient, not a numerical artifact;
  * master length ``L = (1/L_S + 1/L_T + 1/L_B)^-1`` (eq. 52), the harmonic mean of the
    surface (eq. 53), turbulent (eq. 54), buoyancy (eq. 55) length scales;
  * ``L_T = 0.23·∫q z dz / ∫q dz`` (eq. 54) — the faithful discrete quadrature of NN09's
    continuous integral (the dz-weighting IS the integral; a bare point-sum would be the
    approximation). The half-level / end-point quadrature choice is a minor detail;
  * algebraic level-2.5 ``SM``/``SH`` (eqs. 27-28) via the level-2 flux-Richardson
    quadratic (appendix A) and the ``alpha_c`` rescaling (eq. 42);
  * ``Km = L·q·SM``, ``Kh = L·q·SH``, ``Kq = L·q·(3·SM)`` (``Sq = 3·SM``, eq. 67);
  * the NN09 closure constants ``A1``/``A2``/``B1``/``B2``/``C1``-``C5``/``gamma1`` (eq. 66;
    ``gamma1`` in appendix A), matching jax_scm's ``MYNNParams``.
DEPARTURES from NN09 (physics simplifications):
  * **dry θ_v / q_v** buoyancy: NN09 is formulated in liquid-water potential temperature
    ``θ_l`` and total water ``q_w`` with a partial-condensation (cloudy) buoyancy treatment
    (appendix B); this implementation uses dry virtual-potential-temperature gradients and
    ordinary water vapor — no cloud-conditional buoyancy flux;
  * **per-column reference θ**: the buoyancy reference is each column's lowest-height
    (surface-adjacent) ``θ_v``, a surrogate for NN09's reference state ``Θ_0`` (and for
    jax_scm's constant-per-case ``th_ref``);
  * the surface ``qke_sfc = B1^(2/3)·u*²`` is the **MY82/MYNN surface production-dissipation
    balance** (an inherited MY/WRF/jax_scm ground boundary condition), NOT an NN09-specific
    equation — imposed as a post-step surface RESET (see NUMERICS).
NUMERICS (jax_scm-matching or AD-safety — not physics):
  * a **1-2-1 vertical filter** on ``L``, ``SM``, ``SH`` (reflect-padded; matches jax_scm;
    damps the grid-scale noise that ``L_B ∝ q/N`` amplifies) — no such term in NN09;
  * ``Km``/``Kh``/``Kq`` **floored at 0**: a defensive guard on the AD-safe approximations
    (the floored ``D25`` / discriminant / ``1−Rf`` / ``Rf2−Rf``) — the EXACT NN09 level-2.5
    ``SM``/``SH`` are analytically nonnegative (it is the level-3 corrections ``S'_M``/``S'_H``
    that can go negative), so this clamps numerical edge cases, not a stated anti-diffusive
    NN09 algebra;
  * **AD-safety**: ``_safe_pow_pos`` (floored fractional powers), the discriminant /
    ``1−Rf`` / ``Rf2−Rf`` / ``D25`` floors, the Obukhov ``0/0`` double-``where`` guard,
    and the ``_QKE_FLOOR`` / ``_L_FLOOR`` / ``_SMOOTH_EPS`` floors;
  * the stability parameter ``ζ = z/L_obukhov`` is **clipped to ``[−100, 100]``** before the
    eq-53 ``L_S`` formula (bounds L_S in extreme stability / near-zero Obukhov length);
  * **surface qke handling**: implicit ``Kq`` diffusion, then the surface cell is OVERWRITTEN
    with ``qke_sfc`` AFTER the diffusion + production + dissipation update (a post-step reset,
    not a Dirichlet value inside the tridiagonal solve — a small non-conservative surface
    adjustment that can add or remove qke; Phase-D2 follow-up); and the semi-implicit
    linearization of the dissipation.
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_mynn25_faithful.py``.

References
----------
- Nakanishi, M., and H. Niino, 2009: J. Meteor. Soc. Japan 87, 895-912.
- Mellor, G. L., and T. Yamada, 1982: Rev. Geophys. 20, 851-875.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import exner_function, virtual_temperature
from legoesm.atmosphere.physics.turbulence.config import MYNN25Config
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    latent_enthalpy_correction,
    surface_moisture_flux,
    compute_surface_fluxes,
    surface_fluxes_at_lowest_level,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Mellor-Yamada-Nakanishi-Niino level-2.5 (MYNN-2.5) turbulence "
        "closure: prognostic qke = q^2 = 2*TKE with a master length scale and "
        "algebraic level-2.5 stability functions SM, SH set eddy diffusivities "
        "Km, Kh that mix momentum, heat (theta-space) and moisture."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "qke": "m^2/s^2 (q^2 = 2*TKE)",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m", "qke_new": "m^2/s^2",
    },
    "sign_convention": (
        "Down-gradient eddy diffusion, Km = L*q*SM >= 0, Kh = L*q*SH >= 0; "
        "stability functions increase mixing when unstable and suppress it "
        "when stable. The column budget is OPEN: the surface flux (shflx > 0 "
        "upward, lhflx > 0 upward/moistening) is the bottom boundary condition "
        "and the surface qke is reset to qke_sfc = B1^(2/3)*u*^2 after the qke "
        "update (a post-step reset, not a Dirichlet in the solve); "
        "top is zero-flux; z increases upward."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Nakanishi & Niino (2009), J. Meteor. Soc. Japan 87, 895-912; "
        "Mellor & Yamada (1982), Rev. Geophys. 20, 851-875"
    ),
    "idealized_test": (
        "jax_scm oracle parity on GABLS1 / Wangara / Ekman; rest state with "
        "zero surface flux and a neutral column -> near-zero interior "
        "tendency; Km, Kh >= 0; qke stays >= floor."
    ),
}


_QKE_FLOOR = 1e-10        # m²/s²; floor on qke to keep sqrt finite
_L_FLOOR = 1.0            # m; floor on master length scale
_SMOOTH_EPS = 1e-30       # used in safe sqrt / safe divide
# |Ri| is bounded BEFORE it is squared in the level-2 discriminant. G_M is
# exactly zero in a shear-free column, so the 1e-30 divisor floor sends Ri to
# ~1e28-1e30 and Ri*Ri overflows float32 (max 3.4e38) to inf, whose later
# inf/inf is NaN. This module runs x64 in the SCM benchmarks and float32 in a
# global run, so the overflow is a real runtime mode rather than a hypothetical.
# Set as HIGH as float32 allows rather than at a round number: the bound is
# only there to keep the square finite, and every entry it touches is one the
# closure would otherwise have evaluated, so a lower bound perturbs more
# columns for no extra protection. 1e18^2 = 1e36 against a float32 max of
# 3.4e38. MEASURED at 1e15 it moved ekman, wangara and astex at round-off (the
# near-zero shear above the boundary layer); at 1e18 it does not.
_RI_MAX = 1e18
# Two-sided floor on |F1| and |F2|, the level-2 combinations of the closure
# constants that Ri1/Ri2/Ri3 and Rf1/Rf2 divide by. They are ~6.3 and ~5.0 at
# the NN09 defaults, but they are built from TRAINABLE coefficients and F1 hits
# exactly zero inside the declared sigmoid bounds, so the tuner can walk onto
# the singularity.
_F_FLOOR = 1e-6


# MYNN Level-2.5 (Nakanishi-Niino 2009) fixed closure constants.
_MYNN_LS_STABLE_FLOOR = 0.2
_MYNN_LS_STABLE_MID = 2.7
_MYNN_LS_STABLE_HIGH = 3.7
_MYNN_LT_COEFF = 0.23
_MYNN_LB_COEFF = 5.0
_MYNN_PHI_C9 = 9.0
_MYNN_PHI_C12 = 12.0

def _away_from_zero(x: jax.Array, floor: float) -> jax.Array:
    """``x`` pushed out to ``+/-floor`` without changing its sign.

    For denominators that CROSS zero rather than merely approach it. A
    one-sided ``jnp.maximum(x, eps)`` does not regularize such a quantity: it
    leaves the root in place, converts it into a ~1/eps amplification, and
    flips the sign of every negative value. Every entry with
    ``|x| >= floor`` is returned unchanged, so the guard is inert wherever the
    expression was already well posed.

    A PYTHON scalar stays a Python scalar. The level-2 constants F1/F2 are
    plain floats whenever the config is not being traced (i.e. every
    production run), and routing them through ``jnp.where`` moved their
    arithmetic from CPython onto XLA, which reassociates. MEASURED: that alone
    changed ekman and wangara in the last bits of ``v`` -- a field whose true
    value is zero, so the change was pure round-off, but it was a change to
    every existing mynn25 result for no benefit. The guard is only reachable
    when the coefficients are TRACED, which is exactly when they are arrays.
    """
    if isinstance(x, (int, float)):
        return math.copysign(max(abs(x), floor), x) if x else floor
    return jnp.where(x >= 0.0, jnp.maximum(x, floor), jnp.minimum(x, -floor))


def _safe_pow_pos(x: jax.Array, p: float) -> jax.Array:
    """Floored power for AD-safe ``x**p`` with ``p`` fractional and ``p < 1``.

    ``jnp.power`` evaluates both branches under VJP; clipping the base
    above ``_SMOOTH_EPS`` keeps cotangents finite when ``x`` approaches
    zero.  The forward result is bit-identical wherever ``x > eps``.
    """
    return jnp.power(jnp.maximum(x, _SMOOTH_EPS), p)


def _filter_121(x: jax.Array) -> jax.Array:
    """In-place 1-2-1 weighted vertical smoother on the trailing axis.

    Reflect padding at both boundaries gives more aggressive damping at
    the column edges than the natural ``3/4·x[edge]`` weighting and
    matches the jax_scm implementation.
    """
    x_pad = jnp.pad(x, ((0, 0), (1, 1)), mode="reflect")
    return (x_pad[:, :-2] + 2.0 * x_pad[:, 1:-1] + x_pad[:, 2:]) * 0.25


def _full_to_half(x: jax.Array) -> jax.Array:
    """Linear interpolation from full levels ``(ncol, nlev)`` to interior
    half levels ``(ncol, nlev-1)`` (the ``nlev-1`` interfaces between
    consecutive full levels)."""
    return 0.5 * (x[:, :-1] + x[:, 1:])


def _half_to_full(x_half: jax.Array, nlev: int) -> jax.Array:
    """Interpolate half-level quantity ``(ncol, nlev-1)`` to full levels
    ``(ncol, nlev)`` with one-sided fallback at top/surface."""
    interior = 0.5 * (x_half[:, :-1] + x_half[:, 1:])
    return jnp.concatenate(
        [x_half[:, :1], interior, x_half[:, -1:]], axis=-1,
    )


def _compute_master_length(
    q_half: jax.Array,       # (ncol, nlev-1)
    z_half_geom: jax.Array,  # (ncol, nlev-1) — geometric height above surface at interior interfaces
    dz_half: jax.Array,      # (ncol, nlev-1) — spacing between adjacent full levels
    L_obukhov: jax.Array,    # (ncol,)
    dthv_dz_half: jax.Array, # (ncol, nlev-1)
    w_thv_sfc: jax.Array,    # (ncol,)
    th_ref: float,
    config: MYNN25Config,
) -> jax.Array:
    """Master length L (NN09 eq. 52) as harmonic mean of L_S, L_T, L_B.

    All arrays are interior half-level ``(ncol, nlev-1)``.  zeta = z/L
    is clipped to a sane range so the surface-layer L_S formula behaves
    monotonically.  ``dz_half`` weights the L_T vertical integrals
    (NN09 eq. 54) so the turbulent length scale is correct on stretched
    grids.  Output is unfiltered — caller applies the 1-2-1 smoother.
    """
    eps = _SMOOTH_EPS
    # L_obukhov can be ±inf in neutral conditions; jnp handles inf safely
    # in the divisions below, but we clip for AD finiteness.
    L_obu_safe = jnp.where(
        jnp.isfinite(L_obukhov), L_obukhov, jnp.sign(L_obukhov + eps) * 1e30,
    )
    zeta = z_half_geom / jnp.where(jnp.abs(L_obu_safe) > eps, L_obu_safe, eps)[:, None]
    zeta = jnp.clip(zeta, -100.0, 100.0)

    kappa = constants.kappa_vk
    L_S_unstable = kappa * z_half_geom * _safe_pow_pos(
        1.0 - 100.0 * zeta, _MYNN_LS_STABLE_FLOOR,
    )
    # The mid-stability branch is SELECTED only for 0 <= zeta < 1, where the
    # denominator is >= 1, but jnp.where EVALUATES it everywhere and it
    # vanishes at zeta = -1/2.7 = -0.370 -- an ordinary unstable surface-layer
    # value. Forward that is harmless (the entry is discarded), but reverse
    # mode differentiates the division by zero and returns NaN through an
    # unselected branch. Substituting a safe denominator outside the branch's
    # own domain is the same double-where the Obukhov guard above uses, and it
    # leaves the forward result bit-identical. (codex)
    denom_mid = 1.0 + _MYNN_LS_STABLE_MID * zeta
    denom_mid = jnp.where(zeta >= 0.0, denom_mid, 1.0)
    L_S_stable_mid = kappa * z_half_geom / denom_mid
    L_S_stable_high = kappa * z_half_geom / _MYNN_LS_STABLE_HIGH
    L_S = jnp.where(
        zeta < 0.0,
        L_S_unstable,
        jnp.where(zeta < 1.0, L_S_stable_mid, L_S_stable_high),
    )
    L_S = jnp.maximum(L_S, _L_FLOOR)

    # Turbulent length scale L_T = 0.23 * ∫q·z dz / ∫q dz  (NN09 eq. 54).
    # The integrals are dz-WEIGHTED sums over the interior interfaces (WRF
    # module_bl_mynn accumulates ``qkw*zw*dz`` / ``qkw*dz`` the same way);
    # a bare point-sum ratio Σ(q·z)/Σ(q) equals the integral ratio only on a
    # uniform grid and under-weights the (thicker) upper layers on the
    # stretched vertical grids used in SCM/GCM columns.  dz cancels exactly
    # when constant, so uniform-grid results are unchanged.  Output is a
    # per-column scalar broadcast across the half-level axis.
    num = jnp.sum(q_half * z_half_geom * dz_half, axis=-1)
    den = jnp.sum(q_half * dz_half, axis=-1)
    L_T_col = _MYNN_LT_COEFF * num / jnp.maximum(den, eps)
    L_T = jnp.broadcast_to(
        L_T_col[:, None], q_half.shape,
    )
    L_T = jnp.maximum(L_T, _L_FLOOR)

    # Buoyancy length scale L_B (NN09 eq. 55).  N² = g/θ_v · ∂θ_v/∂z.
    N2 = (constants.g / th_ref) * dthv_dz_half
    N = _safe_pow_pos(jnp.maximum(N2, 0.0), 0.5)
    q_c = _safe_pow_pos(
        (constants.g / th_ref) * w_thv_sfc[:, None] * L_T, 1.0 / 3.0,
    )
    L_B_stable_pos = q_half / jnp.maximum(N, eps)
    L_B_unstable = (
        1.0 + _MYNN_LB_COEFF * _safe_pow_pos(q_c / jnp.maximum(L_T * N, eps), 0.5)
    ) * q_half / jnp.maximum(N, eps)
    L_B = jnp.where(
        dthv_dz_half <= 0.0,
        1e30,  # ≈ inf: harmonic mean drops L_B from the budget
        jnp.where(zeta >= 0.0, L_B_stable_pos, L_B_unstable),
    )
    L_B = jnp.maximum(L_B, _L_FLOOR)

    # Harmonic mean.
    L = 1.0 / (1.0 / L_S + 1.0 / L_T + 1.0 / L_B)
    return jnp.maximum(L, _L_FLOOR)


def _compute_SM_SH(
    G_M: jax.Array,        # (ncol, nlev-1)
    G_H: jax.Array,        # (ncol, nlev-1)
    L: jax.Array,          # (ncol, nlev-1)
    q_half: jax.Array,     # (ncol, nlev-1)
    config: MYNN25Config,
) -> tuple[jax.Array, jax.Array]:
    """Algebraic level-2.5 stability functions (NN09 eqs. 27, 28).

    Returns ``(SM25, SH25)`` on the same half-level grid as the inputs.
    The level-2 rescaling factor ``alpha_c`` (eq. 42) bridges the
    closure to weakly-turbulent regimes.
    """
    A1, A2 = config.A1, config.A2
    B1, B2 = config.B1, config.B2
    C1, C2, C3, C5 = config.C1, config.C2, config.C3, config.C5
    gamma1 = config.gamma1

    # Level-2 closure derived constants (NN09 appendix A).
    gamma2 = (2.0 * A1 * (3.0 - 2.0 * C2) + B2 * (1.0 - C3)) / B1
    F1 = B1 * (gamma1 - C1) + 2.0 * A1 * (3.0 - 2.0 * C2) + 3.0 * A2 * (1.0 - C2) * (1.0 - C5)
    F2 = B1 * (gamma1 + gamma2) - 3.0 * A1 * (1.0 - C2)
    # F1 and F2 are divided by four times below and they are built entirely
    # from TRAINABLE coefficients. F1 = 0 is reachable inside the declared
    # sigmoid bounds -- e.g. A1=0.7, A2=0.4, B1=15, B2=8, C1=0.15267, C2=1.4,
    # C3=0.95, C5=0.5, gamma1=0.15 -- so an optimizer exploring those bounds
    # can produce infinities before D25 is even formed. They are ~6.3 and ~5.0
    # at the NN09 defaults, so the two-sided floor is inert there. (codex)
    # NOT wrapped in jnp.asarray: these are built from Python floats when the
    # config is not being traced, and an explicit array creation would give
    # them a STRONG dtype (float64 under jax_enable_x64) that then promotes the
    # whole level-2 block away from the column's float32. jnp.where on weakly
    # typed inputs keeps the weak type and lets the state decide.
    F1 = _away_from_zero(F1, _F_FLOOR)
    F2 = _away_from_zero(F2, _F_FLOOR)
    Rf1 = B1 * (gamma1 - C1) / F1
    Rf2 = B1 * gamma1 / F2
    Rfc = gamma1 / (gamma1 + gamma2)
    Ri1 = 0.5 * A2 * F2 / (A1 * F1)
    Ri2 = 0.5 * Rf1 / Ri1
    Ri3 = (2.0 * Rf2 - Rf1) / Ri1

    # Bounded BEFORE the square below: G_M is exactly 0 with no resolved shear,
    # so the 1e-30 divisor floor makes |Ri| ~ 1e30 and Ri*Ri overflows float32
    # to inf, and inf/inf downstream is NaN. See _RI_MAX. (codex)
    Ri = jnp.clip(-G_H / jnp.maximum(G_M, _SMOOTH_EPS), -_RI_MAX, _RI_MAX)
    # Level-2 flux Richardson, NN09 eq A11.  Discriminant clipped to
    # zero for AD safety in near-neutral regimes.
    disc = jnp.maximum(Ri * Ri - Ri3 * Ri + Ri2 * Ri2, 0.0)
    Rf = Ri1 * (Ri + Ri2 - jnp.sqrt(disc + _SMOOTH_EPS))
    # NN09's level-2 stability functions; degenerate at Rf=1, Rf=Rf2,
    # so we clip Rf at a hair below.
    one_m_Rf = jnp.maximum(1.0 - Rf, _SMOOTH_EPS)
    Rf2_m_Rf = jnp.where(jnp.abs(Rf2 - Rf) > _SMOOTH_EPS, Rf2 - Rf, _SMOOTH_EPS)
    SH2 = 3.0 * A2 * (gamma1 + gamma2) * (Rfc - Rf) / one_m_Rf
    SM2 = (A1 * F1) / (A2 * F2) * (Rf1 - Rf) / Rf2_m_Rf * SH2

    # Level-2 diagnosed qke (NN09 eq A2) — used to derive alpha_c.
    qke_diag = B1 * L * L * SM2 * (1.0 - Rf) * G_M * (q_half * q_half / (L * L + _SMOOTH_EPS))
    qke_diag = jnp.maximum(qke_diag, _QKE_FLOOR)
    q2 = _safe_pow_pos(qke_diag, 0.5)

    alpha_c = jnp.where(q_half < q2, q_half / jnp.maximum(q2, _SMOOTH_EPS), 1.0)
    alpha_c2 = alpha_c * alpha_c

    phi_1 = 1.0 - 3.0 * alpha_c2 * A2 * B2 * (1.0 - C3) * G_H
    phi_2 = 1.0 - _MYNN_PHI_C9 * alpha_c2 * A1 * A2 * (1.0 - C2) * G_H
    phi_3 = phi_1 + _MYNN_PHI_C9 * alpha_c2 * A2 * A2 * (1.0 - C2) * (1.0 - C5) * G_H
    phi_4 = phi_1 - _MYNN_PHI_C12 * alpha_c2 * A1 * A2 * (1.0 - C2) * G_H
    phi_5 = 6.0 * alpha_c2 * A1 * A1 * G_M

    # D25 is the level-2.5 denominator. It equals 1.0 at G_M = G_H = 0 and it
    # PASSES THROUGH ZERO, so a one-sided ``maximum(D25, 1e-30)`` does not
    # regularize it -- it converts the root into a ~1e30 amplification.
    #
    # At exactly zero resolved shear phi_5 = 6*alpha_c^2*A1^2*G_M is 0, so
    # D25 = phi_2*phi_4 with
    #     phi_4 = 1 - [3*A2*B2*(1-C3) + 12*A1*A2*(1-C2)] * G_H,
    # whose root at the NN09 constants is G_H = 0.046. G_H = -L^2 N^2 / qke, so
    # any unstable layer in a shear-free column sweeps straight through it.
    # MEASURED at G_M = 0: SH25 = +5.5e29 just past the root, i.e. Kh ~ 1e25
    # m^2/s -- POSITIVE, so the downstream ``Kh >= 0`` clamp cannot see it, and
    # it enters the qke budget as -Kh*N^2. On the Nieuwstadt CBL (the one
    # tuning case with u_geo = v_geo = f_c = 0, hence S^2 identically zero)
    # that took qke from 5e-6 to 1e24 in a single step and the whole column to
    # NaN by step 10, while all seven other cases and all eight other closures
    # were finite through 2000 steps.
    #
    # The floor is TWO-SIDED and keeps the sign: SM25/SH25 are bit-identical
    # wherever |D25| >= the floor, and past the root they stay negative and are
    # caught by the existing Km/Kh >= 0 clamp exactly as before. Only the
    # neighbourhood of the root changes, which is the only place that was
    # producing 1e25 diffusivities.
    #
    # This bounds the singularity; it is not the NN09/Helfand-Labraga (1988)
    # joint (G_M, G_H) realizability limit, which would additionally keep the
    # closure inside its derived region.
    D25 = _away_from_zero(phi_2 * phi_4 + phi_5 * phi_3, config.d25_floor)
    SM25 = alpha_c * A1 * (phi_3 - 3.0 * C1 * phi_4) / D25
    SH25 = alpha_c * A2 * (phi_2 + 3.0 * C1 * phi_5) / D25
    return SM25, SH25


def mynn25_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    qke: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: MYNN25Config,
) -> tuple[TurbulenceOutput, jax.Array]:
    """Compute MYNN-2.5 turbulence tendencies and updated qke.

    Parameters mirror :func:`legoesm.atmosphere.physics.turbulence.tke.tke_turbulence`
    so the turbulence-integration dispatch sees a uniform signature.
    ``qke`` is ``q² = 2·TKE`` (NN09 convention) carried in the
    :class:`PhysicsState.tke` slot.
    """
    ncol, nlev = T.shape
    # Validated at function entry on the STATIC config value, like every other
    # scheme guard in this package. d25_floor = 0 restores the division by zero
    # at the D25 root, and a NEGATIVE value disables the floor entirely --
    # _away_from_zero then returns the raw denominator on both branches -- so
    # neither can be allowed to pass silently. It is excluded from
    # __param_spec__, so it is never a traced leaf. (codex)
    if not float(config.d25_floor) > 0.0:
        raise ValueError(
            f"MYNN25Config.d25_floor must be > 0, got {config.d25_floor!r}. "
            "Zero reinstates the level-2.5 denominator's pole and a negative "
            "value silently turns the guard off."
        )
    qke = jnp.maximum(qke, _QKE_FLOOR)

    # Heights above surface for L_S and L_T integrals.
    z_sfc = z_full[:, -1:]            # (ncol, 1)
    z_above = jnp.maximum(z_full - z_sfc, 0.0)

    # Surface fluxes via the bulk-flux helper (which honours the SCM
    # prescribed-flux bypass when configured).
    tau_x, tau_y, shflx, lhflx, ustar = surface_fluxes_at_lowest_level(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface, z_full[:, -1] - z_half[:, -1])

    # Kinematic surface fluxes for closure consistency (jax_scm convention).
    rho_sfc = rho[:, -1]
    w_th_s_kin = shflx / (rho_sfc * constants.c_pd)
    w_qv_s_kin = surface_moisture_flux(config.surface, lhflx, T_sfc) / rho_sfc

    # Potential temperature (full levels) via the canonical inverse-Exner
    # helper (exner_pref = 1/Π = (p_ref/p)^κ; same 1 Pa pressure floor).
    exner_pref = 1.0 / exner_function(p_full)
    theta = T * exner_pref
    theta_v = virtual_temperature(theta, q_v)

    # Buoyancy flux (virtual potential temperature) at surface.  Use
    # ``θ_v`` perturbation form ≈ w'θ' + (1/ε − 1) θ̄ w'q_v'.  The virtual-T
    # moisture coefficient is the canonical 1/ε − 1 ≈ 0.6078 (constants.epsilon),
    # not the rounded 0.61 (CLAUDE.md: no hardcoded physical-constant literals).
    th_low = theta[:, -1]
    _vT_coef = 1.0 / constants.epsilon - 1.0
    w_thv_sfc = w_th_s_kin + _vT_coef * th_low * w_qv_s_kin

    # Reference theta for L_B (per-column lowest virtual-θ — surrogate
    # for ``th_ref`` in jax_scm which is a constant per case).  Kept as
    # a JAX array of shape ``(ncol, 1)`` so the closure stays
    # ``jit`` / ``vmap`` / ``grad`` safe (no ``float(...)`` host
    # materialization inside the kernel).  Phase C codex iter-1 high
    # finding.
    th_ref = theta_v[:, -1:]   # (ncol, 1) — broadcasts to (ncol, nlev-1)

    # Obukhov length L = -θ_v·u*³ / (κ·g·w'θ_v').
    #
    # Mask the surface buoyancy flux in the divisor *before* dividing so the
    # dead branch never forms ``num/0`` at zero buoyancy flux (w'θ_v' = 0:
    # exact neutral, SCM-prescribed zero surface fluxes, or cold-start
    # air-surface equilibrium).  Otherwise reverse-mode AD differentiates the
    # unselected ``.../w_thv_sfc`` branch at 0 -> inf cotangent, and the ``where``
    # multiplies it by a zero selector -> ``0*inf = NaN``, poisoning every
    # tendency (zeta -> L_S -> master length -> qke).  Forward is unchanged: the
    # outer ``where`` still returns the 1e30 neutral sentinel (L -> ∞ ⇒ ζ ≈ 0)
    # whenever |w'θ_v'| ≤ eps.
    kappa = constants.kappa_vk
    w_thv_safe = jnp.where(
        jnp.abs(w_thv_sfc) > _SMOOTH_EPS, w_thv_sfc, 1.0
    )
    L_obukhov = jnp.where(
        jnp.abs(w_thv_sfc) > _SMOOTH_EPS,
        -(theta_v[:, -1] * ustar ** 3) / (kappa * constants.g * w_thv_safe),
        1e30,
    )

    # Half-level geometry: interior half-levels are at z = 0.5 (z[k] + z[k+1]).
    # Heights above surface at interior half-levels:
    z_above_half = _full_to_half(z_above)                # (ncol, nlev-1)
    dz_half = jnp.maximum(jnp.abs(z_full[:, :-1] - z_full[:, 1:]), 1.0)

    # Gradients on interior half-levels.
    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half             # ∂u/∂z, K=top-bottom
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    dthv_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half

    # qke on half levels (interior).  Use the prognostic full-level qke
    # interpolated to interfaces; boundary halves are not needed because
    # the closure only computes ``SM, SH`` on interior half-levels and
    # extends to full levels via interpolation.
    qke_half = _full_to_half(qke)                        # (ncol, nlev-1)
    qke_half = jnp.maximum(qke_half, _QKE_FLOOR)
    q_half = _safe_pow_pos(qke_half, 0.5)

    # Master length scale on interior half-levels.
    L = _compute_master_length(
        q_half=q_half,
        z_half_geom=z_above_half,
        dz_half=dz_half,
        L_obukhov=L_obukhov,
        dthv_dz_half=dthv_dz,
        w_thv_sfc=w_thv_sfc,
        th_ref=th_ref,
        config=config,
    )
    L = _filter_121(L)
    L = jnp.maximum(L, _L_FLOOR)

    # Algebraic stability functions.
    S2 = du_dz * du_dz + dv_dz * dv_dz
    N2 = (constants.g / th_ref) * dthv_dz
    G_M = L * L / (qke_half + _SMOOTH_EPS) * S2
    G_H = -L * L / (qke_half + _SMOOTH_EPS) * N2
    SM, SH = _compute_SM_SH(G_M, G_H, L, q_half, config)
    SM = _filter_121(SM)
    SH = _filter_121(SH)

    # Eddy diffusivities on interior half-levels (shape (ncol, nlev-1)).
    Km_half = L * q_half * SM
    Kh_half = L * q_half * SH
    Kq_half = L * q_half * (3.0 * SM)        # NN09 eq 67
    # Sign convention: eddy diffusivities are >= 0 (down-gradient mixing).
    # This clamp is LOAD-BEARING, not defensive. The comment it replaces said
    # the level-2.5 SM/SH are "analytically nonnegative" and that only a
    # numerical edge case could make them "slightly negative"; both halves are
    # false and were measured so. Past the D25 root at G_M = 0 the exact
    # algebra gives SM25 = -0.67 and SH25 = -0.032 at G_H = 1 -- O(1) negative,
    # not slight -- because SH25 reduces to A2/phi_4 there and phi_4 changes
    # sign. So this is a real upgradient branch of the closure being clamped
    # away, and the clamp must stay.
    # It is also NOT sufficient on its own: the same root produces LARGE
    # POSITIVE SM25/SH25 on the other side, which a >= 0 clamp cannot see. That
    # is bounded at the source by the two-sided D25 floor in _compute_SM_SH.
    Km_half = jnp.maximum(Km_half, 0.0)
    Kh_half = jnp.maximum(Kh_half, 0.0)
    Kq_half = jnp.maximum(Kq_half, 0.0)

    # Layer thicknesses at full levels (positive).
    dz_layer = jnp.maximum(jnp.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0)

    # Surface flux contributions per variable (kinematic units shouldn't
    # be confused with mass-weighted; existing helper expects
    # ``surface_flux = ρ K dφ/dz`` in mass-weighted form).
    sflx_u = tau_x
    sflx_v = tau_y
    sflx_q = surface_moisture_flux(config.surface, lhflx, T_sfc)
    # Heat BC carries the latent enthalpy correction (water at L(T) vs L_v).
    sflx_T = (shflx + latent_enthalpy_correction(lhflx, sflx_q)) / constants.c_pd

    # Diffuse u, v, theta, q_v using the existing implicit helpers.
    u_new = implicit_vertical_diffusion(
        u, Km_half, rho, dz_layer, dz_half, dt, sflx_u,
    )
    v_new = implicit_vertical_diffusion(
        v, Km_half, rho, dz_layer, dz_half, dt, sflx_v,
    )
    T_new = implicit_vertical_diffusion_theta(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T,
    )
    q_new = implicit_vertical_diffusion(
        q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q,
    )

    # --- qke prognostic equation ---
    # Production and buoyancy contributions on full levels.
    # Half-level P_S, P_B → averaged to full levels.
    P_S_half = Km_half * S2
    P_B_half = -Kh_half * N2
    P_S = _half_to_full(P_S_half, nlev)
    P_B = _half_to_full(P_B_half, nlev)

    # Master length on full levels (for the dissipation timescale).
    L_full = _half_to_full(L, nlev)
    L_full = jnp.maximum(L_full, _L_FLOOR)
    q_full = _safe_pow_pos(qke, 0.5)
    diss_coeff = q_full / (config.B1 * L_full)         # 1/s; ε = qke^(3/2)/(B1·L)

    # Implicit Kq diffusion with NO surface flux, then OVERWRITE the surface
    # cell with ``qke[-1] = B1^(2/3) · u*²`` AFTER the diffusion + production +
    # dissipation update (a post-step reset, not a Dirichlet value inside the
    # tridiagonal solve).  This is the standard MYNN convention (jax_scm
    # matches).  Approximate: the post-step reset is non-conservative — it can
    # add or remove surface qke vs a true in-solve Dirichlet, acceptable for
    # SCM benchmark fidelity but a candidate for a follow-up Phase D2 hardening.
    qke_diffused = implicit_vertical_diffusion(
        qke, Kq_half, rho, dz_layer, dz_half, dt,
        surface_flux=jnp.zeros(ncol, dtype=qke.dtype),
    )
    qke_after_prod = (
        qke_diffused + 2.0 * dt * (P_S + P_B)
    ) / (1.0 + 2.0 * dt * diss_coeff)
    qke_after_prod = jnp.maximum(qke_after_prod, _QKE_FLOOR)

    qke_sfc = _safe_pow_pos(jnp.asarray(config.B1, dtype=qke.dtype), 2.0 / 3.0) * ustar ** 2
    qke_sfc = jnp.maximum(qke_sfc, _QKE_FLOOR)
    qke_new = qke_after_prod.at[:, -1].set(qke_sfc)

    # Diagnostics: full-level Km, Kh for downstream tools.
    Km_full = _half_to_full(Km_half, nlev)
    Kh_full = _half_to_full(Kh_half, nlev)

    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)

    output = TurbulenceOutput(
        du_dt=(u_new - u) / dt,
        dv_dt=(v_new - v) / dt,
        dT_dt=(T_new - T) / dt,
        dq_v_dt=(q_new - q_v) / dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx,
        ustar=ustar,
        h_pbl=h_pbl,
    )
    return output, qke_new
