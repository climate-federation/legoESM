"""Co-scientist round-2 candidate C1-mem-R (KIND="both", EXPERIMENTAL).

Moist-regime wave-kernel EDMF-TKE with exact transit-time plume memory and an
Ri_eq-anchored stable length. Refinement of round-1 ``r1_c1_wave_kernel_mem``
(parent closure ``round1/r1_c1_wave_kernel.py``, whose helpers ``_g_inf``, ``_sp``,
``_n_plus``, ``_face_to_cell`` and fixed numerics are REUSED by import, not
re-derived). Not a production scheme: loaded by path by the co-scientist harness
(``legoesm.atmosphere.les_suite.candidate``).

Grid: cells i = 0..n-1 surface-first, faces f = 0..n; all returned fluxes are
KINEMATIC, positive up, shape (n+1,). Flux form only: the harness turns them into
tendencies ``-(rho_{f+1} F_{f+1} - rho_f F_f)/m_i``. Index 0 of the scalar fluxes is
the prescribed case flux bit for bit; index 0 of the momentum fluxes is the passed
surface-law stress on a CloudBench column (``surface_stress``; KNOWN DIFFERENCE from
the own law) and the candidate's own MOST stress, implicit in U_1, on an LES column;
index n is exactly 0 for all four.

What changes against the parent (the hypothesis)
------------------------------------------------
(A) Surface: LES lane own MOST stress as parent, but neutral-safe with
    u_reg = 1e-3 m/s and LINEARLY IMPLICIT in U_1: F_U(0) = -C U_1^{n+1},
    C = u*^2/U_g at time n. On a surface-law column C = 0 (no implicit drag) and the
    passed stress is applied explicitly, like the prescribed heat/moisture flux.
(B) Moist regime state at interior faces: Gaussian s-PDF cloud fraction
    (Sommeria-Deardorff 1977; Bechtold et al. 1995) s = a_L (q_t - q_s(T_l,p)),
    sigma_s^2 = c_sig^2 l0^2 [a_L (dq_t/dz - dq_s/dT (dT_l/dz + g/c_pd))]^2 + sigma_0^2,
    f_c = (1 + erf(s/(sqrt2 sigma_s)))/2; buoyancy-flux coefficients
    A = (1-f_c) A_dry + f_c A_sat (Cuijpers-Duynkerke 1993 form):
      A_th,dry = 1 + c_v q_t, A_q,dry = c_v theta, c_v = R_v/R_d - 1,
      A_th,sat = (1 - q_t + (q_s/eps)(1 + L_v/(R_v T)))/(1 + L_v^2 q_s/(c_pd R_v T^2)),
      A_q,sat = L_v A_th,sat/(c_pd Pi) - theta;
    N_eff^2 = (g/theta_v0)(A_th dtheta_l/dz + A_q dq_t/dz) replaces dry N^2 in the
    length, Ri, Pr_t, buoyancy production and the wave gate/transmission.
(C) Stable anchor: c_N = c_k sqrt(Ri_eq/(1 - Ri_eq/Pr_t(Ri_eq))),
    Pr_t(Ri_eq) = Pr_0 + Ri_eq/R_finf, so the z-less TKE balance
    K_m S^2 (1 - Ri/Pr_t) = eps with l = c_N sqrt(e)/N sits at Ri = Ri_eq.
    TKE buoyancy production P_b = (g/theta_v0) w'theta_v' with
    w'theta_v'_ED = -K_h N_eff^2 theta_v0/g and
    w'theta_v'_MF = M (A_th (theta_l,u - theta_l,env) + A_q (q_t,u - q_t,env)).
(D) Plume: as parent.
(E) Memory: M^{n+1} = M* + (M^n - M*) exp(-dt/tau), the exact solution of
    dM/dt = -(M - M*)/tau for M*, tau frozen over the step; tau = c_tau T_top + tau_min,
    T_top = sum_f Pi_f dz_f / sqrt(w2_f + w_d^2) (plume transit time, Pi = the
    parent's termination product). Convex combination: M >= 0, M_0 = M_n = 0.
(F) Wave kernel: as parent with the gate S = g_inf(r - 1), N_eff^2 in r and N_+,
    and the Lindzen saturation bound |F| <= k U_+^3/(2 N_+) enforced smoothly.

Mori-Zwanzig: Markovian = local K(Ri, N_eff, l), cloud-fraction coefficients, M*,
plume excesses, wave transmissions and the Lindzen cap; memory = TKE (implicit) and
the explicit exponential kernel on M; orthogonal noise = 0 (declared lever, inactive).

Conservation (CLAUDE.md RULE 0): every transport (ED, MF, wave incl. its Lindzen
cap) is a face flux returned to the harness, F(n) = 0, scalar F(0) = the prescribed
flux, momentum F(0) = the surface stress actually applied (passed, or own implicit
-C U_1^{n+1}). The implicit solve's storage/coupling rows are those of the harness
(theta_l row in the enthalpy form, storage m Pi, face weights Pi_half), so the
returned flux reproduces the solved state and column content changes by exactly
rho_0 F(0) dt (tested in x64). The wave kernel is zero-sum (F_W(0) = 0, absorb_top
lid). The memory changes only the coefficient M, never the flux form. Moist
coefficients enter only K and P_b. TKE is an auxiliary closure variable, outside the
harness energy budget (as parent); its buoyancy sink is not booked as geopotential.

Deviations from the hypothesis text (each recorded; detail choices decided by vote
author + codex + GLM, picks in the round record)
--------------------------------------------------------------------------------
1. sigma_s uses the NEUTRAL length l0 = 1/(1/(kappa z) + 1/lambda_inf) (vote V1):
   the full l depends on N_eff, which depends on f_c(sigma_s); l0 breaks the loop
   without iteration. sigma_s is therefore an upper estimate in stable layers.
2. Face thermodynamics (V2): face means of cell T_l, T, theta, q_t; q_s and dq_s/dT
   from ``legoesm.thermo`` at face-mean T_l (for s, a_L) and face-mean T (A_sat),
   pressure p_half. N_eff^2 at f_c = 0 equals the dry N^2 only to the linearisation
   of theta_v (codex: the cloud-weighted form is not the full Durran-Klemp N_m^2).
3a. Own law on a law column: fed z0 = 1e-3 z1 (its values are discarded; this keeps
   gradients finite whatever z0 the column carries, NaN or >= z1; votes V19/V20).
   sigma_w = 0 (calm passed stress, strong cooling) and the Lindzen |B| (no squaring,
   fp32 underflow) are double-where / log-norm safe (V18, review round 2).
3. sqrt(e) and U_g are double-where safe square roots (V3, V4): value unchanged
   where the argument is > 0; 0 with zero gradient at exactly 0 (calm surface:
   stress 0, u* = 0, no implicit drag).
4. Lindzen bound (V5-V7): applied to EACH launch's contribution at EACH face with that
   launch's own intrinsic wind U_+,jf = softplus((U_f - c_j).n_j) (face-mean wind,
   c_j = plume velocity at launch j, the parent's phase speed) and N_+ at the face:
   cap_jf = rho_f k U_+,jf^3/(2 N_+,f); with r = |F_jf|/cap_jf and R = the running
   maximum of r along the wave's path from its launch (vote V16: hard max, kinks at
   ties only), F_jf <- F_jf (1 + R^4)^(-1/4) (smoothmin p = 4, vote V6), in log space.
   Saturation persists along the path (no re-emission); the review-round-1 form that
   re-applied the smooth factor at every face compounded with the number of faces
   (0.5 cap after 15 faces at r = 1) and was replaced. The hypothesis'
   "clipped excess deposited in the cell below the face" is the cap itself: capping
   a face flux puts exactly the excess into the cell below through the flux
   divergence; a separate deposit term would double-count it (codex). There is no
   such line, so the hypothesis' revert control ("fails with the excess-deposit line
   removed") does not exist; the cap's own test is the bound, the zero-sum test
   covers its conservation.
5. Memory (V8-V10): c_tau bounds per vote; exp(-dt/tau) is safe at tau = 0
   (double-where, M = M*), so the true memory-free control the meta-review asked for
   is c_tau = 0 with ``_TAU_MIN_S`` = 0 (a source edit); the first call takes
   M = M* (parent's started flag, no spin-up).
6. Transit time (V11): faces f = 1..n, dz = the cell below face f, w2 at face f.
7. Plume launch momentum excess uses the explicit stress estimate
   -u*^2 U_1^n/U_g (law column: the passed stress), V13.
8. Surface-face buoyancy production keeps the parent's B_s from the prescribed
   fluxes (V14); cell N_eff^2 = mean of bounding faces (parent mapping, V15).
9. Saturation adjustment reuses ``les_suite.scm_coupling.saturation_adjust``
   (as parent).
10. q_t positivity at large dt is NOT guaranteed (codex hard gate): the implicit
   upwind environment term and implicit ED form an M-matrix, but the explicit plume
   term M q_u can over-draw a cell at large M dt. No limiter is applied (RULE 0);
   reported as an open risk.

Known differences / broken limits (declared)
-------------------------------------------
* CloudBench: shared surface-law stress, the own surface law (z0, beta_g, psi_m) is
  inert there; beta_g carries gradient only on LES columns.
* Pr_t(Ri_eq) in c_N uses Ri_eq itself, the running Pr_t uses the softplus Ri_+
  (difference r_d softplus - identity, < 1e-10 at Ri_eq >= 0.15).
* Wave KE exchange is signed: the harness returns interior shear work as heat, which
  is negative where the wave flux is up-gradient (energy closes; physical wave
  energetics are not claimed).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from legoesm.atmosphere.les_suite.candidate import (
    absorb_top,
    has_surface_law,
    surface_stress,
)
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.core.bulk_flux import psi_m
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_dT
from legoesm.timestepping.tridiagonal import thomas_solve

from legoesm import constants

_spec = importlib.util.spec_from_file_location(
    "_r2_c1mem_r_parent",
    Path(__file__).resolve().parents[1] / "round1" / "r1_c1_wave_kernel.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_g_inf, _sp, _n_plus, _face_to_cell = _base._g_inf, _base._sp, _base._n_plus, _base._face_to_cell

KIND = "both"

PARAMS = {
    # (C) local TKE closure
    "c_k": (0.52, 0.40, 0.60, "1"),              # K_m = c_k l sqrt(e); dissipation c_k^3
    "lambda_inf": (150.0, 30.0, 500.0, "m"),     # Blackadar asymptotic length
    "Ri_eq": (0.25, 0.15, 0.40, "1"),            # z-less self-regulating Ri (sets c_N)
    "Pr_0": (0.85, 0.6, 1.1, "1"),               # neutral turbulent Prandtl number
    "R_finf": (0.2, 0.1, 0.3, "1"),              # limiting flux Richardson number
    # (B) moist regime
    "c_sig": (1.0, 0.3, 2.0, "1"),               # sigma_s = c_sig l0 |grad s|
    # (A) surface layer (LES lane only; inert on a surface-law column)
    "beta_g": (1.0, 0.3, 1.5, "1"),              # gustiness from lowest-cell TKE
    # (D) plume
    "a_u": (0.1, 0.03, 0.2, "1"),                # plume area fraction at launch
    "beta_u": (1.0, 0.3, 2.0, "1"),              # launch excess beta_u F(0)/sigma_w
    "L_c": (500.0, 50.0, 3000.0, "m"),           # convective launch gate g_inf(-L_c/L)
    "c_ent": (0.4, 0.2, 1.0, "1"),               # entrainment c_ent/z
    "b_w": (2.0, 1.0, 3.0, "1"),                 # drag in the w_u^2 equation
    "c_p": (0.7, 0.3, 0.9, "1"),                 # plume pressure coefficient (momentum)
    "eps_c": (1e-3, 2e-4, 3e-3, "1/m"),          # extra entrainment, condensing plumes
    "delta_c": (2e-3, 5e-4, 5e-3, "1/m"),        # net detrainment, condensing plumes
    # (F) wave kernel
    "phi0": (0.3, 0.0, 0.6, "1"),                # radiated fraction of the reaction
    "lambda_w": (2000.0, 500.0, 6000.0, "m"),    # horizontal wavelength
    "alpha_bg": (1e-4, 1e-5, 1e-3, "1/s"),       # background wave damping rate
    # (E) memory
    "c_tau": (1.0, 0.0, 3.0, "1"),               # memory length / plume transit time
}

# --- fixed numerics (tier 0, never trained); parent's are reused by name ----------
_E_D = _base._E_D                # m2/s2   TKE regularisation in l and the sink
_N_D = _base._N_D                # 1/s     N^2 softplus scale
_N_TAIL_X = _base._N_TAIL_X      # 1       switch of N_+ to its exp(x/2) tail
_S_D = _base._S_D                # 1/s     shear floor in Ri
_R_D = _base._R_D                # 1       Ri softplus scale
_B_D = _base._B_D                # m2/s3   buoyancy-flux softplus scale
_W_D = _base._W_D                # m/s     plume velocity scale
_QC_D = _base._QC_D              # kg/kg   condensate scale in chi_c
_OM_D = _base._OM_D              # 1/s     frequency floor in the plume Froude ratio
_SHEAR_DIR_D = _base._SHEAR_DIR_D  # 1/s   shear scale in the wave direction
_U_D = _base._U_D                # m/s     Doppler-wind softplus scale
_U_PLUS_FRAC = _base._U_PLUS_FRAC  # 1     U_+ floor as a fraction of _U_D
_Q_D = _base._Q_D                # 1       m^2/k^2 softplus scale
_A_W = _base._A_W                # 1       buoyancy coefficient in the w_u^2 equation
_SIGMA_W_PREF = _base._SIGMA_W_PREF  # 1   sigma_w prefactor (parent, 1.3)
_ABSORB_FRAC = _base._ABSORB_FRAC    # 1   absorber base / column top
_E_INIT = _base._E_INIT          # m2/s2   initial TKE
_N_USTAR_ITERS = _base._N_USTAR_ITERS  # fixed MOST iterations
_TAU_CAP = _base._TAU_CAP        # 1       smooth per-cell optical-depth cap
_STABILITY = _base._STABILITY
_U_REG = 1e-3                    # m/s     u*^3 regulariser in 1/L (hypothesis value)
_SIGMA_0 = 1e-5                  # kg/kg   floor of sigma_s
_TAU_MIN_S = 60.0                # s       memory floor (0 = memory-free control)
_LINDZEN_P = 4.0                 # 1       p-norm smoothmin exponent (even: smooth at 0)
_Z0_LAW_RATIO = 1e-3             # 1       placeholder z0/z1 of the discarded own law
_LOG_R_FLOOR = -1.0e3            # 1       log(|F|/cap) of a zero flux (exp underflows to 0)

_CV = 1.0 / constants.epsilon - 1.0    # R_v/R_d - 1


def _ssqrt(x):
    """sqrt(x) for x > 0, 0 with zero gradient for x <= 0 (double-where)."""
    on = x > 0.0
    return jnp.where(on, jnp.sqrt(jnp.where(on, x, 1.0)), 0.0)


def _log_n_plus(N2):
    """log of the parent's N_+ (``_n_plus``), evaluated without forming N_+: on the deep
    unstable tail N_+ = N_d exp(x/2) underflows to 0 (log -> -inf, NaN gradients)."""
    x = N2 / _N_D ** 2
    hi = x > _N_TAIL_X
    x_hi = jnp.where(hi, x, _N_TAIL_X)
    x_lo = jnp.where(hi, _N_TAIL_X, x)
    return jnp.log(_N_D) + jnp.where(hi, 0.5 * jnp.log(jax.nn.softplus(x_hi)), 0.5 * x_lo)


def _lindzen_log_attenuation(log_B, has, log_cap, up):
    """log of the saturation attenuation P_jf in (0, 1] of launch j at face f.

    r_f = |B_f|/cap_f; along each wave's path (upward from launch j for faces f > j,
    downward for f <= j) R_f = max of r over the path so far (hard running max in
    log space, vote V16), and P_f = (1 + R_f^p)^(-1/p) is applied ONCE: a flux
    saturated at face f' stays below cap_f' times the raw transmission above it (no
    re-emission), and a constant r gives the pointwise smooth cap at every face (no
    compounding with the number of faces). B = 0 entries carry r = 0."""
    log_r = jnp.where(has, log_B - log_cap, _LOG_R_FLOOR)

    def scan(mask, reverse):
        def body(lR, xs):
            lr, mk = xs
            lR_new = jnp.where(mk, jnp.maximum(lR, lr), _LOG_R_FLOOR)
            return lR_new, jnp.where(mk, -jax.nn.softplus(_LINDZEN_P * lR_new) / _LINDZEN_P,
                                     0.0)
        _, out = jax.lax.scan(body, jnp.full(log_B.shape[0], _LOG_R_FLOOR, log_B.dtype),
                              (log_r.T, mask.T), reverse=reverse)
        return out.T
    return jnp.where(up, scan(up, False), scan(~up, True))


def _log_norm(B):
    """(log|B|, |B| > 0) over the last (2-vector) axis without squaring |B| (|B|^2
    underflows in fp32 near critical levels) and without a ratio (its JAX derivative
    forms b_max^-2, which overflows for tiny |B|; review round 3):
    log|B| = log b_max + softplus(2 (log b_min - log b_max))/2; log|B| = 0 where B = 0,
    the b_min term 0 where b_min = 0 (double-where). Subnormal components are not
    covered (1/b overflows)."""
    b_abs = jnp.abs(B)
    b_max = jnp.max(b_abs, axis=-1)
    b_min = jnp.min(b_abs, axis=-1)
    has = b_max > 0.0
    two = b_min > 0.0
    l_max = jnp.log(jnp.where(has, b_max, 1.0))
    l_min = jnp.log(jnp.where(two, b_min, 1.0))
    extra = jnp.where(two, 0.5 * jax.nn.softplus(2.0 * (l_min - l_max)), 0.0)
    return jnp.where(has, l_max + extra, 0.0), has


def _scbrt(x):
    """cbrt(x) for x > 0, 0 with zero gradient for x <= 0 (double-where)."""
    on = x > 0.0
    return jnp.where(on, jnp.cbrt(jnp.where(on, x, 1.0)), 0.0)


def init_aux(inputs):
    return {"e": jnp.full_like(inputs.theta_l, _E_INIT),
            "M": jnp.zeros_like(inputs.z_half),
            "started": jnp.zeros((), inputs.theta_l.dtype)}


def fluxes(params, inputs, aux):
    d = closure(params, inputs, aux)
    return d["flux_scalar"], d["flux_momentum"], d["aux"]


def closure(params, inputs, aux):
    """The closure; returns a dict with the fluxes, the new aux and diagnostics."""
    p = params
    inp = inputs
    g = constants.g
    kap = constants.kappa_von_karman
    n = inp.theta_l.shape[0]
    m = inp.mass
    dt = inp.dt
    e = aux["e"]

    # --- thermodynamics of the grid mean ------------------------------------------
    theta = inp.temp / inp.exner
    thv = theta * (1.0 + _CV * inp.q_v - inp.q_c)
    thv0 = thv[0]
    gb = g / thv0
    thv_face = jnp.concatenate([thv[:1], 0.5 * (thv[1:] + thv[:-1]), thv[-1:]])

    # --- (A) surface layer --------------------------------------------------------
    z1 = inp.z_full[0]
    # own law only: on a law column its values are discarded, so it is fed a fixed
    # placeholder roughness (any given z0, NaN or >= z1, would poison the gradient
    # through the discarded branch; votes V19, V20). LES columns use z0 as given.
    z0 = jnp.where(has_surface_law(inp), _Z0_LAW_RATIO * inp.z_full[0], inp.z0_m)
    U1sq = inp.u[0] ** 2 + inp.v[0] ** 2
    Ug = _ssqrt(U1sq + p["beta_g"] ** 2 * (2.0 / 3.0) * e[0])
    calm = Ug > 0.0
    inv_Ug = jnp.where(calm, 1.0 / jnp.where(calm, Ug, 1.0), 0.0)
    wthv_s = inp.w_theta_s * (1.0 + _CV * inp.q_v[0] - inp.q_c[0]) \
        + _CV * theta[0] * inp.w_qt_s
    Bs = gb * wthv_s
    ln_z = jnp.log(z1 / z0)

    def _ustar_iter(_, us):
        inv_L = -kap * Bs / (us ** 3 + _U_REG ** 3)
        den = ln_z - psi_m(z1 * inv_L, _STABILITY) + psi_m(z0 * inv_L, _STABILITY)
        return kap * Ug / den

    ustar_own = jax.lax.fori_loop(0, _N_USTAR_ITERS, _ustar_iter, kap * Ug / ln_z)
    c_own = ustar_own ** 2 * inv_Ug                               # kinematic drag [m/s]
    # explicit stress estimate at U_1^n (plume launch, G_c); a law column passes its own
    uw_ex, vw_ex, ustar = surface_stress(inp, -c_own * inp.u[0], -c_own * inp.v[0],
                                         ustar_own)
    # applied stress: passed explicitly on a law column, own drag implicit in U_1 else
    tau_law_u, tau_law_v, _ = surface_stress(inp, 0.0, 0.0, 0.0)
    C_drag = jnp.where(has_surface_law(inp), 0.0, c_own)
    G_c = _g_inf(p["L_c"] * kap * Bs / (ustar ** 3 + _U_REG ** 3))   # = g_inf(-L_c/L)
    sigma_w = _SIGMA_W_PREF * _scbrt(
        ustar ** 3 + kap * z1 * _B_D * jax.nn.softplus(Bs / _B_D))

    # --- (B) moist regime state on interior faces ----------------------------------
    dzf = inp.z_full[1:] - inp.z_full[:-1]
    zf = inp.z_half[1:-1]
    pf = inp.p_half[1:-1]
    exf = inp.exner_half[1:-1]
    T_l = inp.theta_l * inp.exner
    Tlf = 0.5 * (T_l[1:] + T_l[:-1])
    Tf = 0.5 * (inp.temp[1:] + inp.temp[:-1])
    thf = 0.5 * (theta[1:] + theta[:-1])
    qtf = 0.5 * (inp.q_t[1:] + inp.q_t[:-1])
    dthl = (inp.theta_l[1:] - inp.theta_l[:-1]) / dzf
    dqt = (inp.q_t[1:] - inp.q_t[:-1]) / dzf
    dTl = (T_l[1:] - T_l[:-1]) / dzf
    qs_l = saturation_mixing_ratio(Tlf, pf)
    dqs_l = saturation_mixing_ratio_dT(Tlf, pf)
    a_L = 1.0 / (1.0 + constants.L_v / constants.c_pd * dqs_l)
    s_sat = a_L * (qtf - qs_l)
    grad_s = a_L * (dqt - dqs_l * (dTl + g / constants.c_pd))
    l0f = 1.0 / (1.0 / (kap * zf) + 1.0 / p["lambda_inf"])
    sig_s = jnp.sqrt((p["c_sig"] * l0f * grad_s) ** 2 + _SIGMA_0 ** 2)
    f_c = 0.5 * (1.0 + erf(s_sat / (jnp.sqrt(2.0) * sig_s)))
    qs = saturation_mixing_ratio(Tf, pf)
    A_th_sat = ((1.0 - qtf + qs / constants.epsilon
                 * (1.0 + constants.L_v / (constants.R_v * Tf)))
                / (1.0 + constants.L_v ** 2 * qs / (constants.c_pd * constants.R_v * Tf ** 2)))
    A_q_sat = constants.L_v / (constants.c_pd * exf) * A_th_sat - thf
    A_th = (1.0 - f_c) * (1.0 + _CV * qtf) + f_c * A_th_sat
    A_q = (1.0 - f_c) * (_CV * thf) + f_c * A_q_sat
    N2f = gb * (A_th * dthl + A_q * dqt)                             # N_eff^2 faces

    # --- (C) local coefficients from e^n (interior faces 1..n-1) -------------------
    duf = (inp.u[1:] - inp.u[:-1]) / dzf
    dvf = (inp.v[1:] - inp.v[:-1]) / dzf
    S2f = duf ** 2 + dvf ** 2
    ef = 0.5 * (e[1:] + e[:-1])
    Pr_eq = p["Pr_0"] + p["Ri_eq"] / p["R_finf"]
    c_N = p["c_k"] * jnp.sqrt(p["Ri_eq"] / (1.0 - p["Ri_eq"] / Pr_eq))
    Npf, _ = _n_plus(N2f)
    lf = 1.0 / (1.0 / (kap * zf) + 1.0 / p["lambda_inf"]
                + Npf / (c_N * jnp.sqrt(ef + _E_D)))
    Kmf = p["c_k"] * lf * _ssqrt(ef)
    Ri = N2f / (S2f + _S_D ** 2)
    Ri_p = _R_D * jax.nn.softplus(Ri / _R_D)
    Pr_t = p["Pr_0"] + Ri_p / p["R_finf"]
    Khf = Kmf / Pr_t

    N2c = _face_to_cell(N2f)
    Npc, _ = _n_plus(N2c)
    lc = 1.0 / (1.0 / (kap * inp.z_full) + 1.0 / p["lambda_inf"]
                + Npc / (c_N * jnp.sqrt(e + _E_D)))
    Kmc = p["c_k"] * lc * _ssqrt(e)
    s_cell = _face_to_cell(jnp.stack([duf, dvf], axis=-1))          # (n, 2)

    # --- (D) plume (as parent) -----------------------------------------------------
    X = jnp.stack([inp.theta_l, inp.q_t, inp.u, inp.v], axis=-1)    # (n, 4)
    F0_launch = jnp.stack([inp.w_theta_s, inp.w_qt_s, uw_ex, vw_ex])
    on_w = sigma_w > 0.0       # sigma_w = 0: calm, strongly cooled surface (vote V18)
    Xu1 = X[0] + jnp.where(on_w, p["beta_u"] * F0_launch / jnp.where(on_w, sigma_w, 1.0),
                           0.0)
    th1, qv1, qc1 = saturation_adjust(Xu1[0], Xu1[1], inp.exner_half[1], inp.p_half[1])
    thv_u1 = th1 * (1.0 + _CV * qv1 - qc1)
    w2_1 = sigma_w ** 2
    T_1 = _g_inf(w2_1 / _W_D ** 2 - 1.0)
    chi_1 = qc1 / (qc1 + _QC_D)
    eps_0 = p["c_ent"] / inp.z_full[0] + p["eps_c"] * chi_1
    dz_cell = inp.z_half[1:] - inp.z_half[:-1]

    def _plume_step(carry, xs):
        Xu, w2, a, T, qc_u = carry
        Xe, s_e, z_e, dz, ex_t, p_t, thv_t = xs
        chi = qc_u / (qc_u + _QC_D)
        eps = p["c_ent"] / z_e + p["eps_c"] * chi
        decay = jnp.exp(-eps * dz)
        sc = Xe[:2] + (Xu[:2] - Xe[:2]) * decay
        mom = Xu[2:] + dz * (-eps * (Xu[2:] - Xe[2:]) + p["c_p"] * s_e)
        Xn = jnp.concatenate([sc, mom])
        th_u, qv_u, qc_n = saturation_adjust(sc[0], sc[1], ex_t, p_t)
        thv_u = th_u * (1.0 + _CV * qv_u - qc_n)
        B = gb * (thv_u - thv_t)
        w2n = _sp((w2 + 2.0 * _A_W * B * dz) / (1.0 + 2.0 * p["b_w"] * eps * dz))
        an = a * jnp.exp(-p["delta_c"] * chi * dz)
        Tn = T * _g_inf(w2n / _W_D ** 2 - 1.0)
        return (Xn, w2n, an, Tn, qc_n), (Xn, w2n, an, Tn, eps)

    xs = (X[1:], s_cell[1:], inp.z_full[1:], dz_cell[1:], inp.exner_half[2:],
          inp.p_half[2:], thv_face[2:])
    cdt = jnp.result_type(inp.theta_l, *p.values())
    a_u = jnp.asarray(p["a_u"], cdt)
    carry0 = (Xu1.astype(cdt), jnp.asarray(w2_1, cdt), a_u, jnp.asarray(T_1, cdt),
              jnp.asarray(qc1, cdt))
    _, (Xu_s, w2_s, a_s, T_s, eps_s) = jax.lax.scan(_plume_step, carry0, xs)

    Xu_f = jnp.concatenate([Xu1[None], Xu1[None], Xu_s], axis=0)    # (n+1, 4)
    w2_f = jnp.concatenate([w2_1[None], w2_1[None], w2_s])
    a_f = jnp.concatenate([jnp.broadcast_to(a_u, (2,)), a_s])
    T_f = jnp.concatenate([T_1[None], T_1[None], T_s])
    eps_c_arr = jnp.concatenate([eps_0[None], eps_s])                 # (n,)
    edge = jnp.ones(n + 1, w2_f.dtype).at[0].set(0.0).at[n].set(0.0)
    M_star = edge * G_c * a_f * _ssqrt(w2_f) * T_f                 # (n+1,) >= 0

    # --- (E) memory: exact exponential kernel over the step -----------------------
    T_top = jnp.sum(T_f[1:] * dz_cell / jnp.sqrt(w2_f[1:] + _W_D ** 2))
    tau = p["c_tau"] * T_top + _TAU_MIN_S
    live = tau > 0.0
    keep = jnp.where(live, jnp.exp(-dt / jnp.where(live, tau, 1.0)), 0.0)
    M_mem = M_star + (aux["M"] - M_star) * keep
    M = jnp.where(aux["started"] > 0.0, M_mem, M_star)

    F_mf_u = M[1:-1, None] * Xu_f[1:-1]                              # explicit plume part
    wthv_mf = M[1:-1] * (A_th * (Xu_f[1:-1, 0] - inp.theta_l[1:])
                         + A_q * (Xu_f[1:-1, 1] - inp.q_t[1:]))

    # --- (F) wave kernel (momentum only) -------------------------------------------
    M_c = 0.5 * (M[:-1] + M[1:])
    wu_c = 0.5 * (_ssqrt(w2_f[:-1]) + _ssqrt(w2_f[1:]))
    r = N2c / ((eps_c_arr * wu_c) ** 2 + _OM_D ** 2)
    S = _g_inf(r - 1.0)
    D = (p["phi0"] * S * p["c_p"] * M_c)[:, None] * s_cell           # (n, 2) m/s2
    nvec = s_cell / jnp.sqrt(jnp.sum(s_cell ** 2, axis=-1, keepdims=True)
                             + _SHEAR_DIR_D ** 2)
    c_ph = 0.5 * (Xu_f[:-1, 2:] + Xu_f[1:, 2:])                      # (n, 2)
    U = X[:, 2:]
    Upar = jnp.einsum("jic,jc->ji", U[None, :, :] - c_ph[:, None, :], nvec)
    U_p = _U_D * jax.nn.softplus(Upar / _U_D) + _U_PLUS_FRAC * _U_D
    k = 2.0 * jnp.pi / p["lambda_w"]
    G2 = (Npc[None, :] / (k * U_p)) ** 2
    q = _Q_D * jax.nn.softplus((G2 - 1.0) / _Q_D)
    log_tau = (jnp.log(p["alpha_bg"] + Kmc[None, :] * k ** 2 * q) + jnp.log1p(q)
               - jnp.log(U_p) - 0.5 * jnp.log(q) + jnp.log(dz_cell)[None, :])
    tau_w = _TAU_CAP * jax.nn.sigmoid(log_tau - jnp.log(_TAU_CAP))
    cs_ex = jnp.concatenate([jnp.zeros((n, 1), tau_w.dtype), jnp.cumsum(tau_w, axis=1)],
                            axis=1)
    jj = jnp.arange(n)
    tau_jj = tau_w[jj, jj]
    ff = jnp.arange(n + 1)
    up = ff[None, :] > jj[:, None]
    od_up = 0.5 * tau_jj[:, None] + cs_ex - cs_ex[jj, jj + 1][:, None]
    od_dn = 0.5 * tau_jj[:, None] + cs_ex[jj, jj][:, None] - cs_ex
    od = jnp.where(up, od_up, od_dn)
    kern = jnp.where(up, -jnp.exp(-od), jnp.exp(-od))                 # (n, n+1)
    # per-launch momentum face flux rho F (n_j, n+1, 2), then the Lindzen bound
    B_jf = 0.5 * (m[:, None, None] * kern[:, :, None] * D[:, None, :])
    U_face = jnp.concatenate([U[:1], 0.5 * (U[1:] + U[:-1]), U[-1:]], axis=0)
    Upar_f = jnp.einsum("jfc,jc->jf", U_face[None, :, :] - c_ph[:, None, :], nvec)
    Up_f = _U_D * jax.nn.softplus(Upar_f / _U_D) + _U_PLUS_FRAC * _U_D
    logN_f = _log_n_plus(N2f)
    logN_face = jnp.concatenate([logN_f[:1], logN_f, logN_f[-1:]])
    log_cap = ((jnp.log(0.5 * inp.rho_half * k) - logN_face)[None, :]
               + 3.0 * jnp.log(Up_f))                                # log(rho k U^3/(2N))
    log_B, has_f = _log_norm(B_jf)
    log_P = _lindzen_log_attenuation(log_B, has_f, log_cap, up)
    F_jf = B_jf * jnp.exp(log_P)[:, :, None]                        # capped rho F
    rhoFw = jnp.sum(F_jf, axis=0)
    rhoFw = rhoFw.at[0].set(0.0)                    # downward remainder stays in cell 0
    Fw_raw = rhoFw / inp.rho_half[:, None]
    z_base = _ABSORB_FRAC * inp.z_half[-1]
    Fw_u = absorb_top(Fw_raw[:, 0], inp.z_half, z_base)
    Fw_v = absorb_top(Fw_raw[:, 1], inp.z_half, z_base)

    # --- (G) implicit ED + implicit-upwind MF + explicit wave, implicit drag -------
    F_expl = F_mf_u + jnp.stack([jnp.zeros(n - 1, F_mf_u.dtype),
                                 jnp.zeros(n - 1, F_mf_u.dtype),
                                 Fw_u[1:-1], Fw_v[1:-1]], axis=-1)    # (n-1, 4)
    F0_expl = jnp.stack([inp.w_theta_s, inp.w_qt_s, tau_law_u, tau_law_v])
    zero1 = jnp.zeros((1, 4), F_mf_u.dtype)
    one_c = jnp.ones_like(inp.exner)
    one_h = jnp.ones_like(inp.exner_half)
    w_c = jnp.stack([inp.exner, one_c, one_c, one_c], axis=0)          # (4, n)
    w_h = jnp.stack([inp.exner_half, one_h, one_h, one_h], axis=-1)    # (n+1, 4)
    E = (inp.rho_half[:, None] * w_h
         * jnp.concatenate([F0_expl[None], F_expl, zero1], axis=0))
    Kf = jnp.stack([Khf, Khf, Kmf, Kmf], axis=0)
    kc = inp.rho_half[None, 1:-1] * w_h[1:-1].T * Kf / dzf[None, :]
    z4 = jnp.zeros((4, 1), kc.dtype)
    k_lo = jnp.concatenate([z4, kc], axis=1)
    k_hi = jnp.concatenate([kc, z4], axis=1)
    mc = inp.rho_half[None, 1:-1] * w_h[1:-1].T * M[None, 1:-1]
    mc_lo = jnp.concatenate([z4, mc], axis=1)
    mc_hi = jnp.concatenate([mc, z4], axis=1)
    drag = jnp.zeros((4, n), kc.dtype).at[2:, 0].set(inp.rho_half[0] * C_drag)
    stor = (m / dt)[None, :] * w_c
    diag = stor + k_lo + k_hi + mc_lo + drag
    rhs = stor * X.T - (E[1:] - E[:-1]).T
    X_new = thomas_solve(-k_lo, diag, -(k_hi + mc_hi), rhs)          # (4, n)
    F_ed = -Kf * (X_new[:, 1:] - X_new[:, :-1]) / dzf[None, :]
    F_int = F_ed + F_expl.T - M[None, 1:-1] * X_new[:, 1:]

    def _full(s0, interior):
        return jnp.concatenate([jnp.reshape(s0, (1,)), interior,
                                jnp.zeros((1,), interior.dtype)])

    w_theta = _full(inp.w_theta_s, F_int[0])
    w_qt = _full(inp.w_qt_s, F_int[1])
    u_w = _full(tau_law_u - C_drag * X_new[2, 0], F_int[2])
    v_w = _full(tau_law_v - C_drag * X_new[3, 0], F_int[3])

    # --- TKE: implicit, M-matrix, positive by construction (as parent) ------------
    P_face = jnp.concatenate([jnp.reshape(ustar ** 3 / (kap * z1), (1,)),
                              Kmf * S2f, jnp.zeros((1,), S2f.dtype)])
    Pb_face = jnp.concatenate([jnp.reshape(Bs, (1,)),
                               -Khf * N2f + gb * wthv_mf, jnp.zeros((1,), S2f.dtype)])
    P_s = 0.5 * (P_face[:-1] + P_face[1:])
    P_b = 0.5 * (Pb_face[:-1] + Pb_face[1:])
    P_pos = 0.5 * (P_b + jnp.sqrt(P_b ** 2 + _B_D ** 2))
    P_neg = P_pos - P_b
    sink = P_neg / (e + _E_D) + p["c_k"] ** 3 * jnp.sqrt(e + _E_D) / lc
    ke = inp.rho_half[1:-1] * Kmf / dzf
    ke_lo = jnp.concatenate([jnp.zeros((1,), ke.dtype), ke])
    ke_hi = jnp.concatenate([ke, jnp.zeros((1,), ke.dtype)])
    e_new = thomas_solve(-ke_lo, m / dt + m * sink + ke_lo + ke_hi, -ke_hi,
                         m * e / dt + m * (P_s + P_pos))

    aux_new = {"e": e_new, "M": M, "started": jnp.ones((), inp.theta_l.dtype)}
    return {"flux_scalar": (w_theta, w_qt), "flux_momentum": (u_w, v_w),
            "aux": aux_new, "X_new": X_new, "M_star": M_star, "M": M, "tau": tau,
            "T_top": T_top, "f_c": f_c, "N2_eff": N2f, "c_N": c_N, "S_gate": S,
            "F_wave_launch": F_jf, "B_wave_launch": B_jf, "wave_log_P": log_P,
            "log_cap": log_cap,
            "F_wave": jnp.stack([Fw_u, Fw_v], axis=-1), "C_drag": C_drag,
            "A_th": A_th, "A_q": A_q, "sigma_w": sigma_w,
            "U_plus_face": Up_f, "logN_face": logN_face}
