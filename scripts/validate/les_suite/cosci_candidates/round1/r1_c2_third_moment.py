"""Co-scientist round 1, candidate C2_third_moment (EXPERIMENTAL plug-in, KIND="both").

Advective second-moment closure. Its transport velocity comes from a diagnosed
plume-isolation number Pi = dw^2/(4 w'w'), with V = sqrt(dw^2 - 4 w'w') smoothed.
One upward-upwind operator transports theta_l, q_t, u and v and the scalar
(co)variances. Momentum carries LRR pressure terms and a pressure-drag efficiency
gamma_u. The surface stress comes from a plume-gust MOST law (Beljaars 1995 form).

Contract (packages/atmosphere/legoesm/atmosphere/les_suite/candidate.py)
------------------------------------------------------------------------
* Interfaces k = 0 (surface) .. n (top); full levels j = 0..n-1 between k=j and
  k=j+1. Fluxes are KINEMATIC, positive UP. Scalar fluxes are [K m/s] and [m/s];
  momentum fluxes u'w', v'w' are [m2/s2].
* Index 0: w'theta_l' = inputs.w_theta_s and w'q_t' = inputs.w_qt_s, bit-exact.
  u'w', v'w' = this candidate's own surface stress, -u*^2 (U_1, V_1)/U_eff (drag,
  opposite to the first-level wind). inputs.uw_s / vw_s are never read.
* Index n is exactly 0 for all four fluxes. The momentum flux profile goes through
  ``absorb_top`` (cos^2 taper from Z_ABSORB_FRAC * z_top to the lid), so any
  momentum reaching the top is deposited in the mean flow in flux form.
* Conservation: the harness converts every flux to a tendency in flux form
  (-d(rho F)/dm), so column theta_l (applied as enthalpy), q_t, u and v change
  exactly by the bottom flux. Every smooth floor here acts on DIAGNOSTIC quantities
  (B+-, N+-, P+-, dw, V, variances, correlation, D+), never on a conserved store.
  There is no post-solve flux limiter.
* Energy ledger: heat goes in as an enthalpy flux (column c_p T exact), and the
  harness ``ke_heating`` returns the interior mean-KE loss as heat. TKE ``e`` is a
  SCALE variable outside the total-energy ledger. Its buoyancy production B is
  neither removed from enthalpy nor returned as dissipation heat; its shear
  production is NOT heated here, because the harness already heats the same mean-KE
  loss. So c_p T + L q + |u|^2/2 closes against the boundary inputs with no
  double count (harness e0 check: energy_rel with ke_heating on vs off).

Which buoyancy enters where (review fix 3)
------------------------------------------
* Plume velocity dw (eq 1) and the w'w'/e partition R (eq 0): LAGGED interior
  buoyancy B = beta(A F_thl^{n-1} + Bq F_qt^{n-1}) and lagged shear production. At
  the surface: the CURRENT prescribed B_s, plus the lagged surface-stress work.
* Covariance production (eq 2): lagged fluxes, current mean gradients.
* Momentum buoyancy-pressure term (eq 4): CURRENT F_thv = F_thl + Bq F_qt at the
  same level, solved just before in the same upward sweep.
* N^2: current mean profiles. TKE (eq 6): current fluxes. Surface law: current B_s.

Initial and boundary specification (review fix 3)
-------------------------------------------------
* t = 0: e = 0 (the diagnostics use e + E_MIN), and lagged interior fluxes = 0, so
  production starts at 0. Lagged surface scalar fluxes = prescribed values; lagged
  surface stress = 0.
* Sweep inflow at k = 0: dw_0 = c_0 (B+_s l_inf)^{1/3}, V_0 from dw_0 and
  w2_0 = e_0 c_w(R_0), F_0 = the surface fluxes / stress. Variances and covariance
  v_0 = cov_0 = 0. Reason: scalar third moments V*v vanish at a wall, and the
  top-hat alternative F_s^2/w2_0 diverges at cold start, where e = 0.

Limits (derived in the hypothesis; reproduced by construction here)
-------------------------------------------------------------------
* Neutral: C_u = (1-C2) c_w0 c_ed^{1/3}, so K_m = kappa z u* and Pr0 =
  C_phi/(c_w0 c_ed^{1/3}) (0.89 at the defaults).
* Free convection: Deardorff w*, z_i scaling through dw_0 ~ (B_s l_inf)^{1/3}.
* BROKEN, declared:
  (i) negative skewness (V >= 0 is upward only);
  (ii) Pr_t FALLS with Ri at the defaults: Pr0 = 0.89, Pr_inf = (1-C2)(a_h-b)/a_m
      = 0.72. The defaults are NOT changed (PI ruling pending), and the GABLS
      penalty is accepted;
  (iii) the near-surface free-convection -1/3 law (Paulson psi_m is used);
  (iv) radiated gravity-wave momentum flux;
  (v) cloud-layer buoyancy uses the DRY coefficients (A = 1, Bq = (R_v/R_d - 1)
      theta). The plug-in receives no host SGS-condensation scheme or cloud
      fraction, and the roster precedent (mynn25) also has no cloud-conditional
      buoyancy. So in the BOMEX cloud layer, B < 0 brakes dw wherever the
      theta_l flux is negative.

Implementation deviations from the hypothesis (each one recorded)
---------------------------------------------------------------
* psi_m is the repo's Businger-Dyer/Paulson ``legoesm.core.bulk_flux.psi_m``
  ("dyer1974"; gamma_m traced, beta_m = BETA_M). It splits the branches with a
  HARD ``where`` at zeta = 0 (continuous value, kinked slope) instead of the
  hypothesis' smooth split of width 0.01, and it clips zeta to [-10, 10], which
  gives a zero gradient beyond that range.
* Diagnostics use e + E_MIN (new tier-0 regulariser) so that sqrt(e) is
  differentiable at e = 0. The prognostic e itself is not floored.
* The variance production split P+- uses the scale v_reg/tau_v, which the
  hypothesis does not specify.
* Realised covariance c = r vt vq / sqrt(vt vq + c_reg), not r sqrt(vt vq). The
  two are equal for vt vq >> c_reg, and this form avoids a NaN gradient at 0.
* N^2 is beta times the grid-mean theta_v gradient, as the hypothesis says. The
  plume B uses the dry coefficients (see (v) above).
* U_eff carries an extra U_REG^2 inside the square root, so U_eff > 0 in calm,
  stable air: without it the stress would be 0/0 there. This changes U_eff by
  less than 1e-3 m/s.
* sqrt(N+^2) and (B+_s)^{1/3} are evaluated through an underflow-safe log-softplus
  (Y_TAIL), so their gradients stay finite where the softplus underflows. The
  values are unchanged.
* Parameters are cast to the column dtype, so the scan carry and aux keep the
  column precision.
* Requires n >= 2 levels.
* TKE source S = P + B is split as S+ = B0 sp(S/B0) (explicit) and S- = B0 sp(-S/B0)
  (implicit, as S- e_new/(e_old + E_SINK)). This keeps e_new >= 0. In steady
  state it differs from S by S- E_SINK/(e + E_SINK).
* The covariance sweep transports the REALISED (bounded) covariance c from k-1
  to k, as the variance sweep transports the realised v. The hypothesis' "same
  recursion" is ambiguous on whether it carries c* or c.
* TKE production per layer is the mean of its two bounding interfaces.
* The TKE dissipation length at full levels uses N+ averaged from the bounding
  interfaces; the lowest and highest levels copy the nearest interface value. The
  surface interface carries the stress work -(F_u0 U_1 + F_v0 V_1)/z_1.
* The absorbing layer for momentum spans the top (1 - Z_ABSORB_FRAC) of the column.
* SUB-STEPS (user 2026-10-30): the SCM steps every 10 min, and this explicit operator
  blows up there (12 of 14 cases within 6-12 steps at default parameters). It takes
  N_SUB = 60 internal steps of 10 s per SCM step (``candidate.substepped``) and
  reports their mean flux. Measured at default parameters on the 14 scored cases:
  sub-steps of 300, 120, 60 and 30 s blew up in 12, 11, 5 and 1 cases; 10 s in none.
  KNOWN DIFFERENCE (PI 2026-10-30, round 1): the reported flux is the mean over the
  SCM step, and scoring compares the applied flux with the LES snapshot at the step's
  start (as for the implicit candidates).

Diagnostics (review fix 6)
--------------------------
``aux["dw"]`` (plume velocity [m/s]) and ``aux["Pi"]`` (isolation number) hold
the last step's interface profiles. The BOMEX check, F_thl at 1-1.3 km vs the LES
-0.024 K m/s, reads the recorded w'theta_l' flux.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.atmosphere.les_suite.candidate import absorb_top, substepped
from legoesm.atmosphere.physics._shared import buoyancy_coefficient, virtual_temperature
from legoesm.core.bulk_flux import psi_m
from legoesm.timestepping.tridiagonal import thomas_solve

KIND = "both"

# gamma_u = sigmoid(gamma_u_logit), so 0 < gamma_u < 1 by construction (review fix 5).
_GAMMA_U_DEFAULT = 0.7
_GAMMA_U_LOGIT_DEFAULT = math.log(_GAMMA_U_DEFAULT / (1.0 - _GAMMA_U_DEFAULT))

PARAMS = {
    # --- pressure correlations (LRR / Moeng-Wyngaard) ---
    "C2": (0.65, 0.4, 0.85, "1"),          # rapid shear-pressure on momentum
    "C3": (0.4, 0.05, 0.9, "1"),           # buoyancy pressure-scalar
    "C_B": (0.3, 0.0, 0.6, "1"),           # buoyancy pressure on u'w'
    "C_phi": (0.8, 0.4, 2.0, "1"),         # Rotta slow scalar term
    "C_v": (1.5, 0.5, 4.0, "1"),           # scalar (co)variance dissipation
    "C_ut": (2.0, 0.5, 4.0, "1"),          # u'theta_v' dissipation
    # --- TKE and its partition ---
    "c_w0": (0.45, 0.3, 0.6, "1"),         # w'w'/e under shear
    "c_w1": (1.1, 0.9, 1.3, "1"),          # w'w'/e under pure buoyancy
    "c_ed": (8.0, 5.0, 13.0, "1"),         # TKE dissipation e^{3/2}/(c_ed l)
    "c_k": (0.2, 0.05, 0.5, "1"),          # down-gradient TKE transport
    "c_T": (0.6, 0.3, 1.0, "1"),           # plume TKE flux fraction V w2/2
    # --- mixing length ---
    "alpha_l": (0.3, 0.1, 0.6, "1"),       # l_inf scale
    "c_N": (0.75, 0.3, 1.0, "1"),          # stable length sqrt(e)/N
    # --- plume ---
    "c_b": (1.0, 0.3, 1.5, "1"),           # plume buoyancy efficiency
    "eps_p": (0.05, 0.0, 0.5, "1"),        # plume drag per mixing length
    "c_0": (1.0, 0.5, 2.0, "1"),           # launch dw_0 = c_0 (B_s l_inf)^{1/3}
    "gamma_u_logit": (_GAMMA_U_LOGIT_DEFAULT, -6.0, 6.0, "1"),  # momentum 3rd-moment eff.
    # --- surface law ---
    "c_g": (1.0, 0.5, 1.5, "1"),           # plume-gust factor
    "gamma_m": (16.0, 14.0, 20.0, "1"),    # Paulson unstable coefficient
}

# --- frozen (tier 0) ---
# Stable psi_m slope. It has ZERO gradient on all 14 scored cases (no stable
# surface layer), so it is held out of PARAMS until GABLS3/4 finish.
BETA_M = 5.0
# --- fixed regularisers (tier 0; bounds on their effect are in the hypothesis) ---
B0 = 1e-5             # [m2 s-3] buoyancy / production / TKE-source split scale
N0 = 1e-3             # [s-1]    N^2 split scale (as N0^2)
DELTA_PI = 0.05       # [-]      isolation-number smoothing width
DW_REG = 0.01         # [m s-1]  plume-brake denominator regulariser
V_REG_THL = 1e-6      # [K2]     theta_l variance regulariser
V_REG_QT = 1e-12      # [(kg/kg)2] q_t variance regulariser
C_REG = V_REG_THL * V_REG_QT   # [K2 (kg/kg)2] correlation regulariser
U_REG = 1e-3          # [m s-1]  Obukhov-length regulariser
D_MIN = 1.0           # [-]      log-law denominator floor scale
N_SURFACE_ITER = 4    # static fixed-point pass count of the surface law
# DEVIATION (not in the hypothesis): the diagnostics use e + E_MIN, so sqrt(e)
# stays differentiable at e = 0. This is not a floor on the prognostic e.
E_MIN = 1e-4          # [m2 s-2]
# DEVIATION (unspecified by the hypothesis): the absorbing layer for top-reaching
# momentum flux spans the top (1 - Z_ABSORB_FRAC) of the column.
Z_ABSORB_FRAC = 0.8   # [-]
# Switch point of the underflow-safe log-softplus (numerics only, no physics).
Y_TAIL = -30.0        # [-]
# Denominator floor of the implicit negative-TKE-source term S- e_new/(e_old + E_SINK).
# The steady source error is S- E_SINK/(e + E_SINK), which matters only where e ~ E_SINK.
E_SINK = 1e-10        # [m2 s-2]

_VIRT = 1.0 / constants.epsilon - 1.0   # R_v/R_d - 1


def _split(x, scale):
    """Smooth positive/negative parts, both >= 0 in floating point:
    x+ = s sp(x/s), x- = s sp(-x/s), with x+ - x- = x."""
    return scale * jax.nn.softplus(x / scale), scale * jax.nn.softplus(-x / scale)


def _log_softplus(y):
    """log(sp(y)), finite in value AND gradient for every y. In float64, sp(y)
    underflows to 0 for y < ~-745, and log/sqrt/cbrt of that 0 give an inf
    derivative times a 0 softplus slope = NaN. Below Y_TAIL, log sp(y) = y to
    within exp(Y_TAIL)/2 relative."""
    hi = jnp.where(y < Y_TAIL, Y_TAIL, y)     # safe argument; exact slope at the switch
    return jnp.where(y < Y_TAIL, y, jnp.log(jax.nn.softplus(hi)))


def _avg(x):
    return 0.5 * (x[1:] + x[:-1])


def _mixing_length(z, n_plus, sqrt_e, l_inf, p):
    """1/l = 1/(kappa z) + 1/(alpha_l l_inf) + N+/(c_N e^{1/2})."""
    inv = (1.0 / (constants.kappa_von_karman * z) + 1.0 / (p["alpha_l"] * l_inf)
           + n_plus / (p["c_N"] * sqrt_e))
    return 1.0 / inv


def _transport_velocity(dw, w2):
    """Pi = dw^2/(4 w2); V = 2 w2^{1/2} [delta_Pi sp((Pi-1)/delta_Pi)]^{1/2} >= 0."""
    pi_iso = dw ** 2 / (4.0 * w2)
    v = 2.0 * jnp.sqrt(w2) * jnp.sqrt(DELTA_PI * jax.nn.softplus((pi_iso - 1.0) / DELTA_PI))
    return v, pi_iso


def init_aux(inputs):
    n = inputs.theta_l.shape[0]
    zh = jnp.zeros((n + 1,), inputs.theta_l.dtype)
    return {
        "e": jnp.zeros_like(inputs.theta_l),                       # TKE [m2 s-2]
        "F_thl": zh.at[0].set(inputs.w_theta_s),                   # lagged fluxes
        "F_qt": zh.at[0].set(inputs.w_qt_s),
        "F_u": zh,
        "F_v": zh,
        "dw": zh,                                                  # diagnostic
        "Pi": zh,                                      # diagnostic
    }


def _fluxes_one_step(params, inputs, aux):
    dtype = inputs.theta_l.dtype
    p = {k: jnp.asarray(v, dtype) for k, v in params.items()}
    kappa = constants.kappa_von_karman
    gamma_u = jax.nn.sigmoid(p["gamma_u_logit"])
    C_u = (1.0 - p["C2"]) * p["c_w0"] * jnp.cbrt(p["c_ed"])   # log-law constraint

    zf, zh = inputs.z_full, inputs.z_half
    dz_lay = zh[1:] - zh[:-1]          # (n,)  interface k-1 -> k, k = 1..n
    dz_f = zf[1:] - zf[:-1]            # (n-1,) between full levels (interior interfaces)
    z_int = zh[1:-1]

    theta = inputs.temp / inputs.exner
    thv = (virtual_temperature(inputs.temp, inputs.q_v) - inputs.temp * inputs.q_c) / inputs.exner
    beta_i = buoyancy_coefficient(_avg(thv))            # (n-1,) g/theta_v at interfaces
    bq_i = _VIRT * _avg(theta)                          # dry Bq; A = 1 (deviation (v))
    beta_s = buoyancy_coefficient(thv[0])
    bq_s = _VIRT * theta[0]

    g_thl = (inputs.theta_l[1:] - inputs.theta_l[:-1]) / dz_f
    g_qt = (inputs.q_t[1:] - inputs.q_t[:-1]) / dz_f
    g_u = (inputs.u[1:] - inputs.u[:-1]) / dz_f
    g_v = (inputs.v[1:] - inputs.v[:-1]) / dz_f
    n2 = beta_i * (thv[1:] - thv[:-1]) / dz_f           # grid-mean dTheta_v/dz
    n2p, n2m = _split(n2, N0 ** 2)
    log_n2p = jnp.log(N0 ** 2) + _log_softplus(n2 / N0 ** 2)
    n_plus = jnp.exp(0.5 * log_n2p)                     # sqrt(N+^2), NaN-free gradient

    # --- (0) shared fields from the lagged state ---
    e_old = aux["e"]
    e_eff = e_old + E_MIN
    se_f = jnp.sqrt(e_eff)
    se_h = jnp.sqrt(_avg(e_eff))
    l_inf = jnp.sum(zf * se_f * dz_lay) / jnp.sum(se_f * dz_lay)

    Fthl_l, Fqt_l = aux["F_thl"][1:-1], aux["F_qt"][1:-1]
    Fu_l, Fv_l = aux["F_u"][1:-1], aux["F_v"][1:-1]
    bp, bm = _split(beta_i * (Fthl_l + bq_i * Fqt_l), B0)
    pp, _ = _split(-(Fu_l * g_u + Fv_l * g_v), B0)
    cw = p["c_w0"] + (p["c_w1"] - p["c_w0"]) * bp / (bp + pp + B0)
    w2 = _avg(e_eff) * cw
    ell = _mixing_length(z_int, n_plus, se_h, l_inf, p)
    tau_phi = ell / (p["C_phi"] * se_h)
    tau_v = ell / (p["C_v"] * se_h)
    tau_ut = ell / (p["C_ut"] * se_h)
    tau_u = ell / (C_u * se_h)

    # covariance production from lagged fluxes (current gradients)
    p_tt = -2.0 * Fthl_l * g_thl
    p_qq = -2.0 * Fqt_l * g_qt
    p_tq = -(Fthl_l * g_qt + Fqt_l * g_thl)
    ptt_p, ptt_m = _split(p_tt, V_REG_THL / tau_v)
    pqq_p, pqq_m = _split(p_qq, V_REG_QT / tau_v)

    # --- (5) surface law: plume-gust MOST, static fixed point ---
    b_s = beta_s * (inputs.w_theta_s + bq_s * inputs.w_qt_s)
    bp_s, _ = _split(b_s, B0)
    # c_0 (B+_s l_inf)^{1/3} via logs: finite gradient when B+_s underflows (stable)
    dw0 = p["c_0"] * jnp.exp((jnp.log(B0 * l_inf) + _log_softplus(b_s / B0)) / 3.0)
    u1, v1, z1 = inputs.u[0], inputs.v[0], zf[0]
    z0 = inputs.z0_m
    # U_REG keeps U_eff > 0 in calm stable air (deviation; < 1e-3 m/s effect)
    u_eff = jnp.sqrt(u1 ** 2 + v1 ** 2 + (p["c_g"] * dw0) ** 2 + U_REG ** 2)
    log_z = jnp.log(z1 / z0)

    def psi(zeta):
        return psi_m(zeta, "dyer1974", unstable_gamma=p["gamma_m"], stable_beta=BETA_M)

    ust = kappa * u_eff / log_z
    for _ in range(N_SURFACE_ITER):
        zeta = -kappa * z1 * b_s / (ust ** 3 + U_REG ** 3)
        d_log = log_z - psi(zeta) + psi(zeta * z0 / z1)
        ust = kappa * u_eff / (D_MIN * jax.nn.softplus(d_log / D_MIN))
    fu0 = -ust ** 2 * u1 / u_eff
    fv0 = -ust ** 2 * v1 / u_eff

    # surface plume state (inflow of the sweep)
    pp_0, _ = _split(-(aux["F_u"][0] * u1 + aux["F_v"][0] * v1) / z1, B0)
    w2_0 = e_eff[0] * (p["c_w0"] + (p["c_w1"] - p["c_w0"]) * bp_s / (bp_s + pp_0 + B0))
    v_0, pi_0 = _transport_velocity(dw0, w2_0)

    # --- (1)-(4) one upward sweep over interior interfaces k = 1..n-1 ---
    def up(carry, x):
        dw_m, V_m, vt_m, vq_m, c_m, ft_m, fq_m, fu_m, fv_m = carry
        (dz, bp_k, bm_k, w2_k, l_k, tphi, tv, tut, tu, n2p_k, n2m_k, beta_k, bq_k,
         gt, gq, gu, gv, pttp, pttm, pqqp, pqqm, ptq, fu_lag, fv_lag) = x
        # (1) plume velocity, implicit upwind, negative buoyancy as a linear brake
        dw = (dw_m + dz * p["c_b"] * bp_k / w2_k) / (
            1.0 + dz * (p["eps_p"] / l_k + p["c_b"] * bm_k / (w2_k * (dw_m + DW_REG))))
        V, pi_k = _transport_velocity(dw, w2_k)
        adv = V / dz
        # (2) variances and covariance, same advective operator
        vt = (V_m * vt_m / dz + pttp) / (1.0 / tv + adv + pttm / (vt_m + V_REG_THL))
        vq = (V_m * vq_m / dz + pqqp) / (1.0 / tv + adv + pqqm / (vq_m + V_REG_QT))
        c_star = (V_m * c_m / dz + ptq) / (1.0 / tv + adv)
        prod = vt * vq
        r = jnp.tanh(c_star / jnp.sqrt(prod + C_REG))
        # |c| <= sqrt(vt vq) and differentiable at prod = 0 (deviation: the
        # hypothesis writes r sqrt(vt vq); equal for prod >> C_REG)
        c = r * prod / jnp.sqrt(prod + C_REG)
        thv_t = vt + bq_k * c                       # <theta_l' theta_v'>, A = 1
        thv_q = c + bq_k * vq                       # <q_t' theta_v'>
        # (3) scalar fluxes
        den_s = 1.0 / tphi + adv
        ft = (V_m * ft_m / dz - w2_k * gt + (1.0 - p["C3"]) * beta_k * thv_t) / den_s
        fq = (V_m * fq_m / dz - w2_k * gq + (1.0 - p["C3"]) * beta_k * thv_q) / den_s
        f_thv = ft + bq_k * fq
        # (4) momentum, LRR pressure terms (beta restored on F_thv: review fix 1)
        den_m = 1.0 / tu + (1.0 - p["C_B"]) * tut * n2p_k + gamma_u * adv
        buoy_shear = (1.0 - p["C_B"]) * (1.0 - p["C2"]) * tut * beta_k * f_thv
        fu = (gamma_u * V_m * fu_m / dz - (1.0 - p["C2"]) * w2_k * gu - buoy_shear * gu
              + (1.0 - p["C_B"]) * tut * n2m_k * fu_lag) / den_m
        fv = (gamma_u * V_m * fv_m / dz - (1.0 - p["C2"]) * w2_k * gv - buoy_shear * gv
              + (1.0 - p["C_B"]) * tut * n2m_k * fv_lag) / den_m
        return (dw, V, vt, vq, c, ft, fq, fu, fv), (dw, V, pi_k, ft, fq, fu, fv)

    zero = jnp.zeros_like(dw0)
    carry0 = (dw0, v_0, zero, zero, zero, inputs.w_theta_s, inputs.w_qt_s, fu0, fv0)
    xs = (dz_lay[:-1], bp, bm, w2, ell, tau_phi, tau_v, tau_ut, tau_u, n2p, n2m, beta_i,
          bq_i, g_thl, g_qt, g_u, g_v, ptt_p, ptt_m, pqq_p, pqq_m, p_tq, Fu_l, Fv_l)
    _, (dw_i, V_i, pi_i, ft_i, fq_i, fu_i, fv_i) = lax.scan(up, carry0, xs)

    def full(s0, interior):
        return jnp.concatenate([jnp.reshape(s0, (1,)), interior, jnp.zeros((1,), interior.dtype)])

    F_thl = full(inputs.w_theta_s, ft_i)
    F_qt = full(inputs.w_qt_s, fq_i)
    z_base = Z_ABSORB_FRAC * zh[-1]
    F_u = absorb_top(full(fu0, fu_i), zh, z_base)
    F_v = absorb_top(full(fv0, fv_i), zh, z_base)

    # --- (6) TKE: flux form, implicit, M-matrix (e_new >= 0) ---
    dt = inputs.dt
    prod_if = jnp.concatenate([
        jnp.reshape(-(F_u[0] * u1 + F_v[0] * v1) / z1, (1,)),
        -(F_u[1:-1] * g_u + F_v[1:-1] * g_v), jnp.zeros((1,), g_u.dtype)])
    buoy_if = jnp.concatenate([
        jnp.reshape(b_s, (1,)), beta_i * (F_thl[1:-1] + bq_i * F_qt[1:-1]),
        jnp.zeros((1,), g_u.dtype)])
    src_p, src_m = _split(_avg(prod_if) + _avg(buoy_if), B0)
    np_ext = jnp.concatenate([n_plus[:1], n_plus, n_plus[-1:]])
    l_full = _mixing_length(zf, _avg(np_ext), se_f, l_inf, p)
    diss = se_f / (p["c_ed"] * l_full)                         # [s-1], lagged sqrt(e)
    rho_int = inputs.rho_half[1:-1]
    zpad = jnp.zeros((1,), g_u.dtype)
    ra = jnp.concatenate([zpad, rho_int * 0.5 * p["c_T"] * V_i * cw, zpad])  # upwind adv
    rk = jnp.concatenate([zpad, rho_int * p["c_k"] * ell * se_h / dz_f, zpad])  # diffusion
    w = dt / inputs.mass
    sub = -w * (ra[:-1] + rk[:-1])
    diag = (1.0 + dt * diss + dt * src_m / (e_old + E_SINK)
            + w * (ra[1:] + rk[1:] + rk[:-1]))
    sup = -w * rk[1:]
    e_new = thomas_solve(sub, diag, sup, e_old + dt * src_p)

    aux_new = {
        "e": e_new, "F_thl": F_thl, "F_qt": F_qt, "F_u": F_u, "F_v": F_v,
        "dw": full(dw0, dw_i), "Pi": full(pi_0, pi_i),
    }
    return (F_thl, F_qt), (F_u, F_v), aux_new


# Internal steps per SCM step (deviation SUB-STEPS above): the longest sub-step that
# stayed finite on all 14 scored cases at default parameters, measured 2026-10-30.
N_SUB = 60
fluxes = substepped(_fluxes_one_step, N_SUB, KIND)
