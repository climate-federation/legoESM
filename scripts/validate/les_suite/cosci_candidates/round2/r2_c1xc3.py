"""Co-scientist round-2 offspring r2_c1xc3 (crossover of r1_c1_wave_kernel_mem and
r1_c3_transilient; KIND="both", EXPERIMENTAL, loaded by path by the harness
``legoesm.atmosphere.les_suite.candidate``).

Memory EDMF-TKE with buoyancy-barrier return downdrafts and Gregory momentum, no
wave kernel. Entraining plume + TKE from C1 (code copied from
``round1/r1_c1_wave_kernel.py``), exponential plume-mass-flux memory (C1-mem lineage,
exact kernel of the round-2 refinement), C3's parcel-buoyancy return-path barrier
(helpers imported from ``round1/r1_c3_transilient.py``) re-based on the entraining
plume, and C3's single-Ri stable law.

Grid: cells j = 0..n-1 surface-first, faces f = 0..n. Returned fluxes are
KINEMATIC, +up, shape (n+1,). Scalar index 0 = the prescribed case flux exactly;
index n = 0 for all four. The harness converts flux to tendency
``-(rho_{f+1} F_{f+1} - rho_f F_f)/m_i``, so every column total changes only by the
surface flux.

(A) SURFACE. Scalars prescribed. Momentum: on a surface-law column (finite
    ``uw_s``/``vw_s``, CloudBench) the passed stress is applied explicitly at the
    surface and u* = |tau|^(1/2) feeds every use of u* (plume launch, sigma_w,
    gate, lowest-cell shear production); no implicit drag. On an LES column (NaN)
    own neutral-safe MOST: U_g = sqrt(|U_1|^2 + beta_g^2 (2/3) e_1), u* from 6 fixed
    iterations with the shared ``psi_m``, 1/L = -kappa B_s/(u*^3 + u_reg^3), stress
    -c_s U_1^{n+1} with c_s = u*^2/U_g frozen at t^n (linearly implicit in U_1).
(B) MOIST STABILITY. Gaussian s-PDF: s = a_L (q_t - q_s(T_l, p)),
    a_L = 1/(1 + L_v q_s'/c_pd), sigma_s^2 = l_1^2 [a_L(dq_t/dz - q_s'(dT_l/dz +
    g/c_pd))]^2 + sigma_0^2 (c_sig = 1), Q1 = s/sigma_s, f_c = (1 + erf(Q1/sqrt2))/2.
    w'theta_v' = A_th w'theta_l' + A_q w'q_t', A = (1-f_c) A_dry + f_c A_sat,
    derived from the model's own theta_v = theta (1 + c_v q_v - q_c):
    A_th,dry = 1 + c_v q_t, A_q,dry = c_v theta; saturated (q_v = q_s(T,p)):
    C = 1 + c_v q_s - (q_t - q_s) + (1 + c_v) T q_s', A_th,sat = C/(1 + L_v q_s'/c_pd),
    A_q,sat = A_th,sat L_v/(c_pd Pi) - theta. N_eff^2 = (g/theta_v0)(A_th dtheta_l/dz +
    A_q dq_t/dz) replaces N^2 in the length, Ri, Pr_t and ED buoyancy production.
(C) LOCAL. Prognostic TKE e (cells). 1/l = 1/(kappa z) + 1/lambda_inf +
    N_+/(c_N sqrt(e + e_d)), K_m = c_k l sqrt(e), eps = c_k^3 e^{3/2}/l,
    Ri = N_eff^2/(S^2 + S_d^2), Ri_+ = r_d softplus(Ri/r_d),
    Pr_t = Pr_0 (1 + Ri_+/Ri_eq), K_h = K_m/Pr_t,
    c_N = c_k sqrt(2 Pr_0 Ri_eq/(2 Pr_0 - Ri_eq)) (derived, not free).
(D) PLUME (C1, diagnostic, lax.scan up): launch gate G_c = g_inf(-L_c/L),
    sigma_w = 1.3 (u*^3 + kappa z_1 B_d softplus(B_s/B_d))^{1/3}, launch excess
    beta_u F_X(0)/sigma_w, eps_u = c_ent/z + eps_c chi_c, implicit-drag w2, area
    detrainment delta_c chi_c, smooth termination product Pi_f, Kershaw-Gregory
    momentum U_u <- U_u + dz[-eps_u (U_u - U) + c_p dU/dz]. M*_f = G_c a_f
    sqrt(w2_f) Pi_f >= 0 [m/s], M*_0 = M*_n = 0.
(E) MEMORY. M^{n+1} = M* + (M^n - M*) exp(-dt/tau), tau = c_tau T_top + tau_min,
    T_top = sum_f Pi_f dz_f / sqrt(w2_f + w_d^2) (plume transit time). Convex: M >= 0.
(F) RETURN BARRIER. Mm_f = rho_f M_f [kg m-2 s-1]. delta_j = sp(Mm_j - Mm_{j+1})
    (width M_d). Parcel from cell j at level k < j: b_dn(j,k) (C3 ``_buoyancy``,
    saturation adjustment via ``scm_coupling.saturation_adjust``);
    xd(j,n) = 1 - (1/K_0) sum_{k=n..j-1} b_dn(j,k)(z_{k+1} - z_k),
    K_0 = sigma_w^2/2 + K_eps; mn_j = C3 capped dz-weighted soft-min (weights
    dz_n/Z_ref, anchor 1); A_j = sig(mn_j/dK)/sig(1/dK) in [0, 1].
    Return weights W_{f,j} = alpha_s Mm_f A_j delta_j/(sum_{j'>=f} delta_j' + M_d)
    for j >= f; R_f = sum_j W_{f,j} <= alpha_s Mm_f; D_f = Mm_f - R_f >= 0.
(G) NONLOCAL FLUX (mass units at interior faces, phi_f := cell above face f):
    scalars  F = Mm phi_u - sum_j W_{f,j} phi_j - D phi_f;
    momentum F = Mm U_u - (1 - c_p) sum_j W_{f,j} U_j - (D + c_p R) U_f.
    Coefficients sum to zero: a uniform field has zero nonlocal flux exactly.
(H) TOTAL F = F_ED + F_NL, solved linearly implicit (dense n x n, coefficients at
    t^n): plume term explicit, ED/return/subsidence/drag implicit. The theta_l row
    is the harness' enthalpy form (storage m Pi, face weights Pi_half), so the
    returned flux applied under heat_form="enthalpy" reproduces the solved state.
    TKE: implicit M-matrix (C1), P_b = -K_h N_eff^2 + (g/theta_v0)(A_th F^NL_thl +
    A_q F^NL_qn) at faces (F^NL explicit at t^n; water part rain-free, deviation
    17), sign-split.

Conservation (RULE 0): every transport is an interface flux; F(0) is the surface
input (scalars prescribed, momentum the applied stress), F(n) = 0 is set exactly.
Air mass: Mm - R - D = 0 at every face (D defined as Mm - R), so the scheme moves
no net air and a uniform field has zero flux; the Gregory term on the return stream
keeps the momentum coefficients zero-sum. Nothing is clipped. Energy: harness
enthalpy form + KE dissipation returned as heat; TKE is an auxiliary closure
variable outside the audited budget (its buoyancy/geopotential exchange is NOT
booked; declared, as in the parent).

Mori-Zwanzig: Markovian = K(Ri, N_eff, l), M*, A_j, plume excesses; memory = TKE
(implicit) + exponential kernel on M (length c_tau x transit time); R inherits the
memory through its proportionality to M; noise = 0.

Deviations from the hypothesis text (each recorded)
---------------------------------------------------
1. Units (codex hard gate): the plume M is kinematic [m/s] (C1); delta, R, D, W and
   M_d are formed in MASS units Mm = rho_half M [kg m-2 s-1], consistent with M_d.
2. Return normaliser (both reviewers): the undefined "M_d phi-weight" is removed.
   R_f phi_ret,f is written directly as sum_j W_{f,j} phi_j, i.e.
   phi_ret = sum A delta phi / sum A delta (exactly convex, never divided by a
   vanishing sum); uniform phi -> zero flux exactly. (VOTED V5.)
3. Momentum formula: the final zero-sum form of (G) is implemented; the plume
   pressure term is in U_u (C1) and the return stream carries c_p again as written
   (whether that double-counts is an OPEN physics question, not decided here).
4. Z_ref = current-step plume depth sum_f Pi_f dz_f + Z_D (no one-step lag, no
   carried state; the plume is computed before the barrier). (VOTED V1, V2.)
5. sigma_s uses l_1, the parent-C1 length with the grid-mean theta_v dry N^2 (the
   moist l needs f_c: circular); sigma_0 = 1e-5 kg/kg floor kept from the
   refinement (codex: sigma_s = 0 in a uniform column). (VOTED V3.)
6. The saturated buoyancy coefficients are DERIVED from the model's theta_v
   definition at the grid-mean T (formula above), not quoted from a paper.
7. Nonlocal buoyancy flux in P_b uses the (B) coefficients on the explicit F^NL
   (return and subsidence included). (VOTED V4.)
8. Memory cold start: the first call uses M = M* (parent C1-mem). (VOTED V6.)
9. u_reg = 1e-3 m/s (refinement text; parent code had 1e-2). (VOTED V7.)
10. Time scheme: plume term explicit at t^n, rest implicit (parent C1). (VOTED V8.)
11. The M-matrix / positivity claim of the hypothesis is NOT made: the return terms
   put both signs on off-diagonals (codex). Not enforced, not clipped.
12. sqrt(e) is used as written: e > 0 strictly by construction of the TKE solve
   (P_+ >= B_d/2 > 0, M-matrix), so it is differentiable wherever reached.
13. beta_g enters only the own MOST law: inert on a surface-law column (frozen out
   on CloudBench, as the hypothesis states).
14. sigma_w carries + u_reg^3 inside the cube root (review r1 P1, both reviewers):
   with a passed zero stress (u* = 0) and a cooling surface softplus(B_s/B_d)
   underflows to 0, sigma_w was 0 and the launch excess inf/NaN. Floor 1.3e-3 m/s
   keeps sigma_w^2 below w_d^2, so the plume stays dead there. (VOTED V12.)
15. Three mixing lengths, all of the (C) form: l_1 (cells, dry theta_v N^2, only
   for sigma_s), l_f (faces, N_eff^2, face e: K_m, K_h), l_c (cells, N_eff^2
   averaged to cells, cell e: dissipation). The (C) formula is one law; its
   placement is C1's (C1 deviation 7).
16. Launch excess beta_u F0/sigma_w uses one beta_u for all four variables, so for
   u, v it is (m2/s2)/(m/s) = m/s (inherited from C1, declared here). The launch
   gate carries the same u_reg^3 as 1/L: G_c = g_inf(L_c kappa B_s/(u*^3+u_reg^3)).
17. Rain (review r2, codex P1): the contract's q_t includes rain. All thermodynamics
   use the non-precipitating water q_n = q_v + q_c (s-PDF, A coefficients and their
   gradient, the C3 descending parcel); the plume carries a separate
   non-precipitating water q_n,u (launch q_n,0 + beta_u w'q_t'_s/sigma_w, entrains
   environment q_n with the same exp(-eps dz)) used ONLY for its saturation
   adjustment (review r3, codex P1: subtracting the local rain from the entrained
   total water left d (q_r,below - q_r,here)). The transported plume total water
   q_t,u is the hypothesis' entraining q_t (conservation untouched); rain is
   passive in theta_v (no loading, as both parents' theta_v). The nonlocal
   buoyancy flux in P_b applies A_q to the non-precipitating F^NL (plume q_n,u,
   environment q_n; rain excluded; codex r4 P2, VOTED 2026-11-15 3-0).
   (VOTED V15, V20.)
18. Surface buoyancy flux B_s = (g/theta_v0)(A_th,0 w'theta_l'_s + A_q,0 w'q_t'_s)
   with the cell-0 moist coefficients (review r2, codex P1; equals C1's form in
   clear air without rain). (VOTED V16.)
   c_N needs Ri_eq < 2 Pr_0: guaranteed by the PARAMS bounds (2 Pr_0 - Ri_eq >= 0.8),
   no guard added. (VOTED V22.)
19. On a law column the discarded own MOST law runs on z0 = 0.01 z_1, so a NaN
   roughness cannot put 0*NaN into the gradient (review r2, codex P1). (VOTED V17.)

KNOWN DEFECT, measured, not hidden (PLAUSIBLE beyond the two synthetic test
columns): the return stream is nearly shut. The plume detrains where it stops
(inversion, cloud layer), and the CELL-MEAN air there, brought down undiluted, is
warmer in theta_v than the stably stratified air below, so xd << 0 and A ~ 1e-16;
A ~ 1 only inside the well-mixed layer, where delta ~ 0. At defaults R/Mm ~ 4e-10
(dry CBL, 16 levels) and ~1e-10 (BOMEX-like, 24 levels), d loss/d alpha_s ~ 1e-11
relative: alpha_s is near-inert there. The hypothesis' "chi ~ 0, returns carry
inversion air down" in mixed layers does not occur on these columns. Implemented
as written; the physics question is open.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from legoesm.atmosphere.les_suite.candidate import has_surface_law, surface_stress
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.core.bulk_flux import psi_m
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_dT
from legoesm.timestepping.tridiagonal import thomas_solve

from legoesm import constants

_R1 = Path(__file__).resolve().parents[1] / "round1"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_r2c1xc3_{name}", _R1 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_c1 = _load("r1_c1_wave_kernel")
_c3 = _load("r1_c3_transilient")
_g_inf, _n_plus, _face_to_cell, _sp_w2 = _c1._g_inf, _c1._n_plus, _c1._face_to_cell, _c1._sp
_buoyancy, _gate, _softmin_one, _divergence = (_c3._buoyancy, _c3._gate,
                                               _c3._softmin_one, _c3._divergence)

KIND = "both"

PARAMS = {
    # (C) local TKE closure
    "c_k": (0.52, 0.40, 0.60, "1"),              # K_m = c_k l sqrt(e); dissipation c_k^3
    "lambda_inf": (100.0, 30.0, 300.0, "m"),     # Blackadar asymptotic length
    "Ri_eq": (0.25, 0.15, 0.40, "1"),            # stable self-regulating Ri (sets c_N, Pr_t)
    "Pr_0": (0.85, 0.6, 1.1, "1"),               # neutral turbulent Prandtl number
    # (A) surface (LES lane only; inert on a surface-law column)
    "beta_g": (1.0, 0.3, 1.5, "1"),              # gustiness from lowest-cell TKE
    # (D) plume
    "a_u": (0.1, 0.03, 0.2, "1"),                # plume area fraction at launch
    "beta_u": (1.0, 0.3, 2.0, "1"),              # launch excess beta_u F(0)/sigma_w
    "L_c": (500.0, 50.0, 3000.0, "m"),           # launch gate g_inf(-L_c/L)
    "c_ent": (0.4, 0.2, 1.0, "1"),               # entrainment c_ent/z
    "b_w": (2.0, 1.0, 3.0, "1"),                 # drag in the w_u^2 equation
    "c_p": (0.6, 0.3, 0.9, "1"),                 # Gregory pressure coeff (plume + return)
    "eps_c": (1e-3, 2e-4, 3e-3, "1/m"),          # extra entrainment, condensing plumes
    "delta_c": (2e-3, 5e-4, 5e-3, "1/m"),        # area detrainment, condensing plumes
    # (F) return barrier
    "alpha_s": (0.5, 0.0, 1.0, "1"),             # max direct-return fraction (sigmoid)
    # (E) memory
    "c_tau": (1.0, 0.3, 3.0, "1"),               # memory length in plume transit times
}

# --- fixed numerics (tier 0, never trained) ---------------------------------------
_E_D = 1e-6            # m2/s2   TKE regularisation in l and the sink
_S_D = 1e-5            # 1/s     shear floor in Ri
_R_D = 0.01            # 1       Ri softplus scale
_B_D = 1e-6            # m2/s3   buoyancy-flux softplus scale (sigma_w, P_b split)
_W_D = 0.01            # m/s     plume velocity scale (sp, termination, transit time)
_QC_D = 1e-5           # kg/kg   condensate scale in chi_c
_SIGMA_0 = 1e-5        # kg/kg   s-PDF width floor
_U_REG = 1e-3          # m/s     u*^3 regulariser in 1/L (refinement value)
_TAU_MIN = 60.0        # s       memory-length floor
_DELTA_K = 0.25        # 1       return-barrier soft-min / gate width
_K_EPS = 1e-8          # m2/s2   parcel-energy floor in K_0
_M_D = 1e-6            # kg/m2/s detrainment smooth-positive-part width and normaliser
_Z_D = 1.0             # m       Z_ref floor (dead plume: keeps dz/Z_ref finite)
_Z0_LAW_FRAC = 0.01    # 1       dummy z0/z_1 for the discarded own law on law columns
_A_W = 1.0             # 1       buoyancy coefficient in the w_u^2 equation
_SIGMA_W_PREF = 1.3    # 1       sigma_w prefactor
_E_INIT = 0.01         # m2/s2   initial TKE
_N_USTAR_ITERS = 6     # fixed MOST iterations
_STABILITY = "beljaars_holtslag1991"   # stable psi_m; unstable Businger-Dyer/Paulson

_CV = 1.0 / constants.epsilon - 1.0    # R_v/R_d - 1


def _sp(x, d):
    """(x + sqrt(x^2 + d^2))/2 > 0, cancellation-free on both branches."""
    on = x > 0.0
    xp = jnp.where(on, x, 0.0)
    xn = jnp.where(on, 0.0, x)
    pos = 0.5 * (xp + jnp.sqrt(xp * xp + d * d))
    neg = 0.5 * d * d / (jnp.sqrt(xn * xn + d * d) - xn)
    return jnp.where(on, pos, neg)


def init_aux(inputs):
    return {"e": jnp.full_like(inputs.theta_l, _E_INIT),
            "M": jnp.zeros_like(inputs.z_half),
            "started": jnp.zeros((), inputs.theta_l.dtype)}


def _moist_coefficients(inp, theta, q_n, l_cell, dthl_c, dqn_c, dTl_c):
    """Cell f_c, A_th, A_q from the Gaussian s-PDF and the theta_v linearisation, on
    the non-precipitating water q_n = q_v + q_c (rain passive, deviation 17)."""
    g, Lv, cp = constants.g, constants.L_v, constants.c_pd
    T_l = inp.theta_l * inp.exner
    qs_l = saturation_mixing_ratio(T_l, inp.p_full)
    dqs_l = saturation_mixing_ratio_dT(T_l, inp.p_full)
    a_L = 1.0 / (1.0 + Lv * dqs_l / cp)
    s = a_L * (q_n - qs_l)
    grad_s = a_L * (dqn_c - dqs_l * (dTl_c + g / cp))
    sigma_s = jnp.sqrt((l_cell * grad_s) ** 2 + _SIGMA_0 ** 2)
    f_c = 0.5 * (1.0 + erf(s / (sigma_s * jnp.sqrt(2.0))))
    a_th_d = 1.0 + _CV * q_n
    a_q_d = _CV * theta
    qs = saturation_mixing_ratio(inp.temp, inp.p_full)
    dqs = saturation_mixing_ratio_dT(inp.temp, inp.p_full)
    c_sat = 1.0 + _CV * qs - (q_n - qs) + (1.0 + _CV) * inp.temp * dqs
    a_th_s = c_sat / (1.0 + Lv * dqs / cp)
    a_q_s = a_th_s * Lv / (cp * inp.exner) - theta
    return f_c, (1.0 - f_c) * a_th_d + f_c * a_th_s, (1.0 - f_c) * a_q_d + f_c * a_q_s


def kernel(params, inputs, aux):
    """Everything the fluxes need at t^n (also the falsifier entry point). Dict."""
    p = params
    inp = inputs
    g = constants.g
    kap = constants.kappa_von_karman
    n = inp.theta_l.shape[0]
    e = aux["e"]
    dt = inp.dt

    theta = inp.temp / inp.exner
    thv = theta * (1.0 + _CV * inp.q_v - inp.q_c)
    thv0 = thv[0]
    gb = g / thv0
    thv_face = jnp.concatenate([thv[:1], 0.5 * (thv[1:] + thv[:-1]), thv[-1:]])
    c_N = p["c_k"] * jnp.sqrt(2.0 * p["Pr_0"] * p["Ri_eq"] / (2.0 * p["Pr_0"] - p["Ri_eq"]))

    # --- (B)/(C) local coefficients from e^n --------------------------------------
    dzf = inp.z_full[1:] - inp.z_full[:-1]
    zf = inp.z_half[1:-1]
    dthl = (inp.theta_l[1:] - inp.theta_l[:-1]) / dzf
    q_n = inp.q_v + inp.q_c                             # non-precipitating water
    dqn = (q_n[1:] - q_n[:-1]) / dzf
    dTl = (inp.theta_l[1:] * inp.exner[1:] - inp.theta_l[:-1] * inp.exner[:-1]) / dzf
    duf = (inp.u[1:] - inp.u[:-1]) / dzf
    dvf = (inp.v[1:] - inp.v[:-1]) / dzf
    S2f = duf ** 2 + dvf ** 2
    # first-pass (dry theta_v) cell length for the s-PDF width (deviation 5)
    N2c_dry = _face_to_cell(gb * (thv[1:] - thv[:-1]) / dzf)
    l1 = 1.0 / (1.0 / (kap * inp.z_full) + 1.0 / p["lambda_inf"]
                + _n_plus(N2c_dry)[0] / (c_N * jnp.sqrt(e + _E_D)))
    f_c, a_th, a_q = _moist_coefficients(inp, theta, q_n, l1, _face_to_cell(dthl),
                                         _face_to_cell(dqn), _face_to_cell(dTl))
    a_th_f = 0.5 * (a_th[1:] + a_th[:-1])
    a_q_f = 0.5 * (a_q[1:] + a_q[:-1])
    # --- (A) surface layer --------------------------------------------------------
    z1 = inp.z_full[0]
    law = has_surface_law(inp)
    # the own law is discarded on a law column: finite dummy roughness there so its
    # (unused) branch never puts 0*NaN into the gradient (deviation 19)
    z0 = jnp.where(law, z1 * _Z0_LAW_FRAC, inp.z0_m)
    U1sq = inp.u[0] ** 2 + inp.v[0] ** 2
    Ug = jnp.sqrt(U1sq + p["beta_g"] ** 2 * (2.0 / 3.0) * e[0])
    Bs = gb * (a_th[0] * inp.w_theta_s + a_q[0] * inp.w_qt_s)   # moist, deviation 18
    ln_z = jnp.log(z1 / z0)

    def _ustar_iter(_, us):
        inv_L = -kap * Bs / (us ** 3 + _U_REG ** 3)
        den = ln_z - psi_m(z1 * inv_L, _STABILITY) + psi_m(z0 * inv_L, _STABILITY)
        return kap * Ug / den

    ust_own = jax.lax.fori_loop(0, _N_USTAR_ITERS, _ustar_iter, kap * Ug / ln_z)
    c_s_own = ust_own ** 2 / Ug
    uw_s, vw_s, ustar = surface_stress(inp, -c_s_own * inp.u[0], -c_s_own * inp.v[0],
                                       ust_own)
    c_s = jnp.where(law, 0.0, c_s_own)                 # implicit drag only on own law
    tau_law = jnp.stack(surface_stress(inp, 0.0, 0.0, 0.0)[:2])   # explicit, or 0
    G_c = _g_inf(p["L_c"] * kap * Bs / (ustar ** 3 + _U_REG ** 3))   # = g_inf(-L_c/L)
    sigma_w = _SIGMA_W_PREF * jnp.cbrt(                 # + u_reg^3: deviation 14
        ustar ** 3 + kap * z1 * _B_D * jax.nn.softplus(Bs / _B_D) + _U_REG ** 3)

    N2f = gb * (a_th_f * dthl + a_q_f * dqn)            # N_eff^2 at interior faces
    ef = 0.5 * (e[1:] + e[:-1])
    Npf, _ = _n_plus(N2f)
    lf = 1.0 / (1.0 / (kap * zf) + 1.0 / p["lambda_inf"] + Npf / (c_N * jnp.sqrt(ef + _E_D)))
    Kmf = p["c_k"] * lf * jnp.sqrt(ef)
    Ri = N2f / (S2f + _S_D ** 2)
    Ri_p = _R_D * jax.nn.softplus(Ri / _R_D)
    Pr_t = p["Pr_0"] * (1.0 + Ri_p / p["Ri_eq"])
    Khf = Kmf / Pr_t
    Npc, _ = _n_plus(_face_to_cell(N2f))
    lc = 1.0 / (1.0 / (kap * inp.z_full) + 1.0 / p["lambda_inf"]
                + Npc / (c_N * jnp.sqrt(e + _E_D)))
    s_cell = _face_to_cell(jnp.stack([duf, dvf], axis=-1))          # (n, 2)

    # --- (D) plume (copied from r1_c1_wave_kernel, phi0 removed) -------------------
    X = jnp.stack([inp.theta_l, inp.q_t, inp.u, inp.v], axis=-1)    # (n, 4)
    F0 = jnp.stack([inp.w_theta_s, inp.w_qt_s, uw_s, vw_s])         # (4,)
    Xu1 = X[0] + p["beta_u"] * F0 / sigma_w
    qn1 = q_n[0] + p["beta_u"] * inp.w_qt_s / sigma_w        # plume non-precip water
    th1, qv1, qc1 = saturation_adjust(Xu1[0], qn1, inp.exner_half[1], inp.p_half[1])
    w2_1 = sigma_w ** 2
    T_1 = _g_inf(w2_1 / _W_D ** 2 - 1.0)
    dz_cell = inp.z_half[1:] - inp.z_half[:-1]

    def _plume_step(carry, xs):
        Xu, w2, a, T, qc_u, qn_u = carry
        Xe, s_e, z_e, dz, ex_t, p_t, thv_t, qn_e = xs
        chi = qc_u / (qc_u + _QC_D)
        eps = p["c_ent"] / z_e + p["eps_c"] * chi
        decay = jnp.exp(-eps * dz)
        sc = Xe[:2] + (Xu[:2] - Xe[:2]) * decay                      # theta_l, q_t
        mom = Xu[2:] + dz * (-eps * (Xu[2:] - Xe[2:]) + p["c_p"] * s_e)   # u, v (KG)
        Xn = jnp.concatenate([sc, mom])
        qn_n = qn_e + (qn_u - qn_e) * decay                          # thermo only
        th_u, qv_u, qc_n = saturation_adjust(sc[0], qn_n, ex_t, p_t)
        thv_u = th_u * (1.0 + _CV * qv_u - qc_n)
        B = gb * (thv_u - thv_t)
        w2n = _sp_w2((w2 + 2.0 * _A_W * B * dz) / (1.0 + 2.0 * p["b_w"] * eps * dz))
        an = a * jnp.exp(-p["delta_c"] * chi * dz)
        Tn = T * _g_inf(w2n / _W_D ** 2 - 1.0)
        return (Xn, w2n, an, Tn, qc_n, qn_n), (Xn, w2n, an, Tn, qc_n, qn_n)

    xs = (X[1:], s_cell[1:], inp.z_full[1:], dz_cell[1:], inp.exner_half[2:],
          inp.p_half[2:], thv_face[2:], q_n[1:])
    cdt = jnp.result_type(inp.theta_l, *p.values())
    a_u = jnp.asarray(p["a_u"], cdt)
    carry0 = (Xu1.astype(cdt), jnp.asarray(w2_1, cdt), a_u, jnp.asarray(T_1, cdt),
              jnp.asarray(qc1, cdt), jnp.asarray(qn1, cdt))
    _, (Xu_s, w2_s, a_s, T_s, qcu_s, qnu_s) = jax.lax.scan(_plume_step, carry0, xs)
    Xu_f = jnp.concatenate([Xu1[None], Xu1[None], Xu_s], axis=0)    # (n+1, 4)
    w2_f = jnp.concatenate([w2_1[None], w2_1[None], w2_s])
    a_f = jnp.concatenate([jnp.broadcast_to(a_u, (2,)), a_s])
    Pi_f = jnp.concatenate([T_1[None], T_1[None], T_s])
    qcu_f = jnp.concatenate([qc1[None], qc1[None], qcu_s])
    qnu_f = jnp.concatenate([qn1[None], qn1[None], qnu_s])
    edge = jnp.ones(n + 1, w2_f.dtype).at[0].set(0.0).at[n].set(0.0)
    M_star = edge * G_c * a_f * jnp.sqrt(w2_f) * Pi_f              # (n+1,) >= 0

    # --- (E) memory ----------------------------------------------------------------
    T_top = jnp.sum(Pi_f[1:] * dz_cell / jnp.sqrt(w2_f[1:] + _W_D ** 2))
    tau = p["c_tau"] * T_top + _TAU_MIN
    M_mem = M_star + (aux["M"] - M_star) * jnp.exp(-dt / tau)
    M = jnp.where(aux["started"] > 0.0, M_mem, M_star)

    # --- (F) return barrier ---------------------------------------------------------
    Mm = inp.rho_half * M                                          # kg m-2 s-1
    delta = _sp(Mm[:-1] - Mm[1:], _M_D)                            # (n,) cell j
    Z_ref = jnp.sum(Pi_f[1:] * dz_cell) + _Z_D
    bmat, _thv_e = _buoyancy(inp._replace(q_t=q_n))      # rain-free parcel
    K0 = 0.5 * sigma_w ** 2 + _K_EPS
    dzu = jnp.concatenate([dzf, jnp.zeros((1,), dzf.dtype)])        # z_{k+1} - z_k
    below = jnp.arange(n)[None, :] < jnp.arange(n)[:, None]         # [j, k]: k < j
    wd = jnp.where(below, bmat * dzu[None, :], 0.0)
    crev = jnp.cumsum(wd[:, ::-1], axis=1)[:, ::-1]                 # sum_{k=n..j-1}
    xd = 1.0 - crev / K0
    w_dn = jnp.where(below, (dz_cell / Z_ref)[None, :], 0.0)
    A = _gate(_softmin_one(xd, w_dn, _DELTA_K), _DELTA_K)           # (n,) in [0, 1]
    cum_delta = jnp.cumsum(delta[::-1])[::-1]                       # sum_{j>=f}, f=0..n-1
    ff = jnp.arange(1, n)[:, None]
    jj = jnp.arange(n)[None, :]
    W = jnp.where(jj >= ff, (A * delta)[None, :], 0.0) \
        * (p["alpha_s"] * Mm[1:n] / (cum_delta[1:] + _M_D))[:, None]   # (n-1, n)
    R = jnp.sum(W, axis=1)
    D = Mm[1:n] - R

    return dict(Bs=Bs, ustar=ustar, c_s=c_s, tau_law=tau_law, F0=F0, sigma_w=sigma_w,
                G_c=G_c, f_c=f_c, a_th=a_th, a_q=a_q, a_th_f=a_th_f, a_q_f=a_q_f,
                N2f=N2f, S2f=S2f, Kmf=Kmf, Khf=Khf, Pr_t=Pr_t, lc=lc, c_N=c_N,
                Xu_f=Xu_f, w2_f=w2_f, Pi_f=Pi_f, qcu_f=qcu_f, qnu_f=qnu_f, M_star=M_star, M=M,
                tau=tau, Mm=Mm, delta=delta, Z_ref=Z_ref, A=A, chi=1.0 - A, W=W, R=R,
                D=D, gb=gb, z1=z1, dzf=dzf, bmat=bmat)


def _operators(k, c_p, inputs):
    """Interior-face (n-1, n) implicit flux matrices (mass units) for scalars and
    winds, and the nonlocal-only scalar matrix (TKE buoyancy)."""
    n = inputs.theta_l.shape[0]
    dt_ = k["W"].dtype
    e_dn = jnp.eye(n - 1, n, dtype=dt_)               # cell f-1 below face f
    e_up = jnp.eye(n - 1, n, k=1, dtype=dt_)          # cell f above face f
    W, R, D = k["W"], k["R"], k["D"]
    g_nl = -W - D[:, None] * e_up
    g_nl_m = -(1.0 - c_p) * W - (D + c_p * R)[:, None] * e_up
    rk = inputs.rho_half[1:n] / k["dzf"]
    g_s = g_nl - (rk * k["Khf"])[:, None] * (e_up - e_dn)
    g_m = g_nl_m - (rk * k["Kmf"])[:, None] * (e_up - e_dn)
    return g_s, g_m, g_nl


def _solve(lhs_stor, wh_int, g, expl, phi, src0, drag0=None):
    """Solve m w_c (phi' - phi)/dt = w_h F_k - w_h F_{k+1} with interior
    F = expl + g phi' (mass units), surface mass flux src0 (- drag0 phi'_0)."""
    n = phi.shape[0]
    dv = _divergence(n, g.dtype)
    lhs = jnp.diag(lhs_stor) - dv @ (wh_int[:, None] * g)
    if drag0 is not None:
        lhs = lhs.at[0, 0].add(drag0)
    e0 = (jnp.arange(n) == 0).astype(g.dtype)
    rhs = (lhs_stor[:, None] * phi + (dv @ (wh_int[:, None] * expl))
           + e0[:, None] * src0[None, :])
    return jnp.linalg.solve(lhs, rhs)


def fluxes(params, inputs, aux):
    (w_theta, w_qt), (u_w, v_w), aux_new, _x = fluxes_and_state(params, inputs, aux)
    return (w_theta, w_qt), (u_w, v_w), aux_new


def fluxes_and_state(params, inputs, aux):
    """The closure, also returning the solved new state (theta_l, q_t, u, v)."""
    k = kernel(params, inputs, aux)
    inp = inputs
    n = inp.theta_l.shape[0]
    m, dt, rh = inp.mass, inp.dt, inp.rho_half
    g_s, g_m, g_nl = _operators(k, params["c_p"], inp)
    expl = k["Mm"][1:n, None] * k["Xu_f"][1:n]                      # (n-1, 4) plume
    ones_h = jnp.ones(n - 1, m.dtype)

    th_new = _solve(m * inp.exner / dt, inp.exner_half[1:n], g_s, expl[:, :1],
                    inp.theta_l[:, None],
                    jnp.reshape(rh[0] * inp.exner_half[0] * inp.w_theta_s, (1,)))[:, 0]
    q_new = _solve(m / dt, ones_h, g_s, expl[:, 1:2], inp.q_t[:, None],
                   jnp.reshape(rh[0] * inp.w_qt_s, (1,)))[:, 0]
    uv_new = _solve(m / dt, ones_h, g_m, expl[:, 2:], jnp.stack([inp.u, inp.v], axis=1),
                    rh[0] * k["tau_law"], drag0=rh[0] * k["c_s"])
    f_th = (expl[:, 0] + g_s @ th_new) / rh[1:n]
    f_q = (expl[:, 1] + g_s @ q_new) / rh[1:n]
    f_uv = (expl[:, 2:] + g_m @ uv_new) / rh[1:n, None]
    tau = -k["c_s"] * uv_new[0] + k["tau_law"]

    zero = jnp.zeros((1,), m.dtype)

    def full(s0, interior):
        return jnp.concatenate([jnp.reshape(s0, (1,)).astype(m.dtype), interior, zero])

    w_theta = full(inp.w_theta_s, f_th)
    w_qt = full(inp.w_qt_s, f_q)
    u_w = full(tau[0], f_uv[:, 0])
    v_w = full(tau[1], f_uv[:, 1])

    # --- TKE: implicit, M-matrix, positive by construction (C1) -------------------
    e = aux["e"]
    kap = constants.kappa_von_karman
    fnl_th = (expl[:, 0] + g_nl @ inp.theta_l) / rh[1:n]            # explicit F^NL
    # rain-free F^NL for the buoyancy term only (VOTED V2 2026-11-15, 3-0)
    fnl_q = (k["Mm"][1:n] * k["qnu_f"][1:n] + g_nl @ (inp.q_v + inp.q_c)) / rh[1:n]
    wthv_nl = k["a_th_f"] * fnl_th + k["a_q_f"] * fnl_q
    P_face = jnp.concatenate([jnp.reshape(k["ustar"] ** 3 / (kap * k["z1"]), (1,)),
                              k["Kmf"] * k["S2f"], zero])
    Pb_face = jnp.concatenate([jnp.reshape(k["Bs"], (1,)),
                               -k["Khf"] * k["N2f"] + k["gb"] * wthv_nl, zero])
    P_s = 0.5 * (P_face[:-1] + P_face[1:])
    P_b = 0.5 * (Pb_face[:-1] + Pb_face[1:])
    P_pos = 0.5 * (P_b + jnp.sqrt(P_b ** 2 + _B_D ** 2))
    P_neg = P_pos - P_b
    sink = P_neg / (e + _E_D) + params["c_k"] ** 3 * jnp.sqrt(e + _E_D) / k["lc"]
    ke = rh[1:-1] * k["Kmf"] / k["dzf"]
    ke_lo = jnp.concatenate([zero, ke])
    ke_hi = jnp.concatenate([ke, zero])
    e_new = thomas_solve(-ke_lo, m / dt + m * sink + ke_lo + ke_hi, -ke_hi,
                         m * e / dt + m * (P_s + P_pos))
    aux_new = {"e": e_new, "M": k["M"], "started": jnp.ones((), m.dtype)}
    state = dict(theta_l=th_new, q_t=q_new, u=uv_new[:, 0], v=uv_new[:, 1])
    return (w_theta, w_qt), (u_w, v_w), aux_new, state
