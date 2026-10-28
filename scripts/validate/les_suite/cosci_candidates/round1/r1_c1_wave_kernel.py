"""Co-scientist round-1 candidate C1_wave_kernel (KIND="both", EXPERIMENTAL).

EDMF-TKE whose stable-layer plume pressure reaction is partly radiated as obstacle
gravity waves (momentum only), with a local-wave Pr_t(Ri) and its own MOST surface
stress. Not a production scheme: loaded by path by the co-scientist harness
(``legoesm.atmosphere.les_suite.candidate``).

Grid: cells i = 0..n-1 surface-first, faces f = 0..n. All returned fluxes are
KINEMATIC, positive up, shape (n+1,). Flux form only: the harness turns them into
tendencies ``-(rho_{f+1} F_{f+1} - rho_f F_f)/m_i``. Index 0 of the scalar fluxes is
the prescribed case flux bit for bit; index 0 of the momentum fluxes is this
candidate's own MOST stress; index n is exactly 0 for all four.

Parts
-----
(A) Surface layer. B_s from the prescribed theta_l and q_t fluxes; gust speed
    U_g = sqrt(|U_1|^2 + beta_g^2 (2/3) e_1); u* from 6 fixed iterations of the MOST
    log law with psi_m = Businger-Dyer/Paulson (zeta<0) and Beljaars-Holtslag (zeta>0)
    from the shared ``legoesm.core.bulk_flux.psi_m``. The Obukhov length is never
    formed: 1/L = -kappa B_s/(u*^3 + u_reg^3) is neutral-safe. Stress
    (u'w', v'w')_s = -u*^2 U_1/U_g.
(B) Local part: prognostic TKE e (cells, carried in aux). Faces: l, K_m = c_k l sqrt(e),
    Ri, Pr_t = Pr_0 + Ri_+/R_finf, K_h = K_m/Pr_t. Backward-Euler implicit ED for
    theta_l, q_t, u, v with IMPLICIT UPWIND mass flux (deviation 12) and the wave
    flux as an explicit source; the flux it APPLIED is returned. TKE: implicit,
    M-matrix, positive by construction.
(C) Plume (diagnostic, lax.scan over cells, launched at face 1): value-difference
    mass flux for theta_l, q_t, u, v; Kershaw-Gregory pressure term c_p dU/dz on the
    plume momentum; smooth termination as w_u^2 -> w_d^2.
(D) Wave kernel (momentum only): a fraction phi0*S of the environment's local
    reaction to the plume pressure force is radiated by WKB transmission and
    deposited where the optical depth grows (critical levels, turning points,
    turbulent layers, top absorber via ``absorb_top``).
(E) Totals: F = F_ED + F_MF (+ F_W for momentum).

Pressure law (review fix 5, ONE law): the plume always feels the FULL c_p M dU/dz
(it is in u_u, hence in F_MF). In a bulk mass-flux model the grid-mean MF tendency
contains -c_p M dU/dz as the environment's local reaction. The kernel adds
D_j = phi0 S_j c_p M_j s_j back at launch cell j and removes the same total
elsewhere, so the environment feels c_p (1 - phi0 S) locally and phi0 S c_p
remotely. The kernel is a pure face flux with F_W(0) := 0 and F_W(n) = 0, so it is
zero-sum over the column for ANY transmissions; that the remote part equals the
local part removed is exact only to the plume's entrainment memory (D_j uses the
cell-mean mass flux and cell shear, the u_u increment it mirrors decays upward
with exp(-eps dz)); stated, not hidden.
SELF-ABSORPTION (hypothesis transmission law, kept): the launch cell's own net
gain from launch j is D_j exp(-tau_jj/2), not D_j; the remainder
D_j (1 - exp(-tau_jj/2)) is re-deposited in the launch cell itself. So the
EFFECTIVE radiated fraction is phi0 S_j exp(-tau_jj/2) and the environment feels
c_p (1 - phi0 S_j exp(-tau_jj/2)) locally; in optically thick launch cells
(critical level, turning point, strong K_m) nothing is radiated.

Limits (derived; review fix 3)
-------------------------------
* Neutral (B_s = 0): 1/L = 0 exactly, G_c = 0 exactly (no plume, no waves). Near the
  surface l -> kappa z; local balance K_m S^2 = c_k^3 e^{3/2}/l with S = u*/(kappa z)
  gives e = u*^2/c_k^2 and K_m = kappa z u*: the log law, same kappa as the MOST
  stress (the lowest-cell shear production uses u*^3/(kappa z_1), so the discrete
  lowest cell reproduces it; phi_m is not applied there, see deviations).
  Pr_t = Pr_0 + r_d ln2/R_finf, about +0.035 at defaults from the softplus floor.
* Free convection: S = 0 wherever N^2 <= 0; G_c -> 1 as u* -> 0; sigma_w ->
  1.3 (kappa z_1 B_s)^{1/3}; stress -> 0 linearly in U_1 with a finite gust.
  Deardorff mixed-layer scaling is inherited from EDMF, not derived.
* z-less stable (B_s < 0): G_c = 0 exactly. With l -> c_N sqrt(e)/N (e >> e_d) the
  local equilibrium K_m S^2 (1 - Ri/Pr_t) = c_k^3 e^{3/2}/l becomes
      l^2 S^2 (1 - Ri/Pr_t) = c_k^2 e  ->  (c_N^2/Ri)(1 - Ri/Pr_t) = c_k^2,
  in which e CANCELS: a z-less equilibrium exists only at one Ri = Ri_c solving
      Ri_c (c_k^2/c_N^2 + 1/Pr_t(Ri_c)) = 1     (Ri_c ~ 0.78 at defaults).
  Keeping the other length scales, 1/l = A + N/(c_N sqrt e), A = 1/(kappa z)+1/lambda,
  the equilibrium needs A l = 1 - c_k sqrt(Ri)/(c_N sqrt(1 - Ri/Pr_t)) > 0, i.e.
  Ri < Ri_c. BROKEN LIMIT (declared): above Ri_c the local TKE equilibrium
  collapses (e decays to the e_d-regularised floor; e stays > 0 in time), so this
  closure HAS a critical Richardson number, set by c_k/c_N and R_finf. The
  earlier claim "no critical-Ri collapse for R_finf < Pr_0" was false and is
  withdrawn. Rf = Ri/Pr_t at Ri_c is ~0.16 < R_finf.
* Neutral Ekman: BROKEN with fixed lambda_inf (Ekman depth not exactly ~u*/f).
* phi0 = 0 recovers plain EDMF-TKE with Kershaw-Gregory plume momentum.
* Deliberately broken: Reynolds analogy in stable layers.

Realizability scope (review fix 4): TKE e > 0, w_u^2 > 0 and plume area a >= 0 by
construction; M >= 0; S, G_c in [0, 1). NO covariance (|w'phi'|) bounds and no
wave-energy (Lindzen saturation) bound are claimed or enforced.

Deviations from the hypothesis text (each recorded)
---------------------------------------------------
1. w_u^2 drag is implicit: w2 <- sp((w2 + 2 a_w B dz)/(1 + 2 b_w eps dz)). The
   hypothesis' explicit sp(w2 + 2dz(a_w B - b_w eps w2)) reverses sign when
   2 b_w eps dz > 1, which happens in the first plume steps at default
   parameters (eps dz ~ c_ent dz/z ~ 0.27, b_w = 2), killing every plume.
2. sp() is evaluated in its cancellation-free form for x <= 0
   (w_d^4/(2(sqrt(x^2+w_d^4) - x))), double-where; same function.
3. Saturation adjustment reuses ``les_suite.scm_coupling.saturation_adjust``
   (40 damped fixed-point iterations with legoesm.thermo) instead of 3 Newton
   iterations (RULE 4 reuse).
4. Wave optical depth per cell uses a smooth harmonic cap
   tau_c = tau tau_max/(tau + tau_max), tau_max = 1e3, evaluated in log space
   (overflow-safe for any U_+, q; review fix 2). Relative change < tau/1e3.
5. Smooth plume termination (review fix 6): M_f carries a cumulative factor
   T_f = prod g_inf(w2/w_d^2 - 1), exactly 0 once w2 <= w_d^2.
6. The Lindzen ratio |F_W|/(k U_par^3/N) is not reported (no diagnostic channel
   in the plug-in contract; aux holds only the prognostic e).
7. Grid placement (not specified by the hypothesis): N^2, S^2, l, K_m, K_h, Pr_t on
   interior faces with e averaged to faces; cell values (for dissipation, wave
   damping, the gate S) from face averages; theta_v0 = theta_v of the lowest
   cell; plume buoyancy against the environment theta_v interpolated to the face;
   lowest-cell shear production = mean of u*^3/(kappa z_1) and the face-1 value.
8. psi_m is the shared helper, which clips zeta to [-10, 10] (value-only clip of
   the stability parameter, no conserved quantity touched). u* stays positive
   only while ln(z_1/z0) > psi_m(-10) = 2.55, i.e. z_1/z0 > 12.8 (true on every
   finished LES case: z0 = 0.1 m dry, 1e-4 m BOMEX). Not guarded.
9. P_b^+ / P_b^- split is smooth: P^+ = (P_b + sqrt(P_b^2 + B_d^2))/2, P^- = P^+ - P_b
   (the named B_d scale reused).
10. The implicit theta_l row is assembled in the harness' enthalpy form (storage
   m Pi, face weights Pi_half) so the returned flux reproduces the solved state
   under heat_form="enthalpy" (the scored setting). Under heat_form="theta" the
   applied increment differs from the solve by O(dPi/Pi); still flux form.
11. N_+ = sqrt(N_+^2) uses an exp(x/2) tail below x = N^2/N_d^2 = -30 (AD-safe;
   the plain sqrt(softplus) gives NaN gradients for N^2 <~ -1e-4 in fp32,
   <~ -7e-4 in fp64, i.e. in superadiabatic surface layers).

Known regime-gate leak (review fix 7; PLAUSIBLE, offline proxy, not the candidate's
own plume): with LES final-half profiles, w_u ~ (1-2) sqrt(<w'w'>) and
eps = c_ent/z at default c_ent, the gate S averages 0.19-0.56 over 0.1-0.8 z_i
in the 13 dry runs (roadmap target < 0.1) because the LES upper mixed layer is
weakly stable (N^2 ~ 1e-6 s^-2); in the BOMEX cloud layer (700-1500 m) S =
0.996-0.998 (target > 0.9: met). The gate is implemented as hypothesised.

12. Mass-flux transport is IMPLICIT UPWIND (user 2026-10-28, the plain plume
   control's scheme): F_MF(f) = M_f (X_u,f - X_f^{n+1}), the environment value is
   the cell ABOVE face f (compensating subsidence is downward) at the NEW time
   level, X_u from X^n. The written candidate used the cell BELOW at X^n (downwind
   and explicit): on BOMEX its loss jumped by ~0.1 under 1e-12 parameter changes
   and its AD gradients reached 1e16-1e20. The tridiagonal stays an M-matrix for
   any dt. The plume buoyancy flux feeding TKE uses the same cell-above environment
   theta_v (at X^n, like the closure's other production terms; user 2026-10-28).
   ``mass_flux_hook`` (default identity) lets a variant replace the diagnosed
   M(z) before it is used anywhere (transport, TKE buoyancy, wave kernel).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.atmosphere.les_suite.candidate import absorb_top
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.core.bulk_flux import psi_m
from legoesm.timestepping.tridiagonal import thomas_solve

from legoesm import constants

KIND = "both"

PARAMS = {
    # (B) local TKE closure
    "c_k": (0.52, 0.40, 0.60, "1"),              # K_m = c_k l sqrt(e); dissipation c_k^3
    "lambda_inf": (150.0, 30.0, 500.0, "m"),     # Blackadar asymptotic length
    "c_N": (0.5, 0.2, 1.0, "1"),                 # stable length c_N sqrt(e)/N
    "Pr_0": (0.85, 0.6, 1.1, "1"),               # neutral turbulent Prandtl number
    "R_finf": (0.2, 0.1, 0.3, "1"),              # limiting flux Richardson number
    # (A) surface layer
    "beta_g": (1.0, 0.3, 1.5, "1"),              # gustiness from lowest-cell TKE
    # (C) plume
    "a_u": (0.1, 0.03, 0.2, "1"),                # plume area fraction at launch
    "beta_u": (1.0, 0.3, 2.0, "1"),              # launch excess beta_u F(0)/sigma_w
    "L_c": (500.0, 50.0, 3000.0, "m"),           # convective launch gate g_inf(-L_c/L)
    "c_ent": (0.4, 0.2, 1.0, "1"),               # entrainment c_ent/z
    "b_w": (2.0, 1.0, 3.0, "1"),                 # drag in the w_u^2 equation
    "c_p": (0.7, 0.3, 0.9, "1"),                 # plume pressure coefficient (momentum)
    "eps_c": (1e-3, 2e-4, 3e-3, "1/m"),          # extra entrainment, condensing plumes
    "delta_c": (2e-3, 5e-4, 5e-3, "1/m"),        # net detrainment, condensing plumes
    # (D) wave kernel
    "phi0": (0.3, 0.0, 0.6, "1"),                # radiated fraction of the reaction
    "lambda_w": (2000.0, 500.0, 6000.0, "m"),    # horizontal wavelength
    "alpha_bg": (1e-4, 1e-5, 1e-3, "1/s"),       # background wave damping rate
}

# --- fixed numerics (tier 0, never trained) ---------------------------------------
_E_D = 1e-6            # m2/s2   TKE regularisation in l and the sink
_N_D = 1e-3            # 1/s     N^2 softplus scale
_S_D = 1e-5            # 1/s     shear floor in Ri
_R_D = 0.01            # 1       Ri softplus scale
_B_D = 1e-6            # m2/s3   buoyancy-flux softplus scale (sigma_w, P_b split)
_W_D = 0.01            # m/s     plume velocity scale (sp, termination)
_QC_D = 1e-5           # kg/kg   condensate scale in chi_c
_OM_D = 1e-6           # 1/s     frequency floor in the plume Froude discriminant
_SHEAR_DIR_D = 1e-5    # 1/s     shear scale in the wave direction n_j
_U_D = 0.05            # m/s     Doppler-wind softplus scale
_U_PLUS_FRAC = 1e-6    # 1       U_+ floor as a fraction of _U_D
_Q_D = 0.05            # 1       m^2/k^2 softplus scale
_A_W = 1.0             # 1       buoyancy coefficient in the w_u^2 equation
_SIGMA_W_PREF = 1.3    # 1       sigma_w prefactor
_ABSORB_FRAC = 0.75    # 1       absorber base as a fraction of the column top
_E_INIT = 0.01         # m2/s2   initial TKE
_U_REG = 0.01          # m/s     regulariser of u*^3 in 1/L (neutral-safe)
_N_USTAR_ITERS = 6     # fixed MOST iterations
_N_TAIL_X = -30.0      # 1       switch of _n_plus to its exp(x/2) tail
_GATE_X_FLOOR = 1e-3   # 1       exp(-1/x) is exactly 0.0 below this in fp32 and fp64
_TAU_CAP = 1e3         # 1       smooth per-cell optical-depth cap
_STABILITY = "beljaars_holtslag1991"   # stable psi_m; unstable is Businger-Dyer/Paulson

_CV = 1.0 / constants.epsilon - 1.0    # R_v/R_d - 1


def _g_inf(x):
    """C-infinity gate exp(-1/x) for x > 0, exactly 0 for x <= 0 (double-where safe)."""
    on = x > _GATE_X_FLOOR
    xs = jnp.where(on, x, 1.0)
    return jnp.where(on, jnp.exp(-1.0 / xs), 0.0)


def _sp(x):
    """(x + sqrt(x^2 + w_d^4))/2 > 0, cancellation-free on both branches."""
    w4 = _W_D ** 4
    on = x > 0.0
    xp = jnp.where(on, x, 0.0)          # selection keeps derivative 1 on the used branch
    xn = jnp.where(on, 0.0, x)
    pos = 0.5 * (xp + jnp.sqrt(xp * xp + w4))
    neg = 0.5 * w4 / (jnp.sqrt(xn * xn + w4) - xn)
    return jnp.where(on, pos, neg)


def _n_plus(N2):
    """N_+ = N_d sqrt(softplus(N^2/N_d^2)) > 0 and N_+^2, AD-safe on the deep negative
    tail: there softplus(x) underflows to 0 and d sqrt would be inf (0*inf = NaN), so
    sqrt(softplus(x)) is replaced by exp(x/2) for x < _N_TAIL_X (same function to
    relative 1e-13 at the switch); both branches finite in value and derivative."""
    x = N2 / _N_D ** 2
    hi = x > _N_TAIL_X
    x_hi = jnp.where(hi, x, _N_TAIL_X)
    x_lo = jnp.where(hi, _N_TAIL_X, x)
    root = jnp.where(hi, jnp.sqrt(jax.nn.softplus(x_hi)), jnp.exp(0.5 * x_lo))
    return _N_D * root, _N_D ** 2 * jax.nn.softplus(x)


def _face_to_cell(xf_int):
    """Interior-face values (n-1, ...) -> cell values (n, ...): mean of the two
    bounding faces, end faces copied from their interior neighbour."""
    ext = jnp.concatenate([xf_int[:1], xf_int, xf_int[-1:]], axis=0)
    return 0.5 * (ext[:-1] + ext[1:])


def init_aux(inputs):
    return jnp.full_like(inputs.theta_l, _E_INIT)


def fluxes(params, inputs, aux):
    (w_theta, w_qt), (u_w, v_w), e_new, _ = fluxes_with_mass_flux(params, inputs, aux)
    return (w_theta, w_qt), (u_w, v_w), e_new


def fluxes_with_mass_flux(params, inputs, e, mass_flux_hook=None):
    """The closure, also returning the face mass flux M (n+1,) it used.
    ``mass_flux_hook(M_diag) -> M`` replaces the diagnosed profile (identity if None)."""
    p = params
    inp = inputs
    g = constants.g
    kap = constants.kappa_von_karman
    n = inp.theta_l.shape[0]
    m = inp.mass
    dt = inp.dt

    # --- thermodynamics of the grid mean ------------------------------------------
    theta = inp.temp / inp.exner
    thv = theta * (1.0 + _CV * inp.q_v - inp.q_c)
    thv0 = thv[0]
    gb = g / thv0
    thv_face = jnp.concatenate([thv[:1], 0.5 * (thv[1:] + thv[:-1]), thv[-1:]])

    # --- (A) surface layer --------------------------------------------------------
    z1 = inp.z_full[0]
    z0 = inp.z0_m
    U1sq = inp.u[0] ** 2 + inp.v[0] ** 2
    Ug = jnp.sqrt(U1sq + p["beta_g"] ** 2 * (2.0 / 3.0) * e[0])
    wthv_s = inp.w_theta_s * (1.0 + _CV * inp.q_v[0] - inp.q_c[0]) \
        + _CV * theta[0] * inp.w_qt_s
    Bs = gb * wthv_s
    ln_z = jnp.log(z1 / z0)

    def _ustar_iter(_, us):
        inv_L = -kap * Bs / (us ** 3 + _U_REG ** 3)
        den = ln_z - psi_m(z1 * inv_L, _STABILITY) + psi_m(z0 * inv_L, _STABILITY)
        return kap * Ug / den

    ustar = jax.lax.fori_loop(0, _N_USTAR_ITERS, _ustar_iter, kap * Ug / ln_z)
    uw_s = -ustar ** 2 * inp.u[0] / Ug
    vw_s = -ustar ** 2 * inp.v[0] / Ug
    G_c = _g_inf(p["L_c"] * kap * Bs / (ustar ** 3 + _U_REG ** 3))   # = g_inf(-L_c/L)
    sigma_w = _SIGMA_W_PREF * jnp.cbrt(
        ustar ** 3 + kap * z1 * _B_D * jax.nn.softplus(Bs / _B_D))

    # --- (B) local coefficients from e^n (interior faces 1..n-1) -------------------
    dzf = inp.z_full[1:] - inp.z_full[:-1]
    zf = inp.z_half[1:-1]
    N2f = gb * (thv[1:] - thv[:-1]) / dzf
    duf = (inp.u[1:] - inp.u[:-1]) / dzf
    dvf = (inp.v[1:] - inp.v[:-1]) / dzf
    S2f = duf ** 2 + dvf ** 2
    ef = 0.5 * (e[1:] + e[:-1])
    Npf, _ = _n_plus(N2f)
    lf = 1.0 / (1.0 / (kap * zf) + 1.0 / p["lambda_inf"]
                + Npf / (p["c_N"] * jnp.sqrt(ef + _E_D)))
    Kmf = p["c_k"] * lf * jnp.sqrt(ef)
    Ri = N2f / (S2f + _S_D ** 2)
    Ri_p = _R_D * jax.nn.softplus(Ri / _R_D)
    Pr_t = p["Pr_0"] + Ri_p / p["R_finf"]
    Khf = Kmf / Pr_t

    # cell values
    N2c = _face_to_cell(N2f)
    Npc, _ = _n_plus(N2c)
    lc = 1.0 / (1.0 / (kap * inp.z_full) + 1.0 / p["lambda_inf"]
                + Npc / (p["c_N"] * jnp.sqrt(e + _E_D)))
    Kmc = p["c_k"] * lc * jnp.sqrt(e)
    s_cell = _face_to_cell(jnp.stack([duf, dvf], axis=-1))          # (n, 2)

    # --- (C) plume -----------------------------------------------------------------
    X = jnp.stack([inp.theta_l, inp.q_t, inp.u, inp.v], axis=-1)    # (n, 4)
    F0 = jnp.stack([inp.w_theta_s, inp.w_qt_s, uw_s, vw_s])         # (4,)
    Xu1 = X[0] + p["beta_u"] * F0 / sigma_w
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
        sc = Xe[:2] + (Xu[:2] - Xe[:2]) * decay                      # theta_l, q_t
        mom = Xu[2:] + dz * (-eps * (Xu[2:] - Xe[2:]) + p["c_p"] * s_e)   # u, v (KG)
        Xn = jnp.concatenate([sc, mom])
        th_u, qv_u, qc_n = saturation_adjust(sc[0], sc[1], ex_t, p_t)
        thv_u = th_u * (1.0 + _CV * qv_u - qc_n)
        B = gb * (thv_u - thv_t)
        w2n = _sp((w2 + 2.0 * _A_W * B * dz) / (1.0 + 2.0 * p["b_w"] * eps * dz))
        an = a * jnp.exp(-p["delta_c"] * chi * dz)
        Tn = T * _g_inf(w2n / _W_D ** 2 - 1.0)
        return (Xn, w2n, an, Tn, qc_n), (Xn, w2n, an, Tn, thv_u, eps)

    xs = (X[1:], s_cell[1:], inp.z_full[1:], dz_cell[1:], inp.exner_half[2:],
          inp.p_half[2:], thv_face[2:])
    cdt = jnp.result_type(inp.theta_l, *p.values())       # one carry dtype for the scan
    a_u = jnp.asarray(p["a_u"], cdt)
    carry0 = (Xu1.astype(cdt), jnp.asarray(w2_1, cdt), a_u, jnp.asarray(T_1, cdt),
              jnp.asarray(qc1, cdt))
    _, (Xu_s, w2_s, a_s, T_s, thvu_s, eps_s) = jax.lax.scan(_plume_step, carry0, xs)

    # face arrays (n+1): face 0 is a launch copy (never used as a flux), face 1 launch
    Xu_f = jnp.concatenate([Xu1[None], Xu1[None], Xu_s], axis=0)    # (n+1, 4)
    w2_f = jnp.concatenate([w2_1[None], w2_1[None], w2_s])
    a_f = jnp.concatenate([jnp.broadcast_to(a_u, (2,)), a_s])
    T_f = jnp.concatenate([T_1[None], T_1[None], T_s])
    thvu_f = jnp.concatenate([thv_u1[None], thv_u1[None], thvu_s])
    eps_c_arr = jnp.concatenate([eps_0[None], eps_s])                 # (n,) per cell
    edge = jnp.ones(n + 1, w2_f.dtype).at[0].set(0.0).at[n].set(0.0)
    M = edge * G_c * a_f * jnp.sqrt(w2_f) * T_f                      # (n+1,) >= 0
    if mass_flux_hook is not None:
        M = mass_flux_hook(M)
    # implicit upwind (deviation 12): explicit plume part M X_u here, the
    # environment part -M X^{n+1}_above goes into the solve below
    F_mf_u = M[1:-1, None] * Xu_f[1:-1]                              # (n-1, 4) interior
    wthv_mf = M[1:-1] * (thvu_f[1:-1] - thv[1:])

    # --- (D) wave kernel (momentum only) -------------------------------------------
    M_c = 0.5 * (M[:-1] + M[1:])                                      # (n,)
    wu_c = 0.5 * (jnp.sqrt(w2_f[:-1]) + jnp.sqrt(w2_f[1:]))
    r = N2c / ((eps_c_arr * wu_c) ** 2 + _OM_D ** 2)
    S = _g_inf(r)
    D = (p["phi0"] * S * p["c_p"] * M_c)[:, None] * s_cell           # (n, 2) m/s2
    nvec = s_cell / jnp.sqrt(jnp.sum(s_cell ** 2, axis=-1, keepdims=True)
                             + _SHEAR_DIR_D ** 2)
    c_ph = 0.5 * (Xu_f[:-1, 2:] + Xu_f[1:, 2:])                      # (n, 2) plume U
    U = X[:, 2:]
    Upar = jnp.einsum("jic,jc->ji", U[None, :, :] - c_ph[:, None, :], nvec)
    U_p = _U_D * jax.nn.softplus(Upar / _U_D) + _U_PLUS_FRAC * _U_D
    k = 2.0 * jnp.pi / p["lambda_w"]
    # (N_+/(k U_+))^2, not N_+^2/(k U_+)^2: the latter's backward pass scales as
    # N^2/(k U_+)^4 and overflows fp32 at U_+ ~ 5e-8 (critical levels).
    G2 = (Npc[None, :] / (k * U_p)) ** 2
    q = _Q_D * jax.nn.softplus((G2 - 1.0) / _Q_D)                     # m^2/k^2 > 0
    # tau_ji = (alpha + K k^2 q)(1+q)/(U_+ sqrt q) dz_i, capped smoothly in log space
    log_tau = (jnp.log(p["alpha_bg"] + Kmc[None, :] * k ** 2 * q) + jnp.log1p(q)
               - jnp.log(U_p) - 0.5 * jnp.log(q) + jnp.log(dz_cell)[None, :])
    tau = _TAU_CAP * jax.nn.sigmoid(log_tau - jnp.log(_TAU_CAP))      # (n_j, n_i)
    cs_ex = jnp.concatenate([jnp.zeros((n, 1), tau.dtype), jnp.cumsum(tau, axis=1)],
                            axis=1)                                   # (n, n+1)
    jj = jnp.arange(n)
    tau_jj = tau[jj, jj]
    ff = jnp.arange(n + 1)
    up = ff[None, :] > jj[:, None]                                    # static mask
    od_up = 0.5 * tau_jj[:, None] + cs_ex - cs_ex[jj, jj + 1][:, None]
    od_dn = 0.5 * tau_jj[:, None] + cs_ex[jj, jj][:, None] - cs_ex
    od = jnp.where(up, od_up, od_dn)                                  # >= 0 where used
    kern = jnp.where(up, -jnp.exp(-od), jnp.exp(-od))                 # (n, n+1)
    rhoFw = 0.5 * jnp.einsum("j,jf,jc->fc", m, kern, D)               # (n+1, 2)
    rhoFw = rhoFw.at[0].set(0.0)                    # downward remainder stays in cell 0
    Fw_raw = rhoFw / inp.rho_half[:, None]
    z_base = _ABSORB_FRAC * inp.z_half[-1]
    Fw_u = absorb_top(Fw_raw[:, 0], inp.z_half, z_base)
    Fw_v = absorb_top(Fw_raw[:, 1], inp.z_half, z_base)

    # --- (E) implicit ED with explicit MF + wave sources ---------------------------
    F_expl = F_mf_u + jnp.stack([jnp.zeros(n - 1, F_mf_u.dtype),
                                 jnp.zeros(n - 1, F_mf_u.dtype),
                                 Fw_u[1:-1], Fw_v[1:-1]], axis=-1)    # (n-1, 4)
    zero1 = jnp.zeros((1, 4), F_mf_u.dtype)
    # Heat row in the harness' ENTHALPY form (c_p T conserved): storage m Pi_full,
    # face weights Pi_half; q_t, u, v rows unweighted. The returned theta_l flux,
    # applied by the harness as -d(rho Pi_half F)/(m Pi_full), then reproduces the
    # solved theta_l exactly (theta_l still diffuses down its own gradient).
    one_c = jnp.ones_like(inp.exner)
    one_h = jnp.ones_like(inp.exner_half)
    w_c = jnp.stack([inp.exner, one_c, one_c, one_c], axis=0)          # (4, n)
    w_h = jnp.stack([inp.exner_half, one_h, one_h, one_h], axis=-1)    # (n+1, 4)
    E = (inp.rho_half[:, None] * w_h
         * jnp.concatenate([F0[None], F_expl, zero1], axis=0))
    Kf = jnp.stack([Khf, Khf, Kmf, Kmf], axis=0)                      # (4, n-1)
    kc = inp.rho_half[None, 1:-1] * w_h[1:-1].T * Kf / dzf[None, :]   # (4, n-1)
    z4 = jnp.zeros((4, 1), kc.dtype)
    k_lo = jnp.concatenate([z4, kc], axis=1)                          # (4, n) face i
    k_hi = jnp.concatenate([kc, z4], axis=1)                          # (4, n) face i+1
    mc = inp.rho_half[None, 1:-1] * w_h[1:-1].T * M[None, 1:-1]       # (4, n-1) >= 0
    mc_lo = jnp.concatenate([z4, mc], axis=1)     # face i below cell i: X_i^{n+1} leaves
    mc_hi = jnp.concatenate([mc, z4], axis=1)     # face i+1 above: X_{i+1}^{n+1} enters
    stor = (m / dt)[None, :] * w_c
    diag = stor + k_lo + k_hi + mc_lo
    rhs = stor * X.T - (E[1:] - E[:-1]).T
    X_new = thomas_solve(-k_lo, diag, -(k_hi + mc_hi), rhs)          # (4, n)
    F_ed = -Kf * (X_new[:, 1:] - X_new[:, :-1]) / dzf[None, :]        # (4, n-1)
    F_int = F_ed + F_expl.T - M[None, 1:-1] * X_new[:, 1:]

    def _full(s0, interior):
        return jnp.concatenate([jnp.reshape(s0, (1,)), interior,
                                jnp.zeros((1,), interior.dtype)])

    w_theta = _full(inp.w_theta_s, F_int[0])
    w_qt = _full(inp.w_qt_s, F_int[1])
    u_w = _full(uw_s, F_int[2])
    v_w = _full(vw_s, F_int[3])

    # --- TKE: implicit, M-matrix, positive by construction ------------------------
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

    return (w_theta, w_qt), (u_w, v_w), e_new, M
