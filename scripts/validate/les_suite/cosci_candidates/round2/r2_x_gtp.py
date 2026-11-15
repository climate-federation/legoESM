"""Round-2 candidate X-GTP: plume-genealogy transilient matrix (KIND="both").

Crossover of r1_c4_turning_plumes (one TKE, survival hazard, buoyant turning, gust drag)
x r1_codex_pressure_turning_trans (parcel-value nonlocal exchange as a pair network).
One prognostic TKE e and one diagnostic entraining plume set who exchanges with whom;
compensating subsidence returns the mass; one Gregory-Kershaw coefficient c_p for the
nonlocal momentum; the transport is advanced by the exact matrix exponential.

Grid (surface-first): cells k = 0..n-1 (mass m_k = dp/g), interfaces K = 0..n, interface
K between cells K-1 and K. G = density-weighted flux [kg m-2 s-1 x unit], positive up.

(0) Regime (C4 _regime, c_T/c_S dropped). l_g = kappa z L_inf/(kappa z + L_inf);
    N_e^2 = (1 - sigma) N_unsat^2 + sigma N_sat^2 at interior interfaces, with
    N_unsat^2 = (g/theta_v0) d[theta_l (1 + d_v q_t)]/dz (C4) and
    N_sat^2 = (g/theta_v0) [theta_v,env(K) - theta_v,parcel(K-1 -> K)]/dz (reversible
    displacement through les_suite.scm_coupling.saturation_adjust: moist N^2 in cloud),
    sigma_K = mean over cells K-1, K of q_c/(q_c + Q_C_REF) (host saturation fraction);
    N_p^2 = N_ref^2 softplus(N_e^2/N_ref^2); omega_t^2 = (e + e_eps)/l_g^2;
    S = N_p^2/(N_p^2 + omega_t^2); l_m = [1/l_g + N_p/(c_N sqrt(e + e_eps))]^-1;
    K_m = c_K l_m sqrt(e + e_eps); K_h = K_m/[Pr_0 (1 + S)].
(1) Local exchange: r^L_K = rho_K K(K)/dz_K; G^L_K = r^L_K (X_{K-1} - X_K).
(2) Plume genealogy. Launch rho M_1 = rho_1 a_0 w_0 at interface 1 from cell 0,
    w_0 = (z_1 B_ref softplus(B_s/B_ref) + W_floor^3)^(1/3). Marching cell c = 1..n-1
    with origin mass fluxes P_k = rho M phi_k [kg m-2 s-1] (so sum_k P_k = rho M and
    sum_k phi_k = 1 exactly), age A and w^2:
      parcel (theta_l, q_t) = (sum_k P_k X_k + M_REG X_c)/(rho M + M_REG) (regulariser:
      finite where the plume has died); b from saturation_adjust
      at the interface the step starts (C4); b_+- = b_s softplus(+-b/b_s) (b_+ - b_- = b);
      eps = c_eps/z; gamma = 2 b_-/w_hat^2; delta_A = C_A A omega_t^2/w_hat;
      survival f_c = exp(-(gamma + delta_A) dz) r_c, r_c = taper(K=c+1)/taper(K=c)
      (absorb_top, base ABSORB_BASE_FRAC z_top: the surviving plume detrains smoothly
      in the absorbing layer, r = 0 in the top cell so rho M_n = 0 exactly);
      detrained in cell c from origin k: E_kc = P_k (1 - f_c);
      P_out = f_c P + (f_c rho M)(exp(eps dz) - 1) e_c (entrainment = origin c);
      w^2_out = (w^2 + 2 dz b_+)/(1 + 2 eps dz + gamma dz) (Patankar); A += dz/w_hat.
    The transfer T_kj = E_kj (k < j strictly) moves air k -> j carrying X_k; the return
    is compensating subsidence through every interface between k and j carrying the
    value of the cell ABOVE the interface. Summed over pairs crossing K:
      G^NL_K = sum_{k<K<=j} T_kj (X_k - X_K) = sum_k P_k(K) X_k - rho M_K X_K,
    i.e. plume-mixture value minus upwind subsidence: an exact entraining-plume flux.
(3) Momentum: G^u_K = G^L_K(K_m) + (1 - c_p) phi_R,K G^NL_K(U); surface stress below.
(4) TKE (C4 implicit solve, Patankar sinks): de/dt = P + P_sfc + B_e + R_plume - D_e
    + (1/m) d(rho K_m de/dz). P = shear production of the APPLIED momentum flux in the
    harness's own per-interface split (candidate.ke_dissipation_heating at the step's
    midpoint winds), P_sfc = -rho_0 (u'w'_0 U_mid,0 + v'w'_0 V_mid,0)/m_0 (surface-stress
    work, cell 0), so sum m (P + P_sfc) = -(exact discrete mean-KE change)/dt of the
    momentum update; B_e = buoyancy flux of the total applied scalar fluxes (C4
    unsaturated linearisation); R_plume = sum_k E_kc 1/2 w^2_c(entrance)/m_c (detrained
    plume KE, mass-flux units, review fix); D_e = c_K^3 sqrt(e + e_eps) e^{n+1}/l_m.
    Source/sink split for Patankar: s = P_S softplus(-x/P_S), Src = x + s, Snk = s.
    phi_R,K = (e_K + e_eps)/(e_K + e_eps + dt P_-,K), P_-,K = P_S P_delta(-P^NL_K/P_S)
    (r1_codex P_delta: exactly 0 for P^NL >= 0, C-infinity),
    P^NL_K = -(F^NL_u dU/dz + F^NL_v dV/dz) of the explicit-state nonlocal momentum flux.
(5) Surface. Law column (finite inputs.uw_s/vw_s, CloudBench, PI 2026-11-11): the passed
    stress is applied explicitly at the surface face, no implicit drag, no own u*.
    Own law (NaN): C4 gust drag rho_0 C_D sqrt(|U_0|^2 + beta_g^2 W_s^2 + U_eps^2) U_0,
    implicit in the exponential (as a sink in A), C_D = [kappa/den]^2,
    den = s softplus((ln(z_1/z_0) - psi_m)/s), psi_m = sfc_drag_most psi_BD(zeta) (C4,
    PI 2026-10-28: mirrors each LES run's wall law), zeta from the lagged u*,
    W_s = (B_p h_p + W_floor^3)^(1/3), h_p lagged.
(6) Time step: dX/dt = A X + b, A Metzler for theta_l and q_t (all off-diagonals >= 0);
    for u, v Metzler only where phi_R is constant along the column (codex r1: a
    phi_R varying between interfaces can make non-neighbour entries negative; momentum
    needs no positivity, so this costs only the discrete maximum principle); zero row
    sums for q_t and u, v (a uniform field is a fixed point: air mass is
    returned), flux form (column content changes only by the surface face).
    X^{n+1} = expm(dt [[A, b], [0, 0]]) [X^n; 1] (exact, unconditionally stable, q_t >= 0
    for q_t^n >= 0 and w'q_t'_s >= 0, nothing clipped). theta_l is advanced in the
    enthalpy form m Pi dtheta/dt = -div(Pi_half G) (as both parents). The REPORTED
    interior fluxes are recovered from the update by the top-down sum
    (candidate.tendency_to_flux), so the harness's forward-Euler flux-form update with
    them reproduces X^{n+1} to round-off (theta_l: under heat_form="enthalpy").
MZ memory: the transport uses P_mem, relaxed exactly toward the present genealogy,
    P_mem^{n+1} = P* + (P_mem^n - P*) exp(-dt/tau_A), tau_A = sum_c E_c A_c/(sum E + M_REG)
    + TAU_EPS (detrained-mass-weighted plume age at detrainment, parameter-free). P is
    linear in T, so this IS the T_mem relaxation; the first call (aux started = 0) uses P*.

Energy ownership: the harness ke_heating is the ONLY owner of KE -> heat; e is a closure
reservoir outside the conserved ledger (as C4). Its Patankar shortfall is logged as the
signed column residual aux["tke_residual"] [W m-2], never applied to theta_l.

DEVIATIONS from the hypothesis text (reviewer defect fixes and implementation choices):
* Genealogy marched in rho M (codex: anelastic mass genealogy), E_kj in kg m-2 s-1.
* Pressure endpoint: the spec's carried U_k + c_p (U_j - U_k) (destination value) gives
  negative off-diagonals (codex: no Metzler generator). The pressure term is evaluated at
  the interface's upwind (cell-above) value U_K, so the nonlocal momentum operator is
  exactly (1 - c_p) times the scalar genealogy operator (Metzler, constant-preserving).
* "M_mem" and "T_mem" unified: the carried memory is P_mem (n+1, n), equivalent to
  T_mem; it IS used in A (GLM fix 1). No c_tau parameter (hypothesis: parameter-free);
  the memory-free F6 control needs a code edit.
* phi_R uses an explicit-state estimate and is a per-interface factor on the whole
  nonlocal momentum flux (zero effect where that flux is down-gradient); it does NOT
  make the Patankar shortfall vanish (codex): the shortfall is logged (aux).
* D_e = c_K^3 sqrt(e + e_eps) e/l_m (C4 form, zero at e = 0), not (e + e_eps)^{3/2}/l_m.
* Buoyancy production B_e and detrained plume KE have no booked energy donor (codex,
  GLM 3): e is outside the conserved ledger, so this changes mixing, not conservation.
* Detrained plume KE uses w^2 at the cell entrance (C4); tau_A uses the age at the
  cell entrance (same point the hazard uses).
* Plume launched with the cell-0 value (no C_phi excess: the hypothesis has none).
* Codex parent's backward Euler replaced by expm (hypothesis (5)); the parent's rotation
  term, CIN kernel and K_tgt normalisation dropped (hypothesis).
* P_sfc defined (GLM 4) as the surface-stress work above. psi_m per the existing PI
  ruling (C4: sfc_drag_most mirrors the LES wall law); NaN record -> NaN own stress.
* N_sat^2 by reversible parcel displacement (no re-derived moist-N^2 formula).
* R_plume and tau_A use the INSTANTANEOUS plume; the transport, hence the applied
  fluxes and B_e built from them, uses P_mem (codex r1 fix of the record).
* Parcel and stratification thermodynamics use the non-precipitating water q_v + q_c
  (q_t carries rain, which saturation_adjust would treat as vapour: codex r1 P1);
  q_t itself is what is transported; B_e uses the flux of q_v + q_c advanced by the same
  operator (codex r2: a rain flux carries no vapour buoyancy). Rain loading is
  ignored on both sides (the environment theta_v is theta (1 + d_v q_v - q_c), as
  the parents). The surface buoyancy flux B_s (launch w_0, own-law zeta) still
  treats the prescribed total-water flux w'q_t'_s as vapour: the contract supplies
  no rain split at the surface (GLM r3).
* tau_A uses the mid-cell age A + dz/(2 w_hat) (vote r1 V14); the hazard keeps the
  entrance age (C4).
* Measured at defaults on the moist synthetic test column (first call): survival
  rho M/rho M_1 = 1.25 at 240 m, 0.08 at 480 m, 0 by 720 m: the plume dies below the
  cloud layer there (GLM prior warning borne out); see diagnostics()["survival"].
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import expm
from legoesm.atmosphere.les_suite.candidate import (
    absorb_top,
    half_level_average,
    has_surface_law,
    ke_dissipation_heating,
    surface_stress,
    tendency_to_flux,
)
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.core.bulk_flux import psi_m as most_psi_m
from legoesm.timestepping.tridiagonal import thomas_solve

from legoesm import constants

KIND = "both"

# name: (default, lo, hi, units) -- the five free parameters (all tier 2)
PARAMS = {
    "c_K": (0.5, 0.35, 0.7, "1"),      # K_m = c_K l_m sqrt(e); dissipation c_K^3
    "C_A": (0.5, 0.05, 3.0, "1"),      # survival hazard amplitude
    "c_eps": (0.4, 0.2, 1.0, "1"),     # entrainment eps = c_eps/z
    "a_0": (0.1, 0.03, 0.2, "1"),      # launch area fraction, M_0 = a_0 w_0
    "c_p": (0.7, 0.2, 0.9, "1"),       # Gregory-Kershaw pressure coefficient (nonlocal)
}

# --- fixed (tier 0) scales of the hypothesis ---------------------------------------
L_INF_M = 200.0          # m        Blackadar asymptotic length
C_N = 0.76               # 1        stable length constant (as C4)
PR_0 = 0.85              # 1        neutral turbulent Prandtl number
B_REF = 1.0e-7           # m2 s-3   softplus scale of the surface buoyancy flux
N_REF2 = 1.0e-6          # s-2      softplus scale of N^2
B_SCALE = 1.0e-4         # m s-2    softplus scale of the buoyancy split
W_EPS = 0.05             # m s-1    plume velocity regulariser
E_EPS = 1.0e-6           # m2 s-2   TKE regulariser
BETA_G = 0.85            # 1        gust factor (own surface law only)
ABSORB_BASE_FRAC = 0.75  # 1        absorbing-layer base / column top (shared value)
# --- fixed numerics reused from C4 -------------------------------------------------
W_FLOOR = 1.0e-3         # m s-1    cube-root floor of w_0, W_s
U_EPS = 1.0e-3           # m s-1    wind-speed / u* regulariser
CD_DEN_SCALE = 0.5       # 1        smooth floor of the drag-law denominator
SQRT_REG = 1.0e-12       # 1        relative regulariser of sqrt(N_p^2)
E_INIT = 0.01            # m2 s-2   initial TKE
HP_LEN_M = 1.0           # m        length multiplying M_REG in the h_p denominator
# --- fixed numerics of this implementation (voted, see record) ---------------------
Q_C_REF = 1.0e-5         # kg/kg    saturation-fraction scale sigma = q_c/(q_c + Q_C_REF)
M_REG = 1.0e-6           # kg m-2 s-1  mass-flux regulariser (tau_A, h_p denominators)
TAU_EPS_S = 1.0          # s        floor added to tau_A
P_S = 1.0e-8             # m2 s-3   softplus scale of the production split
EXPM_MAX_SQUARINGS = 24  # 1        expm scaling-and-squaring cap (NaN beyond, loud)
_DELTA = 0.05            # 1        P_delta onset scale (r1_codex parent)
_P_CUT = 80.0            # 1        P_delta returns 0 below delta/_P_CUT (r1_codex parent)

_DV = 1.0 / constants.epsilon - 1.0     # R_v/R_d - 1
_sp = jax.nn.softplus


# --- reused from r1_c4_turning_plumes (copied) -------------------------------------

def _psi_m(drag_most, zeta):
    return drag_most * most_psi_m(zeta, "dyer1974")


def _theta_v_parcel(theta_l, q_t, exner, p):
    theta, q_v, q_c = saturation_adjust(theta_l, q_t, exner, p)
    return theta * (1.0 + _DV * q_v - q_c)


def _to_cells(x_int):
    """Interior-interface values (n-1,) -> cell values (n,), edges copied."""
    pad = jnp.concatenate([x_int[:1], x_int, x_int[-1:]])
    return 0.5 * (pad[:-1] + pad[1:])


def _drag_coef(drag_most, z1, z0, zeta):
    kap = constants.kappa_von_karman
    den = jnp.log(z1 / z0) - _psi_m(drag_most, zeta)
    return (kap / (CD_DEN_SCALE * _sp(den / CD_DEN_SCALE))) ** 2


def _pad0(x):
    z = jnp.zeros((1,), x.dtype)
    return jnp.concatenate([z, x, z])


def _regime(z, e, N2):
    """C4 regime without the c_T/c_S blend: (omega_t^2, S, l_m)."""
    kap = constants.kappa_von_karman
    lg = kap * z * L_INF_M / (kap * z + L_INF_M)
    N2p = N_REF2 * _sp(N2 / N_REF2)
    om2 = (e + E_EPS) / lg ** 2
    S = N2p / (N2p + om2)
    Np = jnp.sqrt(N2p + N_REF2 * SQRT_REG)
    lm = 1.0 / (1.0 / lg + Np / (C_N * jnp.sqrt(e + E_EPS)))
    return om2, S, lm


# --- this candidate ----------------------------------------------------------------

def _p_delta(x):
    """r1_codex_pressure_turning_trans P_delta: x exp(-delta/x) for x > 0, else 0
    (C-infinity, exactly 0 for x <= 0; copied with its constants)."""
    pos = x > _DELTA / _P_CUT
    xs = jnp.where(pos, x, 1.0)
    return jnp.where(pos, x * jnp.exp(-_DELTA / xs), 0.0 * x)


def _realizability(e_h, P_nl, dt):
    """phi_R = (e + e_eps)/(e + e_eps + dt P_-), P_- = P_S P_delta(-P/P_S): exactly 1
    where the nonlocal momentum flux is down-gradient (P >= 0)."""
    P_minus = P_S * _p_delta(-P_nl / P_S)
    return (e_h + E_EPS) / (e_h + E_EPS + dt * P_minus)


def init_aux(inputs):
    dt_ = inputs.theta_l.dtype
    n = inputs.theta_l.shape[0]
    kap = constants.kappa_von_karman
    speed = jnp.sqrt(inputs.u[0] ** 2 + inputs.v[0] ** 2 + U_EPS ** 2)
    # a law column may carry no roughness (NaN): mask it before the log so no NaN
    # reaches the gradient of the initial winds; its u* is the passed stress's
    z0 = jnp.where(has_surface_law(inputs), 1.0, inputs.z0_m)
    den = CD_DEN_SCALE * _sp(jnp.log(inputs.z_full[0] / z0) / CD_DEN_SCALE)
    _, _, ust = surface_stress(inputs, 0.0, 0.0, kap / den * speed)
    return {"e": jnp.full((n,), E_INIT, dt_),
            "P_mem": jnp.zeros((n + 1, n), dt_),     # memory of the genealogy operator
            "started": jnp.zeros((), dt_),            # 0 before the first call
            "h_p": jnp.zeros((), dt_),
            "u_star": jnp.asarray(ust, dt_),
            "tke_residual": jnp.zeros((), dt_)}


def _stratification(inp):
    """theta_v0 and the cloud-weighted N_e^2 at interior interfaces (n-1,)."""
    g = constants.g
    zf = inp.z_full
    dzi = zf[1:] - zf[:-1]
    thv = inp.temp / inp.exner * (1.0 + _DV * inp.q_v - inp.q_c)
    thv0 = thv[0]
    q_np = inp.q_v + inp.q_c          # non-precipitating water (q_t includes rain)
    thv_dry = inp.theta_l * (1.0 + _DV * q_np)
    n2_u = g / thv0 * (thv_dry[1:] - thv_dry[:-1]) / dzi
    thv_par = _theta_v_parcel(inp.theta_l[:-1], q_np[:-1], inp.exner[1:], inp.p_full[1:])
    n2_s = g / thv0 * (thv[1:] - thv_par) / dzi
    sig_c = inp.q_c / (inp.q_c + Q_C_REF)
    sig = 0.5 * (sig_c[1:] + sig_c[:-1])
    return thv, thv0, (1.0 - sig) * n2_u + sig * n2_s, sig


def _plume(p, inp, om2_c, thv_h, thv0, B_s):
    """Diagnostic genealogy march. Returns origin mass fluxes at every interface
    Pint (n+1, n), detrained mass E (n, n) [origin, cell], detrained KE per cell
    (n,) [W m-2], tau_A [s], launch rho M_1 and h_p [m]."""
    g = constants.g
    zf, zh, rho_h = inp.z_full, inp.z_half, inp.rho_half
    n = zf.shape[0]
    dtype = zf.dtype
    dzc = zh[1:] - zh[:-1]
    th, qt = inp.theta_l, inp.q_v + inp.q_c   # parcel water: non-precipitating
    Bp = B_REF * _sp(B_s / B_REF)
    w0 = jnp.cbrt(zf[0] * Bp + W_FLOOR ** 3)
    M0 = rho_h[1] * p["a_0"] * w0
    eye = jnp.eye(n, dtype=dtype)
    taper = absorb_top(jnp.ones_like(zh), zh, ABSORB_BASE_FRAC * zh[-1])
    tpos = taper[:-1] > 0.0
    ratio = jnp.where(tpos, taper[1:] / jnp.where(tpos, taper[:-1], 1.0), 0.0)  # cell c

    def step(carry, x):
        P, w2, A = carry
        zc, dz, om2, ex_h, p_h, thv_e, r_c, e_c, th_c, qt_c = x
        M = jnp.sum(P)
        # parcel = origin-weighted mixture; M_REG of the entered cell's air keeps the
        # ratio (and its gradient) finite where the plume has died (M -> 0)
        th_p = (jnp.dot(P, th) + M_REG * th_c) / (M + M_REG)
        qt_p = (jnp.dot(P, qt) + M_REG * qt_c) / (M + M_REG)
        wh = jnp.sqrt(w2 + W_EPS ** 2)
        b = g * (_theta_v_parcel(th_p, qt_p, ex_h, p_h) - thv_e) / thv0
        bm = B_SCALE * _sp(-b / B_SCALE)
        bp = bm + b
        eps = p["c_eps"] / zc
        gam = 2.0 * bm / wh ** 2
        dA = p["C_A"] * A * om2 / wh
        f = jnp.exp(-(gam + dA) * dz) * r_c
        detr = P * (1.0 - f)
        P_new = f * P + (f * M) * jnp.expm1(eps * dz) * e_c
        w2_new = (w2 + 2.0 * dz * bp) / (1.0 + 2.0 * dz * eps + dz * gam)
        out = (P, detr, 0.5 * w2 * M * (1.0 - f), (A + 0.5 * dz / wh) * M * (1.0 - f))
        return (P_new, w2_new, A + dz / wh), out

    sl = slice(1, n)
    xs = (zf[sl], dzc[sl], om2_c[sl], inp.exner_half[sl], inp.p_half[sl],
          thv_h, ratio[sl], eye[sl], th[sl], qt[sl])
    carry0 = (M0 * eye[0], w0 ** 2, jnp.zeros((), dtype))
    (P_top, _, _), (P_in, detr, ke_c, age_c) = lax.scan(step, carry0, xs)
    zero_row = jnp.zeros((1, n), dtype)
    # row n is the plume leaving through the lid: exactly 0 by the taper (r = 0 in the
    # top cell); kept (not overwritten) so a leak would show in the update and tests
    Pint = jnp.concatenate([zero_row, P_in, P_top[None]], axis=0)    # (n+1, n)
    E = jnp.concatenate([jnp.zeros((n, 1), dtype), detr.T], axis=1)  # (origin, cell)
    ke = jnp.concatenate([jnp.zeros((1,), dtype), ke_c])
    tau_A = jnp.sum(age_c) / (jnp.sum(detr) + M_REG) + TAU_EPS_S
    Mi = jnp.sum(Pint[1:n], axis=1)
    dzi = zf[1:] - zf[:-1]
    h_p = 2.0 * jnp.sum(zh[1:n] * Mi * dzi) / (jnp.sum(Mi * dzi) + M_REG * HP_LEN_M)
    return Pint, E, ke, tau_A, M0, h_p, Bp


def _nl_operator(Pint):
    """Interface operator (n+1, n): (row K) . X = sum_k P_k(K) X_k - rho M_K X_K."""
    n = Pint.shape[1]
    M = jnp.sum(Pint, axis=1)
    ret = jnp.zeros_like(Pint).at[jnp.arange(1, n), jnp.arange(1, n)].set(M[1:n])
    return Pint - ret


def _local_operator(r):
    """Interface operator (n+1, n) of r_K (X_{K-1} - X_K) at interior K, r (n-1,)."""
    n = r.shape[0] + 1
    K = jnp.arange(1, n)
    F = jnp.zeros((n + 1, n), r.dtype)
    return F.at[K, K - 1].set(r).at[K, K].set(-r)


def _advance(F, w_c, w_h, m, X0, B, dt):
    """Columns of X0 (n, nf) advanced with source columns B (n, nf)."""
    n = m.shape[0]
    nf = B.shape[1]
    wF = w_h[:, None] * F
    A = -(wF[1:] - wF[:-1]) / (m * w_c)[:, None]
    Z = jnp.zeros((n + nf, n + nf), m.dtype)
    Z = Z.at[:n, :n].set(dt * A).at[:n, n:].set(dt * B / (m * w_c)[:, None])
    E = expm(Z, max_squarings=EXPM_MAX_SQUARINGS)
    return E[:n, :n] @ X0 + E[:n, n:]


def _solve(p, inp, aux):
    g, kap = constants.g, constants.kappa_von_karman
    zf, zh, rho_h, m, dt = inp.z_full, inp.z_half, inp.rho_half, inp.mass, inp.dt
    n = zf.shape[0]
    dtype = zf.dtype
    dzi = zf[1:] - zf[:-1]
    rho_i = rho_h[1:-1]
    th, qt, U, V = inp.theta_l, inp.q_t, inp.u, inp.v
    e = aux["e"]
    ones_c, ones_h = jnp.ones_like(m), jnp.ones_like(rho_h)

    # (0) regime and local exchange
    thv, thv0, N2_i, sig = _stratification(inp)
    thv_h = 0.5 * (thv[1:] + thv[:-1])
    e_h = 0.5 * (e[1:] + e[:-1])
    _, S_h, lm_h = _regime(zh[1:-1], e_h, N2_i)
    om2_c, _, lm_c = _regime(zf, e, _to_cells(N2_i))
    Km = p["c_K"] * lm_h * jnp.sqrt(e_h + E_EPS)
    Kh = Km / (PR_0 * (1.0 + S_h))
    FLh = _local_operator(rho_i * Kh / dzi)
    FLm = _local_operator(rho_i * Km / dzi)

    # (2) plume genealogy and its memory
    wth_s, wq_s = inp.w_theta_s, inp.w_qt_s
    q_np = inp.q_v + inp.q_c          # buoyancy coefficients: non-precipitating water
    B_s = g / thv0 * ((1.0 + _DV * q_np[0]) * wth_s + _DV * th[0] * wq_s)
    Pint, E, ke_detr, tau_A, M0, h_p, Bp = _plume(p, inp, om2_c, thv_h, thv0, B_s)
    w_mem = jnp.exp(-dt / tau_A)
    P_use = jnp.where(aux["started"] > 0.0, Pint + (aux["P_mem"] - Pint) * w_mem, Pint)
    FN = _nl_operator(P_use)

    # (3) realizability factor on the nonlocal momentum flux (explicit-state estimate)
    G_nu = (1.0 - p["c_p"]) * (FN @ U)[1:n]
    G_nv = (1.0 - p["c_p"]) * (FN @ V)[1:n]
    P_nl = -(G_nu * (U[1:] - U[:-1]) + G_nv * (V[1:] - V[:-1])) / (rho_i * dzi)
    phi_R = _realizability(e_h, P_nl, dt)
    phiR_full = jnp.concatenate([jnp.ones((1,), dtype), phi_R, jnp.ones((1,), dtype)])
    FM = FLm + (1.0 - p["c_p"]) * phiR_full[:, None] * FN

    # (5) surface stress: passed law stress, or own implicit gust drag
    has_law = has_surface_law(inp)
    z0 = jnp.where(has_law, 1.0, inp.z0_m)
    drag_most = jnp.where(has_law, 1.0, inp.sfc_drag_most)
    ust_old = jnp.where(has_law, 1.0, aux["u_star"])
    zeta = -kap * zf[0] * B_s / ust_old ** 3
    CD = _drag_coef(drag_most, zf[0], z0, zeta)
    Ws = jnp.cbrt(Bp * aux["h_p"] + W_FLOOR ** 3)
    c_s = CD * jnp.sqrt(U[0] ** 2 + V[0] ** 2 + (BETA_G * Ws) ** 2 + U_EPS ** 2)
    c_s = jnp.where(has_law, 0.0, c_s)
    uw_law, vw_law, _ = surface_stress(inp, 0.0, 0.0, 0.0)
    FM = FM.at[0, 0].add(-rho_h[0] * c_s)

    # (6) exact exponential update
    e0 = jnp.zeros((n, 1), dtype).at[0, 0].set(1.0)
    X_th = _advance(FLh + FN, inp.exner, inp.exner_half, m, th[:, None],
                    e0 * (inp.exner_half[0] * rho_h[0] * wth_s), dt)[:, 0]
    # q_t (transported, rain included) and, with the same operator, the
    # non-precipitating water whose flux is the one that carries buoyancy (B_e)
    X_qq = _advance(FLh + FN, ones_c, ones_h, m, jnp.stack([qt, q_np], axis=1),
                    e0 * (rho_h[0] * wq_s), dt)
    X_q, X_np = X_qq[:, 0], X_qq[:, 1]
    X_uv = _advance(FM, ones_c, ones_h, m, jnp.stack([U, V], axis=1),
                    e0 * (rho_h[0] * jnp.stack([uw_law, vw_law])[None, :]), dt)
    U1, V1 = X_uv[:, 0], X_uv[:, 1]

    def flux(X1, X0, w_c, w_h):
        return tendency_to_flux(w_c * (X1 - X0) / dt, rho_h * w_h, m)

    top = jnp.zeros((1,), dtype)
    Fth = jnp.concatenate([jnp.reshape(wth_s, (1,)).astype(dtype),
                           flux(X_th, th, inp.exner, inp.exner_half)[1:n], top])
    Fq = jnp.concatenate([jnp.reshape(wq_s, (1,)).astype(dtype),
                          flux(X_q, qt, ones_c, ones_h)[1:n], top])
    fu, fv = flux(U1, U, ones_c, ones_h), flux(V1, V, ones_c, ones_h)
    Fu = jnp.concatenate([jnp.where(has_law, uw_law, fu[0])[None], fu[1:n], top])
    Fv = jnp.concatenate([jnp.where(has_law, vw_law, fv[0])[None], fv[1:n], top])

    # (4) TKE: implicit diffusion, Patankar sinks
    Um_, Vm_ = 0.5 * (U + U1), 0.5 * (V + V1)
    Ps = ke_dissipation_heating(Fu, Fv, Um_, Vm_, rho_h, m)
    Psfc = jnp.zeros_like(th).at[0].set(-rho_h[0] * (Fu[0] * Um_[0] + Fv[0] * Vm_[0]) / m[0])
    th_hf = half_level_average(th, th[0])
    qt_hf = half_level_average(q_np, q_np[0])
    Fnp = jnp.concatenate([jnp.reshape(wq_s, (1,)).astype(dtype),
                           flux(X_np, q_np, ones_c, ones_h)[1:n], top])
    Bf = g / thv0 * ((1.0 + _DV * qt_hf) * Fth + _DV * th_hf * Fnp)
    Bc = 0.5 * (Bf[1:] + Bf[:-1])
    R_plume = ke_detr / m
    x_tot = Ps + Psfc + Bc + R_plume
    snk = P_S * _sp(-x_tot / P_S)
    src = x_tot + snk
    diss = p["c_K"] ** 3 * jnp.sqrt(e + E_EPS) / lm_c
    Ge = _pad0(rho_i * Km / dzi)
    e1 = thomas_solve(-dt * Ge[:-1],
                      m * (1.0 + dt * (snk / (e + E_EPS) + diss)) + dt * (Ge[1:] + Ge[:-1]),
                      -dt * Ge[1:], m * (e + dt * src))
    # signed Patankar shortfall = sum m snk (1 - e1/(e + e_eps)) [W m-2] (diffusion and
    # dissipation are booked at the level they are solved at, e1)
    tke_res = jnp.sum(m * ((e1 - e) / dt - (x_tot - diss * e1)))

    ust = (Fu[0] ** 2 + Fv[0] ** 2 + U_EPS ** 4) ** 0.25
    aux_new = {"e": e1, "P_mem": P_use, "started": jnp.ones((), dtype), "h_p": h_p,
               "u_star": ust, "tke_residual": tke_res}
    diag = dict(X_new=jnp.stack([X_th, X_q, U1, V1]), Pint=Pint, P_use=P_use, E=E,
                tau_A=tau_A, M0=M0, phi_R=phi_R, snk=snk, e1=e1, P_nl=P_nl,
                S_h=S_h, sigma=sig, Km=Km, Kh=Kh,
                ke_detr=ke_detr, Ps=Ps, Psfc=Psfc, U1=U1, V1=V1)
    return (Fth, Fq), (Fu, Fv), aux_new, diag


def fluxes(params, inputs, aux):
    """((w'theta_l', w'q_t'), (u'w', v'w'), aux_new): kinematic, (n+1,), +up."""
    fs, fm, aux_new, _ = _solve(params, inputs, aux)
    return fs, fm, aux_new


def diagnostics(params, inputs, aux):
    """Pre-registered diagnostics (not fed back): plume survival fraction
    rho M_K / rho M_1 at every interface (GLM 2), tau_A, S, sigma, phi_R."""
    _, _, _, d = _solve(params, inputs, aux)
    M = jnp.sum(d["Pint"], axis=1)
    return {"z": inputs.z_half, "survival": M / d["M0"], "tau_A": d["tau_A"],
            "S": d["S_h"], "sigma": d["sigma"], "phi_R": d["phi_R"]}
