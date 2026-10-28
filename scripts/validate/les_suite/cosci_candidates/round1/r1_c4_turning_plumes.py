"""C4 turning plumes (co-scientist round 1, KIND="both").

Buoyancy-launched updrafts that turn smoothly into a descending population, survive
by age x environmental turbulence frequency, carry theta_l, q_t, u and v under one
stratification-gated Gregory-Kershaw pressure coefficient, on one prognostic TKE,
with a plume-gust bulk surface drag law.

Equations (hypothesis numbering; surface-first; interfaces i = 0..n, cells k = 0..n-1,
interface i sits between cells i-1 and i):

(0) l_g = kappa z L_inf/(kappa z + L_inf);  N2 = (g/thv0) d(theta_l (1 + d_v q_t))/dz
    (unsaturated grid-mean formula), N2_p = N_ref2 softplus(N2/N_ref2);
    omega_t^2 = (e + e_eps)/l_g^2;  S = N2_p/(N2_p + omega_t^2);
    c_p = (1-S) c_T + S c_S;  Pr_t = Pr_0 (1 + r_Pr S);
    l_m = [1/l_g + N_p/(c_N sqrt(e+e_eps))]^-1;  K_m = c_K l_m sqrt(e+e_eps), K_h = K_m/Pr_t;
    D_e = c_K^3 e sqrt(e+e_eps)/l_m.
(1) F_X = -K dX/dz + M_u X_u - D X_d - (M_u - D) X_upwind (compensating flux, implicit,
    upwind via the smooth split M_net = M_+ - M_-), one tridiagonal solve per field.
    The heat row is assembled in ENTHALPY form (interface fluxes weighted by Pi_half,
    cell capacity mass*Pi), the round-1 scoring heat form. The reported flux is
    evaluated at the solution, so the harness' forward-Euler update with it
    reproduces the implicit solve exactly (heat_form="enthalpy", closure stepped
    alone; heat_form="theta" differs at O(dPi)).
(2) Launch at interface 1: B_p = B_ref softplus(B_s/B_ref), w_0 = (z_1 B_p + W_floor^3)^(1/3),
    M_u0 = a_0 w_0, phi_u0 = phibar_0 + C_phi F_phi,s/w_hat_0, u_u0 = U_0, w_u^2 = w_0^2, A = 0.
(3) Updraft across cell c (interface c -> c+1): eps = c_eps/z, gamma = 2 b_-/w_hat^2,
    delta_A = C_A A omega_t^2/w_hat; turned mass T = M (1 - e^{-gamma dz}),
    hazard detrainment, then entrainment growth e^{eps dz}; implicit entraining relaxation
    of phi, u (with + c_p dU/dz), Patankar w^2 = (w^2 + 2 dz b_+)/(1 + 2 eps dz + gamma dz).
(4) Descending population (reverse scan) sourced ONLY by T, same laws with roles of b_+
    and b_- swapped and pressure term c_p dU/dzeta; D = 0 at interface 0, so the mass
    arriving at interface 1 detrains into cell 0 through the flux divergence.
(5) tau/rho = -C_D sqrt(|U_0|^2 + beta_g^2 W_s^2 + U_eps^2) U_0 (implicit in U_0, lagged
    coefficient), C_D = [kappa/den_s]^2, den_s = s softplus((ln(z_1/z_0) - psi_m)/s),
    W_s = (B_p h_p + W_floor^3)^(1/3), h_p lagged (aux).
(6) e^{n+1}: implicit TKE with Patankar sinks, diffusion K_m (zero flux at ground and top),
    sources = shear production of the TOTAL applied momentum flux (same discrete form the
    harness uses for its KE-to-heat return), surface drag work in cell 0, total buoyancy
    flux, and the plume KE lost by turning, hazard detrainment, bottom detrainment and
    absorbing-layer detrainment.
(7) Scoring diagnostics: :func:`diagnostics` (not fed back, not read by the harness).

Energy ownership (orchestrator fix 2): the harness ``ke_heating`` is the ONLY owner of
kinetic-energy-to-heat. This closure applies no heat source: total energy (enthalpy +
mean KE) closes in the harness ledger by flux form. The TKE e is a closure reservoir
outside that ledger; its Patankar shortfall is logged in ``aux["tke_residual"]``
[W m^-2] as the signed column residual (fix 1), never applied to theta_l (applying it
would double-count heat the harness already returned).

Deviations from the hypothesis text are listed in DEVIATIONS below.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from legoesm.atmosphere.les_suite.candidate import (
    absorb_top,
    half_level_average,
    ke_dissipation_heating,
)
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.core.bulk_flux import psi_m as most_psi_m
from legoesm.timestepping.tridiagonal import thomas_solve

from legoesm import constants

KIND = "both"

# name: (default, lo, hi, units) -- every free parameter (all tier 2 in the hypothesis)
PARAMS = {
    "C_A": (0.5, 0.05, 3.0, "1"),        # survival hazard coefficient
    "c_T": (0.7, 0.2, 0.9, "1"),         # pressure coefficient, turbulent regime
    "c_S": (0.7, 0.2, 1.6, "1"),         # pressure coefficient, stratified regime
    "c_eps": (0.4, 0.2, 1.0, "1"),       # entrainment eps = c_eps/z
    "a_0": (0.1, 0.03, 0.2, "1"),        # launch area fraction
    "C_phi": (1.0, 0.3, 2.0, "1"),       # launch excess in units of F_s/w_hat_0
    "beta_g": (0.85, 0.3, 2.0, "1"),     # gust factor
    "c_K": (0.5, 0.35, 0.7, "1"),        # K_m = c_K l sqrt(e); dissipation c_K^3
    "Pr_0": (0.85, 0.7, 1.0, "1"),       # neutral turbulent Prandtl number
    "r_Pr": (1.0, 0.0, 3.0, "1"),        # Pr_t rise with S
}

# --- fixed (tier 0) scales of the hypothesis --------------------------------------
L_INF_M = 200.0          # m        Blackadar asymptotic length (gate calibrated on LES TKE)
C_N = 0.76               # 1        Deardorff (1980) stable length constant
B_REF = 1.0e-7           # m2 s-3   softplus scale of the surface buoyancy flux
N_REF2 = 1.0e-6          # s-2      softplus scale of N^2
B_SCALE = 1.0e-4         # m s-2    softplus scale of the buoyancy split b = b_+ - b_-
W_EPS = 0.05             # m s-1    plume velocity regulariser w_hat = sqrt(w^2 + w_eps^2)
E_EPS = 1.0e-6           # m2 s-2   TKE regulariser
M_EPS = 1.0e-6           # m s-1    mass-flux regulariser
HP_LEN_M = 1.0           # m        length multiplying M_eps in the h_p denominator
# --- fixed scales ADDED by this implementation (see DEVIATIONS) --------------------
W_FLOOR = 1.0e-3         # m s-1    cube-root floor: w_0, W_s = (x + W_floor^3)^(1/3)
U_EPS = 1.0e-3           # m s-1    wind-speed / u* regulariser
CD_DEN_SCALE = 0.5       # 1        smooth floor scale of the drag-law denominator
SQRT_REG = 1.0e-12       # 1        relative regulariser of sqrt(N2_p)
ABSORB_BASE_FRAC = 0.75  # 1        absorbing-layer base / column top (same as r1_c1)
E_INIT = 0.01            # m2 s-2   initial TKE (same as r1_c1)

# Surface-layer stability function mirrors each LES run's own wall-drag law (user
# 2026-10-28): psi_m = sfc_drag_most * psi_m_BD(zeta), with inputs.sfc_drag_most read
# by the harness from the run record (1 = Businger-Dyer, 0 = neutral log law). A column
# without a record carries NaN, so the fluxes are NaN (never a silent law).

_DV = 1.0 / constants.epsilon - 1.0     # R_v/R_d - 1
_sp = jax.nn.softplus

DEVIATIONS = (
    "Heat: theta_l is transported (harness converts to enthalpy), not s_l.",
    "No heat source applied by the closure: harness ke_heating owns KE->heat (fix 2); "
    "TKE Patankar shortfall logged as signed column residual aux['tke_residual'] "
    "(fix 1), not applied to theta_l (would double-count).",
    "Plume KE returned to e: turning (net), hazard detrainment of both populations, "
    "bottom detrainment and updraft absorbing-layer detrainment (fix 3); "
    "entrainment-drag KE loss (-2 eps w^2 term) is NOT returned.",
    "sqrt(e) -> sqrt(e + e_eps) in K_m, D_e, omega_t^2 (keeps NaN-free AD at e = 0).",
    "w_0 and W_s cube roots take + W_floor^3 (1e-9 m3 s-3) so AD is finite when "
    "softplus underflows; drag speed takes + U_eps^2.",
    "C_D denominator smoothly floored: s softplus(den/s), s = 0.5 (fix 5).",
    "Absorbing layer: the updraft mass flux is multiplied by the absorb_top taper "
    "(base 0.75 of column top), so its heat, moisture and momentum excess deposit "
    "smoothly there; the descending population is sourced by the tapered turning.",
    "Buoyancy production uses the unsaturated linearisation of the theta_v flux "
    "((1 + d_v q_t) F_theta_l + d_v theta_l F_qt); liquid-water term omitted.",
    "Plume buoyancy uses grid-mean theta_v from the carried temp, q_v, q_c (saturated "
    "where cloudy); N^2 uses the unsaturated formula as written.",
    "q_t >= 0 is NOT guaranteed by construction: the plume flux part is explicit "
    "(lagged plume values); the implicit K + compensating part is an M-matrix.",
    "Plume buoyancy is evaluated at the interface where each step starts.",
    "Compensating flux upwinding uses the smooth split M_+ = (M_net + sqrt(M_net^2 + "
    "M_eps^2))/2 (adds an M_eps/2 exchange, 5e-7 m/s).",
    "Descending properties mix with the environment at weight M_eps/(D + T + M_eps) "
    "each cell (zero-mass regulariser); matters only where D is O(M_eps) and then "
    "its flux is negligible; the KE of that pull is not budgeted.",
    "TKE shear production uses the closure's own midpoint winds (U + U1)/2; the "
    "harness KE-to-heat uses midpoint winds incl. Coriolis/other forcing. TKE is "
    "outside the conserved ledger, so this only changes mixing, not conservation.",
    "Turning returns the NET plume KE 1/2 T (w_u^2 - w_eps^2) (signed: positive part "
    "to the TKE source, negative to the Patankar sink); surface work and plume KE "
    "are split the same way so the TKE RHS stays >= 0.",
    "Heat solve assembled in enthalpy form (scoring form); under heat_form='theta' "
    "the applied update departs from the implicit solve at O(dPi).",
    "theta_v0 (buoyancy reference) = grid-mean theta_v of the lowest cell; h_p uses "
    "the applied (tapered) updraft mass flux.",
    "u_star in aux = sqrt(|tau/rho|) of the applied stress (no gust term); in "
    "free convection with U_0 -> 0 zeta saturates at the psi_m clip (-10).",
)


class _Out(NamedTuple):
    flux_scalar: tuple
    flux_mom: tuple
    aux: dict
    x_new: jnp.ndarray       # (4, n) implicit solution theta_l, q_t, u, v
    mass_flux_sys: tuple     # (G, Up, Um, Pm, wh), each (4, n-1), weighted, self-test
    Mt: jnp.ndarray          # tapered updraft mass flux, interior interfaces (n-1,)
    Dt: jnp.ndarray          # descending mass flux (n-1,), sourced by tapered turning
    up: tuple                # updraft (theta_l, q_t, u, v, w^2) at interior interfaces
    dn: tuple                # descending (theta_l, q_t, u, v, w^2)
    Km: jnp.ndarray
    Kh: jnp.ndarray
    lm_h: jnp.ndarray
    cp_h: jnp.ndarray
    S_h: jnp.ndarray


def _psi_m(drag_most, zeta):
    return drag_most * most_psi_m(zeta, "dyer1974")


def _theta_v_parcel(theta_l, q_t, exner, p):
    theta, q_v, q_c = saturation_adjust(theta_l, q_t, exner, p)
    return theta * (1.0 + _DV * q_v - q_c)


def _to_cells(x_int):
    """Interior-interface values (n-1,) -> cell values (n,), edges copied."""
    pad = jnp.concatenate([x_int[:1], x_int, x_int[-1:]])
    return 0.5 * (pad[:-1] + pad[1:])


def _regime(z, e, N2, p):
    kap = constants.kappa_von_karman
    lg = kap * z * L_INF_M / (kap * z + L_INF_M)
    N2p = N_REF2 * _sp(N2 / N_REF2)
    om2 = (e + E_EPS) / lg ** 2
    S = N2p / (N2p + om2)
    Np = jnp.sqrt(N2p + N_REF2 * SQRT_REG)
    lm = 1.0 / (1.0 / lg + Np / (C_N * jnp.sqrt(e + E_EPS)))
    cp = (1.0 - S) * p["c_T"] + S * p["c_S"]
    return om2, S, lm, cp


def _drag_coef(drag_most, z1, z0, zeta):
    kap = constants.kappa_von_karman
    den = jnp.log(z1 / z0) - _psi_m(drag_most, zeta)
    return (kap / (CD_DEN_SCALE * _sp(den / CD_DEN_SCALE))) ** 2


def _applied_mass_flux(G, Up, Um, Pm, X):
    """rho F at the interior interfaces for cell values X (n,)."""
    return -G * (X[1:] - X[:-1]) - Up * X[1:] + Um * X[:-1] + Pm


def _pad0(x):
    z = jnp.zeros((1,), x.dtype)
    return jnp.concatenate([z, x, z])


def init_aux(inputs):
    dt_ = inputs.theta_l.dtype
    n = inputs.theta_l.shape[0]
    kap = constants.kappa_von_karman
    speed = jnp.sqrt(inputs.u[0] ** 2 + inputs.v[0] ** 2 + U_EPS ** 2)
    den = CD_DEN_SCALE * _sp(jnp.log(inputs.z_full[0] / inputs.z0_m) / CD_DEN_SCALE)
    ust = kap / den * speed                                   # neutral first guess, > 0
    return {"e": jnp.full((n,), E_INIT, dt_),
            "h_p": jnp.zeros((), dt_),
            "u_star": jnp.asarray(ust, dt_),
            "tke_residual": jnp.zeros((), dt_)}


def _solve(p, inp, aux):
    g, kap = constants.g, constants.kappa_von_karman
    zf, zh, rho_h, m, dt = inp.z_full, inp.z_half, inp.rho_half, inp.mass, inp.dt
    n = zf.shape[0]
    dzc = zh[1:] - zh[:-1]                 # (n,) cell thickness
    dzi = zf[1:] - zf[:-1]                 # (n-1,) spacing at interior interfaces
    z_i = zh[1:-1]
    rho_i = rho_h[1:-1]
    th, qt, U, V = inp.theta_l, inp.q_t, inp.u, inp.v
    e = aux["e"]

    # grid-mean buoyancy reference and stratification
    thv = inp.temp / inp.exner * (1.0 + _DV * inp.q_v - inp.q_c)
    thv0 = thv[0]
    thv_h = 0.5 * (thv[1:] + thv[:-1])
    thv_dry = th * (1.0 + _DV * qt)
    N2_i = g / thv0 * (thv_dry[1:] - thv_dry[:-1]) / dzi
    e_h = 0.5 * (e[1:] + e[:-1])
    _, S_h, lm_h, cp_h = _regime(z_i, e_h, N2_i, p)
    om2_c, _, lm_c, cp_c = _regime(zf, e, _to_cells(N2_i), p)
    Pr_h = p["Pr_0"] * (1.0 + p["r_Pr"] * S_h)
    Km = p["c_K"] * lm_h * jnp.sqrt(e_h + E_EPS)
    Kh = Km / Pr_h
    gU = _to_cells((U[1:] - U[:-1]) / dzi)
    gV = _to_cells((V[1:] - V[:-1]) / dzi)

    # (2) buoyancy-based launch at interface 1
    wth_s, wq_s = inp.w_theta_s, inp.w_qt_s
    B_s = g / thv0 * ((1.0 + _DV * qt[0]) * wth_s + _DV * th[0] * wq_s)
    Bp = B_REF * _sp(B_s / B_REF)
    z1 = zf[0]
    w0 = jnp.cbrt(z1 * Bp + W_FLOOR ** 3)
    wh0 = jnp.sqrt(w0 ** 2 + W_EPS ** 2)
    zero = jnp.zeros((), th.dtype)
    carry0 = (p["a_0"] * w0, th[0] + p["C_phi"] * wth_s / wh0,
              qt[0] + p["C_phi"] * wq_s / wh0, U[0], V[0], w0 ** 2, zero)

    sl = slice(1, n - 1)                   # cells 1..n-2 the scans step across
    env = (zf[sl], dzc[sl], th[sl], qt[sl], U[sl], V[sl], om2_c[sl], cp_c[sl],
           gU[sl], gV[sl])

    def split_b(b):
        bm = B_SCALE * _sp(-b / B_SCALE)
        return bm + b, bm                  # b_+, b_- (exact: b_+ - b_- = b)

    # (3) updraft, upward
    def up(c, x):
        M, thu, qtu, uu, vu, w2, A = c
        zc, dz, th_e, qt_e, U_e, V_e, om2, cp, gu, gv, ex_h, p_h, thv_e = x
        wh = jnp.sqrt(w2 + W_EPS ** 2)
        bp, bm = split_b(g * (_theta_v_parcel(thu, qtu, ex_h, p_h) - thv_e) / thv0)
        eps = p["c_eps"] / zc
        gam = 2.0 * bm / wh ** 2
        dA = p["C_A"] * A * om2 / wh
        Mr = M * jnp.exp(-gam * dz)
        T = M - Mr                          # turned -> descending population
        Mh = Mr * jnp.exp(-dA * dz)
        haz = Mr - Mh                       # hazard-detrained
        r = 1.0 + eps * dz
        new = (Mh * jnp.exp(eps * dz), (thu + eps * dz * th_e) / r,
               (qtu + eps * dz * qt_e) / r, (uu + dz * (eps * U_e + cp * gu)) / r,
               (vu + dz * (eps * V_e + cp * gv)) / r,
               (w2 + 2.0 * dz * bp) / (1.0 + 2.0 * dz * eps + dz * gam), A + dz / wh)
        return new, (c, T, 0.5 * (w2 - W_EPS ** 2) * T + 0.5 * w2 * haz)

    xs_up = env + (inp.exner_half[1:n - 1], inp.p_half[1:n - 1], thv_h[0:n - 2])
    fin, (pre, T, loss_up) = lax.scan(up, carry0, xs_up)
    upi = tuple(jnp.concatenate([a, f[None]]) for a, f in zip(pre, fin))
    Mu, thu, qtu, uu, vu, w2u, Au = upi   # interior interfaces 1..n-1

    # (4) descending population, downward, sourced only by T
    def dn(c, x):
        D, thd, qtd, ud, vd, w2d, Ad = c
        (Tc, ths, qts, us, vs, As, zc, dz, th_e, qt_e, U_e, V_e, om2, cp, gu, gv,
         ex_h, p_h, thv_e) = x
        W = D + Tc + M_EPS                  # convex mixing; -> environment as D, T -> 0
        thd = (D * thd + Tc * ths + M_EPS * th_e) / W
        qtd = (D * qtd + Tc * qts + M_EPS * qt_e) / W
        ud = (D * ud + Tc * us + M_EPS * U_e) / W
        vd = (D * vd + Tc * vs + M_EPS * V_e) / W
        w2d = (D * w2d + (Tc + M_EPS) * W_EPS ** 2) / W
        Ad = (D * Ad + Tc * As) / W
        D = D + Tc
        whd = jnp.sqrt(w2d + W_EPS ** 2)
        bp, bm = split_b(g * (_theta_v_parcel(thd, qtd, ex_h, p_h) - thv_e) / thv0)
        eps = p["c_eps"] / zc
        gam = 2.0 * bp / whd ** 2           # positively buoyant descent stops, detrains
        dA = p["C_A"] * Ad * om2 / whd
        Dr = D * jnp.exp(-(gam + dA) * dz)
        r = 1.0 + eps * dz
        new = (Dr * jnp.exp(eps * dz), (thd + eps * dz * th_e) / r,
               (qtd + eps * dz * qt_e) / r, (ud + dz * (eps * U_e - cp * gu)) / r,
               (vd + dz * (eps * V_e - cp * gv)) / r,
               (w2d + 2.0 * dz * bm) / (1.0 + 2.0 * dz * eps + dz * gam), Ad + dz / whd)
        return new, (new, 0.5 * w2d * (D - Dr))

    # absorbing layer (absorb_top taper): the updraft deposits smoothly above the base;
    # the descending population is sourced by the TAPERED turning, D itself untapered
    tau = absorb_top(jnp.ones_like(zh), zh, ABSORB_BASE_FRAC * zh[-1])
    tau_i = tau[1:-1]
    src = (tau[1:n - 1] * T, pre[1], pre[2], pre[3], pre[4], pre[6])
    xs_dn = src + env + (inp.exner_half[2:n], inp.p_half[2:n], thv_h[1:n - 1])
    dcarry0 = (zero, th[-1], qt[-1], U[-1], V[-1], zero + W_EPS ** 2, zero)
    _, (dout, loss_dn) = lax.scan(dn, dcarry0, xs_dn, reverse=True)
    dni = tuple(jnp.concatenate([a, f[None]]) for a, f in zip(dout, dcarry0))
    Dd, thd, qtd, ud, vd, w2d, _ = dni    # interior interfaces 1..n-1 (D = 0 at n-1)

    Mt, Dt = tau_i * Mu, Dd

    # net plume KE lost (kinematic, m3 s-3, signed) per cell -> TKE source/sink (fix 3)
    loss = jnp.zeros_like(th)
    loss = loss.at[1:n - 1].add(tau[1:n - 1] * loss_up + loss_dn)
    # absorbing-layer detrainment of the SURVIVING updraft (after turning and hazard):
    # cell c loses (tau_c - tau_{c+1}) of the flux leaving it; the top cell takes the rest
    loss = loss.at[1:n - 1].add((tau[1:n - 1] - tau[2:n]) * Mu[1:] * 0.5 * w2u[1:])
    loss = loss.at[n - 1].add(tau[n - 1] * Mu[-1] * 0.5 * w2u[-1])
    loss = loss.at[0].add(0.5 * Dt[0] * w2d[0])          # bottom detrainment
    R_plume = inp.rho_full * loss / m

    # (5) surface stress: bulk drag with plume gustiness, lagged coefficient
    hp_new = 2.0 * jnp.sum(z_i * Mt * dzi) / (jnp.sum(Mt * dzi) + M_EPS * HP_LEN_M)
    Ws = jnp.cbrt(Bp * aux["h_p"] + W_FLOOR ** 3)
    zeta = -kap * z1 * B_s / aux["u_star"] ** 3
    CD = _drag_coef(inp.sfc_drag_most, z1, inp.z0_m, zeta)
    c_s = CD * jnp.sqrt(U[0] ** 2 + V[0] ** 2 + (p["beta_g"] * Ws) ** 2 + U_EPS ** 2)

    # (1) one implicit solve per field (batched): K + compensating flux implicit
    Mnet = Mt - Dt
    Mp = 0.5 * (Mnet + jnp.sqrt(Mnet ** 2 + M_EPS ** 2))
    Up, Um = rho_i * Mp, rho_i * (Mp - Mnet)
    Gh, Gm = rho_i * Kh / dzi, rho_i * Km / dzi
    X = jnp.stack([th, qt, U, V])
    Xu = jnp.stack([thu, qtu, uu, vu])
    Xd = jnp.stack([thd, qtd, ud, vd])
    Pm = rho_i * (Mt * Xu - Dt * Xd)                     # (4, n-1)
    # heat row in enthalpy form: interface weight Pi_half, cell capacity mass*Pi
    ones_i = jnp.ones_like(rho_i)
    wh = jnp.stack([inp.exner_half[1:-1], ones_i, ones_i, ones_i])
    wc = jnp.stack([inp.exner, jnp.ones_like(m), jnp.ones_like(m), jnp.ones_like(m)])
    G = jnp.stack([Gh, Gh, Gm, Gm]) * wh
    Upw, Umw, Pm = Up * wh, Um * wh, Pm * wh
    GUm = jax.vmap(_pad0)(G + Umw)
    GUp = jax.vmap(_pad0)(G + Upw)
    sfc = jnp.stack([rho_h[0] * wth_s * inp.exner_half[0], rho_h[0] * wq_s, zero, zero])
    Pm_full = jnp.concatenate([sfc[:, None], Pm, jnp.zeros((4, 1), th.dtype)], axis=1)
    a = -dt * GUm[:, :-1]
    cc = -dt * GUp[:, 1:]
    drag = jnp.stack([zero, zero, rho_h[0] * c_s, rho_h[0] * c_s])
    b = m * wc + dt * (GUm[:, 1:] + GUp[:, :-1])
    b = b.at[:, 0].add(dt * drag)
    d = m * wc * X - dt * (Pm_full[:, 1:] - Pm_full[:, :-1])
    X1 = thomas_solve(a, b, cc, d)

    mf = jax.vmap(_applied_mass_flux)(G, Upw, Umw, Pm, X1)
    F_int = mf / (rho_i * wh)
    one = jnp.ones((1,), th.dtype)
    top = jnp.zeros((1,), th.dtype)
    Fth = jnp.concatenate([jnp.reshape(wth_s, (1,)), F_int[0], top])
    Fq = jnp.concatenate([jnp.reshape(wq_s, (1,)), F_int[1], top])
    Fu = jnp.concatenate([-c_s * X1[2, 0] * one, F_int[2], top])
    Fv = jnp.concatenate([-c_s * X1[3, 0] * one, F_int[3], top])

    # (6) TKE: implicit diffusion, Patankar sinks
    Um_, Vm_ = 0.5 * (U + X1[2]), 0.5 * (V + X1[3])
    Ps = ke_dissipation_heating(Fu, Fv, Um_, Vm_, rho_h, m)
    Psfc = jnp.zeros_like(th).at[0].set(
        -rho_h[0] * (Fu[0] * Um_[0] + Fv[0] * Vm_[0]) / m[0])
    th_hf = half_level_average(th, th[0])
    qt_hf = half_level_average(qt, qt[0])
    Bf = g / thv0 * ((1.0 + _DV * qt_hf) * Fth + _DV * th_hf * Fq)
    Bc = 0.5 * (Bf[1:] + Bf[:-1])
    relu = jax.nn.relu
    Src = relu(Ps) + relu(Psfc) + relu(Bc) + relu(R_plume)
    Snk = relu(-Ps) + relu(-Psfc) + relu(-Bc) + relu(-R_plume)
    diss = p["c_K"] ** 3 * jnp.sqrt(e + E_EPS) / lm_c
    Ge = _pad0(rho_i * Km / dzi)
    e1 = thomas_solve(-dt * Ge[:-1],
                      m * (1.0 + dt * (Snk / (e + E_EPS) + diss)) + dt * (Ge[1:] + Ge[:-1]),
                      -dt * Ge[1:], m * (e + dt * Src))
    tke_res = jnp.sum(m * ((e1 - e) / dt - (Src - Snk - diss * e)))   # W m-2, signed

    ust = (Fu[0] ** 2 + Fv[0] ** 2 + U_EPS ** 4) ** 0.25
    aux_new = {"e": e1, "h_p": hp_new, "u_star": ust, "tke_residual": tke_res}
    return _Out((Fth, Fq), (Fu, Fv), aux_new, X1, (G, Upw, Umw, Pm, wh), Mt, Dt,
                (thu, qtu, uu, vu, w2u), (thd, qtd, ud, vd, w2d), Km, Kh, lm_h, cp_h, S_h)


def fluxes(params, inputs, aux):
    o = _solve(params, inputs, aux)
    return o.flux_scalar, o.flux_mom, o.aux


def diagnostics(params, inputs, aux):
    """(7) Scoring diagnostics at the interior interfaces (not fed back)."""
    p = params
    o = _solve(p, inputs, aux)
    e_h = 0.5 * (aux["e"][1:] + aux["e"][:-1])
    dzi = inputs.z_full[1:] - inputs.z_full[:-1]
    th_h = 0.5 * (inputs.theta_l[1:] + inputs.theta_l[:-1])
    whu = jnp.sqrt(o.up[4] + W_EPS ** 2)
    whd = jnp.sqrt(o.dn[4] + W_EPS ** 2)
    Pp = 0.5 * (o.Mt * whu + o.Dt * whd)
    fp = Pp / (Pp + e_h + E_EPS)
    ww = (2.0 / 3.0) * e_h * (1.0 - fp) + 2.0 * e_h * fp
    dth = (inputs.theta_l[1:] - inputs.theta_l[:-1]) / dzi
    wwth = o.Mt * whu * (o.up[0] - th_h) + o.Dt * whd * (o.dn[0] - th_h)
    thth = (o.Mt / whu * (o.up[0] - th_h) ** 2 + o.Dt / whd * (o.dn[0] - th_h) ** 2
            + o.lm_h / jnp.sqrt(e_h + E_EPS) * o.Kh * dth ** 2)
    slow = p["c_K"] ** 3 * jnp.sqrt(e_h + E_EPS) / o.lm_h
    out = {"z": inputs.z_half[1:-1], "ww": ww, "wwtheta": wwth, "thetatheta": thth,
           "S": o.S_h, "c_p": o.cp_h, "M_u": o.Mt, "D": o.Dt}
    for name, X in (("uw", inputs.u), ("vw", inputs.v)):
        g = (X[1:] - X[:-1]) / dzi
        out["Pi_" + name] = -o.cp_h * ww * g - slow * (-o.Km * g)
    return out


# --- self-test (compute node only: JAX_ENABLE_X64=1 python <this file>) -------------

def _synthetic_column(n=40, dz=50.0):
    from legoesm.atmosphere.les_suite import candidate as cmod
    g, Rd = constants.g, constants.R_d
    zh = jnp.arange(n + 1) * dz
    zf = 0.5 * (zh[1:] + zh[:-1])
    H = Rd * 290.0 / g
    ph, pf = 1.0e5 * jnp.exp(-zh / H), 1.0e5 * jnp.exp(-zf / H)
    exf = (pf / constants.p_ref) ** constants.kappa
    exh = (ph / constants.p_ref) ** constants.kappa
    th = 300.0 + 0.006 * jnp.maximum(zf - 1000.0, 0.0)
    qt = 0.012 - 0.004 * jnp.clip((zf - 900.0) / 400.0, 0.0, 1.0)
    thh = half_level_average(th, th[0])
    return cmod.column_inputs(
        z_full=zf, z_half=zh, rho_full=pf / (Rd * th * exf), rho_half=ph / (Rd * thh * exh),
        mass=(ph[:-1] - ph[1:]) / g, exner=exf, exner_half=exh, p_full=pf, p_half=ph,
        theta_l=th, q_t=qt, u=5.0 + 0.002 * zf, v=-1.0 + 0.001 * zf, w_theta_s=0.1,
        w_qt_s=5.0e-5, uw_s=jnp.nan, vw_s=jnp.nan, f_c=1.0e-4, dt=60.0, z0_m=0.1,
        z0h_m=0.1)


def _selftest():
    from legoesm.atmosphere.les_suite import candidate as cmod
    from legoesm.atmosphere.les_suite.candidate_eval import e0_column_checks
    base = _synthetic_column()
    nan_law = fluxes(cmod.Candidate("r1_c4", KIND, PARAMS, init_aux, fluxes,
                                    __file__).default_params(), base, init_aux(base))
    if bool(jnp.isfinite(nan_law[1][0][0])):
        raise AssertionError("a column without a recorded drag law must give NaN stress")
    # the law acts: 0 is the plain log law, 1 the Businger-Dyer corrected one
    kap, z1, z0 = constants.kappa_von_karman, 10.0, 0.1
    for zeta in (-2.0, 0.5):
        for w, psi in ((0.0, 0.0), (1.0, float(most_psi_m(zeta, "dyer1974")))):
            ref = (kap / (CD_DEN_SCALE * _sp((math.log(z1 / z0) - psi) / CD_DEN_SCALE))) ** 2
            got = float(_drag_coef(jnp.asarray(w), z1, z0, jnp.asarray(zeta)))
            assert abs(got - ref) <= 1e-14 * ref, (zeta, w, got, ref)
        assert abs(float(_drag_coef(1.0, z1, z0, zeta)) - float(_drag_coef(0.0, z1, z0, zeta))
                   ) > 1e-2 * float(_drag_coef(0.0, z1, z0, zeta)), zeta
    stress = [fluxes(cmod.Candidate("r1_c4", KIND, PARAMS, init_aux, fluxes, __file__)
                     .default_params(), base._replace(sfc_drag_most=jnp.asarray(w)),
                     init_aux(base))[1][0][0] for w in (0.0, 1.0)]
    assert abs(float(stress[1] - stress[0])) > 1e-6 * abs(float(stress[0])), stress
    for mode in (0.0, 1.0):
        inp = base._replace(sfc_drag_most=jnp.asarray(mode, base.u.dtype))
        fn = fluxes
        cand = cmod.Candidate("r1_c4", KIND, PARAMS, init_aux, fn, __file__)
        prm = cand.default_params()
        for hf in cmod.HEAT_FORMS:
            r = e0_column_checks(cand, prm, inp, heat_form=hf, ke_heating=True)
            assert r["finite"] and r["surface_exact"] and r["top_zero"], r
            assert max(r["closure_rel"].values()) < 1e-12, r["closure_rel"]
            assert r["jit_parity"] < 1e-12 and r["surface_stress_power"] < 0.0, r
        # total energy (enthalpy + KE) closes only in the enthalpy form, and only with
        # the harness' dissipation heating (single owner): both directions asserted
        on = e0_column_checks(cand, prm, inp, heat_form="enthalpy", ke_heating=True)
        assert on["energy_rel"] < 1e-12, on["energy_rel"]
        off = e0_column_checks(cand, prm, inp, heat_form="enthalpy", ke_heating=False)
        assert off["energy_rel"] > 1e-9, off["energy_rel"]       # non-vacuous
        # reported flux reproduces the implicit solve; flux at the OLD state does not
        o = _solve(prm, inp, init_aux(inp))
        G, Upw, Umw, Pmw, wh = o.mass_flux_sys
        X = jnp.stack([inp.theta_l, inp.q_t, inp.u, inp.v])
        fl = [o.flux_scalar[0], o.flux_scalar[1], o.flux_mom[0], o.flux_mom[1]]

        def heat_step(f):
            return inp.theta_l + inp.dt * cmod.heat_flux_to_theta_tendency(
                f, inp, "enthalpy")
        for k in range(4):
            Xe = (heat_step(fl[0]) if k == 0 else
                  X[k] + inp.dt * cmod.flux_to_tendency(fl[k], inp.rho_half, inp.mass))
            err = jnp.max(jnp.abs(Xe - o.x_new[k])) / jnp.max(jnp.abs(X[k]))
            assert err < 1e-12, (k, float(err))
        bad = _applied_mass_flux(G[0], Upw[0], Umw[0], Pmw[0], inp.theta_l) / (
            inp.rho_half[1:-1] * wh[0])
        bad = jnp.concatenate([fl[0][:1], bad, fl[0][-1:]])
        assert jnp.max(jnp.abs(heat_step(bad) - o.x_new[0])) > 1e-9   # non-vacuous
        # realizability and AD over a short stand-alone rollout (enthalpy form);
        # temp, q_v, q_c are refreshed by the host saturation adjustment every step
        def roll(prm_, steps=8, big_dt=600.0):
            col, aux = inp._replace(dt=jnp.asarray(big_dt)), init_aux(inp)
            for _ in range(steps):
                (ft, fq), (fu, fv), aux = fn(prm_, col, aux)
                tend = cmod.candidate_tendencies("both", (ft, fq), (fu, fv), col,
                                                 heat_form="enthalpy", ke_heating=True)
                thl = col.theta_l + big_dt * tend.dtheta
                q = col.q_t + big_dt * tend.dqt
                theta, q_v, q_c = saturation_adjust(thl, q, col.exner, col.p_full)
                col = col._replace(theta_l=thl, q_t=q, temp=theta * col.exner, q_v=q_v,
                                   q_c=q_c, u=col.u + big_dt * tend.du,
                                   v=col.v + big_dt * tend.dv)
            return col, aux
        col, aux = roll(prm)
        assert bool(jnp.all(aux["e"] >= 0.0)) and bool(jnp.all(jnp.isfinite(col.u)))
        print("sfc_drag_most", mode, "min q_t after rollout", float(jnp.min(col.q_t)),
              "tke_residual", float(aux["tke_residual"]))
        def loss(q):
            c4 = roll(q, steps=4)[0]
            return jnp.sum(c4.u ** 2) + jnp.sum(c4.v ** 2) + jnp.sum(c4.theta_l ** 2)
        grads = jax.grad(loss)(prm)
        inert = [k for k, v in grads.items()
                 if not (jnp.isfinite(v) and abs(float(v)) > 0.0)]  # 0, NaN or inf
        assert not inert, inert
    print("r1_c4_turning_plumes selftest: PASS")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    _selftest()
