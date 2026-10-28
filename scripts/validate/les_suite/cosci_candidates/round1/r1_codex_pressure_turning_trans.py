"""Round-1 candidate: pressure-turning transilient exchange with a buoyancy barrier.

EXPERIMENTAL co-scientist closure (KIND = "both"): one exchange network moves
theta_l, q_t, u and v; momentum differences are additionally rotated by a
work-free, pressure-mediated turning; the surface stress is a Monin-Obukhov drag.

Equations (surface-first, cell i has mass m_i [kg m-2], interface K separates
cells K-1 and K, G = density-weighted flux [kg m-2 s-1 x unit], positive up)
---------------------------------------------------------------------------
Pair exchange for i < j with conductance r_ij >= 0 [kg m-2 s-1]:
    G_X,K = sum_{i<K<=j} r_ij (X_i - X_j)                         (scalars)
    G_U,K = sum_{i<K<=j} r_ij [(U_i - U_j) + b_ij J (U_i - U_j)]   (momentum)
    J(x, y) = (-y, x);  b_ij = a_P cross(V_i, V_j) /
              sqrt((|V_i|^2 + v_f^2)(|V_j|^2 + v_f^2)),  V = U - U_s, U_s = 0
              (ground velocity; the SCM ground is at rest).
    Pairwise rotation work: (U_i-U_j).J(U_i-U_j) = 0, so the b-term does no work.
r_ij = r_loc,ij + r_nl,ij:
  local:  w_ij = m_i m_j [(1-chi_ij) exp(-d^2/(l^2+D_i^2))
                          + chi_ij exp(-d^2/(l_st^2+D_i^2))],
          l_ij = c_l min(z_m + z0, h), z_m = pair midpoint, d = z_j - z_i,
          D_i = grid spacing above cell i (resolution floor),
          r_loc = s_ij w_ij with s_ij interpolated at z_m from the interface ratio
          K_tgt,K / K_raw,K, K_raw,K = sum_{i<K<=j} w_ij d_ij / rho_K, so a linear
          profile sees  K = K_tgt  at every interface (exact where s is constant
          over the crossing pairs, e.g. nearest-neighbour dominated kernels):
          K_tgt = kappa V_0 [(1-chi) min(z+z0, h) + chi l_st]
          tau_loc = l / V_0 is the local exchange time (review fix 1); neutral,
          z << h: chi = 0, w_c = 0, V_0 -> u_*  =>  K = kappa u_* (z + z0).
  nonlocal: r_nl,ij = a_N (w_c^2/V_0^2)(V_0/h) m_i g_i m_j W_ij exp(-E_ij/V_0^2) / N_i,
          g_i = exp(-z_i/(0.1 h)) (launch distribution), W_ij = exp(-d_ij/h),
          N_i = sum_{j>i} m_j W_ij (barrier-free normalisation, so the barrier
          REMOVES exchange rather than redistributing it),
          E_ij = sum_{i<k<=j} dz_k (V_0^2/h) P_delta(-b^p_ik h / V_0^2) [m2 s-2],
          b^p_ik = g (theta_v,parcel(i->k) - theta_v,env(k)) / theta_v,env(k), the
          buoyancy of cell i's (theta_l, q_t) parcel lifted reversibly to level k
          (scm_coupling.saturation_adjust; CIN-like barrier, review fix 3).
Regime gate (per interface, interpolated to z_m for pairs):
    chi = [1 - exp(-P_delta(x_N))] exp(-P_delta(x_B)),
    x_N = N_u^2 h^2 / V_0^2 (UNSATURATED environment N^2, theta_vu = theta_l(1+eps_v q_t)),
    x_B = B_loc h / V_0^3, B_loc = previous-step modelled buoyancy flux (aux).
    P_delta(x) = x exp(-delta/x) for x > 0, 0 otherwise (C-infinity at 0).
    l_st = c_st h / sqrt(1 + P_delta(x_N))  (= c_st V_0 / sqrt(N_+^2 + V_0^2/h^2)).
Velocity scales (smooth, named scale v_f):
    w_c^3 = v_f^3 P_delta(B_s h / v_f^3);  w_c^2 + v_f^2 = (w_c^3 + v_f^3)^(2/3);
    V_0^2 = u_*^2 + w_c^2 + v_f^2;  u_* = sqrt(C_D) S_1.
Surface: S_1 = sqrt(|U_1|^2 + v_f^2), C_D = kappa^2 / [ln((z_1+z0)/z0) - psi_m(z_1/L)]^2
    (Businger-Dyer/Dyer psi_m, core.bulk_flux.psi_m, fixed point, 5 sweeps),
    G_U,0 = -rho_0 C_D S_1 U_1^{new} (a_s = 0: review fix 4). Heat/water: prescribed.
Time stepping: backward Euler of the whole network (one dense solve each for
theta_l, q_t, and the coupled (u, v) pair); the reported interface fluxes are the
ones evaluated at the implicit state, so harness tendency == implicit update.
theta_l is solved in the enthalpy form m Pi dtheta = -(Pi_half G) divergence, so the
harness's heat_form="enthalpy" conversion reproduces the implicit state exactly.

Deviations from the hypothesis text (all also listed in the round record)
--------------------------------------------------------------------------
* tau = h/V_0 and "normalised by the quadrature-weighted column sum" replaced by
  the discrete second-moment normalisation to K_tgt above (fix 1); K_tgt's
  stable branch uses kappa V_0 l_st (choice).
* Kernel width floored by the local grid spacing D_i (l^2 + D_i^2) so a
  sub-grid Gaussian cannot underflow to zero exchange.
* E_bar replaced by the lifted-parcel CIN-like integral (fix 3); chi and l_st use
  the UNSATURATED environment N^2 (fix 3; the hypothesis used moist N_m^2 in l_st).
* B_loc is carried as aux (previous-step interface buoyancy flux, dry-air
  linearisation (1+eps_v q_t) w'theta_l' + eps_v theta_l w'q_t'; saturated
  coefficients NOT used); first step B_loc = 0.
* h = smooth first bulk-Richardson crossing (pbl_height helpers, Ri_crit and
  sharpness from PBLHeightConfig defaults), smooth floor sqrt(h_c^2 + h_min^2).
* U_g renamed U_s = 0 (fix 4); surface turning law removed (a_s = 0, fix 4);
  surface drag implicit in U_1 and uses S_1 (gust floor v_f) not |V_1|.
* KE removed by the exchange is returned as heat by the HARNESS
  (ke_dissipation_heating, per interface), not per pair as the hypothesis states.
* No diagnostic variances (hypothesis: optional).
* Pairs integrated by one implicit network solve, not exact pair exponentials
  (hypothesis allows "implicit exchange update"). Consequence (review): the
  rotation is work-free only instantaneously; over a backward-Euler step an
  isolated equal-mass pair's |D|^2 is multiplied by 1/[(1+alpha)^2 + (alpha b)^2],
  alpha = 2 r dt/m, so nonzero b adds finite-step damping (returned as heat by
  the harness, so total energy still closes).
* The K_tgt normalisation is APPROXIMATE where the interface ratio K_tgt/K_raw
  varies across the interfaces a pair crosses (wide kernels, nonlocal pairs);
  the neutral limit uses V_0 = sqrt(u_*^2 + v_f^2), not u_* alone.
* psi_m is the repo's Businger-Dyer/Dyer helper, which caps zeta to [-10, 10].
  The log uses (z_1+z_0)/z_0 but zeta uses z_1 (hypothesis text). The drag has a
  pole where ln((z_1+z_0)/z_0) = psi_m(-10) ~ 2.55, i.e. z_1/z_0 < ~12; scored
  cases have z_1/z_0 >= 83 (LES first level 8.3 m, z0 0.1 m). Five undamped
  fixed-point sweeps: a scalar Python mirror converged to <0.3% on the scored
  dry/BOMEX surface states (stable cases not checked).
"""
from __future__ import annotations

import jax.numpy as jnp
from legoesm.atmosphere.les_suite.candidate import half_level_average
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    PBLHeightConfig,
    compute_bulk_richardson,
    first_crossing_height,
)
from legoesm.core.bulk_flux import psi_m

from legoesm import constants

KIND = "both"
PARAMS = {
    "a_N": (1.0, 0.1, 5.0, "1"),     # nonlocal (launch->column) exchange amplitude
    "a_P": (0.5, -2.0, 2.0, "1"),    # pressure-turning of momentum differences
    "c_l": (0.4, 0.15, 0.8, "1"),    # local kernel width l = c_l min(z+z0, h)
    "c_st": (0.2, 0.05, 0.5, "1"),   # stable kernel width l_st = c_st V0/sqrt(N+^2+V0^2/h^2)
}

# --- fixed, not fitted (hypothesis) ---
_DELTA = 0.05          # [1] onset scale of P_delta
_V_FLOOR = 0.01        # [m/s] velocity floor v_f
_LAUNCH_FRAC = 0.1     # [1] launch depth / h
_P_CUT = 80.0          # [1] P_delta returns 0 below x = delta/_P_CUT (jump < 1e-38)
_N_MO_ITERS = 5        # fixed Monin-Obukhov fixed-point sweeps (loop count, not a param)
_PBL = PBLHeightConfig()   # Ri_crit, sharpness, h_min for the h diagnosis
_EPS_V = 1.0 / constants.epsilon - 1.0   # R_v/R_d - 1


def init_aux(inputs):
    """Previous-step interface buoyancy flux B_loc [m2 s-3], (n+1,); zero at start."""
    return jnp.zeros_like(inputs.z_half)


def _p_delta(x):
    """P_delta(x) = x exp(-delta/x) for x > 0, else 0 (smooth, safe gradient).

    Below x = delta/_P_CUT the value (< delta/_P_CUT e^-_P_CUT ~ 1e-38) is returned
    as 0 so the float32 gradient never forms 0*inf; NaN input propagates (0*NaN)."""
    pos = x > _DELTA / _P_CUT
    xs = jnp.where(pos, x, 1.0)
    return jnp.where(pos, x * jnp.exp(-_DELTA / xs), 0.0 * x)


def _crossing_parts(c):
    """For upper-triangular pair coefficients c (n, n): (S, P, below), where
    S[K, l] = sum_{j>=K} c[l, j], P[K, l] = sum_{i<K} c[i, l], below[K, l] = l < K."""
    n = c.shape[0]
    s_rev = jnp.flip(jnp.cumsum(jnp.flip(c, 1), 1), 1)
    s_rev = jnp.concatenate([s_rev, jnp.zeros((n, 1), c.dtype)], 1)        # (n, n+1)
    p_exc = jnp.concatenate([jnp.zeros((1, n), c.dtype), jnp.cumsum(c, 0)], 0)  # (n+1, n)
    below = jnp.arange(n)[None, :] < jnp.arange(n + 1)[:, None]
    return s_rev.T, p_exc, below


def _crossing_matrix(c):
    """B (n+1, n) with (B @ X)_K = sum_{i<K<=j} c_ij (X_i - X_j); rows 0 and n are 0."""
    s, p, below = _crossing_parts(c)
    return jnp.where(below, s, -p)


def _crossing_sum(c):
    """sum_{i<K<=j} c_ij for every interface K, (n+1,)."""
    s, _, below = _crossing_parts(c)
    return jnp.sum(jnp.where(below, s, 0.0), axis=1)


def _div(gm):
    """Rows of an interface operator (n+1, .) -> cell divergence G_{k+1} - G_k (n, .)."""
    return gm[1:] - gm[:-1]


def _implicit_scalar(r, x, m, w_cell, w_half, g0, dt):
    """Backward-Euler exchange of x. Cell equation:
    m w_cell (x_new - x)/dt = w_half_k G_k - w_half_{k+1} G_{k+1}, G_0 = g0 prescribed,
    G_n = 0, interior G = B x_new. Returns G (n+1,) with G[0] = g0 and G[n] = 0."""
    n = x.shape[0]
    bm = _crossing_matrix(r)
    a = jnp.diag(m * w_cell) + dt * _div(w_half[:, None] * bm)
    rhs = (m * w_cell * x).at[0].add(dt * w_half[0] * g0)
    x_new = jnp.linalg.solve(a, rhs)
    g_int = (bm @ x_new)[1:n]
    zero = jnp.zeros((1,), x.dtype)
    return jnp.concatenate([jnp.reshape(g0, (1,)).astype(x.dtype), g_int, zero])


def _surface_drag(inputs, b_s):
    """rho_0 C_D S_1 [kg m-2 s-1] and u_* [m/s] from Monin-Obukhov at level 1."""
    kappa = constants.kappa_von_karman
    z1, z0 = inputs.z_full[0], inputs.z0_m
    s1 = jnp.sqrt(inputs.u[0] ** 2 + inputs.v[0] ** 2 + _V_FLOOR ** 2)
    ln_z = jnp.log((z1 + z0) / z0)
    c_d = kappa ** 2 / ln_z ** 2
    for _ in range(_N_MO_ITERS):
        u_star = jnp.sqrt(c_d) * s1
        zeta = -kappa * z1 * b_s / u_star ** 3
        c_d = kappa ** 2 / (ln_z - psi_m(zeta)) ** 2
    return inputs.rho_half[0] * c_d * s1, jnp.sqrt(c_d) * s1


def _operator(params, inputs):
    """Everything except the time update: conductances, rotation, drag, scales."""
    a_n, a_p, c_l, c_st = params["a_N"], params["a_P"], params["c_l"], params["c_st"]
    g = constants.g
    zf, zh, m = inputs.z_full, inputs.z_half, inputs.mass
    n = zf.shape[0]
    dtype = zf.dtype
    tri = jnp.triu(jnp.ones((n, n), dtype), 1)
    z0 = inputs.z0_m

    # Surface buoyancy flux B_s [m2 s-3] (surface air carries no liquid flux).
    theta0 = inputs.temp[0] / inputs.exner[0]
    thv0 = theta0 * (1.0 + _EPS_V * inputs.q_v[0] - inputs.q_c[0])
    b_s = g * (inputs.w_theta_s * (1.0 + _EPS_V * inputs.q_v[0])
               + _EPS_V * theta0 * inputs.w_qt_s) / thv0

    # Mixed-layer depth h: smooth first bulk-Ri crossing, smooth floor h_min.
    def rev(a):
        return a[::-1][None]
    ri_b, _ = compute_bulk_richardson(rev(inputs.temp), rev(inputs.q_v), rev(inputs.u),
                                      rev(inputs.v), rev(inputs.p_full), rev(zf))
    h_c = first_crossing_height(ri_b, rev(zf), _PBL.Ri_crit, _PBL.sharpness)[0]
    h = jnp.sqrt(h_c ** 2 + _PBL.h_min ** 2)

    # Velocity scales.
    c_s, u_star = _surface_drag(inputs, b_s)
    wc3 = _V_FLOOR ** 3 * _p_delta(b_s * h / _V_FLOOR ** 3)
    a3 = (wc3 + _V_FLOOR ** 3) ** (1.0 / 3.0)
    wcf2 = a3 ** 2                                        # w_c^2 + v_f^2
    # w_c^2 = a3^2 - v_f^2 written without cancellation, so it is >= 0 exactly.
    wc2 = wc3 * (a3 + _V_FLOOR) / (a3 ** 2 + a3 * _V_FLOOR + _V_FLOOR ** 2)
    v02 = u_star ** 2 + wcf2
    v0 = jnp.sqrt(v02)

    # Unsaturated-environment N^2 at interior interfaces 1..n-1.
    thvu = inputs.theta_l * (1.0 + _EPS_V * inputs.q_t)
    dzf = zf[1:] - zf[:-1]
    n2 = g * (thvu[1:] - thvu[:-1]) / (0.5 * (thvu[1:] + thvu[:-1]) * dzf)
    return dict(a_n=a_n, a_p=a_p, c_l=c_l, c_st=c_st, tri=tri, z0=z0, b_s=b_s, h=h,
                c_s=c_s, u_star=u_star, wc2=wc2, v02=v02, v0=v0, n2=n2, dzf=dzf,
                thvu=thvu, n=n, dtype=dtype)


def _conductance(o, inputs, aux):
    """r_ij (upper triangular, kg m-2 s-1) and the interface diagnostics."""
    g, kappa = constants.g, constants.kappa_von_karman
    zf, zh, m = inputs.z_full, inputs.z_half, inputs.mass
    n, tri, h, v0, v02, z0 = o["n"], o["tri"], o["h"], o["v0"], o["v02"], o["z0"]
    zi = zh[1:n]                                     # interior interfaces

    # Regime gate chi and stable width at interior interfaces.
    x_n = o["n2"] * h ** 2 / v02
    x_b = aux[1:n] * h / (v0 * v02)
    chi_k = (1.0 - jnp.exp(-_p_delta(x_n))) * jnp.exp(-_p_delta(x_b))
    lst_k = o["c_st"] * h / jnp.sqrt(1.0 + _p_delta(x_n))
    lam_k = (1.0 - chi_k) * jnp.minimum(zi + z0, h) + chi_k * lst_k   # mixing length
    k_tgt = kappa * v0 * lam_k                                         # [m2/s]

    # Pair geometry.
    d = jnp.abs(zf[None, :] - zf[:, None])
    z_m = 0.5 * (zf[None, :] + zf[:, None])
    dz_floor = jnp.concatenate([o["dzf"], o["dzf"][-1:]])[:, None]   # D_i (lower cell)
    chi = jnp.interp(z_m, zi, chi_k)
    l_st = jnp.interp(z_m, zi, lst_k)
    l_loc = o["c_l"] * jnp.minimum(z_m + z0, h)
    mm = m[:, None] * m[None, :]
    w = tri * mm * ((1.0 - chi) * jnp.exp(-d ** 2 / (l_loc ** 2 + dz_floor ** 2))
                    + chi * jnp.exp(-d ** 2 / (l_st ** 2 + dz_floor ** 2)))
    k_raw = _crossing_sum(w * d)[1:n] / inputs.rho_half[1:n]          # [kg]
    s = jnp.interp(z_m, zi, k_tgt / k_raw)
    r_loc = s * w

    # Nonlocal launch exchange through the lifted-parcel buoyancy barrier.
    th_p, qv_p, qc_p = saturation_adjust(
        jnp.broadcast_to(inputs.theta_l[:, None], (n, n)),
        jnp.broadcast_to(inputs.q_t[:, None], (n, n)),
        jnp.broadcast_to(inputs.exner[None, :], (n, n)),
        jnp.broadcast_to(inputs.p_full[None, :], (n, n)))
    thv_p = th_p * (1.0 + _EPS_V * qv_p - qc_p)          # parcel i at level k
    thv_e = jnp.diagonal(thv_p)                          # environment = own parcel
    b_par = g * (thv_p - thv_e[None, :]) / thv_e[None, :]
    dz_cell = (zh[1:] - zh[:-1])[None, :]
    neg = tri * dz_cell * (v02 / h) * _p_delta(-b_par * h / v02)
    e_bar = jnp.cumsum(neg, axis=1)                       # E_ij [m2 s-2]
    w_d = jnp.exp(-d / h)
    norm = jnp.sum(tri * m[None, :] * w_d, axis=1)
    norm = jnp.where(norm > 0.0, norm, 1.0)              # top cell has no pairs
    g_launch = jnp.exp(-zf / (_LAUNCH_FRAC * h))
    r_nl = (o["a_n"] * (o["wc2"] / v02) * (v0 / h) * tri
            * (m * g_launch / norm)[:, None] * m[None, :] * w_d * jnp.exp(-e_bar / v02))
    return r_loc + r_nl, dict(chi_k=chi_k, lam_k=lam_k, k_tgt=k_tgt, e_bar=e_bar)


def _rotation_coeff(a_p, u, v):
    """b_ij = a_P cross(V_i, V_j)/sqrt((|V_i|^2+v_f^2)(|V_j|^2+v_f^2)), V = U (U_s = 0)."""
    cross = u[:, None] * v[None, :] - v[:, None] * u[None, :]
    s2 = u ** 2 + v ** 2 + _V_FLOOR ** 2
    return a_p * cross / jnp.sqrt(s2[:, None] * s2[None, :])


def _implicit_momentum(r, rb, inputs, c_s):
    """Backward Euler of the coupled (u, v) exchange + rotation + implicit drag.
    Returns (G_u, G_v, G_rot_u, G_rot_v) (n+1,) mass-weighted, top exactly 0."""
    m, dt = inputs.mass, inputs.dt
    n = m.shape[0]
    b_r = _crossing_matrix(r)
    b_b = _crossing_matrix(rb)
    e00 = jnp.zeros((n + 1, n), m.dtype).at[0, 0].set(1.0)
    gu_mat = jnp.concatenate([b_r - c_s * e00, -b_b], axis=1)   # (n+1, 2n)
    gv_mat = jnp.concatenate([b_b, b_r - c_s * e00], axis=1)
    a = jnp.diag(jnp.concatenate([m, m])) + dt * jnp.concatenate(
        [_div(gu_mat), _div(gv_mat)], axis=0)
    x = jnp.linalg.solve(a, jnp.concatenate([m * inputs.u, m * inputs.v]))
    u_new, v_new = x[:n], x[n:]
    zero = jnp.zeros((1,), m.dtype)

    def pack(g0, g):
        return jnp.concatenate([jnp.reshape(g0, (1,)), g[1:n], zero])

    g_u = pack(-c_s * u_new[0], gu_mat @ x)
    g_v = pack(-c_s * v_new[0], gv_mat @ x)
    g_rot_u = pack(zero[0], -(b_b @ v_new))
    g_rot_v = pack(zero[0], b_b @ u_new)
    return g_u, g_v, g_rot_u, g_rot_v


def _solve(params, inputs, aux):
    o = _operator(params, inputs)
    r, diag = _conductance(o, inputs, aux)
    m, dt, rho_h = inputs.mass, inputs.dt, inputs.rho_half
    ones_c, ones_h = jnp.ones_like(m), jnp.ones_like(rho_h)
    g_th = _implicit_scalar(r, inputs.theta_l, m, inputs.exner, inputs.exner_half,
                            rho_h[0] * inputs.w_theta_s, dt)
    g_q = _implicit_scalar(r, inputs.q_t, m, ones_c, ones_h, rho_h[0] * inputs.w_qt_s, dt)
    rb = r * _rotation_coeff(o["a_p"], inputs.u, inputs.v)
    g_u, g_v, g_ru, g_rv = _implicit_momentum(r, rb, inputs, o["c_s"])
    return o, diag, (g_th, g_q, g_u, g_v, g_ru, g_rv)


def fluxes(params, inputs, aux):
    """((w'theta_l', w'q_t'), (u'w', v'w'), B_loc_new): kinematic, (n+1,), +up."""
    _, _, (g_th, g_q, g_u, g_v, _, _) = _solve(params, inputs, aux)
    rho_h = inputs.rho_half
    zero = jnp.zeros((1,), rho_h.dtype)
    wth = jnp.concatenate([jnp.reshape(inputs.w_theta_s, (1,)).astype(rho_h.dtype),
                           g_th[1:-1] / rho_h[1:-1], zero])
    wqt = jnp.concatenate([jnp.reshape(inputs.w_qt_s, (1,)).astype(rho_h.dtype),
                           g_q[1:-1] / rho_h[1:-1], zero])
    uw = jnp.concatenate([g_u[:-1] / rho_h[:-1], zero])
    vw = jnp.concatenate([g_v[:-1] / rho_h[:-1], zero])
    # Carried state: modelled interface buoyancy flux (dry-air linearisation).
    th_h = half_level_average(inputs.theta_l, inputs.theta_l[0])
    qt_h = half_level_average(inputs.q_t, inputs.q_t[0])
    thv_h = th_h * (1.0 + _EPS_V * qt_h)
    b_new = constants.g * ((1.0 + _EPS_V * qt_h) * wth + _EPS_V * th_h * wqt) / thv_h
    return (wth, wqt), (uw, vw), b_new


def pressure_budget_terms(params, inputs, aux):
    """Map the modelled momentum flux onto the LES pressure terms (review fix 6).

    LES ``pres_uw`` = <u' dp'/dz + w' dp'/dx> (kinematic p), i.e. MINUS the pressure
    term of the u'w' budget. Under the local-equilibrium Rotta reading of an
    exchange closure, d<u'w'>/dt = P - F/tau + Pi_turn = 0 with tau the local
    exchange time tau_K = lambda_K / V_0, the flux splits as F = F_norot + F_rot with
    F_norot = tau P and F_rot = tau Pi_turn, so the TOTAL pressure term is
    -F_norot/tau and LES-convention pres_uw_model = +F_norot/tau. The turning share
    Pi_turn = F_rot/tau is the part a_P controls. Interior interfaces only (index 0
    and n set to 0). Units m2 s-3. Diagnostic: the harness never calls it.
    """
    o, diag, (_, _, g_u, g_v, g_ru, g_rv) = _solve(params, inputs, aux)
    n = o["n"]
    rho = inputs.rho_half[1:n]
    tau = diag["lam_k"] / o["v0"]
    zero = jnp.zeros((1,), rho.dtype)

    def pad(x):
        return jnp.concatenate([zero, x, zero])

    f_ru, f_rv = g_ru[1:n] / rho, g_rv[1:n] / rho
    f_nu, f_nv = g_u[1:n] / rho - f_ru, g_v[1:n] / rho - f_rv
    return dict(pres_uw=pad(f_nu / tau), pres_vw=pad(f_nv / tau),
                turn_uw=pad(f_ru / tau), turn_vw=pad(f_rv / tau))


def _selftest_x64():
    """Review fix 8 (NOT RUN on the login node; run on a compute node with
    JAX_ENABLE_X64=1). Asserts (a) column momentum changes by exactly the surface
    flux, (b) the pairwise rotation does zero work, and (c) a planted
    non-antisymmetric rotation (J U_i instead of J (U_i - U_j)) is DETECTED."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.atmosphere.les_suite.candidate import column_inputs, flux_to_tendency

    n = 24
    zh = jnp.linspace(0.0, 2400.0, n + 1)
    zf = 0.5 * (zh[1:] + zh[:-1])
    rho_f, rho_h = 1.2 * jnp.exp(-zf / 8000.0), 1.2 * jnp.exp(-zh / 8000.0)
    p_f, p_h = constants.p_ref * jnp.exp(-zf / 8000.0), constants.p_ref * jnp.exp(-zh / 8000.0)
    ex_f, ex_h = (p_f / constants.p_ref) ** constants.kappa, (p_h / constants.p_ref) ** constants.kappa
    th = 300.0 + 0.004 * jnp.maximum(zf - 800.0, 0.0)
    inp = column_inputs(
        z_full=zf, z_half=zh, rho_full=rho_f, rho_half=rho_h, mass=rho_f * (zh[1:] - zh[:-1]),
        exner=ex_f, exner_half=ex_h, p_full=p_f, p_half=p_h, theta_l=th,
        q_t=0.012 * jnp.exp(-zf / 2000.0), u=5.0 + 0.002 * zf, v=2.0 * jnp.sin(zf / 500.0),
        w_theta_s=0.05, w_qt_s=5e-5, uw_s=jnp.nan, vw_s=jnp.nan, f_c=1e-4, dt=30.0,
        z0_m=0.1, z0h_m=0.1)
    params = {k: jnp.asarray(v[0]) for k, v in PARAMS.items()}
    aux = init_aux(inp)
    (wth, wqt), (uw, vw), _ = fluxes(params, inp, aux)
    du = flux_to_tendency(uw, rho_h, inp.mass)
    col = jnp.sum(inp.mass * du)
    ok_mom = abs(float(col - rho_h[0] * uw[0])) <= 1e-12 * abs(float(rho_h[0] * uw[0]))
    ok_ends = (float(wth[0]) == 0.05 and float(wqt[0]) == 5e-5
               and all(float(f[-1]) == 0.0 for f in (wth, wqt, uw, vw)))

    o = _operator(params, inp)
    r, _ = _conductance(o, inp, aux)
    rb = r * _rotation_coeff(o["a_p"], inp.u, inp.v)
    du_k, dv_k = inp.u[1:] - inp.u[:-1], inp.v[1:] - inp.v[:-1]

    def rel_work(gu, gv):
        w = gu[1:n] * du_k + gv[1:n] * dv_k
        return float(jnp.abs(jnp.sum(w)) / jnp.sum(jnp.abs(w)))

    bb = _crossing_matrix(rb)
    good = rel_work(-(bb @ inp.v), bb @ inp.u)
    s, _, below = _crossing_parts(rb)
    planted_b = jnp.where(below, s, 0.0)                 # sum r b J U_i only
    planted = rel_work(-(planted_b @ inp.v), planted_b @ inp.u)

    # Implicit residual: the state the harness builds from the reported fluxes
    # (u + dt du) must reproduce those fluxes through the operator. A planted zero
    # interior flux must FAIL this (the telescoping check above cannot see it).
    br = _crossing_matrix(r)

    def residual(uw_, vw_):
        un = inp.u + inp.dt * flux_to_tendency(uw_, rho_h, inp.mass)
        vn = inp.v + inp.dt * flux_to_tendency(vw_, rho_h, inp.mass)
        gu = jnp.concatenate([(-o["c_s"] * un[0])[None], (br @ un - bb @ vn)[1:n]])
        gv = jnp.concatenate([(-o["c_s"] * vn[0])[None], (bb @ un + br @ vn)[1:n]])
        ref = jnp.concatenate([rho_h[:n] * uw[:n], rho_h[:n] * vw[:n]])
        got = jnp.concatenate([gu, gv])
        mine = jnp.concatenate([rho_h[:n] * uw_[:n], rho_h[:n] * vw_[:n]])
        return float(jnp.max(jnp.abs(got - mine)) / jnp.max(jnp.abs(ref)))

    res_good = residual(uw, vw)
    zero_int = jnp.zeros(n - 1)
    res_planted = residual(jnp.concatenate([uw[:1], zero_int, uw[-1:]]),
                           jnp.concatenate([vw[:1], zero_int, vw[-1:]]))
    result = dict(momentum_closes=ok_mom, ends_exact=ok_ends, rot_work_rel=good,
                  planted_rot_work_rel=planted, implicit_residual=res_good,
                  planted_implicit_residual=res_planted)
    assert (ok_mom and ok_ends and good < 1e-12 and planted > 1e-6
            and res_good < 1e-10 and res_planted > 1e-6), result
    return result


if __name__ == "__main__":
    print(_selftest_x64())
