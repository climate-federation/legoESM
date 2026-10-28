"""Round-1 candidate C3_transilient (co-scientist harness plug-in, KIND="both").

Asymmetric buoyancy-barrier transilient matrix: ONE parcel-energy kernel moves
theta_l, q_t, u and v; momentum differs from scalars only by a Gregory pressure
coefficient C_u and by Pr_t in the local part. The candidate computes its own
surface stress (MOST + convective gustiness). Surface heat/moisture fluxes are the
prescribed ones, bit-exact.

Grid (0-based, surface-first): cells k = 0..N-1, interior interfaces i = 1..N-1
(interface i sits above cell n = i-1). Mass flux F = rho w'phi' [kg m^-2 s^-1 phi],
positive up.

(0) Surface stress. B_0 = (g/theta_v0)(w'theta'_s + (1/eps - 1) theta_0 w'q'_s);
    zeta = -kappa z_1 B_0 / (u_*^3 + U_REG^3); U_eff = (|U_1|^2 + (beta_g w_L)^2
    + U_REG^2)^(1/2); u_* = kappa U_eff / [ln(z_1/z_0) - psi_m(zeta) + psi_m(zeta z_0/z_1)]
    (Paulson / Dyer psi_m from legoesm.core.bulk_flux, gamma 16, beta 5).
    Kinematic stress u'w'_0 = -(u_*^2/U_eff) u_0, implicit in u_0 (drag coefficient
    frozen at t^n), same for v.
(1) Launch. b_s = zeta_s (u_*^3 + U_REG^3)/(kappa Z_b); B_0+ = H(B_0) B_0^3/(B_0^2 + b_s^2)
    (C2 at B_0 = 0); w_L = c_w [(B_0+ Z_b + W_REG^3)^(1/3) - W_REG]; K_0 = w_L^2/2 + K_EPS;
    M_L = C_M rho_0 w_L.
(2) Upward survival. b_k = g (theta_v^p(theta_l0, q_t0, p_k) - theta_v,k)/theta_v,k
    (saturation via les_suite.scm_coupling.saturation_adjust -> thermo).
    x_0 = 1, x_n = 1 + (a_b/K_0) sum_{k=1..n} b_k (z_k - z_{k-1}).
    Soft running min, dz-weighted and capped at 1:
    m_n = -dK ln[exp(-1/dK) + sum_{1<=k<=n} (dz_k/Z_ref) exp(-x_k/dK)] <= 1, Z_ref = Z_b
    (the launch level k = 0 is represented by the anchor alone, so m_0 = 1, S_0 = 1).
    g_n = sig(m_n/dK)/sig(1/dK) in (0, 1]; Z_b = sum_{n<N-1} g_n (z_{n+1} - z_n).
    S_n = exp(-c_d (z_n - z_0)/Z_b) g_n (n < N-1), S_{N-1} = 0; P_j = S_{j-1} - S_j >= 0
    (S_{-1} = 1). Upward long-jump mass above cell n: M_n = M_L S_n.
(3) Return path. xd(j,n) = 1 - (a_b/K_0) sum_{k=n..j-1} b_dn(j,k)(z_{k+1} - z_k),
    b_dn(j,k) = buoyancy of an undiluted parcel with cell-j (theta_l, q_t) at level k.
    A_j = sig(smin_j/dK)/sig(1/dK) with smin_j the same capped dz-weighted soft-min over
    n < j, so A_j <= 1. r_j = alpha_s A_j <= 1; chi_j = 1 - A_j (regime diagnostic).
    R_n = sum_{j>n} r_j M_L P_j; D_n = M_n - R_n >= 0 (since r_j <= 1).
(4) Nonlocal flux above cell n, for each phi with C = 0 (theta_l, q_t) or C_u (u, v):
    F_NL = (1-C)[M_n phi_0 - sum_{j>n} r_j M_L P_j phi_j - D_n phi_{n+1}]
           + C M_n (phi_n - phi_{n+1}).
(5) Local flux F_L = -rho_h K (phi_{n+1} - phi_n)/(z_{n+1} - z_n);
    1/l = 1/(kappa z_h) + 1/lambda_inf; Ri = N^2/(S^2 + S_EPS^2);
    K_m = l^2 [S^2 + S_EPS^2 + c_c H(-N^2)|N^2|^3/(N^4 + S^4 + S_EPS^4)]^(1/2) G(Ri),
    G = [softplus((1 - R(Ri)/Ri_c)/DELTA_F)/softplus(1/DELTA_F)]^2, R(x) = H(x) x^3/(x^2+1),
    K_h = K_m / (Pr_t0 (1 + R(Ri)/Ri_c)).
(6) Linearly implicit update with all coefficients frozen at t^n:
    m (phi^{n+1} - phi^n)/dt = F^{n+1}_below - F^{n+1}_above, dense N x N solves for
    q_t, for theta_l and for (u, v). theta_l uses the operator the harness applies under
    heat_form="enthalpy": m_k Pi_k dtheta_k/dt = Pi_h,k F_k - Pi_h,k+1 F_{k+1}. The
    REPORTED flux is F^{n+1} = G phi^{n+1}, so the harness's forward-Euler flux-form
    application reproduces phi^{n+1} to round-off (theta_l: only under "enthalpy"; under
    "theta" the internal theta_l^{n+1} differs from the applied one by O(dPi)).

Conservation (RULE 0): every term is an interface flux; the top flux is exactly 0
(S_{N-1} = 0 -> M = R = D = 0 at the lid, K = 0 there), the surface scalar fluxes are
the prescribed values bit-exact, and the harness converts flux to tendency, so column
theta_l/enthalpy, q_t and momentum change only by the surface flux. Net mass exchange
across every interface is M - R - D = 0. All transfer coefficients ((1-C) M, r_j M_L P_j,
D, C M, rho K/dz, surface drag) are >= 0 for C in [0, 1], so diag(m) - dt A is an
M-matrix with positive column sums (column-diagonally dominant): q_t^{n+1} >= 0 for any
dt when q_t^n >= 0 and the surface moisture flux is >= 0. Nothing is clipped. Kinetic
energy removed by momentum mixing is returned as heat by the harness (ke_heating).

Limits (as implemented):
- Neutral (B_0 = 0): B_0+ = 0 exactly, w_L = c_w[cbrt(W_REG^3) - W_REG] = 0 to round-off,
  so M_L = 0; K_m = l^2 S G(Ri<=0) = l^2 S, Pr_t = Pr_t0; with l -> kappa z for
  z << lambda_inf this is the log law, consistent with the MOST stress.
- Free convection: Z_ref = Z_b keeps the dz-weighted soft-min a function of z/Z_b, so the
  nonlocal kernel is self-similar in z/h (to the extent the fixed point converges); the
  local K = l^2 sqrt(c_c)|N| carries lambda_inf (dimensional) and breaks similarity there.
- Stable: R(Ri) ~ Ri^3 for Ri << 1 and ~ Ri for Ri >> 1, so the effective cutoff is NOT
  Ri_c but Ri* solving R(Ri*) = Ri_c (Ri* ~ 0.73 at Ri_c = 0.25); beyond Ri*, G decays
  like exp(-2(R/Ri_c - 1)/DELTA_F). Pr_t = Pr_t0 (1 + R/Ri_c) = 2 Pr_t0 at Ri*. The
  round-0 z-less claim (K_m -> u_*^2/S, beta_eff = 2 Pr_t0/Ri_c) does not follow from this
  G and is NOT claimed; for z >> lambda_inf, l -> lambda_inf gives z-independent K, the
  only z-less mechanism here. Rossby-number similarity of the neutral Ekman layer is
  broken (lambda_inf fixed, not ~ u_*/f).
- Stated breaks: launch from cell 0 only (grid dependent); no propagating wave momentum
  flux (top flux 0 by construction, absorb_top is the identity and is not called);
  local N^2 from grid-mean theta_v; undiluted parcels.

Deviations from the hypothesis text (each also listed in the implementation record):
- theta_l, not s_l, is the transported heat variable: the harness flux variable is
  w'theta_l' and the harness applies it as an enthalpy flux (heat_form="enthalpy").
- Fix 1: the value 1 is in both soft-min sets, so m_n <= 1 and A_j <= 1.
- Fix 2: soft-mins are dz-weighted (weights dz_k/Z_ref) with Z_ref = Z_b of the previous
  fixed-point iterate (Z_ref = column depth on the K_EPS initial pass).
- Fix 3: U_REG in 1/L, b_s and U_eff; W_REG smooth cube root; S_EPS^2 inside the sqrt of
  K_m and S_EPS^4 in the convective denominator.
- Fix 4: the TKE diagnostic is dropped (no score for TKE from this candidate).
- The theta_l implicit solve uses the harness's enthalpy-form operator (see (6)); the
  w_p diagnostic carries W_REG^2 inside its sqrt (finite gradient where softplus
  underflows).
- Buoyancy-work path increments use full-level spacing (z_k - z_{k-1}) up and
  (z_{k+1} - z_k) down, not the cell thickness dz_k; cell thickness is the soft-min weight.
- Local N^2 uses the interface mean of the neighbouring theta_v as theta_v0.
- psi_m clips zeta to [-10, 10] (shared bulk_flux helper), on BOTH arguments. This is
  a departure from Paulson/Dyer that is active in strong free convection (light wind,
  large B_0: zeta = z_1/L << -10 is reachable) and in strong stability; there u_* differs
  from the unclipped law (OPEN, needs an unclipped option in the shared helper).
- Diagnostics w_p, w'w'_NL, w'w'theta'_NL (explicit, at phi^n) via :func:`diagnostics`;
  aux carries nothing (no prognostic state).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.core.bulk_flux import psi_m

from legoesm import constants

KIND = "both"

PARAMS = {
    "C_M": (0.10, 0.02, 0.5, "1"),            # launch area-velocity fraction
    "c_w": (1.0, 0.3, 2.0, "1"),              # launch velocity coefficient
    "a_b": (0.5, 0.2, 1.0, "1"),              # buoyancy-work efficiency
    "delta_K": (0.25, 0.05, 1.0, "1"),        # parcel-energy spread (barrier width)
    "c_d": (1.0, 0.1, 3.0, "1"),              # lateral detrainment, eps_d = c_d/Z_b
    "alpha_s": (0.7, 0.0, 1.0, "1"),          # max direct-return fraction
    "C_u": (0.5, 0.0, 1.0, "1"),              # Gregory pressure coefficient (u, v)
    "zeta_s": (0.3, 0.05, 2.0, "1"),          # convective gate scale on -Z_b/L
    "Pr_t0": (0.8, 0.5, 1.2, "1"),            # neutral turbulent Prandtl number
    "Ri_c": (0.25, 0.15, 1.0, "1"),           # Richardson scale of the stable ramp
    "c_c": (1.0, 0.1, 10.0, "1"),             # free-convective local mixing
    "lambda_inf": (60.0, 10.0, 200.0, "m"),   # Blackadar asymptotic mixing length
    "beta_g": (1.0, 0.3, 1.5, "1"),           # surface gustiness coefficient
}

# --- fixed numerics (not trainable) -------------------------------------------------
N_FP = 4            # fixed-point iterations for (Z_b, u_*, w_L)
DELTA_F = 0.05      # width of the stable-cutoff softplus [1]
K_EPS = 1e-8        # parcel-energy floor [m^2 s^-2] (rate coefficient only)
S_EPS = 1e-6        # shear floor [s^-1] (rate coefficient only)
U_REG = 1e-2        # calm-wind / calm-u_* regulariser [m s^-1]
W_REG = 1e-3        # smooth cube-root offset for w_L [m s^-1]
_VT = 1.0 / constants.epsilon - 1.0   # virtual-temperature coefficient (~0.608)


def init_aux(inputs):
    return ()


# --- small helpers ------------------------------------------------------------------

def _ramp(x):
    """H(x) x^3/(x^2+1), written to avoid x^3 overflow; C2 at 0, finite both branches."""
    x2 = x * x
    return jnp.where(x > 0.0, x * (x2 / (x2 + 1.0)), 0.0)


def _gate(m, delta):
    """sig(m/delta)/sig(1/delta): in (0, 1] whenever m <= 1."""
    return jax.nn.sigmoid(m / delta) / jax.nn.sigmoid(1.0 / delta)


def _softmin_one(x, w, delta):
    """-delta ln[exp(-1/delta) + sum_k w_k exp(-x_k/delta)] over the last axis.

    ``w >= 0`` (0 excludes an entry). The leading 1 caps the result at <= 1."""
    lead = jnp.broadcast_to(-1.0 / delta, x.shape[:-1] + (1,)).astype(x.dtype)
    a = jnp.concatenate([lead, -x / delta], axis=-1)
    b = jnp.concatenate([jnp.ones(x.shape[:-1] + (1,), x.dtype), w], axis=-1)
    return -delta * logsumexp(a, axis=-1, b=b)


def _theta_v(theta, q_v, q_c):
    return theta * (1.0 + _VT * q_v - q_c)


def _grid(inputs):
    z, zh = inputs.z_full, inputs.z_half
    dzf = z[1:] - z[:-1]                                   # (N-1,) full-level spacing
    zero = jnp.zeros((1,), z.dtype)
    dzl = jnp.concatenate([zero, dzf])                     # z_k - z_{k-1}, 0 at k=0
    dzu = jnp.concatenate([dzf, zero])                     # z_{k+1} - z_k, 0 at top
    dzc = zh[1:] - zh[:-1]                                 # cell thickness
    return dzf, dzl, dzu, dzc


def _buoyancy(inputs):
    """b[j, k]: buoyancy [m s^-2] of an undiluted parcel with cell-j (theta_l, q_t)
    placed at level k, vs the grid-mean theta_v at k. Row 0 is the launch parcel."""
    n = inputs.theta_l.shape[0]
    shp = (n, n)
    th_p, qv_p, qc_p = saturation_adjust(
        jnp.broadcast_to(inputs.theta_l[:, None], shp),
        jnp.broadcast_to(inputs.q_t[:, None], shp),
        jnp.broadcast_to(inputs.exner[None, :], shp),
        jnp.broadcast_to(inputs.p_full[None, :], shp))
    thv_e = _theta_v(inputs.temp / inputs.exner, inputs.q_v, inputs.q_c)
    return constants.g * (_theta_v(th_p, qv_p, qc_p) - thv_e[None, :]) / thv_e[None, :], thv_e


def _up_gate(b_up, K0, z_ref, a_b, dK, dzl, dzc):
    """Parcel energy x_n and the capped running-min survival gate g_n (N,)."""
    n = b_up.shape[0]
    x = 1.0 + a_b / K0 * jnp.cumsum(b_up * dzl)
    kk, nn = jnp.arange(n)[None, :], jnp.arange(n)[:, None]
    low = (kk <= nn) & (kk >= 1)                                     # 1 <= k <= n
    w = jnp.where(low, (dzc / z_ref)[None, :], 0.0)
    m = _softmin_one(jnp.broadcast_to(x[None, :], (n, n)), w, dK)
    return x, _gate(m, dK)


def _ustar(u1sq, w_l, b0, ust_prev, z1, z0, beta_g):
    kap = constants.kappa_von_karman
    u_eff = jnp.sqrt(u1sq + (beta_g * w_l) ** 2 + U_REG ** 2)
    zeta = -kap * z1 * b0 / (ust_prev ** 3 + U_REG ** 3)
    den = jnp.log(z1 / z0) - psi_m(zeta) + psi_m(zeta * z0 / z1)
    return kap * u_eff / den, u_eff


def kernel(params, inputs):
    """Everything the fluxes need, at t^n (also the falsifier entry point: evaluate it on
    LES mean profiles). Returns a dict of arrays."""
    p = params
    dK = p["delta_K"]
    kap = constants.kappa_von_karman
    z, zh = inputs.z_full, inputs.z_half
    n = z.shape[0]
    _dzf, dzl, dzu, dzc = _grid(inputs)
    bmat, thv_e = _buoyancy(inputs)
    b_up = bmat[0]

    theta0 = inputs.temp[0] / inputs.exner[0]
    b0 = constants.g / thv_e[0] * (inputs.w_theta_s + _VT * theta0 * inputs.w_qt_s)
    z1, z0 = z[0], inputs.z0_m
    u1sq = inputs.u[0] ** 2 + inputs.v[0] ** 2

    def depth(gate):
        return jnp.sum(gate[:-1] * dzu[:-1])

    # initial pass: classical parcel-method depth (K_0 = K_EPS), neutral u_*, w_L = 0
    _x, g0 = _up_gate(b_up, K_EPS, zh[-1], p["a_b"], dK, dzl, dzc)
    zb = depth(g0)
    ust = kap * jnp.sqrt(u1sq + U_REG ** 2) / jnp.log(z1 / z0)
    w_l = jnp.zeros_like(zb)
    for _ in range(N_FP):                     # static trip count, no traced branching
        b_s = p["zeta_s"] * (ust ** 3 + U_REG ** 3) / (kap * zb)
        b0p = jnp.where(b0 > 0.0, b0 * (b0 * b0 / (b0 * b0 + b_s * b_s)), 0.0)
        w_l_new = p["c_w"] * (jnp.cbrt(b0p * zb + W_REG ** 3) - W_REG)
        K0 = 0.5 * w_l_new ** 2 + K_EPS
        _x, g = _up_gate(b_up, K0, zb, p["a_b"], dK, dzl, dzc)
        zb_new = depth(g)
        ust, _ = _ustar(u1sq, w_l_new, b0, ust, z1, z0, p["beta_g"])
        zb, w_l = zb_new, w_l_new

    K0 = 0.5 * w_l ** 2 + K_EPS
    x, g = _up_gate(b_up, K0, zb, p["a_b"], dK, dzl, dzc)
    u_eff = jnp.sqrt(u1sq + (p["beta_g"] * w_l) ** 2 + U_REG ** 2)   # as in _ustar
    c_s = ust ** 2 / u_eff                   # kinematic drag coefficient [m s^-1]

    top = jnp.arange(n) == n - 1
    S = jnp.where(top, 0.0, jnp.exp(-p["c_d"] * (z - z[0]) / zb) * g)
    P = jnp.concatenate([jnp.ones((1,), S.dtype), S[:-1]]) - S
    M_L = p["C_M"] * inputs.rho_full[0] * w_l

    # return path: xd[j, n] for n < j
    below = jnp.arange(n)[None, :] < jnp.arange(n)[:, None]         # [j, k]: k < j
    wd = jnp.where(below, bmat * dzu[None, :], 0.0)
    crev = jnp.cumsum(wd[:, ::-1], axis=1)[:, ::-1]                 # sum_{k>=n, k<j}
    xd = 1.0 - p["a_b"] / K0 * crev
    w_dn = jnp.where(below, (dzc / zb)[None, :], 0.0)
    A = _gate(_softmin_one(xd, w_dn, dK), dK)
    r = p["alpha_s"] * A

    Rj = r * M_L * P                                                 # (N,) by destination
    M = M_L * S[:-1]                                                 # (N-1,) above cell n
    R = jnp.cumsum(Rj[::-1])[::-1][1:]                               # sum_{j>n} Rj
    D = M - R

    # local K at interior interfaces
    du = inputs.u[1:] - inputs.u[:-1]
    dv = inputs.v[1:] - inputs.v[:-1]
    dzi = dzu[:-1]
    S2 = (du * du + dv * dv) / dzi ** 2
    thv_mid = 0.5 * (thv_e[1:] + thv_e[:-1])
    N2 = constants.g * (thv_e[1:] - thv_e[:-1]) / (thv_mid * dzi)
    Ri = N2 / (S2 + S_EPS ** 2)
    ell = 1.0 / (1.0 / (kap * zh[1:n]) + 1.0 / p["lambda_inf"])
    # = -N2^3/(N2^2 + S2^2 + S_EPS^4); written so the AD path never forms the tiny
    # denominator's inverse square (float32 overflow near N2 = S2 = 0)
    conv = jnp.where(N2 < 0.0,
                     -N2 * (N2 / jnp.sqrt(N2 * N2 + S2 * S2 + S_EPS ** 4)) ** 2, 0.0)
    rri = _ramp(Ri) / p["Ri_c"]
    Gri = (jax.nn.softplus((1.0 - rri) / DELTA_F) / jax.nn.softplus(1.0 / DELTA_F)) ** 2
    Km = ell ** 2 * jnp.sqrt(S2 + S_EPS ** 2 + p["c_c"] * conv) * Gri
    Kh = Km / (p["Pr_t0"] * (1.0 + rri))

    # W_REG^2 keeps the sqrt derivative finite where softplus underflows to 0
    w_p = jnp.sqrt(2.0 * K0 * dK * jax.nn.softplus(x / dK) + W_REG ** 2)
    return dict(B0=b0, Z_b=zb, u_star=ust, w_L=w_l, K0=K0, U_eff=u_eff, c_s=c_s,
                x=x, S=S, P=P, A=A, chi=1.0 - A, M_L=M_L, M=M, R=R, D=D, Rj=Rj,
                Km=Km, Kh=Kh, Ri=Ri, w_p=w_p)


def _flux_matrix(k, C, K, inputs):
    """G (N-1, N): interior interface mass flux = G @ phi (nonlocal + local)."""
    n = inputs.theta_l.shape[0]
    dt_ = inputs.theta_l.dtype
    e0 = jnp.eye(n - 1, n, dtype=dt_)
    e1 = jnp.eye(n - 1, n, k=1, dtype=dt_)
    col0 = (jnp.arange(n) == 0).astype(dt_)[None, :]
    upper = jnp.arange(n)[None, :] > jnp.arange(n - 1)[:, None]     # j > n
    M, D, Rj = k["M"], k["D"], k["Rj"]
    nonlocal_ = ((1.0 - C) * (M[:, None] * col0 - jnp.where(upper, Rj[None, :], 0.0)
                              - D[:, None] * e1)
                 + C * M[:, None] * (e0 - e1))
    kcoef = inputs.rho_half[1:n] * K / (inputs.z_full[1:] - inputs.z_full[:-1])
    return nonlocal_ + kcoef[:, None] * (e0 - e1)


def _divergence(n, dtype):
    """Dv (N, N-1): m_k dphi_k/dt contribution from interior interface fluxes."""
    return jnp.eye(n, n - 1, k=-1, dtype=dtype) - jnp.eye(n, n - 1, dtype=dtype)


def fluxes(params, inputs, aux):
    k = kernel(params, inputs)
    n = inputs.theta_l.shape[0]
    dt_ = inputs.theta_l.dtype
    m, dt, rh = inputs.mass, inputs.dt, inputs.rho_half
    dv_ = _divergence(n, dt_)
    zero = jnp.zeros((1,), dt_)

    # scalars: C = 0, K_h, prescribed surface flux (explicit, bit-exact)
    g_s = _flux_matrix(k, 0.0, k["Kh"], inputs)
    e_0 = (jnp.arange(n) == 0).astype(dt_)
    new_q = jnp.linalg.solve(jnp.diag(m) - dt * dv_ @ g_s,
                             m * inputs.q_t + dt * e_0 * rh[0] * inputs.w_qt_s)
    # theta_l: the harness applies the heat flux as an enthalpy flux (Exner-weighted)
    ex, exh = inputs.exner, inputs.exner_half
    a_h = (dv_ * exh[1:n][None, :]) @ g_s / ex[:, None]
    src_h = e_0 * rh[0] * exh[0] * inputs.w_theta_s / ex[0]
    new_th = jnp.linalg.solve(jnp.diag(m) - dt * a_h, m * inputs.theta_l + dt * src_h)
    fs = (g_s @ jnp.stack([new_th, new_q], axis=1)) / rh[1:n, None]

    # winds: C = C_u, K_m, own surface drag implicit in u_0, v_0
    g_w = _flux_matrix(k, params["C_u"], k["Km"], inputs)
    drag = jnp.zeros((n, n), dt_).at[0, 0].set(rh[0] * k["c_s"])
    phi_w = jnp.stack([inputs.u, inputs.v], axis=1)
    new_w = jnp.linalg.solve(jnp.diag(m) - dt * (dv_ @ g_w - drag), m[:, None] * phi_w)
    fw = (g_w @ new_w) / rh[1:n, None]
    tau = -k["c_s"] * new_w[0]                                       # (2,) kinematic

    def assemble(surface, interior):
        return jnp.concatenate([jnp.reshape(surface, (1,)).astype(dt_), interior, zero])

    wtheta = assemble(inputs.w_theta_s, fs[:, 0])
    wqt = assemble(inputs.w_qt_s, fs[:, 1])
    uw = assemble(tau[0], fw[:, 0])
    vw = assemble(tau[1], fw[:, 1])
    return (wtheta, wqt), (uw, vw), aux


def diagnostics(params, inputs):
    """Scoring diagnostics, no feedback, explicit at phi^n, interior interfaces (N-1,):
    w'w'_NL = (M/rho) w_p and w'w'theta'_NL = w_p * w'theta_l'_NL, w_p averaged to the
    interface; plus the regime discriminant chi (N,). TKE is not diagnosed (fix 4)."""
    k = kernel(params, inputs)
    n = inputs.theta_l.shape[0]
    rh = inputs.rho_half[1:n]
    k_nl = dict(k, Kh=jnp.zeros_like(k["Kh"]))
    wth_nl = (_flux_matrix(k_nl, 0.0, k_nl["Kh"], inputs) @ inputs.theta_l) / rh
    w_p_h = 0.5 * (k["w_p"][1:] + k["w_p"][:-1])
    return dict(w_p=k["w_p"], ww_nl=k["M"] / rh * w_p_h, wwtheta_nl=w_p_h * wth_nl,
                wtheta_nl=wth_nl, chi=k["chi"], Z_b=k["Z_b"], u_star=k["u_star"],
                w_L=k["w_L"])
