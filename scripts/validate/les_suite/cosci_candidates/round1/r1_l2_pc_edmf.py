"""PC-EDMF (round 1, r1_l2): one pressure-calibrated plume carries theta_l, q_t, u, v.

Co-scientist round-1 candidate, ``KIND = "both"``. EXPERIMENTAL: never enters the
turbulence dispatch; the harness loads it by path.

Hypothesis (summary)
--------------------
Total kinematic flux at interior interfaces i = 1..n-1 (surface-first, +up)::

    F_phi,i = -K_h,i dphi/dz|_i + M_i (phi_u,i - phi_env,i)       phi in {theta_l, q_t}
    F_U,i   = -K_m,i dU/dz|_i   + M_i (U_u,i   - U_env,i)         U in {u, v}
    M_i = Mhat_i G_i T_i,  Mhat = a w_u/(1-a)  (exact top-hat, zero net mass)
    G = w_u^2/(w_u^2 + W_T^2),  T = 1 - exp(-(z_top - z_i)/L_TOP)  (T_n = 0 exactly)

F(0): theta_l, q_t = the prescribed LES surface fluxes (bit-exact); u, v = the
candidate's own MOST stress. F(n) = 0 for all four. Eddy diffusion from a prognostic
TKE e (K_m = c_K l sqrt(e), Pr = Pr_0 (1 + c_Pr Ri_+), c_eps = c_K^3 derived);
one steady entraining-detraining plume (lax.scan upward) launched from TKE with a
buoyancy-weighted excess; Gregory-Kershaw pressure term (C_u, mu_u) on the plume
momentum; plume buoyancy coefficient a_b = 1 - C3; saturation adjustment for the
plume theta_v.

Contract
--------
``fluxes(params, inputs, aux) -> ((wtheta_l, wq_t), (u'w', v'w'), aux_new)``, each
(n+1,). ``aux = (e, h_p)``: TKE at full levels (n,) [m^2/s^2] and the active plume
depth (scalar) [m] lagged one step (used only by the gustiness term). The plume is
diagnosed fresh every step. ``inputs.uw_s/vw_s`` are never read.

Conservation: theta_l, q_t, u, v move only through the returned interface fluxes; the
harness converts them with -d(rho F)/dm, so column totals change by exactly the
surface flux (index 0) and nothing leaves the top (index n = 0). The implicit solve
below only decides WHICH flux is reported; whatever it is, the harness applies it in
flux form. The stress is parallel to the lowest-level wind U_1 (as the LES wall
model). TKE is a non-ledger intensity (its dissipation is not heated here; the
harness returns the mean-KE removed by the momentum fluxes as heat).

Deviations from the hypothesis text (each also in the review record)
--------------------------------------------------------------------
D1 (review fix 1) Mass flux is IMPLICIT UPWIND: the environment value in the MF term
   is the full level ABOVE the interface (compensating subsidence is downward) at the
   NEW time level, phi_u from phi^n. The matrix is an M-matrix for any dt (column sums
   = layer mass > 0, off-diagonals <= 0) and the scheme is von Neumann stable for any
   dt (uniform M: |1 - c + c e^-ik| <= |1 + c - c e^ik|). It is NOT positivity-
   preserving: the plume outflow dt rho M phi_u stays explicit, so q_t can go negative
   once dt rho M/m exceeds ~1 per layer (layer thinner than M dt; ~12 m at M = 1.2 m/s,
   dt = 10 s). No limiter is applied (would be a choice). Text had explicit MF with the
   interface-mean phibar_i.
D2 (fix 2) Launch w_u,1 = sqrt((2/3)(e_1 + E_S)) sqrt((1-a_0)/a_0); e_1 = TKE at
   interface 1 (mean of the two lowest levels). Cold start: e = l_0^2 sp(S^2 - N^2/Pr_0)
   / c_K^2 with l_0 = kappa z and DEFAULT c_K, Pr_0 (init_aux receives no params),
   the lowest level's shear from the neutral log law U_1/(z_1 ln(z_1/z0)); h_p = 0.
D3 (fix 3) MOST: damped fixed point (relaxation MOST_RELAX, N_MOST_ITERS = 8 instead
   of 3 undamped, because damping halves the per-iteration convergence); zeta bounded
   smoothly, zeta_b = ZETA_BOUND tanh(zeta/ZETA_BOUND); denominator
   ln(z1/z0) - c_psi psi_m floored softly at DENOM_FLOOR. psi_m = Businger-Dyer/Paulson
   (gamma 16) + Dyer (beta 5) from legoesm.core.bulk_flux.psi_m on the smooth split
   zeta_-/zeta_+ (reused, not re-derived).
D4 (fix 4) Lambda uses f(zeta) = zeta^2 sigma(-zeta/s)/sqrt(zeta^2+s^2): exactly 0 at
   zeta = 0, >= 0, ~ -zeta for zeta << -s, -> 0 for stable. (Subtracting s ln2 from the
   softplus would make Lambda negative for stable zeta and unbounded below at small
   zeta_Lambda.)
D5 (fix 5) delta = c_delta sp_{1/L_D}(-B_u/(w_u^2 + w_s^2)); the "+ eps" floor is
   dropped so m grows where eps > delta. The softplus floor c_delta ln2/L_D (~0.007/m
   at c_delta = 1) still competes with eps (~0.0025/m) -- see open risks.
D6 (fix 6) w_g = (g/theta_v0 sp_{F_BS_SCALE}(F_bs) h_p + W_G_FLOOR^3)^(1/3).
D7 Nominal plume mass flux carried as log m; a = A_CAP sigmoid(log m - log(A_CAP w)),
   identical to a = m/(w + m/A_CAP) but cannot overflow (dead plumes have eps ~ 1/m).
   m_1 = a_0 w_1/(1 - a_0/A_CAP) so a_1 = a_0 exactly.
D8 Saturation adjustment via the existing les_suite saturation_adjust (40 damped
   fixed-point iterations, canonical thermo.saturation_mixing_ratio) instead of 3
   Newton steps (reuse rule).
D9 Plume buoyancy in layer k uses the plume state ENTERING the layer at the layer's
   pressure vs the grid-mean theta_v of level k (text left the evaluation point open).
D10 sqrt(e) -> sqrt(e + E_S) in K, l, l_inf and the dissipation rate (AD safety at
   e = 0); N_+ gets an N2_AD_FLOOR inside the sqrt; w_u = sqrt(w_u^2 + W_AD_FLOOR^2).
D11 TKE: buoyancy coefficients are the UNSATURATED ones (A_theta = 1 + C_V q_t,
   A_q = C_V theta) with grid-mean theta, q_t; surface-interface shear production is
   the MOST neutral value u*^3/(kappa z_1). Production at full levels = mean of the
   two bounding interfaces. Patankar split P_+ = sp_{P_S}(P), P_- = sp_{P_S}(-P).
D12 TKE diffusion in mass coordinates, K_e = K_m, zero flux at both ends.
D13 h_p = G_1 z_half[1] + sum_{i=1}^{n-1} G_i dz_i: the layer below the launch
   interface is weighted by the launch gate (text: sum dz G_i, bottom layer unstated).
D14 Lambda is evaluated on the BOUNDED zeta (ZETA_BOUND tanh); identical for |zeta| << 5.
D15 Launch uses the regularised w_u,1 (sqrt(w^2 + W_AD_FLOOR^2)) in both Mhat_1 and a_1,
   so the launch-EXCESS part of the interface-1 flux is exactly Lambda c_MF F_s G_1 T_1
   (the exchange part Mhat_1 G_1 T_1 (phibar_h,1 - phi_new,1) comes on top).

Known breaks (stated, not hidden)
---------------------------------
* NEUTRAL LOG LAW IS BROKEN. Lambda = 0 removes only the launch EXCESS. The plume is
  still launched from TKE (w_u,1 ~ 3 sigma_w at a_0 = 0.1) and develops an excess by
  entrainment lag (scalars) and by -(1 - C_u) dU/dz (momentum), so the mass flux adds
  an effective diffusivity ~ a tau_eps w_u^2/(1-a) to K_m in a neutral surface layer.
  The hypothesis' "no mass-flux flux at neutral" claim does not hold for its own
  equations.
* Deardorff similarity broken by tau_eps (dimensional) -- falsifier (5) tests it.
* No plume precipitation; environment unsaturated in N^2 and in the TKE buoyancy
  coefficients; no wave momentum or environment (non-plume) pressure transport --
  nothing reaches the top, so absorb_top is unused; stress has no cross-wind part.
* L_TOP = 200 m taper is a numerics choice inherited from the text, NOT retuned here
  (review: cuts entrainment-zone MF flux 8-14 % where the column top is low).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.atmosphere.les_suite.candidate import half_level_average
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.core.bulk_flux import psi_m
from legoesm.timestepping.tridiagonal import thomas_solve

from legoesm import constants

KIND = "both"

PARAMS = {
    "c_K": (0.4, 0.25, 0.6, "1"),
    "alpha_l": (0.2, 0.05, 0.6, "1"),
    "c_N": (0.75, 0.3, 1.5, "1"),
    "Pr_0": (0.7, 0.5, 1.0, "1"),
    "c_Pr": (1.5, 0.0, 5.0, "1"),
    "a_0": (0.1, 0.03, 0.4, "1"),
    "c_MF": (0.5, 0.1, 1.0, "1"),
    "c_MFu": (0.5, 0.0, 1.0, "1"),
    "zeta_Lambda": (0.1, 0.01, 1.0, "1"),
    "tau_eps": (400.0, 100.0, 1500.0, "s"),
    "c_delta": (1.0, 0.0, 5.0, "1"),
    "a_b": (0.5, 0.3, 0.9, "1"),
    "b_d": (1.0, 0.3, 2.0, "1"),
    "C_u": (0.6, 0.3, 0.9, "1"),
    "mu_u": (1.0, 0.0, 4.0, "1"),
    "c_psi": (1.0, 0.0, 1.5, "1"),
    "beta_g": (0.5, 0.0, 2.0, "1"),
}

# --- named scales of the hypothesis (fixed, not trained) ---------------------------
W_T = 0.1             # m/s, plume termination-gate velocity
L_TOP = 200.0         # m, top taper e-folding depth (T_n = 0 exactly)
W_S = 0.01            # m/s, plume velocity regularisation (eps, delta, w^2 softplus)
U_S = 0.01            # m/s, u* regularisation in zeta
ZETA_SPLIT = 0.01     # 1, smooth split of zeta into zeta_- / zeta_+ for psi_m
R_S = 0.05            # 1, softplus scale of Ri_+
S_0 = 1e-4            # 1/s, shear regularisation in Ri
N2_S = 1e-6           # 1/s^2, softplus scale of N_+^2
L_D = 100.0           # m, detrainment softplus length (scale 1/L_D)
A_CAP = 0.5           # 1, updraft-fraction cap (top-hat skewness >= 0)
# --- named scales added by the implementation (review fixes / AD safety) ----------
E_S = 1e-4            # m^2/s^2, TKE regularisation (launch, sqrt, Patankar)
P_S = 1e-8            # m^2/s^3, Patankar positive/negative split scale
F_BS_SCALE = 1e-4     # K m/s, softplus scale of the surface buoyancy flux in w_g
W_G_FLOOR = 0.01      # m/s, gustiness floor inside the cube root
U_EFF_FLOOR = 1e-3    # m/s, effective-wind floor (AD safety at calm, no gust)
ZETA_BOUND = 5.0      # 1, smooth |zeta| bound (tanh) in the MOST iteration
DENOM_FLOOR = 1.0     # 1, floor of ln(z1/z0) - c_psi psi_m
DENOM_SCALE = 0.1     # 1, softplus scale of that floor
MOST_RELAX = 0.5      # 1, under-relaxation of the u* fixed point
N_MOST_ITERS = 8      # fixed iteration count (static)
ZETA_GATE_SCALE = 0.01  # 1, smooth positive part of -zeta in Lambda
W_AD_FLOOR = 1e-4     # m/s, floor inside sqrt(w_u^2) (dead plume, G ~ 1e-6)
N2_AD_FLOOR = 1e-14   # 1/s^2, floor inside sqrt(N_+^2)
S2_INIT_SCALE = 1e-6  # 1/s^2, softplus scale of the cold-start production

C_V = 1.0 / constants.epsilon - 1.0   # virtual-temperature coefficient R_v/R_d - 1
KAPPA = constants.kappa_von_karman


def _sp(x, s):
    """Softplus with scale ``s`` (same units as ``x``): s ln(1 + exp(x/s)) >= 0."""
    return s * jnp.logaddexp(0.0, x / s)


def _theta_v(inputs):
    theta = inputs.temp / inputs.exner
    return theta * (1.0 + C_V * inputs.q_v - inputs.q_c)


def _buoy_coeffs(inputs):
    """Unsaturated d theta_v / d theta_l and d theta_v / d q_t at full levels."""
    theta = inputs.temp / inputs.exner
    return 1.0 + C_V * inputs.q_t, C_V * theta


def _grad(f, dzh):
    return (f[1:] - f[:-1]) / dzh


def _avg(f):
    return 0.5 * (f[1:] + f[:-1])


def _length(z, n2, e, l_inf, c_n):
    """1/l = 1/(kappa z) + 1/l_inf + c_N N_+/sqrt(e + E_S)."""
    n_plus = jnp.sqrt(_sp(n2, N2_S) + N2_AD_FLOOR)
    return 1.0 / (1.0 / (KAPPA * z) + 1.0 / l_inf + c_n * n_plus / jnp.sqrt(e + E_S))


def _psi_m_smooth(zeta):
    """psi_Paulson(zeta_-) - 5 zeta_+ on the smooth split (bulk_flux.psi_m, dyer1974)."""
    z_neg = -_sp(-zeta, ZETA_SPLIT)
    z_pos = _sp(zeta, ZETA_SPLIT)
    return psi_m(z_neg) + psi_m(z_pos)


def _most(p, x, h_p, thv, a_th, a_q):
    """Candidate surface stress (damped MOST fixed point). Returns u*, stress, zeta, F_bs."""
    z1 = x.z_full[0]
    bscale = constants.g / thv[0]
    f_bs = a_th[0] * x.w_theta_s + a_q[0] * x.w_qt_s          # K m/s, +up
    w_g = jnp.cbrt(bscale * _sp(f_bs, F_BS_SCALE) * h_p + W_G_FLOOR ** 3)
    u_eff = jnp.sqrt(x.u[0] ** 2 + x.v[0] ** 2 + (p["beta_g"] * w_g) ** 2
                     + U_EFF_FLOOR ** 2)
    ln_z = jnp.log(z1 / x.z0_m)

    def bounded(zeta):
        return ZETA_BOUND * jnp.tanh(zeta / ZETA_BOUND)

    def denom(zeta):
        d = ln_z - p["c_psi"] * _psi_m_smooth(bounded(zeta))
        return DENOM_FLOOR + _sp(d - DENOM_FLOOR, DENOM_SCALE)

    def zeta_of(ust):
        return -KAPPA * z1 * bscale * f_bs / (ust ** 3 + U_S ** 3)

    ust = KAPPA * u_eff / denom(jnp.zeros_like(u_eff))
    for _ in range(N_MOST_ITERS):   # static count: no data-dependent control flow
        ust = (1.0 - MOST_RELAX) * ust + MOST_RELAX * KAPPA * u_eff / denom(zeta_of(ust))
    zeta = bounded(zeta_of(ust))
    tau_u = -ust ** 2 * x.u[0] / u_eff
    tau_v = -ust ** 2 * x.v[0] / u_eff
    return ust, tau_u, tau_v, zeta, f_bs


def _plume(p, x, e, thv, lam, tau_u, tau_v):
    """Steady plume, launched at interface 1, scanned up to interface n.

    Returns per interface 1..n: excess-carrying plume values (theta_l, q_t, U_u, V_u),
    regularised w_u, a, Mhat, and the MF mass flux M = Mhat G T (M at n is exactly 0).
    """
    a0 = p["a_0"]
    th_h = half_level_average(x.theta_l, x.theta_l[0])
    qt_h = half_level_average(x.q_t, x.q_t[0])
    u_h = half_level_average(x.u, x.u[0])
    v_h = half_level_average(x.v, x.v[0])

    e1 = 0.5 * (e[0] + e[1])
    w1 = jnp.sqrt((2.0 / 3.0) * (e1 + E_S)) * jnp.sqrt((1.0 - a0) / a0)
    w1r = jnp.sqrt(w1 ** 2 + W_AD_FLOOR ** 2)
    mhat1 = a0 * w1r / (1.0 - a0)
    logm1 = jnp.log(a0 * w1r / (1.0 - a0 / A_CAP))
    carry0 = (w1 ** 2, logm1,
              th_h[1] + lam * p["c_MF"] * x.w_theta_s / mhat1,
              qt_h[1] + lam * p["c_MF"] * x.w_qt_s / mhat1,
              lam * p["c_MFu"] * tau_u / mhat1,           # D_u = U_u - Ubar
              lam * p["c_MFu"] * tau_v / mhat1)

    dz = x.z_half[2:] - x.z_half[1:-1]                    # layers 1..n-1
    xs = (dz, x.theta_l[1:], x.q_t[1:], thv[1:], x.p_full[1:], x.exner[1:],
          (u_h[2:] - u_h[1:-1]) / dz, (v_h[2:] - v_h[1:-1]) / dz)

    def layer(carry, xk):
        w2, logm, th_u, qt_u, d_u, d_v = carry
        dzk, th_e, qt_e, thv_e, pk, exk, gam_u, gam_v = xk
        theta_u, qv_u, qc_u = saturation_adjust(th_u, qt_u, exk, pk)
        b_u = constants.g * (theta_u * (1.0 + C_V * qv_u - qc_u) - thv_e) / thv_e
        wr2 = w2 + W_S ** 2
        eps = 1.0 / (p["tau_eps"] * jnp.sqrt(wr2))
        delta = p["c_delta"] * _sp(-b_u / wr2, 1.0 / L_D)
        decay = jnp.exp(-eps * dzk)
        th_u = th_e + (th_u - th_e) * decay
        qt_u = qt_e + (qt_u - qt_e) * decay
        lam_u = (1.0 + p["mu_u"]) * eps
        el = jnp.exp(-lam_u * dzk)
        frac = -jnp.expm1(-lam_u * dzk) / lam_u
        d_u = d_u * el - (1.0 - p["C_u"]) * gam_u * frac
        d_v = d_v * el - (1.0 - p["C_u"]) * gam_v * frac
        k_w = 2.0 * p["b_d"] * eps
        w2 = (w2 * jnp.exp(-k_w * dzk)
              + 2.0 * p["a_b"] * b_u * (-jnp.expm1(-k_w * dzk)) / k_w)
        w2 = _sp(w2, W_S ** 2)                             # >= 0 by construction
        logm = logm + (eps - delta) * dzk
        new = (w2, logm, th_u, qt_u, d_u, d_v)
        return new, new

    _, outs = jax.lax.scan(layer, carry0, xs)
    st = [jnp.concatenate([jnp.reshape(c0, (1,)), o]) for c0, o in zip(carry0, outs)]
    w2, logm, th_u, qt_u, d_u, d_v = st                   # interfaces 1..n
    w_u = jnp.sqrt(w2 + W_AD_FLOOR ** 2)
    a = A_CAP * jax.nn.sigmoid(logm - jnp.log(A_CAP * w_u))
    mhat = a * w_u / (1.0 - a)
    gate = w_u ** 2 / (w_u ** 2 + W_T ** 2)
    taper = 1.0 - jnp.exp(-(x.z_half[-1] - x.z_half[1:]) / L_TOP)
    m_flux = mhat * gate * taper
    vals = jnp.stack([th_u, qt_u, u_h[1:] + d_u, v_h[1:] + d_v])
    h_p = gate[0] * x.z_half[1] + jnp.sum(gate[:-1] * dz)
    return vals, w_u, a, mhat, m_flux, h_p


def _implicit(x, fields, k_int, m_int, vals_int, f_sfc, dzh):
    """Backward-Euler ED + implicit upwind MF, mass coordinates, batched over fields.

    fields (F, n); k_int (F, n-1) K at interior interfaces; m_int (n-1,) MF mass flux;
    vals_int (F, n-1) plume values; f_sfc (F,) surface kinematic flux.
    Returns the new fields and the (F, n+1) kinematic interface fluxes applied.
    """
    dt = x.dt
    m = x.mass
    rho_i = x.rho_half[1:-1]
    zf = jnp.zeros(fields.shape[:-1] + (1,), fields.dtype)
    dk = jnp.concatenate([zf, dt * rho_i * k_int / dzh, zf], axis=-1)        # (F, n+1)
    mm = jnp.concatenate([zf, jnp.broadcast_to(dt * rho_i * m_int, k_int.shape), zf],
                         axis=-1)
    mphi = jnp.concatenate([zf, dt * rho_i * m_int * vals_int, zf], axis=-1)
    a = -dk[:, :-1]
    b = m + dk[:, :-1] + dk[:, 1:] + mm[:, :-1]
    c = -(dk[:, 1:] + mm[:, 1:])
    rhs = m * fields + mphi[:, :-1] - mphi[:, 1:]
    rhs = rhs.at[:, 0].add(dt * x.rho_half[0] * f_sfc)
    new = thomas_solve(a / m, b / m, c / m, rhs / m)
    f_int = -k_int * (new[:, 1:] - new[:, :-1]) / dzh + m_int * (vals_int - new[:, 1:])
    zero = jnp.zeros_like(f_int[:, :1])
    return new, jnp.concatenate([f_sfc[:, None], f_int, zero], axis=-1)


def _column(p, x, aux):
    e, h_p = aux
    zf, zh = x.z_full, x.z_half
    dzh = zf[1:] - zf[:-1]
    thv = _theta_v(x)
    a_th, a_q = _buoy_coeffs(x)

    # --- eddy diffusivities from e^n -------------------------------------------------
    sq_e = jnp.sqrt(e + E_S)
    l_inf = p["alpha_l"] * jnp.sum(zf * sq_e * x.mass) / jnp.sum(sq_e * x.mass)
    n2_int = constants.g / _avg(thv) * _grad(thv, dzh)
    s2_int = _grad(x.u, dzh) ** 2 + _grad(x.v, dzh) ** 2
    e_int = _avg(e)
    l_int = _length(zh[1:-1], n2_int, e_int, l_inf, p["c_N"])
    k_m = p["c_K"] * l_int * jnp.sqrt(e_int + E_S)
    ri_p = _sp(n2_int / (s2_int + S_0 ** 2), R_S)
    k_h = k_m / (p["Pr_0"] * (1.0 + p["c_Pr"] * ri_p))
    n2_full = jnp.concatenate([n2_int[:1], _avg(n2_int), n2_int[-1:]])
    l_full = _length(zf, n2_full, e, l_inf, p["c_N"])

    # --- surface stress and launch weight -------------------------------------------
    ust, tau_u, tau_v, zeta, f_bs = _most(p, x, h_p, thv, a_th, a_q)
    f_pos = zeta ** 2 * jax.nn.sigmoid(-zeta / ZETA_GATE_SCALE) / jnp.sqrt(
        zeta ** 2 + ZETA_GATE_SCALE ** 2)
    lam = f_pos / (f_pos + p["zeta_Lambda"])

    # --- plume -----------------------------------------------------------------------
    vals, w_u, a, mhat, m_flux, h_p_new = _plume(p, x, e, thv, lam, tau_u, tau_v)

    # --- implicit ED + implicit upwind MF for theta_l, q_t, u, v --------------------
    fields = jnp.stack([x.theta_l, x.q_t, x.u, x.v])
    k_all = jnp.stack([k_h, k_h, k_m, k_m])
    f_sfc = jnp.stack([jnp.asarray(x.w_theta_s, fields.dtype),
                       jnp.asarray(x.w_qt_s, fields.dtype), tau_u, tau_v])
    new, flux = _implicit(x, fields, k_all, m_flux[:-1], vals[:, :-1], f_sfc, dzh)
    flux = flux.at[:, 0].set(f_sfc).at[:, -1].set(0.0)

    # --- TKE (Patankar-positive implicit) -------------------------------------------
    thv_h = half_level_average(thv, thv[0])
    ath_h = half_level_average(a_th, a_th[0])
    aq_h = half_level_average(a_q, a_q[0])
    p_b = constants.g / thv_h * (ath_h * flux[0] + aq_h * flux[1])        # (n+1,)
    p_s_int = -(flux[2, 1:-1] * _grad(new[2], dzh) + flux[3, 1:-1] * _grad(new[3], dzh))
    p_s = jnp.concatenate([jnp.reshape(ust ** 3 / (KAPPA * zf[0]), (1,)), p_s_int,
                           jnp.zeros(1, p_s_int.dtype)])
    prod = _avg(p_s + p_b)                                                  # (n,)
    p_plus = _sp(prod, P_S)
    p_minus = _sp(-prod, P_S)
    c_eps = p["c_K"] ** 3
    m = x.mass
    extra = x.dt * (p_minus / (e + E_S) + c_eps * jnp.sqrt(e + E_S) / l_full)
    zero1 = jnp.zeros(1, e.dtype)
    dk = jnp.concatenate([zero1, x.dt * x.rho_half[1:-1] * k_m / dzh, zero1])
    e_new = thomas_solve(-dk[:-1] / m, (m + dk[:-1] + dk[1:]) / m + extra,
                         -dk[1:] / m, e + x.dt * p_plus)

    # MF-only fluxes (for the falsifier diagnostics)
    f_mf = m_flux[:-1] * (vals[:, :-1] - new[:, 1:])
    return {"flux": flux, "e": e_new, "h_p": jnp.asarray(h_p_new, e.dtype),
            "a": a, "w_u": w_u, "mhat": mhat, "m_flux": m_flux,
            "f_mf": f_mf, "ustar": ust, "zeta": zeta, "Lambda": lam, "F_bs": f_bs,
            "K_m": k_m, "K_h": k_h}


def init_aux(inputs):
    """Cold start: local shear/stratification equilibrium TKE with DEFAULT c_K, Pr_0
    (l_0 = kappa z; lowest level from the neutral log-law shear), and h_p = 0."""
    x = inputs
    zf = x.z_full
    dzh = zf[1:] - zf[:-1]
    thv = _theta_v(x)
    n2_int = constants.g / _avg(thv) * _grad(thv, dzh)
    s2_int = _grad(x.u, dzh) ** 2 + _grad(x.v, dzh) ** 2
    s2_sfc = (x.u[0] ** 2 + x.v[0] ** 2) / (zf[0] * jnp.log(zf[0] / x.z0_m)) ** 2
    # full level 0 sits at z_1: the log-law shear there is exactly u*/(kappa z_1)
    s2_full = jnp.concatenate([jnp.reshape(s2_sfc, (1,)), _avg(s2_int), s2_int[-1:]])
    n2_full = jnp.concatenate([n2_int[:1], _avg(n2_int), n2_int[-1:]])
    c_k0, pr0 = PARAMS["c_K"][0], PARAMS["Pr_0"][0]
    e0 = (KAPPA * zf) ** 2 * _sp(s2_full - n2_full / pr0, S2_INIT_SCALE) / c_k0 ** 2
    return (e0, jnp.zeros((), e0.dtype))


def fluxes(params, inputs, aux):
    out = _column(params, inputs, aux)
    f = out["flux"]
    return (f[0], f[1]), (f[2], f[3]), (out["e"], out["h_p"])


def diagnostics(params, inputs, aux):
    """Falsifier quantities (not used by the harness): plume a, w_u, Mhat, M, MF
    fluxes and the predicted MF third moment <w'w'phi'> = (1-2a) w_u/(1-a) F_MF at
    interfaces 1..n-1, plus u*, zeta, Lambda, F_bs, K_m, K_h."""
    out = _column(params, inputs, aux)
    a, w_u = out["a"][:-1], out["w_u"][:-1]
    out["wwphi_mf"] = (1.0 - 2.0 * a) * w_u / (1.0 - a) * out["f_mf"]
    return out
