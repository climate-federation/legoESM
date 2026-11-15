"""Round-2 candidate r2_ptt_r (no top-down): inversion-capped, cloud-aware asymmetric
transilient exchange with a lifted-parcel barrier and turnover-time buoyancy memory.

EXPERIMENTAL co-scientist closure (KIND = "both"), refinement of round-1
``r1_codex_pressure_turning_trans.py`` (the parent; its helpers are imported, not
re-derived). REDUCED VARIANT 'r2_ptt_r (no top-down)' (interim PI instruction): the
hypothesis' radiatively driven (Lock et al. 2000) top-down launch from cloud top is
NOT built, and no switched-off parameter stands in for it. Reason: the cloud-top
radiative jump Delta F_rad = -c_pd sum_k w_k m_k Q_rad,k needs the radiative heating
Q_rad (CloudBench SCMForcing.rad_T_tendency), which is not in ColumnInputs; pending a
PI decision on adding it. Only the bottom-up launch exists.

Equations (surface-first; cell i, mass m_i [kg m-2], interface K between cells K-1 and
K; G = mass-weighted interface flux, positive up; G_0 = surface flux, G_n = 0)
---------------------------------------------------------------------------------
Flux form for X in {theta_l (enthalpy form), q_t, u, v}:
    m_i w_i dX_i/dt = w_K G_K - w_{K+1} G_{K+1},  K = i,
    w = Exner (cell) / Exner_half (interface) for theta_l, so c_pd Pi theta_l =
    c_pd T - L_v q_c is the liquid-water enthalpy H and an interface carries the
    endpoint parcels' h_liq at the interface's own Exner (theta_l is conserved by a
    reversible lift); w = 1 for q_t, u, v.
(0) Scales. B_s surface buoyancy flux; w_c^3 = v_f^3 P_d(B_s h/v_f^3); h = smooth
    first bulk-Ri crossing floored sqrt(h_c^2 + h_min^2) (parent);
    V_0^2 = u_*^2 + w_c^2 + v_f^2. Cloud weight sigma_i = logistic((q_t - q_sat)/
    (w_sig q_sat)), q_sat = legoesm.thermo.saturation_mixing_ratio(T, p) at the
    column's actual T, p: a CANDIDATE-INTERNAL DIAGNOSTIC, not a host cloud
    fraction (none smooth exists); interface value = mean of the adjacent cells.
    N_e^2 = (1-sigma) N_u^2 + sigma N_sat^2: N_u^2 the parent's unsaturated formula;
    N_sat^2 = -b_K/dz_K, b_K the buoyancy of cell K-1's parcel lifted reversibly to
    level K (host saturation adjustment; the CIN parcel matrix).
    Inversion cap g_ez(z) = logistic(-(z-h)/(c_ez h)).
    V_loc^2 = g_ez (u_*^2 + w_c^2) + (l_s S)^2 F(Ri) + v_f^2,
    l_s = kappa z/(1 + kappa z/lambda_0), S^2 = |dU/dz|^2,
    Ri = N_e^2/(S^2 + s_f^2), F = exp(-P_d(Ri)/Ri_c).
    chi = [1 - exp(-P_d(N_e^2 h^2/V_0^2))] exp(-P_d(B_mem h/V_0^3)).
(1) Local. K_tgt = kappa V_loc [(1-chi) lam + chi l_st], lam = softmin(z+z0, h)
    = [(z+z0)^-2 + h^-2]^-1/2, l_st = c_st h/sqrt(1 + P_d(N_e^2 h^2/V_loc^2)).
    Pair weights w_ij = m_i m_j [(1-chi) exp(-d^2/(l^2 + dz_i dz_j))
    + chi exp(-d^2/(l_st^2 + dz_i dz_j))], l = c_l softmin(z_m+z0, h);
    r_ij = s(z_m) w_ij with s the interface ratio K_tgt/K_raw interpolated to the
    pair midpoint, K_raw,K = sum_{i<K<=j} w_ij d_ij/rho_K (parent normalisation:
    the conductance rate scale is defined by sum r_ij d_ij = rho_K K_tgt,K exactly
    where s is constant over the crossing pairs).
    G^L_K = sum_{i<K<=j} r_ij (X_i - X_j).
(2) Nonlocal asymmetric (ACM2 topology, bottom-up only). Upward transfer i -> j > i
    M_ij = a_N (w_c^2/V_0^2)(V_0/h) m_i L_i p_ij [kg m-2 s-1],
    L_i = exp(-z_i/(0.1h)) * 0.1h / sum_k dz_k exp(-z_k/(0.1h)),
    p_ij = m_j exp(-d_ij/h) exp(-CIN_ij/V_0^2) / sum_{j'>i} m_j' exp(-d_ij'/h),
    CIN_ij = sum_{i<k<=j} dz_k b_s softplus(-b_ik/b_s), b_ik the buoyancy of cell
    i's parcel lifted reversibly to level k. Return by compensating subsidence,
    one interface at a time, upwind (value of the cell above):
    G^N_K = sum_{i<K<=j} M_ij X_i - U_K X_K,   U_K = sum_{i<K<=j} M_ij.
    Net mass flux through every interface is zero (a uniform X gives G^N = 0).
(3) Momentum: (1)+(2) for U = (u, v), plus on local pairs only the parent's
    work-free rotation G_U,K += sum r_ij b_ij J (U_i - U_j), b_ij = a_P
    cross(V_i, V_j)/sqrt((|V_i|^2+v_f^2)(|V_j|^2+v_f^2)), V = U (ground at rest).
    Surface turning law removed (a_s = 0). Surface stress: on a column with a
    surface law (finite inputs.uw_s/vw_s, CloudBench) the passed stress, applied
    explicitly at the surface, and u_* from it (candidate.surface_stress); the
    implicit drag is dropped there. Otherwise the own law G_U,0 = -rho_0 C_D S_1 U_1
    (implicit in the update), C_D = kappa^2/D_eff^2, D = ln((z_1+z0)/z0) -
    psi_m(z_1/L), D_eff = D_f softplus(D/D_f), Monin-Obukhov fixed point (parent).
(4) Time step. Per field X^{n+1} = expm(dt A) X^n with the affine surface-forcing
    column (augmented (n+1) or (2n+1) matrix [[dt A, dt c], [0, 0]]); one expm for
    theta_l, one for q_t, one for the coupled (u, v) block (rotation). The reported
    flux is recovered from the state change (M_n = 0, M_K = M_{K+1} + m_K w_K
    dX_K/dt; candidate.tendency_to_flux), index 0 set to the prescribed surface
    flux (passed stress on a law column), so the harness's forward step with the
    reported flux reproduces the expm state exactly. For q_t A is Metzler
    (off-diagonals r, M_ij, U_K >= 0), so expm(dt A) >= 0; nothing is clipped.
(5) Energy: KE removed by the momentum fluxes is returned as heat by the harness.
(MZ) Memory: B_mem (aux, interfaces) relaxes to the current modelled buoyancy flux
    B^n over tau_B = h/V_0, exact exponential B_mem' = B^n + (B_mem - B^n)
    exp(-dt/tau_B); B_mem = 0 at cold start. B^n = the parent's dry-air
    linearisation g[(1+eps q_t) w'theta_l' + eps theta_l w'q_t']/theta_v of the
    fluxes this step reports. B_mem enters chi only.

Deviations from the hypothesis text (each a reviewer-flagged defect fix or a voted
detail; votes recorded in the round record)
--------------------------------------------------------------------------------
* No top-down branch (interim PI instruction, see above).
* min(z+z0, h) -> p=2 soft-min (GLM review: hard switch on traced h).
* N_e+^2 = (V_loc^2/h^2) P_d(N_e^2 h^2/V_loc^2) (codex: one delta cannot regularise a
  dimensional N^2 and dimensionless Ri); P_d(Ri) is dimensionless as written.
* Exponential update with an affine surface column (codex: expm(dt A) X omitted the
  prescribed surface forcing). Separate expm per field (H and Q weights differ).
* L_i normalised by sum dz L = 0.1 h; p_ij mass-weighted (parent) - voted.
* Own law uses S_1 (gust floor v_f) instead of |V_1| (smooth at calm), and the
  smooth denominator floor D_eff (hypothesis asks for one; form voted).
* lambda_0 = 30 m, Ri_c = 0.25, s_f, b_s, delta, v_f as the hypothesis' fixed list.
KNOWN DIFFERENCES / unresolved (also in the round record): the neutral surface
layer gives V_loc^2 ~ 2 u_*^2 (g_ez u_*^2 plus (l_s S)^2 = u_*^2), so K ~ sqrt(2)
kappa u_* z, not kappa u_* z; the logistic g_ez tail and v_f leave a small exchange
above h; c_ez -> inf gives g_ez -> 1/2, not the parent (F1 control); B_mem enters
chi but not the inversion cap.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from legoesm.atmosphere.les_suite.candidate import (half_level_average, has_surface_law,
                                                    surface_stress, tendency_to_flux)
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    compute_bulk_richardson,
    first_crossing_height,
)
from legoesm.core import bulk_flux
from legoesm.thermo import saturation_mixing_ratio

from legoesm import constants

_spec = importlib.util.spec_from_file_location(
    "_r2_ptt_r_parent",
    Path(__file__).resolve().parents[1] / "round1" / "r1_codex_pressure_turning_trans.py")
_parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent)
_p_delta = _parent._p_delta                 # P_d(x) = x exp(-delta/x), delta = 0.05
_crossing_parts = _parent._crossing_parts
_crossing_matrix = _parent._crossing_matrix
_crossing_sum = _parent._crossing_sum
_div = _parent._div
_rotation_coeff = _parent._rotation_coeff
_V_FLOOR = _parent._V_FLOOR                 # v_f = 0.01 m/s
_LAUNCH_FRAC = _parent._LAUNCH_FRAC         # launch depth 0.1 h
_N_MO_ITERS = _parent._N_MO_ITERS           # Monin-Obukhov fixed-point sweeps (count)
_PBL = _parent._PBL                         # Ri_crit, sharpness, h_min for h
_EPS_V = _parent._EPS_V                     # R_v/R_d - 1

KIND = "both"
PARAMS = {
    "a_N": (1.0, 0.1, 5.0, "1"),     # nonlocal exchange amplitude
    "a_P": (0.5, -2.0, 2.0, "1"),    # work-free pressure rotation of local pairs
    "c_l": (0.4, 0.15, 0.8, "1"),    # local kernel width l = c_l softmin(z+z0, h)
    "c_st": (0.2, 0.05, 0.5, "1"),   # stable length factor
    "c_ez": (0.1, 0.03, 0.3, "1"),   # inversion-cap width / h
}

# --- fixed, not fitted (hypothesis tier 0 list) ---
_S_FLOOR = 1.0e-3      # [1/s] shear floor s_f in Ri
_LAMBDA_0 = 30.0       # [m] Blackadar asymptotic length
_RI_C = 0.25           # [1] critical Richardson number of F(Ri)
_B_SCALE = 1.0e-4      # [m s-2] b_s, softplus scale of the negative parcel buoyancy
# --- voted numerics (round record) ---
_SIGMA_WIDTH = 0.01    # [1] w_sig: logistic width of the cloud weight, fraction of q_sat
_D_FLOOR = 0.5         # [1] D_f: softplus floor of the drag-law log denominator
_MAX_SQUARINGS = 40    # expm scaling-and-squaring cap (count, not a parameter)
_Z0_MASKED = 1e-3      # [m] placeholder roughness fed to the own-law drag ONLY where
                       # it is masked out (every law column): keeps the
                       # backward pass finite; never reaches an output (voted V17)


def init_aux(inputs):
    """Memory buoyancy flux B_mem [m2 s-3] at the interfaces, (n+1,); zero at start."""
    return jnp.zeros_like(inputs.z_half)


def _softmin(a, b):
    """p=2 soft-min [a^-2 + b^-2]^-1/2 of two positive lengths."""
    return (a ** -2 + b ** -2) ** -0.5


def _surface_drag(inputs, b_s):
    """Own law: rho_0 C_D S_1 [kg m-2 s-1] and u_* [m/s] (parent's Monin-Obukhov fixed
    point with the smooth denominator floor D_eff = D_f softplus(D/D_f))."""
    kappa = constants.kappa_von_karman
    # double-where: on a law column the drag is masked out below; a NaN or zero z0 there
    # would make the gradient 0 * NaN (codex r3); an own-law column keeps z0_m as given
    z1 = inputs.z_full[0]
    z0 = jnp.where(has_surface_law(inputs), _Z0_MASKED, inputs.z0_m)
    s1 = jnp.sqrt(inputs.u[0] ** 2 + inputs.v[0] ** 2 + _V_FLOOR ** 2)
    ln_z = jnp.log((z1 + z0) / z0)

    def c_d_of(den):
        return kappa ** 2 / (_D_FLOOR * jax.nn.softplus(den / _D_FLOOR)) ** 2

    c_d = c_d_of(ln_z)
    for _ in range(_N_MO_ITERS):
        u_star = jnp.sqrt(c_d) * s1
        zeta = -kappa * z1 * b_s / u_star ** 3
        c_d = c_d_of(ln_z - bulk_flux.psi_m(zeta))
    return inputs.rho_half[0] * c_d * s1, jnp.sqrt(c_d) * s1


def _scales(inputs):
    """Surface buoyancy flux, h, velocity scales, surface stress selection."""
    g = constants.g
    theta0 = inputs.temp[0] / inputs.exner[0]
    thv0 = theta0 * (1.0 + _EPS_V * inputs.q_v[0] - inputs.q_c[0])
    b_s = g * (inputs.w_theta_s * (1.0 + _EPS_V * inputs.q_v[0])
               + _EPS_V * theta0 * inputs.w_qt_s) / thv0

    def rev(a):
        return a[::-1][None]
    ri_b, _ = compute_bulk_richardson(rev(inputs.temp), rev(inputs.q_v), rev(inputs.u),
                                      rev(inputs.v), rev(inputs.p_full), rev(inputs.z_full))
    h_c = first_crossing_height(ri_b, rev(inputs.z_full), _PBL.Ri_crit, _PBL.sharpness)[0]
    h = jnp.sqrt(h_c ** 2 + _PBL.h_min ** 2)

    c_s, u_star = _surface_drag(inputs, b_s)
    # law column: passed stress explicit at the surface, u* from it, no implicit drag
    uw_law, vw_law, u_star = surface_stress(inputs, 0.0, 0.0, u_star)
    tau_law = inputs.rho_half[0] * jnp.stack([uw_law, vw_law])
    c_s = jnp.where(has_surface_law(inputs), 0.0, c_s)

    wc3 = _V_FLOOR ** 3 * _p_delta(b_s * h / _V_FLOOR ** 3)
    a3 = (wc3 + _V_FLOOR ** 3) ** (1.0 / 3.0)
    wc2 = wc3 * (a3 + _V_FLOOR) / (a3 ** 2 + a3 * _V_FLOOR + _V_FLOOR ** 2)  # w_c^2 >= 0
    v02 = u_star ** 2 + wc2 + _V_FLOOR ** 2
    return dict(b_s=b_s, h=h, c_s=c_s, tau_law=tau_law, u_star=u_star, wc2=wc2,
                v02=v02, v0=jnp.sqrt(v02))


def _parcels(inputs):
    """Buoyancy b[i, k] [m s-2] of cell i's (theta_l, q_t) parcel lifted reversibly to
    level k, relative to the environment at k (host saturation adjustment)."""
    n = inputs.z_full.shape[0]
    th_p, qv_p, qc_p = saturation_adjust(
        jnp.broadcast_to(inputs.theta_l[:, None], (n, n)),
        jnp.broadcast_to(inputs.q_t[:, None], (n, n)),
        jnp.broadcast_to(inputs.exner[None, :], (n, n)),
        jnp.broadcast_to(inputs.p_full[None, :], (n, n)))
    thv_p = th_p * (1.0 + _EPS_V * qv_p - qc_p)
    thv_e = jnp.diagonal(thv_p)
    return constants.g * (thv_p - thv_e[None, :]) / thv_e[None, :]


def _cloud_weight(inputs):
    """sigma_i: smooth saturation weight from legoesm.thermo q_sat (internal diagnostic)."""
    q_sat = saturation_mixing_ratio(inputs.temp, inputs.p_full)
    return jax.nn.sigmoid((inputs.q_t - q_sat) / (_SIGMA_WIDTH * q_sat))


def _operator(params, inputs, aux):
    """Local conductances r_loc, nonlocal transfers M (both (n, n), upper), scales."""
    g, kappa = constants.g, constants.kappa_von_karman
    zf, zh, m = inputs.z_full, inputs.z_half, inputs.mass
    n = zf.shape[0]
    tri = jnp.triu(jnp.ones((n, n), zf.dtype), 1)
    # mixing-length roughness: NaN (contract: column has none) -> 0 (voted V16 a/a/a;
    # heights here are > 0, so softmin stays finite); own-law drag keeps z0_m
    z0 = jnp.where(jnp.isnan(inputs.z0_m), 0.0, inputs.z0_m)
    sc = _scales(inputs)
    h, v0, v02, wc2, u_star = sc["h"], sc["v0"], sc["v02"], sc["wc2"], sc["u_star"]
    zi = zh[1:n]
    dzf = zf[1:] - zf[:-1]

    # Effective stratification N_e^2 at interior interfaces.
    thvu = inputs.theta_l * (1.0 + _EPS_V * inputs.q_t)
    n2_u = g * (thvu[1:] - thvu[:-1]) / (0.5 * (thvu[1:] + thvu[:-1]) * dzf)
    b_par = _parcels(inputs)
    n2_s = -jnp.diagonal(b_par, offset=1) / dzf           # parcel K-1 lifted to K
    sig_c = _cloud_weight(inputs)
    sig = 0.5 * (sig_c[1:] + sig_c[:-1])
    n2_e = (1.0 - sig) * n2_u + sig * n2_s

    # Local velocity with the inversion cap.
    shear2 = ((inputs.u[1:] - inputs.u[:-1]) ** 2
              + (inputs.v[1:] - inputs.v[:-1]) ** 2) / dzf ** 2
    ri = n2_e / (shear2 + _S_FLOOR ** 2)
    f_ri = jnp.exp(-_p_delta(ri) / _RI_C)
    l_s = kappa * zi / (1.0 + kappa * zi / _LAMBDA_0)
    g_ez = jax.nn.sigmoid(-(zi - h) / (params["c_ez"] * h))
    vloc2 = g_ez * (u_star ** 2 + wc2) + l_s ** 2 * shear2 * f_ri + _V_FLOOR ** 2
    vloc = jnp.sqrt(vloc2)

    # Regime gate (uses V_0 and the memory buoyancy flux) and target diffusivity.
    chi_k = ((1.0 - jnp.exp(-_p_delta(n2_e * h ** 2 / v02)))
             * jnp.exp(-_p_delta(aux[1:n] * h / (v0 * v02))))
    lst_k = params["c_st"] * h / jnp.sqrt(1.0 + _p_delta(n2_e * h ** 2 / vloc2))
    lam_k = (1.0 - chi_k) * _softmin(zi + z0, h) + chi_k * lst_k
    k_tgt = kappa * vloc * lam_k                                        # [m2/s]

    # Local pairs, normalised to K_tgt (parent's moment normalisation).
    d = jnp.abs(zf[None, :] - zf[:, None])
    z_m = 0.5 * (zf[None, :] + zf[:, None])
    dz = zh[1:] - zh[:-1]
    floor2 = dz[:, None] * dz[None, :]
    chi = jnp.interp(z_m, zi, chi_k)
    l_st = jnp.interp(z_m, zi, lst_k)
    l_loc = params["c_l"] * _softmin(z_m + z0, h)
    w = tri * m[:, None] * m[None, :] * (
        (1.0 - chi) * jnp.exp(-d ** 2 / (l_loc ** 2 + floor2))
        + chi * jnp.exp(-d ** 2 / (l_st ** 2 + floor2)))
    k_raw = _crossing_sum(w * d)[1:n] / inputs.rho_half[1:n]
    r_loc = jnp.interp(z_m, zi, k_tgt / k_raw) * w

    # Nonlocal asymmetric transfers through the lifted-parcel barrier.
    cin = jnp.cumsum(tri * dz[None, :] * _B_SCALE * jax.nn.softplus(-b_par / _B_SCALE),
                     axis=1)
    w_d = jnp.exp(-d / h)
    norm = jnp.sum(tri * m[None, :] * w_d, axis=1)
    norm = jnp.where(norm > 0.0, norm, 1.0)               # top cell launches nothing
    e_l = jnp.exp(-zf / (_LAUNCH_FRAC * h))
    launch = e_l * (_LAUNCH_FRAC * h) / jnp.sum(dz * e_l)
    big_m = (params["a_N"] * (wc2 / v02) * (v0 / h) * tri
             * (m * launch / norm)[:, None] * m[None, :] * w_d * jnp.exp(-cin / v02))
    return dict(sc=sc, r_loc=r_loc, big_m=big_m, chi_k=chi_k, k_tgt=k_tgt, n2_e=n2_e,
                g_ez=g_ez, vloc=vloc)


def _nonlocal_matrix(big_m):
    """B (n+1, n): (B X)_K = sum_{i<K<=j} M_ij X_i - U_K X_K (upwind return)."""
    n = big_m.shape[0]
    s, _, below = _crossing_parts(big_m)
    up = _crossing_sum(big_m)                                  # U_K, zero at 0 and n
    ret = jnp.zeros((n + 1, n), big_m.dtype).at[jnp.arange(n), jnp.arange(n)].set(up[:n])
    return jnp.where(below, s, 0.0) - ret


def _expm_step(gmat, g0, x, m_w, w_half, dt):
    """X_new from m_w dX/dt = -div(w_half (gmat X + g0 e_0)), via the augmented expm."""
    n = x.shape[0]
    a = -_div(w_half[:, None] * gmat) / m_w[:, None]
    c = jnp.zeros((n,), x.dtype).at[0].set(w_half[0] * g0 / m_w[0])
    aug = jnp.zeros((n + 1, n + 1), x.dtype).at[:n, :n].set(dt * a).at[:n, n].set(dt * c)
    z = expm(aug, max_squarings=_MAX_SQUARINGS) @ jnp.concatenate([x, jnp.ones((1,), x.dtype)])
    return z[:n]


def _recover_flux(x_new, x, m_w, w_half, rho_half, dt):
    """Kinematic interface flux whose divergence is the state change (top flux 0)."""
    g_w = tendency_to_flux(m_w * (x_new - x) / dt, jnp.ones_like(rho_half),
                           jnp.ones_like(m_w))                  # w_half G, (n+1,)
    return g_w / (w_half * rho_half)


def step(params, inputs, aux, b_extra=None):
    """Internal expm states and diagnostics: dict with the new theta_l, q_t, u, v.
    ``b_extra`` (n+1, n): an additional mass-flux operator added to the exchange of
    every field (r2_ptt_full's top-down branch); None = this candidate."""
    o = _operator(params, inputs, aux)
    m, dt, rho_h = inputs.mass, inputs.dt, inputs.rho_half
    n = m.shape[0]
    b_loc = _crossing_matrix(o["r_loc"])
    b_nl = _nonlocal_matrix(o["big_m"])
    b_sc = b_loc + b_nl
    if b_extra is not None:
        b_sc = b_sc + b_extra
    ones_h = jnp.ones_like(rho_h)
    th_new = _expm_step(b_sc, rho_h[0] * inputs.w_theta_s, inputs.theta_l,
                        m * inputs.exner, inputs.exner_half, dt)
    qt_new = _expm_step(b_sc, rho_h[0] * inputs.w_qt_s, inputs.q_t, m, ones_h, dt)

    # Momentum: coupled (u, v) with the local-pair rotation and the surface stress.
    b_rot = _crossing_matrix(o["r_loc"] * _rotation_coeff(params["a_P"], inputs.u,
                                                           inputs.v))
    e00 = jnp.zeros((n + 1, n), m.dtype).at[0, 0].set(1.0)
    drag = b_sc - o["sc"]["c_s"] * e00
    gmat = jnp.concatenate([jnp.concatenate([drag, -b_rot], 1),
                            jnp.concatenate([b_rot, drag], 1)], 0)       # (2n+2, 2n)
    a = -jnp.concatenate([_div(gmat[:n + 1]), _div(gmat[n + 1:])], 0) / jnp.concatenate([m, m])[:, None]
    tau = o["sc"]["tau_law"]
    c = jnp.zeros((2 * n,), m.dtype).at[0].set(tau[0] / m[0]).at[n].set(tau[1] / m[0])
    aug = (jnp.zeros((2 * n + 1, 2 * n + 1), m.dtype)
           .at[:2 * n, :2 * n].set(dt * a).at[:2 * n, 2 * n].set(dt * c))
    uv = jnp.concatenate([inputs.u, inputs.v, jnp.ones((1,), m.dtype)])
    uv_new = expm(aug, max_squarings=_MAX_SQUARINGS) @ uv
    return dict(theta_l=th_new, q_t=qt_new, u=uv_new[:n], v=uv_new[n:2 * n], op=o,
                b_nl=b_nl, b_loc=b_loc, b_rot=b_rot)


def fluxes(params, inputs, aux, b_extra=None):
    """((w'theta_l', w'q_t'), (u'w', v'w'), B_mem_new): kinematic, (n+1,), +up."""
    st = step(params, inputs, aux, b_extra)
    m, dt, rho_h = inputs.mass, inputs.dt, inputs.rho_half
    ones_h = jnp.ones_like(rho_h)
    dtype = rho_h.dtype
    wth = _recover_flux(st["theta_l"], inputs.theta_l, m * inputs.exner,
                        inputs.exner_half, rho_h, dt)
    wqt = _recover_flux(st["q_t"], inputs.q_t, m, ones_h, rho_h, dt)
    uw = _recover_flux(st["u"], inputs.u, m, ones_h, rho_h, dt)
    vw = _recover_flux(st["v"], inputs.v, m, ones_h, rho_h, dt)
    wth = wth.at[0].set(jnp.asarray(inputs.w_theta_s, dtype))
    wqt = wqt.at[0].set(jnp.asarray(inputs.w_qt_s, dtype))
    has = has_surface_law(inputs)
    uw = uw.at[0].set(jnp.where(has, jnp.where(has, inputs.uw_s, 0.0), uw[0]))
    vw = vw.at[0].set(jnp.where(has, jnp.where(has, inputs.vw_s, 0.0), vw[0]))

    # Memory: relax B_mem toward this step's modelled buoyancy flux over h/V_0.
    th_h = half_level_average(inputs.theta_l, inputs.theta_l[0])
    qt_h = half_level_average(inputs.q_t, inputs.q_t[0])
    thv_h = th_h * (1.0 + _EPS_V * qt_h)
    b_now = constants.g * ((1.0 + _EPS_V * qt_h) * wth + _EPS_V * th_h * wqt) / thv_h
    sc = st["op"]["sc"]
    decay = jnp.exp(-dt * sc["v0"] / sc["h"])
    b_mem = b_now + (aux - b_now) * decay
    return (wth, wqt), (uw, vw), b_mem


def implied_surface_fluxes(params, inputs, aux, b_extra=None):
    """Index-0 flux recovered from the expm state change BEFORE it is overwritten by
    the prescribed/passed value (conservation diagnostic: equals it to round-off)."""
    st = step(params, inputs, aux, b_extra)
    m, dt, rho_h = inputs.mass, inputs.dt, inputs.rho_half
    ones_h = jnp.ones_like(rho_h)
    return dict(
        wtheta=_recover_flux(st["theta_l"], inputs.theta_l, m * inputs.exner,
                             inputs.exner_half, rho_h, dt)[0],
        wqt=_recover_flux(st["q_t"], inputs.q_t, m, ones_h, rho_h, dt)[0],
        uw=_recover_flux(st["u"], inputs.u, m, ones_h, rho_h, dt)[0],
        vw=_recover_flux(st["v"], inputs.v, m, ones_h, rho_h, dt)[0])
