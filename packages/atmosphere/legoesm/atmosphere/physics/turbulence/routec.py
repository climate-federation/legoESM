"""Route C: a subgrid inversion reconstructed inside the L32 boundary layer.

Grenier and Bretherton (2001, Mon. Wea. Rev. 129, 357-377,
doi:10.1175/1520-0493(2001)129<0357:AMPPFL>2.0.CO;2; "GB01") give every
equation here; Lock (2001, Mon. Wea. Rev. 129, 1148-1163,
doi:10.1175/1520-0493(2001)129<1148:TNROEI>2.0.CO;2) gives the per-step
diagnosis and the point that the numerical entrainment of subsidence across the
inversion has to be removed.  The method, its departures from the papers and
the tests are written in docs/85 section 13.23 of the project that uses it.

All pure functions on ASCENDING column arrays (index 0 nearest the surface,
CLUBB's orientation), shape ``(ncol, nz)`` for layer values and
``(ncol, nz + 1)`` for interface values:

* :func:`diagnose_candidates`: the base of the "ambiguous" layer (GB01
  appendix C, Lock's choice) and the above-inversion mass fraction ``m`` of
  that layer from the layer mean of theta_vl (GB01 C1 to C7), at every
  admissible base interface, with blend weights.
* :func:`inversion_jump` and :func:`entrainment_velocity`: the jump of any
  conserved scalar (C8) and the Turner-Deardorff entrainment rate with
  evaporative enhancement (GB01 eqs. 12, 13, 18 to 21, B1).
* :func:`routec_constrain_rows`: the entrainment flux imposed as two rows of
  CLUBB's xm/w'x' penta solve (eqs. 22, C12 to C14).

KNOWN DIFFERENCE (RULE 5): every switch the published method takes on traced
state (cloudy-layer test, cloud-top layer choice, T versus T + 1, the D0
branch, the slope pair, Lock's double jump, the saturation adjustment, the
enhancement gate and clip) is a logistic / C1 form whose width is a
CLUBBConfig field; the integer ambiguous layer becomes a weighted blend over
candidate base interfaces.  All widths 0 is the published hard form (tests
V1, V4).  The defaults' per-case gaps are stored in tests/atmosphere/test_routec.py.
GB01's corrected advective faces for the host dycore (section 5) are not
ported: route C runs in single-column CLUBB.

Layer indices for a candidate base interface ``kb``: layer M = kb - 1,
ambiguous layer M+1 = kb, M+2 = kb + 1, M+3 = kb + 2; flux levels M-1/2 = kb
- 1, M+1/2 = kb, M+3/2 = kb + 1.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.smooth import positive_part_c1, sigmoid_switch, smooth_clamp

__physics_contract__ = {
    "summary": (
        "Subgrid inversion inside the ambiguous layer (GB01 appendix C), its "
        "entrainment velocity (GB01 eq. 19) and the entrainment flux imposed "
        "on the two flux levels bounding the ambiguous layer (GB01 eqs. 22, C12-C14)."
    ),
    "inputs": {
        "theta_vl": "K", "theta_l": "K", "r_t": "kg/kg", "q_l": "kg/kg",
        "z": "m", "p_half": "Pa", "e_i": "m^2/s^2",
    },
    "outputs": {
        "m": "1 (above-inversion mass fraction)", "jump": "units of X",
        "w_e": "m/s", "E": "1", "entrainment_flux": "units of X times m/s",
    },
    "sign_convention": (
        "Ascending arrays, z and the mass coordinate pi = p_s - p increase "
        "upward; D_i X = X(above) - X(below); the entrainment flux -w_e D_i X "
        "is positive upward; w_e >= 0."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Grenier & Bretherton (2001), Mon. Wea. Rev. 129, 357-377; "
        "Lock (2001), Mon. Wea. Rev. 129, 1148-1163"
    ),
    "idealized_test": "tests/atmosphere/test_routec.py (known inversion column, V1)",
}

# GB01 constants, with the equation each comes from.
D0 = 0.1           # minimum inversion jump in theta_vl [K] (appendix C, condition 4)
A1L = 0.16          # dry entrainment efficiency, local closure (eq. 21, section 6)
A2L = 15.0          # evaporative enhancement coefficient, local closure (eq. 21, section 7a)
LOCK_FRAC_MAX = 10.0  # 1/m above which double grid-level jumps are used (Lock 2001, appendix, after A2)
ETA_L = 0.085      # asymptotic length scale over PBL depth (eq. 13)
KAPPA_VK = 0.4      # von Karman constant (eq. 12)
P_TOP_PA = 70000.0  # the boundary layer top must lie below 700 hPa (GB01 appendix C)
# Not from the papers: what "cloudy" means for a layer.
QL_MIN = 1.0e-6     # grid-mean liquid [kg/kg]
# theta_vl = theta_l (1 + 0.608 q_t) (GB01 section 3a); 0.608 = 1/epsilon - 1
_EPS_V = 1.0 / constants.epsilon - 1.0
_N_BISECT = 60
_N_SATADJ = 30
_DT_NEWTON_K = 1.0e-3    # finite-difference step for dr_s/dT in the Newton adjustment


class RouteCWidths(NamedTuple):
    """RULE 5 widths of every switch on traced state (0 = the published hard
    form): the cloudy-layer test in liquid [kg/kg] and pressure [Pa], the
    theta_vl inequalities [K] (and disc in K^2 via its square), the mass
    fraction m and the slope-pair choice (dimensionless), Lock's double jump in
    m, and the [0, 1] clip of E."""
    ql: float
    p: float
    jump: float
    m: float
    lock: float
    e: float


def widths_from_config(cfg) -> RouteCWidths:
    return RouteCWidths(cfg.routec_ql_width_kg_kg, cfg.routec_p_top_width_pa,
                        cfg.routec_jump_width_k, cfg.routec_m_width,
                        cfg.routec_lock_m_width, cfg.routec_E_width)


def _sharp(width):
    return math.inf if width == 0 else 1.0 / width


def _step(x, width):
    """Heaviside x >= 0, smoothed by a logistic of the given width (0 = hard)."""
    return sigmoid_switch(x, _sharp(width))


def _gt(x, width):
    """Strict x > 0, smoothed by the same logistic (0 = hard)."""
    return 1.0 - _step(-x, width)


def cloud_top_weights(cloudy):
    """P(layer k is the top of the lowest contiguous cloudy deck), (ncol, nz),
    from per-layer cloudy probabilities (ascending).  One-hot at the deck top
    when ``cloudy`` is 0/1 (the argmax form), zero for a cloud-free column."""
    ncol, nz = cloudy.shape
    clear_below = jnp.concatenate(
        [jnp.ones_like(cloudy[:, :1]), jnp.cumprod(1.0 - cloudy, axis=1)[:, :-1]], axis=1)
    first = cloudy * clear_below                                   # P(lowest cloudy = j0)
    j = jnp.arange(nz)
    later = j[None, :, None] < j[None, None, :]                    # i > j0
    run = jnp.cumprod(jnp.where(later, cloudy[:, None, :], 1.0), axis=2)  # prod_{j0<i<=k} c_i
    run = jnp.where(j[None, :, None] <= j[None, None, :], run, 0.0)
    gap = 1.0 - jnp.concatenate([cloudy[:, 1:], jnp.zeros_like(cloudy[:, :1])], axis=1)
    return jnp.einsum("nj,njk->nk", first, run) * gap


class Candidates(NamedTuple):
    """The GB01 reconstruction at every candidate base interface kb, (ncol, ncand)."""
    kb: jax.Array        # static candidate interfaces, (ncand,)
    weight: jax.Array    # blend weight of each candidate (sums to <= 1 per column)
    m: jax.Array
    u_upper: jax.Array   # weight of the (M+2, M+3) slope pair (GB01 s_1 choice)
    pi_c_a: jax.Array
    pi_c_2: jax.Array
    pi_c_3: jax.Array
    pi_top: jax.Array
    pi_bot: jax.Array
    pi_low: jax.Array
    dpi_a: jax.Array
    jump_thvl: jax.Array


def _reconstruct(thvl, pi_half, kb, w: RouteCWidths):
    """GB01 C1 to C7 with flux level M+1/2 at interface ``kb`` (a static int,
    ascending), every switch smoothed by ``w``.  Returns per-column arrays and
    the soft 'active' value."""
    dpi = pi_half[:, 1:] - pi_half[:, :-1]
    pi_c = 0.5 * (pi_half[:, 1:] + pi_half[:, :-1])
    tM, tA, t2, t3 = (thvl[:, k] for k in (kb - 1, kb, kb + 1, kb + 2))
    pcA, pc2, pc3 = pi_c[:, kb], pi_c[:, kb + 1], pi_c[:, kb + 2]
    pi_top, pi_bot, pi_low = pi_half[:, kb + 1], pi_half[:, kb], pi_half[:, kb - 1]
    dpiA = dpi[:, kb]
    s_lo = _safe_div(t2 - tA, pc2 - pcA)       # sigma^{M+3/2}
    s_hi = _safe_div(t3 - t2, pc3 - pc2)       # sigma^{M+5/2}
    # GB01 s_1 = the more negative slope: upper pair where s_hi <= s_lo
    u = _step(_safe_div(s_lo - s_hi, jnp.abs(s_lo) + jnp.abs(s_hi)), w.m)
    s1 = u * s_hi + (1.0 - u) * s_lo
    th_top = t2 - s1 * (pc2 - pi_top)          # (C1)
    a = th_top - tM
    c = tA - tM
    disc = a * a - 2.0 * s1 * dpiA * c
    den_q = a + _sqrt0(disc)
    m_q = _safe_div(2.0 * c, den_q)            # smaller root of (C4)
    jump_q = a - s1 * m_q * dpiA               # (C5)
    q = _step(disc, w.jump ** 2) * _step(jump_q - D0, w.jump) * _gt(den_q, w.jump)
    m_d0 = _safe_div(2.0 * c, a + D0)          # (C7)
    m = q * m_q + (1.0 - q) * m_d0
    jump = q * jump_q + (1.0 - q) * D0
    act = (_gt(c, w.jump) * _gt(a + D0, w.jump) * _gt(m, w.m)
           * _step(1.0 - m, w.m))
    finite = jnp.isfinite(m) & jnp.isfinite(jump)
    return (jnp.where(finite, act, 0.0), jnp.where(finite, m, 1.0), u, pcA, pc2, pc3,
            pi_top, pi_bot, pi_low, dpiA, jnp.where(finite, jump, D0))


def diagnose_candidates(thvl, ql, p_half, w: RouteCWidths) -> Candidates:
    """Lock's ambiguous layer (the cloud-top layer T of the lowest cloudy deck
    below 700 hPa, or T + 1 when the reconstruction in T fails) as soft weights
    over every admissible base interface kb in [1, nz - 3].

    Candidate kb gets ``a(kb) [P_T(kb) + P_T(kb - 1) (1 - a(kb - 1))]``, with
    ``P_T`` from :func:`cloud_top_weights` and ``a`` the soft 'active' of the
    reconstruction at kb; at zero widths this is exactly the hard choice."""
    ncol, nz = thvl.shape
    pi_half = p_half[:, :1] - p_half
    cloudy = (_step(ql - QL_MIN, w.ql) * _gt(p_half[:, 1:] - P_TOP_PA, w.p))
    P_T = cloud_top_weights(cloudy)
    kbs = list(range(1, max(nz - 2, 1)))
    rec = [_reconstruct(thvl, pi_half, k, w) for k in kbs]
    act = jnp.stack([r[0] for r in rec], axis=1)
    weight = []
    for i, k in enumerate(kbs):
        prev = P_T[:, k - 1] * (1.0 - act[:, i - 1]) if i > 0 else P_T[:, k - 1]
        weight.append(act[:, i] * (P_T[:, k] + prev))
    fields = [jnp.stack([r[j] for r in rec], axis=1) for j in range(1, 11)]
    return Candidates(jnp.asarray(kbs), jnp.stack(weight, axis=1), *fields)


def _cand_take(x, kb, off):
    """x (ncol, nz) at layer kb + off for every candidate -> (ncol, ncand)."""
    return x[:, kb + off]


def extrapolated_top(x, cand: Candidates):
    """X at flux level M+3/2 by downward extrapolation from layer M+2 with the
    slope of X over the slope pair theta_vl chose (GB01 C1, C9), per candidate."""
    xA, x2, x3 = (_cand_take(x, cand.kb, o) for o in (0, 1, 2))
    sx = (cand.u_upper * _safe_div(x3 - x2, cand.pi_c_3 - cand.pi_c_2)
          + (1.0 - cand.u_upper) * _safe_div(x2 - xA, cand.pi_c_2 - cand.pi_c_a))
    return x2 - sx * (cand.pi_c_2 - cand.pi_top)


def inversion_jump(x, cand: Candidates, w: RouteCWidths):
    """D_i X = 2 X_1 - X_M - X^{M+3/2}, X_1 = (X_{M+1} - (1 - m) X_M) / m (GB01
    C8), per candidate; blended with Lock's double grid-level jump X_{M+2} - X_M
    where 1/m exceeds 10 (Lock 2001 after eq. A2) by a logistic in m about 0.1
    of width ``w.lock`` (0 = the hard switch).  Linear in X."""
    xM, xA, x2 = (_cand_take(x, cand.kb, o) for o in (-1, 0, 1))
    m = jnp.maximum(cand.m, 1e-6)              # guard: GB form is weighted ~0 there
    gb = 2.0 * (xA - (1.0 - m) * xM) / m - xM - extrapolated_top(x, cand)
    lam = _gt(1.0 / LOCK_FRAC_MAX - cand.m, w.lock)
    return lam * (x2 - xM) + (1.0 - lam) * gb


def _take(x, k):
    return jnp.take_along_axis(x, k[:, None], axis=1)[:, 0]


def _sqrt0(x):
    """sqrt(max(x, 0)) with a finite gradient at and below zero (where-trick)."""
    pos = x > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)


def _safe_div(a, b):
    ok = jnp.abs(b) > 0.0
    return jnp.where(ok, a / jnp.where(ok, b, 1.0), 0.0)


def theta_vl(thl, rt):
    """GB01 section 3a: theta_vl = theta_l (1 + 0.608 q_t)."""
    return thl * (1.0 + _EPS_V * rt)


def _sat_adjust(thl, rt, p, exner, w: RouteCWidths):
    """T, r_v, r_c of (theta_l, r_t) at p by Newton on T (liquid only), on
    CLUBB's Flatau saturation curve; r_c = C1 positive part of r_t - r_s of
    width ``w.ql`` (0 = max(., 0))."""
    from legoesm.atmosphere.physics.turbulence.clubb import sat_mixrat_liq
    lv_cp = constants.L_v / constants.c_pd
    T_l = thl * exner
    sh = _sharp(w.ql)

    def f(T):
        return T - T_l - lv_cp * positive_part_c1(rt - sat_mixrat_liq(p, T), sh)

    def body(_, T):
        fT = f(T)
        dfdT = (f(T + _DT_NEWTON_K) - fT) / _DT_NEWTON_K
        return T - fT / dfdT
    T = jax.lax.fori_loop(0, _N_SATADJ, body, T_l)
    rc = positive_part_c1(rt - sat_mixrat_liq(p, T), sh)
    return T, rt - rc, rc


def theta_v_of(thl, rt, p, exner, w: RouteCWidths):
    """theta_v = theta (1 + 0.608 r_v - r_c) after the saturation adjustment."""
    T, rv, rc = _sat_adjust(thl, rt, p, exner, w)
    return (T / exner) * (1.0 + _EPS_V * rv - rc)


def evaporative_enhancement(thl_c, rt_c, dthl, drt, p, exner, w: RouteCWidths):
    """E = chi_s (1 - D_i b_s / D_i b) (GB01 eqs. 18, B1) for mixtures of the
    cloudy boundary layer air with the above-inversion air at pressure p.
    chi_s solves r_t(chi) = r_s(theta_l(chi) exner, p) (bisection, then one
    Newton step off its stopped gradient so chi_s carries the implicit-function
    derivative).  The gate 'boundary layer air saturated, above-inversion air
    not, D_i b > 0' is a product of logistics (widths ``w.ql``, ``w.jump``) and
    the [0, 1] clip a smooth clamp (``w.e``); zero widths give the hard form.
    Returns (E, chi_s)."""
    from legoesm.atmosphere.physics.turbulence.clubb import sat_mixrat_liq

    def f(chi):
        return (rt_c + chi * drt) - sat_mixrat_liq(p, (thl_c + chi * dthl) * exner)
    f0, f1 = f(jnp.zeros_like(thl_c)), f(jnp.ones_like(thl_c))

    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0.0
        return jnp.where(pos, mid, lo), jnp.where(pos, hi, mid)
    lo, hi = jax.lax.fori_loop(0, _N_BISECT, body,
                               (jnp.zeros_like(thl_c), jnp.ones_like(thl_c)))
    chi_b = jax.lax.stop_gradient(0.5 * (lo + hi))
    f_b, df_b = jax.jvp(f, (chi_b,), (jnp.ones_like(chi_b),))
    chi_s = chi_b - _safe_div(f_b, df_b)
    gate = _gt(f0, w.ql) * _gt(-f1, w.ql)
    thv_c = theta_v_of(thl_c, rt_c, p, exner, w)
    b1 = theta_v_of(thl_c + dthl, rt_c + drt, p, exner, w) - thv_c
    thl_s, rt_s = thl_c + chi_s * dthl, rt_c + chi_s * drt
    bs = thl_s * (1.0 + _EPS_V * rt_s) - thv_c          # just saturated: r_c = 0, r_v = r_t
    E = gate * _gt(b1, w.jump) * smooth_clamp(chi_s - _safe_div(bs, b1), 0.0, 1.0,
                                                 _sharp(w.e))
    return E, gate * chi_s


def entrainment_velocity(e_i, z_i, jump_thvl, theta_v0, E):
    """GB01 eq. (19) with (20), (21), (12), (13):
    w_e = A e^{3/2} / (l D_i b), A = a1L (1 + a2L E), l = kappa z_i / (1 + kappa z_i / l_inf),
    l_inf = eta_l z_i, D_i b = g D_i theta_vl / theta_v0."""
    l_inf = ETA_L * z_i
    l_i = KAPPA_VK * z_i / (1.0 + KAPPA_VK * z_i / l_inf)
    db = constants.g * jump_thvl / theta_v0
    A = A1L * (1.0 + A2L * E)
    e = jnp.maximum(e_i, 0.0)
    return _safe_div(A * e * _sqrt0(e), l_i * db)


def clubb_routec_terms(thlm, rtm, rcm, em, thv_ds_zm, rho_ds_zt, p_sfc, p_in_Pa_zt,
                       exner_zt, gr, w: RouteCWidths):
    """Everything CLUBB's xm/w'x' solve needs for route C, from the
    start-of-step state on CLUBB's ascending grid (zt: nz layers, zm: nz + 1).

    Layer masses are CLUBB's own, g rho_ds dz.  Every candidate base interface
    kb carries its own entrainment flux -w_e D_i X (eq. 22) and xi (C14); the
    constrained-row operator blends them by the candidate weights.  Returns a
    dict: ``cand`` (Candidates), ``xi``, ``F_rt``, ``F_thl`` (ncol, ncand),
    ``W_zm`` (ncol, nz + 1, the total constraint weight on each flux level, for
    the eddy diffusivity), and weight-blended diagnostics ``w_e``, ``E``, ``m``,
    ``z_i``, ``jump_thvl``, ``jump_thl``, ``jump_rt``, ``e_i``, ``active``."""
    dpi = constants.g * rho_ds_zt * gr.dzt
    pi_half = jnp.concatenate([jnp.zeros_like(dpi[:, :1]), jnp.cumsum(dpi, axis=1)], axis=1)
    p_half = p_sfc[:, None] - pi_half
    cand = diagnose_candidates(theta_vl(thlm, rtm), rcm, p_half, w)
    kb = cand.kb
    zt, zb = gr.zm[:, kb + 1], gr.zm[:, kb]
    z_i = zt - cand.m * (zt - zb)
    e_i = em[:, kb]
    th0 = thv_ds_zm[:, kb]
    dthl = inversion_jump(thlm, cand, w)
    drt = inversion_jump(rtm, cand, w)
    E, _ = evaporative_enhancement(thlm[:, kb - 1], rtm[:, kb - 1], dthl, drt,
                                   p_in_Pa_zt[:, kb - 1], exner_zt[:, kb - 1], w)
    w_e = entrainment_velocity(e_i, z_i, cand.jump_thvl, th0, E)
    pi_i = cand.pi_top - cand.m * cand.dpi_a
    xi = _safe_div(cand.pi_bot - cand.pi_low, pi_i - cand.pi_low)
    nzm = pi_half.shape[1]
    onehot = (jnp.arange(nzm)[None, :] == kb[:, None]).astype(dpi.dtype)  # (ncand, nzm)
    W_zm = cand.weight @ onehot + cand.weight @ jnp.roll(onehot, 1, axis=1)
    wsum = jnp.sum(cand.weight, axis=1)
    blend = lambda x: jnp.sum(cand.weight * x, axis=1)  # noqa: E731
    return dict(cand=cand, xi=xi, F_rt=-w_e * drt, F_thl=-w_e * dthl, W_zm=W_zm,
                w_e=blend(w_e), E=blend(E), m=blend(cand.m), z_i=blend(z_i),
                jump_thvl=blend(cand.jump_thvl), jump_thl=blend(dthl),
                jump_rt=blend(drt), e_i=blend(e_i), active=wsum,
                kb_eff=blend(jnp.broadcast_to(kb, cand.weight.shape).astype(dpi.dtype)),
                xi_eff=blend(xi), F_rt_eff=blend(-w_e * drt), F_thl_eff=blend(-w_e * dthl))


def routec_constrain_rows(lhs, rhs, rc, F):
    """Route C: blend two w'x' rows of the coupled xm/w'x' penta system toward
    GB01's constraints (eqs. 22, C12 to C14), per candidate base interface kb
    with weight w: the row of w'x'[kb + 1] toward w'x' = 0 and the row of
    w'x'[kb] toward w'x'[kb] - (1 - xi) w'x'[kb - 1] = xi F.  A row r becomes
    (1 - W_r) row_r + (constraint rows weighted by their candidates), exact at
    one-hot weights (the hard form).  Interleaving as :func:`xm_wpxp_lhs`
    (w'x'[k] at 2k); band 4 couples row r to unknown r - 2.  xm rows untouched."""
    cand, xi = rc["cand"], rc["xi"]
    ncol, ndim = rhs.shape
    nzm = (ndim + 1) // 2
    onehot = (jnp.arange(nzm)[None, :] == cand.kb[:, None]).astype(rhs.dtype)
    wb = cand.weight @ onehot                                   # bottom-row weight at kb
    wt = cand.weight @ jnp.roll(onehot, 1, axis=1)              # top-row weight at kb + 1
    b4 = (cand.weight * (1.0 - xi)) @ onehot
    rb = (cand.weight * xi * F) @ onehot

    def to_rows(z):
        return jnp.zeros((ncol, ndim), rhs.dtype).at[:, 0::2].set(z)
    Wb, Wt, B4, Rb = to_rows(wb), to_rows(wt), to_rows(b4), to_rows(rb)
    W = Wb + Wt
    lhs = lhs * (1.0 - W)[None]
    lhs = lhs.at[2].add(W)
    lhs = lhs.at[4].add(-B4)
    rhs = rhs * (1.0 - W) + Rb
    return lhs, rhs
