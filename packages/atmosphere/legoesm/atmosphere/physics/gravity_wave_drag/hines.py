"""Hines (1997) Doppler-spread non-orographic gravity-wave drag.

Port of LMDZ6 trunk r5891 ``libf/phylmd/hines_gwd.f90`` (McLandress 1995 code,
modified by McFarlane & Manzini for MAECHAM; same lineage as ECHAM/ICON
``mo_gw_hines``), oracle-matched by
``tests/atmosphere/hydrostatic/unit/test_hines_lmdz_oracle.py``.  Routines
ported: ``hines_gwd`` (driver: sigma-coordinate N, density, altitude),
``hines_setup`` (naz = 8, slope = 1), ``hines_extro0``, ``hines_wind``,
``hines_wavnum``, ``hines_sigma``, ``hines_intgrl`` (slope = 1 branch),
``hines_flux``, ``hines_exp`` and ``vert_smooth``.  Not ported, because the
LMDZ driver never reaches them: ``hines_heat`` (``iheatcal = 0``), the
precipitation-enhanced rms wind (``lozpr = .FALSE.``) and the convective
``sigsqmcw`` source (identically zero, so ``f2mod = 1`` and ``sigalpmc = 0``).

Departures from the oracle (each voted, spread_f4/VOTES.md V43-V51):

* launch level: LMDZ launches at the fixed model level ``klev-2`` (third from
  the bottom).  Here the launch level is the level nearest ``launch_p`` in
  each column (default 700 hPa, never the lowest level); the oracle test
  passes ``k_launch`` to select LMDZ's level.
* heating: LMDZ returns ``d_t_hin = 0``.  Here the mean-flow kinetic energy
  removed is returned as frictional heating ``dT_dt = -(u du + v dv)/c_pd``
  (``eps_gwd`` its column integral); the momentum tendencies are unchanged.
* differentiability: the oracle's hard switches (N floor, ``m_trial``
  clamp, turbulent-vs-molecular cutoff, ``umin`` wind floor, series-vs-log
  integral) are smooth at ``config.smooth_sharpness`` (on dimensionless
  ratios, 0.01 relative width at the default 100) and exactly the oracle's
  hard forms at ``smooth_sharpness = inf``.  The 105 km exponential-decay
  mask is a level selection by altitude and stays hard.

Arrays are top-down (index 0 = model top), the LMDZ internal order.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
from legoesm.core.smooth import sigmoid_switch, smooth_max

__physics_contract__ = {
    "summary": (
        "Hines (1997) Doppler-spread non-orographic gravity-wave drag, LMDZ "
        "r5891 hines_gwd.f90 port: an 8-azimuth spectrum launched with rms "
        "wind total_rms_wind at the level nearest launch_p is Doppler-shifted "
        "by the wind relative to the launch level; its cutoff vertical "
        "wavenumber m_alpha shrinks upward with the wave-induced rms wind "
        "(turbulent and molecular cutoffs), and the drag is the vertical "
        "divergence of the resulting momentum flux."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "rho": "kg/m^3", "lat": "rad", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "eps_gwd": "W/m^2",
    },
    "sign_convention": (
        "z up, arrays top-down. flux_u[k] = rho_launch * sum_n ak_n k* m_n[k] "
        "* e_n (eastward momentum flux carried upward); drag[k] = (F[k] - "
        "F[k-1]) / (rho_k (alt_{k-1} - alt_k)) with F[-1] = 0, so the "
        "launched flux is removed one level BELOW the launch level and "
        "deposited aloft (column momentum sum exactly zero, as in LMDZ, "
        "unless the 105 km decay acts). "
        "dT_dt = -(u du_dt + v dv_dt)/c_pd, eps_gwd its column integral."
    ),
    # total energy: the mean-flow KE change is returned exactly as heat.
    # Momentum telescopes to zero only below the 105 km decay, so not claimed.
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "Hines (1997) JASTP 59, 371-386 and 387-400; McLandress (1998); LMDZ6 "
        "r5891 libf/phylmd/hines_gwd.f90 (oracle)"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_hines_lmdz_oracle.py: drag "
        "== LMDZ hines_gwd on 14 columns (smooth_sharpness=inf); column "
        "momentum sum rho*du*dz == 0; finite nonzero gradients at the default "
        "sharpness; c_pd*sum(rho*dT_dt*dz) == eps_gwd (test_hines_gwd_faithful.py)"
    ),
}

# --- hines_setup (LMDZ r5891 hines_gwd.f90:1640-1767) ---
_NAZ = 8                      # azimuths
_SLOPE = 1.0                  # m-spectrum slope (only the slope = 1 branch is ported)
_ALT_CUTOFF_M = 105.0e3       # icutoff = 1: drag decays above this altitude
_SMCO = 2.0                   # vert_smooth centre weight
_NSMAX = 5                    # vert_smooth passes
# --- hines_gwd driver constants ---
_VISC_MOL = 1.5e-5            # molecular viscosity [m2/s] (hines_gwd.f90:325); > visc_min
_DTHETA_FLOOR = 5.0           # dTheta/dsigma <= -5/sigma (hines_gwd.f90:293)
_N_SMOOTH = 5.0               # bvfreq smoothing ratio 5*log(sigma_l/sigma_l-1)
# --- hines_wind / hines_intgrl / hines_exp / hines_flux literals, copied verbatim ---
_COS45 = 0.7071068
_UMIN = 0.001                 # [m/s]
_THIRD = 0.33333333           # molecular-cutoff exponent, as written in hines_wavnum
_Q_MIN = 1.0
_QM_MIN = 0.01
_EXP_HSCALE_M = 5.0e3


def _hines_sigma(sigsqh):
    """hines_sigma, naz = 8: (ncol, 8) -> (sigma_alpha (ncol, 8), sigma_t (ncol,))."""
    s = [sigsqh[:, n] for n in range(_NAZ)]
    sum_odd = (s[0] + s[2] + s[4] + s[6]) / 2.0
    sum_even = (s[1] + s[3] + s[5] + s[7]) / 2.0
    a1 = jnp.sqrt(s[0] + s[4] + sum_even)
    a2 = jnp.sqrt(s[1] + s[5] + sum_odd)
    a3 = jnp.sqrt(s[2] + s[6] + sum_even)
    a4 = jnp.sqrt(s[3] + s[7] + sum_odd)
    sigma_alpha = jnp.stack([a1, a2, a3, a4, a1, a2, a3, a4], axis=1)
    tot = s[0]
    for n in range(1, _NAZ):
        tot = tot + s[n]
    return sigma_alpha, jnp.sqrt(tot)


def _hines_wind(uhs, vhs, sharp, hard):
    """hines_wind, naz = 8: (ncol, nlev) winds -> v_alpha (ncol, nlev, 8)."""
    def floor(w):
        # |w| < umin -> umin (sign NOT kept, as in LMDZ)
        if hard:
            return jnp.where(jnp.abs(w) < _UMIN, _UMIN, w)
        keep = sigmoid_switch(jnp.abs(w) / _UMIN - 1.0, sharp)
        return keep * w + (1.0 - keep) * _UMIN
    u = floor(uhs)
    v = floor(vhs)
    a2 = _COS45 * (v + u)
    a4 = _COS45 * (v - u)
    return jnp.stack([u, a2, v, a4, -u, -a2, -v, -a4], axis=-1)


def _hines_intgrl(v_alpha, m_alpha, bvfb, sharp, hard):
    """hines_intgrl, slope = 1: (ncol, 8) -> i_alpha (ncol, 8)."""
    q = v_alpha / bvfb[:, None]
    qm = q * m_alpha
    q_ok = q != 0.0
    q_safe = jnp.where(q_ok, q, 1.0)
    series = jnp.where(
        q_ok,
        (qm ** 2 / 2.0 + qm ** 3 / 3.0 + qm ** 4 / 4.0 + qm ** 5 / 5.0) / q_safe ** 2,  # coeff-ok: Taylor series of -(log(1-x)+x), hines_intgrl
        m_alpha ** 2 / 2.0)
    if hard:
        use_log = (jnp.abs(q) >= _Q_MIN) & (jnp.abs(qm) >= _QM_MIN)
        w_log = use_log.astype(q.dtype)
    else:
        w_log = (sigmoid_switch(jnp.abs(q) / _Q_MIN - 1.0, sharp)
                 * sigmoid_switch(jnp.abs(qm) / _QM_MIN - 1.0, sharp))
        use_log = w_log > 0.0
    # qm < 1 on every reachable state (m <= N/(f1 sigma + n/m + v)); the where
    # keeps an unselected lane's log finite for the VJP.
    one_m_qm = jnp.where(use_log, 1.0 - qm, 0.5)
    logf = -(jnp.log(one_m_qm) + qm) / q_safe ** 2
    if hard:
        return jnp.where(use_log, logf, series)
    return w_log * logf + (1.0 - w_log) * series


def _vert_smooth(data, k_idx, k_launch):
    """vert_smooth (coeff _SMCO, _NSMAX passes) on levels 0..k_launch, ends fixed.

    data (ncol, nlev, ...); k_launch (ncol,)."""
    sum_wts = _SMCO + 2.0
    inner = (k_idx[None, :] >= 1) & (k_idx[None, :] <= k_launch[:, None] - 1)
    inner = inner.reshape(inner.shape + (1,) * (data.ndim - 2))
    for _ in range(_NSMAX):
        up = jnp.concatenate([data[:, :1], data[:, :-1]], axis=1)      # work(l-1)
        dn = jnp.concatenate([data[:, 1:], data[:, -1:]], axis=1)      # work(l+1)
        data = jnp.where(inner, (dn + _SMCO * data + up) / sum_wts, data)
    return data


def hines_gwd(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    rho: jax.Array,
    lat: jax.Array,
    dt: float,
    config: HinesConfig,
    k_launch: jax.Array | None = None,
) -> GWDOutput:
    """Hines Doppler-spread GWD tendencies (LMDZ r5891 ``hines_gwd``).

    Parameters
    ----------
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
        Standard GWD backend signature, column arrays (ncol, nlev) top-down,
        ``p_half`` (ncol, nlev+1), ``lat`` in radians.  ``z_full``/``rho``
        enter only the frictional-heating diagnostics: like LMDZ the scheme
        builds its own density and altitude from ``p`` and ``T``.
    k_launch : optional (ncol,) int
        Launch level index (0-based, top-down).  ``None`` (production) = the
        level nearest ``config.launch_p``; the oracle test passes LMDZ's
        ``nlev - 3``.  A column whose launch level is its lowest level gets no
        source.

    Returns
    -------
    GWDOutput
    """
    ncol, nlev = u.shape
    dtype = u.dtype
    # a host may hand fp64 hybrid pressures to fp32 columns; the scans below
    # need one dtype in their carries
    p_full, p_half, z_half, rho, lat = (
        jnp.asarray(a, dtype) for a in (p_full, p_half, z_half, rho, lat))
    sharp = config.smooth_sharpness
    hard = isinstance(sharp, (int, float)) and sharp == math.inf
    rd, rg = constants.R_d, constants.g
    rgocp = rd / constants.c_pd
    k_idx = jnp.arange(nlev)

    # --- driver: sigma coordinate, N, density, altitude (hines_gwd.f90:270-345)
    ps = p_half[:, -1:]
    sg = p_full / ps
    dsg = (p_half[:, 1:] - p_half[:, :-1]) / ps
    shx = sg ** rgocp
    thx = T / shx
    dttdsf = (thx[:, 1:] - thx[:, :-1]) / (sg[:, 1:] - sg[:, :-1])
    cap = _DTHETA_FLOOR / sg[:, 1:]
    if hard:
        dttdsf = jnp.minimum(dttdsf, -cap)
    else:
        dttdsf = -cap * smooth_max(-dttdsf / cap, 1.0, sharp)
    bv = jnp.sqrt(-dttdsf * sg[:, 1:] * (sg[:, 1:] ** rgocp) / rd) * rg / T[:, 1:]
    ratio = _N_SMOOTH * jnp.log(sg[:, 1:] / sg[:, :-1])

    def _bv_smooth(prev, x):
        b, r = x
        b = (prev + r * b) / (1.0 + r)
        return b, b

    _, bv_s = jax.lax.scan(_bv_smooth, bv[:, 0], (bv.T, ratio.T))
    bvfreq = jnp.concatenate([bv[:, :1], bv_s.T], axis=1)          # level 0 = raw level 1

    hscal = rd * T / rg
    density = sg * ps / (rg * hscal)
    dalt = hscal * dsg / sg
    alt = jnp.concatenate(
        [jnp.cumsum(dalt[:, -2::-1], axis=1)[:, ::-1], jnp.zeros((ncol, 1), dtype)],
        axis=1)                                                     # alt[nlev-1] = 0

    # --- launch level, rms wind, kstar (hines_setup)
    if k_launch is None:
        k_launch = jnp.argmin(jnp.abs(p_full - config.launch_p), axis=1)
    k_launch = jnp.asarray(k_launch, jnp.int32)
    active = k_launch < nlev - 1                                    # no source at the lowest level
    k_launch = jnp.where(active, k_launch, nlev - 2)               # any valid index; masked below
    take = lambda a: jnp.take_along_axis(a, k_launch[:, None], axis=1)[:, 0]
    bvfb = take(bvfreq)
    densb = take(density)
    kstar = config.kstar_min / (jnp.cos(lat) + config.kstar_min / config.kstar_max)

    upper = k_idx[None, :] <= k_launch[:, None]                     # levels 0..k_launch
    uhs = jnp.where(upper, u - take(u)[:, None], 0.0)
    vhs = jnp.where(upper, v - take(v)[:, None], 0.0)
    v_alpha = _hines_wind(uhs, vhs, sharp, hard)                    # (ncol, nlev, 8)

    # --- hines_wavnum: launch level
    sigsqh_b = jnp.full((ncol, _NAZ), config.total_rms_wind ** 2 / _NAZ, dtype)
    sig_a_b, sig_t_b = _hines_sigma(sigsqh_b)
    m_b = bvfb[:, None] / (config.f1 * sig_a_b + config.f2 * sig_t_b[:, None])
    ak = sigsqh_b / (m_b ** (_SLOPE + 1.0) / (_SLOPE + 1.0))

    def _up(carry, x):
        mmin, sig_a, sig_t = carry
        k, n_l, rho_l, va = x                                       # level k, scanned bottom-up
        # turbulent vs molecular cutoff (f2mod = 1: no convective source)
        nm_turb = config.f2 * sig_t
        m_mol = (n_l * kstar / _VISC_MOL) ** _THIRD / config.f3
        nm_mol = n_l / m_mol
        if hard:
            n_over_m = jnp.where(n_l / nm_turb < m_mol, nm_turb, nm_mol)
        else:
            n_over_m = nm_mol * smooth_max(nm_turb / nm_mol, 1.0, sharp)
        denom = config.f1 * sig_a + n_over_m[:, None] + va
        if hard:
            m_trial = bvfb[:, None] / denom
            m = jnp.where((m_trial <= 0.0) | (m_trial > mmin), mmin, m_trial)
        else:
            c = bvfb[:, None] / mmin
            m = bvfb[:, None] / (c * smooth_max(denom / c, 1.0, sharp))
        i_a = _hines_intgrl(va, m, bvfb, sharp, hard)
        sigfac = densb / rho_l * n_l / bvfb
        sig_a_new, sig_t_new = _hines_sigma(sigfac[:, None] * ak * i_a)
        up = (k < k_launch)                                         # above the launch level
        at = (k == k_launch)
        m_out = jnp.where(up[:, None], m, jnp.where(at[:, None], m_b, 0.0))
        new = (jnp.where(up[:, None], m, mmin),
               jnp.where(up[:, None], sig_a_new, sig_a),
               jnp.where(up, sig_t_new, sig_t))
        return new, m_out

    xs = (k_idx[::-1], bvfreq.T[::-1], density.T[::-1],
          jnp.moveaxis(v_alpha, 1, 0)[::-1])
    _, m_rev = jax.lax.scan(_up, (m_b, sig_a_b, sig_t_b), xs)
    m_alpha = jnp.moveaxis(m_rev[::-1], 0, 1)                       # (ncol, nlev, 8)
    m_alpha = _vert_smooth(m_alpha, k_idx, k_launch)

    # --- hines_flux, slope = 1, naz = 8 (k_alpha = kstar for every azimuth)
    pr = ak[:, None, :] * kstar[:, None, None] * m_alpha
    flux_u = pr[..., 0] - pr[..., 4] + _COS45 * (pr[..., 1] - pr[..., 3] - pr[..., 5] + pr[..., 7])
    flux_v = pr[..., 2] - pr[..., 6] + _COS45 * (pr[..., 1] + pr[..., 3] - pr[..., 5] - pr[..., 7])
    flux_u = flux_u * densb[:, None] / _SLOPE
    flux_v = flux_v * densb[:, None] / _SLOPE

    def _drag(f):
        d_top = f[:, :1] / (density[:, :1] * (alt[:, :1] - alt[:, 1:2]))
        d_in = -(f[:, :-1] - f[:, 1:]) / (density[:, 1:] * (alt[:, :-1] - alt[:, 1:]))
        d = jnp.concatenate([d_top, d_in], axis=1)
        # hines_exp: above alt_cutoff (levels 0..k_launch) decay from the
        # lowest such level's value with a 5 km scale height
        hi = (alt >= _ALT_CUTOFF_M) & upper
        k_lo = jnp.max(jnp.where(hi, k_idx[None, :], -1), axis=1)
        d_zmax = jnp.take_along_axis(d, jnp.maximum(k_lo, 0)[:, None], axis=1)
        d = jnp.where(hi, d_zmax * jnp.exp((_ALT_CUTOFF_M - alt) / _EXP_HSCALE_M), d)
        return jnp.where(active[:, None], d, 0.0)

    du_dt = _drag(flux_u)
    dv_dt = _drag(flux_v)

    # frictional heating: the mean-flow KE removed (V47; LMDZ returns 0)
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
