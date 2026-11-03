"""IFS 49r1 non-orographic gravity-wave drag: Warner-McIntyre-Scinocca (WMS).

ORACLE.  OpenIFS 49r1 (8f6f722) ``arpifs/phys_ec/gwdrag_wms.F90`` with the
settings of ``phys_ec/sugwwms.F90`` (one launch level at 450 hPa standard
pressure, launch flux 3.75e-3 Pa per azimuth, m* = 2000 m, C* = 1, NSLOPE = 1,
2p = 2, Gaussian equatorial reduction GGAUSSB = -0.25 over 20 deg, recompute
every GTPHYGWWMS) and ``GSSEC = 1e-12`` (sugwd.F90:161).  Scinocca (2003)
hydrostatic spectral scheme on 4 azimuths x 32 phase speeds: critical-level
filtering of the launched spectrum, Warner-McIntyre saturation, and a
momentum-deposition limiter.  :func:`wms_drag` ports GWDRAG_WMS in the
oracle's operation order and is checked against the running oracle in
``tests/atmosphere/test_ifs_wms_ozone_oracle.py``.

The launch level is static: :func:`wms_launch_level` mirrors SUGWWMS on the
model's standard full-level pressures (hybrid coordinate at p_s = 101325 Pa).

RULE 5 (vote V7, oracle_audit/ifs_port/wms_ozone/VOTES.md), all widths in
:class:`IFSWMSConfig` (0 = GWDRAG_WMS's hard form):
  N^2 floor        linear smooth_max, width ``n2_floor_rel_width * GSSEC``;
  critical speed   running smooth_max, width ``speed_width_m_s``;
  critical level   sigmoid_switch of (c - c_min), width ``speed_width_m_s``;
  saturation       act*smooth_min(F, Fs) + (1-act)*F, width rel*(|F|+|Fs|);
  deposition cap   smooth_min/max, width ``deposit_rel_width`` * (sum of |operands|);
  mean deposited   (sum c dF + eps ZCRT_below)/(sum dF + eps), eps = 1e-12 * launch flux.
The grid-size factor stays hard (static geometry).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.smooth import sigmoid_switch, smooth_max, smooth_min

__physics_contract__ = {
    "summary": (
        "IFS 49r1 GWDRAG_WMS: Warner-McIntyre-Scinocca non-orographic "
        "gravity-wave drag (4 azimuths x 32 phase speeds, launched at 450 hPa)."
    ),
    "inputs": {
        "T": "K", "u": "m/s", "v": "m/s", "p_full": "Pa", "p_half": "Pa (index 0 = top)",
        "geo_full": "m2/s2 (full-level geopotential)", "lat": "rad",
        "pgaw": "1 (cell area / Earth surface area)",
    },
    "outputs": {
        "dudt": "m/s2", "dvdt": "m/s2",
        "flux_u": "Pa (vertical flux of zonal momentum on half levels)",
        "flux_v": "Pa",
    },
    "sign_convention": (
        "flux positive = upward flux of eastward (northward) momentum; "
        "dudt = g dflux/dp between the half levels bounding each full level."
    ),
    "conserves": ["momentum"],
    "differentiable": True,
    "reference": (
        "Scinocca 2003 (JAS 60); Orr et al. 2010 (J. Climate 23); "
        "OpenIFS 49r1 phys_ec/gwdrag_wms.F90, sugwwms.F90."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ifs_wms_ozone_oracle.py: matches the running "
        "49r1 GWDRAG_WMS at rel 1e-12 (fp64) on the tier-2 audit columns; "
        "column momentum: sum(dudt dp/g) = -launch-level net flux."
    ),
}

__param_spec__ = {
    "IFSWMSConfig": {
        "scheme_key": "atm.gwd.IFSWMSConfig",
        "excluded": {
            "n2_floor_rel_width": "numerics: RULE 5 smoothing width (vote V7), 0 = oracle hard form",
            "speed_width_m_s": "numerics: RULE 5 smoothing width (vote V7), 0 = oracle hard form",
            "saturation_rel_width": "numerics: RULE 5 smoothing width (vote V7), 0 = oracle hard form",
            "deposit_rel_width": "numerics: RULE 5 smoothing width (vote V7), 0 = oracle hard form",
        },
        "params": {
            "flux_launch_pa": {
                "units": "Pa", "bounds": (1.0e-3, 1.0e-2), "tunable_tier": 2, "transform": "sigmoid",
                "category": "gravity-wave source", "reference": "sugwwms.F90:110 GFLUXLAUN", "shape": None,
            },
            "c_star": {
                "units": "1", "bounds": (0.5, 2.0), "tunable_tier": 2, "transform": "sigmoid",
                "category": "gravity-wave saturation", "reference": "sugwwms.F90:122 GCSTAR (McLandress and Scinocca 2005)", "shape": None,
            },
            "m_star_length_m": {
                "units": "m", "bounds": (1000.0, 4000.0), "tunable_tier": 2, "transform": "sigmoid",
                "category": "gravity-wave source", "reference": "sugwwms.F90:123 GMSTAR_L (Scinocca 2003)", "shape": None,
            },
            "gauss_halfwidth_deg": {
                "units": "deg", "bounds": (10.0, 40.0), "tunable_tier": 2, "transform": "sigmoid",
                "category": "gravity-wave source", "reference": "sugwwms.F90:130 GGAUSSA", "shape": None,
            },
            "gauss_amplitude": {
                "units": "1", "bounds": (-0.5, 0.5), "tunable_tier": 2, "transform": "sigmoid",
                "category": "gravity-wave source", "reference": "sugwwms.F90:131 GGAUSSB", "shape": None,
            },
        },
    },
}

# --- GWDRAG_WMS fixed numerics (gwdrag_wms.F90) ---
_N_AZI = 4                 # IAZIDIM
_N_C = 32                  # INCDIM
_C_MIN = 0.50              # ZCIMIN [m/s]
_C_MAX = 100.0             # ZCIMAX [m/s]
_GAM = 0.25                # ZGAM (LGACALC = F)
_RAD_TO_DEG = 57.29577951  # ZRADTODEG (literal in the oracle)
_DX_MIN = 1.0e2            # [m]
_DXA = 1.0 / 29.0e3        # [1/m]
_DXB = 1.0 / 3.5e3         # [1/m]
_GSSEC = 1.0e-12           # [s^-2] sugwd.F90:161
_LAUNCH_P = 45000.0        # [Pa] sugwwms.F90:113 ZLAUNCHP(1)
_RHOUR = 3600.0            # [s] YOMCST RHOUR (exact conversion)
_CRT_EPS_REL = 1.0e-12     # vote V7: ZCRT fallback eps / launch flux


class IFSWMSConfig(NamedTuple):
    # --- launch spectrum (sugwwms.F90) ---
    flux_launch_pa: float = 3.75e-3
    c_star: float = 1.0
    m_star_length_m: float = 2000.0
    gauss_halfwidth_deg: float = 20.0
    gauss_amplitude: float = -0.25
    # --- numerics (RULE 5) ---
    n2_floor_rel_width: float = 0.01
    speed_width_m_s: float = 0.01
    saturation_rel_width: float = 0.01
    deposit_rel_width: float = 0.01


def wms_launch_level(stpre) -> int:
    """SUGWWMS ``NLAUNCHL(1)`` (sugwwms.F90:186-192), 1-based Fortran index:
    the smallest level >= 2 whose standard pressure exceeds 450 hPa."""
    stpre = np.asarray(stpre)
    nlev = stpre.shape[0]
    lev = nlev - 1
    for jk in range(nlev, 1, -1):
        if stpre[jk - 1] > _LAUNCH_P:
            lev = jk
    return lev


def wms_call_interval(dt: float) -> float:
    """SUGWWMS ``GTPHYGWWMS`` [s] (sugwwms.F90:144-156, NCONF = 1)."""
    gt = _RHOUR
    if dt >= _RHOUR:
        gt = 2.0 * dt
    return float(int(gt / dt)) * dt


def _spectrum():
    """Phase-speed grid ZCI/ZDCI (Scinocca 2003 eqs. 28-30), oracle op order, libm."""
    xmax, xmin = 1.0 / _C_MIN, 1.0 / _C_MAX
    xran = xmax - xmin
    dx = xran / float(_N_C - 1)
    x1 = xran / (math.exp(xran / _GAM) - 1.0)
    x2 = xmin - x1
    ci, dci = [], []
    for inc in range(1, _N_C + 1):
        tx = float(inc - 1) * dx + xmin
        x = x1 * math.exp((tx - xmin) / _GAM) + x2
        c = 1.0 / x
        ci.append(c)
        dci.append(c * c * (x1 / _GAM) * math.exp((tx - xmin) / _GAM) * dx)
    return np.array(ci), np.array(dci)


def _azimuths():
    ang = 2.0 * math.pi / _N_AZI
    cos = [math.cos(float(i) * ang) for i in range(_N_AZI)]
    sin = [math.sin(float(i) * ang) for i in range(_N_AZI)]
    norm = 0.0
    for c in cos:
        norm += abs(c)
    return np.array(cos), np.array(sin), 2.0 * 1.0 / norm


def _sharp(w):
    return math.inf if (isinstance(w, (int, float)) and w == 0) else 1.0 / w


def _rel_smooth(op, hard, a, b, rel):
    """``op`` (smooth_min/max) of relative width ``rel`` in units of
    ``|a| + |b|``, on the normalised values so no exponent overflows; where
    both operands vanish the hard ``hard`` (where-before-divide, AD-safe).
    ``rel = 0`` is the hard form."""
    if rel == 0.0:
        return hard(a, b)
    raw = jnp.abs(a) + jnp.abs(b)
    ok = raw > _safe_floor(a)           # scale**2 stays normal in the reverse pass
    scale = jnp.where(ok, raw, 1.0)
    return jnp.where(ok, op(a / scale, b / scale, 1.0 / rel) * scale, hard(a, b))


def _safe_floor(x):
    """Smallest magnitude whose square is a normal number in ``x``'s dtype."""
    return math.sqrt(float(jnp.finfo(jnp.result_type(x)).tiny))


def _rel_smin(a, b, rel):
    return _rel_smooth(smooth_min, jnp.minimum, a, b, rel)


def _rel_smax(a, b, rel):
    return _rel_smooth(smooth_max, jnp.maximum, a, b, rel)


def _seqsum(x, axis):
    """Left-to-right sum (the oracle's DO-loop accumulation order)."""
    x = jnp.moveaxis(x, axis, 0)
    acc = jnp.zeros_like(x[0])
    for i in range(x.shape[0]):
        acc = acc + x[i]
    return acc


def wms_drag(T, u, v, p_full, p_half, geo_full, lat, pgaw, *, launch_level: int,
             dt_gw: float, cfg: IFSWMSConfig = IFSWMSConfig(),
             g: float = constants.g, R_d: float = constants.R_d,
             c_pd: float = constants.c_pd, radius: float = constants.R_earth):
    """GWDRAG_WMS (gwdrag_wms.F90:203-614).  Arrays ``(ncol, nlev)`` top to
    bottom (``p_half``: ``(ncol, nlev+1)``); ``launch_level`` = SUGWWMS
    ``NLAUNCHL(1)`` (1-based); ``dt_gw`` = GTPHYGWWMS.
    Returns ``dudt, dvdt, flux_u, flux_v`` (fluxes on the ``nlev+1`` halves)."""
    T, u, v = jnp.asarray(T), jnp.asarray(u), jnp.asarray(v)
    ncol, nlev = T.shape
    L = launch_level - 1                      # 0-based ILAUNCH
    hard = all(w == 0.0 for w in (cfg.n2_floor_rel_width, cfg.speed_width_m_s,
                                  cfg.saturation_rel_width, cfg.deposit_rel_width))
    ci_np, dci_np = _spectrum()
    ci, dci = jnp.asarray(ci_np, T.dtype), jnp.asarray(dci_np, T.dtype)
    cosa_np, sina_np, az_fct = _azimuths()
    cosa, sina = jnp.asarray(cosa_np, T.dtype), jnp.asarray(sina_np, T.dtype)
    ms = 2.0 * math.pi / cfg.m_star_length_m

    # half-level T, u, v (index k = half level above full level k; k=0 -> full)
    th = jnp.concatenate([T[:, :1], 0.5 * (T[:, :-1] + T[:, 1:])], axis=1)
    uh = jnp.concatenate([u[:, :1], 0.5 * (u[:, :-1] + u[:, 1:])], axis=1)
    vh = jnp.concatenate([v[:, :1], 0.5 * (v[:, :-1] + v[:, 1:])], axis=1)
    cons1 = 1.0 / R_d
    cons2 = g ** 2 / c_pd
    delp_geo = geo_full[:, 1:] - geo_full[:, :-1]                  # k = 1..nlev-1
    rho = p_half[:, 1:nlev] * cons1 / th[:, 1:]
    n2 = cons2 / th[:, 1:] * (1.0 + c_pd * (T[:, 1:] - T[:, :-1]) / delp_geo)
    n2 = smooth_max(n2, jnp.full_like(n2, _GSSEC), _sharp(cfg.n2_floor_rel_width * _GSSEC))
    bvf = jnp.sqrt(n2)
    # pad index 0 (never read; ones, not NaN, so no NaN reaches a cotangent)
    # so arrays are indexed like the oracle's JK (k = 1..nlev-1 valid)
    rho = jnp.concatenate([jnp.ones((ncol, 1), T.dtype), rho], axis=1)
    bvf = jnp.concatenate([jnp.ones((ncol, 1), T.dtype), bvf], axis=1)

    ul = cosa[None, :] * uh[:, L, None] + sina[None, :] * vh[:, L, None]      # (ncol, azi)
    bvfl = bvf[:, L]
    ui = (cosa[None, None, :] * uh[:, :, None] + sina[None, None, :] * vh[:, :, None]
          - ul[:, None, :])                                                   # (ncol, nlev, azi)
    fct = rho / bvf                                                           # (ncol, nlev)

    # launch spectrum (NSLOPE = 1, LGSATL = F)
    bvfl4 = bvfl ** 4
    cin4 = (ms * ci) ** 4
    flux1 = fct[:, L, None] * bvfl4[:, None] * ci[None, :] / (bvfl4[:, None] + cin4[None, :])
    pu_launch = _seqsum(flux1 * dci[None, :], axis=1)

    dxg = jnp.maximum(_DX_MIN, 2.0 * radius * jnp.sqrt(math.pi * jnp.asarray(pgaw)))
    dxs = 1.0 - jnp.minimum(1.0, jnp.arctan((jnp.maximum(1.0 / dxg, _DXA) - _DXA) / (_DXB - _DXA)))
    fluxlaun = cfg.flux_launch_pa * dxs
    latdeg = lat * _RAD_TO_DEG
    gauss = cfg.gauss_amplitude * jnp.exp((-(latdeg * latdeg))
                                          / (2.0 * cfg.gauss_halfwidth_deg * cfg.gauss_halfwidth_deg))
    fluxlaun = (1.0 + gauss) * fluxlaun
    fnorm = fluxlaun / pu_launch
    fct = fnorm[:, None] * fct
    flux0 = jnp.broadcast_to((fnorm[:, None] * flux1)[:, :, None], (ncol, _N_C, _N_AZI))

    s_speed = _sharp(cfg.speed_width_m_s)
    crt_eps = 0.0 if hard else _CRT_EPS_REL * fluxlaun[:, None] + _safe_floor(T)

    def down(carry, k):
        cimin, act, flux, crt_below = carry
        uik = ui[:, k, :]
        cimin = smooth_max(cimin, uik, s_speed)
        x = ci[None, :, None] - cimin[:, None, :]
        if hard:
            atmp = 0.5 + jnp.where(x >= 0.0, 0.5, -0.5)
        else:
            atmp = sigmoid_switch(x, s_speed)
        acc = act - atmp
        act = atmp
        dfl = _seqsum(acc * flux * dci[None, :, None], axis=1)
        num = _seqsum(ci[None, :, None] * acc * flux * dci[None, :, None], axis=1)
        if hard:
            safe = jnp.where(dfl > 0.0, dfl, 1.0)
            crt = jnp.where(dfl > 0.0, (0.0 + num) / safe, crt_below)
        else:
            den = dfl + crt_eps
            ok = den > _safe_floor(den)           # zero-flux column: keep ZCRT_below
            crt = jnp.where(ok, (num + crt_eps * crt_below) / jnp.where(ok, den, 1.0), crt_below)
        fluxs = (cfg.c_star * fct[:, k, None, None]
                 * (ci[None, :, None] - uik[:, None, :]) ** 2 * (1.0 / ci)[None, :, None])
        if hard:
            dep = act * (flux - fluxs)
            flux = jnp.where(dep > 0.0, fluxs, flux)
        else:
            flux = act * _rel_smin(flux, fluxs, cfg.saturation_rel_width) + (1.0 - act) * flux
        pu = _seqsum(act * flux * dci[None, :, None], axis=1)
        return (cimin, act, flux, crt), (dfl, crt, pu)

    carry0 = (jnp.full((ncol, _N_AZI), _C_MIN, T.dtype), jnp.ones((ncol, _N_C, _N_AZI), T.dtype),
              flux0, jnp.zeros((ncol, _N_AZI), T.dtype))
    ks = jnp.arange(L - 1, 0, -1)                      # oracle JK = ILAUNCH-1 .. 2
    _, (dfl_s, crt_s, pu_s) = jax.lax.scan(down, carry0, ks)
    # assemble on 0-based levels 0..L (index 0 and L hold the oracle's initial zeros / launch)
    zeros = jnp.zeros((1, ncol, _N_AZI), T.dtype)
    dfl = jnp.concatenate([zeros, dfl_s[::-1], zeros], axis=0)       # (L+1, ncol, azi)
    crt = jnp.concatenate([zeros, crt_s[::-1], zeros], axis=0)
    pu = jnp.concatenate([zeros, pu_s[::-1],
                          jnp.broadcast_to(fluxlaun[None, :, None], (1, ncol, _N_AZI))], axis=0)

    rgpts = 1.0 / (g * dt_gw)
    s_dep_rel = cfg.deposit_rel_width

    def limit(cngl, k):
        ulm = cosa[None, :] * u[:, k, None] + sina[None, :] * v[:, k, None] - ul
        d = dfl[k - 1] + cngl
        cap = 2.0 * (p_full[:, k - 1, None] - p_full[:, k, None]) * (crt[k - 1] - ulm) * rgpts
        if hard:
            dft = jnp.maximum(jnp.minimum(d, cap), 0.0)
        else:
            dft = _rel_smax(_rel_smin(d, cap, s_dep_rel), jnp.zeros_like(d), s_dep_rel)
        cngl = d - dft
        return cngl, cngl

    _, cngl_s = jax.lax.scan(limit, jnp.zeros((ncol, _N_AZI), T.dtype), jnp.arange(1, L + 1))
    pu = pu.at[1:].add(-cngl_s)

    fu = _seqsum(pu * az_fct * cosa[None, None, :], axis=2)          # (L+1, ncol)
    fv = _seqsum(pu * az_fct * sina[None, None, :], axis=2)
    fu = fu.at[0].set(0.0).T
    fv = fv.at[0].set(0.0).T
    pad = jnp.zeros((ncol, nlev + 1 - (L + 1)), T.dtype)
    flux_u = jnp.concatenate([fu, pad], axis=1)                      # (ncol, nlev+1)
    flux_v = jnp.concatenate([fv, pad], axis=1)
    rdp = g / (p_half[:, 1:] - p_half[:, :-1])
    lev = jnp.arange(nlev) <= L
    dudt = jnp.where(lev, (flux_u[:, 1:] - flux_u[:, :-1]) * rdp, 0.0)
    dvdt = jnp.where(lev, (flux_v[:, 1:] - flux_v[:, :-1]) * rdp, 0.0)
    return dudt, dvdt, flux_u, flux_v


def wms_heating(u, v, dudt, dvdt, dt: float, c_pd: float = constants.c_pd):
    """Frictional heating of the WMS tendencies applied over a step ``dt`` [s].

    49r1 (callpar.F90:1357-1358) uses ``-(u dudt + v dvdt)/c_pd`` with the
    current winds, which returns the kinetic-energy loss only to O(dt).  This
    port uses the energy-exact form for the applied step (vote V11b, a
    DEPARTURE from 49r1, energy conservation outranking oracle fidelity):
    ``-(u dudt + v dvdt + 0.5 dt (dudt^2 + dvdt^2)) / c_pd``, so that
    ``c_pd dT = -(KE(u + dt dudt) - KE(u))`` exactly per unit mass."""
    return -(u * dudt + v * dvdt + 0.5 * dt * (dudt * dudt + dvdt * dvdt)) * (1.0 / c_pd)
