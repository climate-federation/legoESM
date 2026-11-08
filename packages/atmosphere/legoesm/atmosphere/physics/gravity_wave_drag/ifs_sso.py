"""IFS subgrid-scale orographic drag (Lott & Miller 1997) as OpenIFS 49r1 runs it.

Faithful port of OpenIFS (github.com/ecmwf-ifs/openifs @ 8f6f722f73, cycle
49r1) ``ifs-source/arpifs/phys_ec/{gwdrag,gwsetup,gwprofil,sugwd}.F90``:
blocked-flow drag below the dividing-streamline height (returned as the
implicit coefficient PSOBETA, which IFS feeds into the VDF joint implicit
momentum solve) plus the anisotropic orographic gravity-wave stress profile
with Richardson-number saturation and critical-level absorption (returned as
explicit u/v tendencies PSOTEU/PSOTEV).

Configuration ported: LPHYLIN = .FALSE., LEPHYS = .TRUE., empty NAMGWD (49r1
code values GKDRAG = 0.15, GKWAKE = 3, GTENLIM = 80 m/s/h).  The level
constants NKTOPG / NGWDTOP / NGWDLIM come from :func:`sugwd_levels` (a port of
SUGWD's loops) on the run's reference coordinate.

Differentiability (votes V4a/V5a, oracle_audit/ifs_port/vdf_gwd/VOTES.md): the
integer levels (height crossings of n*std, Froude-integral crossings, the
critical level) are kept discrete exactly as the oracle, as in the merged IFS
convection port; every VALUE switch, floor, clip and branch is smooth through
:mod:`legoesm.core.smooth`, each with a named width in
:class:`IFSSSOConfig` (default 0.01 of the switch's natural scale).  All
widths 0 give the oracle's hard form (used only by the oracle-match test).

One column per call, vmapped over columns.  Arrays are padded 1-based
(index 0 and KLEV+1 are dummies) so every index reads as in the Fortran.

Also the turbulent orographic form drag (TOFD, Beljaars, Brown & Wood 2004)
coefficient as VDFMAIN computes it (``vdftofdc.F90``, RTOFDALPHA from
``suvdf.F90``): :func:`ifs_tofd_coefficient`.  Like PSOBETA it enters the VDF
implicit momentum solve, so both are returned as coefficients, not applied.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.smooth import positive_part_c1, sigmoid_switch, smooth_max

__physics_contract__ = {
    "summary": (
        "Lott & Miller (1997) subgrid-orography drag, OpenIFS 49r1 GWDRAG: "
        "low-level blocked-flow drag (implicit coefficient) and anisotropic "
        "orographic gravity-wave stress with Ri-saturation and critical-level "
        "absorption (explicit u/v tendencies)."
    ),
    "inputs": {
        "p_half": "Pa (KLEV+1, top -> surface)", "p_full": "Pa", "geo_full": "m2 s-2 above the surface",
        "T": "K", "u": "m s-1", "v": "m s-1",
        "std": "m (subgrid orography standard deviation)", "gamma": "1 (anisotropy)",
        "theta": "rad (principal-axis orientation)", "sigma": "1 (mean slope)",
        "nktopg/ngwdtop/ngwdlim": "level indices from sugwd_levels",
    },
    "outputs": {
        "soteu": "m s-2", "sotev": "m s-2",
        "sobeta": "s-1 (implicit blocking coefficient: du/dt = -sobeta*u)",
        "tofdc": "s-1 (TOFD implicit coefficient from ifs_tofd_coefficient: du/dt = -tofdc*u)",
    },
    "sign_convention": (
        "levels top -> surface (k = 1 top). Wave stress tau >= 0 on half levels "
        "(tau(KLEV+1) at the surface) along the low-level wind; the tendency "
        "-g*dtau/dp is projected back on (ulow, vlow) through (d1, d2), so it "
        "opposes the low-level flow. Blocking: du/dt = -sobeta*u with sobeta >= 0."
    ),
    # Momentum: below the top region the column integral of the wave tendency
    # equals the stress divergence (a mountain sink); the 49r1 top region
    # (p_full < GRFPLM) spreads tau(NGWDTOP+1)-tau(1) with weights dp/dp0^2,
    # which does not sum to the stress, and the GTENLIM clip removes stress,
    # both as in the oracle. No heating: IFS returns the KE loss as heat in VDF.
    # Only u, v tendencies: dry mass and water are untouched.
    "conserves": ["mass"],
    "differentiable": True,
    "reference": (
        "Lott & Miller (1997) QJRMS 123, 101-127; Beljaars, Brown & Wood (2004) "
        "QJRMS 130, 1327-1347; OpenIFS 49r1 (8f6f722) gwdrag.F90 / gwsetup.F90 / "
        "gwprofil.F90 / sugwd.F90 / vdftofdc.F90 / suvdf.F90"
    ),
    "idealized_test": (
        "tests/atmosphere/test_ifs_sso_oracle.py: 49r1 Fortran "
        "oracle columns (hard form) and smooth-form gradient / momentum checks"
    ),
}

__param_spec__ = {
    "IFSSSOConfig": {
        "scheme_key": "atm.gwd.IFSSSOConfig",
        "excluded": {
            "gtenlim_m_s2": "upper-level tendency limiter (a cap, not a closure)",
            "grfplm_pa": "model-top region pressure (numerics of the top boundary)",
            "gvsec_m_s": "numerical wind floor",
            "gssec": "numerical stability floor",
            "gtsec": "numerical stress / anisotropy floor",
            "std_switch_width_m": "smoothing width", "floor_width_frac": "smoothing width",
            "gvsec_width_m_s": "smoothing width", "ri_width": "smoothing width",
            "gtsec_width": "smoothing width", "tenlim_width_frac": "smoothing width",
            "zeff_width_frac": "smoothing width", "ratio_width": "smoothing width",
            "guard_width": "smoothing width", "top_width_pa": "smoothing width",
            "tau_min_width_frac": "smoothing width",
        },
        "params": {
            # --- Lott & Miller closure (sugwd.F90) ---
            "gkdrag": {"units": "1", "bounds": (0.05, 0.5), "tunable_tier": 1, "transform": "sigmoid", "category": "wave_drag", "reference": "Lott & Miller (1997); OpenIFS 49r1 sugwd GKDRAG", "shape": None},
            "gkwake": {"units": "1", "bounds": (0.5, 6.0), "tunable_tier": 1, "transform": "sigmoid", "category": "blocking", "reference": "Lott & Miller (1997) Cd; OpenIFS 49r1 sugwd GKWAKE", "shape": None},
            "grcrit": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "wave_drag", "reference": "OpenIFS 49r1 sugwd GRCRIT (critical Richardson number)", "shape": None},
            "gfrcrit": {"units": "1", "bounds": (0.2, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "blocking", "reference": "OpenIFS 49r1 sugwd GFRCRIT (critical Froude number)", "shape": None},
        },
    },
    "IFSTOFDConfig": {
        "scheme_key": "atm.gwd.IFSTOFDConfig",
        "excluded": {"zmax_width_m": "smoothing width", "repdu2_width_frac": "smoothing width"},
        "params": {
            "tofd_alpha": {"units": "1", "bounds": (5.0, 50.0), "tunable_tier": 1, "transform": "sigmoid", "category": "form_drag", "reference": "Beljaars et al. (2004); OpenIFS 49r1 suvdf RTOFDALPHA", "shape": None},
        },
    },
}

# --- Lott & Miller (1997) anisotropy fits, gwsetup.F90 / gwdrag.F90 ---
_LM_B1 = 0.18
_LM_B2 = 0.04
_LM_C1 = 0.48
_LM_C2 = 0.3
_LM_EFF_DIV = 9.0          # ZEFF**2/9 in the surface stress (gwdrag.F90)
_FROUDE_CRITH = 1.5        # wave-breaking critical-height Froude integral (gwsetup.F90)
_STD_ON_M = 50.0           # GWDRAG active where PHSTD > 50 m (gwdrag.F90)
_GUARD_ONE = 1.0           # MAX(., 1) guards in gwsetup/gwprofil (m, m2 s-2)
_RI_INIT = 9999.0          # PRI initial value (gwsetup.F90)
# --- SUGWD reference levels (sugwd.F90) ---
_SUGWD_SIGT = 0.94
_SUGWD_PR = 80000.0
_SUGWD_PLIM_HI = 5000.0
_SUGWD_PLIM_LO = 1000.0
_SUGWD_KLEV_SWITCH = 19
# --- TOFD spectral fit (Beljaars et al. 2004; vdftofdc.F90).  The Fortran
# assigns these as default-REAL (single precision) literals, so the oracle
# computes with their float32 values; kept bit-identical here. ---
_f4 = lambda x: float(np.float32(x))
_TOFD_BETA = _f4(1.0)
_TOFD_CMD = _f4(0.005)
_TOFD_CORR = _f4(0.6)
_TOFD_K1 = _f4(0.003)
_TOFD_IH = _f4(0.00102)
_TOFD_KFLT = _f4(0.00035)
_TOFD_N1 = _f4(-1.9)
_TOFD_N2 = _f4(-2.8)
_TOFD_IC = _f4(2.109)
_TOFD_IZ_M = _f4(1500.0)
_TOFD_IN = _f4(-1.2)
_TOFD_EXP = _f4(1.5)
_TOFD_ZMAX_M = _f4(5000.0)
_TOFD_FACT1 = _TOFD_K1 ** (_TOFD_N1 - _TOFD_N2) / (_TOFD_IH * _TOFD_KFLT ** _TOFD_N1)
_REPDU2 = 0.1 ** 2          # m2 s-2, parphy.F90 REPDU2 = (0.1_JPRB)**2


class IFSSSOConfig(NamedTuple):
    """OpenIFS 49r1 GWDRAG settings (sugwd.F90 code values) and smooth widths."""

    # --- Lott & Miller closure (sugwd.F90, 49r1 code values) ---
    gkdrag: float = 0.15
    gkwake: float = 3.0
    grcrit: float = 0.25
    gfrcrit: float = 0.40
    gtenlim_m_s2: float = 80.0 / 3600.0
    grfplm_pa: float = 9.9
    # --- numerical floors (sugwd.F90) ---
    gvsec_m_s: float = 0.10
    gssec: float = 1.0e-12
    gtsec: float = 1.0e-7
    # --- smoothing widths (0 = hard oracle form; votes V4a/V5a) ---
    std_switch_width_m: float = 0.5
    floor_width_frac: float = 0.01
    gvsec_width_m_s: float = 0.001
    ri_width: float = 0.0025
    gtsec_width: float = 1.0e-9
    tenlim_width_frac: float = 0.01
    zeff_width_frac: float = 0.01
    ratio_width: float = 0.01
    guard_width: float = 0.01
    top_width_pa: float = 0.099
    tau_min_width_frac: float = 0.01


class IFSTOFDConfig(NamedTuple):
    """OpenIFS 49r1 TOFD setting (suvdf.F90) and smooth widths (0 = hard)."""

    tofd_alpha: float = 24.0             # RTOFDALPHA, suvdf.F90:128
    zmax_width_m: float = 50.0           # 0.01 x the 5000 m cut-off
    repdu2_width_frac: float = 0.01


class IFSSSOOutput(NamedTuple):
    """GWDRAG outputs per column (levels top -> surface)."""

    soteu: jax.Array     # m s-2, explicit wave tendency (PSOTEU)
    sotev: jax.Array     # m s-2 (PSOTEV)
    sobeta: jax.Array    # s-1, implicit blocking coefficient (PSOBETA)
    tau: jax.Array       # Pa, wave stress on half levels 1..KLEV+1 (diagnostic)
    dir_u: jax.Array     # s m-1: soteu = -g*dtau/dp * dir_u (diagnostic)
    dir_v: jax.Array     # s m-1
    ulow: jax.Array      # m s-1, low-level mean wind (diagnostic)
    vlow: jax.Array
    ints: jax.Array      # (KKCRITH, KCRIT, KKENVH, KKNU, KKNU2), 1-based (diagnostic)
    sat_levels: jax.Array  # Ri-saturated GWPROFIL branch weight per level (diagnostic)


def hard_config(cfg: IFSSSOConfig = IFSSSOConfig()) -> IFSSSOConfig:
    """The oracle's hard form: every smoothing width 0 (oracle-match test only)."""
    return cfg._replace(std_switch_width_m=0.0, floor_width_frac=0.0, gvsec_width_m_s=0.0,
                        ri_width=0.0, gtsec_width=0.0, tenlim_width_frac=0.0,
                        zeff_width_frac=0.0, ratio_width=0.0, guard_width=0.0,
                        top_width_pa=0.0, tau_min_width_frac=0.0)


def hard_tofd_config(cfg: IFSTOFDConfig = IFSTOFDConfig()) -> IFSTOFDConfig:
    """The oracle's hard form (oracle-match test only)."""
    return cfg._replace(zmax_width_m=0.0, repdu2_width_frac=0.0)


def _s(width):
    return math.inf if width == 0.0 else 1.0 / width


def _step(x, width, strict=False):
    """Heaviside(x); hard (width 0): x > 0 if strict else x >= 0, as the oracle's test."""
    if width == 0.0 and strict:
        x = jnp.asarray(x)
        return (x > 0).astype(x.dtype)
    return sigmoid_switch(x, _s(width))


def _max(a, b, width):
    return smooth_max(a, b, _s(width))


def _min_below(a, b, width_frac, b_floor):
    """MIN(a, b) for b >= 0 as b - (b - a)^+: EXACTLY b wherever a >= b.

    GWPROFIL's unsaturated branch returns a == b (up to rounding) at every
    level, so a symmetric smooth min would shave every level and compound
    with height; this one-sided form is exact there and smooth (C1 positive
    part, width = width_frac * max(b, b_floor)) only where the saturated stress
    is lower.  The floor keeps the reverse pass finite for vanishing b.
    """
    if width_frac == 0.0:
        return jnp.minimum(a, b)
    bs = smooth_max(jnp.where(b > 0.0, b, b_floor), b_floor, 1.0 / (width_frac * b_floor))
    return jnp.where(b > 0.0, b - bs * positive_part_c1((b - a) / bs, 1.0 / width_frac),
                     jnp.minimum(a, b))


def sugwd_levels(vah, vbh, stpre, cfg: IFSSSOConfig = IFSSSOConfig(), lephys: bool = True):
    """NKTOPG, NGWDTOP, NGWDLIM as SUGWD sets them (sugwd.F90), static numpy.

    vah, vbh: hybrid coefficients on the KLEV+1 half levels (Pa, 1), top -> surface;
    stpre: standard-atmosphere full-level pressures (Pa).
    """
    vah, vbh, stpre = (np.asarray(a, dtype=np.float64) for a in (vah, vbh, stpre))
    klev = stpre.shape[0]
    zpm1r = 0.5 * (vah[:-1] + vah[1:] + _SUGWD_PR * (vbh[:-1] + vbh[1:]))
    jk = np.arange(1, klev + 1)
    hit = (zpm1r / _SUGWD_PR) >= _SUGWD_SIGT
    nktopg = int(jk[hit].min()) if hit.any() else klev
    top = zpm1r < cfg.grfplm_pa
    ngwdtop = int(jk[top].max()) if top.any() else 1
    zplim = _SUGWD_PLIM_HI if klev > _SUGWD_KLEV_SWITCH else _SUGWD_PLIM_LO
    lim = stpre >= zplim
    ngwdlim = int(jk[lim].min()) if lim.any() else 0
    return nktopg, ngwdtop, (ngwdlim if lephys else 0)


def _ssqrt(x):
    """sqrt with a finite (zero) gradient where x <= 0 (padding / zero-wind rows)."""
    ok = x > 0.0
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, x, 1.0)), 0.0)


def _pad(a):
    z = jnp.zeros((1,), a.dtype)
    return jnp.concatenate([z, a, z])


def _column(ph, pf, geo, T, u, v, std, gamma, theta, sig, nktopg, ngwdtop, ngwdlim,
            cfg: IFSSSOConfig, g, rd, cpd):
    klev = pf.shape[0]
    ilevh = klev // 3
    dt_ = pf.dtype
    pf, geo, T, u, v = (_pad(a) for a in (pf, geo, T, u, v))     # 1..klev
    kk = jnp.arange(klev + 2)
    fw = cfg.floor_width_frac
    c1 = 1.0 / rd
    c2 = g ** 2 / cpd
    z1d2rg = 1.0 / (2.0 * g)

    on = _step(std - _STD_ON_M, cfg.std_switch_width_m, strict=True)      # PHSTD > 50
    std_s = jnp.where(std > 0.0, std, 1.0)       # inactive columns: on ~ 0
    gamma = _max(gamma, cfg.gtsec, fw * cfg.gtsec)

    # --- GWSETUP: n*std height crossings (discrete, as the oracle) ---
    hgeo = jnp.zeros(klev + 2, dt_).at[1:klev + 1].set(
        (geo[1:klev + 1] + jnp.concatenate([jnp.zeros(1, dt_), geo[1:klev]])) * z1d2rg)
    idx = jnp.arange(ilevh, klev + 1)

    def crossing(mult):
        ll = jnp.zeros(klev + 2, bool).at[ilevh:klev + 1].set(hgeo[ilevh:klev + 1] > mult * std)
        cr = ll[ilevh:klev + 1] != ll[ilevh + 1:klev + 2]
        return jnp.min(jnp.where(cr, idx, klev))

    kknu, kknu2, iknub, iknul = crossing(4.0), crossing(3.0), crossing(2.0), crossing(1.0)
    kknu = jnp.minimum(kknu, nktopg)
    iknub = jnp.minimum(iknub, nktopg)
    iknul = jnp.where(iknub == nktopg, klev, iknul)
    iknul = jnp.where(iknub == iknul, iknul + 1, iknul)

    # --- density, stability on half levels k = 2..klev ---
    zdp = jnp.zeros(klev + 2, dt_).at[2:klev + 1].set(pf[2:klev + 1] - pf[1:klev])
    tsum = jnp.ones(klev + 2, dt_).at[2:klev + 1].set(T[2:klev + 1] + T[1:klev])
    dtk = jnp.zeros(klev + 2, dt_).at[2:klev + 1].set(T[2:klev + 1] - T[1:klev])
    phm = jnp.zeros(klev + 2, dt_).at[2:klev + 1].set(ph[1:klev])          # PAPHM1(k-1)
    zdp_s = jnp.where(zdp != 0.0, zdp, 1.0)
    act = (kk >= 2) & (kk <= klev)
    rho = jnp.where(act, 2.0 * phm * c1 / tsum, 0.0)
    stab = 2.0 * c2 / tsum * (1.0 - cpd * rho * dtk / zdp_s)
    stab = jnp.where(act, _max(stab, cfg.gssec, fw * cfg.gssec), 0.0)
    sqst = _ssqrt(stab)

    # low-level stability/density average, k = ilevh..klev ascending (sequential MAX)
    def lowlev(carry, k):
        s, r = carry
        inr = (k >= iknub + 1) & (k <= iknul)
        zst = c2 / T[k] * (1.0 - cpd * rho[k] * (T[k] - T[k - 1]) / zdp_s[k])
        s_new = _max(s + zst * zdp[k], cfg.gssec, fw * cfg.gssec)
        r_new = r + ph[k - 1] * 2.0 * zdp[k] * c1 / (T[k] + T[k - 1])
        return (jnp.where(inr, s_new, s), jnp.where(inr, r_new, r)), None

    (s_b, r_b), _ = jax.lax.scan(lowlev, (jnp.zeros((), dt_), jnp.zeros((), dt_)),
                                 jnp.arange(ilevh, klev + 1))
    pfx = jnp.concatenate([pf, pf[-1:]])        # PAPM1(KLEV+1) read when iknul = klev+1
    dlow = pfx[iknul] - pfx[iknub]
    stab_b = s_b / dlow
    rho_b = r_b / dlow

    # low-level wind, k = klev..ilevh descending
    def lowwind(carry, k):
        ul, vl = carry
        inr = (k >= iknub) & (k <= iknul)
        dph = ph[k] - ph[k - 1]
        return (jnp.where(inr, ul + u[k] * dph, ul), jnp.where(inr, vl + v[k] * dph, vl)), None

    (ulow, vlow), _ = jax.lax.scan(lowwind, (jnp.zeros((), dt_), jnp.zeros((), dt_)),
                                   jnp.arange(klev, ilevh - 1, -1))
    phx = jnp.concatenate([ph, ph[-1:]])
    dlw = phx[iknul] - phx[iknub - 1]
    ulow = ulow / dlw
    vlow = vlow / dlw
    znorm = _max(_ssqrt(ulow ** 2 + vlow ** 2), cfg.gvsec_m_s, cfg.gvsec_width_m_s)
    vph_b = znorm

    def shifted(x):
        """ZU = x + 2*GVSEC where -GVSEC <= x < GVSEC (smooth window)."""
        w = (_step(cfg.gvsec_m_s - x, cfg.gvsec_width_m_s, strict=True)
             * _step(x + cfg.gvsec_m_s, cfg.gvsec_width_m_s))
        return x + 2.0 * cfg.gvsec_m_s * w

    zu = shifted(ulow)
    zu = jnp.where(zu != 0.0, zu, cfg.gvsec_m_s)
    psi_b = theta - jnp.arctan(vlow / zu)
    zb = 1.0 - _LM_B1 * gamma - _LM_B2 * gamma ** 2
    zc = _LM_C1 * gamma + _LM_C2 * gamma ** 2
    d1 = zb - (zb - zc) * (jnp.sin(psi_b) ** 2)
    d2 = (zb - zc) * jnp.sin(psi_b) * jnp.cos(psi_b)
    dmod = jnp.sqrt(d1 ** 2 + d2 ** 2)

    zvt1 = ulow * u + vlow * v
    zvt2 = -vlow * u + ulow * v
    vpf = (zvt1 * d1 + zvt2 * d2) / (znorm * dmod)
    vpfm = jnp.concatenate([jnp.zeros(1, dt_), vpf[:-1]])               # vpf(k-1)
    pfm = jnp.concatenate([jnp.zeros(1, dt_), pf[:-1]])                 # papm1(k-1)
    vph_raw = ((phm - pfm) * vpf + (pf - phm) * vpfm) / zdp_s
    vph = jnp.where(act, _max(vph_raw, cfg.gvsec_m_s, cfg.gvsec_width_m_s), 0.0)
    crit = act & (vph_raw < cfg.gvsec_m_s) & (kk <= iknul)
    kcrit = jnp.max(jnp.where(crit, kk, 1))

    zdwind = _max(jnp.abs(vpf - vpfm), cfg.gvsec_m_s, cfg.gvsec_width_m_s)
    rho_s = jnp.where(act, rho, 1.0)
    pri = stab * (zdp / (g * rho_s * zdwind)) ** 2
    pri = jnp.where(act, _max(pri, cfg.grcrit, cfg.ri_width), _RI_INIT)

    # projected wind |.| floored, as recomputed at every level in the Froude loops
    zwind = (ulow * u + vlow * v) / _max(_ssqrt(ulow ** 2 + vlow ** 2), cfg.gvsec_m_s,
                                         cfg.gvsec_width_m_s)
    zwind = _max(jnp.abs(zwind), cfg.gvsec_m_s, cfg.gvsec_width_m_s)
    sqr = jnp.where(act, sqst / rho_s, 0.0)                            # sqst(k)/rho(k)
    sqrp = jnp.concatenate([sqr[1:], jnp.zeros(1, dt_)])               # at k+1
    dph_all = jnp.zeros(klev + 2, dt_).at[1:klev + 1].set(ph[1:] - ph[:-1])
    dph_top = jnp.zeros(klev + 2, dt_).at[1:klev + 1].set(ph[1:] - pf[1:klev + 1])
    inc_mid = (dph_all / g) * ((sqrp + sqr) / 2.0) / zwind
    inc_first = (dph_top / g) * sqrp / zwind

    def froude(carry, k):
        pnu, kenv = carry
        inc = jnp.where(k == kknu2, inc_first[k], jnp.where(k > kknu2, inc_mid[k], 0.0))
        new = pnu + inc
        hit = (pnu <= cfg.gfrcrit) & (new > cfg.gfrcrit) & (kenv == klev) & (k >= kknu2)
        return (new, jnp.where(hit, k, kenv)), None

    (_, kkenvh), _ = jax.lax.scan(froude, (jnp.zeros((), dt_), jnp.asarray(klev)),
                                  jnp.arange(2, klev))

    def crith(carry, k):
        nup, kcr = carry
        new = jnp.where(k < kkenvh, nup + inc_mid[k], nup)
        hit = (k < kkenvh) & (nup <= _FROUDE_CRITH) & (new > _FROUDE_CRITH) & (kcr == klev)
        return (new, jnp.where(hit, k, kcr)), None

    (_, kkcrith), _ = jax.lax.scan(crith, (jnp.zeros((), dt_), jnp.asarray(klev)),
                                   jnp.arange(klev - 1, 1, -1))
    kkcrith = jnp.minimum(kkcrith, kknu)

    blk = (kk >= ilevh) & (kk <= klev) & (kk >= kkenvh)
    zuk = shifted(u)
    zuk = jnp.where(zuk != 0.0, zuk, cfg.gvsec_m_s)
    psi = jnp.where(blk, theta - jnp.arctan(v / zuk), 0.0)
    gw = cfg.guard_width
    ggeenv = _max(_GUARD_ONE, (geo[kkenvh] + geo[kkenvh - 1]) / 2.0, gw)
    ggeom1 = _max(geo, _GUARD_ONE, gw)
    gvar = _max(std * g, _GUARD_ONE, gw)
    arg = (ggeenv - ggeom1) / (ggeom1 + gvar)
    zdep = jnp.where(blk, _ssqrt(arg), 0.0)

    # --- GWDRAG: surface wave stress ---
    zblock = geo[kkenvh] / g
    x3 = 3.0 * std_s - zblock
    if cfg.zeff_width_frac == 0.0:
        zeff = jnp.maximum(0.0, x3) * 2.0
    else:
        zeff = positive_part_c1(x3 / (3.0 * std_s), 1.0 / cfg.zeff_width_frac) * (3.0 * std_s) * 2.0
    tau_b = cfg.gkdrag * rho_b * sig * zeff ** 2 / _LM_EFF_DIV / std_s * vph_b * dmod * _ssqrt(stab_b)

    # --- GWPROFIL ---
    zoro = sig * dmod / 4.0 / _max(std, _GUARD_ONE, gw)
    gr = cfg.grcrit

    def profil(tau_next, k):
        znrm = cfg.gkdrag * rho[k] * _ssqrt(stab[k]) * vph[k] * zoro
        zdz2 = tau_next / _max(znrm, cfg.gssec, fw * cfg.gssec)
        zsqr = jnp.sqrt(pri[k])
        vk = jnp.where(vph[k] != 0.0, vph[k], cfg.gvsec_m_s)
        sk = jnp.where(stab[k] != 0.0, stab[k], cfg.gssec)
        zalfa = _ssqrt(sk * zdz2) / vk
        zriw = pri[k] * (1.0 - zalfa) / (1 + zalfa * zsqr) ** 2
        zdel = 4.0 / zsqr / gr + 1.0 / gr ** 2 + 4.0 / gr
        zbb = 1.0 / gr + 2.0 / zsqr
        zalpha = 0.5 * (-zbb + jnp.sqrt(zdel))
        zdz2n = (vk * zalpha) ** 2 / sk
        sat = _step(gr - zriw, cfg.ri_width, strict=True)                # ZRIW < GRCRIT
        t = sat * (znrm * zdz2n) + (1.0 - sat) * (znrm * zdz2)
        t = _min_below(t, tau_next, cfg.tau_min_width_frac, cfg.gtsec)
        alive = _step(tau_next - cfg.gtsec, cfg.gtsec_width) * (k > kcrit)
        t = alive * t
        out = jnp.where(k >= kknu2, tau_b, t)
        return out, (out, jnp.where(k >= kknu2, 0.0, sat * alive))

    _, (taus, sats) = jax.lax.scan(profil, tau_b, jnp.arange(klev, 1, -1))
    sat_levels = jnp.zeros(klev, dt_).at[1:].set(sats[::-1])          # levels 2..klev
    ptau = jnp.zeros(klev + 2, dt_).at[2:klev + 1].set(taus[::-1]).at[klev + 1].set(tau_b)
    tenv, tcr = ptau[kkenvh], ptau[kkcrith]
    phk1 = jnp.concatenate([jnp.zeros(1, dt_), ph])                   # PAPHM1(k-1), k = 0..klev+1
    zdelp = phk1 - ph[kkenvh - 1]
    zdelpt = ph[kkcrith - 1] - ph[kkenvh - 1]
    mid = (kk > kkcrith) & (kk < kkenvh) & (kk >= 1) & (kk <= klev)
    zdelpt_s = jnp.where(zdelpt != 0.0, zdelpt, 1.0)
    ptau = jnp.where(mid, tenv + (tcr - tenv) * zdelp / zdelpt_s, ptau)

    # --- GWDRAG: tendencies, k = 1..klev ---
    kl = jnp.arange(1, klev + 1)
    dp = ph[1:] - ph[:-1]
    zdelp0 = ph[ngwdtop] - ph[0]
    t_top = -g * (ptau[ngwdtop + 1] - ptau[1]) * dp / (vph_b * zdelp0 ** 2)
    t_mid = -g * (ptau[2:klev + 2] - ptau[1:klev + 1]) / (dp * vph_b)
    wtop = _step(cfg.grfplm_pa - pf[1:klev + 1], cfg.top_width_pa, strict=True)  # PAPM1 < GRFPLM
    ztemp = wtop * t_top + (1.0 - wtop) * t_mid
    teu = (ulow * d1 - vlow * d2) * ztemp / dmod
    tev = (vlow * d1 + ulow * d2) * ztemp / dmod
    lim = cfg.gtenlim_m_s2

    def clip(x):
        """SIGN(MIN(ABS(x), GTENLIM), x)."""
        if cfg.tenlim_width_frac == 0.0:
            return jnp.sign(x) * jnp.minimum(jnp.abs(x), lim)
        return -jnp.sign(x) * smooth_max(-jnp.abs(x), -lim, 1.0 / (cfg.tenlim_width_frac * lim))

    above = kl < ngwdlim
    teu = jnp.where(above, clip(teu), teu)
    tev = jnp.where(above, clip(tev), tev)

    bl = kl >= kkenvh
    zcs = jnp.cos(psi[1:klev + 1]) ** 2
    zss = 1.0 - zcs
    zconb = cfg.gkwake * sig / (2.0 * std_s)
    zabsv = 0.5 * _ssqrt(u[1:klev + 1] ** 2 + v[1:klev + 1] ** 2)
    zzd1 = zb * zcs + zc * zss
    zratio = jnp.sqrt((zcs + gamma ** 2 * zss) / (gamma ** 2 * zcs + zss))
    if cfg.ratio_width == 0.0:
        fac = jnp.maximum(0.0, 2.0 - 1.0 / zratio)
    else:
        fac = positive_part_c1(2.0 - 1.0 / zratio, 1.0 / cfg.ratio_width)
    beta = fac * zconb * zdep[1:klev + 1] * zzd1 * zabsv
    teu = jnp.where(bl, 0.0, teu) * on
    tev = jnp.where(bl, 0.0, tev) * on
    beta = jnp.where(bl, beta, 0.0) * on
    ints = jnp.stack([kkcrith, kcrit, kkenvh, kknu, kknu2])
    dir_u = (ulow * d1 - vlow * d2) / dmod / vph_b
    dir_v = (vlow * d1 + ulow * d2) / dmod / vph_b
    return IFSSSOOutput(teu, tev, beta, ptau[1:klev + 2] * on, dir_u, dir_v, ulow, vlow, ints,
                        sat_levels * on)


def ifs_sso_tendencies(p_half, p_full, geo_full, T, u, v, std, gamma, theta, sigma,
                       nktopg, ngwdtop, ngwdlim, cfg: IFSSSOConfig = IFSSSOConfig(),
                       g: float = constants.g, r_d: float = constants.R_d,
                       c_pd: float = constants.c_pd):
    """GWDRAG on (ncol, klev) columns, levels top -> surface.

    Returns :class:`IFSSSOOutput`, one row per column.  nktopg / ngwdtop / ngwdlim
    are scalars (one reference coordinate) or (ncol,) arrays.
    """
    ncol = p_full.shape[0]
    lev = [jnp.broadcast_to(jnp.asarray(x), (ncol,)) for x in (nktopg, ngwdtop, ngwdlim)]
    fn = lambda *a: _column(*a, cfg=cfg, g=g, rd=r_d, cpd=c_pd)
    return jax.vmap(fn)(p_half, p_full, geo_full, T, u, v, std, gamma, theta, sigma, *lev)


def ifs_tofd_coefficient(geo_full, u, v, sigflt, cfg: IFSTOFDConfig = IFSTOFDConfig(),
                         g: float = constants.g):
    """VDFTOFDC: TOFD implicit coefficient (s-1) on (ncol, klev) full levels.

    geo_full: geopotential above the surface (m2 s-2); sigflt: standard
    deviation of the filtered subgrid orography (m), one per column.
    """
    fact2 = cfg.tofd_alpha * _TOFD_BETA * _TOFD_CMD * _TOFD_CORR * _TOFD_IC * _TOFD_FACT1
    coef = fact2 * sigflt[:, None] ** 2
    zz = geo_full / g
    usq = u ** 2 + v ** 2
    uabs = jnp.sqrt(_max(_REPDU2, usq, cfg.repdu2_width_frac * _REPDU2))
    zs = jnp.where(zz > 0.0, zz, 1.0)
    c = coef * uabs * jnp.exp(-(zs / _TOFD_IZ_M) ** _TOFD_EXP) * zs ** _TOFD_IN
    return jnp.where(zz > 0.0, c, 0.0) * _step(_TOFD_ZMAX_M - zz, cfg.zmax_width_m)
