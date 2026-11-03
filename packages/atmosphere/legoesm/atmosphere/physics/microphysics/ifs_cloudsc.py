"""IFS cloud microphysics CLOUDSC -- faithful port of OpenIFS cycle 49r1.

Oracle: OpenIFS ``ifs-source/arpifs/phys_ec/cloudsc.F90`` and
``cloud_supersatcheck.F90`` at commit 8f6f722f73 (cycle 49r1), with the
settings of 49r1 ``sucldp.F90``.  One call = one physics time step on a set
of columns: prognostic cloud fraction ``a``, cloud liquid, cloud ice, rain and
snow (``clv`` species order L, I, R, S as IFS NCLDQL..NCLDQS), returning the
local tendencies, the precipitation fraction and the sedimentation /
diagnostic fluxes exactly as CLOUDSC does.

Ported path (vote V19a, oracle_audit/ifs_port/vote_1103b/VOTES.md): the one
49r1 runs at its defaults -- the hard-wired IWARMRAIN=3 (KK2000 implicit),
IRAINACC=1, ITURBEROSION=3, IDEPICE=1 (Rotstayn), ISUBLICE=0, IDEPSNOW=1,
ISUBLSNOW=1, IVARFALL_RAIN=1, IVARFALL_ICE=0, IP_SNOW/ICE_ACCRETES_RAIN=1,
IP_RAIN_ACCRETES_SNOW/ICE=0, IFTLIQICE=1, and the namelist defaults
LAER*=F, LCLOUD_INHOMOG=F, LRAD_CLOUD_INHOMOG=F, SPP off, LMFDSNOW=T,
RMFADVW=0, cloud budgets (section 7) off.  No switch exposes the others.

Levels: index 0 = model top (IFS JK=1).  Arrays are ``[ncol, nlev]``.

Differentiability (user rule, vote V16a/V22a): every traced switch, clip,
min/max uses ``legoesm.core.smooth`` -- min/max through the C1 positive part
(exact outside a window of width ``w``), switches through the C1 cubic
smoothstep centred on the threshold -- with ``w = 0.01*|threshold|`` for a
nonzero threshold, ``0.01 K`` for every temperature threshold, and
``0.01*class scale`` for a zero threshold (fractions/ratios 1, water mixing
ratios ``RLMIN``).  ``hard=True`` (static) gives the oracle's exact
discontinuous forms and is used ONLY by the oracle-match test.  Discrete
inputs (convection flag, type) and level-index tests stay exact.  The
species order of the sink truncation (5.2.1) stays the hard sort (V17a).
Paired source/sink transfers are kept paired, so the smooth path conserves
water and liquid-water enthalpy exactly like the hard path.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core import smooth as _sm
from .._ifs_thermo import IFSThermo, dem_mixed, ew_mixed, foeeice, foeeliq
from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow

__physics_contract__ = {
    "summary": (
        "OpenIFS 49r1 CLOUDSC prognostic cloud scheme: cloud fraction, liquid, "
        "ice, rain, snow; detrainment, subsidence, erosion, sedimentation, "
        "warm rain (KK2000), deposition, melting/freezing, riming, accretion, "
        "evaporation/sublimation, implicit solver, supersaturation check."
    ),
    "inputs": {
        "t": "K", "q": "kg kg-1 (specific humidity)", "a": "1 (cloud fraction)",
        "clv": "kg kg-1 (L, I, R, S per moist mass)", "tend_*": "per s",
        "pap": "Pa (full)", "paph": "Pa (half, nlev+1)", "plude": "kg m-2 s-1",
        "pmfu/pmfd": "kg m-2 s-1", "plu": "kg kg-1",
    },
    "outputs": {
        "tend_t": "K s-1", "tend_q": "s-1", "tend_a": "s-1", "tend_cld": "s-1",
        "fplsl/fplsn": "kg m-2 s-1 (liquid / frozen precipitation flux, + down)",
        "fhpsl/fhpsn": "W m-2 (enthalpy flux of precipitation)",
    },
    "sign_convention": (
        "condensation warms (T + L/cp dq_c), precipitation flux positive "
        "downward, tendencies are d/dt of the local state"
    ),
    # moisture: column total water + surface precipitation; energy: liquid-water
    # enthalpy cp*T - Lv*(ql+qr) - Ls*(qi+qs) with the precipitation enthalpy flux
    "conserves": ["moisture", "energy"],
    "differentiable": True,
    "reference": "OpenIFS 8f6f722 (49r1) cloudsc.F90, cloud_supersatcheck.F90, sucldp.F90",
    "idealized_test": "tests/atmosphere/test_ifs_cloudsc_oracle.py",
}

# Species indices (yoecldp.F90 NCLDQL=1 ... NCLDQV=5, here 0-based).
NCLDQL, NCLDQI, NCLDQR, NCLDQS, NCLDQV = 0, 1, 2, 3, 4
NCLV = 5
_IPHASE = (1, 2, 1, 2, 0)          # cloudsc.F90:821-825
_IMELT = (NCLDQI, NCLDQR, NCLDQS, NCLDQR, -99)  # :831-835 (L->I, R->S, I->R, S->R)

# --- fixed numerics / fits of cloudsc.F90 (49r1), with line numbers ---
_ZEPSEC = 1.0e-14                  # :802
_ZEPSILON = 100.0 * float(np.finfo(np.float64).eps)   # :693 (fp64)
_QSAT_CAP = 0.5                    # :1093-1113 MIN(...,0.5)
_A_INCLOUD_FLOOR = 0.01            # :1321, :2751 MAX(ZA,0.01)
_TINY = 1.0e-30   # floor guard: untaken-branch denominators/powers; below every taken-branch threshold (ZEPSEC=1e-14)
_TW1, _TW2, _TW3, _TW4, _TW5 = 1329.31, 0.0074615, 0.85e5, 40.637, 275.0   # :472-476
_EROS_FAC = 0.333                  # :1727 Morcrette (2012)
_EROS_A_FAC = 0.5                  # :1737
_EROS_SHALLOW_FAC = 20.0           # :1706
_EROS_DEEP_FAC = 2.0               # :1708
_EROS_EIS_MAX = 10.0               # :1705 PEIS < 10
_SIG_RHC = 0.8                     # :2293 eta > 0.8
_SIG_RHC_SPAN = 0.2                # :2294
_RH_EVAP_CAP = 0.9                 # :2300
_FALLCORR_EXP = 0.4                # :1791 (0.4_JPRB)
_FALLCORR_EXP_SP = float(np.float32(0.4))   # :2320, :2980 default-real literal 0.4
_RAINFRAC_LIQ_SP = float(np.float32(0.8))   # :2889 default-real literal 0.8
_COVP_AMAX = 1.0 - 1.0e-6          # :1902
_KA_T0 = 273.0                     # :2332 conductivity fit
_KA_C = 393.0                      # :2332
_KA_T1 = 120.0                     # :2332
_EVAP_BETA_C = 0.15                # :2337
_EVAP_VENT_C = 0.78                # :2338
_DEPICE_DT = 5.0                   # :2436 T < RTT-5
_MEYERS_A = 12.96                  # :2440 Meyers et al. (1992)
_MEYERS_B = 0.639                  # :2440
_MEYERS_N0 = 1000.0                # :2440
_KAIR = 2.4e-2                     # :2446 conductivity of air
_DIFF_C = 2.21                     # :2447
_CVDS_A = 7.8                      # :2448
_CVDS_B = 8.87                     # :2449
_P666 = 0.666                      # :2448, :2459
_INEW_EXP = 1.5                    # :2459
_IN_REF = 15000.0                  # :2483
_RIMING_COVP_MIN = 0.01            # :2985
_FALLDIFF_DIV = 8.0                # :3099
_EFF_PSACR = 1.0                   # :3085
_EFF_PIACR = 0.1                   # :3172
_SUPSAT_TMIN = 160.0               # cloud_supersatcheck.F90:131
_RIME_POLY = (0.5, 2.0, 5.0)       # :3104, :3148, :3187 ((0.5*ZRAT+2)*ZRAT+5) collection-moment polynomial
_SUBL_BETA_EXP = 0.5777            # :3281 ZBETA1**0.5777 (snow sublimation)
_FZR_TEMP_CAP = 50.0               # :2908 MIN(RCL_FZRAB*(T-RTT), 50) EXP security
_DEP_LIQ_OVERLAP = 0.65            # :2724 ZTERM2 = 0.65*RCL_CONST6S (snow deposition)
_W_REL = 0.01                      # smooth-window width, relative to the threshold (user rule: 0.01)

# --- 49r1 SUCLDP derived particle-size-distribution constants (sucldp.F90 at
# 8f6f722, as dumped by the oracle: oracle_audit/ifs_port/cloudsc/out_base/
# settings_49r1.txt).  Fixed published fits, not tunables. ---
RCL_CONST1R = 1.3823007675795089
RCL_CONST2R = 2143.2299120517614
RCL_CONST3R = 0.63499999999999979
RCL_CONST4R = -0.20000000000000018
RCL_CONST5R = 8685252.9650821332
RCL_CONST6R = -4.7999999999999998
RCL_CONST7R = 952.92635997173136
RCL_LAM1R = 691.15046497451056
RCL_LAM2R = 0.55555555555555558
RCL_DR = 0.67000000000000004
RCL_X1R = 0.22000000000000000
RCL_X2R = 2.2000000000000002
RCL_CONST1S = 3.6231880115136998e-06
RCL_CONST2S = 6283185.3071795860
RCL_CONST3S = 596.99984758359983
RCL_CONST4S = 0.66666666666666663
RCL_CONST5S = 0.92116666666666669
RCL_CONST6S = 1.0000000948961185
RCL_CONST7S = 90363515.763510734
RCL_CONST8S = 1.1756666666666666
RCL_LAM1S = 276000.03003493568
RCL_LAM2S = 0.33333333333333331
RCL_X1S = 2000000.0000000000


__param_spec__ = {
    "IFSCloudscConfig": {
        "scheme_key": "atm.micro.IFSCloudscConfig",
        "excluded": {
            "ramid": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcldiff": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcldiff_convi": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "ramin": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rlmin": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rssicefactor": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rclcrit_sea": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rclcrit_land": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_kkaac": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_kkbac": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_kkaau": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_kkbauq": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_kkbaun": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_kk_cloud_num_sea": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_kk_cloud_num_land": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_inhomogaut": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_inhomogacc": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcovpmin": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rprecrhmax": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rpecons": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rvrfactor": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_ka273": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_cdenom1": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_cdenom2": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_cdenom3": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rlcritsnow": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rsnowlin1": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rsnowlin2": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_overlapliqice": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "ricinit": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcldtopcf": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rdepliqrefrate": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rdepliqrefdepth": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rthomo": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rtaumel": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_fzrab": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_apb1": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_apb2": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_apb3": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rcl_effrime": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rvice": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rvrain": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rvsnow": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rxfallspeedrain": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rdensref": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
            "rdenswat": "49r1 oracle setting (SUCLDP), fixed pending an IFS tuning plan (vote V24a)",
        },
        "params": {},
    },
}


class IFSCloudscConfig(NamedTuple):
    """YDECLDP / YDECUMF settings CLOUDSC reads; defaults = 49r1 SUCLDP
    (sucldp.F90 at 8f6f722) after su0phy RCLDTOPP=100 (oracle dump)."""
    # --- cloud fraction / erosion ---
    ramid: float = 0.8                   # critical RH [1]
    rcldiff: float = 6.0e-6              # erosion rate [s-1]
    rcldiff_convi: float = 10.0          # convective erosion factor [1]
    ramin: float = 1.0e-8                # minimum cloud fraction [1]
    rlmin: float = 1.0e-8                # minimum condensate [kg/kg]
    rssicefactor: float = 0.33333333333333331  # supersat-check ice cloud factor [1]
    # --- warm rain (Khairoutdinov and Kogan 2000) ---
    rclcrit_sea: float = 2.5e-4          # [kg/kg] (unused by IWARMRAIN=3; kept for audit)
    rclcrit_land: float = 5.5e-4
    rcl_kkaac: float = 67.0
    rcl_kkbac: float = 1.15
    rcl_kkaau: float = 1350.0
    rcl_kkbauq: float = 2.47
    rcl_kkbaun: float = -1.79
    rcl_kk_cloud_num_sea: float = 50.0   # [cm-3]
    rcl_kk_cloud_num_land: float = 300.0  # [cm-3]
    rcl_inhomogaut: float = 1.5
    rcl_inhomogacc: float = 3.0
    # --- precipitation fraction / evaporation ---
    rcovpmin: float = 0.1
    rprecrhmax: float = 0.7
    rpecons: float = 5.5472561985999302e-05
    rvrfactor: float = 5.0899999999999999e-03
    rcl_ka273: float = 2.4e-2
    rcl_cdenom1: float = 557000000000.0
    rcl_cdenom2: float = 103000000.0
    rcl_cdenom3: float = 204.0
    # --- ice ---
    rlcritsnow: float = 2.0e-5           # [kg/kg]
    rsnowlin1: float = 1.0e-3            # [s-1]
    rsnowlin2: float = 0.03              # [K-1]
    rcl_overlapliqice: float = 0.65
    ricinit: float = 1.0e-12             # RICEINIT [kg]
    rcldtopcf: float = 0.01
    rdepliqrefrate: float = 0.5
    rdepliqrefdepth: float = 500.0       # [m]
    rthomo: float = 235.16000000000003   # [K]
    # --- melting / freezing / riming ---
    rtaumel: float = 7200.0              # [s]
    rcl_fzrab: float = -0.66             # [K-1]
    rcl_apb1: float = 714000000000.0     # ice/snow deposition A+B fit
    rcl_apb2: float = 116000000.0
    rcl_apb3: float = 241.6
    rcl_effrime: float = 1.0
    # --- fall speeds ---
    rvice: float = 0.13                  # [m/s]
    rvrain: float = 4.0
    rvsnow: float = 1.0
    rxfallspeedrain: float = 1.0
    rdensref: float = 1.0                # [kg m-3]
    rdenswat: float = 1000.0             # [kg m-3]
    # --- convection coupling (yoecumf) ---
    lmfdsnow: bool = True                # sucumf.F90:203
    # --- numerics ---
    ncldtop: int = 15                    # first level (1-based) CLOUDSC processes


# ---------------------------------------------------------------------------
# smooth / hard helpers
# ---------------------------------------------------------------------------
class _Sm:
    """Hard (oracle) or smooth forms.  ``w`` is the transition width."""

    def __init__(self, hard: bool):
        self.hard = hard

    def pos(self, x, w):
        return jnp.maximum(x, 0.0) if self.hard else _sm.positive_part_c1(x, 1.0 / w)

    def max(self, a, b, w):
        return jnp.maximum(a, b) if self.hard else b + _sm.positive_part_c1(a - b, 1.0 / w)

    def min(self, a, b, w):
        return jnp.minimum(a, b) if self.hard else a - _sm.positive_part_c1(a - b, 1.0 / w)

    def abs(self, x, w):
        return jnp.abs(x) if self.hard else self.pos(x, w) + self.pos(-x, w)

    def _step(self, d, w, strict):
        """weight of (d > 0) [strict] or (d >= 0)."""
        if self.hard:
            return (d > 0) if strict else (d >= 0)
        return _sm.smoothstep_cubic(d / w + 0.5)

    def gt(self, a, b, w):
        return self._f(self._step(a - b, w, True), a)

    def ge(self, a, b, w):
        return self._f(self._step(a - b, w, False), a)

    def lt(self, a, b, w):
        return self._f(self._step(b - a, w, True), a)

    def le(self, a, b, w):
        return self._f(self._step(b - a, w, False), a)

    def neg1(self, x, w):
        """weight of (x < 0), one-sided: smooth only on [-w, 0], exactly 0 for x >= 0."""
        if self.hard:
            return (x < 0).astype(jnp.result_type(x))
        return _sm.smoothstep_cubic(-x / w)

    @staticmethod
    def _f(m, like):
        return m.astype(jnp.result_type(like)) if m.dtype == jnp.bool_ else m

    def sel(self, m, x, y):
        if self.hard:
            return jnp.where(m > 0.5, x, y)
        return m * x + (1.0 - m) * y


def _safe(x):
    """Inactive-operand substitute (V22b): x where x > _TINY, else 1.0.  Every caller is
    gated by a threshold >= ZEPSEC on x (or a value derived from it), so the active branch
    sees x unchanged and the inactive one stays finite in value and gradient (fp32 too)."""
    return jnp.where(x > _TINY, x, 1.0)


def _safe_gate(x, thr):
    """Inactive-operand substitute for a quantity gated by ``x > thr`` on the
    _wt(thr) window: x where its smooth weight can be nonzero (x > thr - w/2, the
    window's lower edge), else 1.0 (weight identically 0 there, so the substitute
    never reaches an output; keeps fp32 power laws finite, review r4)."""
    return jnp.where(x > thr - 0.5 * _wt(thr), x, 1.0)


def _wfloor(b, floor):
    """Window width for a traced bound b: ~0.01|b|, never below ``floor``; smooth in b
    (VC14b: hypot, not max, so the width has no derivative jump at 0.01|b| = floor)."""
    return jnp.hypot(_W_REL * b, floor)


def _wt(thr):
    return _W_REL * abs(thr)


_W_T = _W_REL    # every temperature threshold [K]
_W_F = _W_REL    # zero threshold, fractions / ratios


def _consts(th):
    if not isinstance(th, IFSThermo):
        raise TypeError("ifs_cloudsc: th must be an _ifs_thermo.IFSThermo bundle")
    return th


def _foealfa(sm, th, t):
    tt = sm.max(th.RTICE, sm.min(th.RTWAT, t, _W_T), _W_T)
    return sm.min(1.0, ((tt - th.RTICE) * th.RTWAT_RTICE_R) ** 2, _W_F)


def _foedelta(sm, th, t):
    # MAX(0, SIGN(1, T-RTT)) = 1 for T >= RTT
    return sm.ge(t, th.RTT, _W_T)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
class CloudscInputs(NamedTuple):
    t: jax.Array          # [ncol, nlev] T at start of callpar
    q: jax.Array          # specific humidity
    a: jax.Array          # cloud fraction
    clv: jax.Array        # [ncol, nlev, 5] L, I, R, S (slot 4 vapour, unused)
    tend_t: jax.Array     # cumulative tendencies (PTENDENCY_CML_*)
    tend_q: jax.Array
    tend_a: jax.Array
    tend_cld: jax.Array   # [ncol, nlev, 5]
    pvfl: jax.Array       # liquid / ice tendency from VDF (diagnostic fluxes only)
    pvfi: jax.Array
    pap: jax.Array        # full-level pressure
    paph: jax.Array       # [ncol, nlev+1]
    plsm: jax.Array       # [ncol] land fraction
    ldcum: jax.Array      # [ncol] bool
    ktype: jax.Array      # [ncol] int
    peis: jax.Array       # [ncol]
    plu: jax.Array
    plude: jax.Array
    pludeli: jax.Array    # [ncol, nlev, 4]
    psnde: jax.Array      # [ncol, nlev, 2] rain, snow
    pmfu: jax.Array
    pmfd: jax.Array


class CloudscOutputs(NamedTuple):
    tend_t: jax.Array
    tend_q: jax.Array
    tend_a: jax.Array
    tend_cld: jax.Array   # [ncol, nlev, 4] L, I, R, S
    plude: jax.Array      # PLUDE as CLOUDSC leaves it (INOUT)
    covptot: jax.Array    # precipitation fraction (0 above NCLDTOP; IFS: undefined)
    prainfrac_toprfz: jax.Array
    fsqlf: jax.Array      # [ncol, nlev+1] diagnostic fluxes (cloudsc.F90 section 8)
    fsqif: jax.Array
    fcqnng: jax.Array
    fcqlng: jax.Array
    fsqrf: jax.Array
    fsqsf: jax.Array
    fcqrng: jax.Array
    fcqsng: jax.Array
    fsqltur: jax.Array
    fsqitur: jax.Array
    fplsl: jax.Array
    fplsn: jax.Array
    fhpsl: jax.Array
    fhpsn: jax.Array


def cloudsc(cfg: IFSCloudscConfig, th, x: CloudscInputs, ptsphy, *, hard: bool = False,
            stages: bool = False):
    """One CLOUDSC call.  ``hard`` (static) = oracle forms; ``stages`` returns
    ``(outputs, per-level stage snapshots)`` for the stage-match tests."""
    th = _consts(th)
    sm = _Sm(bool(hard))
    ncol, nlev = x.t.shape
    k0 = int(cfg.ncldtop) - 1
    if not 1 <= k0 < nlev:
        raise ValueError(f"ifs_cloudsc: ncldtop {cfg.ncldtop} must be in [2, nlev]")
    if not (cfg.rvice > 0 and cfg.rvrain > 0 and cfg.rvsnow > 0):
        raise ValueError("ifs_cloudsc: the ported path has ice, rain and snow all falling")
    dt = jnp.asarray(ptsphy, x.t.dtype)
    zqtmst = 1.0 / dt
    zrdcp = th.RD / th.RCPD
    zrg_r = 1.0 / th.RG
    zrldcp = 1.0 / (th.RALSDCP - th.RALVDCP)
    wq = _W_REL * cfg.rlmin            # zero-threshold width for water mixing ratios

    def wb(b):                       # limit against a traced bound: 0.01*|bound|, floor wq (V26a)
        return _wfloor(b, wq)

    # ---------------- section 1: initial values (cloudsc.F90:963-1115) -------
    ztp1 = x.t + dt * x.tend_t
    zqx = [x.clv[..., m] + dt * x.tend_cld[..., m] for m in range(4)]
    zqx.append(x.q + dt * x.tend_q)
    zqx0 = list(zqx)
    za = x.a + dt * x.tend_a
    zaorig = za

    zfoealfa = _foealfa(sm, th, ztp1)
    zfoeewm = sm.min(ew_mixed(th, zfoealfa, ztp1) / x.pap, _QSAT_CAP, _wt(_QSAT_CAP))
    zqsmix = zfoeewm / (1.0 - th.RETV * zfoeewm)
    e_liq = foeeliq(th, ztp1)
    e_ice = foeeice(th, ztp1)
    zalfa = _foedelta(sm, th, ztp1)
    zfoeew = sm.min((zalfa * e_liq + (1.0 - zalfa) * e_ice) / x.pap, _QSAT_CAP, _wt(_QSAT_CAP))
    zfoeew = sm.min(_QSAT_CAP, zfoeew, _wt(_QSAT_CAP))
    zqsice = zfoeew / (1.0 - th.RETV * zfoeew)
    zfoeeliqt = sm.min(e_liq / x.pap, _QSAT_CAP, _wt(_QSAT_CAP))
    zqsliq = zfoeeliqt / (1.0 - th.RETV * zfoeeliqt)

    # ---------------- section 2: tidy up (:1127-1199) ------------------------
    tlq = jnp.zeros_like(x.t)
    tlt = jnp.zeros_like(x.t)
    zlneg = [jnp.zeros_like(x.t) for _ in range(4)]
    w = 1.0 - ((1.0 - sm.lt(zqx[NCLDQL] + zqx[NCLDQI], cfg.rlmin, _wt(cfg.rlmin)))
               * (1.0 - sm.lt(za, cfg.ramin, _wt(cfg.ramin))))
    for m, lat in ((NCLDQL, th.RALVDCP), (NCLDQI, th.RALSDCP)):
        zlneg[m] = zlneg[m] + w * zqx[m]
        zqadj = zqx[m] * zqtmst
        tlq = tlq + w * zqadj
        tlt = tlt - w * (lat * zqadj)
        zqx[NCLDQV] = zqx[NCLDQV] + w * zqx[m]
        zqx[m] = sm.sel(w, 0.0 * zqx[m], zqx[m]) if hard else (1.0 - w) * zqx[m]
    za = sm.sel(w, 0.0 * za, za) if hard else (1.0 - w) * za
    for m in range(4):
        w = sm.lt(zqx[m], cfg.rlmin, _wt(cfg.rlmin))
        zlneg[m] = zlneg[m] + w * zqx[m]
        zqadj = zqx[m] * zqtmst
        tlq = tlq + w * zqadj
        lat = th.RALVDCP if _IPHASE[m] == 1 else th.RALSDCP
        tlt = tlt - w * (lat * zqadj)
        zqx[NCLDQV] = zqx[NCLDQV] + w * zqx[m]
        zqx[m] = sm.sel(w, 0.0 * zqx[m], zqx[m]) if hard else (1.0 - w) * zqx[m]
    za = sm.max(0.0, sm.min(1.0, za, _W_F), _W_F)
    zli = zqx[NCLDQL] + zqx[NCLDQI]
    wli = sm.gt(zli, cfg.rlmin, _wt(cfg.rlmin))
    zliqfrac = sm.sel(wli, zqx[NCLDQL] / _safe_gate(zli, cfg.rlmin), 0.0)
    zicefrac = sm.sel(wli, 1.0 - zliqfrac, 0.0)

    # ---------------- per-level scan (:1219-4128) ----------------------------
    def up(v):    # value at level k-1 (level k0-1 exists, k0 >= 1)
        return jnp.concatenate([v[:, :1], v[:, :-1]], axis=1)

    def dn(v):    # value at level k+1 (last level duplicated; masked by is_last)
        return jnp.concatenate([v[:, 1:], v[:, -1:]], axis=1)

    zqx_s = jnp.stack(zqx, axis=-1)
    zqx0_s = jnp.stack(zqx0, axis=-1)
    lev = dict(
        ztp1=ztp1, ztp1_up=up(ztp1), za=za, za_up=up(za), zaorig=zaorig,
        zqx=zqx_s, zqr_up=up(zqx[NCLDQR]), zqs_up=up(zqx[NCLDQS]), zqx0=zqx0_s,
        zli=zli, zliqfrac=zliqfrac, zicefrac=zicefrac, zfoealfa=zfoealfa,
        zfoeewm=zfoeewm, zqsmix=zqsmix, zfoeew=zfoeew, zqsice=zqsice,
        zfoeeliqt=zfoeeliqt, zqsliq=zqsliq, foeeliq=e_liq, foeeice=e_ice,
        pap=x.pap, pap_up=up(x.pap), paph=x.paph[:, :-1], paph_dn=x.paph[:, 1:],
        dp_up=up(x.paph[:, 1:] - x.paph[:, :-1]), dp_dn=dn(x.paph[:, 1:] - x.paph[:, :-1]),
        pmfu=x.pmfu, pmfd=x.pmfd, pmfu_dn=dn(x.pmfu), pmfd_dn=dn(x.pmfd),
        plude=x.plude, plu_dn=dn(x.plu), pludeli=x.pludeli, psnde=x.psnde,
        pt=x.t, pq=x.q, pa=x.a, tct=x.tend_t, tcq=x.tend_q, tca=x.tend_a,
        tlt=tlt, tlq=tlq,
    )
    lev = {k: jnp.moveaxis(v, 1, 0)[k0:] for k, v in lev.items()}
    idx = jnp.arange(k0, nlev)
    lev["is_first"] = idx == k0
    lev["is_last"] = idx == nlev - 1
    psurf = x.paph[:, -1]

    # PLSM > 0.5 -> land values; geography input, exact in both modes (V27a)
    zconst = jnp.where(x.plsm > 0.5, cfg.rcl_kk_cloud_num_land, cfg.rcl_kk_cloud_num_sea)
    ktype = x.ktype
    zadvw = jnp.ones((ncol,), x.t.dtype)          # RMFADVW = 0 (sucumf.F90:228)
    zadvwd = jnp.ones((ncol,), x.t.dtype)

    zeros = jnp.zeros((ncol,), x.t.dtype)
    carry0 = dict(
        zqxnm1=jnp.zeros((ncol, NCLV), x.t.dtype), zanewm1=zeros, zcovptot=zeros,
        zcovpmax=zeros, zcldtopdist=zeros, prainfrac=zeros,
        llrainliq=jnp.ones((ncol,), x.t.dtype), zpfplsx=jnp.zeros((ncol, NCLV), x.t.dtype),
    )


    def level(c, v):
        snaps = {}

        def snap(name, **kw):
            if stages:
                snaps[name] = kw

        nf = jnp.logical_not(v["is_first"]).astype(x.t.dtype)   # JK > NCLDTOP
        nl = jnp.logical_not(v["is_last"]).astype(x.t.dtype)    # JK < KLEV
        T = v["ztp1"]
        zqx = v["zqx"]
        zqv = zqx[:, NCLDQV]
        za_k = v["za"]
        zqsmix, zqsliq, zqsice = v["zqsmix"], v["zqsliq"], v["zqsice"]
        pap = v["pap"]
        zqxnm1 = c["zqxnm1"]
        zanewm1 = c["zanewm1"]
        zcovptot = c["zcovptot"]
        zcovpmax = c["zcovpmax"]
        zqxfg = [zqx[:, m] for m in range(NCLV)]
        zsolqa = jnp.zeros((ncol, NCLV, NCLV), x.t.dtype)
        zsolqb = jnp.zeros((ncol, NCLV, NCLV), x.t.dtype)
        zsolab = zeros
        zsolac = zeros
        zconvsrce = [zeros] * NCLV
        zconvsink = [zeros] * NCLV
        zfallsrce = [zeros] * NCLV
        zfallsink = [zeros] * NCLV
        zqpretot = zeros
        zlfinalsum = zeros

        def add(mat, i, j, val):
            return mat.at[:, i, j].add(val)

        def pair(mat, i, j, val):          # source of i, sink of j (and the mirror)
            return mat.at[:, i, j].add(val).at[:, j, i].add(-val)

        zdp = v["paph_dn"] - v["paph"]
        zgdp = th.RG / zdp
        zrho = pap / (th.RD * T)
        zdtgdp = dt * zgdp
        zrdtgdp = zdp * (1.0 / (dt * th.RG))

        zfacw = th.R5LES / ((T - th.R4LES) ** 2)
        zcor = 1.0 / (1.0 - th.RETV * v["zfoeeliqt"])
        zdqsliqdt = zfacw * zcor * zqsliq
        zcorqsliq = 1.0 + th.RALVDCP * zdqsliqdt
        zfaci = th.R5IES / ((T - th.R4IES) ** 2)
        zcor = 1.0 / (1.0 - th.RETV * v["zfoeew"])
        zdqsicedt = zfaci * zcor * zqsice
        zcorqsice = 1.0 + th.RALSDCP * zdqsicedt
        zalfaw = v["zfoealfa"]
        zfac = zalfaw * zfacw + (1.0 - zalfaw) * zfaci
        zcor = 1.0 / (1.0 - th.RETV * v["zfoeewm"])
        zdqsmixdt = zfac * zcor * zqsmix
        zcorqsmix = 1.0 + (zalfaw * th.RALVDCP + (1.0 - zalfaw) * th.RALSDCP) * zdqsmixdt
        zevaplimmix = sm.pos((zqsmix - zqv) / zcorqsmix, wb(zqsmix))
        zevaplimliq = sm.pos((zqsliq - zqv) / zcorqsliq, wb(zqsliq))
        zevaplimice = sm.pos((zqsice - zqv) / zcorqsice, wb(zqsice))
        ztmpa = 1.0 / sm.max(za_k, _A_INCLOUD_FLOOR, _wt(_A_INCLOUD_FLOOR))
        zliqcld = zqx[:, NCLDQL] * ztmpa
        zicecld = zqx[:, NCLDQI] * ztmpa
        zlicld = zliqcld + zicecld

        for m in (NCLDQL, NCLDQI):     # :1332-1344 (overwrite, matrix still zero)
            wv = sm.lt(zqx[:, m], cfg.rlmin, _wt(cfg.rlmin))
            zsolqa = zsolqa.at[:, NCLDQV, m].set(wv * zqx[:, m]).at[:, m, NCLDQV].set(wv * (-zqx[:, m]))
        snap("s1", zsolqa=zsolqa)

        # ---- 3.2 detrainment (:1362-1398), JK < KLEV ----
        plude_s = v["plude"] * zdtgdp
        plu_dn = v["plu_dn"]
        wdet = nl * (x.ldcum.astype(T.dtype) * sm.gt(plude_s, _ZEPSEC, _wt(_ZEPSEC))
                     * sm.gt(plu_dn, _ZEPSEC, _wt(_ZEPSEC)))
        zsolac = zsolac + wdet * (plude_s / _safe_gate(plu_dn, _ZEPSEC))
        for m, sl in ((NCLDQL, 0), (NCLDQI, 1)):
            src = v["pludeli"][:, sl] * zdtgdp
            zconvsrce[m] = wdet * src
            zsolqa = add(zsolqa, m, m, wdet * src)
        if hard:
            plude = jnp.where(nl > 0.5, jnp.where(wdet > 0.5, plude_s, 0.0), v["plude"])
        else:
            plude = nl * (wdet * plude_s) + (1.0 - nl) * v["plude"]
        if cfg.lmfdsnow:
            for m, sl in ((NCLDQR, 0), (NCLDQS, 1)):
                src = v["psnde"][:, sl] * zdtgdp
                zconvsrce[m] = nl * src
                zsolqa = add(zsolqa, m, m, nl * src)
        snap("s32", zsolqa=zsolqa, zsolac=zsolac, plude=plude)

        # ---- 3.3 subsidence (:1419-1514) ----
        # max(0, M*dt*g/dp) = max(M, 0)*dt*g/dp; the smooth form acts on the interface mass
        # flux M with ONE width per interface (V25a) so the source here and the sink of the
        # level above stay paired (water conserved in smooth mode).
        def _mflux_w(dpa, dpb):
            return _W_F * (2.0 * dpa * dpb / (dpa + dpb)) / (th.RG * dt)
        zmf = sm.pos(v["pmfu"] * zadvw + v["pmfd"] * zadvwd, _mflux_w(v["dp_up"], zdp)) * zdtgdp
        zacust = zmf * zanewm1
        zlcust = {m: zmf * zqxnm1[:, m] for m in (NCLDQL, NCLDQI)}
        for m in (NCLDQL, NCLDQI):
            zconvsrce[m] = zconvsrce[m] + nf * zlcust[m]
        zdtdp = zrdcp * 0.5 * (v["ztp1_up"] + T) / v["paph"]
        wcap = (pap - v["pap_up"]) / dt
        zwtot = sm.min(th.RG * (v["pmfu"] * zadvw + v["pmfd"] * zadvwd), wcap,
                       _wfloor(wcap, _TINY))
        zdtforc = zdtdp * dt * zwtot
        zdqs = zanewm1 * zdtforc * zdqsmixdt
        for m in (NCLDQL, NCLDQI):
            zlfinal = sm.pos(zlcust[m] - zdqs, wb(zdqs))
            zevap = sm.min(zlcust[m] - zlfinal, zevaplimmix, wb(zevaplimmix))
            zlfinal = zlcust[m] - zevap
            zlfinalsum = zlfinalsum + nf * zlfinal
            zsolqa = add(zsolqa, m, m, nf * zlcust[m])
            zsolqa = add(zsolqa, NCLDQV, m, nf * zevap)
            zsolqa = add(zsolqa, m, NCLDQV, -(nf * zevap))
        zsolac = zsolac + nf * zacust
        zsolac = zsolac - nf * sm.lt(zlfinalsum, _ZEPSEC, _wt(_ZEPSEC)) * zacust
        zmfdn = sm.pos(v["pmfu_dn"] * zadvw + v["pmfd_dn"] * zadvwd, _mflux_w(zdp, v["dp_dn"])) * zdtgdp
        zsolab = zsolab + nl * zmfdn
        for m in (NCLDQL, NCLDQI):
            zsolqb = add(zsolqb, m, m, nl * zmfdn)
            zconvsink[m] = nl * zmfdn
        snap("s33", zsolqa=zsolqa, zsolqb=zsolqb, zsolac=zsolac, zsolab=zsolab)

        # ---- 3.4 erosion, ITURBEROSION = 3 (:1675-1754) ----
        zldifdt = cfg.rcldiff * dt
        kt = x.ktype
        wp = sm.gt(plude, _ZEPSEC, _wt(_ZEPSEC))
        w1 = (kt >= 2).astype(T.dtype) * wp * sm.lt(x.peis, _EROS_EIS_MAX, _wt(_EROS_EIS_MAX))
        w2 = (1.0 - w1) * (kt == 1).astype(T.dtype) * wp
        zldifdt = sm.sel(w1, _EROS_SHALLOW_FAC * cfg.rcldiff_convi * zldifdt,
                         sm.sel(w2, _EROS_DEEP_FAC * cfg.rcldiff_convi * zldifdt, zldifdt))
        zli = v["zli"]
        weli = sm.gt(zli, _ZEPSEC, _wt(_ZEPSEC))
        zqe = (zqv - za_k * zqsmix) / sm.max(_ZEPSEC, 1.0 - za_k, _wt(_ZEPSEC))
        ze = sm.pos(zqsmix - zqe, wb(zqsmix))
        zleros = _EROS_FAC * za_k * (1.0 - za_k) * zldifdt * ze
        zleros = sm.min(zleros, zevaplimmix, wb(zevaplimmix))
        zleros = sm.min(zleros, zli, wb(zli))
        zaeros = _EROS_A_FAC * zleros / jnp.where(zli > _ZEPSEC - 0.5 * _wt(_ZEPSEC), zlicld, 1.0)   # gate weli on zli
        zsolac = zsolac - weli * zaeros
        zsolqa = add(zsolqa, NCLDQV, NCLDQL, weli * (v["zliqfrac"] * zleros))
        zsolqa = add(zsolqa, NCLDQL, NCLDQV, -(weli * (v["zliqfrac"] * zleros)))
        zsolqa = add(zsolqa, NCLDQV, NCLDQI, weli * (v["zicefrac"] * zleros))
        zsolqa = add(zsolqa, NCLDQI, NCLDQV, -(weli * (v["zicefrac"] * zleros)))
        snap("s34", zsolqa=zsolqa, zsolac=zsolac)

        # ---- 4.1 fall speeds and sedimentation (:1783-1877) ----
        zfallcorr = (cfg.rdensref / zrho) ** _FALLCORR_EXP
        ztemp = 0.5 * (zqx[:, NCLDQR] + zqxnm1[:, NCLDQR])
        zrlambda = safe_pow(zrho * ztemp / RCL_LAM1R, RCL_LAM2R)
        zfallspeed = {NCLDQI: cfg.rvice + 0.0 * T, NCLDQS: cfg.rvsnow + 0.0 * T,
                      NCLDQR: cfg.rxfallspeedrain * zfallcorr * RCL_CONST7R * safe_pow(zrlambda, RCL_DR)}
        for m in (NCLDQI, NCLDQR, NCLDQS):
            zfallsrce[m] = nf * (c["zpfplsx"][:, m] * zdtgdp)
            zsolqa = add(zsolqa, m, m, zfallsrce[m])
            zqxfg[m] = zqxfg[m] + zfallsrce[m]
            zqpretot = zqpretot + nf * zqxfg[m]
            zfallsink[m] = zdtgdp * zrho * zfallspeed[m]
        snap("s41", zsolqa=zsolqa, zqxfg=jnp.stack(zqxfg, -1), zqpretot=zqpretot,
             zfallsink=jnp.stack(zfallsink, -1))

        # ---- 4.2 precipitation fraction (:1898-1919) ----
        wpr = sm.gt(zqpretot, _ZEPSEC, _wt(_ZEPSEC))
        za_up = v["za_up"]
        cov = 1.0 - ((1.0 - zcovptot) * (1.0 - sm.max(za_k, za_up, _W_F))
                     / (1.0 - sm.min(za_up, _COVP_AMAX, _W_REL * (1.0 - _COVP_AMAX))))  # VC12a: gap-scale window
        cov = sm.max(cov, cfg.rcovpmin, _wt(cfg.rcovpmin))
        covs = _safe(cov)
        zcovptot = sm.sel(wpr, cov, 0.0)
        zcovpclr = sm.sel(wpr, sm.pos(cov - za_k, _W_F), 0.0)
        zraincld = sm.sel(wpr, 0.5 * (zqx[:, NCLDQR] + v["zqr_up"]) / covs, 0.0)
        zraincldm1 = sm.sel(wpr, 0.5 * (zqx[:, NCLDQR] + zqxnm1[:, NCLDQR]) / covs, 0.0)
        zsnowcld = sm.sel(wpr, 0.5 * (zqx[:, NCLDQS] + v["zqs_up"]) / covs, 0.0)
        zcovpmax = sm.sel(wpr, sm.max(cov, zcovpmax, _W_F), 0.0)
        snap("s42", zcovptot=zcovptot, zcovpclr=zcovpclr, zraincld=zraincld,
             zraincldm1=zraincldm1, zsnowcld=zsnowcld, zcovpmax=zcovpmax)

        # ---- 4.3 warm rain, IWARMRAIN = 3, IRAINACC = 1 (:1930-2259) ----
        wlc = sm.gt(zliqcld, _ZEPSEC, _wt(_ZEPSEC))
        zrainaut = (cfg.rcl_inhomogaut * dt * cfg.rcl_kkaau * safe_pow(zliqcld, cfg.rcl_kkbauq - 1.0)
                    * zconst ** cfg.rcl_kkbaun)
        zsolqb = add(zsolqb, NCLDQR, NCLDQL, wlc * zrainaut)
        wrc = sm.gt(zraincld, _ZEPSEC, _wt(_ZEPSEC))
        zrainacc = wrc * (cfg.rcl_inhomogacc * dt * cfg.rcl_kkaac
                          * safe_pow(zliqcld, cfg.rcl_kkbac - 1.0) * safe_pow(zraincld, cfg.rcl_kkbac))
        zsolqb = add(zsolqb, NCLDQR, NCLDQL, wlc * zrainacc)
        snap("s43", zsolqb=zsolqb)

        # ---- 4.4 rain evaporation (:2271-2378) ----
        zzrh = cfg.rprecrhmax + (1.0 - cfg.rprecrhmax) * zcovpmax / sm.max(_ZEPSEC, 1.0 - za_k, _wt(_ZEPSEC))
        zzrh = sm.min(sm.max(zzrh, cfg.rprecrhmax, _wt(cfg.rprecrhmax)), 1.0, _wt(1.0))
        zsigk = pap / psurf
        wsg = sm.gt(zsigk, _SIG_RHC, _wt(_SIG_RHC))
        zrhc = sm.sel(wsg, cfg.ramid + (1.0 - cfg.ramid) * ((zsigk - _SIG_RHC) / _SIG_RHC_SPAN) ** 2, cfg.ramid)
        zzrh = sm.min(zrhc, zzrh, _W_F)
        zzrh = sm.min(_RH_EVAP_CAP, zzrh, _wt(_RH_EVAP_CAP))
        zqe = sm.max(0.0, sm.min(zqv, zqsliq, wb(zqsliq)), wq)
        llo1 = (sm.gt(zcovpclr, _ZEPSEC, _wt(_ZEPSEC)) * sm.gt(zraincldm1, _ZEPSEC, _wt(_ZEPSEC))
                * sm.lt(zqe, zzrh * zqsliq, wb(zzrh * zqsliq)))
        zpreclr = _safe_gate(zraincldm1, _ZEPSEC)
        zfallcorr_sp = (cfg.rdensref / zrho) ** _FALLCORR_EXP_SP
        zesatliq = th.RV / th.RD * v["foeeliq"]
        zlambda = (RCL_LAM1R / (zrho * zpreclr)) ** RCL_LAM2R
        zevap_denom = (cfg.rcl_cdenom1 * zesatliq - cfg.rcl_cdenom2 * T * zesatliq
                       + cfg.rcl_cdenom3 * T ** 3.0 * pap)
        zcorr2 = (T / _KA_T0) ** 1.5 * _KA_C / (T + _KA_T1)
        zsubsat = sm.pos(zzrh * zqsliq - zqe, wb(zzrh * zqsliq))
        zbeta = ((_EVAP_BETA_C / zqsliq) * T ** 2.0 * zesatliq * RCL_CONST1R * (zcorr2 / zevap_denom)
                 * (_EVAP_VENT_C / (zlambda ** RCL_CONST4R) + RCL_CONST2R * (zrho * zfallcorr_sp) ** 0.5
                    / (zcorr2 ** 0.5 * zlambda ** RCL_CONST3R)))
        zdenom = 1.0 + zbeta * dt * zcorqsliq
        zdpevap = zcovpclr * zbeta * dt * zsubsat / zdenom
        zevap = llo1 * sm.min(zdpevap, zqxfg[NCLDQR], wb(zqxfg[NCLDQR]))
        zsolqa = add(zsolqa, NCLDQV, NCLDQR, zevap)
        zsolqa = add(zsolqa, NCLDQR, NCLDQV, -zevap)
        zqxfg[NCLDQR] = zqxfg[NCLDQR] - zevap
        snap("s44", zsolqa=zsolqa, zqxfg=jnp.stack(zqxfg, -1))

        # ---- 4.5 ice deposition, IDEPICE = 1 (:2401-2499) ----
        wct = sm.lt(za_up, cfg.rcldtopcf, _wt(cfg.rcldtopcf)) * sm.ge(za_k, cfg.rcldtopcf, _wt(cfg.rcldtopcf))
        zcldtopdist = sm.sel(wct, 0.0 * zdp, c["zcldtopdist"] + zdp / (zrho * th.RG))
        zsupersatice = (zqsliq - zqsice) / zqsice
        wdep = sm.lt(T, th.RTT - _DEPICE_DT, _W_T) * sm.gt(zqxfg[NCLDQL], cfg.rlmin, _wt(cfg.rlmin))
        zvpice = v["foeeice"] * th.RV / th.RD
        zicenuclei = _MEYERS_N0 * jnp.exp(_MEYERS_A * zsupersatice - _MEYERS_B)
        zadd = th.RLSTT * (th.RLSTT / (th.RV * T) - 1.0) / (_KAIR * T)
        zbdd = th.RV * T * pap / (_DIFF_C * zvpice)
        zcvds = _CVDS_A * (zicenuclei / zrho) ** _P666 * zsupersatice / (_CVDS_B * (zadd + zbdd))
        zice0 = sm.max(zicecld, zicenuclei * cfg.ricinit / zrho, wb(zicenuclei * cfg.ricinit / zrho))
        zinew = safe_pow(_P666 * zcvds * dt + safe_pow(zice0, _P666), _INEW_EXP)
        zdepos = sm.pos(cfg.rcl_overlapliqice * za_k * (zinew - zice0), wb(cfg.rcl_overlapliqice * za_k * zice0))
        zdepos = sm.min(zdepos, zqxfg[NCLDQL], wb(zqxfg[NCLDQL]))
        zinfactor = sm.min(zicenuclei / _IN_REF, 1.0, _wt(1.0))
        zdepos = zdepos * sm.min(zinfactor + (1.0 - zinfactor)
                                 * (cfg.rdepliqrefrate + zcldtopdist / cfg.rdepliqrefdepth), 1.0, _wt(1.0))
        zdepos = wdep * zdepos
        zsolqa = add(zsolqa, NCLDQI, NCLDQL, zdepos)
        zsolqa = add(zsolqa, NCLDQL, NCLDQI, -zdepos)
        zqxfg[NCLDQI] = zqxfg[NCLDQI] + zdepos
        zqxfg[NCLDQL] = zqxfg[NCLDQL] - zdepos
        snap("s45", zsolqa=zsolqa, zqxfg=jnp.stack(zqxfg, -1), zcldtopdist=zcldtopdist)

        # ---- 4.7 snow deposition, IDEPSNOW = 1 (:2698-2745) ----
        zss_s = (zqv - zqsmix) / zqsmix
        zss = sm.pos(zss_s, _W_F)
        # gate on the SIGNED supersaturation (MAX(s,0) > 0 <=> s > 0), V29a
        wsd = (sm.lt(T, th.RTT, _W_T) * sm.gt(zsnowcld, _ZEPSEC, _wt(_ZEPSEC))
               * sm.gt(zss_s, 0.0, _W_F))
        zaplusb = cfg.rcl_apb1 * zvpice - cfg.rcl_apb2 * zvpice * T + pap * cfg.rcl_apb3 * T ** 3
        zcorrfac = (1.0 / zrho) ** 0.5
        zcorrfac2 = ((T / _KA_T0) ** 1.5) * (_KA_C / (T + _KA_T1))
        zpr02 = zrho * _safe_gate(zsnowcld, _ZEPSEC) * RCL_CONST1S / (1.0 * 1.0)
        zterm1 = zss * T ** 2 * zvpice * zcorrfac2 * 1.0 * RCL_CONST2S * 1.0 / (zrho * zaplusb)
        zterm2 = (_DEP_LIQ_OVERLAP * RCL_CONST6S * zpr02 ** RCL_CONST4S + RCL_CONST3S * zcorrfac ** 0.5 * zrho ** 0.5
                  * zpr02 ** RCL_CONST5S / zcorrfac2 ** 0.5)
        zpsdep = sm.pos(zcovptot * zterm1 * zterm2 * dt, wq)
        sup = sm.pos(zqv - zqsmix, wb(zqsmix))      # = ZQX-ZQSMIX on the active branch (V29a)
        zpsdep = wsd * (sm.min(zpsdep, sup, wb(sup)) / zcorqsmix)
        zsolqa = add(zsolqa, NCLDQS, NCLDQV, zpsdep)
        zsolqa = add(zsolqa, NCLDQV, NCLDQS, -zpsdep)
        snap("s47", zsolqa=zsolqa)

        ztmpa = 1.0 / sm.max(za_k, _A_INCLOUD_FLOOR, _wt(_A_INCLOUD_FLOOR))
        zliqcld = zqxfg[NCLDQL] * ztmpa
        zicecld = zqxfg[NCLDQI] * ztmpa

        # ---- 4.8 ice -> snow (:2769-2800) ----
        wis = sm.le(T, th.RTT, _W_T) * sm.gt(zicecld, _ZEPSEC, _wt(_ZEPSEC))
        zzco = dt * cfg.rsnowlin1 * jnp.exp(cfg.rsnowlin2 * (T - th.RTT))
        ztemp = -1.0 * (zicecld / cfg.rlcritsnow) ** 2     # <= 0: the "ZTEMP < 50" guard always holds
        zsnowaut = wis * (zzco * (1.0 - jnp.exp(ztemp)))
        zsolqb = add(zsolqb, NCLDQS, NCLDQI, zsnowaut)
        snap("s48", zsolqb=zsolqb)

        # ---- 4.12 melting of ice and snow (:2815-2862) ----
        zicetot = zqxfg[NCLDQI] + zqxfg[NCLDQS]
        wm = sm.gt(zicetot, _ZEPSEC, _wt(_ZEPSEC)) * sm.gt(T, th.RTT, _W_T)
        zsubsat = sm.pos(zqsice - zqv, wb(zqsice))
        ztdmtw0 = T - th.RTT - zsubsat * (_TW1 + _TW2 * (pap - _TW3) - _TW4 * (T - _TW5))
        zcons1 = sm.abs(dt * (1.0 + 0.5 * ztdmtw0) / cfg.rtaumel, _W_F)
        zmeltmax = wm * (sm.pos(ztdmtw0, _W_T) * zcons1 * zrldcp)
        wmm = sm.gt(zmeltmax, _ZEPSEC, _wt(_ZEPSEC)) * sm.gt(zicetot, _ZEPSEC, _wt(_ZEPSEC))
        zicetot_s = _safe_gate(zicetot, _ZEPSEC)
        for m in (NCLDQI, NCLDQS):
            n = _IMELT[m]
            zalfa = zqxfg[m] / zicetot_s
            zmelt = wmm * sm.min(zqxfg[m], zalfa * zmeltmax, wb(zalfa * zmeltmax))
            zqxfg[m] = zqxfg[m] - zmelt
            zqxfg[n] = zqxfg[n] + zmelt
            zsolqa = add(zsolqa, n, m, zmelt)
            zsolqa = add(zsolqa, m, n, -zmelt)
        snap("s412", zsolqa=zsolqa, zqxfg=jnp.stack(zqxfg, -1))

        # ---- 4.13 rain freezing (:2877-2933) ----
        qr = zqx[:, NCLDQR]
        wr = sm.gt(qr, _ZEPSEC, _wt(_ZEPSEC))
        wtop = wr * sm.le(T, th.RTT, _W_T) * sm.gt(v["ztp1_up"], th.RTT, _W_T)
        zqpt = sm.max(zqx[:, NCLDQS] + qr, _ZEPSEC, _wt(_ZEPSEC))
        rf = qr / zqpt
        prainfrac = sm.sel(wtop, rf, c["prainfrac"])
        llrainliq = sm.sel(wtop, sm.gt(rf, _RAINFRAC_LIQ_SP, _wt(_RAINFRAC_LIQ_SP)), c["llrainliq"])
        wcold = wr * sm.lt(T, th.RTT, _W_T)
        zlambda = (RCL_LAM1R / (zrho * _safe_gate(qr, _ZEPSEC))) ** RCL_LAM2R
        ztemp = sm.min(cfg.rcl_fzrab * (T - th.RTT), _FZR_TEMP_CAP, _wt(_FZR_TEMP_CAP))
        zfrz = dt * (RCL_CONST5R / zrho) * (jnp.exp(ztemp) - 1.0) * zlambda ** RCL_CONST6R
        f_liq = sm.pos(zfrz, wq)
        zcons1 = sm.abs(dt * (1.0 + 0.5 * (th.RTT - T)) / cfg.rtaumel, _W_F)
        f_ice = sm.pos(th.RTT - T, _W_T) * zcons1 * zrldcp
        zfrzmax = sm.sel(llrainliq, f_liq, f_ice)
        wfz = wcold * sm.gt(zfrzmax, _ZEPSEC, _wt(_ZEPSEC))
        zfrz = wfz * sm.min(qr, zfrzmax, wb(zfrzmax))
        zsolqa = add(zsolqa, NCLDQS, NCLDQR, zfrz)
        zsolqa = add(zsolqa, NCLDQR, NCLDQS, -zfrz)
        snap("s413", zsolqa=zsolqa, prainfrac=prainfrac, llrainliq=llrainliq)

        # ---- 4.14 freezing of cloud liquid (:2948-2960) ----
        zfrzmax = sm.pos(cfg.rthomo - T, _W_T) * zrldcp
        wfl = sm.gt(zfrzmax, _ZEPSEC, _wt(_ZEPSEC)) * sm.gt(zqxfg[NCLDQL], _ZEPSEC, _wt(_ZEPSEC))
        zfrz = wfl * sm.min(zqxfg[NCLDQL], zfrzmax, wb(zfrzmax))
        zsolqa = add(zsolqa, NCLDQI, NCLDQL, zfrz)
        zsolqa = add(zsolqa, NCLDQL, NCLDQI, -zfrz)
        snap("s414", zsolqa=zsolqa)

        # ---- 4.9 riming, PSD slopes, PSACR, PIACR (:2973-3239) ----
        wrim = (sm.le(T, th.RTT, _W_T) * sm.gt(zliqcld, _ZEPSEC, _wt(_ZEPSEC))
                * sm.gt(zsnowcld, _ZEPSEC, _wt(_ZEPSEC))
                * sm.gt(zcovptot, _RIMING_COVP_MIN, _wt(_RIMING_COVP_MIN)))
        zsnowrime = (cfg.rcl_effrime * zcovptot * dt * RCL_CONST7S * zfallcorr_sp
                     * safe_pow(zrho * zsnowcld * RCL_CONST1S, RCL_CONST8S))
        zsnowrime = wrim * sm.min(zsnowrime, 1.0, _wt(1.0))
        zsolqb = add(zsolqb, NCLDQS, NCLDQL, zsnowrime)
        wrn = sm.gt(zraincld, _ZEPSEC, _wt(_ZEPSEC))
        wsn = sm.gt(zsnowcld, _ZEPSEC, _wt(_ZEPSEC))
        wic = sm.gt(zicecld, _ZEPSEC, _wt(_ZEPSEC))
        # PSD slopes/intercepts unweighted (inactive operands substituted, V22b); the
        # activation weights multiply only the final transfer below -- weighting a
        # slope before its inversion made a near-zero weight a singular rate.
        zlamr = (RCL_LAM1R / (zrho * _safe_gate(zraincld, _ZEPSEC))) ** RCL_LAM2R
        zn0r = RCL_X1R * zlamr ** RCL_X2R
        zlams = (RCL_LAM1S / (zrho * _safe_gate(zsnowcld, _ZEPSEC))) ** RCL_LAM2S
        zn0s = RCL_X1S
        zlami = (RCL_LAM1S / (zrho * _safe_gate(zicecld, _ZEPSEC))) ** RCL_LAM2S
        zn0i = RCL_X1S
        wcl = sm.le(T, th.RTT, _W_T) * sm.lt(prainfrac, _ZEPSEC, _wt(_ZEPSEC)) * wrn
        lr = zlamr
        for m, wsp, lam, n0, eff, src in ((NCLDQS, wsn, zlams, zn0s, _EFF_PSACR, NCLDQS),
                                          (NCLDQI, wic, zlami, zn0i, _EFF_PIACR, NCLDQI)):
            fr, fo = zfallspeed[NCLDQR], zfallspeed[m]
            b = (fr + fo) / _FALLDIFF_DIV
            zfalldiff = sm.max(sm.abs(fr - fo, _wfloor(b, _TINY)), b,
                               _wfloor(b, _TINY))
            ls = lam
            zfac = th.RPI * th.RPI * cfg.rdenswat * eff * zrho * zn0r * n0 / (ls * lr ** 3)
            zrat = lr / ls
            proc = wcl * wsp * (zcovptot * dt * zfac * zfalldiff
                                * ((_RIME_POLY[0] * zrat + _RIME_POLY[1]) * zrat + _RIME_POLY[2]) / lr ** 3)
            zsolqa = add(zsolqa, src, NCLDQR, proc)
            zsolqa = add(zsolqa, NCLDQR, src, -proc)
        snap("s49", zsolqa=zsolqa, zsolqb=zsolqb)

        # ---- 4.15 snow sublimation, ISUBLSNOW = 1 (:3252-3318) ----
        zzrh = cfg.rprecrhmax + (1.0 - cfg.rprecrhmax) * zcovpmax / sm.max(_ZEPSEC, 1.0 - za_k, _wt(_ZEPSEC))
        zzrh = sm.min(sm.max(zzrh, cfg.rprecrhmax, _wt(cfg.rprecrhmax)), 1.0, _wt(1.0))
        zqe = (zqv - za_k * zqsice) / sm.max(_ZEPSEC, 1.0 - za_k, _wt(_ZEPSEC))
        zqe = sm.max(0.0, sm.min(zqe, zqsice, wb(zqsice)), wq)
        llo1 = (sm.gt(zcovpclr, _ZEPSEC, _wt(_ZEPSEC)) * sm.gt(zqxfg[NCLDQS], _ZEPSEC, _wt(_ZEPSEC))
                * sm.lt(zqe, zzrh * zqsice, wb(zzrh * zqsice)))
        xd = zcovptot * zdtgdp
        zpreclr = zqxfg[NCLDQS] * zcovpclr / (jnp.where(xd >= 0, 1.0, -1.0)
                                               * jnp.maximum(jnp.abs(xd), _ZEPSILON))
        zbeta1 = (jnp.sqrt(pap / psurf) / cfg.rvrfactor * zpreclr
                  / sm.max(zcovpclr, _ZEPSEC, _wt(_ZEPSEC)))
        zbeta = th.RG * cfg.rpecons * safe_pow(zbeta1, _SUBL_BETA_EXP)
        zdenom = 1.0 + zbeta * dt * zcorqsice
        zdpr = zcovpclr * zbeta * (zqsice - zqe) / zdenom * zdp * zrg_r
        zdpevap = zdpr * zdtgdp
        zevap = llo1 * sm.min(zdpevap, zqxfg[NCLDQS], wb(zqxfg[NCLDQS]))
        zsolqa = add(zsolqa, NCLDQV, NCLDQS, zevap)
        zsolqa = add(zsolqa, NCLDQS, NCLDQV, -zevap)
        cnew = sm.max(cfg.rcovpmin, zcovptot - sm.pos((zcovptot - za_k) * zevap
                                                      / _safe_gate(zqxfg[NCLDQS], _ZEPSEC), _W_F),
                      _wt(cfg.rcovpmin))
        zcovptot = sm.sel(llo1, cnew, zcovptot)
        zqxfg[NCLDQS] = zqxfg[NCLDQS] - zevap
        for m in (NCLDQI, NCLDQR, NCLDQS):        # :3402-3412
            ws = sm.lt(zqxfg[m], cfg.rlmin, _wt(cfg.rlmin))
            zsolqa = add(zsolqa, NCLDQV, m, ws * zqxfg[m])
            zsolqa = add(zsolqa, m, NCLDQV, -(ws * zqxfg[m]))
        snap("s415", zsolqa=zsolqa, zqxfg=jnp.stack(zqxfg, -1), zcovptot=zcovptot)

        # ---- 5.1 cloud cover (:3429-3438) ----
        zanew = (za_k + zsolac) / (1.0 + zsolab)
        zanew = sm.min(zanew, 1.0, _wt(1.0))
        zanew = sm.sel(sm.lt(zanew, cfg.ramin, _wt(cfg.ramin)), 0.0 * zanew, zanew)
        zda = zanew - v["zaorig"]

        # ---- 5.2.1 truncate explicit sinks (:3455-3536) ----
        def ratio_of(row_sum_neg, qv):
            zmax = sm.max(qv, _ZEPSEC, _wt(_ZEPSEC))
            return zmax / sm.max(row_sum_neg, zmax, wb(zmax))

        zratio = []
        for m in range(NCLV):
            s_ = zeros
            for n in range(NCLV):
                s_ = s_ - zsolqa[:, m, n]
            zratio.append(ratio_of(s_, zqx[:, m]))
        iorder = jnp.argsort(jax.lax.stop_gradient(jnp.stack(zratio, -1)), axis=-1, stable=True)
        ar = jnp.arange(NCLV)
        for jm in range(NCLV):
            jo = iorder[:, jm]
            oh = (jo[:, None] == ar[None, :])                       # [ncol, 5]
            row = jnp.take_along_axis(zsolqa, jo[:, None, None], axis=1)[:, 0, :]
            s_ = zeros
            for n in range(NCLV):
                s_ = s_ - row[:, n]
            qo = jnp.take_along_axis(zqx, jo[:, None], axis=1)[:, 0]
            r = ratio_of(s_, qo)
            neg = [sm.neg1(row[:, n], wq) for n in range(NCLV)]   # V28a one-sided
            for n in range(NCLV):
                cur = jnp.take_along_axis(zsolqa[:, :, n], jo[:, None], axis=1)[:, 0]
                newv = sm.sel(neg[n], cur * r, cur)
                zsolqa = zsolqa.at[:, :, n].set(jnp.where(oh, newv[:, None], zsolqa[:, :, n]))
                cur = jnp.take_along_axis(zsolqa[:, n, :], jo[:, None], axis=1)[:, 0]
                newv = sm.sel(neg[n], cur * r, cur)
                zsolqa = zsolqa.at[:, n, :].set(jnp.where(oh, newv[:, None], zsolqa[:, n, :]))
        snap("s521", zsolqa=zsolqa, zanew=zanew)

        # ---- 5.2.2 implicit solve (:3546-3640) ----
        lhs = [[None] * NCLV for _ in range(NCLV)]
        for jm in range(NCLV):
            for jn in range(NCLV):
                if jn == jm:
                    d = 1.0 + zfallsink[jm]
                    for jo_ in range(NCLV):
                        d = d + zsolqb[:, jo_, jn]
                    lhs[jn][jm] = d
                else:
                    lhs[jn][jm] = -zsolqb[:, jn, jm]
        zqxn = []
        for jm in range(NCLV):
            e = zeros
            for jn in range(NCLV):
                e = e + zsolqa[:, jm, jn]
            zqxn.append(zqx[:, jm] + e)
        for jn in range(NCLV - 1):
            for jm in range(jn + 1, NCLV):
                lhs[jm][jn] = lhs[jm][jn] / lhs[jn][jn]
                for ik in range(jn + 1, NCLV):
                    lhs[jm][ik] = lhs[jm][ik] - lhs[jm][jn] * lhs[jn][ik]
        for jn in range(1, NCLV):
            for jm in range(jn):
                zqxn[jn] = zqxn[jn] - lhs[jn][jm] * zqxn[jm]
        zqxn[NCLV - 1] = zqxn[NCLV - 1] / lhs[NCLV - 1][NCLV - 1]
        for jn in range(NCLV - 2, -1, -1):
            for jm in range(jn + 1, NCLV):
                zqxn[jn] = zqxn[jn] - lhs[jn][jm] * zqxn[jm]
            zqxn[jn] = zqxn[jn] / lhs[jn][jn]
        for jn in range(NCLV - 1):
            ws = sm.lt(zqxn[jn], _ZEPSEC, _wt(_ZEPSEC))
            # VC1b/VC13a, conservation fix (departs from :3633-3639): the solve conserved
            # sum(ZQXN) + sum((ZFALLSINK + ZCONVSINK)*ZQXN), and the fall flux and the
            # subsidence outflow are taken from the tidied ZQXN, so the would-be
            # outflow of the tidied amount goes to vapour too; 49r1 drops it.
            zqxn[NCLDQV] = zqxn[NCLDQV] + ws * zqxn[jn] * (1.0 + zfallsink[jn] + zconvsink[jn])
            zqxn[jn] = sm.sel(ws, 0.0 * zqxn[jn], zqxn[jn]) if hard else (1.0 - ws) * zqxn[jn]
        snap("s522", zqxn=jnp.stack(zqxn, -1))

        # ---- 5.3 fluxes to the next level (:3658-3672) ----
        zpfplsx = jnp.stack([zfallsink[m] * zqxn[m] * zrdtgdp for m in range(NCLV)], -1)
        zqpretot = zpfplsx[:, NCLDQS] + zpfplsx[:, NCLDQR]
        zcovptot = sm.sel(sm.lt(zqpretot, _ZEPSEC, _wt(_ZEPSEC)), 0.0 * zcovptot, zcovptot)

        # ---- 6 tendencies (:3713-3809) ----
        tlt = v["tlt"]
        tlq = v["tlq"]
        tlc = []
        for m in range(NCLV - 1):
            zfluxq = zconvsrce[m] + zfallsrce[m] - (zfallsink[m] + zconvsink[m]) * zqxn[m]
            lat = th.RALVDCP if _IPHASE[m] == 1 else th.RALSDCP
            tlt = tlt + lat * (zqxn[m] - zqx[:, m] - zfluxq) * zqtmst
            tlc.append(0.0 + (zqxn[m] - v["zqx0"][:, m]) * zqtmst)
        tlq = tlq + (zqxn[NCLDQV] - zqx[:, NCLDQV]) * zqtmst
        tla = 0.0 + zda * zqtmst
        snap("s6", tlt=tlt, tlq=tlq, tla=tla, tlc=jnp.stack(tlc, -1))
        q_upd = v["pq"] + dt * (tlq + v["tcq"])
        t_upd = v["pt"] + dt * (tlt + v["tct"])
        a_upd = v["pa"] + dt * (tla + v["tca"])
        t_adj, q_adj, a_adj, l_adj, i_adj = _supersatcheck(sm, cfg, th, wq, t_upd, q_upd, a_upd, pap, psurf)
        tlq = tlq + q_adj * zqtmst
        tlt = tlt + t_adj * zqtmst
        tla = tla + a_adj * zqtmst
        tlc[NCLDQL] = tlc[NCLDQL] + l_adj * zqtmst
        tlc[NCLDQI] = tlc[NCLDQI] + i_adj * zqtmst
        for m, lat in ((NCLDQL, th.RALVDCP), (NCLDQI, th.RALSDCP)):
            z = v["zqx0"][:, m] + tlc[m] * dt
            wneg = sm.neg1(z, wq)
            tlc[m] = tlc[m] - wneg * (z * zqtmst)
            tlq = tlq + wneg * (z * zqtmst)
            tlt = tlt - wneg * (lat * z * zqtmst)

        out = dict(tlt=tlt, tlq=tlq, tla=tla, tlc=jnp.stack(tlc, -1), plude=plude,
                   covptot=zcovptot, zqxn=jnp.stack(zqxn, -1), zpfplsx=zpfplsx)
        cn = dict(zqxnm1=jnp.stack(zqxn, -1), zanewm1=zanew, zcovptot=zcovptot, zcovpmax=zcovpmax,
                  zcldtopdist=zcldtopdist, prainfrac=prainfrac, llrainliq=llrainliq, zpfplsx=zpfplsx)
        return cn, (out, snaps)

    cfin, (lo, snaps) = jax.lax.scan(level, carry0, lev)

    def full(vk, above):          # [nlev-k0, ncol, ...] -> [ncol, nlev, ...]
        return jnp.concatenate([above, jnp.moveaxis(vk, 0, 1)], axis=1)

    tend_t = full(lo["tlt"], tlt[:, :k0])
    tend_q = full(lo["tlq"], tlq[:, :k0])
    tend_a = full(lo["tla"], jnp.zeros_like(x.t[:, :k0]))
    tend_cld = full(lo["tlc"], jnp.zeros_like(x.clv[:, :k0, :4]))
    plude = full(lo["plude"], x.plude[:, :k0])
    covptot = full(lo["covptot"], jnp.zeros_like(x.t[:, :k0]))
    zqxn2d = full(lo["zqxn"], jnp.zeros_like(x.clv[:, :k0]))
    pfplsx = jnp.concatenate([jnp.zeros_like(x.clv[:, :k0 + 1]), jnp.moveaxis(lo["zpfplsx"], 0, 1)], axis=1)

    # ---------------- section 8: fluxes (:4167-4253) -------------------------
    fplsl = pfplsx[..., NCLDQR] + pfplsx[..., NCLDQL]
    fplsn = pfplsx[..., NCLDQS] + pfplsx[..., NCLDQI]
    zgdph_r = -zrg_r * (x.paph[:, 1:] - x.paph[:, :-1]) * zqtmst
    fl = dict(
        lf=(zqxn2d[..., NCLDQL] - zqx0[NCLDQL] + x.pvfl * dt) * zgdph_r + x.pludeli[..., 0],
        lng=zlneg[NCLDQL] * zgdph_r, ltur=x.pvfl * dt * zgdph_r,
        rf=(zqxn2d[..., NCLDQR] - zqx0[NCLDQR]) * zgdph_r, rng=zlneg[NCLDQR] * zgdph_r,
        if_=(zqxn2d[..., NCLDQI] - zqx0[NCLDQI] + x.pvfi * dt) * zgdph_r + x.pludeli[..., 1],
        nng=zlneg[NCLDQI] * zgdph_r, itur=x.pvfi * dt * zgdph_r,
        sf=(zqxn2d[..., NCLDQS] - zqx0[NCLDQS]) * zgdph_r, sng=zlneg[NCLDQS] * zgdph_r,
    )

    def fstep(cc, t):
        lf, if_, lng, nng, ltur, itur = cc
        n = (lf + t["lf"], if_ + t["if_"], lng + t["lng"], nng + t["nng"], ltur + t["ltur"],
             itur + t["itur"])
        return n, n + (lf + t["rf"], if_ + t["sf"], lng + t["rng"], nng + t["sng"])

    z1 = jnp.zeros((ncol,), x.t.dtype)
    _, fo = jax.lax.scan(fstep, (z1,) * 6, {k: jnp.moveaxis(v_, 1, 0) for k, v_ in fl.items()})

    def withtop(a):
        return jnp.concatenate([z1[:, None], jnp.moveaxis(a, 0, 1)], axis=1)

    fsqlf, fsqif, fcqlng, fcqnng, fsqltur, fsqitur, fsqrf, fsqsf, fcqrng, fcqsng = map(withtop, fo)
    out = CloudscOutputs(
        tend_t=tend_t, tend_q=tend_q, tend_a=tend_a, tend_cld=tend_cld, plude=plude,
        covptot=covptot, prainfrac_toprfz=cfin["prainfrac"],
        fsqlf=fsqlf, fsqif=fsqif, fcqnng=fcqnng, fcqlng=fcqlng, fsqrf=fsqrf, fsqsf=fsqsf,
        fcqrng=fcqrng, fcqsng=fcqsng, fsqltur=fsqltur, fsqitur=fsqitur, fplsl=fplsl, fplsn=fplsn,
        fhpsl=-th.RLVTT * fplsl, fhpsn=-th.RLSTT * fplsn)
    if stages:
        return out, snaps
    return out


def _supersatcheck(sm, cfg, th, wq, pt, pq, pa, pp, psurf):
    """cloud_supersatcheck.F90 (49r1), KFTLIQICE = 1."""
    def wb(b):
        return _wfloor(b, wq)

    zt = sm.max(pt, _SUPSAT_TMIN, _W_T)
    za = sm.max(sm.min(pa, 1.0, _W_F), 0.0, _W_F)
    zp_r = 1.0 / pp
    zfoeeliq = sm.min(foeeliq(th, zt) * zp_r,
                      _QSAT_CAP, _wt(_QSAT_CAP))
    zqsliq = zfoeeliq / (1.0 - th.RETV * zfoeeliq)
    d = _foedelta(sm, th, zt)
    foeew = th.R2ES * jnp.exp((th.R3LES * d + th.R3IES * (1.0 - d)) * (zt - th.RTT)
                              / (zt - (th.R4LES * d + th.R4IES * (1.0 - d))))
    zfoeeice = sm.min(foeew * zp_r, _QSAT_CAP, _wt(_QSAT_CAP))
    zqsice = zfoeeice / (1.0 - th.RETV * zfoeeice)
    zfokoop = sm.min(zfoeeice * (th.RKOOP1 - th.RKOOP2 * zt), zfoeeliq, wb(zfoeeliq))
    zqsliqk = zfokoop / (1.0 - th.RETV * zfokoop)
    zfoealfa = _foealfa(sm, th, zt)
    zfoeemix = sm.min(ew_mixed(th, zfoealfa, zt) * zp_r, _QSAT_CAP, _wt(_QSAT_CAP))
    zcor = 1.0 / (1.0 - th.RETV * zfoeemix)
    zqsmix = zfoeemix * zcor
    zcorqsmix = 1.0 + zqsmix * zcor * dem_mixed(th, zfoealfa, zt)
    zqsliqk = 0.5 * (zqsliqk + zqsmix)
    zqslim = za * zqsmix + (1.0 - za) * zqsliqk
    zcond = sm.pos(pq - zqslim, wb(zqslim))
    wc = sm.gt(zcond, _ZEPSEC, _wt(_ZEPSEC))
    zfaci = zfoealfa + (1.0 - zfoealfa) * cfg.rssicefactor
    zda_c = (1.0 - za) * zfaci
    za_upd = za + zda_c
    zqslim2 = za_upd * zqsmix + (1.0 - za_upd) * zqsliqk
    zcond = sm.sel(wc, (pq - zqslim2) / zcorqsmix, zcond)
    zda = wc * zda_c
    t_adj = (zfoealfa * th.RALVDCP + (1.0 - zfoealfa) * th.RALSDCP) * zcond
    return t_adj, -zcond, zda, zcond * zfoealfa, zcond * (1.0 - zfoealfa)


# ---------------------------------------------------------------------------
# legoESM integration: NCLDTOP rule and the standard-contract adapter
# ---------------------------------------------------------------------------
_RCLDTOPP = 100.0                  # sucldp.F90 RCLDTOPP [Pa] (49r1)


def ncldtop_from_reference_pressure(p_full_ref):
    """sucldp.F90:631-634: NCLDTOP = 2, then for JLEV = NFLEVG..2 (1-based)
    NCLDTOP = JLEV wherever STPRE(JLEV) > RCLDTOPP.  ``p_full_ref`` is the
    full-level pressure [Pa] of the standard atmosphere (top first), i.e. the
    lane's level table evaluated at p_s = 101325 Pa.  Static (numpy)."""
    p = np.asarray(p_full_ref, dtype=np.float64).reshape(-1)
    ncldtop = 2
    for jlev in range(p.size, 1, -1):
        if p[jlev - 1] > _RCLDTOPP:
            ncldtop = jlev
    return ncldtop


class CloudscConvection(NamedTuple):
    """CUMASTRN's raw outputs CLOUDSC reads (callpar order), [ncol, nlev]
    unless noted: PLU, PLUDE and PLUDELI (``[ncol, nlev, 4]`` liquid, ice,
    vapour, T), PSNDE (``[ncol, nlev, 2]`` rain, snow), PMFU, PMFD, and the
    per-column LDCUM (bool) and KTYPE (int)."""
    plu: jax.Array
    plude: jax.Array
    pludeli: jax.Array
    psnde: jax.Array
    pmfu: jax.Array
    pmfd: jax.Array
    ldcum: jax.Array
    ktype: jax.Array


def ifs_cloudsc_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt,
                             config, *, cloud_fraction, land_frac, conv):
    """Standard microphysics contract around :func:`cloudsc` (one call, the
    smooth path).  Inputs per the integration votes (vote_1103 VC2-VC21):
    PTENDENCY_CML = 0 (operator split), PVFL = PVFI = 0 and PEIS = 0 (no VDF
    yet), PLSM = ``land_frac``, the convective fields from ``conv``
    (:class:`CloudscConvection`).  Species map L, I, R, S -> q_c, q_i, q_r,
    q_s; q_g and every number get zero tendency.  Surface precipitation =
    FPLSL + FPLSN at the surface half level, its frozen part FPLSN there.
    ``rho`` and ``dz`` are unused (CLOUDSC works in pressure).  Returns a
    ``MicrophysicsOutput`` whose ``cloud_fraction_new`` is A + dt*tend_a."""
    from .output import MicrophysicsOutput
    from .._ifs_thermo import lego_thermo
    del rho, dz
    if not isinstance(conv, CloudscConvection):
        raise TypeError("ifs_cloudsc: conv must be a CloudscConvection (the "
                        "CUMASTRN hand-off); got %r" % (type(conv).__name__,))
    zero = jnp.zeros_like(T)
    h = hydrometeors
    clv = jnp.stack([h.q_c, h.q_i, h.q_r, h.q_s, q_v], axis=-1)
    ncol = T.shape[0]
    x = CloudscInputs(
        t=T, q=q_v, a=cloud_fraction, clv=clv,
        tend_t=zero, tend_q=zero, tend_a=zero, tend_cld=jnp.zeros_like(clv),
        pvfl=zero, pvfi=zero, pap=p_full, paph=p_half,
        plsm=jnp.asarray(land_frac, T.dtype).reshape(ncol),
        ldcum=jnp.asarray(conv.ldcum).reshape(ncol).astype(bool),
        ktype=jnp.asarray(conv.ktype).reshape(ncol),
        peis=jnp.zeros((ncol,), T.dtype),
        plu=conv.plu, plude=conv.plude, pludeli=conv.pludeli, psnde=conv.psnde,
        pmfu=conv.pmfu, pmfd=conv.pmfd)
    o = cloudsc(config, lego_thermo(), x, dt, hard=False)
    tc = o.tend_cld
    return MicrophysicsOutput(
        dT_dt=o.tend_t, dq_v_dt=o.tend_q,
        dq_c_dt=tc[..., NCLDQL], dq_r_dt=tc[..., NCLDQR],
        dq_i_dt=tc[..., NCLDQI], dq_s_dt=tc[..., NCLDQS], dq_g_dt=zero,
        dN_c_dt=zero, dN_r_dt=zero, dN_i_dt=zero,
        precipitation=o.fplsl[:, -1] + o.fplsn[:, -1],
        precip_frozen=o.fplsn[:, -1],
        cloud_fraction_new=cloud_fraction + dt * o.tend_a)
