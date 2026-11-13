"""WRF Kain-Fritsch (KF-Eta) cumulus scheme: faithful JAX port, one column.

Port of WRF v4.8.0 (github.com/wrf-model/WRF @ 06d4240ae989cc3e50af412bb472df3d9048783c)
``phys/module_cu_kfeta.F``: ``KF_eta_PARA`` and every routine it calls
(``TPMIX2``, ``DTFRZNEW``, ``CONDLOAD``, ``PROF5``, ``TPMIX2DD``, ``ENVIRTHT``)
plus the lookup tables of ``KF_LUTAB``.  The statements are transcribed in the
oracle's order (arithmetic association included) so the port reproduces the
compiled Fortran to round-off; the test is
``tests/atmosphere/test_kain_fritsch_oracle.py``.

NOT RULE 5 COMPLIANT: reached only through the opt-in
``KainFritschConfig.variant = "wrf_kfeta"`` (``kain_fritsch_wrf.py``); every
traced-state switch below is hard at the default.  Smoothing tracked in #2280;
every switch is inventoried in ``KF_SWITCHES`` and tagged ``# kfsw: <id>``.

Conventions are the oracle's, NOT legoESM's:

* levels BOTTOM-UP (index 0 = lowest model level);
* ``qv`` is the water-vapour MIXING RATIO the oracle reads (``QV0``);
* ``rhoe`` and ``dzq`` give the layer pressure thickness ``DP = rhoe*G*dzq``;
* ``w0avg`` is the running-mean grid vertical velocity ``W0AVG1D`` [m/s];
* the outputs are the oracle's ``DTDT, DQDT, DQCDT, DQRDT, DQIDT, DQSDT`` (rates
  over the convective time scale ``TIMEC``), ``PRATEC`` [kg m-2 s-1], ``NCA``,
  ``CUTOP``, ``CUBOT`` and ``ISHALL``.

Control flow: every ``EXIT``/``CYCLE``/``RETURN`` of the oracle is a hard switch
(``jnp.where`` on a loop-carried flag), so the port is piecewise differentiable
and equals the oracle exactly.  The data-dependent loops run to a static bound
and are masked: the updraft-source-layer search (at most ``KX + 2`` passes), the
updraft and downdraft level loops (``KX``), the closure iteration (10, as in the
oracle) and the compensating-subsidence sub-steps (``_MAX_SUBSTEPS``; a column
that needs more returns NaN rather than a truncated answer).

Oracle behaviour that cannot be expressed in JAX and is mapped instead:

* ``wrf_error_fatal`` (negative vapour at the lowest level after the
  subsidence borrow, and the ``ISTOP`` mass-balance failure) aborts the WRF
  run; here the column returns NaN tendencies.
* A lookup outside the tables (pressure above 50 hPa or a theta-e outside the
  table window) reads out of bounds in the oracle (undefined); here the index is
  clamped to the table edge.
* ``trigger = 2`` (Ma and Tan 2009) needs the horizontal 9-point neighbourhood
  and is not ported: :func:`kf_eta_para` raises for it.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.convection.config import KainFritschConfig
from legoesm.core.smooth import (
    positive_part_c1,
    smooth_max_c1,
    smoothstep_cubic,
    sqrt_c1,
    validate_smoothing_width,
)

__all__ = ("KF_HARD", "KFConstants", "KFSmoothing", "kf_eta_para", "kf_lookup_tables",
           "kf_smoothing", "prof5")

__physics_contract__ = {
    "summary": (
        "WRF v4.8.0 KF-Eta deep+shallow convection column (KF_eta_PARA): USL "
        "search, Fritsch-Chappell trigger, entraining/detraining updraft with "
        "CONDLOAD fallout and glaciation, RH-controlled downdraft, iterated ABE "
        "closure, upstream compensating subsidence over TIMEC."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "t": "K", "qv": "kg/kg (mixing ratio)",
        "p": "Pa", "dzq": "m", "w0avg": "m/s", "rhoe": "kg/m^3",
        "dt": "s", "dx": "m",
    },
    "outputs": {
        "dtdt": "K/s", "dqdt": "kg/kg/s", "dqcdt": "kg/kg/s", "dqrdt": "kg/kg/s",
        "dqidt": "kg/kg/s", "dqsdt": "kg/kg/s", "pratec": "kg/m^2/s",
    },
    "sign_convention": (
        "levels bottom-up; positive tendency = increase; condensation warms and "
        "dries, PRATEC >= 0 leaves the column at the surface"
    ),
    "conserves": ["moisture"],
    "differentiable": True,
    "reference": "WRF v4.8.0 06d4240 phys/module_cu_kfeta.F (Kain 2004, JAM 43)",
    "idealized_test": "tests/atmosphere/test_kain_fritsch_oracle.py",
}

# RULE 5 switch inventory (#2280).  Every switching statement of the traced
# column code carries ``# kfsw: <id>``; the gate is
# tests/atmosphere/test_kf_eta_switch_inventory.py.  id -> (family, status,
# description); families as issue #2280: A whole-column on/off, B branch
# selection, C integer level selection, D discontinuous piecewise formula,
# E continuous kink (min/max/clip/abs/sqrt), F lookup-table coordinate,
# S state-independent (static index or Python-level dispatch).
KF_SWITCHES = {
    "A0": ("A", "hard", "fired: outputs, NaN fill and divisor guards gated by the column firing"),
    "A1": ("A", "hard", "trigger TLCL+DTLCL+DTRH < TENV per source-layer candidate"),
    "A2": ("A", "hard", "mixed-layer depth DPTHMX < DPMIN returns"),
    "A3": ("A", "hard", "ZLCL above the top model level returns"),
    "A4": ("A", "hard", "no convection: LTOP<=KLCL, LTOP<=KPBL or LET+1<=KPBL"),
    "A5": ("A", "hard", "closure cancels: FABE > 1 (deep), AINC < 0.05"),
    "A6": ("A", "hard", "fail-loud NaN: substep overflow, QG(1)<0 borrow, ISTOP mass balance"),
    "B1": ("B", "hard", "deep vs shallow (cldhgt > CHMIN and ABE > 1) and every ISHALL use"),
    "B2": ("B", "hard", "source-layer candidate loop, shallow restart at argmax CLDHGT"),
    "B3": ("B", "hard", "downdraft: dd_ok, deep_dd, nodd (TDER<1), ddinc cap"),
    "B4": ("B", "hard", "closure iteration branches: cyc1, cyc2, convergence, saturation, FABE=0"),
    "B5": ("B", "hard", "melting below the freezing level: LC < ML"),
    "C-KCHECK": ("C", "hard", "15 hPa source-layer candidates, candidate base LC = KCHECK(NU)"),
    "C-KLCL": ("C", "hard", "LCL level KLCL / K = KLCL-1 and the masks built on it"),
    "C-KLCLg": ("C", "smooth", "first-pass gathers at ZLCL: TENV, QENV, WKL, QSLCL, PLCL"),
    "C-KLCL2": ("C", "hard", "closure re-search of the LCL level on the adjusted sounding"),
    "C-KPBL": ("C", "hard", "mixed-layer top KPBL and the masks built on it"),
    "C-L5": ("C", "hard", "L5 and LLFC pressure levels"),
    "C-L5g": ("C", "smooth", "wind speed at L5 in VCONV"),
    "C-LDB": ("C", "hard", "downdraft base LDB and the downdraft level masks"),
    "C-LET": ("C", "hard", "equilibrium-temperature level LET and the cloud-top detrainment layer"),
    "C-LFS": ("C", "hard", "level of free sink LFS (KLFS) and LDT"),
    "C-LTOP": ("C", "hard", "updraft exits (WTW, UMF-UDR), cloud top LTOP and its masks"),
    "C-ML": ("C", "hard", "melting level ML (T0 > T00 below LTOP)"),
    "C-NIC": ("C", "hard", "TIMEC rounded to NINT(TIMEC/DT) steps, NCA"),
    "C-NSTEP": ("C", "hard", "subsidence sub-step count NSTEP and its step mask"),
    "D1": ("D", "hard", "DTLCL = 0 for WKL < 1e-4"),
    "D2": ("D", "hard", "DTRH three-way split on RHLCL"),
    "D3": ("D", "hard", "WLCL = 1 unless DTTOT > 1e-4"),
    "D4": ("D", "hard", "entrainment selection cold / vb / tiny (ee2, ud2, eqfrc)"),
    "D5": ("D", "hard", "shear sign SHSIGN"),
    "D6": ("D", "hard", "RCBH polynomial / constant splice at CBH 3 and 25 kft"),
    "D7a": ("D", "smooth", "TPMIX2 saturated / enough condensate / dry branches"),
    "D7b": ("D", "smooth", "CONDLOAD g1 < 0 set to 0"),
    "D7c": ("D", "hard", "CONDLOAD mean vertical velocity zero guard"),
    "D7d": ("D", "smooth", "CONDLOAD |WTW| < 1e-4 floor"),
    "D7e": ("D", "smooth", "PROF5 y >= 0 polynomial branch and |y|"),
    "D7f": ("D", "hard", "PROF5 eqf == 0 / eqf == 1 exact-equality overrides"),
    "D7g": ("D", "hard", "glaciation: freezing (frz) and partial (part) ramps, DTFRZNEW"),
    "D7h": ("D", "hard", "downdraft RH adjustment: low (qsrh < qd) and adj (rhh < 1)"),
    "D7i": ("D", "hard", "subsidence moisture borrow where QG < 0"),
    "D7j": ("D", "hard", "closure mixed-layer supersaturation adjustment (sup)"),
    "D7k": ("D", "hard", "CPR > 0 guard of the precipitation feedback ratio FRC2"),
    "E1": ("E", "smooth", "Q0 clipped to [1e-6, QES]"),
    "E2": ("E", "smooth", "TLCL <= TMIX (first pass and closure)"),
    "E3": ("E", "hard", "WKLCL linear ramp capped at wklcl_ref"),
    "E4": ("E", "hard", "WLCL <= 3 m/s"),
    "E5": ("E", "hard", "RAD ramp clipped to [rad_min, rad_max]"),
    "E6": ("E", "hard", "ABE accumulates only positive dilbe"),
    "E7": ("E", "hard", "eqf clipped to [0, 1]"),
    "E8": ("E", "hard", "ee2 >= 0.5"),
    "E9": ("E", "hard", "CHMIN temperature ramp between 273 and 293 K"),
    "E10": ("E", "hard", "upstream direction OMG <= 0 in the subsidence advection"),
    "E11": ("E", "hard", "square root kink at zero: GDT, wind speed, shear"),
    "E11c": ("E", "smooth", "CONDLOAD square roots of WTW and G1"),
    "E12": ("E", "hard", "TIMEC = DX/VCONV clamped to [timec_min, timec_max], VCONV > 0"),
    "E13": ("E", "hard", "PEF clamped to [0.2, 0.9]"),
    "E14": ("E", "hard", "PEFCBH <= 0.9"),
    "E15": ("E", "hard", "inflow limit AINCMX: UER-DER > 1e-3 mask, min reduction, AINC <= 1"),
    "E16": ("E", "hard", "subsidence time step DTT limited by |OMG|*TIMEC > 0.75 DP"),
    "E17": ("E", "hard", "closure qmix >= 0"),
    "E18": ("E", "hard", "DABE >= 0.1 ABE"),
    "E19": ("E", "hard", "new AINC capped at AINCMX"),
    "F1": ("F", "smooth", "ALU log table: coordinate clamp and interval index"),
    "F2": ("F", "smooth", "TTAB/QSTAB tables: coordinate clamps and interval indices"),
    "S": ("S", "static", "static level index, static pad mask, Python-level option dispatch"),
    "H": ("S", "static", "inside a masking/rounding helper; the switch is tagged at each call"),
}

# --- KF_LUTAB lookup tables (module_cu_kfeta.F:3174-3301) ---
_KFNT = 250
_KFNP = 220
_LUT_PLUTOP_PA = 5000.0
_LUT_PBOT_PA = 110000.0
_LUT_DTH_K = 1.0
_LUT_TMIN_K = 150.0
_LUT_TOLER_K = 0.001
_LUT_NITER = 11
_ALU_N = 200
_ALU_ASTRT = 1.0e-3
_ALU_AINC = 0.075
_KPA_TO_PA = 1000.0  # ALIQ = SVP1*1000 (SVP1 in kPa)

# --- thermodynamic numbers hard-coded inside the oracle (not argument constants) ---
_EPS_KF = 0.622  # const-ok: KF-Eta hard-codes 0.622 in QES/QSS/EE (it does not read EP2)
_TV_COEF = 0.608  # virtual-temperature coefficient (TV0, TVU, TVQU, TVD, TVLCL)
_P00_PA = 1.0e5  # P00 (DATA, l.711; ENVIRTHT l.3055)
_T00_K = 273.16  # T00 (DATA, l.711; ENVIRTHT l.3055)
_EXN_KAPPA = 0.2854  # moist Exner exponent 0.2854*(1-0.28*q)
_EXN_Q = 0.28
_THE_C1 = 3374.6525  # Bolton theta-e: EXP((C1/T-C2)*Q*(1+0.81*Q))
_THE_C2 = 2.5403
_THE_C3 = 0.81
_TSAT_A = 0.212  # LCL temperature from the dew point (l.928, l.3077)
_TSAT_B = 1.571e-3
_TSAT_C = 4.36e-4
_RLF = 3.339e5  # latent heat of fusion (DATA, l.712)
_TTFRZ_K = 268.16  # glaciation start (DATA, l.714)
_TBFRZ_K = 248.16  # glaciation end
_RATE = 0.03  # CONDLOAD fallout rate (DATA, l.715)
_CPM_Q = 0.887  # CPM = CP*(1+0.887*Q)
_TPMIX_CP = 1004.5  # TPMIX2/DTFRZNEW CPP = 1004.5*(1+0.89*QU)
_TPMIX_CPQ = 0.89
_TPMIX_QEPS = 1.0e-10
_DTFRZ_RLC0 = 2.5e6  # DTFRZNEW latent heats (l.2833-2835)
_DTFRZ_RLC1 = 2369.276
_DTFRZ_RLS0 = 2833922.0
_DTFRZ_RLS1 = 259.532
_KE_DIV = 1.5  # 2*g*dz*B/1.5 (virtual-mass factor, l.1168, l.2888)
_CL_FRESH = 0.6  # CONDLOAD: fraction of fresh condensate in the conversion
_CL_RETAIN = 0.4  # CONDLOAD: fraction of fresh condensate retained
_CL_DRAG_EXCL = 0.2  # CONDLOAD: PPTDRG = 0.5*(OLDQ+QTOT-0.2*QNEW)
_CL_EPS = 1.0e-8
_WTW_FLOOR = 1.0e-4
# PROF5 (l.2930-2974)
_P5_SQRT2P = 2.506628
_P5_A1 = 0.4361836
_P5_A2 = -0.1201676
_P5_A3 = 0.9372980
_P5_P = 0.33267
_P5_SIGMA = 0.166666667
_P5_FE = 0.202765151
_P5_E45_ARG = -4.5
_P5_T1 = 0.500498
_P5_Y_SCALE = 6.0
_P5_Y_OFF = 3.0

# --- KF_eta_PARA scalars (module_cu_kfeta.F:533-2692) ---
_P300_DEPTH_PA = 30000.0  # P300 = P0(1)-30000 (USL search depth)
_PM15_PA = 15.0e2  # USL candidate spacing
_L5_FRAC = 0.5  # L5: last level with P0 >= 0.5*P0(1)
_WKL_DX_REF_M = 25.0e3  # W0AVG*DX/25.E3
_WKL_MIN = 1.0e-4  # DTLCL = 0 below
_DTTOT_MIN_K = 1.0e-4
_GDT_DEPTH_M = 500.0
_WLCL_MAX = 3.0
_AU0_FRAC = 0.01  # AU0 = 0.01*DXSQ
_WTW_STOP = 1.0e-3
_F1_HI = 0.95
_F1_LO = 0.10
_TVDIFF_MIN = 1.0e-3
_EE2_MIN = 0.5
_UD2_SCALE = 1.5
_UMF_DETR_MIN = 10.0  # IF(UMF(NK)-UDR(NK1).LT.10.)
_CHMIN_TWARM_K = 293.0
_CHMIN_TCOLD_K = 273.0
_CHMIN_DEEP_M = 4.0e3
_CHMIN_SHALLOW_M = 2.0e3
_CHMIN_SLOPE_M_PER_K = 100.0
_ABE_MIN = 1.0  # deep convection needs ABE > 1
_TIMEC_SHALLOW_S = 2400.0
_VWS_SCALE = 1.0e3
_PEF_C0 = 1.591
_PEF_C1 = -0.639
_PEF_C2 = 9.53e-2
_PEF_C3 = 4.96e-3
_PEF_MIN = 0.2
_PEF_MAX = 0.9
_CBH_FT_PER_M = 3.281e-3
_CBH_LOW = 3.0
_CBH_HIGH = 25.0
_RCBH_LOW = 0.02
_RCBH_HIGH = 2.4
_RCBH_C0 = 0.96729352
_RCBH_C1 = -0.70034167
_RCBH_C2 = 0.162179896
_RCBH_C3 = -1.2569798e-2
_RCBH_C4 = 4.2772e-4
_RCBH_C5 = 5.44e-6
_PEFCBH_MAX = 0.9
_LFS_DEPTH_PA = 150.0e2
_DD_MIN_DEPTH_PA = 50.0e2
_DMFFRC_COEF = 2.0
_RHH_DECR = 0.2
_RHH_PER_M = 1000.0
_TDER_MIN = 1.0
_AINCMX0 = 1000.0
_UERDER_MIN = 1.0e-3
_STAB = 0.95
_FABE_HI = 1.05
_FABE_LO = 0.95
_AINC_TOL = 0.0001
_DABE_MIN = 1.0e-4
_DABE_FRAC = 0.1
_AINC_MIN = 0.05
_AINC_REL_MAX = 0.999
_N_ITER = 10
_TKEMAX = 5.0
_EVAC_FRAC = 0.5
_EVAC_TKE = 0.1
_FRDP = 0.75
_QG_FLOOR = 1.0e-9
_TOPOMG_TOL = 1.0e-3
_QSEC_PER_MIN = 60.0
_Q0_FLOOR = 0.000001  # Q0 = AMAX1(0.000001, Q0)
_PCT = 100.0
# Static bound on the compensating-subsidence sub-steps NSTEP = NINT(TIMEC/DTT+1).
# The oracle has no bound; a column needing more returns NaN (never truncated).
_MAX_SUBSTEPS = 256


class KFSmoothing(NamedTuple):
    """RULE 5 widths of the port's smoothed switches (#2280).  0 = the oracle's
    hard switch (the transcribed code runs unchanged); defaults are the
    ``KainFritschConfig.wrf_*`` fields.  Static Python floats, never traced.

    * ``lut_knot_width`` [table index units, 0 or (0, 1]]: half-width of the
      box kernel convolved with the piecewise-(bi)linear ALU, THE0K and
      TTAB/QSTAB interpolants; smooths the knots and the edge clamps (F1, F2).
    * ``tpmix_width_kgkg``: band of the C1 positive parts of TPMIX2's
      condensation / evaporation / deficit split (D7a).
    * ``condload_sqrt_width_m2s2``: bridge of the C1 square roots of WTW and of
      the clipped G1 in CONDLOAD (D7b, E11 there).
    * ``wtw_floor_width_m2s2``: cubic ramps outside the |WTW| < 1e-4 floor
      (D7d).
    * ``prof5_width``: half-width in PROF5's ``y`` of the blend between its
      two polynomial branches (D7e).
    * ``lcl_gather_width_m``: band in height around each level of the soft
      interval weights that interpolate TENV, QENV, WKL, QSLCL and PLCL at ZLCL
      (C-KLCLg).  KLCL and K stay hard.
    * ``l5_gather_width_pa``: band in pressure around 0.5 P0(1) of the soft
      level weights of the L5 wind speed in VCONV (C-L5g).  L5 and LLFC stay hard.
    * ``q0_clip_width_kgkg`` / ``q0_floor_width_kgkg``: bands of the one-sided
      C1 forms of Q0 = min(QES, QV0) (never above QES) and Q0 = max(1e-6, Q0)
      (never below 1e-6) (E1).
    * ``tlcl_cap_width_k``: band of the one-sided C1 TLCL <= TMIX (E2), first
      pass and closure.
    """

    lut_knot_width: float
    tpmix_width_kgkg: float
    condload_sqrt_width_m2s2: float
    wtw_floor_width_m2s2: float
    prof5_width: float
    lcl_gather_width_m: float
    l5_gather_width_pa: float
    q0_clip_width_kgkg: float
    q0_floor_width_kgkg: float
    tlcl_cap_width_k: float


KF_HARD = KFSmoothing(*(0.0 for _ in KFSmoothing._fields))
# Largest PROF5 blend half-width: entrainment and detrainment fractions stay
# non-negative on eq in [0, 1] (tested); wider blends reach the far branch's
# negative values (ud < 0 at width 3).
_PROF5_WIDTH_MAX = 0.1
# kfsw: S
assert _PROF5_WIDTH_MAX < 1.0 / _P5_P  # the branch continuations' pole stays outside the blend


def kf_smoothing(config: KainFritschConfig) -> KFSmoothing:
    """The widths a ``KainFritschConfig`` selects (its ``wrf_*`` fields)."""
    return KFSmoothing(
        config.wrf_lut_knot_width, config.wrf_tpmix_width_kgkg,
        config.wrf_condload_sqrt_width_m2s2, config.wrf_wtw_floor_width_m2s2,
        config.wrf_prof5_width,
        config.wrf_lcl_gather_width_m, config.wrf_l5_gather_width_pa,
        config.wrf_q0_clip_width_kgkg, config.wrf_q0_floor_width_kgkg,
        config.wrf_tlcl_cap_width_k)


def validate_kf_smoothing(sm: KFSmoothing) -> None:
    for name, w in sm._asdict().items():  # kfsw: S
        validate_smoothing_width(f"KFSmoothing.{name}", w)  # kfsw: S
    for name, hi in (("lut_knot_width", 1.0), ("prof5_width", _PROF5_WIDTH_MAX)):
        if getattr(sm, name) > hi:  # kfsw: S
            raise ValueError(f"KFSmoothing.{name} must be 0 or in (0, {hi}]; "
                             f"got {getattr(sm, name)!r}")  # kfsw: S


class _KFTables(NamedTuple):
    """KF_LUTAB tables (the first seven are the oracle's), their padded copies
    for the smoothed lookups (node -1 = linear extrapolation of the first
    interval, as the oracle's INT truncation does on (-1, 0)), and the widths."""

    ttab: jax.Array
    qstab: jax.Array
    the0k: jax.Array
    alu: jax.Array
    plutop: jax.Array
    rdpr: jax.Array
    rdthk: jax.Array
    ttab_pad: jax.Array
    qstab_pad: jax.Array
    the0k_pad: jax.Array
    alu_pad: jax.Array
    sm: KFSmoothing


class KFConstants(NamedTuple):
    """The constants KF_eta_PARA receives through its argument list."""

    g: float
    cp: float
    r: float
    xlv0: float
    xlv1: float
    svp1: float
    svp2: float
    svp3: float
    svpt0: float


def _nint(x):
    """Fortran NINT (round half away from zero) as an int32."""
    return (jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)).astype(jnp.int32)  # kfsw: H


def _ftrunc_i(x):
    """Fortran INT(): truncation toward zero, int32."""
    return jnp.trunc(x).astype(jnp.int32)  # kfsw: H


@functools.lru_cache(maxsize=8)
def kf_lookup_tables(svp1: float, svp2: float, svp3: float, svpt0: float):
    """KF_LUTAB (module_cu_kfeta.F:3174-3301) in float64 numpy.

    Returns ``(ttab, qstab, the0k, alu, plutop, rdpr, rdthk)``; ``ttab[it, kp]``
    and ``qstab[it, kp]`` are 0-based versions of TTAB(IT,KP), QSTAB(IT,KP).
    The secant iteration runs vectorised over the pressure index with the
    oracle's per-element exit, accumulation order preserved.
    """
    aliq = svp1 * _KPA_TO_PA
    bliq = svp2
    cliq = svp2 * svpt0
    dliq = svp3
    plutop = _LUT_PLUTOP_PA
    rdthk = 1.0 / _LUT_DTH_K
    dpr = (_LUT_PBOT_PA - plutop) / float(_KFNP - 1)
    rdpr = 1.0 / dpr

    def thetae(t, p):
        es = aliq * np.exp((bliq * t - cliq) / (t - dliq))
        qs = _EPS_KF * es / (p - es)
        pi = (_P00_PA / p) ** (_EXN_KAPPA * (1.0 - _EXN_Q * qs))
        return t * pi * np.exp((_THE_C1 / t - _THE_C2) * qs * (1.0 + _THE_C3 * qs)), qs

    pvec = np.empty(_KFNP)
    p = plutop - dpr
    for kp in range(_KFNP):
        p = p + dpr
        pvec[kp] = p
    the0k, _ = thetae(np.full(_KFNP, _LUT_TMIN_K), pvec)
    ttab = np.empty((_KFNT, _KFNP))
    qstab = np.empty((_KFNT, _KFNP))
    thes = the0k - _LUT_DTH_K
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for it in range(_KFNT):
            thes = thes + _LUT_DTH_K
            tgues = np.full(_KFNP, _LUT_TMIN_K) if it == 0 else ttab[it - 1].copy()
            thgues, qs = thetae(tgues, pvec)
            f0 = thgues - thes
            t1 = tgues - 0.5 * f0
            t0 = tgues.copy()
            act = np.ones(_KFNP, dtype=bool)
            for _ in range(_LUT_NITER):
                thtgs, qs_new = thetae(t1, pvec)
                qs = np.where(act, qs_new, qs)
                f1 = thtgs - thes
                upd = act & ~(np.abs(f1) < _LUT_TOLER_K)
                dt = f1 * (t1 - t0) / (f1 - f0)
                t0 = np.where(upd, t1, t0)
                f0 = np.where(upd, f1, f0)
                t1 = np.where(upd, t1 - dt, t1)
                act = upd
            ttab[it] = t1
            qstab[it] = qs
    alu = np.empty(_ALU_N)
    a1 = _ALU_ASTRT - _ALU_AINC
    for i in range(_ALU_N):
        a1 = a1 + _ALU_AINC
        alu[i] = np.log(a1)
    return ttab, qstab, the0k, alu, plutop, rdpr, rdthk


def _pad1(a):
    return np.concatenate([[2.0 * a[0] - a[1]], a])


def _pad2(a):
    b = np.concatenate([(2.0 * a[0] - a[1])[None, :], a], axis=0)  # kfsw: S
    return np.concatenate([(2.0 * b[:, 0] - b[:, 1])[:, None], b], axis=1)  # kfsw: S


def _kf_tables(kc: KFConstants, dtype, sm: KFSmoothing) -> _KFTables:
    """The oracle's tables, their padded copies (one leading node per axis, the
    linear extrapolation of the first interval, bilinear at the corner) and
    the widths."""
    ttab, qstab, the0k, alu, plutop, rdpr, rdthk = kf_lookup_tables(
        float(kc.svp1), float(kc.svp2), float(kc.svp3), float(kc.svpt0))  # kfsw: S
    return _KFTables(*(jnp.asarray(x, dtype) for x in (
        ttab, qstab, the0k, alu, plutop, rdpr, rdthk, _pad2(ttab), _pad2(qstab),
        _pad1(the0k), _pad1(alu))), sm)


# --------------------------------------------------------------------------
# Callees (scalars; the column driver vmaps)
# --------------------------------------------------------------------------

def _hat_c1(t, w):
    """The linear-interpolation hat convolved with a box of half-width ``w``:
    C1, compact support |t| < 1 + w, the hats sum to 1 (and reproduce linear
    functions) at every point."""
    return (smooth_max_c1(t + 1.0, 0.0, 2.0 * w) - 2.0 * smooth_max_c1(t, 0.0, 2.0 * w)
            + smooth_max_c1(t - 1.0, 0.0, 2.0 * w))  # kfsw: F1,F2


def _stencil_c1(x, n_nodes, w):
    """Padded-table indices (node -1 .. n_nodes-1 -> 0 .. n_nodes, clamped:
    constant beyond both ends) and weights of the 4 nodes around ``x``."""
    x = jnp.asarray(x)
    i0 = jnp.floor(x).astype(jnp.int32)  # kfsw: F1,F2
    j = i0[..., None] + jnp.arange(-1, 3, dtype=jnp.int32)  # kfsw: F1,F2
    wt = _hat_c1(x[..., None] - j.astype(x.dtype), w)  # kfsw: F1,F2
    return jnp.clip(j, -1, n_nodes - 1) + 1, wt  # kfsw: F1,F2


def _step_c(x, w):
    """C1 step centred on 0, band |x| < w/2: 0 below, 1 above (RULE 5)."""
    return smoothstep_cubic(0.5 + x / w)  # kfsw: C-KLCLg,C-L5g


def _cap_c1(x, cap, w):
    """min(x, cap) (w = 0, the oracle) or its one-sided C1 form: never above
    ``cap``, exactly ``x`` below ``cap - w`` (E1 Q0 <= QES and, negated, Q0 >=
    1e-6; E2 TLCL <= TMIX)."""
    if w:  # kfsw: S
        return cap - positive_part_c1(cap - x, 1.0 / w)  # kfsw: E1,E2
    return jnp.minimum(x, cap)  # kfsw: E1,E2


def _gather_at_height(f, Z0, z, kk, valid, LC, w):
    """f at height z on the 1-based padded column (C-KLCLg): soft interval
    weights of the oracle's linear interpolants on segments (Z0[k-1], Z0[k]],
    k = LC..KX, which sum to 1; the first segment extends down and the last up,
    as the oracle extrapolates.  Width ``w`` > 0 [m]."""
    KX = kk.shape[0] - 2
    seg = valid & (kk >= LC)  # kfsw: C-KLCLg
    zlo = jnp.concatenate([Z0[:1], Z0[:-1]])
    lo = jnp.where(kk == LC, 1.0, _step_c(z - zlo, w))  # kfsw: C-KLCLg
    hi = jnp.where(kk == KX, 0.0, _step_c(z - Z0, w))  # kfsw: S
    frac = (z - zlo) / _sd(Z0 - zlo, seg & (Z0 != zlo))  # kfsw: C-KLCLg
    fl = jnp.concatenate([f[:1], f[:-1]])
    return jnp.sum(jnp.where(seg, lo - hi, 0.0) * (fl + (f - fl) * frac))  # kfsw: C-KLCLg


def _l5_weights(P0, kk, valid, w):
    """Soft one-hot of L5, the last level with P0 >= 0.5 P0(1) (C-L5g); sums to
    1 on a pressure-decreasing column.  Width ``w`` > 0 [Pa]."""
    KX = kk.shape[0] - 2
    on = _step_c(P0 - _L5_FRAC * P0[1], w)
    return jnp.where(valid, on - jnp.where(kk == KX, 0.0, jnp.roll(on, -1)), 0.0)  # kfsw: C-L5g


def _interp_c1(pad, x, w):
    """1-D lookup on the padded table, knots and edges smoothed (width ``w``)."""
    idx, wt = _stencil_c1(x, pad.shape[0] - 1, w)
    return jnp.sum(pad[idx] * wt, axis=-1)  # kfsw: F1,F2


def _tlog_lookup(lut, a1):
    """TLOG = ln(a1) from the ALU table (l.919-926, l.3067-3074)."""
    if lut.sm.lut_knot_width:  # kfsw: S
        return _interp_c1(lut.alu_pad, (a1 - _ALU_ASTRT) / _ALU_AINC, lut.sm.lut_knot_width)
    alu = lut.alu
    # Outside the coordinates where the oracle reads inside ALU (-1 < tp < N-1)
    # WRF reads out of bounds; clamp the coordinate there (edge-continuous).
    a1 = jnp.clip(a1, _ALU_ASTRT - _ALU_AINC, _ALU_ASTRT + (_ALU_N - 1) * _ALU_AINC)  # kfsw: F1
    tp = (a1 - _ALU_ASTRT) / _ALU_AINC
    i0 = jnp.clip(_ftrunc_i(tp), 0, _ALU_N - 2)  # kfsw: F1
    value = i0.astype(a1.dtype) * _ALU_AINC + _ALU_ASTRT  # kfsw: F1
    aintrp = (a1 - value) / _ALU_AINC
    return aintrp * alu[i0 + 1] + (1 - aintrp) * alu[i0]  # kfsw: F1


def _tsat_from_dewpoint(tdpt, t):
    return tdpt - (_TSAT_A + _TSAT_B * (tdpt - _T00_K) - _TSAT_C * (t - _T00_K)) * (t - tdpt)


def _exner(p, q):
    return (_P00_PA / p) ** (_EXN_KAPPA * (1.0 - _EXN_Q * q))


def _thetae_bolton(t, p, q, tsat):
    return t * _exner(p, q) * jnp.exp((_THE_C1 / tsat - _THE_C2) * q * (1.0 + _THE_C3 * q))


def _envirtht(lut, ab, p1, t1, q1):
    """ENVIRTHT (l.3044-3081)."""
    aliq, bliq, cliq, dliq = ab
    ee = q1 * p1 / (_EPS_KF + q1)
    tlog = _tlog_lookup(lut, ee / aliq)
    tdpt = (cliq - dliq * tlog) / (bliq - tlog)
    tsat = _tsat_from_dewpoint(tdpt, t1)
    tht = t1 * _exner(p1, q1)
    return tht * jnp.exp((_THE_C1 / tsat - _THE_C2) * q1 * (1.0 + _THE_C3 * q1))


def _tab_interp(lut, p, thes):
    """Bilinear TTAB/QSTAB lookup shared by TPMIX2 and TPMIX2DD."""
    ttab, qstab, the0k, _alu, plutop, rdpr, rdthk = lut[:7]
    w = lut.sm.lut_knot_width
    if w:  # kfsw: S
        xp = (p - plutop) * rdpr
        xt = (thes - _interp_c1(lut.the0k_pad, xp, w)) * rdthk
        ip, wp = _stencil_c1(xp, _KFNP, w)
        it, wt = _stencil_c1(xt, _KFNT, w)
        wgt = wt[..., :, None] * wp[..., None, :]  # kfsw: F2
        it, ip = it[..., :, None], ip[..., None, :]  # kfsw: F2
        return (jnp.sum(lut.ttab_pad[it, ip] * wgt, axis=(-2, -1)),  # kfsw: F2
                jnp.sum(lut.qstab_pad[it, ip] * wgt, axis=(-2, -1)))  # kfsw: F2
    # The oracle reads inside its tables for -1 < coordinate < N-1 (Fortran INT
    # truncates toward zero, so (-1, 0) extrapolates from the first interval);
    # beyond that WRF reads out of bounds.  There the coordinate is clamped
    # (edge-continuous); inside, the arithmetic is the oracle's.
    tp = jnp.clip((p - plutop) * rdpr, -1.0, _KFNP - 1.0)  # kfsw: F2
    ip = jnp.clip(_ftrunc_i(tp), 0, _KFNP - 2)  # kfsw: F2
    qq = tp - ip.astype(tp.dtype)  # kfsw: F2
    bth = (the0k[ip + 1] - the0k[ip]) * qq + the0k[ip]  # kfsw: F2
    tth = jnp.clip((thes - bth) * rdthk, -1.0, _KFNT - 1.0)  # kfsw: F2
    it = jnp.clip(_ftrunc_i(tth), 0, _KFNT - 2)  # kfsw: F2
    pp = tth - it.astype(tth.dtype)  # kfsw: F2
    t00, t10 = ttab[it, ip], ttab[it + 1, ip]  # kfsw: F2
    t01, t11 = ttab[it, ip + 1], ttab[it + 1, ip + 1]  # kfsw: F2
    q00, q10 = qstab[it, ip], qstab[it + 1, ip]  # kfsw: F2
    q01, q11 = qstab[it, ip + 1], qstab[it + 1, ip + 1]  # kfsw: F2
    temp = t00 + (t10 - t00) * pp + (t01 - t00) * qq + (t00 - t10 - t01 + t11) * pp * qq
    qs = q00 + (q10 - q00) * pp + (q01 - q00) * qq + (q00 - q10 - q01 + q11) * pp * qq
    return temp, qs


def _tpmix2(lut, p, thes, qu, qliq, qice, xlv0, xlv1):
    """TPMIX2 (l.2695-2815): returns tu, qu, qliq, qice, qnewlq, qnewic.

    Smooth form (``tpmix_width_kgkg`` > 0): condensation ``C = (-dq)+``,
    evaporation ``E = min(max(dq, 0), qtot)`` (a C1 form never above it and
    zero while supersaturated) and
    deficit ``D = (dq - qtot)+`` with one-sided C1 positive parts (exact beyond
    the width); the oracle's three branches are its hard limit.
    Water is conserved exactly: the vapour gains the condensate lost.  KNOWN
    DIFFERENCE, all at the oracle's 1e-10 kg/kg guard: the oracle's enough
    branch creates ``E*1e-10/(qtot+1e-10)`` of water (qu reaches qs while the
    condensate loses only ``E*qtot/(qtot+1e-10)``), so the smooth qu sits that
    much below qs; where the oracle evaporates all condensate the smooth form
    keeps ``qliq*1e-10/(qtot+1e-10)`` of it; the dry branch (qtot < 1e-10 keeps
    qu) has no counterpart."""
    temp, qs = _tab_interp(lut, p, thes)
    dq = qs - qu
    w = lut.sm.tpmix_width_kgkg
    if w:  # kfsw: S
        qtot = qliq + qice
        # p = one-sided C1 positive part (0 below 0, <= max(y, 0), exact beyond
        # w); smooth_max_c1(y, 0, w) >= max(y, 0), exact for |y| >= w/2.
        # Condensation p(-dq) <= the excess vapour.  Evaporation
        # p(a - smooth_max_c1(a - qtot)), a = p(dq), lies in [0, min(a, qtot)]:
        # none while supersaturated, never more than the condensate; hard
        # limit min(max(dq, 0), qtot).  Vapour and condensate stay >= 0.
        s = 1.0 / w
        cond = positive_part_c1(-dq, s)  # kfsw: D7a
        dfc = positive_part_c1(dq - qtot, s)  # kfsw: D7a
        a = positive_part_c1(dq, s)  # kfsw: D7a
        evap = positive_part_c1(a - smooth_max_c1(a - qtot, 0.0, w), s)  # kfsw: D7a
        rll = xlv0 - xlv1 * temp
        cpp = _TPMIX_CP * (1.0 + _TPMIX_CPQ * qu)
        lost = evap / (qtot + _TPMIX_QEPS)  # fraction of the condensate evaporated
        # vapour gains exactly the condensate lost: water is conserved
        return (temp + rll * (dfc / (1.0 + dfc)) / cpp, qu + qtot * lost - cond,
                qliq * (1.0 - lost), qice * (1.0 - lost), cond, jnp.zeros_like(cond))
    sat = dq <= 0.0  # kfsw: D7a
    qnew_sat = qu - qs
    qtot = qliq + qice
    enough = qtot >= dq  # kfsw: D7a
    # enough condensate: evaporate to saturation at the wet-bulb temperature
    qliq_e = qliq - dq * qliq / (qtot + _TPMIX_QEPS)
    qice_e = qice - dq * qice / (qtot + _TPMIX_QEPS)
    rll = xlv0 - xlv1 * temp
    cpp = _TPMIX_CP * (1.0 + _TPMIX_CPQ * qu)
    dry = qtot < _TPMIX_QEPS  # kfsw: D7a
    temp_dry = temp + rll * (dq / (1.0 + dq)) / cpp
    temp_part = temp + rll * ((dq - qtot) / (1 + dq - qtot)) / cpp
    sub = ~sat
    tu = jnp.where(sub & ~enough, jnp.where(dry, temp_dry, temp_part), temp)  # kfsw: D7a
    qu_out = jnp.where(sat, qs, jnp.where(enough, qs, jnp.where(dry, qu, qu + qtot)))  # kfsw: D7a
    qliq_out = jnp.where(sub & enough, qliq_e, jnp.where(sub & ~dry, 0.0, qliq))  # kfsw: D7a
    qice_out = jnp.where(sub & enough, qice_e, jnp.where(sub & ~dry, 0.0, qice))  # kfsw: D7a
    qnewlq = jnp.where(sat, qnew_sat, 0.0)  # kfsw: D7a
    return tu, qu_out, qliq_out, qice_out, qnewlq, jnp.zeros_like(qnewlq)


def _tpmix2dd(lut, p, thes):
    """TPMIX2DD (l.2977-3041): saturated temperature and mixing ratio."""
    return _tab_interp(lut, p, thes)


def _dtfrznew(ab, tu, p, qu, qfrz, qice):
    """DTFRZNEW (l.2817-2860): returns tu, theteu, qu, qice."""
    aliq, bliq, cliq, dliq = ab
    rlc = _DTFRZ_RLC0 - _DTFRZ_RLC1 * (tu - _T00_K)
    rls = _DTFRZ_RLS0 - _DTFRZ_RLS1 * (tu - _T00_K)
    rlf = rls - rlc
    cpp = _TPMIX_CP * (1.0 + _TPMIX_CPQ * qu)
    a = (cliq - bliq * dliq) / ((tu - dliq) * (tu - dliq))
    dtfrz = rlf * qfrz / (cpp + rls * qu * a)
    tu = tu + dtfrz
    es = aliq * jnp.exp((bliq * tu - cliq) / (tu - dliq))
    qs = es * _EPS_KF / (p - es)
    dqevap = qs - qu
    qice = qice - dqevap
    qu = qu + dqevap
    pii = (_P00_PA / p) ** (_EXN_KAPPA * (1.0 - _EXN_Q * qu))
    theteu = tu * pii * jnp.exp((_THE_C1 / tu - _THE_C2) * qu * (1.0 + _THE_C3 * qu))
    return tu, theteu, qu, qice


def _safe_sqrt(x):
    ok = x > 0.0  # kfsw: H
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, x, 1.0)), 0.0)  # kfsw: H


def _condload(qliq, qice, wtw, dz, boterm, enterm, qnewlq, qnewic, g, sm):
    """CONDLOAD (l.2863-2927): returns qliq, qice, wtw, qlqout, qicout.

    Smooth form: C1 square roots (``condload_sqrt_width_m2s2``; the G1 clip is
    the root's zero below 0) and cubic ramps outside the WTW floor
    (``wtw_floor_width_m2s2``).  KNOWN DIFFERENCE: the ``wavg != 0`` guard stays
    hard; it cannot bind (the updraft exits before WTW drops below 1e-3)."""
    qtot = qliq + qice
    qnew = qnewlq + qnewic
    qest = 0.5 * (qtot + qnew)
    g1 = wtw + boterm - enterm - 2.0 * g * dz * qest / _KE_DIV
    ws = sm.condload_sqrt_width_m2s2
    if ws:  # kfsw: S
        wavg = 0.5 * (sqrt_c1(wtw, ws) + sqrt_c1(g1, ws))  # kfsw: E11c
    else:
        g1 = jnp.where(g1 < 0.0, 0.0, g1)  # kfsw: D7b
        wavg = 0.5 * (_safe_sqrt(wtw) + _safe_sqrt(g1))  # kfsw: E11c
    conv = _RATE * dz / jnp.where(wavg != 0.0, wavg, 1.0)  # kfsw: D7c
    ratio3 = qnewlq / (qnew + _CL_EPS)
    qtot = qtot + _CL_FRESH * qnew
    oldq = qtot
    ratio4 = (_CL_FRESH * qnewlq + qliq) / (qtot + _CL_EPS)
    qtot = qtot * jnp.exp(-conv)
    dq = oldq - qtot
    qlqout = ratio4 * dq
    qicout = (1.0 - ratio4) * dq
    pptdrg = 0.5 * (oldq + qtot - _CL_DRAG_EXCL * qnew)
    wtw = wtw + boterm - enterm - 2.0 * g * dz * pptdrg / _KE_DIV
    wf = sm.wtw_floor_width_m2s2
    if wf:  # kfsw: S
        inside = 1.0 - smoothstep_cubic((jnp.abs(wtw) - _WTW_FLOOR) / wf)  # kfsw: D7d
        wtw = wtw + (_WTW_FLOOR - wtw) * inside
    else:
        wtw = jnp.where(jnp.abs(wtw) < _WTW_FLOOR, _WTW_FLOOR, wtw)  # kfsw: D7d
    qliq = ratio4 * qtot + ratio3 * _CL_RETAIN * qnew
    qice = (1.0 - ratio4) * qtot + (1.0 - ratio3) * _CL_RETAIN * qnew
    return qliq, qice, wtw, qlqout, qicout


def prof5(eq, width=0.0):
    """PROF5 (l.2930-2974): Gaussian-mixing entrainment/detrainment fractions.

    ``width`` > 0: the two polynomial branches (``y >= 0`` / ``y < 0``, which
    differ by ~1e-5 at ``y = 0``) are each continued analytically through 0
    and blended by a cubic smoothstep over ``|y| < width``; 0 is the oracle."""
    if width:  # kfsw: S
        return _prof5_c1(eq, width)
    x = (eq - 0.5) / _P5_SIGMA  # noqa: F841  (computed but unused in the oracle)
    y = _P5_Y_SCALE * eq - _P5_Y_OFF
    ey = jnp.exp(y * y / (-2))
    e45 = jnp.exp(jnp.asarray(_P5_E45_ARG, eq.dtype))
    t2 = 1.0 / (1.0 + _P5_P * jnp.abs(y))  # kfsw: D7e
    t1 = _P5_T1
    c1 = _P5_A1 * t1 + _P5_A2 * t1 * t1 + _P5_A3 * t1 * t1 * t1
    c2 = _P5_A1 * t2 + _P5_A2 * t2 * t2 + _P5_A3 * t2 * t2 * t2
    s = _P5_SIGMA
    ee_pos = s * (0.5 * (_P5_SQRT2P - e45 * c1 - ey * c2) + s * (e45 - ey)) - e45 * eq * eq / 2.0
    ud_pos = s * (0.5 * (ey * c2 - e45 * c1) + s * (e45 - ey)) - e45 * (0.5 + eq * eq / 2.0 - eq)
    ee_neg = s * (0.5 * (ey * c2 - e45 * c1) + s * (e45 - ey)) - e45 * eq * eq / 2.0
    ud_neg = s * (0.5 * (_P5_SQRT2P - e45 * c1 - ey * c2) + s * (e45 - ey)) - e45 * (0.5 + eq * eq / 2.0 - eq)
    pos = y >= 0.0  # kfsw: D7e
    ee = jnp.where(pos, ee_pos, ee_neg)  # kfsw: D7e
    ud = jnp.where(pos, ud_pos, ud_neg)  # kfsw: D7e
    return ee / _P5_FE, ud / _P5_FE


def _prof5_c1(eq, width):
    y = _P5_Y_SCALE * eq - _P5_Y_OFF
    ey = jnp.exp(y * y / (-2))
    e45 = jnp.exp(jnp.asarray(_P5_E45_ARG, eq.dtype))
    t1 = _P5_T1
    c1 = _P5_A1 * t1 + _P5_A2 * t1 * t1 + _P5_A3 * t1 * t1 * t1
    s = _P5_SIGMA

    def c2(yy):  # |y| replaced by the branch's own sign, clamped clear of the pole
        t2 = 1.0 / (1.0 + _P5_P * yy)
        return _P5_A1 * t2 + _P5_A2 * t2 * t2 + _P5_A3 * t2 * t2 * t2

    c2p = c2(jnp.maximum(y, -width))  # kfsw: D7e
    c2n = c2(jnp.maximum(-y, -width))  # kfsw: D7e
    ee_pos = s * (0.5 * (_P5_SQRT2P - e45 * c1 - ey * c2p) + s * (e45 - ey)) - e45 * eq * eq / 2.0
    ud_pos = s * (0.5 * (ey * c2p - e45 * c1) + s * (e45 - ey)) - e45 * (0.5 + eq * eq / 2.0 - eq)
    ee_neg = s * (0.5 * (ey * c2n - e45 * c1) + s * (e45 - ey)) - e45 * eq * eq / 2.0
    ud_neg = (s * (0.5 * (_P5_SQRT2P - e45 * c1 - ey * c2n) + s * (e45 - ey))
              - e45 * (0.5 + eq * eq / 2.0 - eq))
    b = smoothstep_cubic((y + width) / (2.0 * width))  # kfsw: D7e
    return (ee_neg + b * (ee_pos - ee_neg)) / _P5_FE, (ud_neg + b * (ud_pos - ud_neg)) / _P5_FE


# --------------------------------------------------------------------------
# Column driver
# --------------------------------------------------------------------------

def _w(m, new, old):
    return jnp.where(m, new, old)  # kfsw: H


def _put(a, i, v, m):
    """a(i) = v where m (scalar mask), 1-based padded arrays."""
    return a.at[i].set(jnp.where(m, v, a[i]))  # kfsw: H


def _sd(x, ok):
    """Denominator that is ``x`` where it is used and 1 elsewhere (keeps the
    unselected branch finite so gradients through ``jnp.where`` stay finite)."""
    return jnp.where(ok, x, 1.0)  # kfsw: H


def _scan_levels(body, carry, lo, hi, reverse=False):
    """Run ``body(carry, k)`` for static k = lo..hi (inclusive), ascending or
    descending; the body masks itself."""
    ks = jnp.arange(lo, hi + 1, dtype=jnp.int32)  # kfsw: S
    if reverse:  # kfsw: S
        ks = ks[::-1]

    def f(c, k):
        return body(c, k), None  # kfsw: S

    carry, _ = jax.lax.scan(f, carry, ks)
    return carry


def _column(u0, v0, t0, qv0, p0, dzq, w0, rhoe, *, dt, dx, kc, lut, prm, trigger,
            warm_rain, f_qi, f_qs, cudt):
    """KF_eta_PARA for one column.  Inputs are 0-based bottom-up arrays (KX)."""
    dtype = t0.dtype
    KX = t0.shape[0]
    KL = KX
    NP = KX + 2
    i32 = jnp.int32  # kfsw: S
    zero = jnp.zeros((), dtype)
    one = jnp.ones((), dtype)

    def pad(a):  # 1-based with replicated edge pads
        return jnp.concatenate([a[:1], a, a[-1:]])

    U0, V0, T0, QV0, P0, DZQ, W0A, RHOE = (pad(x) for x in (u0, v0, t0, qv0, p0, dzq, w0, rhoe))
    G, CP, R, XLV0, XLV1 = kc.g, kc.cp, kc.r, kc.xlv0, kc.xlv1
    ALIQ = kc.svp1 * _KPA_TO_PA
    BLIQ = kc.svp2
    CLIQ = kc.svp2 * kc.svpt0
    DLIQ = kc.svp3
    ab = (ALIQ, BLIQ, CLIQ, DLIQ)
    GDRY = -G / CP
    DXSQ = dx * dx
    DPMIN = prm["dpmin"]
    kk = jnp.arange(NP, dtype=i32)  # kfsw: S
    valid = (kk >= 1) & (kk <= KX)  # kfsw: S

    # --- sounding (l.761-797) ---
    ES = ALIQ * jnp.exp((BLIQ * T0 - CLIQ) / (T0 - DLIQ))
    QES = _EPS_KF * ES / (P0 - ES)
    sm = lut.sm
    Q0 = _cap_c1(QV0, QES, sm.q0_clip_width_kgkg)
    Q0 = -_cap_c1(-Q0, -_Q0_FLOOR, sm.q0_floor_width_kgkg)
    RH = Q0 / QES
    TV0 = T0 * (1.0 + _TV_COEF * Q0)
    DP = RHOE * G * DZQ
    L5 = jnp.max(jnp.where(valid & (P0 >= _L5_FRAC * P0[1]), kk, 0))  # kfsw: C-L5
    P300 = P0[1] - _P300_DEPTH_PA
    LLFC = jnp.max(jnp.where(valid & (P0 >= P300), kk, 0))  # kfsw: C-L5

    def zbody(z, k):
        zk = jnp.where(k == 1, 0.5 * DZQ[1], z[k - 1] + 0.5 * (DZQ[k] + DZQ[k - 1]))  # kfsw: S
        return z.at[k].set(zk)  # kfsw: S

    Z0 = _scan_levels(zbody, jnp.zeros(NP, dtype), 1, KX)
    DZA = jnp.zeros(NP, dtype).at[1:KX].set(Z0[2:KX + 1] - Z0[1:KX])  # kfsw: S

    # --- USL candidates every 15 hPa (l.807-816) ---
    def kcbody(c, k):
        ncheck, pm15, kcheck = c
        hit = (k <= LLFC) & (P0[k] < pm15)  # kfsw: C-KCHECK
        kcheck = _put(kcheck, ncheck + 1, k, hit)  # kfsw: C-KCHECK
        return (ncheck + hit.astype(i32), _w(hit, pm15 - _PM15_PA, pm15), kcheck)  # kfsw: C-KCHECK

    # kfsw: S
    NCHECK, _, KCHECK = _scan_levels(
        kcbody, (jnp.asarray(1, i32), P0[1] - _PM15_PA, jnp.zeros(NP, i32).at[1].set(1)), 2, KX)

    # --- persistent state of the USL loop ---
    zf = jnp.zeros(NP, dtype)
    S = dict(
        THETEU=zf, WU=zf, UMF=zf, RATIO2=zf, UER=zf, TU=zf, TVU=zf, QU=zf, EQFRC=zf,
        QLIQ=zf, QICE=zf, QLQOUT=zf, QICOUT=zf, DETLQ=zf, DETIC=zf, PPTLIQ=zf,
        PPTICE=zf, THETEE=zf, TVQU=zf, UDR=zf, DILFRC=jnp.ones(NP, dtype), QDT=zf,
        CLDHGT=zf,
    )
    sc = dict(
        NU=jnp.asarray(0, i32), NUCHM=jnp.asarray(0, i32), ISHALL=jnp.asarray(0, i32),
        FBFRC=zero, done=jnp.asarray(False), ret=jnp.asarray(False),
        LC=jnp.asarray(1, i32), KPBL=jnp.asarray(1, i32), DPTHMX=one, TMIX=zero,
        QMIX=zero, ZMIX=zero, PMIX=one, TLCL=zero, TVLCL=one, ZLCL=zero,
        KLCL=jnp.asarray(2, i32), K=jnp.asarray(1, i32), DLP=zero, TENV=zero, QENV=zero,
        TVEN=one, WKL=zero, WKLCL=zero, DTLCL=zero, DTRH=zero, WLCL=one, PLCL=one,
        LCL=jnp.asarray(2, i32), LET=jnp.asarray(2, i32), RAD=one, AU0=one,
        VMFLCL=one, UPOLD=one, UPNEW=one, ABE=zero, TRPPT=zero,
        LTOP=jnp.asarray(2, i32), CHMIN=zero, IFLAG=jnp.asarray(0, i32),
    )  # kfsw: S

    def evaluate(S, sc, NU, act, sc_nuchm):
        """One pass of the usl loop for candidate NU (l.846-1419)."""
        LC = KCHECK[NU]  # kfsw: B2
        # 50-hPa mixed layer (l.856-884)
        def mbody(c, k):
            dpth, ts, qs, zs, ps, stopped, kpbl = c
            a = (k >= LC) & ~stopped  # kfsw: C-KPBL,C-KCHECK
            dpth = _w(a, dpth + DP[k], dpth)  # kfsw: C-KPBL
            ts = _w(a, ts + DP[k] * T0[k], ts)  # kfsw: C-KPBL
            qs = _w(a, qs + DP[k] * Q0[k], qs)  # kfsw: C-KPBL
            zs = _w(a, zs + DP[k] * Z0[k], zs)  # kfsw: C-KPBL
            ps = _w(a, ps + DP[k] * P0[k], ps)  # kfsw: C-KPBL
            kpbl = _w(a, k, kpbl)  # kfsw: C-KPBL
            stopped = stopped | (a & (dpth > DPMIN))  # kfsw: C-KPBL
            return (dpth, ts, qs, zs, ps, stopped, kpbl)

        DPTHMX, TS, QSM, ZS, PS, _, KPBL = _scan_levels(
            mbody, (zero, zero, zero, zero, zero, jnp.asarray(False), LC), 1, KX)
        ret1 = DPTHMX < DPMIN  # kfsw: A2
        DPTHMX_s = _sd(DPTHMX, ~ret1)  # kfsw: A2
        TMIX = TS / DPTHMX_s
        QMIX = QSM / DPTHMX_s
        ZMIX = ZS / DPTHMX_s
        PMIX = PS / DPTHMX_s
        EMIX = QMIX * PMIX / (_EPS_KF + QMIX)
        tlog = _tlog_lookup(lut, EMIX / ALIQ)
        TDPT = (CLIQ - DLIQ * tlog) / (BLIQ - tlog)
        TLCL = _tsat_from_dewpoint(TDPT, TMIX)
        TLCL = _cap_c1(TLCL, TMIX, sm.tlcl_cap_width_k)
        TVLCL = TLCL * (1.0 + _TV_COEF * QMIX)
        ZLCL = ZMIX + (TLCL - TMIX) / GDRY
        hitz = valid & (kk >= LC) & (ZLCL <= Z0)  # kfsw: C-KLCL,C-KCHECK
        # kfsw: C-KLCL
        KLCL = jnp.where(jnp.any(hitz), jnp.argmax(hitz).astype(i32), jnp.asarray(KL, i32))
        ret2 = ZLCL > Z0[KL]  # kfsw: A3
        K = KLCL - 1
        DLP = (ZLCL - Z0[K]) / _sd(Z0[KLCL] - Z0[K], Z0[KLCL] != Z0[K])  # kfsw: C-KLCL
        wz = sm.lcl_gather_width_m
        if wz:  # kfsw: S
            def gz(f):
                return _gather_at_height(f, Z0, ZLCL, kk, valid, LC, wz)
        else:
            def gz(f):
                return f[K] + (f[KLCL] - f[K]) * DLP  # kfsw: C-KLCLg
        TENV = gz(T0)
        QENV = gz(Q0)
        TVEN = TENV * (1.0 + _TV_COEF * QENV)
        WKLCL = jnp.where(ZLCL < prm["wklcl_zref"], prm["wklcl_ref"] * ZLCL / prm["wklcl_zref"],
                          prm["wklcl_ref"])  # kfsw: E3
        WKL = gz(W0A) * dx / _WKL_DX_REF_M - WKLCL
        small = WKL < _WKL_MIN  # kfsw: D1
        # kfsw: D1
        DTLCL = jnp.where(small, 0.0, prm["dtlcl_coeff"] * _sd(WKL, ~small) ** prm["dtlcl_exponent"])
        if trigger == 3:  # kfsw: S
            QSLCL = gz(QES)
            RHLCL = QENV / QSLCL
            DQSSDT = QMIX * (CLIQ - BLIQ * DLIQ) / ((TLCL - DLIQ) * (TLCL - DLIQ))
            mid = (RHLCL >= prm["rh_u00"]) & (RHLCL <= prm["rh_max"])  # kfsw: D2
            # kfsw: D2
            DTRH = jnp.where(
                mid, prm["rh_slope"] * (RHLCL - prm["rh_u00"]) * QMIX / DQSSDT,
                jnp.where(RHLCL > prm["rh_max"], (1.0 / RHLCL - 1.0) * QMIX / DQSSDT, 0.0))
        else:
            DTRH = zero
        go0 = act & ~ret1 & ~ret2
        ret = act & (ret1 | ret2)
        trig = ~(TLCL + DTLCL + DTRH < TENV)  # kfsw: A1
        go = go0 & trig  # updraft computed for this candidate

        # --- updraft initialisation (l.1034-1121) ---
        th_k = _envirtht(lut, ab, PMIX, TMIX, QMIX)
        DTTOT = DTLCL + DTRH
        big = DTTOT > _DTTOT_MIN_K  # kfsw: D3
        GDT = 2.0 * G * DTTOT * _GDT_DEPTH_M / TVEN
        # kfsw: D3,E4,E11
        WLCL = jnp.where(big, jnp.minimum(1.0 + 0.5 * _safe_sqrt(GDT), _WLCL_MAX), 1.0)
        PLCL = gz(P0)
        WTW = WLCL * WLCL
        TVLCL = TLCL * (1.0 + _TV_COEF * QMIX)
        RHOLCL = PLCL / (R * TVLCL)
        LCL = KLCL
        RAD = jnp.where(WKL < 0.0, prm["rad_min"],
                        jnp.where(WKL > prm["rad_wkl_ref"], prm["rad_max"],
                                  prm["rad_min"] + (prm["rad_max"] - prm["rad_min"]) * WKL
                                  / prm["rad_wkl_ref"]))  # kfsw: E5
        AU0 = _AU0_FRAC * DXSQ
        VMFLCL = RHOLCL * AU0
        u_vmf = VMFLCL
        A = dict(S)
        A["THETEU"] = _put(A["THETEU"], K, th_k, go)  # kfsw: A1,C-KLCL
        A["WU"] = _put(A["WU"], K, WLCL, go)  # kfsw: A1,C-KLCL
        A["UMF"] = _put(A["UMF"], K, VMFLCL, go)  # kfsw: A1,C-KLCL
        for nm, val in (("RATIO2", zero), ("UER", zero), ("TU", TLCL), ("TVU", TVLCL),
                        ("QU", QMIX), ("EQFRC", one), ("QLIQ", zero), ("QICE", zero),
                        ("QLQOUT", zero), ("QICOUT", zero), ("DETLQ", zero), ("DETIC", zero),
                        ("PPTLIQ", zero), ("PPTICE", zero)):
            A[nm] = _put(A[nm], K, val, go)  # kfsw: A1,C-KLCL

        # --- updraft loop DO NK=K,KL-1 (l.1122-1320) ---
        u = dict(WTW=WTW, REI=zero, UPOLD=VMFLCL, UPNEW=VMFLCL, ABE=zero, TRPPT=zero,
                 EE1=one, UD1=zero, DILBE=zero, TTEMP=jnp.asarray(_TTFRZ_K, dtype),
                 IFLAG=jnp.asarray(0, i32), LET=LCL, exited=jnp.asarray(False),
                 LTOP=jnp.asarray(KL, i32))  # kfsw: S

        def ubody(c, NK):
            A, u = c
            a = go & ~u["exited"] & (NK >= K)  # kfsw: C-LTOP
            NK1 = NK + 1
            ratio2 = A["RATIO2"][NK]  # kfsw: S
            theteu = A["THETEU"][NK]  # kfsw: S
            # kfsw: S
            tu, qu, qliq, qice, qnl, qni = _tpmix2(lut, P0[NK1], theteu, A["QU"][NK],
                                                   A["QLIQ"][NK], A["QICE"][NK], XLV0, XLV1)
            # glaciation (l.1141-1161)
            frz = tu <= _TTFRZ_K  # kfsw: D7g
            part = tu > _TBFRZ_K  # kfsw: D7g
            ttemp0 = jnp.where(part & (u["TTEMP"] > _TTFRZ_K), _TTFRZ_K, u["TTEMP"])  # kfsw: D7g
            # kfsw: D7g
            frc1 = jnp.where(part, (ttemp0 - tu) / _sd(ttemp0 - _TBFRZ_K, part & frz), 1.0)
            frc1 = jnp.where(frz, frc1, 0.0)  # kfsw: D7g
            iflag = jnp.where(frz & ~part, 1, u["IFLAG"])  # kfsw: D7g
            ttemp = jnp.where(frz, tu, u["TTEMP"])  # kfsw: D7g
            qfrz = (qliq + qnl) * frc1
            qni_f = qni + qnl * frc1
            qnl_f = qnl - qnl * frc1
            qice_f = qice + qliq * frc1
            qliq_f = qliq - qliq * frc1
            tu_f, theteu_f, qu_f, qice_f2 = _dtfrznew(ab, tu, P0[NK1], qu, qfrz, qice_f)  # kfsw: S
            tu = _w(frz, tu_f, tu)  # kfsw: D7g
            theteu = _w(frz, theteu_f, theteu)  # kfsw: D7g
            qu = _w(frz, qu_f, qu)  # kfsw: D7g
            qice = _w(frz, qice_f2, qice)  # kfsw: D7g
            qliq = _w(frz, qliq_f, qliq)  # kfsw: D7g
            qnl = _w(frz, qnl_f, qnl)  # kfsw: D7g
            qni = _w(frz, qni_f, qni)  # kfsw: D7g
            tvu = tu * (1.0 + _TV_COEF * qu)
            first = NK == K  # kfsw: C-KLCL
            be = jnp.where(first, (TVLCL + tvu) / (TVEN + TV0[NK1]) - 1.0,
                           (A["TVU"][NK] + tvu) / (TV0[NK] + TV0[NK1]) - 1.0)  # kfsw: C-KLCL
            boterm = jnp.where(first, 2.0 * (Z0[NK1] - ZLCL) * G * be / _KE_DIV,
                               2.0 * DZA[NK] * G * be / _KE_DIV)  # kfsw: C-KLCL
            dzz = jnp.where(first, Z0[NK1] - ZLCL, DZA[NK])  # kfsw: C-KLCL
            enterm = 2.0 * u["REI"] * u["WTW"] / u["UPOLD"]
            qliq, qice, wtw, qlqout, qicout = _condload(qliq, qice, u["WTW"], dzz, boterm,
                                                        enterm, qnl, qni, G, lut.sm)
            stop1 = wtw < _WTW_STOP  # kfsw: C-LTOP
            a2 = a & ~stop1
            thetee = _envirtht(lut, ab, P0[NK1], T0[NK1], Q0[NK1])  # kfsw: S
            rei = u_vmf * DP[NK1] * prm["entrain_const"] / RAD  # kfsw: S
            tvqu = tu * (1.0 + _TV_COEF * qu - qliq - qice)
            # kfsw: C-KLCL
            dilbe = jnp.where(first, ((TVLCL + tvqu) / (TVEN + TV0[NK1]) - 1.0) * dzz,
                              ((A["TVQU"][NK] + tvqu) / (TV0[NK] + TV0[NK1]) - 1.0) * dzz)
            abe = jnp.where(dilbe > 0.0, u["ABE"] + dilbe * G, u["ABE"])  # kfsw: E6
            # entrainment / detrainment (l.1206-1263)
            cold = tvqu <= TV0[NK1]  # kfsw: D4
            thttmp = _F1_HI * thetee + (1.0 - _F1_HI) * theteu
            qtmp = _F1_HI * Q0[NK1] + (1.0 - _F1_HI) * qu  # kfsw: S
            tl95 = (1.0 - _F1_HI) * qliq
            ti95 = (1.0 - _F1_HI) * qice
            tt95, qt95, tl95b, ti95b, _, _ = _tpmix2(lut, P0[NK1], thttmp, qtmp, tl95, ti95,
                                                     XLV0, XLV1)  # kfsw: S
            tu95 = tt95 * (1.0 + _TV_COEF * qt95 - tl95b - ti95b)
            vb = tu95 > TV0[NK1]  # kfsw: D4
            thttmp = _F1_LO * thetee + (1.0 - _F1_LO) * theteu
            qtmp = _F1_LO * Q0[NK1] + (1.0 - _F1_LO) * qu  # kfsw: S
            tl10 = (1.0 - _F1_LO) * qliq
            ti10 = (1.0 - _F1_LO) * qice
            tt10, qt10, tl10b, ti10b, _, _ = _tpmix2(lut, P0[NK1], thttmp, qtmp, tl10, ti10,
                                                     XLV0, XLV1)  # kfsw: S
            tu10 = tt10 * (1.0 + _TV_COEF * qt10 - tl10b - ti10b)
            tvdiff = jnp.abs(tu10 - tvqu)  # kfsw: D4
            tiny = tvdiff < _TVDIFF_MIN  # kfsw: D4
            eqf = (TV0[NK1] - tvqu) * _F1_LO / _sd(tu10 - tvqu, ~tiny)  # kfsw: D4
            eqf = jnp.maximum(0.0, eqf)  # kfsw: E7
            eqf = jnp.minimum(1.0, eqf)  # kfsw: E7
            ee_p, ud_p = prof5(eqf, lut.sm.prof5_width)
            ee_s = jnp.where(eqf == 1, 1.0, jnp.where(eqf == 0.0, 0.0, ee_p))  # kfsw: D7f
            ud_s = jnp.where(eqf == 1, 0.0, jnp.where(eqf == 0.0, 1.0, ud_p))  # kfsw: D7f
            ee2 = jnp.where(cold, 0.5, jnp.where(vb | tiny, 1.0, ee_s))  # kfsw: D4
            ud2 = jnp.where(cold, 1.0, jnp.where(vb | tiny, 0.0, ud_s))  # kfsw: D4
            eqfrc = jnp.where(cold, 0.0, jnp.where(vb | tiny, 1.0, eqf))  # kfsw: D4
            let = jnp.where(cold, u["LET"], NK1)  # kfsw: C-LET
            ee2 = jnp.maximum(ee2, _EE2_MIN)  # kfsw: E8
            ud2 = _UD2_SCALE * ud2
            uer = 0.5 * rei * (u["EE1"] + ee2)
            udr = 0.5 * rei * (u["UD1"] + ud2)
            stop2 = A["UMF"][NK] - udr < _UMF_DETR_MIN  # kfsw: C-LTOP
            a3 = a2 & ~stop2
            abe = jnp.where(stop2 & (dilbe > 0.0), abe - dilbe * G, abe)  # kfsw: E6,C-LTOP
            let = jnp.where(stop2, NK, let)  # kfsw: C-LET
            upold = A["UMF"][NK] - udr  # kfsw: S
            upnew = upold + uer
            ups = _sd(upold, a3)  # kfsw: C-LTOP
            upns = _sd(upnew, a3)  # kfsw: C-LTOP
            dilfrc = upnew / ups
            detlq = qliq * udr
            detic = qice * udr
            qdt = qu
            qu_m = (upold * qu + uer * Q0[NK1]) / upns  # kfsw: S
            theteu_m = (theteu * upold + thetee * uer) / upns
            qliq_m = qliq * upold / upns
            qice_m = qice * upold / upns
            pptliq = qlqout * A["UMF"][NK]  # kfsw: S
            pptice = qicout * A["UMF"][NK]  # kfsw: S
            uer_m = jnp.where(NK1 <= KPBL, uer + u_vmf * DP[NK1] / DPTHMX_s, uer)  # kfsw: C-KPBL

            # writes before the WTW exit
            B = dict(A)
            B["RATIO2"] = _put(B["RATIO2"], NK1, ratio2, a)  # kfsw: C-LTOP
            B["TU"] = _put(B["TU"], NK1, tu, a)  # kfsw: C-LTOP
            B["QLQOUT"] = _put(B["QLQOUT"], NK1, qlqout, a)  # kfsw: C-LTOP
            B["QICOUT"] = _put(B["QICOUT"], NK1, qicout, a)  # kfsw: C-LTOP
            B["TVU"] = _put(B["TVU"], NK1, tvu, a)  # kfsw: C-LTOP
            # THETEU/QU/QLIQ/QICE(NK1): pre-mix values, then mixed if accepted
            B["THETEU"] = _put(B["THETEU"], NK1, _w(a3, theteu_m, theteu), a)  # kfsw: C-LTOP
            B["QU"] = _put(B["QU"], NK1, _w(a3, qu_m, qu), a)  # kfsw: C-LTOP
            B["QLIQ"] = _put(B["QLIQ"], NK1, _w(a3, qliq_m, qliq), a)  # kfsw: C-LTOP
            B["QICE"] = _put(B["QICE"], NK1, _w(a3, qice_m, qice), a)  # kfsw: C-LTOP
            # writes between the two exits
            B["WU"] = _put(B["WU"], NK1, _safe_sqrt(wtw), a2)  # kfsw: C-LTOP,E11
            B["THETEE"] = _put(B["THETEE"], NK1, thetee, a2)  # kfsw: C-LTOP
            B["TVQU"] = _put(B["TVQU"], NK1, tvqu, a2)  # kfsw: C-LTOP
            B["EQFRC"] = _put(B["EQFRC"], NK1, eqfrc, a2)  # kfsw: C-LTOP
            B["UER"] = _put(B["UER"], NK1, _w(a3, uer_m, uer), a2)  # kfsw: C-LTOP
            B["UDR"] = _put(B["UDR"], NK1, udr, a2)  # kfsw: C-LTOP
            # accepted level
            B["UMF"] = _put(B["UMF"], NK1, upnew, a3)  # kfsw: C-LTOP
            B["DILFRC"] = _put(B["DILFRC"], NK1, dilfrc, a3)  # kfsw: C-LTOP
            B["DETLQ"] = _put(B["DETLQ"], NK1, detlq, a3)  # kfsw: C-LTOP
            B["DETIC"] = _put(B["DETIC"], NK1, detic, a3)  # kfsw: C-LTOP
            B["QDT"] = _put(B["QDT"], NK1, qdt, a3)  # kfsw: C-LTOP
            B["PPTLIQ"] = _put(B["PPTLIQ"], NK1, pptliq, a3)  # kfsw: C-LTOP
            B["PPTICE"] = _put(B["PPTICE"], NK1, pptice, a3)  # kfsw: C-LTOP
            v = dict(u)
            v["WTW"] = _w(a, wtw, u["WTW"])  # kfsw: C-LTOP
            v["TTEMP"] = _w(a, ttemp, u["TTEMP"])  # kfsw: C-LTOP
            v["IFLAG"] = _w(a, iflag, u["IFLAG"])  # kfsw: C-LTOP
            v["REI"] = _w(a2, rei, u["REI"])  # kfsw: C-LTOP
            v["DILBE"] = _w(a2, dilbe, u["DILBE"])  # kfsw: C-LTOP
            v["ABE"] = _w(a2, abe, u["ABE"])  # kfsw: C-LTOP
            v["LET"] = _w(a2, let, u["LET"])  # kfsw: C-LET
            v["EE1"] = _w(a3, ee2, u["EE1"])  # kfsw: C-LTOP
            v["UD1"] = _w(a3, ud2, u["UD1"])  # kfsw: C-LTOP
            v["UPOLD"] = _w(a3, upold, u["UPOLD"])  # kfsw: C-LTOP
            v["UPNEW"] = _w(a3, upnew, u["UPNEW"])  # kfsw: C-LTOP
            v["TRPPT"] = _w(a3, u["TRPPT"] + pptliq + pptice, u["TRPPT"])  # kfsw: C-LTOP
            ex = a & (stop1 | stop2)
            v["LTOP"] = _w(ex, NK, u["LTOP"])  # kfsw: C-LTOP
            v["exited"] = u["exited"] | ex
            return (B, v)

        A, u = _scan_levels(ubody, (A, u), 1, KX - 1)
        LTOP = u["LTOP"]
        cldhgt = Z0[LTOP] - ZLCL  # kfsw: C-LTOP
        CHMIN = jnp.where(TLCL > _CHMIN_TWARM_K, _CHMIN_DEEP_M,
                          jnp.where(TLCL >= _CHMIN_TCOLD_K,
                                    _CHMIN_SHALLOW_M + _CHMIN_SLOPE_M_PER_K * (TLCL - _CHMIN_TCOLD_K),
                                    _CHMIN_SHALLOW_M))  # kfsw: E9
        noconv = (LTOP <= KLCL) | (LTOP <= KPBL) | (u["LET"] + 1 <= KPBL)  # kfsw: A4
        deep = ~noconv & (cldhgt > CHMIN) & (u["ABE"] > _ABE_MIN)  # kfsw: B1
        shallow = ~noconv & ~deep
        exit_sh = shallow & (NU == sc_nuchm)  # kfsw: B2
        zero_cols = noconv | (shallow & ~exit_sh)
        A["CLDHGT"] = _put(A["CLDHGT"], LC, jnp.where(noconv, 0.0, cldhgt), go)  # kfsw: A4,C-KCHECK
        zr = go & zero_cols & (kk >= K) & (kk <= LTOP)  # kfsw: B2,C-LTOP
        for nm in ("UMF", "UDR", "UER", "DETLQ", "DETIC", "PPTLIQ", "PPTICE"):
            A[nm] = jnp.where(zr, 0.0, A[nm])  # kfsw: B2,C-LTOP
        out = dict(
            LC=LC, KPBL=KPBL, DPTHMX=DPTHMX, TMIX=TMIX, QMIX=QMIX, ZMIX=ZMIX, PMIX=PMIX,
            TLCL=TLCL, TVLCL=TVLCL, ZLCL=ZLCL, KLCL=KLCL, K=K, DLP=DLP, TENV=TENV,
            QENV=QENV, TVEN=TVEN, WKL=WKL, WKLCL=WKLCL, DTLCL=DTLCL, DTRH=DTRH, WLCL=WLCL,
            PLCL=PLCL, LCL=LCL, LET=u["LET"], RAD=RAD, AU0=AU0, VMFLCL=VMFLCL,
            UPOLD=u["UPOLD"], UPNEW=u["UPNEW"], ABE=u["ABE"], TRPPT=u["TRPPT"], LTOP=LTOP,
            CHMIN=CHMIN, IFLAG=u["IFLAG"],
        )
        return A, out, ret, go, deep, shallow, exit_sh

    # --- the usl loop (l.820-1420), at most NCHECK passes + restart + re-pass ---
    def trip(carry, _):
        S, sc = carry
        live = ~sc["done"] & ~sc["ret"]
        NU = sc["NU"] + 1
        exhausted = NU > NCHECK  # kfsw: B2
        restart = live & exhausted & (sc["ISHALL"] == 1)  # kfsw: B2
        give_up = live & exhausted & (sc["ISHALL"] != 1)  # kfsw: B2
        # NCHM / NUCHM: first candidate with the largest positive CLDHGT
        idx = jnp.arange(NP, dtype=i32)  # kfsw: S
        ch = jnp.where((idx >= 1) & (idx <= NCHECK), S["CLDHGT"][KCHECK], 0.0)  # kfsw: B2
        # kfsw: B2
        best = jnp.where(jnp.max(ch) > 0.0, jnp.argmax(ch).astype(i32), jnp.asarray(0, i32))
        act = live & ~exhausted
        A, o, ret, go, deep, shallow, exit_sh = evaluate(S, sc, NU, act, sc["NUCHM"])
        S2 = jax.tree_util.tree_map(lambda new, old: jnp.where(act, new, old), A, S)  # kfsw: B2
        sc2 = dict(sc)
        upd = go
        for key, val in o.items():
            sc2[key] = _w(upd, val, sc[key])  # kfsw: B2
        # kfsw: B1
        sc2["ISHALL"] = jnp.where(go & deep, 0, jnp.where(go & shallow, 1, sc["ISHALL"]))
        sc2["done"] = sc["done"] | (go & (deep | exit_sh))
        sc2["ret"] = sc["ret"] | ret | give_up
        sc2["NU"] = jnp.where(restart, best - 1, jnp.where(act, NU, sc["NU"]))  # kfsw: B2
        sc2["NUCHM"] = jnp.where(restart, best, sc["NUCHM"])  # kfsw: B2
        sc2["FBFRC"] = jnp.where(restart, 1.0, sc["FBFRC"])  # kfsw: B2
        return (S2, sc2), None

    (S, sc), _ = jax.lax.scan(trip, (S, sc), None, length=KX + 2)
    return _post(S, sc, U0, V0, T0, P0, Q0, QES, RH, TV0, DP, Z0, DZA, L5, kk, valid,
                 dt=dt, dx=dx, kc=kc, lut=lut, ab=ab, prm=prm, warm_rain=warm_rain,
                 f_qi=f_qi, f_qs=f_qs, cudt=cudt, KX=KX, GDRY=GDRY, DXSQ=DXSQ)


def _upstream_step(X, OMG, FXM, inc, LTOP, kk, src, dtime, EMSD):
    """One upstream/forward step of the KF vertical advection (l.1990-2018,
    l.2336-2375): THFXIN/THFXOUT built exactly as the oracle accumulates them."""
    le = OMG <= 0.0  # kfsw: E10
    up = jnp.concatenate([X[1:], X[-1:]])  # X(k+1)
    dn = jnp.concatenate([X[:1], X[:-1]])  # X(k-1)
    fxm_up = jnp.concatenate([FXM[1:], FXM[-1:]])
    le_up = jnp.concatenate([le[1:], le[-1:]])
    a_rng = inc & (kk >= 2)                       # levels processed as NK; kfsw: C-LTOP
    b_rng = inc & (kk + 1 <= LTOP)                # levels touched by NK+1; kfsw: C-LTOP
    fin = jnp.where(a_rng & le, -FXM * dn, 0.0)  # kfsw: E10
    fin = jnp.where(b_rng & ~le_up, fin + fxm_up * up, fin)  # kfsw: E10
    fout = jnp.where(a_rng & ~le, FXM * X, 0.0)  # kfsw: E10
    fout = jnp.where(b_rng & le_up, fout + (-fxm_up * X), fout)  # kfsw: E10
    return jnp.where(inc, X + src(fin, fout) * dtime * EMSD, X)  # kfsw: C-LTOP


def _post(S, sc, U0, V0, T0, P0, Q0, QES, RH, TV0, DP, Z0, DZA, L5, kk, valid, *, dt, dx,
          kc, lut, ab, prm, warm_rain, f_qi, f_qs, cudt, KX, GDRY, DXSQ):
    """KF_eta_PARA from the end of the usl loop to the tendencies (l.1421-2692)."""
    dtype = T0.dtype
    i32 = jnp.int32  # kfsw: S
    NP = KX + 2
    KL = KX
    zero = jnp.zeros((), dtype)
    G, CP, R, XLV0, XLV1 = kc.g, kc.cp, kc.r, kc.xlv0, kc.xlv1
    ALIQ, BLIQ, CLIQ, DLIQ = ab
    fired0 = sc["done"] & ~sc["ret"]
    ISHALL = sc["ISHALL"]
    sh = ISHALL == 1  # kfsw: B1
    LC, KPBL, KLCL, K, LCL = sc["LC"], sc["KPBL"], sc["KLCL"], sc["K"], sc["LCL"]
    LTOP = sc["LTOP"]
    DPTHMX = _sd(sc["DPTHMX"], fired0)  # kfsw: A0
    # kfsw: S
    VMFLCL, AU0, TMIX, QMIX, ZMIX, PMIX = (sc[x] for x in ("VMFLCL", "AU0", "TMIX", "QMIX", "ZMIX", "PMIX"))
    UMF, UER, UDR = S["UMF"], S["UER"], S["UDR"]
    DETLQ, DETIC, PPTLIQ, PPTICE = S["DETLQ"], S["DETIC"], S["PPTLIQ"], S["PPTICE"]
    QLIQ, QICE, QLQOUT, QICOUT = S["QLIQ"], S["QICE"], S["QLQOUT"], S["QICOUT"]
    # kfsw: S
    TU, QU, WU, QDT, DILFRC, EQFRC, RATIO2 = (S[x] for x in ("TU", "QU", "WU", "QDT", "DILFRC", "EQFRC", "RATIO2"))
    THETEE = S["THETEE"]
    TRPPT = sc["TRPPT"]
    UPOLD = _sd(sc["UPOLD"], fired0)  # kfsw: A0
    UPNEW = sc["UPNEW"]

    # --- shallow LET; total detrainment at cloud top (l.1421-1475) ---
    LET = jnp.where(sh, jnp.maximum(KPBL, KLCL), sc["LET"])  # kfsw: C-LET
    eq = LET == LTOP  # kfsw: C-LET
    udr_t = UMF[LTOP] + UDR[LTOP] - UER[LTOP]  # kfsw: C-LTOP
    detlq_t = QLIQ[LTOP] * udr_t * UPNEW / UPOLD  # kfsw: C-LTOP
    detic_t = QICE[LTOP] * udr_t * UPNEW / UPOLD  # kfsw: C-LTOP
    UDR = _put(UDR, LTOP, udr_t, eq)  # kfsw: C-LET
    DETLQ = _put(DETLQ, LTOP, detlq_t, eq)  # kfsw: C-LET
    DETIC = _put(DETIC, LTOP, detic_t, eq)  # kfsw: C-LET
    UER = _put(UER, LTOP, zero, eq)  # kfsw: C-LET
    UMF = _put(UMF, LTOP, zero, eq)  # kfsw: C-LET
    nb = ~eq

    def dptt_body(c, k):
        return _w(nb & (k >= LET + 1) & (k <= LTOP), c + DP[k], c)  # kfsw: C-LET

    DPTT = _scan_levels(dptt_body, zero, 1, KX)
    DUMFDP = UMF[LET] / _sd(DPTT, nb & (DPTT != 0.0))  # kfsw: C-LET

    def let_body(c, NK):
        umf, uer, udr, detlq, detic, pl, pi_, trppt = c
        a = nb & (NK >= LET + 1) & (NK <= LTOP)  # kfsw: C-LET
        top = NK == LTOP  # kfsw: C-LET
        umf_n = umf[NK - 1] - DP[NK] * DUMFDP  # kfsw: S
        uer_n = umf_n * (1.0 - 1.0 / DILFRC[NK])  # kfsw: S
        udr_n = umf[NK - 1] - umf_n + uer_n  # kfsw: S
        udr_k = jnp.where(top, umf[NK - 1], udr_n)  # kfsw: C-LET
        uer_k = jnp.where(top, 0.0, uer_n)  # kfsw: C-LET
        umf_k = jnp.where(top, umf[NK], umf_n)  # kfsw: C-LET
        detlq_k = udr_k * QLIQ[NK] * DILFRC[NK]  # kfsw: S
        detic_k = udr_k * QICE[NK] * DILFRC[NK]  # kfsw: S
        b = a & (NK >= LET + 2)  # kfsw: C-LET
        trppt_b = trppt - pl[NK] - pi_[NK]  # kfsw: S
        pl_k = umf[NK - 1] * QLQOUT[NK]  # kfsw: S
        pi_k = umf[NK - 1] * QICOUT[NK]  # kfsw: S
        trppt_b = trppt_b + pl_k + pi_k
        return (_put(umf, NK, umf_k, a), _put(uer, NK, uer_k, a), _put(udr, NK, udr_k, a),
                _put(detlq, NK, detlq_k, a), _put(detic, NK, detic_k, a),
                _put(pl, NK, pl_k, b), _put(pi_, NK, pi_k, b), _w(b, trppt_b, trppt))  # kfsw: C-LET

    UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, TRPPT = _scan_levels(
        let_body, (UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, TRPPT), 1, KX)

    # --- below cloud base / above cloud top (l.1479-1577) ---
    ML = jnp.max(jnp.where(valid & (kk <= LTOP) & (T0 > _T00_K), kk, 0))  # kfsw: C-ML

    def below_body(c, NK):
        umf, uer = c
        a = NK <= K  # kfsw: C-KLCL,C-KPBL
        isLC = NK == LC  # kfsw: C-KLCL,C-KPBL,C-KCHECK
        uer_v = VMFLCL * DP[NK] / DPTHMX  # kfsw: S
        # kfsw: C-KLCL,C-KPBL,C-KCHECK
        umf_v = jnp.where(isLC, uer_v, jnp.where(NK <= KPBL, umf[NK - 1] + uer_v, VMFLCL))
        uer_k = jnp.where(isLC | (NK <= KPBL), uer_v, 0.0)  # kfsw: C-KLCL,C-KPBL,C-KCHECK
        inlc = NK >= LC  # kfsw: C-KLCL,C-KPBL,C-KCHECK
        return (_put(umf, NK, jnp.where(inlc, umf_v, 0.0), a),
                _put(uer, NK, jnp.where(inlc, uer_k, 0.0), a))  # kfsw: C-KLCL,C-KPBL,C-KCHECK

    UMF, UER = _scan_levels(below_body, (UMF, UER), 1, KX)
    blw = valid & (kk <= K)  # kfsw: C-KLCL,C-KPBL
    inlc = kk >= LC  # kfsw: C-KLCL,C-KPBL,C-KCHECK
    # kfsw: C-KLCL,C-KPBL,C-KCHECK
    TU = jnp.where(blw, jnp.where(inlc, TMIX + (Z0 - ZMIX) * GDRY, 0.0), TU)
    QU = jnp.where(blw, jnp.where(inlc, QMIX, 0.0), QU)  # kfsw: C-KLCL,C-KPBL,C-KCHECK
    WU = jnp.where(blw, jnp.where(inlc, sc["WLCL"], 0.0), WU)  # kfsw: C-KLCL,C-KPBL,C-KCHECK
    for_zero = (UDR, QDT, QLIQ, QICE, QLQOUT, QICOUT, PPTLIQ, PPTICE, DETLQ, DETIC, RATIO2)
    (UDR, QDT, QLIQ, QICE, QLQOUT, QICOUT, PPTLIQ, PPTICE, DETLQ, DETIC, RATIO2) = (
        jnp.where(blw, 0.0, a) for a in for_zero)  # kfsw: C-KLCL,C-KPBL
    THETEE = jnp.where(blw, _envirtht(lut, ab, P0, T0, Q0), THETEE)  # kfsw: C-KLCL,C-KPBL
    EQFRC = jnp.where(blw, 1.0, EQFRC)  # kfsw: C-KLCL,C-KPBL
    above = valid & (kk >= LTOP + 1)  # kfsw: C-LTOP
    above2 = valid & (kk > LTOP + 1)  # kfsw: C-LTOP
    # kfsw: C-LTOP
    (UMF, UDR, UER, QDT, QLIQ, QICE, QLQOUT, QICOUT, DETLQ, DETIC, PPTLIQ, PPTICE) = (
        jnp.where(above, 0.0, a) for a in
        (UMF, UDR, UER, QDT, QLIQ, QICE, QLQOUT, QICOUT, DETLQ, DETIC, PPTLIQ, PPTICE))
    TU, QU, WU = (jnp.where(above2, 0.0, a) for a in (TU, QU, WU))  # kfsw: C-LTOP
    inc = valid & (kk <= LTOP)  # kfsw: C-LTOP
    EMS = jnp.where(inc, DP * DXSQ / G, 0.0)  # kfsw: C-LTOP
    EMSD = jnp.where(inc, 1.0 / _sd(EMS, inc), 0.0)  # kfsw: C-LTOP
    THTAU = jnp.where(inc, TU * _exner(P0, QDT), 0.0)  # kfsw: C-LTOP
    THTA0 = jnp.where(inc, T0 * _exner(P0, Q0), 0.0)  # kfsw: C-LTOP
    DDILFRC = jnp.where(inc, 1.0 / _sd(DILFRC, inc), 1.0)  # kfsw: C-LTOP

    # --- convective time scale and precipitation efficiency (l.1588-1632) ---
    def wspd(k):
        return _safe_sqrt(U0[k] * U0[k] + V0[k] * V0[k])  # kfsw: E11

    WS_KLCL, WS_LTOP = wspd(KLCL), wspd(LTOP)
    if lut.sm.l5_gather_width_pa:  # kfsw: S
        wl5 = _l5_weights(P0, kk, valid, lut.sm.l5_gather_width_pa)
        WS_L5 = jnp.sum(wl5 * wspd(kk))  # kfsw: C-L5g
    else:
        WS_L5 = wspd(L5)
    VCONV = 0.5 * (WS_KLCL + WS_L5)
    vok = VCONV > 0.0  # kfsw: E12
    TIMEC = jnp.where(vok, dx / _sd(VCONV, vok), jnp.inf)  # kfsw: E12
    TADVEC = TIMEC
    TIMEC = jnp.maximum(prm["timec_min"], TIMEC)  # kfsw: E12
    TIMEC = jnp.minimum(prm["timec_max"], TIMEC)  # kfsw: E12
    TIMEC = jnp.where(sh, _TIMEC_SHALLOW_S, TIMEC)  # kfsw: B1
    NIC = _nint(TIMEC / dt)  # kfsw: C-NIC
    TIMEC = NIC.astype(dtype) * dt  # kfsw: C-NIC
    SHSIGN = jnp.where(WS_LTOP > WS_KLCL, 1.0, -1.0)  # kfsw: D5
    du = U0[LTOP] - U0[KLCL]  # kfsw: C-LTOP,C-KLCL
    dv = V0[LTOP] - V0[KLCL]  # kfsw: C-LTOP,C-KLCL
    VWS = du * du + dv * dv
    dzc = Z0[LTOP] - Z0[LCL]  # kfsw: C-LTOP
    VWS = _VWS_SCALE * SHSIGN * _safe_sqrt(VWS) / _sd(dzc, dzc != 0.0)  # kfsw: E11,C-LTOP
    PEF = _PEF_C0 + VWS * (_PEF_C1 + VWS * (_PEF_C2 - VWS * _PEF_C3))
    PEF = jnp.minimum(jnp.maximum(PEF, _PEF_MIN), _PEF_MAX)  # kfsw: E13
    CBH = (sc["ZLCL"] - Z0[1]) * _CBH_FT_PER_M
    RCBH = jnp.where(CBH < _CBH_LOW, _RCBH_LOW,
                     _RCBH_C0 + CBH * (_RCBH_C1 + CBH * (_RCBH_C2 + CBH * (
                         _RCBH_C3 + CBH * (_RCBH_C4 - CBH * _RCBH_C5)))))  # kfsw: D6
    RCBH = jnp.where(CBH > _CBH_HIGH, _RCBH_HIGH, RCBH)  # kfsw: D6
    PEFCBH = jnp.minimum(1.0 / (1.0 + RCBH), _PEFCBH_MAX)  # kfsw: E14
    PEFF = 0.5 * (PEF + PEFCBH)

    # --- downdraft (l.1647-1784) ---
    zf = jnp.zeros(NP, dtype)
    KSTART = KPBL + 1
    cand = valid & (kk >= KSTART + 1) & (P0[KSTART] - P0 > _LFS_DEPTH_PA)  # kfsw: C-LFS
    KLFS = jnp.where(jnp.any(cand), jnp.argmax(cand).astype(i32), LET - 1)  # kfsw: C-LFS
    KLFS = jnp.minimum(KLFS, LET - 1)  # kfsw: C-LFS
    LFS = jnp.where(sh, jnp.asarray(1, i32), KLFS)  # kfsw: B1
    dd_ok = ~sh & (P0[KSTART] - P0[LFS] > _DD_MIN_DEPTH_PA)  # kfsw: B3
    tz_l, qss_l = _tpmix2dd(lut, P0[LFS], THETEE[LFS])  # kfsw: C-LFS
    thtad_l = tz_l * _exner(P0[LFS], qss_l)  # kfsw: C-LFS
    tvd_l = tz_l * (1.0 + _TV_COEF * qss_l)
    rdd = P0[LFS] / (R * tvd_l)  # kfsw: C-LFS
    dmf_l = -((1.0 - PEFF) * AU0) * rdd
    THETED = _put(zf, LFS, THETEE[LFS], dd_ok)  # kfsw: B3
    QD = _put(zf, LFS, Q0[LFS], dd_ok)  # kfsw: B3
    TZ = _put(zf, LFS, tz_l, dd_ok)  # kfsw: B3
    THTAD = _put(zf, LFS, thtad_l, dd_ok)  # kfsw: B3
    DMF = _put(zf, LFS, dmf_l, dd_ok)  # kfsw: B3
    DER = _put(zf, LFS, dmf_l, dd_ok)  # kfsw: B3
    DDR = zf
    QSD = zf

    def dd1_body(c, ND):
        dmf, der, thd, qd, dptt, rhbar = c
        a = dd_ok & (ND <= LFS - 1) & (ND >= KSTART)  # kfsw: B3,C-LFS
        der_k = der[LFS] * EMS[ND] / _sd(EMS[LFS], a)  # kfsw: B3,C-LFS
        dmf_k = dmf[ND + 1] + der_k  # kfsw: S
        dmfs = _sd(dmf_k, a)  # kfsw: B3,C-LFS
        thd_k = (thd[ND + 1] * dmf[ND + 1] + THETEE[ND] * der_k) / dmfs  # kfsw: S
        qd_k = (qd[ND + 1] * dmf[ND + 1] + Q0[ND] * der_k) / dmfs  # kfsw: S
        return (_put(dmf, ND, dmf_k, a), _put(der, ND, der_k, a), _put(thd, ND, thd_k, a),
                _put(qd, ND, qd_k, a), _w(a, dptt + DP[ND], dptt),
                _w(a, rhbar + RH[ND] * DP[ND], rhbar))  # kfsw: B3,C-LFS

    # kfsw: C-LFS
    DMF, DER, THETED, QD, DPTT2, RHBAR = _scan_levels(
        dd1_body, (DMF, DER, THETED, QD, DP[LFS], RH[LFS] * DP[LFS]), 1, KX, reverse=True)
    RHBAR = RHBAR / _sd(DPTT2, dd_ok)  # kfsw: B3
    DMFFRC = _DMFFRC_COEF * (1.0 - RHBAR)

    def mlt_body(c, k):
        return _w((k >= KLCL) & (k <= LTOP), c + PPTICE[k], c)  # kfsw: C-KLCL,C-LTOP

    PPTMLT = _scan_levels(mlt_body, zero, 1, KX)
    mel = LC < ML  # kfsw: B5,C-KCHECK
    DTMELT = jnp.where(mel, _RLF * PPTMLT / (CP * _sd(UMF[KLCL], mel & dd_ok)), 0.0)  # kfsw: B5
    LDT = jnp.minimum(LFS - 1, KSTART - 1)  # kfsw: C-LFS
    tzk, _ = _tpmix2dd(lut, P0[KSTART], THETED[KSTART])  # kfsw: C-KPBL
    tzk = tzk - DTMELT
    es = ALIQ * jnp.exp((BLIQ * tzk - CLIQ) / (tzk - DLIQ))
    qssk = _EPS_KF * es / (P0[KSTART] - es)  # kfsw: C-KPBL
    thd_k = tzk * (_P00_PA / P0[KSTART]) ** (_EXN_KAPPA * (1.0 - _EXN_Q * qssk)) * jnp.exp(
        (_THE_C1 / tzk - _THE_C2) * qssk * (1.0 + _THE_C3 * qssk))  # kfsw: C-KPBL
    TZ = _put(TZ, KSTART, tzk, dd_ok)  # kfsw: B3
    THETED = _put(THETED, KSTART, thd_k, dd_ok)  # kfsw: B3

    def dd2_body(c, ND):
        tz, qsd, thd, qd, dpdd, ldb, ex = c
        a = dd_ok & (ND <= LDT) & ~ex  # kfsw: B3,C-LDB
        dpdd_n = dpdd + DP[ND]  # kfsw: S
        thd_n = THETED[KSTART]  # kfsw: C-KPBL
        qd_n = qd[KSTART]  # kfsw: C-KPBL
        t_n, qss = _tpmix2dd(lut, P0[ND], thd_n)  # kfsw: S
        rhh = 1.0 - _RHH_DECR / _RHH_PER_M * (Z0[KSTART] - Z0[ND])  # kfsw: C-KPBL
        dssdt = (CLIQ - BLIQ * DLIQ) / ((t_n - DLIQ) * (t_n - DLIQ))
        rl = XLV0 - XLV1 * t_n
        dtmp = rl * qss * (1.0 - rhh) / (CP + rl * rhh * qss * dssdt)
        t1rh = t_n + dtmp
        es_r = rhh * ALIQ * jnp.exp((BLIQ * t1rh - CLIQ) / (t1rh - DLIQ))
        qsrh = _EPS_KF * es_r / (P0[ND] - es_r)  # kfsw: S
        low = qsrh < qd_n  # kfsw: D7h
        qsrh = jnp.where(low, qd_n, qsrh)  # kfsw: D7h
        t1rh = jnp.where(low, t_n + (qss - qsrh) * rl / CP, t1rh)  # kfsw: D7h
        adj = rhh < 1.0  # kfsw: D7h
        t_n = jnp.where(adj, t1rh, t_n)  # kfsw: D7h
        qsd_n = jnp.where(adj, qsrh, qss)  # kfsw: D7h
        tvd = t_n * (1.0 + _TV_COEF * qsd_n)
        stop = (tvd > TV0[ND]) | (ND == 1)  # kfsw: B3,C-LDB
        # kfsw: B3,C-LDB
        return (_put(tz, ND, t_n, a), _put(qsd, ND, qsd_n, a), _put(thd, ND, thd_n, a),
                _put(qd, ND, qd_n, a), _w(a, dpdd_n, dpdd), _w(a & stop, ND, ldb), ex | (a & stop))

    TZ, QSD, THETED, QD, DPDD, LDB, _ = _scan_levels(
        dd2_body, (TZ, QSD, THETED, QD, zero, LFS, jnp.asarray(False)), 1, KX, reverse=True)
    deep_dd = dd_ok & (P0[LDB] - P0[LFS] > _DD_MIN_DEPTH_PA)  # kfsw: B3

    def dd3_body(c, ND):
        ddr, der, dmf, tder, qd, thtad = c
        a = deep_dd & (ND <= LDT) & (ND >= LDB)  # kfsw: B3,C-LDB
        ddr_k = -dmf[KSTART] * DP[ND] / _sd(DPDD, a)  # kfsw: B3,C-LDB
        dmf_k = dmf[ND + 1] + ddr_k  # kfsw: S
        tder_n = tder + (QSD[ND] - qd[ND]) * ddr_k  # kfsw: S
        thtad_k = TZ[ND] * _exner(P0[ND], QSD[ND])  # kfsw: S
        # kfsw: B3,C-LDB
        return (_put(ddr, ND, ddr_k, a), _put(der, ND, zero, a), _put(dmf, ND, dmf_k, a),
                _w(a, tder_n, tder), _put(qd, ND, QSD[ND], a), _put(thtad, ND, thtad_k, a))

    DDR, DER, DMF, TDER, QD, THTAD = _scan_levels(
        dd3_body, (DDR, DER, DMF, zero, QD, THTAD), 1, KX, reverse=True)

    # --- downdraft acceptance (l.1789-1873) ---
    nodd = TDER < _TDER_MIN  # kfsw: B3
    CPR = TRPPT
    ddinc = -DMFFRC * UMF[KLCL] / _sd(DMF[KSTART], ~nodd)  # kfsw: B3
    big = TDER * ddinc > TRPPT  # kfsw: B3
    ddinc = jnp.where(big, TRPPT / _sd(TDER, ~nodd), ddinc)  # kfsw: B3
    TDER = jnp.where(nodd, 0.0, TDER * ddinc)  # kfsw: B3
    PPTFLX = jnp.where(nodd, TRPPT, TRPPT - TDER)  # kfsw: B3
    LDB = jnp.where(nodd, LFS, LDB)  # kfsw: B3,C-LDB
    scl = ~nodd & (kk >= LDB) & (kk <= LFS)  # kfsw: B3,C-LDB
    DMF, DER, DDR = (jnp.where(scl, a * ddinc, a) for a in (DMF, DER, DDR))  # kfsw: B3,C-LDB
    z_nodd = nodd & valid & (kk <= LTOP)  # kfsw: B3,C-LDB
    z_dd = ~nodd & valid & ((kk <= LDB - 1) | (kk >= LFS + 1))  # kfsw: B3,C-LDB
    z_mid = ~nodd & (kk >= LDT + 1) & (kk <= LFS - 1)  # kfsw: B3,C-LDB
    DMF, DER, DDR = (jnp.where(z_nodd | z_dd, 0.0, a) for a in (DMF, DER, DDR))  # kfsw: B3,C-LDB
    # kfsw: B3,C-LDB
    THTAD, TZ, QD = (jnp.where(z_nodd | z_dd | z_mid, 0.0, a) for a in (THTAD, TZ, QD))

    # --- inflow limit and closure (l.1879-2281) ---
    LMAX = jnp.maximum(KLCL, LFS)  # kfsw: C-KLCL,C-LFS
    udd = UER - DER
    okm = valid & (kk >= LC) & (kk <= LMAX) & (udd > _UERDER_MIN)  # kfsw: E15,C-KCHECK,C-KLCL,C-LFS
    AINCM1 = EMS / (_sd(udd, okm) * TIMEC)  # kfsw: E15
    AINCMX = jnp.minimum(_AINCMX0, jnp.min(jnp.where(okm, AINCM1, _AINCMX0)))  # kfsw: E15
    AINC = jnp.where(AINCMX < 1.0, AINCMX, 1.0)  # kfsw: E15
    TDER2, PPTFL2 = TDER, PPTFLX
    base = dict(UMF=UMF, DMF=DMF, DETLQ=DETLQ, DETIC=DETIC, UDR=UDR, UER=UER, DER=DER, DDR=DDR)
    EVAC = _EVAC_FRAC * _TKEMAX * _EVAC_TKE
    ainc_sh = EVAC * DPTHMX * DXSQ / (_sd(VMFLCL, fired0) * G * TIMEC)  # kfsw: A0
    AINC = jnp.where(sh, ainc_sh, AINC)  # kfsw: B1

    F = {k: jnp.where(sh & inc, v * ainc_sh, v) for k, v in base.items()}  # kfsw: B1
    TDER = jnp.where(sh, TDER2 * ainc_sh, TDER)  # kfsw: B1
    PPTFLX = jnp.where(sh, PPTFL2 * ainc_sh, PPTFLX)  # kfsw: B1
    ABE = _sd(sc["ABE"], fired0 & ~sh)  # kfsw: B1

    def advect(X, src, OMG, FXM, nstep, dtime):
        def step(c, s):
            # kfsw: C-NSTEP
            return jnp.where(s < nstep, _upstream_step(c, OMG, FXM, inc, LTOP, kk, src, dtime, EMSD), c), None
        X, _ = jax.lax.scan(step, X, jnp.arange(_MAX_SUBSTEPS, dtype=i32))  # kfsw: S
        return X

    it0 = dict(
        F=F, TDER=TDER, PPTFLX=PPTFLX, AINC=AINC, AINCOLD=AINC, FABEOLD=zero, FABE=jnp.ones((), dtype),
        NOITR=jnp.asarray(False), stop=jnp.asarray(False), ret=jnp.asarray(False),
        fatal=jnp.asarray(False), overflow=jnp.asarray(False),
        OMG=zf, FXM=zf, NSTEP=jnp.asarray(1, i32), DTIME=jnp.ones((), dtype),
        TG=T0, QG=Q0, KLCL=KLCL, THETEU=S["THETEU"], NCOUNT=jnp.asarray(_N_ITER + 1, i32),
    )  # kfsw: S

    def iterate(c, ncount):
        a = ~c["stop"] & ~c["ret"]
        F = c["F"]
        # compensating subsidence (l.1961-1981)
        # kfsw: C-LTOP
        DOMGDP = jnp.where(inc, -(F["UER"] - F["DER"] - F["UDR"] - F["DDR"]) * EMSD, 0.0)

        def omg_body(cc, NK):
            omg, dtt = cc
            b = (NK >= 2) & (NK <= LTOP)  # kfsw: C-LTOP
            o = omg[NK - 1] - DP[NK - 1] * DOMGDP[NK - 1]  # kfsw: S
            ab_ = jnp.abs(o)  # kfsw: E16
            frdp = _FRDP * DP[NK - 1]  # kfsw: S
            lim = b & (ab_ * TIMEC > frdp)  # kfsw: E16
            dtt = jnp.where(lim, jnp.minimum(dtt, frdp / _sd(ab_, lim)), dtt)  # kfsw: E16
            return (_put(omg, NK, o, b), dtt)  # kfsw: C-LTOP

        OMG, DTT = _scan_levels(omg_body, (c["OMG"], TIMEC), 1, KX)
        NSTEP = _nint(TIMEC / DTT + 1)  # kfsw: C-NSTEP
        DTIME = TIMEC / NSTEP.astype(dtype)  # kfsw: C-NSTEP
        FXM = jnp.where(inc, OMG * DXSQ / G, 0.0)  # kfsw: C-LTOP
        overflow = c["overflow"] | (a & (NSTEP > _MAX_SUBSTEPS))  # kfsw: A6
        th_src = lambda fin, fout: fin + F["UDR"] * THTAU + F["DDR"] * THTAD - fout - (F["UER"] - F["DER"]) * THTA0  # noqa: E731
        q_src = lambda fin, fout: fin + F["UDR"] * QDT + F["DDR"] * QD - fout - (F["UER"] - F["DER"]) * Q0  # noqa: E731
        THPA = advect(jnp.where(inc, THTA0, 0.0), th_src, OMG, FXM, NSTEP, DTIME)  # kfsw: C-LTOP
        QPA = advect(jnp.where(inc, Q0, 0.0), q_src, OMG, FXM, NSTEP, DTIME)  # kfsw: C-LTOP
        THTAG = THPA
        QG = jnp.where(inc, QPA, Q0)  # kfsw: C-LTOP
        KLCL_c = c["KLCL"]

        # borrow moisture where QG < 0 (l.2028-2059)
        def neg_body(cc, NK):
            qg, fatal = cc
            b = (NK <= LTOP) & (qg[NK] < 0.0)  # kfsw: D7i
            fatal = fatal | (b & (NK == 1))  # kfsw: D7i,A6
            nk1 = jnp.where(NK == LTOP, KLCL_c, NK + 1)  # kfsw: D7i,C-LTOP
            tma = qg[nk1] * EMS[nk1]  # kfsw: D7i,C-LTOP
            tmb = qg[NK - 1] * EMS[NK - 1]  # kfsw: S
            tmm = (qg[NK] - _QG_FLOOR) * EMS[NK]  # kfsw: S
            tmbs = _sd(tmb, b)  # kfsw: D7i
            bco = -tmm / _sd((tma * tma) / tmbs + tmb, b)  # kfsw: D7i
            aco = bco * tma / tmbs
            tmb = tmb * (1.0 - bco)
            tma = tma * (1.0 - aco)
            qg = _put(qg, NK, jnp.asarray(_QG_FLOOR, dtype), b)  # kfsw: D7i
            qg = _put(qg, nk1, tma * EMSD[nk1], b)  # kfsw: D7i
            qg = _put(qg, NK - 1, tmb * EMSD[NK - 1], b)  # kfsw: D7i
            return (qg, fatal)

        QG, fatal = _scan_levels(neg_body, (QG, c["fatal"]), 1, KX)
        TOPOMG = (F["UDR"][LTOP] - F["UER"][LTOP]) * DP[LTOP] * EMSD[LTOP]  # kfsw: A6,C-LTOP
        istop = jnp.abs(TOPOMG - OMG[LTOP]) > _TOPOMG_TOL  # kfsw: A6
        TG = jnp.where(inc, THTAG / _exner(P0, QG), T0)  # kfsw: C-LTOP
        TVG = TG * (1.0 + _TV_COEF * QG)
        stop_here = istop | sh

        # new cloud and ABE of the adjusted sounding (l.2090-2174)
        def mix_body(cc, k):
            ts, qs = cc
            b = (k >= LC) & (k <= KPBL)  # kfsw: C-KPBL,C-KCHECK
            return (_w(b, ts + DP[k] * TG[k], ts), _w(b, qs + DP[k] * QG[k], qs))  # kfsw: C-KPBL

        TS, QS = _scan_levels(mix_body, (zero, zero), 1, KX)
        tmix = TS / DPTHMX
        qmix = QS / DPTHMX
        es_ = ALIQ * jnp.exp((tmix * BLIQ - CLIQ) / (tmix - DLIQ))
        qss = _EPS_KF * es_ / (PMIX - es_)
        sup = qmix > qss  # kfsw: D7j
        rl = XLV0 - XLV1 * tmix
        cpm = CP * (1.0 + _CPM_Q * qmix)
        dssdt = qss * (CLIQ - BLIQ * DLIQ) / ((tmix - DLIQ) * (tmix - DLIQ))
        dq = (qmix - qss) / (1.0 + rl * dssdt / cpm)
        tmix_s = tmix + rl / CP * dq
        qmix_s = qmix - dq
        qmix_u = jnp.maximum(qmix, 0.0)  # kfsw: E17
        emix = qmix_u * PMIX / (_EPS_KF + qmix_u)
        tlog = _tlog_lookup(lut, emix / ALIQ)
        tdpt = (CLIQ - DLIQ * tlog) / (BLIQ - tlog)
        tlcl_u = _cap_c1(_tsat_from_dewpoint(tdpt, tmix), tmix, lut.sm.tlcl_cap_width_k)
        tmix = jnp.where(sup, tmix_s, tmix)  # kfsw: D7j
        qmix = jnp.where(sup, qmix_s, qmix_u)  # kfsw: D7j
        tlcl = jnp.where(sup, tmix_s, tlcl_u)  # kfsw: D7j
        tvlcl = tlcl * (1.0 + _TV_COEF * qmix)
        zlcl = ZMIX + (tlcl - tmix) / GDRY
        hit = valid & (kk >= LC) & (zlcl <= Z0)  # kfsw: C-KLCL2,C-KCHECK
        # kfsw: C-KLCL2
        klcl = jnp.where(jnp.any(hit), jnp.argmax(hit).astype(i32), jnp.asarray(KL, i32))
        kq = klcl - 1
        dlp = (zlcl - Z0[kq]) / _sd(Z0[klcl] - Z0[kq], Z0[klcl] != Z0[kq])  # kfsw: C-KLCL2
        tenv = TG[kq] + (TG[klcl] - TG[kq]) * dlp  # kfsw: C-KLCL2
        qenv = QG[kq] + (QG[klcl] - QG[kq]) * dlp  # kfsw: C-KLCL2
        tven = tenv * (1.0 + _TV_COEF * qenv)
        th_k = tmix * (_P00_PA / PMIX) ** (_EXN_KAPPA * (1.0 - _EXN_Q * qmix)) * jnp.exp(
            (_THE_C1 / tlcl - _THE_C2) * qmix * (1.0 + _THE_C3 * qmix))
        theteu = c["THETEU"].at[kq].set(th_k)  # kfsw: C-KLCL2

        def abeg_body(cc, NK):
            th, tvq, abeg = cc
            b = (NK >= kq) & (NK <= LTOP - 1)  # kfsw: C-KLCL2
            NK1 = NK + 1
            thn = th[NK]  # kfsw: S
            tgu, qgu = _tpmix2dd(lut, P0[NK1], thn)  # kfsw: S
            tvqu = tgu * (1.0 + _TV_COEF * qgu - QLIQ[NK1] - QICE[NK1])  # kfsw: S
            first = NK == kq  # kfsw: C-KLCL2
            # kfsw: C-KLCL2
            dilbe = jnp.where(first,
                              ((tvlcl + tvqu) / (tven + TVG[NK1]) - 1.0) * (Z0[klcl] - zlcl),
                              ((tvq[NK] + tvqu) / (TVG[NK] + TVG[NK1]) - 1.0) * DZA[NK])
            abeg = jnp.where(b & (dilbe > 0.0), abeg + dilbe * G, abeg)  # kfsw: E6
            thteeg = _envirtht(lut, ab, P0[NK1], TG[NK1], QG[NK1])  # kfsw: S
            th_new = thn * DDILFRC[NK1] + thteeg * (1.0 - DDILFRC[NK1])  # kfsw: S
            return (_put(th, NK1, th_new, b), _put(tvq, NK1, tvqu, b), abeg)  # kfsw: C-KLCL2

        theteu, _, ABEG = _scan_levels(abeg_body, (theteu, jnp.zeros(NP, dtype), zero), 1, KX - 1)
        # decision logic (l.2179-2280)
        stop_noitr = c["NOITR"]
        DABE = jnp.maximum(ABE - ABEG, _DABE_FRAC * ABE)  # kfsw: E18
        FABE = ABEG / ABE
        ret_fabe = (FABE > 1.0) & (ISHALL == 0)  # kfsw: A5
        later = ncount != 1  # kfsw: S
        same = jnp.abs(c["AINC"] - c["AINCOLD"]) < _AINC_TOL  # kfsw: B4
        dfda = (FABE - c["FABEOLD"]) / _sd(c["AINC"] - c["AINCOLD"], later & ~same)  # kfsw: B4
        cyc1 = later & (same | (dfda > 0.0))  # kfsw: B4
        AINCOLD = c["AINC"]
        FABEOLD = FABE
        sat_ = (c["AINC"] / AINCMX > _AINC_REL_MAX) & (FABE > _FABE_HI - _STAB)  # kfsw: B4
        # kfsw: B4
        conv = ((FABE <= _FABE_HI - _STAB) & (FABE >= _FABE_LO - _STAB)) | (ncount == _N_ITER)
        fz = FABE == 0.0  # kfsw: B4
        cyc2 = ~fz & (DABE < _DABE_MIN)  # kfsw: B4
        # kfsw: B4
        ainc_new = jnp.where(fz, c["AINC"] * 0.5, c["AINC"] * _STAB * ABE / _sd(DABE, ~fz & ~cyc2))
        ainc_new = jnp.minimum(AINCMX, ainc_new)  # kfsw: E19
        ret_small = ainc_new < _AINC_MIN  # kfsw: A5
        # resolve the branch order of the oracle
        p_stop = stop_here                       # ISTOP or shallow: exit after the T conversion
        p_noitr = ~p_stop & stop_noitr           # NOITR: exit after ABEG
        p_ret1 = ~p_stop & ~p_noitr & ret_fabe
        p_cyc1 = ~p_stop & ~p_noitr & ~p_ret1 & cyc1
        rest = ~p_stop & ~p_noitr & ~p_ret1 & ~p_cyc1
        p_exit = rest & (sat_ | conv)
        p_cyc2 = rest & ~p_exit & cyc2
        p_rescale = rest & ~p_exit & ~p_cyc2
        p_ret2 = p_rescale & ret_small
        p_apply = p_rescale & ~ret_small
        n = dict(c)
        n["OMG"] = _w(a, OMG, c["OMG"])  # kfsw: B4
        n["FXM"] = _w(a, FXM, c["FXM"])  # kfsw: B4
        n["NSTEP"] = _w(a, NSTEP, c["NSTEP"])  # kfsw: B4
        n["DTIME"] = _w(a, DTIME, c["DTIME"])  # kfsw: B4
        n["overflow"] = overflow
        n["TG"] = _w(a, TG, c["TG"])  # kfsw: B4
        n["QG"] = _w(a, QG, c["QG"])  # kfsw: B4
        n["fatal"] = _w(a, fatal | istop, c["fatal"])  # kfsw: A6
        nonstop = a & ~p_stop
        n["KLCL"] = _w(nonstop, klcl, c["KLCL"])  # kfsw: B4
        n["THETEU"] = _w(nonstop, theteu, c["THETEU"])  # kfsw: B4
        n["FABE"] = _w(nonstop & ~p_noitr, FABE, c["FABE"])  # kfsw: B4
        n["NOITR"] = c["NOITR"] | (a & (p_cyc1 | p_cyc2))
        n["AINC"] = jnp.where(a & p_cyc1, c["AINCOLD"],
                              jnp.where(a & p_apply, ainc_new, c["AINC"]))  # kfsw: B4
        upd_old = a & (p_exit | p_cyc2 | p_rescale)
        n["AINCOLD"] = _w(upd_old, AINCOLD, c["AINCOLD"])  # kfsw: B4
        n["FABEOLD"] = _w(upd_old, FABEOLD, c["FABEOLD"])  # kfsw: B4
        n["F"] = jax.tree_util.tree_map(lambda new, old: _w(a & p_apply, new, old), {
            k: jnp.where(inc, base[k] * ainc_new, base[k]) for k in base}, F)  # kfsw: B4
        n["TDER"] = _w(a & p_apply, TDER2 * ainc_new, c["TDER"])  # kfsw: B4
        n["PPTFLX"] = _w(a & p_apply, PPTFL2 * ainc_new, c["PPTFLX"])  # kfsw: B4
        ends = a & (p_stop | p_noitr | p_exit)
        n["stop"] = c["stop"] | ends
        n["ret"] = c["ret"] | (a & (p_ret1 | p_ret2))
        n["NCOUNT"] = _w(ends | (a & (p_ret1 | p_ret2)), ncount, c["NCOUNT"])  # kfsw: B4
        return n, None

    C, _ = jax.lax.scan(iterate, it0, jnp.arange(1, _N_ITER + 1, dtype=i32))  # kfsw: S
    F, AINC = C["F"], C["AINC"]
    TDER, PPTFLX = C["TDER"], C["PPTFLX"]
    OMG, FXM, NSTEP, DTIME = C["OMG"], C["FXM"], C["NSTEP"], C["DTIME"]
    TG, QG = C["TG"], C["QG"]

    # --- hydrometeors (l.2318-2382) ---
    FBFRC = sc["FBFRC"]
    cpos = CPR > 0.0  # kfsw: D7k
    FRC2 = jnp.where(cpos, PPTFLX / _sd(CPR * AINC, cpos), 0.0)  # kfsw: D7k
    RAINFB = PPTLIQ * AINC * FBFRC * FRC2
    SNOWFB = PPTICE * AINC * FBFRC * FRC2
    QLG = advect(zf, lambda fin, fout: fin + F["DETLQ"] - fout, OMG, FXM, NSTEP, DTIME)
    QIG = advect(zf, lambda fin, fout: fin + F["DETIC"] - fout, OMG, FXM, NSTEP, DTIME)
    QRG = advect(zf, lambda fin, fout: fin - fout + RAINFB, OMG, FXM, NSTEP, DTIME)
    QSG = advect(zf, lambda fin, fout: fin - fout + SNOWFB, OMG, FXM, NSTEP, DTIME)
    QLG, QIG, QRG, QSG = (jnp.where(inc, x, 0.0) for x in (QLG, QIG, QRG, QSG))  # kfsw: C-LTOP

    # --- feedback (l.2503-2692) ---
    PRATEC = PPTFLX * (1.0 - FBFRC) / DXSQ
    NICa = jnp.where(TADVEC < TIMEC, _nint(TADVEC / dt), NIC)  # kfsw: C-NIC
    NCA = jnp.where(sh, cudt * _QSEC_PER_MIN, NICa.astype(dtype) * dt)  # kfsw: B1,C-NIC
    TIMECf = jnp.where(sh, _TIMEC_SHALLOW_S, TIMEC)  # kfsw: B1
    CPM = CP * (1.0 + _CPM_Q * QG)
    if warm_rain:  # kfsw: S
        TGf = TG - (QIG + QSG) * _RLF / CPM
        DQCDT = (QLG + QIG) / TIMECf
        DQIDT = jnp.zeros_like(TG)
        DQRDT = (QRG + QSG) / TIMECf
        DQSDT = jnp.zeros_like(TG)
    elif not f_qs:  # kfsw: S
        # kfsw: C-ML
        TGf = jnp.where(kk <= ML, TG - (QIG + QSG) * _RLF / CPM, TG + (QLG + QRG) * _RLF / CPM)
        DQCDT = (QLG + QIG) / TIMECf
        DQIDT = jnp.zeros_like(TG)
        DQRDT = (QRG + QSG) / TIMECf
        DQSDT = jnp.zeros_like(TG)
    else:
        TGf = TG
        DQCDT = QLG / TIMECf
        DQRDT = QRG / TIMECf
        if f_qi:  # kfsw: S
            DQSDT = QSG / TIMECf
            DQIDT = QIG / TIMECf
        else:
            DQSDT = QSG / TIMECf + QIG / TIMECf
            DQIDT = jnp.zeros_like(TG)
    DTDT = (TGf - T0) / TIMECf
    DQDT = (QG - Q0) / TIMECf

    fired = fired0 & ~C["ret"]
    bad = fired & (C["fatal"] | C["overflow"])
    nanv = jnp.asarray(jnp.nan, dtype)

    def fin(x, fill=0.0):
        return jnp.where(bad, nanv, jnp.where(fired, x, fill))  # kfsw: A0

    sl = slice(1, KX + 1)  # kfsw: S
    out = dict(
        dtdt=fin(DTDT)[sl], dqdt=fin(DQDT)[sl], dqcdt=fin(DQCDT)[sl], dqrdt=fin(DQRDT)[sl],
        dqidt=fin(DQIDT)[sl], dqsdt=fin(DQSDT)[sl], pratec=fin(PRATEC), nca=fin(NCA),
        cutop=jnp.where(fired, LTOP, 1), cubot=jnp.where(fired, LCL, KX + 1),
        ishall=jnp.where(fired, ISHALL, 2), fired=fired,
        # diagnostics (end-of-routine state; meaningful where fired, levels 1..LTOP)
        umf=fin(F["UMF"])[sl], uer=fin(F["UER"])[sl], udr=fin(F["UDR"])[sl],
        dmf=fin(F["DMF"])[sl], der=fin(F["DER"])[sl], ddr=fin(F["DDR"])[sl],
        detlq=fin(F["DETLQ"])[sl], detic=fin(F["DETIC"])[sl],
        tu=fin(TU)[sl], qu=fin(QU)[sl], qliq=fin(QLIQ)[sl], qice=fin(QICE)[sl],
        pptliq=fin(PPTLIQ)[sl], pptice=fin(PPTICE)[sl], tg=fin(TGf)[sl], qg=fin(QG)[sl],
        eqfrc=fin(EQFRC)[sl], ems=EMS[sl],
        ltop=jnp.where(fired, LTOP, 0), let=jnp.where(fired, LET, 0),
        abe=fin(sc["ABE"]), vmflcl=fin(VMFLCL), ainc=fin(AINC), timec=fin(TIMECf),
        ncount=jnp.where(fired, C["NCOUNT"], 0), fabe=fin(C["FABE"]), tder=fin(TDER),
        trppt=fin(TRPPT), pptflx=fin(PPTFLX), cpr=fin(CPR), frc2=fin(FRC2), fbfrc=fin(FBFRC),
        lfs=jnp.where(fired, LFS, 0), ldb=jnp.where(fired, LDB, 0), wlcl=fin(sc["WLCL"]),
        rad=fin(sc["RAD"]), chmin=fin(sc["CHMIN"]), cldhgt=fin(S["CLDHGT"][LC]),
        lcl=jnp.where(fired, LCL, 0), au0=fin(AU0), nstep=jnp.where(fired, NSTEP, 0),
        kpbl=jnp.where(fired, KPBL, 0), lc=jnp.where(fired, LC, 0),
    )  # kfsw: A0
    return out


# Oracle values of the numbers legoESM exposes as KainFritschConfig fields.
KF_ORACLE_PARAMS = dict(
    dpmin=5.0e3,          # DPMIN (l.746)
    wklcl_ref=0.02,       # Kain (2004) Eq. 2 (l.979-981)
    wklcl_zref=2.0e3,
    dtlcl_coeff=4.64,     # Kain (2004) Eq. 1 (l.987)
    dtlcl_exponent=0.33,
    rh_u00=0.75,          # trigger 3 DTRH (l.1009-1012)
    rh_max=0.95,
    rh_slope=0.25,
    rad_min=1000.0,       # Kain (2004) Eq. 6 (l.1055-1061)
    rad_max=2000.0,
    rad_wkl_ref=0.1,
    entrain_const=0.03,   # Kain (2004) Eq. 5 (l.1194)
    timec_min=1800.0,     # TIMEC clamp (l.1596-1597)
    timec_max=3600.0,
)


def kf_eta_para(u, v, t, qv, p, dzq, w0avg, rhoe, *, dt, dx, constants: KFConstants,
                params=None, trigger: int = 3, warm_rain: bool = True, f_qi: bool = False,
                f_qs: bool = False, cudt: float = 0.0, smoothing: KFSmoothing | None = None):
    """KF_eta_PARA on a batch of columns.

    All profile inputs have shape ``(ncol, nlev)``, levels BOTTOM-UP, ``qv`` a
    mixing ratio.  ``trigger`` is the WRF ``kf_trigger`` option (1 or 3);
    ``warm_rain``/``f_qi``/``f_qs`` select the oracle's hydrometeor-feedback
    branch.  ``cudt`` [min] only sets ``nca`` for shallow convection.
    ``smoothing`` holds the RULE 5 widths; ``None`` = the smooth defaults of
    ``KainFritschConfig`` (:func:`kf_smoothing`), ``KF_HARD`` = the oracle.
    Returns a dict of the oracle outputs (and end-of-routine diagnostics), each
    with a leading column axis.
    """
    if trigger not in (1, 3):
        raise ValueError(
            f"kf_eta_para: trigger={trigger!r} is not ported (WRF kf_trigger 1 or 3; "
            "trigger 2 needs the horizontal neighbourhood)")
    prm = dict(KF_ORACLE_PARAMS if params is None else params)
    missing = set(KF_ORACLE_PARAMS) - set(prm)
    if missing:
        raise ValueError(f"kf_eta_para: params missing {sorted(missing)}")
    # Mixed-precision callers (fp32 T with fp64 pressure) promote to one dtype so
    # the scan carries are uniform; no input loses precision.
    dtype = jnp.result_type(u, v, t, qv, p, dzq, w0avg, rhoe)
    u, v, t, qv, p, dzq, w0avg, rhoe = (
        jnp.asarray(x, dtype) for x in (u, v, t, qv, p, dzq, w0avg, rhoe))

    def num(x):  # python numbers stay weakly typed; arrays/tracers take the column dtype
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        return jnp.asarray(x, dtype)

    constants = KFConstants(*(float(x) for x in constants))
    prm = {k: num(x) for k, x in prm.items()}
    dt, dx = num(dt), num(dx)
    sm = kf_smoothing(KainFritschConfig()) if smoothing is None else KFSmoothing(*smoothing)
    validate_kf_smoothing(sm)
    sm = KFSmoothing(*(float(w) for w in sm))
    lut = _kf_tables(constants, dtype, sm)
    col = functools.partial(
        _column, dt=dt, dx=dx, kc=constants, lut=lut, prm=prm, trigger=trigger,
        warm_rain=warm_rain, f_qi=f_qi, f_qs=f_qs, cudt=cudt)
    return jax.vmap(col)(u, v, t, qv, p, dzq, w0avg, rhoe)
