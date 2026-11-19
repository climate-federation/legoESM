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
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.convection.config import KainFritschConfig
from legoesm.core.smooth import (
    positive_part_c1,
    smooth_max_c1,
    smooth_min_c1,
    smoothstep_cubic,
    smoothstep_quintic,
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
    "B3s": ("B", "smooth", "downdraft acceptance as weights: dd_ok, deep_dd, nodd onset, ddinc cap"),
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
    "C-LDBs": ("C", "smooth", "downdraft base by the descent's survival weights (argmax = LDB diagnostic)"),
    "C-LET": ("C", "hard", "equilibrium-temperature level LET and the cloud-top detrainment layer"),
    "C-LFS": ("C", "hard", "level of free sink LFS (KLFS) and LDT"),
    "C-LFSs": ("C", "smooth", "soft LFS search: first-crossing weights of the 150 hPa depth test"),
    "C-LTOP": ("C", "hard", "updraft exits (WTW, UMF-UDR), cloud top LTOP and its masks"),
    "C-LTOPs": ("C", "smooth", "cloud top by the updraft's survival weights and its converted readers"),
    "C-LETs": ("C", "smooth", "soft LET given the top: joint (LET, LTOP) survival weights"),
    "C-ML": ("C", "hard", "melting level ML (T0 > T00 below LTOP)"),
    "C-MLs": ("C", "smooth", "soft warm-level flags: melting given the top and the freezing-level weights"),
    "C-NIC": ("C", "hard", "TIMEC rounded to NINT(TIMEC/DT) steps, NCA"),
    "C-NSTEP": ("C", "hard", "subsidence sub-step count NSTEP and its step mask"),
    "D1": ("D", "smooth", "DTLCL = 0 for WKL < 1e-4"),
    "D2": ("D", "hard", "DTRH three-way split on RHLCL"),
    "D3": ("D", "smooth", "WLCL = 1 unless DTTOT > 1e-4"),
    "D4": ("D", "smooth", "entrainment selection cold / vb / tiny (ee2, ud2, eqfrc)"),
    "D5": ("D", "hard", "shear sign SHSIGN"),
    "D6": ("D", "hard", "RCBH polynomial / constant splice at CBH 3 and 25 kft"),
    "D7a": ("D", "smooth", "TPMIX2 saturated / enough condensate / dry branches"),
    "D7b": ("D", "smooth", "CONDLOAD g1 < 0 set to 0"),
    "D7c": ("D", "hard", "CONDLOAD mean vertical velocity zero guard"),
    "D7d": ("D", "smooth", "CONDLOAD |WTW| < 1e-4 floor"),
    "D7e": ("D", "smooth", "PROF5 y >= 0 polynomial branch and |y|"),
    "D7f": ("D", "hard", "PROF5 eqf == 0 / eqf == 1 exact-equality overrides"),
    "D7g": ("D", "smooth", "glaciation: freezing (frz) and partial (part) ramps, DTFRZNEW"),
    "D7h": ("D", "hard", "downdraft RH adjustment: low (qsrh < qd) and adj (rhh < 1)"),
    "D7hs": ("D", "smooth", "downdraft RH adjustment low (qsrh < qd) blended"),
    "D7i": ("D", "hard", "subsidence moisture borrow where QG < 0"),
    "D7j": ("D", "smooth", "closure mixed-layer supersaturation adjustment (sup)"),
    "D7k": ("D", "hard", "CPR > 0 guard of the precipitation feedback ratio FRC2 (kept: products with PPTLIQ/PPTICE <= CPR are continuous)"),
    "E1": ("E", "smooth", "Q0 clipped to [1e-6, QES]"),
    "E2": ("E", "smooth", "TLCL <= TMIX (first pass and closure)"),
    "E3": ("E", "smooth", "WKLCL linear ramp capped at wklcl_ref"),
    "E4": ("E", "smooth", "WLCL <= 3 m/s"),
    "E5": ("E", "smooth", "RAD ramp clipped to [rad_min, rad_max]"),
    "E6": ("E", "smooth", "ABE accumulates only positive dilbe"),
    "E7": ("E", "smooth", "eqf clipped to [0, 1]"),
    "E8": ("E", "smooth", "ee2 >= 0.5"),
    "E9": ("E", "smooth", "CHMIN temperature ramp between 273 and 293 K"),
    "E10": ("E", "smooth", "upstream direction OMG <= 0 in the subsidence advection (smooth: face weight on OMG)"),
    "E11": ("E", "hard", "square root kink at zero: GDT, wind speed, shear"),
    "E11c": ("E", "smooth", "CONDLOAD square roots of WTW and G1"),
    "E12": ("E", "smooth", "TIMEC = DX/VCONV clamped to [timec_min, timec_max] (VCONV > 0 stays a divisor guard)"),
    "E13": ("E", "smooth", "PEF clamped to [0.2, 0.9]"),
    "E14": ("E", "smooth", "PEFCBH <= 0.9"),
    "E15": ("E", "smooth", "inflow limit AINCMX: UER-DER > 1e-3 mask, min reduction, AINC <= 1"),
    "E16": ("E", "hard", "subsidence time step DTT limited by |OMG|*TIMEC > 0.75 DP"),
    "E17": ("E", "smooth", "closure qmix >= 0"),
    "E18": ("E", "smooth", "DABE >= 0.1 ABE"),
    "E19": ("E", "smooth", "new AINC capped at AINCMX"),
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
# the C1 CHMIN clamps the ramp, so its top must be the oracle's deep value
# kfsw: S
assert _CHMIN_DEEP_M == _CHMIN_SHALLOW_M + _CHMIN_SLOPE_M_PER_K * (_CHMIN_TWARM_K
                                                                  - _CHMIN_TCOLD_K)
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
    * Updraft (PR 4; cloud top LTOP, LET and the A4 test stay hard):
      ``wklcl_cap_width_m`` (E3 WKLCL ramp cap, symmetric), ``dtlcl_onset_width_ms``
      (D1, DTLCL ramps in above WKL = 1e-4), ``wlcl_onset_width_k`` (D3, WLCL
      ramps in above DTTOT = 1e-4 K), ``wlcl_cap_width_ms`` (E4, one-sided
      WLCL <= 3), ``rad_clip_width`` (E5, RAD ramp clip, in WKL/rad_wkl_ref),
      ``entrain_select_width_k`` (D4 cold and vb weights, centred) and
      ``entrain_tiny_width_k`` (D4 tiny weight, above tvdiff = 1e-3 K),
      ``eqf_clip_width`` (E7), ``ee2_floor_width`` (E8), ``frc_clip_width``
      (D7g frozen-fraction cap at 1) and ``freeze_onset_width_k`` (D7g blend
      of the frozen and unfrozen states below TTFRZ), ``abe_pos_width_m`` (E6
      positive part of dilbe, updraft, its stop2 rollback and the closure),
      ``chmin_clip_width_k`` (E9 CHMIN clamp in TLCL).  D7f (PROF5 endpoint
      overrides, a 2e-11 jump where eqf is constant) stays hard.
    * Cloud top (PR 5a): ``top_stop_wtw_width_m2s2`` / ``top_stop_umf_width_kgs``
      are the bands of the stop1 (WTW < 1e-3) and stop2 (UMF - UDR < 10)
      hazards of the updraft's survival; either nonzero runs the updraft to
      the model top and replaces the hard LTOP/LET by their joint survival
      weights (the LET cold weight is D4's, width ``entrain_select_width_k``).
      The downdraft (PR 5b) and the closure (PR 5c) read these weights.
    * Downdraft (PR 6a): ``lfs_depth_width_pa`` is the band of the first-
      crossing weights of the LFS search (first level 150 hPa above the
      source-layer top; C-LFSs); ``melt_t_width_k`` the band of the soft warm
      flags T0 > T00 (C-MLs) of the melting test (a warm level between the
      source layer and the top) and of the freezing-level correction.  Either
      nonzero runs the survival-weighted path like the cloud-top widths (with
      both top widths 0 its top weights are the oracle's one-hot top).
      KNOWN DIFFERENCE (fixture, max over columns and levels, vs the hard
      limit's gap): the LFS width alone moves the sweep's heating by up to
      2.8e-6 K/s and its precipitation rate by 1.6e-6 (kg/m2/s); the melting
      width moves no fixture case (no level within 0.05 K of T00, the half-width
      band of the C2 step).  The warm
      flags are treated as independent per level.  They set the downdraft's
      melting cooling (each source-level run takes its members' conditional
      mean melting ice, as in PR 5b; the cooling is nonlinear in it) and the
      freezing-level split of the condensate heating.  Each run's downdraft
      evaporation is debited from its precipitation (PPTFLX = TRPPT - TDER),
      so the water budget closes at any melting probability (tested).
    * Downdraft acceptance (PR 6b; any nonzero runs the survival path):
      ``dd_ok_depth_width_pa`` the band of the two 50 hPa depth tests (the
      source above the source layer, dd_ok, and the base below the source,
      deep_dd; one band, so deep_dd's weight is at least dd_ok's at every
      base below the source layer; vote 24), ``ldb_tv_width_k`` the band of the descent's stop hazard
      TVD > TV0 (the base LDB by survival weights), ``nodd_tder_width_kgs``
      the onset band of the acceptance TDER >= 1 kg/s, ``ddinc_cap_width``
      the one-sided band of TDER * DDINC <= TRPPT (in units of TRPPT) and
      ``dd_rh_low_width_kgkg`` the band of the RH-adjustment branch
      qsrh < qd (D7h ``low``).  D7h ``adj`` (rhh < 1) stays hard: the descent
      runs below the source layer, where rhh < 1 always.
      KNOWN DIFFERENCE (fixture, max over columns and levels, vs the hard
      limit's gap of 2.3e-15 K/s): the descent-stop width moves the heating
      by up to 4.4e-7 K/s (warm-rain column at 20 m/s); dd_ok/deep_dd, the
      TDER onset, the DDINC cap and the RH branch move no fixture case (no
      downdraft within their bands; dd_ok is exercised on a column shifted
      into its band in-test).  Each base's run closes its own water budget
      (PPTFLX = TRPPT - TDER per run), so the mixture closes it too (tested).
    * Closure-local clamps (PR 7a, forms vote 19, widths vote 20):
      ``omg_upwind_width_pas`` (E10, upwind side of the subsidence advection,
      the same face weight on both sides), ``timec_clip_width_s`` (E12 TIMEC
      clip), ``pef_clip_width`` (E13), ``pefcbh_cap_width`` (E14, one-sided),
      ``uerder_onset_width_kgs`` (E15 eligibility UER - DER > 1e-3, centred),
      ``ainc_min_width`` (E15 level minimum of the inflow limits and AINC <= 1;
      a relative width, the minima are taken in log space),
      ``ainc_cap_width`` (E19 AINC <= AINCMX, one-sided), ``qmix_floor_width_kgkg``
      (E17 qmix >= 0, one-sided), ``dabe_floor_width_jkg`` (E18 DABE >= 0.1 ABE),
      ``closure_sup_width_kgkg`` (D7j, blend of the supersaturated and
      unsaturated closure mixed layer, centred on qmix = qss).  D7k (CPR > 0
      guard of FRC2) stays hard: every consumer multiplies FRC2 by PPTLIQ or
      PPTICE <= CPR, so the products are continuous (vote 21).
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
    wklcl_cap_width_m: float
    dtlcl_onset_width_ms: float
    wlcl_onset_width_k: float
    wlcl_cap_width_ms: float
    rad_clip_width: float
    entrain_select_width_k: float
    entrain_tiny_width_k: float
    eqf_clip_width: float
    ee2_floor_width: float
    frc_clip_width: float
    freeze_onset_width_k: float
    abe_pos_width_m: float
    chmin_clip_width_k: float
    top_stop_wtw_width_m2s2: float
    top_stop_umf_width_kgs: float
    lfs_depth_width_pa: float
    melt_t_width_k: float
    dd_ok_depth_width_pa: float
    ldb_tv_width_k: float
    nodd_tder_width_kgs: float
    ddinc_cap_width: float
    dd_rh_low_width_kgkg: float
    omg_upwind_width_pas: float
    timec_clip_width_s: float
    pef_clip_width: float
    pefcbh_cap_width: float
    uerder_onset_width_kgs: float
    ainc_min_width: float
    ainc_cap_width: float
    qmix_floor_width_kgkg: float
    dabe_floor_width_jkg: float
    closure_sup_width_kgkg: float


KF_HARD = KFSmoothing(*(0.0 for _ in KFSmoothing._fields))


def _survival_on(sm: KFSmoothing) -> bool:
    """The survival-weighted (mixture) path runs when a cloud-top or a
    downdraft width is set (static)."""
    return bool(sm.top_stop_wtw_width_m2s2 or sm.top_stop_umf_width_kgs  # kfsw: S
                or sm.lfs_depth_width_pa or sm.melt_t_width_k or _soft_downdraft_on(sm))


def _soft_downdraft_on(sm: KFSmoothing) -> bool:
    """The downdraft runs as the mixture over its base and acceptance (PR 6b)."""
    return bool(sm.dd_ok_depth_width_pa or sm.ldb_tv_width_k  # kfsw: S
                or sm.nodd_tder_width_kgs or sm.ddinc_cap_width
                or sm.dd_rh_low_width_kgkg)


def _gt_c2(x, w):
    """Weight of x > 0: the oracle's 0/1 at w = 0, else the C2 step (B3s, C-LDBs, D7hs)."""
    if w:  # kfsw: S
        return _step_c2(x, w)
    return (x > 0.0).astype(jnp.result_type(x))  # kfsw: B3s,C-LDBs,D7hs


def _descent_base_weights(h, lev):
    """P(LDB = b) for stop hazards ``h`` [NP] of a descent tested downward over
    the levels ``lev`` (top first): h_b * prod_{j > b in lev} (1 - h_j)."""
    q = jnp.where(lev, 1.0 - h, 1.0)  # kfsw: C-LDBs
    tail = jnp.concatenate([jnp.cumprod(q[::-1])[::-1][1:], jnp.ones((1,), q.dtype)])  # kfsw: C-LDBs
    return jnp.where(lev, h * tail, 0.0)  # kfsw: C-LDBs
# Largest PROF5 blend half-width: entrainment and detrainment fractions stay
# non-negative on eq in [0, 1] (tested); wider blends reach the far branch's
# negative values (ud < 0 at width 3).
_PROF5_WIDTH_MAX = 0.1
# kfsw: S
assert _PROF5_WIDTH_MAX < 1.0 / _P5_P  # the branch continuations' pole stays outside the blend


def kf_smoothing(config: KainFritschConfig) -> KFSmoothing:
    """The widths a ``KainFritschConfig`` selects (its ``wrf_<field>`` fields)."""
    return KFSmoothing(*(getattr(config, f"wrf_{f}") for f in KFSmoothing._fields))  # kfsw: S


def validate_kf_smoothing(sm: KFSmoothing) -> None:
    for name, w in sm._asdict().items():  # kfsw: S
        validate_smoothing_width(f"KFSmoothing.{name}", w)  # kfsw: S
    # the stop1 band stays at WTW >= 0; the stop2 band stays above the
    # continuation floor (identity from _CONT_HI_KGS up); the acceptance onset band stays at
    # TDER > 0 (the run's DDINC divides by its TDER where it is accepted); the UER - DER
    # eligibility band stays above 0 (the inflow limit divides by it)
    for name, hi in (("lut_knot_width", 1.0), ("prof5_width", _PROF5_WIDTH_MAX),
                     ("top_stop_wtw_width_m2s2", 2.0 * _WTW_STOP),
                     ("top_stop_umf_width_kgs", 2.0 * (_UMF_DETR_MIN - _CONT_HI_KGS)),
                     ("nodd_tder_width_kgs", 2.0 * _TDER_MIN),
                     ("uerder_onset_width_kgs", 2.0 * _UERDER_MIN)):
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
    return smoothstep_cubic(0.5 + x / w)  # kfsw: C-KLCLg,C-L5g,E10


def _step_c2(x, w):
    """C2 (quintic) step centred on 0, band |x| < w/2 (w > 0): 0 below, 1
    above (D4 weights, cloud-top survival hazards and the soft LET)."""
    return smoothstep_quintic(0.5 + x / w)  # kfsw: D4,C-LTOPs,C-LETs,E15,D7j


# Continuation floor of the updraft's mass fluxes past a likely stop2 exit
# (RULE 5 survival): identity above _CONT_HI, C2 quintic join down to _CONT_LO,
# constant below; it only acts where the stop2 hazard is already 1.
_CONT_LO_KGS = 1.0
_CONT_HI_KGS = 5.0
# offset of the downdraft mixture's conditional-mean divisor (finite gradients
# for shares down to 0 in fp32)
_QSHARE_FLOOR = 1.0e-15


def _cont_floor(x):
    """Keeps the continuing plume finite and smooth below the stop2 band."""
    t = (x - _CONT_LO_KGS) / (_CONT_HI_KGS - _CONT_LO_KGS)
    return _CONT_LO_KGS + (x - _CONT_LO_KGS) * smoothstep_quintic(t)  # kfsw: C-LTOPs


def _conv_abe(dabe, ptop):
    """ABE summed over the accepted members: the increment of level k counts
    for the members whose top lies above k (a stop2 top rolls its own back)."""
    return jnp.sum(dabe[:-1] * jnp.cumsum(ptop[::-1])[::-1][1:])  # kfsw: C-LTOPs


def _top_hazards(wtw, upold, sm):
    """Exit probabilities of the continuing updraft at a level: stop1 (WTW
    below 1e-3) and stop2 (UMF - UDR below 10 kg/s); a zero width is the
    oracle's step."""
    h = []
    for x, w in ((_WTW_STOP - jnp.asarray(wtw), sm.top_stop_wtw_width_m2s2),
                 (_UMF_DETR_MIN - jnp.asarray(upold), sm.top_stop_umf_width_kgs)):
        if w:  # kfsw: S
            h.append(_step_c2(x, w))  # kfsw: C-LTOPs
        else:
            h.append((x > 0.0).astype(jnp.result_type(x)))  # kfsw: C-LTOPs
    return h


def _top_weights(E1, E2, s_ex, CW, kk, K, LCL, KX, KPBL, let_fixed=None):
    """Joint weights W[j, t] of (LET = j, LTOP = t) from the survival of the
    continuing updraft (C-LTOP, C-LET).  ``E1``/``E2`` [NP]: stop1/stop2 exit
    masses at level t; ``s_ex``: survival past the last level (oracle LTOP = KX);
    ``CW`` [NP]: soft cold weight of level j (LET is the last non-cold level
    <= t, else the initial LET = LCL); stop2 puts LET at t.  ``let_fixed``
    (traced index or None) overrides LET (the shallow branch).  A member the
    oracle's no-convection test rejects (LTOP <= max(LCL, KPBL) or
    LET + 1 <= KPBL) has zero weight, so the weights sum to P(convection)."""
    j, t = kk[:, None], kk[None, :]  # kfsw: S
    # levels below K hold an earlier candidate's values: no top there
    E1, E2 = (jnp.where(kk >= K, x, 0.0) for x in (E1, E2))  # kfsw: S
    p1 = E1 + jnp.where(kk == KX, s_ex, 0.0)  # kfsw: S
    # P[j, t] = prod_{j < i <= t} CW[i]
    M = jnp.where((kk[:, None] > kk[None, :]), CW[:, None], 1.0)  # kfsw: S
    P = jnp.cumprod(M, axis=0)  # kfsw: C-LETs
    P = P.T  # [j, t]
    cand = (j >= K + 1) & (j <= t)  # kfsw: S
    r = jnp.where(cand, (1.0 - CW[:, None]) * P, 0.0)  # kfsw: C-LETs
    r = r + jnp.where(j == LCL, P[K, :][None, :], 0.0)  # kfsw: S
    W = p1[None, :] * r + jnp.where(j == t, E2[None, :], 0.0)  # kfsw: S
    conv = (t > jnp.maximum(LCL, KPBL)) & (j + 1 > KPBL)  # kfsw: C-LTOPs,C-LETs
    W = jnp.where(conv, W, 0.0)  # kfsw: C-LTOPs
    if let_fixed is None:  # kfsw: S
        return W
    # the override follows the no-convection test: rejected members stay out
    return jnp.where(j == let_fixed, jnp.sum(W, axis=0)[None, :], 0.0)  # kfsw: S


def _expected_top_profile(W, UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, QLIQ, QICE,
                          DILFRC, QLQOUT, QICOUT, UPO, UPN, DP, kk):
    """Expectation over the (LET = j, LTOP = t) members, weights ``W`` [j, t],
    of the oracle's cloud-top post-processing (l.1421-1475, LET..LTOP block)
    applied to the continuing plume arrays (C-LTOP, C-LET).  Member (j, t):
    levels below j keep the plume (j > t is treated as j = t); j < k < t
    taper UMF linearly in pressure from UMF[j] to 0 at t, with UER/UDR/DET and
    the precipitation from j + 2 recomputed; t keeps its stored UMF and
    detrains UMF[t-1]; j = t detrains everything at t (UPN/UPO of t - 1);
    above t is zero.  Plume states do not depend on the member, so every
    flux the downstream uses is an expectation of fluxes.  Returns expected
    UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE."""
    j, t = kk[:, None], kk[None, :]  # kfsw: S
    cdp = jnp.cumsum(DP)  # kfsw: C-LTOPs
    djt = cdp[None, :] - cdp[:, None]  # kfsw: S
    # a member with LET above its top (only where the oracle would not
    # convect) detrains everything at the top, so every member closes its mass
    lt, eq = j < t, j >= t  # kfsw: S
    dsafe = jnp.where(lt, djt, 1.0)  # kfsw: S
    upo1 = jnp.concatenate([UPO[:1], UPO[:-1]])  # UPO at t - 1
    upn1 = jnp.concatenate([UPN[:1], UPN[:-1]])

    def umf_m(k):  # taper / plume value of member (j, t) at level k <= t (k < t when tapered)
        taper = UMF[:, None] * (1.0 - (cdp[k] - cdp[:, None]) / dsafe)  # kfsw: S
        return jnp.where(lt & (k > j), taper, UMF[k])  # kfsw: S

    def level(_, k):
        on = k <= t  # kfsw: S
        top = k == t  # kfsw: S
        um = umf_m(k)
        um1 = umf_m(jnp.maximum(k - 1, 0))  # kfsw: C-LTOPs (level 0: padding, unused)
        mid = lt & (k > j) & (k < t)  # kfsw: S
        uer_mid = um * (1.0 - 1.0 / DILFRC[k])  # kfsw: S
        umf = jnp.where(top & eq, 0.0, jnp.where(mid, um, UMF[k]))  # kfsw: S
        uer = jnp.where(top, 0.0, jnp.where(mid, uer_mid, UER[k]))  # kfsw: S
        udr_eq = UMF[k] + UDR[k] - UER[k]  # kfsw: S
        udr = jnp.where(top & eq, udr_eq, jnp.where(top & lt, um1, jnp.where(
            mid, um1 - um + uer_mid, UDR[k])))  # kfsw: S
        upo_s = jnp.where(upo1 == 0.0, 1.0, upo1)  # kfsw: C-LTOPs
        rat = jnp.where(top & eq, (upn1 / upo_s)[None, :], DILFRC[k])  # kfsw: S
        new = (top & eq) | (lt & (k > j))  # kfsw: S
        detlq = jnp.where(new, udr * QLIQ[k] * rat, DETLQ[k])  # kfsw: S
        detic = jnp.where(new, udr * QICE[k] * rat, DETIC[k])  # kfsw: S
        rec = lt & (k >= j + 2)  # kfsw: S
        pl = jnp.where(rec, um1 * QLQOUT[k], PPTLIQ[k])  # kfsw: S
        pi_ = jnp.where(rec, um1 * QICOUT[k], PPTICE[k])  # kfsw: S
        # kfsw: S
        out = tuple(jnp.sum(jnp.where(on, W * x, 0.0))
                    for x in (umf, uer, udr, detlq, detic, pl, pi_))
        return None, out

    _, outs = jax.lax.scan(level, None, kk)
    return outs


def _first_crossing(s, kk, KX):
    """P(kl0 = k) for crossing probabilities ``s`` [NP] tested upward (the
    first level that crosses; none crosses: KX, the oracle's no-candidate
    value).  One-hot at the oracle's kl0 for 0/1 ``s`` (C-LFS)."""
    q = jnp.cumprod(1.0 - s)  # kfsw: C-LFSs
    pi = s * jnp.concatenate([jnp.ones((1,), s.dtype), q[:-1]])
    return pi + jnp.where(kk == KX, q[-1], 0.0)  # kfsw: S


def _lfs_member_weights(pi, kk):
    """QL[l, j] = P(min(kl0, j - 1) = l), kl0 ~ ``pi``: the downdraft source
    LFS of a member with LET = j (kl0 = l below j - 1; kl0 >= j - 1 at it)."""
    pge = jnp.cumsum(pi[::-1])[::-1]  # P(kl0 >= l)  # kfsw: C-LFSs
    lj, jj = kk[:, None], kk[None, :]  # kfsw: S
    # kfsw: S
    return jnp.where(lj < jj - 1, pi[:, None], 0.0) + jnp.where(
        lj == jj - 1, pge[jnp.clip(jj - 1, 0, kk.shape[0] - 1)], 0.0)


def _any_in_range(v, kk):
    """NW[k, t] = P(some level j in [k, t] is warm) for independent warm
    probabilities ``v`` [NP]; 0 for t < k.  The oracle's 0/1 indicators for
    0/1 ``v`` (C-ML: melting given the top, freezing-level weights)."""
    # kfsw: C-MLs
    return 1.0 - jnp.cumprod(jnp.where(kk[None, :] >= kk[:, None], 1.0 - v[None, :], 1.0), axis=1)


def _cap_c1(x, cap, w):
    """min(x, cap) (w = 0, the oracle) or its one-sided C1 form: never above
    ``cap``, exactly ``x`` below ``cap - w`` (E1 Q0 <= QES and, negated, Q0 >=
    1e-6; E2 TLCL <= TMIX; E4 WLCL <= 3 m/s)."""
    if w:  # kfsw: S
        return cap - positive_part_c1(cap - x, 1.0 / w)  # kfsw: E1,E2,E4,E14,E17,E19
    return jnp.minimum(x, cap)  # kfsw: E1,E2,E4,E14,E17,E19


def _glaciate(ab, p, tu, theteu, qu, qliq, qice, qnl, qni, ttemp_prev, sm):
    """Updraft glaciation at one level (l.1141-1161, DTFRZNEW).  Returns the
    state, the TTEMP memory and the all-frozen flag (IFLAG).  Smooth (D7g): the
    frozen fraction's cap at 1 (``frc_clip_width``) and the frozen/unfrozen
    states blended over ``freeze_onset_width_k`` below TTFRZ (one transfer,
    a convex combination, so water is conserved exactly)."""
    frz = tu <= _TTFRZ_K  # kfsw: D7g
    part = tu > _TBFRZ_K  # kfsw: D7g
    ttemp0 = jnp.where(part & (ttemp_prev > _TTFRZ_K), _TTFRZ_K, ttemp_prev)  # kfsw: D7g
    if sm.frc_clip_width:  # kfsw: S
        # the oracle's frc1 = 1 below TBFRZ is the cap of the ramp at 1.
        # KNOWN DIFFERENCE: with the TTEMP memory already at or below TBFRZ
        # and tu warmed back above it, the oracle divides by ttemp0 - TBFRZ
        # <= 0 (frc1 > 1, inf at equality); this form freezes all (frc1 = 1).
        pos = ttemp0 > _TBFRZ_K  # kfsw: D7g
        ramp = (ttemp0 - tu) / _sd(ttemp0 - _TBFRZ_K, pos)  # kfsw: D7g
        frc1 = jnp.where(pos, smooth_min_c1(ramp, 1.0, sm.frc_clip_width), 1.0)  # kfsw: D7g
    else:
        # kfsw: D7g
        frc1 = jnp.where(part, (ttemp0 - tu) / _sd(ttemp0 - _TBFRZ_K, part & frz), 1.0)
    if sm.freeze_onset_width_k:  # kfsw: S
        # frozen and unfrozen states blended below TTFRZ (exact above it)
        bf = smoothstep_cubic((_TTFRZ_K - tu) / sm.freeze_onset_width_k)  # kfsw: D7g
    else:
        frc1 = jnp.where(frz, frc1, 0.0)  # kfsw: D7g
    ttemp = jnp.where(frz, tu, ttemp_prev)  # kfsw: D7g
    qfrz = (qliq + qnl) * frc1
    qni_f = qni + qnl * frc1
    qnl_f = qnl - qnl * frc1
    qice_f = qice + qliq * frc1
    qliq_f = qliq - qliq * frc1
    tu_f, theteu_f, qu_f, qice_f2 = _dtfrznew(ab, tu, p, qu, qfrz, qice_f)
    if sm.freeze_onset_width_k:  # kfsw: S
        tu, theteu, qu, qice, qliq, qnl, qni = (
            old + bf * (new - old) for new, old in (
                (tu_f, tu), (theteu_f, theteu), (qu_f, qu), (qice_f2, qice),
                (qliq_f, qliq), (qnl_f, qnl), (qni_f, qni)))
    else:
        tu = _w(frz, tu_f, tu)  # kfsw: D7g
        theteu = _w(frz, theteu_f, theteu)  # kfsw: D7g
        qu = _w(frz, qu_f, qu)  # kfsw: D7g
        qice = _w(frz, qice_f2, qice)  # kfsw: D7g
        qliq = _w(frz, qliq_f, qliq)  # kfsw: D7g
        qnl = _w(frz, qnl_f, qnl)  # kfsw: D7g
        qni = _w(frz, qni_f, qni)  # kfsw: D7g
    return tu, theteu, qu, qliq, qice, qnl, qni, ttemp, frz & ~part


def _wklcl(zlcl, ref, zref, w):
    """WKLCL (E3): ref*ZLCL/zref capped at ref above zref (w = 0); the C1 form
    caps ZLCL with smooth_min_c1 over |ZLCL - zref| < w/2."""
    if w:  # kfsw: S
        return smooth_min_c1(zlcl, zref, w) * ref / zref  # kfsw: E3
    return jnp.where(zlcl < zref, ref * zlcl / zref, ref)  # kfsw: E3


def _dtlcl(wkl, coeff, expo, w):
    """DTLCL (D1): the oracle's 0 below WKL = 1e-4 m/s and coeff*WKL**expo
    above, a 0.2 K jump (w = 0); the C1 form ramps it in over (1e-4, 1e-4 + w)."""
    if w:  # kfsw: S
        on = smoothstep_cubic((wkl - _WKL_MIN) / w)  # kfsw: D1
        return on * coeff * jnp.maximum(wkl, _WKL_MIN) ** expo  # kfsw: D1
    small = wkl < _WKL_MIN  # kfsw: D1
    return jnp.where(small, 0.0, coeff * _sd(wkl, ~small) ** expo)  # kfsw: D1


def _wlcl(dttot, gdt, gdt0, w):
    """WLCL before its 3 m/s cap (D3): the oracle's 1 below DTTOT = 1e-4 K and
    1 + sqrt(GDT)/2 above, a 0.03 m/s jump (w = 0); the C1 form ramps the
    excess in over (1e-4, 1e-4 + w) K (``gdt0`` = GDT at 1e-4 K)."""
    if w:  # kfsw: S
        on = smoothstep_cubic((dttot - _DTTOT_MIN_K) / w)  # kfsw: D3
        return 1.0 + on * 0.5 * jnp.sqrt(jnp.maximum(gdt, gdt0))  # kfsw: D3
    big = dttot > _DTTOT_MIN_K  # kfsw: D3
    return jnp.where(big, 1.0 + 0.5 * _safe_sqrt(gdt), 1.0)  # kfsw: D3,E11


def _entrain_select(cold, vb, tiny, dcold, dvb, tvdiff, ee_s, ud_s, eqf, ws, wt):
    """(ee2, ud2, eqfrc) before the ee2 floor and the 1.5 detrainment scaling
    (D4): the oracle's cold (TV0 >= tvqu: 0.5, 1, 0) / vb or tiny (1, 0, 1) /
    PROF5 branches (ws = wt = 0); the C1 form weighs them with centred steps
    of ``dcold`` = TV0 - tvqu and ``dvb`` = tu95 - TV0 (width ws) and a tiny
    weight that ramps out over tvdiff in (1e-3, 1e-3 + wt) K, so PROF5 only
    weighs where the eqf divisor is at least 1e-3 K."""
    if not (ws or wt):  # kfsw: S
        ee2 = jnp.where(cold, 0.5, jnp.where(vb | tiny, 1.0, ee_s))  # kfsw: D4
        ud2 = jnp.where(cold, 1.0, jnp.where(vb | tiny, 0.0, ud_s))  # kfsw: D4
        eqfrc = jnp.where(cold, 0.0, jnp.where(vb | tiny, 1.0, eqf))  # kfsw: D4
        return ee2, ud2, eqfrc
    if ws:  # kfsw: S
        c, svb = _step_c2(dcold, ws), _step_c2(dvb, ws)
    else:
        c, svb = cold.astype(eqf.dtype), vb.astype(eqf.dtype)  # kfsw: D4
    if wt:  # kfsw: S
        sti = 1.0 - smoothstep_cubic((tvdiff - _TVDIFF_MIN) / wt)  # kfsw: D4
    else:
        sti = tiny.astype(eqf.dtype)  # kfsw: D4
    s = 1.0 - (1.0 - svb) * (1.0 - sti)
    return (0.5 * c + (1.0 - c) * (s + (1.0 - s) * ee_s),
            c + (1.0 - c) * (1.0 - s) * ud_s,
            (1.0 - c) * (s + (1.0 - s) * eqf))


def _abe_increment(dilbe, g, w):
    """(mask, increment) of ABE from one layer: the oracle adds g*dilbe where
    dilbe > 0 (w = 0); the C1 form adds g*positive_part_c1(dilbe) everywhere,
    the same in the updraft, its stop2 rollback and the closure (E6)."""
    if w:  # kfsw: S
        return True, g * positive_part_c1(dilbe, 1.0 / w)  # kfsw: E6
    return dilbe > 0.0, dilbe * g  # kfsw: E6


def _clip_c1(x, lo, hi, w):
    """clip(x, lo, hi) (w = 0, the oracle's max-then-min) or its C1 form, exact
    outside the bands |x - lo|, |x - hi| < w/2 (E5, E7, E9)."""
    if w:  # kfsw: S
        return smooth_min_c1(smooth_max_c1(x, lo, w), hi, w)  # kfsw: E5,E7,E9,E12,E13
    return jnp.minimum(jnp.maximum(x, lo), hi)  # kfsw: E5,E7,E9,E12,E13


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


def _survival_inflow_limit(aincm1, wl, cap):
    """A level's inflow limit divided by its survival share ``wl``, capped at
    ``cap`` >= _AINCMX0 (the closure's minimum then ignores it); divides only
    below the cap so a tiny share keeps values and gradients finite."""
    c = wl * cap > aincm1  # kfsw: C-LETs
    return jnp.where(c, aincm1 / _sd(wl, c), cap)  # kfsw: C-LETs


def _inflow_limit_c1(aincm1, elig, w):
    """AINCMX = min(_AINCMX0, the levels' limits) and AINC = min(AINCMX, 1) as
    a fold of C1 minima in log space (E15; ``w`` is a relative width, exact
    where two limits differ by more than a factor exp(w/2)).  Always
    0 < AINC <= min(AINCMX, 1) and AINCMX <= the hard minimum, at most a
    factor exp(-w/8) below it per tied level (vote 25).  ``elig`` in [0, 1]
    blends each level's limit with ``top`` (no-op in the fold)."""
    top = _AINCMX0 * math.exp(w)
    v = elig * jnp.where(elig > 0.0, jnp.minimum(aincm1, top), top) + (1.0 - elig) * top  # kfsw: E15
    if not w:  # kfsw: S
        acc = jnp.minimum(_AINCMX0, jnp.min(v))  # kfsw: E15
        return acc, jnp.minimum(acc, 1.0)  # kfsw: E15
    lacc, _ = jax.lax.scan(lambda a, x: (smooth_min_c1(a, x, w), None),  # kfsw: E15
                           jnp.asarray(math.log(_AINCMX0), aincm1.dtype), jnp.log(v))
    return jnp.exp(lacc), jnp.exp(smooth_min_c1(lacc, 0.0, w))  # kfsw: E15


def _floor0_c1(x, w):
    """max(x, 0) (w = 0, the oracle) or its one-sided C1 form, never below 0
    and exactly ``x`` above ``w`` (E17 closure qmix)."""
    if w:  # kfsw: S
        return -_cap_c1(-x, 0.0, w)
    return jnp.maximum(x, 0.0)  # kfsw: E17


def _sup_select(qmix, qss, sat, unsat, w):
    """The closure mixed layer's supersaturated states where qmix > qss, else
    the unsaturated ones (D7j); w > 0 blends them over |qmix - qss| < w/2."""
    if w:  # kfsw: S
        ws = _step_c2(qmix - qss, w)
        return tuple(ws * a + (1.0 - ws) * b for a, b in zip(sat, unsat))  # kfsw: D7j
    sup = qmix > qss  # kfsw: D7j
    return tuple(jnp.where(sup, a, b) for a, b in zip(sat, unsat))  # kfsw: D7j


def _feedback_ratio(pptflx, cpr, ainc):
    """FRC2 = PPTFLX / (CPR AINC), 0 at CPR = 0 (D7k, kept hard: vote 21).

    Every consumer multiplies it by PPTLIQ or PPTICE (summing to CPR = TRPPT),
    so it enters only through P = p FRC2 with 0 <= p <= CPR and
    0 <= PPTFLX <= CPR AINC, i.e. 0 <= FRC2 <= 1 (fixture-checked; PPTFLX is
    the closure-scaled flux).  Hence 0 <= P <= CPR: P is squeezed to 0 with
    CPR, and where a C1 non-negative CPR reaches 0 its derivative is 0, so P
    is differentiable there with derivative 0 from both sides.  The products
    are as smooth as the precipitation onset itself; the guard adds no kink.
    A ramp on the ratio would remove fed-back water."""
    cpos = cpr > 0.0  # kfsw: D7k
    return jnp.where(cpos, pptflx / _sd(cpr * ainc, cpos), 0.0)  # kfsw: D7k


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
    # RULE 5 survival of the updraft (soft cloud top, PR 5a)
    sv = _survival_on(sm)
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
        CLDHGT=zf, E1=zf, E2=zf, CW=zf, UPO=zf, UPN=zf, DABE=zf,
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
        LTOP=jnp.asarray(2, i32), CHMIN=zero, IFLAG=jnp.asarray(0, i32), SEX=zero,
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
        WKLCL = _wklcl(ZLCL, prm["wklcl_ref"], prm["wklcl_zref"], sm.wklcl_cap_width_m)
        WKL = gz(W0A) * dx / _WKL_DX_REF_M - WKLCL
        DTLCL = _dtlcl(WKL, prm["dtlcl_coeff"], prm["dtlcl_exponent"], sm.dtlcl_onset_width_ms)
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
        GDT = 2.0 * G * DTTOT * _GDT_DEPTH_M / TVEN
        gdt0 = 2.0 * G * _DTTOT_MIN_K * _GDT_DEPTH_M / TVEN  # GDT at the D3 threshold
        WLCL = _cap_c1(_wlcl(DTTOT, GDT, gdt0, sm.wlcl_onset_width_k), _WLCL_MAX,
                       sm.wlcl_cap_width_ms)
        PLCL = gz(P0)
        WTW = WLCL * WLCL
        TVLCL = TLCL * (1.0 + _TV_COEF * QMIX)
        RHOLCL = PLCL / (R * TVLCL)
        LCL = KLCL
        if sm.rad_clip_width:  # kfsw: S
            RAD = prm["rad_min"] + (prm["rad_max"] - prm["rad_min"]) * _clip_c1(
                WKL / prm["rad_wkl_ref"], 0.0, 1.0, sm.rad_clip_width)
        else:
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
                 LTOP=jnp.asarray(KL, i32), SV=one)  # kfsw: S
        if sv:  # kfsw: S
            # this candidate's survival arrays and precipitation start empty
            # (levels it never reaches must not keep an earlier candidate's
            # values: the expectations sum the whole column)
            for nm in ("E1", "E2", "CW", "UPO", "UPN", "DABE", "PPTLIQ", "PPTICE"):
                A[nm] = jnp.where(go, 0.0, A[nm])  # kfsw: C-LTOPs
            A["UPO"] = _put(A["UPO"], K - 1, VMFLCL, go)  # kfsw: C-LTOPs
            A["UPN"] = _put(A["UPN"], K - 1, VMFLCL, go)  # kfsw: C-LTOPs

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
            # kfsw: S
            tu, theteu, qu, qliq, qice, qnl, qni, ttemp, allfrz = _glaciate(
                ab, P0[NK1], tu, theteu, qu, qliq, qice, qnl, qni, u["TTEMP"], sm)
            iflag = jnp.where(allfrz, 1, u["IFLAG"])  # kfsw: D7g
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
            pos_b, dinc = _abe_increment(dilbe, G, sm.abe_pos_width_m)
            abe = u["ABE"] + jnp.where(pos_b, dinc, 0.0)  # kfsw: E6
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
            eqf = _clip_c1(eqf, 0.0, 1.0, sm.eqf_clip_width)
            ee_p, ud_p = prof5(eqf, lut.sm.prof5_width)
            ee_s = jnp.where(eqf == 1, 1.0, jnp.where(eqf == 0.0, 0.0, ee_p))  # kfsw: D7f
            ud_s = jnp.where(eqf == 1, 0.0, jnp.where(eqf == 0.0, 1.0, ud_p))  # kfsw: D7f
            # kfsw: S
            ee2, ud2, eqfrc = _entrain_select(
                cold, vb, tiny, TV0[NK1] - tvqu, tu95 - TV0[NK1], tvdiff, ee_s, ud_s, eqf,
                sm.entrain_select_width_k, sm.entrain_tiny_width_k)
            let = jnp.where(cold, u["LET"], NK1)  # kfsw: C-LET
            if sm.ee2_floor_width:  # kfsw: S
                ee2 = smooth_max_c1(ee2, _EE2_MIN, sm.ee2_floor_width)  # kfsw: E8
            else:
                ee2 = jnp.maximum(ee2, _EE2_MIN)  # kfsw: E8
            ud2 = _UD2_SCALE * ud2
            uer = 0.5 * rei * (u["EE1"] + ee2)
            udr = 0.5 * rei * (u["UD1"] + ud2)
            stop2 = A["UMF"][NK] - udr < _UMF_DETR_MIN  # kfsw: C-LTOP
            a3 = a2 & ~stop2
            abe = jnp.where(stop2 & pos_b, abe - dinc, abe)  # kfsw: E6,C-LTOP
            let = jnp.where(stop2, NK, let)  # kfsw: C-LET
            upold = A["UMF"][NK] - udr  # kfsw: S
            upnew = upold + uer
            if sv:  # kfsw: S
                # the plume continues past every exit while it has survival;
                # once the survival is exactly 0 (every later weight is 0 with
                # zero derivative) it freezes, so no NaN enters the backward pass.
                # A write past an exit with zero surviving mass is skipped, as the
                # oracle skips it: later candidates and the closure read the stale arrays.
                h1, h2 = _top_hazards(wtw, upold, sm)
                sv0 = u["SV"]
                snew = sv0 * (1.0 - h1) * (1.0 - h2)
                wa = go & (NK >= K) & (sv0 > 0.0)  # kfsw: C-LTOPs
                wa2 = wa & (sv0 * (1.0 - h1) > 0.0)  # kfsw: C-LTOPs
                wa3 = wa & (snew > 0.0)  # kfsw: C-LTOPs
                ups = upk = _cont_floor(upold)
                upns = ups + uer
                dilfrc = upns / ups
            else:
                wa, wa2, wa3, upk = a, a2, a3, upold
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
            B["RATIO2"] = _put(B["RATIO2"], NK1, ratio2, wa)  # kfsw: C-LTOP
            B["TU"] = _put(B["TU"], NK1, tu, wa)  # kfsw: C-LTOP
            B["QLQOUT"] = _put(B["QLQOUT"], NK1, qlqout, wa)  # kfsw: C-LTOP
            B["QICOUT"] = _put(B["QICOUT"], NK1, qicout, wa)  # kfsw: C-LTOP
            B["TVU"] = _put(B["TVU"], NK1, tvu, wa)  # kfsw: C-LTOP
            # THETEU/QU/QLIQ/QICE(NK1): pre-mix values, then mixed if accepted
            B["THETEU"] = _put(B["THETEU"], NK1, _w(wa3, theteu_m, theteu), wa)  # kfsw: C-LTOP
            B["QU"] = _put(B["QU"], NK1, _w(wa3, qu_m, qu), wa)  # kfsw: C-LTOP
            B["QLIQ"] = _put(B["QLIQ"], NK1, _w(wa3, qliq_m, qliq), wa)  # kfsw: C-LTOP
            B["QICE"] = _put(B["QICE"], NK1, _w(wa3, qice_m, qice), wa)  # kfsw: C-LTOP
            # writes between the two exits
            B["WU"] = _put(B["WU"], NK1, _safe_sqrt(wtw), wa2)  # kfsw: C-LTOP,E11
            B["THETEE"] = _put(B["THETEE"], NK1, thetee, wa2)  # kfsw: C-LTOP
            B["TVQU"] = _put(B["TVQU"], NK1, tvqu, wa2)  # kfsw: C-LTOP
            B["EQFRC"] = _put(B["EQFRC"], NK1, eqfrc, wa2)  # kfsw: C-LTOP
            B["UER"] = _put(B["UER"], NK1, _w(wa3, uer_m, uer), wa2)  # kfsw: C-LTOP
            B["UDR"] = _put(B["UDR"], NK1, udr, wa2)  # kfsw: C-LTOP
            # accepted level
            B["UMF"] = _put(B["UMF"], NK1, upnew, wa3)  # kfsw: C-LTOP
            B["DILFRC"] = _put(B["DILFRC"], NK1, dilfrc, wa3)  # kfsw: C-LTOP
            B["DETLQ"] = _put(B["DETLQ"], NK1, detlq, wa3)  # kfsw: C-LTOP
            B["DETIC"] = _put(B["DETIC"], NK1, detic, wa3)  # kfsw: C-LTOP
            B["QDT"] = _put(B["QDT"], NK1, qdt, wa3)  # kfsw: C-LTOP
            B["PPTLIQ"] = _put(B["PPTLIQ"], NK1, pptliq, wa3)  # kfsw: C-LTOP
            B["PPTICE"] = _put(B["PPTICE"], NK1, pptice, wa3)  # kfsw: C-LTOP
            v = dict(u)
            v["WTW"] = _w(wa, wtw, u["WTW"])  # kfsw: C-LTOP
            v["TTEMP"] = _w(wa, ttemp, u["TTEMP"])  # kfsw: C-LTOP
            v["IFLAG"] = _w(wa, iflag, u["IFLAG"])  # kfsw: C-LTOP
            v["REI"] = _w(wa2, rei, u["REI"])  # kfsw: C-LTOP
            v["DILBE"] = _w(wa2, dilbe, u["DILBE"])  # kfsw: C-LTOP
            v["ABE"] = _w(a2, abe, u["ABE"])  # kfsw: C-LTOP
            v["LET"] = _w(a2, let, u["LET"])  # kfsw: C-LET
            v["EE1"] = _w(wa3, ee2, u["EE1"])  # kfsw: C-LTOP
            v["UD1"] = _w(wa3, ud2, u["UD1"])  # kfsw: C-LTOP
            v["UPOLD"] = _w(wa3, upk, u["UPOLD"])  # kfsw: C-LTOP
            v["UPNEW"] = _w(wa3, upnew, u["UPNEW"])  # kfsw: C-LTOP
            v["TRPPT"] = _w(wa3, u["TRPPT"] + pptliq + pptice, u["TRPPT"])  # kfsw: C-LTOP
            if sv:  # kfsw: S
                # survival: stop1 / stop2 exit masses at this level, then the
                # accepted-step weight of the ABE increment (stop2 rolls back)
                B["E1"] = _put(B["E1"], NK, sv0 * h1, wa)  # kfsw: C-LTOPs
                B["E2"] = _put(B["E2"], NK, sv0 * (1.0 - h1) * h2, wa)  # kfsw: C-LTOPs
                B["UPO"] = _put(B["UPO"], NK, upold, wa)  # kfsw: C-LTOPs
                B["UPN"] = _put(B["UPN"], NK, upnew, wa)  # kfsw: C-LTOPs
                if sm.entrain_select_width_k:  # kfsw: S
                    cw = _step_c2(TV0[NK1] - tvqu, sm.entrain_select_width_k)  # kfsw: S
                else:
                    cw = cold.astype(dtype)  # kfsw: C-LETs
                B["CW"] = _put(B["CW"], NK1, cw, wa)  # kfsw: C-LETs
                v["SV"] = _w(wa, snew, sv0)  # kfsw: C-LTOPs
                B["DABE"] = _put(B["DABE"], NK, jnp.where(pos_b, dinc, 0.0), wa)  # kfsw: C-LTOPs
            ex = a & (stop1 | stop2)
            v["LTOP"] = _w(ex, NK, u["LTOP"])  # kfsw: C-LTOP
            v["exited"] = u["exited"] | ex
            return (B, v)

        A, u = _scan_levels(ubody, (A, u), 1, KX - 1)
        LTOP = u["LTOP"]
        if sv:  # kfsw: S
            # cloud top, LET, depth and ABE given convection (the weights carry
            # only members the no-convection test accepts; with none, e_top = 0)
            Wt = _top_weights(A["E1"], A["E2"], u["SV"], A["CW"], kk, K, LCL, KX, KPBL)
            ptop = jnp.sum(Wt, axis=0)  # kfsw: C-LTOPs
            pc = jnp.sum(ptop)  # kfsw: C-LTOPs
            pcd = _sd(pc, pc > 0.0)  # kfsw: C-LTOPs
            e_top = jnp.sum(ptop * kk) / pcd  # kfsw: C-LTOPs
            e_let = jnp.sum(jnp.sum(Wt, axis=1) * kk) / pcd  # kfsw: C-LETs
            cldhgt = jnp.sum(ptop * Z0) / pcd - ZLCL  # kfsw: C-LTOPs
        else:
            e_top, e_let = LTOP, u["LET"]
            cldhgt = Z0[LTOP] - ZLCL  # kfsw: C-LTOP
        if sm.chmin_clip_width_k:  # kfsw: S
            CHMIN = _CHMIN_SHALLOW_M + _CHMIN_SLOPE_M_PER_K * (_clip_c1(
                TLCL, _CHMIN_TCOLD_K, _CHMIN_TWARM_K, sm.chmin_clip_width_k) - _CHMIN_TCOLD_K)
        else:
            CHMIN = jnp.where(TLCL > _CHMIN_TWARM_K, _CHMIN_DEEP_M,
                              jnp.where(TLCL >= _CHMIN_TCOLD_K,
                                        _CHMIN_SHALLOW_M
                                        + _CHMIN_SLOPE_M_PER_K * (TLCL - _CHMIN_TCOLD_K),
                                        _CHMIN_SHALLOW_M))  # kfsw: E9
        noconv = (e_top <= KLCL) | (e_top <= KPBL) | (e_let + 1 <= KPBL)  # kfsw: A4
        abe_out = _conv_abe(A["DABE"], ptop) / pcd if sv else u["ABE"]  # kfsw: S
        deep = ~noconv & (cldhgt > CHMIN) & (abe_out > _ABE_MIN)  # kfsw: B1
        shallow = ~noconv & ~deep
        exit_sh = shallow & (NU == sc_nuchm)  # kfsw: B2
        zero_cols = noconv | (shallow & ~exit_sh)
        A["CLDHGT"] = _put(A["CLDHGT"], LC, jnp.where(noconv, 0.0, cldhgt), go)  # kfsw: A4,C-KCHECK
        # kfsw: B2,C-LTOP
        zr = go & zero_cols & (kk >= K) & ((kk <= LTOP) | sv)
        for nm in ("UMF", "UDR", "UER", "DETLQ", "DETIC", "PPTLIQ", "PPTICE"):
            A[nm] = jnp.where(zr, 0.0, A[nm])  # kfsw: B2,C-LTOP
        out = dict(
            LC=LC, KPBL=KPBL, DPTHMX=DPTHMX, TMIX=TMIX, QMIX=QMIX, ZMIX=ZMIX, PMIX=PMIX,
            TLCL=TLCL, TVLCL=TVLCL, ZLCL=ZLCL, KLCL=KLCL, K=K, DLP=DLP, TENV=TENV,
            QENV=QENV, TVEN=TVEN, WKL=WKL, WKLCL=WKLCL, DTLCL=DTLCL, DTRH=DTRH, WLCL=WLCL,
            PLCL=PLCL, LCL=LCL, LET=u["LET"], RAD=RAD, AU0=AU0, VMFLCL=VMFLCL,
            UPOLD=u["UPOLD"], UPNEW=u["UPNEW"], ABE=abe_out,
            TRPPT=u["TRPPT"], LTOP=LTOP, CHMIN=CHMIN, IFLAG=u["IFLAG"], SEX=u["SV"],
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


def _upstream_step(X, OMG, FXM, inc, LTOP, kk, src, dtime, EMSD, w):
    """One upstream/forward step of the KF vertical advection (l.1990-2018,
    l.2336-2375): THFXIN/THFXOUT built exactly as the oracle accumulates them.
    ``w`` > 0 blends the two upwind sides of a face over |OMG| < w/2 with the
    same weight on both levels it couples (conservative)."""
    le = _step_c(-OMG, w) if w else OMG <= 0.0  # kfsw: E10
    up = jnp.concatenate([X[1:], X[-1:]])  # X(k+1)
    dn = jnp.concatenate([X[:1], X[:-1]])  # X(k-1)
    fxm_up = jnp.concatenate([FXM[1:], FXM[-1:]])
    le_up = jnp.concatenate([le[1:], le[-1:]])
    a_rng = inc & (kk >= 2)                       # levels processed as NK; kfsw: C-LTOP
    b_rng = inc & (kk + 1 <= LTOP)                # levels touched by NK+1; kfsw: C-LTOP
    if w:  # kfsw: S
        la, lb = le * a_rng, le_up * b_rng  # face weights inside the ranges
        fin = la * (-FXM * dn) + (b_rng - lb) * fxm_up * up
        fout = (a_rng - la) * FXM * X + lb * (-fxm_up * X)
    else:
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
    sm = lut.sm
    sv = _survival_on(sm)
    if sv:  # kfsw: S
        # expectation over the joint survival weights of (LET, LTOP) of the
        # oracle's cloud-top detrainment applied to the continuing plume
        Wn = _top_weights(S["E1"], S["E2"], sc["SEX"], S["CW"], kk, K, LCL, KX, KPBL)
        Ws = _top_weights(S["E1"], S["E2"], sc["SEX"], S["CW"], kk, K, LCL, KX, KPBL,
                          let_fixed=jnp.maximum(KPBL, KLCL))  # kfsw: C-LETs
        Wtop = jnp.where(sh, Ws, Wn)  # kfsw: B1
        PTOP = jnp.sum(Wtop, axis=0)  # kfsw: C-LTOPs
        PGE = jnp.cumsum(PTOP[::-1])[::-1]  # kfsw: C-LTOPs
        PCD = _sd(PGE[0], PGE[0] > 0.0)  # P(convection), guarded  # kfsw: C-LTOPs
        raw = (UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, QLIQ, QICE, DILFRC,
               QLQOUT, QICOUT, S["UPO"], S["UPN"], DP, kk)
        # a member's mass flux at KLCL (its LET block lies above)
        umf_klcl = UMF[KLCL]  # kfsw: B3
        UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE = _expected_top_profile(Wtop, *raw)
        TRPPT = jnp.sum(PPTLIQ + PPTICE)  # kfsw: C-LTOPs
    else:
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
    warm = valid & (T0 > _T00_K)  # kfsw: C-ML
    ML = jnp.max(jnp.where(warm & (kk <= LTOP), kk, 0))  # kfsw: C-ML
    if sv:  # kfsw: S
        # warm flags as probabilities, independent per level (exact one-hot
        # at width 0); NW[k, t] = P(a warm level in [k, t]), 0 for t < k
        if sm.melt_t_width_k:  # kfsw: S
            v = jnp.where(valid, _step_c2(T0 - _T00_K, sm.melt_t_width_k), 0.0)  # kfsw: C-MLs
        else:
            v = warm.astype(dtype)  # kfsw: C-MLs
        NW = _any_in_range(v, kk)
        # P(melting | top t): a warm level in (LC, t] (the oracle's LC < ML)
        mel_p = _any_in_range(jnp.where(kk > LC, v, 0.0), kk)[0]  # kfsw: C-MLs,C-KCHECK

    # the source feeds only the convecting members
    vmf_src = VMFLCL * jnp.sum(PTOP) if sv else VMFLCL  # kfsw: S

    def below_body(c, NK):
        umf, uer = c
        a = NK <= K  # kfsw: C-KLCL,C-KPBL
        isLC = NK == LC  # kfsw: C-KLCL,C-KPBL,C-KCHECK
        uer_v = vmf_src * DP[NK] / DPTHMX  # kfsw: S
        # kfsw: C-KLCL,C-KPBL,C-KCHECK
        umf_v = jnp.where(isLC, uer_v, jnp.where(NK <= KPBL, umf[NK - 1] + uer_v, vmf_src))
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
    if sv:  # kfsw: S
        # the expected fluxes vanish above every possible top; the plume
        # states stay (they only multiply those fluxes); domain = column
        inc, LTOPD = valid, KX
    else:
        above = valid & (kk >= LTOP + 1)  # kfsw: C-LTOP
        above2 = valid & (kk > LTOP + 1)  # kfsw: C-LTOP
        # kfsw: C-LTOP
        (UMF, UDR, UER, QDT, QLIQ, QICE, QLQOUT, QICOUT, DETLQ, DETIC, PPTLIQ, PPTICE) = (
            jnp.where(above, 0.0, a) for a in
            (UMF, UDR, UER, QDT, QLIQ, QICE, QLQOUT, QICOUT, DETLQ, DETIC, PPTLIQ, PPTICE))
        TU, QU, WU = (jnp.where(above2, 0.0, a) for a in (TU, QU, WU))  # kfsw: C-LTOP
        inc, LTOPD = valid & (kk <= LTOP), LTOP  # kfsw: C-LTOP
    EMS = jnp.where(inc, DP * DXSQ / G, 0.0)  # kfsw: C-LTOP
    EMSD = jnp.where(inc, 1.0 / _sd(EMS, inc), 0.0)  # kfsw: C-LTOP
    THTAU = jnp.where(inc, TU * _exner(P0, QDT), 0.0)  # kfsw: C-LTOP
    THTA0 = jnp.where(inc, T0 * _exner(P0, Q0), 0.0)  # kfsw: C-LTOP
    DDILFRC = jnp.where(inc, 1.0 / _sd(DILFRC, inc), 1.0)  # kfsw: C-LTOP

    # --- convective time scale and precipitation efficiency (l.1588-1632) ---
    def wspd(k):
        return _safe_sqrt(U0[k] * U0[k] + V0[k] * V0[k])  # kfsw: E11

    if sv:  # kfsw: S
        # top scalars given convection (members with a top above KLCL)
        pc = _sd(jnp.sum(PTOP), jnp.sum(PTOP) > 0.0)  # kfsw: C-LTOPs
        WS_KLCL, WS_LTOP = wspd(KLCL), jnp.sum(PTOP * wspd(kk)) / pc  # kfsw: C-LTOPs
        U_T, V_T, Z_T = (jnp.sum(PTOP * x) / pc for x in (U0, V0, Z0))  # kfsw: C-LTOPs
    else:
        WS_KLCL, WS_LTOP = wspd(KLCL), wspd(LTOP)
        U_T, V_T, Z_T = U0[LTOP], V0[LTOP], Z0[LTOP]  # kfsw: C-LTOP
    if lut.sm.l5_gather_width_pa:  # kfsw: S
        wl5 = _l5_weights(P0, kk, valid, lut.sm.l5_gather_width_pa)
        WS_L5 = jnp.sum(wl5 * wspd(kk))  # kfsw: C-L5g
    else:
        WS_L5 = wspd(L5)
    VCONV = 0.5 * (WS_KLCL + WS_L5)
    vok = VCONV > 0.0  # kfsw: E12
    TIMEC = jnp.where(vok, dx / _sd(VCONV, vok), jnp.inf)  # kfsw: E12
    TADVEC = TIMEC
    if sm.timec_clip_width_s:  # kfsw: S
        # VCONV = 0 (TIMEC = inf) takes a finite stand-in past the clip band
        big = prm["timec_max"] + sm.timec_clip_width_s
        TIMEC = _clip_c1(jnp.where(vok, TIMEC, big), prm["timec_min"], prm["timec_max"],  # kfsw: E12
                         sm.timec_clip_width_s)
    else:
        TIMEC = jnp.maximum(prm["timec_min"], TIMEC)  # kfsw: E12
        TIMEC = jnp.minimum(prm["timec_max"], TIMEC)  # kfsw: E12
    TIMEC = jnp.where(sh, _TIMEC_SHALLOW_S, TIMEC)  # kfsw: B1
    NIC = _nint(TIMEC / dt)  # kfsw: C-NIC
    TIMEC = NIC.astype(dtype) * dt  # kfsw: C-NIC
    SHSIGN = jnp.where(WS_LTOP > WS_KLCL, 1.0, -1.0)  # kfsw: D5
    du = U_T - U0[KLCL]  # kfsw: C-KLCL
    dv = V_T - V0[KLCL]  # kfsw: C-KLCL
    VWS = du * du + dv * dv
    dzc = Z_T - Z0[LCL]  # kfsw: C-KLCL
    VWS = _VWS_SCALE * SHSIGN * _safe_sqrt(VWS) / _sd(dzc, dzc != 0.0)  # kfsw: E11,C-LTOP
    PEF = _PEF_C0 + VWS * (_PEF_C1 + VWS * (_PEF_C2 - VWS * _PEF_C3))
    PEF = _clip_c1(PEF, _PEF_MIN, _PEF_MAX, sm.pef_clip_width)
    CBH = (sc["ZLCL"] - Z0[1]) * _CBH_FT_PER_M
    RCBH = jnp.where(CBH < _CBH_LOW, _RCBH_LOW,
                     _RCBH_C0 + CBH * (_RCBH_C1 + CBH * (_RCBH_C2 + CBH * (
                         _RCBH_C3 + CBH * (_RCBH_C4 - CBH * _RCBH_C5)))))  # kfsw: D6
    RCBH = jnp.where(CBH > _CBH_HIGH, _RCBH_HIGH, RCBH)  # kfsw: D6
    PEFCBH = _cap_c1(1.0 / (1.0 + RCBH), _PEFCBH_MAX, sm.pefcbh_cap_width)
    PEFF = 0.5 * (PEF + PEFCBH)

    # --- downdraft (l.1647-1784) ---
    zf = jnp.zeros(NP, dtype)
    KSTART = KPBL + 1
    cand = valid & (kk >= KSTART + 1) & (P0[KSTART] - P0 > _LFS_DEPTH_PA)  # kfsw: C-LFS
    kl0 = jnp.where(jnp.any(cand), jnp.argmax(cand).astype(i32), KX)  # kfsw: C-LFS
    KLFS = jnp.minimum(kl0, LET - 1)  # kfsw: C-LFS
    if sv:  # kfsw: S
        # first-crossing weights pi of kl0 (the remainder goes to KX, the
        # oracle's no-candidate value); exact one-hot at width 0
        if sm.lfs_depth_width_pa:  # kfsw: S
            # kfsw: C-LFSs
            s_c = jnp.where(valid & (kk >= KSTART + 1),
                            _step_c2(P0[KSTART] - P0 - _LFS_DEPTH_PA, sm.lfs_depth_width_pa), 0.0)
        else:
            s_c = cand.astype(dtype)  # kfsw: C-LFSs
        QL = _lfs_member_weights(_first_crossing(s_c, kk, KX), kk)
    LFS = jnp.where(sh, jnp.asarray(1, i32), KLFS)  # kfsw: B1

    def mlt_body(c, k):
        return _w((k >= KLCL) & (k <= LTOP), c + PPTICE[k], c)  # kfsw: C-KLCL,C-LTOP

    def _downdraft_mixture(LFS, LDT, HZ, TZ, QSD, QDK, DMF, DER, DMFFRC, p_ok, TRPPT,
                           UMF_KLCL):
        """Expectation over the base b = LDB (survival weights of the descent)
        of the oracle's dd3 and acceptance (l.1757-1873), in closed form: with
        C[k] = sum DP over [k, LDT] the accepted run b has DDR = -DMFK DP / C[b]
        and DMF = DMFK (1 - C / C[b]) on [b, LDT], scaled by its DDINC_b; DMF
        and DER from the source layer up keep the descent's entrainment
        values times the total accepted DDINC."""
        lev = (kk >= 1) & (kk <= LDT)  # kfsw: C-LDBs
        HZ = jnp.where(kk == 1, 1.0, HZ)  # the oracle stops at level 1  # kfsw: C-LDBs
        PB = _descent_base_weights(HZ, lev)
        C = jnp.cumsum(jnp.where(lev, DP, 0.0)[::-1])[::-1]  # kfsw: C-LDBs
        Cs = _sd(C, lev & (C > 0.0))  # kfsw: C-LDBs
        DMFK = DMF[KSTART]  # kfsw: C-KPBL
        G_ = jnp.cumsum(jnp.where(lev, (QSD - QDK) * DP, 0.0)[::-1])[::-1]  # kfsw: C-LDBs
        TDERB = -DMFK * G_ / Cs  # raw TDER of the run with base b
        # deep_dd shares dd_ok's band (vote 24): its weight >= p_ok's at every base
        deep = _gt_c2(P0 - P0[LFS] - _DD_MIN_DEPTH_PA, sm.dd_ok_depth_width_pa)  # kfsw: B3s,C-LFSs
        if sm.nodd_tder_width_kgs:  # kfsw: S
            acc = _step_c2(TDERB - _TDER_MIN, sm.nodd_tder_width_kgs)
        else:
            acc = (TDERB >= _TDER_MIN).astype(dtype)  # kfsw: B3s
        ddinc = -DMFFRC * UMF_KLCL / _sd(DMFK, DMFK != 0.0)  # kfsw: B3s
        if sm.ddinc_cap_width:  # kfsw: S
            tp = _sd(TRPPT, TRPPT > 0.0)  # kfsw: B3s
            TDERF = TRPPT * _cap_c1(TDERB * ddinc / tp, 1.0, sm.ddinc_cap_width)
        else:
            TDERF = jnp.minimum(TDERB * ddinc, TRPPT)  # kfsw: B3s
        ddincb = TDERF / _sd(TDERB, acc > 0.0)  # kfsw: B3s
        wacc = PB * p_ok * deep * acc
        cb = wacc * ddincb
        S0, S1 = jnp.cumsum(cb), jnp.cumsum(cb / Cs)  # kfsw: C-LDBs
        tot = jnp.sum(cb)  # kfsw: B3s
        up = (kk >= KSTART) & (kk <= LFS)  # kfsw: C-KPBL,C-LFSs
        DDR = jnp.where(lev, -DMFK * DP * S1, 0.0)  # kfsw: C-LDBs
        DMFo = jnp.where(lev, DMFK * (S0 - C * S1), jnp.where(up, DMF * tot, 0.0))  # kfsw: C-LDBs
        DERo = jnp.where(up, DER * tot, 0.0)  # kfsw: C-KPBL,C-LFSs
        THTAD = jnp.where(lev, TZ * _exner(P0, QSD), 0.0)  # kfsw: C-LDBs
        QDo = jnp.where(lev, QSD, 0.0)  # kfsw: C-LDBs
        TDER = jnp.sum(wacc * TDERF)  # kfsw: B3s
        LDB = jnp.argmax(PB).astype(i32)  # kfsw: C-LDBs (diagnostic)
        return (DMFo, DERo, DDR, THTAD, TZ, QDo, TDER, TRPPT - TDER, LDB)

    def downdraft(LFS, mel, PPTMLT, TRPPT, UMF_KLCL, soft=False):
        """The oracle's downdraft and its acceptance for one source level LFS,
        melting flag/ice precipitation, updraft precipitation and base flux.
        ``soft`` (static, PR 6b): the expectation over the descent's base LDB
        and the acceptance weights (dd_ok, deep_dd, nodd; with the DDINC cap and
        the D7h low blend inside each run) of the accepted
        downdraft and the no-downdraft outcome; DMF/DER/DDR and TDER are those
        expectations, THTAD/QD the base-independent states below the source
        layer (the closure reads only DDR*THTAD and DDR*QD), LDB the base of
        largest weight (diagnostic)."""
        if soft:  # kfsw: S
            dep = P0[KSTART] - P0[LFS] - _DD_MIN_DEPTH_PA  # kfsw: C-KPBL,C-LFSs
            p_ok = _gt_c2(dep, sm.dd_ok_depth_width_pa)
            p_ok = jnp.where(sh, 0.0, p_ok)  # kfsw: B1
            dd_ok = jnp.asarray(True)  # the descent runs; p_ok weights its outcome
        else:
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

        # the soft run always descends (p_ok weights it): guard its divisor
        # where the column has no updraft mass at KLCL (weight 0 there)
        mok = mel & (UMF_KLCL != 0.0) if soft else mel & dd_ok  # kfsw: A0
        DTMELT = jnp.where(mel, _RLF * PPTMLT / (CP * _sd(UMF_KLCL, mok)), 0.0)  # kfsw: B5
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
            if soft:  # kfsw: S
                a = ND <= LDT  # every level of the descent  # kfsw: C-LDBs
            else:
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
            if sm.dd_rh_low_width_kgkg:  # kfsw: S
                wlo = _gt_c2(qd_n - qsrh, sm.dd_rh_low_width_kgkg)
                t1rh = t1rh + wlo * (t_n + (qss - qd_n) * rl / CP - t1rh)
                qsrh = qsrh + wlo * (qd_n - qsrh)
            else:
                low = qsrh < qd_n  # kfsw: D7h
                qsrh = jnp.where(low, qd_n, qsrh)  # kfsw: D7h
                t1rh = jnp.where(low, t_n + (qss - qsrh) * rl / CP, t1rh)  # kfsw: D7h
            adj = rhh < 1.0  # kfsw: D7h
            t_n = jnp.where(adj, t1rh, t_n)  # kfsw: D7h
            qsd_n = jnp.where(adj, qsrh, qss)  # kfsw: D7h
            tvd = t_n * (1.0 + _TV_COEF * qsd_n)
            if soft:  # kfsw: S
                # the stop hazard weights the descent's base
                hz_n = _gt_c2(tvd - TV0[ND], sm.ldb_tv_width_k)  # kfsw: C-LDBs
                return (_put(tz, ND, t_n, a), _put(qsd, ND, qsd_n, a), _put(thd, ND, thd_n, a),
                        _put(qd, ND, qd_n, a), dpdd, ldb, _put(ex, ND, hz_n, a))  # kfsw: C-LDBs
            stop = (tvd > TV0[ND]) | (ND == 1)  # kfsw: B3,C-LDB
            # kfsw: B3,C-LDB
            return (_put(tz, ND, t_n, a), _put(qsd, ND, qsd_n, a), _put(thd, ND, thd_n, a),
                    _put(qd, ND, qd_n, a), _w(a, dpdd_n, dpdd), _w(a & stop, ND, ldb), ex | (a & stop))

        ex0 = zf if soft else jnp.asarray(False)  # kfsw: S
        TZ, QSD, THETED, QD, DPDD, LDB, HZ = _scan_levels(
            dd2_body, (TZ, QSD, THETED, QD, zero, LFS, ex0), 1, KX, reverse=True)
        if soft:  # kfsw: S
            # kfsw: C-LDBs,C-KPBL
            return _downdraft_mixture(LFS, LDT, HZ, TZ, QSD, QD[KSTART], DMF, DER, DMFFRC,
                                      p_ok, TRPPT, UMF_KLCL)
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
        ddinc = -DMFFRC * UMF_KLCL / _sd(DMF[KSTART], ~nodd)  # kfsw: B3
        big = TDER * ddinc > TRPPT  # kfsw: B3
        ddinc = jnp.where(big, TRPPT / _sd(TDER, ~nodd), ddinc)  # kfsw: B3
        TDER = jnp.where(nodd, 0.0, TDER * ddinc)  # kfsw: B3
        PPTFLX = jnp.where(nodd, TRPPT, TRPPT - TDER)  # kfsw: B3
        LDB = jnp.where(nodd, LFS, LDB)  # kfsw: B3,C-LDB
        scl = ~nodd & (kk >= LDB) & (kk <= LFS)  # kfsw: B3,C-LDB
        DMF, DER, DDR = (jnp.where(scl, a * ddinc, a) for a in (DMF, DER, DDR))  # kfsw: B3,C-LDB
        # the downdraft arrays are zero above LFS < LET <= LTOP: the oracle's
        # NK <= LTOP bound on this zeroing is the whole column
        z_nodd = nodd & valid
        z_dd = ~nodd & valid & ((kk <= LDB - 1) | (kk >= LFS + 1))  # kfsw: B3,C-LDB
        z_mid = ~nodd & (kk >= LDT + 1) & (kk <= LFS - 1)  # kfsw: B3,C-LDB
        DMF, DER, DDR = (jnp.where(z_nodd | z_dd, 0.0, a) for a in (DMF, DER, DDR))  # kfsw: B3,C-LDB
        # kfsw: B3,C-LDB
        THTAD, TZ, QD = (jnp.where(z_nodd | z_dd | z_mid, 0.0, a) for a in (THTAD, TZ, QD))

        return (DMF, DER, DDR, THTAD, TZ, QD, TDER, PPTFLX, LDB)

    if sv:  # kfsw: S
        # mixture over the downdraft source levels l the (LET, top) members
        # give (LFS = min(KLFS, LET - 1)); each run gets its members'
        # conditional melting ice and precipitation; fluxes average, the
        # states the closure multiplies by DDR average DDR-weighted
        # (a shallow column has no downdraft)
        def member(l):
            Wl = QL[l][:, None] * Wtop  # kfsw: C-LETs,C-LFSs
            ql = jnp.sum(Wl)  # kfsw: C-LETs
            # conditional means: the offset keeps the reverse pass of the
            # division finite for tiny shares; it biases a mean by
            # eps / (q_l + eps), so only shares near eps lose weight in it
            qd = ql + _QSHARE_FLOOR
            prof = _expected_top_profile(mel_p[None, :] * Wl, *raw)  # kfsw: C-LTOPs,C-MLs
            mlt = _scan_levels(lambda c, k: _w(k >= KLCL, c + prof[6][k], c),  # kfsw: C-LTOPs
                               zero, 1, KX)
            plq = _expected_top_profile(Wl, *raw)[5:7]
            trp = jnp.sum(plq[0] + plq[1])  # kfsw: C-LTOPs
            # a level no member selects still runs (fixed cost): run it at the
            # column's own LFS so its (zero-weight) values and gradients stay finite
            lr = jnp.where(ql > 0.0, l, LFS)  # kfsw: C-LETs
            out = downdraft(lr, True, mlt / qd, trp / qd, umf_klcl, soft=_soft_downdraft_on(sm))
            return (ql,) + out

        ql, DMFs, DERs, DDRs, THTADs, _, QDs, TDERs, _, LDBs = jax.vmap(member)(kk)  # kfsw: C-LETs
        mix = lambda x: jnp.tensordot(ql, x, axes=1)  # kfsw: C-LETs  # noqa: E731
        DMF, DER, DDR, TDER = mix(DMFs), mix(DERs), mix(DDRs), mix(TDERs)  # kfsw: C-LETs
        # precipitation flux keeps the exact moments: E[TRPPT] - E[TDER]
        PPTFLX = TRPPT - TDER
        # the closure multiplies the downdraft states only by DDR: it reads the
        # mixture's DDR*THTAD and DDR*QD moments directly (no division)
        DDRTH, DDRQ = mix(DDRs * THTADs), mix(DDRs * QDs)  # kfsw: C-LETs
        THTAD = QD = TZ = None
        LDB = LDBs[LFS]  # kfsw: C-LETs (diagnostic: the run at the hard-LET LFS)
    else:
        PPTMLT = _scan_levels(mlt_body, zero, 1, KX)
        mel = LC < ML  # kfsw: B5,C-KCHECK
        DMF, DER, DDR, THTAD, TZ, QD, TDER, PPTFLX, LDB = downdraft(  # kfsw: B5
            LFS, mel, PPTMLT, TRPPT, UMF[KLCL])
    CPR = TRPPT

    # --- inflow limit and closure (l.1879-2281) ---
    LMAX = jnp.maximum(KLCL, LFS)  # kfsw: C-KLCL,C-LFS
    udd = UER - DER
    if sv:  # kfsw: S
        # a level above KLCL limits the inflow with the share of the members
        # whose downdraft starts at or above it (no member: it drops out)
        rng = valid & (kk >= LC)  # kfsw: C-LETs
    else:
        rng = valid & (kk >= LC) & (kk <= LMAX)  # kfsw: C-KCHECK,C-KLCL,C-LFS
    if sm.uerder_onset_width_kgs:  # kfsw: S
        elig = rng * _step_c2(udd - _UERDER_MIN, sm.uerder_onset_width_kgs)
        okm = elig > 0.0  # kfsw: E15
    else:
        okm = rng & (udd > _UERDER_MIN)  # kfsw: E15
        elig = okm
    AINCM1 = EMS / (_sd(udd, okm) * TIMEC)  # kfsw: E15
    if sv:  # kfsw: S
        wl = jnp.where(kk <= KLCL, 1.0, jnp.cumsum(ql[::-1])[::-1] / PCD)  # kfsw: C-LETs
        AINCM1 = _survival_inflow_limit(AINCM1, wl, _AINCMX0 * math.exp(sm.ainc_min_width))
    if sm.ainc_min_width or sm.uerder_onset_width_kgs:  # kfsw: S
        AINCMX, AINC = _inflow_limit_c1(AINCM1, elig, sm.ainc_min_width)
    else:
        AINCMX = jnp.minimum(_AINCMX0, jnp.min(jnp.where(okm, AINCM1, _AINCMX0)))  # kfsw: E15
        AINC = jnp.where(AINCMX < 1.0, AINCMX, 1.0)  # kfsw: E15
    TDER2, PPTFL2 = TDER, PPTFLX
    base = dict(UMF=UMF, DMF=DMF, DETLQ=DETLQ, DETIC=DETIC, UDR=UDR, UER=UER, DER=DER, DDR=DDR)
    if sv:  # kfsw: S
        base.update(DDRTH=DDRTH, DDRQ=DDRQ)  # kfsw: C-LETs
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
            return jnp.where(s < nstep, _upstream_step(c, OMG, FXM, inc, LTOPD, kk, src, dtime, EMSD,
                                                       sm.omg_upwind_width_pas), c), None
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
            b = (NK >= 2) & (NK <= LTOPD)  # kfsw: C-LTOP
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
        if sv:  # kfsw: S
            dd_th, dd_q = F["DDRTH"], F["DDRQ"]
        else:
            dd_th, dd_q = F["DDR"] * THTAD, F["DDR"] * QD
        th_src = lambda fin, fout: fin + F["UDR"] * THTAU + dd_th - fout - (F["UER"] - F["DER"]) * THTA0  # noqa: E731
        q_src = lambda fin, fout: fin + F["UDR"] * QDT + dd_q - fout - (F["UER"] - F["DER"]) * Q0  # noqa: E731
        THPA = advect(jnp.where(inc, THTA0, 0.0), th_src, OMG, FXM, NSTEP, DTIME)  # kfsw: C-LTOP
        QPA = advect(jnp.where(inc, Q0, 0.0), q_src, OMG, FXM, NSTEP, DTIME)  # kfsw: C-LTOP
        THTAG = THPA
        QG = jnp.where(inc, QPA, Q0)  # kfsw: C-LTOP
        KLCL_c = c["KLCL"]

        # borrow moisture where QG < 0 (l.2028-2059)
        def borrow(qg, b, NK, nk1):
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
            return _put(qg, NK - 1, tmb * EMSD[NK - 1], b)  # kfsw: D7i

        def neg_body(cc, NK):
            qg, fatal = cc
            b = (NK <= LTOPD) & (qg[NK] < 0.0)  # kfsw: D7i
            fatal = fatal | (b & (NK == 1))  # kfsw: D7i,A6
            if sv:  # kfsw: S
                # the donor above is the next level while the cloud continues,
                # the cloud base when the top is here (probability-weighted;
                # fades to the base as the tops run out)
                pup = PGE[NK + 1] / (PGE[NK] + _QSHARE_FLOOR * PCD)  # kfsw: C-LTOPs
                q_up = borrow(qg, b, NK, NK + 1)
                q_wr = borrow(qg, b, NK, KLCL_c)
                qg = q_wr + pup * (q_up - q_wr)
            else:
                nk1 = jnp.where(NK == LTOP, KLCL_c, NK + 1)  # kfsw: D7i,C-LTOP
                qg = borrow(qg, b, NK, nk1)
            return (qg, fatal)

        QG, fatal = _scan_levels(neg_body, (QG, c["fatal"]), 1, KX)
        # kfsw: A6,C-LTOP
        TOPOMG = (F["UDR"][LTOPD] - F["UER"][LTOPD]) * DP[LTOPD] * EMSD[LTOPD]
        istop = jnp.abs(TOPOMG - OMG[LTOPD]) > _TOPOMG_TOL  # kfsw: A6
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
        rl = XLV0 - XLV1 * tmix
        cpm = CP * (1.0 + _CPM_Q * qmix)
        dssdt = qss * (CLIQ - BLIQ * DLIQ) / ((tmix - DLIQ) * (tmix - DLIQ))
        dq = (qmix - qss) / (1.0 + rl * dssdt / cpm)
        tmix_s = tmix + rl / CP * dq
        qmix_s = qmix - dq
        qmix_u = _floor0_c1(qmix, sm.qmix_floor_width_kgkg)
        emix = qmix_u * PMIX / (_EPS_KF + qmix_u)
        tlog = _tlog_lookup(lut, emix / ALIQ)
        tdpt = (CLIQ - DLIQ * tlog) / (BLIQ - tlog)
        tlcl_u = _cap_c1(_tsat_from_dewpoint(tdpt, tmix), tmix, lut.sm.tlcl_cap_width_k)
        tmix, qmix, tlcl = _sup_select(qmix, qss, (tmix_s, qmix_s, tmix_s), (tmix, qmix_u, tlcl_u),
                                       sm.closure_sup_width_kgkg)
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
            if sv:  # kfsw: S
                # the whole column; each increment counts for the convecting
                # members whose top lies above NK
                b = NK >= kq  # kfsw: C-LTOPs
            else:
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
            pos_b, dinc = _abe_increment(dilbe, G, lut.sm.abe_pos_width_m)
            if sv:  # kfsw: S
                dinc = dinc * PGE[NK1] / PCD  # kfsw: C-LTOPs
            abeg = jnp.where(b & pos_b, abeg + dinc, abeg)  # kfsw: E6
            thteeg = _envirtht(lut, ab, P0[NK1], TG[NK1], QG[NK1])  # kfsw: S
            th_new = thn * DDILFRC[NK1] + thteeg * (1.0 - DDILFRC[NK1])  # kfsw: S
            return (_put(th, NK1, th_new, b), _put(tvq, NK1, tvqu, b), abeg)  # kfsw: C-KLCL2

        theteu, _, ABEG = _scan_levels(abeg_body, (theteu, jnp.zeros(NP, dtype), zero), 1, KX - 1)
        # decision logic (l.2179-2280)
        stop_noitr = c["NOITR"]
        DABE = smooth_max_c1(ABE - ABEG, _DABE_FRAC * ABE, sm.dabe_floor_width_jkg)  # kfsw: E18
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
        ainc_new = _cap_c1(ainc_new, AINCMX, sm.ainc_cap_width)
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
    FRC2 = _feedback_ratio(PPTFLX, CPR, AINC)
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
        if sv:  # kfsw: S
            # P(k <= ML_t | convection), ML_t the highest warm level <= top t
            pml = jnp.sum(NW * PTOP[None, :], axis=1) / PCD  # kfsw: C-LTOPs,C-MLs
            pml = pml.at[0].set(jnp.sum(PTOP) / PCD)  # pad level: k <= ML_t always  # kfsw: S
            TGf = (pml * (TG - (QIG + QSG) * _RLF / CPM)
                   + (1.0 - pml) * (TG + (QLG + QRG) * _RLF / CPM))
        else:
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
