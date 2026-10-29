"""WRF v4.8.0 MYNN-EDMF, level-2.5 closure: JAX port of the column core.

Oracle: WRF ``v4.8.0`` (06d4240ae989cc3e50af412bb472df3d9048783c), submodule
``phys/MYNN-EDMF`` 8f8e8b59021249949c04d1686edc5a02ca325627, file
``module_bl_mynnedmf.F90`` (sha256 4aee355aefaaa06c...25a83), subroutine
``mynnedmf`` and everything it calls on the ported path, with the WRF namelist
defaults for every option except the closure level (Registry.EM_COMMON):

  bl_mynn_closure = 2.5, bl_mynn_cloudpdf = 2 (Chaboureau-Bechtold),
  bl_mynn_mixlength = 1, bl_mynn_edmf = 1 (implicit mass flux),
  bl_mynn_edmf_mom = 1, bl_mynn_edmf_tke = 0, bl_mynn_edmf_dd = 0,
  bl_mynn_ess = 1, bl_mynn_cloudmix = 1, bl_mynn_mixqt = 0,
  bl_mynn_tkeadvect = .false., tke_budget = 0, bl_mynn_mixscalars = 0,
  bl_mynn_mixnumcon = 0, spp_pbl = 0, no chemistry, FLAG_QC = FLAG_QI =
  .true., FLAG_QS = FLAG_QNC = FLAG_QNI = FLAG_QNWFA/QNIFA/QNBCA =
  FLAG_OZONE = .false. (snow enters as zero).

:func:`mynnedmf_column` is batched over columns; every level array is
BOTTOM-UP (index 0 is the lowest mass level, the oracle's ``kts``), so the
port follows the Fortran index by index: Python index ``i`` is Fortran
``k = i + 1``, and interface arrays (``zw``, mass fluxes) hold Fortran index
``k`` at Python position ``k - 1``.  Every hard switch of the oracle is a
``jnp.where``; where a branch that is evaluated but not selected would divide
by zero or take a log/root of an invalid argument, it is evaluated on a safe
argument (``_sel``), which leaves every selected value unchanged.  The oracle
has no smoothing knobs on this path.

**NOT RULE 5 COMPLIANT** (user decision 2026-10-28): every switch on traced
state, including the PBL-height first crossing, is hard at the default; there
is no recorded default gap and no gradient test through a switch.  This is an
unwired, opt-in kernel; it must be smoothed before any lane uses it (#2231).

Constants: the scheme's own (module_bl_mynnedmf_common.F90) are module-level
constants below; the host physical constants that module USEs from WRF's
``module_model_constants`` are legoESM's (``legoesm.constants``), which is how
the oracle fixture was generated.

Not ported because they cannot change any output on this path: tke-budget
diagnostics, the downdraft scheme (bl_mynn_edmf_dd = 0), environmental
subsidence/detrainment inside the mass-flux scheme (``env_subs = .false.``,
so its ``envm_*``/``det_*``/``sub_*`` terms are zero), mass-flux transport of
number concentrations, aerosols, chemistry and scalars, the stochastic
perturbations (spp_pbl = 0) and debug prints.  The WRF driver's mixing-ratio
conversion and surface-flux clamps are host glue, not part of the column core.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants as _C

__physics_contract__ = {
    "summary": (
        "WRF v4.8.0 MYNN-EDMF column core (mynnedmf), level-2.5 closure with "
        "the WRF namelist defaults: hybrid PBL height, Chaboureau-Bechtold "
        "subgrid clouds, dynamic multi-plume mass flux, level-2.5 stability "
        "functions, implicit qke prediction and implicit EDMF tendencies."
    ),
    "inputs": {
        "dz": "m", "u": "m/s", "v": "m/s", "w": "m/s", "th": "K", "tk": "K",
        "p": "Pa", "exner": "1", "rho": "kg/m^3", "sqv": "kg/kg (specific)",
        "sqc": "kg/kg", "sqi": "kg/kg", "qke": "m^2/s^2 (2*TKE)",
        "qsq": "kg^2/kg^2", "el": "m", "hfx": "W/m^2", "qfx": "kg/m^2/s",
        "ust": "m/s", "ts": "K", "ps": "Pa", "wspd": "m/s", "dx": "m",
    },
    "outputs": {
        "du": "m/s^2", "dv": "m/s^2", "dth": "K/s", "dqv": "kg/kg/s",
        "dqc": "kg/kg/s", "dqi": "kg/kg/s", "qke": "m^2/s^2",
        "km": "m^2/s", "kh": "m^2/s", "pblh": "m", "el": "m",
    },
    "sign_convention": (
        "z up, arrays bottom-up; hfx, qfx > 0 upward (heat/moisture into the "
        "atmosphere); tendencies are d/dt of the column state; mass flux "
        "s_aw > 0 upward."
    ),
    # The oracle's column water change is not qfx: sum rho*dz*(dqv+dqc+dqi) =
    # max(qfx, 0)*rhosfc/rho(kts) on the fixture (its bottom flux is scaled by
    # rhosfc/rho(kts), and mynn_tendencies' qvflux limiter, as written, zeroes
    # every downward surface moisture flux).  The port reproduces it.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "WRF v4.8.0 phys/MYNN-EDMF/module_bl_mynnedmf.F90 (MYNN-EDMF "
        "8f8e8b59); Olson et al. (2026) NOAA Tech. Memo. OAR GSL-77; "
        "Nakanishi & Niino (2009) JMSJ 87"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_mynn25_oracle.py: match to the "
        "running WRF oracle (fp64) on 30 columns, column water and heat integrals."
    ),
}

# ---------------------------------------------------------------------------
# Host constants (module_model_constants as module_bl_mynnedmf_common USEs
# them) -- legoESM values.
# ---------------------------------------------------------------------------
_KARMAN = _C.kappa_von_karman
_GRAV = _C.g
_P1000MB = _C.p_ref
_CP = _C.c_pd
_R_D = _C.R_d
_R_V = _C.R_v
_RCP = _C.R_d / _C.c_pd
_XLV = _C.L_v
_XLF = _C.L_f
_XLS = _C.L_s
_P608 = _C.R_v / _C.R_d - 1.0
_EP_2 = _C.R_d / _C.R_v
_RVOVRD = _C.R_v / _C.R_d
_CPV = _C.c_pv
_CLIQ = _C.c_pw
_CICE = _C.c_pi
_T0C = _C.T_freeze
_EP_QSAT = _C.epsilon_wrf_mynn          # literal ratio in qsat_blend

# --- module_bl_mynnedmf_common.F90: scheme parameters and derived values ---
_TREF = 300.0            # reference temperature [K]
_TICE = 238.0            # saturation w.r.t. ice below [K]
_TLIQ = 269.0            # all hydrometeors liquid above [K]
_EP_3 = 1.0 - _EP_2
_GTR = _GRAV / _TREF
_TV0 = _P608 * _TREF
_XLSCP = (_XLV + _XLF) / _CP
_XLVCP = _XLV / _CP
_CPHM_UNST = 16.0
_CPHH_ST, _CPHH_UNST = 5.0, 16.0
_QMIN = 0.0
_ZMAX = 1.0
_SQFAC = 3.0
_QKEMIN = 1.0e-5
_PR = 0.74
_G1 = 0.235
_B1 = 24.0
_B2 = 15.0
_C2 = 0.729
_C3 = 0.340
_C5 = 0.2
_A1 = _B1 * (1.0 - 3.0 * _G1) / 6.0
_C1 = _G1 - 1.0 / (3.0 * _A1 * 2.88449914061481660)
_A2 = _A1 * (_G1 - _C1) / (_G1 * _PR)
_G2 = _B2 / _B1 * (1.0 - _C3) + 2.0 * _A1 / _B1 * (3.0 - 2.0 * _C2)
_E1C = 3.0 * _A2 * _B2 * (1.0 - _C3)
_E2C = 9.0 * _A1 * _A2 * (1.0 - _C2)
_E3C = 9.0 * _A2 * _A2 * (1.0 - _C2) * (1.0 - _C5)
_E4C = 12.0 * _A1 * _A2 * (1.0 - _C2)
_E5C = 6.0 * _A1 * _A1
# --- mynnedmf: Canuto/Kitamura switch (CKmod) ---
_CKMOD = 1.0
# --- mym_turbulence: Prandtl-number limits ---
_PRLIMIT_FRE = 6.0
_PRLIMIT_SFC = 4.0
# --- DMP_mf: number of plumes and plume-size parameters ---
_NUP = 8
_ATOT = 0.10
_LMAX = 1000.0
_LMIN = 300.0
_DCUT = 1.2
_PLUME_D = -1.9
_FLUXPORTION = 0.75
_PGFAC = 0.0
# --- boulac_length: soft upper limit [m] ---
_BL_LMAX = 1500.0
# --- condensation_edmf: iteration count and convergence ---
_NITER = 50
_CDIFF = 1.0e-6

# --- Flatau et al. (1992) polynomials of esat_blend / qsat_blend, as written
# in module_bl_mynnedmf.F90 (liquid J0-J8, ice K0-K8) ---
_JL = (.611583699E03, .444606896E02, .143177157E01, .264224321E-1, .299291081E-3,
       .203154182E-5, .702620698E-8, .379534310E-11, -.321582393E-13)
_KI = (.609868993E03, .499320233E02, .184672631E01, .402737184E-1, .565392987E-3,
       .521693933E-5, .307839583E-7, .105785160E-9, .161444444E-12)

_P333 = 1.0 / 3.0
_P666 = 2.0 / 3.0

# --- Literal coefficients of module_bl_mynnedmf.F90 (WRF v4.8.0), named by
# --- the routine that uses them; values exactly as written in Fortran.
# --- esat_blend / qsat_blend: clamps as written ---
_ESAT_P_FRAC = 0.15
_QSAT_DP_MIN = 1e-05
_XC_MIN = 80.0
# --- mym_condensation: Chaboureau-Bechtold subgrid-flux factor Fng ---
_FNG_C = 23.9
_FNG_EXP_A = 0.4
_FNG_EXP_B = 3.8
_FNG_EXP_C = 1.6
_FNG_Q1_A = 1.7
_FNG_Q1_B = 2.5
# --- get_pblh: theta-v threshold method (stable_method = 1) ---
_PBLH_CPBLH_HI = 700.0
_PBLH_CPBLH_LO = 400.0
_PBLH_DTHV_LAND = 1.25
_PBLH_DTHV_WATER = 0.75
_PBLH_MINTHV0 = 9000000000.0
_PBLH_SBL_DAMP = 170.0
_PBLH_SBL_LIM = 200.0
_PBLH_THV_ZMAX = 200.0
# --- scale_aware: Honnert/Shin-Hong partition fits ---
_SA_BL_A = 0.106
_SA_BL_B = 0.066
_SA_BL_C = 0.071
_SA_CU_A = 0.145
_SA_CU_B = 0.172
_SA_CU_C = 0.17
_SA_CU_DEPTH = 500.0
_SA_CU_HMAX = 3500.0
_SA_DX_FAC = 2.5
_SA_EXP = 0.667
_SA_PBLH_MAX = 3000.0
# --- mym_condensation (bl_mynn_cloudpdf = 2): Chaboureau-Bechtold PDF ---
_BETA_TV = 1.61
_CB_CF0_A = 0.35
_CB_CF0_B = 3.6
_CB_CF0_C = 0.05
_CB_CF1_A = 0.39
_CB_CF1_B = 1.6
_CB_CF1_C = 0.55
_CB_CFMAX = 0.6
_CB_CF_DZ = 200.0
_CB_CF_MIN = 0.001
_CB_CLIM_DZ = 250.0
_CB_CLIM_PBL = 0.02
_CB_CLIM_SFC = 0.01
_CB_CLIM_TRP = 0.03
_CB_CLIM_ZSL_DZ = 150.0
_CB_DZ_RANGE = 500.0
_CB_LOGQC_OFF = 5.0
_CB_Q1_MIN_WATER = 2.5
_CB_QC_THR = 1e-05
_CB_QLIM_DZ = 300.0
_CB_QLIM_PBL = 0.02
_CB_QLIM_SFC = 0.008
_CB_QLIM_TRP = 0.025
_CB_QLIM_ZSL_DZ = 200.0
_CB_QL_FRAC = 0.025
_CB_RHADJ_QC = 0.07
_CB_RHADJ_QI = 0.037
_CB_RHCRIT = 0.83
_CB_RHMAX = 1.1
_CB_RH_MIN = 0.001
_CB_SGMC_SAT = 0.02
_CB_SGM_INFL = 0.2
_CB_WT2_DZ = 300.0
_CB_ZSL_FRAC = 0.1
_CB_ZSL_MAX = 150.0
_CB_ZSL_MIN = 50.0
_CB_ZSL_VT_MIN = 25.0
_TROPO_EXP = 0.19
_CB_TC_DTH = 10.0       # tropopause search: d(theta)/dz < 10 K / 1500 m
_CB_TC_DZ = 1500.0
_TROPO_HT_A = 44307.692
_TROPO_HT_MAX = 19000.0
_TROPO_HT_MIN = 4000.0
_TROPO_P0 = 101325.0
# --- DMP_mf: dynamic multi-plume mass flux ---
_K50_UNSET = 999  # integer sentinel (DMP_mf k50 search)
_MF_AC_L_OFF = 0.02
_MF_AC_L_SCALE = 0.05
_MF_AC_W_OFF = 0.012
_MF_AC_W_SCALE = 0.03
_MF_ADJ_MIN = 0.01
_MF_AUP_FAC = 1.5
_MF_BCOEFF_NEG = 0.2
_MF_BCOEFF_POS = 0.15
_MF_CB_INIT = 9000.0
_MF_CB_MF_SCALE = 0.05
_MF_CF_A = 0.36
_MF_CF_AUP_WATER = 1.2
_MF_CF_B = 1.8
_MF_CF_HI = 30.0
_MF_CF_MIN = 0.01
_MF_CF_OFF = 0.2
_MF_CSIGMA = 1.34
_MF_DQWDZ_MIN = 0.004
_MF_DW_DZ = 200.0
_MF_DW_RATE = 1.25
_MF_DXSA_HI = 12000.0
_MF_DXSA_LO = 3000.0
_MF_DZ_MAX = 250.0
_MF_EL_MIN = 0.1
_MF_WMIN_FAC = 0.1
_MF_WMIN_MAX = 0.1
_MF_ENTFAC = 0.21
_MF_ENTFAC_FREE = 0.33
_MF_ENT_DZ_BLEND = 500.0
_MF_ENT_DZ_TOP = 1500.0
_MF_ENT_INIT = 0.001
_MF_ENT_MAX_DZ = 0.9
_MF_ENT_MIN = 0.0003
_MF_ENT_QKE_HI = 1.57
_MF_ENT_QKE_LO = 1.3
_MF_ENT_WMIN0 = 0.3
_MF_ENT_WMIN_L = 0.0005
_MF_ENT_ZMAX = 4000.0
_MF_ENT_ZRATE = 5e-06
_MF_EXC_LAND = 0.58
_MF_EXC_WATER = 9.28
_MF_FLQ_MIN = 1e-05
_MF_FLTV_MIN = 0.008
_MF_FLX_A = 0.6
_MF_FLX_L = 0.04
_MF_FLX_W_OFF = 0.007
_MF_FLX_W_SCALE = 0.02
_MF_HUX_ABOVE = 0.0018
_MF_HUX_LAND = 0.003
_MF_HUX_WATER = 0.0024
_MF_PSIGW_MIN = 0.01
_MF_PWMAX = 0.4
_MF_Q1_MIN = 2.25
_MF_QCFAC0 = 1.2
_MF_QCFAC_OFF = 0.04
_MF_QCMIN_HI = 0.025
_MF_QCMIN_LO = 0.01
_MF_QC_CB = 1e-05
_MF_QC_LAND_A = 1.86
_MF_QC_LAND_B = 2.2e-05
_MF_QC_LAND_THR = 5e-05
_MF_SIGQ_A = 5.0
_MF_SIGQ_SAT = 0.015
_MF_SIGW_B = 0.8
_MF_TAUC = 1800.0
_MF_WIDTH_CLD_MIN = 300.0
_MF_WIDTH_CLD_WATER = 0.8
_MF_WIDTH_PBL = 1.1
_MF_WSPD2_MIN = 0.01
_MF_WSP_ONSET = 13.0
_MF_WSP_RANGE = 12.0
_MF_WSTAR_MIN = 0.01
_MF_W_SLOW = 0.2
_MF_Z0 = 50.0
_MF_Z50 = 50.0
_MF_ZTOP_EXTRA = 500.0
# --- mym_level2 / mym_length: O'Gorman stability-function and ESS terms (bl_mynn_ess = 1) ---
_ESS_CFAC_A = 2.7
_ESS_CFAC_MIN = 0.01
_ESS_CLAM_FREE = 0.2
_ESS_CLAM_LAND = 0.08
_ESS_CLAM_WATER = 1.8
_ESS_CLD_MAX = 0.1
_ESS_DZMIN = 400.0
_ESS_MFI_MAX = 0.2
_ESS_PBLH_FRAC = 0.3
_ESS_QKE_FAC = 0.2
_ESS_TAPER = 0.95
_ESS_UONSET = 15.0
_ESS_URANGE = 20.0
_ESS_ZCHOKE = 20.0
_ESS_ZCHOKE_DZ = 200.0
# --- boulac_length: minimum length [m] ---
_BL_MIN = 0.1
# --- mym_length (bl_mynn_mixlength = 1): BouLac blend ---
_ML_ALP1 = 0.23
_ML_ALP1_COARSE = 0.24
_ML_ALP2 = 0.3
_ML_ALP2_COARSE = 0.4
_ML_ALP3 = 5.0
_ML_ALP4_LAND = 15.0
_ML_ALP4_WATER = 20.0
_ML_ALP5 = 0.3
_ML_ALP6 = 50.0
_ML_BV_MIN = 0.001
_ML_CNS = 3.5
_ML_DX_LO = 4000.0
_ML_DX_RANGE = 12000.0
_ML_ELBL_STABLE = 0.4
_ML_ELB_INF = 10000000000.0
_ML_ELS_EXP = 0.2
_ML_ELT0 = 1e-05
_ML_ELTMAX_LAND = 400.0
_ML_ELTMAX_U0 = 20.0
_ML_ELTMAX_URANGE = 30.0
_ML_ELTMAX_WATER = 350.0
_ML_ELT_DZ_MIN = 200.0
_ML_ELT_MIN = 8.0
_ML_H1_FRAC = 0.3
_ML_H1_MAX = 600.0
_ML_H1_MIN = 300.0
_ML_PBLH_MIN = 200.0
_ML_QKW_ELB_MIN = 0.05
_ML_QKW_MAX = 30.0
_ML_QKW_MIN = 0.01
_ML_QTKE_MIN = 0.005
_ML_UONSET = 15.0
_ML_URANGE = 40.0
_ML_VFLX_OFF = 0.005
_ML_WTU1 = 0.2
# --- mym_turbulence (level 2.5): stability-function limits ---
_PR_KONDO_A = 7.0
_PR_KONDO_B = 6.873
_PR_PBLH_MIN = 200.0
_PR_RI_LO = 0.01
_SHB_MIN = 0.004
_SH_MIN = 1e-05
_S_MIN_MF = 0.04
# --- mym_predict / mynn_tendencies / diss_heat: bounds as written ---
_CPM_QV = 0.84
_DISS_MAX = 0.002
_DISS_P_SCALE = 10000.0
_PMZ_LAND_A = 0.95
_PMZ_LAND_B = 3.5
_QKE_MAX = 150.0
_QSQ_MAX = 6e-06
_QVFLUX_FRAC = 0.9
_RHO_MIN = 0.0001
_XLAND_WATER = 1.5
# --- mym_initialize: cold-start profile ---
_INIT_QKE_A = 5.0
_INIT_QKE_FAC = 1.5
_INIT_QKE_H = 700.0
_INIT_QKE_MAX = 125.0
_INIT_QKE_MIN = 0.01
_INIT_QSQ_MAX = 5e-06
_USTM_MIN = 0.01


def _poly(c, x):
    """c0 + x*(c1 + x*(c2 + ... + x*c8)), the oracle's nesting."""
    r = c[8]
    for j in range(7, -1, -1):
        r = c[j] + x * r
    return r


def _sel(cond, x, safe):
    """``x`` where ``cond`` holds, else ``safe``: the argument guard for a
    branch that ``jnp.where`` evaluates but does not select."""
    return jnp.where(cond, x, safe)


def _sqrt0(x):
    """sqrt of ``x >= 0``: the oracle's value, with derivative 0 instead of
    inf * 0 = NaN where ``x`` is 0 (the zero side of a MAX(0, .) switch)."""
    pos = x > 0.0
    return jnp.where(pos, jnp.sqrt(_sel(pos, x, 1.0)), 0.0)


def _pow0(x, p):
    """``x ** p`` (p > 0) of ``x >= 0``, AD-safe at 0 as :func:`_sqrt0`."""
    pos = x > 0.0
    return jnp.where(pos, _sel(pos, x, 1.0) ** p, 0.0)


def _prev(x):
    """x(k-1) at level k (Python index); level 0 gets x(1), never selected."""
    return jnp.concatenate([x[:1], x[:-1]])


def _next(x):
    """x(k+1) at level k; the last level gets x(kte), never selected."""
    return jnp.concatenate([x[1:], x[-1:]])


def _seq_sum(x):
    """Left-to-right sum, the oracle's accumulation loops."""
    def f(c, xi):
        return c + xi, None
    s, _ = lax.scan(f, jnp.zeros((), x.dtype), x)
    return s


def _cumsum_seq(x):
    """Running sum, the oracle's zw(k+1) = zw(k) + dz(k)."""
    def f(c, d):
        c = c + d
        return c, c
    _, s = lax.scan(f, jnp.zeros((), x.dtype), x)
    return s


def esat_blend(t):  # satcurve-ok: the oracle's own Flatau fit (esat_blend)
    """module_bl_mynnedmf.F90 FUNCTION esat_blend."""
    xc = jnp.maximum(-_XC_MIN, t - _T0C)
    esl = _poly(_JL, xc)
    esi = _poly(_KI, xc)
    chi = ((_T0C - 6.0) - t) / ((_T0C - 6.0) - _TICE)
    return jnp.where(t >= _T0C - 6.0, esl,
                     jnp.where(t <= _TICE, esi, (1.0 - chi) * esl + chi * esi))


def qsat_blend(t, p):  # satcurve-ok: the oracle's own Flatau fit (qsat_blend)
    """module_bl_mynnedmf.F90 FUNCTION qsat_blend."""
    xc = jnp.maximum(-_XC_MIN, t - _T0C)
    esl = jnp.minimum(_poly(_JL, xc), p * _ESAT_P_FRAC)
    esi = jnp.minimum(_poly(_KI, xc), p * _ESAT_P_FRAC)
    rslf = _EP_QSAT * esl / jnp.maximum(p - esl, _QSAT_DP_MIN)
    rsif = _EP_QSAT * esi / jnp.maximum(p - esi, _QSAT_DP_MIN)
    chi = ((_T0C - 6.0) - t) / ((_T0C - 6.0) - _TICE)
    return jnp.where(t >= _T0C - 6.0, rslf,
                     jnp.where(t <= _TICE, rsif, (1.0 - chi) * rslf + chi * rsif))


def xl_blend(t):
    """module_bl_mynnedmf.F90 FUNCTION xl_blend."""
    xlvt = _XLV + (_CPV - _CLIQ) * (t - _T0C)
    xlst = _XLS + (_CPV - _CICE) * (t - _T0C)
    chi = (_T0C - t) / (_T0C - _TICE)
    return jnp.where(t >= _T0C, xlvt,
                     jnp.where(t <= _TICE, xlst, (1.0 - chi) * xlvt + chi * xlst))


def tridiag2(a, b, c, d):
    """module_bl_mynnedmf.F90 tridiag2 (Thomas algorithm; a(1) unused)."""
    cp1 = c[0] / b[0]
    dp1 = d[0] / b[0]

    def fwd(carry, abcd):
        cpm, dpm = carry
        ak, bk, ck, dk = abcd
        m = bk - cpm * ak
        cpk = ck / m
        dpk = (dk - dpm * ak) / m
        return (cpk, dpk), (cpk, dpk)

    _, (cps, dps) = lax.scan(fwd, (cp1, dp1), (a[1:], b[1:], c[1:], d[1:]))
    cps = jnp.concatenate([cp1[None], cps])
    dps = jnp.concatenate([dp1[None], dps])

    def bwd(xn, cd):
        cpk, dpk = cd
        xk = dpk - cpk * xn
        return xk, xk

    _, xs = lax.scan(bwd, dps[-1], (cps[:-1], dps[:-1]), reverse=True)
    return jnp.concatenate([xs, dps[-1:]])


def _fng(q1k):
    """Non-Gaussian transport factor (Bechtold & Siebesma 1998) as written."""
    return jnp.where(
        q1k >= 1.0, 1.0,
        jnp.where((q1k >= -_FNG_Q1_A) & (q1k < 1.0), jnp.exp(-_FNG_EXP_A * (q1k - 1.0)),
                  jnp.where((q1k >= -_FNG_Q1_B) & (q1k < -_FNG_Q1_A), 3.0 + jnp.exp(-_FNG_EXP_B * (q1k + _FNG_Q1_A)),
                            jnp.minimum(_FNG_C + jnp.exp(-_FNG_EXP_C * (q1k + _FNG_Q1_B)), 60.0))))


# ---------------------------------------------------------------------------
# GET_PBLH (stable_method = 1), SCALE_AWARE
# ---------------------------------------------------------------------------
def get_pblh(thv, ust, zw, dz, xland):
    """GET_PBLH.  Returns (pblh, kpbl) with kpbl the oracle's 1-based index."""
    n = thv.shape[0]
    i = jnp.arange(n)
    # do while (zw1(k) <= 200.) from k = kts+1: minimum theta-v
    minthv = jnp.min(jnp.where((i >= 1) & (zw[:n] <= _PBLH_THV_ZMAX), thv, _PBLH_MINTHV0))
    delt_thv = jnp.where(xland - _XLAND_WATER >= 0.0, _PBLH_DTHV_WATER, _PBLH_DTHV_LAND)
    thr = minthv + delt_thv
    val = zw[:n] - _prev(dz) * jnp.minimum(
        (thv - thr) / jnp.maximum(thv - _prev(thv), 1e-6), 1.0)
    # k = kts+1 .. kte-2 exit with pblh > 0; reaching k = kte-1 sets zw(kts+1)
    hit = (thv >= thr) & (val > 0.0) & (i >= 1) & (i <= n - 3)
    pblh = jnp.where(jnp.any(hit), val[jnp.argmax(hit)], zw[1])
    wt = 0.5 * jnp.tanh((pblh - 100.0) / 100.0) + 0.5
    cpblh = _PBLH_CPBLH_LO * (1.0 - wt) + _PBLH_CPBLH_HI * wt
    pblh_stable = jnp.maximum(ust * cpblh, 10.0)
    pblh_stable = jnp.maximum(pblh_stable, pblh - _PBLH_SBL_LIM)
    pblh_stable = jnp.minimum(pblh_stable, pblh + _PBLH_SBL_LIM)
    wt = 0.5 * jnp.tanh((pblh - _PBLH_SBL_LIM) / _PBLH_SBL_DAMP) + 0.5
    pblh = pblh_stable * (1.0 - wt) + pblh * wt
    above = (i >= 1) & (i <= n - 2) & (zw[:n] >= pblh)
    kpbl = jnp.where(jnp.any(above), jnp.argmax(above), 2)
    return pblh, kpbl


def scale_aware(dx, pblh):
    """SCALE_AWARE: (Psig_bl, Psig_shcu)."""
    dxdh = jnp.maximum(_SA_DX_FAC * dx, 10.0) / jnp.minimum(pblh, _SA_PBLH_MAX)
    psig_bl = ((dxdh ** 2) + _SA_BL_A * (dxdh ** _SA_EXP)) / (
        (dxdh ** 2) + _SA_BL_B * (dxdh ** _SA_EXP) + _SA_BL_C)
    dxdh = jnp.maximum(_SA_DX_FAC * dx, 10.0) / jnp.minimum(pblh + _SA_CU_DEPTH, _SA_CU_HMAX)
    psig_shcu = ((dxdh ** 2) + _SA_CU_A * (dxdh ** _SA_EXP)) / (
        (dxdh ** 2) + _SA_CU_B * (dxdh ** _SA_EXP) + _SA_CU_C)
    psig_bl = jnp.where(psig_bl > 1.0, 1.0, psig_bl)
    psig_bl = jnp.where(psig_bl < 0.0, 0.0, psig_bl)
    psig_shcu = jnp.where(psig_shcu > 1.0, 1.0, psig_shcu)
    psig_shcu = jnp.where(psig_shcu < 0.0, 0.0, psig_shcu)
    return psig_bl, psig_shcu


# ---------------------------------------------------------------------------
# mym_condensation, CASE (2) (closure < 2.7: sigma from q'^2 only)
# ---------------------------------------------------------------------------
def mym_condensation(dz, zw, xland, thl, qw, qc, qi, p, exner, qsq, th, pblh1):
    """Returns vt, vq, cldfra, qc_bl, qi_bl (kte: vt/vq copied, clouds 0)."""
    n = th.shape[0]
    i = jnp.arange(n)
    # tropopause: first k from kte-3 down to kts passing the stability test;
    # k = kts-1 when none does (Fortran loop exit value)
    ip2 = jnp.minimum(i + 2, n - 1)
    ht = _TROPO_HT_A * (1.0 - (p / _TROPO_P0) ** _TROPO_EXP)
    tc = (((th[ip2] - th) / _sel(i <= n - 4, ht[ip2] - ht, 1.0)) < _CB_TC_DTH / _CB_TC_DZ) & \
        (ht < _TROPO_HT_MAX) & (ht > _TROPO_HT_MIN) & (i <= n - 4)
    k_tropo = jnp.maximum(3, jnp.max(jnp.where(tc, i + 1, 0)) + 2)

    pblh2 = jnp.maximum(10.0, pblh1)
    zagl = zw[:n] + 0.5 * dz
    t = th * exner
    xl = xl_blend(t)
    qsat_tk = qsat_blend(t, p)
    rh = jnp.maximum(jnp.minimum(_CB_RHMAX, qw / jnp.maximum(1E-10, qsat_tk)), _CB_RH_MIN)
    rsl = xl * qsat_tk / (_R_V * t ** 2)
    cpm = _CP + qw * _CPV
    a = 1.0 / (1.0 + xl * rsl / cpm)
    b = a * rsl
    qmq = qw - qsat_tk
    r3sq = jnp.maximum(qsq, 0.0)
    sgm = jnp.maximum(1e-13, _sqrt0(r3sq))
    sgm = jnp.minimum(sgm, qw * 0.5)
    wt = jnp.minimum(1.0, jnp.maximum(0.0, dz - 100.) / _CB_DZ_RANGE)
    sgm = sgm + sgm * _CB_SGM_INFL * wt
    sgmq = sgm
    sgmc = sgm
    wt = jnp.minimum(1.0, jnp.maximum(0.0, (zagl - (pblh2 + 10.))) / _CB_CLIM_DZ)
    clim = _CB_CLIM_PBL * (1.0 - wt) + _CB_CLIM_TRP * wt
    zsl = jnp.minimum(_CB_ZSL_MAX, jnp.maximum(_CB_ZSL_MIN, _CB_ZSL_FRAC * pblh2))
    wt = jnp.minimum(1.0, jnp.maximum(0.0, zagl - zsl) / _CB_CLIM_ZSL_DZ)
    clim = _CB_CLIM_SFC * (1.0 - wt) + clim * wt
    sgmc = jnp.maximum(sgmc, qw * clim)
    sgmc = jnp.maximum(1e-13, sgmc)
    sgmc = jnp.where(qmq >= 0.0, jnp.maximum(_CB_SGMC_SAT * qw, sgmc), sgmc)
    q1 = qmq / sgmc
    wt2 = jnp.minimum(1.0, jnp.maximum(0.0, zagl - pblh2) / _CB_WT2_DZ)
    rhcrit, rhmax = _CB_RHCRIT, _CB_RHMAX
    ci = (qi > 1.e-10) & (zagl > pblh2)
    rh_adj = jnp.minimum(rhmax, rhcrit + wt2 * _CB_RHADJ_QI * jnp.maximum(
        0.0, 10.0 + jnp.log10(_sel(ci, qi, 1.0))))
    rh_adj = jnp.maximum(rh, rh_adj)
    q1_rh = -3. + 3.0 * (rh_adj - rhcrit) / (1.0 - rhcrit)
    q1 = jnp.where(ci, jnp.maximum(q1_rh, q1), q1)
    cc = (qc > _CB_QC_THR) & (zagl > pblh2)
    rh_adj = jnp.minimum(rhmax, rhcrit + wt2 * _CB_RHADJ_QC * jnp.maximum(
        0.0, _CB_LOGQC_OFF + jnp.log10(_sel(cc, qc, 1.0))))
    rh_adj = jnp.maximum(rh, rh_adj)
    q1_rh = -3. + 3.0 * (rh_adj - rhcrit) / (1.0 - rhcrit)
    q1 = jnp.where(cc, jnp.maximum(q1_rh, q1), q1)
    q1k = q1
    wt2 = jnp.minimum(1.0, jnp.maximum(0.0, (zagl - (pblh1 - 100.)) / _CB_CF_DZ))
    cf0 = jnp.maximum(0.0, jnp.minimum(1.0, 0.5 + _CB_CF0_A * jnp.arctan(_CB_CF0_B * (q1k + _CB_CF0_C))))
    cf1 = jnp.maximum(0.0, jnp.minimum(1.0, 0.5 + _CB_CF1_A * jnp.arctan(_CB_CF1_B * (q1k + _CB_CF1_C))))
    cldfra = cf0 * (1.0 - wt2) + cf1 * wt2
    cldfra = jnp.maximum(cldfra, 0.0)          # max(cldfra_qsq, cldfra_rh = 0)
    wt = jnp.minimum(1.0, jnp.maximum(0.0, (zagl - (pblh2 + 10.0))) / _CB_QLIM_DZ)
    qlim = _CB_QLIM_PBL * (1.0 - wt) + _CB_QLIM_TRP * wt
    wt = jnp.minimum(1.0, jnp.maximum(0.0, zagl - zsl) / _CB_QLIM_ZSL_DZ)
    qlim = _CB_QLIM_SFC * (1.0 - wt) + qlim * wt
    sgmq = jnp.maximum(sgmq, qw * qlim)
    ql_water = jnp.minimum(sgmq, _CB_QL_FRAC * qw) * cldfra
    ql_ice = jnp.minimum(sgmq, _CB_QL_FRAC * qw) * cldfra
    small = cldfra < _CB_CF_MIN
    ql_ice = jnp.where(small, 0.0, ql_ice)
    ql_water = jnp.where(small, 0.0, ql_water)
    cldfra = jnp.where(small, 0.0, cldfra)
    liq_frac = jnp.minimum(1.0, jnp.maximum(0.0, (t - _TICE) / (_TLIQ - _TICE)))
    qc_bl = liq_frac * ql_water
    qi_bl = (1.0 - liq_frac) * ql_ice
    above = (i + 1) >= k_tropo
    cldfra = jnp.where(above, 0.0, cldfra)
    qc_bl = jnp.where(above, 0.0, qc_bl)
    qi_bl = jnp.where(above, 0.0, qi_bl)
    q1k = jnp.where(xland - _XLAND_WATER >= 0.0, jnp.maximum(q1, -_CB_Q1_MIN_WATER), jnp.maximum(q1, -2.0))
    fng = _fng(q1k)
    cfmax = jnp.minimum(cldfra, _CB_CFMAX)
    zsl = jnp.minimum(jnp.maximum(_CB_ZSL_VT_MIN, _CB_ZSL_FRAC * pblh2), 100.0)
    wt = jnp.minimum(zagl / zsl, 1.0)
    cfmax = cfmax * wt
    bb = b * t / th
    qww = 1.0 + _P608 * qw
    alpha = _P608 * th
    beta = (th / t) * (xl / _CP) - _BETA_TV * th
    vt = qww - cfmax * beta * bb * fng - 1.0
    vq = alpha + cfmax * beta * a * fng - _TV0
    top = i == n - 1
    vt = jnp.where(top, vt[n - 2], vt)
    vq = jnp.where(top, vq[n - 2], vq)
    cldfra = jnp.where(top, 0.0, cldfra)
    qc_bl = jnp.where(top, 0.0, qc_bl)
    qi_bl = jnp.where(top, 0.0, qi_bl)
    return vt, vq, cldfra, qc_bl, qi_bl, sgm


# ---------------------------------------------------------------------------
# condensation_edmf
# ---------------------------------------------------------------------------
def condensation_edmf(qt, thl, p, zagl):
    """condensation_edmf: plume (THV, QC) after the oracle's fixed-point
    iteration (at most 50 iterations, stop when |QC - QCOLD| < 1e-6)."""
    exn = (p / _P1000MB) ** _RCP

    def body(_, st):
        qc, done = st
        t = exn * thl + _XLVCP / exn * qc
        qs = qsat_blend(t, p)
        qcn = 0.5 * qc + 0.5 * jnp.maximum((qt - qs), 0.0)
        return jnp.where(done, qc, qcn), done | (jnp.abs(qcn - qc) < _CDIFF)

    qc, _ = lax.fori_loop(0, _NITER, body,
                          (jnp.zeros_like(qt), jnp.zeros(jnp.shape(qt), bool)))
    t = exn * thl + _XLVCP / exn * qc
    qs = qsat_blend(t, p)
    qc = jnp.maximum(qt - qs, 0.0)
    qc = jnp.where(zagl < 100., 0.0, qc)
    thv = (thl + _XLVCP / exn * qc) * (1.0 + qt * (_RVOVRD - 1.0) - _RVOVRD * qc)
    return thv, qc


# ---------------------------------------------------------------------------
# DMP_mf (bl_mynn_edmf = 1, momentum transport on, tke transport off,
# env_subs = .false.)
# ---------------------------------------------------------------------------
class MassFlux(NamedTuple):
    """DMP_mf outputs.  ``s_aw*`` are interface arrays of length nlev+1
    (Fortran kts..kte+1); the rest are mass-level arrays or scalars."""
    s_aw: jax.Array
    s_awthl: jax.Array
    s_awqt: jax.Array
    s_awqv: jax.Array
    s_awqc: jax.Array
    s_awu: jax.Array
    s_awv: jax.Array
    edmf_a: jax.Array
    edmf_w: jax.Array
    edmf_qt: jax.Array
    edmf_thl: jax.Array
    edmf_ent: jax.Array
    edmf_qc: jax.Array
    tkeprod_up: jax.Array
    cldfra: jax.Array
    qc_bl: jax.Array
    vt: jax.Array
    vq: jax.Array
    maxmf: jax.Array
    ztop: jax.Array
    maxwidth: jax.Array
    excess_h: jax.Array
    excess_q: jax.Array


def dmp_mf(zw, dz, p, rho, u, v, w, thl, thv, tk, qt, qv, qc, qke, qsq,
           exner, vt, vq, ust, flt, fltv, flq, pblh, kpbl, dx, xland, th_sfc,
           qc_bl, cldfra, psig_shcu, el):
    """DMP_mf on the ported path.  ``th_sfc`` is what mynnedmf passes as the
    dummy ``ts`` (ts/exner(kts)); ``qt, qv, qc`` are specific contents."""
    n = thl.shape[0]
    i = jnp.arange(n)
    dtype = thl.dtype
    water = xland - _XLAND_WATER >= 0.0
    # ---- loop k = 1, kte-1 (exit above pblh+500): resolved w, k50, PBL qke,
    # cloud base ----
    zagl = zw[:n] + 0.5 * dz
    act = (zagl <= pblh + _MF_ZTOP_EXTRA) & (i <= n - 2)
    wpbl = jnp.where(w < 0.0, 2.0 * w, w)
    maxw = jnp.max(jnp.where(act, jnp.abs(wpbl), 0.0))
    k50m = act & (zagl <= _MF_Z50)
    # k50 is never assigned when no level is below 50 m; the oracle is built
    # with the MYNN-EDMF CI's -finit-integer=-999 (see scripts/validate/oracle)
    k50 = jnp.where(jnp.any(k50m), jnp.max(jnp.where(k50m, i + 1, 0)), -_K50_UNSET)
    qkebl = _seq_sum(jnp.where(act & ((i + 1) <= kpbl), qke, 0.0))
    qkebl = qkebl / kpbl.astype(dtype)
    qc_sgs = jnp.maximum(qc, qc_bl)
    cbm = act & (qc_sgs > _MF_QC_CB) & (cldfra >= 0.5)
    cloud_base = jnp.where(jnp.any(cbm), zw[jnp.argmax(cbm)], _MF_CB_INIT)
    maxw = jnp.maximum(0.0, maxw - 1.0)
    psig_w = jnp.maximum(0.0, 1.0 - maxw)
    psig_w = jnp.minimum(psig_w, psig_shcu)
    fltv2 = jnp.where((psig_w < _MF_PSIGW_MIN) & (fltv > 0.0), -1. * fltv, fltv)
    # ---- superadiabatic through 50 m ----
    hux = jnp.where(water, -_MF_HUX_WATER, -_MF_HUX_LAND)
    tvs = th_sfc * (1.0 + _P608 * qv[0])
    sfc_sup = (thv[0] - tvs) / (0.5 * dz[0]) < hux
    dthvdz = (thv - _prev(thv)) / (0.5 * (dz + _prev(dz)))
    rng = (i >= 1) & ((i + 1) <= jnp.maximum(2, k50))
    superadiabatic = sfc_sup & jnp.all(jnp.where(rng, dthvdz < -_MF_HUX_ABOVE, True))
    # ---- plume widths ----
    maxwidth_dx = jnp.minimum(dx * _DCUT, _LMAX)
    maxwidth_pbl = jnp.minimum(_MF_WIDTH_PBL * pblh, _LMAX)
    maxwidth_cld = jnp.where(water,
                             jnp.minimum(_LMAX, jnp.maximum(_MF_WIDTH_CLD_WATER * cloud_base, _MF_WIDTH_CLD_MIN)),
                             jnp.minimum(_LMAX, jnp.maximum(0.5 * cloud_base, _MF_WIDTH_CLD_MIN)))
    wspd_pbl = jnp.sqrt(jnp.maximum(u[0] ** 2 + v[0] ** 2, _MF_WSPD2_MIN))
    maxwidth_flx = jnp.where(
        water,
        jnp.maximum(jnp.minimum(1000. * (_MF_FLX_A * jnp.tanh((fltv - _MF_FLX_W_OFF) / _MF_FLX_W_SCALE) + 0.5), 1000.), 0.0),
        jnp.maximum(jnp.minimum(1000. * (_MF_FLX_A * jnp.tanh((fltv - _MF_FLX_L) / _MF_FLX_L) + 0.5), 1000.), 0.0))
    maxwidth = jnp.minimum(maxwidth_dx, maxwidth_pbl)
    maxwidth = jnp.minimum(maxwidth, maxwidth_cld)
    maxwidth = jnp.minimum(maxwidth, maxwidth_flx)
    minwidth = _LMIN
    maxwidth = jnp.where(maxwidth <= minwidth, 0.0, maxwidth)
    active = (flt > 0.0) & (fltv2 > _MF_FLTV_MIN) & (maxwidth > minwidth) & superadiabatic
    # ---- plume set-up (only used where `active`) ----
    # Plume widths only matter where `active`; elsewhere they are evaluated on
    # a valid width so that no unselected branch divides by a zero width.
    dl = (_sel(active, maxwidth, _LMAX) - minwidth) / float(_NUP - 1)
    ls = [minwidth + dl * float(ip) for ip in range(_NUP)]
    cn = jnp.zeros((), dtype)
    for l in ls:
        cn = cn + l ** _PLUME_D * (l * l) / (dx * dx) * dl
    cfac = _ATOT / _sel(active, cn, 1.0)
    acfac = jnp.where(water, 0.5 * jnp.tanh((fltv2 - _MF_AC_W_OFF) / _MF_AC_W_SCALE) + 0.5,
                      0.5 * jnp.tanh((fltv2 - _MF_AC_L_OFF) / _MF_AC_L_SCALE) + 0.5)
    ac_wsp = 1.0 - jnp.minimum((jnp.maximum(wspd_pbl - _MF_WSP_ONSET, 0.0)) / _MF_WSP_RANGE, 1.0)
    acfac = jnp.minimum(acfac, ac_wsp)
    upa0 = jnp.stack([cfac * l ** _PLUME_D * l * l / (dx * dx) * dl * acfac for l in ls])
    pblh_s = _sel(active, pblh, 1.0)
    wstar = jnp.maximum(_MF_WSTAR_MIN, (_GTR * _sel(active, fltv2, 1.0) * pblh_s) ** _P333)
    qstar = jnp.maximum(flq, _MF_FLQ_MIN) / wstar
    thstar = flt / wstar
    csigma = _MF_CSIGMA
    exc_fac = jnp.where(water, _MF_EXC_WATER, _MF_EXC_LAND) * ac_wsp
    z0 = _MF_Z0
    sigmaw = csigma * wstar * (z0 / pblh_s) ** _P333 * (1.0 - _MF_SIGW_B * z0 / pblh_s)
    sigmaqt = csigma * qstar * (z0 / pblh_s) ** _P333
    sigmath = csigma * thstar * (z0 / pblh_s) ** _P333
    wmin = jnp.minimum(sigmaw * _MF_WMIN_FAC, _MF_WMIN_MAX)
    wmax = jnp.minimum(sigmaw * _MF_PWMAX, 0.5)
    ipf = jnp.arange(1, _NUP + 1).astype(dtype)
    upw2 = wmin + ipf / float(_NUP) * (wmax - wmin)
    sigmaw_s = _sel(active, sigmaw, 1.0)
    exc_heat = exc_fac * upw2 * sigmath / sigmaw_s
    exc_moist = exc_fac * upw2 * sigmaqt / sigmaw_s

    def i2(x):      # (x(kts)*dz(kts+1) + x(kts+1)*dz(kts)) / (dz(kts)+dz(kts+1))
        return (x[0] * dz[1] + x[1] * dz[0]) / (dz[0] + dz[1])

    ones = jnp.ones(_NUP, dtype)
    rhoi = jnp.where(i == 0, rho, (_prev(rho) * dz + rho * _prev(dz)) / (_prev(dz) + dz))
    dzi = jnp.where(i == 0, 0.5 * dz, (_prev(dz) * dz + dz * _prev(dz)) / (_prev(dz) + dz))
    dxsa = 1.0 - jnp.minimum(jnp.maximum((_MF_DXSA_HI - dx) / (_MF_DXSA_HI - _MF_DXSA_LO), 0.0), 1.0)
    lsv = jnp.stack(ls)
    wminl = _MF_ENT_WMIN0 + lsv * _MF_ENT_WMIN_L
    entfac0 = _MF_ENTFAC * jnp.minimum(_MF_ENT_QKE_HI, jnp.maximum(_MF_ENT_QKE_LO, _sqrt0(qkebl)))
    zlim = jnp.minimum(pblh + _MF_ENT_DZ_TOP, _MF_ENT_ZMAX)

    def z2(row):
        return jnp.zeros((n + 1, _NUP), dtype).at[1].set(row)

    upw = z2(upw2)
    upthl = z2(i2(thl) + exc_heat)
    upqt = z2(i2(qt) + exc_moist)
    upqc = jnp.zeros((n + 1, _NUP), dtype)
    upu = z2(i2(u) * ones)
    upv = z2(i2(v) * ones)
    upqke = z2(i2(qke) * ones)
    upqsq = z2(i2(qsq) * ones)
    upa = jnp.zeros((n + 1, _NUP), dtype).at[0].set(upa0).at[1].set(upa0)
    ent = jnp.full((n, _NUP), _MF_ENT_INIT, dtype)

    # ---- plume integration, Fortran k = kts+1 .. kte-2 (Python j = k-1) ----
    def step(carry, j):
        (upw, upthl, upqt, upqc, upu, upv, upqke, upqsq, upa, ent,
         alive, ktop_pl, ktop, fail) = carry
        k = j + 1
        za = zw[j] + 0.5 * dz[j]
        w_k = upw[j]                               # UPW(k)
        w_km1 = upw[j - 1]                         # UPW(k-1)
        wt2 = jnp.minimum(1.0, jnp.maximum(0.0, za - pblh) / _MF_ENT_DZ_BLEND)
        entfac = entfac0 * (1.0 - wt2) + wt2 * _MF_ENTFAC_FREE
        e = entfac / (jnp.minimum(jnp.maximum(w_km1, wminl), 1.0) * lsv)
        e = jnp.maximum(e, _MF_ENT_MIN)
        e = jnp.where(zw[j] >= zlim, e + (zw[j] - zlim) * _MF_ENT_ZRATE, e)
        e = e * (1.0 - 0.0)                        # (one - pattern_spp_pbl1(k))
        e = jnp.minimum(e, _MF_ENT_MAX_DZ / dz[j])
        uk = (u[j] * dz[j + 1] + u[j + 1] * dz[j]) / (dz[j + 1] + dz[j])
        ukm1 = (u[j - 1] * dz[j] + u[j] * dz[j - 1]) / (dz[j - 1] + dz[j])
        vk = (v[j] * dz[j + 1] + v[j + 1] * dz[j]) / (dz[j + 1] + dz[j])
        vkm1 = (v[j - 1] * dz[j] + v[j] * dz[j - 1]) / (dz[j - 1] + dz[j])
        entexp = e * dz[j]
        entexm = entexp * _P333
        qtn = upqt[j] * (1.0 - entexp) + qt[j] * entexp
        thln = upthl[j] * (1.0 - entexp) + thl[j] * entexp
        un = upu[j] * (1.0 - entexm) + u[j] * entexm + dxsa * _PGFAC * (uk - ukm1)
        vn = upv[j] * (1.0 - entexm) + v[j] * entexm + dxsa * _PGFAC * (vk - vkm1)
        qken = upqke[j] * (1.0 - entexp) + qke[j] * entexp
        qsqn = upqsq[j] * (1.0 - entexp) + qsq[j] * entexp
        pk = (p[j] * dz[j + 1] + p[j + 1] * dz[j]) / (dz[j] + dz[j + 1])
        thvn, qcn = condensation_edmf(qtn, thln, pk, zw[j + 1])
        thvk = (thv[j] * dz[j + 1] + thv[j + 1] * dz[j]) / (dz[j + 1] + dz[j])
        bb = _GRAV * (thvn / thvk - 1.0)
        bcoeff = jnp.where(bb > 0., _MF_BCOEFF_POS, _MF_BCOEFF_NEG)
        dzc = jnp.minimum(dz[j], _MF_DZ_MAX)
        slow = w_km1 < _MF_W_SLOW
        wden = jnp.where(slow, jnp.maximum(w_k, _MF_W_SLOW), _sel(alive, w_k, 1.0))
        wn = w_k + (-2. * e * w_k + bcoeff * bb / wden) * dzc
        dwmax = jnp.minimum(_MF_DW_RATE * dz[j] / _MF_DW_DZ, 2.0)
        wn = jnp.where(wn > w_k + dwmax, w_k + dwmax, wn)
        wn = jnp.where(wn < w_k - dwmax, w_k - dwmax, wn)
        wn = jnp.minimum(jnp.maximum(wn, 0.0), 3.0)
        failk = alive & (k == 2) & (wn < 1e-8)      # nup2 = 0; exit this plume
        go = alive & ~failk & (wn > 0.)
        stop = alive & ~go
        ent = ent.at[j].set(jnp.where(alive, e, ent[j]))

        def put(arr, val):
            return arr.at[j + 1].set(jnp.where(go, val, arr[j + 1]))

        upw = put(upw, wn)
        upthl = put(upthl, thln)
        upqt = put(upqt, qtn)
        upqc = put(upqc, qcn)
        upu = put(upu, un)
        upv = put(upv, vn)
        upqke = put(upqke, qken)
        upqsq = put(upqsq, qsqn)
        upa = put(upa, upa[j])
        ktop = jnp.where(jnp.any(go), jnp.maximum(ktop, k), ktop)
        ktop_pl = jnp.where(stop, k, ktop_pl)
        fail = fail | jnp.any(failk)
        return (upw, upthl, upqt, upqc, upu, upv, upqke, upqsq, upa, ent,
                go, ktop_pl, ktop, fail), None

    carry = (upw, upthl, upqt, upqc, upu, upv, upqke, upqsq, upa, ent,
             jnp.broadcast_to(active, (_NUP,)), jnp.full(_NUP, n - 1, i.dtype),
             jnp.zeros((), i.dtype), jnp.zeros((), bool))
    carry, _ = lax.scan(step, carry, jnp.arange(1, n - 2))
    (upw, upthl, upqt, upqc, upu, upv, upqke, upqsq, upa, ent, _, ktop_pl,
     ktop, fail) = carry
    on = active & ~fail                            # nup2 > 0
    excess_h = jnp.where(active, jnp.maximum(0.0, exc_heat[-1]), 0.0)
    excess_q = jnp.where(active, jnp.maximum(0.0, exc_moist[-1]), 0.0)
    ktop = jnp.minimum(ktop, n - 1)
    ztop = jnp.where(ktop == 0, 0.0, zw[jnp.maximum(ktop - 1, 0)])
    # ---- fluxes at interfaces k = kts .. kte-1 ----
    rhoi1 = jnp.concatenate([rhoi, jnp.zeros(1, dtype)])
    lev = jnp.arange(n + 1) <= n - 2

    def acc(q):
        s = jnp.zeros(n + 1, dtype)
        for ip in range(_NUP):
            t = rhoi1 * upa[:, ip] * upw[:, ip]
            t = t * psig_w if q is None else t * q[:, ip] * psig_w
            s = s + jnp.where(lev, t, 0.0)
        return s

    s_aw = acc(None)
    s_awthl = acc(upthl)
    s_awqt = acc(upqt)
    s_awqc = acc(upqc)
    s_awqv = s_awqt - s_awqc
    s_awu = acc(upu)
    s_awv = acc(upv)
    # flux limiter
    flx1 = jnp.where(jnp.abs(s_aw[1]) > 1e-8,
                     jnp.maximum(s_aw[1] * (thv[0] - thv[1]) / dzi[1], 1.0e-6), 0.0)
    flt2 = jnp.maximum(fltv, 0.0)
    lim = (flx1 > _FLUXPORTION * flt2 / dz[0]) & (flx1 > 0.0)
    adjustment = jnp.where(
        lim, jnp.maximum(_MF_ADJ_MIN, _FLUXPORTION * flt2 / dz[0] / _sel(lim, flx1, 1.0)), 1.0)
    s_aw, s_awthl, s_awqt, s_awqc, s_awqv, s_awu, s_awv = [
        jnp.where(lim, x * adjustment, x)
        for x in (s_aw, s_awthl, s_awqt, s_awqc, s_awqv, s_awu, s_awv)]
    upa = jnp.where(lim, upa * adjustment, upa)
    # ---- mean updraft properties on mass levels ----
    ea = jnp.zeros(n, dtype)
    ew = jnp.zeros(n, dtype)
    eqt = jnp.zeros(n, dtype)
    ethl = jnp.zeros(n, dtype)
    eent = jnp.zeros(n, dtype)
    eqc = jnp.zeros(n, dtype)
    for ip in range(_NUP):                         # k = kts: first interface
        ea = ea.at[0].add(upa[1, ip])
        ew = ew.at[0].add(upa[1, ip] * 0.5 * upw[1, ip])
        eqt = eqt.at[0].add(upa[1, ip] * upqt[1, ip])
        ethl = ethl.at[0].add(upa[1, ip] * upthl[1, ip])
        eent = eent.at[0].add(upa[1, ip] * ent[1, ip])
        eqc = eqc.at[0].add(upa[1, ip] * upqc[1, ip])
    dzi_p = _next(dzi)
    den = dzi_p + dzi
    ent_e = jnp.concatenate([ent, ent[-1:]])

    def intp(x):    # (x(k+1)*dzi(k) + x(k)*dzi(k+1)) / (dzi(k+1)+dzi(k))
        return (x[1:n + 1] * dzi + x[:n] * dzi_p) / den

    for ip in range(_NUP):                         # k = kts+1 .. ktop_plume-1
        inr = (i >= 1) & ((i + 1) <= ktop_pl[ip] - 1)
        upak = intp(upa[:, ip])
        ea = ea + jnp.where(inr, upak, 0.0)
        ew = ew + jnp.where(inr, upak * intp(upw[:, ip]), 0.0)
        eqt = eqt + jnp.where(inr, upak * intp(upqt[:, ip]), 0.0)
        ethl = ethl + jnp.where(inr, upak * intp(upthl[:, ip]), 0.0)
        eent = eent + jnp.where(inr, upak * intp(ent_e[:, ip]), 0.0)
        cb = (jnp.abs(upqc[:n, ip]) < 1e-10) & (upqc[1:n + 1, ip] > 0.0)
        eqc = eqc + jnp.where(inr, jnp.where(cb, upak * upqc[1:n + 1, ip],
                                             upak * intp(upqc[:, ip])), 0.0)
    for ip in range(_NUP):                         # top level of each plume
        kt = ktop_pl[ip] - 1
        ea = ea.at[kt].add(upa[kt, ip])
        ew = ew.at[kt].add(upa[kt, ip] * upw[kt, ip])
        eqt = eqt.at[kt].add(upa[kt, ip] * upqt[kt, ip])
        ethl = ethl.at[kt].add(upa[kt, ip] * upthl[kt, ip])
        eent = eent.at[kt].add(upa[kt, ip] * ent[kt, ip])
        eqc = eqc.at[kt].add(upa[kt, ip] * upqc[kt, ip])
    pos = (ea > 0.) & (i <= n - 2)
    eas = _sel(pos, ea, 1.0)
    ew = jnp.where(pos, ew / eas, ew)
    eqt = jnp.where(pos, eqt / eas, eqt)
    ethl = jnp.where(pos, ethl / eas, ethl)
    eent = jnp.where(pos, eent / eas, eent)
    eqc = jnp.where(pos, eqc / eas, eqc)
    ea = jnp.where(pos, ea * psig_w, ea)
    maxmf = jnp.maximum(jnp.max(jnp.where(i <= n - 2, ea * ew, 0.0)), 0.0)
    tkeprod_up = jnp.where(i <= n - 2, (jnp.abs(ew) ** 3) * ea / (
        _B1 * jnp.maximum(0.5 * (el + _next(el)), _MF_EL_MIN)), 0.0)
    # ---- shallow-cumulus clouds (k = kts+1 .. kte-2, k <= ktop) ----
    edmf_th = ethl + _XLVCP / exner * eqc
    qti_p = (qt * _next(dz) + _next(qt) * dz) / (_next(dz) + dz)   # qti(k+1)
    qti = jnp.concatenate([qt[:1], qti_p[:-1]])                     # qti(k)
    cm = (i >= 1) & (i <= n - 3) & ((i + 1) <= ktop) & (eqc > 0.0)
    mf_at_cb = jnp.where(jnp.any(cm), (ea * ew)[jnp.argmax(cm)], 0.0)
    aup = ea
    dqwdz = (qti - qti_p) / dz
    xl = xl_blend(tk)
    qsat_tk = qsat_blend(tk, p)
    rsl = xl * qsat_tk / (_R_V * tk ** 2)
    cpm = _CP + qt * _CPV
    a = 1.0 / (1.0 + xl * rsl / cpm)
    b9 = a * rsl
    q2p = _XLVCP / exner
    pt = thl + q2p * eqc * aup
    bb = b9 * tk / pt
    qww = 1.0 + _P608 * qt
    alpha = _P608 * pt
    beta = pt * xl / (tk * _CP) - _BETA_TV * pt
    tauc = _MF_TAUC
    sigq = _MF_SIGQ_A * aup * ew * jnp.maximum(_MF_DQWDZ_MIN, tauc * jnp.maximum(0.0, dqwdz))
    sigq = jnp.maximum(sigq, 3.0 * aup * (eqt - qt))
    qmq = qt - qsat_tk
    sigq = jnp.where(qmq > 0.0, jnp.minimum(qsat_tk * _MF_SIGQ_SAT, sigq), sigq)
    q1 = qmq / _sel(cm, sigq, 1.0)
    cffac = jnp.minimum(jnp.maximum(0.0, mf_at_cb - _MF_CB_MF_SCALE) / _MF_CB_MF_SCALE, 1.0)
    cf_mf = jnp.where(
        water,
        jnp.maximum(jnp.minimum(jnp.maximum(
            0.5 + _MF_CF_A * jnp.arctan(_MF_CF_B * (q1 + _MF_CF_OFF + cffac)), _MF_CF_MIN), 1.0), _MF_CF_AUP_WATER * aup),
        jnp.maximum(jnp.minimum(jnp.maximum(
            0.5 + _MF_CF_A * jnp.arctan(_MF_CF_B * (q1 + _MF_CF_OFF)), _MF_CF_MIN), 1.0), _MF_CF_B * aup))
    qcfac = _MF_QCFAC0 + 1.0 * jnp.minimum(jnp.maximum(0.0, mf_at_cb - _MF_QCFAC_OFF) / _MF_CB_MF_SCALE, 1.0)
    qc_mf = jnp.where(water, qcfac * (eqc * aup),
                      jnp.where(eqc * aup > _MF_QC_LAND_THR, _MF_QC_LAND_A * (eqc * aup) - _MF_QC_LAND_B,
                                _MF_QCFAC0 * (eqc * aup)))
    wt2 = 1.0 - jnp.minimum(1.0, jnp.maximum(0.0, (cf_mf - _MF_CF_HI)) / (100.0 - _MF_CF_HI))
    qc_mf_min = wt2 * qsat_tk * _MF_QCMIN_LO * cf_mf + (1.0 - wt2) * qsat_tk * _MF_QCMIN_HI * cf_mf
    qc_mf = jnp.maximum(qc_mf, qc_mf_min)
    cf_new = jnp.maximum(cf_mf, cldfra)
    pct_mf = cf_mf / _sel(cm, cf_new, 1.0)
    qc_new = qc_mf * pct_mf + (1.0 - pct_mf) * qc_bl
    fng = _fng(jnp.maximum(q1, -_MF_Q1_MIN))
    vt_new = qww - (_MF_AUP_FAC * aup) * beta * bb * fng - 1.0
    vq_new = alpha + (_MF_AUP_FAC * aup) * beta * a * fng - _TV0
    cm = cm & on
    cldfra = jnp.where(cm, cf_new, cldfra)
    qc_bl = jnp.where(cm, qc_new, qc_bl)
    vt = jnp.where(cm, vt_new, vt)
    vq = jnp.where(cm, vq_new, vq)
    # ---- everything stays zero when nup2 = 0 ----
    def z(x):
        return jnp.where(on, x, jnp.zeros_like(x))

    s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv = map(
        z, (s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv))
    ea, ew, eqt, ethl, eent, eqc, tkeprod_up, maxmf = map(
        z, (ea, ew, eqt, ethl, eent, eqc, tkeprod_up, maxmf))
    maxqc = jnp.max(jnp.where(i + 1 <= ktop, eqc, -jnp.inf))
    maxmf = jnp.where((ktop > 0) & (maxqc < 1.E-8), -1. * maxmf, maxmf)
    return MassFlux(s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv, ea, ew,
                    eqt, ethl, eent, eqc, tkeprod_up, cldfra, qc_bl, vt, vq, maxmf,
                    ztop, maxwidth, excess_h, excess_q)


# ---------------------------------------------------------------------------
# mym_level2 (bl_mynn_ess = 1)
# ---------------------------------------------------------------------------
def mym_level2(zw, dz, xland, pblh, u, v, thl, thv, thlv, th, p, exner, qke,
               cldfra, edmf_a, edmf_w, qw):
    """Returns dtl, dqw, dtv, gm, gh, sm, sh (level kts = 0)."""
    n = th.shape[0]
    i = jnp.arange(n)
    up = i >= 1
    ugrid = _sqrt0(u[0] ** 2 + v[0] ** 2)
    taper = 1.0 - _ESS_TAPER * jnp.minimum(1.0, jnp.maximum(0.0, ugrid - _ESS_UONSET) / _ESS_URANGE)
    clam0 = jnp.where(xland - _XLAND_WATER >= 0.0, _ESS_CLAM_WATER * taper, _ESS_CLAM_LAND)
    dzk = 0.5 * (dz + _prev(dz))
    duz = (u - _prev(u)) ** 2 + (v - _prev(v)) ** 2
    duz = duz / dzk ** 2
    dtz = (thl - _prev(thl)) / dzk
    dqz = (qw - _prev(qw)) / dzk
    zk = zw[:n]
    wt = jnp.minimum(1.0, jnp.maximum(0.0, zk - (pblh + 100.)) /
                     jnp.maximum(_ESS_DZMIN, _ESS_PBLH_FRAC * pblh))
    clam = clam0 * (1.0 - wt) + _ESS_CLAM_FREE * wt
    clam = clam * jnp.minimum(1.0, jnp.maximum(0.0, zk - _ESS_ZCHOKE) / _ESS_ZCHOKE_DZ)
    thvp = 0.0
    thv1 = thvp * thv + (1.0 - thvp) * thlv
    thv0 = _prev(thv1)
    tabs = th * exner
    eth = th * jnp.exp(xl_blend(tabs) * qsat_blend(tabs, p) / (_CP * tabs))
    eth1, eth0 = eth, _prev(eth)
    mfi = jnp.maximum(0.0, jnp.minimum(_ESS_MFI_MAX, (0.5 * (edmf_a + _prev(edmf_a))) *
                                       (0.5 * (edmf_w + _prev(edmf_w)))))
    qkei = jnp.maximum(_QKEMIN, 0.5 * (qke + _prev(qke)))
    qkei = jnp.minimum(0.5, jnp.sqrt(0.25 * qkei)) * _ESS_QKE_FAC
    cldfrai = 0.5 * (cldfra + _prev(cldfra))
    cfac = jnp.maximum(_ESS_CFAC_MIN, jnp.minimum(_ESS_CFAC_A * (cldfrai - 1.0) ** 2, 1.0))
    cldfrai = jnp.minimum(_ESS_CLD_MAX, cldfrai * cfac) * 10.0
    lam = clam * jnp.maximum(mfi, qkei) * cldfrai
    dtq = (thv1 - thv0) / dzk + lam * jnp.minimum(0.0, (eth1 - eth0) / dzk)
    gm = jnp.where(up, duz, 0.0)
    gh = jnp.where(up, -dtq * _GTR, 0.0)
    ri = jnp.where(up, -gh / jnp.maximum(duz, 1.0e-10), 0.0)
    a2fac = 1.0 / (1.0 + _CKMOD * jnp.maximum(ri, 0.0))
    rfc = _G1 / (_G1 + _G2)
    f1 = _B1 * (_G1 - _C1) + 3.0 * _A2 * a2fac * (1.0 - _C2) * (1.0 - _C5) \
        + 2.0 * _A1 * (3.0 - 2.0 * _C2)
    f2 = _B1 * (_G1 + _G2) - 3.0 * _A1 * (1.0 - _C2)
    rf1 = _B1 * (_G1 - _C1) / f1
    rf2 = _B1 * _G1 / f2
    smc = _A1 / (_A2 * a2fac) * f1 / f2
    shc = 3.0 * (_A2 * a2fac) * (_G1 + _G2)
    ri1 = 0.5 / smc
    ri2 = rf1 * smc
    ri3 = 4.0 * rf2 * smc - 2.0 * ri2
    ri4 = ri2 ** 2
    rf = jnp.minimum(ri1 * (ri + ri2 - jnp.sqrt(ri ** 2 - ri3 * ri + ri4)), rfc)
    sh = shc * (rfc - rf) / (1.0 - rf)
    sm = smc * (rf1 - rf) / (rf2 - rf) * sh
    zero = lambda x: jnp.where(up, x, 0.0)
    return zero(dtz), zero(dqz), zero(dtq), gm, gh, zero(sm), zero(sh)


# ---------------------------------------------------------------------------
# boulac_length, mym_length (bl_mynn_mixlength = 1)
# ---------------------------------------------------------------------------
def boulac_length(zw, dz, qtke, theta):
    """boulac_length: (lb1, lb2) = (min, geometric mean) of the up/down
    displacement lengths."""
    n = theta.shape[0]
    iz = jnp.arange(n)
    beta = _GTR

    def up(carry, s):
        zup, zup_inf, zzz, found, dlu = carry
        izz = iz + s
        ok = (iz < n - 1) & (izz <= n - 2) & ~found
        izc = jnp.clip(izz, 0, n - 2)
        dzt = dz[izc]
        th_z, th_z1, th_i = theta[izc], theta[izc + 1], theta
        zup_n = zup - beta * th_i * dzt
        zup_n = zup_n + beta * (th_z1 + th_z) * dzt * 0.5
        zzz_n = zzz + dzt
        hit = ok & (qtke < zup_n) & (qtke >= zup_inf)
        bbb = (th_z1 - th_z) / dzt
        nz = bbb != 0.
        dth = th_z - th_i
        tl_b = (-beta * dth + _sqrt0(jnp.maximum(
            0.0, (beta * dth) ** 2 + 2.0 * bbb * beta * (qtke - zup_inf)))) / _sel(nz, bbb, 1.0) / beta
        ne = th_z != th_i
        tl_c = (qtke - zup_inf) / (beta * _sel(ne, dth, 1.0))
        tl = jnp.where(nz, tl_b, jnp.where(ne, tl_c, 0.0))
        dlu = jnp.where(hit, zzz_n - dzt + tl, dlu)
        found = found | hit
        zup = jnp.where(ok, zup_n, zup)
        zzz = jnp.where(ok, zzz_n, zzz)
        zup_inf = jnp.where(ok, zup_n, zup_inf)
        return (zup, zup_inf, zzz, found, dlu), None

    zero = jnp.zeros(n, theta.dtype)
    dlu0 = zw[n] - zw[:n] - dz * 0.5
    (_, _, _, _, dlu), _ = lax.scan(up, (zero, zero, zero, jnp.zeros(n, bool), dlu0),
                                    jnp.arange(n))

    def dn(carry, s):
        zdo, zdo_sup, zzz, found, dld = carry
        izz = iz - s
        ok = (iz > 0) & (izz >= 1) & ~found
        izc = jnp.clip(izz, 1, n - 1)
        dzt = dz[izc - 1]
        th_z, th_zm1, th_i = theta[izc], theta[izc - 1], theta
        zdo_n = zdo + beta * th_i * dzt
        zdo_n = zdo_n - beta * (th_zm1 + th_z) * dzt * 0.5
        zzz_n = zzz + dzt
        hit = ok & (qtke < zdo_n) & (qtke >= zdo_sup)
        bbb = (th_z - th_zm1) / dzt
        nz = bbb != 0.
        dth = th_z - th_i
        tl_b = (beta * dth + _sqrt0(jnp.maximum(
            0.0, (beta * dth) ** 2 + 2.0 * bbb * beta * (qtke - zdo_sup)))) / _sel(nz, bbb, 1.0) / beta
        ne = th_z != th_i
        tl_c = (qtke - zdo_sup) / (beta * _sel(ne, dth, 1.0))
        tl = jnp.where(nz, tl_b, jnp.where(ne, tl_c, 0.0))
        dld = jnp.where(hit, zzz_n - dzt + tl, dld)
        found = found | hit
        zdo = jnp.where(ok, zdo_n, zdo)
        zzz = jnp.where(ok, zzz_n, zzz)
        zdo_sup = jnp.where(ok, zdo_n, zdo_sup)
        return (zdo, zdo_sup, zzz, found, dld), None

    (_, _, _, _, dld), _ = lax.scan(dn, (zero, zero, zero, jnp.zeros(n, bool), zw[:n]),
                                    jnp.arange(n))
    dld = jnp.minimum(dld, zw[1:n + 1])
    dlu = jnp.maximum(_BL_MIN, dlu / (1.0 + (dlu / _BL_LMAX)))
    dld = jnp.maximum(_BL_MIN, dld / (1.0 + (dld / _BL_LMAX)))
    lb1 = jnp.minimum(dlu, dld)
    lb2 = jnp.sqrt(dlu * dld)
    lb1 = lb1.at[n - 1].set(lb1[n - 2])
    lb2 = lb2.at[n - 1].set(lb2[n - 2])
    return lb1, lb2


def mym_length(xland, dz, dx, zw, rmol, fltv, u, v, qke, qpe, dtv, pblh, theta,
               psig_bl, edmf_w, edmf_a):
    """mym_length, CASE (1).  Returns el, qkw, qtw."""
    n = theta.shape[0]
    i = jnp.arange(n)
    up = i >= 1
    water = xland - _XLAND_WATER >= 0.0
    ugrid = _sqrt0(u[0] ** 2 + v[0] ** 2)
    uonset = _ML_UONSET
    wt_u1 = 1.0 - _ML_WTU1 * jnp.minimum(1.0, jnp.maximum(0.0, ugrid - uonset) / _ML_URANGE)
    wt_u2 = 1.0 - 0.5 * jnp.minimum(1.0, jnp.maximum(0.0, ugrid - uonset) / _ML_URANGE)
    wt_dx = 1.0 - jnp.minimum(1.0, (jnp.maximum(0.0, dx - _ML_DX_LO) / _ML_DX_RANGE))
    cns = _ML_CNS
    alp1 = _ML_ALP1 * wt_dx + (1.0 - wt_dx) * _ML_ALP1_COARSE
    alp2 = _ML_ALP2 * wt_dx + (1.0 - wt_dx) * _ML_ALP2_COARSE
    alp3 = _ML_ALP3 * wt_u2
    alp4 = jnp.where(water, _ML_ALP4_WATER * wt_u2, _ML_ALP4_LAND)
    alp5 = _ML_ALP5
    alp6 = _ML_ALP6
    pblh2 = jnp.maximum(pblh, _ML_PBLH_MIN)
    h1 = jnp.maximum(_ML_H1_FRAC * pblh2, _ML_H1_MIN)
    h1 = jnp.minimum(h1, _ML_H1_MAX)
    h2 = h1 * 0.5
    afk = dz / (dz + _prev(dz))
    abk = 1.0 - afk
    qkw = jnp.where(up, jnp.sqrt(jnp.maximum(qke * abk + _prev(qke) * afk, _QKEMIN)),
                    jnp.sqrt(jnp.maximum(qke, _QKEMIN)))
    qe = qke + qpe
    qtw = jnp.where(up, jnp.sqrt(jnp.maximum(qe * abk + _prev(qe) * afk, _QKEMIN)),
                    jnp.sqrt(jnp.maximum(qe, _QKEMIN)))
    qtke = jnp.where(up, jnp.maximum(0.5 * (qkw ** 2), _ML_QTKE_MIN),
                     jnp.maximum(0.5 * qke, 0.5 * _QKEMIN))
    thetaw = jnp.where(up, theta * abk + _prev(theta) * afk, theta)
    # elt / vsc integrals: k = kts+1 while zw(k) <= pblh2 + max(200, min(.3 pblh2, 600))
    zk = zw[:n]
    lim = pblh2 + jnp.maximum(_ML_ELT_DZ_MIN, jnp.minimum(_ML_H1_FRAC * pblh2, _ML_H1_MAX))
    m = up & (zk <= lim)
    dzk = 0.5 * (dz + _prev(dz))
    qdz = jnp.minimum(jnp.maximum(qkw, _ML_QKW_MIN), _ML_QKW_MAX) * dzk

    def f(c, x):
        e, s = c
        mk, qd, zz = x
        return (jnp.where(mk, e + qd * zz, e), jnp.where(mk, s + qd, s)), None

    one_e5 = jnp.asarray(_ML_ELT0, theta.dtype)
    (elt, vsc), _ = lax.scan(f, (one_e5, one_e5), (m, qdz, zk))
    elt_max = jnp.where(water, _ML_ELTMAX_WATER + 100. * jnp.minimum(1.0, jnp.maximum(0.0, ugrid - _ML_ELTMAX_U0) / _ML_ELTMAX_URANGE),
                        _ML_ELTMAX_LAND)
    elt = jnp.minimum(jnp.maximum(alp1 * elt / vsc, _ML_ELT_MIN), elt_max)
    vflx = fltv - _ML_VFLX_OFF
    vsc = _pow0(_GTR * elt * jnp.maximum(vflx, 0.0), _P333)
    elblmin, elblavg = boulac_length(zw, dz, qtke, thetaw)
    qkw_mf = jnp.maximum((0.5 * ((edmf_a + _prev(edmf_a)))) * (0.5 * ((edmf_w + _prev(edmf_w)))),
                         0.0)
    cpblh = jnp.minimum((zk + 0.25 * pblh2) / pblh2, 1.0)
    qkw_mf = cpblh * qkw_mf
    alp3z = cpblh * alp3
    pos = dtv > 0.0
    bv = jnp.maximum(jnp.sqrt(_GTR * _sel(pos, dtv, 1.0)), _ML_BV_MIN)
    elb_mf = alp6 * qkw_mf / bv
    elb_mf = elb_mf / (1.0 + (elb_mf / 100.))
    elb = alp2 * (jnp.maximum(qkw, _ML_QKW_ELB_MIN)) / bv * (1.0 + alp3z * _sqrt0(vsc / (bv * elt)))
    elb = jnp.maximum(elb, elb_mf)
    elb = jnp.minimum(elb, zk)
    elf = 1.0 * jnp.maximum(qkw, _ML_QKW_ELB_MIN) / bv
    elf = jnp.maximum(elf, elb_mf)
    elb = jnp.where(pos, elb, _ML_ELB_INF)
    elf = jnp.where(pos, elf, elb)
    elblavg = jnp.where(pos, jnp.maximum(alp5 * elblavg, elb_mf), _ML_ELBL_STABLE * elblavg)
    z_m = jnp.maximum(_KARMAN, zk - 4.0)
    stab = rmol > 0.0
    rp = _sel(stab, rmol, 0.0)
    rn = _sel(stab, 0.0, rmol)
    els_s = _KARMAN * zk / (1.0 + cns * jnp.minimum(zk * rp, _ZMAX))
    els1_s = _KARMAN * z_m / (1.0 + cns * jnp.minimum(zk * rp, _ZMAX))
    els_u = _KARMAN * zk * (1.0 - alp4 * zk * rn) ** _ML_ELS_EXP
    els1_u = _KARMAN * z_m * (1.0 - alp4 * zk * rn) ** _ML_ELS_EXP
    els = jnp.where(stab, els_s, els_u)
    els1 = jnp.where(stab, els1_s, els1_u)
    wt = 0.5 * jnp.tanh((zk - (pblh2 + h1)) / h2) + 0.5
    el_unstab = els / (1.0 + (els1 / elt))
    el = jnp.minimum(el_unstab, elb)
    el = jnp.minimum(el, elf)
    el = jnp.where(water, el * wt_u1, el)
    el = el * (1.0 - wt) + elblavg * wt
    el_les = 0.25 * 0.5 * (dz + _prev(dz))
    el = el * psig_bl + (1.0 - psig_bl) * jnp.minimum(el_les, el)
    el = jnp.where(up, el, 0.0)
    return el, qkw, qtw


# ---------------------------------------------------------------------------
# mym_turbulence (closure 2.5)
# ---------------------------------------------------------------------------
class Turb(NamedTuple):
    el: jax.Array
    sh: jax.Array
    sm: jax.Array
    dfm: jax.Array
    dfh: jax.Array
    dfq: jax.Array
    pdk: jax.Array
    pdt: jax.Array
    pdq: jax.Array
    pdc: jax.Array
    tcd: jax.Array
    qcd: jax.Array


def mym_turbulence(xland, dz, dx, zw, p, exner, u, v, thl, thv, thlv, qw, qke,
                   rmol, fltv, pblh, theta, psig_bl, cldfra, edmf_w, edmf_a,
                   tkeprod_up):
    n = theta.shape[0]
    i = jnp.arange(n)
    up = i >= 1
    qpe = jnp.zeros_like(qke)                          # closure < 2.7
    dtl, dqw, dtv, gm, gh, sm, sh = mym_level2(
        zw, dz, xland, pblh, u, v, thl, thv, thlv, theta, p, exner, qke, cldfra,
        edmf_a, edmf_w, qw)
    el, qkw, qtw = mym_length(xland, dz, dx, zw, rmol, fltv, u, v, qke, qpe, dtv,
                              pblh, theta, psig_bl, edmf_w, edmf_a)
    dzk = 0.5 * (dz + _prev(dz))
    elsq = el ** 2
    q3sq = qkw ** 2
    q2sq = _B1 * elsq * (sm * gm + sh * gh)
    sh = jnp.maximum(sh, _SH_MIN)
    duz = (u - _prev(u)) ** 2 + (v - _prev(v)) ** 2
    duz = duz / dzk ** 2
    ri = -gh / jnp.maximum(duz, 1.0e-10)
    a2fac = 1.0 / (1.0 + _CKMOD * jnp.maximum(ri, 0.0))
    b2 = (ri >= _PR_RI_LO) & (ri <= 1.0)
    rs = _sel(b2, ri, 1.0)
    prlim = jnp.where(ri >= 1.0, jnp.minimum(_PR_KONDO_A * ri, _PRLIMIT_FRE),
                      jnp.where(b2, jnp.minimum(_PR_KONDO_B * rs + 1.0 / (_PR_KONDO_B * rs), _PRLIMIT_FRE),
                                _PRLIMIT_FRE))
    wt = jnp.minimum(1.0, zw[:n] / jnp.maximum(pblh, _PR_PBLH_MIN))
    prlim = wt * prlim + (1.0 - wt) * jnp.minimum(_PRLIMIT_SFC, prlim)
    gmel = gm * elsq
    ghel = gh * elsq
    dlsq = elsq
    q3sq = jnp.where(q3sq / _sel(up, dlsq, 1.0) < -gh, -dlsq * gh, q3sq)
    hl = q3sq < q2sq
    qdiv = jnp.sqrt(q3sq / _sel(hl, q2sq, 1.0))
    e1 = q3sq - _E1C * ghel * a2fac
    e2 = q3sq - _E2C * ghel * a2fac
    e3 = e1 + _E3C * ghel * a2fac ** 2
    e4 = e1 - _E4C * ghel * a2fac
    eden = jnp.maximum(e2 * e4 + e3 * _E5C * gmel, 1.0e-20)
    # sm25/sh25 are selected only where up & ~hl; elsewhere divide by 1 so the
    # derivative of an unselected quotient cannot overflow (1/eden**2, fp32)
    eden = _sel(up & ~hl, eden, 1.0)
    sm25 = q3sq * _A1 * (e3 - 3.0 * _C1 * e4) / eden
    sh25 = q3sq * (_A2 * a2fac) * (e2 + 3.0 * _C1 * _E5C * gmel) / eden
    sh = jnp.where(hl, sh * qdiv, sh25)
    sm = jnp.where(hl, sm * qdiv, sm25)
    qdiv = jnp.where(hl, qdiv, 1.0)
    sh = jnp.where(sh > 4.0, 4.0, sh)
    sh = jnp.where(sh < 0.0, 0.0, sh)
    shb = jnp.maximum(sh, _SHB_MIN)
    sm = jnp.minimum(sm, prlim * shb)
    cldavg = 0.5 * (_prev(cldfra) + cldfra)
    mfmax = jnp.maximum(0.5 * (_prev(edmf_a) + edmf_a) * 0.5 * (_prev(edmf_w) + edmf_w), 0.0)
    sm = jnp.maximum(sm, _S_MIN_MF * jnp.minimum(10. * mfmax, 1.0))
    sh = jnp.maximum(sh, _S_MIN_MF * jnp.minimum(10. * mfmax, 1.0))
    sm = jnp.maximum(sm, _S_MIN_MF * jnp.minimum(cldavg, 0.5))
    sh = jnp.maximum(sh, _S_MIN_MF * jnp.minimum(cldavg, 0.5))
    elq = el * qkw
    elqt = el * qtw
    elh = elq * qdiv
    gamt = gamq = gamv = 0.0
    pdk = elq * (sm * gm + sh * gh + gamv) + 0.5 * 0.0 + 0.5 * tkeprod_up
    pdt = elh * (sh * dtl + gamt) * dtl
    pdq = elh * (sh * dqw + gamq) * dqw
    pdc = elh * (sh * dtl + gamt) * dqw * 0.5 + elh * (sh * dqw + gamq) * dtl * 0.5
    dfm = elqt * sm / dzk
    dfh = elq * sh / dzk
    z = lambda x: jnp.where(up, x, 0.0)
    zero = jnp.zeros_like(qke)
    # tcd = qcd = 0 at level 2.5 (their vertical differences are zero)
    return Turb(el, z(sh), z(sm), z(dfm), z(dfh), z(dfm), z(pdk), z(pdt), z(pdq),
                z(pdc), zero, zero)


# ---------------------------------------------------------------------------
# mym_predict (closure 2.5, bl_mynn_edmf = 1, bl_mynn_edmf_tke = 0)
# ---------------------------------------------------------------------------
def mym_predict(delt, dz, ust, pmz, el, dfq, rho, pdk, pdt, pdq, pdc, qke, s_aw):
    """Returns qke, tsq, qsq, cov at the new time level."""
    n = qke.shape[0]
    i = jnp.arange(n)
    onoff = 0.0
    vkz = _KARMAN * 0.5 * dz[0]
    qkw = jnp.sqrt(jnp.maximum(qke, _QKEMIN))
    df3q = _SQFAC * dfq
    dtz = delt / dz
    rhoz = jnp.where(i == 0, rho,
                     jnp.maximum((rho * _prev(dz) + _prev(rho) * dz) / (_prev(dz) + dz), _RHO_MIN))
    rhoinv = jnp.where(i == 0, 1.0 / rho, 1.0 / jnp.maximum(rho, _RHO_MIN))
    kqdz = jnp.concatenate([rhoz * df3q, (rhoz[-1] * df3q[-1])[None]])
    kmdz = jnp.concatenate([rhoz * dfq, (rhoz[-1] * dfq[-1])[None]])
    mid = (jnp.arange(n + 1) >= 1) & (jnp.arange(n + 1) <= n - 2)
    s_awp = _next(s_aw)
    kqdz = jnp.where(mid, jnp.maximum(kqdz, 0.5 * s_aw), kqdz)
    kqdz = jnp.where(mid, jnp.maximum(kqdz, -0.5 * (s_aw - s_awp)), kqdz)
    kmdz = jnp.where(mid, jnp.maximum(kmdz, 0.5 * s_aw), kmdz)
    kmdz = jnp.where(mid, jnp.maximum(kmdz, -0.5 * (s_aw - s_awp)), kmdz)
    pdk1 = 2.0 * ust ** 3 * pmz / (vkz)
    pdk = pdk.at[0].set(pdk1 - pdk[1])
    pdt = pdt.at[0].set(pdt[1])
    pdq = pdq.at[0].set(pdq[1])
    pdc = pdc.at[0].set(pdc[1])
    b1l = _B1 * 0.5 * (_next(el) + el)
    bp = jnp.where(i <= n - 2, 2.0 * qkw / b1l, 0.0)
    rp = _next(pdk) + pdk
    kq, kq1 = kqdz[:n], kqdz[1:]
    sa, sa1 = s_aw[:n], s_aw[1:]
    a = -dtz * kq * rhoinv + 0.5 * dtz * rhoinv * sa * onoff
    b = 1. + dtz * (kq + kq1) * rhoinv + 0.5 * dtz * rhoinv * (sa - sa1) * onoff + bp * delt
    c = -dtz * kq1 * rhoinv - 0.5 * dtz * rhoinv * sa1 * onoff
    d = rp * delt + qke + dtz * rhoinv * (0.0 - 0.0) * onoff
    top = i == n - 1
    a = jnp.where(top, 0., a)
    b = jnp.where(top, 1., b)
    c = jnp.where(top, 0., c)
    d = jnp.where(top, qke[n - 1], d)
    x = tridiag2(a, b, c, d)
    qke_new = jnp.minimum(jnp.maximum(x, _QKEMIN), _QKE_MAX)
    # level 2.5: q'^2, T'^2 and T'q' from the level-2 diagnostic
    b2l = _B2 * 0.25 * (_next(el) + el) / qkw
    qsq = jnp.minimum(_QSQ_MAX, jnp.maximum(1e-17, b2l * (_next(pdq) + pdq)))
    tsq = jnp.minimum(3.0, jnp.maximum(0.0, b2l * (_next(pdt) + pdt)))
    cov = b2l * (_next(pdc) + pdc)
    qsq = jnp.where(top, qsq[n - 2], qsq)
    tsq = jnp.where(top, tsq[n - 2], tsq)
    cov = jnp.where(top, cov[n - 2], cov)
    return qke_new, tsq, qsq, cov


# ---------------------------------------------------------------------------
# mym_initialize (cold start, INITIALIZE_QKE = .true.)
# ---------------------------------------------------------------------------
def mym_initialize(xland, pblh, dz, dx, zw, p, exner, u, v, thl, qw, theta,
                   thv, thlv, ust, psig_bl, qke):
    """mym_initialize with rmol = 0, zero clouds and mass flux.  Returns
    (sh, sm, el, qke, tsq, qsq, cov)."""
    n = theta.shape[0]
    i = jnp.arange(n)
    zeros = jnp.zeros(n, theta.dtype)
    pmz = phh = 1.
    flt = flq = 0.
    ustm = jnp.maximum(_USTM_MIN, ust)
    dtl, dqw, dtv, gm, gh, sm, sh = mym_level2(
        zw, dz, xland, pblh, u, v, thl, thv, thlv, theta, p, exner, qke, zeros,
        zeros, zeros, qw)
    qke1 = _INIT_QKE_FAC * ustm ** 2 * (_B1 * pmz) ** _P666
    qke = jnp.where(i == 0, qke1,
                    qke1 * jnp.maximum((ust * _INIT_QKE_H - zw[:n]) / (ustm * _INIT_QKE_H), _INIT_QKE_MIN))
    for _ in range(5):
        el, qkw, _ = mym_length(xland, dz, dx, zw, 0.0, 0.0, u, v, qke, zeros, dtv,
                                pblh, theta, psig_bl, zeros, zeros)
        elq = el * qkw
        pdk = elq * (sm * gm + sh * gh)
        pdt = elq * sh * dtl ** 2
        pdq = elq * sh * dqw ** 2
        pdc = elq * sh * dtl * dqw
        vkz = _KARMAN * 0.5 * dz[0]
        elv = 0.5 * (el[1] + el[0]) / vkz
        qke0 = 1.0 * ustm ** 2 * (_B1 * pmz * elv) ** _P666
        phm = phh * _B2 / (_B1 * pmz / elv ** 2) ** _P333
        tsq0 = jnp.minimum(3.0, phm * (max(0.0, flt) / ustm) ** 2)
        qsq0 = jnp.minimum(_INIT_QSQ_MAX, phm * (max(0.0, flq) / ustm) ** 2)
        cov0 = phm * (flt / ustm) * (flq / ustm)
        b1l = _B1 * 0.25 * (_next(el) + el)
        tmpq = jnp.minimum(jnp.maximum(b1l * (_next(pdk) + pdk), _QKEMIN), _INIT_QKE_MAX)
        qkem = tmpq ** _P666
        b2l = jnp.where(qkem <= 0.0, 0.0, _B2 * (b1l / _B1) / jnp.sqrt(_sel(qkem > 0.0, qkem, 1.0)))
        tsqm = jnp.minimum(3.0, jnp.maximum(0.0, b2l * (_next(pdt) + pdt)))
        qsqm = jnp.minimum(_INIT_QSQ_MAX, jnp.maximum(0.0, b2l * (_next(pdq) + pdq)))
        covm = b2l * (_next(pdc) + pdc)
        mid = (i >= 1) & (i <= n - 2)
        qke = jnp.where(i == 0, qke0, jnp.where(mid, qkem, qke))
        tsq = jnp.where(i == 0, tsq0, jnp.where(mid, tsqm, 0.0))
        qsq = jnp.where(i == 0, qsq0, jnp.where(mid, qsqm, 0.0))
        cov = jnp.where(i == 0, cov0, jnp.where(mid, covm, 0.0))
    qke = qke.at[0].set(0.5 * (qke[0] + qke[1]))
    qke = qke.at[n - 1].set(qke[n - 2])
    tsq = tsq.at[n - 1].set(tsq[n - 2])
    qsq = qsq.at[n - 1].set(qsq[n - 2])
    cov = cov.at[n - 1].set(cov[n - 2])
    return sh, sm, el, qke, tsq, qsq, cov


# ---------------------------------------------------------------------------
# moisture_check, mynn_tendencies (implicit EDMF, cloudmix = 1, mixqt = 0)
# ---------------------------------------------------------------------------
def moisture_check(delt, dp, exner, qv, qc, qi, th, dqv, dqc, dqi):
    """moisture_check (qs = 0).  Returns (qv, qc, qi, th, dqv, dqc, dqi)."""
    n = qv.shape[0]
    qvmin = 1e-20

    def body(carry, k):
        pq, pd = carry                         # pending qv / dqv change for level k
        qvk = qv[k] + pq
        dqvk = dqv[k] + pd
        dqc2 = jnp.maximum(0.0, 0.0 - qc[k])
        dqi2 = jnp.maximum(0.0, 0.0 - qi[k])
        dqs2 = jnp.maximum(0.0, 0.0 - 0.0)
        dqck = dqc[k] + dqc2 / delt
        dqik = dqi[k] + dqi2 / delt
        dqvk = dqvk - (dqc2 + dqi2 + dqs2) / delt
        qck = qc[k] + dqc2
        qik = qi[k] + dqi2
        qvk = qvk - dqc2 - dqi2 - dqs2
        thk = th[k] + _XLVCP / exner[k] * dqc2 + _XLSCP / exner[k] * (dqi2 + dqs2)
        dqv2 = jnp.maximum(0.0, qvmin - qvk)
        dqvk = dqvk + dqv2 / delt
        qvk = qvk + dqv2
        notbot = k != 0
        dpkm1 = dp[jnp.maximum(k - 1, 0)]
        pq2 = jnp.where(notbot, -(dqv2 * dp[k] / dpkm1), 0.0)
        pd2 = jnp.where(notbot, -(dqv2 * dp[k] / dpkm1 / delt), 0.0)
        qvk = jnp.maximum(qvk, qvmin)
        qck = jnp.maximum(qck, 0.0)
        qik = jnp.maximum(qik, 0.0)
        return (pq2, pd2), (qvk, qck, qik, thk, dqvk, dqck, dqik, dqv2)

    z = jnp.zeros((), qv.dtype)
    _, (qv, qc, qi, th, dqv, dqc, dqi, dqv2s) = lax.scan(
        body, (z, z), jnp.arange(n - 1, -1, -1))
    rev = lambda x: x[::-1]
    qv, qc, qi, th, dqv, dqc, dqi = map(rev, (qv, qc, qi, th, dqv, dqc, dqi))
    dqv2 = dqv2s[-1]                           # value left by k = 1
    pos = qv > 2.0 * qvmin
    tot = _seq_sum(jnp.where(pos, qv * dp, 0.0))
    aa = dqv2 * dp[0] / jnp.maximum(1.e-20, tot)
    fix = (dqv2 > 1.e-20) & (aa < 0.5) & pos
    dum = aa * qv
    qv = jnp.where(fix, qv - dum, qv)
    dqv = jnp.where(fix, dqv - dum / delt, dqv)
    return qv, qc, qi, th, dqv, dqc, dqi


def mynn_tendencies(delt, dz, rho, u, v, th, tk, qv_mr, psfc, exner, thl, sqv, sqc,
                    sqi, ust, flt, flqv, wspd, uoce, voce, dfm, dfh, tcd, qcd,
                    diss_heat, mf):
    """mynn_tendencies on the ported path; returns du, dv, dth, dqv, dqc, dqi."""
    n = dz.shape[0]
    i = jnp.arange(n)
    top = i == n - 1
    bot = i == 0
    flqc = 0.0
    onoff = 1.0                                         # bl_mynn_edmf_mom = 1
    ustovwsp = jnp.where(wspd <= 1e-6, 0.0, ust / _sel(wspd > 1e-6, wspd, 1.0))
    rhosfc = psfc / (_R_D * (tk[0] + _P608 * qv_mr[0]))
    dtz = delt / dz
    rhoz = jnp.where(bot, rho,
                     jnp.maximum((rho * _prev(dz) + _prev(rho) * dz) / (_prev(dz) + dz), _RHO_MIN))
    rhoinv = jnp.where(bot, 1.0 / rho, 1.0 / jnp.maximum(rho, _RHO_MIN))
    khdz = jnp.concatenate([rhoz * dfh, (rhoz[-1] * dfh[-1])[None]])
    kmdz = jnp.concatenate([rhoz * dfm, (rhoz[-1] * dfm[-1])[None]])
    delp = rho * _GRAV * dz
    delp = delp.at[0].set(jnp.where(delp[0] < _P333 * delp[1], _P333 * delp[1], delp[0]))
    s_aw = mf.s_aw
    s_awp = _next(s_aw)
    mid = (jnp.arange(n + 1) >= 1) & (jnp.arange(n + 1) <= n - 2)
    khdz = jnp.where(mid, jnp.maximum(khdz, 0.5 * (s_aw + 0.0)), khdz)
    khdz = jnp.where(mid, jnp.maximum(khdz, -0.5 * (s_aw - s_awp) - 0.5 * (0.0 - 0.0)), khdz)
    kmdz = jnp.where(mid, jnp.maximum(kmdz, 0.5 * (s_aw + 0.0)), kmdz)
    kmdz = jnp.where(mid, jnp.maximum(kmdz, -0.5 * (s_aw - s_awp) - 0.5 * (0.0 - 0.0)), kmdz)
    sa, sa1 = s_aw[:n], s_aw[1:]

    def solve(kdz, d_bot, d_mid, phi, mfon, b_extra):
        """Implicit EDMF system: k = kts row, interior rows, prescribed top."""
        k0, k1 = kdz[:n], kdz[1:]
        a = jnp.where(bot, -dtz * k0 * rhoinv,
                      -dtz * k0 * rhoinv + 0.5 * dtz * rhoinv * sa * mfon)
        b = jnp.where(bot, 1. + dtz * (k1 + b_extra) * rhoinv - 0.5 * dtz * rhoinv * sa1 * mfon,
                      1. + dtz * (k0 + k1) * rhoinv + 0.5 * dtz * rhoinv * (sa - sa1) * mfon)
        c = -dtz * k1 * rhoinv - 0.5 * dtz * rhoinv * sa1 * mfon
        d = jnp.where(bot, d_bot, d_mid)
        a = jnp.where(top, 0.0, a)
        b = jnp.where(top, 1.0, b)
        c = jnp.where(top, 0.0, c)
        d = jnp.where(top, phi[n - 1], d)
        return tridiag2(a, b, c, d)

    def mfdiv(s):          # dtz*rhoinv*(s(k)-s(k+1))*onoff ; kts row: -dtz*rhoinv*s(k+1)*onoff
        return dtz * rhoinv * (s[:n] - s[1:])

    def mfbot(s):
        return dtz * rhoinv * s[1:]

    # u, v (kts row: surface drag in b; v uses ust**2/wspd, as written)
    du_bot = u + dtz * uoce * ust ** 2 / wspd - mfbot(mf.s_awu) * onoff
    x = solve(kmdz, du_bot, u + mfdiv(mf.s_awu) * onoff, u, onoff, rhosfc * ust * ustovwsp)
    du = (x - u) / delt
    dv_bot = v + dtz * voce * ust ** 2 / wspd - mfbot(mf.s_awv) * onoff
    x = solve(kmdz, dv_bot, v + mfdiv(mf.s_awv) * onoff, v, onoff, rhosfc * ust ** 2 / wspd)
    dv = (x - v) / delt
    # theta_l (b at kts: khdz(k+1)+khdz(k), khdz(kts) = 0)
    d_bot = (thl + dtz * rhosfc * flt * rhoinv + tcd * delt - mfbot(mf.s_awthl)
             + diss_heat * delt)
    d_mid = thl + tcd * delt + mfdiv(mf.s_awthl) + diss_heat * delt
    thl_new = solve(khdz, d_bot, d_mid, thl, 1.0, khdz[0])
    # cloud water
    d_bot = sqc + dtz * rhosfc * flqc * rhoinv + qcd * delt - mfbot(mf.s_awqc)
    d_mid = sqc + qcd * delt + mfdiv(mf.s_awqc)
    sqc2 = solve(khdz, d_bot, d_mid, sqc, 1.0, khdz[0])
    # water vapour (limit large negative surface fluxes)
    qvflux = jnp.where(flqv < 0.0,
                       jnp.maximum(flqv, (jnp.minimum(_QVFLUX_FRAC * sqv[0] - 1e-8, 0.0) / dtz[0])),
                       flqv)
    d_bot = sqv + dtz * rhosfc * qvflux * rhoinv + qcd * delt - mfbot(mf.s_awqv)
    d_mid = sqv + qcd * delt + mfdiv(mf.s_awqv)
    sqv2 = solve(khdz, d_bot, d_mid, sqv, 1.0, khdz[0])
    # cloud ice (no updraft transport)
    sqi2 = solve(khdz, sqi, sqi, sqi, 0.0, khdz[0])
    dqv = (sqv2 - sqv) / delt
    dqc = (sqc2 - sqc) / delt
    dqi = (sqi2 - sqi) / delt
    sqv2, sqc2, sqi2, thl_new, dqv, dqc, dqi = moisture_check(
        delt, delp, exner, sqv2, sqc2, sqi2, thl_new, dqv, dqc, dqi)
    dth = (thl_new + _XLVCP / exner * sqc + _XLSCP / exner * (sqi) - th) / delt
    return du, dv, dth, dqv, dqc, dqi


# ---------------------------------------------------------------------------
# mynnedmf: one column, one time step
# ---------------------------------------------------------------------------
class MYNNOutput(NamedTuple):
    """Outputs of :func:`mynnedmf_column` (bottom-up level arrays).

    Tendencies ``du, dv, dth, dqv, dqc, dqi`` (specific contents); the new
    turbulence state ``qke, tsq, qsq, cov, el, sh, sm`` (qke, qsq and el are
    what the next step reads); ``km, kh`` [m^2/s] at the interface below each
    level (index 0 is the surface, 0); subgrid clouds; mean-updraft profiles;
    ``pblh`` [m] and ``kpbl`` (the oracle's 1-based level index); mass-flux
    diagnostics."""
    du: jax.Array
    dv: jax.Array
    dth: jax.Array
    dqv: jax.Array
    dqc: jax.Array
    dqi: jax.Array
    qke: jax.Array
    tsq: jax.Array
    qsq: jax.Array
    cov: jax.Array
    el: jax.Array
    sh: jax.Array
    sm: jax.Array
    km: jax.Array
    kh: jax.Array
    cldfra_bl: jax.Array
    qc_bl: jax.Array
    qi_bl: jax.Array
    edmf_a: jax.Array
    edmf_w: jax.Array
    edmf_qt: jax.Array
    edmf_thl: jax.Array
    edmf_ent: jax.Array
    edmf_qc: jax.Array
    pblh: jax.Array
    kpbl: jax.Array
    maxmf: jax.Array
    maxwidth: jax.Array
    ztop_plume: jax.Array
    excess_h: jax.Array
    excess_q: jax.Array


def _column(dz, u, v, w, th, tk, p, exner, rho, sqv, sqc, sqi, qke, qsq, el,
            xland, ps, ts, ust, hfx, qfx, wspd, uoce, voce, dt, dx, initflag,
            want_inter):
    n = th.shape[0]
    zw = jnp.concatenate([jnp.zeros(1, th.dtype), _cumsum_seq(dz)])
    sqw = sqv + sqc + sqi
    thl = th - _XLVCP / exner * sqc - _XLSCP / exner * (sqi)
    thv = th * (1.0 + _P608 * sqv - (sqc + sqi))
    inter = {}
    if initflag:
        thlv0 = thl * (1.0 + _P608 * sqv)
        qke = _INIT_QKE_A * ust * jnp.maximum((ust * _INIT_QKE_H - zw[:n]) / (jnp.maximum(ust, _INIT_QKE_MIN) * _INIT_QKE_H), _INIT_QKE_MIN)
        pblh0, _ = get_pblh(thv, ust, zw, dz, xland)
        psig_bl0, _ = scale_aware(dx, pblh0)
        # mynnedmf passes sqv1 (not sqw1) as mym_initialize's qw
        _, _, el, qke, _, qsq, _ = mym_initialize(
            xland, pblh0, dz, dx, zw, p, exner, u, v, thl, sqv, th, thv, thlv0,
            ust, psig_bl0, qke)
    qv_mr = sqv / (1.0 - sqv)
    pblh, kpbl = get_pblh(thv, ust, zw, dz, xland)
    psig_bl, psig_shcu = scale_aware(dx, pblh)
    cpm = _CP * (1.0 + _CPM_QV * jnp.maximum(sqv[0], 1e-8))
    flqv = qfx / rho[0]
    flqc = 0.0
    th_sfc = ts / exner[0]
    flq = flqv + flqc
    flt = hfx / (rho[0] * cpm) - _XLVCP * flqc / exner[0]
    fltv = flt + flqv * _P608 * th_sfc
    rmol = -_KARMAN * _GTR * fltv / jnp.maximum(ust ** 3, 1.0e-6)
    zet = 0.5 * dz[0] * rmol
    zet = jnp.maximum(zet, -10.)
    zet = jnp.minimum(zet, 10.)
    stab = zet >= 0.0
    zu = _sel(stab, -1.0, zet)
    water = xland - _XLAND_WATER >= 0.0
    pmz = jnp.where(water,
                    jnp.where(stab, 1.0 + 3.0 * zet, 1.0 / (1.0 - _CPHM_UNST * zu) ** 0.25 - zet),
                    jnp.where(stab, _PMZ_LAND_A + _PMZ_LAND_B * zet, _PMZ_LAND_A / (1.0 - _CPHM_UNST * zu) ** 0.25 - zet))
    phh = jnp.where(stab, 1.0 + _CPHH_ST * zet, 1.0 / jnp.sqrt(1.0 - _CPHH_UNST * zu))
    vt, vq, cldfra, qc_bl, qi_bl, sgm = mym_condensation(
        dz, zw, xland, thl, sqw, sqc, sqi, p, exner, qsq, th, pblh)
    if want_inter:
        inter.update(pblh=pblh, kpbl=kpbl, psig_bl=psig_bl, psig_shcu=psig_shcu,
                     flt=flt, fltv=fltv, flq=flq, rmol=rmol, pmz=pmz, phh=phh,
                     vt_cond=vt, vq_cond=vq, cldfra_cond=cldfra, qc_bl_cond=qc_bl,
                     qi_bl_cond=qi_bl, sgm=sgm)
    mf = dmp_mf(zw, dz, p, rho, u, v, w, thl, thv, tk, sqw, sqv, sqc, qke, qsq,
                exner, vt, vq, ust, flt, fltv, flq, pblh, kpbl, dx, xland, th_sfc,
                qc_bl, cldfra, psig_shcu, el)
    cldfra, qc_bl = mf.cldfra, mf.qc_bl
    if want_inter:
        inter.update(s_aw=mf.s_aw, s_awthl=mf.s_awthl, s_awqt=mf.s_awqt, s_awqv=mf.s_awqv,
                     s_awqc=mf.s_awqc, s_awu=mf.s_awu, s_awv=mf.s_awv,
                     edmf_a_mf=mf.edmf_a, edmf_w_mf=mf.edmf_w, edmf_qt_mf=mf.edmf_qt,
                     edmf_thl_mf=mf.edmf_thl, edmf_ent_mf=mf.edmf_ent,
                     edmf_qc_mf=mf.edmf_qc, tkeprod_up=mf.tkeprod_up,
                     cldfra_mf=cldfra, qc_bl_mf=qc_bl, vt_mf=mf.vt, vq_mf=mf.vq,
                     maxmf=mf.maxmf, ztop_plume=mf.ztop, maxwidth=mf.maxwidth,
                     excess_h=mf.excess_h, excess_q=mf.excess_q)
    qc_tot = jnp.maximum(qc_bl, sqc)
    qi_tot = jnp.maximum(qi_bl, sqi + 0.0)
    thlv = (th - _XLVCP / exner * qc_tot - _XLSCP / exner * qi_tot) * (1.0 + _P608 * sqv)
    thl_tot = th - _XLVCP / exner * qc_tot - _XLSCP / exner * qi_tot
    thv = th * (1.0 + _P608 * sqv - (qc_tot + qi_tot))
    tb = mym_turbulence(xland, dz, dx, zw, p, exner, u, v, thl, thv, thlv, sqw, qke,
                        rmol, fltv, pblh, th, psig_bl, cldfra, mf.edmf_w, mf.edmf_a,
                        mf.tkeprod_up)
    if want_inter:
        inter.update(thlv_tot=thlv, thv_tot=thv, thl_tot=thl_tot, qc_tot=qc_tot,
                     qi_tot=qi_tot, el_turb=tb.el, sh_turb=tb.sh, sm_turb=tb.sm,
                     dfm=tb.dfm, dfh=tb.dfh, dfq=tb.dfq, pdk=tb.pdk, pdt=tb.pdt,
                     pdq=tb.pdq, pdc=tb.pdc, tcd=tb.tcd, qcd=tb.qcd)
    qke_new, tsq, qsq_new, cov = mym_predict(dt, dz, ust, pmz, tb.el, tb.dfq, rho,
                                             tb.pdk, tb.pdt, tb.pdq, tb.pdc, qke, mf.s_aw)
    el_new = tb.el
    i = jnp.arange(n)
    diss_heat = jnp.minimum(jnp.maximum((qke_new ** 1.5) / (
        _B1 * jnp.maximum(0.5 * (el_new + _next(el_new)), 1.0)) / _CP, 0.0), _DISS_MAX)
    diss_heat = diss_heat * jnp.exp(-_DISS_P_SCALE / jnp.maximum(p, 1.0))
    diss_heat = jnp.where(i == n - 1, 0.0, diss_heat)
    if want_inter:
        inter.update(qke_pred=qke_new, tsq_pred=tsq, qsq_pred=qsq_new, cov_pred=cov,
                     diss_heat=diss_heat)
    du, dv, dth, dqv, dqc, dqi = mynn_tendencies(
        dt, dz, rho, u, v, th, tk, qv_mr, ps, exner, thl, sqv, sqc, sqi, ust, flt,
        flqv, wspd, uoce, voce, tb.dfm, tb.dfh, tb.tcd, tb.qcd, diss_heat, mf)
    dzk = 0.5 * (dz + _prev(dz))
    km = jnp.where(i == 0, 0.0, tb.dfm * dzk)
    kh = jnp.where(i == 0, 0.0, tb.dfh * dzk)
    out = MYNNOutput(du, dv, dth, dqv, dqc, dqi, qke_new, tsq, qsq_new, cov, el_new,
                     tb.sh, tb.sm, km, kh, cldfra, qc_bl, qi_bl, mf.edmf_a, mf.edmf_w,
                     mf.edmf_qt, mf.edmf_thl, mf.edmf_ent, mf.edmf_qc, pblh, kpbl,
                     mf.maxmf, mf.maxwidth, mf.ztop, mf.excess_h, mf.excess_q)
    return out, inter


def mynnedmf_column(dz, u, v, w, th, tk, p, exner, rho, sqv, sqc, sqi,
                    qke, qsq, el, xland, ps, ts, ust, hfx, qfx, wspd, dt, dx, *,
                    initflag=False, uoce=None, voce=None, return_intermediates=False):
    """One MYNN-EDMF step for a batch of columns (see module docstring).

    Level arrays are (ncol, nlev) BOTTOM-UP; ``xland`` (1 land, 2 water),
    ``ps`` [Pa], ``ts`` skin temperature [K], ``ust`` [m/s], ``hfx`` [W/m^2],
    ``qfx`` [kg/m^2/s], ``wspd`` [m/s], ``dx`` [m] are (ncol,); ``dt`` [s]
    is a scalar or (ncol,).  ``qke``, ``qsq``, ``el`` are the turbulence state
    carried from the previous step (ignored when ``initflag``: the oracle's
    cold start, mym_initialize, then builds them).  ``uoce``/``voce`` (ocean
    surface current) default to zero.  Returns :class:`MYNNOutput` (and the
    oracle's intermediate fields when ``return_intermediates``).

    Input domain is the oracle's: finite inputs, ``qke >= 0``, ``qsq >= 0``
    and ``wspd > 0`` (the oracle's surface drag divides by ``wspd``; at
    ``wspd = 0`` it produces 0/0 and so does this port).  A negative surface
    moisture flux is dropped by the oracle (mynn_tendencies' ``qvflux``
    limiter), and the port reproduces that.
    """
    ncol = th.shape[0]
    dtype = th.dtype
    z1 = jnp.zeros(ncol, dtype)
    uoce = z1 if uoce is None else uoce
    voce = z1 if voce is None else voce
    dt = jnp.broadcast_to(jnp.asarray(dt, dtype), (ncol,))
    dx = jnp.broadcast_to(jnp.asarray(dx, dtype), (ncol,))

    def f(*a):
        return _column(*a, bool(initflag), bool(return_intermediates))

    out, inter = jax.vmap(f)(dz, u, v, w, th, tk, p, exner, rho, sqv, sqc, sqi, qke,
                             qsq, el, xland, ps, ts, ust, hfx, qfx, wspd, uoce, voce,
                             dt, dx)
    return (out, inter) if return_intermediates else out
