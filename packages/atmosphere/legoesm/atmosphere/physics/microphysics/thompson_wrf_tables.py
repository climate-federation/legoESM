"""WRF v4.8.0 Thompson microphysics: constants, init values and lookup tables.

Faithful NumPy port of the ``thompson_init`` half of WRF
``phys/module_mp_thompson.F`` (tag v4.8.0, commit
06d4240ae989cc3e50af412bb472df3d9048783c) for ``mp_physics=8`` (neither
aerosol- nor hail-aware): the module PARAMETERs, the gamma-function
constants ``cce/ccg/cie/cig/cre/crg/cse/csg/cge/cgg``, the diameter bins and
the lookup tables the column scheme reads
(``qr_acr_qg``, ``qr_acr_qs``, ``freezeH2O``, ``qi_aut_qs``, ``table_Efrw``,
``table_Efsw``).  Names follow the oracle's variables.

The tables are built in memory once per process (WRF builds them in memory at
init when no ``*.dat`` cache exists; this port has no file cache).  Each bin
double sum is evaluated as matrix products of the oracle's own bin arrays,
so entries agree with the Fortran loops to summation-order round-off.
Two tables are evaluated only where ``mp_physics=8`` can read them:
``qr_acr_qg`` for the one graupel density ``idx_bg1`` (``dimNRHG=1``, exactly
as WRF allocates it) and ``freezeH2O`` for the one ice-nuclei index the
scheme looks up (``xni = 1000 /m^3`` is a constant when not aerosol-aware);
:func:`freeze_h2o` evaluates any index for the oracle comparison.
``tpc_wev/tnc_wev`` (aerosol-aware evaporation only) and ``tnccn_act`` are not
built.  ``WRF_THOMPSON_TABLES`` lists every table this module provides.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from legoesm import constants

# --- Module PARAMETERs (module_mp_thompson.F:66-330) ---------------------------
PI = 3.1415926536                    # :67 (truncated pi, used everywhere in the scheme)
T_0 = constants.T_freeze             # :66 T_0
RHO_W = constants.rho_water          # :70 1000
RHO_I = 890.0                        # :72 cloud-ice density
NRHG = 9                             # :73
RHO_G = np.array([50., 100., 200., 300., 400., 500., 600., 700., 800.])  # :77-78
IDX_BG1 = 5                          # :79 (1-based) graupel density index for mp=8
NT_C = 100.0e6                       # :88 prescribed droplet number [/m^3]
NT_C_MAX = 1999.0e6                  # :89
NA_IN1 = 0.5e6                       # :95 (nifa1d default, never consumed with mp=8)
MU_R = 0.0
MU_G = 0.0
MU_I = 0.0
MU_S = 0.6357                        # Field et al. (2005) two-gamma snow PSD
KAP0 = 490.6
KAP1 = 17.46
LAM0 = 20.78
LAM1 = 3.29
GONV_MIN = 1.0e2
GONV_MAX = 1.0e6
AM_R = PI * RHO_W / 6.0
BM_R = 3.0
AM_S = 0.069
BM_S = 2.0
AM_G = PI * RHO_G / 6.0
BM_G = 3.0
AM_I = PI * RHO_I / 6.0
BM_I = 3.0
AV_R = 4854.0                        # Ferrier (1994) rain fall speed
BV_R = 1.0
FV_R = 195.0
AV_S = 40.0
BV_S = 0.55
FV_S = 100.0
AV_G_OLD = 442.0
BV_G_OLD = 0.89
# Heymsfield Best-Reynolds graupel fall speeds; thompson_init
# overwrites entry IDX_BG1 with the old (442, 0.89) law when the
# scheme is not hail-aware, BEFORE the gamma constants are built.
AV_G = np.array([45.9173813, 67.0867386, 98.0158463, 122.353378, 143.204224,
                 161.794724, 178.762115, 194.488785, 209.225876])
BV_G = np.full(9, 0.640961647)
AV_I = 1493.9
BV_I = 1.0
AV_C = 0.316946e8
BV_C = 2.0
C_CUBE = 0.5
C_SQRD = 0.15
EF_SI = 0.05
EF_RS = 0.95
EF_RG = 0.75
EF_RI = 0.95
R1 = 1.0e-12
R2 = 1.0e-6
EPS = 1.0e-15
TNO = 5.0                            # Cooper (1986)
ATO = 0.304
RHO_NOT = constants.p_atm_std / (constants.R_d * 298.0)  # rho_not (same literals)
SC = 0.632                           # Schmidt number
HGFR = 235.16                        # homogeneous freezing temperature [K]
RV = constants.R_v_wrf_thompson
ORV = 1.0 / RV
R_DRY = constants.R_d_wrf_thompson
CP = constants.c_p_wrf_thompson
LSUB = constants.L_s_wrf_thompson
LVAP0 = constants.L_v0_wrf_thompson
LFUS = LSUB - LVAP0
OLFUS = 1.0 / LFUS
XM0I = 1.0e-12
D0C = 1.0e-6
D0R = 50.0e-6
D0S = 300.0e-6
D0G = 350.0e-6
EPS_WRF = constants.epsilon_wrf_thompson  # the scheme's epsilon literal
# Ventilation coefficients of the rate prefactors
RAIN_VENT_A = 0.78                   # rain evaporation
RAIN_VENT_B = 0.308
VENT_A = 0.86                        # snow/graupel deposition and melting
VENT_B = 0.28

# Lookup-table dimensions
NBINS = 100
NBC = NBI = NBR = NBS = NBG = NBINS
NTB_C, NTB_I, NTB_R, NTB_S, NTB_G = 37, 64, 37, 37, 37
NTB_G1, NTB_R1, NTB_I1, NTB_T, NTB_IN = 37, 37, 55, 9, 55
N_TC = 45                            # temperature index of the freezing tables


def _decades(first, ndec):
    """1,2,...,9 x 10^k for ``ndec`` decades starting at ``first``, plus the
    closing power (the oracle's r_c/r_i/r_r/... tables)."""
    out = [float(f"{m}.e{e}") for e in range(first, first + ndec) for m in range(1, 10)]
    return np.array(out + [float(f"1.e{first + ndec}")])


R_C = _decades(-6, 4)                # cloud water content [kg/m^3]
R_I = np.concatenate([np.array([1.e-10, 2.e-10, 3.e-10, 4.e-10]),
                      _decades(-10, 7)[4:]])
R_R = _decades(-6, 4)
R_G = _decades(-6, 4)
R_S = _decades(-6, 4)
N0R_EXP = _decades(6, 4)             # [/m^4]
N0G_EXP = _decades(2, 4)             # [/m^4]
NT_I = _decades(0, 6)                # [/m^3]
NT_IN = _decades(0, 6)               # [/m^3]
# Field et al. (2005) snow moment fits
SA = np.array([5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
               0.31255, 0.000204, 0.003199, 0.0, -0.015952])
SB = np.array([0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
               0.060366, 0.000079, 0.000594, 0.0, -0.003577])
TC_TAB = np.array([-0.01, -5., -10., -15., -20., -25., -30., -35., -40.])

# Rain terminal speed polynomial in the collection tables and Ef_rw
#: v = c0 + c1 D + c2 D^2 + c3 D^3 + c4 D^4.
_VR_POLY = (-0.1021, 4.932e3, -0.9551e6, 0.07934e9, -0.002362e12)
_VR_RATIO_SNOW = 1.5                 # snow speed factor, mass ratio
_BIGG_A = -120.0                     # Bigg (1953) exp(-120 V 5.2e-4 Texp)
_BIGG_B = 5.2e-4
_T_ADJ_MAX = 3.0                     # IN temperature adjustment bounds
_D_RAIN_MAX = 0.005                  # rain bins up to 5 mm
_D_SNOW_MAX = 0.02                   # snow bins up to 2 cm
_D_GRAUPEL_MAX = 0.05                # graupel bins up to 5 cm
_NC_BIN_MAX = 3000.0                 # droplet-number bins 1..3000 /cc
_PER_CC = 1.0e6
_DC_STEP = 1.0e-6                    # cloud-water bin width
_NU_C_REF = 1000.0e6                 # nu_c = min(15, nint(1e9/N_c)+2)
_NU_C_MAX = 15
DI_AUT_MAX = 5.0                    # cloud ice > 5 D0s converts entirely
# Collision efficiency fits Beard & Grover 1974; P&K Fig 14-9)
_EF_RW_DR_LIM = (75.e-6, 125.e-6, 175.e-6, 250.e-6, 350.e-6)
_EF_RW_LIN = (0.026794, -0.20604)
_EF_RW_Q2 = (-0.00066842, 0.061542, -0.37089)
_EF_RW_Q4 = (4.091e-06, -0.00030908, 0.0066237, -0.0013687, -0.073022)
_EF_RW_C3A = (9.6719e-5, -0.0068901, 0.17305, -0.65988)
_EF_RW_C3B = (9.0488e-5, -0.006585, 0.16606, -0.56125)
_EF_RW_C3C = (0.00010721, -0.0072962, 0.1704, -0.46929)
_EF_RW_DC_MIN = 3.e-6
_EF_SW_DC_MIN = 6.e-6
_EF_P_MAX = 0.25
_EF_MAX = 0.95
_STOKES_C = 9.0                      # Stokes number St = D^2 V rho_w / (9 mu D)
_VENT_EXP_C = 5.0                    # cre(11), cge(11,:) = 0.5 (bv + 5 + 2 mu)
_BM_S_TOL = 1.0e-3                   # the oracle's bm_s == 2 test window
_VISC_EF = 1.718e-5                  # air viscosity in the Stokes number
_EF_G = (-0.1007, -0.358, 0.0261)
_EF_H = (0.1465, 1.302, -0.607, 0.293)
_EF_K0_FLOOR = 1.0e-15
_VTC_A = 1.19e4                      # droplet Stokes speed 1.19e4 (1e4 D^2/4)
_VTC_B = 1.0e4
_VTS_MIN = 1.e-3
# Numerical Recipes gammln / gammp
_GAMMLN_STP = 2.5066282746310005
_GAMMLN_COF = (76.18009172947146, -86.50532032941677, 24.01409824083091,
               -1.231739572450155, .1208650973866179e-2, -.5395239384953e-5)
_GAMMLN_SER0 = 1.000000000190015
_GAMMLN_SHIFT = 5.5
_GAM_ITMAX = 100
_GAM_EPS = 3.0e-7
_GAM_FPMIN = 1.0e-30


def gammln(xx):
    """NR ``GAMMLN``, ln Gamma(xx) for xx > 0."""
    x = np.asarray(xx, dtype=np.float64)
    y = x.copy()
    tmp = x + _GAMMLN_SHIFT
    tmp = (x + 0.5) * np.log(tmp) - tmp
    ser = np.full_like(x, _GAMMLN_SER0)
    for c in _GAMMLN_COF:
        y = y + 1.0
        ser = ser + c / y
    return tmp + np.log(_GAMMLN_STP * ser / x)


def wgamma(y):
    """``WGAMMA`` = exp(GAMMLN(y))."""
    return np.exp(gammln(y))


def _gser(a, x):
    gln = float(gammln(a))
    if x <= 0.0:
        return 0.0
    ap = a
    s = 1.0 / a
    de = s
    for _ in range(_GAM_ITMAX):
        ap = ap + 1.0
        de = de * x / ap
        s = s + de
        if abs(de) < abs(s) * _GAM_EPS:
            break
    return s * math.exp(-x + a * math.log(x) - gln)


def _gcf(a, x):
    gln = float(gammln(a))
    b = x + 1.0 - a
    c = 1.0 / _GAM_FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _GAM_ITMAX + 1):
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        if abs(d) < _GAM_FPMIN:
            d = _GAM_FPMIN
        c = b + an / c
        if abs(c) < _GAM_FPMIN:
            c = _GAM_FPMIN
        d = 1.0 / d
        de = d * c
        h = h * de
        if abs(de - 1.0) < _GAM_EPS:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def gammp(a, x):
    """NR ``GAMMP``: regularised lower incomplete gamma P(a, x)."""
    if x < 0.0 or a <= 0.0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def _log_bins(d_first, d_last, n):
    """Geometric bin edges/centres/widths."""
    xdx = np.empty(n + 1)
    xdx[0] = d_first
    xdx[n] = d_last
    k = np.arange(1, n, dtype=np.float64)
    xdx[1:n] = np.exp(k / float(n) * np.log(xdx[n] / xdx[0]) + np.log(xdx[0]))
    return np.sqrt(xdx[:-1] * xdx[1:]), xdx[1:] - xdx[:-1]


def nint(x):
    """Fortran NINT: round half away from zero."""
    x = np.asarray(x, dtype=np.float64)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.int64)


def nu_c_of(nc):
    """Droplet gamma shape ``MIN(15, NINT(1000.E6/nc) + 2)``."""
    return np.minimum(_NU_C_MAX, nint(_NU_C_REF / np.asarray(nc)) + 2)


@functools.lru_cache(maxsize=1)
def init_constants():
    """The values ``thompson_init`` computes before the tables.

    Returns a dict of NumPy float64 scalars/arrays.  Gamma constants keep
    WRF's 1-based layout shifted to 0-based (``ccg[1, n-1]`` is ccg(2,n)).
    """
    c = {}
    av_g = AV_G.copy()
    bv_g = BV_G.copy()
    av_g[IDX_BG1 - 1] = AV_G_OLD
    bv_g[IDX_BG1 - 1] = BV_G_OLD
    c["av_g"], c["bv_g"] = av_g, bv_g
    c["Sc3"] = SC ** (1.0 / 3.0)
    c["D0i"] = (XM0I / AM_I) ** (1.0 / BM_I)
    c["xm0s"] = AM_S * D0S ** BM_S
    c["xm0g"] = AM_G[NRHG - 1] * D0G ** BM_G
    n = np.arange(1, 16, dtype=np.float64)
    cce = np.stack([n + 1.0, BM_R + n + 1.0, BM_R + n + 4.0,
                    n + BV_C + 1.0, BM_R + n + BV_C + 1.0])
    ccg = wgamma(cce)
    c["cce"], c["ccg"] = cce, ccg
    c["ocg1"], c["ocg2"] = 1.0 / ccg[0], 1.0 / ccg[1]
    cie = np.array([MU_I + 1., BM_I + MU_I + 1., BM_I + MU_I + BV_I + 1.,
                    MU_I + BV_I + 1., MU_I + 2., BM_I * 0.5 + MU_I + BV_I + 1.,
                    BM_I * 0.5 + MU_I + 1.])
    cig = wgamma(cie)
    c["cie"], c["cig"] = cie, cig
    c["oig1"], c["oig2"], c["obmi"] = 1.0 / cig[0], 1.0 / cig[1], 1.0 / BM_I
    cre = np.array([BM_R + 1., MU_R + 1., BM_R + MU_R + 1., BM_R * 2. + MU_R + 1.,
                    MU_R + BV_R + 1., BM_R + MU_R + BV_R + 1.,
                    BM_R * 0.5 + MU_R + BV_R + 1., BM_R + MU_R + BV_R + 3.,
                    MU_R + BV_R + 3., MU_R + 2., 0.5 * (BV_R + _VENT_EXP_C + 2. * MU_R),
                    BM_R * 0.5 + MU_R + 1., BM_R * 2. + MU_R + BV_R + 1.])
    crg = wgamma(cre)
    c["cre"], c["crg"] = cre, crg
    c["obmr"], c["ore1"] = 1.0 / BM_R, 1.0 / cre[0]
    c["org1"], c["org2"], c["org3"] = 1.0 / crg[0], 1.0 / crg[1], 1.0 / crg[2]
    cse = np.array([BM_S + 1., BM_S + 2., BM_S * 2., BM_S + BV_S + 1.,
                    BM_S * 2. + BV_S + 1., BM_S * 2. + 1., BM_S + MU_S + 1.,
                    BM_S + MU_S + 2., BM_S + MU_S + 3., BM_S + MU_S + BV_S + 1.,
                    BM_S * 2. + MU_S + BV_S + 1., BM_S * 2. + MU_S + 1.,
                    BV_S + 2., BM_S + BV_S, MU_S + 1., 1.0 + (1.0 + BV_S) / 2.,
                    BM_S + BV_S + 2.])
    csg = wgamma(cse)
    c["cse"], c["csg"] = cse, csg
    c["oams"], c["obms"] = 1.0 / AM_S, 1.0 / BM_S
    c["ocms"] = c["oams"] ** c["obms"]
    cge = np.zeros((12, NRHG))
    cge[0] = BM_G + 1.
    cge[1] = MU_G + 1.
    cge[2] = BM_G + MU_G + 1.
    cge[3] = BM_G * 2. + MU_G + 1.
    cge[9] = MU_G + 2.
    cge[11] = BM_G * 0.5 + MU_G + 1.
    cge[4] = BM_G * 2. + MU_G + bv_g + 1.
    cge[5] = BM_G + MU_G + bv_g + 1.
    cge[6] = BM_G * 0.5 + MU_G + bv_g + 1.
    cge[7] = MU_G + bv_g + 1.
    cge[8] = MU_G + bv_g + 3.
    cge[10] = 0.5 * (bv_g + _VENT_EXP_C + 2. * MU_G)
    cgg = wgamma(cge)
    c["cge"], c["cgg"] = cge, cgg
    c["obmg"] = 1.0 / BM_G
    c["oamg"] = 1.0 / AM_G
    c["ocmg"] = c["oamg"] ** c["obmg"]
    c["oge1"], c["ogg1"] = 1.0 / cge[0, 0], 1.0 / cgg[0, 0]
    c["ogg2"], c["ogg3"] = 1.0 / cgg[1, 0], 1.0 / cgg[2, 0]
    # precomputed rate prefactors
    c["t1_qr_qc"] = PI * .25 * AV_R * crg[8]
    c["t1_qr_qi"] = PI * .25 * AV_R * crg[8]
    c["t2_qr_qi"] = PI * .25 * AM_R * AV_R * crg[7]
    c["t1_qs_qc"] = PI * .25 * AV_S
    c["t1_qs_qi"] = PI * .25 * AV_S
    c["t1_qr_ev"] = RAIN_VENT_A * crg[9]
    c["t2_qr_ev"] = RAIN_VENT_B * c["Sc3"] * np.sqrt(AV_R) * crg[10]
    c["t1_qs_sd"] = VENT_A
    c["t2_qs_sd"] = VENT_B * c["Sc3"] * np.sqrt(AV_S)
    c["t1_qs_me"] = PI * 4. * C_SQRD * OLFUS * VENT_A
    c["t2_qs_me"] = PI * 4. * C_SQRD * OLFUS * VENT_B * c["Sc3"] * np.sqrt(AV_S)
    c["t1_qg_sd"] = VENT_A * cgg[9, 0]
    c["t1_qg_me"] = PI * 4. * C_CUBE * OLFUS * VENT_A * cgg[9, 0]
    # lookup-index helpers (NINT(ALOG10(first entry)))
    for key, tab in (("nic2", R_C), ("nii2", R_I), ("nii3", NT_I), ("nir2", R_R),
                     ("nir3", N0R_EXP), ("nis2", R_S), ("nig2", R_G),
                     ("nig3", N0G_EXP), ("niIN2", NT_IN)):
        c[key] = int(nint(np.log10(tab[0])))
    # bins
    dc = np.empty(NBC)
    dc[0] = D0C
    for n_ in range(1, NBC):
        dc[n_] = dc[n_ - 1] + _DC_STEP
    c["Dc"] = dc
    c["dtc"] = np.concatenate([[D0C], dc[1:] - dc[:-1]])
    c["Di"], c["dti"] = _log_bins(c["D0i"], 2.0 * D0S, NBI)
    c["Dr"], c["dtr"] = _log_bins(D0R, _D_RAIN_MAX, NBR)
    c["Ds"], c["dts"] = _log_bins(D0S, _D_SNOW_MAX, NBS)
    c["Dg"], c["dtg"] = _log_bins(D0G, _D_GRAUPEL_MAX, NBG)
    tnc, _ = _log_bins(1.0, _NC_BIN_MAX, NBC)
    c["t_Nc"] = tnc * _PER_CC
    c["nic1"] = int(np.log(c["t_Nc"][-1] / c["t_Nc"][0]))   # INTEGER nic1
    return c


def _rain_psd(c):
    """N_r(n2) for every (N0r_exp(k), r_r(m)), shape (ntb_r1, ntb_r, nbr)."""
    n0e = N0R_EXP[:, None]
    rr = R_R[None, :]
    lam_exp = (n0e * AM_R * c["crg"][0] / rr) ** c["ore1"]
    lamr = lam_exp * (c["crg"][2] * c["org2"] * c["org1"]) ** c["obmr"]
    n0r = n0e / (c["crg"][1] * lam_exp) * lamr ** c["cre"][1]
    dr = c["Dr"]
    return (n0r[..., None] * dr ** MU_R * np.exp(-lamr[..., None] * dr) * c["dtr"])


def _vr(dr):
    return (_VR_POLY[0] + _VR_POLY[1] * dr + _VR_POLY[2] * dr * dr
            + _VR_POLY[3] * dr * dr * dr + _VR_POLY[4] * dr * dr * dr * dr)


def _bilinear(nx, kern, nr):
    """sum_{n2,n} kern[n2, n] * nx[..., n] * nr[..., n2] over two index sets."""
    a = nx.reshape(-1, nx.shape[-1])
    b = nr.reshape(-1, nr.shape[-1])
    out = a @ (b @ kern).T
    return out.reshape(nx.shape[:-1] + nr.shape[:-1])


@functools.lru_cache(maxsize=1)
def qr_acr_qg():
    """Rain-graupel collection tables for graupel density IDX_BG1.

    Returns tcg_racg, tmr_racg, tcr_gacr, tnr_racg, tnr_gacr, each shaped
    (ntb_g1, ntb_g, ntb_r1, ntb_r) (WRF's (i, j, n3=1, k, m) without n3).
    """
    c = init_constants()
    dr, dg = c["Dr"], c["Dg"]
    ib = IDX_BG1 - 1
    vr = _vr(dr)
    vg = c["av_g"][ib] * dg ** c["bv_g"][ib]
    nr = _rain_psd(c)
    n0e = N0G_EXP[:, None]
    rg = R_G[None, :]
    lam_exp = (n0e * AM_G[ib] * c["cgg"][0, 0] / rg) ** c["oge1"]
    lamg = lam_exp * (c["cgg"][2, 0] * c["ogg2"] * c["ogg1"]) ** c["obmg"]
    n0g = n0e / (c["cgg"][1, 0] * lam_exp) * lamg ** c["cge"][1, 0]
    ng = n0g[..., None] * dg ** MU_G * np.exp(-lamg[..., None] * dg) * c["dtg"]
    massr = (AM_R * dr ** BM_R)[:, None]
    massg = (AM_G[ib] * dg ** BM_G)[None, :]
    dv = vr[:, None] - vg[None, :]
    dvg = 0.5 * (dv + np.abs(dv))
    dvr = 0.5 * (-dv + np.abs(-dv))
    k0 = PI * .25 * EF_RG * (dg[None, :] + dr[:, None]) ** 2
    t1 = _bilinear(ng, k0 * dvg * massg, nr)
    z1 = _bilinear(ng, k0 * dvg * massr, nr)
    y1 = _bilinear(ng, k0 * dvg, nr)
    t2 = _bilinear(ng, k0 * dvr * massr, nr)
    y2 = _bilinear(ng, k0 * dvr, nr)
    return (t1, np.minimum(z1, R_R[None, None, None, :]), t2, y1, y2)


def _snow_moments_tab(c):
    """N_s(n) for every (r_s(i), Tc(j)), shape (ntb_s, ntb_t, nbs)."""
    m2 = (R_S * c["oams"])[:, None]
    tc = TC_TAB[None, :]

    def fit(coef, cse_x):
        return (coef[0] + coef[1] * tc + coef[2] * cse_x + coef[3] * tc * cse_x
                + coef[4] * tc * tc + coef[5] * cse_x * cse_x
                + coef[6] * tc * tc * cse_x + coef[7] * tc * cse_x * cse_x
                + coef[8] * tc * tc * tc + coef[9] * cse_x * cse_x * cse_x)

    if BM_S > 2.0 - _BM_S_TOL and BM_S < 2.0 + _BM_S_TOL:
        # (sic) the oracle transforms when bm_s == 2
        a_ = 10.0 ** fit(SA, BM_S)
        b_ = fit(SB, BM_S)
        second = (m2 / a_) ** (1. / b_)
    else:
        second = m2 + 0.0 * tc
    a_ = 10.0 ** fit(SA, c["cse"][0])
    b_ = fit(SB, c["cse"][0])
    m3 = a_ * second ** b_
    om3 = 1. / m3
    mrat = m2 * (m2 * om3) * (m2 * om3) * (m2 * om3)
    m0 = (m2 * om3) ** MU_S
    slam1 = m2 * om3 * LAM0
    slam2 = m2 * om3 * LAM1
    ds = c["Ds"]
    return (mrat[..., None] * (KAP0 * np.exp(-slam1[..., None] * ds)
                               + KAP1 * m0[..., None] * ds ** MU_S
                               * np.exp(-slam2[..., None] * ds)) * c["dts"])


@functools.lru_cache(maxsize=1)
def qr_acr_qs():
    """Rain-snow collection tables.

    Returns the twelve tables tcs_racs1, tmr_racs1, tcs_racs2, tmr_racs2,
    tcr_sacr1, tms_sacr1, tcr_sacr2, tms_sacr2, tnr_racs1, tnr_racs2,
    tnr_sacr1, tnr_sacr2, each (ntb_s, ntb_t, ntb_r1, ntb_r).
    """
    c = init_constants()
    dr, ds = c["Dr"], c["Ds"]
    vr = _vr(dr)
    vs = _VR_RATIO_SNOW * AV_S * ds ** BV_S * np.exp(-FV_S * ds)
    nr = _rain_psd(c)
    ns = _snow_moments_tab(c)
    massr = (AM_R * dr ** BM_R)[:, None]
    masss = (AM_S * ds ** BM_S)[None, :]
    big = massr > _VR_RATIO_SNOW * masss
    dv = vr[:, None] - vs[None, :]
    dvs = 0.5 * (dv + np.abs(dv))
    dvr = 0.5 * (-dv + np.abs(-dv))
    k0 = PI * .25 * EF_RS * (ds[None, :] + dr[:, None]) ** 2
    w1 = np.where(big, 1.0, 0.0)
    w2 = 1.0 - w1
    t1 = _bilinear(ns, k0 * dvs * masss * w1, nr)
    z1 = _bilinear(ns, k0 * dvs * massr * w1, nr)
    y1 = _bilinear(ns, k0 * dvs * w1, nr)
    t3 = _bilinear(ns, k0 * dvs * masss * w2, nr)
    z3 = _bilinear(ns, k0 * dvs * massr * w2, nr)
    y3 = _bilinear(ns, k0 * dvs * w2, nr)
    t2 = _bilinear(ns, k0 * dvr * massr * w1, nr)
    y2 = _bilinear(ns, k0 * dvr * w1, nr)
    z2 = _bilinear(ns, k0 * dvr * masss * w1, nr)
    t4 = _bilinear(ns, k0 * dvr * massr * w2, nr)
    y4 = _bilinear(ns, k0 * dvr * w2, nr)
    z4 = _bilinear(ns, k0 * dvr * masss * w2, nr)
    return (t1, np.minimum(z1, R_R[None, None, None, :]), t3, z3, t2, z2, t4, z4,
            y1, y3, y2, y4)


def _first_exceed(cum, thresh):
    """Index of the first bin at which a sequential sum reaches ``thresh``
    (the loop EXITs there after adding that bin); last bin if never."""
    hit = cum >= thresh
    idx = np.argmax(hit, axis=-1)
    return np.where(hit.any(axis=-1), idx, cum.shape[-1] - 1)


def _bigg_prob(vol, texp):
    """MAX(0, 1 - DEXP(-120 vol 5.2e-4 Texp)).  For small drops this
    is 1 minus a number within an ulp of 1, so it is evaluated with the C
    library ``exp`` (``math.exp``, the routine the Fortran calls) rather than
    NumPy's vectorised one, which can differ by an ulp and move these tiny
    probabilities by O(1) relative."""
    return np.array([max(0.0, 1.0 - math.exp(_BIGG_A * v * _BIGG_B * texp)) for v in vol])


def freeze_h2o(m_index):
    """Bigg (1953) freezing tables for ice-nuclei index ``m_index``
    (1-based, WRF's ``m``) at every temperature index ``k`` = 1..45.

    Returns tpi_qrfz, tni_qrfz, tpg_qrfz, tnr_qrfz shaped (ntb_r, ntb_r1, 45)
    and tpi_qcfz, tni_qcfz shaped (ntb_c, nbc, 45).
    """
    c = init_constants()
    dr, dc = c["Dr"], c["Dc"]
    massr = AM_R * dr ** BM_R
    massc = AM_R * dc ** BM_R
    orho_w = 1. / RHO_W
    t_adjust = max(-_T_ADJ_MAX, min(_T_ADJ_MAX - math.log10(NT_IN[m_index - 1]), _T_ADJ_MAX))
    nr = np.transpose(_rain_psd(c), (1, 0, 2))[..., ::-1]           # (ntb_r, ntb_r1, nbr), largest bin first
    tnc = c["t_Nc"]
    nu_c = nu_c_of(tnc)                                            # (nbc,)
    lamc = (tnc[None, :] * AM_R * c["ccg"][1, nu_c - 1] * c["ocg1"][nu_c - 1]
            / R_C[:, None]) ** c["obmr"]                           # (ntb_c, nbc)
    n0c = tnc[None, :] * c["ocg1"][nu_c - 1] * lamc ** c["cce"][0, nu_c - 1]
    ncb = (n0c[..., None] * dc[None, None, :] ** nu_c[None, :, None]
           * np.exp(-lamc[..., None] * dc) * c["dtc"])[..., ::-1]  # (ntb_c, nbc, nbc), largest first
    small = (massr < c["xm0g"])[::-1]
    massr_d, massc_d = massr[::-1], massc[::-1]
    out = [np.empty((NTB_R, NTB_R1, N_TC)) for _ in range(4)]
    out += [np.empty((NTB_C, NBC, N_TC)) for _ in range(2)]

    def pick(a, stop):
        return np.take_along_axis(a, stop[..., None], axis=-1)[..., 0]

    for k in range(N_TC):
        texp = math.exp(float(k + 1) - t_adjust * 1.0) - 1.0
        # rain, summed from the largest bin down (loop EXITs once the frozen
        # mass reaches r_r(i))
        prob = _bigg_prob(massr_d * orho_w, texp)
        pn = prob * nr
        pm = pn * massr_d
        cs1 = np.cumsum(np.where(small, pm, 0.0), axis=-1)
        cs2 = np.cumsum(np.where(small, 0.0, pm), axis=-1)
        stop = _first_exceed(cs1 + cs2, R_R[:, None, None])
        out[0][..., k] = pick(cs1, stop)
        out[1][..., k] = pick(np.cumsum(np.where(small, pn, 0.0), axis=-1), stop)
        out[2][..., k] = pick(cs2, stop)
        out[3][..., k] = pick(np.cumsum(np.where(small, 0.0, pn), axis=-1), stop)
        # cloud water
        probc = _bigg_prob(massc_d * orho_w, texp)
        pnc = probc * ncb
        cm = np.cumsum(pnc * massc_d, axis=-1)
        stop = _first_exceed(cm, R_C[:, None, None])
        out[4][..., k] = pick(cm, stop)
        out[5][..., k] = pick(np.minimum(tnc[None, :, None], np.cumsum(pnc, axis=-1)), stop)
    tpi_qrfz, tni_qrfz, tpg_qrfz, tnr_qrfz, tpi_qcfz, tni_qcfz = out
    return tpi_qrfz, tni_qrfz, tpg_qrfz, tnr_qrfz, tpi_qcfz, tni_qcfz


@functools.lru_cache(maxsize=1)
def qi_aut_qs():
    """Cloud ice -> snow tables: tps_iaus, tni_iaus, tpi_ide (ntb_i, ntb_i1)."""
    c = init_constants()
    tps = np.zeros((NTB_I, NTB_I1))
    tni = np.zeros((NTB_I, NTB_I1))
    tpi = np.zeros((NTB_I, NTB_I1))
    di, dti = c["Di"], c["dti"]
    for j in range(NTB_I1):
        for i in range(NTB_I):
            lami = (AM_I * c["cig"][1] * c["oig1"] * NT_I[j] / R_I[i]) ** c["obmi"]
            di_mean = (BM_I + MU_I + 1.) / lami
            n0i = NT_I[j] * c["oig1"] * lami ** c["cie"][0]
            if di_mean > DI_AUT_MAX * D0S:
                tps[i, j], tni[i, j], tpi[i, j] = R_I[i], NT_I[j], 0.0
            elif di_mean < c["D0i"]:
                tps[i, j], tni[i, j], tpi[i, j] = 0.0, 0.0, 1.0
            else:
                tpi[i, j] = gammp(MU_I + 2.0, lami * D0S)
                n_i = n0i * di ** MU_I * np.exp(-lami * di) * dti
                t1 = 0.0
                t2 = 0.0
                for n2 in np.nonzero(di >= D0S)[0]:
                    t1 = t1 + n_i[n2] * AM_I * di[n2] ** BM_I
                    t2 = t2 + n_i[n2]
                tps[i, j], tni[i, j] = t1, t2
    return tps, tni, tpi


def _ef_bg(stokes, p):
    """Beard & Grover (1974) efficiency shared by Ef_rw and Ef_sw."""
    reynolds = _STOKES_C * stokes / (p * p * RHO_W)
    f = np.log(reynolds)
    g = _EF_G[0] + _EF_G[1] * f + _EF_G[2] * f * f
    k0 = np.exp(g)
    z = np.log(stokes / (k0 + _EF_K0_FLOOR))
    h = _EF_H[0] + _EF_H[1] * z + _EF_H[2] * z * z + _EF_H[3] * z * z * z
    yc0 = 2.0 / PI * np.arctan(h)
    return (yc0 + p) * (yc0 + p) / ((1. + p) * (1. + p))


@functools.lru_cache(maxsize=1)
def table_efrw():
    """Rain-cloud collision efficiency t_Efrw (nbr, nbc)."""
    c = init_constants()
    dr = c["Dr"][:, None]
    dc = c["Dc"][None, :]
    p = dc / dr
    x = dc * 1.e6
    lim = _EF_RW_DR_LIM
    zero = 0.0 * dr
    poly = np.select(
        [dr < lim[0], dr < lim[1], dr < lim[2], dr < lim[3], dr < lim[4]],
        [_EF_RW_LIN[0] * x + _EF_RW_LIN[1] + zero,
         _EF_RW_Q2[0] * x * x + _EF_RW_Q2[1] * x + _EF_RW_Q2[2] + zero,
         _EF_RW_Q4[0] * x * x * x * x + _EF_RW_Q4[1] * x * x * x
         + _EF_RW_Q4[2] * x * x + _EF_RW_Q4[3] * x + _EF_RW_Q4[4] + zero,
         _EF_RW_C3A[0] * x * x * x + _EF_RW_C3A[1] * x * x + _EF_RW_C3A[2] * x
         + _EF_RW_C3A[3] + zero,
         _EF_RW_C3B[0] * x * x * x + _EF_RW_C3B[1] * x * x + _EF_RW_C3B[2] * x
         + _EF_RW_C3B[3] + zero],
        _EF_RW_C3C[0] * x * x * x + _EF_RW_C3C[1] * x * x + _EF_RW_C3C[2] * x
        + _EF_RW_C3C[3] + zero)
    vtr = _vr(dr)
    stokes = dc * dc * vtr * RHO_W / (_STOKES_C * _VISC_EF * dr)
    with np.errstate(invalid="ignore", divide="ignore"):
        bg = _ef_bg(stokes, p)
    ef = np.where(p > _EF_P_MAX, poly, bg)
    out = np.maximum(0.0, np.minimum(ef, _EF_MAX))
    return np.where((dr < D0R) | (dc < _EF_RW_DC_MIN), 0.0, out)


@functools.lru_cache(maxsize=1)
def table_efsw():
    """Snow-cloud collision efficiency t_Efsw (nbs, nbc)."""
    c = init_constants()
    ds = c["Ds"][:, None]
    dc = c["Dc"][None, :]
    vtc = _VTC_A * (_VTC_B * dc * dc * 0.25)
    vts = AV_S * ds ** BV_S * np.exp(-FV_S * ds) - vtc
    ds_m = (AM_S * ds ** BM_S / AM_R) ** c["obmr"]
    p = dc / ds_m
    off = (p > _EF_P_MAX) | (ds < D0S) | (dc < _EF_SW_DC_MIN) | (vts < _VTS_MIN)
    stokes = dc * dc * vts * RHO_W / (_STOKES_C * _VISC_EF * ds_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        ef = _ef_bg(stokes, p)
    return np.where(off, 0.0, np.maximum(0.0, np.minimum(ef, _EF_MAX)))


WRF_THOMPSON_TABLES = (
    "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2", "tcr_sacr1",
    "tms_sacr1", "tcr_sacr2", "tms_sacr2", "tnr_racs1", "tnr_racs2",
    "tnr_sacr1", "tnr_sacr2", "tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz",
    "tpi_qcfz", "tni_qcfz", "tps_iaus", "tni_iaus", "tpi_ide", "t_Efrw",
    "t_Efsw")
