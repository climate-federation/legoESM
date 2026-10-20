"""Faithful JAX port of WRF v4.8.0 Thompson microphysics (``mp_physics=8``).

Oracle: WRF ``phys/module_mp_thompson.F``, tag v4.8.0, commit
06d4240ae989cc3e50af412bb472df3d9048783c: the per-column part of
``mp_gt_driver`` (lines 1070-1500) and the whole of ``mp_thompson``
(lines 1513-4082), for the configuration WRF calls ``mp_physics=8``: one-moment
cloud water with the prescribed droplet number ``Nt_c``, two-moment rain and
cloud ice, one-moment snow (Field et al. 2005 moments) and graupel (diagnosed
intercept, density ``rho_g(idx_bg1)``), not aerosol-aware, not hail-aware.
Constants, gamma values and lookup tables come from
:mod:`.thompson_wrf_tables` (the ``thompson_init`` half).

Everything is in the ORACLE's representation: water species are MIXING
RATIOS [kg/kg dry air], ``ni``/``nr`` are per kg, arrays are ``(ncol, nz)``
with level 0 the BOTTOM (WRF's ``kts``), ``T`` in K, ``p`` in Pa, ``dz`` in m,
``w`` in m/s.  The step returns the oracle's own outputs: the updated state,
the tendency arrays as they stand before the final update, and the four
surface precipitation amounts [kg/m^2 per step].  How legoESM's specific-
humidity state maps onto this is NOT decided here (see the ``thompson``
scheme's entry point).

Every Fortran IF is a ``jnp.where`` on the same condition (piecewise-
differentiable, identical values); lookup-table indices are integers, so the
tables enter as piecewise constants.  The only departure from a line-by-line
transcription is the sedimentation sub-step loop: WRF runs ``nstep`` sub-steps
per species and column, here a static ``n_sed_max`` masked loop runs; it is
exact while ``nstep <= n_sed_max`` and a column needing more is an error
(user decision 2026-10-20: the Morrison scheme's strict fixed-maximum gate).  Dead code for ``mp_physics=8`` (aerosol scavenging and
activation, Koop freezing, graupel number/volume sedimentation, radar and
effective radii) is not ported because none of it reaches these outputs.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.microphysics import thompson_wrf_tables as W

__physics_contract__ = {
    "summary": (
        "WRF v4.8.0 Thompson bulk microphysics, mp_physics=8, one column step "
        "in the oracle's own variables (mixing ratios, per-kg numbers, level 0 "
        "at the surface): warm rain, ice/snow/graupel processes, saturation "
        "adjustment, rain evaporation, sub-stepped sedimentation."
    ),
    "inputs": {
        "T": "K", "p": "Pa", "dz": "m", "w": "m/s", "qv": "kg/kg",
        "qc": "kg/kg", "qi": "kg/kg", "qr": "kg/kg", "qs": "kg/kg",
        "qg": "kg/kg", "ni": "1/kg", "nr": "1/kg", "dt": "s",
    },
    "outputs": {
        "T": "K", "qv": "kg/kg", "qc": "kg/kg", "qi": "kg/kg", "qr": "kg/kg",
        "qs": "kg/kg", "qg": "kg/kg", "ni": "1/kg", "nr": "1/kg",
        "tten": "K/s", "qvten": "kg/kg/s", "pptrain": "kg/m^2",
        "pptsnow": "kg/m^2", "pptgraul": "kg/m^2", "pptice": "kg/m^2",
    },
    "sign_convention": (
        "z up, level 0 = surface (WRF kts). Tendencies are positive into the "
        "species; phase changes heat with lvap(T), lsub, lsub-lvap(T) over "
        "the moist cp(1+0.887 qv), exactly as the oracle. Precipitation "
        "amounts (>= 0) leave the column at the bottom level."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Thompson, Field, Rasmussen & Hall (2008), Mon. Wea. Rev. 136, "
        "5095-5115; WRF v4.8.0 phys/module_mp_thompson.F"
    ),
    "idealized_test": (
        "tests/atmosphere/microphysics/unit/test_thompson_wrf_oracle.py: "
        "columns run through the compiled WRF v4.8.0 scheme reproduce its "
        "final state, tendencies and precipitation to 1e-10 relative (fp64)."
    ),
}

# Literals of mp_thompson / mp_gt_driver (line numbers in module_mp_thompson.F)
_QV_MIN = 1.0e-10              # qv = MAX(1.E-10, qv1d)
_MVD_R_DEFAULT = 1.0e-3        # rain mvd when nr unset
_MVD_R_MAX = 2.5e-3
_MVD_R_MIN_FRAC = 0.75         # D0r*0.75
_MVD_G_DEFAULT = 1.5e-3
_MVD_G_MAX = 25.4e-3
_MVD_OFFSET = 0.672            # (3 + mu + 0.672)/lam
_DI_MIN = 5.0e-6               # cloud-ice mean size bounds
_DI_MAX = 300.0e-6
_NI_MAX = 999.0e3              # [/m^3]
_NC_MIN = 2.0
_NC_SED_MIN = 10.0
_GRA_Y_MIN = 1.0e-9            # graupel intercept diagnosis
_GRA_ZANS0 = 3.0
_GRA_ZANS1 = 2.0 / 7.0
_GRA_ZANS_OFF = 8.0
_GRA_ZANS_MIN = 2.0
_GRA_ZANS_MAX = 6.0
_TC_MAX_MOM = -0.1             # snow-moment temperature cap
_DIFFU_A = 2.11e-5             # vapour diffusivity
_DIFFU_B = 1.94
_VISCO_A = 1.718               # dynamic viscosity
_VISCO_B = 0.0049
_VISCO_C = 1.2e-5
_VISCO_S = 1.0e-5
_CP_QV = 0.887                 # moist cp
_TCOND_A = 5.69                # thermal conductivity
_TCOND_B = 0.0168
_TCOND_S = 1.0e-5 * 418.936
_SATW_TWET = 0.999             # wet-bulb only below this RH
_DEW_DT = 0.001
_EF_RR_A = 2300.0              # Verlinde & Cotton break-up
_EF_RR_B = 1950.0e-6
_AUTO_QC = 0.01e-3             # Berry & Reinhardt autoconversion threshold
_BR_ZETA_A = 6.25e-6
_BR_ZETA_B = 0.4
_BR_ZETA_C = 0.027
_BR_TAU_A = 7.5
_BR_TAU_B = 3.72
_BR_NR = 10.0                  # am_r*nu_c*10*D0r^3
_UM_PER_M = 1.0e6
_IDX_T_OFF = 2.5               # temperature table index
_IDX_T_STEP = 5.0
_ALPH_MIN = 1.0e-9
_SC_C3 = 5.0                   # Srivastava & Coen (1992) series, cubic term
_NR_GACR_WARM = 1.5            # warm rain-graupel number source factor
_STOKES_C = 9.0                # Stokes number 9 mu D
_XSAT_MIN = 1.0e-9
_STOKE_LO = 0.4                # graupel riming efficiency
_STOKE_HI = 10.0
_EF_GW_A = 0.55
_EF_GW_B = 2.51
_EF_GW_HI = 0.77
_EF_GW_WARM = 0.1
_RATE_MAX_FRAC = 0.999
_XNI_DEFAULT = 1000.0          # IN [/m^3] when not aerosol-aware
_SSATI_DEPNUC = 0.25
_T_DEPNUC = 253.15             # # const-ok: WRF literal -20 C threshold, not T_freeze
_XNC_MAX = 250.0e3
_FRAC_AUT_Q = 0.99             # ice -> snow when the PSD is snow-sized
_FRAC_AUT_N = 0.95
_DI_AUT_MAX_D0S = W.DI_AUT_MAX * W.D0S  # xDi > 5 D0s: all cloud ice converts
_DI_AUT_MIN = 0.1              # xDi < 0.1 D0s: no conversion
_CSNOW_T0 = 1.5                # C_snow ramp
_CSNOW_T1 = -30.0
_HM_TMAX = -8.0                # Hallett-Mossop
_HM_T1 = -5.0
_HM_T2 = -3.0
_HM_SLOPE2 = 0.33333333
_HM_RATE = 3.5e8
_RIME_RATIO = 2.0              # rimed snow -> graupel
_RIME_RFRAC_MAX = 30.0
_RIME_G0 = 0.15
_RIME_G1 = 0.028
_RIME_GMAX = 0.95
_RIME_B0 = 1.1
_RIME_B1 = 0.014
_RIME_BMAX = 1.5
_RIME_HALF_UM = 0.5e6          # Cober & List rime density
_RIME_RI_MIN = 0.1
_RIME_RI_MAX = 10.0
_RIME_D0 = 0.051
_RIME_D1 = 0.114
_RIME_D2 = 0.0055
_RIME_KG = 1000.0
_RIME_DENS_MIN = 150.0
_SML_NR_EXP = -0.25
_GML_N_MIN = 1.0e-4
_GML_NR_EXP = -0.33
_DT_RIME_WARM = 120.0          # dt above which warm riming goes to rain
_RHOF_MIN_VT = 1.0e-3          # sedimentation velocity threshold
_HGT_CLOUD_SED = 500.0
_W_CLOUD_SED = 1.0e-1
_PPT_FRAC = 1000.0             # precip counted when rr(kts) > R1*1000
_NEWTON_ITERS = 3
_EVAP_RH_FAST = 0.95
_EVAP_RR_FAST = 1.0e-8
_EVAP_F0 = 0.01
_EVAP_F1 = 0.99
_EVAP_FT = 20.0
_NR_EVAP_FRAC = 0.99           # pnr_rev <= 0.99 nr/dt
# Bolton (1980) / Smithsonian helpers
_THE_A = 0.2854
_THE_B = 0.28
_THE_P0 = 100000.0
_THE_C = 3.376
_THE_D = 0.00254
_THE_E = 1000.0
_THE_F = 0.81
_THE_RR = 1.0e-8
_TLCL_A = 56.0
_TLCL_B = 800.0
_TDEW_A = 35.86
_TDEW_B = 4947.2325
_TDEW_C = 23.6837
_CT_T0 = 270.0
_CT_EXP = 1.05
_CT_PEXP = 0.2
_CT_EPS = 0.01
_CT_ITERS = 100
_CT_FALLBACK_EXP = 0.286
_THW_MAX = 475.0
_THW_SPLIT = 335.5
_THW_C = (-1.00922292e-10, -1.47945344e-8, -1.7303757e-6, -0.00012709,
          1.15849867e-6, -3.518296861e-9, 3.5741522e-12)
_THW_D = (0.00000000, -3.5223513e-10, -5.7250807e-8, -5.83975422e-6,
          4.72445163e-8, -1.13402845e-10, 8.729580402e-14)
# Flatau et al. (1992) saturation fits
_RSLF_C = (.611583699E03, .444606896E02, .143177157E01, .264224321E-1,
           .299291081E-3, .203154182E-5, .702620698E-8, .379534310E-11,
           -.321582393E-13)
_RSIF_C = (.609868993E03, .499320233E02, .184672631E01, .402737184E-1,
           .565392987E-3, .521693933E-5, .307839583E-7, .105785160E-9,
           .161444444E-12)
_SAT_X_MIN = -80.0
_SAT_P_FRAC = 0.15
_RSIF_DEN_MIN = 1.0e-4
_QV_REPAIR_MIN = 1.0e-7        # mp_gt_driver negative-qv repair
_NU_C_REF = 1000.0e6
_NU_C_MAX = 15

_IDX_IN = 28                   # Nt_IN index of xni = 1000 /m^3 (see _decade_index)
_POW10 = np.array([float(10 ** k) if k >= 0 else 1.0 / float(10 ** (-k))
                   for k in range(-40, 41)])


class ThompsonWRFStep(NamedTuple):
    """Outputs of one WRF Thompson step (oracle representation, level 0 = bottom)."""

    T: jax.Array
    qv: jax.Array
    qc: jax.Array
    qi: jax.Array
    qr: jax.Array
    qs: jax.Array
    qg: jax.Array
    ni: jax.Array
    nr: jax.Array
    tten: jax.Array
    qvten: jax.Array
    qcten: jax.Array
    qiten: jax.Array
    qrten: jax.Array
    qsten: jax.Array
    qgten: jax.Array
    niten: jax.Array
    nrten: jax.Array
    ncten: jax.Array
    pptrain: jax.Array
    pptsnow: jax.Array
    pptgraul: jax.Array
    pptice: jax.Array
    nstep_required: jax.Array   # (ncol, 4): rain, ice, snow, graupel sub-steps


@functools.lru_cache(maxsize=4)
def _tables(dtype_name: str):
    """Lookup tables as NumPy arrays of ``dtype_name`` (host cache; they enter a
    trace as constants)."""
    tg = W.qr_acr_qg()
    ts = W.qr_acr_qs()
    tf = W.freeze_h2o(_IDX_IN)
    ta = W.qi_aut_qs()
    vals = tg + ts + tf + ta + (W.table_efrw(), W.table_efsw())
    return {n: np.asarray(v, dtype=dtype_name) for n, v in zip(W.WRF_THOMPSON_TABLES, vals)}


def _nint(x):
    return (jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)).astype(jnp.int32)


def _int(x):
    return jnp.trunc(x).astype(jnp.int32)


def _nu_c(nc):
    """``MIN(15, NINT(1000.E6/nc) + 2)``."""
    return jnp.minimum(_NU_C_MAX, _nint(_NU_C_REF / nc) + 2)


def _p10(n, like):
    return jnp.asarray(_POW10, dtype=like.dtype)[n + 40]


def _decade_index(x, n2, ntb):
    """WRF's decade-table index : find the decade of ``x``
    among NINT(log10 x)+{-1,0,+1}, then idx = INT(x/10^n) + 9 (n - n2)."""
    nic = _nint(jnp.log10(x))
    n = nic + 1
    for nn in (nic + 1, nic, nic - 1):
        r = x / _p10(nn, x)
        n = jnp.where((r >= 1.0) & (r < 10.0), nn, n)
    idx = _int(x / _p10(n, x)) + 10 * (n - n2) - (n - n2)
    return jnp.clip(idx, 1, ntb)


def rslf(p, t):
    """Flatau et al. (1992) liquid saturation mixing ratio."""
    c = _RSLF_C
    x = jnp.maximum(_SAT_X_MIN, t - constants.T_flatau_wrf_thompson)
    esl = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (
        c[6] + x * (c[7] + x * c[8])))))))
    esl = jnp.minimum(esl, p * _SAT_P_FRAC)
    return W.EPS_WRF * esl / (p - esl)


def rsif(p, t):
    """Flatau et al. (1992) ice saturation mixing ratio."""
    c = _RSIF_C
    x = jnp.maximum(_SAT_X_MIN, t - constants.T_flatau_wrf_thompson)
    esi = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (
        c[6] + x * (c[7] + x * c[8])))))))
    esi = jnp.minimum(esi, p * _SAT_P_FRAC)
    return W.EPS_WRF * esi / jnp.maximum(_RSIF_DEN_MIN, p - esi)


def _theta_e(pres, temp, w_non, tlcl):
    rr = w_non + _THE_RR
    power = (_THE_A * (1.0 - (_THE_B * rr)))
    xx = temp * (_THE_P0 / pres) ** power
    p1 = (_THE_C / tlcl) - _THE_D
    p2 = (rr * _THE_E) * (1.0 + _THE_F * rr)
    return xx * jnp.exp(p1 * p2)


def _t_lcl(temp, tdew):
    denom = (1.0 / (tdew - _TLCL_A)) + (jnp.log(temp / tdew) / _TLCL_B)
    return (1.0 / denom) + _TLCL_A


def _t_dew(pres, w_non):
    rr = w_non + _THE_RR
    es = pres * rr / (W.EPS_WRF + rr)
    esln = jnp.log(es)
    return (_TDEW_A * esln - _TDEW_B) / (esln - _TDEW_C)


def _theta_wetb(thetae):
    x = jnp.minimum(_THW_MAX, thetae)

    def poly(c):
        return c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * c[6])))))

    return jnp.where(x <= _THW_SPLIT, poly(_THW_C), poly(_THW_D)) + W.T_0


def _comp_t_fr_the(thelcl, pres):
    """Temperature on the moist adiabat of theta-e: up to 100
    secant steps, returning at the first |cor| < 0.01, else the theta-w fit."""
    guess = ((thelcl - 0.5 * (jnp.maximum(thelcl - _CT_T0, 0.0)) ** _CT_EXP)
             * (pres / _THE_P0) ** _CT_PEXP)
    done = jnp.zeros(guess.shape, dtype=bool)

    def body(_, carry):
        g, d = carry
        w1 = rslf(pres, g)
        w2 = rslf(pres, g + 1.0)
        tenu = _theta_e(pres, g, w1, g)
        tenup = _theta_e(pres, g + 1.0, w2, g + 1.0)
        cor = (thelcl - tenu) / (tenup - tenu)
        g_new = jnp.where(d, g, g + cor)
        d_new = d | ((cor < _CT_EPS) & (-cor < _CT_EPS))
        return g_new, d_new

    guess, done = jax.lax.fori_loop(0, _CT_ITERS, body, (guess, done))
    fallback = _theta_wetb(thelcl) * ((pres / _THE_P0) ** _CT_FALLBACK_EXP)
    return jnp.where(done, guess, fallback)


def _snow_fit(coef, tc0, x):
    """Field et al. (2005) moment relation, term order of the oracle."""
    return (coef[0] + coef[1] * tc0 + coef[2] * x + coef[3] * tc0 * x
            + coef[4] * tc0 * tc0 + coef[5] * x * x + coef[6] * tc0 * tc0 * x
            + coef[7] * tc0 * x * x + coef[8] * tc0 * tc0 * tc0
            + coef[9] * x * x * x)


def _snow_moment(smo2, tc0, x):
    a_ = 10.0 ** _snow_fit(W.SA, tc0, x)
    b_ = _snow_fit(W.SB, tc0, x)
    return a_ * smo2 ** b_


def _inherit_from_above(valid, v):
    """vt(k) = v(k) where ``valid`` else vt(k+1) (top-down loop, vt(kte+1)=0)."""
    nz = v.shape[-1]
    k = jnp.arange(nz)
    src = jnp.where(valid, k, nz)                         # nz = "none above"
    src = jax.lax.cummin(src, axis=v.ndim - 1, reverse=True)
    vpad = jnp.concatenate([v, jnp.zeros_like(v[..., :1])], axis=-1)
    return jnp.take_along_axis(vpad, src, axis=-1)


def _ksed_nstep(vt, dz, dt, nz):
    """ksed1 (0-based top-most level with vt > 1e-3, default 0, capped at
    nz-2) and nstep (max INT(dt/(dz/vt) + 1) there, 0 if none)."""
    k = jnp.arange(nz)
    act = vt > _RHOF_MIN_VT
    ksed = jnp.max(jnp.where(act, k, 0), axis=-1)
    ksed = jnp.where(ksed == nz - 1, nz - 2, ksed)
    vt_safe = jnp.where(act, vt, 1.0)
    delta_tp = dz / vt_safe
    ns = jnp.where(act, _int(dt / delta_tp + 1.0), 0)
    return ksed, jnp.max(ns, axis=-1)


def thompson_wrf_step(T, p, dz, w, qv, qc, qi, qr, qs, qg, ni, nr, dt: float,
                      *, n_sed_max: int) -> ThompsonWRFStep:
    """One WRF v4.8.0 ``mp_gt_driver`` column step (``mp_physics=8``).

    Arrays are ``(ncol, nz)``, level 0 = bottom, oracle units (module docstring).
    ``dt`` is a Python float (WRF's ``dt_in``; it also selects the
    ``dt > 120 s`` warm-riming branch).  ``n_sed_max`` is the static length of
    the sedimentation sub-step loop; the result equals the oracle's only while
    every column's required count (``nstep_required``) is ``<= n_sed_max``.
    """
    dt = float(dt)
    dtype = T.dtype
    tab = {k: jnp.asarray(v) for k, v in _tables(jnp.dtype(dtype).name).items()}
    c = W.init_constants()
    ncol, nz = T.shape
    kidx = jnp.arange(nz)
    odt = 1.0 / dt
    odts = odt
    dtsave = dt
    ib = W.IDX_BG1 - 1
    am_gb = W.AM_G[ib]
    av_gb = c["av_g"][ib]
    bv_gb = c["bv_g"][ib]
    cgg = c["cgg"]
    cge = c["cge"]
    # gamma tables indexed by the (traced) droplet shape nu_c
    ccg, cce, ocg1, ocg2 = (jnp.asarray(c[k], dtype=dtype)
                            for k in ("ccg", "cce", "ocg1", "ocg2"))
    cig, cie, oig1, oig2, obmi = c["cig"], c["cie"], c["oig1"], c["oig2"], c["obmi"]
    crg, cre, org1, org2, org3, obmr = (c["crg"], c["cre"], c["org1"], c["org2"],
                                        c["org3"], c["obmr"])
    cse, csg, oams = c["cse"], c["csg"], c["oams"]
    ogg1, ogg2, ogg3, oge1, obmg = c["ogg1"], c["ogg2"], c["ogg3"], c["oge1"], c["obmg"]
    R1, R2, EPS = W.R1, W.R2, W.EPS
    T_0 = W.T_0
    AM_R, BM_R, AM_I, BM_I = W.AM_R, W.BM_R, W.AM_I, W.BM_I
    MU_R, MU_G, MU_I = W.MU_R, W.MU_G, W.MU_I

    # ---------------- mp_gt_driver per-column preamble -------
    rho_d = W.EPS_WRF * p / (W.R_DRY * T * (qv + W.EPS_WRF))
    nc1d = W.NT_C / rho_d
    qg_pos = qg > R1
    qg_s = jnp.where(qg_pos, qg, 1.0)
    ygra1 = jnp.log10(jnp.maximum(_GRA_Y_MIN, qg_s * rho_d))
    zans1 = _GRA_ZANS0 + _GRA_ZANS1 * (ygra1 + _GRA_ZANS_OFF)
    zans1 = jnp.maximum(_GRA_ZANS_MIN, jnp.minimum(zans1, _GRA_ZANS_MAX))
    n0_exp = 10.0 ** zans1
    lam_exp = (n0_exp * am_gb * cgg[0, 0] / (rho_d * qg_s)) ** oge1
    lamg = lam_exp * (cgg[2, 0] * ogg2 * ogg1) ** obmg
    ng1d = cgg[1, 0] * ogg3 * rho_d * qg_s * lamg ** W.BM_G / am_gb
    ng1d = jnp.where(qg_pos, jnp.maximum(R2, ng1d / rho_d), 0.0)

    # ---------------- mp_thompson: local column ---------------
    temp = T
    qv_l = jnp.maximum(_QV_MIN, qv)
    pres = p
    rho = W.EPS_WRF * pres / (W.R_DRY * temp * (qv_l + W.EPS_WRF))

    L_qc = qc > R1
    rc = jnp.where(L_qc, qc * rho, R1)
    nc = jnp.where(L_qc, W.NT_C, _NC_MIN)     # Nt_c when not aerosol-aware
    qc1d = jnp.where(L_qc, qc, 0.0)
    nc1d = jnp.where(L_qc, nc1d, 0.0)

    L_qi = qi > R1
    ri = jnp.where(L_qi, qi * rho, R1)
    ri_s = jnp.where(L_qi, ri, 1.0)
    ni_ = jnp.maximum(R2, ni * rho)
    lami0 = cie[1] / _DI_MIN
    ni_unset = ni_ <= R2
    ni_ = jnp.where(ni_unset, jnp.minimum(_NI_MAX, cig[0] * oig2 * ri_s / AM_I * lami0 ** BM_I),
                    ni_)
    lami = (AM_I * cig[1] * oig1 * ni_ / ri_s) ** obmi
    xdi = (BM_I + MU_I + 1.0) * (1.0 / lami)
    lami_hi = cie[1] / _DI_MAX
    ni_ = jnp.where(xdi < _DI_MIN,
                    jnp.minimum(_NI_MAX, cig[0] * oig2 * ri_s / AM_I * lami0 ** BM_I),
                    jnp.where(xdi > _DI_MAX, cig[0] * oig2 * ri_s / AM_I * lami_hi ** BM_I, ni_))
    ni_k = jnp.where(L_qi, ni_, R2)
    qi1d = jnp.where(L_qi, qi, 0.0)
    ni1d = jnp.where(L_qi, ni, 0.0)

    L_qr = qr > R1
    rr = jnp.where(L_qr, qr * rho, R1)
    rr_s = jnp.where(L_qr, rr, 1.0)
    mvdk = 3.0 + MU_R + _MVD_OFFSET
    nr_ = jnp.maximum(R2, nr * rho)
    lam_d = mvdk / _MVD_R_DEFAULT
    nr_ = jnp.where(nr_ <= R2, crg[1] * org3 * rr_s * lam_d ** BM_R / AM_R, nr_)
    lamr = (AM_R * crg[2] * org2 * nr_ / rr_s) ** obmr
    mvd = mvdk / lamr
    lam_hi = mvdk / _MVD_R_MAX
    lam_lo = mvdk / (W.D0R * _MVD_R_MIN_FRAC)
    nr_ = jnp.where(mvd > _MVD_R_MAX, crg[1] * org3 * rr_s * lam_hi ** BM_R / AM_R,
                    jnp.where(mvd < W.D0R * _MVD_R_MIN_FRAC,
                              crg[1] * org3 * rr_s * lam_lo ** BM_R / AM_R, nr_))
    nr_k = jnp.where(L_qr, nr_, R2)
    qr1d = jnp.where(L_qr, qr, 0.0)
    nr1d = jnp.where(L_qr, nr, 0.0)

    L_qs = qs > R1
    rs = jnp.where(L_qs, qs * rho, R1)
    qs1d = jnp.where(L_qs, qs, 0.0)

    # graupel; idx_bg(k) = idx_bg1 throughout for mp=8 (the driver's
    # qb1d = qg/rho_g(idx_bg1) makes NINT(qg/qb*0.01)+1 = idx_bg1)
    L_qg = qg > R1
    rg = jnp.where(L_qg, qg * rho, R1)
    rg_s = jnp.where(L_qg, rg, 1.0)
    ng_ = jnp.maximum(R2, ng1d * rho)
    mvdg = 3.0 + MU_G + _MVD_OFFSET
    lamg_d = mvdg / _MVD_G_DEFAULT
    ng_ = jnp.where(ng_ <= R2, cgg[1, 0] * ogg3 * rg_s * lamg_d ** W.BM_G / am_gb, ng_)
    lamg = (am_gb * cgg[2, 0] * ogg2 * ng_ / rg_s) ** obmg
    mvd_g = mvdg / lamg
    lamg_hi = mvdg / _MVD_G_MAX
    lamg_lo = mvdg / W.D0R
    ng_ = jnp.where(mvd_g > _MVD_G_MAX, cgg[1, 0] * ogg3 * rg_s * lamg_hi ** W.BM_G / am_gb,
                    jnp.where(mvd_g < W.D0R, cgg[1, 0] * ogg3 * rg_s * lamg_lo ** W.BM_G / am_gb,
                              ng_))
    ng = jnp.where(L_qg, ng_, R2)
    qg1d = jnp.where(L_qg, qg, 0.0)

    # ---------------- thermodynamics -------------------------
    tempc = temp - T_0
    rhof = jnp.sqrt(W.RHO_NOT / rho)
    rhof2 = jnp.sqrt(rhof)
    qvs = rslf(pres, temp)
    delqvs = jnp.maximum(0.0, rslf(pres, T_0 + 0.0 * temp) - qv_l)
    warm = tempc > 0.0
    qvsi = jnp.where(warm, qvs, rsif(pres, temp))
    k_melting = jnp.max(jnp.where(warm, kidx + 1, 0), axis=-1)       # 1-based, 0 = none
    satw = qv_l / qvs
    sati = qv_l / qvsi
    ssatw = satw - 1.0
    ssati = sati - 1.0
    ssatw = jnp.where(jnp.abs(ssatw) < EPS, 0.0, ssatw)
    ssati = jnp.where(jnp.abs(ssati) < EPS, 0.0, ssati)
    no_micro = ~(jnp.any(L_qc | L_qi | L_qr | L_qs | L_qg, axis=-1)
                 | jnp.any(ssati > 0.0, axis=-1))
    diffu = _DIFFU_A * (temp / T_0) ** _DIFFU_B * (constants.p_atm_std / pres)
    visco = jnp.where(tempc >= 0.0, (_VISCO_A + _VISCO_B * tempc) * _VISCO_S,
                      (_VISCO_A + _VISCO_B * tempc - _VISCO_C * tempc * tempc) * _VISCO_S)
    ocp = 1.0 / (W.CP * (1.0 + _CP_QV * qv_l))
    vsc2 = jnp.sqrt(rho / visco)
    lvap = W.LVAP0 + (constants.c_pi - constants.c_pw) * tempc
    tcond = (_TCOND_A + _TCOND_B * tempc) * _TCOND_S
    m_tw = (kidx + 1 <= k_melting[:, None]) & (satw < _SATW_TWET)
    dew_t = jnp.minimum(temp - _DEW_DT, _t_dew(pres, qv_l))
    tlcl = _t_lcl(temp, dew_t)
    the = _theta_e(pres, temp, qv_l, tlcl)
    twet = jnp.where(m_tw, jnp.minimum(temp, _comp_t_fr_the(the, pres)), temp)

    # ---------------- snow moments ---------------------------
    tc0 = jnp.minimum(_TC_MAX_MOM, temp - T_0)
    smob = jnp.where(L_qs, rs * oams, 0.0)
    smob_s = jnp.where(L_qs, smob, 1.0)
    smo2 = smob_s            # bm_s == 2
    smo0 = jnp.where(L_qs, _snow_moment(smo2, tc0, 0.0), 0.0)
    smo1 = jnp.where(L_qs, _snow_moment(smo2, tc0, 1.0), 0.0)
    smoc = jnp.where(L_qs, _snow_moment(smo2, tc0, cse[0]), 0.0)
    smoc_s = jnp.where(L_qs, smoc, 1.0)
    smoe = jnp.where(L_qs, _snow_moment(smo2, tc0, cse[12]), 0.0)
    smof = jnp.where(L_qs, _snow_moment(smo2, tc0, cse[15]), 0.0)

    # ---------------- graupel and rain slopes ----------------
    lamg = (am_gb * cgg[2, 0] * ogg2 * ng / rg) ** obmg
    ilamg = 1.0 / lamg
    n0_g = ng * ogg2 * lamg ** cge[1, 0]
    lamr = (AM_R * crg[2] * org2 * nr_k / rr) ** obmr
    ilamr = 1.0 / lamr
    mvd_r = mvdk / lamr
    n0_r = nr_k * org2 * lamr ** cre[1]

    # ---------------- warm rain ------------------------------
    ef_rr = 1.0 - jnp.exp(_EF_RR_A * (mvd_r - _EF_RR_B))
    pnr_rcr = jnp.where(L_qr & (mvd_r > W.D0R), ef_rr * 2.0 * nr_k * rr, 0.0)
    nu_c = _nu_c(nc)
    rc_s = jnp.where(L_qc, rc, 1.0)
    xdc = jnp.maximum(W.D0C * _UM_PER_M, ((rc_s / (AM_R * nc)) ** obmr) * _UM_PER_M)
    lamc = (nc * AM_R * ccg[1][nu_c - 1] * ocg1[nu_c - 1] / rc_s) ** obmr
    mvd_c = jnp.maximum(W.D0C, jnp.minimum((3.0 + nu_c + _MVD_OFFSET) / lamc, W.D0R))
    mvd_c = jnp.where(L_qc, mvd_c, W.D0C)
    auto = rc > _AUTO_QC
    dc_g = ((ccg[2][nu_c - 1] * ocg2[nu_c - 1]) ** obmr / lamc) * _UM_PER_M
    dcb_arg = xdc * xdc * xdc * dc_g * dc_g * dc_g - xdc * xdc * xdc * xdc * xdc * xdc
    dc_b = jnp.where(auto, dcb_arg, 1.0) ** (1.0 / 6.0)
    z1a = _BR_ZETA_A * xdc * dc_b * dc_b * dc_b - _BR_ZETA_B
    zeta1 = 0.5 * (z1a + jnp.abs(z1a))
    # Forward-identical AD-safety: lanes without autoconversion are zeroed below; giving
    # them rc = taud = 1 keeps 1/(rc*taud) (and its cotangent) finite in fp32.
    rc_a = jnp.where(auto, rc, 1.0)
    zeta = _BR_ZETA_C * rc_a * zeta1
    tda = 0.5 * dc_b - _BR_TAU_A
    taud = jnp.where(auto, 0.5 * (tda + jnp.abs(tda)) + R1, 1.0)
    tau = _BR_TAU_B / (rc_a * taud)
    prr_wau = jnp.minimum(rc * odts, zeta / tau)
    pnr_wau = prr_wau / (AM_R * nu_c * _BR_NR * W.D0R * W.D0R * W.D0R)
    pnc_wau = jnp.minimum(nc * odts, prr_wau / (AM_R * mvd_c * mvd_c * mvd_c))
    prr_wau = jnp.where(auto, prr_wau, 0.0)
    pnr_wau = jnp.where(auto, pnr_wau, 0.0)
    pnc_wau = jnp.where(auto, pnc_wau, 0.0)

    rcw = L_qr & (mvd_r > W.D0R) & (mvd_c > W.D0C)
    lamr_b = 1.0 / ilamr
    dr_ = c["Dr"]
    idx = 1 + _int(W.NBR * jnp.log(jnp.where(rcw, mvd_r, dr_[0]) / dr_[0])
                   / np.log(dr_[-1] / dr_[0]))
    idx = jnp.minimum(idx, W.NBR)
    jdx = _int(mvd_c * _UM_PER_M)           # mvd_c in [D0c, D0r]: 1..50
    ef_rw = tab["t_Efrw"][idx - 1, jdx - 1]
    fr9 = (lamr_b + W.FV_R) ** (-cre[8])
    prr_rcw = jnp.where(rcw, jnp.minimum(rc * odts, rhof * c["t1_qr_qc"] * ef_rw * rc * n0_r * fr9),
                        0.0)
    pnc_rcw = jnp.where(rcw, jnp.minimum(nc * odts, rhof * c["t1_qr_qc"] * ef_rw * nc * n0_r * fr9),
                        0.0)

    # ---------------- frozen species -------------------------
    xds = jnp.where(L_qs, smoc_s / smob_s, 0.0)
    idx_tc = jnp.maximum(1, jnp.minimum(_nint(-tempc), W.N_TC))
    idx_t = _int((tempc - _IDX_T_OFF) / _IDX_T_STEP) - 1
    idx_t = jnp.minimum(jnp.maximum(1, -idx_t), W.NTB_T)
    idx_c = jnp.where(rc > W.R_C[0], _decade_index(jnp.where(rc > W.R_C[0], rc, 1.0), c["nic2"],
                                                   W.NTB_C), 1)
    idx_n = _nint(1.0 + float(W.NBC) * jnp.log(nc / c["t_Nc"][0]) / c["nic1"])
    idx_n = jnp.maximum(1, jnp.minimum(idx_n, W.NBC))
    idx_i = jnp.where(ri > W.R_I[0], _decade_index(jnp.where(ri > W.R_I[0], ri, 1.0), c["nii2"],
                                                   W.NTB_I), 1)
    idx_i1 = jnp.where(ni_k > W.NT_I[0], _decade_index(jnp.where(ni_k > W.NT_I[0], ni_k, 1.0),
                                                       c["nii3"], W.NTB_I1), 1)
    has_r = rr > W.R_R[0]
    idx_r = jnp.where(has_r, _decade_index(jnp.where(has_r, rr, 1.0), c["nir2"], W.NTB_R), 1)
    lam_exp_r = lamr_b * (crg[2] * org2 * org1) ** BM_R
    n0_exp_r = org1 * rr / AM_R * lam_exp_r ** cre[0]
    idx_r1 = jnp.where(has_r, _decade_index(jnp.where(has_r, n0_exp_r, 1.0), c["nir3"], W.NTB_R1),
                       W.NTB_R1)
    has_s = rs > W.R_S[0]
    idx_s = jnp.where(has_s, _decade_index(jnp.where(has_s, rs, 1.0), c["nis2"], W.NTB_S), 1)
    has_g = rg > W.R_G[0]
    idx_g = jnp.where(has_g, _decade_index(jnp.where(has_g, rg, 1.0), c["nig2"], W.NTB_G), 1)
    lamg_b = 1.0 / ilamg
    lam_exp_g = lamg_b * (cgg[2, 0] * ogg2 * ogg1) ** W.BM_G
    n0_exp_g = ogg1 * rg / am_gb * lam_exp_g ** cge[0, 0]
    idx_g1 = jnp.where(has_g, _decade_index(jnp.where(has_g, n0_exp_g, 1.0), c["nig3"], W.NTB_G1),
                       W.NTB_G1)

    # Srivastava & Coen (1992) deposition prefactor
    otemp = 1.0 / temp
    rvs = rho * qvsi
    lsub = W.LSUB
    orv = W.ORV
    rvs_p = rvs * otemp * (lsub * otemp * orv - 1.0)
    rvs_pp = rvs * (otemp * (lsub * otemp * orv - 1.0)
                    * otemp * (lsub * otemp * orv - 1.0)
                    + (-2.0 * lsub * otemp * otemp * otemp * orv)
                    + otemp * otemp)
    gamsc = lsub * diffu / tcond * rvs_p
    alphsc = (0.5 * (gamsc / (1.0 + gamsc)) * (gamsc / (1.0 + gamsc))
              * rvs_pp / rvs_p * rvs / rvs_p)
    alphsc = jnp.maximum(_ALPH_MIN, alphsc)
    xsat = jnp.where(jnp.abs(ssati) < _XSAT_MIN, 0.0, ssati)
    t1_subl = (4.0 * W.PI * (1.0 - alphsc * xsat
                             + 2.0 * alphsc * alphsc * xsat * xsat
                             - _SC_C3 * alphsc * alphsc * alphsc * xsat * xsat * xsat)
               / (1.0 + gamsc))

    # snow collecting cloud water
    ds_ = c["Ds"]
    scw = L_qc & (mvd_c > W.D0C) & (xds > W.D0S)
    idxs = 1 + _int(W.NBS * jnp.log(jnp.where(scw, xds, ds_[0]) / ds_[0]) / np.log(ds_[-1] / ds_[0]))
    idxs = jnp.minimum(idxs, W.NBS)
    ef_sw = tab["t_Efsw"][idxs - 1, jdx - 1]
    prs_scw = jnp.where(scw, jnp.minimum(rc * odts, rhof * c["t1_qs_qc"] * ef_sw * rc * smoe), 0.0)
    pnc_scw = jnp.where(scw, jnp.minimum(nc * odts, rhof * c["t1_qs_qc"] * ef_sw * nc * smoe), 0.0)

    # graupel collecting cloud water
    gcw = L_qc & (mvd_c > W.D0C) & (rg >= W.R_G[0])
    xdg = (W.BM_G + MU_G + 1.0) * ilamg
    vtg = rhof * av_gb * cgg[5, ib] * ogg3 * ilamg ** bv_gb
    stoke_g = mvd_c * mvd_c * vtg * W.RHO_W / (_STOKES_C * visco * xdg)
    ef_gw = jnp.where((stoke_g >= _STOKE_LO) & (stoke_g <= _STOKE_HI),
                      _EF_GW_A * jnp.log10(_EF_GW_B * jnp.where(stoke_g >= _STOKE_LO, stoke_g, 1.0)),
                      jnp.where(stoke_g < _STOKE_LO, 0.0, _EF_GW_HI))
    ef_gw = jnp.where(twet > T_0, ef_gw * _EF_GW_WARM, ef_gw)
    t1_qg_qc = W.PI * .25 * av_gb * cgg[8, ib]
    gpow9 = ilamg ** cge[8, ib]
    prg_gcw = jnp.where(gcw, rhof * t1_qg_qc * ef_gw * rc * n0_g * gpow9, 0.0)
    pnc_gcw = jnp.where(gcw, jnp.minimum(nc * odts, rhof * t1_qg_qc * ef_gw * nc * n0_g * gpow9),
                        0.0)

    # rain collecting snow
    tsn = lambda n: tab[n][idx_s - 1, idx_t - 1, idx_r1 - 1, idx_r - 1]  # noqa: E731
    rcs = (rr >= W.R_R[0]) & (rs >= W.R_S[0])
    cold_w = twet < T_0
    tmr_racs2, tcr_sacr2 = tsn("tmr_racs2"), tsn("tcr_sacr2")
    tmr_racs1, tcr_sacr1 = tsn("tmr_racs1"), tsn("tcr_sacr1")
    tcs_racs1, tms_sacr1 = tsn("tcs_racs1"), tsn("tms_sacr1")
    prr_rcs_c = jnp.maximum(-rr * odts, -(tmr_racs2 + tcr_sacr2 + tmr_racs1 + tcr_sacr1))
    prs_rcs_c = jnp.maximum(-rs * odts, tmr_racs2 + tcr_sacr2 - tcs_racs1 - tms_sacr1)
    prg_rcs_c = jnp.minimum((rr + rs) * odts, tmr_racs1 + tcr_sacr1 + tcs_racs1 + tms_sacr1)
    pnr_rcs_c = jnp.minimum(nr_k * odts, tsn("tnr_racs1") + tsn("tnr_racs2")
                            + tsn("tnr_sacr1") + tsn("tnr_sacr2"))
    prs_rcs_w = jnp.maximum(-rs * odts, -tcs_racs1 - tms_sacr1 + tmr_racs2 + tcr_sacr2)
    prr_rcs = jnp.where(rcs, jnp.where(cold_w, prr_rcs_c, -prs_rcs_w), 0.0)
    prs_rcs = jnp.where(rcs, jnp.where(cold_w, prs_rcs_c, prs_rcs_w), 0.0)
    prg_rcs = jnp.where(rcs & cold_w, prg_rcs_c, 0.0)
    pnr_rcs = jnp.where(rcs & cold_w, pnr_rcs_c, 0.0)

    # rain collecting graupel.  The oracle reads tcg_racg(idx_g1, idx_g,
    # idx_bg(k), idx_r1, idx_r) with idx_bg(k) = idx_bg1 = 5 although for
    # mp_physics=8 that axis is allocated with extent dimNRHG = 1: an
    # out-of-bounds read.  Column-major, it lands on the entry 4 rain-intercept
    # rows further on, (idx_g1, idx_g, idx_r1 + 4, idx_r), carrying into the
    # rain-mass index; this reproduces that read.  Past the end of the
    # allocation (idx_r = ntb_r and idx_r1 > ntb_r1 - 4) the oracle reads memory
    # that is not the table, so the port returns NaN there.
    r1_oob = idx_r1 - 1 + (W.IDX_BG1 - 1)
    r_oob = idx_r - 1 + r1_oob // W.NTB_R1
    r1_oob = r1_oob % W.NTB_R1
    past_end = r_oob >= W.NTB_R

    def tgr(n):
        v = tab[n][idx_g1 - 1, idx_g - 1, r1_oob, jnp.minimum(r_oob, W.NTB_R - 1)]
        return jnp.where(past_end, jnp.nan, v)
    rcg = (rr >= W.R_R[0]) & (rg >= W.R_G[0])
    prg_rcg_c = jnp.minimum(rr * odts, tgr("tmr_racg") + tgr("tcr_gacr"))
    pnr_rcg_c = jnp.minimum(nr_k * odts, tgr("tnr_racg") + tgr("tnr_gacr"))
    prr_rcg_w = jnp.minimum(rg * odts, tgr("tcg_racg"))
    prg_rcg = jnp.where(rcg, jnp.where(cold_w, prg_rcg_c, -prr_rcg_w), 0.0)
    prr_rcg = jnp.where(rcg, jnp.where(cold_w, -prg_rcg_c, prr_rcg_w), 0.0)
    pnr_rcg = jnp.where(rcg, jnp.where(cold_w, pnr_rcg_c, -_NR_GACR_WARM * tgr("tnr_gacr")), 0.0)

    # ---- below 0 C
    cold = temp < T_0
    rate_max = (qv_l - qvsi) * rho * odts * _RATE_MAX_FRAC
    tfz = lambda n, i0, i1: tab[n][i0 - 1, i1 - 1, idx_tc - 1]  # noqa: E731
    rfz_tab = cold & (rr > W.R_R[0])
    rfz_hom = cold & ~(rr > W.R_R[0]) & (rr > R1) & (temp < W.HGFR)
    prg_rfz = jnp.where(rfz_tab, tfz("tpg_qrfz", idx_r, idx_r1) * odts, 0.0)
    pri_rfz = jnp.where(rfz_tab, tfz("tpi_qrfz", idx_r, idx_r1) * odts,
                        jnp.where(rfz_hom, rr * odts, 0.0))
    pni_rfz = jnp.where(rfz_tab, tfz("tni_qrfz", idx_r, idx_r1) * odts,
                        jnp.where(rfz_hom, nr_k * odts, 0.0))
    pnr_rfz = jnp.where(rfz_tab, jnp.minimum(nr_k * odts, tfz("tnr_qrfz", idx_r, idx_r1) * odts),
                        0.0)
    wfz_tab = cold & (rc > W.R_C[0])
    wfz_hom = cold & ~(rc > W.R_C[0]) & (rc > R1) & (temp < W.HGFR)
    pri_wfz_t = jnp.minimum(rc * odts, tfz("tpi_qcfz", idx_c, idx_n) * odts)
    pni_wfz_t = jnp.minimum(jnp.minimum(nc * odts, pri_wfz_t / (2.0 * W.XM0I)),
                            tfz("tni_qcfz", idx_c, idx_n) * odts)
    pri_wfz = jnp.where(wfz_tab, pri_wfz_t, jnp.where(wfz_hom, rc * odts, 0.0))
    pni_wfz = jnp.where(wfz_tab, pni_wfz_t, jnp.where(wfz_hom, nc * odts, 0.0))

    nuc = cold & ((ssati >= _SSATI_DEPNUC) | ((ssatw > EPS) & (temp < _T_DEPNUC)))
    xnc_n = jnp.minimum(_XNC_MAX, W.TNO * jnp.exp(W.ATO * (T_0 - jnp.where(nuc, temp, T_0))))
    xni_n = ni_k + (pni_rfz + pni_wfz) * dtsave
    pni_inu = 0.5 * (xnc_n - xni_n + jnp.abs(xnc_n - xni_n)) * odts
    pri_inu = jnp.where(nuc, jnp.minimum(rate_max, W.XM0I * pni_inu), 0.0)
    pni_inu = pri_inu / W.XM0I

    # cloud ice deposition / sublimation and ice -> snow
    ri_q = jnp.where(L_qi, ri, 1.0)
    lami = (AM_I * cig[1] * oig1 * ni_k / ri_q) ** obmi
    ilami = 1.0 / lami
    xdi = jnp.maximum(c["D0i"], (BM_I + MU_I + 1.0) * ilami)
    xmi = AM_I * xdi ** BM_I
    oxmi = 1.0 / xmi
    pri_ide0 = W.C_CUBE * t1_subl * diffu * ssati * rvs * oig1 * cig[4] * ni_k * ilami
    neg = pri_ide0 < 0.0
    pri_ide_n = jnp.maximum(jnp.maximum(-ri * odts, pri_ide0), rate_max)
    pni_ide_n = jnp.maximum(-ni_k * odts, pri_ide_n * oxmi)
    pri_ide_p = jnp.minimum(pri_ide0, rate_max)
    tpi = tab["tpi_ide"][idx_i - 1, idx_i1 - 1]
    prs_ide_p = (1.0 - tpi) * pri_ide_p
    pri_ide_p = tpi * pri_ide_p
    ide_on = cold & L_qi
    pri_ide = jnp.where(ide_on, jnp.where(neg, pri_ide_n, pri_ide_p), 0.0)
    pni_ide = jnp.where(ide_on & neg, pni_ide_n, 0.0)
    prs_ide = jnp.where(ide_on & ~neg, prs_ide_p, 0.0)
    all_snow = (idx_i == W.NTB_I) | (xdi > _DI_AUT_MAX_D0S)
    no_aut = xdi < _DI_AUT_MIN * W.D0S
    prs_iau = jnp.where(all_snow, ri * _FRAC_AUT_Q * odts,
                        jnp.where(no_aut, 0.0,
                                  jnp.minimum(ri * _FRAC_AUT_Q * odts,
                                              tab["tps_iaus"][idx_i - 1, idx_i1 - 1] * odts)))
    pni_iau = jnp.where(all_snow, ni_k * _FRAC_AUT_N * odts,
                        jnp.where(no_aut, 0.0,
                                  jnp.minimum(ni_k * _FRAC_AUT_N * odts,
                                              tab["tni_iaus"][idx_i - 1, idx_i1 - 1] * odts)))
    prs_iau = jnp.where(ide_on, prs_iau, 0.0)
    pni_iau = jnp.where(ide_on, pni_iau, 0.0)

    # snow deposition / sublimation
    c_snow = W.C_SQRD + (tempc + _CSNOW_T0) * (W.C_CUBE - W.C_SQRD) / (_CSNOW_T1 + _CSNOW_T0)
    c_snow = jnp.maximum(W.C_SQRD, jnp.minimum(c_snow, W.C_CUBE))
    sde_vent = c["t1_qs_sd"] * smo1 + c["t2_qs_sd"] * rhof2 * vsc2 * smof
    prs_sde0 = c_snow * t1_subl * diffu * ssati * rvs * sde_vent
    prs_sde_cold = jnp.where(prs_sde0 < 0.0,
                             jnp.maximum(jnp.maximum(-rs * odts, prs_sde0), rate_max),
                             jnp.minimum(prs_sde0, rate_max))
    prs_sde = jnp.where(cold & L_qs, prs_sde_cold, 0.0)

    # graupel sublimation below 0 C
    t2_qg_sd = W.VENT_B * c["Sc3"] * np.sqrt(av_gb) * cgg[10, ib]
    gde_vent = (c["t1_qg_sd"] * ilamg ** cge[9, 0]
                + t2_qg_sd * vsc2 * rhof2 * ilamg ** cge[10, ib])
    prg_gde0 = W.C_CUBE * t1_subl * diffu * ssati * rvs * n0_g * gde_vent
    prg_gde_cold = jnp.where(prg_gde0 < 0.0,
                             jnp.maximum(jnp.maximum(-rg * odts, prg_gde0), rate_max),
                             jnp.minimum(prg_gde0, rate_max))
    prg_gde = jnp.where(cold & L_qg & (ssati < -EPS), prg_gde_cold, 0.0)

    # snow and rain collecting cloud ice
    sci = cold & L_qi & (rs >= W.R_S[0])
    prs_sci = jnp.where(sci, c["t1_qs_qi"] * rhof * W.EF_SI * ri * smoe, 0.0)
    pni_sci = jnp.where(sci, prs_sci * oxmi, 0.0)
    rci = cold & L_qi & (rr >= W.R_R[0]) & (mvd_r > 4.0 * xdi)
    pri_rci = jnp.where(rci, rhof * c["t1_qr_qi"] * W.EF_RI * ri * n0_r * fr9, 0.0)
    pnr_rci = jnp.where(rci, jnp.minimum(nr_k * odts,
                                         rhof * c["t1_qr_qi"] * W.EF_RI * ni_k * n0_r * fr9), 0.0)
    pni_rci = jnp.where(rci, pri_rci * oxmi, 0.0)
    fr8 = (lamr_b + W.FV_R) ** (-cre[7])
    prr_rci = jnp.where(rci, jnp.minimum(rr * odts, rhof * c["t2_qr_qi"] * W.EF_RI * ni_k * n0_r
                                         * fr8), 0.0)
    prg_rci = pri_rci + prr_rci

    # Hallett & Mossop (1974) rime splinters
    hm = cold & (prg_gcw > EPS) & (tempc > _HM_TMAX)
    tf = jnp.where((tempc >= _HM_T1) & (tempc < _HM_T2), 0.5 * (_HM_T2 - tempc),
                   jnp.where((tempc > _HM_TMAX) & (tempc < _HM_T1), _HM_SLOPE2 * (-_HM_TMAX + tempc),
                             0.0))
    pni_ihm = jnp.where(hm, _HM_RATE * tf * prg_gcw, 0.0)
    pri_ihm = W.XM0I * pni_ihm
    den_hm = jnp.where(hm, prs_scw + prg_gcw, 1.0)
    prs_ihm = jnp.where(hm, prs_scw / den_hm * pri_ihm, 0.0)
    prg_ihm = jnp.where(hm, prg_gcw / den_hm * pri_ihm, 0.0)

    # rimed snow partly to graupel
    rim = cold & (prs_scw > _RIME_RATIO * prs_sde) & (prs_sde > EPS)
    r_frac = jnp.minimum(_RIME_RFRAC_MAX, prs_scw / jnp.where(rim, prs_sde, 1.0))
    g_frac = jnp.minimum(_RIME_GMAX, _RIME_G0 + (r_frac - 2.0) * _RIME_G1)
    vts_boost = jnp.where(rim, jnp.minimum(_RIME_BMAX, _RIME_B0 + (r_frac - 2.0) * _RIME_B1), 1.0)
    xds_r = jnp.where(rim, xds, W.D0S)          # D0s: placeholder where unused (finite AD)
    vts_r = W.AV_S * xds_r ** W.BV_S * jnp.exp(-W.FV_S * xds_r)
    const_ri = -1.0 * (mvd_c * _RIME_HALF_UM) * vts_r / jnp.minimum(_TC_MAX_MOM, tempc)
    const_ri = jnp.maximum(_RIME_RI_MIN, jnp.minimum(const_ri, _RIME_RI_MAX))
    rime_dens = (_RIME_D0 + _RIME_D1 * const_ri - _RIME_D2 * const_ri * const_ri) * _RIME_KG
    g_frac = jnp.where(rime_dens < _RIME_DENS_MIN, 0.0, g_frac)
    prg_scw = jnp.where(rim, g_frac * prs_scw, 0.0)
    prs_scw = jnp.where(rim, (1.0 - g_frac) * prs_scw, prs_scw)

    # ---- at or above 0 C
    hot = ~cold
    # Forward-identical AD-safety: WRF evaluates the 10**(c*(twet-T_0)) melt-number
    # factors only above 0 C; cold lanes get twet = T_0 so the fp32 power cannot overflow.
    twet_h = jnp.where(hot, twet, T_0)
    melt_drive = tempc * tcond - W.LVAP0 * diffu * delqvs
    sml = hot & L_qs
    prr_sml = melt_drive * (c["t1_qs_me"] * smo1 + c["t2_qs_me"] * rhof2 * vsc2 * smof)
    prr_sml = jnp.where(prr_sml > 0.0, prr_sml + constants.c_pw * W.OLFUS * (twet - T_0)
                        * (prr_rcs + prs_scw), prr_sml)
    prr_sml = jnp.minimum(rs * odts, jnp.maximum(0.0, prr_sml))
    rs_q = jnp.where(L_qs, rs, 1.0)
    pnr_sml = jnp.where(sml & (prr_sml > 0.0),
                        smo0 / rs_q * prr_sml * 10.0 ** (_SML_NR_EXP * (twet_h - T_0)), 0.0)
    sde_hot = jnp.maximum(-rs * odts, W.C_SQRD * t1_subl * diffu * ssati * rvs * sde_vent)
    prs_sde = jnp.where(sml & ~(prr_sml > 0.0) & (ssati < 0.0), sde_hot, prs_sde)
    prr_sml = jnp.where(sml, prr_sml, 0.0)

    gml = hot & L_qg
    n0_melt = jnp.where(rg * ng < _GML_N_MIN, (_GML_N_MIN / rg) * ogg2 * lamg_b ** cge[1, 0], n0_g)
    t2_qg_me = W.PI * 4.0 * W.C_CUBE * W.OLFUS * W.VENT_B * c["Sc3"] * np.sqrt(av_gb) * cgg[10, ib]
    prr_gml = melt_drive * n0_melt * (c["t1_qg_me"] * ilamg ** cge[9, 0]
                                      + t2_qg_me * rhof2 * vsc2 * ilamg ** cge[10, ib])
    prr_gml = jnp.minimum(rg * odts, jnp.maximum(0.0, prr_gml))
    pnr_gml = jnp.where(gml & (prr_gml > 0.0),
                        prr_gml * ng / rg * 10.0 ** (_GML_NR_EXP * (twet_h - T_0)), 0.0)
    gde_hot = jnp.maximum(-rg * odts, prg_gde0)
    prg_gde = jnp.where(gml & ~(prr_gml > 0.0) & (ssati < 0.0), gde_hot, prg_gde)
    prr_gml = jnp.where(gml, prr_gml, 0.0)

    if dt > _DT_RIME_WARM:
        prr_rcw = jnp.where(hot, prr_rcw + prs_scw + prg_gcw, prr_rcw)
        prs_scw = jnp.where(hot, 0.0, prs_scw)
        prg_gcw = jnp.where(hot, 0.0, prg_gcw)

    # ---------------- conservation limits --------------------
    sump = pri_inu + pri_ide + prs_ide + prs_sde + prg_gde
    rate_max = (qv_l - qvsi) * rho * odts * _RATE_MAX_FRAC
    lim = ((sump > EPS) & (sump > rate_max)) | ((sump < -EPS) & (sump < rate_max))
    ratio = jnp.where(lim, rate_max / jnp.where(lim, sump, 1.0), 1.0)
    pri_inu, pri_ide, pni_ide = pri_inu * ratio, pri_ide * ratio, pni_ide * ratio
    prs_ide, prs_sde, prg_gde = prs_ide * ratio, prs_sde * ratio, prg_gde * ratio

    def limit(sump, rate, flag, terms):
        on = (sump < rate) & flag
        r = jnp.where(on, rate / jnp.where(on, sump, 1.0), 1.0)
        return [jnp.where(on, t * r, t) for t in terms]

    sump = -prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw - prg_gcw
    prr_wau, pri_wfz, prr_rcw, prs_scw, prg_scw, prg_gcw = limit(
        sump, -rc * odts, L_qc, [prr_wau, pri_wfz, prr_rcw, prs_scw, prg_scw, prg_gcw])
    sump = pri_ide - prs_iau - prs_sci - pri_rci
    pri_ide, prs_iau, prs_sci, pri_rci = limit(
        sump, -ri * odts, L_qi, [pri_ide, prs_iau, prs_sci, pri_rci])
    sump = -prg_rfz - pri_rfz - prr_rci + prr_rcs + prr_rcg
    prg_rfz, pri_rfz, prr_rci, prr_rcs, prr_rcg = limit(
        sump, -rr * odts, L_qr, [prg_rfz, pri_rfz, prr_rci, prr_rcs, prr_rcg])
    sump = prs_sde - prs_ihm - prr_sml + prs_rcs
    prs_sde, prs_ihm, prr_sml, prs_rcs = limit(
        sump, -rs * odts, L_qs, [prs_sde, prs_ihm, prr_sml, prs_rcs])
    sump = prg_gde - prg_ihm - prr_gml + prg_rcg
    prg_gde, prg_ihm, prr_gml, prg_rcg = limit(
        sump, -rg * odts, L_qg, [prg_gde, prg_ihm, prr_gml, prg_rcg])
    pri_ihm = prs_ihm + prg_ihm
    ratio = jnp.minimum(jnp.abs(prr_rcg), jnp.abs(prg_rcg))
    prr_rcg = ratio * jnp.where(jnp.signbit(prr_rcg), -1.0, 1.0)
    prg_rcg = -prr_rcg
    ratio = jnp.minimum(jnp.abs(prr_rcs), jnp.abs(prs_rcs))
    prr_rcs_e = ratio * jnp.where(jnp.signbit(prr_rcs), -1.0, 1.0)
    wet_hot = twet > T_0
    prr_rcs = jnp.where(wet_hot, prr_rcs_e, prr_rcs)
    prs_rcs = jnp.where(wet_hot, -prr_rcs_e, prs_rcs)

    # ---------------- tendencies -----------------------------
    orho = 1.0 / rho
    lfus2 = lsub - lvap
    qvten = (-pri_inu - pri_ide - prs_ide - prs_sde - prg_gde) * orho
    qcten = (-prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw - prg_gcw) * orho
    ncten = (-pnc_wau - pnc_rcw - pni_wfz - pnc_scw - pnc_gcw) * orho
    xrc = jnp.maximum(R1, (qc1d + qcten * dtsave) * rho)
    xnc = jnp.maximum(_NC_MIN, (nc1d + ncten * dtsave) * rho)
    nu_x = _nu_c(xnc)
    lamc = (xnc * AM_R * ccg[1][nu_x - 1] * ocg1[nu_x - 1] / rc) ** obmr
    xdc = (BM_R + nu_x + 1.0) / lamc
    lam_lo_c = cce[1][nu_x - 1] / W.D0C
    lam_hi_c = cce[1][nu_x - 1] / (W.D0R * 2.0)
    xnc_lo = ccg[0][nu_x - 1] * ocg2[nu_x - 1] * xrc / AM_R * lam_lo_c ** BM_R
    xnc_hi = ccg[0][nu_x - 1] * ocg2[nu_x - 1] * xrc / AM_R * lam_hi_c ** BM_R
    ncten_c = jnp.where(xdc < W.D0C, (xnc_lo - nc1d * rho) * odts * orho,
                        jnp.where(xdc > W.D0R * 2.0, (xnc_hi - nc1d * rho) * odts * orho, ncten))
    ncten = jnp.where(xrc > R1, ncten_c, -nc1d * odts)
    xnc = jnp.maximum(0.0, (nc1d + ncten * dtsave) * rho)
    ncten = jnp.where(xnc > W.NT_C_MAX, (W.NT_C_MAX - nc1d * rho) * odts * orho, ncten)

    qiten = (pri_inu + pri_ihm + pri_wfz + pri_rfz + pri_ide - prs_iau - prs_sci - pri_rci) * orho
    niten = (pni_inu + pni_ihm + pni_wfz + pni_rfz + pni_ide - pni_iau - pni_sci - pni_rci) * orho
    xri = jnp.maximum(R1, (qi1d + qiten * dtsave) * rho)
    xni = jnp.maximum(R2, (ni1d + niten * dtsave) * rho)
    lamx = (AM_I * cig[1] * oig1 * xni / xri) ** obmi
    xdx = (BM_I + MU_I + 1.0) * (1.0 / lamx)
    xni_lo = jnp.minimum(_NI_MAX, cig[0] * oig2 * xri / AM_I * lami0 ** BM_I)
    xni_hi = cig[0] * oig2 * xri / AM_I * lami_hi ** BM_I
    niten_c = jnp.where(xdx < _DI_MIN, (xni_lo - ni1d * rho) * odts * orho,
                        jnp.where(xdx > _DI_MAX, (xni_hi - ni1d * rho) * odts * orho, niten))
    niten = jnp.where(xri > R1, niten_c, -ni1d * odts)
    xni = jnp.maximum(0.0, (ni1d + niten * dtsave) * rho)
    niten = jnp.where(xni > _NI_MAX, (_NI_MAX - ni1d * rho) * odts * orho, niten)

    qrten = (prr_wau + prr_rcw + prr_sml + prr_gml + prr_rcs + prr_rcg - prg_rfz - pri_rfz
             - prr_rci) * orho
    nrten = (pnr_wau + pnr_sml + pnr_gml - (pnr_rfz + pnr_rcr + pnr_rcg + pnr_rcs + pnr_rci
                                            + pni_rfz)) * orho
    xrr = jnp.maximum(R1, (qr1d + qrten * dtsave) * rho)
    xnr = jnp.maximum(R2, (nr1d + nrten * dtsave) * rho)
    lamx = (AM_R * crg[2] * org2 * xnr / xrr) ** obmr
    mvdx = mvdk / lamx
    xnr_hi = crg[1] * org3 * xrr * lam_hi ** BM_R / AM_R
    xnr_lo = crg[1] * org3 * xrr * lam_lo ** BM_R / AM_R
    nrten_c = jnp.where(mvdx > _MVD_R_MAX, (xnr_hi - nr1d * rho) * odts * orho,
                        jnp.where(mvdx < W.D0R * _MVD_R_MIN_FRAC, (xnr_lo - nr1d * rho) * odts * orho,
                                  nrten))
    rain_left = xrr > R1
    nrten = jnp.where(rain_left, nrten_c, -nr1d * odts)
    qrten = jnp.where(rain_left, qrten, -qr1d * odts)

    qsten = (prs_iau + prs_sde + prs_sci + prs_scw + prs_rcs + prs_ide - prs_ihm - prr_sml) * orho
    qgten = (prg_scw + prg_rfz + prg_gde + prg_rcg + prg_gcw + prg_rci + prg_rcs - prg_ihm
             - prr_gml) * orho
    xrg = jnp.maximum(R1, (qg1d + qgten * dtsave) * rho)
    qgten = jnp.where(xrg > R1, qgten, -qg1d * odts)

    tten_c = (lsub * ocp * (pri_inu + pri_ide + prs_ide + prs_sde + prg_gde)
              + lfus2 * ocp * (pri_wfz + pri_rfz + prg_rfz + prs_scw + prg_scw + prg_gcw
                               + prg_rcs + prs_rcs + prr_rci + prg_rcg)) * orho
    tten_w = (W.LFUS * ocp * (-prr_sml - prr_gml - prr_rcg - prr_rcs)
              + lsub * ocp * (prs_sde + prg_gde)) * orho
    tten = jnp.where(cold, tten_c, tten_w)

    # ---------------- state at TAU+1 -------------------------
    temp = T + dt * tten
    otemp = 1.0 / temp
    tempc = temp - T_0
    qv_n = jnp.maximum(_QV_MIN, qv + dt * qvten)
    rho = W.EPS_WRF * pres / (W.R_DRY * temp * (qv_n + W.EPS_WRF))
    rhof = jnp.sqrt(W.RHO_NOT / rho)
    rhof2 = jnp.sqrt(rhof)
    qvs = rslf(pres, temp)
    ssatw = qv_n / qvs - 1.0
    ssatw = jnp.where(jnp.abs(ssatw) < EPS, 0.0, ssatw)
    diffu = _DIFFU_A * (temp / T_0) ** _DIFFU_B * (constants.p_atm_std / pres)
    visco = jnp.where(tempc >= 0.0, (_VISCO_A + _VISCO_B * tempc) * _VISCO_S,
                      (_VISCO_A + _VISCO_B * tempc - _VISCO_C * tempc * tempc) * _VISCO_S)
    vsc2 = jnp.sqrt(rho / visco)
    lvap = W.LVAP0 + (constants.c_pi - constants.c_pw) * tempc
    tcond = (_TCOND_A + _TCOND_B * tempc) * _TCOND_S
    ocp = 1.0 / (W.CP * (1.0 + _CP_QV * qv_n))
    lvt2 = lvap * lvap * ocp * orv * otemp * otemp

    qc_n = qc1d + qcten * dt
    L_qc = qc_n > R1
    rc = jnp.where(L_qc, qc_n * rho, R1)
    nc = jnp.where(L_qc, W.NT_C, _NC_MIN)
    qi_n = qi1d + qiten * dt
    L_qi = qi_n > R1
    ri = jnp.where(L_qi, qi_n * rho, R1)
    ni_k = jnp.where(L_qi, jnp.maximum(R2, (ni1d + niten * dt) * rho), R2)
    qr_n = qr1d + qrten * dt
    L_qr = qr_n > R1
    rr = jnp.where(L_qr, qr_n * rho, R1)
    rr_s = jnp.where(L_qr, rr, 1.0)
    nr_ = jnp.maximum(R2, (nr1d + nrten * dt) * rho)
    lamx = (AM_R * crg[2] * org2 * nr_ / rr_s) ** obmr
    mvdx = mvdk / lamx
    nr_ = jnp.where(mvdx > _MVD_R_MAX, crg[1] * org3 * rr_s * lam_hi ** BM_R / AM_R,
                    jnp.where(mvdx < W.D0R * _MVD_R_MIN_FRAC,
                              crg[1] * org3 * rr_s * lam_lo ** BM_R / AM_R, nr_))
    nr_k = jnp.where(L_qr, nr_, R2)
    qs_n = qs1d + qsten * dt
    L_qs = qs_n > R1
    rs = jnp.where(L_qs, qs_n * rho, R1)
    qg_n = qg1d + qgten * dt
    g_on = qg_n > R1
    rg = jnp.where(g_on, qg_n * rho, R1)
    rg_s = jnp.where(g_on, rg, 1.0)
    ygra1 = jnp.log10(jnp.maximum(_GRA_Y_MIN, rg_s))
    zans1 = _GRA_ZANS0 + _GRA_ZANS1 * (ygra1 + _GRA_ZANS_OFF)
    zans1 = jnp.maximum(_GRA_ZANS_MIN, jnp.minimum(zans1, _GRA_ZANS_MAX))
    n0_exp = 10.0 ** zans1
    lam_exp = (n0_exp * am_gb * cgg[0, 0] / rg_s) ** oge1
    lamg = lam_exp * (cgg[2, 0] * ogg2 * ogg1) ** obmg
    ng = jnp.where(g_on, cgg[1, 0] * ogg3 * rg_s * lamg ** W.BM_G / am_gb, R2)
    L_qg = jnp.where(g_on, L_qg, False)       # (sic) never set .true. here

    tc0 = jnp.minimum(_TC_MAX_MOM, temp - T_0)
    smob = jnp.where(L_qs, rs * oams, 0.0)
    smob_s = jnp.where(L_qs, smob, 1.0)
    smoc = jnp.where(L_qs, _snow_moment(smob_s, tc0, cse[0]), 0.0)

    lamg = (am_gb * cgg[2, 0] * ogg2 * ng / rg) ** obmg
    ilamg = 1.0 / lamg
    lamr = (AM_R * crg[2] * org2 * nr_k / rr) ** obmr
    ilamr = 1.0 / lamr
    n0_r = nr_k * org2 * lamr ** cre[1]

    # ---------------- condensation / evaporation -------------
    cnd = (ssatw > EPS) | ((ssatw < -EPS) & L_qc)
    clap = (qv_n - qvs) / (1.0 + lvt2 * qvs)
    for _ in range(_NEWTON_ITERS):
        e = jnp.exp(lvt2 * clap)
        fcd = qvs * e - qv_n + clap
        dfcd = qvs * lvt2 * e + 1.0
        clap = clap - fcd / dfcd
    orho = 1.0 / rho
    xrc = rc + clap * rho
    pos = xrc > R1
    prw_vcd = jnp.where(pos, clap * odt, -rc * orho * odt)
    nucl = pos & (clap > EPS)
    pnc_wcd = jnp.where(pos, jnp.where(nucl, 0.5 * (W.NT_C - nc + jnp.abs(W.NT_C - nc)) * odts
                                       * orho, 0.0),
                        -nc * orho * odt)
    prw_vcd = jnp.where(cnd, prw_vcd, 0.0)
    pnc_wcd = jnp.where(cnd, pnc_wcd, 0.0)
    qvten = qvten - prw_vcd
    qcten = qcten + prw_vcd
    ncten = ncten + pnc_wcd
    tten = tten + lvap * ocp * prw_vcd
    rc_c = jnp.maximum(R1, (qc1d + dt * qcten) * rho)
    rc = jnp.where(cnd, rc_c, rc)
    L_qc = jnp.where(cnd & (rc_c == R1), False, L_qc)
    nc = jnp.where(cnd, W.NT_C, nc)
    qv_c = jnp.maximum(_QV_MIN, qv + dt * qvten)
    temp_c = T + dt * tten
    rho_c = W.EPS_WRF * pres / (W.R_DRY * temp_c * (qv_c + W.EPS_WRF))
    qvs_c = rslf(pres, temp_c)
    qv_n = jnp.where(cnd, qv_c, qv_n)
    temp = jnp.where(cnd, temp_c, temp)
    rho = jnp.where(cnd, rho_c, rho)
    qvs = jnp.where(cnd, qvs_c, qvs)
    ssatw = jnp.where(cnd, qv_c / qvs_c - 1.0, ssatw)

    # ---------------- rain evaporation -----------------------
    ev = (ssatw < -EPS) & L_qr & ~(prw_vcd > 0.0)
    tempc = temp - T_0
    otemp = 1.0 / temp
    orho = 1.0 / rho
    rhof_e = jnp.sqrt(W.RHO_NOT * orho)
    rhof2_e = jnp.sqrt(rhof_e)
    diffu = _DIFFU_A * (temp / T_0) ** _DIFFU_B * (constants.p_atm_std / pres)
    visco = jnp.where(tempc >= 0.0, (_VISCO_A + _VISCO_B * tempc) * _VISCO_S,
                      (_VISCO_A + _VISCO_B * tempc - _VISCO_C * tempc * tempc) * _VISCO_S)
    vsc2 = jnp.sqrt(rho / visco)
    lvap_e = W.LVAP0 + (constants.c_pi - constants.c_pw) * tempc
    tcond = (_TCOND_A + _TCOND_B * tempc) * _TCOND_S
    ocp_e = 1.0 / (W.CP * (1.0 + _CP_QV * qv_n))
    rvs = rho * qvs
    rvs_p = rvs * otemp * (lvap_e * otemp * orv - 1.0)
    rvs_pp = rvs * (otemp * (lvap_e * otemp * orv - 1.0)
                    * otemp * (lvap_e * otemp * orv - 1.0)
                    + (-2.0 * lvap_e * otemp * otemp * otemp * orv)
                    + otemp * otemp)
    gamsc = lvap_e * diffu / tcond * rvs_p
    alphsc = (0.5 * (gamsc / (1.0 + gamsc)) * (gamsc / (1.0 + gamsc))
              * rvs_pp / rvs_p * rvs / rvs_p)
    alphsc = jnp.maximum(_ALPH_MIN, alphsc)
    xsat = jnp.minimum(-_XSAT_MIN, ssatw)
    t1_evap = (2.0 * W.PI * (1.0 - alphsc * xsat
                             + 2.0 * alphsc * alphsc * xsat * xsat
                             - _SC_C3 * alphsc * alphsc * alphsc * xsat * xsat * xsat)
               / (1.0 + gamsc))
    lamr_b = 1.0 / ilamr
    fast = (qv_n / qvs < _EVAP_RH_FAST) & (rr * orho <= _EVAP_RR_FAST)
    prv = (t1_evap * diffu * (-ssatw) * n0_r * rvs
           * (c["t1_qr_ev"] * ilamr ** cre[9]
              + c["t2_qr_ev"] * vsc2 * rhof2_e * ((lamr_b + 0.5 * W.FV_R) ** (-cre[10]))))
    rate_max = jnp.minimum((rr * orho * odts), (qvs - qv_n) * odts)
    prv = jnp.minimum(rate_max, prv * orho)
    eva_factor = jnp.minimum(1.0, _EVAP_F0 + (_EVAP_F1 - _EVAP_F0) * (tempc / _EVAP_FT))
    prv = jnp.where(prr_gml > 0.0, prv * eva_factor, prv)
    prv_rev = jnp.where(ev, jnp.where(fast, rr * orho * odts, prv), 0.0)
    pnr_rev = jnp.where(ev, jnp.minimum(nr_k * _NR_EVAP_FRAC * orho * odts, prv_rev * nr_k / rr),
                        0.0)
    qrten = qrten - prv_rev
    qvten = qvten + prv_rev
    nrten = nrten - pnr_rev
    tten = tten - lvap_e * ocp_e * prv_rev
    rr = jnp.where(ev, jnp.maximum(R1, (qr1d + dt * qrten) * rho), rr)
    nr_k = jnp.where(ev, jnp.maximum(R2, (nr1d + dt * nrten) * rho), nr_k)
    qv_e = jnp.maximum(_QV_MIN, qv + dt * qvten)
    temp_e = T + dt * tten
    qv_n = jnp.where(ev, qv_e, qv_n)
    temp = jnp.where(ev, temp_e, temp)
    rho = jnp.where(ev, W.EPS_WRF * pres / (W.R_DRY * temp_e * (qv_e + W.EPS_WRF)), rho)
    rhof = jnp.where(ev, rhof_e, rhof)
    lvap = jnp.where(ev, lvap_e, lvap)
    ocp = jnp.where(ev, ocp_e, ocp)

    # ---------------- fall speeds and sub-step counts --------
    any_r = jnp.any(L_qr, axis=-1, keepdims=True)
    rhof = jnp.where(any_r, jnp.sqrt(W.RHO_NOT / rho), rhof)
    r_on = rr > R1
    rr_q = jnp.where(r_on, rr, 1.0)
    lamx = (AM_R * crg[2] * org2 * jnp.where(r_on, nr_k, 1.0) / rr_q) ** obmr
    vtr = rhof * W.AV_R * crg[5] * org3 * lamx ** cre[2] * ((lamx + W.FV_R) ** (-cre[5]))
    vtnr = rhof * W.AV_R * crg[6] / crg[11] * lamx ** cre[11] * ((lamx + W.FV_R) ** (-cre[6]))
    vtrk = jnp.where(any_r, _inherit_from_above(r_on, vtr), 0.0)
    vtnrk = jnp.where(any_r, _inherit_from_above(r_on, vtnr), 0.0)
    ksed_r, ns_r = _ksed_nstep(jnp.maximum(vtrk, vtnrk), dz, dt, nz)

    any_c = jnp.any(L_qc, axis=-1, keepdims=True)
    cum = jnp.cumsum(dz, axis=-1)
    cum_prev = jnp.concatenate([jnp.zeros_like(cum[:, :1]), cum[:, :-1]], axis=-1)
    scanned = (cum_prev <= _HGT_CLOUD_SED) & (kidx < nz - 1)
    ksed_c = jnp.max(jnp.where(scanned & (rc > R2), kidx, 0), axis=-1)
    c_on = (kidx <= ksed_c[:, None]) & (rc > R1) & (w < _W_CLOUD_SED) & any_c
    nu_v = _nu_c(nc)
    rc_q = jnp.where(c_on, rc, 1.0)
    lamx = (nc * AM_R * ccg[1][nu_v - 1] * ocg1[nu_v - 1] / rc_q) ** obmr
    ilamc = 1.0 / lamx
    vtck = jnp.where(c_on, rhof * W.AV_C * ccg[4][nu_v - 1] * ocg2[nu_v - 1] * ilamc ** W.BV_C, 0.0)
    vtnck = jnp.where(c_on, rhof * W.AV_C * ccg[3][nu_v - 1] * ocg1[nu_v - 1] * ilamc ** W.BV_C,
                      0.0)

    any_i = jnp.any(L_qi, axis=-1, keepdims=True)
    i_on = ri > R1
    ri_q = jnp.where(i_on, ri, 1.0)
    lamx = (AM_I * cig[1] * oig1 * jnp.where(i_on, ni_k, 1.0) / ri_q) ** obmi
    ilx = 1.0 / lamx
    vti = rhof * W.AV_I * cig[2] * oig2 * ilx ** W.BV_I
    vtni = rhof * W.AV_I * cig[5] / cig[6] * ilx ** W.BV_I
    vtik = jnp.where(any_i, _inherit_from_above(i_on, vti), 0.0)
    vtnik = jnp.where(any_i, _inherit_from_above(i_on, vtni), 0.0)
    ksed_i, ns_i = _ksed_nstep(vtik, dz, dt, nz)

    any_s = jnp.any(L_qs, axis=-1, keepdims=True)
    s_on = rs > R1
    xds_v = jnp.where(s_on, smoc / jnp.where(s_on, smob, 1.0), 1.0)
    mr = 1.0 / xds_v
    ils1 = 1.0 / (mr * W.LAM0 + W.FV_S)
    ils2 = 1.0 / (mr * W.LAM1 + W.FV_S)
    t1_vts = W.KAP0 * csg[3] * ils1 ** cse[3]
    t2_vts = W.KAP1 * mr ** W.MU_S * csg[9] * ils2 ** cse[9]
    ils1 = 1.0 / (mr * W.LAM0)
    ils2 = 1.0 / (mr * W.LAM1)
    t3_vts = W.KAP0 * csg[0] * ils1 ** cse[0]
    t4_vts = W.KAP1 * mr ** W.MU_S * csg[6] * ils2 ** cse[6]
    vts = rhof * W.AV_S * (t1_vts + t2_vts) / (t3_vts + t4_vts)
    sr_ = rs / (rs + rr)
    vts = jnp.where(prr_sml > 0.0, vts * sr_ + (1.0 - sr_) * vtrk, vts * vts_boost)
    vtsk = jnp.where(any_s, _inherit_from_above(s_on, vts), 0.0)
    ksed_s, ns_s = _ksed_nstep(vtsk, dz, dt, nz)

    any_g = jnp.any(L_qg, axis=-1, keepdims=True)
    g_on = rg > R1
    vtg = rhof * W.AV_G_OLD * cgg[5, ib] * ogg3 * ilamg ** W.BV_G_OLD
    vtgk = jnp.where(any_g, _inherit_from_above(g_on, vtg), 0.0)
    ksed_g, ns_g = _ksed_nstep(vtgk, dz, dt, nz)

    # ---------------- sedimentation --------------------------
    odzq = 1.0 / dz
    orho = 1.0 / rho

    def sed_step(q, ten, vt, ksed, onst, active, floor):
        """One sub-step: top level loses its flux; levels ksed..0 gain the
        flux from above and lose their own."""
        sed = vt * q
        sed_up = jnp.concatenate([sed[:, 1:], jnp.zeros_like(sed[:, :1])], axis=-1)
        top = kidx == nz - 1
        body = kidx <= ksed[:, None]
        dten = jnp.where(top, -sed * odzq * onst * orho,
                         jnp.where(body, (sed_up - sed) * odzq * onst * orho, 0.0))
        dq = jnp.where(top, q - sed * odzq * dt * onst,
                       jnp.where(body, q + (sed_up - sed) * odzq * dt * onst, q))
        dq = jnp.where(top | body, jnp.maximum(floor, dq), q)
        act = active[:, None]
        return jnp.where(act, dq, q), jnp.where(act, ten + dten, ten), sed[:, 0]

    n_r = jnp.maximum(ns_r, 1)
    n_i = jnp.maximum(ns_i, 1)
    n_s = jnp.maximum(ns_s, 1)
    n_g = jnp.maximum(ns_g, 1)
    on_r = 1.0 / n_r.astype(dtype)[:, None]
    on_i = 1.0 / n_i.astype(dtype)[:, None]
    on_s = 1.0 / n_s.astype(dtype)[:, None]
    on_g = 1.0 / n_g.astype(dtype)[:, None]
    zero_c = jnp.zeros((ncol,), dtype)

    def sub(n, carry):
        (rr, nr_k, ri, ni_k, rs, rg, qrten, nrten, qiten, niten, qsten, qgten,
         pr, pi_, ps, pg) = carry
        a = (n < n_r) & any_r[:, 0]
        rr_new, qrten, sr0 = sed_step(rr, qrten, vtrk, ksed_r, on_r, a, R1)
        nr_k, nrten, _ = sed_step(nr_k, nrten, vtnrk, ksed_r, on_r, a, R2)
        pr = pr + jnp.where(a & (rr_new[:, 0] > R1 * _PPT_FRAC), sr0 * dt * on_r[:, 0], 0.0)
        a = (n < n_i) & any_i[:, 0]
        ri_new, qiten, si0 = sed_step(ri, qiten, vtik, ksed_i, on_i, a, R1)
        ni_k, niten, _ = sed_step(ni_k, niten, vtnik, ksed_i, on_i, a, R2)
        pi_ = pi_ + jnp.where(a & (ri_new[:, 0] > R1 * _PPT_FRAC), si0 * dt * on_i[:, 0], 0.0)
        a = (n < n_s) & any_s[:, 0]
        rs_new, qsten, ss0 = sed_step(rs, qsten, vtsk, ksed_s, on_s, a, R1)
        ps = ps + jnp.where(a & (rs_new[:, 0] > R1 * _PPT_FRAC), ss0 * dt * on_s[:, 0], 0.0)
        a = (n < n_g) & any_g[:, 0]
        rg_new, qgten, sg0 = sed_step(rg, qgten, vtgk, ksed_g, on_g, a, R1)
        pg = pg + jnp.where(a & (rg_new[:, 0] > R1 * _PPT_FRAC), sg0 * dt * on_g[:, 0], 0.0)
        return (rr_new, nr_k, ri_new, ni_k, rs_new, rg_new, qrten, nrten, qiten, niten,
                qsten, qgten, pr, pi_, ps, pg)

    carry = (rr, nr_k, ri, ni_k, rs, rg, qrten, nrten, qiten, niten, qsten, qgten,
             zero_c, zero_c, zero_c, zero_c)
    carry = jax.lax.fori_loop(0, int(n_sed_max), sub, carry)
    (rr, nr_k, ri, ni_k, rs, rg, qrten, nrten, qiten, niten, qsten, qgten,
     pptrain, pptice, pptsnow, pptgraul) = carry

    # cloud water: one pass over the lowest ~500 m
    sed_c = vtck * rc
    sed_n = vtnck * nc
    sed_c_up = jnp.concatenate([sed_c[:, 1:], jnp.zeros_like(sed_c[:, :1])], axis=-1)
    sed_n_up = jnp.concatenate([sed_n[:, 1:], jnp.zeros_like(sed_n[:, :1])], axis=-1)
    c_body = (kidx <= ksed_c[:, None]) & any_c
    qcten = jnp.where(c_body, qcten + (sed_c_up - sed_c) * odzq * orho, qcten)
    ncten = jnp.where(c_body, ncten + (sed_n_up - sed_n) * odzq * orho, ncten)

    # ---------------- instant melt / freeze ------------------
    xri = jnp.maximum(0.0, qi1d + qiten * dt)
    mlt = (temp > T_0) & (xri > 0.0)
    qcten = jnp.where(mlt, qcten + xri * odt, qcten)
    ncten = jnp.where(mlt, ncten + ni1d * odt, ncten)
    qiten = jnp.where(mlt, qiten - xri * odt, qiten)
    niten = jnp.where(mlt, -ni1d * odt, niten)
    tten = jnp.where(mlt, tten - W.LFUS * ocp * xri * odt, tten)
    xrc = jnp.maximum(0.0, qc1d + qcten * dt)
    frz = (temp < W.HGFR) & (xrc > 0.0)
    lfus2 = lsub - lvap
    xnc = nc1d + ncten * dt
    qiten = jnp.where(frz, qiten + xrc * odt, qiten)
    niten = jnp.where(frz, niten + xnc * odt, niten)
    qcten = jnp.where(frz, qcten - xrc * odt, qcten)
    ncten = jnp.where(frz, ncten - xnc * odt, ncten)
    tten = jnp.where(frz, tten + lfus2 * ocp * xrc * odt, tten)

    # ---------------- final update ---------------------------
    t_out = T + tten * dt
    qv_out = jnp.maximum(_QV_MIN, qv + qvten * dt)
    qc_out = qc1d + qcten * dt
    qc_out = jnp.where(qc_out <= R1, 0.0, qc_out)
    qi_out = qi1d + qiten * dt
    ni_out = jnp.maximum(R2 / rho, ni1d + niten * dt)
    qi_pos = qi_out > R1
    qi_q = jnp.where(qi_pos, qi_out, 1.0)
    lamx = (AM_I * cig[1] * oig1 * jnp.where(qi_pos, ni_out, 1.0) / qi_q) ** obmi
    xdx = (BM_I + MU_I + 1.0) * (1.0 / lamx)
    lamx = jnp.where(xdx < _DI_MIN, lami0, jnp.where(xdx > _DI_MAX, lami_hi, lamx))
    ni_out = jnp.where(qi_pos, jnp.minimum(cig[0] * oig2 * qi_q / AM_I * lamx ** BM_I,
                                           _NI_MAX / rho), 0.0)
    qi_out = jnp.where(qi_pos, qi_out, 0.0)
    qr_out = qr1d + qrten * dt
    nr_out = jnp.maximum(R2 / rho, nr1d + nrten * dt)
    qr_pos = qr_out > R1
    qr_q = jnp.where(qr_pos, qr_out, 1.0)
    lamx = (AM_R * crg[2] * org2 * jnp.where(qr_pos, nr_out, 1.0) / qr_q) ** obmr
    mvdx = mvdk / lamx
    mvdx = jnp.where(mvdx > _MVD_R_MAX, _MVD_R_MAX,
                     jnp.where(mvdx < W.D0R * _MVD_R_MIN_FRAC, W.D0R * _MVD_R_MIN_FRAC, mvdx))
    lamx = mvdk / mvdx
    nr_out = jnp.where(qr_pos, crg[1] * org3 * qr_q * lamx ** BM_R / AM_R, 0.0)
    qr_out = jnp.where(qr_pos, qr_out, 0.0)
    qs_out = qs1d + qsten * dt
    qs_out = jnp.where(qs_out <= R1, 0.0, qs_out)
    qg_out = qg1d + qgten * dt
    qg_out = jnp.where(qg_out <= R1, 0.0, qg_out)

    # ---------------- no_micro early return + driver repair -------
    nm = no_micro[:, None]

    def sel(full, early):
        return jnp.where(nm, early, full)

    t_out = sel(t_out, T)
    qv_out = sel(qv_out, qv)
    qc_out = sel(qc_out, qc1d)
    qi_out = sel(qi_out, qi1d)
    ni_out = sel(ni_out, ni1d)
    qr_out = sel(qr_out, qr1d)
    nr_out = sel(nr_out, nr1d)
    qs_out = sel(qs_out, qs1d)
    qg_out = sel(qg_out, qg1d)
    z = jnp.zeros_like(T)
    tens = [sel(x, z) for x in (tten, qvten, qcten, qiten, qrten, qsten, qgten, niten, nrten,
                                ncten)]
    zc = jnp.zeros((ncol,), dtype)
    ppt = [jnp.where(no_micro, zc, x) for x in (pptrain, pptsnow, pptgraul, pptice)]
    qv_out = _repair_negative_qv(qv_out, qv)
    nsteps = jnp.stack([jnp.where(any_r[:, 0], n_r, 0), jnp.where(any_i[:, 0], n_i, 0),
                        jnp.where(any_s[:, 0], n_s, 0), jnp.where(any_g[:, 0], n_g, 0)], axis=-1)
    nsteps = jnp.where(no_micro[:, None], 0, nsteps)
    floats = [t_out, qv_out, qc_out, qi_out, qr_out, qs_out, qg_out, ni_out, nr_out,
              *tens, *ppt]
    # A column needing more sub-steps than the static loop holds is an error
    # (user decision 2026-10-20), raised with the Morrison scheme's strict
    # sedimentation gate unchanged (morrison.py, sed_cfl_substeps_strict):
    # equinox error_if on the outputs, plus a NaN poison (x*1 + 0 on the clean
    # path) so the abort also reaches derivatives and every consumer.
    overflow = jnp.any(nsteps > int(n_sed_max))
    floats = eqx.error_if(
        floats, overflow,
        "WRF Thompson sedimentation: a column needs more sub-steps than "
        f"n_sed_max={int(n_sed_max)}; shorten the physics step or raise the cap")
    poison = jnp.where(overflow, jnp.nan, 1.0).astype(dtype)
    bias = jnp.where(overflow, jnp.nan, 0.0).astype(dtype)
    floats = [x * poison + bias for x in floats]
    return ThompsonWRFStep(*floats, nsteps)



def _repair_negative_qv(qv_out, qv_in):
    """mp_gt_driver's repair of a negative returned vapour:
    interior levels take MAX(1e-7, mean of the already-written level below
    and the not-yet-written input above), the two lowest and three highest
    levels take 1e-7.  Only an input column with qv < 0 and no microphysics
    can reach it (mp_thompson otherwise returns qv >= 1e-10)."""
    nz = qv_out.shape[-1]

    def step(prev, k):
        q = qv_out[:, k]
        above = qv_in[:, jnp.minimum(k + 1, nz - 1)]
        interior = (k + 1 < nz - 2) & (k + 1 > 2)
        rep = jnp.where(interior, jnp.maximum(_QV_REPAIR_MIN, 0.5 * (prev + above)),
                        _QV_REPAIR_MIN)
        out = jnp.where(q < 0.0, rep, q)
        return out, out

    _, cols = jax.lax.scan(step, jnp.zeros_like(qv_out[:, 0]), jnp.arange(nz))
    return jnp.moveaxis(cols, 0, -1)
