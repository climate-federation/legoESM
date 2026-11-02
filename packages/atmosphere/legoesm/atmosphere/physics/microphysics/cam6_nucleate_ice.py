"""CAM6 cirrus ice nucleation: Liu and Penner (2005) with pre-existing ice (Shi et al. 2015).

Port of ``nucleate_ice.F90`` (``nucleati``, ``hf``, ``hetero``, ``Vpreice``, ``frachom``) and of the
per-level part of ``nucleate_ice_cam.F90::nucleate_ice_cam_calc`` with the modal-aerosol branch, at
ESCOMP/CAM ``cam_cesm2_1_rel_60``, under CAM6's namelist (namelist_defaults_cam.xml:1262-1279, phys=cam6,
microphys=mg2): ``use_preexisting_ice`` true, ``use_hetfrz_classnuc`` true (so the Meyers deposition term is
off; mixed-phase heterogeneous freezing is a separate scheme), ``nucleate_ice_incloud`` false,
``nucleate_ice_subgrid`` = ``nucleate_ice_subgrid_strat`` = 1.2, ``nucleate_ice_strat`` 1.0 with
``nucleate_ice_use_troplev`` true; ``mincld`` 1e-4 (microp_aero.F90:73).  The sub-grid ice updraft is
``microp_aero.F90``'s ``wsubi`` from CLUBB ``tke = 1.5 wp2`` (:517, :532-547).

Output ``naai`` [1/kg] is MG2's activated ice number: MG2 nucleates ``(naai - ni/icldm) icldm`` per step
where ``naai > 0``, ``T < tmelt - 5`` and ``relhum esl/esi > 1.05`` (micro_mg2_0.F90:1117-1129).

Every branch of the Fortran is a ``jnp.where`` on the same conditions, evaluated in the same order;
IEEE semantics are kept (``log(0)`` gives ``-inf`` and the NaN regime threshold compares false exactly as in
the Fortran when no dust is present).  Not differentiable through the branch thresholds.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import (
    saturation_vapor_pressure_goff_gratch,
    saturation_vapor_pressure_ice_goff_gratch,
)

__physics_contract__ = {
    "summary": "CAM6 Liu-Penner cirrus ice nucleation with pre-existing ice (nucleate_ice.F90 + nucleate_ice_cam_calc).",
    "inputs": {"T": "K", "q_v": "kg/kg", "pmid": "Pa", "q_c": "kg/kg", "q_i": "kg/kg", "N_i": "1/kg",
               "ast": "1", "wp2_iface": "m^2/s^2", "aerosol numbers": "1/kg", "aerosol masses": "kg/kg"},
    "outputs": {"naai": "1/kg"},
    "sign_convention": "none (a number of activated ice nuclei)",
    "conserves": ["none"],
    "differentiable": False,
    "reference": "CAM cam_cesm2_1_rel_60 nucleate_ice.F90, nucleate_ice_cam.F90; Liu and Penner (2005, "
                 "doi:10.1127/0941-2948/2005/0059); Shi et al. (2015, doi:10.5194/acp-15-1503-2015)",
    "idealized_test": "gfortran build of unmodified nucleate_ice.F90 with a driver transcribing "
                      "nucleate_ice_cam_calc (scripts/validate/oracle/cam6_nucleate_ice)",
}

# --- provenance block: nucleate_ice.F90 (cam_cesm2_1_rel_60) -----------------------------------------------
SHET = 1.3                     # het freezing threshold (:52)
RHOICE = 0.5e3                 # kg/m3 (:53)
MINWEFF = 0.001                # m/s (:54)
GAMMA4 = 6.0                   # (:58)
CI = RHOICE * math.pi / 6.0    # (:87)
SHOM_PRE = -1.5                # Shom < 1: recomputed from Ren and McKenzie (:157)
NI_PRE_MIN = 10.0              # pre-existing ice threshold, 1/m3 (:155)
QI_PRE_MIN = 1.0e-10           # kg/kg (:155)
RI_PRE_MIN = 1.0e-8            # m (:161)
DETA_T_W = 0.23                # detaT = wbar / 0.23 (:182)
RHI_ACT = 1.2                  # RHimean threshold (:197)
TC_MIXED_MAX = 0.0             # (:199)
TC_HOM = -37.0                 # (:199, :373)
QC_SMALL = 1.0e-12             # (:199)
N_SMALL = 1.0e-10              # aerosol number floors (:195, :203)
REGM_A = (-1.4938, 12.884)     # A = a0 ln(Ns) + a1 (:202)
REGM_B = (-10.41, -67.69)      # B = b0 ln(Ns) + b1 (:203)
TC_HOM_ONLY = -40.0            # exclude T < -40 C and W > 1 m/s from hetero (:206)
W_HOM_ONLY = 1.0
NIHF_OCCUR = 1.0e-3            # hom occurs (:212)
REGM_RAMP = 5.0                # (:229, :260)
NUCI_MAX = 9999.0              # (:288)
PER_CM3 = 1.0e6                # #/cm3 -> #/m3
# hetero (:316-341)
HET = dict(A11=0.0263, A12=-0.0185, A21=2.758, A22=1.3221, B11=-0.008, B12=-0.0468, B21=-0.2667, B22=-1.4588)
# hf (:350-411)
HF = dict(A1f=0.0231, A21f=-1.6387, A22f=-6.045, B1f=-0.008, B21f=-0.042, B22f=-0.112, C1f=0.0739, C2f=1.2372,
          A1s=-0.3949, A2s=1.282, B1s=-0.0156, B2s=0.0111, B3s=0.0217, C1s=0.120, C2s=2.312,
          RHwA=(6.0e-4, 6.6e-3), RHwB=(6.0e-2, 1.052), RHwC=(1.68, 129.35), regm=(6.07, -55.0), T64=-64.0)
PERCENT = 0.01
# Vpreice (:417-484)
VP = dict(ALPHAc=0.5, FA1c=0.601272523, FA2c=0.000342181855, FA3c=1.49236645e-12, WVP1c=3.6e10, WVP2c=6145.0,
          FVTHc=11713803.0, THOUBKc=7.24637701e18, SVOLc=3.23e-23, FDc=249.239822, FPIVOLc=3.89051704e23,
          Sa=2.349, Sb=259.0, B2exp=1.94)
# frachom (:488-523)
SETA = 6132.9                  # K
NBIN = 200
FHOM_NORM = 0.997
# --- nucleate_ice_cam / microp_aero under CAM6 -------------------------------------------------------------
MINCLD = 1.0e-4                # microp_aero.F90:73
SUBGRID = 1.2                  # nucleate_ice_subgrid(_strat), namelist_defaults_cam.xml:1264, :1268
STRAT = 1.0                    # nucleate_ice_strat (:1278)
T_NUC_MAX_OFFSET = 5.0         # t < tmelt - 5 (nucleate_ice_cam.F90:623)
WSUBI_MIN = 0.001              # microp_aero.F90:525, :545
WSUB_CAP = 10.0                # microp_aero.F90:533
TKE_PER_WP2 = 1.5              # clubb tke = 3/2 wp2 (microp_aero.F90:517)


def _svp(T):
    return saturation_vapor_pressure_goff_gratch(T), saturation_vapor_pressure_ice_goff_gratch(T)


def _hf(T, ww, RH, Na):
    """``hf``: homogeneous freezing number [#/cm3]; T in deg C."""
    lw = jnp.log(ww)
    A = HF["RHwA"][0] * lw + HF["RHwA"][1]
    B = HF["RHwB"][0] * lw + HF["RHwB"][1]
    C = HF["RHwC"][0] * lw + HF["RHwC"][1]
    RHw = (A * T * T + B * T + C) * PERCENT
    regm = HF["regm"][0] * lw + HF["regm"][1]
    warm64 = T > HF["T64"]
    A2f = jnp.where(warm64, HF["A21f"], HF["A22f"])
    B2f = jnp.where(warm64, HF["B21f"], HF["B22f"])
    k1f = jnp.exp(A2f + B2f * T + HF["C2f"] * lw)
    k2f = HF["A1f"] + HF["B1f"] * T + HF["C1f"] * lw
    k1s = jnp.exp(HF["A2s"] + (HF["B2s"] + HF["B3s"] * lw) * T + HF["C2s"] * lw)
    k2s = HF["A1s"] + HF["B1s"] * T + HF["C1s"] * lw
    Nf = jnp.minimum(k1f * Na ** k2f, Na)
    Ns = jnp.minimum(k1s * Na ** k2s, Na)
    on = (T <= TC_HOM) & (RH >= RHw)
    return jnp.where(on, jnp.where(T >= regm, Nf, Ns), 0.0)


def _hetero(T, ww, Ns):
    """``hetero``: immersion freezing number [#/cm3] (deposition nucleation off below -37 C)."""
    h = HET
    lN = jnp.log(Ns)
    B = (h["A11"] + h["B11"] * lN) * jnp.log(ww) + (h["A12"] + h["B12"] * lN)
    C = h["A21"] + h["B21"] * lN
    Nis = math.exp(h["A22"]) * Ns ** h["B22"] * jnp.exp(B * T) * ww ** C
    return jnp.minimum(Nis, Ns)


def _vpreice(P_in, T, R_in, C_in, S_in):
    """``Vpreice``: vertical velocity reduction by pre-existing ice [m/s]."""
    v = VP
    P = P_in * 1e-2  # coeff-ok: Pa -> hPa (nucleate_ice.F90 Vpreice)
    S = jnp.where(S_in < 1.0, v["Sa"] - T / v["Sb"], S_in)
    R = R_in * 1e2
    C = C_in * 1e-6
    T_1 = 1.0 / T
    PICE = v["WVP1c"] * jnp.exp(-(v["WVP2c"] * T_1))
    ALP4 = 0.25 * v["ALPHAc"]
    FLUX = ALP4 * jnp.sqrt(v["FVTHc"] * T)
    CISAT = v["THOUBKc"] * PICE * T_1
    A1 = (v["FA1c"] * T_1 - v["FA2c"]) * T_1
    A2 = 1.0 / CISAT
    A3 = v["FA3c"] * T_1 / P
    B1 = FLUX * v["SVOLc"] * CISAT * (S - 1.0)
    B2 = FLUX * v["FDc"] * P * T_1 ** v["B2exp"]
    DLOSS = v["FPIVOLc"] * C * B1 * R ** 2 / (1.0 + B2 * R)
    VICE = (A2 + A3 * S) * DLOSS / (A1 * S)
    return VICE * 1e-2  # coeff-ok: cm/s -> m/s (nucleate_ice.F90 Vpreice)


def _frachom(Tmean, RHimean, detaT):
    """``frachom``: fraction of the cloud reaching the homogeneous threshold.  The Fortran sums bins from
    the top down and exits at the first bin below threshold; Sbin rises with the bin index (detaT > 0),
    so that equals the sum over every bin at or above threshold, taken here in the same top-down order."""
    Sihom = VP["Sa"] - Tmean / VP["Sb"]
    i = jnp.arange(NBIN, 0, -1, dtype=Tmean.dtype)            # i = Nbin .. 1, the loop order
    deta = (i - 0.5 - NBIN / 2) * 6.0 / NBIN
    Sbin = RHimean[..., None] * jnp.exp(deta * detaT[..., None] * SETA / Tmean[..., None] ** 2.0)
    pdf = jnp.exp(-deta ** 2.0 / 2.0) * 6.0 / (jnp.sqrt(2.0 * math.pi) * NBIN)
    live = jnp.cumprod(Sbin >= Sihom[..., None], axis=-1).astype(bool)   # stops at the first failure
    fhom = jnp.zeros_like(Tmean)
    for k in range(NBIN):                                     # sequential, as the Fortran accumulates
        fhom = fhom + jnp.where(live[..., k], pdf[k], 0.0)
    return jnp.minimum(1.0, fhom / FHOM_NORM)


def nucleati(wbar, tair, pmid, relhum, cldn, qc, qi, ni_in, rhoair, so4_num, dst_num, soot_num, subgrid):
    """``nucleati`` for CAM6 settings.  Returns ``(nuci [1/kg], nihf, niimm, oso4_num)`` with nihf and niimm in
    #/cm3 (CAM's per-volume outputs before the final unit change) and oso4_num in #/cm3."""
    esl, esi = _svp(tair)
    RHimean = relhum * esl / esi * subgrid
    # pre-existing ice (:149-185)
    Ni_preice = ni_in * rhoair / jnp.maximum(MINCLD, cldn)
    pre = (Ni_preice > NI_PRE_MIN) & (qi > QI_PRE_MIN)
    lami = (GAMMA4 * CI * ni_in / jnp.where(pre, qi, 1.0)) ** (1.0 / 3.0)
    Ri = jnp.maximum(0.5 / jnp.where(pre, lami, 1.0), RI_PRE_MIN)
    wpice = jnp.where(pre, _vpreice(pmid, tair, Ri, Ni_preice, SHOM_PRE), 0.0)
    wpicehet = jnp.where(pre, _vpreice(pmid, tair, Ri, Ni_preice, SHET), 0.0)
    weff = jnp.maximum(wbar - wpice, MINWEFF)
    weffhet = jnp.maximum(wbar - wpicehet, MINWEFF)
    wbar1, wbar2 = weff, weffhet
    fhom = _frachom(tair, RHimean, wbar / DETA_T_W)
    tc = tair - constants.T_freeze
    dsum = soot_num + dst_num
    so4_ok = so4_num >= N_SMALL
    active = ((so4_ok | (dsum >= N_SMALL)) & (cldn > 0.0) & (RHimean >= RHI_ACT)
              & (((tc <= TC_MIXED_MAX) & (tc >= TC_HOM) & (qc < QC_SMALL)) | (tc <= TC_HOM)))
    A = REGM_A[0] * jnp.log(dsum) + REGM_A[1]
    B = REGM_B[0] * jnp.log(dsum) + REGM_B[1]
    regm = A * jnp.log(wbar1) + B
    rhs = relhum * subgrid
    # the homogeneous block repeated three times in the Fortran (:207-221, :232-243, :247-258)
    nihf_raw = _hf(tc, wbar1, rhs, so4_num)
    hom_occ = nihf_raw > NIHF_OCCUR
    hom_niimm = jnp.where(hom_occ, dsum, 0.0)
    hom_nihf = nihf_raw * fhom
    hom_oso4 = jnp.where(hom_occ, nihf_raw, 0.0) * fhom
    hom_n1 = hom_nihf + hom_niimm
    # heterogeneous at the actual temperature (:222-228)
    het_niimm = _hetero(tc, wbar2, dsum)
    # mixed regime (:259-271)
    mix_nihf_raw = _hf(regm - REGM_RAMP, wbar1, rhs, so4_num)
    mix_niimm = _hetero(regm, wbar2, dsum)
    mix_oso4 = jnp.where(mix_nihf_raw > NIHF_OCCUR, mix_nihf_raw, 0.0)
    ramp = ((regm - tc) / REGM_RAMP) ** 2
    mix_nihf = mix_nihf_raw * fhom * ramp
    mix_oso4 = mix_oso4 * fhom * ramp
    mix_n1 = mix_niimm + mix_nihf
    hom_only = (tc < TC_HOM_ONLY) & (wbar1 > W_HOM_ONLY)
    b1 = (tc > regm) | (so4_num < N_SMALL)
    b2 = (tc < regm - REGM_RAMP) | (dsum < N_SMALL)
    b1_hom = hom_only & so4_ok
    n1 = jnp.where(b1, jnp.where(b1_hom, hom_n1, het_niimm),
                   jnp.where(b2, hom_n1, jnp.where(hom_only, hom_n1, mix_n1)))
    nihf = jnp.where(b1, jnp.where(b1_hom, hom_nihf, 0.0),
                     jnp.where(b2, hom_nihf, jnp.where(hom_only, hom_nihf, mix_nihf)))
    niimm = jnp.where(b1, jnp.where(b1_hom, hom_niimm, het_niimm),
                      jnp.where(b2, hom_niimm, jnp.where(hom_only, hom_niimm, mix_niimm)))
    oso4 = jnp.where(b1, jnp.where(b1_hom, hom_oso4, 0.0),
                     jnp.where(b2, hom_oso4, jnp.where(hom_only, hom_oso4, mix_oso4)))
    ni = jnp.where(active, n1 + Ni_preice * 1e-6, 0.0)
    nihf = jnp.where(active, nihf, 0.0)
    niimm = jnp.where(active, niimm, 0.0)
    oso4 = jnp.where(active, oso4, 0.0)
    nuci = jnp.where((ni > NUCI_MAX) | (ni < 0.0), 0.0, ni)  # nimey = 0 under use_hetfrz_classnuc
    return nuci * PER_CM3 / rhoair, nihf, niimm, oso4


def wsubi_from_wp2(wp2_iface, top_lev):
    """``microp_aero.F90`` wsubi on layers from CLUBB wp2 on interfaces (use_preexisting_ice: no 0.2 cap)."""
    tke = TKE_PER_WP2 * wp2_iface
    arg = 0.5 * (tke[:, :-1] + tke[:, 1:]) * (2.0 / 3.0)
    wsub = jnp.minimum(jnp.sqrt(jnp.maximum(arg, 0.0)), WSUB_CAP)
    above = jnp.arange(arg.shape[1]) < top_lev
    return jnp.where(above, WSUBI_MIN, jnp.maximum(WSUBI_MIN, wsub))


def cam6_naai(T, q_v, pmid, q_c, q_i, N_i, ast, wsubi, num_accum, num_aitken, num_coarse,
              coarse_dust, coarse_nacl, coarse_so4, troplev, top_lev):
    """``nucleate_ice_cam_calc`` per level for the modal aerosol (separate_dust false: the coarse mode holds
    dust, sea salt and sulfate).  All fields (ncol, nlev), top first; aerosol numbers [1/kg], masses [kg/kg];
    ``troplev`` (ncol,) 0-based CAM tropopause layer; ``top_lev`` 0-based.  Returns naai [1/kg]."""
    rho = pmid / (constants.R_d * T)
    esl = saturation_vapor_pressure_goff_gratch(T)
    eps = constants.epsilon
    qs = jnp.where(pmid - esl <= 0.0, 1.0, eps * esl / jnp.where(pmid - esl <= 0.0, 1.0, pmid - (1.0 - eps) * esl))
    relhum = q_v / qs
    icldm = jnp.maximum(ast, MINCLD)
    to_cm3 = rho * 1.0e-6
    dmc, ssmc, so4mc = coarse_dust * rho, coarse_nacl * rho, coarse_so4 * rho
    tot = ssmc + dmc + so4mc
    dst_num = jnp.where(dmc > 0.0, dmc / jnp.where(dmc > 0.0, tot, 1.0) * num_coarse * to_cm3, 0.0)
    so4_num_cr = jnp.where(so4mc > 0.0, so4mc / jnp.where(so4mc > 0.0, tot, 1.0) * num_coarse * to_cm3, 0.0)
    so4_num = num_aitken * to_cm3
    naai, _, _, oso4 = nucleati(wsubi, T, pmid, relhum, icldm, q_c, q_i, N_i, rho,
                                so4_num, dst_num, jnp.zeros_like(T), SUBGRID)
    k = jnp.arange(T.shape[1])
    strat = (k[None, :] < troplev[:, None]) & (oso4 > 0.0)
    so4_num_ac = num_accum * to_cm3
    dso4 = jnp.maximum(0.0, STRAT * (so4_num_cr + so4_num_ac) - oso4) * PER_CM3 / rho
    naai = jnp.where(strat, naai + dso4, naai)
    on = (k[None, :] >= top_lev) & (T < constants.T_freeze - T_NUC_MAX_OFFSET)
    return jnp.where(on, naai, 0.0)
