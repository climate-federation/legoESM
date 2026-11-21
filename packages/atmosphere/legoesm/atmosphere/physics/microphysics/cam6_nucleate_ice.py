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

Every branch of the Fortran is kept, in the same order.  KNOWN DIFFERENCE (RULE 5): each switch on model
state is a one-sided C1 ramp (:func:`ge_ramp` / :func:`le_ramp`) that equals the Fortran exactly on the side the
condition holds and blends over one width on the other side: temperature thresholds over ``wT`` [K]
(``MorrisonConfig.cirrus_T_width_K``, default 0.1 K), relative-humidity, updraft, ice and number thresholds
over ``wr`` times the threshold (``MorrisonConfig.cirrus_rel_width``, default 0.01).  ``wT = wr = 0`` is the
Fortran's hard switches (the oracle test).  Switches on the prescribed aerosol (not model state), static
level indices (incl. the CAM chem tropopause level, stop-gradient'd as on main's CLUBB lane) and division
guards stay hard.  ``oso4 > 0`` (a non-negative number) ramps on the side it holds (:func:`_pos`).  Where no dust is present the Fortran's regime threshold is
NaN and compares false; here it is replaced by a large value that does the same, so gradients stay finite.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.smooth import smoothstep_cubic
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
    "differentiable": True,
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


_REGM_NO_DUST = 1.0e3            # stands in for the Fortran's NaN regime threshold (compares false)


def ge_ramp(x, thr, width, scale=1.0):
    """1 where ``x >= thr`` (exactly, as the Fortran), a C1 ramp from 0 at ``thr - width*scale`` below;
    ``width <= 0`` (a Python float) is the hard switch."""
    if width <= 0.0:
        return (x >= thr).astype(jnp.result_type(x, float))
    return smoothstep_cubic((x - thr) / (width * scale) + 1.0)


def le_ramp(x, thr, width, scale=1.0):
    """1 where ``x <= thr``, a C1 ramp to 0 at ``thr + width*scale`` above; ``width <= 0`` is hard."""
    return ge_ramp(-x, -thr, width, scale)


def _pos(x, width, scale):
    """``x > 0`` for ``x >= 0``: there is no off side to ramp on, so a C1 ramp from 0 at 0 to 1 at
    ``width*scale`` (the one switch here whose gap is on the side the condition holds)."""
    if width <= 0.0:
        return (x > 0.0).astype(jnp.result_type(x, float))
    return smoothstep_cubic(x / (width * scale))


def _pick(s, a, b):
    """``a`` where the switch weight ``s`` is 1, ``b`` where 0, blended between."""
    return b + s * (a - b)


def _svp(T):
    return saturation_vapor_pressure_goff_gratch(T), saturation_vapor_pressure_ice_goff_gratch(T)


def _hf(T, ww, RH, Na, wT, wr):
    """``hf``: homogeneous freezing number [#/cm3]; T in deg C."""
    lw = jnp.log(ww)
    A = HF["RHwA"][0] * lw + HF["RHwA"][1]
    B = HF["RHwB"][0] * lw + HF["RHwB"][1]
    C = HF["RHwC"][0] * lw + HF["RHwC"][1]
    RHw = (A * T * T + B * T + C) * PERCENT
    regm = HF["regm"][0] * lw + HF["regm"][1]
    warm64 = ge_ramp(T, HF["T64"], wT)
    A2f = _pick(warm64, HF["A21f"], HF["A22f"])
    B2f = _pick(warm64, HF["B21f"], HF["B22f"])
    k1f = jnp.exp(A2f + B2f * T + HF["C2f"] * lw)
    k2f = HF["A1f"] + HF["B1f"] * T + HF["C1f"] * lw
    k1s = jnp.exp(HF["A2s"] + (HF["B2s"] + HF["B3s"] * lw) * T + HF["C2s"] * lw)
    k2s = HF["A1s"] + HF["B1s"] * T + HF["C1s"] * lw
    Nf = jnp.minimum(k1f * Na ** k2f, Na)
    Ns = jnp.minimum(k1s * Na ** k2s, Na)
    on = le_ramp(T, TC_HOM, wT) * ge_ramp(RH, RHw, wr, jnp.abs(RHw))
    return on * _pick(ge_ramp(T, regm, wT), Nf, Ns)


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


def _frachom(Tmean, RHimean, detaT, wr):
    """``frachom``: fraction of the cloud reaching the homogeneous threshold.  The Fortran sums bins from
    the top down and exits at the first bin below threshold; Sbin rises with the bin index (detaT > 0),
    so that equals the sum over every bin at or above threshold, taken here in the same top-down order."""
    Sihom = VP["Sa"] - Tmean / VP["Sb"]
    i = jnp.arange(NBIN, 0, -1, dtype=Tmean.dtype)            # i = Nbin .. 1, the loop order
    deta = (i - 0.5 - NBIN / 2) * 6.0 / NBIN
    Sbin = RHimean[..., None] * jnp.exp(deta * detaT[..., None] * SETA / Tmean[..., None] ** 2.0)
    pdf = jnp.exp(-deta ** 2.0 / 2.0) * 6.0 / (jnp.sqrt(2.0 * math.pi) * NBIN)
    live = jnp.cumprod(ge_ramp(Sbin, Sihom[..., None], wr, jnp.abs(Sihom[..., None])),
                       axis=-1)                               # stops at the first failure
    fhom = jnp.zeros_like(Tmean)
    for k in range(NBIN):                                     # sequential, as the Fortran accumulates
        fhom = fhom + live[..., k] * pdf[k]
    return jnp.minimum(1.0, fhom / FHOM_NORM)


def nucleati(wbar, tair, pmid, relhum, cldn, qc, qi, ni_in, rhoair, so4_num, dst_num, soot_num, subgrid,
             wT=0.0, wr=0.0):
    """``nucleati`` for CAM6 settings.  Returns ``(nuci [1/kg], nihf, niimm, oso4_num)`` with nihf and niimm in
    #/cm3 (CAM's per-volume outputs before the final unit change) and oso4_num in #/cm3."""
    esl, esi = _svp(tair)
    RHimean = relhum * esl / esi * subgrid
    # pre-existing ice (:149-185)
    Ni_preice = ni_in * rhoair / jnp.maximum(MINCLD, cldn)
    # the Fortran's strict > is the >= ramp here: identical except exactly at the threshold
    pre = ge_ramp(Ni_preice, NI_PRE_MIN, wr, NI_PRE_MIN) * ge_ramp(qi, QI_PRE_MIN, wr, QI_PRE_MIN)
    live = pre > 0.0
    qi_s = jnp.where(live, qi, 1.0)
    ni_s = jnp.where(live, ni_in, 1.0)
    lami = (GAMMA4 * CI * ni_s / qi_s) ** (1.0 / 3.0)
    Ri = jnp.maximum(0.5 / lami, RI_PRE_MIN)
    Ni_s = jnp.where(live, Ni_preice, 1.0)
    wpice = jnp.where(live, pre * _vpreice(pmid, tair, Ri, Ni_s, SHOM_PRE), 0.0)
    wpicehet = jnp.where(live, pre * _vpreice(pmid, tair, Ri, Ni_s, SHET), 0.0)
    weff = jnp.maximum(wbar - wpice, MINWEFF)
    weffhet = jnp.maximum(wbar - wpicehet, MINWEFF)
    wbar1, wbar2 = weff, weffhet
    fhom = _frachom(tair, RHimean, wbar / DETA_T_W, wr)
    tc = tair - constants.T_freeze
    dsum = soot_num + dst_num
    # aerosol (prescribed, not model state): hard, as the Fortran
    so4_ok = so4_num >= N_SMALL
    dust_ok = dsum >= N_SMALL
    aer_ok = (so4_ok | dust_ok).astype(tair.dtype)
    # safe aerosol numbers where a mode is absent: the branches that read them are not selected there
    so4_s = jnp.where(so4_ok, so4_num, 1.0)
    dsum_s = jnp.where(dust_ok, dsum, 1.0)
    # cldn = max(ast, mincld) > 0 always: the Fortran's cldn > 0 test cannot fail and is dropped
    mixed = le_ramp(tc, TC_MIXED_MAX, wT) * ge_ramp(tc, TC_HOM, wT) * le_ramp(qc, QC_SMALL, wr, QC_SMALL)
    cold = le_ramp(tc, TC_HOM, wT)
    active = aer_ok * ge_ramp(RHimean, RHI_ACT, wr, RHI_ACT) * (mixed + cold - mixed * cold)
    A = REGM_A[0] * jnp.log(dsum_s) + REGM_A[1]
    B = REGM_B[0] * jnp.log(dsum_s) + REGM_B[1]
    regm = jnp.where(dust_ok, A * jnp.log(wbar1) + B, _REGM_NO_DUST)
    rhs = relhum * subgrid
    # the homogeneous block repeated three times in the Fortran (:207-221, :232-243, :247-258)
    nihf_raw = jnp.where(so4_ok, _hf(tc, wbar1, rhs, so4_s, wT, wr), 0.0)
    hom_occ = ge_ramp(nihf_raw, NIHF_OCCUR, wr, NIHF_OCCUR)
    hom_niimm = hom_occ * dsum
    hom_nihf = nihf_raw * fhom
    hom_oso4 = hom_occ * nihf_raw * fhom
    hom_n1 = hom_nihf + hom_niimm
    # heterogeneous at the actual temperature (:222-228)
    het_niimm = jnp.where(dust_ok, _hetero(tc, wbar2, dsum_s), 0.0)
    # mixed regime (:259-271)
    mix_nihf_raw = jnp.where(so4_ok, _hf(regm - REGM_RAMP, wbar1, rhs, so4_s, wT, wr), 0.0)
    mix_niimm = jnp.where(dust_ok, _hetero(regm, wbar2, dsum_s), 0.0)
    mix_oso4 = ge_ramp(mix_nihf_raw, NIHF_OCCUR, wr, NIHF_OCCUR) * mix_nihf_raw
    ramp = ((regm - tc) / REGM_RAMP) ** 2
    mix_nihf = mix_nihf_raw * fhom * ramp
    mix_oso4 = mix_oso4 * fhom * ramp
    mix_n1 = mix_niimm + mix_nihf
    hom_only = le_ramp(tc, TC_HOM_ONLY, wT) * ge_ramp(wbar1, W_HOM_ONLY, wr, W_HOM_ONLY)
    so4_small = 1.0 - so4_ok.astype(tair.dtype)
    dust_small = 1.0 - dust_ok.astype(tair.dtype)
    b1t = ge_ramp(tc, regm, wT)
    b1 = b1t + so4_small - b1t * so4_small
    b2t = le_ramp(tc, regm - REGM_RAMP, wT)
    b2 = b2t + dust_small - b2t * dust_small
    b1_hom = hom_only * so4_ok.astype(tair.dtype)

    def tree(v_hom, v_het, v_mix):
        return _pick(b1, _pick(b1_hom, v_hom, v_het),
                     _pick(b2, v_hom, _pick(hom_only, v_hom, v_mix)))

    n1 = tree(hom_n1, het_niimm, mix_n1)
    nihf = tree(hom_nihf, 0.0, mix_nihf)
    niimm = tree(hom_niimm, het_niimm, mix_niimm)
    oso4 = tree(hom_oso4, 0.0, mix_oso4)
    ni = active * (n1 + Ni_preice * 1e-6)
    nihf = active * nihf
    niimm = active * niimm
    oso4 = active * oso4
    # ni >= 0 always; the cap at 9999/cm3 zeroes (nimey = 0 under use_hetfrz_classnuc)
    nuci = ni * (1.0 - ge_ramp(ni, NUCI_MAX, wr, NUCI_MAX))
    return nuci * PER_CM3 / rhoair, nihf, niimm, oso4


def wsubi_from_wp2(wp2_iface, top_lev):
    """``microp_aero.F90`` wsubi on layers from CLUBB wp2 on interfaces (use_preexisting_ice: no 0.2 cap)."""
    tke = TKE_PER_WP2 * wp2_iface
    arg = 0.5 * (tke[:, :-1] + tke[:, 1:]) * (2.0 / 3.0)
    wsub = jnp.minimum(jnp.sqrt(jnp.maximum(arg, 0.0)), WSUB_CAP)
    above = jnp.arange(arg.shape[1]) < top_lev
    return jnp.where(above, WSUBI_MIN, jnp.maximum(WSUBI_MIN, wsub))


def cam6_naai(T, q_v, pmid, q_c, q_i, N_i, ast, wsubi, num_accum, num_aitken, num_coarse,
              coarse_dust, coarse_nacl, coarse_so4, troplev, top_lev, wT=0.0, wr=0.0):
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
                                so4_num, dst_num, jnp.zeros_like(T), SUBGRID, wT, wr)
    k = jnp.arange(T.shape[1])
    so4_num_ac = num_accum * to_cm3
    so4_strat = STRAT * (so4_num_cr + so4_num_ac)
    # oso4 > 0 ramped over wr times the stratospheric sulfate number it is compared with
    strat = (k[None, :] < troplev[:, None]) * _pos(
        oso4, wr, jnp.where(so4_strat > 0.0, so4_strat, 1.0))
    dso4 = jnp.maximum(0.0, so4_strat - oso4) * PER_CM3 / rho
    naai = naai + strat * dso4
    on = (k[None, :] >= top_lev) * le_ramp(T, constants.T_freeze - T_NUC_MAX_OFFSET, wT)
    return on * naai
