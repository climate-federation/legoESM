"""MARBL autotroph physiology (CESM2 PFTs: sp, diat, diaz).

Ports, from ``marbl_interior_tendency_mod.F90``:
compute_autotroph_elemental_ratios (fixed N:C, variable P:C),
compute_temperature_functional_form (q_10), compute_Pprime,
compute_autotroph_uptake, compute_autotroph_photosynthesis (Geider et al.
1998 with photoacclimation), compute_autotroph_nutrient_uptake,
compute_autotroph_calcification (implicit calcifier), compute_autotroph_nfixation
and compute_autotroph_loss.

All per-PFT arrays are stacked on a leading axis of length 3 (sp, diat, diaz):
shape (3, ..., nlev).  PAR arrays carry a sub-column axis: (n_subcol, ..., nlev).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.biogeochemistry.marbl.config import (
    DIAT, DIAZ, SP, MarblConfig, per_pft,
)

# MARBL marbl_constants_mod
EPS_C = 1.0e-8        # small C concentration [mmol C/m^3]
EPS_T_INV = 3.17e-8   # small inverse time scale [1/s]

_DPS = 1.0 / constants.seconds_per_day


def pft_mask(index: int, ndim_trailing: int, dtype) -> jnp.ndarray:
    """1.0 at PFT ``index``, 0.0 elsewhere; shape (3, 1, ..., 1)."""
    return per_pft(tuple(1.0 if i == index else 0.0 for i in range(3)), ndim_trailing, dtype)


def temperature_function(T: jnp.ndarray, Tref: float, Q_10: float) -> jnp.ndarray:
    """Q10 scaling of physiological rates (MARBL temp_func_form 'q_10')."""
    return Q_10 ** ((T - Tref) / 10.0)


def loss_threshold_factor(zt: jnp.ndarray, z1: float, z2: float) -> jnp.ndarray:
    """1 above z1, 0 below z2, linear in between (compute_Pprime / compute_Zprime)."""
    return jnp.clip((z2 - zt) / (z2 - z1), 0.0, 1.0)


class AutotrophTerms(NamedTuple):
    thetaC: jnp.ndarray
    Qfe: jnp.ndarray
    Qsi: jnp.ndarray
    Qp: jnp.ndarray
    QCaCO3: jnp.ndarray
    gQfe: jnp.ndarray
    gQsi: jnp.ndarray
    gQp: jnp.ndarray
    Pprime: jnp.ndarray
    VNO3: jnp.ndarray
    VNH4: jnp.ndarray
    VPO4: jnp.ndarray
    VDOP: jnp.ndarray
    f_nut: jnp.ndarray
    light_lim: jnp.ndarray
    photoC: jnp.ndarray
    photoacc: jnp.ndarray
    NO3_V: jnp.ndarray
    NH4_V: jnp.ndarray
    PO4_V: jnp.ndarray
    DOP_V: jnp.ndarray
    photoFe: jnp.ndarray
    photoSi: jnp.ndarray
    CaCO3_form: jnp.ndarray
    Nfix: jnp.ndarray
    Nexcrete: jnp.ndarray
    auto_loss: jnp.ndarray
    auto_loss_poc: jnp.ndarray
    auto_loss_dic: jnp.ndarray
    auto_loss_doc: jnp.ndarray
    auto_agg: jnp.ndarray


def zero_consistency(C, Chl, P, Fe, Si, CaCO3):
    """If any pool of a PFT is zero, zero all of its pools (autotroph_zero_consistency_enforce).

    Si / CaCO3 enter the test only for the silicifier / calcifier.
    """
    trailing = C.ndim - 1
    is_si = pft_mask(DIAT, trailing, C.dtype) > 0.0
    zero = (Chl == 0.0) | (C == 0.0) | (P == 0.0) | (Fe == 0.0) | (is_si & (Si == 0.0))
    keep = jnp.where(zero, 0.0, 1.0)
    return C * keep, Chl * keep, P * keep, Fe * keep, Si * keep, CaCO3 * keep


def compute_autotrophs(
    C: jnp.ndarray, Chl: jnp.ndarray, P: jnp.ndarray, Fe_auto: jnp.ndarray,
    Si: jnp.ndarray, CaCO3: jnp.ndarray,
    PO4: jnp.ndarray, DOP: jnp.ndarray, NO3: jnp.ndarray, NH4: jnp.ndarray,
    Fe: jnp.ndarray, SiO3: jnp.ndarray,
    temperature: jnp.ndarray, zt: jnp.ndarray,
    PAR_avg: jnp.ndarray, PAR_col_frac: jnp.ndarray,
    cfg: MarblConfig,
) -> AutotrophTerms:
    """Autotroph growth, uptake and loss terms.

    Autotroph pools C, Chl, P, Fe_auto, Si, CaCO3 are (3, ..., nlev) with Si
    nonzero only for diatoms and CaCO3 only for sp.  Nutrients, temperature
    [degC, in situ] and zt [m, layer-centre depth] are (..., nlev).
    PAR_avg is (n_subcol, ..., nlev), PAR_col_frac (n_subcol, ...).
    """
    trailing = C.ndim - 1
    dt = C.dtype
    pp = lambda v: per_pft(v, trailing, dt)  # noqa: E731
    is_sp = pft_mask(SP, trailing, dt)
    is_diat = pft_mask(DIAT, trailing, dt)
    is_diaz = pft_mask(DIAZ, trailing, dt)

    # --- elemental ratios -------------------------------------------------------
    inv_C = 1.0 / (C + EPS_C)
    thetaC = Chl * inv_C
    Qfe = Fe_auto * inv_C
    Qp = P * inv_C
    Qsi = is_diat * jnp.minimum(Si * inv_C, cfg.gQsi_max)
    QCaCO3 = is_sp * jnp.minimum(CaCO3 * inv_C, cfg.QCaCO3_max)

    FeOpt = pp(cfg.FeOpt)
    gQfe_max = pp(cfg.gQfe_max)
    gQfe = jnp.where(Fe < FeOpt,
                     jnp.maximum(gQfe_max * Fe / FeOpt, pp(cfg.gQfe_min)),
                     gQfe_max)

    SiOpt = pp(cfg.SiOpt)
    Fe_safe = jnp.where(Fe > 0.0, Fe, 1.0)
    gQsi = jnp.where((Fe < FeOpt) & (Fe > 0.0) & (SiO3 > SiOpt),
                     jnp.minimum(cfg.gQsi_0 * FeOpt / Fe_safe, cfg.gQsi_max),
                     jnp.where(Fe == 0.0, cfg.gQsi_max, cfg.gQsi_0))
    SiOpt_safe = jnp.where(SiOpt > 0.0, SiOpt, 1.0)
    gQsi = jnp.where(SiO3 < SiOpt,
                     jnp.maximum(gQsi * SiO3 / SiOpt_safe, cfg.gQsi_min), gQsi)
    gQsi = is_diat * gQsi

    POpt = pp(cfg.POpt)
    gQp_max = pp(cfg.gQp_max)
    Ptot = PO4 + DOP
    gQp = jnp.where(Ptot < POpt,
                    jnp.maximum(gQp_max * Ptot / POpt, pp(cfg.gQp_min)),
                    gQp_max)
    gQn = cfg.Q

    # --- temperature, loss threshold ------------------------------------------
    Tfunc = temperature_function(temperature, cfg.Tref_auto, cfg.Q_10)
    f_loss = loss_threshold_factor(zt, cfg.thres_z1_auto, cfg.thres_z2_auto)
    cold = temperature < pp(cfg.temp_thres)
    C_loss_thres = f_loss * jnp.where(cold, pp(cfg.loss_thres2), pp(cfg.loss_thres))
    Pprime = jnp.maximum(C - C_loss_thres, 0.0)

    # --- nutrient limitation --------------------------------------------------------
    nNO3 = NO3 / pp(cfg.kNO3)
    nNH4 = NH4 / pp(cfg.kNH4)
    VNO3 = nNO3 / (1.0 + nNO3 + nNH4)
    VNH4 = nNH4 / (1.0 + nNO3 + nNH4)
    VNtot = jnp.where(is_diaz > 0.0, 1.0, VNO3 + VNH4)
    VFe = Fe / (Fe + pp(cfg.kFe))
    nPO4 = PO4 / pp(cfg.kPO4)
    nDOP = DOP / pp(cfg.kDOP)
    VPO4 = nPO4 / (1.0 + nPO4 + nDOP)
    VDOP = nDOP / (1.0 + nPO4 + nDOP)
    VPtot = VPO4 + VDOP
    kSiO3_safe = jnp.where(is_diat > 0.0, pp(cfg.kSiO3), 1.0)
    VSiO3 = jnp.where(is_diat > 0.0, SiO3 / (SiO3 + kSiO3_safe), 1.0)
    f_nut = jnp.minimum(jnp.minimum(jnp.minimum(VNtot, VFe), VPtot), VSiO3)

    # --- photosynthesis (Geider et al. 1998), per PAR sub-column ----------------
    PCmax = jnp.where(cold, 0.0, pp(cfg.PCref_per_day) * _DPS * f_nut * Tfunc)
    alphaPI = pp(cfg.alphaPI_per_day) * _DPS
    work = alphaPI[:, None] * thetaC[:, None] * PAR_avg[None]       # (3, nsub, ..., nlev)
    lit = work > 0.0
    light_lim_sub = jnp.where(lit, -jnp.expm1(-work / (PCmax[:, None] + EPS_T_INV)), 0.0)
    PCphoto_sub = PCmax[:, None] * light_lim_sub
    work_safe = jnp.where(lit, work, 1.0)
    pChl_sub = jnp.where(lit, pp(cfg.thetaN_max)[:, None] * PCphoto_sub / work_safe, 0.0)
    # photoacc = pChl PCphoto gQn / thetaC * Chl, with Chl / thetaC = C + epsC
    photoacc_sub = pChl_sub * PCphoto_sub * gQn * (C + EPS_C)[:, None]
    w = PAR_col_frac[None, ..., None]                                 # (1, nsub, ..., 1)
    has_chl = thetaC > 0.0
    light_lim = jnp.where(has_chl, jnp.sum(w * light_lim_sub, axis=1), 0.0)
    PCphoto = jnp.where(has_chl, jnp.sum(w * PCphoto_sub, axis=1), 0.0)
    photoacc = jnp.where(has_chl, jnp.sum(w * photoacc_sub, axis=1), 0.0)
    photoC = PCphoto * C

    # --- nutrient uptake ---------------------------------------------------------------
    VNtot_safe = jnp.where(VNtot > 0.0, VNtot, 1.0)
    N_uptake = photoC * gQn
    NO3_V = jnp.where(VNtot > 0.0, VNO3 / VNtot_safe * N_uptake, 0.0)
    NH4_V = jnp.where(VNtot > 0.0, VNH4 / VNtot_safe * N_uptake, 0.0)
    VPtot_safe = jnp.where(VPtot > 0.0, VPtot, 1.0)
    P_uptake = photoC * gQp
    PO4_V = jnp.where(VPtot > 0.0, VPO4 / VPtot_safe * P_uptake, 0.0)
    DOP_V = jnp.where(VPtot > 0.0, VDOP / VPtot_safe * P_uptake, 0.0)
    photoFe = photoC * gQfe
    photoSi = photoC * gQsi

    # --- calcification (sp, implicit calcifier) ------------------------------------
    CaCO3_form = cfg.parm_f_prod_sp_CaCO3 * photoC * f_nut * f_nut
    CaCO3_form = jnp.where(
        temperature < cfg.CaCO3_temp_thres1,
        CaCO3_form * jnp.maximum(temperature - cfg.CaCO3_temp_thres2, 0.0)
        / (cfg.CaCO3_temp_thres1 - cfg.CaCO3_temp_thres2),
        CaCO3_form)
    CaCO3_form = jnp.where(
        C > cfg.CaCO3_sp_thres,
        jnp.minimum(CaCO3_form * C / cfg.CaCO3_sp_thres, cfg.f_photosp_CaCO3 * photoC),
        CaCO3_form)
    CaCO3_form = is_sp * CaCO3_form

    # --- N fixation (diaz) --------------------------------------------------------------
    Nfix = is_diaz * (N_uptake * cfg.r_Nfix_photo - NO3_V - NH4_V)
    Nexcrete = is_diaz * (Nfix + NO3_V + NH4_V - N_uptake)

    # --- losses ---------------------------------------------------------------------------
    auto_loss = pp(cfg.mort_per_day) * _DPS * Pprime * Tfunc
    auto_agg = jnp.minimum(pp(cfg.agg_rate_max) * _DPS * Pprime,
                           pp(cfg.mort2_per_day) * _DPS * Pprime ** cfg.auto_mort2_exp)
    auto_agg = jnp.maximum(pp(cfg.agg_rate_min) * _DPS * Pprime, auto_agg)
    auto_loss_poc = jnp.where(is_sp > 0.0, QCaCO3 * auto_loss, pp(cfg.loss_poc) * auto_loss)
    auto_loss_doc = (1.0 - cfg.parm_labile_ratio) * (auto_loss - auto_loss_poc)
    auto_loss_dic = cfg.parm_labile_ratio * (auto_loss - auto_loss_poc)

    return AutotrophTerms(
        thetaC=thetaC, Qfe=Qfe, Qsi=Qsi, Qp=Qp, QCaCO3=QCaCO3,
        gQfe=gQfe, gQsi=gQsi, gQp=gQp, Pprime=Pprime,
        VNO3=VNO3, VNH4=VNH4, VPO4=VPO4, VDOP=VDOP, f_nut=f_nut,
        light_lim=light_lim, photoC=photoC, photoacc=photoacc,
        NO3_V=NO3_V, NH4_V=NH4_V, PO4_V=PO4_V, DOP_V=DOP_V,
        photoFe=photoFe, photoSi=photoSi, CaCO3_form=CaCO3_form,
        Nfix=Nfix, Nexcrete=Nexcrete,
        auto_loss=auto_loss, auto_loss_poc=auto_loss_poc,
        auto_loss_dic=auto_loss_dic, auto_loss_doc=auto_loss_doc, auto_agg=auto_agg,
    )
