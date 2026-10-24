"""MARBL interior tendency (marbl_interior_tendency_compute) for the base tracers.

Call order follows MARBL: local tracer copies (negatives as zero), in-situ
temperature, carbonate chemistry, PAR, autotrophs, zooplankton / grazing /
routing, DOM, sinking particles with iron scavenging and sediments, ligand,
nitrification, denitrification, then compute_local_tendencies.

Not ported (not used by the CESM2 base configuration in a 1-D column):
carbon isotopes, abiotic DIC, tracer restoring, the o2_consumption_scalef /
p_remin_scalef forcings (both 1), bury-coefficient adjustment
(ladjust_bury_coeff = .false.), and dust production from sediment / vent Fe
(MARBL multiplies it by zero).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.biogeochemistry.marbl.carbonate_marbl import (
    carbonate_coefficients,
    carbonate_speciation,
    co3_saturation,
)
from legoesm.ocean.biogeochemistry.marbl.autotrophs import (
    compute_autotrophs,
    zero_consistency,
)
from legoesm.ocean.biogeochemistry.marbl.config import (
    MarblConfig,
    MarblInteriorDiagnostics,
    MarblInteriorForcing,
    MarblTracers,
)
from legoesm.ocean.biogeochemistry.marbl.dom import compute_dom
from legoesm.ocean.biogeochemistry.marbl.iron import (
    compute_ligand,
    doc_scavenging_factor,
    iron_speciation,
)
from legoesm.ocean.biogeochemistry.marbl.light import compute_par
from legoesm.ocean.biogeochemistry.marbl.nitrogen import (
    compute_denitrification,
    compute_nitrification,
)
from legoesm.ocean.biogeochemistry.marbl.particles import (
    ParticleProduction,
    compute_particles,
    default_bot_flux_to_tend,
)
from legoesm.ocean.biogeochemistry.marbl.surface import MASS_TO_VOL
from legoesm.ocean.biogeochemistry.marbl.zooplankton import compute_zooplankton
from legoesm.ocean.eos import potential_temperature


def redfield_o2_ratios(cfg: MarblConfig) -> tuple:
    """(C:O2 production, C:O2 remineralization, C:O2 production by diazotrophs).

    marbl_settings_set_all_derived with lvariable_NtoC = .false.:
    Red_D_O2_P = Remin_D_O2_P + 32, diazotroph O2:P = 150.
    """
    red_D_O2_P = cfg.parm_Remin_D_O2_P + 32.0
    return (cfg.parm_Red_D_C_P / red_D_O2_P,
            cfg.parm_Red_D_C_P / cfg.parm_Remin_D_O2_P,
            cfg.parm_Red_D_C_P / 150.0)


def marbl_interior_tendencies(
    tracers: MarblTracers,
    forcing: MarblInteriorForcing,
    dz: jnp.ndarray,
    zt: jnp.ndarray,
    zw: jnp.ndarray,
    kmt: jnp.ndarray,
    cfg: MarblConfig,
    bot_flux_to_tend: jnp.ndarray | None = None,
    n_iter: int = 50,
) -> tuple[MarblTracers, MarblInteriorDiagnostics]:
    """Source/sink tendencies [conc/s] of the 32 MARBL tracers.

    Parameters
    ----------
    tracers : MarblTracers, each (..., nlev)
    forcing : MarblInteriorForcing
    dz, zt, zw : (..., nlev) layer thickness, centre depth, bottom-interface depth [m, > 0]
    kmt : (...,) int, number of active levels
    bot_flux_to_tend : (..., nlev) weights converting a bottom flux to a tendency,
        sum(dz * bot_flux_to_tend) = 1.  Default: all into the deepest active layer.
    n_iter : carbonate solver iterations (static).
    """
    if bot_flux_to_tend is None:
        bot_flux_to_tend = default_bot_flux_to_tend(dz, kmt)
    nlev = dz.shape[-1]
    active = jnp.arange(nlev) < jnp.asarray(kmt)[..., None]
    t = jax.tree_util.tree_map(lambda a: jnp.where(active, jnp.maximum(a, 0.0), 0.0), tracers)

    # --- autotroph pools stacked (sp, diat, diaz) -------------------------------------
    zero = jnp.zeros_like(t.spC)
    C, Chl, P, Fe_auto, Si, CaCO3 = zero_consistency(
        jnp.stack([t.spC, t.diatC, t.diazC]),
        jnp.stack([t.spChl, t.diatChl, t.diazChl]),
        jnp.stack([t.spP, t.diatP, t.diazP]),
        jnp.stack([t.spFe, t.diatFe, t.diazFe]),
        jnp.stack([zero, t.diatSi, zero]),
        jnp.stack([t.spCaCO3, zero, zero]),
    )

    # --- in-situ temperature, carbonate chemistry --------------------------------------
    S = forcing.salinity
    p_bar = forcing.pressure_bar
    T = potential_temperature(S, forcing.potemp, jnp.zeros_like(p_bar), p_bar)
    coeffs = carbonate_coefficients(T, S, p_bar)
    to_kg = 1.0 / MASS_TO_VOL
    spec = carbonate_speciation(t.DIC * to_kg, t.ALK * to_kg, t.PO4 * to_kg, t.SiO3 * to_kg,
                                coeffs, n_iter=n_iter)
    spec_alt = carbonate_speciation(t.DIC_ALT_CO2 * to_kg, t.ALK_ALT_CO2 * to_kg,
                                    t.PO4 * to_kg, t.SiO3 * to_kg, coeffs, n_iter=n_iter)
    CO3 = spec.CO3 * MASS_TO_VOL
    CO3_alt = spec_alt.CO3 * MASS_TO_VOL
    co3_sat_calc, co3_sat_arag = co3_saturation(T, S, p_bar)
    co3_sat_calc = co3_sat_calc * MASS_TO_VOL
    co3_sat_arag = co3_sat_arag * MASS_TO_VOL

    # --- light ------------------------------------------------------------------------------
    PAR = compute_par(forcing.surf_shortwave, forcing.PAR_col_frac, jnp.sum(Chl, axis=0), dz, cfg)

    # --- plankton -----------------------------------------------------------------------------
    auto = compute_autotrophs(C, Chl, P, Fe_auto, Si, CaCO3,
                              t.PO4, t.DOP, t.NO3, t.NH4, t.Fe, t.SiO3,
                              T, zt, PAR.avg, PAR.col_frac, cfg)
    zoo, grz = compute_zooplankton(t.zooC, auto.Pprime, auto.QCaCO3, auto.Qp, auto.Qfe,
                                   auto.auto_loss, auto.auto_loss_poc, auto.auto_agg,
                                   T, zt, cfg)
    dom = compute_dom(t.DOC, t.DON, t.DOP, t.DOCr, t.DONr, t.DOPr,
                      zoo.zoo_loss_doc, auto.auto_loss_doc, grz.auto_graze_doc,
                      grz.remaining_N_don, grz.remaining_P_dop, PAR, dz, cfg)

    # --- particle production (compute_large_detritus_prod) --------------------------------
    sum0 = lambda a: jnp.sum(a, axis=0)  # noqa: E731
    POC_prod = (zoo.zoo_loss_poc + sum0(grz.auto_graze_poc) + sum0(auto.auto_agg)
                + sum0(auto.auto_loss_poc))
    POP_prod = cfg.Qp_zoo * zoo.zoo_loss_poc + sum0(grz.remaining_P_pop)
    DOP_loss_P_bal = jnp.maximum(-POP_prod, 0.0)
    POP_prod = jnp.maximum(POP_prod, 0.0)
    PON_prod = cfg.Q * zoo.zoo_loss_poc + sum0(grz.remaining_N_pon)
    DON_loss_N_bal = jnp.maximum(-PON_prod, 0.0)
    PON_prod = jnp.maximum(PON_prod, 0.0)
    P_iron_base = zoo.zoo_loss_poc * cfg.Qfe_zoo + sum0(grz.remaining_Fe_pfe)
    CaCO3_prod = sum0(((1.0 - cfg.f_graze_CaCO3_remin) * grz.auto_graze
                       + auto.auto_loss + auto.auto_agg) * auto.QCaCO3)
    SiO2_prod = sum0(auto.Qsi * ((1.0 - cfg.f_graze_si_remin) * grz.auto_graze + auto.auto_agg
                                 + jnp.asarray(cfg.loss_poc).reshape((3,) + (1,) * zero.ndim)
                                 * auto.auto_loss))

    # --- particles, iron, sediments -------------------------------------------------------
    fe_spec = iron_speciation(t.Fe, t.Lig, cfg)
    part = compute_particles(
        ParticleProduction(POC=POC_prod, POP=POP_prod, PON=PON_prod, CaCO3=CaCO3_prod,
                           SiO2=SiO2_prod, P_iron_base=P_iron_base),
        dz, zw, kmt, bot_flux_to_tend, t.O2, t.NO3, fe_spec.Fefree, fe_spec.FeLig1,
        doc_scavenging_factor(t.DOC, t.DOCr), CO3, CO3_alt, co3_sat_calc,
        forcing.dust_flux, forcing.fesedflux, forcing.feventflux, cfg)

    lig = compute_ligand(t.Lig, part.Lig_prod_vent, part.POC_remin, dom.DOC_prod,
                         part.Lig_scavenge, sum0(auto.photoFe), PAR, dz, cfg)
    nitrif = compute_nitrification(t.NH4, PAR, cfg)
    POCremin_refract = cfg.DOCprod_refract * cfg.POCremin_refract_frac
    PONremin_refract = cfg.DONprod_refract * cfg.PONremin_refract_frac
    POPremin_refract = cfg.DOPprod_refract * cfg.POPremin_refract_frac
    denitrif = compute_denitrification(t.O2, t.NO3, dom.DOC_remin, dom.DOCr_remin,
                                       part.POC_remin, part.other_remin, part.sed_denitrif,
                                       POCremin_refract, cfg)

    # --- tendencies (compute_local_tendencies) ----------------------------------------------
    auto_sum = grz.auto_graze + auto.auto_loss + auto.auto_agg
    d_auto_C = auto.photoC - auto_sum
    d_auto_P = auto.PO4_V + auto.DOP_V - auto.Qp * auto_sum
    d_auto_Chl = auto.photoacc - auto.thetaC * auto_sum
    d_auto_Fe = auto.photoFe - auto.Qfe * auto_sum
    d_auto_Si = auto.photoSi - auto.Qsi * auto_sum
    d_auto_CaCO3 = auto.CaCO3_form - auto.QCaCO3 * auto_sum

    d_NO3 = nitrif - denitrif - part.sed_denitrif - sum0(auto.NO3_V)
    d_NH4 = (dom.DON_remin + dom.DONr_remin - sum0(auto.NH4_V) - nitrif
             + (1.0 - PONremin_refract) * part.PON_remin + sum0(grz.remaining_N_din)
             + cfg.Q * zoo.zoo_loss_dic + sum0(auto.Nexcrete))
    d_PO4 = (dom.DOP_remin + dom.DOPr_remin - sum0(auto.PO4_V)
             + (1.0 - POPremin_refract) * part.POP_remin + sum0(grz.remaining_P_dip)
             + cfg.Qp_zoo * zoo.zoo_loss_dic)
    d_Fe = (part.P_iron_remin - sum0(auto.photoFe) - part.Fe_scavenge
            + sum0(grz.remaining_Fe_dfe) + cfg.Qfe_zoo * (zoo.zoo_loss_dic + zoo.zoo_loss_doc))
    d_Lig = lig.Lig_prod - lig.Lig_loss
    loss_poc = jnp.asarray(cfg.loss_poc).reshape((3,) + (1,) * zero.ndim)
    d_SiO3 = part.SiO2_remin + sum0(-auto.photoSi + auto.Qsi * (
        cfg.f_graze_si_remin * grz.auto_graze + (1.0 - loss_poc) * auto.auto_loss))
    d_zooC = zoo.x_graze_zoo - zoo.zoo_loss

    d_DOC = dom.DOC_prod * (1.0 - cfg.DOCprod_refract) - dom.DOC_remin
    d_DOCr = dom.DOC_prod * cfg.DOCprod_refract - dom.DOCr_remin + part.POC_remin * POCremin_refract
    d_DON = dom.DON_prod * (1.0 - cfg.DONprod_refract) - dom.DON_remin - DON_loss_N_bal
    d_DONr = dom.DON_prod * cfg.DONprod_refract - dom.DONr_remin + part.PON_remin * PONremin_refract
    d_DOP = (dom.DOP_prod * (1.0 - cfg.DOPprod_refract) - dom.DOP_remin - sum0(auto.DOP_V)
             - DOP_loss_P_bal)
    d_DOPr = dom.DOP_prod * cfg.DOPprod_refract - dom.DOPr_remin + part.POP_remin * POPremin_refract

    calcite_gut = sum0(cfg.f_graze_CaCO3_remin * grz.auto_graze * auto.QCaCO3 - auto.CaCO3_form)
    d_DIC = (sum0(auto.auto_loss_dic) + sum0(grz.auto_graze_dic) - sum0(auto.photoC)
             + dom.DOC_remin + part.POC_remin * (1.0 - POCremin_refract) + zoo.zoo_loss_dic
             + part.CaCO3_remin + dom.DOCr_remin + calcite_gut)
    d_DIC_alt = d_DIC + (part.CaCO3_ALT_CO2_remin - part.CaCO3_remin)
    d_ALK = -d_NO3 + d_NH4 + 2.0 * part.CaCO3_remin + 2.0 * calcite_gut
    d_ALK_alt = d_ALK + 2.0 * (part.CaCO3_ALT_CO2_remin - part.CaCO3_remin)

    red_C_O2, remin_C_O2, red_C_O2_diaz = redfield_o2_ratios(cfg)
    N_src = auto.NO3_V + auto.NH4_V + auto.Nfix          # Nfix = 0 except diaz
    N_src_safe = jnp.where(N_src > 0.0, N_src, 1.0)
    O2_per_C = (auto.NO3_V / red_C_O2 + auto.NH4_V / remin_C_O2 + auto.Nfix / red_C_O2_diaz) / N_src_safe
    O2_production = sum0(jnp.where(N_src > 0.0, auto.photoC * O2_per_C, 0.0))
    o2_avail = jnp.clip((t.O2 - cfg.parm_o2_min) / cfg.parm_o2_min_delta, 0.0, 1.0)
    O2_consumption = o2_avail * (
        (part.POC_remin * (1.0 - POCremin_refract) + dom.DOC_remin + dom.DOCr_remin
         - part.sed_denitrif * cfg.denitrif_C_N - part.other_remin
         + zoo.zoo_loss_dic + sum0(auto.auto_loss_dic) + sum0(grz.auto_graze_dic)) / remin_C_O2
        + 2.0 * nitrif)
    d_O2 = O2_production - O2_consumption

    tend = MarblTracers(
        PO4=d_PO4, NO3=d_NO3, SiO3=d_SiO3, NH4=d_NH4, Fe=d_Fe, Lig=d_Lig, O2=d_O2,
        DIC=d_DIC, DIC_ALT_CO2=d_DIC_alt, ALK=d_ALK, ALK_ALT_CO2=d_ALK_alt,
        DOC=d_DOC, DON=d_DON, DOP=d_DOP, DOPr=d_DOPr, DONr=d_DONr, DOCr=d_DOCr,
        zooC=d_zooC,
        spChl=d_auto_Chl[0], spC=d_auto_C[0], spP=d_auto_P[0], spFe=d_auto_Fe[0],
        spCaCO3=d_auto_CaCO3[0],
        diatChl=d_auto_Chl[1], diatC=d_auto_C[1], diatP=d_auto_P[1], diatFe=d_auto_Fe[1],
        diatSi=d_auto_Si[1],
        diazChl=d_auto_Chl[2], diazC=d_auto_C[2], diazP=d_auto_P[2], diazFe=d_auto_Fe[2],
    )
    tend = jax.tree_util.tree_map(lambda a: jnp.where(active, a, 0.0), tend)

    PAR_avg = jnp.sum(PAR.col_frac[..., None] * PAR.avg, axis=0)
    diags = MarblInteriorDiagnostics(
        temperature=T, pH=spec.pH, CO3=CO3,
        CO3_sat_calcite=co3_sat_calc, CO3_sat_aragonite=co3_sat_arag,
        PAR_avg=PAR_avg, photoC=auto.photoC, f_nut=auto.f_nut, light_lim=auto.light_lim,
        auto_graze=grz.auto_graze, CaCO3_form=sum0(auto.CaCO3_form), Nfix=sum0(auto.Nfix),
        nitrif=nitrif, denitrif=denitrif, sed_denitrif=part.sed_denitrif,
        Fe_scavenge=part.Fe_scavenge,
        POC_flux_out=part.POC_flux_out, CaCO3_flux_out=part.CaCO3_flux_out,
        SiO2_flux_out=part.SiO2_flux_out, POC_remin=part.POC_remin,
        O2_production=O2_production, O2_consumption=O2_consumption,
        POC_sed_loss=part.POC_sed_loss, CaCO3_sed_loss=part.CaCO3_sed_loss,
        SiO2_sed_loss=part.SiO2_sed_loss, POP_sed_loss=part.POP_sed_loss,
        PON_sed_loss=part.PON_sed_loss, P_iron_sed_loss=part.P_iron_sed_loss,
        dust_Fe_release=part.dust_Fe_release,
        pH_alt_co2=spec_alt.pH, CO3_alt_co2=CO3_alt,
        CaCO3_ALT_CO2_sed_loss=part.CaCO3_ALT_CO2_sed_loss,
    )
    return tend, diags
