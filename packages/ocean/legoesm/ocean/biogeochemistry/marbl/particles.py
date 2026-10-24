"""MARBL sinking particles, iron scavenging and sediment exchange.

Ports compute_scavenging (rate part), compute_large_detritus_prod (P_iron
part), compute_particulate_terms and set_surface_particulate_terms from
``marbl_interior_tendency_mod.F90`` (Armstrong et al. 2000 ballast model
with soft / hard sub-classes, O2- and depth-dependent dissolution length
scales, Dunne et al. 2007 POC burial, Bohlen et al. 2012 sediment
denitrification).

Particulate pools are not tracers: fluxes are integrated top-down through
each column with ``lax.scan`` over levels, and remineralization in each
layer follows from flux_in - flux_out + prod dz - remin dz = 0.

P_CaCO3_ALT_CO2 has the same production and dissolution as P_CaCO3 in the
water column, so its in-column fluxes are identical; the two differ only in
the saturation-state-based burial at the sea floor.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.biogeochemistry.marbl.config import MarblConfig

_SPD = constants.seconds_per_day
_SPY = constants.seconds_per_day * constants.days_per_year
# MARBL unit_system 'mks' conversions of cgs-specified parameters
_CONC_FLUX2NMOL_CM2S = 1.0e2     # mmol/m^2 -> nmol/cm^2
_LEN2CM = 1.0e2                  # m -> cm
_DUST_SCAV_SCALE = 1.0e6         # mg dust per kg dust (sinking mass in mg/m^2/s)


class ParticleProduction(NamedTuple):
    """Production of particulate pools [mmol/m^3/s], each (..., nlev)."""
    POC: jnp.ndarray
    POP: jnp.ndarray
    PON: jnp.ndarray
    CaCO3: jnp.ndarray
    SiO2: jnp.ndarray
    P_iron_base: jnp.ndarray     # P_iron production excluding scavenged Fe


class ParticleTerms(NamedTuple):
    POC_remin: jnp.ndarray
    POP_remin: jnp.ndarray
    PON_remin: jnp.ndarray
    CaCO3_remin: jnp.ndarray
    CaCO3_ALT_CO2_remin: jnp.ndarray
    SiO2_remin: jnp.ndarray
    P_iron_remin: jnp.ndarray
    dust_Fe_release: jnp.ndarray
    Fe_scavenge: jnp.ndarray
    Lig_scavenge: jnp.ndarray
    Lig_prod_vent: jnp.ndarray
    sed_denitrif: jnp.ndarray
    other_remin: jnp.ndarray
    POC_flux_out: jnp.ndarray
    CaCO3_flux_out: jnp.ndarray
    SiO2_flux_out: jnp.ndarray
    POC_sed_loss: jnp.ndarray
    POP_sed_loss: jnp.ndarray
    PON_sed_loss: jnp.ndarray
    CaCO3_sed_loss: jnp.ndarray
    CaCO3_ALT_CO2_sed_loss: jnp.ndarray
    SiO2_sed_loss: jnp.ndarray
    P_iron_sed_loss: jnp.ndarray
    dust_sed_loss: jnp.ndarray


def default_bot_flux_to_tend(dz: jnp.ndarray, kmt: jnp.ndarray) -> jnp.ndarray:
    """All bottom flux into the deepest active layer: 1/dz at k = kmt-1, else 0."""
    k = jnp.arange(dz.shape[-1])
    is_bottom = k == (jnp.asarray(kmt)[..., None] - 1)
    dz_safe = jnp.where(dz > 0.0, dz, 1.0)
    return jnp.where(is_bottom, 1.0 / dz_safe, 0.0)


def compute_particles(
    prod: ParticleProduction,
    dz: jnp.ndarray,
    zw: jnp.ndarray,
    kmt: jnp.ndarray,
    bot_flux_to_tend: jnp.ndarray,
    O2: jnp.ndarray,
    NO3: jnp.ndarray,
    Fefree: jnp.ndarray,
    FeLig1: jnp.ndarray,
    doc_scav_factor: jnp.ndarray,
    CO3: jnp.ndarray,
    CO3_alt_co2: jnp.ndarray,
    CO3_sat_calcite: jnp.ndarray,
    dust_flux: jnp.ndarray,
    fesedflux: jnp.ndarray,
    feventflux: jnp.ndarray,
    cfg: MarblConfig,
) -> ParticleTerms:
    """Sinking fluxes, remineralization and sediment losses for each column.

    Level fields are (..., nlev) with k = 0 at the surface; ``zw`` is the
    depth of each layer's bottom interface [m]; ``kmt`` (...,) is the number
    of active levels.  ``dust_flux`` [kg/m^2/s], ``fesedflux`` /
    ``feventflux`` [mmol/m^2/s per layer].
    """
    nlev = dz.shape[-1]
    dtype = dz.dtype
    k_idx = jnp.arange(nlev)
    active = k_idx < jnp.asarray(kmt)[..., None]
    dz_safe = jnp.where(dz > 0.0, dz, 1.0)
    dzr = jnp.where(active, 1.0 / dz_safe, 0.0)

    # QA (ballast) mass ratios [mmol C per mmol ballast or per kg dust]
    rho_CaCO3 = cfg.parm_hPOC_CaCO3_ratio * constants.M_CaCO3 / constants.M_C
    rho_SiO2 = cfg.parm_hPOC_SiO2_ratio * constants.M_SiO2 / constants.M_C
    rho_dust = cfg.parm_hPOC_dust_ratio * 1.0e6 / constants.M_C
    dust_to_Fe = cfg.dust_Fe_frac / constants.M_Fe * 1.0e6   # mmol Fe per kg dust

    Fe_scav_rate0 = cfg.parm_Fe_scavenge_rate0 * _CONC_FLUX2NMOL_CM2S / _SPY
    Lig_scav_rate0 = cfg.parm_Lig_scavenge_rate0 * _CONC_FLUX2NMOL_CM2S / _SPY
    FeLig_scav_rate0 = cfg.parm_FeLig_scavenge_rate0 * _CONC_FLUX2NMOL_CM2S / _SPY
    Fe_desorption = cfg.parm_Fe_desorption_rate0 * _LEN2CM

    # depth-dependent scale length (linear between prescribed depths, clamped)
    scalelength = jnp.interp(zw, jnp.asarray(cfg.parm_scalelen_z, dtype=dtype),
                             jnp.asarray(cfg.parm_scalelen_vals, dtype=dtype))
    o2_frac = jnp.minimum(1.0, (cfg.o2_sf_o2_range_hi - O2)
                          / (cfg.o2_sf_o2_range_hi - cfg.o2_sf_o2_range_lo))
    o2_scalefactor = jnp.where(O2 < cfg.o2_sf_o2_range_hi,
                               1.0 + (cfg.o2_sf_val_lo_o2 - 1.0) * o2_frac, 1.0)
    diss_factor = scalelength * o2_scalefactor
    poc_diss = cfg.parm_POC_diss * diss_factor
    sio2_diss = cfg.parm_SiO2_diss * diss_factor
    caco3_diss = cfg.parm_CaCO3_diss * diss_factor
    dust_diss = cfg.dust_diss * diss_factor
    decay_POC = jnp.exp(-dz / poc_diss)
    decay_SiO2 = jnp.exp(-dz / sio2_diss)
    decay_CaCO3 = jnp.exp(-dz / caco3_diss)
    decay_dust = jnp.exp(-dz / dust_diss)
    decay_hard = jnp.exp(-dz / cfg.hard_diss)
    decay_hard_dust = jnp.exp(-dz / cfg.hard_dust_diss)
    dzr_mod = jnp.minimum(1.0, dz_safe ** cfg.Fe_desorption_dz_exp)

    # surface boundary: only dust enters from above (hard sub-class for gamma = 1)
    zero = jnp.zeros_like(dz[..., 0])
    dust_s0 = (1.0 - cfg.dust_gamma) * dust_flux + zero
    dust_h0 = cfg.dust_gamma * dust_flux + zero
    carry0 = dict(POC_s=zero, POC_h=zero, POP_s=zero, PON_s=zero,
                  Ca_s=zero, Ca_h=zero, Si_s=zero, Si_h=zero,
                  dust_s=dust_s0, dust_h=dust_h0, Fe_s=zero,
                  QA=rho_dust * (dust_s0 + dust_h0))

    level_fields = dict(
        active=active, dz=dz_safe, dzr=dzr, dzr_mod=dzr_mod,
        POC_prod=prod.POC, POP_prod=prod.POP, PON_prod=prod.PON,
        Ca_prod=prod.CaCO3, Si_prod=prod.SiO2, Fe_prod_base=prod.P_iron_base,
        poc_diss=poc_diss, sio2_diss=sio2_diss, caco3_diss=caco3_diss,
        decay_POC=decay_POC, decay_SiO2=decay_SiO2, decay_CaCO3=decay_CaCO3,
        decay_dust=decay_dust, decay_hard=decay_hard, decay_hard_dust=decay_hard_dust,
        Fefree=Fefree, FeLig1=FeLig1, doc_factor=doc_scav_factor,
        fesed=fesedflux, fevent=feventflux,
    )
    xs = jax.tree_util.tree_map(lambda a: jnp.moveaxis(a, -1, 0), level_fields)

    gCa = cfg.parm_CaCO3_gamma
    gSi = cfg.parm_SiO2_gamma

    def level(c, x):
        dz_k = x["dz"]
        dzr_k = x["dzr"]

        # Fe scavenging, driven by the sinking mass entering the layer
        sinking_mass = ((c["POC_s"] + c["POC_h"]) * cfg.POC_mass_for_scav
                        + (c["Ca_s"] + c["Ca_h"]) * constants.M_CaCO3
                        + (c["Si_s"] + c["Si_h"]) * constants.M_SiO2
                        + (c["dust_s"] + c["dust_h"]) * _DUST_SCAV_SCALE) * x["doc_factor"]
        Fe_scavenge = (x["Fefree"] * Fe_scav_rate0 + x["FeLig1"] * FeLig_scav_rate0) * sinking_mass
        Lig_scavenge = x["FeLig1"] * Lig_scav_rate0 * sinking_mass
        Fe_prod = x["Fe_prod_base"] + Fe_scavenge

        # ballast minerals and dust
        Ca_s_out = (c["Ca_s"] * x["decay_CaCO3"]
                    + x["Ca_prod"] * (1.0 - gCa) * (1.0 - x["decay_CaCO3"]) * x["caco3_diss"])
        Ca_h_out = c["Ca_h"] * x["decay_hard"] + x["Ca_prod"] * gCa * dz_k
        Si_s_out = (c["Si_s"] * x["decay_SiO2"]
                    + x["Si_prod"] * (1.0 - gSi) * (1.0 - x["decay_SiO2"]) * x["sio2_diss"])
        Si_h_out = c["Si_h"] * x["decay_hard"] + x["Si_prod"] * gSi * dz_k
        dust_s_out = c["dust_s"] * x["decay_dust"]          # no dust production
        dust_h_out = c["dust_h"] * x["decay_hard_dust"]

        # POC available for the excess (soft) flux after ballast-associated POC
        avail = x["POC_prod"] - rho_CaCO3 * x["Ca_prod"] - rho_SiO2 * x["Si_prod"]
        dust_in = c["dust_s"] + c["dust_h"]
        dust_in_safe = jnp.where(dust_in > 0.0, dust_in, 1.0)
        QA_new = jnp.where(c["QA"] > 0.0,
                           c["QA"] * (dust_s_out + dust_h_out) / dust_in_safe, 0.0)
        reduce = (QA_new > 0.0) & (x["POC_prod"] > 0.0)
        QA_red = QA_new - avail * dz_k
        avail = jnp.where(reduce, jnp.where(QA_red < 0.0, -QA_red * dzr_k, 0.0), avail)
        QA_new = jnp.where(reduce, jnp.maximum(QA_red, 0.0), QA_new)

        POC_h_out = jnp.maximum(
            rho_CaCO3 * (Ca_s_out + Ca_h_out) + rho_SiO2 * (Si_s_out + Si_h_out)
            + rho_dust * (dust_s_out + dust_h_out) - QA_new, 0.0)
        POC_h_out = jnp.where((c["POC_h"] == 0.0) & (x["POC_prod"] == 0.0), 0.0, POC_h_out)
        POC_s_out = c["POC_s"] * x["decay_POC"] + avail * (1.0 - x["decay_POC"]) * x["poc_diss"]

        Ca_remin = x["Ca_prod"] + ((c["Ca_s"] - Ca_s_out) + (c["Ca_h"] - Ca_h_out)) * dzr_k
        Si_remin = x["Si_prod"] + ((c["Si_s"] - Si_s_out) + (c["Si_h"] - Si_h_out)) * dzr_k
        POC_remin = x["POC_prod"] + ((c["POC_s"] - POC_s_out) + (c["POC_h"] - POC_h_out)) * dzr_k
        dust_remin = ((c["dust_s"] - dust_s_out) + (c["dust_h"] - dust_h_out)) * dzr_k

        # iron: remineralized in proportion to POC, plus desorption
        POC_in = c["POC_s"] + c["POC_h"]
        POC_in_safe = jnp.where(POC_in > 0.0, POC_in, 1.0)
        Fe_remin = jnp.where(POC_in == 0.0, POC_remin * cfg.parm_Red_Fe_C,
                             POC_remin * c["Fe_s"] / POC_in_safe)
        Fe_remin = Fe_remin + c["Fe_s"] * Fe_desorption * x["dzr_mod"]
        Fe_s_out = c["Fe_s"] + dz_k * (Fe_prod - Fe_remin)
        Fe_neg = Fe_s_out < 0.0
        Fe_remin = jnp.where(Fe_neg, c["Fe_s"] * dzr_k + Fe_prod, Fe_remin)
        Fe_s_out = jnp.maximum(Fe_s_out, 0.0)
        dust_Fe = dust_remin * dust_to_Fe
        Fe_remin = Fe_remin + dust_Fe + (x["fesed"] + x["fevent"]) * dzr_k
        Lig_prod_vent = x["fevent"] * dzr_k * cfg.Lig_vent_ratio

        # POP and PON follow POC remineralization
        POC_prod_safe = jnp.where(x["POC_prod"] > 0.0, x["POC_prod"], 1.0)

        def follow(s_in, p):
            ratio = jnp.where(POC_in > 0.0, s_in / POC_in_safe,
                              jnp.where(x["POC_prod"] > 0.0, p / POC_prod_safe, 0.0))
            remin = POC_remin * ratio
            out = s_in + dz_k * (p - remin)
            neg = out < 0.0
            return jnp.maximum(out, 0.0), jnp.where(neg, s_in * dzr_k + p, remin)

        POP_s_out, POP_remin = follow(c["POP_s"], x["POP_prod"])
        PON_s_out, PON_remin = follow(c["PON_s"], x["PON_prod"])

        on = x["active"]
        out_carry = dict(POC_s=POC_s_out, POC_h=POC_h_out, POP_s=POP_s_out, PON_s=PON_s_out,
                         Ca_s=Ca_s_out, Ca_h=Ca_h_out, Si_s=Si_s_out, Si_h=Si_h_out,
                         dust_s=dust_s_out, dust_h=dust_h_out, Fe_s=Fe_s_out, QA=QA_new)
        new_carry = {key: jnp.where(on, v, 0.0) for key, v in out_carry.items()}
        outputs = dict(
            POC_remin=POC_remin, POP_remin=POP_remin, PON_remin=PON_remin,
            Ca_remin=Ca_remin, Si_remin=Si_remin, Fe_remin=Fe_remin, dust_Fe=dust_Fe,
            Fe_scavenge=Fe_scavenge, Lig_scavenge=Lig_scavenge, Lig_prod_vent=Lig_prod_vent,
            POC_out=POC_s_out + POC_h_out, POP_out=POP_s_out, PON_out=PON_s_out,
            Ca_out=Ca_s_out + Ca_h_out, Si_out=Si_s_out + Si_h_out,
            Fe_out=Fe_s_out, dust_out=dust_s_out + dust_h_out,
        )
        outputs = {key: jnp.where(on, v, 0.0) for key, v in outputs.items()}
        return new_carry, outputs

    _, ys = jax.lax.scan(level, carry0, xs)
    ys = jax.tree_util.tree_map(lambda a: jnp.moveaxis(a, 0, -1), ys)

    # ---- sea floor (deepest active layer) ---------------------------------------
    is_bottom = k_idx == (jnp.asarray(kmt)[..., None] - 1)
    at_bottom = lambda a: jnp.sum(jnp.where(is_bottom, a, 0.0), axis=-1)  # noqa: E731
    bftt = bot_flux_to_tend
    POC_to_floor = at_bottom(ys["POC_out"])
    POP_to_floor = at_bottom(ys["POP_out"])
    PON_to_floor = at_bottom(ys["PON_out"])
    Ca_to_floor = at_bottom(ys["Ca_out"])
    Si_to_floor = at_bottom(ys["Si_out"])
    Fe_to_floor = at_bottom(ys["Fe_out"])
    dust_to_floor = at_bottom(ys["dust_out"])
    O2_b = at_bottom(O2)
    NO3_b = at_bottom(NO3)

    # POC burial efficiency (Dunne et al. 2007), flux in mmol/m^2/day
    flux_day = POC_to_floor * _SPD
    bury_frac = 0.013 + 0.53 * flux_day * flux_day / (7.0 + flux_day) ** 2
    POM_bury = jnp.minimum(cfg.POM_bury_frac_max, cfg.POC_bury_coeff * bury_frac)
    POC_sed = POC_to_floor * POM_bury
    PON_sed = PON_to_floor * cfg.PON_bury_coeff * POM_bury
    POP_sed = POP_to_floor * jnp.minimum(cfg.POM_bury_frac_max, cfg.POP_bury_coeff * bury_frac)

    # sediment denitrification (Bohlen et al. 2012) and other anaerobic remin
    has_poc = POC_to_floor > 0.0
    sed_denitrif = (bftt * cfg.parm_sed_denitrif_coeff * POC_to_floor[..., None]
                    * (0.06 + 0.19 * 0.99 ** (O2_b - NO3_b)[..., None]) * 1.5)
    sed_denitrif = jnp.where(NO3 < 1.0, NO3 * sed_denitrif, sed_denitrif)
    poc_remin_sed = bftt * (POC_to_floor - POC_sed)[..., None]
    flux_cm2_yr = POC_to_floor * 1.0e-4 * _SPY               # mmol/cm^2/yr
    other_remin = jnp.minimum(
        bftt * jnp.minimum(0.1 + flux_cm2_yr, 0.5)[..., None] * (POC_to_floor - POC_sed)[..., None],
        poc_remin_sed - sed_denitrif * cfg.denitrif_C_N)
    other_remin = jnp.where(O2 < 1.0, poc_remin_sed - sed_denitrif * cfg.denitrif_C_N, other_remin)
    sed_denitrif = jnp.where(has_poc[..., None], sed_denitrif, 0.0)
    other_remin = jnp.where(has_poc[..., None], other_remin, 0.0)

    # biogenic silica burial (Ragueneau et al. 2000)
    Si_bury_frac = jnp.where(Si_to_floor * _SPD > 2.0, 0.2, 0.04)
    SiO2_sed = Si_to_floor * jnp.minimum(cfg.bSi_bury_coeff * Si_bury_frac, cfg.bSi_bury_frac_max)

    # CaCO3 preserved where bottom water is supersaturated w.r.t. calcite
    CO3_sat_b = at_bottom(CO3_sat_calcite)
    CaCO3_sed = jnp.where(at_bottom(CO3) > cfg.caco3_bury_thres_omega_calc * CO3_sat_b,
                          Ca_to_floor, 0.0)
    CaCO3_alt_sed = jnp.where(at_bottom(CO3_alt_co2) > cfg.caco3_bury_thres_omega_calc * CO3_sat_b,
                              Ca_to_floor, 0.0)

    def add_floor(remin, to_floor, sed):
        return remin + bftt * (to_floor - sed)[..., None]

    return ParticleTerms(
        POC_remin=add_floor(ys["POC_remin"], POC_to_floor, POC_sed),
        POP_remin=add_floor(ys["POP_remin"], POP_to_floor, POP_sed),
        PON_remin=add_floor(ys["PON_remin"], PON_to_floor, PON_sed),
        CaCO3_remin=add_floor(ys["Ca_remin"], Ca_to_floor, CaCO3_sed),
        CaCO3_ALT_CO2_remin=add_floor(ys["Ca_remin"], Ca_to_floor, CaCO3_alt_sed),
        SiO2_remin=add_floor(ys["Si_remin"], Si_to_floor, SiO2_sed),
        P_iron_remin=ys["Fe_remin"],
        dust_Fe_release=ys["dust_Fe"],
        Fe_scavenge=ys["Fe_scavenge"],
        Lig_scavenge=ys["Lig_scavenge"],
        Lig_prod_vent=ys["Lig_prod_vent"],
        sed_denitrif=sed_denitrif,
        other_remin=other_remin,
        POC_flux_out=ys["POC_out"],
        CaCO3_flux_out=ys["Ca_out"],
        SiO2_flux_out=ys["Si_out"],
        POC_sed_loss=POC_sed,
        POP_sed_loss=POP_sed,
        PON_sed_loss=PON_sed,
        CaCO3_sed_loss=CaCO3_sed,
        CaCO3_ALT_CO2_sed_loss=CaCO3_alt_sed,
        SiO2_sed_loss=SiO2_sed,
        P_iron_sed_loss=Fe_to_floor,
        dust_sed_loss=dust_to_floor,
    )
