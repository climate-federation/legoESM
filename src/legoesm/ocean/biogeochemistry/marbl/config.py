"""MARBL configuration, tracer state and diagnostics.

Port of MARBL (https://github.com/marbl-ecosys/MARBL, commit f00d642) with
the CESM2 plankton functional types (small phytoplankton ``sp``, diatoms
``diat``, diazotrophs ``diaz``; one zooplankton ``zoo``), fixed N:C
(``lvariable_NtoC = .false.``) and variable P:C.

Defaults come from ``defaults/settings_cesm2.1.yaml``; parameters MARBL
hard-codes come from ``src/marbl_settings_mod.F90`` and
``src/marbl_pft_mod.F90`` at that commit.  Units follow MARBL's mks unit
system: concentrations mmol/m^3 (Chl mg/m^3, ALK meq/m^3), lengths m,
time s (rates given per day here are converted in the process code).

Per-PFT parameters are tuples ordered (sp, diat, diaz).  All numeric
fields are pytree leaves, so ``jax.grad`` with respect to a ``MarblConfig``
gives sensitivities to every parameter.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

# PFT indices (CESM2 PFT set; the process code relies on these roles:
# sp = implicit calcifier, diat = silicifier, diaz = N2 fixer).
SP, DIAT, DIAZ = 0, 1, 2
N_AUTOTROPHS = 3

# Q10 reference temperature for the q_10 temperature functional form
# (marbl_interior_tendency_mod: compute_temperature_functional_form).
_TREF_Q10 = 30.0


class MarblConfig(NamedTuple):
    """MARBL parameters (CESM2.1 defaults)."""

    # ---- temperature dependence -------------------------------------------
    Q_10: float = 1.7                       # marbl_settings_mod
    Tref_auto: float = _TREF_Q10            # [degC]
    Tref_zoo: float = _TREF_Q10             # [degC]

    # ---- stoichiometry ----------------------------------------------------
    Q: float = 16.0 / 112.0                 # N/C of phyto and zoo (marbl_settings_mod)
    Qp_zoo: float = 1.0 / 117.0             # P/C of zoo (marbl_pft_mod)
    Qfe_zoo: float = 6.0e-6                 # Fe/C of zoo
    parm_Red_D_C_P: float = 117.0           # cesm2.1 (lvariable_NtoC = .false.)
    parm_Remin_D_O2_P: float = 138.0        # cesm2.1
    parm_Red_Fe_C: float = 3.0e-6
    r_Nfix_photo: float = 1.25              # N fixed relative to C fixed
    denitrif_C_N: float = 0.9375            # MARBL overwrites its flux-based
    #   value with this constant at every level (interior_tendency_mod:3421)

    # ---- autotrophs (sp, diat, diaz) ---------------------------------------
    kFe: tuple = (0.03e-3, 0.07e-3, 0.045e-3)
    kPO4: tuple = (0.01, 0.05, 0.015)
    kDOP: tuple = (0.3, 0.5, 0.075)
    kNO3: tuple = (0.25, 0.5, 2.0)
    kNH4: tuple = (0.01, 0.05, 0.2)
    kSiO3: tuple = (0.0, 0.7, 0.0)
    SiOpt: tuple = (0.0, 10.0, 0.0)
    gQfe_max: tuple = (30e-6, 30e-6, 60e-6)
    gQfe_min: tuple = (2.5e-6, 2.5e-6, 2.5e-6)
    FeOpt: tuple = (1.8e-3, 1.8e-3, 2.0e-3)
    gQp_max: tuple = (1.1e-2, 1.1e-2, 8.3e-3)
    gQp_min: tuple = (7.5e-3, 8.3e-3, 3.7e-3)
    POpt: tuple = (0.8, 0.8, 0.8)
    alphaPI_per_day: tuple = (0.39, 0.28, 0.39)   # mmol C m^2 / (mg Chl W day)
    PCref_per_day: tuple = (5.0, 5.0, 2.5)
    thetaN_max: tuple = (2.5, 4.0, 2.5)           # mg Chl / mmol N
    loss_thres: tuple = (0.01, 0.02, 0.02)
    loss_thres2: tuple = (0.0, 0.0, 0.001)
    temp_thres: tuple = (-10.0, -10.0, 15.0)       # [degC]
    mort_per_day: tuple = (0.1, 0.1, 0.1)
    mort2_per_day: tuple = (0.01, 0.01, 0.01)
    agg_rate_max: tuple = (0.5, 0.5, 0.5)          # [1/day]
    agg_rate_min: tuple = (0.01, 0.02, 0.01)       # [1/day]
    loss_poc: tuple = (0.0, 0.0, 0.0)
    gQsi_0: float = 0.137
    gQsi_max: float = 0.822
    gQsi_min: float = 0.0457
    QCaCO3_max: float = 0.40
    auto_mort2_exp: float = 1.75
    thres_z1_auto: float = 50.0             # [m]
    thres_z2_auto: float = 100.0            # [m]

    # ---- calcification ----------------------------------------------------
    parm_f_prod_sp_CaCO3: float = 0.07
    CaCO3_temp_thres1: float = 4.0          # [degC]
    CaCO3_temp_thres2: float = -2.0         # [degC]
    CaCO3_sp_thres: float = 2.5             # [mmol C/m^3]
    f_photosp_CaCO3: float = 0.40

    # ---- zooplankton and grazing (prey: sp, diat, diaz) --------------------
    z_mort_0_per_day: float = 0.1
    z_mort2_0_per_day: float = 0.4
    basal_respiration_rate_per_day: float = 0.0
    zoo_loss_thres: float = 0.075
    zoo_mort2_exp: float = 1.5
    thres_z1_zoo: float = 40.0              # [m]
    thres_z2_zoo: float = 80.0              # [m]
    z_umax_0_per_day: tuple = (3.3, 3.15, 3.3)
    z_grz: tuple = (1.2, 1.2, 1.2)          # [mmol C/m^3]
    graze_zoo: tuple = (0.3, 0.25, 0.3)
    graze_poc: tuple = (0.0, 0.39, 0.1)
    graze_doc: tuple = (0.06, 0.06, 0.06)
    f_zoo_detr: tuple = (0.12, 0.24, 0.12)
    caco3_poc_min: float = 0.40
    spc_poc_fac: float = 0.04
    f_graze_sp_poc_lim: float = 0.33
    f_graze_CaCO3_remin: float = 0.33
    f_graze_si_remin: float = 0.22
    parm_labile_ratio: float = 0.94
    f_toDON: float = 0.05
    f_toDOP: float = 0.15

    # ---- dissolved organic matter -------------------------------------------
    DOC_reminR_light_per_day: float = 1.0 / (365.0 * 15.0)
    DON_reminR_light_per_day: float = 1.0 / (365.0 * 15.0)
    DOP_reminR_light_per_day: float = 1.0 / (365.0 * 60.0)
    DOC_reminR_dark_per_day: float = 1.0 / (365.0 * 6.0)
    DON_reminR_dark_per_day: float = 1.0 / (365.0 * 5.5)
    DOP_reminR_dark_per_day: float = 1.0 / (365.0 * 4.5)
    DOCr_reminR0_per_day: float = 1.0 / (365.0 * 16000.0)
    DONr_reminR0_per_day: float = 1.0 / (365.0 * 9500.0)
    DOPr_reminR0_per_day: float = 1.0 / (365.0 * 5500.0)
    DOMr_reminR_photo_per_day: float = 1.0 / (365.0 * 18.0)
    DOM_remin_light_PAR: float = 1.0        # [W/m^2] PAR above which "light" rates apply
    DOMr_photo_ref_depth: float = 10.0      # [m] photodegradation scaled by 10 m / dz1
    DOMr_photo_lnPAR_coeff: float = 0.4373
    DOCprod_refract: float = 0.01
    DONprod_refract: float = 0.0115
    DOPprod_refract: float = 0.003
    POCremin_refract_frac: float = 0.06     # POCremin_refract = DOCprod_refract * this
    PONremin_refract_frac: float = 0.03
    POPremin_refract_frac: float = 0.06

    # ---- light --------------------------------------------------------------
    f_qsw_par: float = 0.45                 # PAR fraction of shortwave
    PAR_threshold: float = 1.0e-19          # [W/m^2]

    # ---- nitrogen cycle -------------------------------------------------------
    parm_kappa_nitrif_per_day: float = 0.06
    parm_nitrif_par_lim: float = 1.0        # [W/m^2]
    parm_o2_min: float = 5.0                # [mmol/m^3]
    parm_o2_min_delta: float = 5.0          # [mmol/m^3]
    parm_sed_denitrif_coeff: float = 1.0

    # ---- iron and ligand ------------------------------------------------------
    parm_Fe_scavenge_rate0: float = 22.0    # [cm^2/ng s/yr] (converted to mks in code)
    parm_Lig_scavenge_rate0: float = 0.015
    parm_FeLig_scavenge_rate0: float = 1.2
    parm_Lig_degrade_rate0: float = 0.000094
    parm_Fe_desorption_rate0: float = 1.0e-6  # [1/cm]
    KFeLig1: float = 1.0e8                  # [(mmol/m^3)^-1] (10e13 L/mol * 1e-6)
    remin_to_Lig: float = 0.0001
    Lig_vent_ratio: float = 0.15            # ligand per vent Fe
    Lig_uptake_ratio: float = 0.4           # ligand loss per photoFe
    Lig_low_thres: float = 1.25e-3          # [mmol/m^3]
    Lig_photochem_per_day: float = 0.018
    dust_Fe_frac: float = 0.035             # g Fe / g dust
    dust_P_frac: float = 0.00105            # P mass fraction of dust
    dust_P_solubility: float = 0.15
    dust_Si_frac: float = 0.308
    dust_Si_solubility: float = 0.075

    # ---- particles (Armstrong et al. 2000 ballast model) ---------------------
    parm_POC_diss: float = 100.0            # [m]
    parm_SiO2_diss: float = 650.0           # [m]
    parm_CaCO3_diss: float = 500.0          # [m]
    dust_diss: float = 400.0                # [m]
    hard_diss: float = 4.0e4                # [m] hard ballast dissolution length
    hard_dust_diss: float = 3.22291e6       # [m]
    parm_SiO2_gamma: float = 0.0
    parm_CaCO3_gamma: float = 0.02
    dust_gamma: float = 1.0
    parm_hPOC_SiO2_ratio: float = 0.01
    parm_hPOC_CaCO3_ratio: float = 0.01
    parm_hPOC_dust_ratio: float = 0.01
    POC_mass_for_scav: float = 24.02        # g POM per mol POC in sinking mass
    parm_scalelen_z: tuple = (100.0, 250.0, 500.0, 1000.0)   # [m]
    parm_scalelen_vals: tuple = (1.0, 3.6, 4.7, 4.8)
    o2_sf_o2_range_hi: float = 45.0         # [mmol/m^3]
    o2_sf_o2_range_lo: float = 5.0
    o2_sf_val_lo_o2: float = 2.6
    Fe_desorption_dz_exp: float = -0.36

    # ---- sediments ------------------------------------------------------------
    POC_bury_coeff: float = 2.54
    POP_bury_coeff: float = 0.36
    bSi_bury_coeff: float = 1.53
    PON_bury_coeff: float = 0.5
    POM_bury_frac_max: float = 0.8
    bSi_bury_frac_max: float = 1.0
    caco3_bury_thres_omega_calc: float = 0.89


class MarblTracers(NamedTuple):
    """The 32 MARBL base tracers, each shaped (..., nlev).

    Units: mmol/m^3, except ALK / ALK_ALT_CO2 (meq/m^3) and *Chl (mg/m^3).
    Field order is MARBL's tracer order for the CESM2 PFTs.
    """
    PO4: jnp.ndarray
    NO3: jnp.ndarray
    SiO3: jnp.ndarray
    NH4: jnp.ndarray
    Fe: jnp.ndarray
    Lig: jnp.ndarray
    O2: jnp.ndarray
    DIC: jnp.ndarray
    DIC_ALT_CO2: jnp.ndarray
    ALK: jnp.ndarray
    ALK_ALT_CO2: jnp.ndarray
    DOC: jnp.ndarray
    DON: jnp.ndarray
    DOP: jnp.ndarray
    DOPr: jnp.ndarray
    DONr: jnp.ndarray
    DOCr: jnp.ndarray
    zooC: jnp.ndarray
    spChl: jnp.ndarray
    spC: jnp.ndarray
    spP: jnp.ndarray
    spFe: jnp.ndarray
    spCaCO3: jnp.ndarray
    diatChl: jnp.ndarray
    diatC: jnp.ndarray
    diatP: jnp.ndarray
    diatFe: jnp.ndarray
    diatSi: jnp.ndarray
    diazChl: jnp.ndarray
    diazC: jnp.ndarray
    diazP: jnp.ndarray
    diazFe: jnp.ndarray


TRACER_NAMES: tuple = MarblTracers._fields

TRACER_UNITS: dict = {
    name: ("meq/m^3" if name.startswith("ALK")
           else "mg/m^3" if name.endswith("Chl")
           else "mmol/m^3")
    for name in TRACER_NAMES
}


class MarblInteriorForcing(NamedTuple):
    """Per-column forcing for the interior tendency.

    Shapes: column fields (...,), level fields (..., nlev), sub-column
    fields (n_subcol, ...).
    """
    potemp: jnp.ndarray            # potential temperature [degC] (..., nlev)
    salinity: jnp.ndarray          # [PSU] (..., nlev)
    pressure_bar: jnp.ndarray      # sea pressure [bar] (..., nlev)
    surf_shortwave: jnp.ndarray    # net shortwave into ocean [W/m^2] (n_subcol, ...)
    PAR_col_frac: jnp.ndarray      # sub-column area fractions (n_subcol, ...), sum 1
    dust_flux: jnp.ndarray         # dust deposition [kg/m^2/s] (...,)
    fesedflux: jnp.ndarray         # sedimentary Fe source [mmol/m^2/s] per level (..., nlev)
    feventflux: jnp.ndarray        # hydrothermal Fe source [mmol/m^2/s] per level (..., nlev)


class MarblSurfaceForcing(NamedTuple):
    """Surface-flux forcing, column fields shaped (...,)."""
    sst: jnp.ndarray               # [degC]
    sss: jnp.ndarray               # [PSU]
    u10_sqr: jnp.ndarray           # mean squared 10-m wind [m^2/s^2]
    atm_press: jnp.ndarray         # [atm]
    ifrac: jnp.ndarray             # sea-ice fraction [0-1]
    xco2: jnp.ndarray              # atmospheric CO2, dry-air mole fraction [ppmv]
    xco2_alt_co2: jnp.ndarray      # alternative-CO2 mole fraction [ppmv]
    dust_flux: jnp.ndarray         # [kg/m^2/s]
    iron_flux: jnp.ndarray         # soluble Fe deposition [mmol/m^2/s]
    nox_flux: jnp.ndarray          # NOx deposition [mmol N/m^2/s]
    nhy_flux: jnp.ndarray          # NHy deposition [mmol N/m^2/s]


class MarblSurfaceDiagnostics(NamedTuple):
    pH_surf: jnp.ndarray
    pCO2surf: jnp.ndarray          # [ppmv]
    dpCO2: jnp.ndarray             # pCO2 ocean - atm [ppmv]
    flux_co2: jnp.ndarray          # [mmol/m^2/s], + into ocean
    flux_o2: jnp.ndarray           # [mmol/m^2/s], + into ocean
    o2sat: jnp.ndarray             # [mmol/m^3]
    nhx_surface_emis: jnp.ndarray  # [mmol N/m^2/s], + out of ocean
    pv_co2: jnp.ndarray            # [m/s]
    pv_o2: jnp.ndarray             # [m/s]


class MarblInteriorDiagnostics(NamedTuple):
    """Selected interior diagnostics (per level unless noted)."""
    temperature: jnp.ndarray       # in-situ T [degC]
    pH: jnp.ndarray
    CO3: jnp.ndarray               # [mmol/m^3]
    CO3_sat_calcite: jnp.ndarray
    CO3_sat_aragonite: jnp.ndarray
    PAR_avg: jnp.ndarray           # col_frac-weighted layer-mean PAR [W/m^2]
    photoC: jnp.ndarray            # (3, ..., nlev) C fixation [mmol C/m^3/s]
    f_nut: jnp.ndarray             # (3, ..., nlev) nutrient limitation
    light_lim: jnp.ndarray         # (3, ..., nlev)
    auto_graze: jnp.ndarray        # (3, ..., nlev)
    CaCO3_form: jnp.ndarray        # [mmol/m^3/s]
    Nfix: jnp.ndarray              # diazotroph N fixation [mmol N/m^3/s]
    nitrif: jnp.ndarray
    denitrif: jnp.ndarray
    sed_denitrif: jnp.ndarray
    Fe_scavenge: jnp.ndarray
    POC_flux_out: jnp.ndarray      # sinking POC flux at layer bottom [mmol/m^2/s]
    CaCO3_flux_out: jnp.ndarray
    SiO2_flux_out: jnp.ndarray
    POC_remin: jnp.ndarray
    O2_production: jnp.ndarray
    O2_consumption: jnp.ndarray
    POC_sed_loss: jnp.ndarray      # (...,) [mmol/m^2/s]
    CaCO3_sed_loss: jnp.ndarray
    SiO2_sed_loss: jnp.ndarray
    POP_sed_loss: jnp.ndarray
    PON_sed_loss: jnp.ndarray
    P_iron_sed_loss: jnp.ndarray
    dust_Fe_release: jnp.ndarray   # (..., nlev) Fe from dust dissolution [mmol/m^3/s]
    pH_alt_co2: jnp.ndarray
    CO3_alt_co2: jnp.ndarray
    CaCO3_ALT_CO2_sed_loss: jnp.ndarray


def per_pft(values, ndim_trailing: int, dtype=None) -> jnp.ndarray:
    """Tuple of per-PFT values -> array (3, 1, ..., 1) broadcastable to (3, ..., nlev)."""
    arr = jnp.asarray(values, dtype=dtype)
    return arr.reshape(arr.shape + (1,) * ndim_trailing)
