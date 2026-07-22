"""
JAX translation of MLCanopyFluxesType Fortran module.

Defines the immutable :class:`mlcanopy_type` NamedTuple that is the
central data container for all multilayer canopy physics, together with
factory and initialisation helpers.

Public API
----------
- :class:`mlcanopy_type`: NamedTuple of JAX arrays.
- :func:`create_mlcanopy`: allocate and return a fully-initialised
  instance (replaces Fortran ``InitAllocate`` + ``Init``).
- :func:`init_cold`: cold-start leaf-water-potential and intercepted-
  water initialisation (replaces Fortran ``InitCold``).
- :func:`init_history`: stub — CLM history-file infrastructure is not
  translated.
- :func:`restart`: stub — CLM restart-file infrastructure is not
  translated.

Naming conventions preserved from Fortran:
  - ``*_canopy``  — single-level canopy variable
  - ``*_soil``    — single-level soil variable
  - ``*_forcing`` — single-level atmospheric forcing variable
  - ``*_profile`` — multi-level variable at each canopy layer
  - ``*_leaf``    — multi-level variable for sunlit and shaded leaves

Original Fortran module: MLCanopyFluxesType
Fortran lines 1-560
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import ispval, spval  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd, numrad  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import nrk  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun, nleaf, nlevmlcan  # noqa: F401

# ---------------------------------------------------------------------------
# NamedTuple definition
# ---------------------------------------------------------------------------


class mlcanopy_type(NamedTuple):
    """
    Immutable container for all multilayer canopy state and flux variables.

    Mirrors Fortran ``type :: mlcanopy_type`` (lines 18-290).

    All arrays are JAX arrays stored with 1-based patch and layer indexing
    (index 0 is allocated but unused unless the Fortran dimension starts at
    0, e.g. ``zw_profile`` and ``tbi_profile`` which run ``0:nlevmlcan``).

    Array shapes use the following conventions, where
    ``np = endp + 1`` covers 1-based patch indices:

    - ``(np,)``                   : ``*_canopy``, ``*_soil``, ``*_forcing``
    - ``(np, nlevmlcan+1)``       : ``*_profile`` (index 0 unused unless
                                    Fortran range starts at 0)
    - ``(np, nlevmlcan+1, nleaf+1)`` : ``*_leaf``
    - ``(np, numrad+1)``          : spectral canopy/soil variables
    - ``(np, nlevmlcan+1, numrad+1)`` : spectral profile variables
    - ``(np, nlevmlcan+1, nrk+1)`` : Runge-Kutta increment arrays
    - ``(np, nlevmlcan+1, nleaf+1, nrk+1)`` : leaf RK increments
    - ``(np, nlevgrnd+1)``        : soil-layer variables
    - ``(np, 3)``                 : beta-distribution parameters (index 1-2)
    """

    # ------------------------------------------------------------------
    # Vegetation input variables                         shape: (np,)
    # ------------------------------------------------------------------
    ztop_canopy: jnp.ndarray  # Canopy foliage top height (m)
    zbot_canopy: jnp.ndarray  # Canopy foliage bottom height (m)
    lai_canopy: jnp.ndarray  # Leaf area index (m2/m2)
    sai_canopy: jnp.ndarray  # Stem area index (m2/m2)
    root_biomass_canopy: jnp.ndarray  # Fine root biomass (g/m2)
    pbeta_lai_canopy: jnp.ndarray  # Beta distribution parameters for LAI (-); shape (np,3)
    pbeta_sai_canopy: jnp.ndarray  # Beta distribution parameters for SAI (-); shape (np,3)

    # ------------------------------------------------------------------
    # Atmospheric forcing variables (required input)     shape: (np,) or (np, numrad+1)
    # ------------------------------------------------------------------
    zref_forcing: jnp.ndarray  # Reference height (m)
    tref_forcing: jnp.ndarray  # Air temperature at zref, interpolated (K)
    tref_bef_forcing: jnp.ndarray  # Air temperature, previous CLM step (K)
    tref_cur_forcing: jnp.ndarray  # Air temperature, current CLM step (K)
    tref_next_forcing: jnp.ndarray  # Air temperature, next CLM step (K)
    qref_forcing: jnp.ndarray  # Specific humidity at zref, interpolated (kg/kg)
    qref_bef_forcing: jnp.ndarray  # Specific humidity, previous CLM step (kg/kg)
    qref_cur_forcing: jnp.ndarray  # Specific humidity, current CLM step (kg/kg)
    qref_next_forcing: jnp.ndarray  # Specific humidity, next CLM step (kg/kg)
    uref_forcing: jnp.ndarray  # Wind speed at zref, interpolated (m/s)
    uref_bef_forcing: jnp.ndarray  # Wind speed, previous CLM step (m/s)
    uref_cur_forcing: jnp.ndarray  # Wind speed, current CLM step (m/s)
    uref_next_forcing: jnp.ndarray  # Wind speed, next CLM step (m/s)
    pref_forcing: jnp.ndarray  # Air pressure, interpolated (Pa)
    pref_bef_forcing: jnp.ndarray  # Air pressure, previous CLM step (Pa)
    pref_cur_forcing: jnp.ndarray  # Air pressure, current CLM step (Pa)
    pref_next_forcing: jnp.ndarray  # Air pressure, next CLM step (Pa)
    co2ref_forcing: jnp.ndarray  # Atmospheric CO2, interpolated (umol/mol)
    co2ref_bef_forcing: jnp.ndarray  # Atmospheric CO2, previous CLM step (umol/mol)
    co2ref_cur_forcing: jnp.ndarray  # Atmospheric CO2, current CLM step (umol/mol)
    co2ref_next_forcing: jnp.ndarray  # Atmospheric CO2, next CLM step (umol/mol)
    o2ref_forcing: jnp.ndarray  # Atmospheric O2 (mmol/mol)
    swskyb_forcing: jnp.ndarray  # Direct beam SW, interpolated (W/m2); shape (np, numrad+1)
    swskyb_bef_forcing: jnp.ndarray  # Direct beam SW, previous CLM step (W/m2)
    swskyb_cur_forcing: jnp.ndarray  # Direct beam SW, current CLM step (W/m2)
    swskyb_next_forcing: jnp.ndarray  # Direct beam SW, next CLM step (W/m2)
    swskyd_forcing: jnp.ndarray  # Diffuse SW, interpolated (W/m2); shape (np, numrad+1)
    swskyd_bef_forcing: jnp.ndarray  # Diffuse SW, previous CLM step (W/m2)
    swskyd_cur_forcing: jnp.ndarray  # Diffuse SW, current CLM step (W/m2)
    swskyd_next_forcing: jnp.ndarray  # Diffuse SW, next CLM step (W/m2)
    lwsky_forcing: jnp.ndarray  # Atmospheric LW, interpolated (W/m2)
    lwsky_bef_forcing: jnp.ndarray  # Atmospheric LW, previous CLM step (W/m2)
    lwsky_cur_forcing: jnp.ndarray  # Atmospheric LW, current CLM step (W/m2)
    lwsky_next_forcing: jnp.ndarray  # Atmospheric LW, next CLM step (W/m2)
    qflx_rain_forcing: jnp.ndarray  # Rainfall (kg H2O/m2/s)
    qflx_snow_forcing: jnp.ndarray  # Snowfall (kg H2O/m2/s)
    tacclim_forcing: jnp.ndarray  # Average temperature for acclimation (K)

    # ------------------------------------------------------------------
    # Derived atmospheric forcing variables              shape: (np,)
    # ------------------------------------------------------------------
    eref_forcing: jnp.ndarray  # Vapor pressure at zref (Pa)
    thref_forcing: jnp.ndarray  # Potential temperature at zref (K)
    thvref_forcing: jnp.ndarray  # Virtual potential temperature at zref (K)
    rhoair_forcing: jnp.ndarray  # Air density at zref (kg/m3)
    rhomol_forcing: jnp.ndarray  # Molar density at zref (mol/m3)
    mmair_forcing: jnp.ndarray  # Molecular mass of air (kg/mol)
    cpair_forcing: jnp.ndarray  # Specific heat of air, const-P (J/mol/K)
    solar_zen_forcing: jnp.ndarray  # Solar zenith angle (radians)

    # ------------------------------------------------------------------
    # Canopy flux variables (per m2 ground)              shape: (np,) or (np, numrad+1)
    # ------------------------------------------------------------------
    swveg_canopy: jnp.ndarray  # Absorbed SW: vegetation (W/m2); shape (np, numrad+1)
    swvegsun_canopy: jnp.ndarray  # Absorbed SW: sunlit canopy (W/m2)
    swvegsha_canopy: jnp.ndarray  # Absorbed SW: shaded canopy (W/m2)
    lwveg_canopy: jnp.ndarray  # Absorbed LW: vegetation (W/m2)
    lwvegsun_canopy: jnp.ndarray  # Absorbed LW: sunlit canopy (W/m2)
    lwvegsha_canopy: jnp.ndarray  # Absorbed LW: shaded canopy (W/m2)
    shveg_canopy: jnp.ndarray  # Sensible heat: vegetation (W/m2)
    shvegsun_canopy: jnp.ndarray  # Sensible heat: sunlit canopy (W/m2)
    shvegsha_canopy: jnp.ndarray  # Sensible heat: shaded canopy (W/m2)
    lhveg_canopy: jnp.ndarray  # Latent heat: vegetation (W/m2)
    lhvegsun_canopy: jnp.ndarray  # Latent heat: sunlit canopy (W/m2)
    lhvegsha_canopy: jnp.ndarray  # Latent heat: shaded canopy (W/m2)
    etveg_canopy: jnp.ndarray  # Water vapor: vegetation (mol H2O/m2/s)
    etvegsun_canopy: jnp.ndarray  # Water vapor: sunlit canopy (mol H2O/m2/s)
    etvegsha_canopy: jnp.ndarray  # Water vapor: shaded canopy (mol H2O/m2/s)
    trveg_canopy: jnp.ndarray  # Transpiration (mol H2O/m2/s)
    evveg_canopy: jnp.ndarray  # Canopy evaporation (mol H2O/m2/s)
    gppveg_canopy: jnp.ndarray  # GPP: vegetation (umol CO2/m2/s)
    gppvegsun_canopy: jnp.ndarray  # GPP: sunlit canopy (umol CO2/m2/s)
    gppvegsha_canopy: jnp.ndarray  # GPP: shaded canopy (umol CO2/m2/s)
    vcmax25veg_canopy: jnp.ndarray  # Vcmax25: total canopy (umol/m2/s)
    vcmax25sun_canopy: jnp.ndarray  # Vcmax25: sunlit canopy (umol/m2/s)
    vcmax25sha_canopy: jnp.ndarray  # Vcmax25: shaded canopy (umol/m2/s)
    gsveg_canopy: jnp.ndarray  # Stomatal conductance: canopy (mol H2O/m2/s)
    gsvegsun_canopy: jnp.ndarray  # Stomatal conductance: sunlit (mol H2O/m2/s)
    gsvegsha_canopy: jnp.ndarray  # Stomatal conductance: shaded (mol H2O/m2/s)
    windveg_canopy: jnp.ndarray  # Wind speed: canopy (m/s)
    windvegsun_canopy: jnp.ndarray  # Wind speed: sunlit canopy (m/s)
    windvegsha_canopy: jnp.ndarray  # Wind speed: shaded canopy (m/s)
    tlveg_canopy: jnp.ndarray  # Leaf temperature: canopy (K)
    tlvegsun_canopy: jnp.ndarray  # Leaf temperature: sunlit (K)
    tlvegsha_canopy: jnp.ndarray  # Leaf temperature: shaded (K)
    taveg_canopy: jnp.ndarray  # Air temperature: canopy (K)
    tavegsun_canopy: jnp.ndarray  # Air temperature: sunlit (K)
    tavegsha_canopy: jnp.ndarray  # Air temperature: shaded (K)
    laisun_canopy: jnp.ndarray  # PAI: sunlit canopy (m2/m2)
    laisha_canopy: jnp.ndarray  # PAI: shaded canopy (m2/m2)
    albcan_canopy: jnp.ndarray  # Albedo above canopy (-); shape (np, numrad+1)
    lwup_canopy: jnp.ndarray  # Upward LW above canopy (W/m2)
    rnet_canopy: jnp.ndarray  # Net radiation including soil (W/m2)
    shflx_canopy: jnp.ndarray  # Sensible heat including soil (W/m2)
    lhflx_canopy: jnp.ndarray  # Latent heat including soil (W/m2)
    etflx_canopy: jnp.ndarray  # Water vapor including soil (mol H2O/m2/s)
    stflx_air_canopy: jnp.ndarray  # Canopy air storage heat flux (W/m2)
    stflx_veg_canopy: jnp.ndarray  # Canopy biomass storage heat flux (W/m2)
    ustar_canopy: jnp.ndarray  # Friction velocity (m/s)
    gac_to_hc_canopy: jnp.ndarray  # Aerodynamic conductance above canopy (mol/m2/s)
    qflx_intr_canopy: jnp.ndarray  # Intercepted precipitation (kg H2O/m2/s)
    qflx_tflrain_canopy: jnp.ndarray  # Rain throughfall onto ground (kg H2O/m2/s)
    qflx_tflsnow_canopy: jnp.ndarray  # Snow throughfall onto ground (kg H2O/m2/s)

    # ------------------------------------------------------------------
    # Canopy diagnostic variables                        shape: (np,)
    # ------------------------------------------------------------------
    uaf_canopy: jnp.ndarray  # Wind speed at canopy top (m/s)
    taf_canopy: jnp.ndarray  # Air temperature at canopy top for ObuFunc (K)
    qaf_canopy: jnp.ndarray  # Specific humidity at canopy top for ObuFunc (kg/kg)
    fracminlwp_canopy: jnp.ndarray  # Fraction of canopy that is water-stressed

    # ------------------------------------------------------------------
    # Canopy aerodynamic variables                       shape: (np,)
    # ------------------------------------------------------------------
    obu_canopy: jnp.ndarray  # Obukhov length (m)
    beta_canopy: jnp.ndarray  # u* / u at canopy top (-)
    PrSc_canopy: jnp.ndarray  # Prandtl (Schmidt) number at canopy top (-)
    Lc_canopy: jnp.ndarray  # Canopy density length scale (m)
    zdisp_canopy: jnp.ndarray  # Displacement height (m)
    z0m_canopy: jnp.ndarray  # Roughness length for momentum (m)

    # ------------------------------------------------------------------
    # Canopy stomatal conductance variables              shape: (np,)
    # ------------------------------------------------------------------
    g0_canopy: jnp.ndarray  # Ball-Berry/Medlyn minimum conductance (mol H2O/m2/s)
    g1_canopy: jnp.ndarray  # Ball-Berry/Medlyn slope parameter

    # ------------------------------------------------------------------
    # Soil energy balance variables                      shape: (np,) or (np, numrad+1) or (np, nrk+1)
    # ------------------------------------------------------------------
    albsoib_soil: jnp.ndarray  # Direct beam albedo of ground (-); shape (np, numrad+1)
    albsoid_soil: jnp.ndarray  # Diffuse albedo of ground (-); shape (np, numrad+1)
    swsoi_soil: jnp.ndarray  # Absorbed SW: ground (W/m2); shape (np, numrad+1)
    lwsoi_soil: jnp.ndarray  # Absorbed LW: ground (W/m2)
    rnsoi_soil: jnp.ndarray  # Net radiation: ground (W/m2)
    shsoi_soil: jnp.ndarray  # Sensible heat: ground (W/m2)
    lhsoi_soil: jnp.ndarray  # Latent heat: ground (W/m2)
    etsoi_soil: jnp.ndarray  # Water vapor: ground (mol H2O/m2/s)
    gsoi_soil: jnp.ndarray  # Soil heat flux (W/m2)
    tg_soil: jnp.ndarray  # Soil surface temperature (K)
    tg_bef_soil: jnp.ndarray  # Soil surface temperature, previous step (K)
    dtg_soil: jnp.ndarray  # Change in tg over RK step (K); shape (np, nrk+1)
    eg_soil: jnp.ndarray  # Soil surface vapor pressure (Pa)
    rhg_soil: jnp.ndarray  # Relative humidity at soil surface (-)
    gac0_soil: jnp.ndarray  # Aerodynamic conductance for soil (mol/m2/s)
    soil_t_soil: jnp.ndarray  # Temperature of first snow/soil layer (K)
    soil_dz_soil: jnp.ndarray  # Depth to first snow/soil layer (m)
    soil_tk_soil: jnp.ndarray  # Thermal conductivity of first layer (W/m/K)
    soilres_soil: jnp.ndarray  # Soil evaporative resistance (s/m)

    # ------------------------------------------------------------------
    # Soil moisture variables                            shape: (np,) or (np, nlevgrnd+1)
    # ------------------------------------------------------------------
    btran_soil: jnp.ndarray  # Soil wetness factor for photosynthesis (-)
    psis_soil: jnp.ndarray  # Weighted soil water potential (MPa)
    rsoil_soil: jnp.ndarray  # Soil hydraulic resistance (MPa.s.m2/mmol H2O)
    soil_et_loss_soil: (
        jnp.ndarray
    )  # Fraction of ET from each soil layer (-); shape (np, nlevgrnd+1)

    # ------------------------------------------------------------------
    # Canopy layer indices                               shape: (np,) int
    # ------------------------------------------------------------------
    ncan_canopy: jnp.ndarray  # Number of aboveground layers
    ntop_canopy: jnp.ndarray  # Index for top leaf layer
    nbot_canopy: jnp.ndarray  # Index for bottom leaf layer

    # ------------------------------------------------------------------
    # Canopy layer variables                             shape: (np, nlevmlcan+1) or with extra dims
    # ------------------------------------------------------------------
    dlai_frac_profile: jnp.ndarray  # Layer LAI (fraction of canopy total)
    dsai_frac_profile: jnp.ndarray  # Layer SAI (fraction of canopy total)
    dlai_profile: jnp.ndarray  # Layer LAI (m2/m2)
    dsai_profile: jnp.ndarray  # Layer SAI (m2/m2)
    dpai_profile: jnp.ndarray  # Layer PAI (m2/m2)
    zs_profile: jnp.ndarray  # Layer height for scalar concentration (m)
    zw_profile: jnp.ndarray  # Layer interface height; shape (np, nlevmlcan+1), index 0..nlevmlcan
    dz_profile: jnp.ndarray  # Layer thickness (m)

    vcmax25_profile: jnp.ndarray  # Layer vcmax at 25C (umol/m2/s)
    jmax25_profile: jnp.ndarray  # Layer jmax at 25C (umol/m2/s)
    kp25_profile: jnp.ndarray  # Layer C4 kp at 25C (mol/m2/s)
    rd25_profile: jnp.ndarray  # Layer rd at 25C (umol/m2/s)
    cpleaf_profile: jnp.ndarray  # Layer leaf heat capacity (J/m2 leaf/K)

    fracsun_profile: jnp.ndarray  # Layer sunlit fraction (-)
    kb_profile: jnp.ndarray  # Direct beam extinction coefficient (-)
    tb_profile: jnp.ndarray  # Layer transmittance of direct beam (-)
    td_profile: jnp.ndarray  # Layer transmittance of diffuse radiation (-)
    tbi_profile: (
        jnp.ndarray
    )  # Cumulative transmittance of direct beam; shape (np, nlevmlcan+1), index 0..nlevmlcan
    swbeam_profile: (
        jnp.ndarray
    )  # Direct beam SW above layer (W/m2); shape (np, nlevmlcan+1, numrad+1)
    swupw_profile: (
        jnp.ndarray
    )  # Upward diffuse SW above layer (W/m2); shape (np, nlevmlcan+1, numrad+1)
    swdwn_profile: (
        jnp.ndarray
    )  # Downward diffuse SW above layer (W/m2); shape (np, nlevmlcan+1, numrad+1)
    lwupw_profile: jnp.ndarray  # Upward LW above layer (W/m2); shape (np, nlevmlcan+1)
    lwdwn_profile: jnp.ndarray  # Downward LW above layer (W/m2); shape (np, nlevmlcan+1)

    swsrc_profile: (
        jnp.ndarray
    )  # Layer source: absorbed SW (W/m2); shape (np, nlevmlcan+1, numrad+1)
    lwsrc_profile: jnp.ndarray  # Layer source: absorbed LW (W/m2)
    rnsrc_profile: jnp.ndarray  # Layer source: net radiation (W/m2)
    stsrc_profile: jnp.ndarray  # Layer source: storage heat flux (W/m2)
    shsrc_profile: jnp.ndarray  # Layer source: sensible heat (W/m2)
    lhsrc_profile: jnp.ndarray  # Layer source: latent heat (W/m2)
    etsrc_profile: jnp.ndarray  # Layer source: water vapor (mol H2O/m2/s)
    trsrc_profile: jnp.ndarray  # Layer source: transpiration (mol H2O/m2/s)
    evsrc_profile: jnp.ndarray  # Layer source: evaporation (mol H2O/m2/s)
    fco2src_profile: jnp.ndarray  # Layer source: CO2 (umol CO2/m2/s)

    wind_profile: jnp.ndarray  # Layer wind speed (m/s)
    tair_profile: jnp.ndarray  # Layer air temperature (K)
    eair_profile: jnp.ndarray  # Layer vapor pressure (Pa)
    cair_profile: jnp.ndarray  # Layer CO2 (umol/mol)
    tair_bef_profile: jnp.ndarray  # Layer air temperature, previous step (K)
    eair_bef_profile: jnp.ndarray  # Layer vapor pressure, previous step (Pa)
    cair_bef_profile: jnp.ndarray  # Layer CO2, previous step (umol/mol)
    dtair_profile: jnp.ndarray  # Change in tair over RK step (K); shape (np, nlevmlcan+1, nrk+1)
    deair_profile: jnp.ndarray  # Change in eair over RK step (Pa); shape (np, nlevmlcan+1, nrk+1)
    wind_data_profile: jnp.ndarray  # Layer wind speed FROM DATASET (m/s)
    tair_data_profile: jnp.ndarray  # Layer air temperature FROM DATASET (K)
    eair_data_profile: jnp.ndarray  # Layer vapor pressure FROM DATASET (Pa)

    shair_profile: jnp.ndarray  # Layer air sensible heat flux (W/m2)
    etair_profile: jnp.ndarray  # Layer air water vapor flux (mol H2O/m2/s)
    stair_profile: jnp.ndarray  # Layer air storage heat flux (W/m2)
    mflx_profile: jnp.ndarray  # Layer momentum flux (m2/s2)
    gac_profile: jnp.ndarray  # Layer aerodynamic conductance (mol/m2/s)
    kc_eddy_profile: jnp.ndarray  # Layer eddy diffusivity, HF (m2/s)

    swleaf_mean_profile: (
        jnp.ndarray
    )  # Layer mean: absorbed SW per leaf (W/m2 leaf); shape (np, nlevmlcan+1, numrad+1)
    lwleaf_mean_profile: jnp.ndarray  # Layer mean: absorbed LW per leaf (W/m2 leaf)
    rnleaf_mean_profile: jnp.ndarray  # Layer mean: net radiation per leaf (W/m2 leaf)
    stleaf_mean_profile: jnp.ndarray  # Layer mean: storage heat per leaf (W/m2 leaf)
    shleaf_mean_profile: jnp.ndarray  # Layer mean: sensible heat per leaf (W/m2 leaf)
    lhleaf_mean_profile: jnp.ndarray  # Layer mean: latent heat per leaf (W/m2 leaf)
    etleaf_mean_profile: jnp.ndarray  # Layer mean: water vapor per leaf (mol H2O/m2 leaf/s)
    trleaf_mean_profile: jnp.ndarray  # Layer mean: transpiration per leaf (mol H2O/m2 leaf/s)
    evleaf_mean_profile: jnp.ndarray  # Layer mean: evaporation per leaf (mol H2O/m2 leaf/s)
    fco2_mean_profile: jnp.ndarray  # Layer mean: net photosynthesis (umol CO2/m2 leaf/s)
    apar_mean_profile: jnp.ndarray  # Layer mean: absorbed PAR (umol/m2 leaf/s)
    gs_mean_profile: jnp.ndarray  # Layer mean: stomatal conductance (mol H2O/m2 leaf/s)
    tleaf_mean_profile: jnp.ndarray  # Layer mean: leaf temperature (K)
    lwp_mean_profile: jnp.ndarray  # Layer mean: leaf water potential (MPa)

    lsc_profile: jnp.ndarray  # Layer leaf-specific conductance (mmol H2O/m2 leaf/s/MPa)
    h2ocan_profile: jnp.ndarray  # Layer intercepted water (kg H2O/m2)
    h2ocan_bef_profile: jnp.ndarray  # Layer intercepted water, previous step (kg H2O/m2)
    dh2ocan_profile: jnp.ndarray  # Change in h2ocan over RK step; shape (np, nlevmlcan+1, nrk+1)
    fwet_profile: jnp.ndarray  # Layer wet fraction of PAI (-)
    fdry_profile: jnp.ndarray  # Layer green-dry fraction of PAI (-)

    # ------------------------------------------------------------------
    # Sunlit / shaded leaf variables                     shape: (np, nlevmlcan+1, nleaf+1) or with extra dims
    # ------------------------------------------------------------------
    tleaf_leaf: jnp.ndarray  # Leaf temperature (K)
    tleaf_bef_leaf: jnp.ndarray  # Leaf temperature, previous step (K)
    dtleaf_leaf: (
        jnp.ndarray
    )  # Change in tleaf over RK step (K); shape (np, nlevmlcan+1, nleaf+1, nrk+1)
    tleaf_hist_leaf: jnp.ndarray  # Leaf temperature for history files (K)
    swleaf_leaf: (
        jnp.ndarray
    )  # Leaf absorbed SW (W/m2 leaf); shape (np, nlevmlcan+1, nleaf+1, numrad+1)
    lwleaf_leaf: jnp.ndarray  # Leaf absorbed LW (W/m2 leaf)
    rnleaf_leaf: jnp.ndarray  # Leaf net radiation (W/m2 leaf)
    stleaf_leaf: jnp.ndarray  # Leaf storage heat flux (W/m2 leaf)
    shleaf_leaf: jnp.ndarray  # Leaf sensible heat flux (W/m2 leaf)
    lhleaf_leaf: jnp.ndarray  # Leaf latent heat flux (W/m2 leaf)
    trleaf_leaf: jnp.ndarray  # Leaf transpiration (mol H2O/m2 leaf/s)
    evleaf_leaf: jnp.ndarray  # Leaf evaporation (mol H2O/m2 leaf/s)

    gbh_leaf: jnp.ndarray  # Boundary layer conductance: heat (mol/m2 leaf/s)
    gbv_leaf: jnp.ndarray  # Boundary layer conductance: H2O (mol H2O/m2 leaf/s)
    gbc_leaf: jnp.ndarray  # Boundary layer conductance: CO2 (mol CO2/m2 leaf/s)

    vcmax25_leaf: jnp.ndarray  # Leaf vcmax at 25C (umol/m2/s)
    jmax25_leaf: jnp.ndarray  # Leaf jmax at 25C (umol/m2/s)
    kp25_leaf: jnp.ndarray  # Leaf C4 kp at 25C (mol/m2/s)
    rd25_leaf: jnp.ndarray  # Leaf rd at 25C (umol/m2/s)

    kc_leaf: jnp.ndarray  # Michaelis-Menten constant for CO2 (umol/mol)
    ko_leaf: jnp.ndarray  # Michaelis-Menten constant for O2 (mmol/mol)
    cp_leaf: jnp.ndarray  # CO2 compensation point (umol/mol)
    vcmax_leaf: jnp.ndarray  # Vcmax at leaf temperature (umol/m2/s)
    jmax_leaf: jnp.ndarray  # Jmax at leaf temperature (umol/m2/s)
    kp_leaf: jnp.ndarray  # C4 kp at leaf temperature (mol/m2/s)
    ceair_leaf: jnp.ndarray  # Vapor pressure constrained for stomata (Pa)
    leaf_esat_leaf: jnp.ndarray  # Leaf saturation vapor pressure (Pa)

    apar_leaf: jnp.ndarray  # Absorbed PAR (umol photon/m2 leaf/s)
    je_leaf: jnp.ndarray  # Electron transport rate (umol/m2/s)
    ac_leaf: jnp.ndarray  # Rubisco-limited gross An (umol CO2/m2 leaf/s)
    aj_leaf: jnp.ndarray  # RuBP-limited gross An (umol CO2/m2 leaf/s)
    ap_leaf: jnp.ndarray  # Product/CO2-limited gross An (umol CO2/m2 leaf/s)
    agross_leaf: jnp.ndarray  # Gross photosynthesis (umol CO2/m2 leaf/s)
    anet_leaf: jnp.ndarray  # Net photosynthesis (umol CO2/m2 leaf/s)
    rd_leaf: jnp.ndarray  # Leaf respiration rate (umol CO2/m2 leaf/s)
    ci_leaf: jnp.ndarray  # Intercellular CO2 (umol/mol)
    cs_leaf: jnp.ndarray  # Leaf surface CO2 (umol/mol)

    lwp_leaf: jnp.ndarray  # Leaf water potential (MPa)
    lwp_bef_leaf: jnp.ndarray  # Leaf water potential, previous step (MPa)
    dlwp_leaf: (
        jnp.ndarray
    )  # Change in lwp over RK step (MPa); shape (np, nlevmlcan+1, nleaf+1, nrk+1)
    lwp_hist_leaf: jnp.ndarray  # Leaf water potential for history files (MPa)
    hs_leaf: jnp.ndarray  # Fractional humidity at leaf surface (-)
    vpd_leaf: jnp.ndarray  # Leaf vapor pressure deficit (Pa)
    gs_leaf: jnp.ndarray  # Stomatal conductance (mol H2O/m2 leaf/s)
    gspot_leaf: jnp.ndarray  # Stomatal conductance without water stress (mol H2O/m2 leaf/s)


# ---------------------------------------------------------------------------
# Factory function: replaces InitAllocate + Init
# ---------------------------------------------------------------------------


def create_mlcanopy(begp: int, endp: int) -> mlcanopy_type:
    """
    Allocate and initialise a :class:`mlcanopy_type` instance.

    Mirrors Fortran subroutine ``InitAllocate`` (lines 345-480) called
    from ``Init`` (lines 320-333).

    All ``real(r8)`` arrays are filled with ``spval``; integer arrays
    are filled with ``ispval``.  The ``init_cold`` function must be
    called subsequently to set physically meaningful initial values for
    leaf water potential and intercepted water.

    Dimension conventions (1-based indexing preserved throughout):

    ============================================  =======================================
    Fortran declaration                           Python shape
    ============================================  =======================================
    ``(begp:endp)``                               ``(endp+1,)``
    ``(begp:endp, 1:nlevmlcan)``                  ``(endp+1, nlevmlcan+1)``
    ``(begp:endp, 0:nlevmlcan)``                  ``(endp+1, nlevmlcan+1)``
    ``(begp:endp, 1:numrad)``                     ``(endp+1, numrad+1)``
    ``(begp:endp, 1:nleaf)``                      ``(endp+1, nleaf+1)``
    ``(begp:endp, 1:nrk)``                        ``(endp+1, nrk+1)``
    ``(begp:endp, 1:nlevmlcan, 1:nleaf)``         ``(endp+1, nlevmlcan+1, nleaf+1)``
    ``(begp:endp, 1:nlevmlcan, 1:numrad)``        ``(endp+1, nlevmlcan+1, numrad+1)``
    ``(begp:endp, 0:nlevmlcan, 1:numrad)``        ``(endp+1, nlevmlcan+1, numrad+1)``
    ``(begp:endp, 1:nlevmlcan, 1:nrk)``           ``(endp+1, nlevmlcan+1, nrk+1)``
    ``(begp:endp, 1:nlevmlcan, 1:nleaf, 1:nrk)`` ``(endp+1, nlevmlcan+1, nleaf+1, nrk+1)``
    ``(begp:endp, 1:2)``                          ``(endp+1, 3)``
    ``(begp:endp, 1:nlevgrnd)``                   ``(endp+1, nlevgrnd+1)``
    ============================================  =======================================

    Args:
        begp: First patch index (typically 1).
        endp: Last patch index.

    Returns:
        Fully allocated :class:`mlcanopy_type` with all values set to
        ``spval`` (real) or ``ispval`` (integer).
    """
    np_ = endp + 1  # patch dimension
    nl = nlevmlcan + 1  # level dimension (0..nlevmlcan or 1..nlevmlcan)
    nf = nleaf + 1  # leaf dimension
    nr = numrad + 1  # spectral dimension
    nrk_ = nrk + 1  # Runge-Kutta stage dimension
    ng = nlevgrnd + 1  # soil layer dimension

    def _r(*shape):
        """Float array filled with spval."""
        return jnp.full(shape, spval, dtype=jnp.float64)

    def _i(*shape):
        """Integer array filled with ispval."""
        return jnp.full(shape, ispval, dtype=jnp.int32)

    return mlcanopy_type(
        # Vegetation input
        ztop_canopy=_r(np_),
        zbot_canopy=_r(np_),
        lai_canopy=_r(np_),
        sai_canopy=_r(np_),
        root_biomass_canopy=_r(np_),
        pbeta_lai_canopy=_r(np_, 3),
        pbeta_sai_canopy=_r(np_, 3),
        # Atmospheric forcing
        zref_forcing=_r(np_),
        tref_forcing=_r(np_),
        tref_bef_forcing=_r(np_),
        tref_cur_forcing=_r(np_),
        tref_next_forcing=_r(np_),
        qref_forcing=_r(np_),
        qref_bef_forcing=_r(np_),
        qref_cur_forcing=_r(np_),
        qref_next_forcing=_r(np_),
        uref_forcing=_r(np_),
        uref_bef_forcing=_r(np_),
        uref_cur_forcing=_r(np_),
        uref_next_forcing=_r(np_),
        pref_forcing=_r(np_),
        pref_bef_forcing=_r(np_),
        pref_cur_forcing=_r(np_),
        pref_next_forcing=_r(np_),
        co2ref_forcing=_r(np_),
        co2ref_bef_forcing=_r(np_),
        co2ref_cur_forcing=_r(np_),
        co2ref_next_forcing=_r(np_),
        o2ref_forcing=_r(np_),
        swskyb_forcing=_r(np_, nr),
        swskyb_bef_forcing=_r(np_, nr),
        swskyb_cur_forcing=_r(np_, nr),
        swskyb_next_forcing=_r(np_, nr),
        swskyd_forcing=_r(np_, nr),
        swskyd_bef_forcing=_r(np_, nr),
        swskyd_cur_forcing=_r(np_, nr),
        swskyd_next_forcing=_r(np_, nr),
        lwsky_forcing=_r(np_),
        lwsky_bef_forcing=_r(np_),
        lwsky_cur_forcing=_r(np_),
        lwsky_next_forcing=_r(np_),
        qflx_rain_forcing=_r(np_),
        qflx_snow_forcing=_r(np_),
        tacclim_forcing=_r(np_),
        # Derived atmospheric forcing
        eref_forcing=_r(np_),
        thref_forcing=_r(np_),
        thvref_forcing=_r(np_),
        rhoair_forcing=_r(np_),
        rhomol_forcing=_r(np_),
        mmair_forcing=_r(np_),
        cpair_forcing=_r(np_),
        solar_zen_forcing=_r(np_),
        # Canopy fluxes
        swveg_canopy=_r(np_, nr),
        swvegsun_canopy=_r(np_, nr),
        swvegsha_canopy=_r(np_, nr),
        lwveg_canopy=_r(np_),
        lwvegsun_canopy=_r(np_),
        lwvegsha_canopy=_r(np_),
        shveg_canopy=_r(np_),
        shvegsun_canopy=_r(np_),
        shvegsha_canopy=_r(np_),
        lhveg_canopy=_r(np_),
        lhvegsun_canopy=_r(np_),
        lhvegsha_canopy=_r(np_),
        etveg_canopy=_r(np_),
        etvegsun_canopy=_r(np_),
        etvegsha_canopy=_r(np_),
        trveg_canopy=_r(np_),
        evveg_canopy=_r(np_),
        gppveg_canopy=_r(np_),
        gppvegsun_canopy=_r(np_),
        gppvegsha_canopy=_r(np_),
        vcmax25veg_canopy=_r(np_),
        vcmax25sun_canopy=_r(np_),
        vcmax25sha_canopy=_r(np_),
        gsveg_canopy=_r(np_),
        gsvegsun_canopy=_r(np_),
        gsvegsha_canopy=_r(np_),
        windveg_canopy=_r(np_),
        windvegsun_canopy=_r(np_),
        windvegsha_canopy=_r(np_),
        tlveg_canopy=_r(np_),
        tlvegsun_canopy=_r(np_),
        tlvegsha_canopy=_r(np_),
        taveg_canopy=_r(np_),
        tavegsun_canopy=_r(np_),
        tavegsha_canopy=_r(np_),
        laisun_canopy=_r(np_),
        laisha_canopy=_r(np_),
        albcan_canopy=_r(np_, nr),
        lwup_canopy=_r(np_),
        rnet_canopy=_r(np_),
        shflx_canopy=_r(np_),
        lhflx_canopy=_r(np_),
        etflx_canopy=_r(np_),
        stflx_air_canopy=_r(np_),
        stflx_veg_canopy=_r(np_),
        ustar_canopy=_r(np_),
        gac_to_hc_canopy=_r(np_),
        qflx_intr_canopy=_r(np_),
        qflx_tflrain_canopy=_r(np_),
        qflx_tflsnow_canopy=_r(np_),
        # Canopy diagnostics
        uaf_canopy=_r(np_),
        taf_canopy=_r(np_),
        qaf_canopy=_r(np_),
        fracminlwp_canopy=_r(np_),
        # Canopy aerodynamics
        obu_canopy=_r(np_),
        beta_canopy=_r(np_),
        PrSc_canopy=_r(np_),
        Lc_canopy=_r(np_),
        zdisp_canopy=_r(np_),
        z0m_canopy=_r(np_),
        # Stomatal conductance
        g0_canopy=_r(np_),
        g1_canopy=_r(np_),
        # Soil energy balance
        albsoib_soil=_r(np_, nr),
        albsoid_soil=_r(np_, nr),
        swsoi_soil=_r(np_, nr),
        lwsoi_soil=_r(np_),
        rnsoi_soil=_r(np_),
        shsoi_soil=_r(np_),
        lhsoi_soil=_r(np_),
        etsoi_soil=_r(np_),
        gsoi_soil=_r(np_),
        tg_soil=_r(np_),
        tg_bef_soil=_r(np_),
        dtg_soil=_r(np_, nrk_),
        eg_soil=_r(np_),
        rhg_soil=_r(np_),
        gac0_soil=_r(np_),
        soil_t_soil=_r(np_),
        soil_dz_soil=_r(np_),
        soil_tk_soil=_r(np_),
        soilres_soil=_r(np_),
        # Soil moisture
        btran_soil=_r(np_),
        psis_soil=_r(np_),
        rsoil_soil=_r(np_),
        soil_et_loss_soil=_r(np_, ng),
        # Canopy layer indices (integer)
        ncan_canopy=_i(np_),
        ntop_canopy=_i(np_),
        nbot_canopy=_i(np_),
        # Canopy layer variables
        dlai_frac_profile=_r(np_, nl),
        dsai_frac_profile=_r(np_, nl),
        dlai_profile=_r(np_, nl),
        dsai_profile=_r(np_, nl),
        dpai_profile=_r(np_, nl),
        zs_profile=_r(np_, nl),
        zw_profile=_r(np_, nl),  # Fortran: 0:nlevmlcan; nl = nlevmlcan+1
        dz_profile=_r(np_, nl),
        vcmax25_profile=_r(np_, nl),
        jmax25_profile=_r(np_, nl),
        kp25_profile=_r(np_, nl),
        rd25_profile=_r(np_, nl),
        cpleaf_profile=_r(np_, nl),
        fracsun_profile=_r(np_, nl),
        kb_profile=_r(np_, nl),
        tb_profile=_r(np_, nl),
        td_profile=_r(np_, nl),
        tbi_profile=_r(np_, nl),  # Fortran: 0:nlevmlcan
        swbeam_profile=_r(np_, nl, nr),
        swupw_profile=_r(np_, nl, nr),
        swdwn_profile=_r(np_, nl, nr),
        lwupw_profile=_r(np_, nl),
        lwdwn_profile=_r(np_, nl),
        swsrc_profile=_r(np_, nl, nr),
        lwsrc_profile=_r(np_, nl),
        rnsrc_profile=_r(np_, nl),
        stsrc_profile=_r(np_, nl),
        shsrc_profile=_r(np_, nl),
        lhsrc_profile=_r(np_, nl),
        etsrc_profile=_r(np_, nl),
        trsrc_profile=_r(np_, nl),
        evsrc_profile=_r(np_, nl),
        fco2src_profile=_r(np_, nl),
        wind_profile=_r(np_, nl),
        tair_profile=_r(np_, nl),
        eair_profile=_r(np_, nl),
        cair_profile=_r(np_, nl),
        tair_bef_profile=_r(np_, nl),
        eair_bef_profile=_r(np_, nl),
        cair_bef_profile=_r(np_, nl),
        dtair_profile=_r(np_, nl, nrk_),
        deair_profile=_r(np_, nl, nrk_),
        wind_data_profile=_r(np_, nl),
        tair_data_profile=_r(np_, nl),
        eair_data_profile=_r(np_, nl),
        shair_profile=_r(np_, nl),
        etair_profile=_r(np_, nl),
        stair_profile=_r(np_, nl),
        mflx_profile=_r(np_, nl),
        gac_profile=_r(np_, nl),
        kc_eddy_profile=_r(np_, nl),
        swleaf_mean_profile=_r(np_, nl, nr),
        lwleaf_mean_profile=_r(np_, nl),
        rnleaf_mean_profile=_r(np_, nl),
        stleaf_mean_profile=_r(np_, nl),
        shleaf_mean_profile=_r(np_, nl),
        lhleaf_mean_profile=_r(np_, nl),
        etleaf_mean_profile=_r(np_, nl),
        trleaf_mean_profile=_r(np_, nl),
        evleaf_mean_profile=_r(np_, nl),
        fco2_mean_profile=_r(np_, nl),
        apar_mean_profile=_r(np_, nl),
        gs_mean_profile=_r(np_, nl),
        tleaf_mean_profile=_r(np_, nl),
        lwp_mean_profile=_r(np_, nl),
        lsc_profile=_r(np_, nl),
        h2ocan_profile=_r(np_, nl),
        h2ocan_bef_profile=_r(np_, nl),
        dh2ocan_profile=_r(np_, nl, nrk_),
        fwet_profile=_r(np_, nl),
        fdry_profile=_r(np_, nl),
        # Sunlit/shaded leaf variables
        tleaf_leaf=_r(np_, nl, nf),
        tleaf_bef_leaf=_r(np_, nl, nf),
        dtleaf_leaf=_r(np_, nl, nf, nrk_),
        tleaf_hist_leaf=_r(np_, nl, nf),
        swleaf_leaf=_r(np_, nl, nf, nr),
        lwleaf_leaf=_r(np_, nl, nf),
        rnleaf_leaf=_r(np_, nl, nf),
        stleaf_leaf=_r(np_, nl, nf),
        shleaf_leaf=_r(np_, nl, nf),
        lhleaf_leaf=_r(np_, nl, nf),
        trleaf_leaf=_r(np_, nl, nf),
        evleaf_leaf=_r(np_, nl, nf),
        gbh_leaf=_r(np_, nl, nf),
        gbv_leaf=_r(np_, nl, nf),
        gbc_leaf=_r(np_, nl, nf),
        vcmax25_leaf=_r(np_, nl, nf),
        jmax25_leaf=_r(np_, nl, nf),
        kp25_leaf=_r(np_, nl, nf),
        rd25_leaf=_r(np_, nl, nf),
        kc_leaf=_r(np_, nl, nf),
        ko_leaf=_r(np_, nl, nf),
        cp_leaf=_r(np_, nl, nf),
        vcmax_leaf=_r(np_, nl, nf),
        jmax_leaf=_r(np_, nl, nf),
        kp_leaf=_r(np_, nl, nf),
        ceair_leaf=_r(np_, nl, nf),
        leaf_esat_leaf=_r(np_, nl, nf),
        apar_leaf=_r(np_, nl, nf),
        je_leaf=_r(np_, nl, nf),
        ac_leaf=_r(np_, nl, nf),
        aj_leaf=_r(np_, nl, nf),
        ap_leaf=_r(np_, nl, nf),
        agross_leaf=_r(np_, nl, nf),
        anet_leaf=_r(np_, nl, nf),
        rd_leaf=_r(np_, nl, nf),
        ci_leaf=_r(np_, nl, nf),
        cs_leaf=_r(np_, nl, nf),
        lwp_leaf=_r(np_, nl, nf),
        lwp_bef_leaf=_r(np_, nl, nf),
        dlwp_leaf=_r(np_, nl, nf, nrk_),
        lwp_hist_leaf=_r(np_, nl, nf),
        hs_leaf=_r(np_, nl, nf),
        vpd_leaf=_r(np_, nl, nf),
        gs_leaf=_r(np_, nl, nf),
        gspot_leaf=_r(np_, nl, nf),
    )


# ---------------------------------------------------------------------------
# Cold-start initialisation
# ---------------------------------------------------------------------------


def init_cold(mlcanopy_inst: mlcanopy_type, begp: int, endp: int) -> mlcanopy_type:
    """
    Cold-start initialisation for leaf water potential and intercepted water.

    Mirrors Fortran subroutine ``InitCold`` (lines 495-520).

    .. warning::
        These values are overwritten by ``initVerticalProfiles``; this
        function sets only the minimal state needed to avoid uninitialised
        reads on the first timestep.

    Sets (Fortran lines 513-516):

    .. code-block:: none

        lwp_leaf(p, ic, isun) = -0.1 MPa
        lwp_leaf(p, ic, isha) = -0.1 MPa
        h2ocan_profile(p, ic) = 0.0 kg H2O/m2

    for all patches ``p ∈ [begp, endp]`` and layers
    ``ic ∈ [1, nlevmlcan]``.

    Args:
        mlcanopy_inst: Container allocated by :func:`create_mlcanopy`.
        begp: First patch index.
        endp: Last patch index.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    lwp_leaf = mlcanopy_inst.lwp_leaf
    h2ocan = mlcanopy_inst.h2ocan_profile

    for p in range(begp, endp + 1):  # Fortran: do p = bounds%begp, bounds%endp
        for ic in range(1, nlevmlcan + 1):  # Fortran: do ic = 1, nlevmlcan
            lwp_leaf = lwp_leaf.at[p, ic, isun].set(-0.1)
            lwp_leaf = lwp_leaf.at[p, ic, isha].set(-0.1)
            h2ocan = h2ocan.at[p, ic].set(0.0)

    return mlcanopy_inst._replace(
        lwp_leaf=lwp_leaf,
        h2ocan_profile=h2ocan,
    )


# ---------------------------------------------------------------------------
# Stubs: InitHistory and Restart
# ---------------------------------------------------------------------------


def init_history(mlcanopy_inst: mlcanopy_type, begp: int, endp: int) -> None:
    """
    Register history file variables.

    Mirrors Fortran subroutine ``InitHistory`` (lines 480-494).

    Not translated: CLM history-file infrastructure (``hist_addfld1d``,
    ``hist_addfld2d``) has no equivalent in the JAX code base.  Callers
    that need diagnostic output should read directly from
    :attr:`mlcanopy_type.gppveg_canopy` and
    :attr:`mlcanopy_type.lwp_mean_profile`.
    """
    pass


def restart(
    mlcanopy_inst: mlcanopy_type,
    begp: int,
    endp: int,
    ncid: object,
    flag: str,
) -> mlcanopy_type:
    """
    Read or write restart variables.

    Mirrors Fortran subroutine ``Restart`` (lines 522-545).

    Not translated: CLM restart infrastructure (``restartvar``,
    ``ncd_io``) has no equivalent in the JAX code base.  Callers
    should serialise the NamedTuple arrays directly using standard
    NetCDF or HDF5 I/O.

    Args:
        mlcanopy_inst: Canopy container.
        begp: First patch index.
        endp: Last patch index.
        ncid: Open NetCDF file descriptor (unused stub).
        flag: ``'read'`` or ``'write'`` (unused stub).

    Returns:
        Unchanged ``mlcanopy_inst``.
    """
    return mlcanopy_inst
