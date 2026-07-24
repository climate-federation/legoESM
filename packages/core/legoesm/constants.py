"""Physical constants for legoESM.

All constants are in SI units unless otherwise noted.
"""

import jax.numpy as jnp

# ==============================================================================
# Fundamental Constants
# ==============================================================================
g = 9.80616                     # Gravitational acceleration [m/s^2]
Omega = 7.292e-5                # Earth rotation rate [rad/s]
R_earth = 6.371229e6            # Earth mean radius [m]

# ==============================================================================
# Dry Air Thermodynamics
# ==============================================================================
R_d = 287.05                    # Gas constant for dry air [J/(kg*K)]
c_pd = 1004.64                  # Specific heat at constant pressure [J/(kg*K)]
# Enforce the thermodynamic identity ``R_d = c_pd - c_vd`` exactly so
# the compressible Euler EOS (``pressure_from_eos``: p_0 · (R_d·ρ·θ/p_0)^(c_p/c_v))
# is consistent.  An earlier hardcoded value of 717.56 violated the
# identity by 0.03 J/(kg·K), introducing a ~0.01 % bias in p that
# compounded in tendencies.  Audit cycle iter-39 finding MEDIUM #6.
c_vd = c_pd - R_d               # Specific heat at constant volume [J/(kg*K)] = 717.59
kappa = R_d / c_pd              # Poisson constant R_d/c_pd (~0.2857)
p_ref = 1.0e5                   # Reference pressure [Pa] (1000 hPa)
p_atm_std = 101325.0            # Standard atmosphere [Pa] (1013.25 hPa)

# ==============================================================================
# Water Thermodynamics
# ==============================================================================
R_v = 461.51                    # Gas constant for water vapor [J/(kg*K)]
c_pv = 1846.0                   # Specific heat of water vapor [J/(kg*K)]
c_pw = 4218.0                   # Specific heat of liquid water [J/(kg*K)]
c_pi = 2106.0                   # Specific heat of ice [J/(kg*K)]
L_v = 2.501e6                   # Latent heat of vaporization at 0C [J/kg]
# SST slope of L_v in the NEMO/AeroBulk air-sea convention (sbc_phy
# L_vap: L = (2.501 - 0.00237 (T - T_freeze)) 1e6) — equals L_v at 0 degC
# by construction.  Used by thermo.latent_heat_vaporization_sst (#762).
L_v_sst_slope = 2.370e3         # [J/(kg*K)] dL_v/dT, NEMO sbc_phy / Fairall
L_s = 2.834e6                   # Latent heat of sublimation at 0C [J/kg]
L_f = 3.337e5                   # Latent heat of fusion at 0C [J/kg]
rho_water = 1000.0              # Density of liquid water [kg/m^3]
rho_ice = 917.0                 # Density of ice [kg/m^3]
rho_snow = 330.0                # Density of dry snow on sea ice [kg/m^3] (CICE default)
rho_air = 1.225                 # Reference dry-air density at sea level [kg/m^3]
rho_ocean = 1025.0              # Reference seawater density [kg/m^3] (= ocean.eos.rho_0)
rho_ocean_nemo = 1026.0         # NEMO rau0 [kg/m^3] (phycst.F90; GYRE/DINO/ORCA all use 1026)
rho_soil_particle = 2700.0      # Mineral soil particle (quartz) density [kg/m^3] (de Vries 1963)
c_sw = 3994.0                   # Specific heat of seawater [J/(kg*K)] (Gill 1982)
c_snow = 2090.0                 # Specific heat of snow [J/(kg*K)] (≈ c_pi, CICE default)
k_ice_default = 2.04            # Thermal conductivity of pure ice [W/(m*K)]
k_snow = 0.31                   # Thermal conductivity of dry snow [W/(m*K)] (CICE default)
T_freeze = 273.15               # Freezing point of water [K]
T_min_atmosphere = 200.0        # Lower-bound floor for atmospheric temperature [K]
                                # Used as safety floor in hypsometric psl extrapolation
                                # to avoid division by near-zero T in clear-sky columns.
                                # Matches Held-Suarez T_MIN and DCMIP lower bounds.
T_freeze_ocean = 271.35         # Freezing point of seawater [K] (~-1.8 C)
T_hom_freeze = 233.15           # Homogeneous freezing threshold [K] (~-40 C): all
                                # condensate is ice below; standard mixed-phase
                                # partition ramp spans [T_hom_freeze, T_freeze]
                                # (Pruppacher & Klett 1997; Morrison/IFS-style
                                # linear liquid fraction).
S_ice_bulk_default = 4.0        # Default bulk ice salinity [g/kg or PSU] (CICE-style)
S_ocean_ref = 34.7              # Reference ocean salinity [g/kg or PSU] (~WOA mean)
mu_ice_freeze = 0.054           # Liquidus slope / freezing-point depression [degC/PSU] (Bitz-Lipscomb 1999, CICE)
beta_ice_cond = 0.13            # Brine thermal-conductivity coefficient [W/(m*PSU)] (Untersteiner 1964; k = k0 + beta*S/T)
T_deep_ocean_ref_C = 1.5        # Global mean deep-ocean potential T [degC]
                                # (WOCE / WOA18 abyssal climatology — used as
                                # fallback fill when an interpolated profile
                                # has no valid data, e.g. below bathymetry)
S_deep_ocean_ref_psu = 34.7     # Global mean deep-ocean practical salinity
                                # [PSU] (WOCE / WOA18 abyssal climatology)
# Freshwater EOS local-parabolic fit (Kell 1975 / Jones-Harris)
T_freshwater_max_density = 277.133  # Max-density temperature [K] (~3.983 C)
rho_freshwater_curvature = 8.0e-6   # ρ-anomaly curvature [K^-2] from d²ρ/dT² at T_max

# Molar masses (g/mol) — used for CO2 ↔ mixing-ratio conversions, etc.
# Dry-air mean molar mass (CODATA / Mohr et al. 2024 = 28.96546 g/mol).
# The kg/mol forms ``M_dry`` and ``M_h2o`` (below) are derived from these
# via ``* 1e-3`` so future drift between g/mol and kg/mol forms is impossible.
M_air = 28.96546        # [g/mol] dry air (CODATA)
M_C   = 12.011          # [g/mol] atomic mass of carbon (IUPAC 2021)
M_CO2 = 44.01           # [g/mol] CO2
M_H2O = 18.01528        # [g/mol] water

# ==============================================================================
# Moisture Parameters
# ==============================================================================
epsilon = R_d / R_v              # Molecular weight ratio (~0.622)

# ==============================================================================
# Cloud-microphysics material properties
# (diffusional droplet growth: Super-Droplet Method, bin/bulk condensation)
# ==============================================================================
D_vapor = 2.21e-5               # Water-vapor diffusivity in air [m^2/s]
                                # (Pruppacher & Klett 1997; CONST value)
k_air = 2.40e-2                 # Thermal conductivity of air [W/(m*K)]
                                # (Pruppacher & Klett 1997)
sigma_water = 0.0728            # Surface tension of the water-air interface [N/m]
                                # at ~293 K (Pruppacher & Klett 1997) — Kelvin
                                # curvature term in Köhler droplet growth

# ==============================================================================
# Turbulence
# ==============================================================================
kappa_von_karman = 0.4          # Von Kármán constant for the log-law (Pope 2000)
nu_ocean_molecular = 1.4e-6     # Molecular kinematic viscosity of seawater
                                # [m^2/s] (~10 degC; NEMO phycst/zdfiwm ``rnu``)
kappa_T_ocean_molecular = 1.4e-7  # Molecular thermal diffusivity of seawater
                                  # [m^2/s] (= nu_ocean_molecular / Pr, Pr~10;
                                  # NEMO zdfiwm lower bound on wave-driven Kz)


# ==============================================================================
# Radiation
# ==============================================================================
sigma_sb = 5.670374419e-8       # Stefan-Boltzmann constant [W/(m^2*K^4)]
S_0 = 1361.0                    # Total solar irradiance [W/m^2]

# Present-day (≈ year 2000) Earth orbital elements for the realistic
# (AMIP-II / CMIP) insolation.  Berger (1978) convention as used by CESM
# ``shr_orb_mod`` / climlab: the longitude of perihelion is measured from
# the moving vernal equinox.  Used only when orbital insolation is enabled;
# the idealized default keeps a circular orbit (eccentricity = 0).
orbital_eccentricity = 0.016704         # [-] orbital eccentricity (year ~2000)
orbital_obliquity_deg = 23.439          # [deg] obliquity of the ecliptic
orbital_long_perihelion_deg = 282.895   # [deg] longitude of perihelion from VE

# Broadband longwave emissivities — the source of truth for the CENTRALIZED
# surface-tile emissivity defaults: ``driver/config.py`` (ExperimentConfig),
# ``driver/physics_pipeline.py``, and ``land/config.py`` reference these
# (enforced by test_constants_consistency.
# test_surface_emissivity_defaults_reference_constants).  Scheme-specific
# radiation keeps its own convention (gray uses a black surface eps=1.0);
# ``ice/config.py`` independently shares the 0.97 fresh-ice value.  Sea-water
# and fresh ice/snow are near-blackbody in the thermal-IR window; generic
# land ~0.95.
emissivity_ocean = 0.97         # [-] open ocean / lake water
emissivity_ice = 0.97           # [-] fresh sea ice / fresh snow
emissivity_land = 0.95          # [-] generic land surface
# NEMO/aerobulk sea-water thermal-IR emissivity (sbc_phy.F90 ``emiss_w``),
# used in the KIRCHHOFF net-longwave form ``Q_lw = eps*(LW_down - sigma*T^4)``
# (same eps as IR absorptivity AND emissivity).  The OMIP/CORE-II faithful
# ocean forcing path uses this; ``emissivity_ocean`` (0.97, emission-only
# convention) predates it and keeps its existing consumers.
emissivity_seawater_lw = 0.98   # [-] NEMO sbc_phy emiss_w

# Seawater specific heat used by NEMO (TEOS-10, eosbn2.F90 ``rcp``).  The
# OMIP-faithful surface-flux path uses it for the precipitation/evaporation
# heat-content terms of the non-solar flux so they match NEMO bit-for-bit.
c_p_seawater = 3991.86795711963  # [J/(kg*K)] NEMO TEOS-10 rcp

# Seawater specific heat used by the Jenkins (1991) / ISOMIP+ ice-shelf
# basal-melt intercomparison (Asay-Davis et al. 2016, GMD 9, 2471-2497).
# Deliberately the ISOMIP+ reference value (distinct from Gill-1982 ``c_sw``
# and NEMO's ``c_p_seawater``) so the linearised Jenkins melt rate reproduces
# the ISOMIP+ intercomparison numbers bit-for-bit.
c_p_seawater_isomip = 3974.0     # [J/(kg*K)] Jenkins 1991 / ISOMIP+

# NEMO/aerobulk moist-air heat-capacity pair (sbc_phy.F90 rCp_dry/rCp_vap),
# used by the NCAR bulk algorithm's sensible-heat flux
# ``cp_air(q) = rCp_dry + rCp_vap*q``.  Deliberately separate from the
# atmosphere's ``c_pd`` (1004.64): these are NEMO-parity coefficients of the
# CORE-II/OMIP faithful flux path, not legoESM's atmospheric thermodynamics.
c_p_dry_air_nemo = 1005.0       # [J/(kg*K)] NEMO sbc_phy rCp_dry
c_p_vapor_nemo = 1860.0         # [J/(kg*K)] NEMO sbc_phy rCp_vap

# Remaining NEMO-parity constants of the CORE-II/OMIP faithful flux path
# (values verbatim from NEMO 5.0.1 phycst.F90 / sbc_phy.F90).  They differ
# from legoESM's globals in the 4th-5th digit; the faithful path uses these
# so flux-level golden tests against the NEMO formulas close exactly.
g_nemo = 9.80665                # [m/s^2]  NEMO phycst grav
R_v_nemo = 461.495              # [J/(kg*K)] NEMO sbc_phy R_vap (ours: 461.51)
L_fus_nemo = 0.3333601e6        # [J/kg]   NEMO phycst rLfus (ours L_f: 3.337e5)
L_fus_isf_nemo = 0.334e6        # [J/kg]   NEMO isf_oce rLfusisf (ISF melt latent
                                #          heat — deliberately NOT rLfus)
c_p_ice_nemo = 2096.7           # [J/(kg*K)] NEMO phycst rcpi (ours c_pi: 2106)
# Universal molar constants used by NEMO's barometric 10-m pressure
# (sbc_phy pres_temp): molar gas constant + dry-air/water molar masses.
R_gas_molar = 8.314510          # [J/(mol*K)] universal molar gas constant
M_dry_air = 28.9647e-3          # [kg/mol] dry-air molar mass
M_water = 18.0153e-3            # [kg/mol] water molar mass

# Broadband sea-ice / snow surface albedos used as Delta-Eddington
# fallback or for the constant-albedo configuration.  Values follow
# CICE6 conventions (Briegleb & Light 2007).  Spectral splits:
#   - VIS = 0.2–0.7 μm (visible band)
#   - NIR = 0.7–5.0 μm (near-IR band, decays faster with melt / wetness)
alpha_snow_cold_vis = 0.98      # Dry cold snow, visible band
alpha_snow_cold_nir = 0.70      # Dry cold snow, near-IR band
alpha_snow_melt_vis = 0.80      # Melting snow, visible band
alpha_snow_melt_nir = 0.55      # Melting snow, near-IR band
alpha_ice_cold_vis = 0.78       # Bare cold sea ice, visible band
alpha_ice_cold_nir = 0.36       # Bare cold sea ice, near-IR band
alpha_ice_melt_vis = 0.68       # Melting bare ice, visible band
alpha_ice_melt_nir = 0.30       # Melting bare ice, near-IR band
alpha_pond_max_vis = 0.27       # Deep melt pond, visible band (Briegleb-Light)
alpha_pond_max_nir = 0.07       # Deep melt pond, near-IR band
i0_vis = 0.70                   # Fraction of incident VIS that penetrates bare ice
i0_nir = 0.0                    # NIR has negligible penetration

# Broadband (VIS+NIR integrated) surface shortwave albedos for the coarse OMIP
# ocean-only SW-reduction surrogate (no spectral split): the effective albedo at
# a cell is alpha_ocean_broadband*(1-siconc) + alpha_ice_broadband_cold*siconc,
# with siconc the prescribed sea-ice concentration.  The open-ocean value is the
# broadband reflectance that was previously MISSING (the ocean absorbed 100% of
# downwelling SW); the sea-ice value is the snow-free-to-snow broadband mean
# (CICE6 / Briegleb-Light, ~mean of the cold VIS/NIR ice+snow albedos above).
alpha_ocean_broadband = 0.06        # [-] open-ocean broadband albedo
alpha_ice_broadband_cold = 0.65     # [-] cold (snow-covered) sea-ice broadband
alpha_ice_broadband_warm = 0.45     # [-] melting sea-ice broadband (phase-2 blend)

# ==============================================================================
# Molecular Weights (kg/mol)
# ==============================================================================
# These are needed by forcing loaders to convert between mass-mixing-ratio
# (kg/kg) and volume-mixing-ratio (mol/mol) when an external file uses
# one convention and the radiation kernel expects the other.  CMIP6
# input4MIPs ozone uses ``vmro3`` in mol/mol (no conversion required), but
# older NCAR/E3SM pathways supply ``tro3`` in kg/kg, which has to be
# multiplied by ``M_dry / M_o3`` to land back in mol/mol before the
# radiation solver consumes it.  Values are derived from the g/mol forms
# above via ``* 1e-3`` so the two views never drift.
M_dry = M_air * 1e-3            # [kg/mol] = 0.02896546 (CODATA dry air)
M_o3 = 0.0479982                # [kg/mol] ozone (NIST CODATA / IUPAC)
M_h2o = M_H2O * 1e-3            # [kg/mol] = 0.01801528 (water)

# Avogadro's number (CODATA 2019 SI redefinition — exact value).
N_A = 6.02214076e23             # [1/mol] particles per mole

# Universal gas constant (CODATA 2018 / SI redefinition — exact).
# Used by biochemistry / photosynthesis (Arrhenius temperature factors)
# and any other code that needs R independent of a specific gas (R_d,
# R_v are gas-specific = R / M).
R_universal = 8.314462618       # [J/(mol·K)] ≈ N_A·k_B (value truncated at
                                # 10 digits; exact product is 8.31446261815324)
# Boltzmann constant — exact by SI definition (2019 redefinition). Defined as
# its own literal rather than R_universal/N_A because the R_universal literal
# above is truncated (deriving would be off by ~2e-11 relative).
k_B = 1.380649e-23              # [J/K] (exact, SI)

# Planck constant and speed of light — exact by the 2019 SI redefinition.
# Needed to convert a spectral radiance / irradiance [W/m^2] into a photon flux
# [mol photons / m^2 / s] through the photon energy E = h*c/lambda — e.g. the
# satellite-SIF radiance -> emitted-photon-flux conversion in
# scripts/data/build_sif_observations.py (and any PAR / quantum-yield code).
h_planck = 6.62607015e-34       # [J·s]  Planck constant        (exact, SI 2019)
c_light = 2.99792458e8          # [m/s]  speed of light in vacuum (exact, SI)

# ==============================================================================
# Mathematical Constants
# ==============================================================================
PI = jnp.pi
TWO_PI = 2.0 * jnp.pi
DEG_TO_RAD = jnp.pi / 180.0
RAD_TO_DEG = 180.0 / jnp.pi

# ==============================================================================
# Turbulence
# ==============================================================================
kappa_vk = 0.4                  # von Kármán constant
nu_air = 1.5e-5                 # Kinematic viscosity of air at 15°C [m^2/s]
prandtl_air = 0.71              # Molecular Prandtl number of air ν/α [-] (DNS
                                # heat diffusion κ = ν / Pr). Schmidt number of
                                # water vapour in air is ~0.6 (close); reuse this
                                # for the DNS scalar leg unless set separately.

# ==============================================================================
# Shallow Water Test Case Constants
# ==============================================================================
H_MEAN = 1.0e4                  # Mean fluid depth for shallow water [m]
