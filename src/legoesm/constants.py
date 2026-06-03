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
L_s = 2.834e6                   # Latent heat of sublimation at 0C [J/kg]
L_f = 3.337e5                   # Latent heat of fusion at 0C [J/kg]
rho_water = 1000.0              # Density of liquid water [kg/m^3]
rho_ice = 917.0                 # Density of ice [kg/m^3]
rho_snow = 330.0                # Density of dry snow on sea ice [kg/m^3] (CICE default)
rho_air = 1.225                 # Reference dry-air density at sea level [kg/m^3]
rho_ocean = 1025.0              # Reference seawater density [kg/m^3] (= ocean.eos.rho_0)
c_sw = 3994.0                   # Specific heat of seawater [J/(kg*K)] (Gill 1982)
c_snow = 2090.0                 # Specific heat of snow [J/(kg*K)] (≈ c_pi, CICE default)
k_ice_default = 2.04            # Thermal conductivity of pure ice [W/(m*K)]
k_snow = 0.31                   # Thermal conductivity of dry snow [W/(m*K)] (CICE default)
T_freeze = 273.15               # Freezing point of water [K]
T_freeze_ocean = 271.35         # Freezing point of seawater [K] (~-1.8 C)
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
M_CO2 = 44.01           # [g/mol] CO2
M_H2O = 18.01528        # [g/mol] water

# ==============================================================================
# Moisture Parameters
# ==============================================================================
epsilon = R_d / R_v              # Molecular weight ratio (~0.622)

# ==============================================================================
# Turbulence
# ==============================================================================
kappa_von_karman = 0.4          # Von Kármán constant for the log-law (Pope 2000)


# ==============================================================================
# Radiation
# ==============================================================================
sigma_sb = 5.670374419e-8       # Stefan-Boltzmann constant [W/(m^2*K^4)]
S_0 = 1361.0                    # Total solar irradiance [W/m^2]

# Broadband longwave emissivities (used as defaults when a tile config
# does not specify its own).  Sea-water and most ice surfaces are
# near-blackbody in the thermal-IR window; sand/dry-soil ~0.91.
# Note: ``driver/config.py`` and ``driver/physics_pipeline.py`` carry
# pre-existing ``emissivity_ice = 0.95`` defaults that predate the
# centralisation here.  0.97 is the fresh-sea-ice / fresh-snow
# value used by ``ice/config.py`` and the bare-ice albedo path; 0.95
# represents a melt-pond / weathered ice surface mix.  Reconciliation
# is tracked as a follow-up — see slopbuster review of
# Physical_Consistency PR.
emissivity_ocean = 0.97         # [-] open ocean / lake water
emissivity_ice = 0.97           # [-] fresh sea ice / fresh snow
emissivity_land = 0.95          # [-] generic land surface

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
R_universal = 8.314462618       # [J/(mol·K)] = N_A · k_B

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
