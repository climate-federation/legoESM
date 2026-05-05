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
c_vd = 717.56                   # Specific heat at constant volume [J/(kg*K)]
kappa = R_d / c_pd              # Poisson constant R_d/c_pd (~0.2857)
p_ref = 1.0e5                   # Reference pressure [Pa] (1000 hPa)

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
T_freeze = 273.15               # Freezing point of water [K]
T_freeze_ocean = 271.35         # Freezing point of seawater [K] (~-1.8 C)

# ==============================================================================
# Moisture Parameters
# ==============================================================================
epsilon = R_d / R_v              # Molecular weight ratio (~0.622)

# ==============================================================================
# Radiation
# ==============================================================================
sigma_sb = 5.670374419e-8       # Stefan-Boltzmann constant [W/(m^2*K^4)]
S_0 = 1361.0                    # Total solar irradiance [W/m^2]

# ==============================================================================
# Molecular Weights
# ==============================================================================
# These are needed by forcing loaders to convert between mass-mixing-ratio
# (kg/kg) and volume-mixing-ratio (mol/mol) when an external file uses
# one convention and the radiation kernel expects the other.  CMIP6
# input4MIPs ozone uses ``vmro3`` in mol/mol (no conversion required), but
# older NCAR/E3SM pathways supply ``tro3`` in kg/kg, which has to be
# multiplied by ``M_dry / M_o3`` to land back in mol/mol before the
# radiation solver consumes it.  Values from NIST CODATA / IUPAC.
M_dry = 0.0289644               # Molecular weight of dry air [kg/mol]
M_o3 = 0.0479982                # Molecular weight of ozone [kg/mol]
M_h2o = 0.018015                # Molecular weight of water [kg/mol]

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

# ==============================================================================
# Shallow Water Test Case Constants
# ==============================================================================
H_MEAN = 1.0e4                  # Mean fluid depth for shallow water [m]
