"""
JAX translation of MLclm_varcon Fortran module.

Physical constants and adjustable parameters for the multilayer canopy
(CLMml) model. Covers gas-phase transport, leaf photosynthesis,
stomatal conductance, leaf heat capacity, boundary layer conductance,
canopy interception, solar and longwave radiation, roughness sublayer
parameterization, and RSL psihat look-up table arrays.

Original Fortran module: MLclm_varcon
Fortran lines 1-110
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import spval  # noqa: F401

# ---------------------------------------------------------------------------
# Physical constants — Fortran lines 15-25
# ---------------------------------------------------------------------------

rgas: float = 8.31446  # Universal gas constant (J/K/mol)
mmdry: float = 28.97e-3  # Molecular mass of dry air (kg/mol)
mmh2o: float = 18.02e-3  # Molecular mass of water vapour (kg/mol)
cpd: float = 1005.0  # Specific heat of dry air at constant pressure (J/kg/K)
cpw: float = 1846.0  # Specific heat of water vapour at constant pressure (J/kg/K)
visc0: float = 13.3e-6  # Kinematic viscosity at 0 °C and 1013.25 hPa (m2/s)
dh0: float = 18.9e-6  # Molecular diffusivity, heat, at 0 °C and 1013.25 hPa (m2/s)
dv0: float = 21.8e-6  # Molecular diffusivity, H2O,  at 0 °C and 1013.25 hPa (m2/s)
dc0: float = 13.8e-6  # Molecular diffusivity, CO2,  at 0 °C and 1013.25 hPa (m2/s)
lapse_rate: float = 0.0098  # Temperature lapse rate (K/m)


# ---------------------------------------------------------------------------
# Leaf photosynthesis — Fortran lines 27-73
# ---------------------------------------------------------------------------

# Michaelis-Menten and CO2 compensation — Fortran lines 29-34
kc25: float = 404.9  # Michaelis-Menten constant for CO2 at 25 °C (umol/mol)
kcha: float = 79430.0  # Activation energy for kc (J/mol)
ko25: float = 278.4  # Michaelis-Menten constant for O2 at 25 °C (mmol/mol)
koha: float = 36380.0  # Activation energy for ko (J/mol)
cp25: float = 42.75  # CO2 compensation point at 25 °C (umol/mol)
cpha: float = 37830.0  # Activation energy for cp (J/mol)

# Vcmax temperature response — Fortran lines 36-41
vcmaxha_noacclim: float = 65330.0  # Activation energy for vcmax, no acclimation (J/mol)
vcmaxha_acclim: float = 72000.0  # Activation energy for vcmax, with acclimation (J/mol)
vcmaxhd_noacclim: float = 150000.0  # Deactivation energy for vcmax, no acclimation (J/mol)
vcmaxhd_acclim: float = 200000.0  # Deactivation energy for vcmax, with acclimation (J/mol)
vcmaxse_noacclim: float = 490.0  # Entropy term for vcmax, no acclimation (J/mol/K)
vcmaxse_acclim: float = spval  # Entropy term for vcmax, with acclimation (J/mol/K)

# Jmax temperature response — Fortran lines 43-48
jmaxha_noacclim: float = 43540.0  # Activation energy for jmax, no acclimation (J/mol)
jmaxha_acclim: float = 50000.0  # Activation energy for jmax, with acclimation (J/mol)
jmaxhd_noacclim: float = 150000.0  # Deactivation energy for jmax, no acclimation (J/mol)
jmaxhd_acclim: float = 200000.0  # Deactivation energy for jmax, with acclimation (J/mol)
jmaxse_noacclim: float = 490.0  # Entropy term for jmax, no acclimation (J/mol/K)
jmaxse_acclim: float = spval  # Entropy term for jmax, with acclimation (J/mol/K)

# Rd temperature response — Fortran lines 50-52
rdha: float = 46390.0  # Activation energy for Rd (J/mol)
rdhd: float = 150000.0  # Deactivation energy for Rd (J/mol)
rdse: float = 490.0  # Entropy term for Rd (J/mol/K)

# Jmax:Vcmax and Rd:Vcmax ratios — Fortran lines 54-59
jmax25_to_vcmax25_noacclim: float = 1.67  # Jmax/Vcmax at 25 °C, no acclimation (umol/umol)
jmax25_to_vcmax25_acclim: float = spval  # Jmax/Vcmax at 25 °C, with acclimation (umol/umol)
rd25_to_vcmax25_c3: float = 0.015  # Rd/Vcmax at 25 °C, C3 (umol/umol)
rd25_to_vcmax25_c4: float = 0.025  # Rd/Vcmax at 25 °C, C4 (umol/umol)
kp25_to_vcmax25_c4: float = 0.02  # Kp/Vcmax at 25 °C, C4 (mol/umol)

# Quantum yield and electron transport — Fortran lines 61-65
phi_psII: float = 0.70  # C3: quantum yield of PS II
# phi_psII: float = 0.85  # C3: alternative value (commented out in Fortran line 62)
theta_j: float = 0.90  # C3: curvature parameter for electron transport rate
qe_c4: float = 0.05  # C4: quantum yield (mol CO2 / mol photons)

# Co-limitation curvature parameters — Fortran lines 67-71
colim_c3a: float = 0.98  # C3 co-limitation (Ac, Aj)
colim_c3b: float = spval  # C3 co-limitation (Ap)
colim_c4a: float = 0.80  # C4 co-limitation (Ac, Aj)
colim_c4b: float = 0.95  # C4 co-limitation (Ap)


# ---------------------------------------------------------------------------
# Stomatal conductance — Fortran lines 74-77
# ---------------------------------------------------------------------------

dh2o_to_dco2: float = 1.6  # Diffusivity H2O / Diffusivity CO2
rh_min_BB: float = 0.2  # Minimum RH for Ball-Berry stomatal conductance (fraction)
vpd_min_MED: float = 100.0  # Minimum VPD for Medlyn stomatal conductance (Pa)


# ---------------------------------------------------------------------------
# Leaf heat capacity — Fortran lines 79-82
# ---------------------------------------------------------------------------

cpbio: float = 4188.0 / 3.0  # Specific heat of dry biomass (J/kg/K)
fcarbon: float = 0.5  # Fraction of dry biomass that is carbon (kg C / kg DM)
fwater: float = 0.7  # Fraction of fresh biomass that is water (kg H2O / kg FM)


# ---------------------------------------------------------------------------
# Leaf boundary layer conductance — Fortran lines 84-85
# ---------------------------------------------------------------------------

gb_factor: float = 1.5  # Empirical correction factor for Nusselt number


# ---------------------------------------------------------------------------
# Canopy interception — Fortran lines 87-91
# ---------------------------------------------------------------------------

dewmx: float = 0.1  # Maximum allowed interception (kg H2O / m2 leaf)
maximum_leaf_wetted_fraction: float = 0.05  # Maximum fraction of leaf that can be wet
interception_fraction: float = 1.0  # Fraction of intercepted precipitation
fwet_exponent: float = 0.67  # Exponent for wetted canopy fraction


# ---------------------------------------------------------------------------
# Solar radiation — Fortran lines 93-97
# ---------------------------------------------------------------------------

chil_min: float = -0.4  # Minimum value for xl leaf angle orientation parameter
chil_max: float = 0.6  # Maximum value for xl leaf angle orientation parameter
kb_max: float = 40.0  # Maximum direct beam extinction coefficient
J_to_umol: float = 4.6  # PAR conversion: W/m2 → umol/m2/s (umol/J)


# ---------------------------------------------------------------------------
# Longwave radiation — Fortran lines 99-100
# ---------------------------------------------------------------------------

emg: float = 0.96  # Ground (soil) emissivity


# ---------------------------------------------------------------------------
# Roughness sublayer parameterization — Fortran lines 102-117
# ---------------------------------------------------------------------------

cd: float = 0.25  # Drag coefficient for canopy elements (dimensionless)
beta_neutral_max: float = 0.35  # Maximum beta in neutral conditions
cr: float = 0.3  # Parameter to calculate beta_neutral
c2: float = 0.5  # Depth scale multiplier
Pr0: float = 0.5  # Neutral value for Pr (Sc)
Pr1: float = 0.3  # Magnitude of variation of Pr (Sc) with stability
Pr2: float = 2.0  # Scale of variation of Pr (Sc) with stability
z0mg: float = 0.01  # Roughness length of ground (m)

# a1–a3 parameters for Harman (2012, eq. 13) — Fortran lines 119-120
# Fortran: real(r8) :: aH12(3); data aH12 / 0.89, -0.07, 2.19 /
# Array is 1-based in Fortran; preserved as a 1-D JAX array of length 3.
aH12: np.ndarray = np.array([0.89, -0.07, 2.19], dtype=np.float64)


# ---------------------------------------------------------------------------
# Limits on various variables — Fortran lines 122-128
# ---------------------------------------------------------------------------

wind_forc_min: float = 0.5  # Minimum wind speed at forcing height (m/s)
LcL_min: float = -2.0  # Minimum value for Lc/obu
LcL_max: float = 1.0  # Maximum value for Lc/obu
gbh_min: float = 0.2  # Minimum leaf boundary layer conductance for heat (mol/m2/s)
ra_max: float = 500.0  # Maximum aerodynamic resistance for a canopy layer (s/m)
eta_max: float = 20.0  # Maximum value for "eta" parameter


# ---------------------------------------------------------------------------
# RSL psihat look-up table dimensions and arrays — Fortran lines 130-140
# Populated by LookupPsihatINI; declared here as empty placeholders.
# ---------------------------------------------------------------------------

nZ: int = 276  # Number of z/h grid points in psihat tables
nL: int = 41  # Number of (h-d)/L grid points in psihat tables

# Momentum psihat table — Fortran: zdtgridM(nZ,1), dtLgridM(1,nL), psigridM(nZ,nL)
# Using numpy arrays so element-wise access in _LookupPsihat does not trigger XLA syncs.
zdtgridM: np.ndarray = np.zeros((nZ, 1), dtype=np.float64)  # (z-h)/(h-d) grid, momentum
dtLgridM: np.ndarray = np.zeros((1, nL), dtype=np.float64)  # (h-d)/L    grid, momentum
psigridM: np.ndarray = np.zeros((nZ, nL), dtype=np.float64)  # psihat values,   momentum

# Heat psihat table — Fortran: zdtgridH(nZ,1), dtLgridH(1,nL), psigridH(nZ,nL)
zdtgridH: np.ndarray = np.zeros((nZ, 1), dtype=np.float64)  # (z-h)/(h-d) grid, heat
dtLgridH: np.ndarray = np.zeros((1, nL), dtype=np.float64)  # (h-d)/L    grid, heat
psigridH: np.ndarray = np.zeros((nZ, nL), dtype=np.float64)  # psihat values,   heat
