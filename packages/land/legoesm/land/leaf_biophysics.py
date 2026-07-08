"""Shared leaf-biochemistry temperature responses and Rubisco kinetics.

Single source of the Arrhenius / peaked-Arrhenius temperature-response factors
and the Bernacchi (2001) Rubisco kinetic constants used by BOTH land
photosynthesis paths:

- ``legoesm.land.stomata`` — the big-leaf SimpleSEB / coupled A-gs solver
  (config-based Farquhar C3 + Ball-Berry / Medlyn / Jarvis stomata), and
- ``legoesm.land.canopy.photosynthesis`` — the two-leaf canonical FvCB C3 + C4.

Before this module the two paths each carried their own copy of the Arrhenius
math (with the gas constant hardcoded as ``8.314`` in the canopy copy) and their
own copy of the Bernacchi kinetics (in mismatched umol/mol vs mmol/mol units),
so a fix to one could silently drift from the other.  The factors here are
NORMALISED (== 1 at 25 degC): a caller multiplies by its own 25 degC reference
value (``param25``), which lets the same helper serve both the config-supplied
per-leaf-class capacities (carbon path) and the module-constant capacities
(canopy path).

All functions are pure JAX (differentiable, JIT-friendly).

References
----------
- Bernacchi, C. J. et al. (2001): Improved temperature response functions for
  models of Rubisco-limited photosynthesis. Plant, Cell & Environment, 24,
  253-259.
- Farquhar, von Caemmerer & Berry (1980): Planta 149, 78-90.
- Bonan (2019), "Climate Change and Terrestrial Ecosystem Modeling", ch. 11
  (eq. 11.34 Arrhenius, 11.36 peaked-Arrhenius).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants

# Universal gas constant, the single canonical value (CODATA 2018).  The canopy
# path previously hardcoded a truncated 8.314; both paths now share this.
_R_GAS = constants.R_universal  # [J/(mol·K)]
# Reference temperature for the temperature responses: 25 degC, expressed via
# constants.T_freeze so the Celsius->Kelvin offset is never a bare literal.
T_REF_K = constants.T_freeze + 25.0  # [K] (25 degC)

# --- Bernacchi (2001) Rubisco kinetics at 25 degC (one units-explicit set) ---
# Carried in umol/mol.  Consumers that form only ratios (O2/Ko) are unit-
# independent; consumers that add Kc to a mole-fraction Ci already work in
# umol/mol, so this single unit system serves both photosynthesis paths.
KC25_UMOL_MOL = 404.9          # Michaelis constant for CO2 [umol/mol]
KO25_UMOL_MOL = 278400.0       # Michaelis constant for O2 [umol/mol] (= 278.4 mmol/mol)
GAMMA_STAR25_UMOL_MOL = 42.75  # CO2 compensation point Gamma* at 25 degC [umol/mol]
O2_UMOL_MOL = 209000.0         # Atmospheric / intercellular O2 [umol/mol] (= 209.0 mmol/mol)

# Arrhenius activation energies for the Bernacchi kinetics [J/mol].
HA_KC = 79430.0
HA_KO = 36380.0
HA_GAMMA = 37830.0

# Ratio of H2O to CO2 molecular diffusivity in air (Fick's law).  Appears as the
# Medlyn USO prefactor and in every A -> Ci stomatal-diffusion back-calculation,
# so BOTH the big-leaf (land/stomata.py) and two-leaf canopy
# (canopy/energy_balance.py) paths read it from here rather than each carrying a
# copy.
DIFFUSIVITY_RATIO_H2O_CO2 = 1.6


def arrhenius_factor(T_K: jax.Array, Ha: float | jax.Array) -> jax.Array:
    """Normalised Arrhenius temperature response (Bonan eq. 11.34).

    Returns f(T) = exp(Ha·(T − T_ref) / (T_ref·R·T)), equal to 1 at 25 degC.
    Multiply by the 25 degC reference value to obtain the temperature-adjusted
    quantity.
    """
    return jnp.exp(Ha * (T_K - T_REF_K) / (T_REF_K * _R_GAS * T_K))


def peaked_arrhenius_factor(
    T_K: jax.Array, Ha: float | jax.Array, Hd: float | jax.Array, S: float | jax.Array
) -> jax.Array:
    """Normalised peaked-Arrhenius response for parameters that decline at high T.

    Returns f(T) = arrhenius_factor(T, Ha) · (1 + exp((S·T_ref − Hd)/(R·T_ref)))
    / (1 + exp((S·T − Hd)/(R·T))), equal to 1 at 25 degC (Bonan eq. 11.34·11.36).
    """
    f_T = arrhenius_factor(T_K, Ha)
    num = 1.0 + jnp.exp((S * T_REF_K - Hd) / (_R_GAS * T_REF_K))
    den = 1.0 + jnp.exp((S * T_K - Hd) / (_R_GAS * T_K))
    return f_T * num / den
