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


# --- Water viscosity + density (Huber et al. 2009; Fisher & Dial 1975) ---
# Transcribed verbatim from the rpmodel R source (geco-bern/rpmodel,
# R/subroutines.R: viscosity_h2o, density_h2o): Huber et al. (2009),
# J. Phys. Chem. Ref. Data 38, 101-125, with water density from the Tumlirz
# equation of state (Fisher & Dial 1975, Marine Physical Laboratory Tech.
# Rept., San Diego).  Used by the P model's least-cost xi (cost of water
# transport scales with viscosity).
_VISC_TK_AST = 647.096  # [K] Huber reference temperature
_VISC_RHO_AST = 322.0  # [kg/m3] Huber reference density
_VISC_MU_AST = 1e-6  # [Pa s] Huber reference viscosity
# Huber eq. 11 / Table 2: mu0 denominator, ascending powers of (1/tbar).
_VISC_MU0_COEFFS = (1.67752, 2.20462, 0.6366564, -0.241605)
# Huber Table 3: row j multiplies (rbar-1)^j, column i multiplies ctbar^i.
_VISC_H = (
    (0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0),
    (0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573),
    (-0.281378, -0.906851, -0.772479, -0.489837, -0.257040, 0.0),
    (0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0),
    (-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0),
    (0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264),
)
# Fisher & Dial (1975) Tumlirz-equation polynomials, ascending powers of T [degC].
_TUMLIRZ_LAMBDA = (1788.316, 21.55053, -0.4695911, 3.096363e-3, -7.341182e-6)  # [bar cm3/g]
_TUMLIRZ_PO = (5918.499, 58.05267, -1.1253317, 6.6123869e-3, -1.4661625e-5)  # [bar]
_TUMLIRZ_VINF = (  # [cm3/g]
    0.6980547, -7.435626e-4, 3.704258e-5, -6.315724e-7, 9.829576e-9,
    -1.197269e-10, 1.005461e-12, -5.437898e-15, 1.69946e-17, -2.295063e-20,
)


def _poly_ascending(coeffs: tuple[float, ...], x: jax.Array) -> jax.Array:
    """Horner evaluation of a polynomial given ascending-power coefficients."""
    y = jnp.zeros_like(x)
    for c in reversed(coeffs):
        y = y * x + c
    return y


def water_density(T_K: jax.Array, P_Pa: jax.Array) -> jax.Array:
    """Density of pure water [kg/m3] (Tumlirz equation, Fisher & Dial 1975)."""
    tc = T_K - constants.T_freeze
    lam = _poly_ascending(_TUMLIRZ_LAMBDA, tc)
    po = _poly_ascending(_TUMLIRZ_PO, tc)
    vinf = _poly_ascending(_TUMLIRZ_VINF, tc)
    pbar = 1e-5 * P_Pa  # [Pa] -> [bar]
    v = vinf + lam / (po + pbar)  # specific volume [cm3/g]
    return 1e3 / v


def water_viscosity(T_K: jax.Array, P_Pa: jax.Array) -> jax.Array:
    """Dynamic viscosity of pure water [Pa s] (Huber et al. 2009)."""
    tbar = T_K / _VISC_TK_AST
    rbar = water_density(T_K, P_Pa) / _VISC_RHO_AST
    mu0 = 1e2 * jnp.sqrt(tbar) / _poly_ascending(_VISC_MU0_COEFFS, 1.0 / tbar)
    ctbar = 1.0 / tbar - 1.0
    mu1 = jnp.zeros_like(tbar)
    for i in range(6):
        inner = jnp.zeros_like(tbar)
        for j in range(7):
            inner = inner + _VISC_H[j][i] * (rbar - 1.0) ** j
        mu1 = mu1 + ctbar**i * inner
    mu1 = jnp.exp(rbar * mu1)
    return mu0 * mu1 * _VISC_MU_AST


def water_viscosity_ratio(T_K: jax.Array, P_Pa: jax.Array) -> jax.Array:
    """eta* = eta(T, P) / eta(25 degC, standard atmosphere), the P model's
    relative cost of water transport (Stocker et al. 2020 GMD, eq. 9)."""
    return water_viscosity(T_K, P_Pa) / water_viscosity(
        jnp.asarray(T_REF_K), jnp.asarray(constants.p_atm_std)
    )


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
