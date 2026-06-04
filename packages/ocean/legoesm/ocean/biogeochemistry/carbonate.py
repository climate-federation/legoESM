"""Carbonate chemistry: CO2 solubility, equilibria, pCO2, and pH.

Implements the simplified solver from Follows et al. (2006) for
computing surface ocean pCO2 from DIC and total alkalinity,
using the temperature- and salinity-dependent equilibrium constants
of Lueker et al. (2000) on the total pH scale.

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Follows, M. J., et al. (2006). Solving the carbonate system in
  ocean biogeochemistry models. Ocean Modelling, 12, 290-301.
- Lueker, T. J., et al. (2000). Ocean pCO2 calculated from dissolved
  inorganic carbon, alkalinity, and equations for K1 and K2.
  Marine Chemistry, 70, 105-119.
- Weiss, R. F. (1974). Carbon dioxide in water and seawater: the
  solubility of a non-ideal gas. Marine Chemistry, 2, 203-215.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


# ============================================================================
# Solubility and equilibrium constants
# ============================================================================


def co2_solubility(T_degC: jnp.ndarray, S_psu: jnp.ndarray) -> jnp.ndarray:
    """Henry's law solubility of CO2 in seawater (K0).

    Weiss (1974), Table I, for CO2 in mol/(kg*atm).

    Parameters
    ----------
    T_degC : array
        Sea surface temperature [degC].
    S_psu : array
        Sea surface salinity [PSU].

    Returns
    -------
    K0 : array
        CO2 solubility [mol/(kg*atm)].
    """
    T_K = T_degC + constants.T_freeze
    T_100 = T_K / 100.0

    ln_K0 = (
        -60.2409 + 93.4517 / T_100 + 23.3585 * jnp.log(T_100)
        + S_psu * (0.023517 - 0.023656 * T_100 + 0.0047036 * T_100 ** 2)
    )
    return jnp.exp(ln_K0)


def carbonate_equilibria(
    T_degC: jnp.ndarray,
    S_psu: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """First and second dissociation constants of carbonic acid.

    Lueker et al. (2000), total pH scale, valid for
    T = 2-35 degC, S = 19-43 PSU.

    Parameters
    ----------
    T_degC : array
        Temperature [degC].
    S_psu : array
        Salinity [PSU].

    Returns
    -------
    K1 : array
        First dissociation constant [mol/kg].
    K2 : array
        Second dissociation constant [mol/kg].
    """
    T_K = T_degC + constants.T_freeze

    # pK1 = -log10(K1)
    pK1 = (
        3633.86 / T_K - 61.2172 + 9.6777 * jnp.log(T_K)
        - 0.011555 * S_psu + 0.0001152 * S_psu ** 2
    )
    # pK2 = -log10(K2)
    pK2 = (
        471.78 / T_K + 25.929 - 3.16967 * jnp.log(T_K)
        - 0.01781 * S_psu + 0.0001122 * S_psu ** 2
    )

    K1 = 10.0 ** (-pK1)
    K2 = 10.0 ** (-pK2)

    return K1, K2


def borate_equilibrium(T_degC: jnp.ndarray, S_psu: jnp.ndarray) -> jnp.ndarray:
    """Borate dissociation constant K_B (Dickson 1990, total pH scale).

    Parameters
    ----------
    T_degC : array
        Temperature [degC].
    S_psu : array
        Salinity [PSU].

    Returns
    -------
    K_B : array
        Borate dissociation constant [mol/kg].
    """
    T_K = T_degC + constants.T_freeze
    sqrt_S = jnp.sqrt(jnp.clip(S_psu, 0.0, None))

    ln_KB = (
        (-8966.90 - 2890.53 * sqrt_S - 77.942 * S_psu
         + 1.728 * S_psu * sqrt_S - 0.0996 * S_psu ** 2) / T_K
        + 148.0248 + 137.1942 * sqrt_S + 1.62142 * S_psu
        + (-24.4344 - 25.085 * sqrt_S - 0.2474 * S_psu) * jnp.log(T_K)
        + 0.053105 * sqrt_S * T_K
    )
    return jnp.exp(ln_KB)


def total_borate(S_psu: jnp.ndarray) -> jnp.ndarray:
    """Total dissolved boron concentration [mol/kg].

    Uppström (1974): B_T = 0.000416 * S / 35.
    """
    return 0.000416 * S_psu / 35.0


# ============================================================================
# pCO2 and pH solver (Follows et al. 2006)
# ============================================================================


def solve_carbonate_system(
    DIC: jnp.ndarray,
    ALK: jnp.ndarray,
    T_degC: jnp.ndarray,
    S_psu: jnp.ndarray,
    rho_sw: float = constants.rho_ocean,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the carbonate system for pCO2 and pH.

    Uses the simplified analytic solver of Follows et al. (2006),
    which avoids iterative Newton-Raphson by making a single-step
    cubic approximation. Accurate to ~1 uatm for typical ocean
    conditions.

    Parameters
    ----------
    DIC : array
        Dissolved inorganic carbon [mol C/m^3].
    ALK : array
        Total alkalinity [mol eq/m^3].
    T_degC : array
        Temperature [degC].
    S_psu : array
        Salinity [PSU].
    rho_sw : float
        Seawater reference density [kg/m^3].

    Returns
    -------
    pCO2 : array
        Partial pressure of CO2 [uatm].
    pH : array
        pH on total scale.
    """
    # Convert from mol/m^3 to mol/kg
    DIC_kg = DIC / rho_sw
    ALK_kg = ALK / rho_sw

    K1, K2 = carbonate_equilibria(T_degC, S_psu)
    K0 = co2_solubility(T_degC, S_psu)
    K_B = borate_equilibrium(T_degC, S_psu)
    B_T = total_borate(S_psu)

    # Carbonate alkalinity: A_C = ALK - A_B
    # A_B = B_T * K_B / (K_B + [H+])
    # Use initial guess for [H+] from pH~8.1
    H_init = 10.0 ** (-8.1) * jnp.ones_like(DIC_kg)

    # Follows et al. (2006) single-step approximation:
    # Borate alkalinity using initial [H+]
    A_B = B_T * K_B / (K_B + H_init)

    # Carbonate alkalinity
    A_C = ALK_kg - A_B
    # Clamp to prevent negative/zero
    A_C = jnp.clip(A_C, 1.0e-10, None)

    # From A_C = DIC * (K1*H + 2*K1*K2) / (H^2 + K1*H + K1*K2)
    # Rearrange: gamma = DIC / A_C
    gamma = DIC_kg / A_C

    # Quadratic in [H+]: H^2 + K1*(1-gamma)*H + K1*K2*(1-2*gamma) = 0
    a_coeff = 1.0
    b_coeff = K1 * (1.0 - gamma)
    c_coeff = K1 * K2 * (1.0 - 2.0 * gamma)

    discriminant = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
    discriminant = jnp.clip(discriminant, 0.0, None)

    H = (-b_coeff + jnp.sqrt(discriminant)) / (2.0 * a_coeff)
    H = jnp.clip(H, 1.0e-12, 1.0e-4)

    # pCO2 from [CO2*] = DIC * H^2 / (H^2 + K1*H + K1*K2)
    denom = H ** 2 + K1 * H + K1 * K2
    CO2_aq = DIC_kg * H ** 2 / jnp.clip(denom, 1.0e-30, None)

    # pCO2 = [CO2*] / K0 in atm, convert to uatm
    pCO2 = CO2_aq / jnp.clip(K0, 1.0e-10, None) * 1.0e6

    # pH = -log10([H+])
    pH = -jnp.log10(jnp.clip(H, 1.0e-14, None))

    return pCO2, pH
