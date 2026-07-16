"""Carbonate chemistry: CO2 solubility, equilibria, pCO2, and pH.

Implements the simplified solver from Follows et al. (2006) for
computing surface ocean pCO2 from DIC and total alkalinity,
using the temperature- and salinity-dependent equilibrium constants
of Lueker et al. (2000) on the total pH scale.

All functions are JAX-compatible (differentiable, JIT-friendly).

Faithfulness
------------
``tests/ocean/unit/test_bgc_carbonate_faithful.py`` pins every closed form to
round-off (rel 1e-12) against an INDEPENDENT scalar reimplementation with the
published coefficients, and cross-checks the solver against a fully-converged
same-chemistry solve.

FAITHFUL to the cited references (exact published coefficients + forms):
- ``co2_solubility`` K0 — Weiss (1974) Table I, mol/(kg*atm).
- ``carbonate_equilibria`` K1, K2 — Lueker et al. (2000), total pH scale.
- ``borate_equilibrium`` K_B — Dickson (1990), total pH scale.
- ``total_borate`` B_T — Uppström (1974).
- ``solve_carbonate_system`` — the Follows et al. (2006) borate-corrected
  quadratic-in-[H+] algebra, iterated ``_N_BORATE_ITER`` times toward the
  self-consistent borate correction (a fixed count, not a tolerance loop).

DEPARTURES (documented, canaried in the test):
- The solver's alkalinity balance is CARBONATE + BORATE only.  It neglects the
  water (OH-/H+), phosphate, silicate and (in applicable waters) ammonia,
  sulfide and organic-alkalinity contributions, and treats the Weiss (1974) K0
  partial-pressure vs fugacity distinction as pCO2.  It is therefore NOT a
  substitute for a full solver (CO2SYS): the offset depends on the neglected
  species and is not quantified by these tests.  The ``_N_BORATE_ITER``
  iteration removes only the borate-guess error, not these omissions.
- AD-safety guards clip A_C >= 1e-10, the discriminant >= 0, [H+] to
  [1e-12, 1e-4], and denominators to >= 1e-30.  Across the tested ocean grid
  NONE of these bind (the test asserts the smooth interior path); off the
  physical manifold they bias the result but keep gradients finite.
- The quadratic's larger (``+``) root is the unique positive [H+] only in the
  carbonate regime gamma = DIC/A_C > 0.5 (DIC exceeding carbonate alkalinity),
  which holds across the tested grid (canaried); it is not an unconditional
  positive-root rule.
- ``_N_BORATE_ITER`` is a FIXED iteration count (not a tolerance loop), so the
  solve is JIT-static and AD-trivial but converged only to the tolerance that
  count buys (< 1 uatm across the tested grid).

References
----------
- Follows, M. J., et al. (2006). Solving the carbonate system in
  ocean biogeochemistry models. Ocean Modelling, 12, 290-301.
- Lueker, T. J., et al. (2000). Ocean pCO2 calculated from dissolved
  inorganic carbon, alkalinity, and equations for K1 and K2.
  Marine Chemistry, 70, 105-119.
- Weiss, R. F. (1974). Carbon dioxide in water and seawater: the
  solubility of a non-ideal gas. Marine Chemistry, 2, 203-215.
- Dickson, A. G. (1990). Thermodynamics of the dissociation of boric acid
  in synthetic seawater. Deep-Sea Res., 37, 755-766.
- Uppström, L. R. (1974). The boron/chlorinity ratio of deep-sea water from
  the Pacific Ocean. Deep-Sea Res., 21, 161-162.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants

# Number of Follows et al. (2006) borate-alkalinity iterations.  A SINGLE step
# at the fixed initial guess H_init = 10^-8.1 (the original un-iterated form)
# biases pCO2 by hundreds of uatm when the true pH departs 8.1 (warm low-buffer
# or cold high-pH surface water).  Iterating the borate correction converges the
# solve: over a 135-point ocean grid (T in [2,34] degC, S in [31,37], DIC in
# [1900,2100] umol/kg, ALK = DIC + [200,350] umol/kg) the worst
# |pCO2 - fully_converged| is ~531 uatm at 1 iteration (warm low-buffer corner),
# 5.4 at 5, and 0.25 at 8.
# A loop-iteration COUNT is a module constant, never a config/trainable field.
_N_BORATE_ITER = 8


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


def _borate_quadratic_step(H, DIC_kg, ALK_kg, K1, K2, K_B, B_T):
    """One Follows et al. (2006) borate-corrected pass of the pCO2/pH solver.

    Given the current [H+], update the borate alkalinity A_B = B_T*K_B/(K_B+H),
    form the carbonate-alkalinity residual A_C = ALK - A_B, and solve the
    quadratic  H^2 + K1*(1-gamma)*H + K1*K2*(1-2*gamma) = 0  (gamma = DIC/A_C)
    for the new [H+], taking the larger (positive) root.

    Returns ``(H_new, A_C_raw, disc_raw, H_raw, gamma)``: ``H_new`` is the
    post-clip [H+] the solver carries to the next pass; the ``*_raw`` values are
    the PRE-CLIP carbonate-alkalinity residual, discriminant and root, and
    ``gamma`` the DIC/A_C ratio — exposed so the faithfulness test can verify the
    AD-safety guards (A_C >= 1e-10, discriminant >= 0, [H+] in [1e-12, 1e-4]) are
    inactive on the physical manifold and the root-selection regime gamma > 0.5
    holds, using the SAME code the solver runs (not a reimplementation).  All
    ``*_raw``/``gamma`` are intermediates of ``H_new``, so exposing them is free.
    """
    A_B = B_T * K_B / (K_B + H)
    # Carbonate alkalinity residual (neglects water/phosphate/silicate); clamp to
    # prevent a negative/zero denominator in gamma.
    A_C_raw = ALK_kg - A_B
    A_C = jnp.clip(A_C_raw, 1.0e-10, None)
    gamma = DIC_kg / A_C
    b_coeff = K1 * (1.0 - gamma)
    c_coeff = K1 * K2 * (1.0 - 2.0 * gamma)
    disc_raw = b_coeff ** 2 - 4.0 * c_coeff
    H_raw = (-b_coeff + jnp.sqrt(jnp.clip(disc_raw, 0.0, None))) / 2.0
    H_new = jnp.clip(H_raw, 1.0e-12, 1.0e-4)
    return H_new, A_C_raw, disc_raw, H_raw, gamma


def solve_carbonate_system(
    DIC: jnp.ndarray,
    ALK: jnp.ndarray,
    T_degC: jnp.ndarray,
    S_psu: jnp.ndarray,
    rho_sw: float = constants.rho_ocean,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the carbonate system for pCO2 and pH.

    Uses the analytic solver of Follows et al. (2006), which avoids
    Newton-Raphson by reducing each step to a quadratic in [H+].  The
    borate-alkalinity correction depends on the unknown [H+], so it is
    iterated ``_N_BORATE_ITER`` (=8) times from an initial guess pH ~ 8.1;
    across the tested ocean T/S/DIC/ALK grid the result is within ~1 uatm
    of a fully-converged (200-step) solve of the SAME reduced chemistry.
    A single un-iterated step (the prior form) errs by up to ~531 uatm over
    that grid (the warm, low-buffer corner).  This neglects the water/borate-excluded minor
    alkalinity species (see the module Faithfulness note), so it is not a
    substitute for a full solver such as CO2SYS.

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

    # Follows et al. (2006): the borate alkalinity A_B = B_T*K_B/(K_B+[H+])
    # depends on the unknown [H+], so the carbonate-alkalinity residual
    # A_C = ALK - A_B and the resulting [H+] must be solved self-consistently.
    # Iterate the borate correction (``_borate_quadratic_step``) from an initial
    # guess pH ~ 8.1; a single step (the original form) errs by up to ~531 uatm
    # over the tested grid (warm low-buffer corner), whereas _N_BORATE_ITER passes
    # converge to < 1 uatm.  Fixed count ⇒ JIT-static + AD-trivial (a small
    # unrolled loop of smooth, guarded ops).
    H = 10.0 ** (-8.1) * jnp.ones_like(DIC_kg)
    for _ in range(_N_BORATE_ITER):
        H = _borate_quadratic_step(H, DIC_kg, ALK_kg, K1, K2, K_B, B_T)[0]

    # pCO2 from [CO2*] = DIC * H^2 / (H^2 + K1*H + K1*K2)
    denom = H ** 2 + K1 * H + K1 * K2
    CO2_aq = DIC_kg * H ** 2 / jnp.clip(denom, 1.0e-30, None)

    # pCO2 = [CO2*] / K0 in atm, convert to uatm
    pCO2 = CO2_aq / jnp.clip(K0, 1.0e-10, None) * 1.0e6

    # pH = -log10([H+])
    pH = -jnp.log10(jnp.clip(H, 1.0e-14, None))

    return pCO2, pH
