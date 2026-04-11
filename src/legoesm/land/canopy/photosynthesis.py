"""C3 and C4 Farquhar photosynthesis for the two-leaf canopy model.

C3: Farquhar et al. (1980) with colimitation (DePury & Farquhar 1997).
    Temperature response with growth-temperature acclimation (Leuning 2002).
C4: Simplified C4 model with Q10 temperature response.

All functions are pure JAX, JIT-compatible, and differentiable.
Source: adapted from DifferBESS/process/Photosynthesis.py (Ryu et al.).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Temperature acclimation helpers
# ---------------------------------------------------------------------------

def _compute_vcmax_deltaS(TgC: jax.Array) -> jax.Array:
    """Entropy parameter for Vcmax temperature response (Leuning 2002).

    Parameters
    ----------
    TgC : growth temperature [°C], clipped to [11, 35]

    Returns
    -------
    DeltaS [kJ mol-1 K-1]
    """
    TgC_clip = jnp.clip(TgC, 11.0, 35.0)
    return (-1.07 * TgC_clip + 668.39) / 1000.0


def vcmax_temperature_response(Tf: jax.Array, TgC: jax.Array) -> jax.Array:
    """Normalised Vcmax temperature response with growth-temperature acclimation.

    Leuning (2002) formulation.  Returns the scaling factor f(T) such that
    Vcmax(T) = f(T) * Vcmax25.

    Parameters
    ----------
    Tf  : leaf temperature [K]
    TgC : 30-day mean growth temperature [°C]

    Returns
    -------
    f : dimensionless scaling factor
    """
    Ha  = 72.0    # activation energy [kJ mol-1]
    Hd  = 200.0   # deactivation energy [kJ mol-1]
    R   = 0.008314  # gas constant [kJ mol-1 K-1]
    T0  = 298.15   # reference temperature [K]

    f_up = jnp.exp(Ha * (Tf - T0) / (T0 * R * Tf))
    DeltaS = _compute_vcmax_deltaS(TgC)
    f_down = (
        (1.0 + jnp.exp((T0 * DeltaS - Hd) / (T0 * R)))
        / (1.0 + jnp.exp((Tf * DeltaS - Hd) / (Tf * R)))
    )
    return f_up * f_down


def rd_temperature_response(Tf: jax.Array, TgC: jax.Array) -> jax.Array:
    """Normalised dark respiration temperature response with acclimation.

    Parameters
    ----------
    Tf  : leaf temperature [K]
    TgC : 30-day mean growth temperature [°C]

    Returns
    -------
    f : dimensionless scaling factor
    """
    TgC_clip = jnp.clip(TgC, 11.0, 35.0)
    TaC = Tf - 273.15
    Q10 = jnp.maximum(3.22 - 0.046 * TaC, 1e-3)
    item = (Tf - 298.15) / 10.0
    return (10.0 ** (-0.00794 * (TgC_clip - 25.0))) * (Q10 ** item)


# ---------------------------------------------------------------------------
# C3 photosynthesis (Farquhar et al. 1980, DePury & Farquhar 1997)
# ---------------------------------------------------------------------------

@jax.jit
def c3_photosynthesis(
    Tf: jax.Array,
    Ci: jax.Array,
    APAR: jax.Array,
    Vcmax25: jax.Array,
    Ps: jax.Array,
    alf: jax.Array,
    TgC: jax.Array,
) -> jax.Array:
    """Net assimilation rate for C3 photosynthesis.

    Parameters
    ----------
    Tf     : leaf temperature [K]
    Ci     : intercellular CO2 concentration [μmol mol-1]
    APAR   : absorbed PAR [μmol m-2 s-1]
    Vcmax25: maximum carboxylation rate at 25°C [μmol m-2 s-1]
    Ps     : atmospheric pressure [Pa]
    alf    : quantum yield for electron transport [mol CO2 mol photons-1]
    TgC    : 30-day mean growth temperature [°C]

    Returns
    -------
    An : net assimilation rate [μmol m-2 s-1] (non-negative)
    """
    R = 8.314e-3  # [kJ K-1 mol-1]

    # Partial pressures [Pa]
    O   = Ps * 0.21
    KC  = jnp.exp(38.05 - 79.43 / (R * Tf)) * 1e-6 * Ps   # Michaelis for CO2
    KO  = jnp.exp(20.30 - 36.38 / (R * Tf)) * 1e-3 * Ps   # Michaelis for O2
    GS  = jnp.exp(19.02 - 37.83 / (R * Tf)) * 1e-6 * Ps   # CO2 compensation point
    K   = KC * (1.0 + O / KO)
    Pi  = Ci * 1e-6 * Ps  # intercellular CO2 partial pressure [Pa]

    # Temperature-dependent Vcmax and Rd
    Vcmax = vcmax_temperature_response(Tf, TgC) * Vcmax25
    Rd_o  = 0.015 * Vcmax
    Rd    = Rd_o * rd_temperature_response(Tf, TgC)

    # Three limiting rates
    JC = Vcmax * (Pi - GS) / (Pi + K)
    JE = alf   * APAR * (Pi - GS) / (Pi + 2.0 * GS)
    JS = Vcmax / 2.0

    # Colimitation JC–JE (DePury & Farquhar 1997)
    a1  = 0.98
    b1  = -(JC + JE)
    c1  = JC * JE
    disc1 = jnp.maximum(b1**2 - 4.0 * a1 * c1, 1e-20)
    JCE = (-b1 + jnp.sign(b1) * jnp.sqrt(disc1)) / (2.0 * a1)

    # Colimitation JCE–JS
    a2  = 0.95
    b2  = -(JCE + JS)
    c2  = JCE * JS
    disc2 = jnp.maximum(b2**2 - 4.0 * a2 * c2, 1e-20)
    JCES = (-b2 + jnp.sign(b2) * jnp.sqrt(disc2)) / (2.0 * a2)

    An = jnp.maximum(JCES - Rd, 0.0)
    return An


# ---------------------------------------------------------------------------
# C4 photosynthesis
# ---------------------------------------------------------------------------

@jax.jit
def c4_photosynthesis(
    Tf: jax.Array,
    Ci: jax.Array,
    APAR: jax.Array,
    Vcmax25: jax.Array,
) -> jax.Array:
    """Net assimilation rate for C4 photosynthesis.

    Parameters
    ----------
    Tf     : leaf temperature [K]
    Ci     : intercellular CO2 concentration [μmol mol-1]
    APAR   : absorbed PAR [μmol m-2 s-1]
    Vcmax25: maximum carboxylation rate at 25°C [μmol m-2 s-1]

    Returns
    -------
    An : net assimilation rate [μmol m-2 s-1] (non-negative)
    """
    item = (Tf - 298.15) / 10.0
    Q10  = 2.0

    kp25 = 0.02 * Vcmax25
    k    = kp25 * Q10 ** item   # PEP carboxylase efficiency

    Vcmax_o = Vcmax25 * Q10 ** item
    Vcmax   = Vcmax_o / ((1.0 + jnp.exp(0.3 * (286.15 - Tf)))
                         * (1.0 + jnp.exp(0.3 * (Tf - 309.15))))
    Rd_o = 0.8 * Q10 ** item
    Rd   = Rd_o / (1.0 + jnp.exp(1.3 * (Tf - 328.15)))

    # Three limiting rates
    alf_c4 = 0.067   # quantum yield for C4
    Je = Vcmax
    Ji = alf_c4 * APAR
    ci = Ci * 1e-6
    Jc = ci * k * 1e6

    # Colimitation Je–Ji
    a1   = 0.80
    b1   = -(Je + Ji)
    c1   = Je * Ji
    disc1 = jnp.maximum(b1**2 - 4.0 * a1 * c1, 1e-20)
    Jei  = (-b1 + jnp.sign(b1) * jnp.sqrt(disc1)) / (2.0 * a1)

    # Colimitation Jei–Jc
    a2   = 0.95
    b2   = -(Jei + Jc)
    c2   = Jei * Jc
    disc2 = jnp.maximum(b2**2 - 4.0 * a2 * c2, 1e-20)
    Jeic = (-b2 + jnp.sign(b2) * jnp.sqrt(disc2)) / (2.0 * a2)

    An = jnp.maximum(Jeic - Rd, 0.0)
    return An


# ---------------------------------------------------------------------------
# Mixed C3/C4 photosynthesis
# ---------------------------------------------------------------------------

@jax.jit
def photosynthesis(
    Tf: jax.Array,
    Ci: jax.Array,
    APAR: jax.Array,
    Vcmax25_C3: jax.Array,
    Vcmax25_C4: jax.Array,
    fC4: jax.Array,
    Ps: jax.Array,
    alf: jax.Array,
    TgC: jax.Array,
) -> jax.Array:
    """Net assimilation for a mixed C3/C4 canopy.

    Computes both C3 and C4 rates and combines them using a continuous
    fraction fC4, so the result is differentiable with respect to fC4.

    Parameters
    ----------
    Tf, Ci, APAR : leaf temperature [K], intercellular CO2 [μmol/mol],
                   absorbed PAR [μmol m-2 s-1]
    Vcmax25_C3, Vcmax25_C4 : canopy-integrated Vcmax25 [μmol m-2 s-1]
    fC4     : C4 area fraction [0–1] — traced value, not branched
    Ps      : atmospheric pressure [Pa]
    alf     : quantum yield [mol CO2 mol photons-1] (used for C3)
    TgC     : 30-day mean growth temperature [°C]

    Returns
    -------
    An : net assimilation rate [μmol m-2 s-1]
    """
    An_C3 = c3_photosynthesis(Tf, Ci, APAR, Vcmax25_C3, Ps, alf, TgC)
    An_C4 = c4_photosynthesis(Tf, Ci, APAR, Vcmax25_C4)
    # Continuous weighted average — fully differentiable wrt fC4
    return (1.0 - fC4) * An_C3 + fC4 * An_C4
