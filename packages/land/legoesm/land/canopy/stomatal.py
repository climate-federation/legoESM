"""Leaf-level stomatal conductance primitives for the two-leaf canopy.

Two empirical leaf-boundary stomatal conductance models:

- ``ball_berry_gs``: Ball, Woodrow & Berry (1987) — RH-based.
- ``medlyn_gs``:     Medlyn et al. (2011) — VPD / optimality.

Both return ``gs`` in ``mol H2O / m^2 / s`` at the leaf boundary and take
the slope (``m`` / ``g1``) and intercept (``b0`` / ``g0``) as **explicit
arguments**, not a ``StomataConfig``.  This is what distinguishes them from
the config-based stomatal roster in ``legoesm.land.carbon.stomata`` (used by
the SimpleSEB column path and the coupled Farquhar solver): the two-leaf
canopy passes per-leaf-class ``(m_C3, b0_C3)`` / ``(m_C4, b0_C4)`` values
that cannot be sourced from a single scalar config.

The CO2-independent Jarvis model and the coupled Farquhar-stomata solver
live in ``legoesm.land.carbon.stomata`` (``jarvis_gs``,
``coupled_farquhar_stomata``, ``compute_stomatal_beta``); this module is the
canopy-only leaf-level subset.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.land.leaf_biophysics import DIFFUSIVITY_RATIO_H2O_CO2


def ball_berry_gs(
    An: jax.Array,
    RH: jax.Array,
    Cs: jax.Array,
    m: jax.Array | float,
    b0: jax.Array | float,
) -> jax.Array:
    """Ball-Berry (1987) stomatal conductance.

    gs = b0 + m * max(An, 0) * RH / Cs   [mol H2O / m^2 / s]

    Parameters
    ----------
    An : net assimilation rate [μmol CO2 / m^2 / s]
    RH : relative humidity at the leaf surface [-]
    Cs : CO2 concentration at the leaf surface [μmol / mol]
    m  : Ball-Berry slope [-]; typically 9 for C3, 4 for C4
    b0 : Ball-Berry intercept / residual conductance [mol / m^2 / s];
         typically 0.01 for C3, 0.04 for C4

    Returns
    -------
    gs : stomatal conductance, lower-bounded by ``b0`` [mol H2O / m^2 / s]
    """
    A_pos = jnp.maximum(An, 0.0)
    Cs_safe = jnp.maximum(Cs, 1.0)
    return jnp.maximum(b0 + m * A_pos * RH / Cs_safe, b0)


def medlyn_gs(
    An: jax.Array,
    VPD_kPa: jax.Array,
    Cs: jax.Array,
    g1: jax.Array | float,
    g0: jax.Array | float,
) -> jax.Array:
    """Medlyn et al. (2011) optimal (USO) stomatal conductance.

    gs = g0 + 1.6 * (1 + g1 / sqrt(VPD)) * max(An, 0) / Cs   [mol H2O / m^2 / s]

    Parameters
    ----------
    An      : net assimilation rate [μmol CO2 / m^2 / s]
    VPD_kPa : vapour pressure deficit at the leaf surface [kPa]
    Cs      : CO2 concentration at the leaf surface [μmol / mol]
    g1      : Medlyn slope [kPa^0.5]; typically 4 for C3, 1-2 for C4
    g0      : residual conductance [mol / m^2 / s]

    Returns
    -------
    gs : stomatal conductance, lower-bounded by ``g0`` [mol H2O / m^2 / s]
    """
    A_pos = jnp.maximum(An, 0.0)
    Cs_safe = jnp.maximum(Cs, 1.0)
    # Floor VPD to avoid divergence of 1/sqrt(VPD) as VPD → 0.
    VPD_safe = jnp.maximum(VPD_kPa, 0.05)  # coeff-ok: VPD floor guarding 1/sqrt(VPD) blow-up
    return jnp.maximum(
        g0
        + DIFFUSIVITY_RATIO_H2O_CO2
        * (1.0 + g1 / jnp.sqrt(VPD_safe))
        * A_pos
        / Cs_safe,
        g0,
    )
