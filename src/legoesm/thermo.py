"""Lightweight saturation thermodynamics for legoESM.

This module provides `saturation_mixing_ratio` and
`saturation_mixing_ratio_ice` with *no* dependency on
``atmosphere.physics`` so that ``land/``, ``ice/``, ``ocean/``, and
``coupler/`` modules can import them without pulling in the full
atmosphere physics package.

All operations are pure JAX and compatible with jit, grad, vmap, scan.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants


def saturation_mixing_ratio(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation mixing ratio using Tetens formula.

    e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))   where T_c = T - 273.15
    q_sat = epsilon * e_sat / (p - e_sat)

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Saturation mixing ratio [kg/kg].
    """
    T_c = T - constants.T_freeze  # Celsius
    e_sat = 611.2 * jnp.exp(17.67 * T_c / (T_c + 243.5))
    # Clip denominator to avoid division by zero when e_sat >= p
    denom = jnp.maximum(p - e_sat, 1.0)
    q_sat = constants.epsilon * e_sat / denom
    # Cap at 1.0 kg/kg: prevents singularity at low-pressure levels
    # where e_sat > p (e.g. isothermal 300K init above ~3500 Pa)
    return jnp.minimum(q_sat, 1.0)


def saturation_mixing_ratio_ice(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation mixing ratio over ice (Clausius-Clapeyron).

    e_sat_i = 611.2 * exp(L_s/R_v * (1/T_freeze - 1/T))
    q_sat_i = epsilon * e_sat_i / (p - e_sat_i)

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Ice saturation mixing ratio [kg/kg].
    """
    e_sat_i = 611.2 * jnp.exp(
        constants.L_s / constants.R_v * (1.0 / constants.T_freeze - 1.0 / T)
    )
    denom = jnp.maximum(p - e_sat_i, 1.0)
    q_sat_i = constants.epsilon * e_sat_i / denom
    return jnp.minimum(q_sat_i, 1.0)
