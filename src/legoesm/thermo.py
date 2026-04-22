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


def saturation_vapor_pressure(T: jax.Array) -> jax.Array:
    """Compute saturation vapor pressure using Tetens formula.

    e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))

    Parameters
    ----------
    T : jax.Array
        Temperature [K].

    Returns
    -------
    jax.Array
        Saturation vapor pressure [Pa].
    """
    T_c = T - constants.T_freeze
    return 611.2 * jnp.exp(17.67 * T_c / (T_c + 243.5))


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
    e_sat = saturation_vapor_pressure(T)
    # Smooth floor on denominator: preserves gradients near e_sat ≈ p
    # instead of a hard clip that creates a zero-gradient plateau.
    # softplus(x - 1) + 1 ≈ x for x >> 1, ≈ 1 for x << 1, smooth at x = 1.
    denom = jax.nn.softplus(p - e_sat - 1.0) + 1.0
    q_sat = constants.epsilon * e_sat / denom
    # Smooth cap at 1.0 kg/kg: prevents singularity at low-pressure levels
    # while allowing gradients to flow (unlike hard jnp.minimum).
    # Uses LogSumExp smooth-min: 1 - softplus(β(1 - x))/β with β = 20.
    return 1.0 - jax.nn.softplus(20.0 * (1.0 - q_sat)) / 20.0


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
    denom = jax.nn.softplus(p - e_sat_i - 1.0) + 1.0
    q_sat_i = constants.epsilon * e_sat_i / denom
    return 1.0 - jax.nn.softplus(20.0 * (1.0 - q_sat_i)) / 20.0


def saturation_specific_humidity(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation specific humidity from saturation mixing ratio.

    q = w_sat / (1 + w_sat)

    where w_sat = epsilon * e_sat / (p - e_sat) is the saturation mixing
    ratio.  Use this function when working with specific humidity fields
    (q = m_v / (m_v + m_d)) rather than mixing ratio (w = m_v / m_d).

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Saturation specific humidity [kg/kg].
    """
    w_sat = saturation_mixing_ratio(T, p)
    return w_sat / (1.0 + w_sat)


def saturation_specific_humidity_ice(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation specific humidity over ice.

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Ice saturation specific humidity [kg/kg].
    """
    w_sat_ice = saturation_mixing_ratio_ice(T, p)
    return w_sat_ice / (1.0 + w_sat_ice)
