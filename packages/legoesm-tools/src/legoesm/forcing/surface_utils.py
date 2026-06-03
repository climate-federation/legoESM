"""Surface blending utilities for AMIP-style experiments.

Provides common surface-property computations shared across scripts:
- Sea-ice / ocean temperature blending
- Sea-ice / ocean albedo and emissivity blending
- Column AOD distribution to model layers
"""

from __future__ import annotations

import jax.numpy as jnp


def blend_surface_temperature(
    sst: jnp.ndarray,
    sic: jnp.ndarray,
    T_ice: float,
) -> jnp.ndarray:
    """Blend ocean SST and sea-ice temperature by ice concentration.

    Parameters
    ----------
    sst : array
        Sea surface temperature [K].
    sic : array
        Sea-ice concentration [0, 1].
    T_ice : float
        Sea-ice temperature [K].

    Returns
    -------
    T_sfc : array
        Blended surface temperature [K].
    """
    return sic * T_ice + (1.0 - sic) * sst


def blend_surface_property(
    sic: jnp.ndarray,
    value_ice: float | jnp.ndarray,
    value_ocean: float | jnp.ndarray,
) -> jnp.ndarray:
    """Blend an ice and ocean surface property by ice concentration.

    Useful for albedo, emissivity, or any linearly-blended quantity.

    Parameters
    ----------
    sic : array
        Sea-ice concentration [0, 1].
    value_ice : float or array
        Property value over ice.
    value_ocean : float or array
        Property value over ocean.

    Returns
    -------
    blended : array
        Ice-concentration-weighted blend.
    """
    return sic * value_ice + (1.0 - sic) * value_ocean


def distribute_column_aod_to_layers(
    aod_col: jnp.ndarray,
    p_half_col: jnp.ndarray,
) -> jnp.ndarray:
    """Distribute column aerosol optical depth to layers by pressure thickness.

    Parameters
    ----------
    aod_col : array, shape (ncol,)
        Total column AOD per column.
    p_half_col : array, shape (ncol, nlev+1)
        Half-level pressures.

    Returns
    -------
    aod_layers : array, shape (ncol, nlev)
        Layer-distributed AOD.
    """
    dp = jnp.clip(p_half_col[:, 1:] - p_half_col[:, :-1], 1.0e-12, None)
    w = dp / jnp.sum(dp, axis=1, keepdims=True)
    return jnp.clip(aod_col, 0.0, None)[:, None] * w
