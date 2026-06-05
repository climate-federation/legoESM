"""Surface radiation flux utilities.

Computes net shortwave and longwave radiation at the surface using
Stefan-Boltzmann emission. Shared by slab land, multi-layer land,
sea ice, lake, and ocean tile models.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


def surface_radiation_fluxes(
    sw_down: jnp.ndarray,
    lw_down: jnp.ndarray,
    T_sfc: jnp.ndarray,
    alpha: jnp.ndarray,
    emissivity: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute net surface radiation fluxes.

    Parameters
    ----------
    sw_down : array
        Downward shortwave radiation at surface [W/m2].
    lw_down : array
        Downward longwave radiation at surface [W/m2].
    T_sfc : array
        Surface temperature [K].
    alpha : array or float
        Surface albedo [0, 1].
    emissivity : float or array
        Surface emissivity [0, 1].

    Returns
    -------
    sw_net : array
        Net shortwave flux absorbed at surface [W/m2].
    lw_net : array
        Net longwave flux at surface [W/m2] (positive = warming surface).
    lw_up : array
        Upward longwave emission from surface [W/m2].
    """
    sw_net = (1.0 - alpha) * sw_down
    lw_down_abs = emissivity * lw_down
    lw_emit = emissivity * constants.sigma_sb * T_sfc ** 4
    # Total upward LW = thermal emission + reflected downward LW
    # (consistent with gray-radiation surface BC)
    lw_up = lw_emit + (1.0 - emissivity) * lw_down
    # Net LW at the surface = absorbed - emitted (surface energy budget)
    lw_net = lw_down_abs - lw_emit
    return sw_net, lw_net, lw_up
