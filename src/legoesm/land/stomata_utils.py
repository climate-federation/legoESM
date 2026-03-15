"""Stomatal / carbon dispatch utilities.

Shared by both the slab land and multi-layer land models.
Computes the effective moisture availability beta, with optional
stomatal conductance and carbon coupling.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.stomata import (
    coupled_farquhar_stomata,
    compute_stomatal_beta,
    jarvis_gs,
)


def compute_effective_beta(
    T_surface: jnp.ndarray,
    forcing: AtmToSurface,
    beta_soil: jnp.ndarray,
    config,
    carbon_state: CarbonState | None,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Compute effective moisture availability beta, with optional stomatal/carbon coupling.

    When stomatal conductance is disabled, returns ``beta_soil`` directly.
    When enabled with the differland carbon scheme, uses the coupled
    Farquhar-stomata solver.  Otherwise falls back to the Jarvis model.

    Parameters
    ----------
    T_surface : jnp.ndarray
        Surface temperature [K].
    forcing : AtmToSurface
        Atmospheric forcing fields (provides sw_down, co2_ppmv,
        q_lowest, p_surface).
    beta_soil : jnp.ndarray
        Bucket/soil moisture beta [0-1].
    config : LandConfig or MultiLayerLandConfig
        Land configuration (must have ``stomata`` and ``carbon`` attributes).
    carbon_state : CarbonState or None
        Current carbon pools (needed for differland LAI).
    dt : float
        Time step [s] (reserved for future use).

    Returns
    -------
    (beta, gpp_farq)
        Effective moisture factor [0-1] and Farquhar GPP [gC/m2/s] or None.
    """
    gpp_farq = None

    if config.stomata.enabled:
        if config.carbon.scheme == "differland" and carbon_state is not None:
            LAI = carbon_state.C_fol / config.carbon.LCMA
            gs, gpp_farq = coupled_farquhar_stomata(
                T_surface, forcing.sw_down, forcing.co2_ppmv,
                forcing.q_lowest, forcing.p_surface, LAI, beta_soil,
                config.stomata)
            beta = compute_stomatal_beta(
                gs, LAI, beta_soil, config.stomata)
        else:
            gs = jarvis_gs(
                T_surface, forcing.sw_down, forcing.q_lowest,
                forcing.p_surface, beta_soil, config.stomata)
            beta = compute_stomatal_beta(
                gs, None, beta_soil, config.stomata)
    else:
        beta = beta_soil

    return beta, gpp_farq
