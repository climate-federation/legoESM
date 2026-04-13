"""Stomatal / carbon dispatch utilities.

Shared by the slab land and multi-layer land models when running with
the **SimpleSEB** surface scheme.  Computes an effective moisture
availability ``beta`` — starting from the bucket / root-zone-weighted
soil beta — and optionally down-regulates it via Jarvis (carbon off)
or coupled Leuning Farquhar + Ball-Berry / Medlyn (carbon on).

The TwoLeafCanopy surface scheme bypasses this module and uses its own
Newton closure in ``canopy/solver.py`` — it does not need a beta proxy
because LE is computed directly from leaf ↔ canopy-air humidity gradients.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.land.carbon.config import CarbonState
from legoesm.land.canopy.stomatal import (
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
    land_params=None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Compute effective moisture availability beta, with optional stomatal/carbon coupling.

    When stomatal conductance is disabled, returns ``beta_soil`` directly.
    When enabled with the differland carbon scheme, uses the coupled
    Leuning Farquhar-stomata solver.  Otherwise falls back to the Jarvis
    multiplicative model.

    Parameters
    ----------
    T_surface : jnp.ndarray
        Surface temperature [K].
    forcing : AtmToSurface
        Atmospheric forcing fields (provides sw_down, co2_ppmv,
        q_lowest, p_surface).
    beta_soil : jnp.ndarray
        Bucket / root-zone soil moisture beta [0-1].
    config : LandConfig or MultiLayerLandConfig
        Land configuration (must have ``stomata`` and ``carbon`` attributes).
    carbon_state : CarbonState or None
        Current carbon pools (needed for differland LAI).
    dt : float
        Time step [s] (reserved for future use).
    land_params : LandSurfaceParams or None
        Spatially-varying parameters.  When provided, overrides
        ``config.stomata.Vcmax25_C3``, ``config.stomata.g1_bb``,
        and ``config.carbon.LCMA`` with per-cell arrays.

    Returns
    -------
    (beta, gpp_farq)
        Effective moisture factor [0-1] and Farquhar GPP [gC/m2/s] or None.
    """
    # Override stomatal / carbon config fields with spatial arrays if provided.
    if land_params is not None:
        _stomata = config.stomata._replace(
            Vcmax25_C3=land_params.Vc_max25,
            g1_bb=land_params.g1,
        )
        _carbon = config.carbon._replace(LCMA=land_params.LCMA)
    else:
        _stomata = config.stomata
        _carbon = config.carbon

    gpp_farq = None

    if config.stomata.enabled:
        if config.carbon.scheme == "differland" and carbon_state is not None:
            LAI = carbon_state.C_fol / _carbon.LCMA
            gs, gpp_farq = coupled_farquhar_stomata(
                T_surface, forcing.sw_down, forcing.co2_ppmv,
                forcing.q_lowest, forcing.p_surface, LAI, beta_soil,
                _stomata)
            beta = compute_stomatal_beta(
                gs, LAI, beta_soil, _stomata)
        else:
            gs = jarvis_gs(
                T_surface, forcing.sw_down, forcing.q_lowest,
                forcing.p_surface, beta_soil, _stomata)
            beta = compute_stomatal_beta(
                gs, None, beta_soil, _stomata)
    else:
        beta = beta_soil

    return beta, gpp_farq
