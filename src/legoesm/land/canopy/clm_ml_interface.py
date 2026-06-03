"""Interface between legoESM and the CLM-ML-JAX multilayer canopy model.

This module provides :func:`compute_clm_ml_canopy_fluxes`, which translates
legoESM's ``AtmToSurface`` forcing into the ``mlcanopy_type`` input
container, calls ``MLCanopyFluxes``, and maps the output back to a
``SurfaceFluxOutput``.

All imports of ``clm_ml_jax``  / ``multilayer_canopy`` are **lazy** (inside
function bodies) so that the rest of legoESM continues to import cleanly
without the optional ``canopy`` extra installed.

Phase implementation status
---------------------------
Phase 2 (current): stub — signature is correct but body raises
    ``NotImplementedError``.  Used only to exercise the dispatch path and
    confirm that existing tests are unaffected.
Phase 3: full implementation — SW partitioning, AtmToSurface → mlcanopy_type
    mapping, MLCanopyFluxes call, output mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.canopy.state import CanopyState
from legoesm.land.surface_scheme import SurfaceFluxOutput

if TYPE_CHECKING:
    from legoesm.land.config import MultiLayerLandConfig


def compute_clm_ml_canopy_fluxes(
    T_soil_top: jnp.ndarray,
    forcing: AtmToSurface,
    canopy_config: CLMMLCanopyConfig,
    land_config: "MultiLayerLandConfig",
    land_params: Any | None,
    w_frac_rz: jnp.ndarray,
    wind_speed: jnp.ndarray,
    canopy_state: CanopyState | None,
    dt: float,
) -> tuple[SurfaceFluxOutput, CanopyState]:
    """Compute canopy fluxes via the CLM-ML-JAX multilayer canopy model.

    Parameters
    ----------
    T_soil_top:
        Soil surface temperature [K], shape ``(ncol,)``.
    forcing:
        Atmospheric forcing from the coupler.
    canopy_config:
        Static CLM-ML-JAX configuration.
    land_config:
        Parent ``MultiLayerLandConfig`` (for fallback soil parameters).
    land_params:
        Per-column ``LandSurfaceParams`` (or ``None`` for config defaults).
        Must supply ``LAI``, ``SAI``, ``htop``, ``hbot`` for the canopy
        scheme to be meaningful.
    w_frac_rz:
        Root-zone soil-moisture stress fraction [0–1], shape ``(ncol,)``.
    wind_speed:
        Scalar wind speed [m/s], shape ``(ncol,)``.
    canopy_state:
        Previous-step canopy state (``mlcanopy_type`` wrapper).  ``None``
        on the cold start; the interface will allocate and initialise a
        fresh ``mlcanopy_type``.
    dt:
        Timestep [s].

    Returns
    -------
    surface_out : SurfaceFluxOutput
        Canopy surface energy balance fluxes for the legoESM coupler.
    new_canopy_state : CanopyState
        Updated prognostic state to carry forward to the next step.

    Raises
    ------
    NotImplementedError
        Phase 2 stub — full implementation in Phase 3.
    """
    raise NotImplementedError(
        "CLM-ML-JAX canopy interface is not yet implemented (Phase 2 stub). "
        "Full implementation will be added in Phase 3."
    )
