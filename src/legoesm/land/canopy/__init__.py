"""Canopy biophysics module for the legoESM two-leaf surface scheme.

This package contains the per-leaf radiative transfer, photosynthesis,
stomatal conductance, Monin-Obukhov stability, and Newton-Raphson energy
balance closure that the ``TwoLeafCanopyConfig`` surface scheme consumes.

The user-facing entry point for canopy simulations is::

    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(...))

— not the low-level helpers re-exported below, which exist for tests and
advanced diagnostics.
"""

from legoesm.land.canopy.config import (
    CanopyConfig,
    CanopyLandParams,
    PFT_AERO_PARAMS,
    PFT_CANOPY_HEIGHT,
    PFT_VCMAX25_C3,
    PFT_VCMAX25_C4,
)

__all__ = [
    "CanopyConfig",
    "CanopyLandParams",
    "PFT_AERO_PARAMS",
    "PFT_CANOPY_HEIGHT",
    "PFT_VCMAX25_C3",
    "PFT_VCMAX25_C4",
]
