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
    CLMMLCanopyConfig,
    PFT_AERO_PARAMS,
    PFT_CANOPY_HEIGHT,
    PFT_LEAF_WIDTH,
    PFT_VCMAX25_C3,
    PFT_VCMAX25_C4,
    VCMAX25_C3_DEFAULT,
    VCMAX25_C4_DEFAULT,
    lookup_vcmax25,
)
from legoesm.land.canopy.sif import SIFConfig
from legoesm.land.canopy.state import CanopyState

__all__ = [
    "CanopyConfig",
    "CanopyLandParams",
    "CLMMLCanopyConfig",
    "CanopyState",
    "PFT_AERO_PARAMS",
    "PFT_CANOPY_HEIGHT",
    "PFT_LEAF_WIDTH",
    "PFT_VCMAX25_C3",
    "PFT_VCMAX25_C4",
    "VCMAX25_C3_DEFAULT",
    "VCMAX25_C4_DEFAULT",
    "lookup_vcmax25",
    # Solar-induced fluorescence (optional diagnostic) — the user-facing config.
    # The compute functions live in ``legoesm.land.canopy.sif``.
    "SIFConfig",
]
