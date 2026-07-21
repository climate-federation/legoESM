"""Surface scheme abstraction for legoESM land models.

A *surface scheme* computes the fluxes at the atmosphere-surface interface
given the land state and forcing.  Two schemes are available:

- ``SimpleSEBConfig`` — bulk-flux surface energy balance with a
  homogeneous skin temperature (top soil layer for multilayer, slab T for
  slab).  Optionally coupled to Jarvis / Leuning Farquhar stomatal
  conductance via ``legoesm.land.stomata_utils.compute_effective_beta``.
  This is the default.

- ``TwoLeafCanopyConfig`` — DifferBESS-style two-leaf canopy energy
  balance + Farquhar C3/C4 photosynthesis with Newton-Raphson closure
  and an outer Picard loop reconciling canopy turbulence with soil
  thermal diffusion.

- ``CLMMLCanopyConfig`` — CLM-ML-JAX multilayer canopy model (Bonan
  et al. 2021, GMD, CLM-ML v2). Multi-layer within-canopy radiative
  transfer, turbulence, leaf energy balance, stomatal conductance, and
  plant hydraulics.  Requires ``pip install legoesm[canopy]``.
  **Not** ``jax.jit``-compatible; forward simulation only.

All three schemes produce a common ``SurfaceFluxOutput`` that downstream
land-model post-processing (snow, Richards, soil thermal, carbon,
TileResponse) consumes without knowing which scheme produced it.

Usage::

    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig

    # Default: simple SEB
    cfg = MultiLayerLandConfig()  # surface_scheme=SimpleSEBConfig() implicit

    # Two-leaf canopy
    cfg = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=30, LE_module="BT"))
"""

from legoesm.land.surface_scheme.base import SurfaceFluxOutput
from legoesm.land.surface_scheme.simple_seb import (
    SimpleSEBConfig,
    compute_simple_seb_fluxes,
)
from legoesm.land.surface_scheme.two_leaf_canopy import (
    TwoLeafCanopyConfig,
    compute_two_leaf_canopy_fluxes,
)

__all__ = [
    "SurfaceFluxOutput",
    "SimpleSEBConfig",
    "TwoLeafCanopyConfig",
    "compute_simple_seb_fluxes",
    "compute_two_leaf_canopy_fluxes",
]
