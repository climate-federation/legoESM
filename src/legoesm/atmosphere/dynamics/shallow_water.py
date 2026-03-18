"""Backward-compatible shallow water module.

Re-exports the FV shallow water model (which wraps the CDGrid implementation)
under the legacy names ``ShallowWaterModel``, ``ShallowWaterConfig``, and
``shallow_water_tendencies`` so that existing scripts and tests that import
from ``legoesm.atmosphere.dynamics.shallow_water`` continue to work.

The FV wrapper accepts and returns ``ShallowWaterState`` (Field-based)
objects, which is what legacy code expects.
"""

from legoesm.atmosphere.dynamics.shallow_water_fv import (
    FVShallowWaterModel as ShallowWaterModel,
    FVShallowWaterConfig as ShallowWaterConfig,
    fv_shallow_water_tendencies as shallow_water_tendencies,
)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterState as ShallowWaterState,
)

__all__ = [
    "ShallowWaterModel",
    "ShallowWaterConfig",
    "ShallowWaterState",
    "shallow_water_tendencies",
]
