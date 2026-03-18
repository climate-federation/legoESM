"""Backward-compatible alias for shallow water FV model.

This module re-exports classes from shallow_water_fv3_cdgrid for backward
compatibility with code that imports from the old module name.
"""

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterModel as FVShallowWaterModel,
    CDGridShallowWaterConfig as FVShallowWaterConfig,
    CDGridShallowWaterState as FVShallowWaterState,
    cdgrid_shallow_water_tendencies as fv_shallow_water_tendencies,
)

__all__ = [
    "FVShallowWaterModel",
    "FVShallowWaterConfig",
    "FVShallowWaterState",
    "fv_shallow_water_tendencies",
]
