"""Backward-compatible shallow water module.

Re-exports the CDGrid shallow water model under the legacy names
``ShallowWaterModel``, ``ShallowWaterConfig``, and
``shallow_water_tendencies`` so that existing scripts and tests that import
from ``legoesm.atmosphere.dynamics.shallow_water`` continue to work.
"""

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterModel as ShallowWaterModel,
    CDGridShallowWaterConfig as ShallowWaterConfig,
    CDGridShallowWaterState as ShallowWaterState,
    cdgrid_shallow_water_tendencies as shallow_water_tendencies,
)

__all__ = [
    "ShallowWaterModel",
    "ShallowWaterConfig",
    "ShallowWaterState",
    "shallow_water_tendencies",
]
