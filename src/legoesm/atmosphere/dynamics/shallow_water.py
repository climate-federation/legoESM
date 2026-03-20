"""Deprecated shallow water alias module.

.. deprecated::
    Import from ``legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid``
    instead (``CDGridShallowWaterModel``, ``CDGridShallowWaterConfig``, etc.).

Re-exports the CDGrid shallow water model under the legacy names so that
existing scripts continue to work, but emits ``DeprecationWarning`` on import.
"""

import warnings as _warnings

_warnings.warn(
    "Importing from legoesm.atmosphere.dynamics.shallow_water is deprecated. "
    "Use legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid instead "
    "(CDGridShallowWaterModel, CDGridShallowWaterConfig, etc.).",
    DeprecationWarning,
    stacklevel=2,
)

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
