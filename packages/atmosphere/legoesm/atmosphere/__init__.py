"""Atmosphere component for legoESM."""

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterModel as ShallowWaterModel,
    cdgrid_shallow_water_tendencies as shallow_water_tendencies,
)
