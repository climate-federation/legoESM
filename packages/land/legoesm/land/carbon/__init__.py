"""Land carbon cycle module.

Provides a switchable carbon model with two schemes:

- **differland**: DALEC990-based prognostic 6-pool model (labile, foliage,
  root, wood, litter, SOM) driven by a light-use-efficiency GPP with
  phenology and temperature/moisture-dependent decomposition.  Based on
  DifferLand v1.0 (Fang & Gentine, Columbia).

- **seasonal**: prescribed repeating sinusoidal NEE cycle with
  latitude-dependent amplitude and phase.  No prognostic pools.

Set ``CarbonConfig(scheme="none")`` (the default) to disable.
"""

from legoesm.land.carbon.config import CarbonConfig, CarbonState
from legoesm.land.carbon.carbon_cycle import (
    annual_frozen_fraction,
    compute_gpp,
    compute_phenology,
    init_carbon_state,
    perennial_frost_protection,
    seasonal_co2_flux,
    step_carbon,
    step_carbon_differland,
)

# Stomatal conductance + plant physiology now live in the neutral
# ``legoesm.land.stomata`` module (shared by the two-leaf canopy and the
# big-leaf SimpleSEB paths); import from there, not the carbon package.

__all__ = [
    "CarbonConfig",
    "CarbonState",
    "annual_frozen_fraction",
    "compute_gpp",
    "compute_phenology",
    "init_carbon_state",
    "perennial_frost_protection",
    "seasonal_co2_flux",
    "step_carbon",
    "step_carbon_differland",
]
