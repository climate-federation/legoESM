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
from legoesm.land.carbon.stomata import (
    StomataConfig,
    arrhenius,
    peaked_arrhenius,
    farquhar_photosynthesis,
    ball_berry_gs,
    medlyn_gs,
    jarvis_gs,
    coupled_farquhar_stomata,
    solve_coupled_farquhar_ci,
    CoupledLeafState,
    compute_stomatal_beta,
)

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
    # Stomata / plant physiology
    "StomataConfig",
    "arrhenius",
    "peaked_arrhenius",
    "farquhar_photosynthesis",
    "ball_berry_gs",
    "medlyn_gs",
    "jarvis_gs",
    "coupled_farquhar_stomata",
    "solve_coupled_farquhar_ci",
    "CoupledLeafState",
    "compute_stomatal_beta",
]
