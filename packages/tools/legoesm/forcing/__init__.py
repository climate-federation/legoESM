"""Prescribed forcing for legoESM.

Provides AMIP-style prescribed SST and sea-ice boundary conditions,
experiment configuration, checkpoint/restart, and external forcing scaffolds.
"""

from legoesm.forcing.amip import (
    AMIPForcingConfig,
    AMIPForcing,
    get_amip_preset,
    load_amip_forcing,
    get_forcing_at_time,
)
from legoesm.forcing.amip_config import (
    AMIPExperimentConfig,
    config_to_dict,
    config_from_dict,
    save_config,
    load_config,
    save_checkpoint,
    load_checkpoint,
)
from legoesm.forcing.external import (
    ExternalForcingConfig,
    GHGConfig,
    OzoneConfig,
    AerosolConfig,
    SolarConfig,
    get_solar_forcing_at_time,
)
from legoesm.forcing.experiments import (
    ExperimentTemplate,
    EXPERIMENT_TEMPLATES,
    ghg_at_year,
    get_ghg_for_experiment,
    create_experiment_config,
)
from legoesm.forcing.analytical import analytical_sst_sic
from legoesm.forcing.time_utils import day_to_calendar
from legoesm.forcing.surface_utils import (
    blend_surface_temperature,
    blend_surface_property,
    distribute_column_aod_to_layers,
    place_stratospheric_aod_profile_to_layers,
)

__all__ = [
    "AMIPForcingConfig",
    "AMIPForcing",
    "get_amip_preset",
    "load_amip_forcing",
    "get_forcing_at_time",
    "AMIPExperimentConfig",
    "config_to_dict",
    "config_from_dict",
    "save_config",
    "load_config",
    "save_checkpoint",
    "load_checkpoint",
    "ExternalForcingConfig",
    "GHGConfig",
    "OzoneConfig",
    "AerosolConfig",
    "SolarConfig",
    "get_solar_forcing_at_time",
    # Experiment templates
    "ExperimentTemplate",
    "EXPERIMENT_TEMPLATES",
    "ghg_at_year",
    "get_ghg_for_experiment",
    "create_experiment_config",
    # Analytical forcing
    "analytical_sst_sic",
    # Time utilities
    "day_to_calendar",
    # Surface utilities
    "blend_surface_temperature",
    "blend_surface_property",
    "distribute_column_aod_to_layers",
    "place_stratospheric_aod_profile_to_layers",
]
