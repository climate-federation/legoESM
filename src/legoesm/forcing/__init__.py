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
]
