"""Prescribed forcing for legoESM.

Provides AMIP-style prescribed SST and sea-ice boundary conditions.
"""

from legoesm.forcing.amip import (
    AMIPForcingConfig,
    AMIPForcing,
    get_amip_preset,
    load_amip_forcing,
    get_forcing_at_time,
)

__all__ = [
    "AMIPForcingConfig",
    "AMIPForcing",
    "get_amip_preset",
    "load_amip_forcing",
    "get_forcing_at_time",
]
