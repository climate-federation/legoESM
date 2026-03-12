"""Ocean biogeochemistry: carbon cycle, NPZD ecosystem, air-sea CO2 exchange.

Switchable schemes:
- ``"none"``: Disabled (default).
- ``"abiotic"``: DIC + alkalinity with carbonate chemistry and air-sea CO2.
- ``"npzd"``: Full NPZD ecosystem coupled to the carbon cycle.
"""

from legoesm.ocean.biogeochemistry.config import (
    BiogeoConfig,
    OceanBiogeoState,
    BiogeoTendencies,
    AirSeaCO2Diagnostics,
    init_biogeo_state,
)
from legoesm.ocean.biogeochemistry.carbonate import (
    co2_solubility,
    carbonate_equilibria,
    solve_carbonate_system,
)
from legoesm.ocean.biogeochemistry.gas_exchange import (
    schmidt_number_co2,
    gas_transfer_velocity,
    air_sea_co2_flux,
)
from legoesm.ocean.biogeochemistry.npzd import (
    par_profile,
    npzd_source_sink,
)
from legoesm.ocean.biogeochemistry.carbon_cycle import (
    compute_biogeo_tendencies,
    step_ocean_biogeochemistry,
)

__all__ = [
    "BiogeoConfig",
    "OceanBiogeoState",
    "BiogeoTendencies",
    "AirSeaCO2Diagnostics",
    "init_biogeo_state",
    "co2_solubility",
    "carbonate_equilibria",
    "solve_carbonate_system",
    "schmidt_number_co2",
    "gas_transfer_velocity",
    "air_sea_co2_flux",
    "par_profile",
    "npzd_source_sink",
    "compute_biogeo_tendencies",
    "step_ocean_biogeochemistry",
]
