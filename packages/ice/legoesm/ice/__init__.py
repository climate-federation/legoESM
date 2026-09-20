"""Cryosphere component for legoESM.

Sea ice model with switchable dynamics and multi-category support:
- ``dynamics="none"``: Slab thermodynamics (default, backward compatible).
- ``dynamics="free_drift"``: Heuristic linear-combination velocity.
- ``dynamics="evp"``: EVP rheology (Hunke & Dukowicz 1997).
- ``dynamics="mevp"``: Modified-EVP pseudo-time relaxation
  (Bouillon 2013 / Kimmritz 2015).

Multi-category ice (``n_categories > 1``) uses a simplified category
transfer scheme for ice thickness redistribution.  Snow depth is not
tracked.
"""

from legoesm.ice.brine import (
    SaltBudgetResult,
    aggregate_salt_flux,
    update_salinity_and_salt_flux,
)
from legoesm.ice.c1d_omip_l3 import C1DOMIPL3Card, build_c1d_omip_l3_card
from legoesm.ice.config import (
    BrineConfig,
    MeltPondConfig,
    RidgingConfig,
    SI3ThermoConfig,
    SeaIceConfig,
    SnowConfig,
    validate_si3_bulk_config,
    validate_si3_thermo_config,
)
from legoesm.ice.constants_config import (
    IceConstantsConfig,
    NEMO_SI3_CONSTANTS_CONFIG,
)
from legoesm.ice.dynamics import (
    air_ice_stress,
    evp_solver,
    free_drift_velocity,
    mevp_solver,
    ocean_ice_stress,
    si3_cgrid_deformation,
    stress_divergence,
)
from legoesm.ice.itd import (
    aggregate_state,
    category_bounds,
    distribute_to_categories,
    linear_remap,
    lipscomb_2001_remap,
    upper_bounds,
)
from legoesm.ice.ponds import step_ponds
from legoesm.ice.rheology import (
    delta_deformation,
    evp_stress_update,
    ice_strength,
    mevp_stress_update,
    strain_rates,
    vp_stress,
)
from legoesm.ice.ridging import (
    SI3JPL1RidgingConfig,
    SI3JPL1RidgingLosses,
    SI3JPL1RidgingState,
    apply_ridging,
    apply_si3_jpl1_ridging,
    participation_weights,
)
from legoesm.ice.sea_ice import (
    grid_supports_ice_dynamics,
    grid_supports_ice_transport,
    step_sea_ice,
    uses_new_physics,
)
from legoesm.ice.shortwave import (
    IceSWResult,
    compute_ice_sw,
    delta_eddington_albedo,
    maykut_untersteiner_albedo,
)
from legoesm.ice.snow import (
    accumulate_snowfall,
    combined_conductive_flux,
    consume_from_snow_then_ice,
    consume_sublimation_from_snow_then_ice,
    snow_ice_flooding,
)
from legoesm.ice.state import (
    DynamicSeaIceState,
    SI3ColumnState,
    SeaIceState,
    distribute_dynamic_state_to_categories,
    dynamic_to_slab,
    init_dynamic_ice_state,
    slab_to_dynamic,
)
from legoesm.ice.transport import advect_ice_tracers

__all__ = [
    # Config and state
    "SeaIceConfig",
    "SeaIceState",
    "DynamicSeaIceState",
    "SI3ColumnState",
    "init_dynamic_ice_state",
    "distribute_dynamic_state_to_categories",
    "dynamic_to_slab",
    "slab_to_dynamic",
    # Main step function
    "step_sea_ice",
    "grid_supports_ice_dynamics",
    "grid_supports_ice_transport",
    "uses_new_physics",
    # Rheology
    "ice_strength",
    "strain_rates",
    "delta_deformation",
    "vp_stress",
    "evp_stress_update",
    "mevp_stress_update",
    # Dynamics
    "stress_divergence",
    "evp_solver",
    "mevp_solver",
    "free_drift_velocity",
    "air_ice_stress",
    "ocean_ice_stress",
    "si3_cgrid_deformation",
    # ITD
    "category_bounds",
    "upper_bounds",
    "aggregate_state",
    "distribute_to_categories",
    "linear_remap",
    "lipscomb_2001_remap",
    # Transport
    "advect_ice_tracers",
    # Configs
    "SnowConfig",
    "BrineConfig",
    "RidgingConfig",
    "MeltPondConfig",
    "SI3ThermoConfig",
    "validate_si3_bulk_config",
    "validate_si3_thermo_config",
    "IceConstantsConfig",
    "NEMO_SI3_CONSTANTS_CONFIG",
    # Snow physics
    "accumulate_snowfall",
    "combined_conductive_flux",
    "consume_from_snow_then_ice",
    "consume_sublimation_from_snow_then_ice",
    "snow_ice_flooding",
    # Brine physics
    "update_salinity_and_salt_flux",
    "aggregate_salt_flux",
    "SaltBudgetResult",
    # Ridging
    "apply_ridging",
    "apply_si3_jpl1_ridging",
    "participation_weights",
    "SI3JPL1RidgingConfig",
    "SI3JPL1RidgingLosses",
    "SI3JPL1RidgingState",
    # Shortwave
    "compute_ice_sw",
    "delta_eddington_albedo",
    "maykut_untersteiner_albedo",
    "IceSWResult",
    # Ponds
    "step_ponds",
    "build_c1d_omip_l3_card",
    "C1DOMIPL3Card",
]
