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

from legoesm.ice.config import (
    SeaIceConfig,
    SnowConfig,
    BrineConfig,
    RidgingConfig,
    MeltPondConfig,
)
from legoesm.ice.state import (
    SeaIceState,
    DynamicSeaIceState,
    init_dynamic_ice_state,
    distribute_dynamic_state_to_categories,
    dynamic_to_slab,
    slab_to_dynamic,
)
from legoesm.ice.sea_ice import (
    step_sea_ice,
    grid_supports_ice_dynamics,
    grid_supports_ice_transport,
    uses_new_physics,
)
from legoesm.ice.rheology import (
    ice_strength,
    strain_rates,
    delta_deformation,
    vp_stress,
    evp_stress_update,
    mevp_stress_update,
)
from legoesm.ice.dynamics import (
    stress_divergence,
    evp_solver,
    mevp_solver,
    free_drift_velocity,
    air_ice_stress,
    ocean_ice_stress,
)
from legoesm.ice.itd import (
    category_bounds,
    upper_bounds,
    aggregate_state,
    distribute_to_categories,
    linear_remap,
    lipscomb_2001_remap,
)
from legoesm.ice.transport import advect_ice_tracers
from legoesm.ice.snow import (
    accumulate_snowfall,
    combined_conductive_flux,
    consume_from_snow_then_ice,
    consume_sublimation_from_snow_then_ice,
    snow_ice_flooding,
)
from legoesm.ice.brine import (
    update_salinity_and_salt_flux,
    aggregate_salt_flux,
    SaltBudgetResult,
)
from legoesm.ice.ridging import apply_ridging, participation_weights
from legoesm.ice.shortwave import (
    compute_ice_sw,
    delta_eddington_albedo,
    maykut_untersteiner_albedo,
    IceSWResult,
)
from legoesm.ice.ponds import step_ponds

__all__ = [
    # Config and state
    "SeaIceConfig",
    "SeaIceState",
    "DynamicSeaIceState",
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
    "participation_weights",
    # Shortwave
    "compute_ice_sw",
    "delta_eddington_albedo",
    "maykut_untersteiner_albedo",
    "IceSWResult",
    # Ponds
    "step_ponds",
]
