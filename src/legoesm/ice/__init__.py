"""Cryosphere component for legoESM.

Sea ice model with switchable dynamics and multi-category support:
- ``dynamics="none"``: Slab thermodynamics (default, backward compatible).
- ``dynamics="free_drift"``: Free-drift velocity with tracer advection.
- ``dynamics="evp"``: EVP rheology (Hunke & Dukowicz 1997).

Multi-category ice (``n_categories > 1``) uses linear remapping
(Lipscomb 2001) for ice thickness distribution.
"""

from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import (
    SeaIceState,
    DynamicSeaIceState,
    init_dynamic_ice_state,
    dynamic_to_slab,
    slab_to_dynamic,
)
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.rheology import (
    ice_strength,
    strain_rates,
    delta_deformation,
    vp_stress,
    evp_stress_update,
)
from legoesm.ice.dynamics import (
    stress_divergence,
    evp_solver,
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
)
from legoesm.ice.transport import advect_ice_tracers

__all__ = [
    # Config and state
    "SeaIceConfig",
    "SeaIceState",
    "DynamicSeaIceState",
    "init_dynamic_ice_state",
    "dynamic_to_slab",
    "slab_to_dynamic",
    # Main step function
    "step_sea_ice",
    # Rheology
    "ice_strength",
    "strain_rates",
    "delta_deformation",
    "vp_stress",
    "evp_stress_update",
    # Dynamics
    "stress_divergence",
    "evp_solver",
    "free_drift_velocity",
    "air_ice_stress",
    "ocean_ice_stress",
    # ITD
    "category_bounds",
    "upper_bounds",
    "aggregate_state",
    "distribute_to_categories",
    "linear_remap",
    # Transport
    "advect_ice_tracers",
]
