"""Ocean-coupler surface-flux applicators.

The legoESM ocean dycores expose generic prognostic state containers
(``OceanState`` cube, ``LatLonCGridOceanState``, ``MPASOceanState``);
this module ports atmospheric / land flux output into those state
containers. ``omip2_applicator`` builds the JRA55-do / CORE-II bulk-flux
``OceanSurfaceForcing`` that ``model.step(surface_forcing=...)`` integrates
inside the timestep.
"""

from .omip2_applicator import (
    compute_omip2_surface_forcing,
    compute_omip2_freshwater_forcing,
    sample_omip2_forcing,
)
from .sss_apply import (
    apply_sss_restoring_step,
    apply_sss_restoring_step_fesom,
    apply_sss_restoring_step_mpas,
)
from .runoff_apply import apply_runoff_step, apply_runoff_step_mpas
from .ice_shelf_apply import (
    apply_ice_shelf_basal_step,
    apply_ice_shelf_basal_step_mpas,
)
from .tidal_mixing_apply import apply_tidal_mixing_step
from .geothermal_apply import apply_geothermal_step

__all__ = [
    "compute_omip2_surface_forcing",
    "compute_omip2_freshwater_forcing",
    "sample_omip2_forcing",
    "apply_sss_restoring_step",
    "apply_sss_restoring_step_fesom",
    "apply_sss_restoring_step_mpas",
    "apply_runoff_step",
    "apply_runoff_step_mpas",
    "apply_ice_shelf_basal_step",
    "apply_ice_shelf_basal_step_mpas",
    "apply_tidal_mixing_step",
    "apply_geothermal_step",
]
