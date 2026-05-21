"""Ocean-coupler surface-flux applicators.

The legoESM ocean dycores expose generic prognostic state containers
(``OceanState`` cube, ``LatLonCGridOceanState``, ``MPASOceanState``);
this module ports atmospheric / land flux output into those state
containers as explicit per-timestep top-layer tendencies. The Phase F
climate-scale drivers use ``omip2_applicator`` to wire JRA55-do
forcing + L&Y 2009 bulk fluxes into the ocean state.
"""

from .omip2_applicator import apply_omip2_surface_fluxes
from .sss_apply import apply_sss_restoring_step, apply_sss_restoring_step_mpas
from .runoff_apply import apply_runoff_step, apply_runoff_step_mpas
from .ice_shelf_apply import apply_ice_shelf_basal_step
from .tidal_mixing_apply import apply_tidal_mixing_step

__all__ = [
    "apply_omip2_surface_fluxes",
    "apply_sss_restoring_step",
    "apply_sss_restoring_step_mpas",
    "apply_runoff_step",
    "apply_runoff_step_mpas",
    "apply_ice_shelf_basal_step",
    "apply_tidal_mixing_step",
]
