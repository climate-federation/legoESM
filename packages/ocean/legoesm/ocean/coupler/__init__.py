"""Ocean-coupler surface-flux applicators.

The legoESM ocean dycores expose generic prognostic state containers
(``OceanState`` cube, ``LatLonCGridOceanState``, ``MPASOceanState``);
this module ports atmospheric / land flux output into those state
containers as explicit per-timestep top-layer tendencies. The Phase F
climate-scale drivers use ``omip2_applicator`` to wire JRA55-do
forcing + L&Y 2009 bulk fluxes into the ocean state.
"""

from .omip2_applicator import (
    apply_omip2_surface_fluxes,
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
    "apply_omip2_surface_fluxes",
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
