"""Surface boundary data **consumer**: build per-column land-model inputs.

The runtime loader :mod:`legoesm.land.global_surface_data` reads the harmonized
``legoesm_surfdata`` NetCDF (produced offline by
:mod:`legoesm.land.surface_data`) and regrids it once to the model grid,
returning a :class:`~legoesm.land.global_surface_data.GlobalSurfaceData`.  This
subpackage is the next layer: it turns that ``GlobalSurfaceData`` into the
per-column arrays the land step function consumes
(:class:`~legoesm.land.canopy.config.CanopyLandParams` or
:class:`~legoesm.land.surface_params.LandSurfaceParams`).

By analogy with ``ocean/forcing/`` (time-varying atmospheric drivers) and
``ocean/bathymetry.py`` (static boundary geometry), this package owns the
static / quasi-static **land boundary data** consumer path.  It is NOT
time-varying atmospheric forcing — the monthly LAI cycle is a slow-varying
boundary climatology.

Modules:
  - :mod:`~legoesm.land.boundary_data.builders` — host-side builders that turn a
    :class:`GlobalSurfaceData` into ``CanopyLandParams`` /
    ``LandSurfaceParams`` (incl. the simulation-start entry
    :func:`init_land_surface_data` and the soil-hydraulics Cosby pedotransfer).
  - :mod:`~legoesm.land.boundary_data.gap_fill` — reconciliation with the
    driver's authoritative land mask (bare-soil fallback for uncovered cells).
  - :mod:`~legoesm.land.boundary_data.step_updater` — JAX-native
    ``(theta_top, doy) -> (land_params, lai_col)`` updater for ``lax.scan``
    time loops.
  - :mod:`~legoesm.land.boundary_data.point` — single-point extractor for
    offline / fluxtower runs.
"""

from __future__ import annotations

from legoesm.land.boundary_data.builders import (
    dominant_pft_index,
    glacier_mask,
    build_canopy_params,
    build_soil_hydraulics,
    cover_fracs,
    SurfaceDataParamProvider,
    surface_data_param_provider,
    surface_data_to_land_params,
    prescribed_canopy_structure,
    init_land_surface_data,
)
from legoesm.land.boundary_data.gap_fill import (
    surfdata_covered,
    fill_land_param_gaps,
)
from legoesm.land.boundary_data.step_updater import (
    make_step_land_params_updater,
    CanopyUpdaterInputs, precompute_canopy_updater, apply_canopy_updater,
)
from legoesm.land.boundary_data.point import (
    PointSurfaceParams,
    surface_params_at_point,
)

__all__ = [
    # builders
    "cover_fracs", "dominant_pft_index", "glacier_mask",
    "build_canopy_params", "build_soil_hydraulics",
    "SurfaceDataParamProvider", "surface_data_param_provider",
    "surface_data_to_land_params", "prescribed_canopy_structure",
    "init_land_surface_data",
    # gap fill
    "surfdata_covered", "fill_land_param_gaps",
    # step updater (lax.scan)
    "make_step_land_params_updater",
    "CanopyUpdaterInputs", "precompute_canopy_updater", "apply_canopy_updater",
    # single-point
    "PointSurfaceParams", "surface_params_at_point",
]
