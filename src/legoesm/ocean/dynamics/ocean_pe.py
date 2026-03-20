"""Deprecated alias for ocean primitive equation model.

.. deprecated::
    Import ``ocean_baroclinic_tendencies_cdgrid`` from
    ``legoesm.ocean.dynamics.ocean_pe_cdgrid`` instead.

This module wraps the C-D grid implementation, automatically creating the
CDGrid from the base CubedSphereGrid so callers can use the old 4-argument
signature: ``ocean_baroclinic_tendencies(state, grid, z_coord, config)``.
"""

from __future__ import annotations

import warnings as _warnings

from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
    ocean_baroclinic_tendencies_cdgrid,
    _vertical_advection_ocean,
    OceanState,
    OceanTendencies,
    OceanConfig,
)
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def ocean_baroclinic_tendencies(
    state: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    config: OceanConfig = OceanConfig(),
    physics_fn=None,
) -> OceanTendencies:
    """Compute 3D baroclinic ocean tendencies.

    .. deprecated::
        Use ``ocean_baroclinic_tendencies_cdgrid`` from
        ``legoesm.ocean.dynamics.ocean_pe_cdgrid`` instead,
        passing the CDGrid explicitly.

    Backward-compatible wrapper that auto-creates the CDGrid from the
    base grid.
    """
    _warnings.warn(
        "ocean_baroclinic_tendencies (from ocean_pe) is deprecated; "
        "use ocean_baroclinic_tendencies_cdgrid from "
        "legoesm.ocean.dynamics.ocean_pe_cdgrid instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    cdgrid = create_cubed_sphere_cdgrid(grid)
    return ocean_baroclinic_tendencies_cdgrid(
        state, grid, z_coord, cdgrid, config, physics_fn,
    )


__all__ = [
    "ocean_baroclinic_tendencies",
    "_vertical_advection_ocean",
]
