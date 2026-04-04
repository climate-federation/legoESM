"""Factory for ocean surface forcing physics."""

from __future__ import annotations

from typing import Callable

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig


def make_surface_forcing_physics(
    config: SurfaceForcingConfig,
) -> Callable:
    """Create a surface forcing physics function.

    Parameters
    ----------
    config : SurfaceForcingConfig

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    """
    scheme = config.scheme

    if scheme == "none":
        return _make_none()
    elif scheme == "prescribed":
        return _make_prescribed(config)
    elif scheme == "restoring":
        return _make_restoring(config)
    elif scheme == "bulk_formulas":
        return _make_bulk_formulas(config)
    else:
        raise ValueError(f"Unknown surface forcing scheme: {scheme!r}")


def _make_none() -> Callable:
    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        return _zero_tendencies(state)
    return physics_fn


def _make_prescribed(config: SurfaceForcingConfig) -> Callable:
    from legoesm.ocean.physics.surface_forcing.prescribed import prescribed_surface_forcing
    cfg = config.prescribed

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        out = prescribed_surface_forcing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            z_coord, J, grid, cfg,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_restoring(config: SurfaceForcingConfig) -> Callable:
    from legoesm.ocean.physics.surface_forcing.restoring import restoring_surface_forcing
    cfg = config.restoring

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        out = restoring_surface_forcing(state.T.data, state.S.data, grid, cfg)
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_bulk_formulas(config: SurfaceForcingConfig) -> Callable:
    from legoesm.ocean.physics.surface_forcing.bulk_formulas import bulk_formula_surface_forcing
    cfg = config.bulk_formulas

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        out = bulk_formula_surface_forcing(
            state.T.data, state.S.data, z_coord, J, cfg,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


# --- Helpers ---

def _zero_tendencies(state):
    from legoesm.ocean.physics.combined import zero_ocean_tendencies
    return zero_ocean_tendencies(state)


def _wrap_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state):
    from legoesm.ocean.physics.combined import wrap_ocean_tendencies
    return wrap_ocean_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state)
