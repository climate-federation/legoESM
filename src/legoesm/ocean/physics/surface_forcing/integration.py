"""Factory for ocean surface forcing physics."""

from __future__ import annotations

from typing import Callable

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.surface_forcing.prescribed import prescribed_surface_forcing
from legoesm.ocean.physics.surface_forcing.external import external_surface_forcing
from legoesm.ocean.physics.surface_forcing.restoring import restoring_surface_forcing
from legoesm.ocean.physics.surface_forcing.bulk_formulas import bulk_formula_surface_forcing


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
    elif scheme == "combined":
        return _make_combined(config)
    elif scheme == "bulk_formulas":
        return _make_bulk_formulas(config)
    elif scheme == "external":
        return _make_external(config)
    else:
        raise ValueError(f"Unknown surface forcing scheme: {scheme!r}")


def _make_none() -> Callable:
    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        return _zero_tendencies(state)
    return physics_fn


def _make_prescribed(config: SurfaceForcingConfig) -> Callable:
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


def _make_external(config: SurfaceForcingConfig) -> Callable:
    """Coupler-provided surface forcing: apply the passed OceanSurfaceForcing
    (tau / q_net / freshwater / salt) through the physics path (cubed-sphere /
    MPAS two-way coupling)."""
    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        if surface_forcing is None:
            return _zero_tendencies(state)
        # Partial-cell-aware ACTUAL top-layer thickness (not dz_ref[0]*J) so the
        # flux-to-tendency conversion is conservative on shallow top cells.
        h = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord)
        out = external_surface_forcing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            h[..., 0], surface_forcing,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_restoring(config: SurfaceForcingConfig) -> Callable:
    cfg = config.restoring

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        out = restoring_surface_forcing(state.T.data, state.S.data, grid, cfg)
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_combined(config: SurfaceForcingConfig) -> Callable:
    """Prescribed wind stress + temperature/salinity restoring."""
    cfg_p = config.prescribed
    cfg_r = config.restoring

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        p = prescribed_surface_forcing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            z_coord, J, grid, cfg_p,
        )
        r = restoring_surface_forcing(state.T.data, state.S.data, grid, cfg_r)
        return _wrap_tendencies(
            p.du_dt + r.du_dt,
            p.dv_dt + r.dv_dt,
            p.dT_dt + r.dT_dt,
            p.dS_dt + r.dS_dt,
            state,
        )
    return physics_fn


def _make_bulk_formulas(config: SurfaceForcingConfig) -> Callable:
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
