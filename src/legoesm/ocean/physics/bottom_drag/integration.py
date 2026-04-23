"""Factory for ocean bottom drag physics."""

from __future__ import annotations

from typing import Callable

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig


def make_bottom_drag_physics(
    config: BottomDragConfig,
) -> Callable:
    """Create a bottom drag physics function.

    Parameters
    ----------
    config : BottomDragConfig

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    """
    scheme = config.scheme

    if scheme == "none":
        return _make_none()
    elif scheme == "linear":
        return _make_linear(config)
    elif scheme == "quadratic":
        return _make_quadratic(config)
    else:
        raise ValueError(f"Unknown bottom drag scheme: {scheme!r}")


def _make_none() -> Callable:
    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        return _zero_tendencies(state)
    return physics_fn


def _make_linear(config: BottomDragConfig) -> Callable:
    from legoesm.ocean.physics.bottom_drag.linear import linear_bottom_drag
    cfg = config.linear

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        out = linear_bottom_drag(state.u.data, state.v.data, z_coord, J, cfg)
        return _wrap_tendencies(out.du_dt, out.dv_dt, state)
    return physics_fn


def _make_quadratic(config: BottomDragConfig) -> Callable:
    from legoesm.ocean.physics.bottom_drag.quadratic import quadratic_bottom_drag
    cfg = config.quadratic

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        out = quadratic_bottom_drag(state.u.data, state.v.data, z_coord, J, cfg)
        return _wrap_tendencies(out.du_dt, out.dv_dt, state)
    return physics_fn


# --- Helpers ---

def _zero_tendencies(state):
    from legoesm.ocean.physics.combined import zero_ocean_tendencies
    return zero_ocean_tendencies(state)


def _wrap_tendencies(du_dt, dv_dt, state):
    from legoesm.ocean.physics.combined import wrap_ocean_tendencies
    return wrap_ocean_tendencies(du_dt, dv_dt, None, None, state)
