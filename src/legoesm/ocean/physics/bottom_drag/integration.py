"""Factory for ocean bottom drag physics."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.core.field import Field
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
                   z_coord: OceanZStarCoordinate) -> OceanTendencies:
        return _zero_tendencies(state)
    return physics_fn


def _make_linear(config: BottomDragConfig) -> Callable:
    from legoesm.ocean.physics.bottom_drag.linear import linear_bottom_drag
    cfg = config.linear

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate) -> OceanTendencies:
        out = linear_bottom_drag(state.u.data, state.v.data, cfg)
        return _wrap_tendencies(out.du_dt, out.dv_dt, state)
    return physics_fn


def _make_quadratic(config: BottomDragConfig) -> Callable:
    from legoesm.ocean.physics.bottom_drag.quadratic import quadratic_bottom_drag
    cfg = config.quadratic

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        out = quadratic_bottom_drag(state.u.data, state.v.data, z_coord, J, cfg)
        return _wrap_tendencies(out.du_dt, out.dv_dt, state)
    return physics_fn


# --- Helpers ---

def _zero_tendencies(state: OceanState) -> OceanTendencies:
    z3 = jnp.zeros_like(state.u.data)
    z2 = jnp.zeros_like(state.eta.data)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return OceanTendencies(
        du_dt=Field(data=z3, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=z3, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=z3, name="dT_dt", dims=dims_3d, units="degC/s"),
        dS_dt=Field(data=z3, name="dS_dt", dims=dims_3d, units="PSU/s"),
        deta_dt=Field(data=z2, name="deta_dt", dims=dims_2d, units="m/s"),
        dH_bathy_dt=Field(data=z2, name="dH_bathy_dt", dims=dims_2d, units="m/s"),
        dland_mask_dt=Field(data=z2, name="dland_mask_dt", dims=dims_2d, units="1/s"),
    )


def _wrap_tendencies(du_dt, dv_dt, state):
    z3 = jnp.zeros_like(state.u.data)
    z2 = jnp.zeros_like(state.eta.data)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return OceanTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=z3, name="dT_dt", dims=dims_3d, units="degC/s"),
        dS_dt=Field(data=z3, name="dS_dt", dims=dims_3d, units="PSU/s"),
        deta_dt=Field(data=z2, name="deta_dt", dims=dims_2d, units="m/s"),
        dH_bathy_dt=Field(data=z2, name="dH_bathy_dt", dims=dims_2d, units="m/s"),
        dland_mask_dt=Field(data=z2, name="dland_mask_dt", dims=dims_2d, units="1/s"),
    )
