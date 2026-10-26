"""Shared zero / wrap / no-op tendency helpers for ocean physics.

Leaf module: depends only on ``core.field`` + ``ocean.state`` / ``ocean.vertical``
/ ``grids.cubed_sphere``.  It does NOT import the per-scheme integration
factories or the combiner, so every integration module and ``combined`` can
import these helpers *directly* at module top level.

This breaks the ``combined`` <-> ``integration`` import cycle that previously
forced each integration module to re-implement thin deferred-import wrappers
(``_zero_tendencies`` / ``_wrap_tendencies``) and a byte-identical no-op factory
(``_make_none``).
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate


def zero_ocean_tendencies(state: OceanState) -> OceanTendencies:
    """Return zero tendencies matching *state* shapes."""
    z3 = jnp.zeros_like(state.u.data)
    z2 = jnp.zeros_like(state.eta.data)
    dims_3d = state.u.dims if hasattr(state.u, 'dims') else ("face", "x", "y", "level")
    dims_2d = state.eta.dims if hasattr(state.eta, 'dims') else ("face", "x", "y")
    return OceanTendencies(
        du_dt=Field(data=z3, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=z3, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=z3, name="dT_dt", dims=dims_3d, units="degC/s"),
        dS_dt=Field(data=z3, name="dS_dt", dims=dims_3d, units="PSU/s"),
        deta_dt=Field(data=z2, name="deta_dt", dims=dims_2d, units="m/s"),
        dH_bathy_dt=Field(data=z2, name="dH_bathy_dt", dims=dims_2d, units="m/s"),
        dland_mask_dt=Field(data=z2, name="dland_mask_dt", dims=dims_2d, units="1/s"),
    )


def wrap_ocean_tendencies(
    du_dt, dv_dt, dT_dt, dS_dt, state: OceanState,
) -> OceanTendencies:
    """Wrap raw tendency arrays into an ``OceanTendencies`` NamedTuple.

    2-D fields (deta_dt, dH_bathy_dt, dland_mask_dt) are set to zero.
    Any of du_dt … dS_dt may be ``None``, in which case the corresponding
    tendency is set to zero.
    """
    z3 = jnp.zeros_like(state.u.data)
    z2 = jnp.zeros_like(state.eta.data)
    dims_3d = state.u.dims if hasattr(state.u, 'dims') else ("face", "x", "y", "level")
    dims_2d = state.eta.dims if hasattr(state.eta, 'dims') else ("face", "x", "y")
    return OceanTendencies(
        du_dt=Field(data=du_dt if du_dt is not None else z3, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt if dv_dt is not None else z3, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt if dT_dt is not None else z3, name="dT_dt", dims=dims_3d, units="degC/s"),
        dS_dt=Field(data=dS_dt if dS_dt is not None else z3, name="dS_dt", dims=dims_3d, units="PSU/s"),
        deta_dt=Field(data=z2, name="deta_dt", dims=dims_2d, units="m/s"),
        dH_bathy_dt=Field(data=z2, name="dH_bathy_dt", dims=dims_2d, units="m/s"),
        dland_mask_dt=Field(data=z2, name="dland_mask_dt", dims=dims_2d, units="1/s"),
    )


def make_none_physics_fn() -> Callable:
    """Factory for a no-op ocean physics function (``scheme="none"``).

    Returns ``physics_fn(state, grid, z_coord, surface_forcing=None, dt=None)`` yielding
    zero tendencies — the single shared implementation that the five per-scheme
    integration factories (bottom_drag, vertical_mixing, convection,
    lateral_mixing, surface_forcing) previously each duplicated byte-for-byte.
    """
    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None, dt=None) -> OceanTendencies:
        return zero_ocean_tendencies(state)
    return physics_fn
