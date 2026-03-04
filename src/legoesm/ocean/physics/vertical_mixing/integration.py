"""Factory for ocean vertical mixing physics."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import wright_eos, compute_hydrostatic_pressure, rho_0 as rho_0_ref
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig


def make_vertical_mixing_physics(
    config: VerticalMixingConfig,
) -> Callable:
    """Create a vertical mixing physics function.

    Parameters
    ----------
    config : VerticalMixingConfig

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    """
    scheme = config.scheme

    if scheme == "none":
        return _make_none()
    elif scheme == "constant":
        return _make_constant(config)
    elif scheme == "richardson":
        return _make_richardson(config)
    elif scheme == "kpp":
        return _make_kpp(config)
    else:
        raise ValueError(f"Unknown vertical mixing scheme: {scheme!r}")


def _make_none() -> Callable:
    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate) -> OceanTendencies:
        return _zero_tendencies(state)
    return physics_fn


def _make_constant(config: VerticalMixingConfig) -> Callable:
    from legoesm.ocean.physics.vertical_mixing.constant import constant_vertical_mixing
    cfg = config.constant

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        out = constant_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            z_coord, J, cfg,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_richardson(config: VerticalMixingConfig) -> Callable:
    from legoesm.ocean.physics.vertical_mixing.richardson import richardson_vertical_mixing
    cfg = config.richardson

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        rho = _compute_rho(state, z_coord, J)
        out = richardson_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, z_coord, J, cfg,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_kpp(config: VerticalMixingConfig) -> Callable:
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    cfg = config.kpp

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        rho = _compute_rho(state, z_coord, J)
        out = kpp_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, state.eta.data, z_coord, J, cfg,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


# --- Helpers ---

def _compute_rho(state, z_coord, J, g=9.80616):
    """Compute in-situ density from state."""
    p_hydro = compute_hydrostatic_pressure(
        jnp.full_like(state.T.data, rho_0_ref),
        state.eta.data, z_coord.dz_ref, J, rho_0_ref, g,
    )
    return wright_eos(state.T.data, state.S.data, p_hydro)


def _zero_tendencies(state: OceanState) -> OceanTendencies:
    """Return zero tendencies matching state shapes."""
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


def _wrap_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state):
    """Wrap raw tendency arrays into OceanTendencies."""
    z2 = jnp.zeros_like(state.eta.data)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return OceanTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt, name="dT_dt", dims=dims_3d, units="degC/s"),
        dS_dt=Field(data=dS_dt, name="dS_dt", dims=dims_3d, units="PSU/s"),
        deta_dt=Field(data=z2, name="deta_dt", dims=dims_2d, units="m/s"),
        dH_bathy_dt=Field(data=z2, name="dH_bathy_dt", dims=dims_2d, units="m/s"),
        dland_mask_dt=Field(data=z2, name="dland_mask_dt", dims=dims_2d, units="1/s"),
    )
