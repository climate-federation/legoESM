"""Factory for ocean lateral mixing physics."""

from __future__ import annotations

from typing import Callable

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import compute_ocean_rho as _compute_rho
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig


def make_lateral_mixing_physics(
    config: LateralMixingConfig,
) -> Callable:
    """Create a lateral mixing physics function.

    Parameters
    ----------
    config : LateralMixingConfig

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    """
    scheme = config.scheme

    if scheme == "none":
        return _make_none()
    elif scheme == "harmonic":
        return _make_harmonic(config)
    elif scheme == "biharmonic":
        return _make_biharmonic(config)
    elif scheme == "gm_redi":
        return _make_gm_redi(config)
    else:
        raise ValueError(f"Unknown lateral mixing scheme: {scheme!r}")


def _make_none() -> Callable:
    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        return _zero_tendencies(state)
    return physics_fn


def _make_harmonic(config: LateralMixingConfig) -> Callable:
    from legoesm.ocean.physics.lateral_mixing.harmonic import harmonic_lateral_mixing
    cfg = config.harmonic

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        out = harmonic_lateral_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            state.land_mask.data, grid, cfg,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_biharmonic(config: LateralMixingConfig) -> Callable:
    from legoesm.ocean.physics.lateral_mixing.biharmonic import biharmonic_lateral_mixing
    cfg = config.biharmonic

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        out = biharmonic_lateral_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            state.land_mask.data, grid, cfg,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_gm_redi(config: LateralMixingConfig) -> Callable:
    from legoesm.ocean.physics.lateral_mixing.gm_redi import gm_redi_lateral_mixing
    cfg = config.gm_redi

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        rho = _compute_rho(state, z_coord, J)
        out = gm_redi_lateral_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, z_coord, J, grid, cfg,
        )
        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn



def _zero_tendencies(state):
    from legoesm.ocean.physics.combined import zero_ocean_tendencies
    return zero_ocean_tendencies(state)


def _wrap_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state):
    from legoesm.ocean.physics.combined import wrap_ocean_tendencies
    return wrap_ocean_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state)
