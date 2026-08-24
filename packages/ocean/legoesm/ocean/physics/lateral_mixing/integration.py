"""Factory for ocean lateral mixing physics."""

from __future__ import annotations

from typing import Callable

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.eos import compute_ocean_rho as _compute_rho
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.lateral_mixing.harmonic import harmonic_lateral_mixing
from legoesm.ocean.physics.lateral_mixing.biharmonic import biharmonic_lateral_mixing
from legoesm.ocean.physics.lateral_mixing.gm_redi import (
    gm_redi_lateral_mixing,
    validate_cubed_sphere_gm_redi_config,
)
from legoesm.ocean.physics.tendencies import make_none_physics_fn, wrap_ocean_tendencies


def make_lateral_mixing_physics(
    config: LateralMixingConfig,
    constants_config: ConstantsConfig = ConstantsConfig(),
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
        return make_none_physics_fn()
    elif scheme == "harmonic":
        return _make_harmonic(config)
    elif scheme == "biharmonic":
        return _make_biharmonic(config)
    elif scheme == "gm_redi":
        return _make_gm_redi(config, constants_config=constants_config)
    else:
        raise ValueError(f"Unknown lateral mixing scheme: {scheme!r}")


def _make_harmonic(config: LateralMixingConfig) -> Callable:
    cfg = config.harmonic

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        out = harmonic_lateral_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            state.land_mask.data, grid, cfg,
        )
        return wrap_ocean_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_biharmonic(config: LateralMixingConfig) -> Callable:
    cfg = config.biharmonic

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        out = biharmonic_lateral_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            state.land_mask.data, grid, cfg,
        )
        return wrap_ocean_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn


def _make_gm_redi(config: LateralMixingConfig,
                  constants_config: ConstantsConfig = ConstantsConfig()) -> Callable:
    """Create GM/Redi physics function (cubed-sphere only).

    The lat-lon C-grid ocean model bypasses this factory and calls
    ``gm_redi_latlon_cgrid.gm_redi_tracer_tendency_latlon`` directly
    from ``ocean_model_latlon_cgrid.py``.  Unifying the factory to
    support both grids is tracked as a known gap (see
    ``docs/ocean/experiments/gm_redi_latlon_cgrid_plan.md``).
    """
    cfg = config.gm_redi

    # Fail fast at factory-build time if the config sets a field the
    # cubed-sphere GM/Redi leaf does not honor (those are lat-lon C-grid
    # only).  Validated here on the static Python config so the error is
    # surfaced before the JIT-traced physics_fn ever runs.
    validate_cubed_sphere_gm_redi_config(cfg)

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        if not isinstance(grid, CubedSphereGrid):
            raise TypeError(
                "Factory GM/Redi only supports CubedSphereGrid. "
                "Lat-lon C-grid uses gm_redi_latlon_cgrid directly "
                "(see ocean_model_latlon_cgrid.py)."
            )
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        rho = _compute_rho(state, z_coord, J, g=constants_config.g,
                           rho0=constants_config.rho_0)
        out = gm_redi_lateral_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, z_coord, J, grid, cfg,
        )
        return wrap_ocean_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    return physics_fn



