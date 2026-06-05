"""Factory for ocean convection physics."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import (
    compute_ocean_rho as _compute_rho,
    compute_ocean_rho_and_pressure as _compute_rho_and_pressure,
)
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.convection.enhanced_diffusion import enhanced_diffusion_convection
from legoesm.ocean.physics.convection.plume import plume_convection
from legoesm.ocean.physics.tendencies import make_none_physics_fn, wrap_ocean_tendencies


def make_convection_physics(
    config: OceanConvectionConfig,
    apply_diffusion: bool = True,
) -> Callable:
    """Create an ocean convection physics function.

    Parameters
    ----------
    config : OceanConvectionConfig
    apply_diffusion : bool
        If False, the ``enhanced_diffusion`` scheme returns zero tendency
        but still produces the K_v profile so the dynamics step can apply
        it via an implicit backward-Euler solve (combined with KPP /
        background diffusivities).  The ``plume`` scheme ignores this
        flag (it is not a diffusion).

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    """
    scheme = config.scheme

    if scheme == "none":
        return make_none_physics_fn()
    elif scheme == "enhanced_diffusion":
        return _make_enhanced_diffusion(config, apply_diffusion=apply_diffusion)
    elif scheme == "plume":
        return _make_plume(config)
    else:
        raise ValueError(f"Unknown ocean convection scheme: {scheme!r}")


def _make_enhanced_diffusion(config: OceanConvectionConfig,
                             apply_diffusion: bool = True) -> Callable:
    cfg = config.enhanced_diffusion

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        rho = _compute_rho(state, z_coord, J)
        out = enhanced_diffusion_convection(
            state.T.data, state.S.data, rho, z_coord, J, cfg,
            apply_diffusion=apply_diffusion,
        )
        z3 = jnp.zeros_like(state.u.data)
        t = wrap_ocean_tendencies(z3, z3, out.dT_dt, out.dS_dt, state)
        # When implicit, pass convection K_v through for downstream
        # use by the tridiagonal solve (avoids re-running EOS/N² in
        # compute_vertical_K_profiles).
        if not apply_diffusion and out.K_v is not None:
            t = t._replace(K_v=out.K_v)
        return t
    return physics_fn


def _make_plume(config: OceanConvectionConfig) -> Callable:
    cfg = config.plume

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        rho, p_hydro = _compute_rho_and_pressure(state, z_coord, J)
        out = plume_convection(
            state.T.data, state.S.data, rho, p_hydro, z_coord, J, cfg,
        )
        z3 = jnp.zeros_like(state.u.data)
        return wrap_ocean_tendencies(z3, z3, out.dT_dt, out.dS_dt, state)
    return physics_fn



