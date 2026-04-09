"""Combined ocean physics orchestrator.

Provides OceanPhysicsConfig and make_ocean_physics(), which create a
single physics function combining vertical mixing, lateral mixing,
surface forcing, bottom drag, and convection.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.state import OceanState, OceanSurfaceForcing, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate

from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig,
    shortwave_penetration_tendency,
)

from legoesm.ocean.physics.vertical_mixing.integration import make_vertical_mixing_physics
from legoesm.ocean.physics.lateral_mixing.integration import make_lateral_mixing_physics
from legoesm.ocean.physics.surface_forcing.integration import make_surface_forcing_physics
from legoesm.ocean.physics.bottom_drag.integration import make_bottom_drag_physics
from legoesm.ocean.physics.convection.integration import make_convection_physics


class OceanPhysicsConfig(NamedTuple):
    """Unified ocean physics configuration.

    Set the ``scheme`` field of any sub-config to ``"none"`` to disable
    that module entirely.
    """
    vertical_mixing: VerticalMixingConfig = VerticalMixingConfig()
    lateral_mixing: LateralMixingConfig = LateralMixingConfig()
    surface_forcing: SurfaceForcingConfig = SurfaceForcingConfig()
    bottom_drag: BottomDragConfig = BottomDragConfig()
    convection: OceanConvectionConfig = OceanConvectionConfig()
    shortwave_penetration: ShortwavePenetrationConfig | None = ShortwavePenetrationConfig()


def make_ocean_physics(config: OceanPhysicsConfig) -> Callable:
    """Create a combined ocean physics function.

    The returned function calls each enabled physics module and sums
    their tendencies.

    Parameters
    ----------
    config : OceanPhysicsConfig

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord, surface_forcing=None) -> OceanTendencies
    """
    fns = []

    if config.vertical_mixing.scheme != "none":
        fns.append(make_vertical_mixing_physics(config.vertical_mixing))
    if config.lateral_mixing.scheme != "none":
        fns.append(make_lateral_mixing_physics(config.lateral_mixing))
    if config.surface_forcing.scheme != "none":
        fns.append(make_surface_forcing_physics(config.surface_forcing))
    if config.bottom_drag.scheme != "none":
        fns.append(make_bottom_drag_physics(config.bottom_drag))
    if config.convection.scheme != "none":
        fns.append(make_convection_physics(config.convection))

    sw_config = config.shortwave_penetration

    def physics_fn(
        state: OceanState,
        grid: CubedSphereGrid,
        z_coord: OceanZStarCoordinate,
        surface_forcing: OceanSurfaceForcing | None = None,
    ) -> OceanTendencies:
        if not fns and sw_config is None:
            return _zero_tendencies(state)

        # Sum tendencies from all enabled sub-physics modules.
        if fns:
            first = fns[0](state, grid, z_coord, surface_forcing)
            du_dt = first.du_dt.data
            dv_dt = first.dv_dt.data
            dT_dt = first.dT_dt.data
            dS_dt = first.dS_dt.data
            deta_dt = first.deta_dt.data

            for fn in fns[1:]:
                t = fn(state, grid, z_coord, surface_forcing)
                du_dt = du_dt + t.du_dt.data
                dv_dt = dv_dt + t.dv_dt.data
                dT_dt = dT_dt + t.dT_dt.data
                dS_dt = dS_dt + t.dS_dt.data
                deta_dt = deta_dt + t.deta_dt.data
        else:
            z3 = jnp.zeros_like(state.u.data)
            z2 = jnp.zeros_like(state.eta.data)
            du_dt, dv_dt, dT_dt, dS_dt, deta_dt = z3, z3, z3, z3, z2

        # Shortwave penetration: distribute SW heating through water column.
        if (
            sw_config is not None
            and surface_forcing is not None
            and surface_forcing.sw_down is not None
        ):
            from legoesm.ocean.vertical import compute_ocean_jacobian
            J = compute_ocean_jacobian(
                state.eta.data, state.H_bathy.data, z_coord,
            )
            sw_tend = shortwave_penetration_tendency(
                surface_forcing.sw_down,
                z_coord.dz_ref,
                z_coord.z_half_ref,
                J,
                sw_config,
            )
            dT_dt = dT_dt + sw_tend

        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")
        return OceanTendencies(
            du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
            dT_dt=Field(data=dT_dt, name="dT_dt", dims=dims_3d, units="degC/s"),
            dS_dt=Field(data=dS_dt, name="dS_dt", dims=dims_3d, units="PSU/s"),
            deta_dt=Field(data=deta_dt, name="deta_dt", dims=dims_2d, units="m/s"),
            dH_bathy_dt=Field(
                data=jnp.zeros_like(state.eta.data),
                name="dH_bathy_dt", dims=dims_2d, units="m/s",
            ),
            dland_mask_dt=Field(
                data=jnp.zeros_like(state.eta.data),
                name="dland_mask_dt", dims=dims_2d, units="1/s",
            ),
        )

    return physics_fn


def _zero_tendencies(state: OceanState) -> OceanTendencies:
    return zero_ocean_tendencies(state)


# ---------------------------------------------------------------------------
# Shared tendency helpers (used by all ocean physics integration modules)
# ---------------------------------------------------------------------------

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
