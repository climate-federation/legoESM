"""Backward-compatible alias for shallow water FV model on the cubed-sphere.

Wraps the C-D grid implementation, providing adapters that accept/return
the generic ShallowWaterState (Field-based) used by the test suite and
Williamson test cases.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState, ShallowWaterTendencies
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterModel,
    CDGridShallowWaterConfig,
    CDGridShallowWaterState,
    cdgrid_shallow_water_tendencies,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


# ==============================================================================
# Config alias
# ==============================================================================

class FVShallowWaterConfig(NamedTuple):
    """Configuration for the FV shallow water model (cubed-sphere)."""
    g: float = constants.g
    A_h: float = 0.0
    hyperdiff_coeff: float = 0.0
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    time_integrator: str = "ssp_rk3"


# ==============================================================================
# State conversion helpers
# ==============================================================================

def _sw_to_cdgrid(state: ShallowWaterState, cdgrid: CubedSphereCDGrid) -> CDGridShallowWaterState:
    """Convert generic ShallowWaterState (Field) to CDGridShallowWaterState (raw arrays).

    The generic state has A-grid velocities at cell centres (6, n, n).
    The CDGrid state needs D-grid velocities at cell corners (6, n+1, n+1).
    We interpolate from centres to corners.
    """
    h = state.h.data
    u_center = state.u.data
    v_center = state.v.data
    h_s = state.h_s.data

    # Interpolate cell-centre velocities to cell corners via averaging neighbors
    # Pad with halo for boundary info
    from legoesm.grids.halo import pad_halo, pad_halo_vector
    u_pad, v_pad = pad_halo_vector(
        u_center, v_center,
        cdgrid.base.cos_angle, cdgrid.base.sin_angle,
        cdgrid.base.cos_angle_padded, cdgrid.base.sin_angle_padded,
        interp_offsets=cdgrid.base.halo_interp_offsets,
    )

    # Corners are at intersections of 4 cells: average the 4 surrounding centres
    # padded has shape (6, n+2, n+2); corners at positions [0..n, 0..n] in padded = [i, j]
    u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1] +
                   u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
    v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1] +
                   v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])

    return CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


def _cdgrid_to_sw(cdstate: CDGridShallowWaterState, dims: tuple) -> ShallowWaterState:
    """Convert CDGridShallowWaterState back to generic ShallowWaterState.

    Interpolates D-grid corner velocities back to A-grid cell centres.
    """
    h = cdstate.h
    u_d = cdstate.u_d
    v_d = cdstate.v_d
    h_s = cdstate.h_s

    # Average corners to centres (4-point average)
    u_center = 0.25 * (u_d[:, :-1, :-1] + u_d[:, 1:, :-1] +
                        u_d[:, :-1, 1:] + u_d[:, 1:, 1:])
    v_center = 0.25 * (v_d[:, :-1, :-1] + v_d[:, 1:, :-1] +
                        v_d[:, :-1, 1:] + v_d[:, 1:, 1:])

    return ShallowWaterState(
        h=Field(data=h, name="h", dims=dims, units="m"),
        u=Field(data=u_center, name="u", dims=dims, units="m/s"),
        v=Field(data=v_center, name="v", dims=dims, units="m/s"),
        h_s=Field(data=h_s, name="h_s", dims=dims, units="m"),
    )


# ==============================================================================
# Wrapped tendency function
# ==============================================================================

def fv_shallow_water_tendencies(
    state: ShallowWaterState,
    grid: CubedSphereGrid,
    config: FVShallowWaterConfig | None = None,
) -> ShallowWaterTendencies:
    """Compute FV shallow water tendencies from a generic ShallowWaterState.

    Wraps the CDGrid implementation, handling state conversion.
    """
    if config is None:
        config = FVShallowWaterConfig()

    cdgrid = create_cubed_sphere_cdgrid(grid)
    cd_config = CDGridShallowWaterConfig(
        g=config.g, A_h=config.A_h,
        hyperdiff_coeff=config.hyperdiff_coeff,
    )

    cd_state = _sw_to_cdgrid(state, cdgrid)
    dh_dt, du_d_dt, dv_d_dt = cdgrid_shallow_water_tendencies(cd_state, cdgrid, cd_config)

    # Convert D-grid tendency back to A-grid
    du_dt = 0.25 * (du_d_dt[:, :-1, :-1] + du_d_dt[:, 1:, :-1] +
                     du_d_dt[:, :-1, 1:] + du_d_dt[:, 1:, 1:])
    dv_dt = 0.25 * (dv_d_dt[:, :-1, :-1] + dv_d_dt[:, 1:, :-1] +
                     dv_d_dt[:, :-1, 1:] + dv_d_dt[:, 1:, 1:])

    dims = state.h.dims

    return ShallowWaterTendencies(
        dh_dt=Field(data=dh_dt, name="dh_dt", dims=dims, units="m/s"),
        du_dt=Field(data=du_dt, name="du_dt", dims=dims, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims, units="m/s^2"),
    )


# ==============================================================================
# Model class
# ==============================================================================

class FVShallowWaterModel(IntegrationMixin):
    """FV shallow water model wrapping the C-D grid implementation.

    Accepts and returns generic ShallowWaterState (with Field objects).
    """

    def __init__(self, grid: CubedSphereGrid, config: FVShallowWaterConfig | None = None):
        self.grid = grid
        self.config = config or FVShallowWaterConfig()
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self.cd_config = CDGridShallowWaterConfig(
            g=self.config.g,
            A_h=self.config.A_h,
            hyperdiff_coeff=self.config.hyperdiff_coeff,
            use_conservation_fixer=self.config.use_conservation_fixer,
            fix_mass=self.config.fix_mass,
            time_integrator=self.config.time_integrator,
        )
        # CDGridShallowWaterModel expects the BASE grid, not the CDGrid
        self._cd_model = CDGridShallowWaterModel(grid, self.cd_config)

    def tendencies(self, state: ShallowWaterState) -> ShallowWaterTendencies:
        return fv_shallow_water_tendencies(state, self.grid, self.config)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ShallowWaterState, dt: float) -> ShallowWaterState:
        """Advance one time step, preserving Field metadata for scan compatibility."""
        cd_state = _sw_to_cdgrid(state, self.cdgrid)
        cd_new = self._cd_model.step(cd_state, dt)

        # Preserve metadata from input state (critical for jax.lax.scan)
        h_new = cd_new.h
        u_center = 0.25 * (cd_new.u_d[:, :-1, :-1] + cd_new.u_d[:, 1:, :-1] +
                            cd_new.u_d[:, :-1, 1:] + cd_new.u_d[:, 1:, 1:])
        v_center = 0.25 * (cd_new.v_d[:, :-1, :-1] + cd_new.v_d[:, 1:, :-1] +
                            cd_new.v_d[:, :-1, 1:] + cd_new.v_d[:, 1:, 1:])

        return ShallowWaterState(
            h=state.h.replace(data=h_new),
            u=state.u.replace(data=u_center),
            v=state.v.replace(data=v_center),
            h_s=state.h_s.replace(data=cd_new.h_s),
        )


# Alias for backward compatibility
FVShallowWaterState = CDGridShallowWaterState

__all__ = [
    "FVShallowWaterModel",
    "FVShallowWaterConfig",
    "FVShallowWaterState",
    "fv_shallow_water_tendencies",
]
