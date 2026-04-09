"""Prescribed-wind tracer transport on the lat-lon grid.

Advects an arbitrary number of passive tracers using analytically
prescribed wind fields. The transport equation in sigma coordinates:

    dq_i/dt = -(u dq_i/dx + v dq_i/dy) - sigma_dot dq_i/dsigma + D(q_i)

where D is optional hyperdiffusion for numerical stability.

This model mirrors the cubed-sphere TracerTransportModel but operates
on LatLonGrid fields with shape (n_lat, n_lon, nlev[, n_tracers]).

Time is embedded in the TracerState pytree so that SSP-RK3 evaluates
the prescribed wind function at the correct intermediate times for
each Runge-Kutta stage.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import TracerState
from legoesm.core.operators_latlon_3d import (
    gradient_x_3d,
    gradient_y_3d,
    hyperdiffusion_3d,
)
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.vertical import SigmaCoordinate, vertical_advection
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin


# Type alias for prescribed wind functions.
# Signature: (time, grid, sigma_coord) -> (u, v, sigma_dot)
#   u, v:        shape (n_lat, n_lon, nlev) — geographic wind components
#   sigma_dot:   shape (n_lat, n_lon, nlev+1) — vertical velocity at interfaces
WindFnLatLon = Callable[
    [float, LatLonGrid, SigmaCoordinate],
    tuple[jax.Array, jax.Array, jax.Array],
]


class TracerTransportLatLonConfig(NamedTuple):
    """Configuration for lat-lon tracer transport model."""
    hyperdiff_coeff: float = 0.0
    time_integrator: str = "ssp_rk3"  # "ssp_rk3" | "ssp_rk34"/"ssp34" | "ssp_rk54"/"ssp45"


def tracer_tendencies_latlon(
    state: TracerState,
    grid: LatLonGrid,
    sigma_coord: SigmaCoordinate,
    wind_fn: WindFnLatLon,
    config: TracerTransportLatLonConfig = TracerTransportLatLonConfig(),
) -> TracerState:
    """Compute tracer transport tendencies on a lat-lon grid.

    Returns a TracerState-shaped pytree of tendencies for use with
    ssp_rk3_step (which requires tendency_fn to return the same pytree
    structure as the state).

    Parameters
    ----------
    state : TracerState
        Current state with tracers shape (n_lat, n_lon, nlev, n_tracers)
        and scalar time.
    grid : LatLonGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    wind_fn : WindFnLatLon
        Prescribed wind function: (t, grid, sigma_coord) -> (u, v, sigma_dot).
    config : TracerTransportLatLonConfig
        Optional configuration.

    Returns
    -------
    TracerState
        Tendency pytree: dtracers_dt and dtime_dt = 1.0.
    """
    t = state.time.data  # scalar time
    q = state.tracers.data  # (n_lat, n_lon, nlev, n_tracers)

    # Get prescribed winds at current time
    u, v, sigma_dot = wind_fn(t, grid, sigma_coord)  # (n_lat,n_lon,nlev), (n_lat,n_lon,nlev+1)

    # Compute tendencies for each tracer via vmap over the tracer axis
    def single_tracer_tendency(q_i):
        """Compute dq_i/dt for a single tracer. q_i shape: (n_lat, n_lon, nlev)."""
        # Centered advection: -(u dq/dx + v dq/dy)
        dq_dx = gradient_x_3d(q_i, grid)
        dq_dy = gradient_y_3d(q_i, grid)
        horiz_adv = -(u * dq_dx + v * dq_dy)

        # Vertical advection: -sigma_dot dq/dsigma
        vert_adv = vertical_advection(q_i, sigma_dot, sigma_coord)

        tendency = horiz_adv + vert_adv

        # Optional hyperdiffusion
        if config.hyperdiff_coeff > 0:
            tendency = tendency + hyperdiffusion_3d(q_i, grid, config.hyperdiff_coeff)

        return tendency

    # Move tracer axis to front for vmap: (n_tracers, n_lat, n_lon, nlev)
    q_t = jnp.moveaxis(q, -1, 0)
    dq_dt_t = jax.vmap(single_tracer_tendency)(q_t)  # (n_tracers, n_lat, n_lon, nlev)
    dq_dt = jnp.moveaxis(dq_dt_t, 0, -1)  # (n_lat, n_lon, nlev, n_tracers)

    # Return same pytree structure as state
    return TracerState(
        tracers=state.tracers.replace(data=dq_dt),
        time=state.time.replace(data=jnp.ones_like(t)),  # dtime/dt = 1.0
    )


class TracerTransportLatLonModel(IntegrationMixin):
    """Prescribed-wind tracer transport model on the lat-lon grid.

    Parameters
    ----------
    grid : LatLonGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    wind_fn : WindFnLatLon
        Prescribed wind function: (t, grid, sigma_coord) -> (u, v, sigma_dot).
    config : TracerTransportLatLonConfig, optional
        Model configuration.

    Example
    -------
    >>> grid = create_latlon_grid(64, 128)
    >>> sigma = create_sigma_coordinate(30)
    >>> model = TracerTransportLatLonModel(grid, sigma, prescribed_wind)
    >>> state = init_tracers(grid, sigma)
    >>> state_new = model.step(state, dt=1800.0)
    """

    def __init__(
        self,
        grid: LatLonGrid,
        sigma_coord: SigmaCoordinate,
        wind_fn: WindFnLatLon,
        config: TracerTransportLatLonConfig | None = None,
    ):
        import warnings
        warnings.warn(
            "TracerTransportLatLonModel (A-grid) is deprecated. Use the "
            "cubed-sphere or icosahedral transport models instead. "
            "See #115.",
            FutureWarning, stacklevel=2,
        )
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.wind_fn = wind_fn
        self.config = config or TracerTransportLatLonConfig()

    def tendencies(self, state: TracerState) -> TracerState:
        """Compute tendencies (pure function wrapper)."""
        return tracer_tendencies_latlon(
            state, self.grid, self.sigma_coord, self.wind_fn, self.config
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TracerState, dt: float) -> TracerState:
        """Advance one time step.

        Parameters
        ----------
        state : TracerState
            Current state.
        dt : float
            Time step [seconds].

        Returns
        -------
        TracerState : State after one time step.
        """
        def tendency_fn(s):
            return tracer_tendencies_latlon(
                s, self.grid, self.sigma_coord, self.wind_fn, self.config
            )

        return dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

    # integrate() and integrate_scan() inherited from IntegrationMixin
