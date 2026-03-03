"""Prescribed-wind tracer transport on the cubed-sphere.

Advects an arbitrary number of passive tracers using analytically
prescribed wind fields. The transport equation in sigma coordinates:

    dq_i/dt = -(u dq_i/dx + v dq_i/dy) - sigma_dot dq_i/dsigma + D(q_i)

where D is optional hyperdiffusion for numerical stability.

This model is used for DCMIP-2012 transport test cases (Tests 1-1, 1-2, 1-3)
where the wind field is specified analytically rather than solved prognostically.

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
from legoesm.core.operators_3d import (
    gradient_x_3d,
    gradient_y_3d,
    hyperdiffusion_3d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import SigmaCoordinate, vertical_advection
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step


# Type alias for prescribed wind functions.
# Signature: (time, grid, sigma_coord) -> (u, v, sigma_dot)
#   u, v:        shape (6, n, n, nlev) — grid-aligned wind components
#   sigma_dot:   shape (6, n, n, nlev+1) — vertical velocity at interfaces
WindFn = Callable[
    [float, CubedSphereGrid, SigmaCoordinate],
    tuple[jax.Array, jax.Array, jax.Array],
]


class TracerTransportConfig(NamedTuple):
    """Configuration for tracer transport model."""
    hyperdiff_coeff: float = 0.0
    time_integrator: str = "ssp_rk3"  # "ssp_rk3" | "ssp_rk34"/"ssp34" | "ssp_rk54"/"ssp45"


def tracer_tendencies(
    state: TracerState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
    wind_fn: WindFn,
    config: TracerTransportConfig = TracerTransportConfig(),
) -> TracerState:
    """Compute tracer transport tendencies.

    Returns a TracerState-shaped pytree of tendencies for use with
    ssp_rk3_step (which requires tendency_fn to return the same pytree
    structure as the state).

    Parameters
    ----------
    state : TracerState
        Current state with tracers shape (6, n, n, nlev, n_tracers) and
        scalar time.
    grid : CubedSphereGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    wind_fn : WindFn
        Prescribed wind function: (t, grid, sigma_coord) -> (u, v, sigma_dot).
    config : TracerTransportConfig
        Optional configuration.

    Returns
    -------
    TracerState
        Tendency pytree: dtracers_dt and dtime_dt = 1.0.
    """
    t = state.time.data  # scalar time
    q = state.tracers.data  # (6, n, n, nlev, n_tracers)
    n_tracers = q.shape[-1]

    # Get prescribed winds at current time
    u, v, sigma_dot = wind_fn(t, grid, sigma_coord)  # (6,n,n,nlev), (6,n,n,nlev+1)

    # Compute tendencies for each tracer via vmap over the tracer axis
    def single_tracer_tendency(q_i):
        """Compute dq_i/dt for a single tracer. q_i shape: (6, n, n, nlev)."""
        # Horizontal advection: -(u dq/dx + v dq/dy)
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

    # Move tracer axis to front for vmap: (n_tracers, 6, n, n, nlev)
    q_t = jnp.moveaxis(q, -1, 0)
    dq_dt_t = jax.vmap(single_tracer_tendency)(q_t)  # (n_tracers, 6, n, n, nlev)
    dq_dt = jnp.moveaxis(dq_dt_t, 0, -1)  # (6, n, n, nlev, n_tracers)

    # Return same pytree structure as state
    return TracerState(
        tracers=state.tracers.replace(data=dq_dt),
        time=state.time.replace(data=jnp.ones_like(t)),  # dtime/dt = 1.0
    )


class TracerTransportModel:
    """Prescribed-wind tracer transport model on the cubed-sphere.

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    wind_fn : WindFn
        Prescribed wind function: (t, grid, sigma_coord) -> (u, v, sigma_dot).
    config : TracerTransportConfig, optional
        Model configuration.

    Example
    -------
    >>> grid = create_cubed_sphere(48)
    >>> sigma = create_sigma_coordinate(30)
    >>> model = TracerTransportModel(grid, sigma, dcmip11_wind)
    >>> state = dcmip11_init(grid, sigma)
    >>> state_new = model.step(state, dt=1800.0)
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        wind_fn: WindFn,
        config: TracerTransportConfig | None = None,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.wind_fn = wind_fn
        self.config = config or TracerTransportConfig()

    def tendencies(self, state: TracerState) -> TracerState:
        """Compute tendencies (pure function wrapper)."""
        return tracer_tendencies(
            state, self.grid, self.sigma_coord, self.wind_fn, self.config
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TracerState, dt: float) -> TracerState:
        """Advance one time step using SSP-RK3.

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
            return tracer_tendencies(
                s, self.grid, self.sigma_coord, self.wind_fn, self.config
            )

        integrator = self.config.time_integrator.lower()
        if integrator in ("ssp_rk54", "ssp54", "ssp45", "rk54"):
            return ssp_rk54_step(state, tendency_fn, dt)
        if integrator in ("ssp_rk34", "ssp34", "rk34"):
            return ssp_rk34_step(state, tendency_fn, dt)
        if integrator in ("ssp_rk3", "ssp3", "rk3"):
            return ssp_rk3_step(state, tendency_fn, dt)
        raise ValueError(f"Unsupported time_integrator={self.config.time_integrator!r}")

    def integrate(
        self,
        state: TracerState,
        duration: float,
        dt: float,
        save_every: int = 1,
        callback=None,
    ) -> tuple[TracerState, list[TracerState]]:
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : TracerState
            Initial state.
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.
        callback : callable, optional
            Called as callback(step_number, state) every save_every steps.

        Returns
        -------
        final_state : TracerState
        trajectory : list of TracerState
        """
        n_steps = int(duration / dt)
        trajectory = [state]

        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
                if callback is not None:
                    callback(i + 1, state)

        return state, trajectory

    def integrate_scan(
        self,
        state: TracerState,
        n_steps: int,
        dt: float,
    ) -> tuple[TracerState, TracerState]:
        """Integrate using jax.lax.scan (differentiable, JIT-friendly).

        Parameters
        ----------
        state : TracerState
            Initial state.
        n_steps : int
            Number of time steps.
        dt : float
            Time step [seconds].

        Returns
        -------
        final_state : TracerState
        trajectory : TracerState
            All states, each leaf shape: (n_steps, ...).
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, xs=None, length=n_steps
        )
        return final_state, trajectory
