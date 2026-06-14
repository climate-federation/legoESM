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

from legoesm.core.state import TracerState
from legoesm.core.operators_3d import (
    gradient_x_3d,
    gradient_y_3d,
    hyperdiffusion_3d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo_4d
from legoesm.grids.vertical import SigmaCoordinate, vertical_advection
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin


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

    # Fold the tracer axis into the level axis so the cubed-sphere halo+stencil
    # operators (``gradient_x_3d``, ``gradient_y_3d``, ``hyperdiffusion_3d``)
    # process all tracers in a single ``pad_halo_4d`` call instead of issuing
    # one halo exchange per tracer under vmap.  Multi-GPU MPI exchange dominates
    # the cost of these per-level operators, so collapsing n_tracers separate
    # exchanges into one is a substantial savings under multi-GPU sharding.
    nlev = q.shape[-2]
    q_flat = q.reshape(*q.shape[:3], nlev * n_tracers)  # (6, n, n, nlev*n_tracers)

    # Pre-pad ``q_flat`` once and feed it to both gradient_x_3d and
    # gradient_y_3d via ``padded=``.  Halves the gradient halo cost
    # (1 MPI exchange instead of 2 on the same input).  The pad is
    # also reused inside ``hyperdiffusion_3d``'s inner Laplacian when
    # hyperdiffusion is enabled.
    _dg_q = getattr(grid, 'duogrid', None)
    _offsets_q = None if _dg_q is not None else grid.halo_interp_offsets
    _q_flat_pad = pad_halo_4d(q_flat, interp_offsets=_offsets_q, duogrid=_dg_q)

    dq_dx_flat = gradient_x_3d(q_flat, grid, padded=_q_flat_pad)
    dq_dy_flat = gradient_y_3d(q_flat, grid, padded=_q_flat_pad)
    dq_dx = dq_dx_flat.reshape(*q.shape)  # (6, n, n, nlev, n_tracers)
    dq_dy = dq_dy_flat.reshape(*q.shape)
    horiz_adv = -(u[..., None] * dq_dx + v[..., None] * dq_dy)

    if config.hyperdiff_coeff > 0:
        hyper_flat = hyperdiffusion_3d(
            q_flat, grid, config.hyperdiff_coeff, padded=_q_flat_pad,
        )
        horiz_adv = horiz_adv + hyper_flat.reshape(*q.shape)

    # Vertical advection — local stencil along axis -1, no halo cost.  Use
    # ``jax.vmap`` over the tracer axis (with sigma_dot/sigma_coord captured
    # in the closure) so JAX produces a single batched kernel rather than
    # n_tracers unrolled stencils.
    def _vert_adv_one(q_one):
        return vertical_advection(q_one, sigma_dot, sigma_coord)

    vert_adv = jax.vmap(_vert_adv_one, in_axes=-1, out_axes=-1)(q)

    dq_dt = horiz_adv + vert_adv  # (6, n, n, nlev, n_tracers)

    # Return same pytree structure as state
    return TracerState(
        tracers=state.tracers.replace(data=dq_dt),
        time=state.time.replace(data=jnp.ones_like(t)),  # dtime/dt = 1.0
    )


class TracerTransportModel(IntegrationMixin):
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
            return tracer_tendencies(
                s, self.grid, self.sigma_coord, self.wind_fn, self.config
            )

        return dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

    # integrate() and integrate_scan() inherited from IntegrationMixin
