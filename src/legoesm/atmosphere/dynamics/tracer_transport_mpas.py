"""Prescribed-wind tracer transport on MPAS Voronoi meshes.

Advects an arbitrary number of passive tracers using analytically
prescribed wind fields. The transport equation in sigma coordinates
uses the advective form on the unstructured C-grid:

    dq_i/dt = -(div(q_i * u) - q_i * div(u)) - sigma_dot dq_i/dsigma

The horizontal advection uses the identity:

    u . grad(q) = div(q * u) - q * div(u)

which is exact for constant q and avoids computing explicit gradients
on the Voronoi mesh. Divergence is computed via the TRiSK operator
(Ringler et al. 2010) and q at edges is obtained by simple averaging
of the two adjacent cell values.

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
from legoesm.core.operators_voronoi import (
    divergence_cell,
    cell_to_edge_avg,
)
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.grids.vertical import SigmaCoordinate, vertical_advection
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin


# Type alias for prescribed wind functions on MPAS meshes.
# Signature: (time, mesh, sigma_coord) -> (u_edge, sigma_dot)
#   u_edge:      shape (nEdges, nlev) — normal velocity at edges
#   sigma_dot:   shape (nCells, nlev+1) — vertical velocity at interfaces
WindFnMPAS = Callable[
    [float, VoronoiMesh, SigmaCoordinate],
    tuple[jax.Array, jax.Array],
]


class TracerTransportMPASConfig(NamedTuple):
    """Configuration for MPAS tracer transport model."""
    hyperdiff_coeff: float = 0.0
    time_integrator: str = "ssp_rk3"  # "ssp_rk3" | "ssp_rk34"/"ssp34" | "ssp_rk54"/"ssp45"


def tracer_tendencies_mpas(
    state: TracerState,
    mesh: VoronoiMesh,
    sigma_coord: SigmaCoordinate,
    wind_fn: WindFnMPAS,
    config: TracerTransportMPASConfig = TracerTransportMPASConfig(),
) -> TracerState:
    """Compute tracer transport tendencies on an MPAS Voronoi mesh.

    Returns a TracerState-shaped pytree of tendencies for use with
    ssp_rk3_step (which requires tendency_fn to return the same pytree
    structure as the state).

    Parameters
    ----------
    state : TracerState
        Current state with tracers shape (nCells, nlev, n_tracers) and
        scalar time.
    mesh : VoronoiMesh
        Voronoi mesh.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    wind_fn : WindFnMPAS
        Prescribed wind function: (t, mesh, sigma_coord) -> (u_edge, sigma_dot).
    config : TracerTransportMPASConfig
        Optional configuration.

    Returns
    -------
    TracerState
        Tendency pytree: dtracers_dt and dtime_dt = 1.0.
    """
    t = state.time.data  # scalar time
    q = state.tracers.data  # (nCells, nlev, n_tracers)

    # Get prescribed winds at current time
    u_edge, sigma_dot = wind_fn(t, mesh, sigma_coord)  # (nEdges, nlev), (nCells, nlev+1)

    # Compute tendencies for each tracer via vmap over the tracer axis
    def single_tracer_tendency(q_i):
        """Compute dq_i/dt for a single tracer. q_i shape: (nCells, nlev)."""
        # Horizontal advection using advective form:
        #   u . grad(q) = div(q*u) - q * div(u)
        # Compute per-level via vmap over the level axis.

        def _horiz_adv_level(q_k, u_k):
            """Horizontal advection at a single level.

            q_k: (nCells,), u_k: (nEdges,)
            """
            # Interpolate q to edges
            q_edge = cell_to_edge_avg(q_k, mesh)  # (nEdges,)

            # Flux = q_edge * u_edge
            flux = q_edge * u_k  # (nEdges,)

            # div(q*u)
            div_qu = divergence_cell(flux, mesh)  # (nCells,)

            # div(u)
            div_u = divergence_cell(u_k, mesh)  # (nCells,)

            # Advective form: -(div(q*u) - q * div(u))
            return -(div_qu - q_k * div_u)

        # vmap over vertical levels: q_i is (nCells, nlev), u_edge is (nEdges, nlev)
        # Transpose to (nlev, nCells) and (nlev, nEdges) for vmap
        q_levels = jnp.moveaxis(q_i, -1, 0)        # (nlev, nCells)
        u_levels = jnp.moveaxis(u_edge, -1, 0)      # (nlev, nEdges)

        horiz_adv_levels = jax.vmap(_horiz_adv_level)(q_levels, u_levels)  # (nlev, nCells)
        horiz_adv = jnp.moveaxis(horiz_adv_levels, 0, -1)  # (nCells, nlev)

        # Vertical advection: -sigma_dot dq/dsigma
        vert_adv = vertical_advection(q_i, sigma_dot, sigma_coord)

        tendency = horiz_adv + vert_adv

        # Hyperdiffusion placeholder — MPAS scalar hyperdiffusion is not yet
        # available in operators_voronoi; set tendency contribution to zero.
        # TODO: implement scalar_laplacian_cell and use it here.

        return tendency

    # Move tracer axis to front for vmap: (n_tracers, nCells, nlev)
    q_t = jnp.moveaxis(q, -1, 0)
    dq_dt_t = jax.vmap(single_tracer_tendency)(q_t)  # (n_tracers, nCells, nlev)
    dq_dt = jnp.moveaxis(dq_dt_t, 0, -1)  # (nCells, nlev, n_tracers)

    # Return same pytree structure as state
    return TracerState(
        tracers=state.tracers.replace(data=dq_dt),
        time=state.time.replace(data=jnp.ones_like(t)),  # dtime/dt = 1.0
    )


class TracerTransportMPASModel(IntegrationMixin):
    """Prescribed-wind tracer transport model on an MPAS Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
        Voronoi mesh.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    wind_fn : WindFnMPAS
        Prescribed wind function: (t, mesh, sigma_coord) -> (u_edge, sigma_dot).
    config : TracerTransportMPASConfig, optional
        Model configuration.

    Example
    -------
    >>> mesh = create_voronoi_mesh(2562)
    >>> sigma = create_sigma_coordinate(30)
    >>> model = TracerTransportMPASModel(mesh, sigma, prescribed_wind)
    >>> state = init_tracers(mesh, sigma)
    >>> state_new = model.step(state, dt=1800.0)
    """

    def __init__(
        self,
        mesh: VoronoiMesh,
        sigma_coord: SigmaCoordinate,
        wind_fn: WindFnMPAS,
        config: TracerTransportMPASConfig | None = None,
    ):
        self.mesh = mesh
        self.sigma_coord = sigma_coord
        self.wind_fn = wind_fn
        self.config = config or TracerTransportMPASConfig()

    def tendencies(self, state: TracerState) -> TracerState:
        """Compute tendencies (pure function wrapper)."""
        return tracer_tendencies_mpas(
            state, self.mesh, self.sigma_coord, self.wind_fn, self.config
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
            return tracer_tendencies_mpas(
                s, self.mesh, self.sigma_coord, self.wind_fn, self.config
            )

        return dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

    # integrate() and integrate_scan() inherited from IntegrationMixin
