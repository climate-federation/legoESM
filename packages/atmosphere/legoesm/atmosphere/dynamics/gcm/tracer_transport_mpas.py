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

from legoesm.core.state import TracerState
from legoesm.core.operators_voronoi import (
    divergence_cell_3d,
    cell_to_edge_avg_3d,
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


def tracer_horizontal_advection(q, u_edge, mesh):
    """Advective-form horizontal tracer tendency ``-(div(q u) - q div(u))``
    on the MPAS C-grid.

    SHARED, vertical-coordinate-INDEPENDENT kernel: reused by BOTH the sigma
    standalone ``tracer_advection_tendency`` AND the hybrid coupled PE
    (``primitive_eq_mpas.mpas_hydrostatic_tendencies``), each of which then
    adds its OWN matching vertical advection (``vertical_advection`` for
    sigma, ``vertical_advection_hybrid_van_leer`` for the hybrid dycore).  Keeping the
    horizontal numerics in one place avoids duplicating the gather +
    divergence across the two paths.

    Parameters
    ----------
    q : (nCells, nlev, n_tracers)
    u_edge : (nEdges, nlev)   edge-normal advecting velocity
    mesh : VoronoiMesh

    Returns
    -------
    (nCells, nlev, n_tracers) horizontal advective tendency.
    """
    nCells, nlev, n_tracers = q.shape

    # ``divergence_cell_3d`` and ``cell_to_edge_avg_3d`` accept any trailing
    # axis as passive, so fold tracers into the level axis to run the
    # gather + divergence ONCE for all tracers instead of n_tracers
    # vmap'd graphs.
    q_flat = q.reshape(nCells, nlev * n_tracers)
    q_edge_flat = cell_to_edge_avg_3d(q_flat, mesh)               # (nEdges, nlev*n_tracers)
    n_edges = q_edge_flat.shape[0]
    # Multiply by u_edge via reshape→multiply→reshape so u_edge (nEdges, nlev)
    # broadcasts against the tracer axis without materializing a tile.
    q_edge = q_edge_flat.reshape(n_edges, nlev, n_tracers)
    flux = q_edge * u_edge[..., None]                              # (nEdges, nlev, n_tracers)
    flux_flat = flux.reshape(n_edges, nlev * n_tracers)

    # Batch ``div(u)`` (shared across tracers) with the per-tracer
    # ``div(q*u)`` flux divergences into a single ``divergence_cell_3d``
    # call by concatenating along the trailing axis.  ``div(u)`` claims
    # the first ``nlev`` slots; the per-tracer ``div(q*u)`` claims the
    # rest.  ``n_tracers + 1`` divergences → 1.
    _u_and_flux = jnp.concatenate(
        [u_edge, flux_flat], axis=-1,
    )  # (nEdges, nlev*(1 + n_tracers))
    _u_flux_div = divergence_cell_3d(_u_and_flux, mesh)
    div_u_3d = _u_flux_div[:, :nlev]                # (nCells, nlev)
    div_qu_flat = _u_flux_div[:, nlev:]             # (nCells, nlev*n_tracers)
    div_qu = div_qu_flat.reshape(nCells, nlev, n_tracers)
    return -(div_qu - q * div_u_3d[..., None])


def tracer_advection_tendency(q, u_edge, sigma_dot, mesh, sigma_coord):
    """Full sigma-coordinate advective tracer tendency = shared horizontal
    (:func:`tracer_horizontal_advection`) + sigma vertical advection.  Used by
    the standalone prescribed-wind model; the hybrid PE builds its own
    (horizontal + ``vertical_advection_hybrid_van_leer``)."""
    horiz_adv = tracer_horizontal_advection(q, u_edge, mesh)

    # Vertical advection — local stencil along axis -1, no halo cost.
    # ``vertical_advection`` hard-codes axis -1 as nlev, so vmap over
    # the trailing tracer axis (with sigma_dot/sigma_coord captured).
    def _vert_one(q_one):
        return vertical_advection(q_one, sigma_dot, sigma_coord)

    vert_adv = jax.vmap(_vert_one, in_axes=-1, out_axes=-1)(q)

    return horiz_adv + vert_adv


def tracer_tendencies_mpas(
    state: TracerState,
    mesh: VoronoiMesh,
    sigma_coord: SigmaCoordinate,
    wind_fn: WindFnMPAS,
    config: TracerTransportMPASConfig = TracerTransportMPASConfig(),
) -> TracerState:
    """Compute tracer transport tendencies on an MPAS Voronoi mesh.

    Thin prescribed-wind wrapper around the shared
    :func:`tracer_advection_tendency` kernel.  Returns a TracerState-shaped
    pytree of tendencies for use with ssp_rk3_step (which requires
    tendency_fn to return the same pytree structure as the state).

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

    dq_dt = tracer_advection_tendency(q, u_edge, sigma_dot, mesh, sigma_coord)

    # Hyperdiffusion placeholder — MPAS scalar hyperdiffusion is not yet
    # available in operators_voronoi; tendency contribution is zero.
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
