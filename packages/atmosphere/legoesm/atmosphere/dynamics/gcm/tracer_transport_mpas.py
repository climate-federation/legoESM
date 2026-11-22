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
    (``primitive_eq_mpas.mpas_legacy_pe_tendencies``), each of which then
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


def fct_in_out_sums(out, width):
    """FCT antidiffusive in/out sums over a cell's edges (axis 0; ``out > 0``
    leaves the cell) as a C1 positive part from ABOVE: ``(x + W)^2 / 4W`` on
    ``|x| < W``, exact ``max(x, 0)`` outside, ``W = width`` times the L2 norm of
    the cell's ``out`` (smooth away from all-zero, where an L1 sum ``|out|``
    kinks as one flux reverses).  ``width = 0`` is the hard sum."""
    s2 = jnp.sum(out * out, axis=0)
    band = width * jnp.where(s2 > 0, jnp.sqrt(jnp.where(s2 > 0, s2, 1.0)), 0.0)

    def part_up(x):
        inside = jnp.abs(x) < band
        wb = jnp.where(inside, band, 1.0)
        return jnp.where(inside, (x + wb) ** 2 / (4.0 * wb), jnp.maximum(x, 0.0))

    return jnp.sum(part_up(-out), axis=0), jnp.sum(part_up(out), axis=0)


def tracer_horizontal_flux_fct(q, u_edge, dp, dp_edge, mesh, dt, width=1.0e-4,
                               upwind_width_ms=1.0e-3):
    """Flux-form, mass-consistent horizontal tracer tendency with Zalesak FCT.

    The layer tracer mass ``q dp`` obeys ``d(q dp)/dt = -div(u dp_e q_e)``,
    paired with the SAME layer mass flux ``u dp_e`` whose divergence drives
    the continuity; the state carries ``q``, so this returns
    ``(-div(u dp_e q_e) + q div(u dp_e)) / dp``.  ``q_e``: first-order upwind
    plus the antidiffusive part toward the centred value ``(q_1 + q_2)/2``,
    limited (Zalesak 1979, doi:10.1016/0021-9991(79)90051-2; the limiter MPAS-A
    applies, Skamarock and Gassmann 2011, doi:10.1175/MWR-D-10-05056.1) so the
    forward-Euler update over ``dt`` stays inside the range of ``q`` and of the
    low-order update over the cell and its neighbours.  Port of the ENSO fork
    (Di Carlo, e0226a4a8b) onto the shared PE core.

    KNOWN DIFFERENCE (RULE 5): every switch on the state is C1 or smooth, with
    ``width = upwind_width_ms = 0`` the fork's hard limiter: the upwind
    selection is ``|F| -> F tanh(F / (upwind_width_ms dp_e))`` (no flux at
    rest, less upwind diffusion for ``|u| < ~upwind_width_ms``, so ``q_td``
    can leave the range of ``q`` there); the neighbour bounds and the clips
    ``min(1, R)`` are ``core.smooth`` min/max with a band ``width`` relative
    to the operands (the smooth max is never below the hard one, the min
    never above); the inflow/outflow sums are a C1 positive part from above
    (``(x + W)^2 / 4W`` on ``|x| < W``, ``W`` = width times the L2 norm of
    the cell's fluxes)
    so they over-count; the donor/acceptor factor ramps (``step_rel``,
    ``|anti| < width * a (|q1| + |q2|)/2``) above the smaller of the two
    directions' factors on each side of ``anti = 0``.  Each smooth piece errs
    toward more limiting, so the update stays inside the bounds built from
    ``q_td``; conservation is exact (one limited flux per edge).
    The relative bands vanish with their operands, so a comparison against an
    exactly zero value (an empty cell beside an empty one, ``q_td = q = 0``)
    stays the hard one.  Measured gaps and bound excursions at the defaults
    are stored per case in tests/unit/test_mpas_terrain_damping.py.
    A cell with no antidiffusive inflow (outflow) takes ratio 1, the limit of
    ``min(1, room/p)`` as ``p -> 0+`` (the fork took 0; the limited flux is
    zero there either way at the hard limit).

    ``q`` (nCells, nlev, ntr); ``u_edge`` (nEdges, nlev); ``dp`` (nCells,
    nlev) and ``dp_edge`` (nEdges, nlev) [Pa]; ``dt`` [s] the step the limiter
    guards (> 0).  Returns (nCells, nlev, ntr).
    """
    from legoesm.core.smooth import max_rel, min_rel, step_rel

    if isinstance(dt, (int, float)) and not dt > 0:
        raise ValueError(f"tracer_horizontal_flux_fct needs dt > 0 (got {dt!r})")
    nC, nlev, ntr = q.shape
    nE = u_edge.shape[0]

    def div(f):   # (nEdges, nlev, ntr) per unit length -> (nCells, nlev, ntr) per area
        return divergence_cell_3d(f.reshape(nE, -1), mesh).reshape(nC, nlev, -1)

    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    dp_c = jnp.where(dp != 0, dp, 1.0)[..., None]        # halo padding (p_s = 0)
    fm = (u_edge * dp_edge)[..., None]                   # (nE, nlev, 1)
    if upwind_width_ms == 0:
        a = jnp.abs(fm)
    else:
        wa = (upwind_width_ms * jnp.where(dp_edge > 0, dp_edge, 1.0))[..., None]
        a = fm * jnp.tanh(fm / wa)
    q1, q2 = q[c1], q[c2]
    f_low = 0.5 * fm * (q1 + q2) - 0.5 * a * (q2 - q1)   # upwind at a = |fm|
    anti = 0.5 * a * (q2 - q1)                           # centred minus low order
    div_fm = div(fm)
    q_td = q + dt * (q * div_fm - div(f_low)) / dp_c
    # bounds: q and the low-order update over the cell and its neighbours
    coc = mesh.cellsOnCell
    ok = (coc >= 0)[..., None, None]
    coc_s = jnp.maximum(coc, 0)
    lo, hi = min_rel(q, q_td, width), max_rel(q, q_td, width)
    nb_lo, nb_hi = jnp.where(ok, lo[coc_s], lo), jnp.where(ok, hi[coc_s], hi)
    q_min, q_max = lo, hi
    for k in range(coc.shape[0]):
        q_min, q_max = min_rel(q_min, nb_lo[k], width), max_rel(q_max, nb_hi[k], width)
    # antidiffusive in/out per cell area (> 0 leaves the cell)
    eoc = mesh.edgesOnCell
    # in the state's precision: a mesh may store its geometry in float32
    w = (jnp.where(eoc >= 0, mesh.edgeSignOnCell, 0).astype(q.dtype)
         * mesh.dvEdge[jnp.maximum(eoc, 0)].astype(q.dtype)
         / mesh.areaCell[None, :].astype(q.dtype))[..., None, None]
    out = w * anti[jnp.maximum(eoc, 0)]
    p_in, p_out = fct_in_out_sums(out, width)
    # rooms are >= 0: the bounds include q_td and the smooth max (min) is
    # never below (above) the hard one
    room_in = (q_max - q_td) * dp_c / dt
    room_out = (q_td - q_min) * dp_c / dt

    def ratio(room, p):
        return jnp.where(p > 0, min_rel(1.0, room / jnp.where(p > 0, p, 1.0), width), 1.0)

    r_plus, r_minus = ratio(room_in, p_in), ratio(room_out, p_out)
    # donor/acceptor pair by the flux direction: c1 -> c2 for anti > 0.  Each
    # direction's factor only ramps in on its own side of anti = 0 above the
    # common lower bound, so the factor never exceeds the bound of the side
    # the flux actually takes.
    fwd = min_rel(r_minus[c1], r_plus[c2], width)
    bwd = min_rel(r_plus[c1], r_minus[c2], width)
    if width == 0:
        lim = jnp.where(anti >= 0, fwd, bwd)
    else:
        se = 0.5 * a * (jnp.abs(q1) + jnp.abs(q2))
        both = min_rel(fwd, bwd, width)
        lim = (both + step_rel(anti, se, width) * (fwd - both)
               + step_rel(-anti, se, width) * (bwd - both))
    return (q * div_fm - div(f_low + lim * anti)) / dp_c
