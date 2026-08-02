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
    sigma, ``vertical_advection_hybrid`` for the hybrid dycore).  Keeping the
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


def tracer_flux_form_tendency(
    q, u_edge, dp_edge, dp_cell, div_dp_cell, vert_mass_flux, mesh,
):
    """Mass-CONSISTENT flux-form tracer tendency ``dq/dt`` on the MPAS C-grid.

    Transports the layer tracer MASS ``q·δp`` with the SAME discrete edge mass
    flux ``u·δp_edge`` and the SAME half-level vertical mass flux ``F`` the
    dycore's own continuity closure uses, then converts back to a mixing-ratio
    tendency by subtracting the co-transported layer-mass tendency::

        d(q δp_k)/dt = -div(u q_e δp_e) - (F_{k+1/2} q*_{k+1/2}
                                           - F_{k-1/2} q*_{k-1/2})
        d(δp_k)/dt   = -div(u δp_e)     - (F_{k+1/2} - F_{k-1/2})
        dq_k/dt      = [ d(q δp_k)/dt - q_k · d(δp_k)/dt ] / δp_k

    WHAT THIS BUYS OVER THE ADVECTIVE FORM — be precise, the obvious answer is
    wrong.  ``-(div(q u) - q div(u))`` (:func:`tracer_horizontal_advection`) is
    a CONSISTENT discretisation of the same continuous equation (``∂q/∂t =
    -v·∇q - σ̇ ∂q/∂σ`` IS the flux form minus ``q``× continuity) and it IS
    free-stream preserving — it is identically zero for constant ``q``.  What
    it is not is discretely CONSERVATIVE: it differences ``q`` against the
    *unweighted* ``div(u)`` while this dycore's continuity carries layer mass
    with the *δp-weighted* ``div(u δp_e)``, so the per-cell errors no longer
    telescope and ``Σ_cells areaCell Σ_k q δp`` drifts.  Measured on a level-2
    mesh with a ±80 hPa surface-pressure wave
    (``tests/atmosphere/dycore/unit/test_mpas_flux_form_tracers.py``): the
    advective operator's global tracer-mass tendency is ~1.5e-7 of the total
    tracer mass per SECOND; this operator's is round-off (~1e-17 of the gross
    transport terms).  The error is largest where the layer-mass gradient and
    the divergence are both large — steep terrain under a divergent wind.

    SIGN / COORDINATE CONVENTION (the whole file inherits it from
    :class:`~legoesm.grids.vertical.SigmaCoordinate`):

    * ``k = 0`` is the MODEL TOP; σ (and pressure) INCREASE DOWNWARD, so
      ``δp_k > 0``.
    * Interface index ``j`` in ``vert_mass_flux`` sits ABOVE layer ``j``:
      layer ``k`` is bounded by interfaces ``k`` (top) and ``k+1`` (bottom).
    * ``F_j > 0`` means mass moving DOWNWARD (toward larger σ/p) through
      interface ``j`` [Pa/s] — the same sign convention
      :func:`~legoesm.grids.vertical.vertical_advection` upwinds against
      (``σ̇ > 0`` selects the backward/above difference).
    * ``F_0 = F_nlev = 0`` (rigid lid, material lower boundary).  The vertical
      flux difference therefore TELESCOPES to zero over a column, and the
      horizontal divergence telescopes to zero over the mesh (each edge enters
      its two cells with opposite ``edgeSignOnCell``), so ``Σ_cells areaCell ·
      Σ_k δp_k dq_k/dt + q_k d(δp_k)/dt`` — i.e. the rate of change of global
      column tracer mass — vanishes to round-off with no sources.
    * ``div(u δp_e) > 0`` is mass EXPORT, hence the leading minus signs.

    FREE-STREAM PRESERVATION (necessary, but NOT what distinguishes this from
    the advective form).  For a spatially and vertically CONSTANT ``q``:
    ``q_e = q`` exactly (a two-cell 0.5-average of equal values), ``q* = q``
    exactly (an upwind select between equal values), so the vertical terms
    cancel BIT-exactly and the horizontal terms cancel to the round-off of the
    divergence reduction.  No wind field, terrain or coordinate can then
    manufacture a tracer extremum out of a uniform field.

    LIMITER.  Like the advective form this operator uses a CENTRED
    (``cell_to_edge_avg_3d``) edge value and an UPWIND interface value; it is
    conservative and free-stream preserving but NOT monotone, so it can still
    undershoot.  The MPAS floors stage (``conservative_positive_clamp``) is
    still required, and with it the chain conserves column tracer mass.

    Parameters
    ----------
    q : jax.Array, (nCells, nlev, n_tracers)
        Mixing ratios [per unit MASS].  Per-VOLUME fields (``N_c``/``N_r``,
        [#/m³]) must NOT be routed here — a δp-weighted transport conserves
        ``∫ q dp/g`` which has no meaning for them; the caller filters with
        ``legoesm.core.conservation.is_borrow_eligible_tracer``.
    u_edge : jax.Array, (nEdges, nlev)
        Edge-normal advecting velocity [m/s] — the dycore's own ``u``.
    dp_edge : jax.Array, (nEdges, nlev)
        Layer thickness at edges [Pa] — the SAME array the continuity's
        ``div(u δp_e)`` was formed from.
    dp_cell : jax.Array, (nCells, nlev)
        Layer thickness at cells [Pa], strictly positive.
    div_dp_cell : jax.Array, (nCells, nlev)
        ``div(u δp_e)`` at cells [Pa/s] — the dycore's flux-form layer-mass
        divergence, passed in so the tracer differences against the SAME
        discrete operator (re-deriving it would reintroduce the defect).
    vert_mass_flux : jax.Array, (nCells, nlev+1)
        Half-level vertical mass flux ``F`` [Pa/s], positive DOWNWARD, zero at
        both boundaries.  σ coordinate: ``p_s·σ̇``.  Hybrid: ``mass_flux``.
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, (nCells, nlev, n_tracers)
        Mixing-ratio tendency [1/s].
    """
    n_cells, nlev, n_tracers = q.shape

    # --- Horizontal: div(u · q_edge · δp_edge) --------------------------
    # Fold tracers into the level axis so the gather + divergence run ONCE
    # for all tracers (same batching trick as tracer_horizontal_advection).
    q_flat = q.reshape(n_cells, nlev * n_tracers)
    q_edge = cell_to_edge_avg_3d(q_flat, mesh)                # (nEdges, nlev*ntr)
    n_edges = q_edge.shape[0]
    q_edge = q_edge.reshape(n_edges, nlev, n_tracers)
    # (nEdges, nlev, 1): the edge MASS flux the continuity already uses.
    mass_flux_edge = (u_edge * dp_edge)[..., None]
    div_q_mass = divergence_cell_3d(
        (q_edge * mass_flux_edge).reshape(n_edges, nlev * n_tracers), mesh,
    ).reshape(n_cells, nlev, n_tracers)                        # [Pa/s]

    # --- Vertical: Δ_k( F · q* ) ----------------------------------------
    # Interface j lies above layer j.  F_j > 0 (downward) ⇒ the mass crossing
    # interface j came from layer j-1 (ABOVE) ⇒ upwind value q_{j-1}.
    f_interior = vert_mass_flux[:, 1:-1, None]                 # (nCells, nlev-1, 1)
    q_upwind = jnp.where(f_interior > 0, q[:, :-1, :], q[:, 1:, :])
    # Pad the two boundary interfaces; F = 0 there, so the padded value is
    # multiplied by zero and never enters the budget.
    q_iface = jnp.pad(q_upwind, ((0, 0), (1, 1), (0, 0)))       # (nCells, nlev+1, ntr)
    vflux = vert_mass_flux[..., None] * q_iface                 # (nCells, nlev+1, ntr)
    d_vflux = vflux[:, 1:, :] - vflux[:, :-1, :]                # bottom minus top
    d_f = (vert_mass_flux[:, 1:] - vert_mass_flux[:, :-1])[..., None]

    # d(q δp)/dt and the co-transported d(δp)/dt, from the SAME fluxes.
    d_qdp_dt = -div_q_mass - d_vflux
    d_dp_dt = -div_dp_cell[..., None] - d_f

    # δp_k > 0 by construction (p_s is clipped to [p_floor, p_ceil] upstream
    # and every coordinate layer has positive thickness), so no guard here —
    # a degenerate coordinate must fail loudly, not be silently floored.
    return (d_qdp_dt - q * d_dp_dt) / dp_cell[..., None]


def tracer_advection_tendency(q, u_edge, sigma_dot, mesh, sigma_coord):
    """Full sigma-coordinate advective tracer tendency = shared horizontal
    (:func:`tracer_horizontal_advection`) + sigma vertical advection.  Used by
    the standalone prescribed-wind model; the hybrid PE builds its own
    (horizontal + ``vertical_advection_hybrid``)."""
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
