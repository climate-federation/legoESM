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
of the two adjacent cell values (``horizontal_scheme="centred2"``).

``horizontal_scheme="mpas_fct"`` is the MPAS-Atmosphere scalar transport
instead: flux form on a prognostic layer air mass ``dp`` (state
:class:`TracerAirMassState`), the 3rd/4th-order edge reconstruction of
Skamarock & Gassmann (2011) and the Wicker-Skamarock RK3 of
``atm_srk3`` with the monotonic flux limiter of ``atm_advance_scalars_mono``
(Wang, Skamarock & Feingold 2009) on the final stage only
(:func:`mpas_rk3_fct_step`).

Time is embedded in the TracerState pytree so that SSP-RK3 evaluates
the prescribed wind function at the correct intermediate times for
each Runge-Kutta stage.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
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
    time_integrator: str = "ssp_rk3"  # "ssp_rk3" | "ssp_rk34"/"ssp34" | "ssp_rk54"/"ssp45" | "mpas_rk3"
    # Horizontal scheme: "centred2" (advective form, centred two-cell edge
    # average, any ``dispatch_integrator`` scheme) or "mpas_fct" (MPAS-A
    # scalar transport; needs ``time_integrator="mpas_rk3"`` and a
    # :class:`TracerAirMassState`).
    horizontal_scheme: str = "centred2"
    # MPAS ``config_coef_3rd_order`` (Registry default 0.25): 0 = 4th-order
    # centred, 1 = 3rd-order upwind edge reconstruction.
    coef_3rd_order: float = 0.25
    # RULE 5 widths of the mpas_fct limiter switches (as the PE dycore's
    # ``fct_smoothing_width`` / ``fct_upwind_width_ms``); both 0 = hard MPAS.
    fct_smoothing_width: float = 1.0e-4
    fct_upwind_width_ms: float = 1.0e-3


HORIZONTAL_SCHEMES = ("centred2", "mpas_fct")
# atm_advance_scalars_mono's eps in the scale factors' denominators
MPAS_MONO_EPS = 1.0e-20


def validate_tracer_transport_mpas_config(config: TracerTransportMPASConfig) -> None:
    """Raise on an unknown horizontal scheme or an integrator it cannot use."""
    if config.horizontal_scheme not in HORIZONTAL_SCHEMES:
        raise ValueError(f"unknown horizontal_scheme {config.horizontal_scheme!r}; "
                         f"expected one of {HORIZONTAL_SCHEMES}")
    if (config.horizontal_scheme == "mpas_fct") != (config.time_integrator == "mpas_rk3"):
        raise ValueError("horizontal_scheme='mpas_fct' runs (only) with "
                         "time_integrator='mpas_rk3' and vice versa; got "
                         f"{config.horizontal_scheme!r} / {config.time_integrator!r}")
    if config.horizontal_scheme == "mpas_fct" and config.hyperdiff_coeff != 0.0:
        raise ValueError("horizontal_scheme='mpas_fct' has no hyperdiffusion; "
                         f"hyperdiff_coeff must be 0 (got {config.hyperdiff_coeff!r})")
    if not 0.0 <= config.coef_3rd_order <= 1.0:
        raise ValueError(f"coef_3rd_order must be in [0, 1]; got {config.coef_3rd_order!r}")
    if not 0.0 <= config.fct_smoothing_width <= FCT_MAX_SMOOTHING_WIDTH:
        raise ValueError("fct_smoothing_width must be in [0, sqrt(5)-1]; got "
                         f"{config.fct_smoothing_width!r}")
    if not (math.isfinite(config.fct_upwind_width_ms) and config.fct_upwind_width_ms >= 0.0):
        raise ValueError("fct_upwind_width_ms must be finite and >= 0; got "
                         f"{config.fct_upwind_width_ms!r}")


class TracerAirMassState(NamedTuple):
    """Prescribed-wind tracer state with a prognostic layer air mass.

    ``tracers`` (nCells, nlev, ntr) mixing ratios, ``air_mass`` (nCells,
    nlev) layer pressure thickness ``dp`` [Pa] (the MPAS ``rho_zz`` analogue),
    ``time`` [s].  The tracer mass of a layer is ``q dp areaCell / g``."""
    tracers: Field
    air_mass: Field
    time: Field


def air_mass_state(state: TracerState, p_s, sigma_coord: SigmaCoordinate) -> TracerAirMassState:
    """Attach the layer air mass ``dp = p_s dsigma`` [Pa] of a sigma column
    (``p_s`` [Pa], scalar or (nCells,)) to a :class:`TracerState`."""
    q = state.tracers.data
    p_s = jnp.broadcast_to(jnp.asarray(p_s, q.dtype), q.shape[:1])
    dp = p_s[:, None] * jnp.asarray(sigma_coord.dsigma, q.dtype)[None, :]
    return TracerAirMassState(
        tracers=state.tracers,
        air_mass=Field(data=dp, name="air_mass", dims=("cell", "level"), units="Pa"),
        time=state.time)


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
    vert_adv = tracer_vertical_advection(q, sigma_dot, sigma_coord)
    return horiz_adv + vert_adv


def tracer_vertical_advection(q, sigma_dot, sigma_coord):
    """Sigma vertical advective tendency of (nCells, nlev, ntr) tracers."""
    # Vertical advection — local stencil along axis -1, no halo cost.
    # ``vertical_advection`` hard-codes axis -1 as nlev, so vmap over
    # the trailing tracer axis (with sigma_dot/sigma_coord captured).
    def _vert_one(q_one):
        return vertical_advection(q_one, sigma_dot, sigma_coord)

    return jax.vmap(_vert_one, in_axes=-1, out_axes=-1)(q)


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
    if config.horizontal_scheme != "centred2":
        raise ValueError("tracer_tendencies_mpas is the centred2 method-of-lines form; "
                         f"horizontal_scheme={config.horizontal_scheme!r} steps with "
                         "mpas_rk3_fct_step")
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
        validate_tracer_transport_mpas_config(self.config)
        self.adv_coefs = None
        if self.config.horizontal_scheme == "mpas_fct":
            from legoesm.core.operators_voronoi import mpas_advection_coefficients
            self.adv_coefs = mpas_advection_coefficients(mesh)

    def tendencies(self, state: TracerState) -> TracerState:
        """Compute tendencies (pure function wrapper; ``centred2`` only)."""
        if self.config.horizontal_scheme != "centred2":
            raise ValueError("tendencies() is the centred2 method-of-lines form; "
                             "mpas_fct limits inside the step (use step())")
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
        if self.config.horizontal_scheme == "mpas_fct":
            return mpas_rk3_fct_step(state, dt, self.mesh, self.sigma_coord,
                                     self.wind_fn, self.config, self.adv_coefs)

        def tendency_fn(s):
            return tracer_tendencies_mpas(
                s, self.mesh, self.sigma_coord, self.wind_fn, self.config
            )

        return dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

    # integrate() and integrate_scan() inherited from IntegrationMixin


# Largest fct_smoothing_width keeping every limiter factor >= 0: the relative
# min of (1, R) stays >= 0 for every R >= 0 iff width <= sqrt(5) - 1 (GLM
# derivation; codex reproduced reversed antidiffusive fluxes at width 8).
FCT_MAX_SMOOTHING_WIDTH = 5.0 ** 0.5 - 1.0


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
    from legoesm.core.smooth import max_rel, min_rel

    if isinstance(dt, (int, float)) and not dt > 0:
        raise ValueError(f"tracer_horizontal_flux_fct needs dt > 0 (got {dt!r})")
    if not 0.0 <= width <= FCT_MAX_SMOOTHING_WIDTH:
        raise ValueError(f"tracer_horizontal_flux_fct width must be in [0, sqrt(5)-1] "
                         f"(limiter factors stay >= 0); got {width!r}")
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
    p_in, p_out = fct_in_out_sums(fct_cell_outflow(anti, mesh, q.dtype), width)
    # rooms are >= 0: the bounds include q_td and the smooth max (min) is
    # never below (above) the hard one
    room_in = (q_max - q_td) * dp_c / dt
    room_out = (q_td - q_min) * dp_c / dt
    lim = fct_flux_factor(anti, a, q1, q2, room_in, p_in, room_out, p_out, mesh, width)
    return (q * div_fm - div(f_low + lim * anti)) / dp_c


def fct_cell_outflow(anti, mesh, dtype):
    """Per-cell-area antidiffusive flux of each edge of a cell (axis 0 over
    ``edgesOnCell``), ``> 0`` leaving the cell: ``sign dvEdge anti / area``."""
    eoc = mesh.edgesOnCell
    # in the state's precision: a mesh may store its geometry in float32
    w = (jnp.where(eoc >= 0, mesh.edgeSignOnCell, 0).astype(dtype)
         * mesh.dvEdge[jnp.maximum(eoc, 0)].astype(dtype)
         / mesh.areaCell[None, :].astype(dtype))[..., None, None]
    return w * anti[jnp.maximum(eoc, 0)]


def fct_flux_factor(anti, a, q1, q2, room_in, p_in, room_out, p_out, mesh, width,
                    mpas_eps=None):
    """Zalesak limiting factor of each edge's antidiffusive flux ``anti``
    (positive c1 -> c2): the cell factors ``R+ = min(1, room_in / p_in)`` and
    ``R- = min(1, room_out / p_out)`` (1 where no antidiffusive inflow /
    outflow), then ``min(R-[c1], R+[c2])`` for ``anti > 0`` and
    ``min(R+[c1], R-[c2])`` for ``anti < 0``.  With ``mpas_eps`` the cell
    factor is MPAS ``atm_advance_scalars_mono``'s ``min(1, max(0, room / (p +
    eps)))`` instead (no switch on ``p``; the max as a C1 positive part).
    The smooth forms (RULE 5, ``width``) are those documented on
    :func:`tracer_horizontal_flux_fct`; ``width = 0`` is the hard limiter."""
    from legoesm.core.smooth import min_rel, positive_part_rel, stable_div, step_rel

    def ratio(room, p):
        if mpas_eps is not None:
            # stable_div: no (p + eps)**-2 in the derivative (fp32 NaN at p = 0)
            r = positive_part_rel(stable_div(room, p + mpas_eps), 1.0, width)
            return min_rel(1.0, r, width)
        return jnp.where(p > 0, min_rel(1.0, room / jnp.where(p > 0, p, 1.0), width), 1.0)

    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    r_plus, r_minus = ratio(room_in, p_in), ratio(room_out, p_out)
    # donor/acceptor pair by the flux direction: c1 -> c2 for anti > 0.  Each
    # direction's factor only ramps in on its own side of anti = 0 above the
    # common lower bound, so the factor never exceeds the bound of the side
    # the flux actually takes.
    fwd = min_rel(r_minus[c1], r_plus[c2], width)
    bwd = min_rel(r_plus[c1], r_minus[c2], width)
    if width == 0:
        return jnp.where(anti >= 0, fwd, bwd)
    se = 0.5 * a * (jnp.abs(q1) + jnp.abs(q2))
    both = min_rel(fwd, bwd, width)
    return (both + step_rel(anti, se, width) * (fwd - both)
            + step_rel(-anti, se, width) * (bwd - both))


def _mpas_mass_fluxes(q, dp, u_edge, sigma_dot, mesh, sigma_coord, coefs, config):
    """Fluxes of one MPAS RK3 stage from the stage state ``(q, dp)``;
    ``sigma_dot`` on the nlev+1 interfaces, (nCells, nlev+1).

    Returns ``(fm, a, f_ho, vm, vt)``: edge air-mass flux ``fm = u dp_e``
    (``dp_e`` the two-cell mean, as MPAS ``ru``) and its smooth magnitude
    ``a`` (RULE 5, ``|fm|`` at ``fct_upwind_width_ms = 0``), (nE, nlev, 1);
    the high-order tracer-mass flux ``f_ho = fm q4 + coef_3rd a q3`` (MPAS
    ``uhAvg (adv_coefs + sign(u) coef_3rd adv_coefs_3rd) q``), (nE, nlev,
    ntr); the vertical air-mass tendency ``vm = -p_s (sdot_{k+1/2} -
    sdot_{k-1/2})`` (nC, nlev) with ``p_s = sum(dp) / sum(dsigma)``; and the vertical
    tracer-mass tendency ``vt = dp V(q) + q vm`` with ``V`` the shared
    sigma vertical advection (so a constant ``q`` stays constant)."""
    from legoesm.core.operators_voronoi import mpas_edge_reconstruction
    if sigma_dot.shape != dp.shape[:-1] + (dp.shape[-1] + 1,):
        raise ValueError("sigma_dot must be on the nlev+1 interfaces, shape "
                         f"{dp.shape[:-1] + (dp.shape[-1] + 1,)}; got {sigma_dot.shape}")
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    fm = (u_edge * 0.5 * (dp[c1] + dp[c2]))[..., None]
    if config.fct_upwind_width_ms == 0:
        a = jnp.abs(fm)
    else:
        # smooth |fm| from ABOVE, so the low-order flux 0.5 fm (q1 + q2) -
        # 0.5 a (q2 - q1) keeps non-negative donor/acceptor weights 0.5 (a +-
        # fm) (positivity under the upwind CFL); at rest it is a diffusion
        # at the speed fct_upwind_width_ms
        dp_e = 0.5 * (dp[c1] + dp[c2])
        wa = (config.fct_upwind_width_ms * jnp.where(dp_e > 0, dp_e, 1.0))[..., None]
        a = jnp.hypot(fm, wa)
    q4, q3 = mpas_edge_reconstruction(q, coefs)
    f_ho = fm * q4 + config.coef_3rd_order * a * q3
    # p = sigma p_s with a model top at sigma_half[0] > 0 (DCMIP: p_top/p0):
    # the column holds sum(dp) = p_s (1 - sigma_top) = p_s sum(dsigma)
    p_s = (jnp.sum(dp, axis=-1, keepdims=True)
           / jnp.sum(jnp.asarray(sigma_coord.dsigma, dp.dtype)))
    vm = -p_s * (sigma_dot[..., 1:] - sigma_dot[..., :-1])
    vt = dp[..., None] * tracer_vertical_advection(q, sigma_dot, sigma_coord) + q * vm[..., None]
    return fm, a, f_ho, vm, vt


def mpas_rk3_fct_step(state: TracerAirMassState, dt, mesh, sigma_coord, wind_fn,
                      config: TracerTransportMPASConfig, coefs) -> TracerAirMassState:
    """One MPAS-A scalar transport step (``atm_srk3`` RK3 + monotonic limiter).

    Wicker-Skamarock RK3, each stage from the step's start state ``(q0,
    dp0)`` over ``dt/3, dt/2, dt`` with the fluxes of the previous stage's
    state (wind at that state's time), in flux form on the prognostic air
    mass: ``dp' = dp0 + tau (-div(fm) + vm)``, ``(q dp)' = q0 dp0 + tau
    (-div(F) + vt)``.  Stages 1-2 use the high-order flux (MPAS
    ``atm_advance_scalars``).  Stage 3 is ``atm_advance_scalars_mono``
    (Wang, Skamarock & Feingold 2009, doi:10.1175/2009MWR2869.1) applied to
    the HORIZONTAL flux: first-order upwind flux from ``q0``, antidiffusive
    flux ``F_ho - F_up``, cell bounds ``s_max/s_min`` = extremes of ``q0``
    over the cell and its neighbours, scale factors ``R+ = clip((s_max dp_h
    - Q_td) / Q_in, 0, 1)``, ``R- = clip((Q_td - s_min dp_h) / Q_out, 0,
    1)`` with ``Q_td = q0 dp0 - dt div(F_up)`` and ``dp_h = dp0 - dt
    div(fm)``, and each edge flux scaled by the donor/acceptor minimum; the
    vertical tracer-mass tendency of the stage-2 state is added unlimited.

    KNOWN DIFFERENCES from MPAS-A (each recorded with its reason):
    * MPAS limits the horizontal and vertical antidiffusive fluxes jointly in
      3-D (bounds over the vertical neighbours too); here only the
      horizontal one, the vertical scheme being the shared sigma operator.
    * MPAS copies back ``max(0, q)``; dropped (it creates tracer mass, RULE
      0); the limited horizontal update is positive without it.
    * RULE 5: the hard ``min``/``max``/clips of the limiter are the smooth
      forms of :func:`fct_flux_factor` (relative band ``fct_smoothing_width``,
      the scale-factor clip ``max(0, .)`` a C1 positive part; the bounds
      ``s_max``/``s_min`` as C1 extrema taken from INSIDE the hard interval,
      so the smooth limiter never admits values beyond the hard bounds); ``|u|`` of the
      upwind split and of the 3rd-order term is ``hypot(fm, w dp_e)``,
      ``w = fct_upwind_width_ms`` (from above, so the low-order update keeps
      non-negative weights; at rest a diffusion at speed ``w``).  Both widths
      0 is the hard MPAS limiter (tested against a loop transcription).  The
      donor/acceptor blend's scale ``a (|q1| + |q2|)/2`` vanishes where both
      cells are exactly empty, leaving a one-sided derivative there (as on
      the PE dycore's limiter).  Likewise every relative band (bounds, blend,
      in/out sums) collapses where the values it compares are exactly 0, so
      the step is NOT differentiable at exact zeros of ``q`` (a 0/1 field on
      ico2, default widths: AD directional derivative 667 vs one-sided finite
      differences 297 / 549 at eps 1e-6 / 1e-8, measured); an absolute floor
      on the bands would fix it (open design question, not done).
    * The vertical tracer-mass tendency ``dp V(q) + q vm`` telescopes only if
      the shared vertical operator ``V`` is in face-minus-cell form on the
      interface mass flux ``p_s sigma_dot``; the default advective upwind
      ``V`` is not, so with vertical motion the column tracer mass is NOT
      conserved (measured: -0.50 % over DCMIP 1-1 at ico5, 12 days).  The
      horizontal transport conserves to round-off.

    Conservation: every horizontal flux is one value per edge, so the global
    tracer mass ``sum areaCell q dp`` changes only through ``vt``.
    """
    from legoesm.core.smooth import positive_part_rel

    if isinstance(dt, (int, float)) and not dt > 0:
        raise ValueError(f"mpas_rk3_fct_step needs dt > 0 (got {dt!r})")
    if not isinstance(state, TracerAirMassState):
        raise TypeError("horizontal_scheme='mpas_fct' carries the air mass: pass a "
                        "TracerAirMassState (see air_mass_state)")
    width = config.fct_smoothing_width
    t0 = state.time.data
    # working precision: the default float (fp64 under x64), whatever the
    # state stores (the DCMIP MPAS IC is float32)
    wdt = jnp.promote_types(state.tracers.data.dtype, jax.dtypes.canonicalize_dtype(jnp.float64))
    q0, dp0 = state.tracers.data.astype(wdt), state.air_mass.data.astype(wdt)
    nC, nlev, ntr = q0.shape
    nE = mesh.nEdges

    def div(f):   # (nE, nlev, k) per unit length -> (nC, nlev, k) per area
        return divergence_cell_3d(f.reshape(nE, -1), mesh).reshape(nC, nlev, -1)

    qdp0 = q0 * dp0[..., None]
    q, dp, t = q0, dp0, t0
    for tau in (dt / 3.0, dt / 2.0):
        u_edge, sigma_dot = wind_fn(t, mesh, sigma_coord)
        fm, a, f_ho, vm, vt = _mpas_mass_fluxes(q, dp, u_edge, sigma_dot, mesh,
                                                sigma_coord, coefs, config)
        dp = dp0 + tau * (-div(fm)[..., 0] + vm)
        q = (qdp0 + tau * (-div(f_ho) + vt)) / dp[..., None]
        t = t0 + tau

    # stage 3: monotonic limiter over the full step (atm_advance_scalars_mono)
    u_edge, sigma_dot = wind_fn(t, mesh, sigma_coord)
    fm, a, f_ho, vm, vt = _mpas_mass_fluxes(q, dp, u_edge, sigma_dot, mesh,
                                            sigma_coord, coefs, config)
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    q1, q2 = q0[c1], q0[c2]
    # donor/acceptor weights first (no cancellation of a small donor value);
    # at a = |fm| exactly MPAS max(0, uh) q1 + min(0, uh) q2
    f_up = 0.5 * (fm + a) * q1 - 0.5 * (a - fm) * q2
    anti = f_ho - f_up
    dp_h = dp0 - dt * div(fm)[..., 0]
    qdp_td = qdp0 - dt * div(f_up)
    coc = mesh.cellsOnCell
    ok = (coc >= 0)[..., None, None]
    nb = q0[jnp.maximum(coc, 0)]
    # smooth extrema from INSIDE the hard interval: s + pp(nb - s) with pp the
    # C1 positive part from below (exact outside its band), so q0 <= s_max <=
    # the hard max and q0 >= s_min >= the hard min.  max_rel/min_rel (from
    # outside) let a plateau creep up step after step (measured, DCMIP 1-3
    # step-function tracer, 12 d: max 1.0188 vs 1.0 hard).
    # Outside the band (and at width 0) the winning endpoint is selected
    # directly, never rebuilt as s + (nb - s) (round-off could leave the
    # neighbour interval).
    def toward(s, n, sign):          # sign +1: max side, -1: min side
        if width == 0:
            return jnp.maximum(s, n) if sign > 0 else jnp.minimum(s, n)
        x, scale = sign * (n - s), jnp.abs(s) + jnp.abs(n)
        return jnp.where(x >= width * scale, n, s + sign * positive_part_rel(x, scale, width))

    s_min, s_max = q0, q0
    for k in range(coc.shape[0]):     # a padded slot (pentagon) leaves the bound as is
        s_min = jnp.where(ok[k], toward(s_min, nb[k], -1), s_min)
        s_max = jnp.where(ok[k], toward(s_max, nb[k], 1), s_max)
    # per-step amounts, as MPAS scale_arr (its eps = 1e-20 in those units)
    p_in, p_out = fct_in_out_sums(fct_cell_outflow(dt * anti, mesh, wdt), width)
    room_in = s_max * dp_h[..., None] - qdp_td
    room_out = qdp_td - s_min * dp_h[..., None]
    lim = fct_flux_factor(anti, a, q1, q2, room_in, p_in, room_out, p_out, mesh, width,
                          mpas_eps=MPAS_MONO_EPS)
    dp_new = dp_h + dt * vm
    qdp_new = qdp_td - dt * div(lim * anti) + dt * vt
    # stored in the state's precision (as the dispatch integrators do)
    return TracerAirMassState(
        tracers=state.tracers.replace(data=(qdp_new / dp_new[..., None]).astype(state.tracers.data.dtype)),
        air_mass=state.air_mass.replace(data=dp_new.astype(state.air_mass.data.dtype)),
        time=state.time.replace(data=jnp.asarray(t0 + dt, jnp.result_type(t0))))
