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
from legoesm.grids.vertical import (
    SigmaCoordinate, check_tracer_vertical_scheme, vertical_advection,
    vertical_remap_sigma_fv3)
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
    #: "upwind" (in-tendency) | "fv3_remap": the horizontal SSP-RK step runs
    #: alone, then a split FV3 Lagrangian vertical remap
    #: (``grids.vertical.vertical_remap_sigma_fv3``, sigma_dot at mid-step).
    #: ``horizontal_scheme="mpas_fct"`` takes "mpas_flux" (MPAS-A flux-form
    #: vertical transport inside the 3-D monotone limiter) or "fv3_remap"
    #: (horizontal-only stages, then the FV3 remap of the Lagrangian layers);
    #: see :func:`mpas_rk3_fct_step`.
    vertical_scheme: str = "upwind"
    # Horizontal scheme: "centred2" (advective form, centred two-cell edge
    # average, any ``dispatch_integrator`` scheme) or "mpas_fct" (MPAS-A
    # scalar transport; needs ``time_integrator="mpas_rk3"`` and a
    # :class:`TracerAirMassState`).
    horizontal_scheme: str = "centred2"
    # MPAS ``config_coef_3rd_order`` (Registry default 0.25): 0 = 4th-order
    # centred, 1 = 3rd-order upwind edge reconstruction.
    coef_3rd_order: float = 0.25
    # RULE 5 widths of the mpas_fct limiter switches (as the PE dycore's
    # ``fct_smoothing_width`` / ``fct_upwind_width_ms``); these two and
    # ``fct_vertical_upwind_width_pas`` all 0 = hard MPAS.
    fct_smoothing_width: float = 1.0e-4
    fct_upwind_width_ms: float = 1.0e-3
    # RULE 5 width [Pa/s] of the smooth |F| of the mpas_fct vertical air-mass
    # flux (upwind split and 3rd-order term); 0 = hard MPAS.
    fct_vertical_upwind_width_pas: float = 1.0e-5


HORIZONTAL_SCHEMES = ("centred2", "mpas_fct")
MPAS_FCT_VERTICAL_SCHEMES = ("mpas_flux", "fv3_remap")
# atm_advance_scalars_mono's eps in the scale factors' denominators
MPAS_MONO_EPS = 1.0e-20
# ulps by which mpas_rk3_fct_step shrinks the limiter's cell factors (a
# floating-point margin, not a physical parameter)
FCT_ROUNDOFF_MARGIN = 16


def validate_tracer_transport_mpas_config(config: TracerTransportMPASConfig) -> None:
    """Raise on an unknown horizontal scheme or an integrator it cannot use."""
    if config.horizontal_scheme == "mpas_fct":
        if config.vertical_scheme not in MPAS_FCT_VERTICAL_SCHEMES:
            raise ValueError("horizontal_scheme='mpas_fct' takes vertical_scheme in "
                             f"{MPAS_FCT_VERTICAL_SCHEMES}; got {config.vertical_scheme!r}")
    else:
        check_tracer_vertical_scheme(config.vertical_scheme)
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
    if not (math.isfinite(config.fct_vertical_upwind_width_pas)
            and config.fct_vertical_upwind_width_pas >= 0.0):
        raise ValueError("fct_vertical_upwind_width_pas must be finite and >= 0; got "
                         f"{config.fct_vertical_upwind_width_pas!r}")


class TracerAirMassState(NamedTuple):
    """Prescribed-wind tracer state with a prognostic layer air mass.

    ``tracers`` (nCells, nlev, ntr) mixing ratios, ``air_mass`` (nCells,
    nlev) layer pressure thickness ``dp`` [Pa] (the MPAS ``rho_zz`` analogue),
    ``time`` [s].  The tracer mass of a layer is ``q dp areaCell / g``."""
    tracers: Field
    air_mass: Field
    time: Field


def air_mass_state(state: TracerState, p_s, sigma_coord: SigmaCoordinate) -> TracerAirMassState:
    """Attach the layer air mass ``dp = p_s dsigma`` [Pa] of a sigma column,
    or ``dp = dA p_ref + dB p_s`` of a hybrid one (``p_s`` [Pa], scalar or
    (nCells,)), to a :class:`TracerState`.

    Hybrid precondition: ``p_s`` must exceed the level set's minimum valid
    surface pressure everywhere, so every layer mass is positive.  Concrete
    (eager) input violating it raises ``ValueError``; traced input (under
    ``jit``/``grad``) is not checked here and must be validated by the caller.
    """
    from legoesm.grids.vertical import HybridSigmaPressureCoordinate, dp_from_hybrid
    q = state.tracers.data
    p_s = jnp.broadcast_to(jnp.asarray(p_s, q.dtype), q.shape[:1])
    if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
        from legoesm.grids.vertical import hybrid_min_valid_surface_pressure
        # dA p_ref + dB p_s > 0 only for p_s strictly above the shared
        # threshold: refuse a collapsed/inverted layer.  Host check on
        # concrete input (every caller initialises eagerly); traced input
        # keeps the calculation traceable and is validated by its caller
        p_min = hybrid_min_valid_surface_pressure(
            sigma_coord.A_half, sigma_coord.B_half, sigma_coord.p_ref)
        dp = dp_from_hybrid(sigma_coord, p_s).astype(q.dtype)
        if not isinstance(dp, jax.core.Tracer) and not bool(
                jnp.all(jnp.isfinite(p_s)) and jnp.all(p_s > p_min)
                and jnp.all(jnp.isfinite(dp)) and jnp.all(dp > 0)):
            raise ValueError(f"hybrid layer air mass dA p_ref + dB p_s must be strictly "
                             f"positive: p_s must exceed {p_min:.6g} Pa")
    else:
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
    check_tracer_vertical_scheme(config.vertical_scheme)
    t = state.time.data  # scalar time
    q = state.tracers.data  # (nCells, nlev, n_tracers)

    # Get prescribed winds at current time
    u_edge, sigma_dot = wind_fn(t, mesh, sigma_coord)  # (nEdges, nlev), (nCells, nlev+1)

    if config.vertical_scheme == "fv3_remap":
        dq_dt = tracer_horizontal_advection(q, u_edge, mesh)
    else:
        dq_dt = tracer_advection_tendency(q, u_edge, sigma_dot, mesh,
                                          sigma_coord)

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
        *,
        mass_flux=None,
    ):
        self.mesh = mesh
        self.mass_flux = mass_flux
        self.sigma_coord = sigma_coord
        self.wind_fn = wind_fn
        self.config = config or TracerTransportMPASConfig()
        validate_tracer_transport_mpas_config(self.config)
        self.adv_coefs = None
        if self.config.horizontal_scheme == "mpas_fct":
            from legoesm.core.operators_voronoi import mpas_advection_coefficients
            self.adv_coefs = mpas_advection_coefficients(mesh)
        if mass_flux is not None and (self.config.horizontal_scheme != "mpas_fct"
                                      or self.config.vertical_scheme != "mpas_flux"):
            raise ValueError("mass_flux runs with horizontal_scheme='mpas_fct' and "
                             "vertical_scheme='mpas_flux'")

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
                                     self.wind_fn, self.config, self.adv_coefs,
                                     self.mass_flux)

        def tendency_fn(s):
            return tracer_tendencies_mpas(
                s, self.mesh, self.sigma_coord, self.wind_fn, self.config
            )

        new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )
        if self.config.vertical_scheme != "fv3_remap":
            return new
        _, sigma_dot = self.wind_fn(state.time.data + 0.5 * dt, self.mesh,
                                    self.sigma_coord)
        q = vertical_remap_sigma_fv3(
            new.tracers.data, sigma_dot, self.sigma_coord, dt)
        return new._replace(tracers=new.tracers.replace(data=q))

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
    can leave the range of ``q`` there); the neighbour bounds are extrema
    taken from INSIDE the hard interval, C1 at ``width > 0`` except where
    both compared values are exactly 0 (see the helper) (:func:`fct_bound_toward`, between
    ``q_td`` and the hard max/min; changed 2026-11-24 from the outside
    ``max_rel``/``min_rel``, which let a 0/1 plateau creep to 1.0081 in 200
    steps); the clips ``min(1, R)`` are ``core.smooth`` min with a band
    ``width`` relative to the operands; the inflow/outflow sums are a C1 positive part from above
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
    # smooth extrema from INSIDE the hard interval (fct_bound_toward), folded
    # from q_td so the rooms stay >= 0; max_rel/min_rel (from outside) let a
    # 0/1 plateau creep up (measured: 1.0081 after 200 steps, 1.0 now)
    lo, hi = fct_bound_toward(q_td, q, -1, width), fct_bound_toward(q_td, q, 1, width)
    nb_lo, nb_hi = jnp.where(ok, lo[coc_s], lo), jnp.where(ok, hi[coc_s], hi)
    q_min, q_max = lo, hi
    for k in range(coc.shape[0]):
        q_min = fct_bound_toward(q_min, nb_lo[k], -1, width)
        q_max = fct_bound_toward(q_max, nb_hi[k], 1, width)
    p_in, p_out = fct_in_out_sums(fct_cell_outflow(anti, mesh, q.dtype), width)
    # rooms are >= 0: the bounds are folded from q_td and only move outward
    # from it (never past the hard extremes)
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


def fct_cell_factors(room_in, p_in, room_out, p_out, width, mpas_eps=None):
    """Zalesak cell factors ``(R+, R-)``: ``R+ = min(1, room_in / p_in)``,
    ``R- = min(1, room_out / p_out)`` (1 where no antidiffusive inflow /
    outflow).  With ``mpas_eps`` the factor is MPAS
    ``atm_advance_scalars_mono``'s ``min(1, max(0, room / (p + eps)))``
    instead (no switch on ``p``; the max as a C1 positive part).  Smooth
    forms (RULE 5, ``width``) as documented on
    :func:`tracer_horizontal_flux_fct`; ``width = 0`` is the hard limiter."""
    from legoesm.core.smooth import min_rel, positive_part_rel, stable_div

    def ratio(room, p):
        if mpas_eps is not None:
            # stable_div: no (p + eps)**-2 in the derivative (fp32 NaN at p = 0)
            r = positive_part_rel(stable_div(room, p + mpas_eps), 1.0, width)
            return min_rel(1.0, r, width)
        return jnp.where(p > 0, min_rel(1.0, room / jnp.where(p > 0, p, 1.0), width), 1.0)

    return ratio(room_in, p_in), ratio(room_out, p_out)


def fct_pair_factor(anti, a, q1, q2, r_plus_1, r_minus_1, r_plus_2, r_minus_2, width):
    """Limiting factor of an antidiffusive flux ``anti`` between a cell pair
    (positive 1 -> 2): ``min(R-_1, R+_2)`` for ``anti > 0`` and ``min(R+_1,
    R-_2)`` for ``anti < 0``; ``a`` the flux's smooth magnitude scale and
    ``q1, q2`` the pair's values (the RULE 5 direction blend)."""
    from legoesm.core.smooth import min_rel, step_rel
    # donor/acceptor pair by the flux direction: 1 -> 2 for anti > 0.  Each
    # direction's factor only ramps in on its own side of anti = 0 above the
    # common lower bound, so the factor never exceeds the bound of the side
    # the flux actually takes.
    fwd = min_rel(r_minus_1, r_plus_2, width)
    bwd = min_rel(r_plus_1, r_minus_2, width)
    if width == 0:
        return jnp.where(anti >= 0, fwd, bwd)
    se = 0.5 * a * (jnp.abs(q1) + jnp.abs(q2))
    both = min_rel(fwd, bwd, width)
    return (both + step_rel(anti, se, width) * (fwd - both)
            + step_rel(-anti, se, width) * (bwd - both))


def fct_flux_factor(anti, a, q1, q2, room_in, p_in, room_out, p_out, mesh, width,
                    mpas_eps=None):
    """Zalesak limiting factor of each edge's antidiffusive flux ``anti``
    (positive c1 -> c2): :func:`fct_cell_factors` per cell, then
    :func:`fct_pair_factor` over ``cellsOnEdge``."""
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    r_plus, r_minus = fct_cell_factors(room_in, p_in, room_out, p_out, width, mpas_eps)
    return fct_pair_factor(anti, a, q1, q2, r_plus[c1], r_minus[c1], r_plus[c2],
                           r_minus[c2], width)


def fct_bound_toward(s, n, sign, width):
    """Running FCT bound folded toward ``n`` (``sign`` +1: max, -1: min),
    from INSIDE the hard interval.  ``width = 0`` is the hard max/min (a kink
    at every tie); for ``width > 0`` it is C1 except where ``|s| + |n| = 0`` (both
    operands exactly zero: the band collapses and the hard max/min is taken,
    e.g. ``max(s, 0)`` at an empty-cell tie) and, for widths > 1, where an
    operand crosses zero inside the ramp (the ``|.|`` of the scale): ``s + sign pp(sign (n - s))`` with
    ``pp`` the C1 positive part from below (:func:`positive_part_rel`, band
    relative to ``|s| + |n|``), so the result lies between ``s`` and the
    hard ``max/min(s, n)``; outside the band (and at ``width = 0``) the
    winning endpoint is selected directly, never rebuilt as ``s + (n - s)``
    (round-off could leave the interval).  Outside-smoothed bounds
    (``max_rel``/``min_rel``) let a 0/1 plateau creep up step after step
    (measured on DCMIP 1-3 q3, 12 d: max 1.0188 vs 1.0 hard)."""
    from legoesm.core.smooth import positive_part_rel
    if width == 0:
        return jnp.maximum(s, n) if sign > 0 else jnp.minimum(s, n)
    x, scale = sign * (n - s), jnp.abs(s) + jnp.abs(n)
    return jnp.where(x >= width * scale, n, s + sign * positive_part_rel(x, scale, width))


def mpas_vertical_mass_flux(div_h, sigma_coord):
    """Interface air-mass fluxes ``F`` (..., nlev+1) [Pa/s], positive toward
    larger sigma (downward), from the DISCRETE column continuity of the sigma
    coordinate with the horizontal air-mass divergence ``div_h`` (..., nlev)
    [Pa/s] the scheme actually applies: the layer tendency ``-div_h_k + F_k
    - F_{k+1}`` equals ``dsigma_k dp_s/dt`` with ``dp_s/dt sum(dsigma) =
    -sum_k div_h_k``, and ``F = 0`` at the top and the surface (no air
    leaves the column).  So a state with ``dp = p_s dsigma`` keeps that
    profile at every stage (an off-profile offset is carried unchanged), and
    the column air mass changes only by the horizontal divergence."""
    dsig = jnp.asarray(sigma_coord.dsigma, div_h.dtype)
    tend_ps = -jnp.sum(div_h, axis=-1, keepdims=True) / jnp.sum(dsig)
    inner = -jnp.cumsum(div_h[..., :-1] + dsig[:-1] * tend_ps, axis=-1)
    zero = jnp.zeros_like(div_h[..., :1])
    return jnp.concatenate([zero, inner, zero], axis=-1)


def mpas_vertical_flux_ho(q, F, a_v, sigma_coord, coef3):
    """MPAS-A high-order vertical tracer flux at the nlev+1 interfaces
    (``atm_advance_scalars`` / ``_mono`` ``wdtn``), positive downward:
    interior interfaces ``flux3`` (4th-order ``F (7 (q_{i-1} + q_i) -
    (q_{i-2} + q_{i+1})) / 12`` plus ``coef3 |F| ((q_{i+1} - q_{i-2}) - 3
    (q_i - q_{i-1})) / 12``, the uniform-grid formula MPAS applies), the
    interfaces next to the top and the surface layer the linear
    interpolation ``F (dsigma_i q_{i-1} + dsigma_{i-1} q_i) / (dsigma_{i-1}
    + dsigma_i)`` (MPAS ``fzm/fzp``, ``config_interface_projection =
    linear_interpolation``), zero at top and surface.  ``|F|`` is the smooth
    ``a_v`` (RULE 5).  ``q`` (nC, nlev, ntr); ``F``, ``a_v`` (nC, nlev+1, 1)."""
    nlev = q.shape[1]
    dsig = jnp.asarray(sigma_coord.dsigma, q.dtype)[None, :, None]
    # linear interpolation at every interior interface i = 1..nlev-1
    lin = (dsig[:, 1:] * q[:, :-1] + dsig[:, :-1] * q[:, 1:]) / (dsig[:, :-1] + dsig[:, 1:])
    g = F[:, 1:-1] * lin
    if nlev >= 4:
        qm2, qm1, q0_, qp1 = q[:, :-3], q[:, 1:-2], q[:, 2:-1], q[:, 3:]   # i = 2..nlev-2
        f3 = (F[:, 2:-2] * (7.0 * (qm1 + q0_) - (qm2 + qp1)) / 12.0
              + coef3 * a_v[:, 2:-2] * ((qp1 - qm2) - 3.0 * (q0_ - qm1)) / 12.0)
        g = jnp.concatenate([g[:, :1], f3, g[:, -1:]], axis=1)
    zero = jnp.zeros_like(q[:, :1])
    return jnp.concatenate([zero, g, zero], axis=1)



def _mpas_stage_fluxes(q, dp, u_edge, mesh, sigma_coord, coefs, config, mass_flux=None):
    """Fluxes of one MPAS RK3 stage from the stage state ``(q, dp)``.

    Returns ``(fm, a, f_ho, div_h, F, a_v, g_ho)``: edge air-mass flux ``fm
    = u dp_e`` (``dp_e`` the two-cell mean, as MPAS ``ru``) and its smooth
    magnitude ``a`` (RULE 5, ``|fm|`` at ``fct_upwind_width_ms = 0``), (nE,
    nlev, 1); the high-order horizontal tracer-mass flux ``f_ho = fm q4 +
    coef_3rd a q3`` (MPAS ``uhAvg (adv_coefs + sign(u) coef_3rd
    adv_coefs_3rd) q``), (nE, nlev, ntr); ``div_h`` = div(fm) (nC, nlev);
    the interface air-mass flux ``F`` of :func:`mpas_vertical_mass_flux`
    and its smooth magnitude ``a_v`` (nC, nlev+1, 1); and the high-order
    vertical tracer flux ``g_ho`` (nC, nlev+1, ntr).  The last three are
    ``None`` for ``vertical_scheme="fv3_remap"``, whose stages are
    horizontal only.

    ``mass_flux`` (a steady :class:`~legoesm.atmosphere.dynamics.gcm.
    consistent_mass_flux.PrescribedMassFlux`): ``fm`` and ``F`` are its
    consistent edge and interface air fluxes (``u_edge`` unused), ``F``
    open at the top and the surface, where the tracer flux (high and low
    order alike) is ``F`` times the adjacent layer's value."""
    from legoesm.core.operators_voronoi import mpas_edge_reconstruction
    nC, nlev = dp.shape
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    dp_e = 0.5 * (dp[c1] + dp[c2])
    if mass_flux is None:
        fm = (u_edge * dp_e)[..., None]
    else:
        fm = mass_flux.flux_h.astype(dp.dtype)[..., None]
    if config.fct_upwind_width_ms == 0:
        a = jnp.abs(fm)
    else:
        # smooth |fm| from ABOVE, so the low-order flux 0.5 fm (q1 + q2) -
        # 0.5 a (q2 - q1) keeps non-negative donor/acceptor weights 0.5 (a +-
        # fm) (positivity under the upwind CFL); at rest it is a diffusion
        # at the speed fct_upwind_width_ms
        wa = (config.fct_upwind_width_ms * jnp.where(dp_e > 0, dp_e, 1.0))[..., None]
        a = jnp.hypot(fm, wa)
    q4, q3 = mpas_edge_reconstruction(q, coefs)
    f_ho = fm * q4 + config.coef_3rd_order * a * q3
    div_h = divergence_cell_3d(fm[..., 0], mesh)
    if config.vertical_scheme != "mpas_flux":
        return fm, a, f_ho, div_h, None, None, None
    if mass_flux is None:
        F = mpas_vertical_mass_flux(div_h, sigma_coord)[..., None]
    else:
        F = mass_flux.w.astype(dp.dtype)[..., None]
    if config.fct_vertical_upwind_width_pas == 0:
        a_v = jnp.abs(F)
    else:   # smooth |F| from above (as a); zero at top and surface
        a_v = jnp.hypot(F, config.fct_vertical_upwind_width_pas)
        a_v = a_v.at[:, 0].set(0.0).at[:, -1].set(0.0)
    g_ho = mpas_vertical_flux_ho(q, F, a_v, sigma_coord, config.coef_3rd_order)
    if mass_flux is not None:
        g_ho = g_ho.at[:, 0].set(F[:, 0] * q[:, 0]).at[:, -1].set(F[:, -1] * q[:, -1])
    return fm, a, f_ho, div_h, F, a_v, g_ho


def _cell_bounds(q0, mesh, width, vertical):
    """MPAS ``s_min``/``s_max``: extremes of ``q0`` over the cell, its
    horizontal neighbours at the same level and (``vertical``) the layers
    directly above and below (``atm_advance_scalars_mono_work``), folded
    with :func:`fct_bound_toward`; a padded (pentagon) slot leaves the bound."""
    coc = mesh.cellsOnCell
    ok = (coc >= 0)[..., None, None]
    nb = q0[jnp.maximum(coc, 0)]
    s_min, s_max = q0, q0
    if vertical:
        up = jnp.concatenate([q0[:, :1], q0[:, :-1]], axis=1)    # top layer: itself
        dn = jnp.concatenate([q0[:, 1:], q0[:, -1:]], axis=1)    # surface layer: itself
        for n in (up, dn):
            s_min, s_max = fct_bound_toward(s_min, n, -1, width), fct_bound_toward(s_max, n, 1, width)
    for k in range(coc.shape[0]):
        s_min = jnp.where(ok[k], fct_bound_toward(s_min, nb[k], -1, width), s_min)
        s_max = jnp.where(ok[k], fct_bound_toward(s_max, nb[k], 1, width), s_max)
    return s_min, s_max


def mpas_rk3_fct_step(state: TracerAirMassState, dt, mesh, sigma_coord, wind_fn,
                      config: TracerTransportMPASConfig, coefs,
                      mass_flux=None) -> TracerAirMassState:
    """One MPAS-A scalar transport step (``atm_srk3`` RK3 + monotonic limiter).

    Wicker-Skamarock RK3 as ``atm_srk3`` / ``atm_advance_scalars_work``: each
    stage from the step's start state ``(q0, dp0)`` over ``dt/3, dt/2, dt``
    with the fluxes of the previous stage's state (wind at that state's
    time), in flux form on the prognostic layer air mass.  By default
    (``mass_flux=None``) the vertical air-mass flux ``F`` is NOT the wind
    function's ``sigma_dot`` (ignored here): it is diagnosed each stage from the discrete column continuity of
    the horizontal mass flux actually applied
    (:func:`mpas_vertical_mass_flux`, zero at top and surface), so the air
    mass is exactly compatible with the transport (user decision
    2026-11-24; the analytic DCMIP 1-3 sigma_dot drove dp to zero).

    ``vertical_scheme="mpas_flux"`` (MPAS-A): ``dp' = dp0 + tau (-div(fm) +
    F_k - F_{k+1})``, ``(q dp)' = q0 dp0 + tau (-div(f_ho) - (G_{k+1} -
    G_k))`` with the high-order vertical flux ``G`` of
    :func:`mpas_vertical_flux_ho`.  Stage 3 is ``atm_advance_scalars_mono``
    (Wang, Skamarock & Feingold 2009, doi:10.1175/2009MWR2869.1) in 3-D:
    first-order upwind horizontal AND vertical fluxes from ``q0`` with the
    stage mass fluxes, antidiffusive fluxes ``f_ho - f_up`` and ``G_ho -
    G_up``, bounds ``s_max/s_min`` = extremes of ``q0`` over the cell, its
    horizontal neighbours and the layers above and below, ONE in/out budget
    per cell from both flux sets, ``R+ = clip((s_max dp_new - Q_td) / Q_in,
    0, 1)``, ``R- = clip((Q_td - s_min dp_new) / Q_out, 0, 1)`` with ``Q_td
    = q0 dp0 - dt (div(f_up) + delta G_up)``, each edge flux scaled by its
    cell pair's minimum and each interface flux by its layer pair's.

    ``vertical_scheme="fv3_remap"`` (opt-in comparison, FV3 style): the
    three stages are horizontal only (``dp`` moves by ``-div(fm)``,
    Lagrangian layers), stage 3 is the horizontal limiter (bounds over the
    cell and its horizontal neighbours, ``dp_h = dp0 - dt div(fm)``), then
    the column is remapped from the Lagrangian layers ``dp_h`` to the
    target layers ``dp_new = p_s' dsigma``, ``p_s' = sum(dp_h) / sum(dsigma)``
    (returned as the new air mass, whatever profile ``dp0`` had) with the
    shared FV3 tracer remap
    (:func:`legoesm.grids.vertical.vertical_remap_sigma_fv3`, kord 9) in the
    normalised coordinate ``sigma_top + cumsum(dp_h) / p_s'``.

    ``mass_flux`` (steady prescribed air fluxes of
    :func:`legoesm.atmosphere.dynamics.gcm.consistent_mass_flux.
    consistent_mass_flux_mpas`, DCMIP 1-3; needs ``vertical_scheme=
    "mpas_flux"``): the edge flux ``fm`` and the interface flux ``F`` are
    the prescribed consistent ones (``div fm + delta F = 0`` to round-off,
    so ``dp`` stays at its steady value and a uniform tracer stays uniform;
    the wind function is not called; ponytail: the state's ``air_mass`` must
    equal ``mass_flux.dp``, not checked inside the traced step -- the matrix
    row builds both from one ``dp``); a nonzero ``F`` at the top or the
    surface is an open boundary, where the tracer flux is ``F`` times the
    adjacent layer's ``q`` in the low-order flux (no antidiffusive part
    there), and the tracer mass changes only by that boundary flux (the
    DCMIP 1-3 row's fluxes are closed there: zero).

    KNOWN DIFFERENCES from MPAS-A (each recorded with its reason):
    * MPAS copies back ``max(0, q)``; dropped (it creates tracer mass, RULE
      0); instead the cell factors carry a ``FCT_ROUNDOFF_MARGIN``-ulp
      margin on the antidiffusive correction (measured: exactly
      non-negative on the tested fields; bounds hold to round-off).
    * MPAS takes ``wwAvg`` from its dynamics; here the prescribed-wind model
      diagnoses ``F`` from the column continuity above (DCMIP's analytic
      ``sigma_dot`` is not used).
    * RULE 5: the hard ``min``/``max``/clips of the limiter are the smooth
      forms of :func:`fct_cell_factors` / :func:`fct_pair_factor` (relative
      band ``fct_smoothing_width``, the scale-factor clip ``max(0, .)`` a C1
      positive part; the bounds as C1 extrema taken from INSIDE the hard
      interval, :func:`fct_bound_toward`); ``|u|`` of the upwind split and
      of the 3rd-order term is ``hypot(fm, w dp_e)``, ``w =
      fct_upwind_width_ms``, and ``|F|`` is ``hypot(F,
      fct_vertical_upwind_width_pas)`` (from above, so the low-order update
      keeps non-negative weights; at rest a diffusion at that speed).  All
      widths 0 is the hard MPAS limiter (tested against a loop
      transcription).  Every relative band collapses where the values it
      compares are exactly 0 (both operands zero, e.g. two empty cells),
      where the switch is the hard one: there the step CAN be
      non-differentiable (flux products or cancellation may still remove the
      kink), and AD returns the derivative of the branch taken, which can
      differ from the one-sided directional derivatives (a 0/1 field on ico2, default widths:
      AD directional derivative 667 vs one-sided finite differences 297 /
      549 at eps 1e-6 / 1e-8, measured).  The direction blend's scale
      ``a (|q1| + |q2|) / 2`` also has a kink where ONE of the pair's values
      is exactly 0, even with the band open, so one-sided derivatives can
      differ there too.  User decision 2026-11-24: no absolute floor.

    Positivity / monotonicity of the low-order (upwind) update, and so of the
    limited one, need in every cell at every stage ``tau * (total donor
    outflow of that stage's fluxes, horizontal plus vertical, smooth
    magnitudes included) < dp0`` (the update's base, ``tau = dt/3, dt/2,
    dt``) with a margin of a few ulps (the low-order update is the
    subtraction ``q0 dp0 - dt (div f_up + delta g_up)``, which can round
    below 0 when the outflow is within round-off of ``dp0``; the limiter's
    margin below protects only the antidiffusive correction), so every
    stage and final layer mass stays strictly positive; not enforced (as in MPAS-A, where ``dt`` is the user's).  The limiter's cell
    factors are shrunk by ``FCT_ROUNDOFF_MARGIN`` ulps, which keeps the
    antidiffusive correction from spending more than a cell's room through
    round-off (a 0/1 field stays >= 0 exactly; -3.5e-18 without it,
    measured); local bounds in general hold to round-off.  MPAS's max(0, q)
    copy-back, which creates mass, is dropped.

    Conservation (RULE 0): every horizontal flux is one value per edge and
    every vertical flux one value per interface (zero at top and surface),
    so the global tracer mass ``sum areaCell q dp`` is conserved to
    round-off, and the global air mass likewise; the FV3 remap conserves
    the column tracer mass over the Lagrangian layers.
    """
    from legoesm.grids.vertical import vertical_remap_sigma_fv3

    if isinstance(dt, (int, float)) and not dt > 0:
        raise ValueError(f"mpas_rk3_fct_step needs dt > 0 (got {dt!r})")
    if not isinstance(state, TracerAirMassState):
        raise TypeError("horizontal_scheme='mpas_fct' carries the air mass: pass a "
                        "TracerAirMassState (see air_mass_state)")
    if config.vertical_scheme not in MPAS_FCT_VERTICAL_SCHEMES:
        raise ValueError(f"mpas_rk3_fct_step: vertical_scheme {config.vertical_scheme!r}; "
                         f"expected one of {MPAS_FCT_VERTICAL_SCHEMES}")
    flux_vertical = config.vertical_scheme == "mpas_flux"
    if mass_flux is not None and not flux_vertical:
        raise ValueError("mass_flux needs vertical_scheme='mpas_flux'")
    width = config.fct_smoothing_width
    t0 = state.time.data
    # working precision: the default float (fp64 under x64), whatever the
    # state stores
    wdt = jnp.promote_types(state.tracers.data.dtype, jax.dtypes.canonicalize_dtype(jnp.float64))
    q0, dp0 = state.tracers.data.astype(wdt), state.air_mass.data.astype(wdt)
    nC, nlev, ntr = q0.shape
    nE = mesh.nEdges

    def div(f):   # (nE, nlev, k) per unit length -> (nC, nlev, k) per area
        return divergence_cell_3d(f.reshape(nE, -1), mesh).reshape(nC, nlev, -1)

    def vdiv(g):  # interface flux (nC, nlev+1, k), positive down -> layer net outflow
        return g[:, 1:] - g[:, :-1]

    qdp0 = q0 * dp0[..., None]
    q, dp, t = q0, dp0, t0
    for tau in (dt / 3.0, dt / 2.0):
        u_edge = None if mass_flux is not None else wind_fn(t, mesh, sigma_coord)[0]
        fm, a, f_ho, div_h, F, a_v, g_ho = _mpas_stage_fluxes(q, dp, u_edge, mesh, sigma_coord,
                                                              coefs, config, mass_flux)
        if flux_vertical:
            dp = dp0 + tau * (-div_h - vdiv(F)[..., 0])
            q = (qdp0 + tau * (-div(f_ho) - vdiv(g_ho))) / dp[..., None]
        else:
            dp = dp0 - tau * div_h
            q = (qdp0 - tau * div(f_ho)) / dp[..., None]
        t = t0 + tau

    # stage 3: monotonic limiter over the full step (atm_advance_scalars_mono)
    u_edge = None if mass_flux is not None else wind_fn(t, mesh, sigma_coord)[0]
    fm, a, f_ho, div_h, F, a_v, g_ho = _mpas_stage_fluxes(q, dp, u_edge, mesh, sigma_coord,
                                                          coefs, config, mass_flux)
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    q1, q2 = q0[c1], q0[c2]
    # donor/acceptor weights first (no cancellation of a small donor value);
    # at a = |fm| exactly MPAS max(0, uh) q1 + min(0, uh) q2
    f_up = 0.5 * (fm + a) * q1 - 0.5 * (a - fm) * q2
    anti = f_ho - f_up
    out_h = fct_cell_outflow(dt * anti, mesh, wdt)           # per-step amounts, as MPAS
    if flux_vertical:
        qa, qb = q0[:, :-1], q0[:, 1:]                       # layer above / below interface
        g_up = 0.5 * (F[:, 1:-1] + a_v[:, 1:-1]) * qa - 0.5 * (a_v[:, 1:-1] - F[:, 1:-1]) * qb
        zero = jnp.zeros_like(q0[:, :1])
        if mass_flux is None:
            g_up = jnp.concatenate([zero, g_up, zero], axis=1)
        else:   # open top / surface: F times the adjacent layer (low order only;
            # ponytail: first order in time there (step-start q0); the RK
            # stage value + limited correction is the ceiling.  The DCMIP 1-3
            # row's fluxes are closed (F = 0) at both boundaries)
            g_up = jnp.concatenate([F[:, :1] * q0[:, :1], g_up, F[:, -1:] * q0[:, -1:]],
                                   axis=1)
        anti_v = g_ho - g_up
        if mass_flux is not None:
            # the open top / surface carry the low-order flux F q0 only
            anti_v = anti_v.at[:, 0].set(0.0).at[:, -1].set(0.0)
        dp_new = dp0 + dt * (-div_h - vdiv(F)[..., 0])
        dp_lim = dp_new
        qdp_td = qdp0 - dt * (div(f_up) + vdiv(g_up))
        # one budget per cell: horizontal edges, then the top and bottom faces
        # (> 0 leaves the layer: up through its top if anti_v_k < 0, down
        # through its bottom if anti_v_{k+1} > 0)
        out_v = jnp.stack([-dt * anti_v[:, :-1], dt * anti_v[:, 1:]])
        p_in, p_out = fct_in_out_sums(jnp.concatenate([out_h, out_v], axis=0), width)
    else:
        dp_lim = dp0 - dt * div_h
        qdp_td = qdp0 - dt * div(f_up)
        p_in, p_out = fct_in_out_sums(out_h, width)
    s_min, s_max = _cell_bounds(q0, mesh, width, flux_vertical)
    room_in = s_max * dp_lim[..., None] - qdp_td
    room_out = qdp_td - s_min * dp_lim[..., None]
    r_plus, r_minus = fct_cell_factors(room_in, p_in, room_out, p_out, width,
                                       mpas_eps=MPAS_MONO_EPS)
    # floating-point margin (FCT_ROUNDOFF_MARGIN ulps) on the antidiffusive
    # correction only: its limited in/out sums then do not spend more than a
    # cell's room through round-off (measured: a 0/1 field stays >= 0
    # exactly, where -3.5e-18 appeared without it).  Local bounds in general
    # hold to round-off; the low-order update is not covered (see the
    # docstring).  Replaces MPAS's mass-creating max(0, q) copy-back.
    safety = 1.0 - FCT_ROUNDOFF_MARGIN * jnp.finfo(wdt).eps
    r_plus, r_minus = r_plus * safety, r_minus * safety
    lim = fct_pair_factor(anti, a, q1, q2, r_plus[c1], r_minus[c1], r_plus[c2],
                          r_minus[c2], width)
    qdp_new = qdp_td - dt * div(lim * anti)
    if flux_vertical:
        lim_v = fct_pair_factor(anti_v[:, 1:-1], a_v[:, 1:-1], qa, qb,
                                r_plus[:, :-1], r_minus[:, :-1], r_plus[:, 1:], r_minus[:, 1:],
                                width)
        lim_anti_v = jnp.concatenate([zero, lim_v * anti_v[:, 1:-1], zero], axis=1)
        qdp_new = qdp_new - dt * vdiv(lim_anti_v)
        q_new = qdp_new / dp_new[..., None]
    else:
        q_h = qdp_new / dp_lim[..., None]
        # remap the Lagrangian layers dp_lim onto dp_new in the normalised
        # coordinate s = sigma_top + cumsum(dp) / p_s'.  Target interfaces
        # sigma_top + cumsum(dsigma): the coordinate's dsigma, which dp_new is
        # proportional to (its stored sigma_half may differ from them by
        # float32 rounding, which would break the column budget at 1e-8).
        dsig = jnp.asarray(sigma_coord.dsigma, wdt)
        s0 = jnp.asarray(sigma_coord.sigma_half, wdt)[:1]
        s_tgt = jnp.concatenate([s0, s0 + jnp.cumsum(dsig)])
        p_s_new = jnp.sum(dp_lim, axis=-1, keepdims=True) / jnp.sum(dsig)
        s_src = s0 + jnp.concatenate(
            [jnp.zeros_like(dp_lim[:, :1]), jnp.cumsum(dp_lim, axis=-1)], axis=-1) / p_s_new
        disp = (s_src - s_tgt).at[:, 0].set(0.0).at[:, -1].set(0.0)
        q_new = vertical_remap_sigma_fv3(q_h, disp / dt, sigma_coord._replace(sigma_half=s_tgt), dt)
        # the state returns on the remap's target layers (FV3: back to the
        # reference coordinate), so sum(q dp) is the remapped column mass
        # even for a state whose dp0 is off the p_s dsigma profile
        dp_new = p_s_new * dsig[None, :]
    # stored in the state's precision
    return TracerAirMassState(
        tracers=state.tracers.replace(data=q_new.astype(state.tracers.data.dtype)),
        air_mass=state.air_mass.replace(data=dp_new.astype(state.air_mass.data.dtype)),
        time=state.time.replace(data=jnp.asarray(t0 + dt, jnp.result_type(t0))))
