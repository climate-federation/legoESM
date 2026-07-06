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


def advective_tracer_tendency(q, u, v, grid, vertical_fn, *, hyperdiff_coeff=0.0,
                              horizontal=True):
    """Advective-form tracer tendency on the cubed sphere, batched over tracers.

    The SHARED cubed-sphere tracer-advection core: the moisture/tracer transport
    numerics live in ONE place, reused by the prescribed-wind
    :class:`TracerTransportModel` (:func:`tracer_tendencies`) AND the FV3
    hydrostatic dycore's moist tracer transport (``fv3_hydrostatic_tendencies``).
    Advective (not flux) form, consistent with how the FV3 hydrostatic core
    transports temperature.

    Parameters
    ----------
    q : jax.Array, shape ``(6, n, n, nlev, n_tracers)``
        Tracer mixing ratios, packed along a trailing tracer axis.
    u, v : jax.Array, shape ``(6, n, n, nlev)``
        Cell-centre grid-aligned wind components (shared by all tracers).
    grid : CubedSphereGrid
    vertical_fn : callable
        Per-tracer vertical advection of a single ``(6, n, n, nlev)`` field with
        the caller's vertical velocity bound, e.g.
        ``lambda q1: vertical_advection(q1, sigma_dot, sigma_coord)`` (sigma) or
        ``lambda q1: vertical_advection_hybrid(q1, mass_flux, p_s, sigma_coord)``
        (hybrid).  ``vmap``-ped over the trailing tracer axis here.
    hyperdiff_coeff : float, optional
        Biharmonic hyperdiffusion coefficient (>0 enables it); shares the single
        batched halo pad with the gradient stencils.
    horizontal : bool, optional
        When ``False``, SKIP the horizontal advective ``-(u·∇q)`` term (and the
        horizontal hyperdiffusion) and return ONLY the vertical tendency.  Used
        by the #771 flux-form moisture path, which transports horizontally with
        the mass-conserving :func:`flux_form_tracer_step` as a post-RK3 substep
        instead of this advective, non-conserving horizontal form.  The vertical
        advection (already mass-flux-consistent) stays in the RK3 tendency.

    Returns
    -------
    jax.Array, shape ``(6, n, n, nlev, n_tracers)``
        ``-(u·∇x q + v·∇y q) + vertical + hyperdiff`` (or ``vertical`` alone when
        ``horizontal=False``).  The tracer axis is folded into the level axis so
        the cubed-sphere halo + gradient/hyperdiff stencils run in a SINGLE
        ``pad_halo_4d`` exchange instead of one per tracer (multi-GPU MPI
        exchange dominates these per-level operators).
    """
    # Vertical advection — local stencil along the level axis, no halo cost.
    # vmap over the trailing tracer axis (vertical velocity captured in the
    # closure) so JAX emits a single batched kernel.
    vert = jax.vmap(vertical_fn, in_axes=-1, out_axes=-1)(q)
    if not horizontal:
        return vert

    n_tracers = q.shape[-1]
    nlev = q.shape[-2]
    q_flat = q.reshape(*q.shape[:3], nlev * n_tracers)  # (6, n, n, nlev*n_tracers)

    # Pre-pad q_flat ONCE and feed it to gradient_x_3d / gradient_y_3d (and the
    # hyperdiffusion inner Laplacian) via ``padded=`` — halves the halo cost.
    _dg = getattr(grid, "duogrid", None)
    _offsets = None if _dg is not None else grid.halo_interp_offsets
    q_pad = pad_halo_4d(q_flat, interp_offsets=_offsets, duogrid=_dg)

    dq_dx = gradient_x_3d(q_flat, grid, padded=q_pad).reshape(*q.shape)
    dq_dy = gradient_y_3d(q_flat, grid, padded=q_pad).reshape(*q.shape)
    horiz = -(u[..., None] * dq_dx + v[..., None] * dq_dy)

    if hyperdiff_coeff > 0:
        horiz = horiz + hyperdiffusion_3d(
            q_flat, grid, hyperdiff_coeff, padded=q_pad,
        ).reshape(*q.shape)

    return horiz + vert


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

    # Get prescribed winds at current time
    u, v, sigma_dot = wind_fn(t, grid, sigma_coord)  # (6,n,n,nlev), (6,n,n,nlev+1)

    # Delegate to the SHARED cubed-sphere advective-tracer core (one batched
    # halo pad for the horizontal stencils; per-tracer vertical advection with
    # the prescribed sigma_dot bound).  Same numerics now used by the FV3
    # hydrostatic dycore's moist tracer transport.
    def _vert_adv_one(q_one):
        return vertical_advection(q_one, sigma_dot, sigma_coord)

    dq_dt = advective_tracer_tendency(
        q, u, v, grid, _vert_adv_one,
        hyperdiff_coeff=config.hyperdiff_coeff,
    )  # (6, n, n, nlev, n_tracers)

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
