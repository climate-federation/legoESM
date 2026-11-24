"""Prescribed-wind tracer transport on the lat-lon C-grid.

Advects an arbitrary number of passive tracers using analytically
prescribed wind fields.  The transport equation in sigma coordinates:

    dq_i/dt = -(u dq_i/dx + v dq_i/dy) - sigma_dot dq_i/dsigma

Horizontal advection uses the C-grid PPM operator
(cgrid_fv_scalar_advection_latlon_3d) in advective form.  The
advective form is correct for prescribed-wind transport because there
is no companion continuity equation to be mass-consistent with, and
it preserves uniform tracers exactly regardless of wind divergence.
Vertical advection uses the shared upwind operator from
grids.vertical (``vertical_scheme="upwind"``, default), or -- with
``vertical_scheme="fv3_remap"`` -- the FV3 way: the horizontal SSP-RK step
runs alone, then the layer interfaces are displaced by ``dt * sigma_dot``
(sigma_dot at the step mid-time) and the tracer is remapped back to the
fixed levels with the certified duo remap
(``grids.vertical.vertical_remap_sigma_fv3``).  Lie splitting, horizontal
then vertical, as FV3 does within each remap step.

Prescribed winds are evaluated directly at C-grid face coordinates
(u at longitude interfaces, v at latitude interfaces) to avoid
cell-center averaging that changes the velocity field before advection.
sigma_dot is kept at cell centers.

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
from legoesm.core.operators_fv_latlon_3d import (
    cgrid_fv_flux_divergence_latlon_3d, cgrid_fv_scalar_advection_latlon_3d)
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.vertical import (
    SigmaCoordinate, check_tracer_vertical_scheme, vertical_advection,
    vertical_remap_sigma_fv3)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin


# Type alias for prescribed wind functions on the lat-lon grid.
# Signature: (time, lon2d, lat2d, sigma_coord) -> (u_east, v_north, sigma_dot)
#   u_east:      shape matching lon2d/lat2d + (nlev,)
#   v_north:     shape matching lon2d/lat2d + (nlev,)
#   sigma_dot:   shape matching lon2d/lat2d + (nlev+1,)
WindFnLatLon = Callable[
    [float, jax.Array, jax.Array, SigmaCoordinate],
    tuple[jax.Array, jax.Array, jax.Array],
]


class TracerTransportLatLonConfig(NamedTuple):
    """Configuration for lat-lon C-grid tracer transport model."""
    time_integrator: str = "ssp_rk3"
    #: "upwind" (in-tendency donor cell) | "fv3_remap" (split Lagrangian
    #: remap after the horizontal step; see module docstring).
    vertical_scheme: str = "upwind"


def uface_coords(grid: LatLonGrid):
    """Compute 2D (lon, lat) coordinates at interior u-face positions.

    Returns only the *n_lon* unique longitude interfaces (no periodic
    wrap duplicate).  The caller must append the wrap column after
    evaluating the wind function to avoid passing out-of-range
    coordinates to non-periodic interpolators.

    All returned longitudes are in [0, 2π).

    Shape: (n_lat, n_lon).
    """
    lon = grid.lon  # (n_lon,)
    dlon = grid.dlon
    # Interior interfaces: midpoints between consecutive cell centers.
    # lon_iface[k] = 0.5 * (lon[k-1] + lon[k]) for k=1..n_lon-1.
    # The first interface wraps: midpoint of (lon[-1], lon[0]+2π).
    lon_iface = 0.5 * (jnp.roll(lon, 1) + lon)  # (n_lon,)
    # Fix the wrap interface: roll puts lon[-1] at position 0, so
    # lon_iface[0] = 0.5*(lon[-1]+lon[0]).  The correct value is
    # 0.5*(lon[-1] + lon[0]+2π) mod 2π = lon[0] - dlon/2 mod 2π.
    wrap_lon = (lon[0] - 0.5 * dlon) % (2.0 * jnp.pi)
    lon_iface = lon_iface.at[0].set(wrap_lon)
    lat_u = grid.lat  # (n_lat,)
    return jnp.broadcast_to(lon_iface[None, :], (grid.n_lat, grid.n_lon)), \
           jnp.broadcast_to(lat_u[:, None], (grid.n_lat, grid.n_lon))


def vface_coords(grid: LatLonGrid):
    """Compute 2D (lon, lat) coordinates at v-face positions.

    V-faces sit at latitude interfaces between cell centers.
    Shape: (n_lat+1, n_lon).
    """
    # Band-correct v-face latitudes carried by the grid (pre-sliced under
    # MPI band decomposition; a hard-coded ±π/2 pad would mislabel interior
    # band cuts as poles when evaluating the prescribed wind).
    lat_v = grid.lat_v  # (n_lat+1,)
    lon_v = grid.lon  # (n_lon,) — same longitudes as cell centers
    return jnp.broadcast_to(lon_v[None, :], (grid.n_lat + 1, grid.n_lon)), \
           jnp.broadcast_to(lat_v[:, None], (grid.n_lat + 1, grid.n_lon))


def tracer_tendencies_latlon(
    state: TracerState,
    grid: LatLonGrid,
    sigma_coord: SigmaCoordinate,
    wind_fn: WindFnLatLon,
    config: TracerTransportLatLonConfig = TracerTransportLatLonConfig(),
) -> TracerState:
    """Compute tracer transport tendencies on the lat-lon C-grid.

    Prescribed winds are evaluated directly at C-grid face coordinates
    to avoid cell-center averaging.  sigma_dot is at cell centers.

    Parameters
    ----------
    state : TracerState
        Current state with tracers shape (n_lat, n_lon, nlev, n_tracers)
        and scalar time.
    grid : LatLonGrid
    sigma_coord : SigmaCoordinate
    wind_fn : WindFnLatLon
        ``(t, lon2d, lat2d, sigma_coord) → (u, v, sigma_dot)``.
        Called three times: once for u-face coords, once for v-face
        coords, and once for cell-center coords (sigma_dot only).
    config : TracerTransportLatLonConfig

    Returns
    -------
    TracerState  — tendency pytree (dtracers_dt, dtime_dt=1).
    """
    check_tracer_vertical_scheme(config.vertical_scheme)
    t = state.time.data
    q = state.tracers.data  # (n_lat, n_lon, nlev, n_tracers)
    n_lat, n_lon, nlev, n_tracers = q.shape

    # Evaluate u at the n_lon unique longitude-face coordinates, then
    # append the periodic wrap column.  This avoids passing a 2π+ε
    # longitude to non-periodic wind_fn interpolation.
    lon_u, lat_u = uface_coords(grid)  # (n_lat, n_lon) — no wrap
    u_interior, _, _ = wind_fn(t, lon_u, lat_u, sigma_coord)
    u_face = jnp.concatenate([u_interior, u_interior[:, 0:1]], axis=1)

    # Evaluate v at latitude-face coordinates
    lon_v, lat_v = vface_coords(grid)
    _, v_face, _ = wind_fn(t, lon_v, lat_v, sigma_coord)

    # sigma_dot at cell centers
    _, _, sigma_dot = wind_fn(t, grid.lon2d, grid.lat2d, sigma_coord)

    # Horizontal advection — fold tracer axis into level axis so the
    # operator's halo pad + PPM reconstruction runs once for all tracers.
    # ``pad_halo_latlon_3d`` and ``_ppm_reconstruct_*_3d`` only operate
    # on lat/lon; the trailing combined axis is passively carried
    # through.  ``u_face`` and ``v_face`` are the same for every tracer,
    # so they are broadcast across the trailing axis using ``jnp.repeat``:
    # the ``q_flat`` reshape interleaves levels and tracers as
    # ``[lev0/trc0, lev0/trc1, ..., lev1/trc0, ...]``, so each level's
    # velocity must be duplicated ``n_tracers`` times to align — using
    # ``jnp.tile`` would concatenate the entire array and mis-align
    # tracer ↔ level.
    q_flat = q.reshape(n_lat, n_lon, nlev * n_tracers)
    if n_tracers == 1:
        u_face_b, v_face_b = u_face, v_face
    else:
        u_face_b = jnp.repeat(u_face, n_tracers, axis=-1)
        v_face_b = jnp.repeat(v_face, n_tracers, axis=-1)
    horiz_adv_flat = cgrid_fv_scalar_advection_latlon_3d(
        q_flat, u_face_b, v_face_b, grid,
    )
    horiz_adv = horiz_adv_flat.reshape(n_lat, n_lon, nlev, n_tracers)

    # Vertical advection — local stencil along axis -1, no halo cost.
    # ``vertical_advection`` hard-codes axis -1 as nlev, so vmap over
    # the trailing tracer axis (with sigma_dot/sigma_coord captured in
    # the closure) so JAX produces one batched kernel.
    def _vert_one(q_one):
        return vertical_advection(q_one, sigma_dot, sigma_coord)

    if config.vertical_scheme == "fv3_remap":
        dq_dt = horiz_adv           # vertical done by the split remap
    else:
        vert_adv = jax.vmap(_vert_one, in_axes=-1, out_axes=-1)(q)
        dq_dt = horiz_adv + vert_adv

    return TracerState(
        tracers=state.tracers.replace(data=dq_dt),
        time=state.time.replace(data=jnp.ones_like(t)),
    )


def tracer_tendencies_latlon_mass_flux(state: TracerState, grid: LatLonGrid,
                                       mass_flux, sigma_coord: SigmaCoordinate) -> TracerState:
    """Flux-form tracer tendency on a steady prescribed air mass with the
    discretely consistent air fluxes of
    :mod:`legoesm.atmosphere.dynamics.gcm.consistent_mass_flux`::

        dq_k/dt = (-div(F_k q) - (g_{k+1/2} - g_{k-1/2})) / dp_k

    horizontal PPM C-grid flux form (the face air flux ``F`` in place of the
    velocity), vertical monotone van-Leer flux ``g`` of the interface air
    flux (``consistent_mass_flux.van_leer_vertical_flux``; positive down; nonzero at the top/surface = open).  ``div F_k + delta W_k = 0`` keeps a
    uniform tracer uniform; ``sum area dp q`` changes only by the tracer flux
    through an open top or surface."""
    from legoesm.atmosphere.dynamics.gcm.consistent_mass_flux import (
        van_leer_vertical_flux)
    t = state.time.data
    q = state.tracers.data
    n_lat, n_lon, nlev, ntr = q.shape
    fu, fv = mass_flux.flux_h
    q_flat = q.reshape(n_lat, n_lon, nlev * ntr)
    if ntr > 1:
        fu, fv = jnp.repeat(fu, ntr, axis=-1), jnp.repeat(fv, ntr, axis=-1)
    neg_div = cgrid_fv_flux_divergence_latlon_3d(q_flat, fu, fv, grid)
    neg_div = neg_div.reshape(n_lat, n_lon, nlev, ntr)
    g = van_leer_vertical_flux(q, mass_flux.w, sigma_coord)
    dq_dt = (neg_div - (g[..., 1:, :] - g[..., :-1, :])) / mass_flux.dp[..., None]
    return TracerState(
        tracers=state.tracers.replace(data=dq_dt),
        time=state.time.replace(data=jnp.ones_like(t)),
    )


class TracerTransportLatLonModel(IntegrationMixin):
    """Prescribed-wind tracer transport on the lat-lon C-grid.

    Parameters
    ----------
    grid : LatLonGrid
    sigma_coord : SigmaCoordinate
    wind_fn : WindFnLatLon
        ``(t, lon2d, lat2d, sigma_coord) → (u, v, sigma_dot)``.
    config : TracerTransportLatLonConfig, optional
    mass_flux : PrescribedMassFlux, optional
        Steady consistent air fluxes (``consistent_mass_flux_latlon``): the
        step is then the flux form of
        :func:`tracer_tendencies_latlon_mass_flux` (``wind_fn`` unused; its
        vertical flux is van Leer); needs ``vertical_scheme="upwind"`` (the
        config's in-tendency choice; ponytail: no separate selector).  ``None`` (default): unchanged.
    """

    def __init__(
        self,
        grid: LatLonGrid,
        sigma_coord: SigmaCoordinate,
        wind_fn: WindFnLatLon,
        config: TracerTransportLatLonConfig | None = None,
        *,
        mass_flux=None,
    ):
        self.grid = grid
        self.mass_flux = mass_flux
        self.sigma_coord = sigma_coord
        self.wind_fn = wind_fn
        self.config = config or TracerTransportLatLonConfig()
        check_tracer_vertical_scheme(self.config.vertical_scheme)
        if mass_flux is not None and self.config.vertical_scheme != "upwind":
            raise ValueError("mass_flux runs the flux-form (van Leer) vertical "
                             "transport: vertical_scheme must be 'upwind'")

    def tendencies(self, state: TracerState) -> TracerState:
        if self.mass_flux is not None:
            return tracer_tendencies_latlon_mass_flux(state, self.grid, self.mass_flux,
                                                      self.sigma_coord)
        return tracer_tendencies_latlon(
            state, self.grid, self.sigma_coord, self.wind_fn, self.config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TracerState, dt: float) -> TracerState:
        """Advance one time step."""
        if self.mass_flux is not None:
            return dispatch_integrator(
                state, lambda s: tracer_tendencies_latlon_mass_flux(
                    s, self.grid, self.mass_flux, self.sigma_coord),
                dt, self.config.time_integrator)

        def tendency_fn(s):
            return tracer_tendencies_latlon(
                s, self.grid, self.sigma_coord, self.wind_fn, self.config,
            )
        new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )
        if self.config.vertical_scheme != "fv3_remap":
            return new
        _, _, sigma_dot = self.wind_fn(
            state.time.data + 0.5 * dt, self.grid.lon2d, self.grid.lat2d,
            self.sigma_coord)
        q = vertical_remap_sigma_fv3(
            new.tracers.data, sigma_dot, self.sigma_coord, dt)
        return new._replace(tracers=new.tracers.replace(data=q))
