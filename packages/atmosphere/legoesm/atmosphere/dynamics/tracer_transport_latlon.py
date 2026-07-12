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
grids.vertical.

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
from legoesm.core.operators_fv_latlon_3d import cgrid_fv_scalar_advection_latlon_3d
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.vertical import SigmaCoordinate, vertical_advection
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


def _uface_coords(grid: LatLonGrid):
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


def _vface_coords(grid: LatLonGrid):
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
    t = state.time.data
    q = state.tracers.data  # (n_lat, n_lon, nlev, n_tracers)
    n_lat, n_lon, nlev, n_tracers = q.shape

    # Evaluate u at the n_lon unique longitude-face coordinates, then
    # append the periodic wrap column.  This avoids passing a 2π+ε
    # longitude to non-periodic wind_fn interpolation.
    lon_u, lat_u = _uface_coords(grid)  # (n_lat, n_lon) — no wrap
    u_interior, _, _ = wind_fn(t, lon_u, lat_u, sigma_coord)
    u_face = jnp.concatenate([u_interior, u_interior[:, 0:1]], axis=1)

    # Evaluate v at latitude-face coordinates
    lon_v, lat_v = _vface_coords(grid)
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

    vert_adv = jax.vmap(_vert_one, in_axes=-1, out_axes=-1)(q)

    dq_dt = horiz_adv + vert_adv

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
    """

    def __init__(
        self,
        grid: LatLonGrid,
        sigma_coord: SigmaCoordinate,
        wind_fn: WindFnLatLon,
        config: TracerTransportLatLonConfig | None = None,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.wind_fn = wind_fn
        self.config = config or TracerTransportLatLonConfig()

    def tendencies(self, state: TracerState) -> TracerState:
        return tracer_tendencies_latlon(
            state, self.grid, self.sigma_coord, self.wind_fn, self.config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TracerState, dt: float) -> TracerState:
        """Advance one time step."""
        def tendency_fn(s):
            return tracer_tendencies_latlon(
                s, self.grid, self.sigma_coord, self.wind_fn, self.config,
            )
        return dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )
