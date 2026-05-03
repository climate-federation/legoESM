"""Arakawa C-grid Shallow Water Equations on the latitude-longitude grid.

True staggered C-grid discretisation for the shallow water equations:

* u lives at longitude interfaces: shape (n_lat, n_lon+1)
* v lives at latitude interfaces:  shape (n_lat+1, n_lon)
* h, h_s at cell centers:         shape (n_lat, n_lon)

Key design choices (following MITgcm / Sadourny 1975):
- Vector-invariant momentum form with Bernoulli function at cell centers.
- Sadourny (1975) energy-conserving Coriolis from the ocean C-grid operators.
- Conservative FV mass flux divergence using compact C-grid stencils.
- Pole treatment: v = 0 at poles (wall BC), periodic longitude.
- No checkerboard null space (compact 1-cell gradient/divergence, not 2Δx).

Operator reuse:
- gradient_x/y_cgrid, divergence_cgrid, coriolis_cgrid, vector_laplacian_cgrid
  are imported from the ocean lat-lon C-grid operator module.
- Halo exchange uses the existing lat-lon periodic + pole-folding backend.

References
----------
- Sadourny (1975): The dynamics of finite-difference models of the
  shallow water equations. J. Atmos. Sci., 32, 680-689.
- Arakawa & Lamb (1977): Computational Design of the Basic Dynamical
  Processes of the UCLA General Circulation Model.
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (sections on
  lat-lon C-grid shallow water).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    gradient_x_cgrid,
    gradient_y_cgrid,
    divergence_cgrid,
    curl_vertex_cgrid,
    vector_laplacian_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
    cell_to_cgrid_winds,
)
from legoesm.core.operators_fv_latlon import cgrid_fv_flux_divergence_latlon
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import (
    compute_polar_filter_mask,
    fourier_filter,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm.core.precision import cast_pytree
from legoesm.core.conservation import _accumulation_dtype
from legoesm import constants


# ==============================================================================
# State and Config
# ==============================================================================

class CGridLatLonShallowWaterState(NamedTuple):
    """Shallow water state on the lat-lon Arakawa C-grid.

    h   : (n_lat, n_lon)   -- fluid depth at cell centres [m]
    u   : (n_lat, n_lon+1)  -- zonal velocity at longitude interfaces [m/s]
    v   : (n_lat+1, n_lon)  -- meridional velocity at latitude interfaces [m/s]
    h_s : (n_lat, n_lon)   -- surface topography at cell centres [m]
    """
    h: jax.Array
    u: jax.Array
    v: jax.Array
    h_s: jax.Array


class CGridLatLonShallowWaterConfig(NamedTuple):
    """Configuration for the C-grid lat-lon shallow water model.

    Time integrator options: "ssp_rk3", "ssp_rk34", "ssp_rk54", "rk4".
    """
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    time_integrator: str = "ssp_rk3"
    fix_mass: bool = True
    use_ppm_transport: bool = True  # PPM (4th-order) vs simple averaging for mass flux
    use_polar_filter: bool = False
    polar_filter_cutoff_deg: float = 60.0
    polar_filter_max_wave_speed: float = 300.0


# ==============================================================================
# C-grid interpolation helpers
# ==============================================================================

def absolute_vorticity_coriolis(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Coriolis-like acceleration using absolute vorticity (ζ+f).

    The vector-invariant momentum equation requires the full absolute
    vorticity η = ζ + f, not just the planetary vorticity f.  This
    function computes ζ at vertices via ``curl_vertex_cgrid``, adds the
    planetary vorticity, averages η to velocity faces, and returns the
    cross-product terms.

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    cor_u, cor_v : same shapes as u, v
    """
    is_3d = u.ndim == 3
    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # --- Relative vorticity at vertices ---
    zeta = curl_vertex_cgrid(u, v, grid)  # (n_lat+1, n_lon+1[, nlev])

    # --- Planetary vorticity at vertices ---
    # sin(±π/2) = ±1 exactly, so build f_vert directly from the
    # interior sin via Pad with constant_values = ±2Ω.  Single
    # Pad HLO op replaces alloc-2-singletons + concatenate-of-three +
    # sin tower.
    lat = grid.lat  # cell-center latitudes
    lat_int = 0.5 * (lat[:-1] + lat[1:])
    twoOmega = 2.0 * constants.Omega
    f_vert = jnp.pad(
        twoOmega * jnp.sin(lat_int),
        (1, 1), constant_values=(-twoOmega, twoOmega),
    )  # (n_lat+1,)

    # Absolute vorticity at vertices
    if is_3d:
        eta = zeta + f_vert[:, jnp.newaxis, jnp.newaxis]
    else:
        eta = zeta + f_vert[:, jnp.newaxis]

    # --- Average η to u-faces ---
    # u-face[i, j] is flanked by vertex[i, j] (south) and vertex[i+1, j] (north)
    eta_at_u = 0.5 * (eta[:-1] + eta[1:])  # (n_lat, n_lon+1[, nlev])

    # --- Average v to u-faces (4-point, same as coriolis_cgrid) ---
    v_west = jnp.roll(v, 1, axis=1)
    v_avg = 0.25 * (v[:-1] + v[1:] + v_west[:-1] + v_west[1:])
    if is_3d:
        v_at_u = jnp.concatenate([v_avg, v_avg[:, 0:1, :]], axis=1)
    else:
        v_at_u = jnp.concatenate([v_avg, v_avg[:, 0:1]], axis=1)

    # --- Average η to v-faces ---
    # v-face[i, j] is flanked by vertex[i, j] (west) and vertex[i, j+1] (east)
    # vertex[:, n_lon] wraps to vertex[:, 0]
    eta_at_v = 0.5 * (eta[:, :-1] + eta[:, 1:])  # (n_lat+1, n_lon[, nlev])

    # --- Average u to v-faces (4-point, same as coriolis_cgrid) ---
    # Use ``jnp.pad`` on the leading axis instead of allocating
    # ``zero_row`` twice and concatenating — one HLO Pad op vs
    # alloc + concat.
    u_avg_interior = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
    if is_3d:
        u_at_v = jnp.pad(u_avg_interior, ((1, 1), (0, 0), (0, 0)))
    else:
        u_at_v = jnp.pad(u_avg_interior, ((1, 1), (0, 0)))

    # --- Coriolis terms ---
    cor_u = eta_at_u * v_at_u
    cor_v = -eta_at_v * u_at_v

    return cor_u, cor_v


def _kinetic_energy_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Kinetic energy at cell centers from C-grid face velocities.

    KE = 0.5 * (ū² + v̄²) where ū, v̄ are averaged to cell centers.

    Parameters
    ----------
    u : (n_lat, n_lon+1)  -- zonal velocity at lon interfaces
    v : (n_lat+1, n_lon)  -- meridional velocity at lat interfaces

    Returns
    -------
    KE : (n_lat, n_lon)
    """
    u_c = 0.5 * (u[:, :-1] + u[:, 1:])   # (n_lat, n_lon)
    v_c = 0.5 * (v[:-1, :] + v[1:, :])   # (n_lat, n_lon)
    return 0.5 * (u_c**2 + v_c**2)


# ==============================================================================
# Tendency computation
# ==============================================================================

def cgrid_latlon_sw_tendencies(
    state: CGridLatLonShallowWaterState,
    grid: LatLonGrid,
    config: CGridLatLonShallowWaterConfig = CGridLatLonShallowWaterConfig(),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute C-grid shallow water tendencies on the lat-lon grid.

    Equations (vector-invariant form):
        dh/dt = -∇·(h v)                           (mass conservation)
        du/dt = -∂B/∂x + f v̄  + viscosity          (zonal momentum)
        dv/dt = -∂B/∂y - f ū  + viscosity          (meridional momentum)

    where B = g(h + h_s) + KE is the Bernoulli function.

    Parameters
    ----------
    state : CGridLatLonShallowWaterState
    grid : LatLonGrid
    config : CGridLatLonShallowWaterConfig

    Returns
    -------
    (dh_dt, du_dt, dv_dt) : tuple of jax.Array
        Tendencies at (cell centers, lon faces, lat faces).
    """
    h, u, v, h_s = state
    g = config.g

    # --- 1. Mass flux divergence ---
    if config.use_ppm_transport:
        # PPM (4th-order) reconstruction of h at faces for upwind flux
        dh_dt = cgrid_fv_flux_divergence_latlon(h, u, v, grid)
    else:
        # Simple 2nd-order averaging of h to faces
        h_u = interp_cell_to_uface(h)
        h_v = interp_cell_to_vface(h)
        F_u = h_u * u
        F_v = h_v * v
        dh_dt = -divergence_cgrid(F_u, F_v, grid)

    # --- 2. Bernoulli function at cell centers ---
    KE = _kinetic_energy_cgrid(u, v)   # (n_lat, n_lon)
    B = g * (h + h_s) + KE             # (n_lat, n_lon)

    # --- 3. Pressure gradient + KE gradient (Bernoulli gradient) ---
    du_dt = -gradient_x_cgrid(B, grid)   # (n_lat, n_lon+1)
    dv_dt = -gradient_y_cgrid(B, grid)   # (n_lat+1, n_lon)

    # --- 4. Coriolis using absolute vorticity (ζ+f) ---
    cor_u, cor_v = absolute_vorticity_coriolis(u, v, grid)
    du_dt = du_dt + cor_u
    dv_dt = dv_dt + cor_v

    # --- 5. Laplacian viscosity (optional) ---
    if config.A_h > 0.0:
        lap_u, lap_v = vector_laplacian_cgrid(u, v, grid)
        du_dt = du_dt + config.A_h * lap_u
        dv_dt = dv_dt + config.A_h * lap_v

    # Enforce zero tendency at poles (wall BC) so intermediate RK
    # stages never see nonzero v at poles feeding into divergence/Coriolis.
    # Single ``Pad`` HLO op replaces two ``ScatterUpdate`` ops on the
    # lat axis — same per-RK-stage pattern as the lat-lon C-grid PE.
    dv_dt = jnp.pad(dv_dt[1:-1, :], ((1, 1), (0, 0)))

    return dh_dt, du_dt, dv_dt


# ==============================================================================
# Model class
# ==============================================================================

class CGridLatLonShallowWaterModel(IntegrationMixin):
    """C-grid shallow water model on the latitude-longitude grid.

    Features:
    - Compact-stencil C-grid operators (no checkerboard mode).
    - Sadourny (1975) energy-conserving Coriolis.
    - Conservative mass flux divergence.
    - Wall BC at poles (v = 0), periodic longitude.
    - Optional Laplacian viscosity and mass conservation fixer.

    Parameters
    ----------
    grid : LatLonGrid
        Horizontal grid.
    config : CGridLatLonShallowWaterConfig, optional
    """

    def __init__(
        self,
        grid: LatLonGrid,
        config: CGridLatLonShallowWaterConfig | None = None,
        dt: float = 600.0,
    ):
        self.grid = grid
        self.config = config or CGridLatLonShallowWaterConfig()
        # Precompute polar filter mask (cached, not traced)
        if self.config.use_polar_filter:
            self._polar_mask = compute_polar_filter_mask(
                grid, dt=dt,
                max_wave_speed=self.config.polar_filter_max_wave_speed,
                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
            )
        else:
            self._polar_mask = None

    def compute_mass(self, state: CGridLatLonShallowWaterState) -> jax.Array:
        """Compute total mass (for conservation fixer target)."""
        acc = _accumulation_dtype()
        return jnp.sum(state.h.astype(acc) * self.grid.area.astype(acc))

    def tendencies(
        self, state: CGridLatLonShallowWaterState,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute tendencies (pure function wrapper)."""
        return cgrid_latlon_sw_tendencies(state, self.grid, self.config)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: CGridLatLonShallowWaterState,
        dt: float,
        target_mass: jax.Array | None = None,
    ) -> CGridLatLonShallowWaterState:
        """Advance one time step.

        Parameters
        ----------
        state : CGridLatLonShallowWaterState
        dt : float
        target_mass : jax.Array or None
            If provided, the mass fixer corrects to this target.
            Compute once via ``model.compute_mass(state0)`` and pass
            as an explicit traced argument (avoids JIT recompilation
            when the target changes).
        """
        state_c = cast_pytree(state, None, "compute")

        def tendency_fn(s):
            dh, du, dv = cgrid_latlon_sw_tendencies(
                s, self.grid, self.config,
            )
            # Polar filter: damp high-frequency modes near poles
            if self._polar_mask is not None:
                dh = fourier_filter(dh, self.grid, self._polar_mask)
                du_interior = fourier_filter(du[:, :-1], self.grid, self._polar_mask)
                du = jnp.concatenate([du_interior, du_interior[:, 0:1]], axis=1)
            return CGridLatLonShallowWaterState(
                h=dh, u=du, v=dv,
                h_s=jnp.zeros_like(s.h_s),
            )

        state_new = dispatch_integrator(
            state_c, tendency_fn, dt, self.config.time_integrator,
        )

        # Enforce v = 0 at poles via a single Pad HLO op.
        v_new = jnp.pad(state_new.v[1:-1, :], ((1, 1), (0, 0)))
        state_new = state_new._replace(v=v_new)

        # Conservation fixer (use float64 accumulation for precision)
        if self.config.fix_mass:
            acc = _accumulation_dtype()
            area = self.grid.area.astype(acc)
            total_area = jnp.sum(area)
            if target_mass is not None:
                mass_target = target_mass
            else:
                mass_target = jnp.sum(state.h.astype(acc) * area)
            mass_new = jnp.sum(state_new.h.astype(acc) * area)
            correction = (mass_target - mass_new) / total_area
            h_fixed = state_new.h + correction.astype(state_new.h.dtype)
            # Re-clamp after mass correction to prevent negative depth
            h_fixed = jnp.maximum(h_fixed, 0.0)
            state_new = state_new._replace(h=h_fixed)

        return cast_pytree(state_new, None, "storage")


# ==============================================================================
# Initial condition helpers
# ==============================================================================

# cell_to_cgrid_winds is imported from
# legoesm.ocean.dynamics.latlon_cgrid_operators (shared with ocean).
# The shared version handles both 2D and 3D via ndim dispatch.


def williamson_test2_cgrid(
    grid: LatLonGrid,
) -> CGridLatLonShallowWaterState:
    """Williamson Test 2 initial condition on the C-grid.

    Global steady-state nonlinear zonal geostrophic flow (12-day rotation).
    Since the flow is purely zonal and zonally uniform, u at lon faces
    equals u at cell centers.
    """
    R = grid.radius
    Omega = constants.Omega
    g = constants.g

    u_0 = 2.0 * jnp.pi * R / (12.0 * 86400.0)
    gh_0 = 2.94e4
    h_0 = gh_0 / g

    lat = grid.lat2d   # (n_lat, n_lon)

    # Height at cell centers (geostrophic balance)
    h = h_0 - (R * Omega * u_0 + u_0**2 / 2.0) * jnp.sin(lat)**2 / g

    # Zonal velocity at cell centers (zonally uniform)
    u_cell = u_0 * jnp.cos(lat)

    # Convert to C-grid: u at lon faces, v = 0 at lat faces
    n_lat, n_lon = grid.n_lat, grid.n_lon
    # u at lon faces: same as cell centers since u is zonally uniform
    u_face = jnp.concatenate([u_cell, u_cell[:, 0:1]], axis=1)
    v_face = jnp.zeros((n_lat + 1, n_lon), dtype=h.dtype)

    return CGridLatLonShallowWaterState(
        h=h, u=u_face, v=v_face, h_s=jnp.zeros_like(h),
    )


def williamson_test2_exact_cgrid(
    grid: LatLonGrid,
    t: float,
) -> CGridLatLonShallowWaterState:
    """Exact solution for Williamson Test 2 at time t (steady-state)."""
    return williamson_test2_cgrid(grid)


def williamson_test5_cgrid(
    grid: LatLonGrid,
) -> CGridLatLonShallowWaterState:
    """Williamson Test 5: Zonal flow over isolated mountain on C-grid."""
    R = grid.radius
    Omega = constants.Omega
    g = constants.g

    u_0 = 20.0  # m/s
    gh_0 = 5960.0 * g

    lat = grid.lat2d
    lon = grid.lon2d
    n_lat, n_lon = grid.n_lat, grid.n_lon

    # Velocity (solid body rotation)
    u_cell = u_0 * jnp.cos(lat)

    # Mountain topography
    lon_c = 3.0 * jnp.pi / 2.0
    lat_c = jnp.pi / 6.0
    R_m = jnp.pi / 9.0
    h_s0 = 2000.0

    r = jnp.arccos(jnp.clip(
        jnp.sin(lat_c) * jnp.sin(lat)
        + jnp.cos(lat_c) * jnp.cos(lat) * jnp.cos(lon - lon_c),
        -1.0, 1.0,
    ))
    h_s = jnp.where(r < R_m, h_s0 * (1.0 - r / R_m), 0.0)

    # Height: h = h_free - h_s
    h_free = (gh_0 - (R * Omega * u_0 + u_0**2 / 2.0) * jnp.sin(lat)**2) / g
    h = h_free - h_s

    # C-grid winds
    u_face = jnp.concatenate([u_cell, u_cell[:, 0:1]], axis=1)
    v_face = jnp.zeros((n_lat + 1, n_lon), dtype=h.dtype)

    return CGridLatLonShallowWaterState(
        h=h, u=u_face, v=v_face, h_s=h_s,
    )


def compute_error_norms_cgrid(
    state: CGridLatLonShallowWaterState,
    reference: CGridLatLonShallowWaterState,
    grid: LatLonGrid,
) -> dict[str, float]:
    """Compute L1, L2, L∞ error norms for height field."""
    err = state.h - reference.h
    ref = reference.h
    area = grid.area

    l1 = jnp.sum(jnp.abs(err) * area) / jnp.sum(jnp.abs(ref) * area)
    l2 = jnp.sqrt(jnp.sum(err**2 * area) / jnp.sum(ref**2 * area))
    linf = jnp.max(jnp.abs(err)) / jnp.max(jnp.abs(ref))

    return {"l1": float(l1), "l2": float(l2), "linf": float(linf)}
