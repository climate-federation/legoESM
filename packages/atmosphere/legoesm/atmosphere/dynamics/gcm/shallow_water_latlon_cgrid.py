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

from legoesm.grids.operator_adapters import latlon_cgrid_operators
from legoesm.grids.operators_latlon_cgrid import (
    get_band_mpi_cut_layout,
    vector_laplacian_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
    pad_lon_cgrid,
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
from legoesm.core.conservation import conservation_accumulator
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
    anchor_mass_to_initial: bool = False  # Mirror PE: anchor fixer to initial mass
    use_ppm_transport: bool = True  # PPM (4th-order) vs simple averaging for mass flux
    use_polar_filter: bool = False
    polar_filter_cutoff_deg: float = 60.0
    polar_filter_max_wave_speed: float = 300.0
    # --- New fields APPENDED (codex review: inserting before existing
    # fields breaks positional NamedTuple construction for callers) ---
    # Biharmonic (del-4) viscosity [m^4/s].  Scale-selective: damps
    # grid-scale noise ~ (k*dx)^4 while leaving resolved scales nearly
    # untouched — unlike A_h, whose k^2 law measurably damps planetary
    # waves at 2.5 deg (Rossby-Haurwitz wave-4 e-folds in ~6 days at
    # A_h ~ 8e5 m^2/s).  Requires passing ``dt`` to the tendency
    # function: the per-latitude-row coefficient is capped at
    # ``nu_del4_cfl_frac * 1/(dt*lam_row^2)`` (lam_row = max Laplacian
    # eigenvalue of the row) so the shrinking zonal spacing toward the
    # poles cannot violate the explicit del-4 stability bound.
    nu_del4: float = 0.0
    # Numerics safety cap fraction for the del-4 diffusive CFL (SSP-RK3
    # real-axis stability ~ 2.5; 0.25 gives 10x margin).  Not a tunable.
    nu_del4_cfl_frac: float = 0.25


# ==============================================================================
# C-grid interpolation helpers
# ==============================================================================

def absolute_vorticity_coriolis(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    u_lat_pad: jnp.ndarray | None = None,
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

    # Static band-MPI dispatch (None ⟺ serial / single-rank band /
    # non-band topology — the historical code path runs verbatim).
    # Under latitude-band MPI with interior cuts, the band's end
    # vertex/face rows are NOT poles, so the two pole-constant
    # constructions below (planetary vorticity ±2Ω, u_at_v zero wall
    # rows) must instead use the neighbour rank's rows.
    _band = get_band_mpi_cut_layout()
    # Single-program SPMD twin: under the lat-band shard_map backend
    # ``get_band_mpi_cut_layout()`` is None (it recognizes only the 'mpi'
    # backend), so the f_vert / u_at_v pole constants must instead be selected
    # DATA-dependently per band — interior band cuts are NOT poles. Returns
    # ``(south_mask, north_mask)`` traced booleans inside the shard_map, or None
    # for serial/MPI/cube (those paths keep their static branch — additive).
    # Mirrors curl_vertex_cgrid's SPMD pole handling.
    from legoesm.parallel.latlon_spmd import spmd_pole_end_masks
    _spmd_pm = spmd_pole_end_masks()

    # --- Relative vorticity at vertices (via the B2 operator interface) ---
    # ``u_lat_pad`` (optional, exactly pad_with_pole_bc_lat(u, halo=1,
    # 0, 0)) is shared between the curl's circulation pad and the
    # 4-pt u->v-face average below — the atm tendency fuses it with
    # its other entry-level pads (census probe 8460424: u was padded
    # twice per RK stage).
    zeta = latlon_cgrid_operators(grid).vorticity(
        u, v, u_ext=u_lat_pad,
    )  # (n_lat+1, n_lon+1[, nlev])

    # --- Planetary vorticity at vertices ---
    lat = grid.lat  # cell-center latitudes
    # Rotation rate from the GRID, not constants.Omega: a non-rotating
    # grid (create_latlon_grid(..., omega=0.0) — the colliding-modons
    # case, #521) must yield f_vert = 0, but the hardcoded
    # constants.Omega silently kept the planet rotating regardless of
    # the grid omega (codex 2026-07-03 HIGH).  ``grid.omega`` is the
    # stored construction scalar (appended NamedTuple field), so this
    # is exact on any MPI band (no recovery from local ``f`` rows —
    # a band owning only the equator row has sin(lat) = 0), stays in
    # JAX space if the grid is ever passed as a dynamic pytree, and is
    # bit-identical for every rotating caller (grids built with the
    # default omega = constants.Omega).
    twoOmega = 2.0 * grid.omega
    if _band is None and _spmd_pm is None:
        # Serial / single-rank: sin(±π/2) = ±1 exactly, so build f_vert directly
        # from the interior sin via Pad with constant_values = ±2Ω.  Single
        # Pad HLO op replaces alloc-2-singletons + concatenate-of-three + sin.
        lat_int = 0.5 * (lat[:-1] + lat[1:])
        f_vert = jnp.pad(
            twoOmega * jnp.sin(lat_int),
            (1, 1), constant_values=(-twoOmega, twoOmega),
        )  # (n_lat+1,)
    else:
        # Band-decomposed (MPI or single-program SPMD): end vertex rows at
        # interior cuts sit at the midpoint latitude ACROSS the cut.  Pad
        # ``lat`` one row (backend-aware: MPI 1-D sendrecv / SPMD lat-band
        # ppermute at cuts; the constant ghost at a pole-touching end never
        # reaches the output — overwritten below), midpoint to vertex rows,
        # then restore the exact ±2Ω pole values at PHYSICAL pole ends only.
        # Interior vertex rows evaluate the identical
        # ``2Ω·sin(0.5·(lat[j-1]+lat[j]))`` chain — bit-identical to serial.
        from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
        lat_pad = pad_with_pole_bc_lat(
            lat, halo=1, south_value=0.0, north_value=0.0,
        )
        lat_vert = 0.5 * (lat_pad[:-1] + lat_pad[1:])  # (n_lat_local+1,)
        f_vert = twoOmega * jnp.sin(lat_vert)
        if _spmd_pm is not None:
            # SPMD: select the ±2Ω clamp DATA-dependently per band (a static
            # ``if`` would clamp every interior cut). south then north,
            # sequenced like the MPI path so a single band clamps both ends.
            _south_m, _north_m = _spmd_pm
            f_vert = jnp.where(_south_m, f_vert.at[0].set(-twoOmega), f_vert)
            f_vert = jnp.where(_north_m, f_vert.at[-1].set(twoOmega), f_vert)
        else:
            # MPI: static per-rank pole answer (None ⟺ this rank owns the pole).
            if _band.south_rank is None:
                f_vert = f_vert.at[0].set(-twoOmega)
            if _band.north_rank is None:
                f_vert = f_vert.at[-1].set(twoOmega)

    # Absolute vorticity at vertices
    if is_3d:
        eta = zeta + f_vert[:, jnp.newaxis, jnp.newaxis]
    else:
        eta = zeta + f_vert[:, jnp.newaxis]

    # --- Average η to u-faces ---
    # u-face[i, j] is flanked by vertex[i, j] (south) and vertex[i+1, j] (north)
    eta_at_u = 0.5 * (eta[:-1] + eta[1:])  # (n_lat, n_lon+1[, nlev])

    # --- Average v to u-faces (4-point, same as coriolis_cgrid) ---
    # v sits on lat-faces, cell-aligned in lon; the u-face 4-pt average needs
    # v's lon WEST-neighbour cell, so pad-then-average through the dispatched
    # lon halo (local wrap / 2-D ring) -> the full n_lon+1 u-faces directly.
    # Bit-identical to ``roll(v,1)`` + wrap-column concat at proc_lon==1, and
    # spans lon partition cuts under a 2-D split (THIS local roll was the
    # missed lon op that made the 2-D u-momentum decomposition-dependent).
    v_pad = pad_lon_cgrid(v, halo=1)
    ve = v_pad[:, 1:]    # cell j   bordering u-face j
    vw = v_pad[:, :-1]   # cell j-1 bordering u-face j
    v_at_u = 0.25 * (ve[:-1] + ve[1:] + vw[:-1] + vw[1:])  # (n_lat, n_lon+1)

    # --- Average η to v-faces ---
    # v-face[i, j] is flanked by vertex[i, j] (west) and vertex[i, j+1] (east)
    # vertex[:, n_lon] wraps to vertex[:, 0]
    eta_at_v = 0.5 * (eta[:, :-1] + eta[:, 1:])  # (n_lat+1, n_lon[, nlev])

    # --- Average u to v-faces (4-point, same as coriolis_cgrid) ---
    if _band is None and _spmd_pm is None:
        # Serial / single-rank: ``jnp.pad`` on the leading axis (one HLO Pad op
        # vs alloc + concat).  The zero end rows are the pole wall BC.
        u_avg_interior = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
        if is_3d:
            u_at_v = jnp.pad(u_avg_interior, ((1, 1), (0, 0), (0, 0)))
        else:
            u_at_v = jnp.pad(u_avg_interior, ((1, 1), (0, 0)))
    else:
        # Band-decomposed (MPI or SPMD): the end v-face rows at interior cuts
        # carry the genuine 4-point average spanning the cut, not the pole wall
        # zero.  Pad ``u`` one lat row (backend-aware: MPI sendrecv / SPMD
        # lat-band ppermute; u-face fields with the n_lon+1 wrap column are
        # supported), average on the padded array — interior rows the identical
        # operand chain as serial — then restore the zero wall rows at PHYSICAL
        # pole ends only.
        from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
        if u_lat_pad is None:
            u_lat_pad = pad_with_pole_bc_lat(
                u, halo=1, south_value=0.0, north_value=0.0,
            )  # (n_lat_local+2, n_lon+1[, nlev])
        u_at_v = 0.25 * (
            u_lat_pad[:-1, :-1] + u_lat_pad[:-1, 1:]
            + u_lat_pad[1:, :-1] + u_lat_pad[1:, 1:]
        )  # (n_lat_local+1, n_lon[, nlev])
        if _spmd_pm is not None:
            # SPMD: zero the pole-wall v-face DATA-dependently per band.
            _south_m, _north_m = _spmd_pm
            u_at_v = jnp.where(
                _south_m, u_at_v.at[0].set(jnp.zeros_like(u_at_v[0])), u_at_v)
            u_at_v = jnp.where(
                _north_m, u_at_v.at[-1].set(jnp.zeros_like(u_at_v[-1])), u_at_v)
        else:
            # MPI: static per-rank pole answer.
            if _band.south_rank is None:
                u_at_v = u_at_v.at[0].set(jnp.zeros_like(u_at_v[0]))
            if _band.north_rank is None:
                u_at_v = u_at_v.at[-1].set(jnp.zeros_like(u_at_v[-1]))

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

def _nu_del4_row_profiles(
    grid: LatLonGrid,
    config: CGridLatLonShallowWaterConfig,
    dt,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-latitude-row biharmonic coefficients with pole stability cap.

    The zonal spacing ``dx = R*dlon*cos(lat)`` shrinks toward the poles,
    so a globally-constant ``nu_del4`` sized for the interior violates
    the explicit del-4 diffusive CFL near the poles by many orders of
    magnitude.  Cap the coefficient per row at

        nu_max(row) = cfl_frac / (dt * lam_row^2),
        lam_row     = 4/dx_row^2 + 4/dy^2

    (``lam_row`` = max eigenvalue of the discrete 5-point Laplacian on
    that row), which keeps ``nu*dt*lam^2 <= cfl_frac`` everywhere.  The
    interior rows keep the full ``config.nu_del4``; only rows poleward
    of the crossover (about 79 deg at 2.5 deg resolution, 48 h grid
    e-folding) are reduced.

    Returns
    -------
    (nu_u, nu_v) : ((n_lat, 1), (n_lat+1, 1)) coefficient columns for
        the u-face (cell-center-lat) and v-face (lat-interface) rows.
    """
    # Per-row cell height: ``grid.dy`` is the 2-cell distance (see
    # LatLonGrid docstring), so the single-cell height of row j is
    # dy[j]/2 — scalar ``grid.dlat`` is only a diagnostic minimum on
    # Mercator/stretched grids and would over-cap interior rows there
    # (codex review).
    dy_u = 0.5 * grid.dy                                 # (n_lat,)
    # v-face (interface) rows: mean of the adjacent cell heights;
    # boundary interfaces reuse the edge-row height.
    dy_v = jnp.concatenate([
        dy_u[0:1], 0.5 * (dy_u[:-1] + dy_u[1:]), dy_u[-1:],
    ])                                                   # (n_lat+1,)
    dx_u = grid.radius * grid.dlon * grid.cos_lat        # (n_lat,)
    dx_v = grid.radius * grid.dlon * grid.cos_lat_v      # (n_lat+1,)
    lam_u = 4.0 / dx_u ** 2 + 4.0 / dy_u ** 2
    lam_v = 4.0 / dx_v ** 2 + 4.0 / dy_v ** 2
    nu_u = jnp.minimum(config.nu_del4,
                       config.nu_del4_cfl_frac / (dt * lam_u ** 2))
    nu_v = jnp.minimum(config.nu_del4,
                       config.nu_del4_cfl_frac / (dt * lam_v ** 2))
    return nu_u[:, None], nu_v[:, None]


def cgrid_latlon_sw_tendencies(
    state: CGridLatLonShallowWaterState,
    grid: LatLonGrid,
    config: CGridLatLonShallowWaterConfig = CGridLatLonShallowWaterConfig(),
    dt=None,
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

    # The grid-dispatched operator interface (B2): the differential operators flow
    # through ``ops.divergence``/``gradient``/``vorticity`` rather than the bare free
    # functions, so the core calls operators without naming the grid (design L2).
    # The adapter delegates to the same lat-lon C-grid operators, so this is
    # byte-identical to the direct calls.  The cell->face interpolations stay direct:
    # the adapter's ``interpolate(.., "vface")`` uses the grid-aware ``pad_ns_scalar``
    # north/south form, whereas this SW dycore uses the legacy ``grid=None`` (pole-
    # inert) form, so routing it would change the north-fold value on a tripolar grid.
    ops = latlon_cgrid_operators(grid)

    # --- 1. Mass flux divergence ---
    if config.use_ppm_transport:
        # PPM (4th-order) reconstruction of h at faces for upwind flux
        dh_dt = cgrid_fv_flux_divergence_latlon(h, u, v, grid)
    else:
        # Simple 2nd-order averaging of h to faces (legacy grid=None pole handling)
        h_u = interp_cell_to_uface(h)
        h_v = interp_cell_to_vface(h)
        F_u = h_u * u
        F_v = h_v * v
        dh_dt = -ops.divergence(F_u, F_v)

    # --- 2. Bernoulli function at cell centers ---
    KE = _kinetic_energy_cgrid(u, v)   # (n_lat, n_lon)
    B = g * (h + h_s) + KE             # (n_lat, n_lon)

    # --- 3. Pressure gradient + KE gradient (Bernoulli gradient) ---
    grad_x, grad_y = ops.gradient(B)
    du_dt = -grad_x   # (n_lat, n_lon+1)
    dv_dt = -grad_y   # (n_lat+1, n_lon)

    # --- 4. Coriolis using absolute vorticity (ζ+f) ---
    cor_u, cor_v = absolute_vorticity_coriolis(u, v, grid)
    du_dt = du_dt + cor_u
    dv_dt = dv_dt + cor_v

    # --- 5. Laplacian viscosity (optional) ---
    # Sign convention: viscosity enters as +A_h*lap(u) (Laplacian of a
    # local extremum is negative -> damping).
    if config.A_h > 0.0:
        lap_u, lap_v = vector_laplacian_cgrid(u, v, grid)
        du_dt = du_dt + config.A_h * lap_u
        dv_dt = dv_dt + config.A_h * lap_v

    # --- 5b. Biharmonic (del-4) viscosity (optional) ---
    # Sign convention: biharmonic damping is -nu4*lap(lap(u)) — for a
    # Fourier mode e^{ikx}, lap^2 -> +k^4, so the term is -nu4*k^4*u
    # (decay).  A "+" here would be anti-diffusive and blow up at the
    # grid scale.  Static Python branch on the config float (feature
    # gating exception — not jnp.where).
    if config.nu_del4 > 0.0:
        if dt is None:
            raise ValueError(
                "cgrid_latlon_sw_tendencies: nu_del4 > 0 requires the "
                "dt argument (per-row pole stability cap needs it).")
        lap_u, lap_v = vector_laplacian_cgrid(u, v, grid)
        lap2_u, lap2_v = vector_laplacian_cgrid(lap_u, lap_v, grid)
        nu_u, nu_v = _nu_del4_row_profiles(grid, config, dt)
        du_dt = du_dt - nu_u * lap2_u
        dv_dt = dv_dt - nu_v * lap2_v

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
        # Constructor dt, kept as the fallback for the no-argument
        # ``tendencies(state)`` path (the nu_del4 pole cap needs a dt).
        self._dt = dt
        # Precompute polar filter masks (cached, not traced): one for
        # cell-centered rows (dh, du after lon-trim) and one for v-face
        # rows (dv).  The v-face mask uses ``grid.cos_lat_v`` / the
        # half-cell-offset lat-interface coordinates so the wavenumber
        # cutoff matches the actual v-face CFL — mirrors the PE dycore
        # (``primitive_eq_latlon_cgrid``), which fixed exactly this:
        # filtering only dh/du leaves unfiltered high-k dv modes at the
        # pole-adjacent v rows to violate CFL at the relaxed dt.
        if self.config.use_polar_filter:
            self._polar_mask = compute_polar_filter_mask(
                grid, dt=dt,
                max_wave_speed=self.config.polar_filter_max_wave_speed,
                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
            )
            self._polar_mask_v = compute_polar_filter_mask(
                grid, dt=dt,
                max_wave_speed=self.config.polar_filter_max_wave_speed,
                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
                is_v_face=True,
            )
        else:
            self._polar_mask = None
            self._polar_mask_v = None

        # Iter-4: anchored mass target (fp64, lazy on first step()).
        # Mirrors ``CGridLatLonPrimitiveEquationModel._target_mass``.
        self._target_mass: jax.Array | None = None

    def reset_target_mass(self) -> None:
        """Clear the anchored mass target.

        iter-18: lets the caller re-snapshot the initial mass on the
        next ``step()`` call.  Useful when restarting integration from a
        different initial state without rebuilding the model object.
        """
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target.

        iter-19: complement to ``reset_target_mass`` — bypasses the
        lazy snapshot path when the caller already knows the target
        (e.g. restart from a checkpoint that recorded the original
        initial-condition mass).  Accepts any fp64 scalar.
        """
        self._target_mass = target_mass

    def compute_mass(self, state: CGridLatLonShallowWaterState) -> jax.Array:
        """Compute total mass (for conservation fixer target).

        Uses the fp64 conservation accumulator unconditionally — mass
        budgets need higher precision than the per-step compute dtype
        (``_accumulation_dtype`` may be fp32 on fp32 policies, which
        leaks ~10^-7 reduction noise into the anchored target and
        defeats the fixer; ``conservation_accumulator`` promotes to
        fp64 whenever x64 is enabled).
        """
        acc = conservation_accumulator()
        return jnp.sum(state.h.astype(acc) * self.grid.area.astype(acc))

    def tendencies(
        self, state: CGridLatLonShallowWaterState,
        dt: float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute tendencies (pure function wrapper).

        ``dt`` is only consumed when ``config.nu_del4 > 0`` (the del-4
        pole stability cap is dt-dependent); when omitted it falls back
        to the constructor ``dt`` so the no-argument
        ``model.tendencies(state)`` path (e.g. ``DycoreComponent``)
        keeps working with the biharmonic enabled.
        """
        if dt is None:
            dt = self._dt
        return cgrid_latlon_sw_tendencies(state, self.grid, self.config, dt)

    def step(
        self,
        state: CGridLatLonShallowWaterState,
        dt: float,
        target_mass: jax.Array | None = None,
    ) -> CGridLatLonShallowWaterState:
        """Outer wrapper: snapshots initial mass on first call when
        ``anchor_mass_to_initial`` is on (fp64, outside JIT), then
        delegates to the JIT'd inner step."""
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None
                and target_mass is None):
            self._target_mass = self.compute_mass(state)
        if (target_mass is None
                and self.config.fix_mass
                and self.config.anchor_mass_to_initial):
            target_mass = self._target_mass
        return self._step_jit(state, dt, target_mass)

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(
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
                s, self.grid, self.config, dt,
            )
            # Polar filter: damp high-frequency modes near poles
            if self._polar_mask is not None:
                dh = fourier_filter(dh, self.grid, self._polar_mask)
                du_interior = fourier_filter(du[:, :-1], self.grid, self._polar_mask)
                du = jnp.concatenate([du_interior, du_interior[:, 0:1]], axis=1)
                # dv on lat-interface (v-face) rows, with its own mask —
                # mirrors the PE dycore's polar_mask_v treatment.
                dv = fourier_filter(dv, self.grid, self._polar_mask_v)
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

        # Conservation fixer — use the fp64 budget accumulator so the
        # mass integrals aren't contaminated by fp32 reduction noise
        # (fp32 ``jnp.sum`` over 16k+ cells leaks ~N·eps relative
        # error and produces spurious O(1e-6) drift even when the
        # anchored target is exact).
        if self.config.fix_mass:
            acc = conservation_accumulator()
            area = self.grid.area.astype(acc)
            total_area = jnp.sum(area)
            if target_mass is not None:
                mass_target = target_mass
                mass_new = jnp.sum(state_new.h.astype(acc) * area)
            else:
                # Both mass integrals share the ``* area`` weight on
                # the same horizontal axes — stack and reduce once.
                _h_pair = jnp.stack(
                    [state.h.astype(acc), state_new.h.astype(acc)], axis=-1,
                ) * area[..., None]
                _mass_pair = jnp.sum(
                    _h_pair, axis=tuple(range(area.ndim)),
                )
                mass_target, mass_new = _mass_pair[0], _mass_pair[1]
            correction = (mass_target - mass_new) / total_area
            # Iter-4: drop ``.astype(state_new.h.dtype)`` so the
            # fp64 correction promotes the add (matches
            # ``fix_ps_mass`` semantics on cubed-sphere PE).  The
            # end-of-step ``cast_pytree(..., "storage")`` rounds back
            # to fp32 once instead of compounding fp32 quantization
            # every step.
            h_fixed = state_new.h + correction
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
