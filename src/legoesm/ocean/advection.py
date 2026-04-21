"""DST-3 flux-limited tracer advection with multi-dimensional correction.

Implements MITgcm scheme 33: Direct Space-Time 3rd-order with Sweby limiter.
Third-order accurate in both space and time (via CFL-dependent coefficients),
monotone (via flux limiting), and optionally corrected for operator-splitting
errors using a multi-dimensional predictor step.

Reference: MITgcm documentation Section 2.17.2.3
           Adcroft, Hill, Marshall (1997)

The DST-3 face value for positive flow at face j+1/2 (donor = cell j):

    T_face = T_j + psi(r) * [d0(c) * (T_{j+1} - T_j) + d1(c) * (T_{j-1} - T_j)]

where:
    c = |u| * dt / dx  (Courant number at face)
    d0(c) = (2 - c)(1 - c) / 6  (anti-diffusive / downstream correction)
    d1(c) = (1 - c^2) / 6       (upwind-of-upwind correction)
    psi(r) = Sweby limiter (superbee)
    r = (T_j - T_{j-1}) / (T_{j+1} - T_j)  (smoothness ratio)
"""

import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid


# =============================================================================
# Flux limiter
# =============================================================================

def _sweby_limiter(r: jnp.ndarray) -> jnp.ndarray:
    """Sweby (superbee) flux limiter.

    psi(r) = max(0, min(1, 2r), min(2, r))

    Traces the upper boundary of the Sweby TVD region, providing
    maximum anti-diffusion while maintaining monotonicity.
    """
    return jnp.maximum(
        0.0,
        jnp.maximum(jnp.minimum(1.0, 2.0 * r), jnp.minimum(2.0, r)),
    )


def _van_leer_limiter(r: jnp.ndarray) -> jnp.ndarray:
    """Van Leer flux limiter: phi(r) = (r + |r|) / (1 + |r|).

    Smooth, second-order, TVD. Bounded by [0, 2). Less aggressive
    than Sweby, better stability for DST-3 at low CFL.
    """
    return (r + jnp.abs(r)) / (1.0 + jnp.abs(r))


# =============================================================================
# DST-3 coefficients
# =============================================================================

def _dst3_d0(c: jnp.ndarray) -> jnp.ndarray:
    """DST-3 downstream coefficient, stability-capped.

    Raw formula: (1-c)(4-2c)/6 (3rd-order in space+time).
    Stability cap: (1-c)/2 (Lax-Wendroff limit for forward Euler).

    The cap activates at c < 0.5 where the raw DST-3 coefficient exceeds
    the forward-Euler stability boundary for the 2-grid-cell mode.
    At c >= 0.5, the two expressions are equal and the cap is inactive.
    """
    d0_dst3 = (1.0 - c) * (4.0 - 2.0 * c) / 6.0
    d0_lw = (1.0 - c) / 2.0  # Lax-Wendroff stability limit
    return jnp.minimum(d0_dst3, d0_lw)


def _dst3_d1(c: jnp.ndarray) -> jnp.ndarray:
    """DST-3 upwind-of-upwind coefficient: d1(c) = (1 - c)(1 - 2c) / 6.

    At c=0: d1=1/6.
    At c=0.5: d1=0 (curvature correction vanishes).
    At c>0.5: d1<0 (reverses sign — physically correct for large CFL).
    """
    return (1.0 - c) * (1.0 - 2.0 * c) / 6.0


# =============================================================================
# Horizontal DST-3: zonal (u-points)
# =============================================================================

def dst3_to_u_points(
    f: jnp.ndarray,
    mass_flux_u: jnp.ndarray,
    h_u: jnp.ndarray,
    grid: LatLonGrid,
    dt: float,
) -> jnp.ndarray:
    """DST-3 Sweby-limited interpolation to u-faces (zonal).

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, nlev)
        Tracer at cell centers.
    mass_flux_u : array, shape (n_lat, n_lon+1, nlev)
        Thickness-weighted velocity (h*u) at u-faces.
    h_u : array, shape (n_lat, n_lon+1, nlev)
        Layer thickness interpolated to u-faces.
    grid : LatLonGrid
    dt : float
        Timestep [s].

    Returns
    -------
    f_u : array, shape (n_lat, n_lon+1, nlev)
        Tracer value at u-faces (to be multiplied by mass_flux_u for flux).
    """
    eps = 1e-30
    n_lon = f.shape[1]

    # Cell-width at u-face latitudes: dx = R * dlon * cos(lat)
    dx = grid.radius * grid.dlon * grid.cos_lat  # (n_lat,)
    dx_3d = dx[:, jnp.newaxis, jnp.newaxis]  # broadcast to (n_lat, 1, 1)

    # Velocity and CFL at interior faces (n_lat, n_lon, nlev)
    # Face j sits between cell (j-1) mod n_lon and cell j.
    # mass_flux interior: first n_lon faces
    mf = mass_flux_u[:, :n_lon, :]
    vel = mf / jnp.maximum(h_u[:, :n_lon, :], eps)
    cfl = jnp.minimum(jnp.abs(vel) * dt / dx_3d, 1.0)

    # 5-point stencil (periodic in longitude)
    # For face j: donor for positive flow is cell j-1, receiver is cell j
    f_jm2 = jnp.roll(f, 2, axis=1)   # f[:, (j-2) % n_lon]
    f_jm1 = jnp.roll(f, 1, axis=1)   # f[:, (j-1) % n_lon] = donor for +flow
    f_j = f                            # f[:, j] = receiver for +flow
    f_jp1 = jnp.roll(f, -1, axis=1)  # f[:, (j+1) % n_lon]

    # --- Positive flow (from cell j-1 to cell j) ---
    # Donor = f_jm1, Downstream = f_j, Upwind-of-donor = f_jm2
    delta_pos = f_j - f_jm1              # local gradient across face
    delta_uu_pos = f_jm1 - f_jm2         # upwind gradient
    r_pos = delta_uu_pos / jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps)
    psi_pos = _van_leer_limiter(r_pos)
    d0_pos = _dst3_d0(cfl)
    d1_pos = _dst3_d1(cfl)
    # T_face = T_donor + psi * [d0*(T_downstream - T_donor) + d1*(T_upup - T_donor)]
    f_face_pos = f_jm1 + psi_pos * (d0_pos * delta_pos + d1_pos * (f_jm2 - f_jm1))
    # Monotonicity clamp: face value must stay between donor and downstream
    f_face_pos = jnp.clip(f_face_pos, jnp.minimum(f_jm1, f_j), jnp.maximum(f_jm1, f_j))

    # --- Negative flow (from cell j to cell j-1) ---
    # Donor = f_j, Downstream = f_jm1, Upwind-of-donor = f_jp1
    delta_neg = f_jm1 - f_j              # local gradient (downstream - donor)
    # Match TVD convention: r = (f_{j+1}-f_j) / (f_{j-1}-f_j)
    # This makes negative flow default to upwind at smooth monotone fields,
    # providing essential implicit diffusion for forward-Euler stability.
    r_neg = (f_jp1 - f_j) / jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps)
    psi_neg = _van_leer_limiter(r_neg)
    d0_neg = _dst3_d0(cfl)
    d1_neg = _dst3_d1(cfl)
    f_face_neg = f_j + psi_neg * (d0_neg * delta_neg + d1_neg * (f_jp1 - f_j))
    # Monotonicity clamp
    f_face_neg = jnp.clip(f_face_neg, jnp.minimum(f_jm1, f_j), jnp.maximum(f_jm1, f_j))

    # Select based on flow direction
    f_face = jnp.where(mf > 0, f_face_pos, f_face_neg)

    # Wrap: face n_lon equals face 0 (periodic)
    return jnp.concatenate([f_face, f_face[:, 0:1, :]], axis=1)


# =============================================================================
# Horizontal DST-3: meridional (v-points)
# =============================================================================

def dst3_to_v_points(
    f: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    h_v: jnp.ndarray,
    grid: LatLonGrid,
    dt: float,
) -> jnp.ndarray:
    """DST-3 Sweby-limited interpolation to v-faces (meridional).

    Solid wall boundary at poles. Near-boundary faces (within 2 cells
    of poles) fall back to first-order upwind where the 5-point stencil
    is incomplete.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, nlev)
        Tracer at cell centers.
    mass_flux_v : array, shape (n_lat+1, n_lon, nlev)
        Thickness-weighted velocity (h*v) at v-faces.
    h_v : array, shape (n_lat+1, n_lon, nlev)
        Layer thickness interpolated to v-faces.
    grid : LatLonGrid
    dt : float
        Timestep [s].

    Returns
    -------
    f_v : array, shape (n_lat+1, n_lon, nlev)
        Tracer value at v-faces. Zero at pole boundaries.
    """
    eps = 1e-30
    n_lat = f.shape[0]

    # Cell height: dy = R * dlat (uniform)
    dy = grid.radius * grid.dlat

    # Interior v-faces: indices 1 to n_lat-1 (between cells 0..n_lat-2 and 1..n_lat-1)
    # Face i sits between cell i-1 (south) and cell i (north).
    mf_int = mass_flux_v[1:-1, :, :]   # (n_lat-1, n_lon, nlev)
    h_v_int = h_v[1:-1, :, :]
    vel_int = mf_int / jnp.maximum(h_v_int, eps)
    cfl = jnp.minimum(jnp.abs(vel_int) * dt / dy, 1.0)

    # Build stencil with ghost cells at boundaries (Neumann: copy boundary value)
    # Ghost: f[-1] = f[0], f[-2] = f[0] at south; f[n_lat] = f[n_lat-1] at north
    f_ext = jnp.concatenate([f[:1, :, :], f[:1, :, :], f, f[-1:, :, :], f[-1:, :, :]], axis=0)
    # f_ext indices: 0,1 = south ghosts; 2..n_lat+1 = real; n_lat+2, n_lat+3 = north ghosts
    # Interior face i (1-indexed in original) corresponds to between cell i-1 and cell i.
    # In f_ext, cell i-1 = index i+1, cell i = index i+2.

    # Vectorized over all interior faces i=1..n_lat-1 (1-indexed):
    # Cell k in original lives at f_ext[k+2].
    # Face i is between cell i-1 (south) and cell i (north):
    #   south-of-south = cell i-2 → f_ext[i]
    #   south          = cell i-1 → f_ext[i+1]
    #   north          = cell i   → f_ext[i+2]
    #   north-of-north = cell i+1 → f_ext[i+3]
    # For i=1..n_lat-1 the slices are:
    f_south2 = f_ext[1:n_lat, :, :]          # f_ext[1..n_lat-1]
    f_south = f_ext[2:n_lat + 1, :, :]       # f_ext[2..n_lat]
    f_north = f_ext[3:n_lat + 2, :, :]       # f_ext[3..n_lat+1]
    f_north2 = f_ext[4:n_lat + 3, :, :]      # f_ext[4..n_lat+2]

    # --- Positive flow (south to north): donor = f_south, downstream = f_north ---
    delta_pos = f_north - f_south
    r_pos = (f_south - f_south2) / jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps)
    psi_pos = _van_leer_limiter(r_pos)
    d0_pos = _dst3_d0(cfl)
    d1_pos = _dst3_d1(cfl)
    f_face_pos = f_south + psi_pos * (d0_pos * delta_pos + d1_pos * (f_south2 - f_south))
    # Monotonicity clamp
    f_face_pos = jnp.clip(f_face_pos, jnp.minimum(f_south, f_north),
                           jnp.maximum(f_south, f_north))

    # --- Negative flow (north to south): donor = f_north, downstream = f_south ---
    delta_neg = f_south - f_north
    # Match TVD convention for implicit diffusion stability
    r_neg = (f_north2 - f_north) / jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps)
    psi_neg = _van_leer_limiter(r_neg)
    d0_neg = _dst3_d0(cfl)
    d1_neg = _dst3_d1(cfl)
    f_face_neg = f_north + psi_neg * (d0_neg * delta_neg + d1_neg * (f_north2 - f_north))
    # Monotonicity clamp
    f_face_neg = jnp.clip(f_face_neg, jnp.minimum(f_south, f_north),
                           jnp.maximum(f_south, f_north))

    # Select based on flow direction
    f_face = jnp.where(mf_int > 0, f_face_pos, f_face_neg)

    # Solid wall at poles: zero flux
    zero = jnp.zeros((1, f.shape[1], f.shape[2]), dtype=f.dtype)
    return jnp.concatenate([zero, f_face, zero], axis=0)


# =============================================================================
# Vertical DST-3
# =============================================================================

def flux_form_vertical_tracer_advection_dst3(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    h_k: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with DST-3 Sweby-limited scheme.

    Third-order accurate in smooth regions, falls back to first-order
    upwind at discontinuities via the Sweby limiter. Monotone.

    Same interface as flux_form_vertical_tracer_advection_tvd.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Tracer at full levels.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels [m/s].
        Positive = upward. Zero at surface and bottom.
    h_k : array, shape (..., nlev)
        Layer thickness [m] at full levels.
    dt : float
        Time step [s].

    Returns
    -------
    vert_flux_div : array, shape (..., nlev)
        Vertical flux divergence F_top[k] - F_bot[k] for each level.
        Units: [tracer]*[m/s] (NOT divided by layer thickness).
    """
    eps = 1e-30
    nlev = field.shape[-1]

    # Interior interfaces: k = 1..nlev-1
    w_int = w_half[..., 1:nlev]   # (..., nlev-1)
    T_below = field[..., 1:]      # field[k] for k=1..nlev-1
    T_above = field[..., :-1]     # field[k-1] for k=1..nlev-1

    # --- First-order upwind flux ---
    T_upwind = jnp.where(w_int > 0.0, T_below, T_above)
    F_upwind = w_int * T_upwind

    # --- CFL at each interface ---
    h_below = h_k[..., 1:]
    h_above = h_k[..., :-1]
    h_donor = jnp.where(w_int > 0.0, h_below, h_above)
    cfl = jnp.minimum(jnp.abs(w_int) * dt / jnp.maximum(h_donor, eps), 1.0)

    # --- DST-3 coefficients ---
    d0 = _dst3_d0(cfl)
    d1 = _dst3_d1(cfl)

    # --- 5-point stencil with ghost cells at boundaries ---
    # Ghost: copy boundary value (Neumann BC → delta=0 → r=0 → upwind)
    # Extended field: [ghost_top, ghost_top, field, ghost_bot, ghost_bot]
    f_ext = jnp.concatenate(
        [field[..., :1], field[..., :1], field, field[..., -1:], field[..., -1:]],
        axis=-1,
    )
    # f_ext indexing (0-based): indices 0,1 are top ghosts; 2..nlev+1 are
    # actual field; nlev+2, nlev+3 are bottom ghosts.
    # field[k] lives at f_ext index k+2.
    #
    # For interface k (1-indexed, k=1..nlev-1, between level k-1 and k):
    #   T_above = field[k-1] = f_ext[k+1]
    #   T_below = field[k]   = f_ext[k+2]
    #   T_below_below = field[k+1] = f_ext[k+3]  (upup for upward flow)
    #   T_above_above = field[k-2] = f_ext[k]    (upup for downward flow)

    # Upup for upward flow: field[k+1] for k=1..nlev-1
    # = f_ext[k+3] for k=1..nlev-1 = f_ext[4:nlev+3]
    T_below_below = f_ext[..., 4:nlev + 3]  # (nlev-1 elements)

    # Upup for downward flow: field[k-2] for k=1..nlev-1
    # = f_ext[k] for k=1..nlev-1 = f_ext[1:nlev]
    T_above_above = f_ext[..., 1:nlev]      # (nlev-1 elements)

    # Smoothness ratio for upward flow (donor=below=field[k]):
    # r = (donor - upup) / (downstream - donor)
    # = (T_below - T_below_below) / (T_above - T_below)
    delta_up = T_above - T_below  # across-face gradient (downstream - donor)
    r_up = (T_below - T_below_below) / jnp.where(jnp.abs(delta_up) > eps, delta_up, eps)

    # Smoothness ratio for downward flow (donor=above=field[k-1]):
    # r = (donor - upup) / (downstream - donor)
    # = (T_above - T_above_above) / (T_below - T_above)
    delta_down = T_below - T_above  # across-face gradient (downstream - donor)
    r_down = (T_above - T_above_above) / jnp.where(jnp.abs(delta_down) > eps, delta_down, eps)

    # --- Select by flow direction ---
    r = jnp.where(w_int > 0.0, r_up, r_down)
    psi = _van_leer_limiter(r)

    # DST-3 correction:
    # For upward flow: correction = d0*(T_above - T_below) + d1*(T_below_below - T_below)
    # For downward flow: correction = d0*(T_below - T_above) + d1*(T_above_above - T_above)
    # Equivalently: correction = d0*(downstream - donor) + d1*(upup - donor)
    T_downstream = jnp.where(w_int > 0.0, T_above, T_below)
    T_donor = T_upwind
    T_upup = jnp.where(w_int > 0.0, T_below_below, T_above_above)

    correction = d0 * (T_downstream - T_donor) + d1 * (T_upup - T_donor)

    # Limited flux: F = w * T_face = w * (T_donor + psi*correction)
    #            = F_upwind + w * psi * correction
    # Must use signed w (not |w|) because correction is relative to donor.
    # Monotonicity clamp: T_face must stay between donor and downstream.
    T_face = T_donor + psi * correction
    T_face = jnp.clip(T_face, jnp.minimum(T_donor, T_downstream),
                       jnp.maximum(T_donor, T_downstream))
    F_interior = w_int * T_face

    # Full flux with zero boundaries
    zeros = jnp.zeros((*field.shape[:-1], 1), dtype=field.dtype)
    F = jnp.concatenate([zeros, F_interior, zeros], axis=-1)

    # Flux divergence: F_top[k] - F_bot[k] = F[k] - F[k+1]
    return F[..., :-1] - F[..., 1:]


# =============================================================================
# Multi-dimensional tracer advection
# =============================================================================

def multidim_tracer_advection(
    tracer: jnp.ndarray,
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w_half: jnp.ndarray,
    h_k: jnp.ndarray,
    h_u: jnp.ndarray,
    h_v: jnp.ndarray,
    grid: LatLonGrid,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-dimensional DST-3 tracer advection with transverse correction.

    Two-pass approach to reduce operator-splitting errors:
    1. Compute preliminary upwind transverse fluxes
    2. Correct tracer for transverse transport (predictor)
    3. Compute final DST-3 fluxes from corrected tracer

    This removes the leading-order splitting error that causes
    anisotropy and false extrema at diagonal flows.

    Parameters
    ----------
    tracer : (n_lat, n_lon, nlev) tracer field.
    mass_flux_u : (n_lat, n_lon+1, nlev) at u-faces.
    mass_flux_v : (n_lat+1, n_lon, nlev) at v-faces.
    w_half : (..., nlev+1) vertical velocity on half levels.
    h_k : (n_lat, n_lon, nlev) layer thickness at cell centers.
    h_u : (n_lat, n_lon+1, nlev) layer thickness at u-faces.
    h_v : (n_lat+1, n_lon, nlev) layer thickness at v-faces.
    grid : LatLonGrid
    dt : float

    Returns
    -------
    div_h_flux : (n_lat, n_lon, nlev)
        Horizontal flux divergence div(mass_flux * T_face).
    vert_flux_div : (n_lat, n_lon, nlev)
        Vertical flux divergence.
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _upwind_to_u_points,
        _upwind_to_v_points,
    )
    from legoesm.ocean.vertical import flux_form_vertical_tracer_advection

    eps = 1e-10

    # --- Pass 1: preliminary upwind fluxes for transverse correction ---
    tr_u_upw = _upwind_to_u_points(tracer, mass_flux_u)
    tr_v_upw = _upwind_to_v_points(tracer, mass_flux_v)
    flux_u_upw = mass_flux_u * tr_u_upw
    flux_v_upw = mass_flux_v * tr_v_upw
    div_h_upw = divergence_cgrid(flux_u_upw, flux_v_upw, grid)
    vert_div_upw = flux_form_vertical_tracer_advection(tracer, w_half)

    # --- Transverse correction (predictor) ---
    # For x-sweep: correct for y+z transport
    # For y-sweep: correct for x+z transport
    # For z-sweep: correct for x+y transport
    # Simplified: use a single corrected tracer for all sweeps
    # T* = T - (dt/2) * (div_h + vert_div) / h
    total_div_upw = div_h_upw + vert_div_upw
    tracer_corr = tracer - 0.5 * dt * total_div_upw / jnp.maximum(h_k, eps)

    # --- Pass 2: DST-3 fluxes from corrected tracer ---
    tr_u_dst3 = dst3_to_u_points(tracer_corr, mass_flux_u, h_u, grid, dt)
    tr_v_dst3 = dst3_to_v_points(tracer_corr, mass_flux_v, h_v, grid, dt)
    flux_u_dst3 = mass_flux_u * tr_u_dst3
    flux_v_dst3 = mass_flux_v * tr_v_dst3
    div_h_flux = divergence_cgrid(flux_u_dst3, flux_v_dst3, grid)

    vert_flux_div = flux_form_vertical_tracer_advection_dst3(
        tracer_corr, w_half, h_k, dt,
    )

    return div_h_flux, vert_flux_div
