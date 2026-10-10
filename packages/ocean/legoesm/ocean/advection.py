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

import jax
import jax.numpy as jnp
from legoesm.core.weno import weno5_z, weno7_z, weno_upwind
from legoesm.grids.latlon import LatLonGrid
from legoesm.core.flux_limiters import grad_safe_ratio, ratio_grad_floor
from legoesm.ocean.dynamics._flux_limiters import (
    sweby_limiter as _superbee_limiter,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    divergence_cgrid,
    is_tripolar,
    vface_zonal_cos_lat,
)

# =============================================================================
# Flux limiter — DST-3 uses Van Leer (less aggressive than Sweby, better
# stability for DST-3 at low CFL). Centralized in ``core.flux_limiters``;
# Sweby (``_superbee_limiter``) imported above for the Veros-faithful path.
# =============================================================================

# Van Leer limiter is the canonical core kernel (redundancy audit) — import it
# instead of re-deriving phi(r) = (r+|r|)/(1+|r|); aliased to the local private
# name so call sites are unchanged.
from legoesm.core.flux_limiters import van_leer_limiter as _van_leer_limiter

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

    # Cell-width at u-face latitudes
    if is_tripolar(grid):
        dx_3d = grid.dx_u[:, :, jnp.newaxis]  # (n_lat, n_lon+1, 1)
    else:
        dx = grid.radius * grid.dlon * grid.cos_lat  # (n_lat,)
        dx_3d = dx[:, jnp.newaxis, jnp.newaxis]

    # Velocity and CFL at interior faces (n_lat, n_lon, nlev)
    # Face j sits between cell (j-1) mod n_lon and cell j.
    # mass_flux interior: first n_lon faces
    mf = mass_flux_u[:, :n_lon, :]
    t_grad = ratio_grad_floor(f.dtype)
    h_face = h_u[:, :n_lon, :]
    vel = grad_safe_ratio(mf, jnp.maximum(h_face, eps), h_face > t_grad)
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
    r_pos = grad_safe_ratio(
        delta_uu_pos,
        jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps),
        jnp.abs(delta_pos) > t_grad,
    )
    psi_pos = _van_leer_limiter(r_pos)
    d0_pos = _dst3_d0(cfl)
    d1_pos = _dst3_d1(cfl)
    # T_face = T_donor + psi * [d0*(T_downstream - T_donor) + d1*(T_upup - T_donor)]
    f_face_pos = f_jm1 + psi_pos * (d0_pos * delta_pos + d1_pos * (f_jm2 - f_jm1))
    # Monotonicity clamp: face value must stay between donor and downstream
    f_face_pos = jnp.clip(f_face_pos, jnp.minimum(f_jm1, f_j), jnp.maximum(f_jm1, f_j))

    # --- Negative flow (from cell j to cell j-1) ---
    # Donor = f_j, Downstream = f_jm1, Upwind-of-donor (upup) = f_jp1
    delta_neg = f_jm1 - f_j              # local gradient (downstream - donor)
    # Canonical DST-3 smoothness ratio (matches the vertical sibling
    # ``flux_form_vertical_tracer_advection_dst3`` and the positive-flow branch
    # above):  r = (donor - upup) / (downstream - donor).  On a smooth monotone
    # field donor-upup == downstream-donor -> r=+1 -> psi(1)=1 -> full 3rd-order,
    # SYMMETRIC with the u>0 branch.  The prior numerator (upup - donor) was
    # negated, giving r=-1 -> van_leer(-1)=0 -> 1st-order/over-diffusive for u<0
    # ONLY (direction-asymmetric diffusion).
    r_neg = grad_safe_ratio(
        f_j - f_jp1,
        jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps),
        jnp.abs(delta_neg) > t_grad,
    )
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

    # Face-to-face distance at interior v-faces.
    if is_tripolar(grid):
        # Tripolar: per-cell meridional spacing from 2D metrics.
        dy_v_int = grid.dy_v[1:-1, 0]  # (n_lat-1,) from interior rows
    else:
        # Regular or Mercator: variable-dy safe.
        dy_h = grid.dy * 0.5                                # (n_lat,)
        dy_v_int = 0.5 * (dy_h[1:] + dy_h[:-1])              # (n_lat-1,)

    # Interior v-faces: indices 1 to n_lat-1 (between cells 0..n_lat-2 and 1..n_lat-1)
    # Face i sits between cell i-1 (south) and cell i (north).
    mf_int = mass_flux_v[1:-1, :, :]   # (n_lat-1, n_lon, nlev)
    h_v_int = h_v[1:-1, :, :]
    t_grad = ratio_grad_floor(f.dtype)
    vel_int = grad_safe_ratio(
        mf_int, jnp.maximum(h_v_int, eps), h_v_int > t_grad)
    cfl = jnp.minimum(jnp.abs(vel_int) * dt / dy_v_int[:, jnp.newaxis, jnp.newaxis], 1.0)

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
    r_pos = grad_safe_ratio(
        f_south - f_south2,
        jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps),
        jnp.abs(delta_pos) > t_grad,
    )
    psi_pos = _van_leer_limiter(r_pos)
    d0_pos = _dst3_d0(cfl)
    d1_pos = _dst3_d1(cfl)
    f_face_pos = f_south + psi_pos * (d0_pos * delta_pos + d1_pos * (f_south2 - f_south))
    # Monotonicity clamp
    f_face_pos = jnp.clip(f_face_pos, jnp.minimum(f_south, f_north),
                           jnp.maximum(f_south, f_north))

    # --- Negative flow (north to south): donor = f_north, downstream = f_south,
    #     upwind-of-donor (upup) = f_north2 ---
    delta_neg = f_south - f_north        # downstream - donor
    # Canonical DST-3 smoothness ratio (matches the vertical sibling and the
    # positive-flow branch above):  r = (donor - upup) / (downstream - donor).
    # The prior numerator (upup - donor) = (f_north2 - f_north) was negated,
    # giving r=-1 on a smooth monotone field -> van_leer(-1)=0 -> 1st-order/
    # over-diffusive for v<0 ONLY (direction-asymmetric diffusion).
    r_neg = grad_safe_ratio(
        f_north - f_north2,
        jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps),
        jnp.abs(delta_neg) > t_grad,
    )
    psi_neg = _van_leer_limiter(r_neg)
    d0_neg = _dst3_d0(cfl)
    d1_neg = _dst3_d1(cfl)
    f_face_neg = f_north + psi_neg * (d0_neg * delta_neg + d1_neg * (f_north2 - f_north))
    # Monotonicity clamp
    f_face_neg = jnp.clip(f_face_neg, jnp.minimum(f_south, f_north),
                           jnp.maximum(f_south, f_north))

    # Select based on flow direction
    f_face = jnp.where(mf_int > 0, f_face_pos, f_face_neg)

    # Solid wall at poles: zero flux.  Single Pad HLO op replaces
    # alloc-zeros + concatenate-of-three (DST-3 hot path).
    return jnp.pad(f_face, ((1, 1), (0, 0), (0, 0)))


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

    # --- CFL at each interface ---
    h_below = h_k[..., 1:]
    h_above = h_k[..., :-1]
    h_donor = jnp.where(w_int > 0.0, h_below, h_above)
    t_grad = ratio_grad_floor(field.dtype)
    cfl = jnp.minimum(
        grad_safe_ratio(jnp.abs(w_int) * dt,
                        jnp.maximum(h_donor, eps),
                        h_donor > t_grad),
        1.0)

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
    r_up = grad_safe_ratio(
        T_below - T_below_below,
        jnp.where(jnp.abs(delta_up) > eps, delta_up, eps),
        jnp.abs(delta_up) > t_grad,
    )

    # Smoothness ratio for downward flow (donor=above=field[k-1]):
    # r = (donor - upup) / (downstream - donor)
    # = (T_above - T_above_above) / (T_below - T_above)
    delta_down = T_below - T_above  # across-face gradient (downstream - donor)
    r_down = grad_safe_ratio(
        T_above - T_above_above,
        jnp.where(jnp.abs(delta_down) > eps, delta_down, eps),
        jnp.abs(delta_down) > t_grad,
    )

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

    # Full flux with zero boundaries — single Pad HLO op vs alloc
    # ``(..., 1)`` zeros + 3-array concat.  Hot per scan step.
    pad_axes = ((0, 0),) * (F_interior.ndim - 1)
    F = jnp.pad(F_interior, (*pad_axes, (1, 1)))

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
        upwind_to_u_points,
        upwind_to_v_points,
    )
    from legoesm.ocean.vertical import flux_form_vertical_tracer_advection

    eps = 1e-10

    # --- Pass 1: preliminary upwind fluxes for transverse correction ---
    tr_u_upw = upwind_to_u_points(tracer, mass_flux_u)
    tr_v_upw = upwind_to_v_points(tracer, mass_flux_v)
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


# =============================================================================
# PPM (Piecewise Parabolic Method) — 4th-order reconstruction, monotone
# =============================================================================

def ppm_to_u_points(
    f: jnp.ndarray,
    mass_flux_u: jnp.ndarray,
) -> jnp.ndarray:
    """PPM-reconstructed tracer at u-faces (zonal).

    Uses 4th-order edge values with Colella-Woodward monotonicity limiter.
    No CFL dependence — works at any Courant number with forward Euler.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, nlev)
        Tracer at cell centers.
    mass_flux_u : array, shape (n_lat, n_lon+1, nlev)
        Thickness-weighted velocity at u-faces (sign determines upwind).

    Returns
    -------
    f_u : array, shape (n_lat, n_lon+1, nlev)
        PPM face values at u-points.
    """
    from legoesm.core.operators_fv import ppm_edge_values, ppm_limit

    n_lat, n_lon, nlev = f.shape

    # Pad longitude with halo=2 (periodic)
    f_pad = jnp.concatenate([f[:, -2:, :], f, f[:, :2, :]], axis=1)
    # f_pad shape: (n_lat, n_lon+4, nlev)

    # PPM edge values along longitude (axis=-2 must be the reconstruction dir)
    # Rearrange to (n_lat, n_lon+4, nlev) → axis=-2 is already longitude ✓
    # But ppm_edge_values operates on axis=-2 with shape (..., M, K)
    # We need (nlev, n_lat, n_lon+4) then compute along axis=-2=n_lon+4... no.
    # Actually ppm_edge_values needs shape (..., M, K) where M=n_lon+4 is the
    # direction. Our shape is (n_lat, n_lon+4, nlev): axis=-2=n_lon+4 ✓!
    q_hat = ppm_edge_values(f_pad)  # (n_lat, n_lon+3, nlev)

    # Left/right edge values for each cell
    a_L = q_hat[:, :-1, :]   # (n_lat, n_lon+2, nlev)
    a_R = q_hat[:, 1:, :]    # (n_lat, n_lon+2, nlev)
    q_c = f_pad[:, 1:-1, :]  # (n_lat, n_lon+2, nlev) — cell averages

    # Colella-Woodward limiter
    a_L, a_R = ppm_limit(q_c, a_L, a_R)

    # At face j (between cell j-1 and cell j):
    # - positive flow → use right edge of cell j-1 = a_R[j-1] (in padded coords: a_R[j])
    # - negative flow → use left edge of cell j = a_L[j] (in padded coords: a_L[j+1])
    # Interior faces in original coords: j=0..n_lon (n_lon+1 faces, wrapping)
    # In padded+reconstructed coords: j=0..n_lon maps to a_R[1..n_lon+1], a_L[2..n_lon+2]
    q_R_left = a_R[:, 1:n_lon + 2, :]   # (n_lat, n_lon+1, nlev)
    q_L_right = a_L[:, 2:n_lon + 3, :]  # (n_lat, n_lon+1, nlev)

    # But we have n_lon+2 elements in a_R/a_L. Let me recheck...
    # a_R shape = (n_lat, n_lon+2, nlev). Indices 0..n_lon+1.
    # For n_lon+1 faces (including periodic wrap):
    # Face j=0..n_lon: use a_R[1:n_lon+2] and a_L[2:n_lon+3]...
    # But a_L only has n_lon+2 elements (indices 0..n_lon+1), so a_L[2:n_lon+3]
    # would exceed bounds for large n_lon. Let me fix:
    # Face j in original (0-indexed, j=0..n_lon):
    #   Left cell = (j-1) mod n_lon → in padded: index j+1 (halo offset)
    #   Right cell = j → in padded: index j+2
    # a_R for left cell: a_R[j+1-1] = a_R[j] (right edge of padded cell j+1, but...)

    # Actually simpler: after PPM on (n_lat, n_lon+4, nlev), we get n_lon+3 edges.
    # Remove the outermost edges (fully in halo): keep inner n_lon+1 edges.
    # These correspond to faces 0..n_lon in the original grid.
    q_hat[:, 1:-1, :]  # (n_lat, n_lon+1, nlev) — interior edges

    # For each edge, the left cell's right-edge is the edge value approached from left,
    # and the right cell's left-edge is approached from right.
    # With monotone limiting, the upwind face value is:
    #   positive flow → right-edge of left cell
    #   negative flow → left-edge of right cell

    # Re-derive from the limited a_L, a_R:
    # a_L[i], a_R[i] are left/right edges of the i-th cell in the padded array.
    # Padded cells: 0(halo), 1(halo), 2..n_lon+1(real), n_lon+2(halo), n_lon+3(halo)
    # Real cells in padded: indices 2..n_lon+1
    # Faces between real cells: face j (0-indexed) is between real cell j and j+1
    #   = between padded cells j+2 and j+3
    #   Left cell right edge = a_R[j+2-1] = a_R[j+1]... no, a_R[k] is right edge of
    #   padded cell k. So right edge of padded cell j+2 = a_R[j+2].
    #   But wait: a_R has shape (n_lon+2) — indices 0..n_lon+1.
    #   Padded cells from which a_L/a_R are computed: cells 1..(n_lon+2) in f_pad
    #   (from q_c = f_pad[:, 1:-1, :] which is cells 1..n_lon+2, i.e. n_lon+2 cells)
    #   So a_L[k], a_R[k] for k=0..n_lon+1 correspond to padded cells 1..n_lon+2.
    #   Real data cells in padded: 2..n_lon+1 → a_L/a_R indices 1..n_lon.
    #   Face j between real cells j and j+1:
    #     = between a_L/a_R indices j+1 and j+2
    #     → left cell right edge = a_R[j+1]
    #     → right cell left edge = a_L[j+2]
    #   For j=0..n_lon-1: a_R[1..n_lon] and a_L[2..n_lon+1]
    #   For periodic face j=n_lon (=face 0): same as face 0.

    # Upwind selection for n_lon interior faces + 1 periodic wrap:
    q_R_left = a_R[:, 1:n_lon + 1, :]    # (n_lat, n_lon, nlev) — right edge of left cell
    q_L_right = a_L[:, 2:n_lon + 2, :]   # (n_lat, n_lon, nlev) — left edge of right cell

    mf = mass_flux_u[:, :n_lon, :]  # interior n_lon faces
    f_face = jnp.where(mf > 0, q_R_left, q_L_right)

    # Monotonicity clamp (local bounds of adjacent cells)
    f_left = jnp.roll(f, 1, axis=1)  # cell to left of face
    f_right = f                       # cell to right of face
    f_face = jnp.clip(f_face, jnp.minimum(f_left, f_right),
                       jnp.maximum(f_left, f_right))

    # Periodic wrap: face n_lon = face 0
    return jnp.concatenate([f_face, f_face[:, 0:1, :]], axis=1)


def ppm_to_v_points(
    f: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
) -> jnp.ndarray:
    """PPM-reconstructed tracer at v-faces (meridional).

    Solid wall at poles. Uses 4th-order PPM with Colella-Woodward limiter.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, nlev)
        Tracer at cell centers.
    mass_flux_v : array, shape (n_lat+1, n_lon, nlev)
        Thickness-weighted velocity at v-faces.

    Returns
    -------
    f_v : array, shape (n_lat+1, n_lon, nlev)
        PPM face values at v-points. Zero at pole boundaries.
    """
    from legoesm.core.operators_fv import ppm_limit

    n_lat, n_lon, nlev = f.shape

    # Extended field with ghost cells (Neumann BC: reflect boundary rows)
    f_ext = jnp.concatenate(
        [f[1::-1, :, :], f, f[-1:-3:-1, :, :]], axis=0)
    # f_ext shape: (n_lat+4, n_lon, nlev)
    # Indices: 0,1=south ghosts; 2..n_lat+1=real; n_lat+2,n_lat+3=north ghosts

    # 4th-order edge values at interior faces i=1..n_lat-1:
    # Face i is between cell i-1 and cell i in original.
    # In f_ext: cell i-1=index i+1, cell i=index i+2.
    # a_i = (7/12)*(f_ext[i+1]+f_ext[i+2]) - (1/12)*(f_ext[i]+f_ext[i+3])
    s0 = f_ext[1:n_lat, :, :]      # f_ext[i] for i=1..n_lat-1
    s1 = f_ext[2:n_lat + 1, :, :]  # cell i-1
    s2 = f_ext[3:n_lat + 2, :, :]  # cell i
    s3 = f_ext[4:n_lat + 3, :, :]  # f_ext[i+3]

    a_int = (7.0 / 12.0) * (s1 + s2) - (1.0 / 12.0) * (s0 + s3)
    # Monotone clamp between neighbors
    a_int = jnp.clip(a_int, jnp.minimum(s1, s2), jnp.maximum(s1, s2))
    # shape: (n_lat-1, n_lon, nlev)

    # Build full edge array (n_lat+1 interfaces):
    # Face 0=south wall, faces 1..n_lat-1=interior, face n_lat=north wall
    a_full = jnp.concatenate([f[:1, :, :], a_int, f[-1:, :, :]], axis=0)
    # shape: (n_lat+1, n_lon, nlev)

    # Left/right edges per cell:
    # Cell j: T_L=a_full[j] (south edge), T_R=a_full[j+1] (north edge)
    T_L = a_full[:-1, :, :]  # (n_lat, n_lon, nlev)
    T_R = a_full[1:, :, :]

    # Colella-Woodward limiter
    T_L, T_R = ppm_limit(f, T_L, T_R)

    # Upwind face value at interior faces:
    # Positive flow (south→north): donor=cell i-1, use T_R of cell i-1
    # Negative flow (north→south): donor=cell i, use T_L of cell i
    T_R_south = T_R[:-1, :, :]  # T_R of cells 0..n_lat-2
    T_L_north = T_L[1:, :, :]   # T_L of cells 1..n_lat-1

    mf_int = mass_flux_v[1:-1, :, :]
    f_face = jnp.where(mf_int > 0, T_R_south, T_L_north)

    # Monotonicity clamp
    f_south = f[:-1, :, :]
    f_north = f[1:, :, :]
    f_face = jnp.clip(f_face, jnp.minimum(f_south, f_north),
                       jnp.maximum(f_south, f_north))

    # Solid wall at poles: zero flux.  Single Pad HLO op replaces
    # alloc-zeros + concatenate-of-three (PPM hot path).
    return jnp.pad(f_face, ((1, 1), (0, 0), (0, 0)))


def flux_form_vertical_tracer_advection_ppm(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    h_k: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with PPM reconstruction.

    4th-order edge values with Colella-Woodward monotonicity limiter.
    Same interface as flux_form_vertical_tracer_advection_tvd.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Tracer at full levels.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels [m/s].
    h_k : array, shape (..., nlev)
        Layer thickness [m] at full levels.
    dt : float
        Time step [s] (unused — PPM doesn't need CFL).

    Returns
    -------
    vert_flux_div : array, shape (..., nlev)
        Vertical flux divergence F_top[k] - F_bot[k].
    """
    from legoesm.core.operators_fv import ppm_edge_values, ppm_limit

    nlev = field.shape[-1]

    # Pad with halo=2 in the vertical (Neumann: copy boundary values)
    f_pad = jnp.concatenate(
        [field[..., 1::-1], field, field[..., -1:-3:-1]], axis=-1)
    # shape: (..., nlev+4)

    # For ppm_edge_values, we need shape (..., M, K) where M=nlev+4 is
    # the reconstruction direction. Add a dummy trailing dimension.
    f_pad_2d = f_pad[..., jnp.newaxis]  # (..., nlev+4, 1)
    q_hat_2d = ppm_edge_values(f_pad_2d)  # (..., nlev+3, 1)
    q_hat = q_hat_2d[..., 0]  # (..., nlev+3)

    # Left/right edges
    a_L = q_hat[..., :-1]   # (..., nlev+2)
    a_R = q_hat[..., 1:]    # (..., nlev+2)
    q_c = f_pad[..., 1:-1]  # (..., nlev+2)

    # Add trailing dim for ppm_limit (needs consistent shapes)
    a_L_2d = a_L[..., jnp.newaxis]
    a_R_2d = a_R[..., jnp.newaxis]
    q_c_2d = q_c[..., jnp.newaxis]
    a_L_2d, a_R_2d = ppm_limit(q_c_2d, a_L_2d, a_R_2d)
    a_L = a_L_2d[..., 0]
    a_R = a_R_2d[..., 0]

    # Interior interfaces: k=1..nlev-1
    # Real cells in padded: indices 2..nlev+1 → a_L/a_R indices 1..nlev
    # Interface k is between level k-1 (above) and level k (below):
    #   = between a_L/a_R indices k and k+1
    #   above cell right edge (bottom) = a_R[k]
    #   below cell left edge (top) = a_L[k+1]
    q_R_above = a_R[..., 1:nlev]       # (..., nlev-1) — bottom edge of cell above
    q_L_below = a_L[..., 2:nlev + 1]   # (..., nlev-1) — top edge of cell below

    w_int = w_half[..., 1:nlev]  # interior interfaces

    # Upward flow (w>0): fluid comes from below → use top edge of below cell
    # Downward flow (w<0): fluid comes from above → use bottom edge of above cell
    T_face = jnp.where(w_int > 0.0, q_L_below, q_R_above)

    # Monotonicity clamp
    T_above = field[..., :-1]
    T_below = field[..., 1:]
    T_face = jnp.clip(T_face, jnp.minimum(T_above, T_below),
                       jnp.maximum(T_above, T_below))

    # Flux at interfaces
    F_interior = w_int * T_face

    # Zero-flux boundaries — single Pad HLO op.
    pad_axes_b = ((0, 0),) * (F_interior.ndim - 1)
    F = jnp.pad(F_interior, (*pad_axes_b, (1, 1)))

    # Flux divergence
    return F[..., :-1] - F[..., 1:]


# =============================================================================
# FCT (Flux-Corrected Transport) — PPM accuracy with upwind stability
# =============================================================================

def centred2_to_u_points(f: jnp.ndarray) -> jnp.ndarray:
    """2nd-order centred tracer at u-faces: 0.5·(T_west + T_east).

    NEMO ``traadv_fct`` high-order horizontal flux with ``nn_fct_h=2``
    (``0.5*pU*(pt(ji)+pt(ji+1))``).  Periodic in longitude; returns
    (n_lat, n_lon+1, nlev) with the wrap column appended.
    """
    f_face = 0.5 * (jnp.roll(f, 1, axis=1) + f)          # face j: cells j-1, j
    return jnp.concatenate([f_face, f_face[:, 0:1, :]], axis=1)


def centred2_to_v_points(f: jnp.ndarray) -> jnp.ndarray:
    """2nd-order centred tracer at v-faces: 0.5·(T_south + T_north).

    Wall faces (j=0, n_lat) copy the adjacent cell — their mass flux is
    zero so the value only needs to be finite. Returns (n_lat+1, n_lon,
    nlev).
    """
    f_int = 0.5 * (f[:-1, :, :] + f[1:, :, :])           # interior n_lat-1 faces
    return jnp.concatenate([f[:1, :, :], f_int, f[-1:, :, :]], axis=0)


FCT_HIGH_ORDER_SCHEMES = ("ppm", "centred2")
NEMO_FCT_TRACE_FIELDS = (
    "first_u", "first_v", "first_w", "first_div", "midpoint",
    "average_u", "average_v", "average_w", "upstream_div",
    "rhs_after_up", "anti_pre_u", "anti_pre_v", "anti_pre_w",
    "coef_u", "coef_v", "coef_w", "anti_post_u", "anti_post_v",
    "anti_post_w", "final_div", "divisor", "rhs_final",
)
NEMO_FCT_BETA_TRACE_FIELDS = (
    "zup", "zdo", "zpos", "zneg", "zbt",
    "zbetup_literal", "zbetdo_literal", "r_in", "r_out",
    "coef_u", "coef_v", "coef_w",
)
NEMO_FCT_STENCIL_TRACE_FIELDS = (
    "zbup_center", "zbup_west", "zbup_east", "zbup_south",
    "zbup_north", "zbup_above", "zbup_below",
    "pbef", "paft", "wet", "zup",
)
NEMO_FCT_UP1_TRACE_FIELDS = (
    "first_u_raw", "first_v_raw", "first_w_raw", "first_div",
    "midpoint", "average_u_raw", "average_v_raw", "average_w_raw",
    "explicit_ztra", "implicit_ztra", "total_ztra", "base_content",
    "dt_ztra", "numerator", "after_thickness", "paft",
)


def fct_tracer_advection(
    tracer: jnp.ndarray,
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w_half: jnp.ndarray,
    h_k: jnp.ndarray,
    grid: "LatLonGrid",
    dt: float,
    high_order: str = "ppm",
    tracer_before: jnp.ndarray | None = None,
    active_mask: jnp.ndarray | None = None,
    fixed_thickness: bool = False,
    low_order_predictor: str = "one_step",
    base_thickness: jnp.ndarray | None = None,
    after_thickness: jnp.ndarray | None = None,
    implicit_w: jnp.ndarray | None = None,
    return_nemo_split: bool = False,
    return_nemo_trace: bool = False,
    return_nemo_beta_trace: bool = False,
    return_nemo_stencil_trace: bool = False,
    return_nemo_up1_trace: bool = False,
    return_limiter_activity: bool = False,
) -> tuple:
    """FCT tracer advection: high-order accuracy with guaranteed monotonicity.

    Combines first-order upwind (inherently stable) with a high-order flux
    using a Zalesak limiter that adds maximum anti-diffusion without
    creating new extrema.

    The scheme is:
    - Conservative (flux-form)
    - Monotone (Zalesak bounds on total tendency)
    - Stable at any CFL (starts from upwind)
    - Higher accuracy than TVD at fronts

    Parameters
    ----------
    tracer : (n_lat, n_lon, nlev)
    mass_flux_u : (n_lat, n_lon+1, nlev)
    mass_flux_v : (n_lat+1, n_lon, nlev)
    w_half : (n_lat, n_lon, nlev+1)
    h_k : (n_lat, n_lon, nlev) layer thickness
    grid : LatLonGrid
    dt : float
    high_order : {"ppm", "centred2"}
        High-order flux: "ppm" (4th-order PPM reconstruction, the
        historical ``ppm_fct``) or "centred2" (plain 2nd-order centred
        mean, NEMO ``traadv_fct`` with ``nn_fct_h = nn_fct_v = 2`` — the
        DINO / ORCA1 namelist selection). The centred face value is NOT
        pre-clamped to local bounds (NEMO doesn't); the Zalesak step
        supplies all the monotonicity.
    tracer_before : (n_lat, n_lon, nlev) or None
        BEFORE-level tracer (Kbb) for the leapfrog outer step.  Under the
        modified leap-frog the FCT-limited advective increment is applied to
        the BEFORE state (``T(Naa) = T(Nbb) + 2dt·RHS``), so the Zalesak
        monotonicity base — the first-order upwind low-order flux, the
        provisional low-order update ``q_td``, and the local ``q_min``/``q_max``
        bounds — must be taken from the BEFORE level, exactly as NEMO
        ``traadv_fct``: ``fct_up1(...,pt(Kbb))`` for the upstream flux and
        ``nonosc(Kbb, ..., p2dt=2dt)`` for the bounds, while the high-order
        ANTIDIFFUSIVE face is still built from the NOW level (Kmm) via
        ``0.5·pU·(pt(Kmm)+pt(Kmm))``.  ``None`` (forward-Euler / AB2 path)
        ⇒ base == ``tracer`` (Kbb == Kmm) ⇒ byte-identical to the FE-certified
        scheme.
    active_mask : (n_lat, n_lon, nlev) or None
        Per-cell wet mask (``is_active`` / ``active_3d``), truthy where
        wet.  NEMO's ``nonosc`` masks the per-point bound to
        ``MERGE(max(pbef,paft), -zbig, tmask==1)`` / ``MERGE(min(...),
        +zbig, tmask==1)`` (traadv_fct.F90:911-915) BEFORE the 7-point
        neighbourhood max/min, so a dry cell's ``q_td`` (an unconstrained
        ``h_k→0`` division that legoESM does not bother to make sane,
        since it is masked out of the tracer update anyway) never widens
        a WET neighbour's box.  ``None`` (default) skips the mask — the
        historical behaviour, which lets a dry cell's ``q_td`` blow-up
        (``0/eps``) leak into the neighbourhood stencil and, downstream,
        into that neighbour's ``R_in``/``R_out`` — the root cause of
        legoESM's w-face clip count running ~1.9x NEMO's on the DINO
        oracle (#1226 item 8: the flux values already matched NEMO at
        corr > 0.9999 pre-fix; only the boundedness of the LIMITER inputs
        at dry cells was unfaithful).  Passing the mask is a strict
        no-op away from dry/wet boundaries.
    fixed_thickness : bool
        Static Python bool.  True under key_linssh (fixed layer
        thicknesses; the caller adds the surface concentration/dilution
        flux separately after limiting): the limiter certifies against
        ``h_k`` itself.  False (z-star default): the AFTER thickness
        ``h_new = h_k - dt*div(mf)`` is derived in the body and the
        Zalesak box is certified against it -- see the h_new block.
    return_limiter_activity : bool
        Private write-only diagnostic. If true, append a cell-centred bool
        map for cells incident to a non-zero antidiffusive face whose limiter
        coefficient is below one. False preserves the ordinary two-array
        return.
    return_nemo_trace : bool
        Private write-only fidelity trace of the values corresponding to
        NEMO's compiled two-step FCT stores.  This requires
        ``low_order_predictor="nemo_rk3_two_step"`` and cannot be combined
        with another diagnostic return.  False preserves the ordinary return.
    return_nemo_beta_trace : bool
        Private write-only readout of the source-aligned limiter-cell operands
        and the live face coefficients.  Its duplicated arithmetic is kept
        behind an XLA optimization barrier; the ordinary limiter graph remains
        the one that supplies the two production outputs.
    return_nemo_stencil_trace : bool
        Private write-only readout of the seven source-ordered ``zbup`` inputs,
        their ``pbef``/``paft`` sources, wet mask, and resulting ``zup``.  False
        preserves the ordinary return.

    Returns
    -------
    div_h_flux : (n_lat, n_lon, nlev) horizontal flux divergence
    vert_flux_div : (n_lat, n_lon, nlev) vertical flux divergence
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        upwind_to_u_points, upwind_to_v_points,
    )

    if high_order not in FCT_HIGH_ORDER_SCHEMES:
        raise ValueError(
            f"Unknown FCT high_order scheme '{high_order}'; "
            f"expected one of {FCT_HIGH_ORDER_SCHEMES}")
    if low_order_predictor not in ("one_step", "nemo_rk3_two_step"):
        raise ValueError(
            f"Unknown FCT low_order_predictor {low_order_predictor!r}; "
            "expected 'one_step' or 'nemo_rk3_two_step'")
    trace_returns = (
        return_nemo_trace, return_nemo_beta_trace, return_nemo_stencil_trace,
        return_nemo_up1_trace)
    if any(trace_returns):
        if return_nemo_split or return_limiter_activity:
            raise ValueError(
                "NEMO trace returns cannot be combined with another "
                "diagnostic return")
        if sum(trace_returns) > 1:
            raise ValueError("NEMO trace returns are mutually exclusive")
        if low_order_predictor != "nemo_rk3_two_step":
            raise ValueError(
                "NEMO trace returns require the NEMO RK3 two-step predictor")
    if return_nemo_stencil_trace and active_mask is None:
        raise ValueError("NEMO stencil trace requires the compiled wet mask")

    eps = 1e-30

    # Monotonicity base (Kbb under leapfrog; == tracer on the FE/AB2 path).
    # The low-order upwind flux, the vertical upwind interface flux, the
    # provisional low-order update ``q_td``, and the ``q_min``/``q_max`` stencil
    # bounds are ALL built from ``base`` — NEMO ``fct_up1(pt(Kbb))`` +
    # ``nonosc(Kbb)``.  The high-order antidiffusive faces stay on ``tracer``
    # (Kmm).  ``tracer_before is None`` ⇒ ``base is tracer`` ⇒ byte-identical.
    base = tracer if tracer_before is None else tracer_before

    # --- Step 1: Horizontal face fluxes (low and high order) ---
    tr_u_low = upwind_to_u_points(base, mass_flux_u)
    tr_v_low = upwind_to_v_points(base, mass_flux_v)
    flux_u_low = mass_flux_u * tr_u_low
    flux_v_low = mass_flux_v * tr_v_low
    div_h_low = divergence_cgrid(flux_u_low, flux_v_low, grid)
    if return_nemo_trace:
        trace_first_u = flux_u_low * jnp.asarray(grid.dy_u)[..., None]
        trace_first_v = flux_v_low * jnp.asarray(grid.dx_v)[..., None]

    if high_order == "ppm":
        tr_u_hi = ppm_to_u_points(tracer, mass_flux_u)
        tr_v_hi = ppm_to_v_points(tracer, mass_flux_v)
    else:
        tr_u_hi = centred2_to_u_points(tracer)
        tr_v_hi = centred2_to_v_points(tracer)
    flux_u_hi = mass_flux_u * tr_u_hi
    flux_v_hi = mass_flux_v * tr_v_hi

    # --- Step 2: Vertical interface fluxes (low and high order) ---
    nlev = tracer.shape[-1]
    w_int = w_half[..., 1:nlev]  # interior interfaces (..., nlev-1)
    T_below = tracer[..., 1:]    # (..., nlev-1) NOW level (Kmm), high-order
    T_above = tracer[..., :-1]   # (..., nlev-1)

    # Upwind interface flux from the BEFORE (Kbb) base level
    Tb_below = base[..., 1:]     # (..., nlev-1)
    Tb_above = base[..., :-1]    # (..., nlev-1)
    T_face_low = jnp.where(w_int > 0.0, Tb_below, Tb_above)
    F_vert_low_int = w_int * T_face_low  # (..., nlev-1)

    if high_order == "ppm":
        # PPM interface flux
        from legoesm.core.operators_fv import ppm_edge_values, ppm_limit
        f_pad = jnp.concatenate(
            [tracer[..., 1::-1], tracer, tracer[..., -1:-3:-1]], axis=-1)
        f_pad_2d = f_pad[..., jnp.newaxis]
        q_hat = ppm_edge_values(f_pad_2d)[..., 0]
        a_L = q_hat[..., :-1]
        a_R = q_hat[..., 1:]
        q_c = f_pad[..., 1:-1]
        a_L_2d, a_R_2d = ppm_limit(
            q_c[..., jnp.newaxis], a_L[..., jnp.newaxis],
            a_R[..., jnp.newaxis])
        a_L, a_R = a_L_2d[..., 0], a_R_2d[..., 0]
        q_R_above = a_R[..., 1:nlev]
        q_L_below = a_L[..., 2:nlev + 1]
        T_face_hi = jnp.where(w_int > 0.0, q_L_below, q_R_above)
        T_face_hi = jnp.clip(T_face_hi,
                             jnp.minimum(T_above, T_below),
                             jnp.maximum(T_above, T_below))
    else:
        # NEMO nn_fct_v=2: plain centred interface mean, no clamp
        T_face_hi = 0.5 * (T_above + T_below)
    F_vert_hi_int = w_int * T_face_hi

    # Vertical divergences for Zalesak bounds computation — pad
    # (single Pad HLO each) instead of allocating a fresh ``(..., 1)``
    # zero buffer and concatenating it on both ends.
    pad_axes_v = ((0, 0),) * (F_vert_low_int.ndim - 1)
    F_vert_low = jnp.pad(F_vert_low_int, (*pad_axes_v, (1, 1)))
    vert_div_low = F_vert_low[..., :-1] - F_vert_low[..., 1:]
    if return_nemo_up1_trace:
        up1_first_u = flux_u_low
        up1_first_v = flux_v_low
        up1_first_w = F_vert_low
        up1_first_div = -(div_h_low + vert_div_low)
    if return_nemo_trace:
        trace_first_w = F_vert_low * jnp.asarray(grid.area_T)[..., None]
        trace_first_div = -(div_h_low + vert_div_low)

    # NEMO key_RK3 does not use the ordinary one-step upstream predictor.
    # traadv_fct.F90:493-537 first advances Kbb by pDt/2 with upstream
    # fluxes to a Kmm-thickness midpoint.  Lines 560-607 then replace every
    # upstream face flux by the arithmetic mean of the Kbb and midpoint
    # upstream fluxes.  That averaged flux supplies BOTH the low-order guess
    # and the antidiffusive difference consumed by nonosc.
    h_base = h_k if base_thickness is None else base_thickness
    if low_order_predictor == "nemo_rk3_two_step":
        implicit_mass_div = 0.0
        if implicit_w is not None:
            if implicit_w.shape != tracer.shape[:-1] + (nlev + 1,):
                raise ValueError("implicit_w must contain nlev+1 interfaces")
            # Resolved nn_fct_imp=1: traadv_fct.F90:528-536 subtracts
            # (wi_top-wi_bottom)*T(Kbb) in the half-step predictor; the same
            # zero-order term is used again at :598-607 for the full predictor.
            implicit_mass_div = (
                implicit_w[..., :-1] - implicit_w[..., 1:]) * base
        q_mid = grad_safe_ratio(
            h_base * base - (0.5 * dt) * (
                div_h_low + vert_div_low + implicit_mass_div),
            jnp.maximum(h_k, eps),
            h_k > ratio_grad_floor(tracer.dtype),
        )
        if active_mask is not None:
            q_mid = jnp.where(active_mask > 0.5, q_mid, base)
        qmid_u = upwind_to_u_points(q_mid, mass_flux_u)
        qmid_v = upwind_to_v_points(q_mid, mass_flux_v)
        flux_u_low = 0.5 * (flux_u_low + mass_flux_u * qmid_u)
        flux_v_low = 0.5 * (flux_v_low + mass_flux_v * qmid_v)
        div_h_low = divergence_cgrid(flux_u_low, flux_v_low, grid)
        qmid_below = q_mid[..., 1:]
        qmid_above = q_mid[..., :-1]
        qmid_face = jnp.where(w_int > 0.0, qmid_below, qmid_above)
        F_vert_low_int = 0.5 * (
            F_vert_low_int + w_int * qmid_face)
        F_vert_low = jnp.pad(F_vert_low_int, (*pad_axes_v, (1, 1)))
        vert_div_low = F_vert_low[..., :-1] - F_vert_low[..., 1:]

    if return_nemo_up1_trace:
        up1_explicit_ztra = -(div_h_low + vert_div_low)
        up1_implicit_ztra = -implicit_mass_div
        up1_total_ztra = up1_explicit_ztra + up1_implicit_ztra
        up1_base_content = h_base * base
        up1_dt_ztra = dt * up1_total_ztra
        up1_numerator = up1_base_content + up1_dt_ztra

    if return_nemo_trace:
        trace_average_u = flux_u_low * jnp.asarray(grid.dy_u)[..., None]
        trace_average_v = flux_v_low * jnp.asarray(grid.dx_v)[..., None]
        trace_average_w = F_vert_low * jnp.asarray(grid.area_T)[..., None]
        trace_upstream_div = -(div_h_low + vert_div_low)

    # --- Step 3: True sign-split Zalesak (1979) limiter (issue #212) ---
    # Anti-diffusive face fluxes:
    ad_flux_u = flux_u_hi - flux_u_low      # (n_lat, n_lon+1, nlev)
    ad_flux_v = flux_v_hi - flux_v_low      # (n_lat+1, n_lon, nlev)
    ad_vert_int = F_vert_hi_int - F_vert_low_int  # (..., nlev-1)
    if return_nemo_trace:
        trace_anti_pre_u = ad_flux_u * jnp.asarray(grid.dy_u)[..., None]
        trace_anti_pre_v = ad_flux_v * jnp.asarray(grid.dx_v)[..., None]
        trace_anti_pre_w = jnp.pad(
            ad_vert_int, (*pad_axes_v, (1, 1))) * jnp.asarray(
                grid.area_T)[..., None]

    # AFTER thickness from the SAME advecting fluxes: under z-star the
    # caller's flux-form update divides by h_new = h_k - dt*div(mf), so the
    # provisional low-order update and the Zalesak budgets MUST be
    # normalised by h_new -- NEMO traadv_fct's ``zwi = (e3t(Kbb)*pt(Kbb) -
    # p2dt*ztra) / e3t(Kaa)``.  Certifying the box against h_OLD (the
    # pre-2026-08-10 behaviour) left the ACTUAL update outside the box by
    # exactly T*dt*div(mf)/h under divergent flow: measured overshoot
    # 1.18365e-3 == 30 * 3.945e-5 (= T*max|dt*div/h|) on the repro, and
    # +0.12/-0.03 K per day at the lock-exchange front (eta and the front
    # are correlated, so the mis-sizing is systematic, not noise).
    # Solenoidal flow: h_new == h_k, bit-identical to the old behaviour.
    #
    # ``fixed_thickness`` (key_linssh, static Python bool): the coordinate
    # keeps thicknesses FIXED and the caller adds the surface
    # concentration/dilution flux separately AFTER limiting, so the
    # after-thickness the update divides by IS h_k -- deriving h_new from
    # the interior fluxes there would mis-certify (codex 2026-08-10).
    #
    # LEAPFROG CAVEAT: under the outer leapfrog (tracer_before set,
    # dt = 2*rdt), the caller passes the NOW-eta thickness as h_k while
    # NEMO's zwi uses e3t(Kbb) -> e3t(Kaa); the certification there is
    # approximate (same class as the pre-fix behaviour on ALL paths).
    # Exact leapfrog certification needs the BEFORE thickness threaded --
    # flagged, not fixed here.  Only the EULER consumer is exactly
    # certified; inner AB2 extrapolates this limited divergence with a
    # history term before the thickness division, which no single-step
    # certificate covers (the pre-existing AB2 limitation).
    if after_thickness is not None:
        h_new = after_thickness
    elif fixed_thickness:
        h_new = h_k
    else:
        div_mf_h = divergence_cgrid(mass_flux_u, mass_flux_v, grid)
        w_full = jnp.pad(w_int, (*pad_axes_v, (1, 1)))
        vert_div_mf = w_full[..., :-1] - w_full[..., 1:]
        h_new = h_k - dt * (div_mf_h + vert_div_mf)
    t_grad_h = ratio_grad_floor(tracer.dtype)

    # Provisional low-order (upwind) update in AFTER-thickness form.
    q_td = grad_safe_ratio(
        h_base * base - dt * (
            div_h_low + vert_div_low + (
                (implicit_w[..., :-1] - implicit_w[..., 1:]) * base
                if implicit_w is not None else 0.0)),
        jnp.maximum(h_new, eps),
        h_new > t_grad_h,
    )
    if return_nemo_up1_trace:
        up1_trace = jax.lax.optimization_barrier((
            up1_first_u, up1_first_v, up1_first_w, up1_first_div,
            q_mid, flux_u_low, flux_v_low, F_vert_low,
            up1_explicit_ztra, up1_implicit_ztra, up1_total_ztra,
            up1_base_content, up1_dt_ztra, up1_numerator, h_new, q_td,
        ))

    # Local min / max over the (cell + 6 neighbours) stencil.  For non-
    # cyclic latitude the boundary cell is its own south/north neighbour
    # (copy BC); periodic in lon; vertical clamps to top/bottom layer.
    #
    # NEMO nonosc (traadv_fct.F90:876-880, 912-920): the PER-POINT bound at
    # each stencil cell is ``bnd_up = max(pbef, paft)`` / ``bnd_do =
    # min(pbef, paft)`` where ``paft`` is ``zta_up1`` — the upstream
    # provisional guess, i.e. exactly this function's ``q_td`` — NOT ``pbef``
    # (Kbb/``base``) alone.  The 7-point neighbourhood max/min is then taken
    # over that per-point ``bnd_up``/``bnd_do`` field.  Building the
    # neighbourhood from ``base`` alone (the prior legoESM behaviour) drops
    # the ``q_td`` contribution to the bound at every one of the 7 stencil
    # points — under #1226 item 8's stage-by-stage oracle comparison this
    # under/over-tightens the box at ~40-45% of wet cells (median diff tiny
    # at nit000 since q_td ~ base after one step, but non-negligible: max
    # 0.039 degC on the DINO Y5 restart) and is the first stage at which the
    # legoESM limiter deviates from a faithful nonosc transcription.
    bnd_up = jnp.maximum(base, q_td)
    bnd_do = jnp.minimum(base, q_td)
    wet = None if active_mask is None else active_mask > 0.5
    if wet is not None:
        # #1226 item 8: faithful dry-cell mask (traadv_fct.F90:911-915
        # ``MERGE(..., -zbig/+zbig, tmask==1)``) BEFORE the neighbourhood
        # max/min — a dry cell's ``q_td`` is an unconstrained ``h_k→0``
        # division (legoESM never bothered to make it sane there since the
        # tracer update masks the cell out anyway) and must not widen a
        # WET neighbour's box.  ``zbig`` finite-sentineled to the dtype's
        # max (not ``inf``) so float32 callers stay finite under AD.
        zbig = jnp.asarray(0.5, dtype=bnd_up.dtype) * jnp.finfo(bnd_up.dtype).max
        bnd_up = jnp.where(wet, bnd_up, -zbig)
        bnd_do = jnp.where(wet, bnd_do, zbig)
    tr_west = jnp.roll(bnd_up, 1, axis=1)
    tr_east = jnp.roll(bnd_up, -1, axis=1)
    tr_south = jnp.concatenate([bnd_up[:1, :, :], bnd_up[:-1, :, :]], axis=0)
    tr_north = jnp.concatenate([bnd_up[1:, :, :], bnd_up[-1:, :, :]], axis=0)
    tr_above = jnp.concatenate([bnd_up[..., :1], bnd_up[..., :-1]], axis=-1)
    tr_below = jnp.concatenate([bnd_up[..., 1:], bnd_up[..., -1:]], axis=-1)
    q_max = jnp.maximum(
        jnp.maximum(jnp.maximum(bnd_up, tr_west), jnp.maximum(tr_east, tr_south)),
        jnp.maximum(jnp.maximum(tr_north, tr_above), tr_below),
    )
    if return_nemo_stencil_trace:
        # Materialise the observer after an optimization barrier.  Returning
        # the raw intermediates lets XLA fuse their consumers back into the
        # ordinary divergence graph and changes the bits being observed.
        stencil_trace = jax.lax.optimization_barrier((
            bnd_up, tr_west, tr_east, tr_south, tr_north, tr_above, tr_below,
            base, q_td, wet, q_max,
        ))
    tr_west_do = jnp.roll(bnd_do, 1, axis=1)
    tr_east_do = jnp.roll(bnd_do, -1, axis=1)
    tr_south_do = jnp.concatenate([bnd_do[:1, :, :], bnd_do[:-1, :, :]], axis=0)
    tr_north_do = jnp.concatenate([bnd_do[1:, :, :], bnd_do[-1:, :, :]], axis=0)
    tr_above_do = jnp.concatenate([bnd_do[..., :1], bnd_do[..., :-1]], axis=-1)
    tr_below_do = jnp.concatenate([bnd_do[..., 1:], bnd_do[..., -1:]], axis=-1)
    q_min = jnp.minimum(
        jnp.minimum(jnp.minimum(bnd_do, tr_west_do), jnp.minimum(tr_east_do, tr_south_do)),
        jnp.minimum(jnp.minimum(tr_north_do, tr_above_do), tr_below_do),
    )

    # h_new, not h_k: the budgets Q/P are increments of the AFTER field,
    # which the caller normalises by the AFTER thickness (see h_new above).
    alpha_u_full, alpha_v, alpha_vert_face = _zalesak_signsplit_face_alphas(
        ad_flux_u, ad_flux_v, ad_vert_int,
        q_td, q_min, q_max, h_new, dt, grid, eps,
    )
    if return_nemo_beta_trace:
        # Keep the observer out of the production limiter graph.  Without this
        # explicit barrier XLA can fuse the extra source-aligned readouts back
        # into the live alpha computation and move the bits being observed.
        diagnostic_inputs = jax.lax.optimization_barrier((
            ad_flux_u, ad_flux_v, ad_vert_int, q_td, q_min, q_max, h_new,
        ))
        _, _, _, beta_trace = _zalesak_signsplit_face_alphas(
            *diagnostic_inputs[:3],
            diagnostic_inputs[3], diagnostic_inputs[4], diagnostic_inputs[5],
            diagnostic_inputs[6], dt, grid, eps,
            return_nemo_beta_trace=True,
        )
    limited_u, limited_v = alpha_u_full * ad_flux_u, alpha_v * ad_flux_v
    div_h_fct = divergence_cgrid(flux_u_low + limited_u, flux_v_low + limited_v, grid)
    div_h_anti = divergence_cgrid(limited_u, limited_v, grid)
    limited_w = alpha_vert_face * ad_vert_int
    F_vert_fct = jnp.pad(F_vert_low_int + limited_w, (*pad_axes_v, (1, 1)))
    vert_div_fct = F_vert_fct[..., :-1] - F_vert_fct[..., 1:]
    anti_full = jnp.pad(limited_w, (*pad_axes_v, (1, 1)))
    if return_nemo_trace:
        mask = (
            jnp.ones_like(h_k) if active_mask is None else active_mask)
        safe_h = jnp.maximum(h_k, jnp.asarray(1.0e-10, h_k.dtype))
        anti_vert_div = anti_full[..., :-1] - anti_full[..., 1:]
        trace_final_div = -(div_h_anti + anti_vert_div)
        trace_rhs_after_up = trace_upstream_div / safe_h * mask
        trace_rhs_final = (
            trace_rhs_after_up + trace_final_div / safe_h * mask)
        trace = (
            trace_first_u, trace_first_v, trace_first_w, trace_first_div,
            q_mid, trace_average_u, trace_average_v, trace_average_w,
            trace_upstream_div, trace_rhs_after_up,
            trace_anti_pre_u, trace_anti_pre_v, trace_anti_pre_w,
            alpha_u_full, alpha_v,
            jnp.pad(alpha_vert_face, (*pad_axes_v, (1, 1)),
                    constant_values=1.0),
            limited_u * jnp.asarray(grid.dy_u)[..., None],
            limited_v * jnp.asarray(grid.dx_v)[..., None],
            anti_full * jnp.asarray(grid.area_T)[..., None],
            trace_final_div, h_k, trace_rhs_final,
        )
        return div_h_fct, vert_div_fct, trace
    if return_nemo_beta_trace:
        return div_h_fct, vert_div_fct, beta_trace + (
            alpha_u_full,
            alpha_v,
            jnp.pad(alpha_vert_face, (*pad_axes_v, (1, 1)),
                    constant_values=1.0),
        )
    if return_nemo_stencil_trace:
        return div_h_fct, vert_div_fct, stencil_trace
    if return_nemo_up1_trace:
        return div_h_fct, vert_div_fct, up1_trace
    if return_limiter_activity:
        # WRITE-only branch census for the developed-state fidelity walk.
        # A cell is active when a non-zero antidiffusive flux on any incident
        # face is multiplied by an alpha below one.  The ordinary return and
        # every default caller remain byte-for-byte unchanged.
        limited_u = (alpha_u_full < 1.0) & (ad_flux_u != 0.0)
        limited_v = (alpha_v < 1.0) & (ad_flux_v != 0.0)
        limited_w = (alpha_vert_face < 1.0) & (ad_vert_int != 0.0)
        cell_activity = (
            limited_u[:, :-1, :] | limited_u[:, 1:, :]
            | limited_v[:-1, :, :] | limited_v[1:, :, :]
            | jnp.pad(limited_w, ((0, 0), (0, 0), (0, 1)))
            | jnp.pad(limited_w, ((0, 0), (0, 0), (1, 0)))
        )
        if active_mask is not None:
            cell_activity = cell_activity & (active_mask > 0.5)
        if return_nemo_split:
            return (div_h_fct, vert_div_fct,
                    (div_h_low, vert_div_low, div_h_anti,
                     anti_full[..., :-1] - anti_full[..., 1:]),
                    cell_activity)
        return div_h_fct, vert_div_fct, cell_activity
    if return_nemo_split:
        return div_h_fct, vert_div_fct, (div_h_low, vert_div_low, div_h_anti,
            anti_full[..., :-1] - anti_full[..., 1:])
    return div_h_fct, vert_div_fct


# =============================================================================
# WENO-Z tracer advection (Phase 2a of Silvestri et al. 2024 WENO-ILES plan)
# =============================================================================
#
# High-order essentially non-oscillatory reconstruction at cell faces using
# the WENO-Z kernels from ``legoesm.core.weno``.  Unlike DST-3 and PPM, WENO
# uses nonlinear weights (not explicit limiters) to suppress oscillations near
# discontinuities, so no CFL, limiter, or monotonicity clamp is needed.
#
# WENO5: 5th-order, 6-point stencil (3 cells each side of face).
# WENO7: 7th-order, 8-point stencil (4 cells each side of face).

def _weno_to_u_points(
    f: jnp.ndarray,
    mass_flux_u: jnp.ndarray,
    order: int,
) -> jnp.ndarray:
    """WENO-Z interpolation to u-faces (zonal).

    Periodic in longitude. No CFL dependence — reconstruction is
    purely spatial.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, nlev)
        Tracer at cell centers.
    mass_flux_u : array, shape (n_lat, n_lon+1, nlev)
        Thickness-weighted velocity at u-faces (sign determines upwind).
    order : {5, 7}
        WENO order.

    Returns
    -------
    f_u : array, shape (n_lat, n_lon+1, nlev)
        WENO face values at u-points.
    """
    weno_fn = {5: weno5_z, 7: weno7_z}[order]
    hw = {5: 3, 7: 4}[order]
    n_lon = f.shape[1]

    # Convert point values to cell averages. WENO reconstruction is
    # a finite-volume method expecting cell-average inputs; passing
    # point values caps the order at O(dx^3). Conversion order must
    # match or exceed the WENO order for full accuracy.
    from legoesm.core.weno import point_to_cellavg_periodic
    conv_order = {5: 6, 7: 8}[order]
    f_avg = point_to_cellavg_periodic(f, axis=1, order=conv_order)

    # Build stencil for all faces simultaneously (periodic longitude).
    # Face j between cell j-1 and cell j: WENO face at I+1/2 where I=j-1.
    # Need cells j-hw to j+(hw-1), obtained via roll offsets hw..-(hw-1).
    stencil = [jnp.roll(f_avg, hw - j, axis=1) for j in range(2 * hw)]

    f_plus, f_minus = weno_fn(stencil)

    mf = mass_flux_u[:, :n_lon, :]
    f_face = weno_upwind(f_plus, f_minus, mf)

    # Periodic wrap: face n_lon = face 0
    return jnp.concatenate([f_face, f_face[:, 0:1, :]], axis=1)


def _weno_to_v_points(
    f: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    order: int,
) -> jnp.ndarray:
    """WENO-Z interpolation to v-faces (meridional).

    Solid wall at poles. Ghost cells use Neumann BC (copy boundary value),
    which degrades the reconstruction to lower order near boundaries.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, nlev)
        Tracer at cell centers.
    mass_flux_v : array, shape (n_lat+1, n_lon, nlev)
        Thickness-weighted velocity at v-faces.
    order : {5, 7}
        WENO order.

    Returns
    -------
    f_v : array, shape (n_lat+1, n_lon, nlev)
        WENO face values at v-points. Zero at pole boundaries.
    """
    weno_fn = {5: weno5_z, 7: weno7_z}[order]
    hw = {5: 3, 7: 4}[order]
    n_lat = f.shape[0]

    # Convert point values to cell averages (meridional, bounded).
    from legoesm.core.weno import point_to_cellavg_bounded
    conv_order = {5: 6, 7: 8}[order]
    f_avg = point_to_cellavg_bounded(f, axis=0, order=conv_order)

    # Ghost cells (Neumann BC: copy boundary value)
    f_ext = jnp.concatenate(
        [f_avg[:1, :, :]] * hw + [f_avg] + [f_avg[-1:, :, :]] * hw, axis=0
    )

    # Stencil for interior faces i=1..n_lat-1.
    # Cell k in original = f_ext[k + hw].
    # Face i: WENO at I+1/2 where I = i-1. Need cells i-hw..i+(hw-1).
    # In f_ext: indices i..i+(2*hw-1).
    stencil = [f_ext[1 + j: n_lat + j, :, :] for j in range(2 * hw)]

    f_plus, f_minus = weno_fn(stencil)

    mf_int = mass_flux_v[1:-1, :, :]
    f_face = weno_upwind(f_plus, f_minus, mf_int)

    # Solid wall at poles: zero flux
    zero = jnp.zeros((1, f.shape[1], f.shape[2]), dtype=f.dtype)
    return jnp.concatenate([zero, f_face, zero], axis=0)


def _flux_form_vertical_tracer_advection_weno(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    h_k: jnp.ndarray,
    dt: float,
    order: int,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with WENO-Z reconstruction.

    Same interface as ``flux_form_vertical_tracer_advection_dst3``.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Tracer at full levels.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels [m/s].
        Positive = upward. Zero at surface and bottom.
    h_k : array, shape (..., nlev)
        Layer thickness [m] (unused — WENO is purely spatial).
    dt : float
        Time step [s] (unused — WENO doesn't need CFL).
    order : {5, 7}
        WENO order.

    Returns
    -------
    vert_flux_div : array, shape (..., nlev)
        Vertical flux divergence F_top[k] - F_bot[k] for each level.
    """
    weno_fn = {5: weno5_z, 7: weno7_z}[order]
    hw = {5: 3, 7: 4}[order]
    nlev = field.shape[-1]

    # Ghost cells (Neumann BC: copy boundary value)
    f_ext = jnp.concatenate(
        [field[..., :1]] * hw + [field] + [field[..., -1:]] * hw, axis=-1
    )

    # Stencil for interior interfaces k=1..nlev-1.
    # field[k] = f_ext[..., k + hw].
    # Interface k between level k-1 (above) and level k (below):
    #   WENO at I+1/2 where I = k-1. Need cells k-hw..k+(hw-1).
    #   In f_ext: indices k..k+(2*hw-1).
    stencil = [f_ext[..., 1 + j: nlev + j] for j in range(2 * hw)]

    w_int = w_half[..., 1:nlev]
    f_plus, f_minus = weno_fn(stencil)

    # Stencil is ordered top-to-bottom (increasing level index).
    # f_plus = left-biased (from above), f_minus = right-biased (from below).
    # Upward flow (w > 0): donor is below → use f_minus.
    # Downward flow (w <= 0): donor is above → use f_plus.
    # Tie at w==0 -> f_plus (donor-above), matching the dst3/ppm/fct vertical
    # convention (all split on ``w_int > 0.0``).  At w==0 the flux is zero
    # regardless of the pick, so the choice only fixes a consistent convention.
    T_face = jnp.where(w_int > 0, f_minus, f_plus)

    F_interior = w_int * T_face

    # Zero-flux boundaries
    zeros = jnp.zeros((*field.shape[:-1], 1), dtype=field.dtype)
    F = jnp.concatenate([zeros, F_interior, zeros], axis=-1)

    # Flux divergence: F_top[k] - F_bot[k]
    return F[..., :-1] - F[..., 1:]


# --- Public API: WENO5 ---

def weno5_to_u_points(
    f: jnp.ndarray,
    mass_flux_u: jnp.ndarray,
) -> jnp.ndarray:
    """WENO5-Z interpolation to u-faces (zonal).

    5th-order essentially non-oscillatory reconstruction. Periodic in
    longitude. See ``_weno_to_u_points`` for details.
    """
    return _weno_to_u_points(f, mass_flux_u, order=5)


def weno5_to_v_points(
    f: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
) -> jnp.ndarray:
    """WENO5-Z interpolation to v-faces (meridional).

    5th-order with solid wall BCs at poles.
    See ``_weno_to_v_points`` for details.
    """
    return _weno_to_v_points(f, mass_flux_v, order=5)


def flux_form_vertical_tracer_advection_weno5(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    h_k: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with WENO5-Z.

    Same interface as ``flux_form_vertical_tracer_advection_dst3``.
    *h_k* and *dt* are unused (WENO is purely spatial) but kept for
    interface compatibility.
    """
    return _flux_form_vertical_tracer_advection_weno(field, w_half, h_k, dt, order=5)


# --- Public API: WENO7 ---

def weno7_to_u_points(
    f: jnp.ndarray,
    mass_flux_u: jnp.ndarray,
) -> jnp.ndarray:
    """WENO7-Z interpolation to u-faces (zonal).

    7th-order essentially non-oscillatory reconstruction. Periodic in
    longitude. See ``_weno_to_u_points`` for details.
    """
    return _weno_to_u_points(f, mass_flux_u, order=7)


def weno7_to_v_points(
    f: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
) -> jnp.ndarray:
    """WENO7-Z interpolation to v-faces (meridional).

    7th-order with solid wall BCs at poles.
    See ``_weno_to_v_points`` for details.
    """
    return _weno_to_v_points(f, mass_flux_v, order=7)


def flux_form_vertical_tracer_advection_weno7(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    h_k: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with WENO7-Z.

    Same interface as ``flux_form_vertical_tracer_advection_dst3``.
    *h_k* and *dt* are unused (WENO is purely spatial) but kept for
    interface compatibility.
    """
    return _flux_form_vertical_tracer_advection_weno(field, w_half, h_k, dt, order=7)


def _zalesak_signsplit_face_alphas(
    ad_flux_u: jnp.ndarray,
    ad_flux_v: jnp.ndarray,
    ad_vert_int: jnp.ndarray,
    q_td: jnp.ndarray,
    q_min: jnp.ndarray,
    q_max: jnp.ndarray,
    h_k: jnp.ndarray,
    dt: float,
    grid: "LatLonGrid",
    eps: float = 1e-30,
    *,
    return_nemo_beta_trace: bool = False,
) -> tuple:
    """Zalesak (1979) sign-split FCT face-flux limiter.

    Replaces the conservation-preserving ``alpha_face = min(alpha_left,
    alpha_right)`` heuristic with the proper monotone limiter.  For each
    cell ``c`` we compute the incoming and outgoing anti-diffusive
    increments separately (``P+_c``, ``P-_c``), pair them with the
    incoming/outgoing budgets ``Q+_c = q_max_c - q_td_c`` and
    ``Q-_c = q_td_c - q_min_c``, and form the per-cell ratios

        R+_c = min(1, Q+_c / max(P+_c, eps))
        R-_c = min(1, Q-_c / max(P-_c, eps)).

    Each anti-diffusive face flux is then limited by the smaller of the
    *receiving* cell's R+ and the *sending* cell's R-, with sender /
    receiver determined by the sign of the face flux.  This is the
    monotone version of Zalesak's algorithm and removes the symmetric
    ``min`` looseness of the old face-min heuristic that allowed grid-
    scale T noise to leak through under strong-frontal forcing
    (issue #212; reference Zalesak 1979 J. Comp. Phys. 31, 335; Kuzmin,
    *Flux-corrected transport*, 2012).

    Parameters
    ----------
    ad_flux_u : array (n_lat, n_lon+1, nlev) — anti-diffusive u-face flux
    ad_flux_v : array (n_lat+1, n_lon, nlev) — anti-diffusive v-face flux
    ad_vert_int : array (n_lat, n_lon, nlev-1) — anti-diffusive interior
        vertical interface flux (positive = upward).
    q_td : array (n_lat, n_lon, nlev) — provisional low-order update.
    q_min, q_max : array (n_lat, n_lon, nlev) — local stencil bounds.
    h_k : array (n_lat, n_lon, nlev) — the thickness the caller's update
        normalises by (the AFTER thickness ``h_new`` under z-star; equal to
        the old thickness only for non-divergent flow — see the h_new block
        in ``fct_tracer_advection``).
    dt : float — baroclinic time step.
    grid : LatLonGrid.
    eps : float — divide-by-zero guard for empty P+/P-.

    Returns
    -------
    alpha_u_full : (n_lat, n_lon+1, nlev) — alpha for each u-face, with
        ``face[n_lon] == face[0]`` for periodic-x.
    alpha_v : (n_lat+1, n_lon, nlev) — alpha for each v-face; the two
        wall faces carry placeholder 1.0 (their face flux is zero by
        the wall mask, so any alpha is harmless).
    alpha_vert_face : (n_lat, n_lon, nlev-1) — alpha for each interior
        vertical interface.
    """
    # Local positive / negative parts of the anti-diffusive face fluxes.
    F_u_pos = jnp.maximum(ad_flux_u, 0.0)
    F_u_neg = jnp.maximum(-ad_flux_u, 0.0)
    F_v_pos = jnp.maximum(ad_flux_v, 0.0)
    F_v_neg = jnp.maximum(-ad_flux_v, 0.0)
    F_w_pos = jnp.maximum(ad_vert_int, 0.0)
    F_w_neg = jnp.maximum(-ad_vert_int, 0.0)

    # Spherical face metrics (mirroring divergence_cgrid).
    if is_tripolar(grid):
        # Tripolar: use full 2D metrics — column-0 extraction is NOT
        # valid on the bipolar cap where dy_u/dx_v vary in longitude.
        face_dy = grid.dy_u                               # (n_lat, n_lon+1)
        face_dx = grid.dx_v                               # (n_lat+1, n_lon)
        _is_2d_dy = True
        _is_2d_dx = True
    else:
        R_planet = grid.radius
        dlon = grid.dlon
        # face_dy at h-points: cell-row meridional extent (1D, Mercator-safe).
        face_dy = (grid.dy * 0.5)[:, jnp.newaxis, jnp.newaxis]  # (n_lat,1,1)
        # #516: single-source v-face zonal cos(lat_v) (interior
        # cos(0.5·(lat[j]+lat[j+1])), poles 0) — routes through the shared
        # backend-aware helper so an MPI lat-band cut keeps the neighbour-rank
        # metric instead of the old serial jnp.pad (which zeroed local band
        # edges).  Bit-identical on serial.
        face_dx = R_planet * dlon * vface_zonal_cos_lat(grid)  # (n_lat+1,)
        _is_2d_dy = False
        _is_2d_dx = False
    area = grid.area[..., jnp.newaxis]            # (n_lat, n_lon, 1)

    # Per-cell magnitudes of incoming / outgoing horizontal flux.
    # u-face j is the WEST face of cell j and EAST face of cell j-1.
    # Per cell c (index j):
    #   incoming  = F_u_pos at WEST face (eastward in)  + F_u_neg at EAST face (westward in)
    #   outgoing  = F_u_neg at WEST face (westward out) + F_u_pos at EAST face (eastward out)
    if _is_2d_dy:
        # Per-face dy weighting: west face = face_dy[:, :-1], east = face_dy[:, 1:].
        dy_w = face_dy[:, :-1, jnp.newaxis]              # (n_lat, n_lon, 1)
        dy_e = face_dy[:, 1:, jnp.newaxis]               # (n_lat, n_lon, 1)
        in_u_w  = F_u_pos[:, :-1, :] * dy_w + F_u_neg[:, 1:, :] * dy_e
        out_u_w = F_u_neg[:, :-1, :] * dy_w + F_u_pos[:, 1:, :] * dy_e
    else:
        in_u  = F_u_pos[:, :-1, :] + F_u_neg[:, 1:, :]
        out_u = F_u_neg[:, :-1, :] + F_u_pos[:, 1:, :]

    # v-face j is the SOUTH face of cell j and NORTH face of cell j-1; weighted by face_dx[j].
    if _is_2d_dx:
        dx_s = face_dx[:-1, :, jnp.newaxis]              # (n_lat, n_lon, 1)
        dx_n = face_dx[1:, :, jnp.newaxis]               # (n_lat, n_lon, 1)
    else:
        dx_s = face_dx[:-1, jnp.newaxis, jnp.newaxis]
        dx_n = face_dx[1:, jnp.newaxis, jnp.newaxis]
    in_v_w = F_v_pos[:-1, :, :] * dx_s + F_v_neg[1:, :, :] * dx_n
    out_v_w = F_v_neg[:-1, :, :] * dx_s + F_v_pos[1:, :, :] * dx_n

    # Horizontal-incoming / outgoing tracer increment per cell (same units as ad·dt).
    if _is_2d_dy:
        P_in_h = (in_u_w + in_v_w) / area
        P_out_h = (out_u_w + out_v_w) / area
        if return_nemo_beta_trace:
            zpos_h = in_u_w + in_v_w
            zneg_h = out_u_w + out_v_w
    else:
        P_in_h = (in_u * face_dy + in_v_w) / area
        P_out_h = (out_u * face_dy + out_v_w) / area
        if return_nemo_beta_trace:
            zpos_h = in_u * face_dy + in_v_w
            zneg_h = out_u * face_dy + out_v_w

    # Vertical: pad with zeros at the top / bottom (rigid lid + floor) so
    # cell-c indexing is uniform.  ad_vert_int has shape (n_lat, n_lon,
    # nlev-1) for interfaces 0..nlev-2 between cell k (above) and k+1
    # (below); F > 0 = upward.  Pad (single HLO op) instead of
    # alloc-zeros + 3-array concatenate.
    pad_axes_v = ((0, 0),) * (F_w_pos.ndim - 1)
    F_w_pos_full = jnp.pad(F_w_pos, (*pad_axes_v, (1, 1)))
    F_w_neg_full = jnp.pad(F_w_neg, (*pad_axes_v, (1, 1)))
    # For cell k:
    #   TOP    interface index k:   F>0 = upward = leaving k upward, F<0 = entering k from above.
    #   BOTTOM interface index k+1: F>0 = upward = entering k from below, F<0 = leaving k downward.
    P_in_w  = F_w_neg_full[..., :-1] + F_w_pos_full[..., 1:]
    P_out_w = F_w_pos_full[..., :-1] + F_w_neg_full[..., 1:]

    # Gradient-underflow gates (grad_safe_ratio): the eps floors keep the
    # PRIMAL finite, but in float32 compute the division derivative's
    # 1/den**2 underflows (den ~ 1e-30..1e-20 -> den**2 = 0 -> inf*0 = NaN)
    # whenever a tracer is near-uniform (Q, inc -> 0) or h_k -> 0. The
    # all-NaN reverse gradients of ppm_fct rollouts pinned in
    # tests/ocean/unit/test_advection_grad_underflow.py came from here.
    t_grad = ratio_grad_floor(q_td.dtype)
    h_safe = jnp.maximum(h_k, eps)
    h_ok = h_k > t_grad
    inc_in = grad_safe_ratio((P_in_h + P_in_w) * dt, h_safe, h_ok)
    inc_out = grad_safe_ratio((P_out_h + P_out_w) * dt, h_safe, h_ok)

    # Per-cell budgets and ratios.
    Q_up = jnp.maximum(q_max - q_td, 0.0)
    Q_dn = jnp.maximum(q_td - q_min, 0.0)
    R_in = jnp.minimum(1.0, grad_safe_ratio(
        Q_up, jnp.maximum(inc_in, eps), inc_in > t_grad))
    R_out = jnp.minimum(1.0, grad_safe_ratio(
        Q_dn, jnp.maximum(inc_out, eps), inc_out > t_grad))

    # #1226 item 8: dry-cell (h_k ~ 0) ratios are NOT a real Zalesak
    # constraint — NEMO's own ``nonosc`` gives a dry point zbetup=zbetdo=
    # zbig there (traadv_fct.F90: zpos/zneg are exactly 0 once the
    # antidiffusive fluxes are wmask'ed, so the ``zpos/=0.`` guard falls
    # through to the "no local extremum" branch, zcoef=1, no clip).
    # legoESM's ad_vert_int/ad_flux_u/ad_flux_v are likewise ~0 at a dry
    # cell (the advecting mass flux is masked upstream), but Q_up/Q_dn and
    # inc_in/inc_out are each an O(1e-19)/O(h_k) ratio of that same
    # float-noise residual over an h_k that floors to eps=1e-30 -- the
    # *ratio* of two independent noise floors is unconstrained garbage
    # (observed up to ~1e17 on the DINO oracle), NOT a small number, so it
    # does not cancel in R_in/R_out and instead saturates one of them to 0.
    # A near-zero R at a dry cell then forces alpha=0 (full clip) on the
    # WET neighbour's face sharing that dry cell as sender/receiver --
    # i.e. every subsurface-topography w-face over-clips, inflating
    # legoESM's w-face clip count ~1.9x vs NEMO (662) purely from this
    # noise, with no signal in the antidiffusive flux itself (the clipped
    # face fluxes already matched NEMO at corr>0.9999 pre-fix). Force the
    # faithful zbig-equivalent (unclipped, R=1) at dry cells.
    R_in = jnp.where(h_ok, R_in, 1.0)
    R_out = jnp.where(h_ok, R_out, 1.0)

    if return_nemo_beta_trace:
        # Private source-aligned readout of compiled traadv_fct.f90:849-878.
        # ``zpos``/``zneg`` retain integrated-face units; ``zbt`` restores the
        # area*thickness/time factor before the NEMO-literal division order.
        # The live production ratios remain ``R_in``/``R_out`` below, so this
        # diagnostic cannot alter the limiter arithmetic it observes.
        zpos = zpos_h + P_in_w * area
        zneg = zneg_h + P_out_w * area
        zbt = area * h_k / dt
        zbig_beta = (
            jnp.asarray(0.5, dtype=q_td.dtype) * jnp.finfo(q_td.dtype).max)
        zbetup_literal = jnp.where(
            (q_max != -zbig_beta) & (zpos != 0.0),
            (q_max - q_td) / zpos * zbt,
            zbig_beta,
        )
        zbetdo_literal = jnp.where(
            (q_min != zbig_beta) & (zneg != 0.0),
            (q_td - q_min) / zneg * zbt,
            zbig_beta,
        )
        beta_trace = (
            q_max, q_min, zpos, zneg, zbt,
            zbetup_literal, zbetdo_literal, R_in, R_out,
        )

    # ---- Per-face alpha selection ----
    # u-face j: cell L = (j-1)%n_lon (west), cell R = j (east).
    # F > 0  → flow east, into R, out of L  → α = min(R+_R, R-_L).
    # F < 0  → flow west, into L, out of R  → α = min(R+_L, R-_R).
    n_lon = ad_flux_u.shape[1] - 1
    R_in_R_u = R_in                                 # (n_lat, n_lon, nlev)
    R_in_L_u = jnp.roll(R_in, 1, axis=1)
    R_out_R_u = R_out
    R_out_L_u = jnp.roll(R_out, 1, axis=1)
    ad_face_u_int = ad_flux_u[:, :n_lon, :]
    alpha_u_pos = jnp.minimum(R_in_R_u, R_out_L_u)
    alpha_u_neg = jnp.minimum(R_in_L_u, R_out_R_u)
    alpha_u_int = jnp.where(
        ad_face_u_int > 0.0, alpha_u_pos,
        jnp.where(ad_face_u_int < 0.0, alpha_u_neg, 1.0),
    )
    # Periodic wrap: face n_lon == face 0.
    alpha_u_full = jnp.concatenate(
        [alpha_u_int, alpha_u_int[:, :1, :]], axis=1,
    )

    # v-face j (interior, 1 ≤ j ≤ n_lat-1): cell S = j-1, cell N = j.
    R_in_N_v  = R_in[1:, :, :]
    R_in_S_v  = R_in[:-1, :, :]
    R_out_N_v = R_out[1:, :, :]
    R_out_S_v = R_out[:-1, :, :]
    ad_v_int = ad_flux_v[1:-1, :, :]  # interior faces
    alpha_v_pos = jnp.minimum(R_in_N_v, R_out_S_v)
    alpha_v_neg = jnp.minimum(R_in_S_v, R_out_N_v)
    alpha_v_int_face = jnp.where(
        ad_v_int > 0.0, alpha_v_pos,
        jnp.where(ad_v_int < 0.0, alpha_v_neg, 1.0),
    )
    # Wall faces (south & north): the wall mass flux is zero, so any
    # alpha is harmless; use 1 as a neutral placeholder.
    walls_shape = (1, ad_flux_v.shape[1], ad_flux_v.shape[2])
    walls = jnp.ones(walls_shape, dtype=ad_flux_v.dtype)
    alpha_v = jnp.concatenate([walls, alpha_v_int_face, walls], axis=0)

    # Vertical interface k between cell k (above) and cell k+1 (below).
    # F > 0 = upward → out of (k+1) below, into k above → α = min(R+_above, R-_below).
    # F < 0 = downward → out of k above, into k+1 below → α = min(R+_below, R-_above).
    R_in_above_w  = R_in[..., :-1]   # cell k above interface k
    R_in_below_w  = R_in[..., 1:]    # cell k+1 below interface k
    R_out_above_w = R_out[..., :-1]
    R_out_below_w = R_out[..., 1:]
    alpha_w_pos = jnp.minimum(R_in_above_w, R_out_below_w)
    alpha_w_neg = jnp.minimum(R_in_below_w, R_out_above_w)
    alpha_vert_face = jnp.where(
        ad_vert_int > 0.0, alpha_w_pos,
        jnp.where(ad_vert_int < 0.0, alpha_w_neg, 1.0),
    )
    if return_nemo_beta_trace:
        return alpha_u_full, alpha_v, alpha_vert_face, beta_trace
    return alpha_u_full, alpha_v, alpha_vert_face


# =============================================================================
# Veros W-grid superbee advection (for interface-resident energy fields)
# =============================================================================
#
# Faithful port of Veros's ``enable_tke_superbee_advection`` machinery for
# fields living on the W-grid (TKE/EKE-style energies at the ``M = nlev-1``
# interior interfaces):
#
#   - ``calculate_velocity_on_wgrid`` (veros/core/advection.py:117-217):
#     dz-weighted average of the cell-centre velocities onto the W-levels,
#     with the bottom T-cell's lower half absorbed into the deepest W-cell,
#     and the W-grid vertical velocity rebuilt FROM CONTINUITY of the W-grid
#     horizontal velocities (so a constant field is advected without spurious
#     interior sources).
#   - ``_adv_superbee`` (veros/core/advection.py:22-48): the MITgcm
#     CFL-dependent superbee flux
#         F = vel*(var_d + var_u)/2 − |vel|*((1−cr) + uCFL*cr)*rj/2,
#     cr = superbee-limited slope ratio, uCFL = |vel|*dt_tracer/dx
#     (Veros uses ``settings.dt_tracer`` in uCFL even for the W-grid fluxes —
#     advection.py:47).  The limiter is the canonical ``sweby_limiter``
#     (identical to Veros's ``limiter(cr)=max(clip(2cr,0,1), clip(cr,0,2))``).
#   - ``adv_flux_superbee_wgrid`` (veros/core/advection.py:221-244) + the
#     flux-divergence assembly of ``integrate_tke``
#     (veros/core/tke.py:286-311).
#
# W-grid layout mapping (legoESM vs Veros) — load-bearing, read carefully:
#
#   Veros carries TKE on ``nz`` W-levels ordered bottom→top: levels
#   ``k=0..nz-2`` are the interior interfaces (between T-cells k and k+1) and
#   level ``nz-1`` is the SURFACE half-cell (thickness ``0.5*dzw[-1]``).
#   legoESM carries interface energies on the ``M = nlev-1`` INTERIOR
#   interfaces only, ordered top→bottom (index 0 = shallowest interior
#   interface); the Veros surface half-cell level has NO legoESM counterpart
#   (the same truncation the prognostic-TKE implicit solve and the 3-D EKE
#   vertical diffusion already use: an independent M-level W-column with
#   zero-flux ends, the surface TKE flux BC applied at interface 0).
#
#   Index map: legoESM interface ``i``  ⟷  Veros W-level ``k = nz-2-i``.
#
#   Veros's vertical flux-divergence special cases (tke.py:304-310) map as:
#     * Veros bottom level k=0 (``-flux_top[0]/dzw[0]``): legoESM i = M-1 —
#       reproduced EXACTLY (no flux below the deepest W-cell).
#     * Veros interior (``-(flux_top[k]-flux_top[k-1])/dzw[k]``): exact.
#     * Veros surface level k=nz-1 (``/(0.5*dzw[-1])``): the surface half-cell
#       is NOT carried, so the flux between interface 0 and the (absent)
#       surface level is set to ZERO and legoESM interface 0 absorbs the
#       W-grid column-divergence residual — playing the role Veros's surface
#       half-cell plays (whose own top flux is also zero, adv_ft[...,-1]=0).
#       Column-integral conservation Σ dE·dzw·area = 0 holds EXACTLY either
#       way (both flux ends are zero ⇒ vertical telescoping; the horizontal
#       divergence is flux-form).
#
# Metrics use the REFERENCE 1-D dz (``z_coord.dz_ref``): the dzt/dzw ratios in
# the W-grid velocity are jacobian-independent under z* (uniform column
# stretching), and the faithful recipes run a rigid lid (J = 1) — matching
# Veros's fixed dzt/dzw exactly there.
#
# Masks: legoESM wet masks are 2-D columns (no kbot-style per-level masking —
# bathymetry enters via column stretching), so Veros's maskW products reduce
# to the 2-D u/v face masks and the topography redirect of
# ``calculate_velocity_on_wgrid`` (interior masked-W folding) has no
# counterpart.  Veros's halo-interior updates are covered by periodic-lon
# wrap + zero-flux walls.
#
# Horizontal flux divergence goes through the canonical ``divergence_cgrid``
# (legoESM's shared cell-area convention: exact spherical sin-band areas) vs
# Veros's ``cost·dxt·dyt`` rectangle areas — on identical fluxes the
# tendencies agree to machine precision, and the area conventions differ by a
# smooth ≲1.6% factor at 4°.  Per the oracle-recipe doctrine the shared
# operator wins; this is the documented residual vs a bit-exact Veros
# tendency.
#
# The limiter is the canonical ``sweby_limiter`` (imported at the top as
# ``_superbee_limiter``) — identical to Veros's ``limiter(cr)``.

# Veros ``_calc_cr`` division guard (veros/core/advection.py:12) — a pure
# numerical floor, exempt from the named-constant rule like the other eps
# floors in this module.
_SUPERBEE_CR_EPS = 1e-20


def _veros_superbee_face_flux(vel, var_m1, var_0, var_1, var_2,
                              fm_m1, fm_0, fm_1, u_cfl):
    """Veros ``_adv_superbee`` flux at one face (vectorised over faces).

    ``vel`` is the face velocity; positive transports from ``var_0`` (donor
    for ``vel > 0``) to ``var_1``.  ``var_m1``/``var_2`` extend the stencil
    one cell beyond the donor/downstream cell.  ``fm_*`` are the FACE wet
    masks at the previous / this / next face along the + direction (Veros's
    ``maskUtr``-style products), zeroing slope contributions through walls.
    ``u_cfl = |vel|*dt_tracer/dx`` is the CFL-dependent anti-diffusion weight.
    """
    rjp = (var_2 - var_1) * fm_1
    rj = (var_1 - var_0) * fm_0
    rjm = (var_0 - var_m1) * fm_m1
    cr = _superbee_limiter(grad_safe_ratio(
        jnp.where(vel > 0.0, rjm, rjp),
        jnp.where(jnp.abs(rj) < _SUPERBEE_CR_EPS, _SUPERBEE_CR_EPS, rj),
        jnp.abs(rj) > ratio_grad_floor(rj.dtype),
    ))
    return (vel * (var_1 + var_0) * 0.5
            - jnp.abs(vel) * ((1.0 - cr) + u_cfl * cr) * rj * 0.5)


def wgrid_velocities_latlon_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    dz_ref: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Horizontal advecting velocities on the W-grid (Veros
    ``calculate_velocity_on_wgrid``, veros/core/advection.py:123-195).

    ``u_w[..., i]`` is the dz-weighted average of ``u`` at cells ``i`` and
    ``i+1`` onto interface ``i``; the deepest interface (i = M-1) additionally
    absorbs the lower half of the bottom T-cell (Veros's bottom redirect,
    advection.py:169-179 at its k=0).  The Veros surface W-level
    (``u[-1]*0.5*dzt[-1]/dzw[-1]``) belongs to the surface half-cell legoESM
    does not carry.

    Parameters
    ----------
    u : (n_lat, n_lon+1, nlev), v : (n_lat+1, n_lon, nlev) — cell-centre-depth
        face velocities (the C-grid prognostic u/v).
    dz_ref : (nlev,) reference layer thicknesses.
    u_mask : (n_lat, n_lon+1), v_mask : (n_lat+1, n_lon) — face wet masks.

    Returns
    -------
    u_w : (n_lat, n_lon+1, M), v_w : (n_lat+1, n_lon, M) with M = nlev-1.
    """
    dz = jnp.asarray(dz_ref, dtype=u.dtype)
    dzw = 0.5 * (dz[:-1] + dz[1:])                       # (M,) Veros dzw (interior)
    w_up = 0.5 * dz[:-1] / dzw                           # weight on cell i
    w_dn = 0.5 * dz[1:] / dzw                            # weight on cell i+1
    u_w = u[..., :-1] * w_up + u[..., 1:] * w_dn
    v_w = v[..., :-1] * w_up + v[..., 1:] * w_dn
    # Bottom redirect: the deepest W-cell absorbs the bottom T-cell's lower
    # half (Veros u_wgrid[:, :, 0] += u[:, :, 0]*0.5*dzt[0]/dzw[0]).
    bottom_extra = 0.5 * dz[-1] / dzw[-1]
    u_w = u_w.at[..., -1].add(u[..., -1] * bottom_extra)
    v_w = v_w.at[..., -1].add(v[..., -1] * bottom_extra)
    u_w = u_w * u_mask[:, :, jnp.newaxis]
    v_w = v_w * v_mask[:, :, jnp.newaxis]
    # Enforce the periodic-lon wrap column (face n_lon ≡ face 0) so the
    # superbee wrap flux and the continuity divergence see the SAME face
    # velocity even if the caller's u carries a stale wrap column — the
    # column-integral conservation of the advective tendency depends on it.
    u_w = u_w.at[:, -1].set(u_w[:, 0])
    return u_w, v_w


def wgrid_vertical_velocity_latlon_cgrid(
    u_w: jnp.ndarray,
    v_w: jnp.ndarray,
    dz_ref: jnp.ndarray,
    grid: "LatLonGrid",
) -> jnp.ndarray:
    """W-grid vertical velocity FROM CONTINUITY (Veros advection.py:197-215).

    Integrates ``∂w/∂z = -div_h(u_w, v_w)`` upward from zero below the
    deepest W-cell, returning ``w`` at the ``M-1`` vertical flux positions
    (position ``j`` sits at T-cell centre ``j+1``, between interfaces ``j``
    and ``j+1``; positive = upward).  Reuses the conservative
    ``divergence_cgrid`` (the same metric divergence Veros forms explicitly).
    """
    dz = jnp.asarray(dz_ref, dtype=u_w.dtype)
    dzw = 0.5 * (dz[:-1] + dz[1:])                       # (M,)
    div = divergence_cgrid(u_w, v_w, grid)               # (n_lat, n_lon, M)
    col = div * dzw
    # rcs[..., i] = sum over W-cells i..M-1 (bottom-up partial sums).
    rcs = jnp.cumsum(col[..., ::-1], axis=-1)[..., ::-1]
    # w at the top of W-cell j+1 = -sum of divergence below it.
    return -rcs[..., 1:]                                 # (n_lat, n_lon, M-1)


def adv_flux_superbee_wgrid_latlon_cgrid(
    E: jnp.ndarray,
    u_w: jnp.ndarray,
    v_w: jnp.ndarray,
    w_w: jnp.ndarray,
    grid: "LatLonGrid",
    dz_ref: jnp.ndarray,
    dt_tracer: float,
    land_mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Superbee advective fluxes of a W-grid field (Veros
    ``adv_flux_superbee_wgrid``, veros/core/advection.py:221-244).

    Returns ``(fe, fn, ft)``:
      fe : (n_lat, n_lon+1, M) zonal flux at u-faces (periodic wrap),
      fn : (n_lat+1, n_lon, M) meridional flux at v-faces (zero at walls),
      ft : (n_lat, n_lon, M-1) vertical flux at the interior flux positions
           (positive = upward); the fluxes above interface 0 and below
           interface M-1 are zero by construction (see module note).

    ``dt_tracer`` enters ONLY the uCFL anti-diffusion weight (Veros uses
    ``settings.dt_tracer`` there, advection.py:47).
    """
    if is_tripolar(grid):
        # The meridional superbee stencil here uses simple edge replication at
        # the north boundary (no fold-partner exchange, cf. _tvd_to_v_points)
        # and the uCFL metrics assume the regular 1-D lat-lon spacings —
        # both silently wrong across a tripolar fold. Fail fast (static grid
        # metadata, raises at trace time) until a fold-aware variant exists.
        raise NotImplementedError(
            "W-grid superbee advection does not support tripolar grids "
            "(fold-aware meridional stencil + 2-D metrics not implemented).")
    dtype = E.dtype
    dz = jnp.asarray(dz_ref, dtype=dtype)
    dzw = 0.5 * (dz[:-1] + dz[1:])                       # (M,)
    n_lon = E.shape[1]
    lm3 = land_mask[:, :, jnp.newaxis]

    # ---- Zonal (periodic): face j between cells (j-1)%n_lon and j ----
    vel_x = u_w[:, :n_lon, :]
    var_0 = jnp.roll(E, 1, axis=1)                       # west cell (donor, vel>0)
    var_1 = E                                            # east cell
    var_m1 = jnp.roll(E, 2, axis=1)
    var_2 = jnp.roll(E, -1, axis=1)
    fm = u_mask[:, :n_lon]
    fm_0 = fm[:, :, jnp.newaxis]
    fm_m1 = jnp.roll(fm, 1, axis=1)[:, :, jnp.newaxis]
    fm_1 = jnp.roll(fm, -1, axis=1)[:, :, jnp.newaxis]
    # Veros uCFL_x = |vel|*dt/(cost*dxt) — dxt of the face's WEST cell, which
    # on the uniform-dlon lat-lon grid equals the east cell's (exact).
    dx_cell = (grid.radius * grid.dlon * grid.cos_lat)[:, jnp.newaxis, jnp.newaxis]
    u_cfl_x = jnp.abs(vel_x) * dt_tracer / dx_cell.astype(dtype)
    fe_core = _veros_superbee_face_flux(
        vel_x, var_m1, var_0, var_1, var_2, fm_m1, fm_0, fm_1, u_cfl_x)
    fe = jnp.concatenate([fe_core, fe_core[:, :1, :]], axis=1)

    # ---- Meridional: interior face i between cells i-1 (south) and i ----
    vel_y = v_w[1:-1]
    var_0 = E[:-1]                                       # south (donor, vel>0)
    var_1 = E[1:]                                        # north
    var_m1 = jnp.concatenate([E[:1], E[:-2]], axis=0)    # edge replicate south
    var_2 = jnp.concatenate([E[2:], E[-1:]], axis=0)     # edge replicate north
    fm_0 = v_mask[1:-1][:, :, jnp.newaxis]
    fm_m1 = v_mask[:-2][:, :, jnp.newaxis]
    fm_1 = v_mask[2:][:, :, jnp.newaxis]
    # Veros uCFL_y = |v·cosu(face)|*dt/(cost(south)*dyt(south)) — the face/
    # centre cosine ratio is kept for faithfulness (advection.py:44-47).
    # Face latitudes from grid.lat (same construction divergence_cgrid uses;
    # LatLonCGridGeometry now carries cos_lat_v; this path predates it).
    dy_cell = (0.5 * grid.dy)                            # (n_lat,) cell heights
    cos_face = jnp.cos(0.5 * (grid.lat[:-1] + grid.lat[1:]))
    cos_ratio = cos_face / grid.cos_lat[:-1]
    inv_dy = (cos_ratio / dy_cell[:-1])[:, jnp.newaxis, jnp.newaxis]
    u_cfl_y = jnp.abs(vel_y) * dt_tracer * inv_dy.astype(dtype)
    fn_core = _veros_superbee_face_flux(
        vel_y, var_m1, var_0, var_1, var_2, fm_m1, fm_0, fm_1, u_cfl_y)
    fn = jnp.pad(fn_core, ((1, 1), (0, 0), (0, 0)))      # zero flux at walls

    # ---- Vertical: + direction = UPWARD (decreasing legoESM index).
    # Flux position j between interfaces j (above) and j+1 (below).
    vel_z = w_w                                          # (..., M-1), >0 upward
    var_0 = E[..., 1:]                                   # below (donor, vel>0)
    var_1 = E[..., :-1]                                  # above
    var_m1 = jnp.concatenate([E[..., 2:], E[..., -1:]], axis=-1)  # below-below
    var_2 = jnp.concatenate([E[..., :1], E[..., :-2]], axis=-1)   # above-above
    # Veros uCFL_z dx = dzw of the W-cell BELOW the flux position
    # (advection.py:34, dx = dzw[:-1] in bottom-up indexing).
    u_cfl_z = jnp.abs(vel_z) * dt_tracer / dzw[1:]
    ft = _veros_superbee_face_flux(
        vel_z, var_m1, var_0, var_1, var_2, lm3, lm3, lm3, u_cfl_z)
    return fe, fn, ft


def wgrid_advection_tendency_latlon_cgrid(
    E: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: "LatLonGrid",
    dz_ref: jnp.ndarray,
    dt_tracer: float,
    land_mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Superbee W-grid advective tendency ``dE/dt`` [field-units/s] of an
    interface-resident energy field (Veros ``integrate_tke``'s ``dtke``
    assembly, veros/core/tke.py:286-311).

    Orchestrates: W-grid velocities (incl. continuity ``w``) → superbee
    fluxes → conservative flux divergence.  The volume integral
    ``Σ dE·dzw·area`` over the closed/periodic domain vanishes to machine
    precision (flux-form horizontal divergence + zero-ended vertical
    telescoping).

    Parameters
    ----------
    E : (n_lat, n_lon, M) field at the interior interfaces (M = nlev-1).
    u, v : pre-step C-grid velocities (Veros ``u[..., tau]``).
    dz_ref : (nlev,) reference layer thicknesses.
    dt_tracer : tracer timestep [s] — the uCFL weight (Veros dt_tracer).
    land_mask, u_mask, v_mask : 2-D wet masks.
    """
    dtype = E.dtype
    dz = jnp.asarray(dz_ref, dtype=dtype)
    dzw = 0.5 * (dz[:-1] + dz[1:])                       # (M,)
    u_w, v_w = wgrid_velocities_latlon_cgrid(u, v, dz, u_mask, v_mask)
    w_w = wgrid_vertical_velocity_latlon_cgrid(u_w, v_w, dz, grid)
    fe, fn, ft = adv_flux_superbee_wgrid_latlon_cgrid(
        E, u_w, v_w, w_w, grid, dz, dt_tracer, land_mask, u_mask, v_mask)
    # Horizontal: -div(F) per level (Veros tke.py:293-303, maskW applied).
    adv_h = -divergence_cgrid(fe, fn, grid)
    # Vertical: -(F_above - F_below)/dzw with zero fluxes at both column ends
    # (Veros tke.py:304-310; see the module note on the surface mapping).
    pad_axes = ((0, 0),) * (ft.ndim - 1)
    F = jnp.pad(ft, (*pad_axes, (1, 1)))                 # (..., M+1)
    adv_v = -(F[..., :-1] - F[..., 1:]) / dzw
    return (adv_h + adv_v) * land_mask[:, :, jnp.newaxis]
