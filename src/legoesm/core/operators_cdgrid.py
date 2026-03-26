"""C-D grid operators on the cubed-sphere — FV3-faithful rewrite.

Provides core operators for the FV3-style C-D grid discretisation,
unified for both 2D (shallow water) and 3D (atmosphere PE / ocean PE):

* PPM (Piecewise Parabolic Method) transport — 4th-order in smooth regions
* D-grid vorticity via circulation (exact, no interpolation artefacts)
* D-to-C grid interpolation with non-orthogonality correction (d2a2c)
* C-grid divergence and mass flux
* Vector-invariant momentum tendencies with Arakawa-Lamb gradient
* Divergence damping (2nd- and 4th-order, with adaptive Smagorinsky option)
* Laplacian and biharmonic diffusion

All operators handle both 2D ``(6, n+1, n+1)`` and 3D ``(6, n+1, n+1, nlev)``
inputs automatically.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo


# ==============================================================================
# Internal: halo padding that works for both 2D and 3D
# ==============================================================================

def _pad_halo_auto(field, cdgrid):
    """Pad halo=1 for a 2D or 3D cell-centre field.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n) or (6, n, n, nlev)

    Returns
    -------
    jax.Array, shape (6, n+2, n+2) or (6, n+2, n+2, nlev)
    """
    if field.ndim == 3:
        return pad_halo(field, interp_offsets=cdgrid.base.halo_interp_offsets)
    # 3D: vmap over levels
    f_t = jnp.moveaxis(field, -1, 0)

    def pad_one(fk):
        return pad_halo(fk, interp_offsets=cdgrid.base.halo_interp_offsets)

    f_pad_t = jax.vmap(pad_one)(f_t)
    return jnp.moveaxis(f_pad_t, 0, -1)


def _pad_halo_auto_h2(field, cdgrid):
    """Pad halo=2 for a 2D or 3D cell-centre field.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n) or (6, n, n, nlev)

    Returns
    -------
    jax.Array, shape (6, n+4, n+4) or (6, n+4, n+4, nlev)
    """
    if field.ndim == 3:
        return pad_halo(field, halo=2,
                        interp_offsets=cdgrid.base.halo_interp_offsets_h2)
    f_t = jnp.moveaxis(field, -1, 0)

    def pad_one(fk):
        return pad_halo(fk, halo=2,
                        interp_offsets=cdgrid.base.halo_interp_offsets_h2)

    f_pad_t = jax.vmap(pad_one)(f_t)
    return jnp.moveaxis(f_pad_t, 0, -1)


def _broadcast_metric(metric, field):
    """Broadcast a 2D metric (6, ...) to match field's trailing nlev dim."""
    if field.ndim > metric.ndim:
        return metric[..., None]
    return metric


# ==============================================================================
# PPM (Piecewise Parabolic Method) transport
# ==============================================================================

def _ppm_reconstruct_1d(q):
    """PPM face-value reconstruction along the LAST axis.

    Given cell averages q[..., i], compute left and right face values
    (q_L[i], q_R[i]) for each cell using 4th-order interpolation with
    monotonicity constraints (Colella & Woodward 1984).

    Parameters
    ----------
    q : jax.Array, shape (..., N)
        Cell averages.  Requires N >= 4 for the 4th-order stencil.

    Returns
    -------
    q_L : jax.Array, shape (..., N)
        Left face value for each cell.
    q_R : jax.Array, shape (..., N)
        Right face value for each cell.
    """
    N = q.shape[-1]

    # Pad with 2 ghost cells on each side (edge extrapolation)
    q_pad = jnp.pad(q, [(0, 0)] * (q.ndim - 1) + [(2, 2)], mode='edge')

    # 4th-order face values at i+1/2 (between cell i and i+1 in padded coords)
    # q_face[i] = face value between original cell i-1 and i (0-indexed)
    # We need face values at i-1/2 and i+1/2 for each cell i.
    # q_face = (7*(q[i] + q[i+1]) - (q[i-1] + q[i+2])) / 12
    # In padded coords, original cell i maps to padded index i+2.
    # Face between padded i and i+1 for all valid faces:
    q_face = (7.0 * (q_pad[..., 1:-2] + q_pad[..., 2:-1])
              - (q_pad[..., :-3] + q_pad[..., 3:])) / 12.0
    # q_face has shape (..., N+1): face values at positions -1/2, 1/2, ..., N-1/2

    # Left and right face values for each cell
    q_L = q_face[..., :-1]  # face at i-1/2 → left face of cell i
    q_R = q_face[..., 1:]   # face at i+1/2 → right face of cell i

    # --- Monotonicity constraints (Colella & Woodward 1984) ---
    # 1. Detect local extrema: if (q_R - q)(q - q_L) <= 0, flatten
    delta = (q_R - q) * (q - q_L)
    is_extremum = delta <= 0.0
    q_L = jnp.where(is_extremum, q, q_L)
    q_R = jnp.where(is_extremum, q, q_R)

    # 2. Overshoot limiting: ensure the parabola doesn't create new extrema
    # q_6 = 6*(q - 0.5*(q_L + q_R))
    q_6 = 6.0 * (q - 0.5 * (q_L + q_R))
    # If q_6 * (q_R - q_L) > (q_R - q_L)^2, limit q_L
    dq = q_R - q_L
    cond_L = q_6 > dq * dq
    # If adjustment needed: q_L = 3*q - 2*q_R
    q_L_adj = 3.0 * q - 2.0 * q_R
    q_L = jnp.where(cond_L & ~is_extremum, q_L_adj, q_L)
    # If -q_6 > dq*dq, limit q_R
    cond_R = -q_6 > dq * dq
    q_R_adj = 3.0 * q - 2.0 * q_L
    q_R = jnp.where(cond_R & ~is_extremum, q_R_adj, q_R)

    return q_L, q_R


def _ppm_flux_1d(q, courant):
    """Compute PPM flux along the last axis.

    Parameters
    ----------
    q : jax.Array, shape (..., N)
        Cell averages of the transported quantity.
    courant : jax.Array, shape (..., N+1)
        Courant number at each face: c = u * dt / dx.
        Positive means flow in the +i direction.

    Returns
    -------
    flux : jax.Array, shape (..., N+1)
        Mass-weighted flux at each face.
    """
    q_L, q_R = _ppm_reconstruct_1d(q)

    # Pad q_L, q_R to get values at ghost faces
    # For face i+1/2, if c > 0 we use the upwind cell i (q_R[i], q_L[i])
    # If c < 0 we use cell i+1 (q_L[i+1], q_R[i+1])
    q_L_pad = jnp.pad(q_L, [(0, 0)] * (q.ndim - 1) + [(1, 1)], mode='edge')
    q_R_pad = jnp.pad(q_R, [(0, 0)] * (q.ndim - 1) + [(1, 1)], mode='edge')
    q_pad = jnp.pad(q, [(0, 0)] * (q.ndim - 1) + [(1, 1)], mode='edge')

    # Face i+1/2 (0-indexed face j, j=0..N)
    # If c > 0: upwind cell is j-1 (padded index j), use q_R of that cell
    # If c < 0: upwind cell is j (padded index j+1), use q_L of that cell
    c = courant  # (..., N+1)
    c_abs = jnp.abs(c)

    # Upwind cell's parabola parameters
    # c > 0: from cell on the left (padded j), integrate from right face inward
    q_R_up = q_R_pad[..., :-1]  # q_R of left cell at each face
    q_L_up = q_L_pad[..., :-1]  # q_L of left cell
    q_avg_up = q_pad[..., :-1]  # q of left cell

    # c < 0: from cell on the right (padded j+1), integrate from left face inward
    q_R_dn = q_R_pad[..., 1:]
    q_L_dn = q_L_pad[..., 1:]
    q_avg_dn = q_pad[..., 1:]

    # PPM flux integration:
    # For c > 0, integrating from right face (q_R) of upwind cell:
    #   F = q_R - c/2 * (q_R - q_L - (1 - 2c/3) * q_6)
    #   where q_6 = 6*(q_avg - 0.5*(q_L + q_R))
    c_safe = jnp.clip(c_abs, 0.0, 1.0)

    # Positive direction (c > 0)
    q6_pos = 6.0 * (q_avg_up - 0.5 * (q_L_up + q_R_up))
    flux_pos = q_R_up - 0.5 * c_safe * (
        q_R_up - q_L_up - (1.0 - 2.0 / 3.0 * c_safe) * q6_pos
    )

    # Negative direction (c < 0)
    q6_neg = 6.0 * (q_avg_dn - 0.5 * (q_L_dn + q_R_dn))
    flux_neg = q_L_dn + 0.5 * c_safe * (
        q_R_dn - q_L_dn + (1.0 - 2.0 / 3.0 * c_safe) * q6_neg
    )

    flux = jnp.where(c >= 0, flux_pos, flux_neg)
    return flux


# ==============================================================================
# Cell-centre <-> D-grid corner vector conversion
# ==============================================================================

def center_to_dgrid_vector(u_cc, v_cc, cdgrid):
    """Interpolate cell-centre wind vectors to D-grid corner positions.

    Uses ``pad_halo_vector`` for proper rotation of vector components
    across cubed-sphere face boundaries, then 4-point averages to corners.

    Works for both 2D (6, n, n) and 3D (6, n, n, nlev) inputs.

    Parameters
    ----------
    u_cc, v_cc : jax.Array, shape (6, n, n[, nlev])
        Cell-centre velocity components.

    Returns
    -------
    u_d, v_d : jax.Array, shape (6, n+1, n+1[, nlev])
    """
    from legoesm.grids.halo import pad_halo_vector

    grid = cdgrid.base
    if u_cc.ndim == 3:
        u_pad, v_pad = pad_halo_vector(
            u_cc, v_cc,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
            interp_offsets=grid.halo_interp_offsets,
        )
        u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1]
                       + u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
        v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1]
                       + v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
        return u_d, v_d

    # 3D: vmap over levels
    u_t = jnp.moveaxis(u_cc, -1, 0)
    v_t = jnp.moveaxis(v_cc, -1, 0)

    def convert_one(args):
        uk, vk = args
        return center_to_dgrid_vector(uk, vk, cdgrid)

    u_d_t, v_d_t = jax.vmap(convert_one)((u_t, v_t))
    return jnp.moveaxis(u_d_t, 0, -1), jnp.moveaxis(v_d_t, 0, -1)


def dgrid_to_center_vector(u_d, v_d):
    """D-grid corner velocities -> cell-centre velocities (4-point average).

    No cross-face rotation needed since corners within a face share
    the same local coordinate system.

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev).
    """
    if u_d.ndim == 3:
        u_cc = 0.25 * (u_d[:, :-1, :-1] + u_d[:, 1:, :-1]
                        + u_d[:, :-1, 1:] + u_d[:, 1:, 1:])
        v_cc = 0.25 * (v_d[:, :-1, :-1] + v_d[:, 1:, :-1]
                        + v_d[:, :-1, 1:] + v_d[:, 1:, 1:])
    else:
        u_cc = 0.25 * (u_d[:, :-1, :-1, :] + u_d[:, 1:, :-1, :]
                        + u_d[:, :-1, 1:, :] + u_d[:, 1:, 1:, :])
        v_cc = 0.25 * (v_d[:, :-1, :-1, :] + v_d[:, 1:, :-1, :]
                        + v_d[:, :-1, 1:, :] + v_d[:, 1:, 1:, :])
    return u_cc, v_cc


# ==============================================================================
# D-grid -> C-grid interpolation (d2a2c)
# ==============================================================================

def dgrid_to_cgrid(u_d, v_d, cdgrid):
    """D-grid corner winds -> C-grid edge-normal velocities.

    Accounts for non-orthogonality of the cubed-sphere grid:
    - x-face normal velocity: u_c = u_d*sin(alpha) - v_d*cos(alpha)
    - y-face normal velocity: v_c = v_d  (e_perp IS the y-face normal)

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev).
    """
    # x-face (at constant i): average along j, then project onto face normal
    u_avg = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])    # (6, n+1, n[, nlev])
    v_avg_x = 0.5 * (v_d[:, :, :-1] + v_d[:, :, 1:])  # (6, n+1, n[, nlev])
    cosa_u = _broadcast_metric(cdgrid.cosa_u, u_avg)
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, 1e-12))
    u_c = u_avg * sina_u - v_avg_x * cosa_u

    # y-face (at constant j): e_perp is the outward normal, so v_c = v_d
    v_c = 0.5 * (v_d[:, :-1] + v_d[:, 1:])             # (6, n, n+1[, nlev])
    return u_c, v_c


def cgrid_to_dgrid(u_c, v_c, cdgrid):
    """Interpolate C-grid edge velocities to D-grid corners.

    Inverse of dgrid_to_cgrid accounting for non-orthogonality:
    - v_d from v_c (y-face): v_d = v_c (since v_c = v_d)
    - u_d from u_c (x-face): u_c = u_d*sin(alpha) - v_d*cos(alpha)
      => u_d = (u_c + v_d*cos(alpha)) / sin(alpha)

    Works for both 2D and 3D inputs.

    Parameters
    ----------
    u_c : jax.Array, shape (6, n+1, n[, nlev])
    v_c : jax.Array, shape (6, n, n+1[, nlev])

    Returns
    -------
    u_d, v_d : jax.Array, shape (6, n+1, n+1[, nlev])
    """
    # v_d from v_c (y-face normal = e_perp, so v_c = v_d)
    pad_v = [(0, 0), (1, 1), (0, 0)] + [(0, 0)] * (v_c.ndim - 3)
    v_c_pad = jnp.pad(v_c, pad_v, mode='edge')
    v_d = 0.5 * (v_c_pad[:, :-1] + v_c_pad[:, 1:])

    # u_d from u_c (x-face) with inverse non-orthogonality correction
    pad_u = [(0, 0), (0, 0), (1, 1)] + [(0, 0)] * (u_c.ndim - 3)
    u_c_pad = jnp.pad(u_c, pad_u, mode='edge')
    u_c_avg = 0.5 * (u_c_pad[:, :, :-1] + u_c_pad[:, :, 1:])

    cosa = _broadcast_metric(cdgrid.cosa_corner, u_c_avg)
    sina = jnp.sqrt(jnp.maximum(1.0 - cosa**2, 1e-12))

    u_d = (u_c_avg + v_d * cosa) / jnp.maximum(sina, 1e-12)

    return u_d, v_d


# ==============================================================================
# D-grid vorticity (circulation form)
# ==============================================================================

def dgrid_vorticity(u_d, v_d, cdgrid):
    """Relative vorticity at cell centres from D-grid corner winds.

    Uses the integral circulation form: zeta = (1/A) oint v . dl, which is
    exact for the D-grid and avoids the Hollingsworth-Kallberg instability.

    D-grid winds are in the (e_i, e_perp) orthogonal basis where e_perp is
    perpendicular to e_i (90 deg CCW).  For the circulation integral:
    - i-edges (south/north, tangent e_i): v . e_i = u_d
    - j-edges (east/west, tangent e_j): v . e_j = u_d*cos(alpha) + v_d*sin(alpha)
      where alpha is the angle between e_i and e_j (non-orthogonality).

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev) inputs.

    Parameters
    ----------
    u_d, v_d : jax.Array, shape (6, n+1, n+1[, nlev])

    Returns
    -------
    zeta : jax.Array, shape (6, n, n[, nlev])
    """
    cosa = _broadcast_metric(cdgrid.cosa_corner, u_d)
    sina = jnp.sqrt(jnp.maximum(1.0 - cosa**2, 1e-12))

    # South edge (i-direction): v . e_i = u_d
    u_south = 0.5 * (u_d[:, :-1, :-1] + u_d[:, 1:, :-1])
    # North edge (i-direction): v . e_i = u_d
    u_north = 0.5 * (u_d[:, :-1, 1:] + u_d[:, 1:, 1:])

    # East edge (j-direction): v . e_j = u_d*cos(alpha) + v_d*sin(alpha)
    u_east_raw = 0.5 * (u_d[:, 1:, :-1] + u_d[:, 1:, 1:])
    v_east_raw = 0.5 * (v_d[:, 1:, :-1] + v_d[:, 1:, 1:])
    cosa_east = 0.5 * (cosa[:, 1:, :-1] + cosa[:, 1:, 1:])
    sina_east = 0.5 * (sina[:, 1:, :-1] + sina[:, 1:, 1:])
    v_cov_east = u_east_raw * cosa_east + v_east_raw * sina_east

    # West edge (j-direction): v . e_j = u_d*cos(alpha) + v_d*sin(alpha)
    u_west_raw = 0.5 * (u_d[:, :-1, :-1] + u_d[:, :-1, 1:])
    v_west_raw = 0.5 * (v_d[:, :-1, :-1] + v_d[:, :-1, 1:])
    cosa_west = 0.5 * (cosa[:, :-1, :-1] + cosa[:, :-1, 1:])
    sina_west = 0.5 * (sina[:, :-1, :-1] + sina[:, :-1, 1:])
    v_cov_west = u_west_raw * cosa_west + v_west_raw * sina_west

    dx_south = _broadcast_metric(cdgrid.dx_edge_y[:, :, :-1], u_d)
    dx_north = _broadcast_metric(cdgrid.dx_edge_y[:, :, 1:], u_d)
    dy_west = _broadcast_metric(cdgrid.dy_edge_x[:, :-1, :], u_d)
    dy_east = _broadcast_metric(cdgrid.dy_edge_x[:, 1:, :], u_d)

    circ = (u_south * dx_south + v_cov_east * dy_east
            - u_north * dx_north - v_cov_west * dy_west)

    area = _broadcast_metric(cdgrid.base.area, u_d)
    return circ / area


# ==============================================================================
# C-grid divergence
# ==============================================================================

def cgrid_divergence(u_c, v_c, cdgrid):
    """Exact flux-form divergence at cell centres.

    Works for both 2D and 3D inputs.

    Parameters
    ----------
    u_c : jax.Array, shape (6, n+1, n[, nlev])
    v_c : jax.Array, shape (6, n, n+1[, nlev])

    Returns
    -------
    div : jax.Array, shape (6, n, n[, nlev])
    """
    dy = _broadcast_metric(cdgrid.dy_edge_x, u_c)
    dx = _broadcast_metric(cdgrid.dx_edge_y, v_c)
    flux_x = u_c * dy
    flux_y = v_c * dx
    net_x = flux_x[:, 1:] - flux_x[:, :-1]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]
    area = _broadcast_metric(cdgrid.base.area, net_x)
    return (net_x + net_y) / area


# ==============================================================================
# C-grid mass flux with PPM transport
# ==============================================================================

def cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid):
    """Conservative mass flux divergence using PPM face reconstruction.

    Uses the Piecewise Parabolic Method (Colella & Woodward 1984) for
    4th-order accurate face values in smooth regions, with monotonicity
    constraints to prevent oscillations near discontinuities.

    Requires halo=2 data for the PPM stencil.

    Works for both 2D and 3D inputs.

    Parameters
    ----------
    h : jax.Array, shape (6, n, n[, nlev])
    u_c : jax.Array, shape (6, n+1, n[, nlev])
    v_c : jax.Array, shape (6, n, n+1[, nlev])

    Returns
    -------
    dh_dt : jax.Array, shape (6, n, n[, nlev])
    """
    if h.ndim == 4:
        # 3D: apply per level via vmap
        h_t = jnp.moveaxis(h, -1, 0)
        u_c_t = jnp.moveaxis(u_c, -1, 0)
        v_c_t = jnp.moveaxis(v_c, -1, 0)

        def flux_div_one(args):
            hk, uk, vk = args
            return cgrid_mass_flux_divergence(hk, uk, vk, cdgrid)

        result_t = jax.vmap(flux_div_one)((h_t, u_c_t, v_c_t))
        return jnp.moveaxis(result_t, 0, -1)

    # 2D case: PPM face reconstruction with halo=2
    h_pad = _pad_halo_auto_h2(h, cdgrid)  # (6, n+4, n+4)

    dy = cdgrid.dy_edge_x   # (6, n+1, n)
    dx = cdgrid.dx_edge_y   # (6, n, n+1)
    n = cdgrid.n

    # --- X-direction PPM ---
    # For each j, reconstruct h along i-direction and compute flux at
    # each x-interface.  h_pad[:, :, j+2] for j in [0, n-1] gives the
    # i-strip at interior j, with 2 halo cells on each side.
    # x-interface (i, j) for i in [0, n] needs cells i-2..i+1 in the
    # original grid, which maps to padded indices i..i+3.

    # Extract strips along i for each j: shape (6, n+4, n) from padded
    h_x_strips = h_pad[:, :, 2:-2]  # (6, n+4, n)

    # PPM reconstruction along axis=1 (i-direction)
    q_L_x, q_R_x = _ppm_reconstruct_1d(h_x_strips)  # each (6, n+4, n)

    # Face values at x-interfaces: we need n+1 faces for interior cells
    # Face (i) is between padded cells (i+1) and (i+2), i.e. original cells i-1 and i
    # For the upwind flux, use q_R of the left cell or q_L of the right cell

    # q_L, q_R are defined for each cell in padded array (6, n+4, n)
    # Face index f in [0, n] corresponds to:
    #   left cell: padded index f+1, right cell: padded index f+2
    q_R_left = q_R_x[:, 1:n+2, :]    # (6, n+1, n) - q_R of left cell
    q_L_right = q_L_x[:, 2:n+3, :]   # (6, n+1, n) - q_L of right cell

    h_face_x = jnp.where(u_c > 0, q_R_left, q_L_right)

    # --- Y-direction PPM ---
    h_y_strips = h_pad[:, 2:-2, :]  # (6, n, n+4)

    q_L_y, q_R_y = _ppm_reconstruct_1d(h_y_strips)

    q_R_bottom = q_R_y[:, :, 1:n+2]   # (6, n, n+1)
    q_L_top = q_L_y[:, :, 2:n+3]      # (6, n, n+1)

    h_face_y = jnp.where(v_c > 0, q_R_bottom, q_L_top)

    # --- Flux divergence ---
    flux_x = h_face_x * u_c * dy
    flux_y = h_face_y * v_c * dx

    net_x = flux_x[:, 1:] - flux_x[:, :-1]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]

    return -(net_x + net_y) / cdgrid.base.area


# ==============================================================================
# Scalar advection on C-grid (PPM)
# ==============================================================================

def cgrid_scalar_advection(q, u_c, v_c, cdgrid):
    """Advective transport of scalar q by C-grid velocities (PPM).

    Works for both 2D and 3D inputs.
    """
    return cgrid_mass_flux_divergence(q, u_c, v_c, cdgrid)


# ==============================================================================
# Arakawa-Lamb gradient at D-grid corners
# ==============================================================================

def _arakawa_lamb_gradient(B, cdgrid):
    """4-point Arakawa-Lamb gradient of cell-centre field at D-grid corners.

    Works for both 2D (6, n, n) and 3D (6, n, n, nlev) inputs.

    Parameters
    ----------
    B : jax.Array, shape (6, n, n[, nlev])

    Returns
    -------
    dB_dx, dB_dy : jax.Array, shape (6, n+1, n+1[, nlev])
    """
    B_pad = _pad_halo_auto(B, cdgrid)

    if B.ndim == 3:
        B_sw = B_pad[:, :-1, :-1]
        B_se = B_pad[:, 1:, :-1]
        B_nw = B_pad[:, :-1, 1:]
        B_ne = B_pad[:, 1:, 1:]
    else:
        B_sw = B_pad[:, :-1, :-1, :]
        B_se = B_pad[:, 1:, :-1, :]
        B_nw = B_pad[:, :-1, 1:, :]
        B_ne = B_pad[:, 1:, 1:, :]

    # Use proper halo exchange for metrics (not mode='edge' which fails
    # at equatorial-polar face boundaries where axes swap).
    dx_pad = _pad_halo_auto(cdgrid.base.dx, cdgrid)
    dy_pad = _pad_halo_auto(cdgrid.base.dy, cdgrid)
    dx_dual = 0.25 * (dx_pad[:, :-1, :-1] + dx_pad[:, 1:, :-1]
                       + dx_pad[:, :-1, 1:] + dx_pad[:, 1:, 1:])
    dy_dual = 0.25 * (dy_pad[:, :-1, :-1] + dy_pad[:, 1:, :-1]
                       + dy_pad[:, :-1, 1:] + dy_pad[:, 1:, 1:])

    dx_dual_b = _broadcast_metric(dx_dual, B_sw)
    dy_dual_b = _broadcast_metric(dy_dual, B_sw)

    dB_dx = ((B_se + B_ne) - (B_sw + B_nw)) / jnp.maximum(dx_dual_b, 1e-10)
    dB_dy = ((B_nw + B_ne) - (B_sw + B_se)) / jnp.maximum(dy_dual_b, 1e-10)

    return dB_dx, dB_dy


# ==============================================================================
# Interpolation helpers
# ==============================================================================

def _interp_center_to_corner(field, cdgrid):
    """Interpolate cell-centre field to D-grid corners (4-point average).

    Works for both 2D (6, n, n) and 3D (6, n, n, nlev) inputs.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n[, nlev])

    Returns
    -------
    jax.Array, shape (6, n+1, n+1[, nlev])
    """
    f_pad = _pad_halo_auto(field, cdgrid)

    if field.ndim == 3:
        return 0.25 * (f_pad[:, :-1, :-1] + f_pad[:, 1:, :-1]
                        + f_pad[:, :-1, 1:] + f_pad[:, 1:, 1:])
    return 0.25 * (f_pad[:, :-1, :-1, :] + f_pad[:, 1:, :-1, :]
                    + f_pad[:, :-1, 1:, :] + f_pad[:, 1:, 1:, :])


def _interp_corner_to_center(field_d):
    """Interpolate D-grid corners to cell centres (4-point average).

    Parameters
    ----------
    field_d : jax.Array, shape (6, n+1, n+1[, nlev])

    Returns
    -------
    jax.Array, shape (6, n, n[, nlev])
    """
    if field_d.ndim == 3:
        return 0.25 * (field_d[:, :-1, :-1] + field_d[:, 1:, :-1]
                        + field_d[:, :-1, 1:] + field_d[:, 1:, 1:])
    return 0.25 * (field_d[:, :-1, :-1, :] + field_d[:, 1:, :-1, :]
                    + field_d[:, :-1, 1:, :] + field_d[:, 1:, 1:, :])


# ==============================================================================
# Divergence damping
# ==============================================================================

def _divergence_damping(u_c, v_c, cdgrid, d2_coeff=0.0, d4_coeff=0.0,
                        dddmp=0.0, div_field=None):
    """Compute divergence damping tendencies at D-grid corners.

    Applies 2nd-order (d2) and/or 4th-order (d4) divergence damping,
    optionally with adaptive Smagorinsky-like scaling.

    Parameters
    ----------
    u_c, v_c : C-grid velocities
    cdgrid : CubedSphereCDGrid
    d2_coeff : float
        2nd-order divergence damping coefficient [m^2/s].
    d4_coeff : float
        4th-order divergence damping coefficient [m^4/s].
    dddmp : float
        Smagorinsky adaptive coefficient (0 = off, 0.2 = typical).
    div_field : jax.Array or None
        Pre-computed divergence. If None, computed from u_c, v_c.

    Returns
    -------
    dd_dx, dd_dy : jax.Array at D-grid corners
        Divergence damping tendency for u and v.
    """
    if div_field is None:
        div_field = cgrid_divergence(u_c, v_c, cdgrid)

    dd_dx = jnp.zeros_like(div_field)
    dd_dy = jnp.zeros_like(div_field)

    # Non-orthogonality correction helper for v-equation gradient
    cosa_c = _broadcast_metric(cdgrid.cosa_corner, dd_dx)
    sina_c = jnp.sqrt(jnp.maximum(1.0 - cosa_c**2, 1e-12))

    def _correct_v_grad(grad_x, grad_y):
        """Correct gradient from e_j to e_perp direction."""
        return (grad_y - cosa_c * grad_x) / jnp.maximum(sina_c, 1e-12)

    # 2nd-order: du += d2_coeff * grad(div)
    if d2_coeff > 0:
        ddiv_dx, ddiv_dy = _arakawa_lamb_gradient(div_field, cdgrid)
        ddiv_dy_perp = _correct_v_grad(ddiv_dx, ddiv_dy)
        # Adaptive coefficient
        if dddmp > 0:
            area_min = jnp.min(cdgrid.base.area)
            d2_bg = d2_coeff / area_min
            div_abs = jnp.abs(div_field)
            adaptive = area_min * jnp.maximum(
                d2_bg, jnp.minimum(0.20, dddmp * div_abs))
            adaptive_corner = _interp_center_to_corner(adaptive, cdgrid)
            return adaptive_corner * ddiv_dx, adaptive_corner * ddiv_dy_perp
        else:
            return d2_coeff * ddiv_dx, d2_coeff * ddiv_dy_perp

    # 4th-order: du -= d4_coeff * grad(lap(div))
    if d4_coeff > 0:
        from legoesm.core.operators import laplacian_compact
        if div_field.ndim == 3:
            lap_div = laplacian_compact(div_field, cdgrid.base)
        else:
            # 3D: vmap
            div_t = jnp.moveaxis(div_field, -1, 0)
            lap_div_t = jax.vmap(
                lambda d: laplacian_compact(d, cdgrid.base)
            )(div_t)
            lap_div = jnp.moveaxis(lap_div_t, 0, -1)
        dlap_dx, dlap_dy = _arakawa_lamb_gradient(lap_div, cdgrid)
        dlap_dy_perp = _correct_v_grad(dlap_dx, dlap_dy)
        return -d4_coeff * dlap_dx, -d4_coeff * dlap_dy_perp

    return dd_dx, dd_dy


# ==============================================================================
# Laplacian at D-grid corners
# ==============================================================================

def _laplacian_dgrid(u_d, cdgrid):
    """Laplacian of a D-grid field via cell-centre round-trip.

    Interpolates D-grid -> cell centres, applies the compact
    cell-centre Laplacian (which uses proper inter-face halo exchange),
    then interpolates back to D-grid corners.

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev).
    For 3D, applies the Laplacian level-by-level.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n+1, n+1[, nlev])

    Returns
    -------
    jax.Array, shape (6, n+1, n+1[, nlev])
    """
    if u_d.ndim == 4:
        u_t = jnp.moveaxis(u_d, -1, 0)

        def lap_one(uk):
            return _laplacian_dgrid(uk, cdgrid)

        result = jax.vmap(lap_one)(u_t)
        return jnp.moveaxis(result, 0, -1)

    # 1. D-grid -> cell centres: (6, n+1, n+1) -> (6, n, n)
    u_cc = _interp_corner_to_center(u_d)

    # 2. Cell-centre Laplacian with proper halo exchange
    from legoesm.core.operators import laplacian_compact
    lap_a = laplacian_compact(u_cc, cdgrid.base)  # (6, n, n)

    # 3. Cell centres -> D-grid: (6, n, n) -> (6, n+1, n+1)
    return _interp_center_to_corner(lap_a, cdgrid)


# ==============================================================================
# Vector-invariant momentum tendencies (unified 2D/3D)
# ==============================================================================

def _extrapolate_boundary_corners(du, dv, n):
    """Blend D-grid boundary-corner tendencies toward interior values.

    The D-grid vorticity flux and Bernoulli gradient suffer from poor
    geostrophic cancellation at face-boundary corners because the
    vorticity stencil uses face-local D-grid data while the gradient
    stencil uses cross-face haloed cell-centre data.  The mismatch is
    largest at cube vertices where cos(alpha) = +/-0.5.

    Blends boundary-row tendency values with the nearest interior
    neighbour (w=0.5).  Vertices (where 3 faces meet) are replaced with
    the average of the two adjacent edge values.

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev) inputs.

    Parameters
    ----------
    du, dv : jax.Array, shape (6, n+1, n+1[, nlev])
    n : int  (face tile size; corner indices run 0..n)

    Returns
    -------
    du, dv : jax.Array with blended boundary values
    """
    w = 0.5

    # West (i=0): blend with i=1
    du = du.at[:, 0, 1:-1].set((1 - w) * du[:, 0, 1:-1] + w * du[:, 1, 1:-1])
    dv = dv.at[:, 0, 1:-1].set((1 - w) * dv[:, 0, 1:-1] + w * dv[:, 1, 1:-1])
    # East (i=n): blend with i=n-1
    du = du.at[:, n, 1:-1].set((1 - w) * du[:, n, 1:-1] + w * du[:, n - 1, 1:-1])
    dv = dv.at[:, n, 1:-1].set((1 - w) * dv[:, n, 1:-1] + w * dv[:, n - 1, 1:-1])
    # South (j=0): blend with j=1
    du = du.at[:, 1:-1, 0].set((1 - w) * du[:, 1:-1, 0] + w * du[:, 1:-1, 1])
    dv = dv.at[:, 1:-1, 0].set((1 - w) * dv[:, 1:-1, 0] + w * dv[:, 1:-1, 1])
    # North (j=n): blend with j=n-1
    du = du.at[:, 1:-1, n].set((1 - w) * du[:, 1:-1, n] + w * du[:, 1:-1, n - 1])
    dv = dv.at[:, 1:-1, n].set((1 - w) * dv[:, 1:-1, n] + w * dv[:, 1:-1, n - 1])

    # Vertex corners: average of the two adjacent edge values
    corners = [
        ((0, 0), (1, 0), (0, 1)),
        ((n, 0), (n - 1, 0), (n, 1)),
        ((0, n), (1, n), (0, n - 1)),
        ((n, n), (n - 1, n), (n, n - 1)),
    ]
    for (ci, cj), (n1i, n1j), (n2i, n2j) in corners:
        du = du.at[:, ci, cj].set(0.5 * (du[:, n1i, n1j] + du[:, n2i, n2j]))
        dv = dv.at[:, ci, cj].set(0.5 * (dv[:, n1i, n1j] + dv[:, n2i, n2j]))

    return du, dv


def cdgrid_momentum_tendencies(
    h_or_p, u_d, v_d, h_s_or_p_prime, cdgrid,
    g=9.80616, A_h=0.0, hyperdiff_coeff=0.0, div_damp=0.0,
    rho_0=None, div_v=None, f_3d=None,
    u_prime=None, v_prime=None,
):
    """D-grid momentum tendencies (vector-invariant form).

    Unified for both shallow water (2D) and 3D primitive equations.

    For 2D (shallow water):
        du_d/dt = +zeta_abs * v_d - dB/dx + viscosity + div_damping
        dv_d/dt = -zeta_abs * u_d - dB/dy + viscosity + div_damping
        where B = KE + g*(h + h_s)

    For 3D (primitive equations / ocean):
        du_d/dt = zeta*v_d + f*v' - dKE/dx - (1/rho_0)*dp'/dx + viscosity
        dv_d/dt = -zeta*u_d - f*u' - dKE/dy - (1/rho_0)*dp'/dy + viscosity

    Parameters
    ----------
    h_or_p : jax.Array, shape (6, n, n[, nlev])
    u_d, v_d : jax.Array, shape (6, n+1, n+1[, nlev])
    h_s_or_p_prime : jax.Array, shape (6, n, n)
    cdgrid : CubedSphereCDGrid
    g : float
    A_h : float
    hyperdiff_coeff : float
    div_damp : float
        Divergence damping coefficient.
    rho_0 : float or None
    div_v : jax.Array or None
    f_3d : jax.Array or None
    u_prime, v_prime : jax.Array or None

    Returns
    -------
    du_d_dt, dv_d_dt : jax.Array, shape (6, n+1, n+1[, nlev])
    """
    is_3d = u_d.ndim == 4

    # 1. Vorticity at cell centres
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)
    if is_3d:
        zeta_abs = zeta
    else:
        zeta_abs = zeta + cdgrid.base.f

    # 2. KE at cell centres from D-grid corners (orthogonal basis)
    u_cc, v_cc = dgrid_to_center_vector(u_d, v_d)
    KE = 0.5 * (u_cc ** 2 + v_cc ** 2)

    # Also need C-grid velocities for divergence / mass flux
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)

    # 3. Gradients at corners (Arakawa-Lamb)
    if is_3d:
        dKE_dx, dKE_dy = _arakawa_lamb_gradient(KE, cdgrid)
        dp_dx, dp_dy = _arakawa_lamb_gradient(h_or_p, cdgrid)
    else:
        B = KE + g * (h_or_p + h_s_or_p_prime)
        dB_dx, dB_dy = _arakawa_lamb_gradient(B, cdgrid)

    # 4. Vorticity at corners
    zeta_corner = _interp_center_to_corner(zeta_abs, cdgrid)

    # Non-orthogonality correction for gradient in v-equation:
    # The Arakawa-Lamb gradient gives (dB/ds_i, dB/ds_j).
    # u-equation needs dB/ds_i (no correction).
    # v-equation needs dB/ds_perp = (dB/ds_j - cos(alpha)*dB/ds_i) / sin(alpha).
    cosa_c = _broadcast_metric(cdgrid.cosa_corner, u_d)
    sina_c = jnp.sqrt(jnp.maximum(1.0 - cosa_c**2, 1e-12))

    # 5. Tendencies
    if is_3d:
        # Correct v-equation gradient from e_j to e_perp
        dKE_dy_perp = (dKE_dy - cosa_c * dKE_dx) / jnp.maximum(sina_c, 1e-12)

        du_d_dt = zeta_corner * v_d - dKE_dx
        dv_d_dt = -zeta_corner * u_d - dKE_dy_perp

        if rho_0 is not None:
            dp_dy_perp = (dp_dy - cosa_c * dp_dx) / jnp.maximum(sina_c, 1e-12)
            du_d_dt = du_d_dt - dp_dx / rho_0
            dv_d_dt = dv_d_dt - dp_dy_perp / rho_0

        if f_3d is not None and u_prime is not None and v_prime is not None:
            f_corner = cdgrid.f_corner[..., None]
            du_d_dt = du_d_dt + f_corner * v_prime
            dv_d_dt = dv_d_dt - f_corner * u_prime

        if div_v is not None:
            div_corner = _interp_center_to_corner(div_v, cdgrid)
            du_d_dt = du_d_dt - 0.5 * u_d * div_corner
            dv_d_dt = dv_d_dt - 0.5 * v_d * div_corner
    else:
        # Correct v-equation gradient from e_j to e_perp
        dB_dy_perp = (dB_dy - cosa_c * dB_dx) / jnp.maximum(sina_c, 1e-12)
        du_d_dt = zeta_corner * v_d - dB_dx
        dv_d_dt = -zeta_corner * u_d - dB_dy_perp

    # 6. Laplacian viscosity
    if A_h > 0:
        du_d_dt = du_d_dt + A_h * _laplacian_dgrid(u_d, cdgrid)
        dv_d_dt = dv_d_dt + A_h * _laplacian_dgrid(v_d, cdgrid)

    # 7. Biharmonic hyperdiffusion
    if hyperdiff_coeff > 0:
        du_d_dt = du_d_dt - hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(u_d, cdgrid), cdgrid)
        dv_d_dt = dv_d_dt - hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(v_d, cdgrid), cdgrid)

    # 8. Divergence damping (FV3-style adaptive Smagorinsky)
    if div_damp > 0:
        div_field = cgrid_divergence(u_c, v_c, cdgrid)
        area_min = jnp.min(cdgrid.base.area)
        d2_bg = div_damp / area_min
        dddmp = 0.2
        div_abs = jnp.abs(div_field)
        div_abs_corner = _interp_center_to_corner(div_abs, cdgrid)
        adaptive_coeff = area_min * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * div_abs_corner))
        ddiv_dx, ddiv_dy = _arakawa_lamb_gradient(div_field, cdgrid)
        # Correct v-equation gradient from e_j to e_perp
        ddiv_dy_perp = (ddiv_dy - cosa_c * ddiv_dx) / jnp.maximum(sina_c, 1e-12)
        du_d_dt = du_d_dt + adaptive_coeff * ddiv_dx
        dv_d_dt = dv_d_dt + adaptive_coeff * ddiv_dy_perp

    return du_d_dt, dv_d_dt


# ==============================================================================
# FV3 edge-midpoint D-grid operators
# ==============================================================================
#
# In the FV3 edge-midpoint stagger, prognostic winds live at edge midpoints:
#   u_d : (6, n, n+1) -- x-velocity at midpoint of x-edge
#   v_d : (6, n+1, n) -- y-velocity at midpoint of y-edge

def fv3_vorticity(u_d, v_d, cdgrid):
    """Relative vorticity at cell corners from edge-midpoint D-grid winds.

    Uses the exact circulation form: the four edges surrounding each corner
    contribute directly without any spatial interpolation.

    At face boundaries, the halo line integrals are reconstructed from
    halo-exchanged cell-centre winds to avoid the incorrect ``mode='edge'``
    padding that would repeat same-face edge values.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n, n+1)
    v_d : jax.Array, shape (6, n+1, n)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    vort : jax.Array, shape (6, n+1, n+1)
    """
    from legoesm.grids.halo import pad_halo_vector

    dx = cdgrid.dx_edge_y   # (6, n, n+1)
    dy = cdgrid.dy_edge_x   # (6, n+1, n)
    n = cdgrid.n

    u_dx = u_d * dx          # (6, n, n+1)
    v_dy = v_d * dy          # (6, n+1, n)

    # --- Halo-aware padding ------------------------------------------------
    # Interior: direct line integrals (no change).
    # Boundary halo: reconstruct from halo-exchanged cell-centre winds so
    # that the circulation at shared cube-face corners is physically
    # consistent.

    # Cell-centre winds from edge-midpoint D-grid (simple average)
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])   # (6, n, n)
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])   # (6, n, n)

    grid = cdgrid.base
    u_pad, v_pad = pad_halo_vector(
        u_cc, v_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )   # each (6, n+2, n+2)

    # Reconstruct halo u_dx from padded cell-centre u ----------------------
    # x-edge at position (i, j) sits between cells (i, j-1) and (i, j).
    # In padded coords cell (i, j) -> padded (i+1, j+1).
    # Halo edge i=-1 (padded row 0): average padded[:, 0, j] and [:, 0, j+1]
    # Halo edge i=n  (padded row n+1): average padded[:, n+1, j] and [:, n+1, j+1]
    u_dx_west_halo = (0.5 * (u_pad[:, 0:1, :-1] + u_pad[:, 0:1, 1:])
                      * dx[:, 0:1, :])              # (6, 1, n+1)
    u_dx_east_halo = (0.5 * (u_pad[:, -1:, :-1] + u_pad[:, -1:, 1:])
                      * dx[:, -1:, :])              # (6, 1, n+1)

    u_dx_pad = jnp.concatenate([u_dx_west_halo, u_dx, u_dx_east_halo],
                               axis=1)              # (6, n+2, n+1)

    # Reconstruct halo v_dy from padded cell-centre v ----------------------
    # y-edge at position (i, j) sits between cells (i-1, j) and (i, j).
    # Halo edge j=-1 (padded col 0): average padded[:, i, 0] and [:, i+1, 0]
    # Halo edge j=n  (padded col n+1): average padded[:, i, n+1] and [:, i+1, n+1]
    v_dy_south_halo = (0.5 * (v_pad[:, :-1, 0:1] + v_pad[:, 1:, 0:1])
                       * dy[:, :, 0:1])             # (6, n+1, 1)
    v_dy_north_halo = (0.5 * (v_pad[:, :-1, -1:] + v_pad[:, 1:, -1:])
                       * dy[:, :, -1:])             # (6, n+1, 1)

    v_dy_pad = jnp.concatenate([v_dy_south_halo, v_dy, v_dy_north_halo],
                               axis=2)              # (6, n+1, n+2)

    # --- Circulation -------------------------------------------------------
    u_south = u_dx_pad[:, :-1, :]
    u_north = u_dx_pad[:, 1:, :]
    v_west  = v_dy_pad[:, :, :-1]
    v_east  = v_dy_pad[:, :, 1:]

    circ = u_south - u_north + v_east - v_west

    return circ / cdgrid.area_corner


def fv3_d2cc(u_d, v_d, cdgrid):
    """Edge-midpoint D-grid winds to cell-centre velocities.

    Simple average of the two opposing edge velocities to the cell centre.
    D-grid winds use the orthogonal-rotation convention (geographic wind
    projected using the grid angle), so no non-orthogonality correction
    is needed.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n, n+1)
    v_d : jax.Array, shape (6, n+1, n)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    u_cc, v_cc : jax.Array, shape (6, n, n)
    """
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    return u_cc, v_cc


def fv3_cc2c(u_cc, v_cc, cdgrid):
    """Cell-centre velocities to C-grid edge-normal velocities.

    Uses halo exchange of cell-centre velocities followed by 2nd-order
    interpolation to edge midpoints.

    Parameters
    ----------
    u_cc, v_cc : jax.Array, shape (6, n, n)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    u_c : jax.Array, shape (6, n+1, n)
    v_c : jax.Array, shape (6, n, n+1)
    """
    from legoesm.grids.halo import pad_halo_vector

    grid = cdgrid.base
    u_pad, v_pad = pad_halo_vector(
        u_cc, v_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )

    # Average cell-centre velocities to face positions.
    # The halo exchange already handles cross-face vector rotation,
    # so the interpolated values are in the correct local basis.
    u_c = 0.5 * (u_pad[:, :-1, 1:-1] + u_pad[:, 1:, 1:-1])  # (6, n+1, n)
    v_c = 0.5 * (v_pad[:, 1:-1, :-1] + v_pad[:, 1:-1, 1:])  # (6, n, n+1)

    return u_c, v_c


def fv3_d2cc2c(u_d, v_d, cdgrid):
    """Edge-midpoint D-grid winds to both cell-centre and C-grid velocities.

    Combines :func:`fv3_d2cc` and :func:`fv3_cc2c`.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n, n+1)
    v_d : jax.Array, shape (6, n+1, n)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    u_cc, v_cc : jax.Array, shape (6, n, n)
    u_c : jax.Array, shape (6, n+1, n)
    v_c : jax.Array, shape (6, n, n+1)
    """
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)
    return u_cc, v_cc, u_c, v_c


def fv3_sw_tendencies(
    h, u_d, v_d, h_s, cdgrid,
    g=9.80616, div_damp=0.0, hyperdiff_coeff=0.0,
):
    """Shallow water tendencies on the FV3 edge-midpoint D-grid.

    Uses a corner-based momentum computation for stability (compact
    stencil avoids the wide-stencil computational mode), with
    edge-midpoint mass transport via PPM:

    1. Edge-midpoint D-grid -> cell-centre -> C-grid (mass transport)
    2. Corner winds from D-grid edge averages (pad + average to corners)
    3. Corner momentum tendencies (Arakawa-Lamb gradient + vorticity)
    4. Corner tendencies averaged back to edge-midpoint positions
    5. Biharmonic hyperdiffusion at cell centres projected to edges

    Parameters
    ----------
    h : jax.Array, shape (6, n, n)
    u_d : jax.Array, shape (6, n, n+1)
    v_d : jax.Array, shape (6, n+1, n)
    h_s : jax.Array, shape (6, n, n)
    cdgrid : CubedSphereCDGrid
    g, div_damp, hyperdiff_coeff : float

    Returns
    -------
    dh_dt : (6, n, n), du_d_dt : (6, n, n+1), dv_d_dt : (6, n+1, n)
    """
    n = cdgrid.n

    # (a) D -> cell-centre -> C-grid velocities for mass transport
    u_cc, v_cc, u_c, v_c = fv3_d2cc2c(u_d, v_d, cdgrid)

    # (b) Height tendency (PPM mass flux divergence)
    dh_dt = cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)

    # Zero-mean correction for mass conservation
    total_area = jnp.sum(cdgrid.base.area)
    dh_dt = dh_dt - jnp.sum(dh_dt * cdgrid.base.area) / total_area

    # (c) Corner winds from D-grid edge midpoints
    # u_d (n, n+1) -> pad in i -> average to get (n+1, n+1)
    u_d_pad = jnp.pad(u_d, [(0, 0), (1, 1), (0, 0)], mode='edge')
    u_corner = 0.5 * (u_d_pad[:, :-1, :] + u_d_pad[:, 1:, :])
    # v_d (n+1, n) -> pad in j -> average to get (n+1, n+1)
    v_d_pad = jnp.pad(v_d, [(0, 0), (0, 0), (1, 1)], mode='edge')
    v_corner = 0.5 * (v_d_pad[:, :, :-1] + v_d_pad[:, :, 1:])

    # (d) Corner-based momentum tendencies (stable compact stencil)
    du_corner, dv_corner = cdgrid_momentum_tendencies(
        h, u_corner, v_corner, h_s, cdgrid,
        g=g, A_h=0.0,
        hyperdiff_coeff=hyperdiff_coeff,
        div_damp=div_damp,
    )

    # (e) Average corner tendencies to edge-midpoint positions
    du_d_dt = 0.5 * (du_corner[:, :-1, :] + du_corner[:, 1:, :])   # (6, n, n+1)
    dv_d_dt = 0.5 * (dv_corner[:, :, :-1] + dv_corner[:, :, 1:])   # (6, n+1, n)

    return dh_dt, du_d_dt, dv_d_dt


# ==============================================================================
# Legacy aliases for backward compatibility
# ==============================================================================

def dgrid_vorticity_3d(u_d, v_d, cdgrid):
    """Alias: dgrid_vorticity handles both 2D and 3D."""
    return dgrid_vorticity(u_d, v_d, cdgrid)


def cgrid_divergence_3d(u_c, v_c, cdgrid):
    """Alias: cgrid_divergence handles both 2D and 3D."""
    return cgrid_divergence(u_c, v_c, cdgrid)


def cgrid_mass_flux_divergence_3d(h, u_c, v_c, cdgrid):
    """Alias: cgrid_mass_flux_divergence handles both 2D and 3D."""
    return cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)


def _arakawa_lamb_gradient_3d(B, cdgrid):
    """Alias: _arakawa_lamb_gradient handles both 2D and 3D."""
    return _arakawa_lamb_gradient(B, cdgrid)


def _interp_center_to_corner_3d(field, cdgrid):
    """Alias: _interp_center_to_corner handles both 2D and 3D."""
    return _interp_center_to_corner(field, cdgrid)


def cdgrid_momentum_tendencies_3d(
    u_d, v_d, p_prime, cdgrid, rho_0,
    A_h=0.0, div_v=None, f_3d=None,
    u_prime=None, v_prime=None,
):
    """Alias: cdgrid_momentum_tendencies handles both 2D and 3D."""
    return cdgrid_momentum_tendencies(
        p_prime, u_d, v_d, jnp.zeros(cdgrid.base.area.shape), cdgrid,
        rho_0=rho_0, A_h=A_h, div_v=div_v, f_3d=f_3d,
        u_prime=u_prime, v_prime=v_prime,
    )


# Keep _smooth_boundary_cells for any code that imports it, but make it a no-op
# since FV3-faithful operators should not need boundary smoothing.
def _smooth_boundary_cells(field, cdgrid):
    """No-op: FV3-faithful operators handle boundaries via d2a2c."""
    return field
