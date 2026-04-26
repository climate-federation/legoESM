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
from legoesm.grids.halo import pad_halo, pad_halo_4d

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


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
    dg = cdgrid.base.duogrid
    offsets = None if dg is not None else cdgrid.base.halo_interp_offsets
    if field.ndim == 3:
        return pad_halo(field, interp_offsets=offsets, duogrid=dg)
    return pad_halo_4d(field, interp_offsets=offsets, duogrid=dg)


def _pad_halo_auto_h2(field, cdgrid):
    """Pad halo=2 for a 2D or 3D cell-centre field.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n) or (6, n, n, nlev)

    Returns
    -------
    jax.Array, shape (6, n+4, n+4) or (6, n+4, n+4, nlev)
    """
    dg = cdgrid.base.duogrid
    offsets = None if dg is not None else cdgrid.base.halo_interp_offsets_h2
    if field.ndim == 3:
        return pad_halo(field, halo=2, interp_offsets=offsets, duogrid=dg)
    return pad_halo_4d(field, halo=2, interp_offsets=offsets, duogrid=dg)


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
    from legoesm.grids.halo import pad_halo_vector, pad_halo_vector_4d

    grid = cdgrid.base
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    if u_cc.ndim == 3:
        u_pad, v_pad = pad_halo_vector(
            u_cc, v_cc,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
            interp_offsets=offsets, duogrid=dg,
        )
        u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1]
                       + u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
        v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1]
                       + v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
        return u_d, v_d

    # 4D: native single-message vector halo via pad_halo_vector_4d.
    # Replaces nlev separate ``pad_halo_vector`` MPI calls under the
    # previous per-level vmap.  4-point averaging then proceeds on axes
    # 1, 2 (i, j), with the trailing nlev axis carried through passively.
    u_pad, v_pad = pad_halo_vector_4d(
        u_cc, v_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    u_d = 0.25 * (u_pad[:, :-1, :-1, :] + u_pad[:, 1:, :-1, :]
                   + u_pad[:, :-1, 1:, :] + u_pad[:, 1:, 1:, :])
    v_d = 0.25 * (v_pad[:, :-1, :-1, :] + v_pad[:, 1:, :-1, :]
                   + v_pad[:, :-1, 1:, :] + v_pad[:, 1:, 1:, :])
    return u_d, v_d


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


def dgrid_to_center_geographic(u_d, v_d, cdgrid):
    """D-grid corner velocities -> cell-centre east/north (geographic) winds.

    Rotates each corner velocity to geographic (east, north) coordinates
    BEFORE averaging to cell centres.  This is essential for accurate
    diagnostics: averaging in face-local coordinates then rotating
    produces spurious v_north for solid-body rotation (0.85 m/s at C16),
    while rotating first then averaging gives v_north = 0 to machine
    precision.

    Works for 2D (6, n+1, n+1) only.

    Parameters
    ----------
    u_d, v_d : jax.Array, shape (6, n+1, n+1)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    u_east, v_north : jax.Array, shape (6, n, n) — geographic winds at centres
    """
    ca = cdgrid.cos_angle_corner  # (6, n+1, n+1)
    sa = cdgrid.sin_angle_corner  # (6, n+1, n+1)

    # Rotate to geographic at each corner
    ue = ca * u_d - sa * v_d  # u_east at corners
    vn = sa * u_d + ca * v_d  # v_north at corners

    # Average geographic winds to cell centres
    u_east = 0.25 * (ue[:, :-1, :-1] + ue[:, 1:, :-1]
                      + ue[:, :-1, 1:] + ue[:, 1:, 1:])
    v_north = 0.25 * (vn[:, :-1, :-1] + vn[:, 1:, :-1]
                       + vn[:, :-1, 1:] + vn[:, 1:, 1:])
    return u_east, v_north


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
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, _EPS))
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
    sina = jnp.sqrt(jnp.maximum(1.0 - cosa**2, _EPS))

    u_d = (u_c_avg + v_d * cosa) / jnp.maximum(sina, _EPS)

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
    sina = jnp.sqrt(jnp.maximum(1.0 - cosa**2, _EPS))

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
# C-grid compact gradient (cell centre → edge midpoints)
# ==============================================================================

def cgrid_gradient_2d(eta, cdgrid):
    """Compact C-grid gradient of a cell-centre scalar to edge midpoints.

    Uses single-cell differences scaled by centre-to-centre distances
    (``dxc``, ``dyc``), matching the FV3 Bernoulli gradient stencil.

    Parameters
    ----------
    eta : jax.Array, shape (6, n, n)
        Cell-centre scalar (e.g. free-surface height).
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    deta_dx : jax.Array, shape (6, n+1, n)
        Gradient at x-edge (u) midpoints.
    deta_dy : jax.Array, shape (6, n, n+1)
        Gradient at y-edge (v) midpoints.
    """
    eta_pad = _pad_halo_auto(eta, cdgrid)
    # eta_pad shape: (6, n+2, n+2)  (1-cell halo on each side)

    # x-gradient at u-points: (eta[i,j] - eta[i-1,j]) / dxc
    # In padded coords: interior is [1:-1, 1:-1], so u-faces run 0..n
    deta_dx = (eta_pad[:, 1:, 1:-1] - eta_pad[:, :-1, 1:-1]) * cdgrid.rdxc

    # y-gradient at v-points: (eta[i,j] - eta[i,j-1]) / dyc
    deta_dy = (eta_pad[:, 1:-1, 1:] - eta_pad[:, 1:-1, :-1]) * cdgrid.rdyc

    return deta_dx, deta_dy


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
        # Native 4D path: ``_pad_halo_auto_h2`` dispatches to ``pad_halo_4d``
        # for 4D input, so all vertical levels get one MPI exchange instead of
        # nlev under per-level vmap.  Move nlev to the front so the rest of
        # the 2D body's slicing along axes -2/-1 still operates on (i, j).
        h_pad = _pad_halo_auto_h2(h, cdgrid)  # (6, n+4, n+4, nlev)
        n = cdgrid.n
        dy = cdgrid.dy_edge_x  # (6, n+1, n)
        dx = cdgrid.dx_edge_y  # (6, n, n+1)

        h_pad_t = jnp.moveaxis(h_pad, -1, 0)  # (nlev, 6, n+4, n+4)
        u_c_t = jnp.moveaxis(u_c, -1, 0)      # (nlev, 6, n+1, n)
        v_c_t = jnp.moveaxis(v_c, -1, 0)      # (nlev, 6, n, n+1)

        # X-direction PPM (axis -1 = j after slicing transverse halo).
        h_x_strips_t = h_pad_t[..., 2:-2]              # (nlev, 6, n+4, n)
        q_L_x_t, q_R_x_t = _ppm_reconstruct_1d(h_x_strips_t)
        q_R_left_t = q_R_x_t[..., 1:n+2, :]            # (nlev, 6, n+1, n)
        q_L_right_t = q_L_x_t[..., 2:n+3, :]
        h_face_x_t = jnp.where(u_c_t > 0, q_R_left_t, q_L_right_t)

        # Y-direction PPM (axis -1 = j with halo).
        h_y_strips_t = h_pad_t[..., 2:-2, :]           # (nlev, 6, n, n+4)
        q_L_y_t, q_R_y_t = _ppm_reconstruct_1d(h_y_strips_t)
        q_R_bottom_t = q_R_y_t[..., 1:n+2]             # (nlev, 6, n, n+1)
        q_L_top_t = q_L_y_t[..., 2:n+3]
        h_face_y_t = jnp.where(v_c_t > 0, q_R_bottom_t, q_L_top_t)

        flux_x_t = h_face_x_t * u_c_t * dy             # (nlev, 6, n+1, n)
        flux_y_t = h_face_y_t * v_c_t * dx             # (nlev, 6, n, n+1)

        dg = cdgrid.base.duogrid
        if dg is not None and dg.ng >= 2:
            from legoesm.grids.halo import synchronize_cgrid_fluxes
            flux_x_t, flux_y_t = jax.vmap(
                lambda fx, fy: synchronize_cgrid_fluxes(fx, fy, n),
            )(flux_x_t, flux_y_t)

        net_x_t = flux_x_t[..., 1:, :] - flux_x_t[..., :-1, :]
        net_y_t = flux_y_t[..., 1:] - flux_y_t[..., :-1]
        result_t = -(net_x_t + net_y_t) / cdgrid.base.area
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

    # Duogrid flux synchronization: average boundary fluxes between adjacent
    # faces so that mass leaving face A = mass entering face B.  Required for
    # duogrid where each face independently computes boundary fluxes from its
    # own extended grid.  Matches FV3 dyn_core.F90:853-900.
    # NOT applied for non-duogrid: PPM boundary asymmetry is a feature of
    # the higher-order reconstruction, and averaging reduces accuracy (tested:
    # unconditional sync causes 110x W2 regression).
    dg = cdgrid.base.duogrid
    if dg is not None and dg.ng >= 2:
        from legoesm.grids.halo import synchronize_cgrid_fluxes
        flux_x, flux_y = synchronize_cgrid_fluxes(flux_x, flux_y, n)

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
# Flux-corrected transport (FCT) for monotone tracer advection
# ==============================================================================

def _cgrid_fct_fluxes_2d(q, u_c, v_c, cdgrid):
    """Compute FCT-limited tracer fluxes on the C-D grid (2D only).

    Uses a two-stage approach for monotone transport:

    1. PPM face-value reconstruction with Colella-Woodward limiter,
       followed by clipping each face value to [min, max] of the two
       cells sharing the face.  This prevents the 1D PPM reconstruction
       from producing face values outside the local range (which happens
       near cubed-sphere panel edges due to halo interpolation errors).

    2. First-order upwind fallback blending: after computing the
       face-value-clipped PPM flux divergence and the first-order
       upwind flux divergence, blend them so that the resulting
       tendency cannot push any cell outside the local
       [q_min, q_max] range.  This handles the multidimensional
       aspect — even if each 1D face value is bounded, the combined
       x + y update can still overshoot.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
        Tracer at cell centres.
    u_c : jax.Array, shape (6, n+1, n)
        C-grid x-velocity at x-faces.
    v_c : jax.Array, shape (6, n, n+1)
        C-grid y-velocity at y-faces.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    dq_dt : jax.Array, shape (6, n, n)
        Monotone advective tendency.

    References
    ----------
    - Colella & Woodward (1984): PPM reconstruction.
    - Lin (2004): FV3 transport.
    - Zalesak (1979): Fully multidimensional FCT.
    """
    n = cdgrid.n
    area = cdgrid.base.area   # (6, n, n)
    dy = cdgrid.dy_edge_x     # (6, n+1, n)
    dx = cdgrid.dx_edge_y     # (6, n, n+1)

    # Halo-padded field for neighbour lookups (halo=1)
    q_pad = _pad_halo_auto(q, cdgrid)  # (6, n+2, n+2)

    # ----------------------------------------------------------------
    # Step 1: First-order upwind fluxes (inherently monotone for CFL<1)
    # ----------------------------------------------------------------
    q_left_x = q_pad[:, :-1, 1:-1]    # (6, n+1, n)
    q_right_x = q_pad[:, 1:, 1:-1]    # (6, n+1, n)
    q_face_low_x = jnp.where(u_c > 0, q_left_x, q_right_x)

    q_below_y = q_pad[:, 1:-1, :-1]   # (6, n, n+1)
    q_above_y = q_pad[:, 1:-1, 1:]    # (6, n, n+1)
    q_face_low_y = jnp.where(v_c > 0, q_below_y, q_above_y)

    flux_low_x = q_face_low_x * u_c * dy
    flux_low_y = q_face_low_y * v_c * dx

    net_low_x = flux_low_x[:, 1:] - flux_low_x[:, :-1]
    net_low_y = flux_low_y[:, :, 1:] - flux_low_y[:, :, :-1]
    dq_low = -(net_low_x + net_low_y) / area

    # ----------------------------------------------------------------
    # Step 2: PPM face values with face-value clipping
    # ----------------------------------------------------------------
    q_pad_h2 = _pad_halo_auto_h2(q, cdgrid)  # (6, n+4, n+4)

    # X-direction PPM
    q_x_strips = q_pad_h2[:, :, 2:-2]
    q_L_x, q_R_x = _ppm_reconstruct_1d(q_x_strips)
    q_R_left = q_R_x[:, 1:n+2, :]
    q_L_right = q_L_x[:, 2:n+3, :]
    q_face_hi_x = jnp.where(u_c > 0, q_R_left, q_L_right)

    # Clip to local bounds of adjacent cells
    q_face_min_x = jnp.minimum(q_left_x, q_right_x)
    q_face_max_x = jnp.maximum(q_left_x, q_right_x)
    q_face_hi_x = jnp.clip(q_face_hi_x, q_face_min_x, q_face_max_x)

    # Y-direction PPM
    q_y_strips = q_pad_h2[:, 2:-2, :]
    q_L_y, q_R_y = _ppm_reconstruct_1d(q_y_strips)
    q_R_bottom = q_R_y[:, :, 1:n+2]
    q_L_top = q_L_y[:, :, 2:n+3]
    q_face_hi_y = jnp.where(v_c > 0, q_R_bottom, q_L_top)

    # Clip to local bounds of adjacent cells
    q_face_min_y = jnp.minimum(q_below_y, q_above_y)
    q_face_max_y = jnp.maximum(q_below_y, q_above_y)
    q_face_hi_y = jnp.clip(q_face_hi_y, q_face_min_y, q_face_max_y)

    flux_hi_x = q_face_hi_x * u_c * dy
    flux_hi_y = q_face_hi_y * v_c * dx

    net_hi_x = flux_hi_x[:, 1:] - flux_hi_x[:, :-1]
    net_hi_y = flux_hi_y[:, :, 1:] - flux_hi_y[:, :, :-1]
    dq_hi = -(net_hi_x + net_hi_y) / area

    # ----------------------------------------------------------------
    # Step 3: Zalesak-style blending (high ← low fallback)
    #
    # Compute anti-diffusive tendency = dq_hi - dq_low.
    # Limit it so that the total tendency dq_low + alpha * (dq_hi - dq_low)
    # does not push q outside [q_min, q_max] for any cell, even for a
    # unit-CFL step where the tendency acts as a full update.
    #
    # Since the actual dt is always ≤ dx/u (CFL), limiting for dt=1
    # (i.e. treating the tendency as the update) is strictly conservative.
    # ----------------------------------------------------------------
    ad = dq_hi - dq_low  # anti-diffusive tendency

    # Local min/max including all face-adjacent neighbours
    q_min = q
    q_max = q
    q_min = jnp.minimum(q_min, q_pad[:, :-2, 1:-1])   # west
    q_min = jnp.minimum(q_min, q_pad[:, 2:, 1:-1])    # east
    q_min = jnp.minimum(q_min, q_pad[:, 1:-1, :-2])   # south
    q_min = jnp.minimum(q_min, q_pad[:, 1:-1, 2:])    # north
    q_max = jnp.maximum(q_max, q_pad[:, :-2, 1:-1])
    q_max = jnp.maximum(q_max, q_pad[:, 2:, 1:-1])
    q_max = jnp.maximum(q_max, q_pad[:, 1:-1, :-2])
    q_max = jnp.maximum(q_max, q_pad[:, 1:-1, 2:])

    # How much room does the low-order update leave?
    # After applying dq_low, q would be at q + dq_low (for unit "dt").
    # We allow the anti-diffusive part to bring it to at most q_max
    # and at least q_min.
    q_td = q + dq_low   # provisional (unit-step low-order update)

    room_up = q_max - q_td     # how much we can still increase
    room_dn = q_td - q_min     # how much we can still decrease

    # Per-cell blending factor alpha ∈ [0, 1]:
    # If ad > 0 (high-order wants to increase), alpha = room_up / ad
    # If ad < 0 (high-order wants to decrease), alpha = room_dn / |ad|
    # If ad == 0, alpha = 1 (no correction needed)
    #
    # NOTE: jnp.where evaluates BOTH branches for all elements before
    # selecting.  Division by ad when ad ≈ 0 produces inf/NaN values
    # that are discarded in the forward pass but propagate through
    # jax.grad.  Use safe denominators clamped away from zero so that
    # the unevaluated branch never divides by zero.
    eps = 1.0e-30
    safe_ad_pos = jnp.maximum(ad, eps)    # always > 0 — safe for branch ad > eps
    safe_ad_neg = jnp.minimum(ad, -eps)   # always < 0 — safe for branch ad < -eps
    alpha = jnp.where(
        ad > eps,
        jnp.minimum(1.0, room_up / safe_ad_pos),
        jnp.where(
            ad < -eps,
            jnp.minimum(1.0, room_dn / (-safe_ad_neg)),
            1.0,
        ),
    )
    alpha = jnp.clip(alpha, 0.0, 1.0)

    return dq_low + alpha * ad


def _make_fct_2d_differentiable(cdgrid):
    """Create a differentiable FCT function that closes over cdgrid.

    Returns a ``custom_jvp``-wrapped function whose forward pass uses the
    full FCT limiter (monotone) and whose JVP linearizes through the
    unlimited PPM scheme (always differentiable).

    ``cdgrid`` is captured by closure so that JAX never traces its
    integer fields (``n``, etc.) as differentiable primals.

    This is the standard approach for non-smooth limiters in
    differentiable simulation: the limiter is a nonlinear correction
    whose linearization is the unlimited high-order scheme.
    """

    @jax.custom_jvp
    def _fct(q, u_c, v_c):
        return _cgrid_fct_fluxes_2d(q, u_c, v_c, cdgrid)

    @_fct.defjvp
    def _fct_jvp(primals, tangents):
        q, u_c, v_c = primals
        dq, du_c, dv_c = tangents
        primal_out = _cgrid_fct_fluxes_2d(q, u_c, v_c, cdgrid)
        _, tangent_out = jax.jvp(
            lambda q_, u_, v_: cgrid_mass_flux_divergence(q_, u_, v_, cdgrid),
            (q, u_c, v_c),
            (dq, du_c, dv_c),
        )
        return primal_out, tangent_out

    return _fct


def cgrid_tracer_advection_fct(q, u_c, v_c, cdgrid):
    """Monotone tracer advection using PPM with local-bounds clipping.

    Combines high-order PPM reconstruction with face-value clipping
    to ensure that face values at each interface lie within [min, max]
    of the two adjacent cells. This prevents the creation of new
    extrema that unlimited PPM produces on the cubed-sphere, especially
    near panel edges where halo interpolation introduces errors.

    The scheme is:
    - Conservative (flux-form divergence)
    - Monotone (face values bounded by adjacent cell values)
    - dt-independent (no time step required for the limiter)
    - Differentiable (custom JVP linearizes through unlimited PPM)
    - JAX-compatible (pure array operations, no Python control flow)

    Works for both 2D (6, n, n) and 3D (6, n, n, nlev) inputs.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n[, nlev])
        Tracer at cell centres.
    u_c : jax.Array, shape (6, n+1, n[, nlev])
        C-grid x-velocity at x-faces.
    v_c : jax.Array, shape (6, n, n+1[, nlev])
        C-grid y-velocity at y-faces.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    dq_dt : jax.Array, shape (6, n, n[, nlev])
        Monotone tracer advection tendency.
    """
    fct_fn = _make_fct_2d_differentiable(cdgrid)

    if q.ndim == 4:
        # 3D: vmap over levels
        q_t = jnp.moveaxis(q, -1, 0)
        u_c_t = jnp.moveaxis(u_c, -1, 0)
        v_c_t = jnp.moveaxis(v_c, -1, 0)

        def fct_one(args):
            qk, uk, vk = args
            return fct_fn(qk, uk, vk)

        result_t = jax.vmap(fct_one)((q_t, u_c_t, v_c_t))
        return jnp.moveaxis(result_t, 0, -1)

    return fct_fn(q, u_c, v_c)


# ==============================================================================
# Arakawa-Lamb gradient at D-grid corners
# ==============================================================================

def _arakawa_lamb_gradient(B, cdgrid, padded=None):
    """4-point Arakawa-Lamb gradient at D-grid corners.

    Returns the gradient in physical (e_x, e_perp) coordinates using a
    precomputed transformation matrix derived from 3D Cartesian geometry.
    This eliminates the separate non-orthogonality correction and gives
    correct gradients at face boundaries and cube vertices where face-local
    metrics are inconsistent across faces.

    Works for both 2D (6, n, n) and 3D (6, n, n, nlev) inputs.

    Parameters
    ----------
    B : jax.Array, shape (6, n, n[, nlev])
    cdgrid : CubedSphereCDGrid
    padded : jax.Array, optional
        Pre-padded field (6, n+2, n+2[, nlev]).  Skips internal halo
        exchange when provided (stage-level packing).

    Returns
    -------
    dB_dx, dB_dy_perp : jax.Array, shape (6, n+1, n+1[, nlev])
        Gradient along face-local e_x and perpendicular to e_x.
    """
    B_pad = padded if padded is not None else _pad_halo_auto(B, cdgrid)

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

    # Raw 4-point finite-difference quantities
    dB_raw_x = (B_se + B_ne) - (B_sw + B_nw)  # east − west
    dB_raw_y = (B_nw + B_ne) - (B_sw + B_se)  # north − south

    # Precomputed 2×2 gradient matrix (3D Cartesian → face-local)
    c00 = _broadcast_metric(cdgrid.grad_c00, dB_raw_x)
    c01 = _broadcast_metric(cdgrid.grad_c01, dB_raw_x)
    c10 = _broadcast_metric(cdgrid.grad_c10, dB_raw_x)
    c11 = _broadcast_metric(cdgrid.grad_c11, dB_raw_x)

    dB_dx = c00 * dB_raw_x + c01 * dB_raw_y
    dB_dy_perp = c10 * dB_raw_x + c11 * dB_raw_y

    return dB_dx, dB_dy_perp


# ==============================================================================
# Interpolation helpers
# ==============================================================================

def _interp_center_to_corner(field, cdgrid, padded=None):
    """Interpolate cell-centre field to D-grid corners (4-point average).

    Works for both 2D (6, n, n) and 3D (6, n, n, nlev) inputs.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n[, nlev])
    cdgrid : CubedSphereCDGrid
    padded : jax.Array, optional
        Pre-padded field (6, n+2, n+2[, nlev]).  When provided, the
        internal halo exchange is skipped — used by stage-level
        packing to avoid redundant collectives.

    Returns
    -------
    jax.Array, shape (6, n+1, n+1[, nlev])
    """
    f_pad = padded if padded is not None else _pad_halo_auto(field, cdgrid)

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

    # 2nd-order: du += d2_coeff * grad(div)
    if d2_coeff > 0:
        ddiv_dx, ddiv_dy_perp = _arakawa_lamb_gradient(div_field, cdgrid)
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
        dlap_dx, dlap_dy_perp = _arakawa_lamb_gradient(lap_div, cdgrid)
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
    """Fix momentum tendencies at cube-vertex corners.

    The Arakawa-Lamb gradient at the 8 cube vertices (where 3 faces
    meet) has O(dx) error because all 4 stencil cells are from
    different faces with halo interpolation errors.  Edge-interior
    boundary corners use 2 on-face + 2 halo cells and have O(dx^2)
    accuracy (only ~2x worse than deep interior).

    Vertex corners are replaced using bilinear extrapolation from the
    3 nearest edge/interior corners:
        tend(0,0) = tend(1,0) + tend(0,1) - tend(1,1)

    This gives O(dx^2) accuracy because the 3 source points are
    O(dx^2) accurate, and bilinear extrapolation preserves the order.

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev) inputs.

    Parameters
    ----------
    du, dv : jax.Array, shape (6, n+1, n+1[, nlev])
    n : int  (face tile size; corner indices run 0..n)

    Returns
    -------
    du, dv : jax.Array with fixed boundary values
    """
    # Bilinear extrapolation at vertex corners from 3 nearby points.
    # For corner (0,0): use (1,0), (0,1), (1,1):
    #   tend(0,0) = tend(1,0) + tend(0,1) - tend(1,1)
    corners = [
        # (vertex, edge_nb1, edge_nb2, interior_diag)
        ((0, 0), (1, 0), (0, 1), (1, 1)),
        ((n, 0), (n - 1, 0), (n, 1), (n - 1, 1)),
        ((0, n), (1, n), (0, n - 1), (1, n - 1)),
        ((n, n), (n - 1, n), (n, n - 1), (n - 1, n - 1)),
    ]
    for (ci, cj), (e1i, e1j), (e2i, e2j), (di, dj) in corners:
        du = du.at[:, ci, cj].set(
            du[:, e1i, e1j] + du[:, e2i, e2j] - du[:, di, dj])
        dv = dv.at[:, ci, cj].set(
            dv[:, e1i, e1j] + dv[:, e2i, e2j] - dv[:, di, dj])

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
        dKE_dx, dKE_dy_perp = _arakawa_lamb_gradient(KE, cdgrid)
        dp_dx, dp_dy_perp = _arakawa_lamb_gradient(h_or_p, cdgrid)
    else:
        B = KE + g * (h_or_p + h_s_or_p_prime)
        dB_dx, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid)

    # 4. Vorticity at corners
    zeta_corner = _interp_center_to_corner(zeta_abs, cdgrid)

    # 5. Tendencies (gradient already in physical e_x / e_perp coordinates)
    if is_3d:
        du_d_dt = zeta_corner * v_d - dKE_dx
        dv_d_dt = -zeta_corner * u_d - dKE_dy_perp

        if rho_0 is not None:
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
        du_d_dt = zeta_corner * v_d - dB_dx
        dv_d_dt = -zeta_corner * u_d - dB_dy_perp

    # 6. Laplacian viscosity
    # 7. Biharmonic hyperdiffusion
    #
    # For 2D shallow water, apply diffusion DIRECTLY at D-grid corners
    # in GEOGRAPHIC (east/north) coordinates.  The 5-point Laplacian
    # operates on the (n+1, n+1) corner grid, which sees the 2Δx
    # computational mode that the old cell-center path misses (D→A
    # averaging kills the 2Δx mode before the Laplacian can damp it).
    # Boundary corners (i=0, n; j=0, n) are left untouched (handled
    # by _extrapolate_boundary_corners); interior corners use on-face data.
    if (A_h > 0 or hyperdiff_coeff > 0) and not is_3d:
        from legoesm.core.operators import laplacian_compact
        # Geographic-frame diffusion at CELL CENTRES.  Geographic winds
        # are smooth across face boundaries AND at the poles (unlike
        # face-local or geographic at corners).  The D→A averaging kills
        # the 2Δx computational mode, so this diffusion only smooths
        # resolved scales.  The 2Δx mode is handled separately by a
        # light filter in the model step function.
        ca_c = cdgrid.cos_angle_corner
        sa_c = cdgrid.sin_angle_corner
        ue = ca_c * u_d - sa_c * v_d
        vn = sa_c * u_d + ca_c * v_d
        ue_cc = 0.25 * (ue[:, :-1, :-1] + ue[:, 1:, :-1]
                        + ue[:, :-1, 1:] + ue[:, 1:, 1:])
        vn_cc = 0.25 * (vn[:, :-1, :-1] + vn[:, 1:, :-1]
                        + vn[:, :-1, 1:] + vn[:, 1:, 1:])
        lap_ue = laplacian_compact(ue_cc, cdgrid.base)
        lap_vn = laplacian_compact(vn_cc, cdgrid.base)

        if A_h > 0:
            cos_a = jnp.cos(cdgrid.base.angle)
            sin_a = jnp.sin(cdgrid.base.angle)
            lap_u_local = cos_a * lap_ue + sin_a * lap_vn
            lap_v_local = -sin_a * lap_ue + cos_a * lap_vn
            lap_u_corner = _interp_center_to_corner(lap_u_local, cdgrid)
            lap_v_corner = _interp_center_to_corner(lap_v_local, cdgrid)
            du_d_dt = du_d_dt + A_h * lap_u_corner
            dv_d_dt = dv_d_dt + A_h * lap_v_corner

        if hyperdiff_coeff > 0:
            bilap_ue = laplacian_compact(lap_ue, cdgrid.base)
            bilap_vn = laplacian_compact(lap_vn, cdgrid.base)
            cos_a = jnp.cos(cdgrid.base.angle)
            sin_a = jnp.sin(cdgrid.base.angle)
            bilap_u_local = cos_a * bilap_ue + sin_a * bilap_vn
            bilap_v_local = -sin_a * bilap_ue + cos_a * bilap_vn
            bilap_u_corner = _interp_center_to_corner(bilap_u_local, cdgrid)
            bilap_v_corner = _interp_center_to_corner(bilap_v_local, cdgrid)
            du_d_dt = du_d_dt - hyperdiff_coeff * bilap_u_corner
            dv_d_dt = dv_d_dt - hyperdiff_coeff * bilap_v_corner

    elif A_h > 0 or hyperdiff_coeff > 0:
        # 3D fallback: use original _laplacian_dgrid (for ocean/PE models)
        if A_h > 0:
            du_d_dt = du_d_dt + A_h * _laplacian_dgrid(u_d, cdgrid)
            dv_d_dt = dv_d_dt + A_h * _laplacian_dgrid(v_d, cdgrid)
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
        ddiv_dx, ddiv_dy_perp = _arakawa_lamb_gradient(div_field, cdgrid)
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
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    u_pad, v_pad = pad_halo_vector(
        u_cc, v_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
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
    interpolation to edge midpoints.  When duogrid is active, the vector
    halo exchange uses the duogrid scalar remap for smoother cross-face
    data.

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
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    u_pad, v_pad = pad_halo_vector(
        u_cc, v_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )

    # Average cell-centre velocities to C-grid face positions with
    # non-orthogonality correction for the edge-normal projection
    # (same correction that dgrid_to_cgrid applies for corner D-grid).
    u_avg = 0.5 * (u_pad[:, :-1, 1:-1] + u_pad[:, 1:, 1:-1])  # (6, n+1, n)
    v_at_u = 0.5 * (v_pad[:, :-1, 1:-1] + v_pad[:, 1:, 1:-1])  # v at u_c pos
    cosa_u = _broadcast_metric(cdgrid.cosa_u, u_avg)
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, _EPS))
    u_c = u_avg * sina_u - v_at_u * cosa_u

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
    boundary_fix=False,
    zero_mean_correction=False,
):
    """Shallow water tendencies on the FV3 edge-midpoint D-grid.

    Computes momentum at D-grid corners via the Arakawa-Lamb gradient
    and circulation-based vorticity, then averages to edge-midpoint
    positions.  Mass transport uses PPM via fv3_cc2c physical C-grid
    velocities.

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

    # (a) Cell-centre and C-grid velocities for mass transport
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)

    # (b) Height tendency (PPM mass flux divergence)
    dh_dt = cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)
    if zero_mean_correction:
        total_area = jnp.sum(cdgrid.base.area)
        dh_dt = dh_dt - jnp.sum(dh_dt * cdgrid.base.area) / total_area

    # (c) Bernoulli function using physical cell-centre winds
    KE = 0.5 * (u_cc ** 2 + v_cc ** 2)
    B = KE + g * (h + h_s)

    # (d) Arakawa-Lamb gradient at D-grid corners
    dB_dx, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid)

    # (e) Corner winds from halo-exchanged cell-centre velocities.
    # Both vorticity and gradient use haloed cell-centre data, giving
    # CONSISTENT interpolation errors that cancel in geostrophic balance
    # (tested: D-grid circulation vorticity breaks this cancellation,
    # causing 3x W2 regression despite 4x W5 improvement).
    from legoesm.grids.halo import pad_halo_vector
    grid = cdgrid.base
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    u_cc_pad, v_cc_pad = pad_halo_vector(
        u_cc, v_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    u_corner = 0.25 * (u_cc_pad[:, :-1, :-1] + u_cc_pad[:, 1:, :-1]
                        + u_cc_pad[:, :-1, 1:] + u_cc_pad[:, 1:, 1:])
    v_corner = 0.25 * (v_cc_pad[:, :-1, :-1] + v_cc_pad[:, 1:, :-1]
                        + v_cc_pad[:, :-1, 1:] + v_cc_pad[:, 1:, 1:])

    # (f) Cell-centre vorticity from corner winds (consistent with gradient).
    zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
    zeta_abs = zeta + cdgrid.base.f
    zeta_corner = _interp_center_to_corner(zeta_abs, cdgrid)

    # (g) Momentum tendencies at CELL CENTRES (better geostrophic balance).
    # Computing both gradient and vorticity at the same stagger (cell centres)
    # gives 2.6x better cancellation than at D-grid corners, because the
    # halo-exchanged fields have consistent interpolation errors at cell centres.
    dB_dx_cc = _interp_corner_to_center(dB_dx)
    dB_dy_cc = _interp_corner_to_center(dB_dy_perp)
    du_cc = zeta_abs * v_cc - dB_dx_cc      # (6, n, n)
    dv_cc = -zeta_abs * u_cc - dB_dy_cc     # (6, n, n)

    # (h) Divergence damping at cell centres
    if div_damp > 0:
        div_field = cgrid_divergence(u_c, v_c, cdgrid)
        area_min = jnp.min(cdgrid.base.area)
        d2_bg = div_damp / area_min
        dddmp = 0.2
        div_abs = jnp.abs(div_field)
        adaptive_coeff = area_min * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * div_abs))
        ddiv_dx, ddiv_dy_perp_cc = _arakawa_lamb_gradient(div_field, cdgrid)
        du_cc = du_cc + adaptive_coeff * _interp_corner_to_center(ddiv_dx)
        dv_cc = dv_cc + adaptive_coeff * _interp_corner_to_center(ddiv_dy_perp_cc)

    # (i) Biharmonic hyperdiffusion (cell-centre geographic path)
    if hyperdiff_coeff > 0:
        from legoesm.core.operators import laplacian_compact
        cos_a = jnp.cos(cdgrid.base.angle)
        sin_a = jnp.sin(cdgrid.base.angle)
        ue_cc = cos_a * u_cc - sin_a * v_cc
        vn_cc = sin_a * u_cc + cos_a * v_cc
        lap_ue = laplacian_compact(ue_cc, cdgrid.base)
        bilap_ue = laplacian_compact(lap_ue, cdgrid.base)
        lap_vn = laplacian_compact(vn_cc, cdgrid.base)
        bilap_vn = laplacian_compact(lap_vn, cdgrid.base)
        bilap_u_local = cos_a * bilap_ue + sin_a * bilap_vn
        bilap_v_local = -sin_a * bilap_ue + cos_a * bilap_vn
        du_cc = du_cc - hyperdiff_coeff * bilap_u_local
        dv_cc = dv_cc - hyperdiff_coeff * bilap_v_local

    # (j) Smooth face-boundary tendencies. The geostrophic imbalance at boundary
    # cells (rows 0 and n-1) is 7.7x larger than interior due to halo
    # interpolation error in corner winds and gradient. Blending with the
    # adjacent interior reduces this without affecting balanced flows.
    if boundary_fix and n > 2:
        du_cc = du_cc.at[:, 0, :].set(0.5 * (du_cc[:, 0, :] + du_cc[:, 1, :]))
        du_cc = du_cc.at[:, n-1, :].set(0.5 * (du_cc[:, n-1, :] + du_cc[:, n-2, :]))
        du_cc = du_cc.at[:, :, 0].set(0.5 * (du_cc[:, :, 0] + du_cc[:, :, 1]))
        du_cc = du_cc.at[:, :, n-1].set(0.5 * (du_cc[:, :, n-1] + du_cc[:, :, n-2]))
        dv_cc = dv_cc.at[:, 0, :].set(0.5 * (dv_cc[:, 0, :] + dv_cc[:, 1, :]))
        dv_cc = dv_cc.at[:, n-1, :].set(0.5 * (dv_cc[:, n-1, :] + dv_cc[:, n-2, :]))
        dv_cc = dv_cc.at[:, :, 0].set(0.5 * (dv_cc[:, :, 0] + dv_cc[:, :, 1]))
        dv_cc = dv_cc.at[:, :, n-1].set(0.5 * (dv_cc[:, :, n-1] + dv_cc[:, :, n-2]))

    # (k) Project cell-centre tendencies to D-grid edge-midpoints via halo exchange
    du_cc_pad, dv_cc_pad = pad_halo_vector(
        du_cc, dv_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    du_d_dt = 0.5 * (du_cc_pad[:, 1:-1, :-1] + du_cc_pad[:, 1:-1, 1:])   # (6, n, n+1)
    dv_d_dt = 0.5 * (dv_cc_pad[:, :-1, 1:-1] + dv_cc_pad[:, 1:, 1:-1])   # (6, n+1, n)

    return dh_dt, du_d_dt, dv_d_dt




# ==============================================================================
# Overlapped (async) halo variants for MPI compute-communication overlap
# ==============================================================================

def _overlapped_interp_center_to_corner(field, cdgrid, masks=None):
    """Like _interp_center_to_corner but with interior/boundary overlap.

    Computes interior stencil before halo exchange, boundary after.
    Only beneficial under MPI where halo exchange has latency.

    Only supports 4D fields (6, n, n, nlev) — the 3D case is handled
    by the standard function since 2D halos are very cheap.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n, nlev)
    cdgrid : CubedSphereCDGrid
    masks : InteriorBoundaryMasks, optional
        Pre-computed masks.  Created on-the-fly if None.

    Returns
    -------
    jax.Array, shape (6, n+1, n+1, nlev)
    """
    if field.ndim == 3:
        return _interp_center_to_corner(field, cdgrid)

    from legoesm.parallel.async_halo import overlapped_halo_compute

    def _stencil_body(f_pad):
        """4-point average on padded (6, n+2, n+2) field -> (6, n+1, n+1)."""
        return 0.25 * (f_pad[:, :-1, :-1] + f_pad[:, 1:, :-1]
                        + f_pad[:, :-1, 1:] + f_pad[:, 1:, 1:])

    # Apply per-level via vmap over trailing axis
    # overlapped_halo_compute works on 2D (6, n, n) fields
    field_t = jnp.moveaxis(field, -1, 0)  # (nlev, 6, n, n)
    result_t = jax.vmap(
        lambda f: overlapped_halo_compute(f, _stencil_body, halo_width=1, masks=masks)
    )(field_t)
    return jnp.moveaxis(result_t, 0, -1)  # (6, n+1, n+1, nlev)


def _overlapped_arakawa_lamb_gradient(B, cdgrid, masks=None):
    """Like _arakawa_lamb_gradient but with interior/boundary overlap.

    Only supports 4D fields (6, n, n, nlev).

    Parameters
    ----------
    B : jax.Array, shape (6, n, n, nlev)
    cdgrid : CubedSphereCDGrid
    masks : InteriorBoundaryMasks, optional

    Returns
    -------
    dB_dx, dB_dy_perp : each (6, n+1, n+1, nlev)
    """
    if B.ndim == 3:
        return _arakawa_lamb_gradient(B, cdgrid)

    from legoesm.parallel.async_halo import overlapped_halo_compute

    c00 = cdgrid.grad_c00
    c01 = cdgrid.grad_c01
    c10 = cdgrid.grad_c10
    c11 = cdgrid.grad_c11

    def _stencil_body(f_pad):
        """Arakawa-Lamb gradient stencil -> (6, n+1, n+1, 2) packed dx/dy."""
        B_sw = f_pad[:, :-1, :-1]
        B_se = f_pad[:, 1:, :-1]
        B_nw = f_pad[:, :-1, 1:]
        B_ne = f_pad[:, 1:, 1:]
        dB_raw_x = (B_se + B_ne) - (B_sw + B_nw)
        dB_raw_y = (B_nw + B_ne) - (B_sw + B_se)
        dB_dx = c00 * dB_raw_x + c01 * dB_raw_y
        dB_dy = c10 * dB_raw_x + c11 * dB_raw_y
        return jnp.stack([dB_dx, dB_dy], axis=-1)  # (6, n+1, n+1, 2)

    # Apply per-level
    B_t = jnp.moveaxis(B, -1, 0)  # (nlev, 6, n, n)
    result_t = jax.vmap(
        lambda f: overlapped_halo_compute(f, _stencil_body, halo_width=1, masks=masks)
    )(B_t)  # (nlev, 6, n+1, n+1, 2)
    result = jnp.moveaxis(result_t, 0, -2)  # (6, n+1, n+1, nlev, 2)
    return result[..., 0], result[..., 1]
