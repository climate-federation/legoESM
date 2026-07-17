"""C-D grid operators on cubed-sphere — FV3-faithful rewrite. Lin 2004, Putman & Lin 2007, Colella & Woodward 1984.

PPM transport, D-grid vorticity, d2a2c, C-grid divergence/mass flux, A-L gradient, div damping, Laplacian/biharmonic.
Handles 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev) automatically.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.operators import laplacian_compact
from legoesm.core.operators_3d import laplacian_compact_3d
from legoesm.grids.halo import (
    pad_halo,
    pad_halo_4d,
    pad_halo_vector,
    pad_halo_vector_4d,
    synchronize_cgrid_fluxes,
)

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


# ==============================================================================
# Internal: halo padding that works for both 2D and 3D
# ==============================================================================

def pad_halo_auto(field, cdgrid):
    """Pad halo=1 for 2D/3D cell-centre field."""
    dg = cdgrid.base.duogrid
    offsets = None if dg is not None else cdgrid.base.halo_interp_offsets
    if field.ndim == 3:
        return pad_halo(field, interp_offsets=offsets, duogrid=dg)
    return pad_halo_4d(field, interp_offsets=offsets, duogrid=dg)


def _pad_halo_auto_h2(field, cdgrid):
    """Pad halo=2 for 2D/3D cell-centre field."""
    dg = cdgrid.base.duogrid
    offsets = None if dg is not None else cdgrid.base.halo_interp_offsets_h2
    if field.ndim == 3:
        return pad_halo(field, halo=2, interp_offsets=offsets, duogrid=dg)
    return pad_halo_4d(field, halo=2, interp_offsets=offsets, duogrid=dg)


# Public alias of the halo=2 cc pad (companion to the public ``pad_halo_auto``):
# the sub-face tiled stage (legoesm.parallel.tiled_production_cdgrid) needs it
# cross-package for the deep-h mass pre-pad, and the no-private-cross-import CI
# ratchet forbids importing the ``_``-prefixed name.
pad_halo_auto_h2 = _pad_halo_auto_h2


def _broadcast_metric(metric, field):
    """Broadcast a 2D metric (6, ...) to match field's trailing nlev dim."""
    if field.ndim > metric.ndim:
        return metric[..., None]
    return metric


# ==============================================================================
# PPM (Piecewise Parabolic Method) transport
# ==============================================================================

def _ppm_reconstruct_1d(q, *, axis: int,
                        apply_fortran_xppm_boundary: bool = False,
                        n_interior: int | None = None):
    """PPM face-value reconstruction along axis (Colella & Woodward 1984).

    iter-509 (Codex): axis is REQUIRED kwonly to prevent the iter-505/506/508 axis-default bug.
    apply_fortran_xppm_boundary: tp_core.F90:357-369 iord<7 cube-edge override (constants tp_core.F90:63-65).
    Partial fix using uniform-grid xt; full dxa-weighted formula deferred.
    """
    # Normalize negative axis to positive for clarity.
    orig_axis = q.ndim + axis if axis < 0 else axis
    if not 0 <= orig_axis < q.ndim:
        raise ValueError(
            f"_ppm_reconstruct_1d: axis={axis} out of range for "
            f"input of ndim={q.ndim}.")
    if q.shape[orig_axis] < 4:
        raise ValueError(
            f"_ppm_reconstruct_1d: requires size along axis>=4 for the "
            f"4th-order stencil, got {q.shape[orig_axis]} along axis "
            f"{orig_axis}.")
    # If axis is not last, move it to last, run the reconstruction,
    # then move it back.  This consolidates the swapaxes pattern that
    # iter-505 and iter-506 had to repeat at every call site.
    moved = orig_axis != q.ndim - 1
    if moved:
        q = jnp.moveaxis(q, orig_axis, -1)

    q.shape[-1]

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
    # q_face shape (..., N+1): face values at positions -1/2, 1/2, ..., N-1/2

    # iter-892: Fortran-faithful cube-edge boundary overrides (tp_core.F90:357-369 iord<7)
    # Overrides 4 cube-edge faces (al(1)/al(2)/al(npx-1)/al(npx)); al(0)/al(npx+1) need halo=3.
    # iter-899: production strip is q[k]=q1(k-2) (Hypothesis A; q_pad[k]=q1(k-4), q_face[k]=al(k-2))
    if apply_fortran_xppm_boundary and n_interior is not None:
        c1 = -2.0 / 14.0
        c2 = 11.0 / 14.0
        c3 = 5.0 / 14.0
        n_int = int(n_interior)

        # iter-892 LEFT (al(1)/al(2)). 1-cell shift bug (iter-899 Hypothesis B vs A) kept as
        # production default — measured 17% W2 v_ll_Linf reduction (iter-893).
        xt_L = (0.75 * (q_pad[..., 3] + q_pad[..., 4])
                - 0.25 * (q_pad[..., 2] + q_pad[..., 5]))
        q_lo_L = jnp.minimum(jnp.minimum(q_pad[..., 2], q_pad[..., 3]),
                              jnp.minimum(q_pad[..., 4], q_pad[..., 5]))
        q_hi_L = jnp.maximum(jnp.maximum(q_pad[..., 2], q_pad[..., 3]),
                              jnp.maximum(q_pad[..., 4], q_pad[..., 5]))
        face_al1 = jnp.clip(xt_L, q_lo_L, q_hi_L)
        face_al2 = (c3 * q_pad[..., 4] + c2 * q_pad[..., 5]
                    + c1 * q_pad[..., 6])

        q_face = q_face.at[..., 2].set(face_al1)
        q_face = q_face.at[..., 3].set(face_al2)

        # iter-892 RIGHT (al(npx-1)/al(npx)). 1-cell shift bug (iter-899) kept as production default.
        # al(npx-1) = c1*q1(npx-3)+c2*q1(npx-2)+c3*q1(npx-1)
        face_alnm1 = (c1 * q_pad[..., n_int + 1]
                      + c2 * q_pad[..., n_int + 2]
                      + c3 * q_pad[..., n_int + 3])
        # al(npx) = xt clipped to min/max(q1(npx-2..npx+1)) = q_pad[n+2..n+5]
        xt_R = (0.75 * (q_pad[..., n_int + 3] + q_pad[..., n_int + 4])
                - 0.25 * (q_pad[..., n_int + 2] + q_pad[..., n_int + 5]))
        q_lo_R = jnp.minimum(
            jnp.minimum(q_pad[..., n_int + 2], q_pad[..., n_int + 3]),
            jnp.minimum(q_pad[..., n_int + 4], q_pad[..., n_int + 5]))
        q_hi_R = jnp.maximum(
            jnp.maximum(q_pad[..., n_int + 2], q_pad[..., n_int + 3]),
            jnp.maximum(q_pad[..., n_int + 4], q_pad[..., n_int + 5]))
        face_aln = jnp.clip(xt_R, q_lo_R, q_hi_R)

        q_face = q_face.at[..., n_int + 1].set(face_alnm1)
        q_face = q_face.at[..., n_int + 2].set(face_aln)

    # Left and right face values for each cell
    q_L = q_face[..., :-1]  # face at i-1/2 → left face of cell i
    q_R = q_face[..., 1:]   # face at i+1/2 → right face of cell i

    # --- Monotonicity constraints (Colella & Woodward 1984, eq. 1.10) ---
    # 1. Detect local extrema: if (q_R - q)(q - q_L) <= 0, flatten
    delta = (q_R - q) * (q - q_L)
    is_extremum = delta <= 0.0
    q_L = jnp.where(is_extremum, q, q_L)
    q_R = jnp.where(is_extremum, q, q_R)

    # 2. Overshoot limiting (CW84 eq. 1.10). iter-878: restored ``dq`` factor on LHS
    # to match Fortran pert_ppm (tp_core.F90:1199-1205) ``a6da = 3*(al+ar)*da1`` vs ``da2 = da1²``.
    q_6 = 6.0 * (q - 0.5 * (q_L + q_R))
    dq = q_R - q_L
    q6_dq = q_6 * dq
    dq_sq = dq * dq
    cond_L = q6_dq > dq_sq
    q_L_adj = 3.0 * q - 2.0 * q_R
    q_L = jnp.where(cond_L & ~is_extremum, q_L_adj, q_L)
    cond_R = q6_dq < -dq_sq
    q_R_adj = 3.0 * q - 2.0 * q_L
    q_R = jnp.where(cond_R & ~is_extremum, q_R_adj, q_R)

    # Restore the original axis ordering if the caller passed a non-
    # default `axis`.
    if moved:
        q_L = jnp.moveaxis(q_L, -1, orig_axis)
        q_R = jnp.moveaxis(q_R, -1, orig_axis)

    return q_L, q_R


# ==============================================================================
# Cell-centre <-> D-grid corner vector conversion
# ==============================================================================

def _a2b_ord4_corner_from_padded(f_pad, n):
    """FV3_3D iter 696: 4th-order A→B cc→corner cascade from already-padded array.

    Accepts padded array of shape ``(6, n+4, n+4)`` (3-D) or
    ``(6, n+4, n+4, nlev)`` (4-D).  Same stencil as
    :func:`interp_center_to_corner_a2b_ord4`, but lifted out so it
    can be reused with a pre-rotated vector halo (halo=2).

    Returns corner array of shape ``(6, n+1, n+1[, nlev])``.
    """
    a1 = 9.0 / 16.0
    a2 = -1.0 / 16.0
    b1 = 7.0 / 12.0
    b2 = -1.0 / 12.0
    if f_pad.ndim == 4:
        qx = (b2 * (f_pad[:, 0:n + 1, :, :] + f_pad[:, 3:n + 4, :, :])
              + b1 * (f_pad[:, 1:n + 2, :, :] + f_pad[:, 2:n + 3, :, :]))
        qy = (b2 * (f_pad[:, :, 0:n + 1, :] + f_pad[:, :, 3:n + 4, :])
              + b1 * (f_pad[:, :, 1:n + 2, :] + f_pad[:, :, 2:n + 3, :]))
        qxx = (a2 * (qx[:, :, 0:n + 1, :] + qx[:, :, 3:n + 4, :])
               + a1 * (qx[:, :, 1:n + 2, :] + qx[:, :, 2:n + 3, :]))
        qyy = (a2 * (qy[:, 0:n + 1, :, :] + qy[:, 3:n + 4, :, :])
               + a1 * (qy[:, 1:n + 2, :, :] + qy[:, 2:n + 3, :, :]))
    else:
        qx = (b2 * (f_pad[:, 0:n + 1, :] + f_pad[:, 3:n + 4, :])
              + b1 * (f_pad[:, 1:n + 2, :] + f_pad[:, 2:n + 3, :]))
        qy = (b2 * (f_pad[:, :, 0:n + 1] + f_pad[:, :, 3:n + 4])
              + b1 * (f_pad[:, :, 1:n + 2] + f_pad[:, :, 2:n + 3]))
        qxx = (a2 * (qx[:, :, 0:n + 1] + qx[:, :, 3:n + 4])
               + a1 * (qx[:, :, 1:n + 2] + qx[:, :, 2:n + 3]))
        qyy = (a2 * (qy[:, 0:n + 1, :] + qy[:, 3:n + 4, :])
               + a1 * (qy[:, 1:n + 2, :] + qy[:, 2:n + 3, :]))
    return 0.5 * (qxx + qyy)


def center_to_dgrid_vector(u_cc, v_cc, cdgrid, use_fv3_a2b_ord4: bool = False):
    """Interpolate cell-centre wind vectors to D-grid corners.

    Default (``use_fv3_a2b_ord4=False``): halo=1 vector pad + 4-pt
    arithmetic average (2nd-order).

    FV3_3D iter 696 path (``use_fv3_a2b_ord4=True``): halo=2 vector
    pad with cross-face rotation, then 4th-order PPM-volume +
    Lagrange cascade (Fortran-faithful A→B via
    ``a2b_edge.F90:a2b_ord4`` duogrid path).  Use to close the last
    documented PE-vs-NH FV3-fidelity asymmetry on the NH path's
    cell-centre → D-corner u, v lift (suspect for the residual
    cube-imprint floor at C24-C32).
    """
    grid = cdgrid.base
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    offsets_h2 = (None if dg is not None
                  else getattr(grid, "halo_interp_offsets_h2", None))

    if u_cc.ndim == 3:
        if use_fv3_a2b_ord4:
            u_pad, v_pad = pad_halo_vector(
                u_cc, v_cc,
                grid.cos_angle, grid.sin_angle,
                grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
                interp_offsets=offsets_h2, halo=2, duogrid=dg,
            )
            n = u_cc.shape[1]
            u_d = _a2b_ord4_corner_from_padded(u_pad, n)
            v_d = _a2b_ord4_corner_from_padded(v_pad, n)
            return u_d, v_d
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

    # 4D
    if use_fv3_a2b_ord4:
        u_pad, v_pad = pad_halo_vector_4d(
            u_cc, v_cc,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
            interp_offsets=offsets_h2, halo=2, duogrid=dg,
        )
        n = u_cc.shape[1]
        u_d = _a2b_ord4_corner_from_padded(u_pad, n)
        v_d = _a2b_ord4_corner_from_padded(v_pad, n)
        return u_d, v_d
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
    """D-grid corner → cc (4-pt avg). No cross-face rotation (same basis within face)."""
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
    """D-grid corner → cc east/north (geographic). Rotate BEFORE averaging.

    Averaging in face-local then rotating gives spurious v_north (0.85 m/s at C16 for solid-body).
    Handles 2D ((6,n+1,n+1)) and 3D ((6,n+1,n+1,nlev)) D-grid arrays — the corner
    rotation matrices are 2D (per-face), broadcast over the trailing level axis.
    """
    ca = cdgrid.cos_angle_corner  # (6, n+1, n+1)
    sa = cdgrid.sin_angle_corner  # (6, n+1, n+1)

    if u_d.ndim == 4:
        # 3D D-grid: broadcast 2D angles over the trailing level axis.
        ca = ca[..., None]
        sa = sa[..., None]

    # Rotate to geographic at each corner
    ue = ca * u_d - sa * v_d  # u_east at corners
    vn = sa * u_d + ca * v_d  # v_north at corners

    # Average geographic winds to cell centres (works for 2D or 3D last-spatial-dim slice).
    if u_d.ndim == 3:
        u_east = 0.25 * (ue[:, :-1, :-1] + ue[:, 1:, :-1]
                          + ue[:, :-1, 1:] + ue[:, 1:, 1:])
        v_north = 0.25 * (vn[:, :-1, :-1] + vn[:, 1:, :-1]
                           + vn[:, :-1, 1:] + vn[:, 1:, 1:])
    else:
        u_east = 0.25 * (ue[:, :-1, :-1, :] + ue[:, 1:, :-1, :]
                          + ue[:, :-1, 1:, :] + ue[:, 1:, 1:, :])
        v_north = 0.25 * (vn[:, :-1, :-1, :] + vn[:, 1:, :-1, :]
                           + vn[:, :-1, 1:, :] + vn[:, 1:, 1:, :])
    return u_east, v_north


def cubed_to_latlon(u_d, v_d, cdgrid):
    """FV3_3D iter 607: alias for ``dgrid_to_center_geographic``.

    Matches FV3 ``cubed_to_latlon`` (fv_grid_utils.F90:2386) naming
    so users porting FV3 code find the expected entry point.

    Faithful to FV3 c2l_ord2 (line 2547) semantics: D-grid (u, v)
    on edges → cell-center (ua, va) in geographic (east, north)
    frame.  FV3's ``c2l_ord=2`` (2nd order) is the default; ord=4
    (covariant-to-latlon via a11/a12/a21/a22 matrix) is the more
    accurate variant per ``c2l_ord4`` at fv_grid_utils.F90:2407 —
    legoESM's iter-326 ``cos_angle_corner``/``sin_angle_corner``
    rotation matches the c2l_ord4 metric semantics for the
    cubed-sphere grid_type<4 branch.

    Parameters
    ----------
    u_d, v_d : jax.Array
        D-grid winds (shape (6, n+1, n+1) or (6, n+1, n+1, nlev)).
    cdgrid : CubedSphereCDGrid
        Provides cos_angle_corner / sin_angle_corner.

    Returns
    -------
    ua, va : jax.Array
        Cell-centered geographic winds (east, north).
    """
    return dgrid_to_center_geographic(u_d, v_d, cdgrid)


# ==============================================================================
# D-grid -> C-grid interpolation (d2a2c)
# ==============================================================================

def dgrid_to_cgrid_core(u_d, v_d, cosa_u_metric):
    """Pure-array core of :func:`dgrid_to_cgrid` — D-grid corner -> C-grid
    edge-normal with the ``cosa_u`` metric passed explicitly.  Leading-axis-
    agnostic and PURELY LOCAL (x-face = j-avg of adjacent corners + the face-
    normal projection; y-face = i-avg), so it is shared by the global wrapper
    AND the sub-face tile kernel
    (``legoesm.parallel.tiled_production_cdgrid.dgrid_to_cgrid_tile_2d``) — no
    halo, no cross-face rotation (within-face projection).  Shapes (``...``
    leading + optional trailing nlev): ``u_d``/``v_d`` ``(F, A+1, B+1)`` corners;
    ``cosa_u_metric`` ``(F, A+1, B)``; returns ``u_c`` ``(F, A+1, B)`` (x-face) +
    ``v_c`` ``(F, A, B+1)`` (y-face)."""
    _u = u_d.data if hasattr(u_d, 'data') else u_d
    _v = v_d.data if hasattr(v_d, 'data') else v_d
    u_avg = 0.5 * (_u[:, :, :-1] + _u[:, :, 1:])    # (6, n+1, n[, nlev])
    v_avg_x = 0.5 * (_v[:, :, :-1] + _v[:, :, 1:])  # (6, n+1, n[, nlev])
    cosa_u = _broadcast_metric(cosa_u_metric, u_avg)
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, _EPS))
    u_c = u_avg * sina_u - v_avg_x * cosa_u

    # y-face (at constant j): e_perp is the outward normal, so v_c = v_d
    v_c = 0.5 * (_v[:, :-1, :] + _v[:, 1:, :])      # (6, n, n+1[, nlev])
    return u_c, v_c


def dgrid_to_cgrid(u_d, v_d, cdgrid):
    """D-grid corner → C-grid edge-normal. u_c = u_d*sin(α) - v_d*cos(α); v_c = v_d (e_perp = y-normal).

    Thin wrapper over :func:`dgrid_to_cgrid_core` (the tile kernel reuses the
    core with a per-tile ``cosa_u`` slice — no dup numerics)."""
    return dgrid_to_cgrid_core(u_d, v_d, cdgrid.cosa_u)


def cgrid_to_dgrid(u_c, v_c, cdgrid):
    """C-grid edge → D-grid corner (inverse of dgrid_to_cgrid). u_d = (u_c + v_d*cos(α))/sin(α)."""
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

def dgrid_vorticity_core(u_d, v_d, cosa_corner, dx_edge_y, dy_edge_x, area):
    """Pure-array core of :func:`dgrid_vorticity` — relative vorticity at cc
    from D-grid corner winds via exact circulation, with the RAW metrics passed
    explicitly (``cosa_corner``/``dx_edge_y``/``dy_edge_x``/``area``).  Shared by
    the global wrapper and the sub-face tile kernel
    (:func:`legoesm.parallel.tiled_production_cdgrid.dgrid_vorticity_tile_2d`)
    so the circulation numerics are NOT duplicated.  Purely local: cc cell
    ``(i,j)`` reads only the 2x2 corner block ``[i:i+2, j:j+2]``.

    Non-orthogonality: j-edges use v.e_j = u_d*cos(α) + v_d*sin(α).
    """
    cosa = _broadcast_metric(cosa_corner, u_d)
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

    dx_south = _broadcast_metric(dx_edge_y[:, :, :-1], u_d)
    dx_north = _broadcast_metric(dx_edge_y[:, :, 1:], u_d)
    dy_west = _broadcast_metric(dy_edge_x[:, :-1, :], u_d)
    dy_east = _broadcast_metric(dy_edge_x[:, 1:, :], u_d)

    circ = (u_south * dx_south + v_cov_east * dy_east
            - u_north * dx_north - v_cov_west * dy_west)

    area_b = _broadcast_metric(area, u_d)
    return circ / area_b


def dgrid_vorticity(u_d, v_d, cdgrid):
    """Relative vorticity at cc from D-grid corners via exact circulation
    (avoids Hollingsworth-Kallberg).  Thin wrapper over
    :func:`dgrid_vorticity_core` with the cdgrid metrics."""
    return dgrid_vorticity_core(
        u_d, v_d, cdgrid.cosa_corner, cdgrid.dx_edge_y, cdgrid.dy_edge_x,
        cdgrid.base.area)


# ==============================================================================
# C-grid divergence
# ==============================================================================

def cgrid_divergence_local(u_c, v_c, dy_edge_x, dx_edge_y, area):
    """Exact flux-form divergence — leading-axis-agnostic CORE.

    Operates on whatever leading structure the caller supplies: the
    global ``(6, n, n)`` cube (``cgrid_divergence``) OR a SINGLE
    ``(nl, nl)`` tile inside the tiled ``shard_map`` stage (P4
    phase-1b), where the staggered ``u_c (nl+1, nl)`` / ``v_c (nl,
    nl+1)`` blocks already carry the tile's boundary faces (duplicated
    shared face, Pace layout) so NO halo exchange is needed — flux-form
    divergence reads only a cell's own four surrounding faces.

    Shapes (``...`` = leading axes; last two are horizontal, trailing
    optional vertical):
      u_c        (..., A+1, B[, nlev])   x-face normal velocity
      v_c        (..., A,   B+1[, nlev]) y-face normal velocity
      dy_edge_x  (..., A+1, B)   x-face length
      dx_edge_y  (..., A,   B+1) y-face length
      area       (..., A,   B)   cell area
    Returns div (..., A, B[, nlev]).
    """
    dy = _broadcast_metric(dy_edge_x, u_c)
    dx = _broadcast_metric(dx_edge_y, v_c)
    flux_x = u_c * dy
    flux_y = v_c * dx
    net_x = flux_x[:, 1:] - flux_x[:, :-1]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]
    area_b = _broadcast_metric(area, net_x)
    return (net_x + net_y) / area_b


def cgrid_divergence(u_c, v_c, cdgrid):
    """Exact flux-form divergence at cell centres. 2D and 3D.

    Thin wrapper over :func:`cgrid_divergence_local` (one shared body —
    the tiled stage calls the local core with per-tile metrics)."""
    return cgrid_divergence_local(
        u_c, v_c, cdgrid.dy_edge_x, cdgrid.dx_edge_y, cdgrid.base.area)


def cgrid_wet_face_masks(wet_cc, cdgrid):
    """C-grid face wet/dry masks from a cell-centre wet mask.

    A C-grid face is wet iff BOTH adjacent cell centres are wet, so the
    flux-form mass/tracer divergence carries ZERO transport across a wet/dry
    interface (coastline OR partial-cell seafloor step).  This STRICTLY closes
    the face, rather than relying on a zeroed cell-centre velocity that the
    d->c average can leave nonzero at the interface (the documented
    ocean_pe_cdgrid seafloor/coastline leak).  Halo-correct across cube-face
    seams: it uses the same duogrid halo (``pad_halo_auto``) as the
    gradient/divergence operators, so a face on a panel edge sees the true
    neighbouring-panel wet state, not a zero-padded ghost.

    Multiply ``u_c`` by ``mask_u`` and ``v_c`` by ``mask_v`` BEFORE the mass
    flux divergence, the velocity divergence, and the tracer advection — all
    three consume the C-grid velocities, so one masking point closes every
    flux.

    Parameters
    ----------
    wet_cc : array, shape (6, n, n) or (6, n, n, nlev)
        Cell-centre wet mask (1.0 wet, 0.0 dry).  For a partial-cell column
        pass ``land_mask * is_active`` so both coastline and below-seafloor
        cells are dry.
    cdgrid : cubed-sphere C-D grid.

    Returns
    -------
    mask_u : array, shape (6, n+1, n[, nlev])  -- x-face (u-point) wet mask
    mask_v : array, shape (6, n, n+1[, nlev])  -- y-face (v-point) wet mask
    """
    wet_pad = pad_halo_auto(wet_cc, cdgrid)  # halo=1, duogrid-synced
    # x-faces between padded cells (i-1, i); y-faces between (j-1, j).
    # Same index convention as ``cgrid_gradient_2d``.
    mask_u = wet_pad[:, 1:, 1:-1] * wet_pad[:, :-1, 1:-1]
    mask_v = wet_pad[:, 1:-1, 1:] * wet_pad[:, 1:-1, :-1]
    return mask_u, mask_v


# ==============================================================================
# C-grid compact gradient (cell centre → edge midpoints)
# ==============================================================================

def cgrid_gradient_2d_local(eta_pad, rdxc, rdyc):
    """Compact C-grid gradient CORE — leading-axis-agnostic.

    Takes an ALREADY-halo-padded ``eta_pad`` (..., A+2, B+2) and the
    inverse edge lengths; differences cc -> edge midpoints.  Shared by
    the global cube (:func:`cgrid_gradient_2d`, which pads the whole
    (6,n,n) then calls this) and a single (1, nl+2, nl+2) tile inside
    the tiled ``shard_map`` stage (P4 phase-1b), where ``eta_pad`` is
    the tile's padded block from ``make_tiled_pad_body`` (the halo is a
    scalar pad — eta is a scalar field, so no rotation).

      eta_pad (..., A+2, B+2)
      rdxc    (..., A+1, B)   1/dxc at u-faces
      rdyc    (..., A,   B+1) 1/dyc at v-faces
    Returns ``(deta_dx (..., A+1, B), deta_dy (..., A, B+1))``.
    """
    deta_dx = (eta_pad[:, 1:, 1:-1] - eta_pad[:, :-1, 1:-1]) * rdxc
    deta_dy = (eta_pad[:, 1:-1, 1:] - eta_pad[:, 1:-1, :-1]) * rdyc
    return deta_dx, deta_dy


def cgrid_gradient_2d(eta, cdgrid):
    """Compact C-grid gradient cc → edge midpoints (FV3 Bernoulli stencil using dxc/dyc).

    Thin wrapper over :func:`cgrid_gradient_2d_local` (one shared body —
    the tiled stage calls the local core with per-tile metrics)."""
    eta_pad = pad_halo_auto(eta, cdgrid)  # (6, n+2, n+2)
    return cgrid_gradient_2d_local(eta_pad, cdgrid.rdxc, cdgrid.rdyc)


def cgrid_flux_divergence_sync(h_u, h_v, u_c, v_c, cdgrid):
    """Flux-form divergence div(h·v) from face-interpolated ``h``, seam-exact.

    Non-duogrid: identical to ``cgrid_divergence(h_u*u_c, h_v*v_c, cdgrid)``
    (each panel's seam flux uses its own interpolated halo — the seam
    mismatch is the cross-face interpolation error).  Duogrid (ng>=2): the
    boundary fluxes of adjacent panels are averaged to a SINGLE shared value
    via ``synchronize_cgrid_fluxes`` (same gating as
    :func:`cgrid_mass_flux_divergence`), so the global area integral of the
    divergence telescopes to machine zero — the PE flux-form continuity
    relies on this for exact dry-mass conservation.  Returns +div (positive
    = mass export).
    """
    dg = cdgrid.base.duogrid
    if dg is None or dg.ng < 2:
        return cgrid_divergence(h_u * u_c, h_v * v_c, cdgrid)
    # Duogrid: build the raw face fluxes, average the shared panel-boundary
    # fluxes, then difference — mirrors cgrid_mass_flux_divergence's PPM path.
    dy = _broadcast_metric(cdgrid.dy_edge_x, u_c)
    dx = _broadcast_metric(cdgrid.dx_edge_y, v_c)
    flux_x = (h_u * u_c) * dy
    flux_y = (h_v * v_c) * dx
    flux_x, flux_y = synchronize_cgrid_fluxes(flux_x, flux_y, cdgrid.n)
    net_x = flux_x[:, 1:] - flux_x[:, :-1]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]
    return (net_x + net_y) / _broadcast_metric(cdgrid.base.area, net_x)


def cgrid_interp_cc_to_faces_local(f_pad):
    """Cell-centre → C-grid face 2-point average — leading-axis-agnostic CORE.

    Takes an ALREADY-halo-padded cc field ``f_pad`` (..., A+2, B+2[, nlev])
    and averages adjacent cells onto the C-grid faces (the same face/index
    convention as :func:`cgrid_gradient_2d_local`, with the difference
    replaced by the mean).  Cube analogue of the lat-lon C-grid
    ``interp_cell_to_uface`` / ``interp_cell_to_vface_halo`` pair — used by
    the PE flux-form continuity to put the layer thickness dp on the faces
    the divergence operator consumes.  Shared by the global dycore (padded
    via ``pad_halo_auto`` / the packed stage halo) and the tiled shard_map
    stage (padded via the in-stage ``make_tiled_pad_body``), so the stencil
    lives in ONE place.

    Returns ``(f_u (..., A+1, B[, nlev]), f_v (..., A, B+1[, nlev]))``.
    """
    f_u = 0.5 * (f_pad[:, 1:, 1:-1] + f_pad[:, :-1, 1:-1])
    f_v = 0.5 * (f_pad[:, 1:-1, 1:] + f_pad[:, 1:-1, :-1])
    return f_u, f_v


# ==============================================================================
# C-grid mass flux with PPM transport
# ==============================================================================

def cgrid_ppm_fluxes_core(
    h_pad, u_c, v_c, dy, dx, n_local, *, halo_in=2,
    effective_xppm_boundary=False,
):
    """Pure-array PPM upwind C-grid fluxes from an ALREADY-padded ``h``.

    Operates on 3D ``(F, X, Y)`` blocks (F = the 6-face leading axis on the
    global path, 1 per device in the tiled stage, or the vmapped 6-face block
    per level on the 4D path).  Core shared by the global wrapper
    (:func:`_cgrid_ppm_fluxes_2d_no_sync`, ``halo_in=2``, the production
    halo=2 pad) and the sub-face tile kernel
    (:func:`legoesm.parallel.tiled_production_cdgrid.cgrid_mass_divergence_tile_2d`,
    ``halo_in=3`` — one deeper ring) so the PPM reconstruction + upwind
    numerics are NOT duplicated.

    ``halo_in`` is the cell-halo depth of ``h_pad`` on EACH axis
    (``h_pad`` is ``(..., n_local+2*halo_in, n_local+2*halo_in)``).  The
    used reconstruction faces are ``q_R[..., halo_in-1 : halo_in-1+n_local+1]``
    (left-of-face cell's right value) and ``q_L[..., halo_in : ...+1]``
    (right-of-face cell's left value), which collapse to the historical
    ``[1:n+2]`` / ``[2:n+3]`` at ``halo_in=2`` — bit-identical to the
    pre-factor global path.

    Why the tile needs ``halo_in=3`` while the global gets away with
    ``halo_in=2``: PPM reconstruction of the boundary cell (local ``-1``)
    reads cells ``[-3..1]``; in the GLOBAL face that ``-3`` is the cube
    edge (the internal ``mode='edge'`` pad supplies it, as on a real
    boundary), but at an INTERIOR tile cut ``-3`` is a real neighbour
    cell, so the tile must carry it — one ring deeper.  The deeper pad's
    outermost ring is itself edge-extended from the halo=2 pad, so a
    FACE-edge tile reproduces the global's ``mode='edge'`` value exactly.

    Shapes (``...`` leading): ``h_pad (..., L, L)`` with
    ``L=n_local+2*halo_in``; ``u_c (..., n_local+1, n_local)``;
    ``v_c (..., n_local, n_local+1)``; ``dy (..., n_local+1, n_local)``
    (x-face length); ``dx (..., n_local, n_local+1)`` (y-face length).
    Returns ``flux_x (..., n_local+1, n_local)``,
    ``flux_y (..., n_local, n_local+1)``.
    """
    n = n_local
    h = halo_in
    # x-faces: keep the full x-halo, trim y to the cc extent.
    h_x_strips = h_pad[:, :, h:-h]
    q_L_x, q_R_x = _ppm_reconstruct_1d(
        h_x_strips, axis=1,
        apply_fortran_xppm_boundary=effective_xppm_boundary,
        n_interior=n,
    )
    q_R_left = q_R_x[:, h - 1:h - 1 + n + 1, :]
    q_L_right = q_L_x[:, h:h + n + 1, :]
    h_face_x = jnp.where(u_c > 0, q_R_left, q_L_right)

    h_y_strips = h_pad[:, h:-h, :]
    q_L_y, q_R_y = _ppm_reconstruct_1d(
        h_y_strips, axis=2,
        apply_fortran_xppm_boundary=effective_xppm_boundary,
        n_interior=n,
    )
    q_R_bottom = q_R_y[:, :, h - 1:h - 1 + n + 1]
    q_L_top = q_L_y[:, :, h:h + n + 1]
    h_face_y = jnp.where(v_c > 0, q_R_bottom, q_L_top)

    flux_x = h_face_x * u_c * dy
    flux_y = h_face_y * v_c * dx
    return flux_x, flux_y


def _cgrid_ppm_fluxes_2d_no_sync(
    h, u_c, v_c, h_pad, cdgrid,
    apply_fortran_xppm_boundary=False,
):
    """2D PPM flux computation WITHOUT duogrid synchronization.

    Used by the 4D ``cgrid_mass_flux_divergence`` to compute per-level
    fluxes inside ``jax.vmap``.  The 4D entry then synchronizes the
    stacked 4D fluxes (one MPI sendrecv exchange total, vs ``nlev``
    inside vmap which mpi4jax's batch-axis rule refuses).

    Thin wrapper over :func:`cgrid_ppm_fluxes_core` (``halo_in=2``); no
    dup numerics (the tiled stage reuses the core with ``halo_in=3``).
    """
    effective_xppm_boundary = (
        apply_fortran_xppm_boundary
        and not cdgrid.base.bounded_domain
    )
    return cgrid_ppm_fluxes_core(
        h_pad, u_c, v_c, cdgrid.dy_edge_x, cdgrid.dx_edge_y, cdgrid.n,
        halo_in=2,
        effective_xppm_boundary=effective_xppm_boundary,
    )


def cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid,
                                apply_fortran_xppm_boundary=False,
                                h_pad=None):
    """Conservative mass flux divergence via PPM (Colella & Woodward 1984). Halo=2. 2D/3D.

    FV3_3D iter-1042: 3D path pre-pads halos OUTSIDE the per-level
    ``jax.vmap`` and threads the padded array through the
    ``h_pad`` kwarg.  Required for MPI fidelity: ``mpi4jax``'s
    sendrecv batching rule asserts matching batch axes on the
    send/recv buffers, which fails when ``pad_halo`` is invoked
    inside ``vmap``.  Mirrors the pre-existing pattern in
    :func:`_cgrid_fct_fluxes_2d` (pad-once-then-vmap).
    """
    if h.ndim == 4:
        # 3D: pad halos ONCE for all levels (avoids MPI sendrecv inside vmap)
        h_pad_4d = _pad_halo_auto_h2(h, cdgrid)  # (6, n+4, n+4, nlev)
        h_t = jnp.moveaxis(h, -1, 0)
        u_c_t = jnp.moveaxis(u_c, -1, 0)
        v_c_t = jnp.moveaxis(v_c, -1, 0)
        h_pad_t = jnp.moveaxis(h_pad_4d, -1, 0)

        # FV3_3D iter-1049: per-level vmap computes fluxes only (no
        # sync, no divergence) — the duogrid flux synchronization
        # involves an MPI sendrecv that cannot run inside ``vmap``
        # (mpi4jax batch-axis assertion).  Lift sync + divergence to
        # the 4D level after the vmap.
        def flux_per_level(args):
            hk, uk, vk, hk_pad = args
            return _cgrid_ppm_fluxes_2d_no_sync(
                hk, uk, vk, hk_pad, cdgrid,
                apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
            )

        flux_x_t, flux_y_t = jax.vmap(flux_per_level)(
            (h_t, u_c_t, v_c_t, h_pad_t)
        )
        # (nlev, 6, n+1, n) → (6, n+1, n, nlev) and (6, n, n+1, nlev)
        flux_x_4d = jnp.moveaxis(flux_x_t, 0, -1)
        flux_y_4d = jnp.moveaxis(flux_y_t, 0, -1)

        # Duogrid flux sync at the 4D level (one MPI exchange total,
        # not nlev of them).  ``synchronize_cgrid_fluxes`` is
        # shape-polymorphic — operates on the trailing nlev axis via
        # broadcasting.
        dg = cdgrid.base.duogrid
        n = cdgrid.n
        if dg is not None and dg.ng >= 2:
            flux_x_4d, flux_y_4d = synchronize_cgrid_fluxes(
                flux_x_4d, flux_y_4d, n,
            )

        net_x_4d = flux_x_4d[:, 1:] - flux_x_4d[:, :-1]
        net_y_4d = flux_y_4d[:, :, 1:] - flux_y_4d[:, :, :-1]
        return -(net_x_4d + net_y_4d) / cdgrid.base.area[..., None]

    # 2D case: PPM face reconstruction with halo=2
    if h_pad is None:
        h_pad = _pad_halo_auto_h2(h, cdgrid)  # (6, n+4, n+4)

    n = cdgrid.n

    # iter-889b: gate Fortran xppm boundary on .not.(bounded_domain.or.duogrid) per tp_core.F90:357
    effective_xppm_boundary = (
        apply_fortran_xppm_boundary
        and not cdgrid.base.bounded_domain)

    # PPM upwind C-grid fluxes via the shared core (halo_in=2 = the
    # production halo=2 pad; the sub-face tile kernel reuses the SAME core
    # with halo_in=3).  Bit-identical to the prior inline x/y reconstruction.
    flux_x, flux_y = cgrid_ppm_fluxes_core(
        h_pad, u_c, v_c, cdgrid.dy_edge_x, cdgrid.dx_edge_y, n,
        halo_in=2,
        effective_xppm_boundary=effective_xppm_boundary)

    # Duogrid flux sync: avg boundary fluxes for mass conservation (FV3 dyn_core.F90:853-900).
    # NOT for non-duogrid: PPM boundary asymmetry is a feature; sync gives 110x W2 regression.
    # FV3_3D iter-1049: ``synchronize_cgrid_fluxes`` reads neighbor face
    # values directly (``fx[nbr_face, ...]``).  Under MPI in replicated
    # mode the non-owned face flux values were computed with zero halo
    # and are WRONG, contaminating owned-face boundary averages.  The
    # MPI-aware variant exchanges boundary flux strips across ranks
    # before averaging, restoring bit-for-bit fidelity vs the local
    # backend on owned faces.
    dg = cdgrid.base.duogrid
    if dg is not None and dg.ng >= 2:
        flux_x, flux_y = synchronize_cgrid_fluxes(flux_x, flux_y, n)

    net_x = flux_x[:, 1:] - flux_x[:, :-1]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]

    return -(net_x + net_y) / cdgrid.base.area


# ==============================================================================
# Scalar advection on C-grid (PPM)
# ==============================================================================

# ==============================================================================
# Flux-corrected transport (FCT) for monotone tracer advection
# ==============================================================================

def _cgrid_fct_fluxes_2d(q, u_c, v_c, cdgrid):
    """FCT-limited tracer fluxes on C-D grid (2-stage monotone transport).

    Stage 1: PPM face-value reconstruction + clip to [min,max] of adjacent cells (cube-edge halo fix).
    Stage 2: 1st-order upwind blending for multidim bound preservation.
    Zalesak (1979) + Colella & Woodward (1984) + Lin (2004).
    """
    n = cdgrid.n
    area = cdgrid.base.area   # (6, n, n)
    dy = cdgrid.dy_edge_x     # (6, n+1, n)
    dx = cdgrid.dx_edge_y     # (6, n, n+1)

    # Halo-padded fields (halo=1 for upwind, halo=2 for PPM)
    q_pad_full = pad_halo_auto(q, cdgrid)        # (6, n+2, n+2[, nlev])
    q_pad_h2_full = _pad_halo_auto_h2(q, cdgrid)  # (6, n+4, n+4[, nlev])

    # 4D: move nlev to leading so spatial slicing uses [..., (i,j)]
    is_4d = q.ndim == 4
    if is_4d:
        q_t = jnp.moveaxis(q, -1, 0)
        q_pad = jnp.moveaxis(q_pad_full, -1, 0)
        q_pad_h2 = jnp.moveaxis(q_pad_h2_full, -1, 0)
        u_c_t = jnp.moveaxis(u_c, -1, 0)
        v_c_t = jnp.moveaxis(v_c, -1, 0)
    else:
        q_t = q
        q_pad = q_pad_full
        q_pad_h2 = q_pad_h2_full
        u_c_t = u_c
        v_c_t = v_c

    # ----------------------------------------------------------------
    # Step 1: First-order upwind fluxes (inherently monotone for CFL<1)
    # ----------------------------------------------------------------
    q_left_x = q_pad[..., :-1, 1:-1]
    q_right_x = q_pad[..., 1:, 1:-1]
    q_face_low_x = jnp.where(u_c_t > 0, q_left_x, q_right_x)

    q_below_y = q_pad[..., 1:-1, :-1]
    q_above_y = q_pad[..., 1:-1, 1:]
    q_face_low_y = jnp.where(v_c_t > 0, q_below_y, q_above_y)

    flux_low_x = q_face_low_x * u_c_t * dy
    flux_low_y = q_face_low_y * v_c_t * dx

    net_low_x = flux_low_x[..., 1:, :] - flux_low_x[..., :-1, :]
    net_low_y = flux_low_y[..., 1:] - flux_low_y[..., :-1]
    dq_low = -(net_low_x + net_low_y) / area

    # ----------------------------------------------------------------
    # Step 2: PPM face values with face-value clipping
    # ----------------------------------------------------------------
    # NOTE: ``q_pad_h2`` was assigned above (and moveaxis'd for 4D
    # inputs).  Do *not* re-pad here — that would discard the leading
    # ``nlev`` axis and break the 4D path with a shape mismatch when
    # the clip step (line ~960) compares against ``q_left_x`` /
    # ``q_right_x`` (which use the rotated ``q_pad``).

    # X-direction PPM — strips are taken along the trailing ``(i, j)``
    # axes regardless of rank.  Use negative axes so the helper acts on
    # the correct PPM (i) axis whether ``q`` is 3D ``(6, ny, nx)`` or
    # 4D moved to ``(nlev, 6, ny, nx)``.
    q_x_strips = q_pad_h2[..., :, 2:-2]                 # (..., n+4, n)
    q_L_x, q_R_x = _ppm_reconstruct_1d(q_x_strips, axis=-2)
    q_R_left = q_R_x[..., 1:n+2, :]
    q_L_right = q_L_x[..., 2:n+3, :]
    q_face_hi_x = jnp.where(u_c_t > 0, q_R_left, q_L_right)

    # Clip to local bounds of adjacent cells
    q_face_min_x = jnp.minimum(q_left_x, q_right_x)
    q_face_max_x = jnp.maximum(q_left_x, q_right_x)
    q_face_hi_x = jnp.clip(q_face_hi_x, q_face_min_x, q_face_max_x)

    # Y-direction PPM — strip shape (..., n, n+4) puts the halo-padded
    # j-axis at axis=-1.
    q_y_strips = q_pad_h2[..., 2:-2, :]
    q_L_y, q_R_y = _ppm_reconstruct_1d(q_y_strips, axis=-1)
    q_R_bottom = q_R_y[..., :, 1:n+2]
    q_L_top = q_L_y[..., :, 2:n+3]
    q_face_hi_y = jnp.where(v_c_t > 0, q_R_bottom, q_L_top)

    # Clip to local bounds of adjacent cells
    q_face_min_y = jnp.minimum(q_below_y, q_above_y)
    q_face_max_y = jnp.maximum(q_below_y, q_above_y)
    q_face_hi_y = jnp.clip(q_face_hi_y, q_face_min_y, q_face_max_y)

    flux_hi_x = q_face_hi_x * u_c_t * dy
    flux_hi_y = q_face_hi_y * v_c_t * dx

    net_hi_x = flux_hi_x[..., 1:, :] - flux_hi_x[..., :-1, :]
    net_hi_y = flux_hi_y[..., 1:] - flux_hi_y[..., :-1]
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
    q_min = q_t
    q_max = q_t
    q_min = jnp.minimum(q_min, q_pad[..., :-2, 1:-1])   # west
    q_min = jnp.minimum(q_min, q_pad[..., 2:, 1:-1])    # east
    q_min = jnp.minimum(q_min, q_pad[..., 1:-1, :-2])   # south
    q_min = jnp.minimum(q_min, q_pad[..., 1:-1, 2:])    # north
    q_max = jnp.maximum(q_max, q_pad[..., :-2, 1:-1])
    q_max = jnp.maximum(q_max, q_pad[..., 2:, 1:-1])
    q_max = jnp.maximum(q_max, q_pad[..., 1:-1, :-2])
    q_max = jnp.maximum(q_max, q_pad[..., 1:-1, 2:])

    # FCT blending: cap anti-diffusion to room_up/room_dn
    q_td = q_t + dq_low   # provisional low-order

    room_up = q_max - q_td
    room_dn = q_td - q_min

    # alpha ∈ [0,1]: ad>0 → room_up/ad, ad<0 → room_dn/|ad|, ad≈0 → 1
    # Safe denominators (eps) for grad-safety: jnp.where evaluates both branches
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

    result = dq_low + alpha * ad
    if is_4d:
        return jnp.moveaxis(result, 0, -1)
    return result


def _make_fct_2d_differentiable(cdgrid):
    """Differentiable FCT closure: forward uses FCT limiter; JVP linearizes through unlimited PPM."""

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
    """Monotone PPM tracer advection with local-bounds face clipping. 2D/3D, conservative, differentiable."""
    fct_fn = _make_fct_2d_differentiable(cdgrid)
    return fct_fn(q, u_c, v_c)


# ==============================================================================
# Arakawa-Lamb gradient at D-grid corners
# ==============================================================================

def _al_grad_matrix(dB_raw_x, dB_raw_y, grad_c00, grad_c01, grad_c10, grad_c11):
    """Apply the precomputed 2x2 (3D Cartesian -> face-local) A-L gradient
    matrix to the raw 4-pt finite differences.  Shared by every
    arakawa_lamb_gradient branch + the sub-face tile kernel (no dup numerics)."""
    c00 = _broadcast_metric(grad_c00, dB_raw_x)
    c01 = _broadcast_metric(grad_c01, dB_raw_x)
    c10 = _broadcast_metric(grad_c10, dB_raw_x)
    c11 = _broadcast_metric(grad_c11, dB_raw_x)
    return c00 * dB_raw_x + c01 * dB_raw_y, c10 * dB_raw_x + c11 * dB_raw_y


def arakawa_lamb_gradient_core(B_pad, grad_c00, grad_c01, grad_c10, grad_c11):
    """Default-path Arakawa-Lamb corner gradient (NO cube-vertex specials): the
    2x2 box 4-pt finite-diff + :func:`_al_grad_matrix`.  ``B_pad[:, :-1, :-1]``
    keeps the trailing axis so this is ndim-agnostic (3D + 4D — bit-identical to
    the old explicit ``B.ndim`` branch).  Shared by the global wrapper
    and the sub-face tile kernel
    (``tiled_production_cdgrid.arakawa_lamb_gradient_tile_2d``)."""
    B_sw = B_pad[:, :-1, :-1]
    B_se = B_pad[:, 1:, :-1]
    B_nw = B_pad[:, :-1, 1:]
    B_ne = B_pad[:, 1:, 1:]
    dB_raw_x = (B_se + B_ne) - (B_sw + B_nw)   # east - west
    dB_raw_y = (B_nw + B_ne) - (B_sw + B_se)   # north - south
    return _al_grad_matrix(dB_raw_x, dB_raw_y, grad_c00, grad_c01, grad_c10,
                           grad_c11)


def arakawa_lamb_gradient(B, cdgrid, padded=None,
                           fortran_dir_aware_corners=False):
    """4-pt Arakawa-Lamb gradient at D-grid corners via precomputed 3D Cartesian matrix.

    Returns (dB/dx, dB/dy_perp) in face-local basis; handles non-orthogonality at cube vertices.
    fortran_dir_aware_corners (iter-765): dir=1 (x-grad) / dir=2 (y-grad) inner fills at cube vertices.
    """
    B_pad = padded if padded is not None else pad_halo_auto(B, cdgrid)

    if fortran_dir_aware_corners:
        # Two padded variants at 4 cube vertices: dir1 (x-grad: i=0-col, one-j-inward);
        # dir2 (y-grad: j=0-row, one-i-inward). Naming per halo.py::fill_corners_h1.
        p = B_pad
        if B.ndim == 3:
            p1 = p.at[:, 0, 0].set(p[:, 0, 1])
            p1 = p1.at[:, 0, -1].set(p[:, 0, -2])
            p1 = p1.at[:, -1, 0].set(p[:, -1, 1])
            p1 = p1.at[:, -1, -1].set(p[:, -1, -2])
            p2 = p.at[:, 0, 0].set(p[:, 1, 0])
            p2 = p2.at[:, 0, -1].set(p[:, 1, -1])
            p2 = p2.at[:, -1, 0].set(p[:, -2, 0])
            p2 = p2.at[:, -1, -1].set(p[:, -2, -1])
            B_sw_x = p1[:, :-1, :-1]; B_se_x = p1[:, 1:, :-1]
            B_nw_x = p1[:, :-1, 1:];  B_ne_x = p1[:, 1:, 1:]
            B_sw_y = p2[:, :-1, :-1]; B_se_y = p2[:, 1:, :-1]
            B_nw_y = p2[:, :-1, 1:];  B_ne_y = p2[:, 1:, 1:]
        else:
            p1 = p.at[:, 0, 0, :].set(p[:, 0, 1, :])
            p1 = p1.at[:, 0, -1, :].set(p[:, 0, -2, :])
            p1 = p1.at[:, -1, 0, :].set(p[:, -1, 1, :])
            p1 = p1.at[:, -1, -1, :].set(p[:, -1, -2, :])
            p2 = p.at[:, 0, 0, :].set(p[:, 1, 0, :])
            p2 = p2.at[:, 0, -1, :].set(p[:, 1, -1, :])
            p2 = p2.at[:, -1, 0, :].set(p[:, -2, 0, :])
            p2 = p2.at[:, -1, -1, :].set(p[:, -2, -1, :])
            B_sw_x = p1[:, :-1, :-1, :]; B_se_x = p1[:, 1:, :-1, :]
            B_nw_x = p1[:, :-1, 1:, :];  B_ne_x = p1[:, 1:, 1:, :]
            B_sw_y = p2[:, :-1, :-1, :]; B_se_y = p2[:, 1:, :-1, :]
            B_nw_y = p2[:, :-1, 1:, :];  B_ne_y = p2[:, 1:, 1:, :]
        dB_raw_x = (B_se_x + B_ne_x) - (B_sw_x + B_nw_x)
        dB_raw_y = (B_nw_y + B_ne_y) - (B_sw_y + B_se_y)
        return _al_grad_matrix(
            dB_raw_x, dB_raw_y, cdgrid.grad_c00, cdgrid.grad_c01,
            cdgrid.grad_c10, cdgrid.grad_c11)

    # Default path (no cube-vertex specials).
    return arakawa_lamb_gradient_core(
        B_pad, cdgrid.grad_c00, cdgrid.grad_c01, cdgrid.grad_c10,
        cdgrid.grad_c11)


# ==============================================================================
# Interpolation helpers
# ==============================================================================

def interp_center_to_corner(field, cdgrid, padded=None):
    """Cell-centre → D-grid corner (4-pt avg). 2D/3D. padded= skips internal halo (stage-pack)."""
    f_pad = padded if padded is not None else pad_halo_auto(field, cdgrid)

    if field.ndim == 3:
        return 0.25 * (f_pad[:, :-1, :-1] + f_pad[:, 1:, :-1]
                        + f_pad[:, :-1, 1:] + f_pad[:, 1:, 1:])
    return 0.25 * (f_pad[:, :-1, :-1, :] + f_pad[:, 1:, :-1, :]
                    + f_pad[:, :-1, 1:, :] + f_pad[:, 1:, 1:, :])


def cgrid_corner_min(field, cdgrid, padded=None):
    """Cell-centre → D-grid corner via 4-point MIN (not average).

    Returns, at each D-grid corner, the minimum of the four surrounding
    cell-centre values — using the SAME four cells and halo convention as
    :func:`interp_center_to_corner` and the default branch of
    :func:`arakawa_lamb_gradient` (``B_pad[:-1,:-1]``, ``[1:,:-1]``,
    ``[:-1,1:]``, ``[1:,1:]``), so a corner value here is co-located with that
    operator's gradient output.  Halo-correct across cube seams (duogrid pad).

    Used by the Adcroft-Campin partial-cell PGF correction on the cd-grid: the
    corner-common reference depth is the shallowest (minimum) of the four
    surrounding cell centroid depths.  Shape (6, n, n[, nlev]) → (6, n+1, n+1
    [, nlev]).
    """
    f_pad = padded if padded is not None else pad_halo_auto(field, cdgrid)
    if field.ndim == 3:
        return jnp.minimum(
            jnp.minimum(f_pad[:, :-1, :-1], f_pad[:, 1:, :-1]),
            jnp.minimum(f_pad[:, :-1, 1:], f_pad[:, 1:, 1:]),
        )
    return jnp.minimum(
        jnp.minimum(f_pad[:, :-1, :-1, :], f_pad[:, 1:, :-1, :]),
        jnp.minimum(f_pad[:, :-1, 1:, :], f_pad[:, 1:, 1:, :]),
    )


def interp_center_to_corner_a2b_ord4(field, cdgrid):
    """iter-971: 4th-order A→B cc→corner (FV3 a2b_ord4 a2b_edge.F90:50-330 duogrid path).

    Cascaded 4-pt stencils: qx → qxx + qy → qyy → qout = 0.5(qxx+qyy).
    a1=9/16, a2=-1/16 (Lagrange 4-pt); b1=7/12, b2=-1/12 (PPM volume mean).
    Used by iter-959/963 Smag d_sw5 callers (Fortran-faithful vs 2nd-order interp_center_to_corner).
    """
    n = field.shape[1]
    # Halo=2 for 4-pt stencil on both i and j
    f_pad = _pad_halo_auto_h2(field, cdgrid)  # (6, n+4, n+4)
    a1 = 9.0 / 16.0
    a2 = -1.0 / 16.0
    b1 = 7.0 / 12.0
    b2 = -1.0 / 12.0

    # qx at i-face k ∈ [0, n]: cells (k-2..k+1) → padded (k..k+3). qx shape (6, n+1, n+4)
    qx = (b2 * (f_pad[:, 0:n + 1, :] + f_pad[:, 3:n + 4, :])
          + b1 * (f_pad[:, 1:n + 2, :] + f_pad[:, 2:n + 3, :]))

    # qy: symmetric on j-axis. qy shape (6, n+4, n+1)
    qy = (b2 * (f_pad[:, :, 0:n + 1] + f_pad[:, :, 3:n + 4])
          + b1 * (f_pad[:, :, 1:n + 2] + f_pad[:, :, 2:n + 3]))

    # qxx: 4-pt y-stencil on qx.  qx j-axis has (n+4) cells indexed
    # 0..n+3.  For corner j ∈ [0, n] (n+1 corners), the 4-pt stencil
    # reads j-cells (j-2, j-1, j, j+1) → padded indices (j, j+1, j+2,
    # j+3).
    qxx = (a2 * (qx[:, :, 0:n + 1] + qx[:, :, 3:n + 4])
           + a1 * (qx[:, :, 1:n + 2] + qx[:, :, 2:n + 3]))
    # qxx shape: (6, n+1, n+1)

    # qyy: 4-pt x-stencil on qy.
    qyy = (a2 * (qy[:, 0:n + 1, :] + qy[:, 3:n + 4, :])
           + a1 * (qy[:, 1:n + 2, :] + qy[:, 2:n + 3, :]))
    # qyy shape: (6, n+1, n+1)

    return 0.5 * (qxx + qyy)


def interp_corner_to_center(field_d):
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

# ==============================================================================
# Laplacian at D-grid corners
# ==============================================================================

def laplacian_dgrid(u_d, cdgrid):
    """D-grid Laplacian via cc round-trip (uses inter-face halo). 4D shares halo via pad_halo_4d."""
    # 1. D-grid -> cell centres: (6, n+1, n+1[, nlev]) -> (6, n, n[, nlev])
    u_cc = interp_corner_to_center(u_d)

    # 2. Cell-centre Laplacian with proper halo exchange.  Use the
    # native-4D variant on 3D inputs so all levels share one
    # ``pad_halo_4d`` MPI exchange.
    if u_d.ndim == 4:
        lap_a = laplacian_compact_3d(u_cc, cdgrid.base)  # (6, n, n, nlev)
    else:
        lap_a = laplacian_compact(u_cc, cdgrid.base)  # (6, n, n)

    # 3. Cell centres -> D-grid: (6, n, n[, nlev]) -> (6, n+1, n+1[, nlev])
    return interp_center_to_corner(lap_a, cdgrid)


# ==============================================================================
# Vector-invariant momentum tendencies (unified 2D/3D)
# ==============================================================================

def extrapolate_boundary_corners(du, dv, n):
    """Fix tendency at 8 cube vertices via bilinear extrapolation from 3 nearest corners.

    A-L gradient at vertices has O(dx) error (4 stencil cells from different faces).
    tend(0,0) = tend(1,0) + tend(0,1) - tend(1,1) gives O(dx²) accuracy.
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
    g=constants.g, A_h=0.0, hyperdiff_coeff=0.0, div_damp=0.0,
    rho_0=None, div_v=None, f_3d=None,
    u_prime=None, v_prime=None,
    dddmp=0.0,
):
    """D-grid momentum tendencies (vector-invariant). 2D SW: B = KE + g*(h+h_s); 3D PE/ocean: dKE + dp'/rho_0 + f*v'."""
    is_3d = u_d.ndim == 4

    # 1. Relative vorticity at cell centres (just ζ, no f yet)
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)

    # 2. KE at cell centres from D-grid corners (orthogonal basis)
    u_cc, v_cc = dgrid_to_center_vector(u_d, v_d)
    KE = 0.5 * (u_cc ** 2 + v_cc ** 2)

    # Also need C-grid velocities for divergence / mass flux
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)

    # 3. Gradients at corners (Arakawa-Lamb)
    if is_3d:
        dKE_dx, dKE_dy_perp = arakawa_lamb_gradient(KE, cdgrid)
        dp_dx, dp_dy_perp = arakawa_lamb_gradient(h_or_p, cdgrid)
    else:
        B = KE + g * (h_or_p + h_s_or_p_prime)
        dB_dx, dB_dy_perp = arakawa_lamb_gradient(B, cdgrid)

    # 4. Absolute vorticity at corners: interp ζ only, add f_corner directly (FV3 convention).
    # interp(f_cc) adds O(dx²) sin(lat) nonlinearity error vs exact f_corner.
    if is_3d:
        zeta_corner = interp_center_to_corner(zeta, cdgrid)
    else:
        zeta_corner = (interp_center_to_corner(zeta, cdgrid)
                       + cdgrid.f_corner)

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
            div_corner = interp_center_to_corner(div_v, cdgrid)
            du_d_dt = du_d_dt - 0.5 * u_d * div_corner
            dv_d_dt = dv_d_dt - 0.5 * v_d * div_corner
    else:
        du_d_dt = zeta_corner * v_d - dB_dx
        dv_d_dt = -zeta_corner * u_d - dB_dy_perp

    # 6/7. Laplacian + biharmonic diffusion at corners in geographic (E/N) coords.
    # D→A avg kills the 2Δx mode (smooths resolved only); 2Δx handled by separate filter.
    if (A_h > 0 or hyperdiff_coeff > 0) and not is_3d:
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
            lap_u_corner = interp_center_to_corner(lap_u_local, cdgrid)
            lap_v_corner = interp_center_to_corner(lap_v_local, cdgrid)
            du_d_dt = du_d_dt + A_h * lap_u_corner
            dv_d_dt = dv_d_dt + A_h * lap_v_corner

        if hyperdiff_coeff > 0:
            bilap_ue = laplacian_compact(lap_ue, cdgrid.base)
            bilap_vn = laplacian_compact(lap_vn, cdgrid.base)
            cos_a = jnp.cos(cdgrid.base.angle)
            sin_a = jnp.sin(cdgrid.base.angle)
            bilap_u_local = cos_a * bilap_ue + sin_a * bilap_vn
            bilap_v_local = -sin_a * bilap_ue + cos_a * bilap_vn
            bilap_u_corner = interp_center_to_corner(bilap_u_local, cdgrid)
            bilap_v_corner = interp_center_to_corner(bilap_v_local, cdgrid)
            du_d_dt = du_d_dt - hyperdiff_coeff * bilap_u_corner
            dv_d_dt = dv_d_dt - hyperdiff_coeff * bilap_v_corner

    elif A_h > 0 or hyperdiff_coeff > 0:
        # 3D fallback: laplacian_dgrid for ocean/PE
        if A_h > 0:
            du_d_dt = du_d_dt + A_h * laplacian_dgrid(u_d, cdgrid)
            dv_d_dt = dv_d_dt + A_h * laplacian_dgrid(v_d, cdgrid)
        if hyperdiff_coeff > 0:
            du_d_dt = du_d_dt - hyperdiff_coeff * laplacian_dgrid(
                laplacian_dgrid(u_d, cdgrid), cdgrid)
            dv_d_dt = dv_d_dt - hyperdiff_coeff * laplacian_dgrid(
                laplacian_dgrid(v_d, cdgrid), cdgrid)

    # 8. Div damp (FV3 Smag). iter-758c revert: tendency form *dt over-damps; iter-757b da_min_c metric retained.
    # iter-872c-take4: narrow gate (div_damp > 0). iter-872c-take5: warn when dddmp>0 + div_damp=0 silent no-op.
    if dddmp > 0 and div_damp == 0:
        import warnings
        warnings.warn(
            f"`cdgrid_momentum_tendencies` called with `dddmp={dddmp!r}` and "
            f"`div_damp=0`. The narrow gate silently no-ops `dddmp`; set "
            f"`div_damp > 0` to enable adaptive damping.",
            stacklevel=2,
        )
    if div_damp > 0:
        div_field = cgrid_divergence(u_c, v_c, cdgrid)
        da_min_c = jnp.min(cdgrid.area_corner)    # Fortran da_min_c
        d2_bg = div_damp / da_min_c
        div_abs = jnp.abs(div_field)
        div_abs_corner = interp_center_to_corner(div_abs, cdgrid)
        adaptive_coeff = da_min_c * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * div_abs_corner))
        ddiv_dx, ddiv_dy_perp = arakawa_lamb_gradient(div_field, cdgrid)
        du_d_dt = du_d_dt + adaptive_coeff * ddiv_dx
        dv_d_dt = dv_d_dt + adaptive_coeff * ddiv_dy_perp

    return du_d_dt, dv_d_dt


# ==============================================================================
# FV3 edge-midpoint D-grid operators (u_d at x-edge midpoint, v_d at y-edge)
# ==============================================================================

def fv3_vorticity(u_d, v_d, cdgrid):
    """Relative vorticity at corners from edge-midpoint D-grid (exact circulation).

    Face boundaries: halo line integrals reconstructed from halo-exchanged cc winds
    to avoid same-face mode='edge' repetition.
    """
    dx = cdgrid.dx_edge_y   # (6, n, n+1)
    dy = cdgrid.dy_edge_x   # (6, n+1, n)

    u_dx = u_d * dx          # (6, n, n+1)
    v_dy = v_d * dy          # (6, n+1, n)

    # Halo-aware padding: interior direct, boundary halo from cc winds (cube-corner consistency)
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

    # Reconstruct halo u_dx from padded cc u: x-edge between cells (i,j-1) and (i,j)
    u_dx_west_halo = (0.5 * (u_pad[:, 0:1, :-1] + u_pad[:, 0:1, 1:])
                      * dx[:, 0:1, :])              # (6, 1, n+1)
    u_dx_east_halo = (0.5 * (u_pad[:, -1:, :-1] + u_pad[:, -1:, 1:])
                      * dx[:, -1:, :])              # (6, 1, n+1)

    u_dx_pad = jnp.concatenate([u_dx_west_halo, u_dx, u_dx_east_halo],
                               axis=1)              # (6, n+2, n+1)

    # Reconstruct halo v_dy: y-edge between cells (i-1,j) and (i,j)
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
    """Edge-midpoint D-grid → cc (simple avg, orthogonal-rotation convention; no non-orthogonality fix)."""
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    return u_cc, v_cc


def fv3_cc2c_core(u_pad, v_pad, cosa_u_metric):
    """Pure-array core of :func:`fv3_cc2c` — the cc -> C-face avg + non-orthogonality
    projection on the ALREADY vector-halo-padded cc winds.  Shared by the global
    wrapper + the sub-face tile kernel
    (``tiled_production_cdgrid.fv3_cc2c_tile_2d``); no dup numerics.

    ``u_pad``/``v_pad`` ``(F, W, W[, nlev])`` (W=n+2 full, or nl+2 per tile —
    the ``[1:-1]`` j-trim makes the same window slice tile cleanly);
    ``cosa_u_metric`` ``(F, W-1, W-2)``.  Returns ``u_c`` ``(F, W-1, W-2)``
    (x-face) + ``v_c`` ``(F, W-2, W-1)`` (y-face)."""
    # cc -> C-face avg with non-orthogonality projection (matches dgrid_to_cgrid)
    u_avg = 0.5 * (u_pad[:, :-1, 1:-1] + u_pad[:, 1:, 1:-1])
    v_at_u = 0.5 * (v_pad[:, :-1, 1:-1] + v_pad[:, 1:, 1:-1])
    cosa_u = _broadcast_metric(cosa_u_metric, u_avg)
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, _EPS))
    u_c = u_avg * sina_u - v_at_u * cosa_u

    v_c = 0.5 * (v_pad[:, 1:-1, :-1] + v_pad[:, 1:-1, 1:])
    return u_c, v_c


def fv3_cc2c(u_cc, v_cc, cdgrid):
    """cc → C-grid edge-normal. Vector halo + 2nd-order interp; duogrid scalar remap if active."""
    grid = cdgrid.base
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    u_pad, v_pad = pad_halo_vector(
        u_cc, v_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )

    u_c, v_c = fv3_cc2c_core(u_pad, v_pad, cdgrid.cosa_u)  # (6,n+1,n) / (6,n,n+1)

    # iter-839: u_c face-normal projection / v_c plain avg asymmetry is LOAD-BEARING.
    # Symmetric v_c projection caused 230× W2 regression (test_boundary_fix_is_load_bearing_for_w2_l2 failure).
    # Fortran d2a2c_vect uses covariant averages then derives contravariant downstream;
    # our path collapses D→A→C — fix needs cgrid_mass_flux_divergence refactor (iter-840+).
    return u_c, v_c


def fv3_sw_tendencies(
    h, u_d, v_d, h_s, cdgrid,
    g=constants.g, div_damp=0.0, hyperdiff_coeff=0.0,
    boundary_fix=False,
    zero_mean_correction=False,
    fortran_dir_aware_corners=False,
    dddmp=0.0,
    apply_fortran_xppm_boundary=False,
    d4_bg=0.0,
    d4_nord=1,
):
    """SW tendencies on FV3 edge-midpoint D-grid. Momentum via A-L + circulation; PPM mass transport.

    Biharmonic hyperdiffusion (``hyperdiff_coeff``) IS applied at
    cell centres downstream — see step (i) below.  Per
    new_test_dycores iter-41, this is the path that stabilizes
    cube W5 15-day + cube W6 14-day in the matrix runner
    (without it, both BLOW UP per iter-31/33 measurements).

    The original iter-1019 warning saying ``hyperdiff_coeff`` is
    "signature-only NO-OP, use CDGridShallowWaterModel" was stale
    — a later iter added the cell-centre biharmonic path without
    retiring the warning.  Removed in iter-41 after empirical
    verification that the kwarg has real algorithmic effect.
    """
    n = cdgrid.n

    # (a) Cell-centre and C-grid velocities for mass transport
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)

    # (b) Height tendency
    # iter-889: forward xppm_boundary for tp_core.F90:357-369 iord<7 overrides
    dh_dt = cgrid_mass_flux_divergence(
        h, u_c, v_c, cdgrid,
        apply_fortran_xppm_boundary=apply_fortran_xppm_boundary)
    if zero_mean_correction:
        total_area = jnp.sum(cdgrid.base.area)
        dh_dt = dh_dt - jnp.sum(dh_dt * cdgrid.base.area) / total_area

    # (c) Bernoulli function using physical cell-centre winds
    KE = 0.5 * (u_cc ** 2 + v_cc ** 2)
    B = KE + g * (h + h_s)

    # (d) Arakawa-Lamb gradient at D-grid corners.
    # iter-765: dir-aware halo=1 corner fill (sw_core.F90:3856-3915)
    dB_dx, dB_dy_perp = arakawa_lamb_gradient(
        B, cdgrid,
        fortran_dir_aware_corners=fortran_dir_aware_corners)

    # (e) Corner winds from halo-exchanged cc velocities.
    # Same stagger for vorticity + gradient: consistent interp errors cancel in geostrophic balance
    # (D-grid circulation vort breaks this: 3x W2 regression).
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

    # (f) Absolute vorticity at cell centres (same stagger as momentum below; no f interp).
    zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
    zeta_abs = zeta + cdgrid.base.f

    # (g) Momentum at cell centres: gradient + vorticity at same stagger → 2.6x better geostrophic cancellation
    dB_dx_cc = interp_corner_to_center(dB_dx)
    dB_dy_cc = interp_corner_to_center(dB_dy_perp)
    du_cc = zeta_abs * v_cc - dB_dx_cc      # (6, n, n)
    dv_cc = -zeta_abs * u_cc - dB_dy_cc     # (6, n, n)

    # (h) Div damping at cc (FV3 sw_core.F90:1720). iter-755b/757: da_min_c = min(area_corner).
    # iter-758c (revert): tendency form *dt over-damps; full d_sw5 port deferred (needs corner div + KE add).
    # iter-872c-take4: narrow gate (div_damp > 0). take5: warn when dddmp>0 + div_damp=0 silent no-op.
    if dddmp > 0 and div_damp == 0:
        import warnings
        warnings.warn(
            f"`fv3_sw_tendencies(dddmp={dddmp!r}, div_damp=0)` silent no-op. "
            f"Set `div_damp > 0` to enable adaptive damping.",
            stacklevel=2,
        )
    if div_damp > 0:
        div_field = cgrid_divergence(u_c, v_c, cdgrid)
        da_min_c = jnp.min(cdgrid.area_corner)    # Fortran da_min_c
        d2_bg = div_damp / da_min_c
        # iter-872c: dddmp default 0.0 (Fortran fv_arrays.F90:360 strict); prod 0.2 via CDGridShallowWaterConfig.dddmp_prod
        div_abs = jnp.abs(div_field)
        adaptive_coeff = da_min_c * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * div_abs))

        # iter-765b: thread dir_aware_corners to ALL A-L calls
        ddiv_dx, ddiv_dy_perp_cc = arakawa_lamb_gradient(
            div_field, cdgrid,
            fortran_dir_aware_corners=fortran_dir_aware_corners)
        du_cc = du_cc + adaptive_coeff * interp_corner_to_center(ddiv_dx)
        dv_cc = dv_cc + adaptive_coeff * interp_corner_to_center(ddiv_dy_perp_cc)

        if d4_bg > 0:
            # FV3 d_sw5 nord>=1 BACKGROUND divergence damping
            # (sw_core.F90:1720-1868; certified translation
            # fv3_native_d_sw.d_sw5 / fv3_native_dsw5): the damp
            # potential gains ddn*lap^nord(div) with
            # ddn = (da_min_c*d4_bg)^(nord+1); its wind-gradient
            # contribution is the del-(2*nord+2) divergence damping.
            # The authoritative duo case configs run nord=2, d4_bg=0.12
            # (Zenodo 8327578 rundir input.nml) with NO vorticity
            # damping.
            # Sign convention (stated per the sign-check mandate): on a
            # Fourier mode grad->div contributes one more Laplacian, so
            # du += s*ddn*grad(lap^nord(div)) gives d(div)/dt =
            # s*ddn*lap^(nord+1)(div), eigenvalue
            # s*(-1)^(nord+1)*k^(2nord+2) — decay requires
            # s = (-1)^nord: nord=1 -> MINUS (del-4), nord=2 -> PLUS
            # (del-6).  Pinned by
            # test_fv3_sw_d4_divergence_damping_decays (nord 1 and 2).
            ddn = (da_min_c * d4_bg) ** (d4_nord + 1)
            sgn = -1.0 if (d4_nord % 2) else 1.0
            lap_div = div_field
            for _ in range(d4_nord):
                lap_div = laplacian_compact(lap_div, cdgrid.base)
            dlap_dx, dlap_dy_perp_cc = arakawa_lamb_gradient(
                lap_div, cdgrid,
                fortran_dir_aware_corners=fortran_dir_aware_corners)
            du_cc = du_cc + sgn * ddn * interp_corner_to_center(dlap_dx)
            dv_cc = dv_cc + sgn * ddn * interp_corner_to_center(
                dlap_dy_perp_cc)

    # (i) Biharmonic hyperdiffusion (cell-centre geographic path)
    if hyperdiff_coeff > 0:
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

    # (j) Smooth face-boundary tendencies — NON-FV3 stabilizer (load-bearing).
    # iter-511 W2 C36 1d: boundary_fix=True L2=5.64e-4 vs False L2=1.36e-3 (2.4x worse).
    # iter-865b: gate on `not bounded_domain` (FV3 fv_arrays.F90:1512 bounded_domain = regional|nested|duogrid).
    # duogrid required for cross_face flag — pad_halo_vector handles boundaries Fortran-faithfully.
    if boundary_fix and (not cdgrid.base.bounded_domain) and n > 2:
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

def overlapped_arakawa_lamb_gradient(B, cdgrid, masks=None):
    """Arakawa-Lamb gradient. Falls through to canonical 4D path (mpi4jax has no non-blocking).

    Per-level overlap turns 1 halo into nlev halos (strictly worse). Signature kept for forward compat.
    """
    return arakawa_lamb_gradient(B, cdgrid)
