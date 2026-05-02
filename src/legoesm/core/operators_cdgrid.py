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

from legoesm import constants
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

def _ppm_reconstruct_1d(q, *, axis: int,
                        apply_fortran_xppm_boundary: bool = False,
                        n_interior: int | None = None,
                        fortran_faithful_ppm_left: bool = False,
                        fortran_faithful_ppm_right: bool = False):
    """PPM face-value reconstruction along ``axis``.

    Given cell averages along ``axis``, compute left and right face
    values (q_L, q_R) for each cell using 4th-order interpolation with
    monotonicity constraints (Colella & Woodward 1984).

    iter-509 (Codex): ``axis`` is a REQUIRED keyword-only argument.
    Earlier iterations (505/506/508) had to repair x-direction call
    sites that silently reconstructed along the wrong axis because
    the reconstruction axis was implicit (LAST axis by default).
    Removing the default forces every caller to declare the axis at
    the call site so the bug class cannot reappear.  The previous
    iter-508 default of ``axis=-1`` was the same value that produced
    the original bug for x-direction strips of shape ``(6, n+4, n)``,
    so leaving it as a default just papered over the issue.

    Parameters
    ----------
    q : jax.Array, shape (..., N, ...)
        Cell averages.  Requires the size along ``axis`` to be N >= 4.
    axis : int
        REQUIRED keyword.  Axis along which to reconstruct face values.
        Use ``axis=1`` for x-direction strips of shape ``(6, n+4, n)``;
        use ``axis=2`` for y-direction strips of shape ``(6, n, n+4)``.
    apply_fortran_xppm_boundary : bool, default False
        Iter-889: when True, AND the strip has a halo=2 cubed-sphere
        layout (`n_interior` provided), overwrite the 6 face-boundary
        ``q_face`` values with Fortran's iord<7 cube-edge boundary
        formulas from `tp_core.F90:357-369`:
        - Left side: ``q_face[1] = c1*q1(-2) + c2*q1(-1) + c3*q1(0)``
          (Fortran al(0)), ``q_face[2] = uniform 4-point xt`` (Fortran
          al(1) — cube-face edge), ``q_face[3] = c3*q1(1) + c2*q1(2)
          + c1*q1(3)`` (Fortran al(2)).  Mirror on the right side.
        Constants from `tp_core.F90:63-65`:
            c1 = -2/14, c2 = 11/14, c3 = 5/14.
        The 4-point xt at the cube-face edge uses the UNIFORM-GRID
        simplification of Fortran's dxa-weighted formula (lines 360-
        361, 366-367):
            xt = 0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))
        and is clipped to ``min/max(q1(-1..2))``.  For non-uniform
        cubed-sphere boundary cells this is a partial Fortran-fidelity
        fix; the full ``dxa``-weighted formula is deferred until
        ``dxa`` plumbing is added.
        Default False preserves prior behaviour bit-for-bit.
    n_interior : int or None, default None
        Number of interior cells along ``axis`` when the strip has
        halo=2 padding (so total size along ``axis`` is
        ``n_interior + 4``).  Required when
        ``apply_fortran_xppm_boundary=True`` because the Fortran
        boundary formulas need to know exactly which 6 ``q_face``
        indices to override (depends on n).  When the kwarg is False
        this is ignored.

    Returns
    -------
    q_L : jax.Array, same shape as ``q``
        Left face value for each cell along ``axis``.
    q_R : jax.Array, same shape as ``q``
        Right face value for each cell along ``axis``.
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

    # Iter-892 (Codex iter-891b stop-time follow-up).  Fortran-faithful
    # cube-edge boundary overrides per tp_core.F90:357-369 (iord<7
    # path).  Only fires when the kwarg is True AND the strip is a
    # known halo=2 cubed-sphere layout (caller provides `n_interior`).
    # Constants from tp_core.F90:63-65: c1 = -2/14, c2 = 11/14, c3 = 5/14.
    #
    # Index map (CORRECTED in iter-892):
    #   q has caller halo=2: q[k] = q1(k-1) for k=0..n_int+3.
    #   q_pad = pad(q, halo=2, 'edge') has length n_int+8.  Indices:
    #     q_pad[0,1] = replicated q[0] = q1(-1)
    #     q_pad[k] = q[k-2] for k=2..n_int+5  =>  q1(k-3) for k=2..n_int+5
    #     q_pad[n_int+6, n_int+7] = replicated q[n_int+3] = q1(n_int+2)
    #   q_face[k] = (7/12)*(q_pad[k+1]+q_pad[k+2]) - (1/12)*(q_pad[k]+q_pad[k+3])
    #     i.e. face between q_pad[k+1] and q_pad[k+2]
    #     = face between q1(k-2) and q1(k-1)
    #     = Fortran al(k-1).   So al(i) → q_face[i+1].
    #
    # Fortran al(0) needs q1(-2) and al(npx+1) needs q1(npx+2).  Both
    # are halo depth 2 cells unavailable with our halo=2 input.
    # q_pad[0] / q_pad[1] are mode='edge' replicas of q1(-1), NOT
    # actual q1(-2); same on the right.  Including those degenerate
    # values would produce non-Fortran-faithful results.  iter-892
    # therefore overrides ONLY the 4 cube-edge faces that are
    # computable Fortran-faithfully with halo=2:
    #   q_face[2]      = al(1)     (left 4-pt xt clipped)
    #   q_face[3]      = al(2)     (left c3/c2/c1 mirror)
    #   q_face[n+1]    = al(npx-1) (right c1/c2/c3)
    #   q_face[n+2]    = al(npx)   (right 4-pt xt clipped)
    # q_face[1] (al(0)) and q_face[n+3] (al(npx+1)) are LEFT
    # UNTOUCHED on the standard 4th-order edge formula.
    #
    # Pre-iter-892 iter-889 used left-side cells (q_pad[2..7]) shifted
    # +1 from the correct Fortran mapping (q_pad[2..6] for the al(1)
    # /al(2) overrides).  iter-892 corrects the cell indexing.
    if apply_fortran_xppm_boundary and n_interior is not None:
        c1 = -2.0 / 14.0
        c2 = 11.0 / 14.0
        c3 = 5.0 / 14.0
        n_int = int(n_interior)

        if fortran_faithful_ppm_left:
            # Iter-900 LEFT-side Fortran-faithful overrides at the
            # CORRECTED q_face indices.  iter-899 investigation
            # (`scripts/diag_iter899_ppm_strip_layout.py`) established
            # that production strip is q[k]=q1(k-2) (Hypothesis A,
            # full halo=2 per side), NOT q[k]=q1(k-1) as the iter-892
            # docstring claims.  Under Hypothesis A, q_pad[k]=q1(k-4)
            # for k=2..n+5, so:
            #   q_pad[2]=q1(-2), q_pad[3]=q1(-1), q_pad[4]=q1(0),
            #   q_pad[5]=q1(1),  q_pad[6]=q1(2),  q_pad[7]=q1(3),
            # and q_face[k] corresponds to Fortran al(k-2):
            #   q_face[2]=al(0), q_face[3]=al(1), q_face[4]=al(2).
            #
            # Fortran formulas (`tp_core.F90:359-362`):
            #   al(0) = c1*q1(-2) + c2*q1(-1) + c3*q1(0)
            #         = c1*q_pad[2] + c2*q_pad[3] + c3*q_pad[4]
            #   al(1) = xt 4-pt clipped using q1(-1..2) = q_pad[3..6]
            #         xt = 0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))
            #            = 0.75*(q_pad[4]+q_pad[5]) - 0.25*(q_pad[3]+q_pad[6])
            #         clip to min/max(q1(-1..2)) = min/max(q_pad[3..6]).
            #   al(2) = c3*q1(1) + c2*q1(2) + c1*q1(3)
            #         = c3*q_pad[5] + c2*q_pad[6] + c1*q_pad[7]
            face_al0 = (c1 * q_pad[..., 2] + c2 * q_pad[..., 3]
                        + c3 * q_pad[..., 4])
            xt_L = (0.75 * (q_pad[..., 4] + q_pad[..., 5])
                    - 0.25 * (q_pad[..., 3] + q_pad[..., 6]))
            q_lo_L = jnp.minimum(
                jnp.minimum(q_pad[..., 3], q_pad[..., 4]),
                jnp.minimum(q_pad[..., 5], q_pad[..., 6]))
            q_hi_L = jnp.maximum(
                jnp.maximum(q_pad[..., 3], q_pad[..., 4]),
                jnp.maximum(q_pad[..., 5], q_pad[..., 6]))
            face_al1 = jnp.clip(xt_L, q_lo_L, q_hi_L)
            face_al2 = (c3 * q_pad[..., 5] + c2 * q_pad[..., 6]
                        + c1 * q_pad[..., 7])

            q_face = q_face.at[..., 2].set(face_al0)
            q_face = q_face.at[..., 3].set(face_al1)
            q_face = q_face.at[..., 4].set(face_al2)
        else:
            # Iter-892 LEFT cube-edge boundary overrides (al(1) and
            # al(2) only; al(0) at q_face[1] left untouched — needs
            # halo=3).  KNOWN 1-CELL SHIFT BUG documented in iter-899:
            # the q_pad indices used here correspond to Hypothesis B
            # (q[k]=q1(k-1)) but production gives Hypothesis A
            # (q[k]=q1(k-2)).  Despite the bug, iter-893 measured a
            # 17 % W2 v_ll_Linf reduction with this default — keep it
            # as the production default until iter-900+ measurement
            # confirms the strict-Fortran path is at least as good.
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

        if fortran_faithful_ppm_right:
            # Iter-903 RIGHT-side Fortran-faithful overrides at the
            # CORRECTED q_face indices.  Symmetric to iter-900's
            # LEFT-side fix.  Under Hypothesis A (production strip
            # layout q[k]=q1(k-2)), q_pad[k]=q1(k-4) for k=2..n+5,
            # and q_face[k]=al(k-2).  Therefore:
            #   q_face[n+2] = al(n)  = al(npx-1)
            #   q_face[n+3] = al(n+1) = al(npx)
            #   q_face[n+4] = al(n+2) = al(npx+1)  (NOT placed - halo=3)
            #
            # Fortran formulas (`tp_core.F90:365-368`):
            #   al(npx-1) = c1*q1(npx-3) + c2*q1(npx-2) + c3*q1(npx-1)
            #             = c1*q_pad[n+2] + c2*q_pad[n+3] + c3*q_pad[n+4]
            #   al(npx) = xt 4-pt clipped using q1(npx-2..npx+1)
            #           = q_pad[n+3..n+6]
            #     where q_pad[n+6] is mode='edge' replica of q1(n+1).
            #     Fortran's al(npx) needs q1(npx+1) = q1(n+2) which is
            #     OUTSIDE halo=2; the xt formula here uses q_pad[n+6] =
            #     q1(n+1) instead (PARTIAL faithfulness — the same
            #     "halo=2 limit" caveat as for al(npx+1)).
            #   al(npx+1) needs q1(n+2) and q1(n+3) — NOT plumbed.
            face_alnm1_faithful = (
                c1 * q_pad[..., n_int + 2]
                + c2 * q_pad[..., n_int + 3]
                + c3 * q_pad[..., n_int + 4])
            xt_R_faithful = (
                0.75 * (q_pad[..., n_int + 4] + q_pad[..., n_int + 5])
                - 0.25 * (q_pad[..., n_int + 3] + q_pad[..., n_int + 6]))
            q_lo_R_f = jnp.minimum(
                jnp.minimum(q_pad[..., n_int + 3], q_pad[..., n_int + 4]),
                jnp.minimum(q_pad[..., n_int + 5], q_pad[..., n_int + 6]))
            q_hi_R_f = jnp.maximum(
                jnp.maximum(q_pad[..., n_int + 3], q_pad[..., n_int + 4]),
                jnp.maximum(q_pad[..., n_int + 5], q_pad[..., n_int + 6]))
            face_aln_faithful = jnp.clip(
                xt_R_faithful, q_lo_R_f, q_hi_R_f)

            q_face = q_face.at[..., n_int + 2].set(face_alnm1_faithful)
            q_face = q_face.at[..., n_int + 3].set(face_aln_faithful)
            # Iter-892's q_face[n+1] override (placing c1/c2/c3 at
            # the al(n-1) slot which Fortran does NOT specially treat)
            # is REMOVED in this branch — that slot reverts to the
            # standard 4th-order interior stencil.
        else:
            # Iter-892 RIGHT cube-edge boundary overrides (al(npx-1)
            # and al(npx) only; al(npx+1) at q_face[n+3] left
            # untouched — needs halo=3).  KNOWN 1-CELL SHIFT BUG
            # documented in iter-899: places formulas at q_face[n+1,
            # n+2] but those are al(n-1) and al(n) under Hypothesis
            # A (not the al(npx-1, npx) the formulas were designed
            # for).  Despite the bug, iter-893 measured a 17 % W2
            # v_ll_Linf reduction with this default — keep it as the
            # production default until iter-901+ measurement
            # confirms the strict-Fortran path is at least as good.
            #
            # al(npx-1) = c1*q1(npx-3) + c2*q1(npx-2) + c3*q1(npx-1)
            # With npx = n_int + 1: q1(npx-3) = q1(n-2) = q_pad[n+1],
            # q1(npx-2) = q1(n-1) = q_pad[n+2], q1(npx-1) = q1(n) = q_pad[n+3].
            face_alnm1 = (c1 * q_pad[..., n_int + 1]
                          + c2 * q_pad[..., n_int + 2]
                          + c3 * q_pad[..., n_int + 3])
            # al(npx) = uniform 4-point xt clipped to min/max(q1(npx-2..npx+1)).
            # q1(npx-2) = q_pad[n+2], q1(npx-1) = q_pad[n+3],
            # q1(npx) = q_pad[n+4], q1(npx+1) = q_pad[n+5].
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

    # 2. Overshoot limiting: ensure the parabola doesn't create new
    # extrema.  Per CW84 eq. 1.10:
    #   if Δa · a_6 >  (Δa)²:  a_L = 3a - 2a_R
    #   if Δa · a_6 < -(Δa)²:  a_R = 3a - 2a_L
    # where Δa = a_R - a_L and a_6 = 6(a - 0.5(a_L+a_R)).  Iter-878
    # (Fortran-fidelity fix): pre-iter-878 the conditions were
    # ``q_6 > dq*dq`` and ``-q_6 > dq*dq`` (missing the ``dq`` factor
    # on the LHS), which differs from CW84 / Fortran ``pert_ppm``
    # (tp_core.F90:1199-1205) where the test is ``a6da = 3*(al+ar)*
    # da1`` (this includes ``da1`` factor) compared to ``da2 = da1²``.
    # Restoring the ``dq`` factor on the LHS makes the gate match
    # CW84 + Fortran exactly.
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

def cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid,
                                apply_fortran_xppm_boundary=False,
                                fortran_faithful_ppm_left=False,
                                fortran_faithful_ppm_right=False):
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
            return cgrid_mass_flux_divergence(
                hk, uk, vk, cdgrid,
                apply_fortran_xppm_boundary=(
                    apply_fortran_xppm_boundary),
                fortran_faithful_ppm_left=fortran_faithful_ppm_left,
                fortran_faithful_ppm_right=fortran_faithful_ppm_right)

        result_t = jax.vmap(flux_div_one)((h_t, u_c_t, v_c_t))
        return jnp.moveaxis(result_t, 0, -1)

    # 2D case: PPM face reconstruction with halo=2
    h_pad = _pad_halo_auto_h2(h, cdgrid)  # (6, n+4, n+4)

    dy = cdgrid.dy_edge_x   # (6, n+1, n)
    dx = cdgrid.dx_edge_y   # (6, n, n+1)
    n = cdgrid.n

    # Iter-889b (Codex iter-889 stop-time fix): Fortran's iord<7 cube-
    # edge boundary overrides at tp_core.F90:357 are gated on
    # ``.not. (bounded_domain .or. duogrid) .and. grid_type<3``.
    # iter-889 forwarded the kwarg unconditionally; in duogrid /
    # bounded-domain mode the cross-face halo (or Neumann panel BC)
    # already provides Fortran-faithful neighbour-face values so the
    # legacy non-duogrid boundary formula must NOT fire.  Same gate
    # pattern as iter-865's `boundary_fix` and iter-865b's
    # `fortran_vector_corner_fill` corrections.  Compute the EFFECTIVE
    # flag here so the gate lives in one place and the leaf
    # `_ppm_reconstruct_1d` receives a pre-gated boolean.
    effective_xppm_boundary = (
        apply_fortran_xppm_boundary
        and not cdgrid.base.bounded_domain)

    # --- X-direction PPM ---
    # For each j, reconstruct h along i-direction and compute flux at
    # each x-interface.  h_pad[:, :, j+2] for j in [0, n-1] gives the
    # i-strip at interior j, with 2 halo cells on each side.
    # x-interface (i, j) for i in [0, n] needs cells i-2..i+1 in the
    # original grid, which maps to padded indices i..i+3.

    # Extract strips along i for each j: shape (6, n+4, n) from padded.
    # The halo-padded i-axis is axis=1; pass `axis=1` explicitly so
    # `_ppm_reconstruct_1d` reconstructs in the i-direction (iter-508
    # made the axis explicit to prevent the iter-505 silent-bug class).
    h_x_strips = h_pad[:, :, 2:-2]                  # (6, n+4, n)
    q_L_x, q_R_x = _ppm_reconstruct_1d(
        h_x_strips, axis=1,
        apply_fortran_xppm_boundary=effective_xppm_boundary,
        n_interior=n,
        fortran_faithful_ppm_left=fortran_faithful_ppm_left,
        fortran_faithful_ppm_right=fortran_faithful_ppm_right)

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
    # Strip shape (6, n, n+4) puts the halo-padded j-axis at axis=2;
    # pass `axis=2` explicitly per the iter-509 PPM contract.
    h_y_strips = h_pad[:, 2:-2, :]  # (6, n, n+4)
    q_L_y, q_R_y = _ppm_reconstruct_1d(
        h_y_strips, axis=2,
        apply_fortran_xppm_boundary=effective_xppm_boundary,
        n_interior=n,
        fortran_faithful_ppm_left=fortran_faithful_ppm_left,
        fortran_faithful_ppm_right=fortran_faithful_ppm_right)

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

# ==============================================================================
# Flux-corrected transport (FCT) for monotone tracer advection
# ==============================================================================

def _cgrid_fct_fluxes_2d(q, u_c, v_c, cdgrid):
    """Compute FCT-limited tracer fluxes on the C-D grid.

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

    Accepts both 2D ``(6, n, n)`` and 3D ``(6, n, n, nlev)`` tracers.
    The 4D path moves ``nlev`` to the leading position so the spatial
    slicing operates on axes ``-2, -1`` regardless of rank, and the
    halo pad runs once for all levels via ``pad_halo_4d`` /
    ``pad_halo_4d_h2`` (single MPI exchange).  Replaces the prior
    per-level ``vmap`` dispatcher in :func:`cgrid_tracer_advection_fct`
    that issued ``nlev`` separate halo exchanges.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n) or (6, n, n, nlev)
    u_c : jax.Array, shape (6, n+1, n[, nlev])
    v_c : jax.Array, shape (6, n, n+1[, nlev])
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    dq_dt : jax.Array, shape matching ``q``.

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

    # Halo-padded fields (halo=1 for upwind / min-max stencil, halo=2 for PPM).
    # ``_pad_halo_auto*`` already dispatch on ``q.ndim`` so 4D inputs use
    # ``pad_halo_4d`` (one MPI message for all levels).
    q_pad_full = _pad_halo_auto(q, cdgrid)        # (6, n+2, n+2[, nlev])
    q_pad_h2_full = _pad_halo_auto_h2(q, cdgrid)  # (6, n+4, n+4[, nlev])

    # For 4D inputs we move ``nlev`` to the leading position so that the
    # rest of the body's spatial slicing — written with ``[..., ...]``
    # prefixes — operates on the same trailing ``(i, j)`` axes regardless
    # of rank.  This also keeps ``_ppm_reconstruct_1d`` (axis -1) acting on
    # the correct PPM axis (j with halo, n with halo stripped).
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
    q_pad_h2 = _pad_halo_auto_h2(q, cdgrid)  # (6, n+4, n+4)

    # X-direction PPM — pass `axis=1` explicitly to reconstruct along
    # the halo-padded i-direction.  Same iter-508 contract as
    # `cgrid_mass_flux_divergence`.
    q_x_strips = q_pad_h2[:, :, 2:-2]                   # (6, n+4, n)
    q_L_x, q_R_x = _ppm_reconstruct_1d(q_x_strips, axis=1)
    q_R_left = q_R_x[:, 1:n+2, :]
    q_L_right = q_L_x[:, 2:n+3, :]
    q_face_hi_x = jnp.where(u_c > 0, q_R_left, q_L_right)

    # Clip to local bounds of adjacent cells
    q_face_min_x = jnp.minimum(q_left_x, q_right_x)
    q_face_max_x = jnp.maximum(q_left_x, q_right_x)
    q_face_hi_x = jnp.clip(q_face_hi_x, q_face_min_x, q_face_max_x)

    # Y-direction PPM — strip shape (6, n, n+4) puts the halo-padded
    # j-axis at axis=2; pass `axis=2` explicitly per iter-509 contract.
    q_y_strips = q_pad_h2[:, 2:-2, :]
    q_L_y, q_R_y = _ppm_reconstruct_1d(q_y_strips, axis=2)
    q_R_bottom = q_R_y[:, :, 1:n+2]
    q_L_top = q_L_y[:, :, 2:n+3]
    q_face_hi_y = jnp.where(v_c > 0, q_R_bottom, q_L_top)

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

    # How much room does the low-order update leave?
    # After applying dq_low, q would be at q + dq_low (for unit "dt").
    # We allow the anti-diffusive part to bring it to at most q_max
    # and at least q_min.
    q_td = q_t + dq_low   # provisional (unit-step low-order update)

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

    result = dq_low + alpha * ad
    if is_4d:
        return jnp.moveaxis(result, 0, -1)
    return result


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

    Works for both 2D (6, n, n) and 3D (6, n, n, nlev) inputs.  The 4D
    path runs ``_cgrid_fct_fluxes_2d`` natively on the 4D field — a
    single ``pad_halo_4d`` MPI exchange across all levels — replacing
    the prior per-level ``vmap`` that issued ``nlev`` separate halo
    exchanges.  ``cgrid_mass_flux_divergence`` (used in the JVP path of
    ``_make_fct_2d_differentiable``) is already 4D-native, so the
    differentiable ``custom_jvp`` wrapper composes cleanly.

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
    return fct_fn(q, u_c, v_c)


# ==============================================================================
# Arakawa-Lamb gradient at D-grid corners
# ==============================================================================

def _arakawa_lamb_gradient(B, cdgrid, padded=None,
                           fortran_dir_aware_corners=False,
                           fortran_a2b_corner_avg=False):
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
    fortran_dir_aware_corners : bool, default False
        Iter-765: when True, replace the 2-pt-avg cube-corner halo
        cells in the padded field with Fortran-faithful directional
        inner fills (sw_core.F90:3856-3915 halo=1 inner subset —
        iter-764d).  x-gradient component uses dir=1 inner fill;
        y-gradient component uses dir=2 inner fill.  Only affects
        the 4 cube-vertex halo cells per face; the remaining halo
        is untouched.  See iter-764/765 review-doc entries.  False
        preserves the legacy 2-pt-avg behaviour.
    fortran_a2b_corner_avg : bool, default False
        Iter-766: when True, replace the 2-pt-avg cube-corner halo
        cells with Fortran's ``a2b_ord4`` 3-point corner average
        (``a2b_edge.F90:385-388``).  Fortran's 3-pt formula at the
        SW cube vertex is

            qout(1,1) = r3*(qin(1,1) + qin(1,0) + qin(0,1))

        — a SCALAR 3-point average that EXCLUDES the diagonal halo
        cell ``qin(0,0)``.  In our halo=1 convention this maps to

            padded[0,0] = (1/3)*(padded[0,1] + padded[1,0]
                                  + padded[1,1])

        (and symmetric formulas for SE, NE, NW).  Fortran uses this
        formula in the A-to-B scalar interpolation that feeds its
        pressure-gradient machinery — the closest Fortran analogue
        to our A-L gradient's 4-point stencil at D-grid corners.
        UNLIKE ``fortran_dir_aware_corners`` (iter-765 falsified),
        this fill is DIRECTION-NEUTRAL — no sweep-specific variant.
        Symmetric 2D stencils can consume it consistently.  Only
        affects the 4 cube-vertex halo cells per face.  See iter-766
        review-doc entry.  False preserves the legacy 2-pt-avg
        behaviour.

    Returns
    -------
    dB_dx, dB_dy_perp : jax.Array, shape (6, n+1, n+1[, nlev])
        Gradient along face-local e_x and perpendicular to e_x.
    """
    B_pad = padded if padded is not None else _pad_halo_auto(B, cdgrid)

    # Iter-766 Codex 2nd-pass: the two cube-corner opt-ins both
    # overwrite the same 4 cube-vertex halo cells; applied together
    # iter-766's a2b mutation is silently discarded by iter-765's
    # subsequent p1/p2 construction.  Refuse the combination so a
    # future diagnostic cannot silently get a mixed result.
    if fortran_a2b_corner_avg and fortran_dir_aware_corners:
        raise ValueError(
            "`fortran_a2b_corner_avg=True` and "
            "`fortran_dir_aware_corners=True` both overwrite the 4 "
            "cube-corner halo cells of the A-L gradient's padded "
            "field.  Enabling both silently discards the a2b "
            "mutation (iter-765's dir-aware p1/p2 construction wins).  "
            "Pick one diagnostic at a time; do not combine.")

    if fortran_a2b_corner_avg:
        # Fortran's `a2b_ord4` 3-pt average at the 4 cube-vertex
        # A-halo cells (a2b_edge.F90:385-388).  Overwrite only those
        # 4 positions; the remaining halo is unchanged.
        # Fortran -> Python halo=1 index mapping:
        #   qin(0,0)   — Python padded[:, 0, 0]    — SW cube halo
        #   qin(0,1)   — Python padded[:, 0, 1]    — W-edge halo at j=1
        #   qin(1,0)   — Python padded[:, 1, 0]    — S-edge halo at i=1
        #   qin(1,1)   — Python padded[:, 1, 1]    — interior diagonal
        if B.ndim == 3:
            sw = (B_pad[:, 0, 1] + B_pad[:, 1, 0] + B_pad[:, 1, 1]) / 3.0
            se = (B_pad[:, -2, 0] + B_pad[:, -1, 1] + B_pad[:, -2, 1]) / 3.0
            nw = (B_pad[:, 0, -2] + B_pad[:, 1, -1] + B_pad[:, 1, -2]) / 3.0
            ne = (B_pad[:, -2, -1] + B_pad[:, -1, -2] + B_pad[:, -2, -2]) / 3.0
            B_pad = B_pad.at[:, 0, 0].set(sw)
            B_pad = B_pad.at[:, -1, 0].set(se)
            B_pad = B_pad.at[:, 0, -1].set(nw)
            B_pad = B_pad.at[:, -1, -1].set(ne)
        else:
            sw = (B_pad[:, 0, 1, :] + B_pad[:, 1, 0, :]
                  + B_pad[:, 1, 1, :]) / 3.0
            se = (B_pad[:, -2, 0, :] + B_pad[:, -1, 1, :]
                  + B_pad[:, -2, 1, :]) / 3.0
            nw = (B_pad[:, 0, -2, :] + B_pad[:, 1, -1, :]
                  + B_pad[:, 1, -2, :]) / 3.0
            ne = (B_pad[:, -2, -1, :] + B_pad[:, -1, -2, :]
                  + B_pad[:, -2, -2, :]) / 3.0
            B_pad = B_pad.at[:, 0, 0, :].set(sw)
            B_pad = B_pad.at[:, -1, 0, :].set(se)
            B_pad = B_pad.at[:, 0, -1, :].set(nw)
            B_pad = B_pad.at[:, -1, -1, :].set(ne)

    if fortran_dir_aware_corners:
        # Build two padded variants differing only at the 4 cube-
        # corner halo cells per face.  For each corner position,
        # Fortran's dir=1 inner fill uses the i=0-column value one-j-
        # inward; dir=2 inner fill uses the j=0-row value one-i-
        # inward.  The x-gradient stencil uses dir=1; y-gradient
        # uses dir=2.
        # Iter-766 (Codex): the `halo.py::_fill_corners_h1` convention
        # names these 4 cube-vertex halo cells as SW=[0,0], NW=[0,-1],
        # SE=[-1,0], NE=[-1,-1].  Earlier iter-765 labels here
        # transposed NW and SE — runtime behaviour was unaffected
        # (each overwrite still targets the same cell) but combined
        # iter-765/766 reasoning was harder to follow.  Corrected:
        # SW: padded[:, 0, 0]   — dir1 ← padded[:, 0, 1],  dir2 ← padded[:, 1, 0]
        # NW: padded[:, 0, -1]  — dir1 ← padded[:, 0, -2], dir2 ← padded[:, 1, -1]
        # SE: padded[:, -1, 0]  — dir1 ← padded[:, -1, 1], dir2 ← padded[:, -2, 0]
        # NE: padded[:, -1, -1] — dir1 ← padded[:, -1, -2],dir2 ← padded[:, -2, -1]
        # (dims: pad is (6, n+2, n+2); here index -1 means n+1.)
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
    else:
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


def _interp_center_to_corner_a2b_ord4(field, cdgrid):
    """Iter-971: 4th-order A→B (cell-centre → corner) interpolation.

    Port of Fortran ``a2b_ord4`` (a2b_edge.F90:50-330) for the
    duogrid path (lines 98-104, 185-192, 241-258).  The full FV3
    reference uses two cascaded 4-point stencils:

        qx(i, j)  = b2 * (qin(i-2, j) + qin(i+1, j))
                   + b1 * (qin(i-1, j) + qin(i, j))
        qy(i, j)  = b2 * (qin(i, j-2) + qin(i, j+1))
                   + b1 * (qin(i, j-1) + qin(i, j))
        qxx(i, j) = a2 * (qx(i, j-2) + qx(i, j+1))
                   + a1 * (qx(i, j-1) + qx(i, j))
        qyy(i, j) = a2 * (qy(i-2, j) + qy(i+1, j))
                   + a1 * (qy(i-1, j) + qy(i, j))
        qout(i, j) = 0.5 * (qxx(i, j) + qyy(i, j))

    Constants: a1=9/16, a2=-1/16 (Lagrange 4-pt); b1=7/12, b2=-1/12
    (PPM volume mean).

    The qx step is a 4-pt x-direction average from cells to i-faces.
    The qxx step is a 4-pt y-direction average from i-faces to
    corners.  qyy is the symmetric path through y first.  Final
    qout averages the two paths.

    For Smagorinsky-tuned d_sw5 callers (iter-959/963) this is
    more Fortran-faithful than `_interp_center_to_corner` (which
    is a 2nd-order 4-point average).

    Parameters
    ----------
    field : (6, n, n) cell-centre A-grid scalar (e.g., wk vorticity)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    qout : (6, n+1, n+1) corner-staggered B-grid scalar
    """
    n = field.shape[1]
    # Pad input with cross-face halo of depth 2 (need cells [-2, n+1]
    # for the 4-pt stencil applied on both i and j axes).
    f_pad = _pad_halo_auto_h2(field, cdgrid)  # (6, n+4, n+4)
    a1 = 9.0 / 16.0
    a2 = -1.0 / 16.0
    b1 = 7.0 / 12.0
    b2 = -1.0 / 12.0

    # qx(i, j) at i-face i ∈ [0, n] uses cells (i-2, i-1, i, i+1).
    # In padded indexing (cell c at f_pad[c+2]): for I-face k ∈ [0, n],
    # cells (k-2, k-1, k, k+1) → padded (k, k+1, k+2, k+3).
    # qx shape: (n+1) i-faces × full halo'd j-cells (n+4) with j-halo
    # available for the qxx 4-pt y-stencil.
    qx = (b2 * (f_pad[:, 0:n + 1, :] + f_pad[:, 3:n + 4, :])
          + b1 * (f_pad[:, 1:n + 2, :] + f_pad[:, 2:n + 3, :]))
    # qx shape: (6, n+1, n+4)

    # qy(i, j) at j-face j ∈ [0, n] using cells (j-2..j+1) on the
    # j-axis.  Symmetric to qx.
    qy = (b2 * (f_pad[:, :, 0:n + 1] + f_pad[:, :, 3:n + 4])
          + b1 * (f_pad[:, :, 1:n + 2] + f_pad[:, :, 2:n + 3]))
    # qy shape: (6, n+4, n+1)

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

# ==============================================================================
# Laplacian at D-grid corners
# ==============================================================================

def _laplacian_dgrid(u_d, cdgrid):
    """Laplacian of a D-grid field via cell-centre round-trip.

    Interpolates D-grid -> cell centres, applies the compact
    cell-centre Laplacian (which uses proper inter-face halo exchange),
    then interpolates back to D-grid corners.

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev) inputs.
    The 3D path uses ``laplacian_compact_3d`` so the cell-centre halo
    exchange is shared across all vertical levels in a single
    ``pad_halo_4d`` collective — replaces the previous
    ``moveaxis + jax.vmap + moveaxis`` dance that issued ``nlev``
    separate halo calls.  ``_interp_corner_to_center`` and
    ``_interp_center_to_corner`` are already 4D-native, so the whole
    operator is batched with no per-level Python loop.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n+1, n+1[, nlev])

    Returns
    -------
    jax.Array, shape (6, n+1, n+1[, nlev])
    """
    # 1. D-grid -> cell centres: (6, n+1, n+1[, nlev]) -> (6, n, n[, nlev])
    u_cc = _interp_corner_to_center(u_d)

    # 2. Cell-centre Laplacian with proper halo exchange.  Use the
    # native-4D variant on 3D inputs so all levels share one
    # ``pad_halo_4d`` MPI exchange.
    if u_d.ndim == 4:
        from legoesm.core.operators_3d import laplacian_compact_3d
        lap_a = laplacian_compact_3d(u_cc, cdgrid.base)  # (6, n, n, nlev)
    else:
        from legoesm.core.operators import laplacian_compact
        lap_a = laplacian_compact(u_cc, cdgrid.base)  # (6, n, n)

    # 3. Cell centres -> D-grid: (6, n, n[, nlev]) -> (6, n+1, n+1[, nlev])
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
    g=constants.g, A_h=0.0, hyperdiff_coeff=0.0, div_damp=0.0,
    rho_0=None, div_v=None, f_3d=None,
    u_prime=None, v_prime=None,
    dddmp=0.0,
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

    # 1. Relative vorticity at cell centres (just ζ, no f yet)
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)

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

    # 4. Absolute vorticity at corners.  Interpolate only the RELATIVE
    # part ζ from cell centres to corners; add the planetary part f
    # DIRECTLY at corners via cdgrid.f_corner (= 2Ω sin(lat_corner))
    # instead of interp(cdgrid.base.f) = interp(2Ω sin(lat_cc)).  The
    # latter adds an O(dx²) interpolation error from sin(lat)
    # non-linearity; the former uses the exact f at corner positions.
    # Matches FV3 convention where f0 is stored at B-grid corners
    # (sw_core.F90 — absolute vorticity wk = vort + f0 at corners).
    if is_3d:
        zeta_corner = _interp_center_to_corner(zeta, cdgrid)
    else:
        zeta_corner = (_interp_center_to_corner(zeta, cdgrid)
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

    # 8. Divergence damping (FV3-style adaptive Smagorinsky).
    # Iter-758c revert: see fv3_sw_tendencies for the detailed
    # derivation.  Short version: adding `*dt` in the adaptive term
    # alone (iter-758) over-damps by dt× when the cap is active in
    # tendency form.  Compensating with `/dt` (iter-758b) fixes the
    # clip but under-damps the background d2_bg regime by dt×.  A
    # fully Fortran-faithful port needs the structural items 3+4
    # (corner divergence + KE-add) together, deferred to a dedicated
    # iter.  Iter-757b metric fix (da_min_c = area_corner) is
    # retained.
    # Iter-872c-take4 (Codex pass-4): gate reverted to narrow
    # (`if div_damp > 0:`) — see iter-872c-take4 comment in
    # `fv3_sw_tendencies`.  The kwarg `dddmp` (Fortran-strict
    # default 0.0) is kept so advanced users can override the
    # adaptive Smagorinsky coefficient when they explicitly enable
    # divergence damping via `div_damp > 0`.
    #
    # Iter-872c-take5 (Codex pass-5): symmetric warning for the
    # narrow-gate silent no-op when `dddmp > 0, div_damp = 0`.
    if dddmp > 0 and div_damp == 0:
        import warnings
        warnings.warn(
            f"`cdgrid_momentum_tendencies` called with `dddmp="
            f"{dddmp!r}` and `div_damp=0`.  The narrow gate "
            f"(iter-872c-take4) silently no-ops `dddmp` whenever "
            f"`div_damp == 0`; adaptive Smagorinsky damping is NOT "
            f"active.  To enable adaptive damping, also set "
            f"`div_damp > 0`.",
            stacklevel=2,
        )
    if div_damp > 0:
        div_field = cgrid_divergence(u_c, v_c, cdgrid)
        da_min_c = jnp.min(cdgrid.area_corner)    # Fortran da_min_c
        d2_bg = div_damp / da_min_c
        div_abs = jnp.abs(div_field)
        div_abs_corner = _interp_center_to_corner(div_abs, cdgrid)
        adaptive_coeff = da_min_c * jnp.maximum(
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

    # Iter-839 (Codex fidelity audit a5ae1319518698356): asymmetry
    # between u_c (face-normal projection) and v_c (plain average) was
    # flagged as a candidate Fortran-fidelity gap.  Applying a symmetric
    # `v_c = v_avg · sina_v − u_at_v · cosa_v` projection catastrophically
    # regressed W2 (L2 5e-02 vs baseline 2.2e-04, 230× worse) — tested at
    # C36 1d and failed `test_boundary_fix_is_load_bearing_for_w2_l2`.
    # The asymmetry is load-bearing: `u_c` has the face-normal
    # projection because mass transport via PPM requires edge-normal
    # velocity; `v_c` does NOT because its convention differs.
    # Fortran's `d2a2c_vect` uses pure covariant averages for BOTH
    # families (`uc = a2*utmp + a1*utmp`, `vc = a2*vtmp + a1*vtmp` —
    # sw_core.F90:3560, 3691) then derives the contravariant transport
    # velocities `ut`, `vt` downstream.  Our Python path collapses the
    # D→A→C sequence into a single `fv3_d2cc` + `fv3_cc2c` pair with a
    # different physical-vs-covariant convention.  A true Fortran-
    # faithful rewrite requires refactoring `cgrid_mass_flux_divergence`
    # to consume COVARIANT uc/vc and apply the `(uc − v·cosa_u)·rsin_u`
    # contravariant conversion internally — multi-iter architectural
    # change out of iter-839 scope.  Retained as open iter-840+
    # candidate.
    return u_c, v_c


def _fortran_agrid_vector_corner_fill(u_pad, v_pad):
    """Fortran `fill_corners_agrid_r8` (fv_mp_mod.F90:1433-1457) applied
    to an already-padded A-grid vector pair (u_pad, v_pad) at the 4
    cube-vertex halo cells of each face.

    Fortran VECTOR fill (mySign = -1) at the SW cube vertex:
        x(0,0) = -y(0,1)
        y(0,0) = -x(1,0)

    — the cube-corner halo cell of one component equals ± the
    edge-halo cell of the OTHER component (cross-component swap
    with sign flip).  The sign pattern across the 4 corners is
    {SW: -, SE: +, NW: +, NE: -}.

    Python halo=1 index mapping (padded shape (6, n+2, n+2),
    interior at [1..n, 1..n]):
      SW cube-corner halo padded[:, 0, 0]  ← Fortran qin(0, 0)
      SE cube-corner halo padded[:, -1, 0] ← Fortran qin(npx, 0)
      NW cube-corner halo padded[:, 0, -1] ← Fortran qin(0, npy)
      NE cube-corner halo padded[:, -1, -1]← Fortran qin(npx, npy)
      W-edge halo at j=1   padded[:, 0, 1]
      S-edge halo at i=1   padded[:, 1, 0]
      E-edge halo at j=1   padded[:, -1, 1]
      S-edge halo at i=npx-1 padded[:, -2, 0]
      (etc. by symmetry — see iter-767 review-doc entry.)

    Parameters
    ----------
    u_pad, v_pad : jax.Array, shape (6, n+2, n+2)
        A-grid padded vector components (grid-aligned, face-local
        frame) — typically the output of `pad_halo_vector`.

    Returns
    -------
    (u_pad_new, v_pad_new) : same shape
        Identical to inputs EXCEPT at the 4 cube-vertex halo cells
        per face, which are overwritten with Fortran's VECTOR
        `fill_corners_agrid_r8` values.
    """
    # SW: x(0,0) = -y(0,1), y(0,0) = -x(1,0)
    u_new_sw = -v_pad[:, 0, 1]
    v_new_sw = -u_pad[:, 1, 0]
    # SE (mySign=+1 for both components):
    #   x(npx, 0) = y(npx, 1),  y(npx, 0) = x(npx-1, 0)
    u_new_se = v_pad[:, -1, 1]
    v_new_se = u_pad[:, -2, 0]
    # NW (mySign=+1 for both):
    #   x(0, npy) = y(0, npy-1),  y(0, npy) = x(1, npy)
    u_new_nw = v_pad[:, 0, -2]
    v_new_nw = u_pad[:, 1, -1]
    # NE (mySign=-1 for both):
    #   x(npx, npy) = -y(npx, npy-1),  y(npx, npy) = -x(npx-1, npy)
    u_new_ne = -v_pad[:, -1, -2]
    v_new_ne = -u_pad[:, -2, -1]

    u_pad = u_pad.at[:, 0, 0].set(u_new_sw)
    u_pad = u_pad.at[:, -1, 0].set(u_new_se)
    u_pad = u_pad.at[:, 0, -1].set(u_new_nw)
    u_pad = u_pad.at[:, -1, -1].set(u_new_ne)
    v_pad = v_pad.at[:, 0, 0].set(v_new_sw)
    v_pad = v_pad.at[:, -1, 0].set(v_new_se)
    v_pad = v_pad.at[:, 0, -1].set(v_new_nw)
    v_pad = v_pad.at[:, -1, -1].set(v_new_ne)
    return u_pad, v_pad


def fv3_sw_tendencies(
    h, u_d, v_d, h_s, cdgrid,
    g=constants.g, div_damp=0.0, hyperdiff_coeff=0.0,
    boundary_fix=False,
    boundary_fix_skip_corners=False,
    zero_mean_correction=False,
    fortran_dir_aware_corners=False,
    fortran_a2b_corner_avg=False,
    fortran_vector_corner_fill=False,
    dddmp=0.0,
    apply_fortran_xppm_boundary=False,
    fortran_faithful_ppm_left=False,
    fortran_faithful_ppm_right=False,
    use_fv3_dsw1_mass_transport=False,
    dt=None,
    dsw1_nord=2,
    dsw1_damp_c=0.06,
    cube_edge_softer_div_damp=False,
    cube_edge_div_damp_factor=0.5,
    cube_edge_div_damp_band=2,
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

    # Iter-1019 Codex finding: `hyperdiff_coeff` appears in this
    # function's signature for API symmetry with
    # `cdgrid_momentum_tendencies` (which DOES implement it at
    # operators_cdgrid.py:1604-1614), but is NEVER applied inside
    # `fv3_sw_tendencies` itself.  Callers passing
    # `hyperdiff_coeff > 0` here would silently see no biharmonic
    # damping.  Warn loudly so users notice and either:
    #  (a) switch to `cdgrid_momentum_tendencies` /
    #      `CDGridShallowWaterModel`, which DOES apply hyperdiff, or
    #  (b) accept that `FV3EdgeShallowWaterModel`'s production path
    #      relies on `div_damp` + `damp_v` for damping (no biharmonic).
    if hyperdiff_coeff > 0:
        import warnings
        warnings.warn(
            f"`fv3_sw_tendencies(hyperdiff_coeff={hyperdiff_coeff!r})` "
            f"is silently ignored on this code path: the production "
            f"FV3 edge-midpoint path uses `div_damp` + `damp_v` for "
            f"damping, not biharmonic.  To get biharmonic hyperdiff "
            f"on a cubed-sphere SW path, use `CDGridShallowWaterModel` "
            f"(which routes through `cdgrid_momentum_tendencies` and "
            f"DOES apply `hyperdiff_coeff`).  This warning was added "
            f"in iter-1019 after a Codex fidelity audit found the "
            f"silent no-op.",
            UserWarning, stacklevel=2,
        )

    # (a) Cell-centre and C-grid velocities for mass transport
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)

    # (b) Height tendency
    if use_fv3_dsw1_mass_transport:
        # Iter-904: opt into the true-FV3 d_sw1 mass transport path
        # per `sw_core.F90:79` -> d_sw1 -> fv_tp_2d.  Uses
        # `_d2a2c_vect` to derive contravariant transport velocities
        # (ut, vt) and `transport_step` (= the FV3-style Lin-Rood
        # finite-volume update with PPM fluxes via `fv_tp_2d`).
        # `dt` MUST be provided in this branch — the FV3 transport is
        # a finite-volume update returning h_new, from which we
        # extract dh_dt = (h_new - h) / dt for the SSP-RK3 caller.
        # Default-OFF preserves bit-equality with the iter-892 path.
        #
        # Iter-904b (Codex iter-904 stop-time fix): forward
        # `dsw1_nord` and `dsw1_damp_c` to `transport_step`'s
        # `nord` / `damp_c` kwargs.  Fortran `sw_core.F90:886-887`
        # calls `fv_tp_2d(delp, ..., nord=nord_v, damp_c=damp_v)`
        # inside d_sw1 so the mass transport picks up the same
        # 4th-order del-n smoother the FB chain applies.  Pre-
        # iter-904b omission of these kwargs gave d_sw1 transport
        # without any damping, which is the likely root cause of
        # the iter-904 W2 NaN at C36 1-day.
        if dt is None:
            raise ValueError(
                "`use_fv3_dsw1_mass_transport=True` requires `dt` to "
                "be passed through `fv3_sw_tendencies`.  The "
                "production caller `FV3EdgeShallowWaterModel.step` "
                "forwards `dt` automatically when the flag is set; "
                "non-production callers must do the same.")
        from legoesm.core.fv3_sw_core import _d2a2c_vect
        from legoesm.core.fv_tp_2d import transport_step
        _, _, _, _, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)
        h_new = transport_step(
            h, ut, vt, dt, cdgrid,
            nord=dsw1_nord,
            damp_c=dsw1_damp_c,
            apply_fortran_xppm_boundary=apply_fortran_xppm_boundary)
        dh_dt = (h_new - h) / dt
    else:
        # Iter-889: forward apply_fortran_xppm_boundary so the
        # production CDGrid mass-flux PPM picks up Fortran's iord<7
        # cube-edge boundary overrides at tp_core.F90:357-369 when
        # opted in.
        dh_dt = cgrid_mass_flux_divergence(
            h, u_c, v_c, cdgrid,
            apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
            fortran_faithful_ppm_left=fortran_faithful_ppm_left,
            fortran_faithful_ppm_right=fortran_faithful_ppm_right)
    if zero_mean_correction:
        total_area = jnp.sum(cdgrid.base.area)
        dh_dt = dh_dt - jnp.sum(dh_dt * cdgrid.base.area) / total_area

    # (c) Bernoulli function using physical cell-centre winds
    KE = 0.5 * (u_cc ** 2 + v_cc ** 2)
    B = KE + g * (h + h_s)

    # (d) Arakawa-Lamb gradient at D-grid corners.
    # Iter-765: optionally use Fortran-faithful halo=1 inner dir-aware
    # corner fill (dir=1 for x-gradient, dir=2 for y-gradient) per
    # sw_core.F90:3856-3915.  See iter-764d for scope details.
    # Iter-766: optionally replace the 2-pt-avg cube-corner halo
    # cells with Fortran's `a2b_ord4` 3-pt-avg (a2b_edge.F90:385-388)
    # — the direction-neutral Fortran scalar corner average used in
    # the A-to-B interpolation feeding the pressure gradient.
    dB_dx, dB_dy_perp = _arakawa_lamb_gradient(
        B, cdgrid,
        fortran_dir_aware_corners=fortran_dir_aware_corners,
        fortran_a2b_corner_avg=fortran_a2b_corner_avg)

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
    # Iter-767: optionally overwrite the 4 cube-vertex halo cells of
    # (u_cc_pad, v_cc_pad) with Fortran's `fill_corners_agrid_r8`
    # VECTOR formula (fv_mp_mod.F90:1433-1457 with mySign=-1).
    # Fortran's direct cross-component swap+sign avoids the rotate-
    # pad-rotate mismatch at the 3-face cube vertex where face-local
    # grid angle is discontinuous.  See iter-767 review-doc entry.
    #
    # Iter-865 / iter-865b: gate this NON-DUOGRID corner override on
    # `not cdgrid.base.bounded_domain` per CLAUDE.md duogrid
    # constraint #2.  Fortran's `fill_corners_agrid_r8` is part of
    # the non-bounded-domain cube-vertex handling; in any
    # bounded_domain mode (duogrid OR regional panel) the cross-face
    # halo from `pad_halo_vector` (or Neumann wall BC) already
    # provides correct cube-vertex values and overwriting them with
    # the legacy cross-component formula would corrupt the
    # bounded-domain path.  iter-865b widens the gate from
    # `duogrid is None` to `not bounded_domain` per Codex stop-time
    # review (regional-panel runs would have hit the legacy path).
    if (fortran_vector_corner_fill
            and not cdgrid.base.bounded_domain):
        u_cc_pad, v_cc_pad = _fortran_agrid_vector_corner_fill(
            u_cc_pad, v_cc_pad)
    u_corner = 0.25 * (u_cc_pad[:, :-1, :-1] + u_cc_pad[:, 1:, :-1]
                        + u_cc_pad[:, :-1, 1:] + u_cc_pad[:, 1:, 1:])
    v_corner = 0.25 * (v_cc_pad[:, :-1, :-1] + v_cc_pad[:, 1:, :-1]
                        + v_cc_pad[:, :-1, 1:] + v_cc_pad[:, 1:, 1:])

    # (f) Absolute vorticity at CELL CENTRES (same stagger as the
    # momentum tendency below — no f interpolation).  Uses f at cell
    # centres directly.  Note: iter-74's cdgrid_momentum_tendencies fix
    # (corner-based tendency) switched to cdgrid.f_corner precisely to
    # avoid sin(lat) interp; here we stay at cell centres throughout so
    # no interp of f is needed and cdgrid.base.f is the right choice.
    zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
    zeta_abs = zeta + cdgrid.base.f

    # (g) Momentum tendencies at CELL CENTRES (better geostrophic balance).
    # Computing both gradient and vorticity at the same stagger (cell centres)
    # gives 2.6x better cancellation than at D-grid corners, because the
    # halo-exchanged fields have consistent interpolation errors at cell centres.
    dB_dx_cc = _interp_corner_to_center(dB_dx)
    dB_dy_cc = _interp_corner_to_center(dB_dy_perp)
    du_cc = zeta_abs * v_cc - dB_dx_cc      # (6, n, n)
    dv_cc = -zeta_abs * u_cc - dB_dy_cc     # (6, n, n)

    # (h) Divergence damping at cell centres.
    # Fortran reference: sw_core.F90:1720
    #   damp = gridstruct%da_min_c * max(d2_bg, min(0.20, dddmp*abs(delpc(i,j)*dt)))
    # Iter-755b/757: use B-grid da_min_c = min(area_corner).
    # Iter-758c (revert from 758/758b): the `*dt` factor in the adaptive
    # Smag term is NOT applied here.  Rationale: applying `*dt` inside
    # the cap alone (iter-758) makes the cap activate at |div|>0.003
    # instead of >1.0, and the RK3 tendency form over-damps by dt×
    # when capped.  Compensating with `/dt` (iter-758b) fixes the
    # clipped regime but UNDER-damps by dt× in the background
    # (d2_bg-dominated) regime.  Neither is correct in isolation —
    # the Fortran d_sw5 applies damp as a per-step u += damp*grad(div)
    # after the main d_sw6 update, which is a structural change
    # beyond the scope of a single-line fix.  Full Fortran-faithful
    # port requires pairing (*dt factor) with items 3 (corner
    # divergence delpc) and 4 (KE-add structure) — deferred to a
    # dedicated iter that ports d_sw5 holistically.
    # Iter-872c-take4 (Codex pass-4): gate is narrow
    # (``if div_damp > 0:``) — same as pre-iter-872b.  iter-872b
    # widened the gate to ``div_damp > 0 or dddmp > 0`` to enable
    # the Fortran-valid regime ``d2_bg = 0, dddmp > 0`` (pure
    # adaptive Smagorinsky), but Codex pass-4 correctly noted that
    # combined with the production `dddmp_prod = 0.2` default this
    # turned on adaptive damping for default-config callers — a
    # silent behavioural change for a code path that the comment
    # below documents as "structurally incomplete" (the *dt factor
    # and corner-divergence stencil are deferred to a dedicated
    # d_sw5 holistic port).  Reverting the gate to narrow restores
    # pre-iter-872b semantics while keeping the kwarg infrastructure
    # for the production path's explicit `dddmp_prod` opt-in.  The
    # widened-gate Fortran-fidelity improvement is deferred until
    # the d_sw5 port is complete; see iter-872c-take4 doc entry.
    #
    # Iter-872c-take5 (Codex pass-5): warn loudly when `dddmp > 0`
    # is supplied with `div_damp = 0`, because the narrow gate
    # silently no-ops `dddmp` in that regime.  Without this warning
    # users could set `dddmp_prod=0.4` and believe adaptive
    # Smagorinsky is active when the entire branch is bypassed.
    if dddmp > 0 and div_damp == 0:
        import warnings
        warnings.warn(
            f"`fv3_sw_tendencies` called with `dddmp={dddmp!r}` and "
            f"`div_damp=0`.  The narrow gate (iter-872c-take4) "
            f"silently no-ops `dddmp` whenever `div_damp == 0`; "
            f"adaptive Smagorinsky damping is NOT active.  To "
            f"enable adaptive damping, also set `div_damp > 0`.  "
            f"The Fortran-valid pure-adaptive regime "
            f"(`div_damp=0, dddmp>0`) is a deferred Fortran-fidelity "
            f"gap pending the holistic d_sw5 port.",
            stacklevel=2,
        )
    if div_damp > 0:
        div_field = cgrid_divergence(u_c, v_c, cdgrid)
        da_min_c = jnp.min(cdgrid.area_corner)    # Fortran da_min_c
        d2_bg = div_damp / da_min_c
        # Iter-872c: `dddmp` is now a kwarg with Fortran-strict
        # default 0.0 (Fortran fv_arrays.F90:360).  Production
        # callers pass 0.2 explicitly via
        # `CDGridShallowWaterConfig.dddmp_prod`
        # (`FV3EdgeShallowWaterModel.step`); direct callers that omit
        # `dddmp` get pure background-only damping when `div_damp>0`
        # and no damping at all when `div_damp=0`, matching Fortran's
        # strict-default semantics.  Pre-iter-872c the literal
        # was hardcoded 0.2 and silently leaked adaptive Smagorinsky
        # contributions into every direct call.
        div_abs = jnp.abs(div_field)
        adaptive_coeff = da_min_c * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * div_abs))

        # Iter-909 (default OFF): cube-edge-aware adaptive_coeff per
        # iter-908b's option (2).  iter-907 identified div_damp as
        # the dominant contributor (29x more than boundary_fix) to
        # t=0 dv_d_dt at the W2 D-grid hot spots (lat ±33.9° face
        # 0/2, i=2-3 row near EW cube edge).  iter-908b confirmed
        # that GLOBAL coefficient sweep cannot improve W2 below the
        # iter-892 0.132 m/s baseline.  iter-909 tests whether
        # SOFTENING div_damp ONLY at the cube-edge boundary cells
        # (i=0..band-1 and i=n-band..n-1, j=0..band-1 and
        # j=n-band..n-1) reduces hot-spot growth without the global
        # trade-off.
        #
        # Default OFF preserves iter-892/iter-893 production
        # behaviour bit-for-bit.  When True, the adaptive_coeff is
        # multiplied by `cube_edge_div_damp_factor` (default 0.5)
        # at boundary cells and by 1.0 at interior cells.  Boundary
        # band depth is `cube_edge_div_damp_band` cells (default 2,
        # covering i=0,1 and i=n-2,n-1).
        if cube_edge_softer_div_damp:
            band = int(cube_edge_div_damp_band)
            factor = float(cube_edge_div_damp_factor)
            mask = jnp.ones_like(adaptive_coeff)
            # Soften the band cells along i and j on each face.
            mask = mask.at[:, :band, :].set(factor)
            mask = mask.at[:, n - band:, :].set(factor)
            mask = mask.at[:, :, :band].set(factor)
            mask = mask.at[:, :, n - band:].set(factor)
            adaptive_coeff = adaptive_coeff * mask

        # Iter-765b: thread fortran_dir_aware_corners flag to this
        # A-L gradient call too, so the flag consistently affects ALL
        # A-L invocations inside fv3_sw_tendencies.
        # Iter-766: same discipline for fortran_a2b_corner_avg.
        ddiv_dx, ddiv_dy_perp_cc = _arakawa_lamb_gradient(
            div_field, cdgrid,
            fortran_dir_aware_corners=fortran_dir_aware_corners,
            fortran_a2b_corner_avg=fortran_a2b_corner_avg)
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

    # (j) Smooth face-boundary tendencies — non-FV3 stabilizer.
    # The geostrophic imbalance at boundary cells (rows 0 and n-1) is
    # ~7.7x larger than interior due to halo interpolation error in
    # corner winds (step (e)) and the Arakawa-Lamb gradient (step (d)).
    # Blending the boundary tendency with the adjacent interior cell
    # reduces this O(dx) error without altering balanced flows.
    #
    # **Iter-511 measurement** on the canonical W2 production harness
    # (C36, dt=60s, 1 day):
    #   boundary_fix=True:  L2 = 5.64e-4, Linf = 4.53e-3
    #   boundary_fix=False: L2 = 1.36e-3, Linf = 8.51e-3   (2.4x worse)
    # Locked by `tests/.../test_w2_alpha0_c16_1day_boundary_fix_load_bearing`
    # (iter-513) so this stabilizer cannot be silently disabled.
    #
    # This is a NON-FV3 hack (the FV3 d_sw_native chain achieves the
    # same effect via Fortran-faithful c_sw + flux-sync + d_sw5 corner
    # divergence damping; our A-L + RK3 production path is not
    # FV3-faithful, so the boundary cells need explicit smoothing).
    # Removing this is gated on the FB chain becoming stable at C36
    # (review-doc item #2: ng=3 halo infrastructure).
    #
    # Iter-865 / iter-865b: Fortran-faithful gating per CLAUDE.md
    # duogrid constraint #2 ("Legacy edge handling must be disabled
    # in duogrid mode via bounded_domain = .true.").  When the
    # bounded-domain flag is True (duogrid OR regional/nested
    # single-face panel), the cross-face halo placed by
    # `pad_halo_vector` (or Neumann wall BC on a panel) already
    # provides Fortran-faithful neighbour-face values at face
    # boundaries; smoothing same-face boundary cells with
    # adjacent-interior cells (this `boundary_fix` block) is the
    # legacy non-FV3 hack that should be bypassed.  In a
    # non-bounded-domain global cubed sphere (LEGACY) the smoothing
    # remains active — iter-511 measured it as load-bearing for W2
    # L2.  Codex iter-865 stop-time review flagged the original
    # iter-865 gate `cdgrid.base.duogrid is None` as hardcoding
    # duogrid-only and missing the regional-panel case; iter-865b
    # uses the proper `bounded_domain` flag from `cdgrid.base` per
    # the Fortran `fv_arrays.F90:1512` definition `bounded_domain =
    # (regional .or. nested .or. duogrid)`.  Single-face regional
    # panels now bypass the legacy edge handling correctly.
    if boundary_fix and (not cdgrid.base.bounded_domain) and n > 2:
        # Iter-769: optionally skip the 4 cube-corner cells [0,0],
        # [0,n-1], [n-1,0], [n-1,n-1].  The cascaded row-0/col-0 (and
        # row-n/col-n) smoothing causes corner cells to receive a
        # DOUBLE update — effectively a 4-point average of the 2×2
        # block at the corner.  Iter-762/768 localize mode A at cells
        # adjacent to the 8 cube vertices (which correspond to these
        # corner cells), so selectively skipping them isolates whether
        # the cascaded corner smoothing contributes to mode A.
        #
        # Fortran has NO post-tendency smoothing analog (confirmed by
        # Codex iter-769 review: no equivalent in sw_core.F90 / d_sw
        # routines).  boundary_fix is a Python-specific stabilizer;
        # reducing its scope is a step toward Fortran faithfulness.
        if boundary_fix_skip_corners:
            # Smooth only the INTERIOR of each boundary row/col, i.e.
            # leave the 4 corner cells [0,0], [0,n-1], [n-1,0],
            # [n-1,n-1] untouched.  Skip indices: col 0 and col n-1
            # for the row operations; row 0 and row n-1 for the col
            # operations.
            du_cc = du_cc.at[:, 0, 1:-1].set(
                0.5 * (du_cc[:, 0, 1:-1] + du_cc[:, 1, 1:-1]))
            du_cc = du_cc.at[:, n-1, 1:-1].set(
                0.5 * (du_cc[:, n-1, 1:-1] + du_cc[:, n-2, 1:-1]))
            du_cc = du_cc.at[:, 1:-1, 0].set(
                0.5 * (du_cc[:, 1:-1, 0] + du_cc[:, 1:-1, 1]))
            du_cc = du_cc.at[:, 1:-1, n-1].set(
                0.5 * (du_cc[:, 1:-1, n-1] + du_cc[:, 1:-1, n-2]))
            dv_cc = dv_cc.at[:, 0, 1:-1].set(
                0.5 * (dv_cc[:, 0, 1:-1] + dv_cc[:, 1, 1:-1]))
            dv_cc = dv_cc.at[:, n-1, 1:-1].set(
                0.5 * (dv_cc[:, n-1, 1:-1] + dv_cc[:, n-2, 1:-1]))
            dv_cc = dv_cc.at[:, 1:-1, 0].set(
                0.5 * (dv_cc[:, 1:-1, 0] + dv_cc[:, 1:-1, 1]))
            dv_cc = dv_cc.at[:, 1:-1, n-1].set(
                0.5 * (dv_cc[:, 1:-1, n-1] + dv_cc[:, 1:-1, n-2]))
        else:
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
    # Iter-767: same Fortran vector cube-corner fill applied to the
    # tendency projection halo exchange — keeps the treatment
    # consistent across every pad_halo_vector call inside this
    # function (both the wind halo in step (e) and the tendency
    # projection here in step (k)).
    # Iter-865 / iter-865b: same `not bounded_domain` gate as the
    # (e)-site call — `fortran_vector_corner_fill` is the non-
    # bounded-domain legacy corner formula and must NOT fire when
    # the grid is bounded_domain (duogrid or regional panel).
    if (fortran_vector_corner_fill
            and not cdgrid.base.bounded_domain):
        du_cc_pad, dv_cc_pad = _fortran_agrid_vector_corner_fill(
            du_cc_pad, dv_cc_pad)
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
