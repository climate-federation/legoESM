"""FV3-inspired shallow water forward-backward core (EXPERIMENTAL).

Implements a c_sw + d_sw forward-backward step adapted from GFDL's
FV3 dynamical core (sw_core.F90).  The operators use FV3's covariant
velocity convention with sin_sg flux scaling.

**Status**: experimental.  The full forward-backward step
(``fv3_fb_sw_step``) is known to be unstable — use ``fv3_sw_tendencies``
(operators_cdgrid.py) or ``fv3_csw_tendencies`` for production work.

The c_sw half-step operates at C-grid face positions (gradient and
vorticity at the same stagger — essential for geostrophic balance).
The d_sw half-step operates at D-grid edge-midpoint positions (reusing
the existing Arakawa-Lamb gradient which is consistent at that stagger).

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Mouallem, Harris & Chen (2023): Duo-Grid edge effect fix
- GFDL sw_core.F90: c_sw (lines 79-488), d2a2c_vect (lines 3006-3345)
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.fv_tp_2d import (
    _pert_ppm,
    compute_transport_quantities,
    fv_tp_2d,
    transport_step,
)
from legoesm.core.operators_cdgrid import (
    _pad_halo_auto,
    cgrid_mass_flux_divergence,
    _interp_center_to_corner,
    _interp_center_to_corner_a2b_ord4,
    cgrid_divergence,
    fv3_cc2c,
    fv3_d2cc,
)
from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.duogrid import ext_vector_dgrid
from legoesm.grids.halo import (
    pad_halo,
    pad_halo_vector,
    synchronize_bgrid_ne_corner_geo,
    synchronize_cgrid_fluxes,
)

_EPS = float(jnp.finfo(jnp.float32).eps)


def _pad_halo_dgrid_for_ppm(u_d, v_d, cdgrid, halo: int = 2):
    """Iter-945: cross-face halo for D-grid winds for PPM transport.

    Equivalent to a single-rank ``mpp_update_domains(u_d, v_d, gridtype=
    DGRID_NE)`` with the duogrid `cube_rmp` interpolation: fills cube-
    face halo cells of (u_d, v_d) with rotated cross-face data using
    the existing ``ext_vector_dgrid`` pipeline.  The interior is
    overwritten EXACTLY with the input u_d/v_d so the function is the
    identity on the interior region.

    Output shapes are sliced to match what `_bgrid_ke_transport` PPM
    transport needs:
      * u_d_ihalo: (6, n + 2*halo, n+1)  — i-cells extended (cell axis 1)
        with j-stagger preserved.  Used for ``_ppm_transport_1d(u_d_ihalo,
        ub, rdx, axis=1, external_halo=halo)`` in the d_sw3 x-sweep.
      * v_d_jhalo: (6, n+1, n + 2*halo) — j-cells extended (cell axis 2)
        with i-stagger preserved.  Used for ``_ppm_transport_1d(v_d_jhalo,
        vb, rdy, axis=2, external_halo=halo)`` in the d_sw3 y-sweep.

    Parameters
    ----------
    u_d : (6, n, n+1) — D-grid x-velocity (i-cells, j-stagger)
    v_d : (6, n+1, n) — D-grid y-velocity (i-stagger, j-cells)
    cdgrid : CubedSphereCDGrid (must have an active duogrid with ng>=halo)
    halo : int — halo width.  Must be ≤ the duogrid's available ng.

    Returns
    -------
    u_d_ihalo : (6, n + 2*halo, n+1)
    v_d_jhalo : (6, n+1, n + 2*halo)
    """
    n = cdgrid.n
    h = halo
    grid = cdgrid.base
    dg = grid.duogrid
    if dg is None or dg.ng < h:
        raise ValueError(
            f"_pad_halo_dgrid_for_ppm: requires duogrid with ng>={h}, "
            f"got ng={None if dg is None else dg.ng}."
        )

    # Same A-grid input prep as _d2a2c_vect_duogrid step 1: 2nd-order
    # length-weighted D→A average to A-grid covariant utmp/vtmp.
    dx_u = cdgrid.dx_edge_y  # (6, n, n+1) — x-length at u_d positions
    dy_v = cdgrid.dy_edge_x  # (6, n+1, n) — y-length at v_d positions
    wu = u_d * dx_u
    wv = v_d * dy_v
    utmp_2nd = (wu[:, :, :-1] + wu[:, :, 1:]) / (
        dx_u[:, :, :-1] + dx_u[:, :, 1:])  # (6, n, n)
    vtmp_2nd = (wv[:, :-1, :] + wv[:, 1:, :]) / (
        dy_v[:, :-1, :] + dy_v[:, 1:, :])  # (6, n, n)
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell

    u_d_full, v_d_full = ext_vector_dgrid(
        utmp_2nd, vtmp_2nd, dg,
        grid.cos_angle, grid.sin_angle,
        cos_sg5, rsin2,
        halo=h,
    )  # u_d_full: (6, n+2h, n+2h-1); v_d_full: (6, n+2h-1, n+2h)

    # Overwrite interior with EXACT input u_d/v_d — mirrors
    # `_d2a2c_vect_duogrid` and matches Fortran's mpp_update_domains
    # which only fills halo cells.
    u_d_full = u_d_full.at[:, h:h + n, h - 1:h + n].set(u_d)
    v_d_full = v_d_full.at[:, h - 1:h + n, h:h + n].set(v_d)

    # u_d_ihalo: (6, n+2h, n+1) — slice j-stagger to interior n+1.
    # u_d_full's j-stagger axis has n+2h-1 elements with index map
    # j_padded = j_cdgrid + (h - 1).  Interior j ∈ [0, n] maps to
    # padded indices [h-1, h+n-1] — slice [h-1 : h+n].
    u_d_ihalo = u_d_full[:, :, h - 1:h + n]

    # v_d_jhalo: (6, n+1, n+2h) — slice i-stagger to interior n+1.
    v_d_jhalo = v_d_full[:, h - 1:h + n, :]

    return u_d_ihalo, v_d_jhalo


def _pad_halo_uc_vc_via_d2a2c(u_d, v_d, cdgrid):
    """Iter-946 (NEGATIVE-RESULT helper): cross-face halo for covariant
    uc, vc derived from OLD u_d, v_d via the 4th-order d2a2c machinery.

    Reuses ``_d2a2c_vect_duogrid``'s 4th-order machinery to derive uc
    with j-halo (1 cell on each side) and vc with i-halo (1 cell on
    each side) WITHOUT going through a lossy cell-centre roundtrip
    (cf. iter-836b/iter-837 inside `_corner_vorticity` which DOES use
    a cell-centre 2-point average — that pattern was tried for d_sw
    in iter-945's pre-commit prototype and rejected because it
    introduced a 1-2-1 smoothing that worsened W2 fidelity).

    Iter-946 found this helper PROBLEMATIC for d_sw: the derived halo
    is from OLD u_d, v_d (passed to `_d_sw_native`) and does NOT
    include the ``c_sw + p_grad_c`` increment that was added to the
    INTERIOR uc, vc.  For W2 solid-body the missing increment is
    O(dt2 * gradient) ~ 15 m/s on vc, comparable to the velocity
    magnitude itself.  Mixing OLD-derived halo cells with
    NEW-modified interior cells in the d_sw1/d_sw3 4-cell averages
    produces a discontinuity at boundaries that gives WORSE results
    than ``mode='edge'`` (which keeps boundary cells consistent at
    the cost of a wrong cross-face geometry).

    Iter-946 measurement on duogrid C36 W2 1-day FB chain:
      - mode='edge' baseline (iter-945):        |u|=78,  |v|=81 m/s
      - iter-946 d2a2c halo in d_sw1+d_sw3:     |u|=73,  |v|=156 m/s ← WORSE
      - iter-946 d2a2c halo in d_sw3 only:      |u|=70,  |v|=150 m/s ← WORSE

    Reverted; the helper is retained for future work that ALSO
    propagates the c_sw + p_grad_c increments to halo (iter-947+
    candidate).  Without that companion fix, this helper introduces
    a Fortran-unfaithful OLD/NEW mismatch.

    Requires duogrid with ng≥3 (for the j-halo extension on the
    4-point j-stencil applied to ``u_d_full``).  Raises ``ValueError``
    on insufficient halo or non-duogrid grids.

    Parameters
    ----------
    u_d : (6, n, n+1) — D-grid x-velocity (i-cells, j-stagger)
    v_d : (6, n+1, n) — D-grid y-velocity (i-stagger, j-cells)
    cdgrid : CubedSphereCDGrid (must have an active duogrid with ng≥3)

    Returns
    -------
    uc_jhalo : (6, n+1, n+2) — covariant uc with j-halo of width 1
        (j=-1 prepended, j=n appended) along axis 2.
    vc_ihalo : (6, n+2, n+1) — covariant vc with i-halo of width 1
        (i=-1 prepended, i=n appended) along axis 1.
    """
    n = cdgrid.n
    grid = cdgrid.base
    dg = grid.duogrid
    if dg is None or dg.ng < 3:
        raise ValueError(
            f"_pad_halo_uc_vc_via_d2a2c: requires duogrid with ng>=3 "
            f"(j-halo extension on the 4-point j-stencil); "
            f"got ng={None if dg is None else dg.ng}."
        )
    h = 3

    # Step 1: same A-grid prep as `_d2a2c_vect_duogrid` (length-weighted
    # 2nd-order D→A average to A-grid covariant utmp_2nd, vtmp_2nd).
    dx_u = cdgrid.dx_edge_y
    dy_v = cdgrid.dy_edge_x
    wu = u_d * dx_u
    wv = v_d * dy_v
    utmp_2nd = (wu[:, :, :-1] + wu[:, :, 1:]) / (
        dx_u[:, :, :-1] + dx_u[:, :, 1:])  # (6, n, n)
    vtmp_2nd = (wv[:, :-1, :] + wv[:, 1:, :]) / (
        dy_v[:, :-1, :] + dy_v[:, 1:, :])  # (6, n, n)
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell

    u_d_full, v_d_full = ext_vector_dgrid(
        utmp_2nd, vtmp_2nd, dg,
        grid.cos_angle, grid.sin_angle,
        cos_sg5, rsin2,
        halo=h,
    )  # u_d_full: (6, n+2h, n+2h-1); v_d_full: (6, n+2h-1, n+2h)

    # Overwrite interior with EXACT input u_d/v_d (matches Fortran's
    # mpp_update_domains which only fills halo cells).
    u_d_full = u_d_full.at[:, h:h + n, h - 1:h + n].set(u_d)
    v_d_full = v_d_full.at[:, h - 1:h + n, h:h + n].set(v_d)

    # Step 2: utmp at i-halo full AND j-halo=1 (cells j ∈ [-1, n]).
    # 4-point j-stencil on u_d_full's j-stagger axis with indices
    # extended by 1 on each side compared to `_d2a2c_vect_duogrid`'s
    # interior-j slicing.  Padded j-stagger range needed: [h-3, h+n+1]
    # (max index h+n+1 = n+4 for h=3); u_d_full j-axis has n+2h-1 =
    # n+5 elements (indices 0..n+4) — exactly fits.  Requires h≥3
    # (h=2 would underflow at [h-3] = -1).
    utmp_T = (
        _A2 * (u_d_full[:, :, h - 3:h - 3 + n + 2]
               + u_d_full[:, :, h:h + n + 2])
        + _A1 * (u_d_full[:, :, h - 2:h - 2 + n + 2]
                 + u_d_full[:, :, h - 1:h - 1 + n + 2])
    )  # (6, n+2h, n+2)

    # Step 3: 4th-order A→C i-stencil on utmp_T → uc with j-halo=1.
    # Same I-face stencil as `_d2a2c_vect_duogrid` Step 4, shape
    # preserved.  The j-axis already carries the halo from utmp_T.
    uc_jhalo = (
        _A2 * (utmp_T[:, h - 2:h - 2 + n + 1, :]
               + utmp_T[:, h + 1:h + 1 + n + 1, :])
        + _A1 * (utmp_T[:, h - 1:h - 1 + n + 1, :]
                 + utmp_T[:, h:h + n + 1, :])
    )  # (6, n+1, n+2)

    # Step 4: vtmp at j-halo full AND i-halo=1 (cells i ∈ [-1, n]).
    # Symmetric to utmp_T construction, swapping the i-axis stencil
    # for the j-axis.
    vtmp_T = (
        _A2 * (v_d_full[:, h - 3:h - 3 + n + 2, :]
               + v_d_full[:, h:h + n + 2, :])
        + _A1 * (v_d_full[:, h - 2:h - 2 + n + 2, :]
                 + v_d_full[:, h - 1:h - 1 + n + 2, :])
    )  # (6, n+2, n+2h)

    # Step 5: 4th-order A→C j-stencil on vtmp_T → vc with i-halo=1.
    vc_ihalo = (
        _A2 * (vtmp_T[:, :, h - 2:h - 2 + n + 1]
               + vtmp_T[:, :, h + 1:h + 1 + n + 1])
        + _A1 * (vtmp_T[:, :, h - 1:h - 1 + n + 1]
                 + vtmp_T[:, :, h:h + n + 1])
    )  # (6, n+2, n+1)

    return uc_jhalo, vc_ihalo


def _pad_halo_uc_vc_new_via_old_delta(uc, vc, u_d, v_d, cdgrid):
    """Iter-947: NEW-uc, NEW-vc halo via OLD-halo cross-face delta.

    Combines the iter-946 d2a2c-derived OLD halo with the NEW interior
    uc, vc passed to ``_d_sw_native`` to produce a halo that:

    1. Carries the cross-face geometric delta (rotation between cube
       faces) from the 4th-order d2a2c machinery.
    2. Includes the c_sw + p_grad_c increment by anchoring on the NEW
       interior boundary cell.

    The formula at each halo cell ``H`` adjacent to interior cell ``B``::

        uc_NEW_halo[H] = uc[B]_NEW + (uc_OLD_halo[H] - uc_OLD[B])
                       = NEW_boundary + cross_face_delta_from_OLD

    For W2 solid-body the OLD-derived cross-face delta and the c_sw +
    p_grad_c increment are nearly orthogonal contributions (the delta
    is smooth across faces; the increment is smooth on a face), so
    summing them gives a good approximation of the NEW cross-face halo.

    Iter-946 (NEGATIVE-RESULT) showed that using the OLD halo directly
    introduces an OLD/NEW mismatch in the d_sw 4-cell averages that
    worsens W2 |v_max|.  This iter-947 helper anchors on NEW interior
    so the halo is consistent with the rest of the d_sw call.

    Requires duogrid with ng>=3 (same as `_pad_halo_uc_vc_via_d2a2c`).

    Parameters
    ----------
    uc : (6, n+1, n) — NEW covariant C-grid u (post-c_sw + p_grad_c)
    vc : (6, n, n+1) — NEW covariant C-grid v
    u_d : (6, n, n+1) — D-grid x-velocity at d_sw entry (OLD u_d)
    v_d : (6, n+1, n) — D-grid y-velocity at d_sw entry (OLD v_d)
    cdgrid : CubedSphereCDGrid (must have an active duogrid with ng>=3)

    Returns
    -------
    uc_pad : (6, n+1, n+2) — uc with NEW-consistent j-halo (j=-1, j=n)
    vc_pad : (6, n+2, n+1) — vc with NEW-consistent i-halo (i=-1, i=n)
    """
    n = cdgrid.n

    # OLD halo via the iter-946 d2a2c machinery.
    uc_old_jhalo, vc_old_ihalo = _pad_halo_uc_vc_via_d2a2c(u_d, v_d, cdgrid)
    # uc_old_jhalo: (6, n+1, n+2) — index 0 → j=-1, index n+1 → j=n.
    # vc_old_ihalo: (6, n+2, n+1) — index 0 → i=-1, index n+1 → i=n.

    # OLD interior uc, vc (the d2a2c output on OLD u_d, v_d).  Used
    # ONLY at the boundary cells to compute the cross-face geometric
    # delta; the NEW interior is preserved everywhere else.  Iter-948
    # tested a 2-point linear extrapolation of the c_sw + p_grad_c
    # increment to halo cells (in place of the constant iter-947
    # extrapolation) and found it WORSENED W2 |v_max| at C36 from
    # 75 → 87 m/s — the increment varies non-linearly along the
    # face's j-direction near cube vertices, so a linear stencil
    # overshoots.  Constant (iter-947) extrap retained.
    _, _, uc_old_int, vc_old_int, _, _ = _d2a2c_vect(u_d, v_d, cdgrid)
    # uc_old_int: (6, n+1, n) — interior j ∈ [0, n-1]
    # vc_old_int: (6, n, n+1) — interior i ∈ [0, n-1]

    # uc south halo (j=-1): cross-face delta = uc_OLD_jhalo[j=-1] - uc_OLD[j=0]
    delta_uc_south = uc_old_jhalo[:, :, 0:1] - uc_old_int[:, :, 0:1]
    # uc north halo (j=n):   cross-face delta = uc_OLD_jhalo[j=n] - uc_OLD[j=n-1]
    delta_uc_north = uc_old_jhalo[:, :, n + 1:n + 2] - uc_old_int[:, :, n - 1:n]

    uc_new_south = uc[:, :, 0:1] + delta_uc_south    # NEW boundary + cross-face delta
    uc_new_north = uc[:, :, n - 1:n] + delta_uc_north
    uc_pad = jnp.concatenate([uc_new_south, uc, uc_new_north], axis=2)
    # uc_pad: (6, n+1, n+2)

    # vc west halo (i=-1):
    delta_vc_west = vc_old_ihalo[:, 0:1, :] - vc_old_int[:, 0:1, :]
    # vc east halo (i=n):
    delta_vc_east = vc_old_ihalo[:, n + 1:n + 2, :] - vc_old_int[:, n - 1:n, :]

    vc_new_west = vc[:, 0:1, :] + delta_vc_west
    vc_new_east = vc[:, n - 1:n, :] + delta_vc_east
    vc_pad = jnp.concatenate([vc_new_west, vc, vc_new_east], axis=1)
    # vc_pad: (6, n+2, n+1)

    return uc_pad, vc_pad


def _d_sw1_recompute_ut_vt(uc, vc, cdgrid, dt,
                            u_d_old=None, v_d_old=None):
    """FV3 d_sw1 transport velocity recomputation with boundary handling.

    Recomputes contravariant transport velocities (ut, vt) from covariant
    C-grid velocities (uc, vc) using FV3's 4-cell cross-velocity average,
    face-boundary overrides (sin_sg upwind), adjacent strip recomputation,
    and corner 2×2 solve. Matches sw_core.F90:618-812.

    This replaces the ut/vt from d2a2c_vect with more accurate transport
    velocities that have cross-face consistent boundary handling.

    Parameters
    ----------
    uc : (6, n+1, n) covariant C-grid u
    vc : (6, n, n+1) covariant C-grid v
    cdgrid : CubedSphereCDGrid
    dt : float — time step (for upwind sign test)
    u_d_old, v_d_old : optional D-grid winds at d_sw entry (OLD u_d,
        OLD v_d).  Iter-947: when both are provided AND duogrid ng>=3,
        the vc/uc halo cells used in the 4-cell averages are sourced
        from `_pad_halo_uc_vc_new_via_old_delta` (NEW-corrected
        cross-face halo) instead of `mode='edge'`.

    Returns
    -------
    ut : (6, n+1, n) contravariant transport u
    vt : (6, n, n+1) contravariant transport v
    """
    n = cdgrid.n
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2

    cosa_u = cdgrid.cosa_u  # (6, n+1, n)
    cosa_v = cdgrid.cosa_v  # (6, n, n+1)
    rsin_u = cdgrid.rsin_u  # (6, n+1, n)
    rsin_v = cdgrid.rsin_v  # (6, n, n+1)
    sg = cdgrid.sin_sg

    # === Part 1: Interior ut/vt from 4-cell vc/uc average ===
    # ut(I,j) = (uc(I,j) - 0.25*cosa_u*(vc(I-1,j)+vc(I,j)+vc(I-1,j+1)+vc(I,j+1)))*rsin_u
    # Need vc padded in axis 1 (rows) for the I-1 stencil; uc padded in
    # axis 2 (cols) for the J-1 stencil.
    #
    # Iter-946 (Fortran-fidelity fix): on the duogrid path with ng>=3,
    # source the vc/uc halo cells from the OLD u_d, v_d via
    # `_pad_halo_uc_vc_via_d2a2c` (4th-order d2a2c machinery) instead
    # of `mode='edge'` same-face replication.  HALO ROWS only are
    # replaced; the interior is preserved exactly so the 4-cell
    # averages at interior I-faces / J-faces are bit-identical to
    # pre-iter-946.  Caveat: derived halo lacks the c_sw + p_grad_c
    # increment that was added to interior uc, vc — small for W2
    # geostrophic balance, but iter-947+ may close that O(dt2 * grad)
    # gap by extending c_sw / p_grad_c to halo positions.
    # iter-947 (Fortran-fidelity fix): use NEW-corrected halo via
    # `_pad_halo_uc_vc_new_via_old_delta`.  See helper docstring
    # for the formula NEW_halo = NEW_boundary + (OLD_halo - OLD_boundary)
    # which carries the OLD cross-face geometric delta while
    # preserving the c_sw + p_grad_c increment.  Requires u_d, v_d
    # (the OLD inputs to d_sw — already available as parameters of
    # `_d_sw1_recompute_ut_vt`'s caller `_d_sw_native`).  Iter-947's
    # caller in `_d_sw_native` forwards them via `u_d_old`/`v_d_old`
    # kwargs introduced for this purpose.
    if (use_duogrid and dg.ng >= 3
            and u_d_old is not None and v_d_old is not None):
        uc_pad, vc_pad = _pad_halo_uc_vc_new_via_old_delta(
            uc, vc, u_d_old, v_d_old, cdgrid)
    else:
        vc_pad = jnp.pad(vc, [(0, 0), (1, 1), (0, 0)], mode='edge')
        uc_pad = jnp.pad(uc, [(0, 0), (0, 0), (1, 1)], mode='edge')
    # vc_pad[:, I, :] = vc at padded row I → original row I-1
    # 4-cell average at each u-face (I, j): vc(I-1,j)+vc(I,j)+vc(I-1,j+1)+vc(I,j+1)
    vc_avg = (vc_pad[:, :-1, :-1] + vc_pad[:, 1:, :-1]
              + vc_pad[:, :-1, 1:] + vc_pad[:, 1:, 1:])  # (6, n+1, n)
    ut = (uc - 0.25 * cosa_u * vc_avg) * rsin_u

    # vt(i,J) = (vc(i,J) - 0.25*cosa_v*(uc(i,J-1)+uc(i+1,J-1)+uc(i,J)+uc(i+1,J)))*rsin_v
    uc_avg = (uc_pad[:, :-1, :-1] + uc_pad[:, 1:, :-1]
              + uc_pad[:, :-1, 1:] + uc_pad[:, 1:, 1:])  # (6, n, n+1)
    vt = (vc - 0.25 * cosa_v * uc_avg) * rsin_v

    # Iter-953 (NEGATIVE-RESULT): tested running Parts 2/3/4 for
    # duogrid too (early return removed).  Result on duogrid C36 W2
    # 1-day: v_ll_Linf 55.6 → 127.2 m/s (regression of 130 %).
    # Parts 2/3/4 use `grid.halo_interp_offsets` (non-duogrid mode)
    # for sin_sg padding, which conflicts with the duogrid cube_rmp
    # halo of uc, vc.
    #
    # Iter-954 (NEGATIVE-RESULT): tried just the sin_sg-upwind
    # override at I=0/n, J=0/n with duogrid-aware sin_sg padding —
    # ALSO regressed v_ll_Linf to 133 m/s.  The Part 2 formula
    # ``ut = uc / sin_sg(upwind)`` replaces Part 1's 4-cell average
    # but DROPS the cross-velocity ``-0.25*cosa_u*vc_avg`` term;
    # with iter-947's correct halo, Part 1's full formula is
    # preferable to Part 2's simpler boundary override.  Reverted.
    if use_duogrid:
        return ut, vt

    # === Part 2: Non-duogrid face-boundary overrides ===
    # West face (I=0): ut = uc / sin_sg(upwind)
    sin_w_left = sg[:, :, :, 2]   # E-edge of cell to the left
    sin_w_right = sg[:, :, :, 0]  # W-edge of cell to the right
    # Pad sin_sg for cross-face upwind selection
    grid = cdgrid.base
    offsets = grid.halo_interp_offsets
    se_pad = pad_halo(sin_w_left, interp_offsets=offsets)
    sw_pad = pad_halo(sin_w_right, interp_offsets=offsets)

    # Face boundary at I=0 (Python) = Fortran I=1
    sin_upwind_w = jnp.where(uc[:, 0, :] * dt > 0,
                             se_pad[:, :n+1, 1:-1][:, 0, :],
                             sw_pad[:, 1:n+2, 1:-1][:, 0, :])
    ut = ut.at[:, 0, :].set(uc[:, 0, :] / jnp.maximum(jnp.abs(sin_upwind_w), _EPS))

    # East face (I=n)
    sin_upwind_e = jnp.where(uc[:, n, :] * dt > 0,
                             se_pad[:, :n+1, 1:-1][:, n, :],
                             sw_pad[:, 1:n+2, 1:-1][:, n, :])
    ut = ut.at[:, n, :].set(uc[:, n, :] / jnp.maximum(jnp.abs(sin_upwind_e), _EPS))

    # South face (J=0): vt = vc / sin_sg(upwind)
    sin_s_below = sg[:, :, :, 3]  # N-edge of cell below
    sin_s_above = sg[:, :, :, 1]  # S-edge of cell above
    sn_pad = pad_halo(sin_s_below, interp_offsets=offsets)
    ss_pad = pad_halo(sin_s_above, interp_offsets=offsets)

    sin_upwind_s = jnp.where(vc[:, :, 0] * dt > 0,
                             sn_pad[:, 1:-1, :n+1][:, :, 0],
                             ss_pad[:, 1:-1, 1:n+2][:, :, 0])
    vt = vt.at[:, :, 0].set(vc[:, :, 0] / jnp.maximum(jnp.abs(sin_upwind_s), _EPS))

    # North face (J=n)
    sin_upwind_n = jnp.where(vc[:, :, n] * dt > 0,
                             sn_pad[:, 1:-1, :n+1][:, :, n],
                             ss_pad[:, 1:-1, 1:n+2][:, :, n])
    vt = vt.at[:, :, n].set(vc[:, :, n] / jnp.maximum(jnp.abs(sin_upwind_n), _EPS))

    # === Part 3: Adjacent strip recomputation (vectorized) ===
    # FV3 sw_core.F90:666-726. Recompute vt/ut near face boundaries
    # using corrected boundary ut/vt values.
    if n > 4:
        jlo = 2; jhi = n - 1  # interior J range for vt recomputation

        # West: vt at rows 0, 1 for J=jlo..jhi
        for row in [0, 1]:
            # ut 4-cell average at each v-face J: ut(row,J-1)+ut(row+1,J-1)+ut(row,J)+ut(row+1,J)
            avg = (ut[:, row, jlo-1:jhi] + ut[:, row+1, jlo-1:jhi]
                   + ut[:, row, jlo:jhi+1] + ut[:, row+1, jlo:jhi+1])  # (6, jhi-jlo+1)
            vt = vt.at[:, row, jlo:jhi+1].set(
                vc[:, row, jlo:jhi+1] - 0.25 * cosa_v[:, row, jlo:jhi+1] * avg)

        # East: vt at rows n-2, n-1
        for row in [n-2, n-1]:
            if row >= 0 and row + 1 <= n:
                avg = (ut[:, row, jlo-1:jhi] + ut[:, row+1, jlo-1:jhi]
                       + ut[:, row, jlo:jhi+1] + ut[:, row+1, jlo:jhi+1])
                vt = vt.at[:, row, jlo:jhi+1].set(
                    vc[:, row, jlo:jhi+1] - 0.25 * cosa_v[:, row, jlo:jhi+1] * avg)

        # South: ut at cols 0, 1 for I=ilo..ihi
        ilo = 2; ihi = n - 1
        for col in [0, 1]:
            avg = (vt[:, ilo-1:ihi, col] + vt[:, ilo:ihi+1, col]
                   + vt[:, ilo-1:ihi, col+1] + vt[:, ilo:ihi+1, col+1])
            ut = ut.at[:, ilo:ihi+1, col].set(
                uc[:, ilo:ihi+1, col] - 0.25 * cosa_u[:, ilo:ihi+1, col] * avg)

        # North: ut at cols n-1, n
        for col in [n-1, n]:
            if col - 1 >= 0 and col <= n:
                avg = (vt[:, ilo-1:ihi, col-1] + vt[:, ilo:ihi+1, col-1]
                       + vt[:, ilo-1:ihi, col] + vt[:, ilo:ihi+1, col])
                ut = ut.at[:, ilo:ihi+1, col].set(
                    uc[:, ilo:ihi+1, col] - 0.25 * cosa_u[:, ilo:ihi+1, col] * avg)

    # === Part 4: Corner 2×2 solve ===
    # At each cube vertex, solve a coupled system for the ut/vt values
    # near the corner. FV3 sw_core.F90:739-811.
    # For the SW corner:
    #   damp = 1/(1 - 0.0625*cosa_u(2,1)*cosa_v(1,2))
    #   ut(2,1) = (uc(2,1) - 0.25*cosa_u(2,1)*(vt(1,1)+vt(2,1)+vt(2,2)+vc(1,2)
    #              - 0.25*cosa_v(1,2)*(ut(1,1)+ut(1,2)+ut(2,2)))) * damp
    # In Python 0-based: Fortran (2,1) → Python (1,0)

    # SW corner: Fortran (2,1) → Python ut(1, 0), vt(0, 1)
    cu = cosa_u  # (6, n+1, n)
    cv = cosa_v  # (6, n, n+1)

    # Interior solve: Fortran ut(2,1) → Python ut[:, 1, 0]
    damp = 1.0 / (1.0 - 0.0625 * cu[:, 1, 0] * cv[:, 0, 1])
    ut = ut.at[:, 1, 0].set(
        (uc[:, 1, 0] - 0.25 * cu[:, 1, 0] * (
            vt[:, 0, 0] + vt[:, 1, 0] + vt[:, 1, 1] + vc[:, 0, 1]
            - 0.25 * cv[:, 0, 1] * (ut[:, 0, 0] + ut[:, 0, 1] + ut[:, 1, 1])
        )) * damp)
    vt = vt.at[:, 0, 1].set(
        (vc[:, 0, 1] - 0.25 * cv[:, 0, 1] * (
            ut[:, 0, 0] + ut[:, 0, 1] + ut[:, 1, 1] + uc[:, 1, 0]
            - 0.25 * cu[:, 1, 0] * (vt[:, 0, 0] + vt[:, 1, 0] + vt[:, 1, 1])
        )) * damp)

    # SE corner: Fortran ut(npx-1,1) → Python ut[:, n-1, 0], vt(npx-1,2) → vt[:, n-2, 1]
    damp = 1.0 / (1.0 - 0.0625 * cu[:, n-1, 0] * cv[:, n-2, 1])
    ut = ut.at[:, n-1, 0].set(
        (uc[:, n-1, 0] - 0.25 * cu[:, n-1, 0] * (
            vt[:, n-2, 0] + vt[:, n-3, 0] + vt[:, n-3, 1] + vc[:, n-2, 1]
            - 0.25 * cv[:, n-2, 1] * (ut[:, n, 0] + ut[:, n, 1] + ut[:, n-1, 1])
        )) * damp)
    vt = vt.at[:, n-2, 1].set(
        (vc[:, n-2, 1] - 0.25 * cv[:, n-2, 1] * (
            ut[:, n, 0] + ut[:, n, 1] + ut[:, n-1, 1] + uc[:, n-1, 0]
            - 0.25 * cu[:, n-1, 0] * (vt[:, n-2, 0] + vt[:, n-3, 0] + vt[:, n-3, 1])
        )) * damp)

    # NE corner: Fortran ut(npx-1,npy-1) → Python ut[:, n-1, n-1], vt(npx-1,npy-1) → vt[:, n-2, n-1]
    damp = 1.0 / (1.0 - 0.0625 * cu[:, n-1, n-1] * cv[:, n-2, n-1])
    ut = ut.at[:, n-1, n-1].set(
        (uc[:, n-1, n-1] - 0.25 * cu[:, n-1, n-1] * (
            vt[:, n-2, n] + vt[:, n-3, n] + vt[:, n-3, n-1] + vc[:, n-2, n-1]
            - 0.25 * cv[:, n-2, n-1] * (ut[:, n, n-1] + ut[:, n, n-2] + ut[:, n-1, n-2])
        )) * damp)
    vt = vt.at[:, n-2, n-1].set(
        (vc[:, n-2, n-1] - 0.25 * cv[:, n-2, n-1] * (
            ut[:, n, n-1] + ut[:, n, n-2] + ut[:, n-1, n-2] + uc[:, n-1, n-1]
            - 0.25 * cu[:, n-1, n-1] * (vt[:, n-2, n] + vt[:, n-3, n] + vt[:, n-3, n-1])
        )) * damp)

    # NW corner: Fortran ut(2,npy-1) → Python ut[:, 1, n-1], vt(1,npy-1) → vt[:, 0, n-1]
    damp = 1.0 / (1.0 - 0.0625 * cu[:, 1, n-1] * cv[:, 0, n-1])
    ut = ut.at[:, 1, n-1].set(
        (uc[:, 1, n-1] - 0.25 * cu[:, 1, n-1] * (
            vt[:, 0, n] + vt[:, 1, n] + vt[:, 1, n-1] + vc[:, 0, n-1]
            - 0.25 * cv[:, 0, n-1] * (ut[:, 0, n-1] + ut[:, 0, n-2] + ut[:, 1, n-2])
        )) * damp)
    vt = vt.at[:, 0, n-1].set(
        (vc[:, 0, n-1] - 0.25 * cv[:, 0, n-1] * (
            ut[:, 0, n-1] + ut[:, 0, n-2] + ut[:, 1, n-2] + uc[:, 1, n-1]
            - 0.25 * cu[:, 1, n-1] * (vt[:, 0, n] + vt[:, 1, n] + vt[:, 1, n-1])
        )) * damp)

    return ut, vt


def _edge_interpolate4(ua4, dxa4):
    """FV3 non-uniform 4-point interpolation to the interface between cells 2 and 3.

    Exact port of ``edge_interpolate4`` from sw_core.F90.
    """
    t1 = dxa4[..., 0] + dxa4[..., 1]
    t2 = dxa4[..., 2] + dxa4[..., 3]
    return 0.5 * (
        ((t1 + dxa4[..., 1]) * ua4[..., 1] - dxa4[..., 1] * ua4[..., 0]) / t1
        + ((t2 + dxa4[..., 2]) * ua4[..., 2] - dxa4[..., 2] * ua4[..., 3]) / t2
    )


# FV3 interpolation constants (from sw_core.F90)
_A1 = 0.5625     # 4th-order Lagrange
_A2 = -0.0625
_C1 = -2.0 / 14.0  # one-sided cubic at face boundary
_C2 = 11.0 / 14.0
_C3 = 5.0 / 14.0


# ==============================================================================
# d2a2c_vect: D-grid → A-grid → C-grid (FV3 covariant convention)
# ==============================================================================

def _d2a2c_vect_duogrid(u_d, v_d, cdgrid):
    """D→A→C with Duo-Grid: FV3-faithful via unified D-grid halo exchange.

    Matches the ``gridstruct%dg%is_initialized`` branch in FV3
    ``sw_core.F90:3419-3454``:
    1. Full 2D D-grid staggered halo exchange via ``ext_vector`` pipeline
       (c2l_ord2 + scalar lat/lon halo + ``cubed_a2d_halo``).  This is
       the FV3 duogrid equivalent of ``mpp_update_domains(DGRID_NE)`` and
       populates u_d / v_d halo in BOTH axes.  Iter-59 identified
       cross-axis halo as the missing piece required by the 4th-order
       A→C stencil; iter-60 wires a single consistent halo path for both
       cross-axis and same-axis halo cells (no stitching across different
       halo sources).
    2. 4th-order D→A averaging on the fully-haloed domain gives utmp /
       vtmp at full (i, j) halo range using one consistent stencil on
       one consistent halo field (matching Fortran exactly).
    3. Covariant→contravariant via cosa_s/rsin2 (FV3 line 3451-3452).
    4. 4th-order A→C interpolation on the full-halo utmp / vtmp.
    """
    n = cdgrid.n
    grid = cdgrid.base
    dg = grid.duogrid
    # Iter-654: upgrade halo depth from h=2 to h=3 when the duogrid
    # structure has at least 3 halo cells (ng>=3).  This is the
    # FB-chain ng=3 wiring documented in docs/fv3_fortran_fidelity_review.md
    # architectural item #2: `_c_sw` first-order upwind at cube edges
    # amplifies face-boundary halo divergence, and Fortran FV3 delivers
    # halo-quality data 3 rings deep via `mpp_update_domains(DGRID_NE)`.
    # Fall back to h=2 for small grids where ng<3.
    h = 3 if (dg is not None and dg.ng >= 3) else 2
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell

    # ---- Step 1: Full 2D D-grid halo via ext_vector pipeline ----
    # FV3 ext_vector DGRID case (fv_duogrid.F90:741-826):
    #   c2l_ord2 (2-point length-weighted D→A average) → A-grid covariant
    #   → lat/lon rotation → scalar halo (pad_halo with cube_rmp) →
    #   cubed_a2d_halo back to D-grid at full 2D halo domain.
    #
    # ``ext_vector_dgrid`` takes A-grid COVARIANT utmp/vtmp as input
    # (in OUR convention these are cell-centre covariant winds, i.e.,
    # V · e_x / |e_x|² for normalised grid basis).  Use the length-
    # weighted mean form:
    #   utmp(i, j_cell) =
    #     (u(i,j)*dx(i,j) + u(i,j+1)*dx(i,j+1)) / (dx(i,j) + dx(i,j+1))
    # For constant u, utmp = u exactly (length-weighted weights sum to 1).
    #
    # Note: Fortran c2l_ord2 (fv_grid_utils.F90:2605-2611) uses the same
    # formula BUT with a leading factor of 2 compensated by a11/a22
    # having built-in factor 0.5.  Our ext_vector_dgrid works on
    # UNCOMPENSATED covariant A-grid input, so we drop the 2 for
    # constant-state preservation.
    #
    # Length at u_d stagger (i-cell, j-edge) is dx_edge_y (shape matches
    # u_d exactly).  Length at v_d stagger (i-edge, j-cell) is dy_edge_x.
    dx_u = cdgrid.dx_edge_y  # (6, n, n+1) — x-length at u_d positions
    dy_v = cdgrid.dy_edge_x  # (6, n+1, n) — y-length at v_d positions
    wu = u_d * dx_u
    wv = v_d * dy_v
    utmp_2nd = (wu[:, :, :-1] + wu[:, :, 1:]) / (
        dx_u[:, :, :-1] + dx_u[:, :, 1:])  # (6, n, n)
    vtmp_2nd = (wv[:, :-1, :] + wv[:, 1:, :]) / (
        dy_v[:, :-1, :] + dy_v[:, 1:, :])  # (6, n, n)
    u_d_full, v_d_full = ext_vector_dgrid(
        utmp_2nd, vtmp_2nd, dg,
        grid.cos_angle, grid.sin_angle,
        cos_sg5, rsin2,
        halo=h,
    )  # u_d_full: (6, n+2h, n+2h-1), v_d_full: (6, n+2h-1, n+2h)

    # Overwrite interior with EXACT original u_d/v_d — mirrors Fortran
    # where mpp_update_domains only fills halo cells; the interior is
    # never modified by ext_vector.  Index map: u_d_full[:, i_pad, j_pad]
    # has i_pad=i_cdgrid+h, j_pad=j_edge_cdgrid+h-1.  So u_d
    # (i=0..n-1, j=0..n) maps to u_d_full[:, h:h+n, h-1:h+n].
    u_d_full = u_d_full.at[:, h:h + n, h - 1:h + n].set(u_d)
    v_d_full = v_d_full.at[:, h - 1:h + n, h:h + n].set(v_d)

    # ---- Step 2: 4th-order D→A on fully-haloed domain ----
    # utmp(i, j_cell) = a2*(u(i,j-1)+u(i,j+2)) + a1*(u(i,j)+u(i,j+1))
    # Same 4th-order stencil applied uniformly to ALL cells (interior +
    # i-halo) using ONE halo field.  Matches FV3 sw_core.F90:3421-3435
    # duogrid branch exactly.
    #
    # Iter-654 generalisation: u_d_full has shape (6, n+2h, n+2h-1).
    # j-edge padded index p corresponds to cdgrid edge (p - (h-1)).
    # So cdgrid edges [j_cell-1, j_cell, j_cell+1, j_cell+2] map to
    # padded [j_cell + (h-2), j_cell + (h-1), j_cell + h, j_cell + (h+1)].
    # For j_cell in [0, n-1]: padded ranges
    #   [h-2:h-2+n, h-1:h-1+n, h:h+n, h+1:h+1+n]
    # At h=2 these collapse to [0:n, 1:n+1, 2:n+2, 3:n+3] (original code).
    if n > 3:
        utmp_full = (
            _A2 * (u_d_full[:, :, h - 2:h - 2 + n]
                   + u_d_full[:, :, h + 1:h + 1 + n])
            + _A1 * (u_d_full[:, :, h - 1:h - 1 + n]
                     + u_d_full[:, :, h:h + n])
        )  # (6, n+2h, n) — i-halo full, j-interior only
        vtmp_full = (
            _A2 * (v_d_full[:, h - 2:h - 2 + n, :]
                   + v_d_full[:, h + 1:h + 1 + n, :])
            + _A1 * (v_d_full[:, h - 1:h - 1 + n, :]
                     + v_d_full[:, h:h + n, :])
        )  # (6, n, n+2h) — j-halo full, i-interior only
    else:
        # Tiny grid fallback: 2-point average without 4th-order stencil.
        # Read edges [j_cell, j_cell+1] → padded [j_cell+(h-1), j_cell+h].
        utmp_full = 0.5 * (u_d_full[:, :, h - 1:h - 1 + n]
                            + u_d_full[:, :, h:h + n])
        vtmp_full = 0.5 * (v_d_full[:, h - 1:h - 1 + n, :]
                            + v_d_full[:, h:h + n, :])

    # ---- Step 3: Covariant→contravariant at cell centres (interior) ----
    # FV3 sw_core.F90:3451-3452:
    #   ua(i,j) = (utmp(i,j)-vtmp(i,j)*cosa_s(i,j)) * rsin2(i,j)
    # Take the i-interior slice of utmp_full / j-interior of vtmp_full
    # (ua/va only needs interior for downstream operators).
    utmp_int = utmp_full[:, h:h + n, :]
    vtmp_int = vtmp_full[:, :, h:h + n]
    ua = (utmp_int - vtmp_int * cos_sg5) * rsin2
    va = (vtmp_int - utmp_int * cos_sg5) * rsin2

    # ---- Step 4: 4th-order A→C interpolation ----
    # uc(i+1/2, j) = a2*(utmp(i-1,j)+utmp(i+2,j)) + a1*(utmp(i,j)+utmp(i+1,j))
    # utmp_full has shape (6, n+2h, n).  Padded i-index p maps to cell
    # (p - h).  For u-face k in [0, n], the stencil reads cells
    # [k-2, k-1, k, k+1] → padded [k-2+h, k-1+h, k+h, k+1+h].
    # For k in [0, n] (n+1 faces): padded ranges
    #   [h-2:h-1+n, h-1:h+n, h:h+n+1, h+1:h+2+n]
    # At h=2 these collapse to [0:n+1, 1:n+2, 2:n+3, 3:n+4] == the
    # original `[:-3, 1:-2, 2:-1, 3:]` slicing of length (n+4).
    uc = (_A2 * (utmp_full[:, h - 2:h - 1 + n, :]
                 + utmp_full[:, h + 1:h + 2 + n, :])
          + _A1 * (utmp_full[:, h - 1:h + n, :]
                   + utmp_full[:, h:h + n + 1, :]))  # (6, n+1, n)

    ut = (uc - v_d * cdgrid.cosa_u) * cdgrid.rsin_u

    vc = (_A2 * (vtmp_full[:, :, h - 2:h - 1 + n]
                 + vtmp_full[:, :, h + 1:h + 2 + n])
          + _A1 * (vtmp_full[:, :, h - 1:h + n]
                   + vtmp_full[:, :, h:h + n + 1]))  # (6, n, n+1)

    vt = (vc - u_d * cdgrid.cosa_v) * cdgrid.rsin_v

    return ua, va, uc, vc, ut, vt


def _apply_fortran_d2a2c_corner_overrides(utmp_pad, vtmp_pad, n):
    """Iter-938 port of Fortran sw_core.F90:3527-3545 + 3620-3639
    cube-corner sign-flip overrides on utmp_pad / vtmp_pad.

    Fortran applies four corner overrides per axis (SW/SE/NE/NW),
    each writing 3 halo cells (i=-2..0 or i=0..2) at the boundary
    halo row/col.  Our Python pad_halo_vector(halo=2) only provides
    2 halo cells per side, so we port the 2 deepest cells (depth
    -1 and -0 in Fortran indexing → padded-index depth-2 and
    depth-1).  The third Fortran cell (depth-3) is OUT of our
    halo=2 reach and is skipped — a documented partial port.

    Index map (with halo=2, padded shape (n+4, n+4)):
      Fortran i=-1 → padded 0   (depth-2 west halo)
      Fortran i=0  → padded 1   (depth-1 west halo)
      Fortran i=1  → padded 2   (first interior, west boundary)
      Fortran i=npx → padded n+2 (depth-1 east halo)
      Fortran i=npx+1 → padded n+3 (depth-2 east halo)

    The Fortran utmp/vtmp interior values are NOT modified — only
    halo cells.  This matches our intent of correcting `pad_halo_vector`'s
    `_fill_corners_h2` 2-point AVERAGE with Fortran's sign-flipped
    cross-component copy at the cube vertex.

    The vtmp overrides read from utmp_pad INTERIOR cells (which the
    utmp overrides do NOT modify), so the two override blocks are
    independent in input/output and may be applied in either order.

    Parameters
    ----------
    utmp_pad : (6, n+4, n+4) — pad_halo_vector output
    vtmp_pad : (6, n+4, n+4) — pad_halo_vector output
    n : int — interior grid size

    Returns
    -------
    utmp_pad, vtmp_pad : same shapes, with cube-vertex halo cells
        overwritten using Fortran's sign-flip cross-component values.
    """
    # ---- utmp x-direction overrides (Fortran 3527-3545) ----
    # SW corner: utmp(i=-1..0, j=0) = -vtmp(0, 1-i)
    utmp_pad = utmp_pad.at[:, 0, 1].set(-vtmp_pad[:, 1, 3])  # i=-1
    utmp_pad = utmp_pad.at[:, 1, 1].set(-vtmp_pad[:, 1, 2])  # i=0
    # SE corner: utmp(npx+i, 0) = +vtmp(npx, i+1)  for i in {0, 1}
    utmp_pad = utmp_pad.at[:, n+2, 1].set(+vtmp_pad[:, n+2, 2])
    utmp_pad = utmp_pad.at[:, n+3, 1].set(+vtmp_pad[:, n+2, 3])
    # NE corner: utmp(npx+i, npy) = -vtmp(npx, npy-1-i) for i in {0, 1}
    utmp_pad = utmp_pad.at[:, n+2, n+2].set(-vtmp_pad[:, n+2, n+1])
    utmp_pad = utmp_pad.at[:, n+3, n+2].set(-vtmp_pad[:, n+2, n])
    # NW corner: utmp(i=-1..0, npy) = +vtmp(0, npy-1+i+1)
    utmp_pad = utmp_pad.at[:, 0, n+2].set(+vtmp_pad[:, 1, n])     # i=-1
    utmp_pad = utmp_pad.at[:, 1, n+2].set(+vtmp_pad[:, 1, n+1])   # i=0

    # ---- vtmp y-direction overrides (Fortran 3620-3639) ----
    # SW corner: vtmp(0, j=-1..0) = -utmp(1-j, 0)
    vtmp_pad = vtmp_pad.at[:, 1, 0].set(-utmp_pad[:, 3, 1])  # j=-1, reads utmp(2, 0)
    vtmp_pad = vtmp_pad.at[:, 1, 1].set(-utmp_pad[:, 2, 1])  # j=0,  reads utmp(1, 0)
    # NW corner: vtmp(0, npy+j) = +utmp(j+1, npy)  for j in {0, 1}
    vtmp_pad = vtmp_pad.at[:, 1, n+2].set(+utmp_pad[:, 2, n+2])
    vtmp_pad = vtmp_pad.at[:, 1, n+3].set(+utmp_pad[:, 3, n+2])
    # SE corner: vtmp(npx, j=-1..0) = +utmp(ie+j, 0)
    vtmp_pad = vtmp_pad.at[:, n+2, 0].set(+utmp_pad[:, n,   1])   # j=-1 → utmp(npx-2, 0)
    vtmp_pad = vtmp_pad.at[:, n+2, 1].set(+utmp_pad[:, n+1, 1])   # j=0  → utmp(npx-1, 0)
    # NE corner: vtmp(npx, npy+j) = -utmp(ie-j, npy) for j in {0, 1}
    vtmp_pad = vtmp_pad.at[:, n+2, n+2].set(-utmp_pad[:, n+1, n+2])
    vtmp_pad = vtmp_pad.at[:, n+2, n+3].set(-utmp_pad[:, n,   n+2])

    return utmp_pad, vtmp_pad


def _d2a2c_vect(u_d, v_d, cdgrid):
    """FV3 D-grid → A-grid → C-grid vector conversion.

    Adapted from GFDL sw_core.F90 d2a2c_vect.  Returns C-grid
    velocities in FV3's COVARIANT convention (uc = interpolated covariant
    utmp, NOT the physical face-normal velocity).

    When the Duo-Grid is active on the base grid, dispatches to the
    FV3-faithful Duo-Grid path (``_d2a2c_vect_duogrid``) which uses
    4th-order interpolation everywhere and skips all edge/corner specials.
    This matches the ``dg%is_initialized`` branch in FV3 sw_core.F90.

    Parameters
    ----------
    u_d : (6, n, n+1) D-grid x-velocity at x-edge midpoints
    v_d : (6, n+1, n) D-grid y-velocity at y-edge midpoints
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    ua, va : (6, n, n) A-grid contravariant
    uc : (6, n+1, n) C-grid covariant u
    vc : (6, n, n+1) C-grid covariant v
    ut : (6, n+1, n) C-grid contravariant u (transport)
    vt : (6, n, n+1) C-grid contravariant v (transport)
    """
    # Dispatch to Duo-Grid path when active (FV3 sw_core.F90:3419).
    # FV3 gates on `dg%is_initialized` which is true when the duogrid
    # structure is fully allocated with sufficient halo. Here the `ng >= 2`
    # guard ensures the k2e-remapped halo covers both depths needed by the
    # 4th-order A→C stencil (halo=2). For tiny grids (n<4) where
    # ng = n//2 < 2, fall through to the legacy path.
    dg = cdgrid.base.duogrid
    if dg is not None and dg.ng >= 2:
        return _d2a2c_vect_duogrid(u_d, v_d, cdgrid)

    n = cdgrid.n
    npt = min(4, n // 2)

    # --- Fortran parity note (iter-108, Priority 3 audit) --------------
    # sw_core.F90:3527-3545 and 3620-3640 apply cube-vertex corner
    # OVERRIDES on utmp/vtmp and ua/va at the 4 cube corners (sw/se/ne/
    # nw), gated on `.not. dg%is_initialized`.  Each override writes
    # HALO cells of utmp/vtmp (not interior cells) with sign-flipped
    # copies of the OTHER component on the adjacent face, e.g.:
    #   utmp(i=-2..0, j=0)   = -vtmp(0, 1-i)       ! SW corner halo row
    #   vtmp(i=0, j=-2..0)   = -utmp(1-j, 0)       ! SW corner halo col
    # The intent is to give utmp/vtmp sensible values in the 3-face
    # cube-vertex halo region, where ordinary 2-face halo exchange
    # (copy/interpolate from a single neighbor) is ambiguous.
    #
    # NOT PORTED in Python's non-duogrid path.  Python relies on:
    #   - `pad_halo_vector` for 2-face cross-face halo interpolation
    #     along panel edges (works cleanly away from cube vertices), and
    #   - `_fill_corners_h1` / `_fill_corners_h2` inside the halo layer
    #     for the cube-vertex 2x2 blocks (2-point AVERAGES of adjacent
    #     edge halos, NOT the Fortran sign-flip copy from the other
    #     component).
    #
    # These two approaches give DIFFERENT values at the cube-vertex
    # cells in the non-duogrid path — the delta is O(1) on random
    # input but typically O(dx²) on smooth fields.  The impact on the
    # non-duogrid FB path has NOT been quantified (the FB path is
    # already experimental/unstable at C36 for independent reasons).
    #
    # Duogrid path (via `_d2a2c_vect_duogrid`, which Fortran also
    # skips via `dg%is_initialized`) is unaffected by this gap.
    # ------------------------------------------------------------------

    # ---- Step 1: D-grid → covariant cell centres (utmp, vtmp) ----
    utmp = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])   # (6, n, n)
    vtmp = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])   # (6, n, n)

    # 4th-order interior (at least npt cells from each edge).
    # Guard: needs n > 2*npt AND npt > 0 (i.e., n >= 2).
    if n > 2 * npt and npt > 0:
        u4 = (_A2 * (u_d[:, :, :-3] + u_d[:, :, 3:])
              + _A1 * (u_d[:, :, 1:-2] + u_d[:, :, 2:-1]))
        utmp = utmp.at[:, :, npt:n - npt].set(u4[:, :, npt - 1:n - npt - 1])
        v4 = (_A2 * (v_d[:, :-3, :] + v_d[:, 3:, :])
              + _A1 * (v_d[:, 1:-2, :] + v_d[:, 2:-1, :]))
        vtmp = vtmp.at[:, npt:n - npt, :].set(v4[:, npt - 1:n - npt - 1, :])

    # ---- Step 2: Halo-exchange COVARIANT utmp/vtmp ----
    # halo=2 for edge_interpolate4 at face boundaries (FV3 sw_core.F90:3587).
    grid = cdgrid.base
    h = 2
    utmp_pad, vtmp_pad = pad_halo_vector(
        utmp, vtmp,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        interp_offsets=grid.halo_interp_offsets_h2,
        halo=h,
    )  # each (6, n+4, n+4)

    # Iter-938 / iter-938b: the Fortran sw_core.F90:3527-3545 + 3620-
    # 3639 cube-corner sign-flip overrides on utmp_pad/vtmp_pad are
    # available as `_apply_fortran_d2a2c_corner_overrides(utmp_pad,
    # vtmp_pad, n)` but are NOT wired into the main path because the
    # downstream edge_interpolate4 j-slicing reads only interior j ∈
    # [h, n+h-1] = padded [2, n+1], while the Fortran overrides write
    # to padded j=1 and j=n+2 (south/north halo).  Wiring without the
    # edge_interpolate4 j-slice extension would be output-dead (Codex
    # iter-938 stop-time finding).  The helper + its mapping unit
    # tests are kept as documentation of the Fortran arithmetic for
    # iter-939+ to wire in once the edge_interpolate4 refactor lands.

    # ---- Step 3: Contravariant at cell centres (including halo) ----
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')

    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad
    va_pad = (vtmp_pad - utmp_pad * cos_sg5_pad) * rsin2_pad

    ua = ua_pad[:, h:-h, h:-h]  # (6, n, n)
    va = va_pad[:, h:-h, h:-h]

    # ---- Step 4a: A→C x-direction (covariant utmp → uc) ----
    # With halo=2: padded cell k → utmp_pad[:, k+h, ...]
    # u-face i sits between cells i-1 and i → padded indices i+h-1 and i+h
    uc = 0.5 * (utmp_pad[:, h-1:n+h, h:-h]
                + utmp_pad[:, h:n+h+1, h:-h])  # (6, n+1, n)

    if n > 2 * npt + 2:
        uc_4th = (_A2 * (utmp_pad[:, h-2:n+h-1, h:-h]
                         + utmp_pad[:, h+1:n+h+2, h:-h])
                  + _A1 * (utmp_pad[:, h-1:n+h, h:-h]
                           + utmp_pad[:, h:n+h+1, h:-h]))
        i_lo = npt + 1
        i_hi = n - npt
        uc = uc.at[:, i_lo:i_hi, :].set(uc_4th[:, i_lo - 1:i_hi - 1, :])

    # One-sided c1/c2/c3 stencil at i=1, n-1 (FV3 sw_core.F90:3586,3594)
    if n > 3:
        uc = uc.at[:, 1, :].set(
            _C1 * utmp_pad[:, h + 2, h:-h] + _C2 * utmp_pad[:, h + 1, h:-h]
            + _C3 * utmp_pad[:, h, h:-h])
        uc = uc.at[:, n - 1, :].set(
            _C1 * utmp_pad[:, n + h - 3, h:-h] + _C2 * utmp_pad[:, n + h - 2, h:-h]
            + _C3 * utmp_pad[:, n + h - 1, h:-h])

    # AT face boundary (i=0, i=n): edge_interpolate4 on CONTRAVARIANT ua
    # (FV3 sw_core.F90:3587,3603). With halo=2 the 4-point stencil
    # straddles the face boundary correctly.
    #
    # The Fortran uses sin_sg from the HALO cell for the upwind conversion:
    #   ut>0 → sin_sg(i-1,j,3) where i-1 is in the halo for i=is (face boundary)
    #   ut≤0 → sin_sg(i,j,1) where i is the first interior cell
    # We use pad_halo to get the correct sin_sg at halo positions.
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (h, h), (0, 0)], mode='edge')
    sin_east = cdgrid.sin_sg[:, :, :, 2]   # E-edge
    sin_west = cdgrid.sin_sg[:, :, :, 0]   # W-edge
    offsets = grid.halo_interp_offsets
    se_pad_x = pad_halo(sin_east, interp_offsets=offsets)  # (6, n+2, n+2)
    sw_pad_x = pad_halo(sin_west, interp_offsets=offsets)
    for i_bdy in ([0, n] if n >= 2 else []):
        i_p = i_bdy + h  # padded offset: cell i → padded index i+h
        # ua_pad stencil: 4 cells centred on u-face i_bdy
        ua4 = jnp.stack([ua_pad[:, i_p - 1, h:-h], ua_pad[:, i_p, h:-h],
                         ua_pad[:, i_p + 1, h:-h], ua_pad[:, i_p + 2, h:-h]],
                        axis=-1)
        dxa4 = jnp.stack([dxc_pad_x[:, i_p - 1, :], dxc_pad_x[:, i_p, :],
                          dxc_pad_x[:, i_p + 1, :], dxc_pad_x[:, i_p + 2, :]],
                         axis=-1)
        ut_bdy = _edge_interpolate4(ua4, dxa4)

        # FV3 sw_core.F90:3589-3592: sin_sg at halo cell for upwind
        # se_pad has halo=1: se_pad[:, i, j+1] = sin_east at cell (i-1, j)
        # For i_bdy=0: left cell is halo(-1) → se_pad[:, 0, 1:-1]
        # For i_bdy=n: left cell is n-1 → se_pad[:, n, 1:-1]
        sin_left = se_pad_x[:, i_bdy, 1:-1]   # E-edge of cell to LEFT of face i_bdy
        sin_right = sw_pad_x[:, i_bdy + 1, 1:-1]  # W-edge of cell to RIGHT of face i_bdy
        uc_bdy = jnp.where(ut_bdy > 0, ut_bdy * sin_left, ut_bdy * sin_right)
        uc = uc.at[:, i_bdy, :].set(uc_bdy)

    # Contravariant ut from covariant uc (FV3 rsin_u = 1/sin²)
    ut = (uc - v_d * cdgrid.cosa_u) * cdgrid.rsin_u

    # At face boundaries (i=0, n): FV3 sets ut = edge_interpolate4(ua) DIRECTLY
    # (sw_core.F90:3587,3603), not via (uc - v*cos)*rsin_u.
    # Since uc = ut_ei4*sin_sg, dividing by the same sin_sg recovers ut_ei4.
    # Use haloed sin_sg (same as for uc above) for consistent upwind.
    for i_bdy in ([0, n] if n >= 2 else []):
        sin_left = se_pad_x[:, i_bdy, 1:-1]
        sin_right = sw_pad_x[:, i_bdy + 1, 1:-1]
        sin_upwind = jnp.where(uc[:, i_bdy, :] > 0, sin_left, sin_right)
        ut = ut.at[:, i_bdy, :].set(
            uc[:, i_bdy, :] / jnp.maximum(sin_upwind, _EPS))

    # ---- Step 4b: A→C y-direction (covariant vtmp → vc) ----
    # vtmp_pad axis 2 has n+2h elements.  v-face j → padded j+h-1 and j+h.
    vc = 0.5 * (vtmp_pad[:, h:-h, h-1:n+h] + vtmp_pad[:, h:-h, h:n+h+1])  # (6, n, n+1)

    if n > 2 * npt + 2:
        vc_4th = (_A2 * (vtmp_pad[:, h:-h, h-2:n+h-1]
                         + vtmp_pad[:, h:-h, h+1:n+h+2])
                  + _A1 * (vtmp_pad[:, h:-h, h-1:n+h]
                           + vtmp_pad[:, h:-h, h:n+h+1]))
        j_lo = npt + 1
        j_hi = n - npt
        vc = vc.at[:, :, j_lo:j_hi].set(vc_4th[:, :, j_lo - 1:j_hi - 1])

    # One-sided c1/c2/c3 stencil at j=1, n-1
    if n > 3:
        vc = vc.at[:, :, 1].set(
            _C1 * vtmp_pad[:, h:-h, h + 2] + _C2 * vtmp_pad[:, h:-h, h + 1]
            + _C3 * vtmp_pad[:, h:-h, h])
        vc = vc.at[:, :, n - 1].set(
            _C1 * vtmp_pad[:, h:-h, n + h - 3] + _C2 * vtmp_pad[:, h:-h, n + h - 2]
            + _C3 * vtmp_pad[:, h:-h, n + h - 1])

    # AT face boundary (j=0, j=n): edge_interpolate4 on va (halo=2 straddles boundary)
    # Use haloed sin_sg for correct upwind at face boundaries (same logic as x-dir).
    dyc_pad_y = jnp.pad(grid.dy, [(0, 0), (0, 0), (h, h)], mode='edge')
    sin_north = cdgrid.sin_sg[:, :, :, 3]  # N-edge
    sin_south = cdgrid.sin_sg[:, :, :, 1]  # S-edge
    sn_pad_y = pad_halo(sin_north, interp_offsets=offsets)
    ss_pad_y = pad_halo(sin_south, interp_offsets=offsets)
    for j_bdy in ([0, n] if n >= 2 else []):
        j_p = j_bdy + h
        va4 = jnp.stack([va_pad[:, h:-h, j_p - 1], va_pad[:, h:-h, j_p],
                         va_pad[:, h:-h, j_p + 1], va_pad[:, h:-h, j_p + 2]],
                        axis=-1)
        dya4 = jnp.stack([dyc_pad_y[:, :, j_p - 1], dyc_pad_y[:, :, j_p],
                          dyc_pad_y[:, :, j_p + 1], dyc_pad_y[:, :, j_p + 2]],
                         axis=-1)
        vt_bdy = _edge_interpolate4(va4, dya4)

        # FV3: sin_sg from halo cell for upwind at face boundary
        sin_below = sn_pad_y[:, 1:-1, j_bdy]     # N-edge of cell BELOW face j_bdy
        sin_above = ss_pad_y[:, 1:-1, j_bdy + 1]  # S-edge of cell ABOVE face j_bdy
        vc_bdy = jnp.where(vt_bdy > 0, vt_bdy * sin_below, vt_bdy * sin_above)
        vc = vc.at[:, :, j_bdy].set(vc_bdy)

    vt = (vc - u_d * cdgrid.cosa_v) * cdgrid.rsin_v

    # Override only at edge_interpolate4 positions (j=0, n) to recover
    # edge_interpolate4 result: vt = vc/sin_sg. Use haloed sin_sg.
    for j_bdy in [0, n]:
        sin_below = sn_pad_y[:, 1:-1, j_bdy]
        sin_above = ss_pad_y[:, 1:-1, j_bdy + 1]
        sin_upwind = jnp.where(vc[:, :, j_bdy] > 0, sin_below, sin_above)
        vt = vt.at[:, :, j_bdy].set(
            vc[:, :, j_bdy] / jnp.maximum(sin_upwind, _EPS))

    # FV3 non-duogrid adjacent-strip vt recomputation (sw_core.F90:670-691).
    # On west (i=0) and east (i=n-1) boundary cells, recompute vt using a
    # 4-point ut average instead of the standard (vc - u*cosa)*rsin_v.
    # Restricted to j_face in [2, n-2] per Fortran `max(3,js), min(npy-2,je+1)`.
    # Same-face ut indices are (i_cell=0,1) for west; (i_cell=n-2,n-1) for east.
    # Fortran halo vt(0,j), vt(npx,j) columns are not addressable in Python's
    # interior-only layout — only vt(1,j) / vt(npx-1,j) are ported.
    if n >= 4:
        j_lo, j_hi = 2, n - 1  # j_face range [j_lo, j_hi) → [2, n-2]
        # West: Fortran vt(1, j) → Python vt[:, 0, j_lo:j_hi]
        ut_w = (ut[:, 0, j_lo - 1:j_hi - 1] + ut[:, 1, j_lo - 1:j_hi - 1]
                + ut[:, 0, j_lo:j_hi] + ut[:, 1, j_lo:j_hi])
        vt_w_new = (vc[:, 0, j_lo:j_hi]
                    - 0.25 * cdgrid.cosa_v[:, 0, j_lo:j_hi] * ut_w)
        vt = vt.at[:, 0, j_lo:j_hi].set(vt_w_new)
        # East: Fortran vt(npx-1, j) → Python vt[:, n-1, j_lo:j_hi]
        ut_e = (ut[:, n - 1, j_lo - 1:j_hi - 1]
                + ut[:, n, j_lo - 1:j_hi - 1]
                + ut[:, n - 1, j_lo:j_hi]
                + ut[:, n, j_lo:j_hi])
        vt_e_new = (vc[:, n - 1, j_lo:j_hi]
                    - 0.25 * cdgrid.cosa_v[:, n - 1, j_lo:j_hi] * ut_e)
        vt = vt.at[:, n - 1, j_lo:j_hi].set(vt_e_new)

    # FV3 non-duogrid adjacent-strip ut recomputation (sw_core.F90:701-707,
    # 716-722). South (j=0) and north (j=n-1) boundary cells: recompute ut
    # using a 4-point vt average. Restricted to i_face in [2, n-2].
    #
    # Dependency-ordering note: the south block reads vt at i_cell ∈ [1, n-3]
    # (matching Fortran's i_fortran-1 ∈ [2, npx-3]).  The WEST 4-point block
    # above only writes vt at i_cell=0; the EAST block only writes vt at
    # i_cell=n-1.  Neither overlaps the i_cell range read here, so the
    # south/north blocks read interior (vc - u*cosa)*rsin_v values for
    # j_face=1 and the j_bdy sin_sg-overridden values for j_face=0 — exactly
    # what Fortran does.  The Fortran halo-column updates vt(0, j), vt(npx, j)
    # have no functional impact on the south/north reads because those halo
    # i-cells are not referenced by the south/north 4-point formula.
    if n >= 4:
        i_lo, i_hi = 2, n - 1
        # South: Fortran ut(i, 1) → Python ut[:, i_lo:i_hi, 0]
        vt_s = (vt[:, i_lo - 1:i_hi - 1, 0] + vt[:, i_lo:i_hi, 0]
                + vt[:, i_lo - 1:i_hi - 1, 1] + vt[:, i_lo:i_hi, 1])
        ut_s_new = (uc[:, i_lo:i_hi, 0]
                    - 0.25 * cdgrid.cosa_u[:, i_lo:i_hi, 0] * vt_s)
        ut = ut.at[:, i_lo:i_hi, 0].set(ut_s_new)
        # North: Fortran ut(i, npy-1) → Python ut[:, i_lo:i_hi, n-1]
        vt_n = (vt[:, i_lo - 1:i_hi - 1, n - 1]
                + vt[:, i_lo:i_hi, n - 1]
                + vt[:, i_lo - 1:i_hi - 1, n]
                + vt[:, i_lo:i_hi, n])
        ut_n_new = (uc[:, i_lo:i_hi, n - 1]
                    - 0.25 * cdgrid.cosa_u[:, i_lo:i_hi, n - 1] * vt_n)
        ut = ut.at[:, i_lo:i_hi, n - 1].set(ut_n_new)

    return ua, va, uc, vc, ut, vt


# ==============================================================================
# Shared FV3 c_sw helpers (used by both _c_sw and fv3_csw_tendencies)
# ==============================================================================


def _sina_u_v_from_sin_sg(cdgrid):
    """Return `sina_u` (6, n+1, n) and `sina_v` (6, n, n+1) constructed
    from FV3 sub-grid `sin_sg` per ``fv_grid_utils.F90:505-518``.

    Interior faces:
      sina_u(i,j) = 0.5*(sin_sg(i-1,j,3) + sin_sg(i,j,1))
      sina_v(i,j) = 0.5*(sin_sg(i,j-1,4) + sin_sg(i,j,2))

    Panel-edge faces: use the single-side sin_sg at the outermost cell,
    matching the sina_u/sina_v construction in `cubed_sphere_cdgrid.py`
    and `_d_sw5_corner_divergence`.

    Using this formulation instead of ``sqrt(1 - cosa_u**2)`` is the
    Fortran-faithful convention — the two are only identical when
    ``cosa**2 + sina**2 = 1`` exactly, which is NOT the case for
    halo-averaged ``cosa_u = 0.5*(cos_sg(E) + cos_sg(W))``.
    """
    sg = cdgrid.sin_sg  # (6, n, n, 9): 0=W, 1=S, 2=E, 3=N, 4=center, ...
    sin_E = sg[:, :, :, 2]
    sin_W = sg[:, :, :, 0]
    sin_N = sg[:, :, :, 3]
    sin_S = sg[:, :, :, 1]

    sina_u_int = 0.5 * (sin_E[:, :-1, :] + sin_W[:, 1:, :])  # (6, n-1, n)
    sina_u = jnp.concatenate(
        [sin_W[:, :1, :], sina_u_int, sin_E[:, -1:, :]], axis=1,
    )  # (6, n+1, n)

    sina_v_int = 0.5 * (sin_N[:, :, :-1] + sin_S[:, :, 1:])  # (6, n, n-1)
    sina_v = jnp.concatenate(
        [sin_S[:, :, :1], sina_v_int, sin_N[:, :, -1:]], axis=2,
    )  # (6, n, n+1)

    return sina_u, sina_v


def _ke_upwind(uc, vc, ua, va, u_d, v_d, cdgrid, use_duogrid):
    """FV3 c_sw KE upwind selection (sw_core.F90:303-365).

    Returns ke_u, ke_v — the upwind-selected covariant velocities for KE.
    Non-duogrid path applies sin_sg/cos_sg conversion at face boundaries.
    """
    n = cdgrid.n
    ke_u = jnp.where(ua > 0, uc[:, :-1, :], uc[:, 1:, :])
    ke_v = jnp.where(va > 0, vc[:, :, :-1], vc[:, :, 1:])

    if not use_duogrid:
        sg = cdgrid.sin_sg
        cg = cdgrid.cos_sg
        # West edge (cell 0, ua > 0): uc*sin_sg(W) + v*cos_sg(W)
        ke_bdy_l = uc[:, 0, :] * sg[:, 0, :, 0] + v_d[:, 0, :] * cg[:, 0, :, 0]
        ke_u = ke_u.at[:, 0, :].set(
            jnp.where(ua[:, 0, :] > 0, ke_bdy_l, ke_u[:, 0, :]))
        # East edge (cell n-1, ua <= 0): uc*sin_sg(E) + v*cos_sg(E)
        ke_bdy_r = uc[:, n, :] * sg[:, n-1, :, 2] + v_d[:, n, :] * cg[:, n-1, :, 2]
        ke_u = ke_u.at[:, n-1, :].set(
            jnp.where(ua[:, n-1, :] > 0, ke_u[:, n-1, :], ke_bdy_r))
        # South edge (cell j=0, va > 0): vc*sin_sg(S) + u*cos_sg(S)
        ke_bdy_b = vc[:, :, 0] * sg[:, :, 0, 1] + u_d[:, :, 0] * cg[:, :, 0, 1]
        ke_v = ke_v.at[:, :, 0].set(
            jnp.where(va[:, :, 0] > 0, ke_bdy_b, ke_v[:, :, 0]))
        # North edge (cell j=n-1, va <= 0): vc*sin_sg(N) + u*cos_sg(N)
        ke_bdy_t = vc[:, :, n] * sg[:, :, n-1, 3] + u_d[:, :, n] * cg[:, :, n-1, 3]
        ke_v = ke_v.at[:, :, n-1].set(
            jnp.where(va[:, :, n-1] > 0, ke_v[:, :, n-1], ke_bdy_t))

    return ke_u, ke_v


def _del6_vt_flux(nord, damp, q, cdgrid, use_duogrid=False):
    """FV3 del6_vt_flux: del-n damping for relative vorticity (sw_core.F90:2008-2121).

    Same Laplacian operator as ``_deln_flux`` but returns raw diffusive fluxes
    instead of adding them to transport fluxes.  Used in d_sw6 for vorticity
    damping when ``damp_v > 1e-5``.

    Halo path: uses ``pad_halo`` for cell-centre scalar exchange (equivalent
    to Fortran's MPI ``mpp_update_domains``).  For duogrid, the Fortran skips
    ``copy_corners`` (gated on ``bounded_domain``, sw_core.F90:2060-2061);
    ``pad_halo`` already includes corner-safe cross-face exchange, so no
    special duogrid gating is needed in the halo exchange itself.

    Parameters
    ----------
    nord : int — damping order (0=del-2, 1=del-4, 2=del-6)
    damp : float — pre-scaled coefficient: (damp_v * da_min_c)^(nord+1)
    q : (6, n, n) — relative vorticity at cell centres
    cdgrid : CubedSphereCDGrid
    use_duogrid : bool — True when duogrid is active (matches Fortran
        ``bounded_domain`` gating: copy_corners skipped for duogrid)

    Returns
    -------
    fx2 : (6, n+1, n) — x-direction diffusive vorticity flux
    fy2 : (6, n, n+1) — y-direction diffusive vorticity flux
    """
    n = cdgrid.n
    grid = cdgrid.base
    sg = cdgrid.sin_sg
    dy = cdgrid.dy_edge_x
    dx = cdgrid.dx_edge_y
    rdxc = cdgrid.rdxc
    rdyc = cdgrid.rdyc
    rarea = 1.0 / grid.area

    # Route halo exchanges through the duogrid remap when `use_duogrid`,
    # matching the Fortran `bounded_domain` path in `del6_vt_flux` which
    # skips `copy_corners` and relies on the duogrid MPI update.  Without
    # this the function silently fell back to the non-duogrid `interp_offsets`
    # path even when called from the d_sw6 duogrid branch.
    dg = grid.duogrid if use_duogrid else None
    _offs = None if use_duogrid else grid.halo_interp_offsets

    # Iter-937b (Codex iter-937 stop-time fix): factor `damp` out of
    # the iteration and apply at the final flux-output stage.  iter-937
    # only fixed the sibling implementation in `fv3_del6_vt_flux.py`
    # (called from `fv3_del6_vorticity_damping`, which the production
    # `FV3EdgeShallowWaterModel.step` post-step damp_v hook uses); the
    # ACTIVE FB-chain path is via `_d_sw_native` (line 2341), which
    # calls THIS implementation.  Same float32-overflow class as
    # iter-934 (`_deln_flux`): `d2 = damp * q` blows past float32 max
    # at low resolution where `damp = (damp_v * da_min_c)^(nord+1)` is
    # huge.  All operations between Step 1 and the return are LINEAR
    # in d2, so the result is mathematically identical with the damp
    # factor applied at the end instead.
    d2 = q

    # Laplacian diffusive fluxes (USE_SG path, sw_core.F90:2064-2082)
    d2_pad = pad_halo(d2, interp_offsets=_offs, duogrid=dg)
    sin_E = sg[:, :, :, 2]
    sin_W = sg[:, :, :, 0]
    sin_N = sg[:, :, :, 3]
    sin_S = sg[:, :, :, 1]
    se_pad = pad_halo(sin_E, interp_offsets=_offs, duogrid=dg)
    sw_pad = pad_halo(sin_W, interp_offsets=_offs, duogrid=dg)
    sn_pad = pad_halo(sin_N, interp_offsets=_offs, duogrid=dg)
    ss_pad = pad_halo(sin_S, interp_offsets=_offs, duogrid=dg)

    sin_uv_x = 0.5 * (se_pad[:, :n+1, 1:-1] + sw_pad[:, 1:n+2, 1:-1])
    sin_uv_y = 0.5 * (sn_pad[:, 1:-1, :n+1] + ss_pad[:, 1:-1, 1:n+2])

    fx2 = sin_uv_x * dy * (d2_pad[:, :-1, 1:-1] - d2_pad[:, 1:, 1:-1]) * rdxc
    fy2 = sin_uv_y * dx * (d2_pad[:, 1:-1, :-1] - d2_pad[:, 1:-1, 1:]) * rdyc

    # Higher-order iteration (sw_core.F90:2084-2119)
    for _it in range(nord):
        d2 = (fx2[:, :-1, :] - fx2[:, 1:, :] + fy2[:, :, :-1] - fy2[:, :, 1:]) * rarea
        d2_pad = pad_halo(d2, interp_offsets=_offs, duogrid=dg)
        fx2 = sin_uv_x * dy * (d2_pad[:, 1:, 1:-1] - d2_pad[:, :-1, 1:-1]) * rdxc
        fy2 = sin_uv_y * dx * (d2_pad[:, 1:-1, 1:] - d2_pad[:, 1:-1, :-1]) * rdyc

    # Iter-937b: apply the deferred damp factor at the final output
    # stage.  See top-of-function comment for float32-overflow rationale.
    fx2 = damp * fx2
    fy2 = damp * fy2

    return fx2, fy2


def _divergence_corner_duo(u_d, v_d, ua, va, cdgrid):
    """FV3 divergence_corner_duo (sw_core.F90:2345-2447).

    Computes corner divergence for hyperviscosity (nord > 0) on the duogrid
    path.  Uses cross-velocity correction via cos_sg/sin_sg at cell edges
    and applies face-boundary zeroing + 0.25 attenuation at adjacent cells.

    Parameters
    ----------
    u_d : (6, n, n+1) D-grid x-velocity at x-edge midpoints
    v_d : (6, n+1, n) D-grid y-velocity at y-edge midpoints
    ua : (6, n, n) A-grid contravariant u (from d2a2c_vect)
    va : (6, n, n) A-grid contravariant v
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    divg_d : (6, n+1, n+1) corner divergence
    """
    n = cdgrid.n
    sg = cdgrid.sin_sg
    cg = cdgrid.cos_sg
    dxc = cdgrid.dxc   # (6, n+1, n) centre-to-centre in x
    dyc = cdgrid.dyc   # (6, n, n+1) centre-to-centre in y
    rarea_c = cdgrid.rarea_c  # (6, n+1, n+1)

    # Pad ua, va for cross-velocity averages at boundaries.
    # Iter-657 (reverting iter-656 Fix B, per Codex "no-op" finding):
    # the halo-source choice here is a numerical no-op.  uf differs
    # between `pad_halo(cross-face)` and `mode='edge'(same-face)` at
    # j=0 and j=n cells (measured ~1e6 absolute diff on random input
    # at C36), but those cells' contribution to `divg_d` is ZEROED
    # by the face-boundary zeroing step below (`.at[:, 0, :].set(0)`
    # etc.).  Iter-949 re-tested with `pad_halo_vector` instead of
    # `mode='edge'` and confirmed bit-identical FB chain output on
    # duogrid C36 W2 1-day (|u|=76.91, |v|=75.38 either way).
    # mode='edge' retained as the cheaper equivalent.
    ua_pad = jnp.pad(ua, [(0, 0), (1, 1), (0, 0)], mode='edge')  # (6, n+2, n)
    va_pad = jnp.pad(va, [(0, 0), (0, 0), (1, 1)], mode='edge')  # (6, n, n+2)

    # --- uf: u-direction flux at v-face positions (sw_core.F90:2413-2418) ---
    # uf(i,j) = (u(i,j) - 0.25*(va(i,j-1)+va(i,j))*(cos_sg(i,j-1,N)+cos_sg(i,j,S)))
    #           * dyc(i,j) * 0.5*(sin_sg(i,j-1,N)+sin_sg(i,j,S))
    # u_d: (6, n, n+1), va: (6, n, n), sin/cos_sg: (6, n, n, 9)
    # va at (i,j-1) and (i,j): need j from 1..n (Fortran jsd+1..jed → Python 1..n)
    # But in Python, u_d[:, :, j] for j=0..n, va[:, :, j] for j=0..n-1
    # For uf at j=1..n: va(i,j-1) and va(i,j)
    va_below = va_pad[:, :, :-1]   # (6, n, n+1) — va at j-1
    va_above = va_pad[:, :, 1:]    # (6, n, n+1) — va at j

    # cos_sg N-edge (index 3) at cell (i,j-1) and S-edge (index 1) at cell (i,j)
    cos_N = cg[:, :, :, 3]  # (6, n, n)
    cos_S = cg[:, :, :, 1]  # (6, n, n)
    sin_N = sg[:, :, :, 3]  # (6, n, n)
    sin_S = sg[:, :, :, 1]  # (6, n, n)
    cos_N_pad = jnp.pad(cos_N, [(0, 0), (0, 0), (1, 1)], mode='edge')
    cos_S_pad = jnp.pad(cos_S, [(0, 0), (0, 0), (1, 1)], mode='edge')
    sin_N_pad = jnp.pad(sin_N, [(0, 0), (0, 0), (1, 1)], mode='edge')
    sin_S_pad = jnp.pad(sin_S, [(0, 0), (0, 0), (1, 1)], mode='edge')

    cos_sum_u = cos_N_pad[:, :, :-1] + cos_S_pad[:, :, 1:]  # (6, n, n+1)
    sin_sum_u = sin_N_pad[:, :, :-1] + sin_S_pad[:, :, 1:]
    uf = (u_d - 0.25 * (va_below + va_above) * cos_sum_u) * dyc * 0.5 * sin_sum_u

    # --- vf: v-direction flux at u-face positions (sw_core.F90:2420-2425) ---
    # vf(i,j) = (v(i,j) - 0.25*(ua(i-1,j)+ua(i,j))*(cos_sg(i-1,j,E)+cos_sg(i,j,W)))
    #           * dxc(i,j) * 0.5*(sin_sg(i-1,j,E)+sin_sg(i,j,W))
    ua_left = ua_pad[:, :-1, :]    # (6, n+1, n) — ua at i-1
    ua_right = ua_pad[:, 1:, :]    # (6, n+1, n) — ua at i

    cos_E = cg[:, :, :, 2]  # (6, n, n) E-edge
    cos_W = cg[:, :, :, 0]  # (6, n, n) W-edge
    sin_E = sg[:, :, :, 2]
    sin_W = sg[:, :, :, 0]
    cos_E_pad = jnp.pad(cos_E, [(0, 0), (1, 1), (0, 0)], mode='edge')
    cos_W_pad = jnp.pad(cos_W, [(0, 0), (1, 1), (0, 0)], mode='edge')
    sin_E_pad = jnp.pad(sin_E, [(0, 0), (1, 1), (0, 0)], mode='edge')
    sin_W_pad = jnp.pad(sin_W, [(0, 0), (1, 1), (0, 0)], mode='edge')

    cos_sum_v = cos_E_pad[:, :-1, :] + cos_W_pad[:, 1:, :]  # (6, n+1, n)
    sin_sum_v = sin_E_pad[:, :-1, :] + sin_W_pad[:, 1:, :]
    vf = (v_d - 0.25 * (ua_left + ua_right) * cos_sum_v) * dxc * 0.5 * sin_sum_v

    # --- divg_d: corner divergence (sw_core.F90:2427-2442) ---
    # divg_d(i,j) = (vf(i,j-1) - vf(i,j) + uf(i-1,j) - uf(i,j)) * rarea_c(i,j)
    # vf: (6, n+1, n), uf: (6, n, n+1)
    # Need padded vf/uf for the stencil at corner (i,j) ranging 0..n
    vf_pad = jnp.pad(vf, [(0, 0), (0, 0), (1, 1)], mode='edge')  # (6, n+1, n+2)
    uf_pad = jnp.pad(uf, [(0, 0), (1, 1), (0, 0)], mode='edge')  # (6, n+2, n+1)

    divg_d = (vf_pad[:, :, :-1] - vf_pad[:, :, 1:]
              + uf_pad[:, :-1, :] - uf_pad[:, 1:, :]) * rarea_c

    # Face-boundary zeroing (sw_core.F90:2431-2434)
    divg_d = divg_d.at[:, 0, :].set(0.0)
    divg_d = divg_d.at[:, n, :].set(0.0)
    divg_d = divg_d.at[:, :, 0].set(0.0)
    divg_d = divg_d.at[:, :, n].set(0.0)

    # 0.25× attenuation at face-adjacent cells (sw_core.F90:2437-2440)
    divg_d = divg_d.at[:, 1, :].multiply(0.25)
    divg_d = divg_d.at[:, n - 1, :].multiply(0.25)
    divg_d = divg_d.at[:, :, 1].multiply(0.25)
    divg_d = divg_d.at[:, :, n - 1].multiply(0.25)

    return divg_d


def _apply_legacy_d_sw4_corner_ke_fix(
        ke, ut, vt, u_d, v_d, dt,
        bounded_domain: bool):
    """Iter-869 corner-KE fix kernel for Fortran d_sw4 legacy path.

    Applies the four cube-vertex KE overrides from Fortran
    ``sw_core.F90:1442-1465``, gated by Fortran's
    ``.not. bounded_domain .or. .not. flagstruct%duogrid`` (i.e.,
    fires UNLESS BOTH bounded_domain AND duogrid are true — for
    legoESM's global cubed sphere with duogrid we have
    bounded_domain==duogrid so the gate matches
    ``not bounded_domain``).

    Fortran formula (1-indexed; SW corner shown):

    ::

        dt6 = dt / 6.
        if (sw_corner) ke(1, 1) = dt6 * (
            (ut(1, 1) + ut(1, 0)) * u(1, 1) +
            (vt(1, 1) + vt(0, 1)) * v(1, 1) +
            (ut(1, 1) + vt(1, 1)) * u(0, 1) )

    Halo-input scope (Codex iter-869 caveat).  The right-hand side
    references halo cells (``ut(1, 0)``, ``vt(0, 1)``, ``u(0, 1)``,
    etc.).  Our Python pads ut/vt/u_d/v_d with ``mode='edge'``
    (same-face extension) — Fortran would have proper cross-face
    halo via ``mpp_update_domains``.  This is the same halo-quality
    gap as iter-862's d_sw5 corner corrections; the structural
    arithmetic of the fix is Fortran-faithful but the right-hand-side
    data is incomplete at cube vertices until a cross-face D-grid
    edge halo helper lands (deferred to iter-870+).

    Pure JAX-functional: returns a new ke array.  Caller is
    responsible for the duogrid / bounded_domain gate AND the
    iter-869 opt-in flag.

    Parameters
    ----------
    ke : jax.Array, shape (6, n+1, n+1)
        Corner KE field to update.
    ut : jax.Array, shape (6, n+1, n)
        Contravariant transport u at C-grid u-edges.
    vt : jax.Array, shape (6, n, n+1)
        Contravariant transport v at C-grid v-edges.
    u_d : jax.Array, shape (6, n, n+1)
        D-grid u at v-edge midpoints.
    v_d : jax.Array, shape (6, n+1, n)
        D-grid v at u-edge midpoints.
    dt : float
        Full time step.
    bounded_domain : bool
        From the caller.  If True, the function returns ``ke``
        unchanged (Fortran skips this fix in pure-duogrid mode).

    Returns
    -------
    ke_fixed : jax.Array, shape (6, n+1, n+1)
        Same shape as input; only the four cube-vertex corners
        change in the legacy (non-bounded-domain) branch.
    """
    if bounded_domain:
        return ke

    dt6 = dt / 6.0

    # Pad ut, vt with halo=1 mode='edge' so the Fortran formulas at
    # i=0/j=0/i=n/j=n halo positions can be evaluated.  iter-870+
    # tracks the cross-face halo upgrade.
    ut_pad = jnp.pad(ut, [(0, 0), (0, 0), (1, 1)], mode='edge')   # (6, n+1, n+2)
    vt_pad = jnp.pad(vt, [(0, 0), (1, 1), (0, 0)], mode='edge')   # (6, n+2, n+1)
    u_pad = jnp.pad(u_d, [(0, 0), (1, 1), (0, 0)], mode='edge')   # (6, n+2, n+1)
    v_pad = jnp.pad(v_d, [(0, 0), (0, 0), (1, 1)], mode='edge')   # (6, n+1, n+2)

    n = ke.shape[1] - 1  # ke is (6, n+1, n+1)
    # Index translation (Fortran 1-indexed → Python 0-indexed):
    #   Fortran i=1 (interior west boundary) → Python i=0.
    #   Fortran i=npx (interior east boundary) → Python i=n.
    #   Fortran ut at (1, 0) (south halo) → Python ut_pad at (0, 0)
    #     where ut_pad has axis-2 halo so j_pad=0 is the south halo.
    #   Fortran u(0, 1) (west halo, j=1) → Python u_pad at (0, 0)
    #     where u_pad has axis-1 halo so i_pad=0 is the west halo.

    # SW corner: Fortran ke(1,1).  Python: ke[:, 0, 0].
    sw_value = dt6 * (
        # (ut(1,1) + ut(1,0)) * u(1,1): ut at i=0 (Fortran 1), j=0
        # interior + j=-1 halo; u at i=0, j=0.
        (ut[:, 0, 0] + ut_pad[:, 0, 0]) * u_d[:, 0, 0]
        # (vt(1,1) + vt(0,1)) * v(1,1): vt at i=0 interior + i=-1
        # halo, j=0; v at i=0, j=0.
        + (vt[:, 0, 0] + vt_pad[:, 0, 0]) * v_d[:, 0, 0]
        # (ut(1,1) + vt(1,1)) * u(0,1): u at west halo i=-1, j=0.
        + (ut[:, 0, 0] + vt[:, 0, 0]) * u_pad[:, 0, 0]
    )

    # SE corner: Fortran ke(npx, 1).  Python: ke[:, -1, 0].
    se_value = dt6 * (
        # (ut(npx,1) + ut(npx,0)) * u(npx-1,1): ut at i=n, j interior
        # + halo south.  u(npx-1) = u at i=n-1.
        (ut[:, -1, 0] + ut_pad[:, -1, 0]) * u_d[:, -1, 0]
        # (vt(npx,1) + vt(npx-1,1)) * v(npx,1): vt at i=n + i=n-1; v at npx (i=n).
        + (vt_pad[:, -1, 0] + vt[:, -1, 0]) * v_d[:, -1, 0]
        # (ut(npx,1) - vt(npx-1,1)) * u(npx,1): u at east halo i=n.
        # Python u_pad has west-halo at i=0 and east-halo at i=-1
        # (i.e., n+1).  u(npx, 1) is east halo of u_d → u_pad[:, -1, 0].
        + (ut[:, -1, 0] - vt[:, -1, 0]) * u_pad[:, -1, 0]
    )

    # NE corner: Fortran ke(npx, npy).  Python: ke[:, -1, -1].
    ne_value = dt6 * (
        (ut[:, -1, -1] + ut[:, -1, -2]) * u_d[:, -1, -1]
        + (vt[:, -1, -1] + vt[:, -2, -1]) * v_d[:, -1, -1]
        + (ut[:, -1, -2] + vt[:, -2, -1]) * u_pad[:, -1, -1]
    )

    # NW corner: Fortran ke(1, npy).  Python: ke[:, 0, -1].
    nw_value = dt6 * (
        (ut[:, 0, -1] + ut[:, 0, -2]) * u_d[:, 0, -1]
        + (vt[:, 0, -1] + vt_pad[:, 0, -1]) * v_d[:, 0, -1]
        + (ut[:, 0, -2] - vt[:, 0, -1]) * u_pad[:, 0, -1]
    )

    ke_fixed = ke.at[:, 0, 0].set(sw_value)
    ke_fixed = ke_fixed.at[:, -1, 0].set(se_value)
    ke_fixed = ke_fixed.at[:, -1, -1].set(ne_value)
    ke_fixed = ke_fixed.at[:, 0, -1].set(nw_value)
    return ke_fixed


def _apply_legacy_d_sw5_corner_corrections(field_at_corners, edge_halo_field):
    """Iter-862 corner correction kernel for Fortran d_sw5 legacy path.

    Applies the four cube-vertex corner adjustments from
    Fortran sw_core.F90:1709-1715 (delpc/vort) and 1773-1776
    (divg_d/uc) on a (6, n+1, n+1) corner field using a (6, n+1, n+2)
    edge-halo field as the right-hand side:

        field[:, 0, 0]   -=  edge_halo[:, 0, 0]    # SW
        field[:, -1, 0]  -=  edge_halo[:, -1, 0]   # SE
        field[:, -1, -1] +=  edge_halo[:, -1, -1]  # NE
        field[:, 0, -1]  +=  edge_halo[:, 0, -1]   # NW

    Pure JAX-functional: returns a new array.  Caller is responsible
    for gating on `cdgrid.base.duogrid is None` (= Fortran's
    `.not. flagstruct%duogrid`) and the iter-862 opt-in flag.
    Factored out so unit tests can verify exact sign / index /
    magnitude on synthetic inputs that bypass the upstream face-
    boundary zeroing performed by `_divergence_corner_duo`.
    """
    f = field_at_corners
    f = f.at[:, 0, 0].add(-edge_halo_field[:, 0, 0])
    f = f.at[:, -1, 0].add(-edge_halo_field[:, -1, 0])
    f = f.at[:, -1, -1].add(edge_halo_field[:, -1, -1])
    f = f.at[:, 0, -1].add(edge_halo_field[:, 0, -1])
    return f


def _d_sw5_corner_divergence(u_d, v_d, ua, va, cdgrid, dt,
                             d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
                             apply_legacy_corner_corrections=False):
    """FV3 d_sw5 corner divergence damping (sw_core.F90:1641-1821).

    Computes divergence at D-grid corners and returns a damping term
    to be added to the KE field (ke += damp * delpc).

    For nord=0: del-2 damping with adaptive Smagorinsky coefficient.
    For nord>0: higher-order damping with iterated Laplacian.

    Matches Fortran sw_core.F90:1641-1821 for the duogrid path.

    Parameters
    ----------
    u_d : (6, n, n+1) D-grid x-velocity
    v_d : (6, n+1, n) D-grid y-velocity
    ua : (6, n, n) A-grid contravariant u
    va : (6, n, n) A-grid contravariant v
    cdgrid : CubedSphereCDGrid
    dt : float
    d2_bg : float — background del-2 coefficient (Fortran default 0.0)
    dddmp : float — adaptive del-2 coefficient (Fortran default 0.0)
    d4_bg : float — background del-4 coefficient (Fortran default 0.16)
    nord : int — damping order: 0=del-2, 1=del-4, etc.
    apply_legacy_corner_corrections : bool, default False
        Iter-862 opt-in for Check 3 from iter-849 d_sw5 fidelity audit.
        When ``True`` AND ``cdgrid.base.duogrid is None`` (= Fortran's
        ``.not. flagstruct%duogrid``), apply the four cube-vertex corner
        corrections from Fortran ``sw_core.F90:1709-1715`` (nord=0) and
        ``1773-1776`` (nord>=1 n-loop):

        ::

            if (sw_corner) delpc(1,    1) = delpc(1,    1) - vort(1,    0)
            if (se_corner) delpc(npx,  1) = delpc(npx,  1) - vort(npx,  0)
            if (ne_corner) delpc(npx,npy) = delpc(npx,npy) + vort(npx,npy)
            if (nw_corner) delpc(1,  npy) = delpc(1,  npy) + vort(1,  npy)

        DEFAULT IS FALSE — corrections are OFF by default.  Reason
        (Codex iter-862 finding): the right-hand-side ``vort_pad`` /
        ``uc_lap`` values come from the iter-655 ``mode='edge'`` same-
        face halo, which is documented as an O(1) approximation at
        cube vertices; the cross-face D-grid edge halo
        (``mpp_update_domains(DGRID_NE)`` analogue) does not yet
        exist in our Python.  Until that lands, applying the
        corner corrections by default could push legacy FB-chain
        runs FURTHER from Fortran at cube vertices because the
        right-hand-side data quality is incomplete.  The opt-in flag
        keeps the structural Fortran-faithful arithmetic ready for
        future experiments while the default behaviour stays bit-
        identical to the pre-iter-862 baseline.

        When ``False`` or ``cdgrid.base.duogrid is not None``, the
        corrections are skipped — output is bit-identical to the
        pre-iter-862 implementation.

        Production ``fv3_sw_tendencies`` does NOT call this helper, so
        production W2 / W5 / cosine bell sentinels are unaffected by
        either setting.

    Returns
    -------
    ke_damping : (6, n+1, n+1) — damping increment for ke_corner
    """
    n = cdgrid.n
    cosa_u = cdgrid.cosa_u    # (6, n+1, n)
    cosa_v = cdgrid.cosa_v    # (6, n, n+1)
    # sina_u/v from sin_sg sub-grid — factored into a shared helper
    # (iter-87) so `_vorticity_flux` uses the same Fortran-faithful
    # convention instead of `sqrt(1 - cosa**2)`.
    sina_u, sina_v = _sina_u_v_from_sin_sg(cdgrid)

    dxc = cdgrid.dxc          # (6, n+1, n)
    dyc = cdgrid.dyc          # (6, n, n+1)
    rarea_c = cdgrid.rarea_c  # (6, n+1, n+1)
    da_min_c = jnp.min(1.0 / rarea_c)  # minimum corner area

    # Iter-656 (Codex correction on iter-655): move the ua/va padding
    # INSIDE the `if nord == 0:` branch.  Iter-655 placed the
    # pad_halo call at module scope here, which fired unconditionally
    # even on the default `nord=1` path — wasted work plus a silent
    # halo dependency that didn't previously exist on that path.

    if nord == 0:
        # --- Del-2 divergence damping (sw_core.F90:1644-1724) ---
        # Pad ua, va for cross-velocity averages at boundaries.
        # Fortran FV3 fills ua/va halos via mpp_update_domains
        # (DGRID_NE) before d_sw5 fires; `mode='edge'` (the pre-iter-655
        # behaviour here) was a same-face 1D extension, NOT the
        # cross-face halo the Fortran oracle expects.  Iter-655 fixed
        # this for the nord=0 branch (only nord>=1 branch still uses
        # `_divergence_corner_duo` which has its own `mode='edge'`
        # gap flagged as unresolved).
        dg = cdgrid.base.duogrid
        _offs = None if dg is not None else cdgrid.base.halo_interp_offsets
        ua_full = pad_halo(ua, halo=1, interp_offsets=_offs, duogrid=dg)
        va_full = pad_halo(va, halo=1, interp_offsets=_offs, duogrid=dg)
        # Match the shape of the old jnp.pad call to preserve
        # downstream slicing: ua_pad is (6, n+2, n) — halo along
        # axis 1 only; va_pad is (6, n, n+2) — halo along axis 2.
        ua_pad = ua_full[:, :, 1:-1]
        va_pad = va_full[:, 1:-1, :]

        # Duogrid/bounded_domain path (lines 1644-1658):
        # ptc(i,j) = (u(i,j) - 0.5*(va(i,j-1)+va(i,j))*cosa_v(i,j))
        #            * dyc(i,j) * sina_v(i,j)
        va_below = va_pad[:, :, :-1]  # (6, n, n+1) va at j-1
        va_above = va_pad[:, :, 1:]   # (6, n, n+1) va at j
        ptc = (u_d - 0.5 * (va_below + va_above) * cosa_v) * dyc * sina_v

        # vort(i,j) = (v(i,j) - 0.5*(ua(i-1,j)+ua(i,j))*cosa_u(i,j))
        #             * dxc(i,j) * sina_u(i,j)
        ua_left = ua_pad[:, :-1, :]   # (6, n+1, n) ua at i-1
        ua_right = ua_pad[:, 1:, :]   # (6, n+1, n) ua at i
        vort = (v_d - 0.5 * (ua_left + ua_right) * cosa_u) * dxc * sina_u

        # delpc(i,j) = vort(i,j-1) - vort(i,j) + ptc(i-1,j) - ptc(i,j)
        # vort: (6, n+1, n), ptc: (6, n, n+1)
        # Corner stagger: need vort at j-1 and j, ptc at i-1 and i.
        # Iter-655: vort/ptc live on D-grid face midpoints (non-cell-
        # centre).  `pad_halo` is cell-centre; mode='edge' here is a
        # same-face extension that survives pending a proper edge-
        # midpoint halo exchange.  This is an acknowledged gap —
        # upgrading it requires an edge-midpoint halo helper that
        # does cross-face interpolation for u-edge / v-edge fields.
        # Flagged for future iter.
        vort_pad = jnp.pad(vort, [(0, 0), (0, 0), (1, 1)], mode='edge')
        ptc_pad = jnp.pad(ptc, [(0, 0), (1, 1), (0, 0)], mode='edge')

        delpc = (vort_pad[:, :, :-1] - vort_pad[:, :, 1:]
                 + ptc_pad[:, :-1, :] - ptc_pad[:, 1:, :])

        # Iter-862: cube-vertex corner corrections (Check 3 from iter-849
        # d_sw5 audit).  Fortran sw_core.F90:1709-1715 applies four corner
        # adjustments to `delpc` BEFORE the rarea_c scaling, gated by
        # `.not. flagstruct%duogrid`:
        #     if (sw_corner) delpc(1,    1) = delpc(1,    1) - vort(1,    0)
        #     if (se_corner) delpc(npx,  1) = delpc(npx,  1) - vort(npx,  0)
        #     if (ne_corner) delpc(npx,npy) = delpc(npx,npy) + vort(npx,npy)
        #     if (nw_corner) delpc(1,  npy) = delpc(1,  npy) + vort(1,  npy)
        # On a global cubed sphere every face has all four cube-vertex
        # corners, so all six faces apply all four corrections.  The
        # Fortran gate maps to `cdgrid.base.duogrid is None` here — i.e.,
        # legacy non-duogrid mode.  In duogrid mode Fortran skips these
        # adjustments because the duogrid halo for vort already supplies
        # the cross-face contribution implicitly.
        #
        # Index translation (Fortran 1-indexed → Python 0-indexed):
        #   delpc(1, 1)        → delpc[:, 0, 0]      (SW corner)
        #   delpc(npx, 1)      → delpc[:, -1, 0]     (SE corner)
        #   delpc(npx, npy)    → delpc[:, -1, -1]    (NE corner)
        #   delpc(1, npy)      → delpc[:, 0, -1]     (NW corner)
        #   vort(1, 0)         → vort_pad[:, 0, 0]   (south halo, west edge)
        #   vort(npx, 0)       → vort_pad[:, -1, 0]  (south halo, east edge)
        #   vort(npx, npy)     → vort_pad[:, -1, -1] (north halo, east edge)
        #   vort(1, npy)       → vort_pad[:, 0, -1]  (north halo, west edge)
        #
        # Scope of fidelity claim — STRUCTURAL ONLY.  iter-862 ports the
        # Fortran arithmetic STRUCTURE of the four corner adjustments
        # but the right-hand side `vort_pad[corner-halo]` values come
        # from the SAME `mode='edge'` halo whose limitations are
        # documented above (iter-655 same-face fallback at the j=0/j=n
        # halo cells).  At cube vertices the Fortran-faithful right-
        # hand-side would come from a true cross-face D-grid edge halo
        # exchange of `u_d, v_d` (Fortran's `mpp_update_domains` with
        # DGRID_NE) — that helper does not yet exist in our Python.
        # Until it lands, the Python correction's MAGNITUDE differs
        # from Fortran at cube vertices by an amount bounded by
        # `|vort_cross_face - vort_same_face|`.  Compared to the pre-
        # iter-862 baseline (no correction at all) this still moves
        # delpc closer to Fortran on average — it captures the
        # qualitative structure even when the cross-face data is
        # incomplete.  Future iter (iter-863+) tracks porting the
        # cross-face D-grid edge halo so the correction's right-hand
        # side becomes Fortran-faithful too.
        #
        # Production W2 path (`fv3_sw_tendencies`) does NOT call this
        # helper, so no production sentinel changes from this edit; the
        # FB chain (`_d_sw_native`) gains the structural correction in
        # its legacy-mode invocation ONLY when the caller passes
        # ``apply_legacy_corner_corrections=True`` (default False).
        # Arithmetic factored into `_apply_legacy_d_sw5_corner_corrections`
        # so its sign/index contract is testable on synthetic inputs.
        # iter-958 (NEGATIVE-RESULT): tested removing the iter-862
        # `cdgrid.base.duogrid is None` gate (so corrections fire on
        # duogrid too).  Result on duogrid C36 W2 1-day: bit-identical
        # to iter-947 baseline (v_ll_Linf=55.6130) — the corrections
        # are essentially no-ops because divg_d at cube vertices is
        # boundary-zeroed by `_divergence_corner_duo` and uc_lap at
        # the corresponding halo positions is also ~0 for W2 smooth
        # flow.  Original iter-862 gate restored.
        if (apply_legacy_corner_corrections
                and cdgrid.base.duogrid is None):
            delpc = _apply_legacy_d_sw5_corner_corrections(delpc, vort_pad)

        delpc = rarea_c * delpc

        # Adaptive Smagorinsky coefficient (sw_core.F90:1720-1721):
        # damp = da_min_c * max(d2_bg, min(0.20, dddmp*abs(delpc*dt)))
        damp = da_min_c * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * jnp.abs(delpc * dt)))
        ke_damping = damp * delpc

    else:
        # --- Higher-order divergence damping (sw_core.F90:1725-1821) ---
        # Matches Fortran structure: _divergence_corner_duo for initial divg_d,
        # then nord iterations of metric-weighted Laplacian using divg_u/divg_v,
        # plus del-2 + del-(2*nord+2) composite damping.
        divg_d = _divergence_corner_duo(u_d, v_d, ua, va, cdgrid)
        delpc = divg_d  # save for del-2 part

        # del-(2*nord+2) coefficient (sw_core.F90:1811)
        dd8 = (da_min_c * d4_bg) ** (nord + 1)

        # del-2 part (from Smagorinsky, sw_core.F90:1790-1805)
        if dddmp > 1e-5:
            # Interpolate relative vorticity to corners for Smagorinsky
            rarea = 1.0 / cdgrid.base.area
            dx_u = cdgrid.dx_edge_y
            dy_v = cdgrid.dy_edge_x
            wk = rarea * (u_d[:, :, :-1] * dx_u[:, :, :-1]
                          - u_d[:, :, 1:] * dx_u[:, :, 1:]
                          - v_d[:, :-1, :] * dy_v[:, :-1, :]
                          + v_d[:, 1:, :] * dy_v[:, 1:, :])
            # Iter-972: use Fortran-faithful 4th-order a2b_ord4
            # interpolation (matching sw_core.F90:1795 `a2b_ord4` call)
            # instead of the 2nd-order `_interp_center_to_corner`.
            wk_corner = _interp_center_to_corner_a2b_ord4(wk, cdgrid)
            # FV3_3D iter 183: use the JAX double-where trick (same
            # pattern as iter 181) to make the gradient through
            # ``sqrt`` finite at rest state.  Forward pass bit-for-bit
            # unchanged at any nonzero ``delpc² + wk² > 0``; exactly
            # 0 at rest.  Backward pass: gradient finite everywhere.
            _smag_arg = delpc ** 2 + wk_corner ** 2
            _safe_smag_arg = jnp.where(_smag_arg > 0.0, _smag_arg, 1.0)
            _smag_root = jnp.where(
                _smag_arg > 0.0, jnp.sqrt(_safe_smag_arg), 0.0,
            )
            smag_vort = jnp.abs(dt) * _smag_root
            damp2 = da_min_c * jnp.maximum(
                d2_bg, jnp.minimum(0.20, dddmp * smag_vort))
        else:
            damp2 = da_min_c * d2_bg

        # --- Iterated Laplacian using divg_u/divg_v metrics ---
        # Fortran sw_core.F90:1737-1787: nord iterations of the discrete
        # divergence-of-gradient operator on the corner grid.
        #
        # divg_u(i,j) = sina_v * dyc / dx  at (6, n, n+1) positions
        # divg_v(i,j) = sina_u * dxc / dy  at (6, n+1, n) positions
        # (fv_grid_utils.F90:709-735)
        dx = cdgrid.dx_edge_y   # (6, n, n+1)
        dy = cdgrid.dy_edge_x   # (6, n+1, n)
        divg_u_met = sina_v * dyc / jnp.maximum(dx, _EPS)  # (6, n, n+1)
        divg_v_met = sina_u * dxc / jnp.maximum(dy, _EPS)  # (6, n+1, n)

        for _it in range(nord):
            # Iter-132 Fortran-fidelity note: in the FV3 oracle,
            # sw_core.F90:1737-1785 accesses `divg_d(i+1,j)` etc. at
            # halo indices that were populated by the model-level MPI
            # `mpp_update_domains` before `d_sw5` was called.  The
            # duogrid branch (fill_c=False when `gridstruct%duogrid`,
            # lines 1740-1742) relies purely on that MPI halo + the
            # kinked-extended remap for the cross-face values.
            #
            # Python's single-process `mode='edge'` replicates the
            # boundary value — which agrees with Fortran ONLY for
            # boundary-parallel gradients at panel edges.  At CUBE
            # VERTICES and for gradients with non-zero cross-face
            # component, it is an O(1) approximation relative to the
            # Fortran MPI/duogrid halo.
            #
            # Impact: only the `_d_sw5_corner_divergence` path in the
            # FB chain (`fv3_forward_backward_step`, `fv3_fb_sw_step`),
            # which is experimental and unstable at C36 for
            # independent reasons (see docs/fv3_fortran_fidelity_
            # review.md item #2).  Production (A-L + RK3) uses a
            # different divergence damping at operators_cdgrid.py:
            # 1542-1552 and does not reach this path.
            divg_d_pad = jnp.pad(divg_d, [(0, 0), (1, 1), (1, 1)],
                                 mode='edge')
            # Pad metrics for extended gradient stencil
            divg_u_pad = jnp.pad(divg_u_met, [(0, 0), (1, 1), (0, 0)],
                                 mode='edge')  # (6, n+2, n+1)
            divg_v_pad = jnp.pad(divg_v_met, [(0, 0), (0, 0), (1, 1)],
                                 mode='edge')  # (6, n+1, n+2)

            # x-gradient: vc(i,j) = (divg_d(i+1,j)-divg_d(i,j)) * divg_u(i,j)
            # At (6, n+2, n+1) positions (sw_core.F90:1748-1752)
            vc_lap = ((divg_d_pad[:, 1:n+3, 1:n+2]
                       - divg_d_pad[:, 0:n+2, 1:n+2]) * divg_u_pad)

            # y-gradient: uc(i,j) = (divg_d(i,j+1)-divg_d(i,j)) * divg_v(i,j)
            # At (6, n+1, n+2) positions (sw_core.F90:1756-1760)
            uc_lap = ((divg_d_pad[:, 1:n+2, 1:n+3]
                       - divg_d_pad[:, 1:n+2, 0:n+2]) * divg_v_pad)

            # Convergence at corners (6, n+1, n+1) (sw_core.F90:1765-1769)
            # divg_d(i,j) = uc(i,j-1) - uc(i,j) + vc(i-1,j) - vc(i,j)
            divg_d = (uc_lap[:, :, :-1] - uc_lap[:, :, 1:]
                      + vc_lap[:, :-1, :] - vc_lap[:, 1:, :])

            # Iter-862: cube-vertex corner corrections inside the
            # iterated-Laplacian n-loop (Fortran sw_core.F90:1773-1776),
            # gated by `.not. flagstruct%duogrid`.  Same pattern as the
            # nord=0 branch but operating on `uc_lap` (Fortran name `uc`
            # in the n-loop) instead of `vort`:
            #     if (sw_corner) divg_d(1, 1)     -= uc(1, 0)
            #     if (se_corner) divg_d(npx, 1)   -= uc(npx, 0)
            #     if (ne_corner) divg_d(npx, npy) += uc(npx, npy)
            #     if (nw_corner) divg_d(1, npy)   += uc(1, npy)
            # Applied each iteration BEFORE the rarea_c scaling so the
            # subsequent gradient stencil sees corrected divergence at
            # cube vertices.  Like the nord=0 branch this only fires on
            # the legacy non-duogrid path; production unaffected.
            #
            # Halo-input scope (same caveat as the nord=0 branch above):
            # `uc_lap` was constructed from `divg_d_pad` with the
            # `mode='edge'` same-face halo, which the comment block
            # above acknowledges as an O(1) approximation at cube
            # vertices.  iter-862 ports the Fortran arithmetic
            # STRUCTURE of the corner correction but its right-hand
            # side inherits that halo's known imperfection.  Fortran-
            # faithful magnitude requires the cross-face halo path
            # (`mpp_update_domains` analogue) to land first, deferred
            # to iter-863+.  Until then the corrections are gated
            # behind the ``apply_legacy_corner_corrections`` opt-in
            # flag (default False) so callers cannot accidentally
            # apply Fortran-structure with Fortran-incomplete data.
            if (apply_legacy_corner_corrections
                    and cdgrid.base.duogrid is None):
                # iter-958 verified that removing the duogrid gate
                # here makes the corrections bit-identical no-ops on
                # duogrid (uc_lap ~0 at cube-vertex halo for W2).
                # Gate retained.
                divg_d = _apply_legacy_d_sw5_corner_corrections(
                    divg_d, uc_lap)

            # Scale by rarea_c (sw_core.F90:1780-1784)
            divg_d = divg_d * rarea_c

        # Composite damping: del-2 + del-(2*nord+2) (sw_core.F90:1814-1820)
        ke_damping = damp2 * delpc + dd8 * divg_d

    return ke_damping


def _corner_vorticity(uc, vc, cdgrid, use_duogrid):
    """FV3 c_sw corner vorticity from C-grid circulation (sw_core.F90:378-408).

    Returns vort_abs at D-grid corners (6, n+1, n+1).
    """
    n = cdgrid.n
    fx_circ = uc * cdgrid.dxc    # (6, n+1, n)
    fy_circ = vc * cdgrid.dyc    # (6, n, n+1)

    # Boundary padding for the circulation halo.
    # - Non-duogrid path: Fortran (sw_core.F90:396-400) applies linear
    #   extrapolation, so we start from mode='edge' then overwrite.
    # - Duogrid path (iter-836): Fortran relies on cross-face halo-
    #   exchanged uc/vc from `d2a2c_vect` applied to halo-filled u/v
    #   (`ext_vector`).  Plain `mode='edge'` on fx_circ/fy_circ loses
    #   the cross-face rotation and produces a 15.6 %-of-interior-
    #   scale error at cube-vertex corners on W2 IC (iter-836 diag).
    #   Replaced here by `pad_halo_vector` on cell-centre-averaged
    #   (uc, vc) — cross-face rotation-correct to leading order.
    if use_duogrid and n >= 2:
        # Halo-only fix (iter-836b after Codex adversarial review):
        # compute HALO fx/fy values from cross-face-rotated uc/vc
        # cell-centre averages, but PRESERVE interior fx_circ/fy_circ
        # exactly (do NOT smooth interior through an average-pad-average
        # round-trip).  Interior uc/vc came from the 4th-order A→C
        # stencil in `_d2a2c_vect_duogrid`; replacing them with 1-2-1-
        # smoothed versions would introduce an O(dx²) error in the
        # interior vort_abs that is NOT Fortran-faithful.
        uc_cc = 0.5 * (uc[:, :-1, :] + uc[:, 1:, :])   # (6, n, n)
        vc_cc = 0.5 * (vc[:, :, :-1] + vc[:, :, 1:])   # (6, n, n)
        grid = cdgrid.base
        dg = grid.duogrid
        # iter-837 (Codex fidelity fix): uc/vc are FV3 COVARIANT winds
        # (velocity dotted with face basis vectors), NOT grid-aligned
        # physical velocity.  Activate `pad_halo_vector`'s covariant
        # branch by passing `cos_theta`/`sin_theta` (the cell-centre
        # non-orthogonality metrics).  Without these, the default path
        # rotates as if the inputs were grid-aligned — the wrong
        # quantity per Codex WEAKENS finding iter-836b.
        uc_cc_pad, vc_cc_pad = pad_halo_vector(
            uc_cc, vc_cc,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
            interp_offsets=None, duogrid=dg, halo=1,
            cos_theta=cdgrid.cosa_cell,
            sin_theta=cdgrid.sina_cell,
        )  # (6, n+2, n+2) each — covariant-aware cross-face rotation
        # Extract HALO rows only and reconstruct face-staggered values.
        # uc_cc_pad[:, :, 0]   = uc_cc at j_cell = −1 (south halo)
        # uc_cc_pad[:, :, n+1] = uc_cc at j_cell =  n (north halo)
        uc_halo_j_below = 0.5 * (uc_cc_pad[:, :-1, 0] + uc_cc_pad[:, 1:, 0])
        uc_halo_j_above = 0.5 * (uc_cc_pad[:, :-1, n + 1]
                                  + uc_cc_pad[:, 1:, n + 1])
        vc_halo_i_left = 0.5 * (vc_cc_pad[:, 0, :-1] + vc_cc_pad[:, 0, 1:])
        vc_halo_i_right = 0.5 * (vc_cc_pad[:, n + 1, :-1]
                                  + vc_cc_pad[:, n + 1, 1:])
        # Metric halo: dxc/dyc continuous across seams; edge-mode for
        # the halo row (O(dx) error, much smaller than the 15.6 % uc
        # rotation error that mode='edge' on fx_circ produced).
        dxc_halo_j_below = cdgrid.dxc[:, :, 0]    # (6, n+1)
        dxc_halo_j_above = cdgrid.dxc[:, :, -1]   # (6, n+1)
        dyc_halo_i_left = cdgrid.dyc[:, 0, :]     # (6, n+1)
        dyc_halo_i_right = cdgrid.dyc[:, -1, :]   # (6, n+1)

        fx_halo_j_below = uc_halo_j_below * dxc_halo_j_below   # (6, n+1)
        fx_halo_j_above = uc_halo_j_above * dxc_halo_j_above   # (6, n+1)
        fy_halo_i_left = vc_halo_i_left * dyc_halo_i_left      # (6, n+1)
        fy_halo_i_right = vc_halo_i_right * dyc_halo_i_right   # (6, n+1)

        # Interior fx_pad = fx_circ; halo rows at j=-1 and j=n ONLY are
        # replaced with the rotated values.  Interior values unchanged.
        fx_pad = jnp.concatenate([
            fx_halo_j_below[:, :, jnp.newaxis],   # (6, n+1, 1) j=-1
            fx_circ,                               # (6, n+1, n) interior
            fx_halo_j_above[:, :, jnp.newaxis],   # (6, n+1, 1) j=n
        ], axis=2)  # (6, n+1, n+2)
        fy_pad = jnp.concatenate([
            fy_halo_i_left[:, jnp.newaxis, :],    # (6, 1, n+1) i=-1
            fy_circ,                               # (6, n, n+1) interior
            fy_halo_i_right[:, jnp.newaxis, :],   # (6, 1, n+1) i=n
        ], axis=1)  # (6, n+2, n+1)
    else:
        # Non-duogrid path: edge padding + linear extrapolation at the 4
        # panel-edge boundaries (sw_core.F90:396-400).
        fx_pad = jnp.pad(fx_circ, [(0, 0), (0, 0), (1, 1)], mode='edge')
        fy_pad = jnp.pad(fy_circ, [(0, 0), (1, 1), (0, 0)], mode='edge')
        if n > 2:
            fx_pad = fx_pad.at[:, :, 0].set(2 * fx_circ[:, :, 0] - fx_circ[:, :, 1])
            fx_pad = fx_pad.at[:, :, n + 1].set(2 * fx_circ[:, :, n - 1] - fx_circ[:, :, n - 2])
            fy_pad = fy_pad.at[:, 0, :].set(2 * fy_circ[:, 0, :] - fy_circ[:, 1, :])
            fy_pad = fy_pad.at[:, n + 1, :].set(2 * fy_circ[:, n - 1, :] - fy_circ[:, n - 2, :])

    # Direct corner vorticity: vort(i,j) = fx(i,j-1) - fx(i,j) - fy(i-1,j) + fy(i,j)
    vort = (fx_pad[:, :, :-1] - fx_pad[:, :, 1:]
            - fy_pad[:, :-1, :] + fy_pad[:, 1:, :])

    # Corner corrections for non-duogrid (FV3 sw_core.F90:396-400)
    if not use_duogrid:
        vort = vort.at[:, 0, 0].add(fy_pad[:, 0, 0])
        vort = vort.at[:, n, 0].add(-fy_pad[:, n + 1, 0])
        vort = vort.at[:, n, n].add(-fy_pad[:, n + 1, n])
        vort = vort.at[:, 0, n].add(fy_pad[:, 0, n])

    rarea_c = 1.0 / cdgrid.area_corner
    return cdgrid.f_corner + rarea_c * vort


def _vorticity_flux(v_d, u_d, uc, vc, vort_abs, cdgrid, use_duogrid):
    """FV3 c_sw vorticity transport flux (sw_core.F90:416-480).

    Returns fy1, vort_x (x-face) and fx1, vort_y (y-face).
    Uses 1/sin (NOT 1/sin²) per FV3 comment at sw_core.F90:417.

    `sina_u` / `sina_v` come from the sin_sg sub-grid as
    ``0.5*(sin_sg(i-1,j,3) + sin_sg(i,j,1))`` (matching
    `fv_grid_utils.F90:505-518`) rather than ``sqrt(1 - cosa**2)``.
    The two are not identical because `cosa_u` is a halo-averaged
    value of `cos_sg` and the trigonometric identity does not hold
    on averaged quantities.
    """
    n = cdgrid.n
    sina_u, sina_v = _sina_u_v_from_sin_sg(cdgrid)

    fy1 = (v_d - uc * cdgrid.cosa_u) / jnp.maximum(sina_u, _EPS)
    if not use_duogrid:
        fy1 = fy1.at[:, 0, :].set(v_d[:, 0, :])
        fy1 = fy1.at[:, n, :].set(v_d[:, n, :])
    vort_x = jnp.where(fy1 > 0, vort_abs[:, :, :-1], vort_abs[:, :, 1:])

    fx1 = (u_d - vc * cdgrid.cosa_v) / jnp.maximum(sina_v, _EPS)
    if not use_duogrid:
        fx1 = fx1.at[:, :, 0].set(u_d[:, :, 0])
        fx1 = fx1.at[:, :, n].set(u_d[:, :, n])
    vort_y = jnp.where(fx1 > 0, vort_abs[:, :-1, :], vort_abs[:, 1:, :])

    return fy1, vort_x, fx1, vort_y


# ==============================================================================
# c_sw: C-grid half-step
# ==============================================================================

def _c_sw(h, u_d, v_d, h_s, cdgrid, dt, g):
    """FV3 c_sw: C-grid half of the forward-backward step.

    Updates mass (h) via first-order upwind transport and C-grid
    covariant velocities (uc, vc) via vorticity flux + KE gradient.

    Parameters
    ----------
    h : (6, n, n) height at cell centres
    u_d : (6, n, n+1) D-grid x-velocity
    v_d : (6, n+1, n) D-grid y-velocity
    h_s : (6, n, n) surface topography
    cdgrid : CubedSphereCDGrid
    dt : float — full time step (half-step scaling applied internally)
    g : float — gravitational acceleration

    Returns
    -------
    h_star : (6, n, n) transported mass
    uc_new : (6, n+1, n) updated C-grid covariant u
    vc_new : (6, n, n+1) updated C-grid covariant v
    ua : (6, n, n) A-grid contravariant u (for diagnostics)
    va : (6, n, n) A-grid contravariant v
    """
    n = cdgrid.n
    dt2 = 0.5 * dt  # half-step
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2

    # 1. d2a2c_vect
    ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)

    # 2. Scale transport velocities with dt/2 * edge_length * sin_sg_upwind
    # Use proper halo exchange for sin_sg (not mode='edge') so cross-face
    # upwinding is correct at panel boundaries (FV3 fv_grid_utils.F90:570).
    # Route through the duogrid kinked-to-extended remap when duogrid is
    # active, matching compute_transport_quantities and the Fortran
    # `bounded_domain` path (sw_core.F90:830-862 skips copy_corners).
    dy = cdgrid.dy_edge_x   # (6, n+1, n)
    dx = cdgrid.dx_edge_y   # (6, n, n+1)
    sg = cdgrid.sin_sg
    grid = cdgrid.base
    _offs = None if use_duogrid else grid.halo_interp_offsets
    _dg = dg if use_duogrid else None

    # x-direction: ut → scaled flux (same pattern as compute_transport_quantities)
    sin_east = sg[:, :, :, 2]   # E-edge of each cell
    sin_west = sg[:, :, :, 0]   # W-edge of each cell
    se_pad = pad_halo(sin_east, interp_offsets=_offs, duogrid=_dg)
    sw_pad = pad_halo(sin_west, interp_offsets=_offs, duogrid=_dg)
    sin_upwind_x = jnp.where(ut > 0, se_pad[:, :n+1, 1:-1],
                                      sw_pad[:, 1:n+2, 1:-1])
    ut_scaled = dt2 * ut * dy * sin_upwind_x

    # y-direction: vt → scaled flux
    sin_north = sg[:, :, :, 3]  # N-edge of each cell
    sin_south = sg[:, :, :, 1]  # S-edge of each cell
    sn_pad = pad_halo(sin_north, interp_offsets=_offs, duogrid=_dg)
    ss_pad = pad_halo(sin_south, interp_offsets=_offs, duogrid=_dg)
    sin_upwind_y = jnp.where(vt > 0, sn_pad[:, 1:-1, :n+1],
                                      ss_pad[:, 1:-1, 1:n+2])
    vt_scaled = dt2 * vt * dx * sin_upwind_y

    # 3. First-order upwind mass transport
    h_pad = _pad_halo_auto(h, cdgrid)
    h_left = h_pad[:, :-1, 1:-1]   # (6, n+1, n)
    h_right = h_pad[:, 1:, 1:-1]
    fx = jnp.where(ut_scaled > 0, h_left, h_right) * ut_scaled

    h_bot = h_pad[:, 1:-1, :-1]    # (6, n, n+1)
    h_top = h_pad[:, 1:-1, 1:]
    fy = jnp.where(vt_scaled > 0, h_bot, h_top) * vt_scaled

    # Duogrid flux synchronization (FV3 dyn_core.F90:853-900)
    if use_duogrid:
        fx, fy = synchronize_cgrid_fluxes(fx, fy, n)

    rarea = 1.0 / cdgrid.base.area
    h_star = h + (fx[:, :-1, :] - fx[:, 1:, :] + fy[:, :, :-1] - fy[:, :, 1:]) * rarea

    # 4. KE at cell centres (FV3 sw_core.F90:303-372)
    ke_u, ke_v = _ke_upwind(uc, vc, ua, va, u_d, v_d, cdgrid, use_duogrid)
    # c_sw uses ONLY kinetic energy (no g*h) — per FV3 dt4 = 0.25*dt scaling
    ke_total = dt2 * 0.5 * (ua * ke_u + va * ke_v)

    # 5. Vorticity at D-grid corners (FV3 sw_core.F90:378-408)
    vort_abs = _corner_vorticity(uc, vc, cdgrid, use_duogrid)

    # 6. Vorticity flux at C-grid face positions (FV3 sw_core.F90:416-480)
    fy1, vort_x, fx1, vort_y = _vorticity_flux(
        v_d, u_d, uc, vc, vort_abs, cdgrid, use_duogrid)
    # Scale by dt/2 (c_sw half-step)
    fy1 = dt2 * fy1
    fx1 = dt2 * fx1

    # 7. KE gradient at C-grid face positions (2-point difference)
    ke_pad = _pad_halo_auto(ke_total, cdgrid)
    dke_x = cdgrid.rdxc * (ke_pad[:, :-1, 1:-1] - ke_pad[:, 1:, 1:-1])
    dke_y = cdgrid.rdyc * (ke_pad[:, 1:-1, :-1] - ke_pad[:, 1:-1, 1:])

    # 8. Update C-grid covariant velocities
    uc_new = uc + fy1 * vort_x + dke_x
    vc_new = vc - fx1 * vort_y + dke_y

    return h_star, uc_new, vc_new, ua, va


# ==============================================================================
# C-grid tendency for RK3 integration
# ==============================================================================

def fv3_csw_tendencies(h, u_d, v_d, h_s, cdgrid, g=constants.g,
                       div_damp=0.0, hyperdiff_coeff=0.0):
    """FV3 c_sw-style shallow water tendencies for RK3 integration.

    Computes the FULL Bernoulli gradient + vorticity flux at C-grid
    face positions (same stagger → geostrophic balance preserved), then
    projects the TOTAL C-grid tendency to D-grid edge-midpoint positions.

    Projecting the SUM (which is near zero for balanced flow) preserves
    balance, unlike projecting gradient and vorticity separately.

    Parameters
    ----------
    h : (6, n, n) height at cell centres
    u_d : (6, n, n+1) D-grid x-velocity
    v_d : (6, n+1, n) D-grid y-velocity
    h_s : (6, n, n) surface topography
    cdgrid : CubedSphereCDGrid
    g, div_damp, hyperdiff_coeff : float

    Returns
    -------
    dh_dt : (6, n, n)
    du_dt : (6, n, n+1)
    dv_dt : (6, n+1, n)
    """
    n = cdgrid.n

    # 1. d2a2c_vect: D→A→C for KE and vorticity (covariant convention)
    ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)

    # 2. Mass transport uses the PROVEN fv3_cc2c (physical face-normal).
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    uc_mass, vc_mass = fv3_cc2c(u_cc, v_cc, cdgrid)
    dh_dt = cgrid_mass_flux_divergence(h, uc_mass, vc_mass, cdgrid)

    # 3. KE from physical-frame D-grid winds (avoids 1/sin² amplification
    # of the contravariant ua*uc formula at face boundaries).
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2
    utmp_ke = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])   # (6, n, n)
    vtmp_ke = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    ke = 0.5 * (utmp_ke**2 + vtmp_ke**2)
    B = ke + g * (h + h_s)

    # 4. Bernoulli gradient at C-grid face positions (2-point difference)
    B_pad = _pad_halo_auto(B, cdgrid)
    dB_x = cdgrid.rdxc * (B_pad[:, :-1, 1:-1] - B_pad[:, 1:, 1:-1])  # (6, n+1, n)
    dB_y = cdgrid.rdyc * (B_pad[:, 1:-1, :-1] - B_pad[:, 1:-1, 1:])  # (6, n, n+1)

    # 5. Vorticity at D-grid corners (FV3 sw_core.F90:378-408)
    vort_abs = _corner_vorticity(uc, vc, cdgrid, use_duogrid)

    # 6. Vorticity flux at C-grid face positions (FV3 sw_core.F90:416-480)
    fy1, vort_x, fx1, vort_y = _vorticity_flux(
        v_d, u_d, uc, vc, vort_abs, cdgrid, use_duogrid)

    # 7. TOTAL C-grid tendency = vorticity flux + Bernoulli gradient
    duc = fy1 * vort_x + dB_x    # (6, n+1, n)
    dvc = -fx1 * vort_y + dB_y   # (6, n, n+1)

    # 8. Divergence damping (optional).
    # The momentum equation with divergence damping is
    #     ∂u/∂t = ... + K_d · ∂D/∂x
    # so that taking the divergence gives ∂D/∂t = K_d · ∇²D — high-k
    # divergence noise decays as exp(-K_d·k²·t).  ``ddiv_x`` here
    # uses the same negated-gradient stencil as ``dB_x`` above
    # (``(D[:-1] - D[1:])/dxc = -∂D/∂x``), so to add ``+K_d · ∂D/∂x``
    # to ``duc`` we must SUBTRACT ``div_damp * ddiv_x``.  The prior
    # iter-57 audit found ``+ div_damp * ddiv_x`` (anti-damping):
    # ∂D/∂t = -K_d · ∇²D → exp(+K_d·k²·t) → grid-scale divergence
    # noise GROWS exponentially.  This path is reachable only via
    # ``shallow_water_fv3_cdgrid.use_experimental_csw=True`` (legacy
    # RK3 wrapper); the production FB chain uses a separate
    # divergence-damping formulation in ``_d_sw5_corner_divergence``.
    if div_damp > 0:
        div_field = cgrid_divergence(uc, vc, cdgrid)
        div_pad = _pad_halo_auto(div_field, cdgrid)
        ddiv_x = cdgrid.rdxc * (div_pad[:, :-1, 1:-1] - div_pad[:, 1:, 1:-1])
        ddiv_y = cdgrid.rdyc * (div_pad[:, 1:-1, :-1] - div_pad[:, 1:-1, 1:])
        duc = duc - div_damp * ddiv_x
        dvc = dvc - div_damp * ddiv_y

    # 9. Project TOTAL C-grid tendency → D-grid edge midpoints via
    # halo-exchanged cell-centre averaging. The previous edge-copy padding
    # created a linear instability at face corners (blowup at ~2h).
    # Now: C-grid → cell-centre average → vector halo exchange → D-grid.
    grid = cdgrid.base
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    # Step 1: average C-grid to cell centres
    duc_cc = 0.5 * (duc[:, :-1, :] + duc[:, 1:, :])   # (6, n, n)
    dvc_cc = 0.5 * (dvc[:, :, :-1] + dvc[:, :, 1:])   # (6, n, n)
    # Step 2: vector halo exchange (rotates tendencies across face boundaries)
    duc_pad, dvc_pad = pad_halo_vector(
        duc_cc, dvc_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    # Step 3: average haloed cell centres to D-grid edge midpoints
    du_dt = 0.5 * (duc_pad[:, 1:-1, :-1] + duc_pad[:, 1:-1, 1:])   # (6, n, n+1)
    dv_dt = 0.5 * (dvc_pad[:, :-1, 1:-1] + dvc_pad[:, 1:, 1:-1])   # (6, n+1, n)

    return dh_dt, du_dt, dv_dt


# ==============================================================================
# Complete forward-backward step
# ==============================================================================

def fv3_forward_backward_step(h, u_d, v_d, h_s, cdgrid, dt, g=constants.g,
                               div_damp=0.0, hyperdiff_coeff=0.0,
                               apply_legacy_d_sw4_corner_ke_fix=False,
                               apply_legacy_d_sw5_corner_corrections=False,
                               apply_fortran_xppm_boundary=False):
    """EXPERIMENTAL: One complete FV3 forward-backward time step for shallow water.

    Known unstable — produces large errors by step ~50.  Use
    ``fv3_sw_tendencies`` (operators_cdgrid.py) with RK3 integration
    for production work.

    Combines c_sw (C-grid half) + d_sw (D-grid half).  The c_sw half
    uses FV3's covariant velocity convention with sin_sg flux scaling;
    the d_sw half uses the existing corner-based operators (consistent
    at D-grid stagger).  The forward-backward coupling connects them
    without projecting between staggers.

    Parameters
    ----------
    h : (6, n, n) height at cell centres
    u_d : (6, n, n+1) D-grid x-velocity at x-edge midpoints
    v_d : (6, n+1, n) D-grid y-velocity at y-edge midpoints
    h_s : (6, n, n) surface topography
    cdgrid : CubedSphereCDGrid
    dt : float — full time step
    g : float
    div_damp : float — LEGACY, UNUSED in this FB-chain entry point.
        Forwarded to `_d_sw_native` which ignores it.  See
        `_d_sw_native` docstring for the split across paths (three
        LIVE-use paths at `cdgrid_momentum_tendencies`,
        `fv3_sw_tendencies` A-L production, and `fv3_csw_tendencies`
        experimental CSW — they share
        `CDGridShallowWaterConfig.div_damp`).  The FB chain's
        divergence damping comes from d_sw5 coefficients
        (d2_bg/dddmp/d4_bg/nord), not this field.
    hyperdiff_coeff : float — biharmonic hyperdiffusion coefficient

    Returns
    -------
    h_new : (6, n, n)
    u_d_new : (6, n, n+1)
    v_d_new : (6, n+1, n)
    """
    # Phase 1: C-grid half-step (c_sw) — mass transport + velocity update
    h_star, uc_new, vc_new, ua, va = _c_sw(
        h, u_d, v_d, h_s, cdgrid, dt, g)

    # Phase 2: Backward pressure gradient at C-grid (p_grad_c)
    dt2 = 0.5 * dt
    dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt2, g)
    uc_new = uc_new + dp_x
    vc_new = vc_new + dp_y

    # Phase 3: D-grid half-step using FV3-native transport operators
    # (NOT Arakawa-Lamb — uses PPM mass transport + KE/vort transport)
    # Iter-871c: forward iter-869b/iter-871b opt-in flags so a caller
    # of `fv3_forward_backward_step` can opt into the d_sw4 corner-KE
    # fix and the d_sw5 corner corrections.  Default-off preserves
    # bit-identical behaviour for existing callers.
    h_new, u_d_new, v_d_new = _d_sw_native(
        h, u_d, v_d, h_s, uc_new, vc_new, ua, va, cdgrid, dt, g,
        div_damp=div_damp,
        apply_legacy_d_sw4_corner_ke_fix=apply_legacy_d_sw4_corner_ke_fix,
        apply_legacy_d_sw5_corner_corrections=(
            apply_legacy_d_sw5_corner_corrections),
        apply_fortran_xppm_boundary=apply_fortran_xppm_boundary)

    return h_new, u_d_new, v_d_new


# ==============================================================================
# NEW: FV3-faithful forward-backward WITHOUT Arakawa-Lamb gradient
# ==============================================================================

def _p_grad_c(h_star, h_s, cdgrid, dt2, g):
    """Backward pressure gradient at C-grid positions.

    Applies g*grad(h_star + h_s) at C-grid face positions using a 2-point
    divided difference.  This is the "backward-in-time" step that couples
    mass and momentum implicitly, providing stability for gravity waves.

    Parameters
    ----------
    h_star : (6, n, n) — half-step height from c_sw
    h_s : (6, n, n) — surface topography
    cdgrid : CubedSphereCDGrid
    dt2 : float — dt/2
    g : float

    Returns
    -------
    dp_x : (6, n+1, n) — pressure gradient contribution to uc
    dp_y : (6, n, n+1) — pressure gradient contribution to vc
    """
    p = g * (h_star + h_s)
    p_pad = _pad_halo_auto(p, cdgrid)
    # 2-point gradient at C-grid faces (same sign convention as c_sw KE gradient)
    dp_x = dt2 * cdgrid.rdxc * (p_pad[:, :-1, 1:-1] - p_pad[:, 1:, 1:-1])
    dp_y = dt2 * cdgrid.rdyc * (p_pad[:, 1:-1, :-1] - p_pad[:, 1:-1, 1:])
    return dp_x, dp_y


def _ppm_transport_1d(field, courant, rdelta, axis, external_halo: int = 0,
                       apply_d_sw3_boundary_fix: bool = False,
                       boundary_fix_dx_field=None):
    """PPM hord=9 transport of a staggered field along one axis.

    Implements the Fortran ytp_v / xtp_u (sw_core.F90:2897-3353,
    2540-2894) for the jord>=8 branch with jord=9 limiting.
    Used for B-grid KE transport in d_sw3.

    The field has N cells in the sweep direction. The Courant number
    has N+1 interfaces (corners). Transport produces N+1 output values
    at the interface positions.

    For the duogrid path, boundary handling is skipped — the interior
    PPM stencil is applied everywhere, using edge-copy padding where
    the stencil reaches beyond the domain.

    Parameters
    ----------
    field : (6, ..., N + 2*external_halo, ...) — field to transport, optionally
        with cross-face halo cells already filled along the sweep axis.
    courant : (6, ..., N+1, ...) — Courant number at INTERIOR interfaces
        (units of distance).
    rdelta : (6, ..., N, ...) — 1/cell_width at INTERIOR field positions
        (1/dy or 1/dx).
        Used to convert the distance-based courant to dimensionless CFL fraction:
        ``cfl = |courant| * rdelta`` (FV3 sw_core.F90:3342).
    axis : int (1 or 2) — sweep axis.
    external_halo : int (default 0) — iter-945: when > 0, ``field`` is
        considered to already have ``external_halo`` cells of cross-face
        halo along the sweep axis on each side.  The PPM stencil
        internally pads to ``h3=4`` total halo, so ``mode='edge'`` is
        applied only to the gap (``h3 - external_halo``).  When 0,
        behaviour is bit-identical to the pre-iter-945 path
        (mode='edge' for the full h3=4 halo).

    Returns
    -------
    flux : (6, ..., N+1, ...) — transported field at interfaces.
    """
    # Transpose so sweep axis is axis 1 for uniform indexing
    if axis == 1:
        v = field    # (6, N + 2*ext, M)
        c = courant  # (6, N+1, M)
        rd = rdelta  # (6, N, M)
    else:
        v = jnp.swapaxes(field, 1, 2)     # (6, N + 2*ext, M)
        c = jnp.swapaxes(courant, 1, 2)   # (6, N+1, M)
        rd = jnp.swapaxes(rdelta, 1, 2)   # (6, N, M)

    nn = v.shape[1] - 2 * external_halo  # interior cells along sweep axis

    # Bring field to total halo h3=4 along the sweep axis.  External
    # cross-face halo (depth ``external_halo``) is preserved; the remaining
    # ``h3 - external_halo`` cells are filled by ``mode='edge'`` (or, if
    # external_halo > h3, we trim to the inner h3 cells).
    h3 = 4
    if external_halo == h3:
        vp = v
    elif external_halo < h3:
        gap = h3 - external_halo
        vp = jnp.pad(v, [(0, 0), (gap, gap), (0, 0)], mode='edge')
    else:
        trim = external_halo - h3
        vp = v[:, trim:v.shape[1] - trim, :]
    # vp shape: (6, N + 2*h3, M) — same as pre-iter-945 internal layout

    # --- Monotone slopes (FV3 sw_core.F90:3165-3171) ---
    # dm[j] at each padded cell. We compute for padded cells 1..N+4 (need j±1).
    xt = 0.25 * (vp[:, 2:, :] - vp[:, :-2, :])  # (6, N+4, M), at padded cells 1..N+4
    vm = vp[:, 1:-1, :]  # (6, N+4, M)
    vhi = jnp.maximum(jnp.maximum(vp[:, :-2, :], vm), vp[:, 2:, :])
    vlo = jnp.minimum(jnp.minimum(vp[:, :-2, :], vm), vp[:, 2:, :])
    dm = jnp.sign(xt) * jnp.minimum(
        jnp.abs(xt), jnp.minimum(vhi - vm, vm - vlo))
    # dm[k] = slope at padded cell k+1 (k=0..N+3)

    # --- Cell differences (FV3 sw_core.F90:3173-3177) ---
    dq = vp[:, 1:, :] - vp[:, :-1, :]  # (6, N+5, M), dq[k] = v[k+1]-v[k] at padded k

    # --- Edge values (FV3 sw_core.F90:3180-3183) ---
    # al at interface between padded cells k and k+1:
    #   al = 0.5*(v[k]+v[k+1]) + r3*(dm_at_k - dm_at_k+1)
    # dm_at_k = dm[k-1] (dm starts at padded cell 1, so dm index = padded cell - 1)
    r3 = 1.0 / 3.0
    al = (0.5 * (vp[:, 1:-2, :] + vp[:, 2:-1, :])
          + r3 * (dm[:, :-1, :] - dm[:, 1:, :]))
    # al[k] = interface between padded cells k+1 and k+2, k=0..N+2
    # al shape: (6, N+3, M)

    # --- PPM reconstruction jord=9 from ytp_v (sw_core.F90:3194-3204) ---
    # IMPORTANT: ytp_v/xtp_u (B-grid wind transport) uses pmp/lac limiter
    # for jord=9, NOT pert_ppm(iv=0).  The transported fields (D-grid winds)
    # are SIGNED, so the positive-definite constraint would incorrectly zero
    # the reconstruction for negative wind cells.  Only tp_core.F90 xppm/yppm
    # (mass/vorticity transport via fv_tp_2d) uses pert_ppm(iv=0) for iord=9.
    #
    # Need bl/br for cells -1..N (for flux at interfaces 0..N).
    nc = nn + 2  # cells -1..N

    # Indices in padded coordinates for cells -1..N:
    # Cell j (original 0-based) sits at padded index j+h3.
    # Cell -1 → padded h3-1=3; cell N → padded h3+N=nn+4.
    # al[k] = edge between padded cells k+1 and k+2.
    # Left edge of cell j: al[j+h3-2]. Right edge: al[j+h3-1].
    al_l = al[:, 1:1+nc, :]    # al_left for cells -1..N
    al_r = al[:, 2:2+nc, :]    # al_right for cells -1..N
    v_c = vp[:, h3-1:h3-1+nc, :]   # v at cells -1..N

    # pmp/lac coefficients (sw_core.F90:3197-3202):
    #   pmp_1 = -2*dq[j],  lac_1 = pmp_1 + 1.5*dq[j+1]  (bl direction)
    #   pmp_2 = 2*dq[j-1], lac_2 = pmp_2 - 1.5*dq[j-2]  (br direction)
    # where dq[j] = v[j+1]-v[j] (single difference).
    # dq_at_cell(j) = dq[j+h3] in padded indexing (dq[k]=vp[k+1]-vp[k]).
    # For cells -1..N: dq starts at index h3-1.
    p_off = h3 - 1  # dq at cell j is at dq index j + p_off
    pmp_1 = -2.0 * dq[:, p_off:p_off+nc, :]           # -2*dq[j] for j=-1..N
    lac_1 = pmp_1 + 1.5 * dq[:, p_off+1:p_off+1+nc, :]  # + 1.5*dq[j+1]
    pmp_2 = 2.0 * dq[:, p_off-1:p_off-1+nc, :]        # 2*dq[j-1]
    lac_2 = pmp_2 - 1.5 * dq[:, p_off-2:p_off-2+nc, :]  # - 1.5*dq[j-2]

    z = jnp.zeros_like(pmp_1)
    bl = jnp.minimum(
        jnp.maximum(jnp.maximum(z, pmp_1), lac_1),
        jnp.maximum(al_l - v_c,
                     jnp.minimum(jnp.minimum(z, pmp_1), lac_1)))
    br = jnp.minimum(
        jnp.maximum(jnp.maximum(z, pmp_2), lac_2),
        jnp.maximum(al_r - v_c,
                     jnp.minimum(jnp.minimum(z, pmp_2), lac_2)))
    # bl, br: (6, nc, M) for cells -1..N (index 0..nc-1)

    # Iter-967: d_sw3 cube-edge boundary fix (sw_core.F90:3239-3316
    # ytp_v branch; sw_core.F90:2819-2863 xtp_u branch — same formula
    # mirrored across the sweep axis).
    #
    # CRITICAL Fortran detail: d_sw3 calls ytp_v / xtp_u with
    # ``bounded_domain=.false.`` HARDCODED at lines 1315-1316 and
    # 1373-1374, regardless of the global bounded_domain flag.  The
    # boundary fix in xtp_u at line 2819 fires when
    # ``(.not. bounded_domain .or. .not. dg%is_initialized)`` — with
    # the HARDCODED .false., the condition becomes
    # ``(.not. .false. .or. ...) = .true.``, so the fix ALWAYS FIRES
    # for d_sw3 wind transport on cube-face boundaries (regardless
    # of duogrid status).
    #
    # Constants from sw_core.F90:38: s11=11/14, s14=4/7, s15=3/14.
    # Index map: Fortran cell j ↔ Python k = j (when bl/br is indexed
    # 0..nc-1 over Python cells -1..N, where Python cell j = Fortran
    # cell j+1 → bl/br index k_python = Fortran j_fortran).
    #
    # Boundary overrides at js=1 (south boundary):
    #   br(2) = al(3) - v(2)
    #   xt = s15*v(1) + s11*v(2) - s14*dm(2)
    #   br(1) = xt - v(1);  bl(2) = xt - v(2)
    #   bl(0) = s14*dm(-1) - s11*dq(-1)
    #   xt = (length-weighted xt of v(0)/v(-1) and v(1)/v(2) extrap)
    #   bl(1) = xt - v(1);  br(0) = xt - v(0)
    #   pert_ppm(v(2), bl(2), br(2), iv=-1) → standard PPM constraint
    if apply_d_sw3_boundary_fix:
        s11_c = 11.0 / 14.0
        s14_c = 4.0 / 7.0
        s15_c = 3.0 / 14.0
        # Index map (Fortran j_F, 1-indexed) → (Python k):
        #   - vp index for Fortran v(j_F)        : h3 + j_F - 1
        #   - dm index for Fortran dm(j_F)       : h3 + j_F - 2
        #   - dq index for Fortran dq(j_F)       : h3 + j_F - 2
        #   - al index for Fortran al(j_F)       : h3 + j_F - 3
        #   - bl/br index for Fortran bl(j_F)    : j_F (with my k=0 → halo cell -1)
        # Substitutions:
        #   Fortran j_F=npy-X with npy=N+1 → j_F = N+1-X.
        # E.g., v(npy-2) → vp[h3 + (N+1-2) - 1] = vp[h3 + N - 2].

        # SOUTH halo + interior cells (Fortran j_F ∈ {-1, 0, 1, 2, 3}):
        v_jm1 = vp[:, h3 - 2, :]   # Fortran v(-1)
        v_j0 = vp[:, h3 - 1, :]    # v(0)
        v_j1 = vp[:, h3, :]        # v(1)
        v_j2 = vp[:, h3 + 1, :]    # v(2)
        # dm at Fortran cell j_F : dm index = h3 + j_F - 2
        dm_jm1 = dm[:, h3 - 3, :]  # dm(-1)
        dm_j2 = dm[:, h3, :]       # dm(2)
        # dq at Fortran cell j_F : dq index = h3 + j_F - 2
        dq_jm1 = dq[:, h3 - 3, :]  # dq(-1)
        # al(3) — Fortran al(j_F=3) → al index = h3 + 3 - 3 = h3
        al_j3 = al[:, h3, :]

        # NORTH halo + interior cells (Fortran j_F ∈ {npy-2, npy-1, npy, npy+1}
        #                              = {N-1, N, N+1, N+2}):
        v_npy_m2 = vp[:, h3 + nn - 2, :]  # v(npy-2) = v(N-1)
        v_npy_m1 = vp[:, h3 + nn - 1, :]  # v(npy-1) = v(N)
        v_npy = vp[:, h3 + nn, :]         # v(npy) = v(N+1)
        v_npy_p1 = vp[:, h3 + nn + 1, :]  # v(npy+1) = v(N+2)
        # dm(npy-2) = dm(N-1) → index = h3 + (N-1) - 2 = h3+N-3
        dm_npy_m2 = dm[:, h3 + nn - 3, :]
        dm_npy_p1 = dm[:, h3 + nn, :]   # dm(npy+1) = dm(N+2) → index h3+N
        # dq(npy) = dq(N+1) → index = h3 + (N+1) - 2 = h3+N-1
        dq_npy = dq[:, h3 + nn - 1, :]
        # al(npy-2) → al index = h3 + (N-1) - 3 = h3+N-4
        al_npy_m2 = al[:, h3 + nn - 4, :]

        # Optional length-weighted xt via dx (boundary_fix_dx_field).
        # The caller provides INTERIOR shape (no halo); we pad to match
        # vp's h3=4 internal halo via mode='edge' so the boundary
        # formula's halo references work.
        if boundary_fix_dx_field is not None:
            if axis == 2:
                dxf_int = jnp.swapaxes(boundary_fix_dx_field, 1, 2)
            else:
                dxf_int = boundary_fix_dx_field
            # Pad sweep axis to total halo h3 to match vp.
            dxf = jnp.pad(dxf_int, [(0, 0), (h3, h3), (0, 0)],
                           mode='edge')
            dx_m2 = dxf[:, h3 - 2, :]
            dx_m1 = dxf[:, h3 - 1, :]
            dx_1 = dxf[:, h3, :]
            dx_2 = dxf[:, h3 + 1, :]
            dx_npy_m2 = dxf[:, h3 + nn - 2, :]
            dx_npy_m1 = dxf[:, h3 + nn - 1, :]
            dx_npy = dxf[:, h3 + nn, :]
            dx_npy_p1 = dxf[:, h3 + nn + 1, :]
        else:
            dx_m2 = dx_m1 = dx_1 = dx_2 = None
            dx_npy_m2 = dx_npy_m1 = dx_npy = dx_npy_p1 = None

        # SOUTH boundary fix (js==1) — overrides bl/br at Python k = 0, 1, 2
        # (Fortran j = 0, 1, 2).
        # br(2) = al(3) - v(2)
        br = br.at[:, 2, :].set(al_j3 - v_j2)
        # xt = s15*v(1) + s11*v(2) - s14*dm(2)
        xt_s = s15_c * v_j1 + s11_c * v_j2 - s14_c * dm_j2
        br = br.at[:, 1, :].set(xt_s - v_j1)
        bl = bl.at[:, 2, :].set(xt_s - v_j2)
        # bl(0) = s14*dm(-1) - s11*dq(-1)
        bl = bl.at[:, 0, :].set(s14_c * dm_jm1 - s11_c * dq_jm1)
        # ELSE branch (length-weighted xt for bl(1), br(0)):
        if dx_m1 is not None:
            x0L = 0.5 * (
                ((2.0 * dx_m1 + dx_m2) * v_j0 - dx_m1 * v_jm1)
                / jnp.maximum(dx_m1 + dx_m2, _EPS)
            )
            x0R = 0.5 * (
                ((2.0 * dx_1 + dx_2) * v_j1 - dx_1 * v_j2)
                / jnp.maximum(dx_1 + dx_2, _EPS)
            )
            xt_s2 = x0L + x0R
        else:
            xt_s2 = 0.5 * ((1.5 * v_j0 - 0.5 * v_jm1)
                            + (1.5 * v_j1 - 0.5 * v_j2))
        bl = bl.at[:, 1, :].set(xt_s2 - v_j1)
        br = br.at[:, 0, :].set(xt_s2 - v_j0)

        # NORTH boundary fix (je+1==npy) — overrides bl/br at Python k =
        # npy-2, npy-1, npy = N-1, N, N+1 (Fortran j=npy-2, npy-1, npy).
        k_nm2 = nn - 1
        k_nm1 = nn
        k_n = nn + 1

        # bl(npy-2) = al(npy-2) - v(npy-2)
        bl = bl.at[:, k_nm2, :].set(al_npy_m2 - v_npy_m2)
        # xt = s15*v(npy-1) + s11*v(npy-2) + s14*dm(npy-2)
        xt_n = s15_c * v_npy_m1 + s11_c * v_npy_m2 + s14_c * dm_npy_m2
        br = br.at[:, k_nm2, :].set(xt_n - v_npy_m2)
        bl = bl.at[:, k_nm1, :].set(xt_n - v_npy_m1)
        # br(npy) = s11*dq(npy) - s14*dm(npy+1)
        br = br.at[:, k_n, :].set(s11_c * dq_npy - s14_c * dm_npy_p1)
        # ELSE branch (length-weighted xt for br(npy-1), bl(npy)):
        if dx_npy_m1 is not None:
            x0L_n = 0.5 * (
                ((2.0 * dx_npy_m1 + dx_npy_m2) * v_npy_m1
                  - dx_npy_m1 * v_npy_m2)
                / jnp.maximum(dx_npy_m1 + dx_npy_m2, _EPS)
            )
            x0R_n = 0.5 * (
                ((2.0 * dx_npy + dx_npy_p1) * v_npy
                  - dx_npy * v_npy_p1)
                / jnp.maximum(dx_npy + dx_npy_p1, _EPS)
            )
            xt_n2 = x0L_n + x0R_n
        else:
            xt_n2 = 0.5 * ((1.5 * v_npy_m1 - 0.5 * v_npy_m2)
                            + (1.5 * v_npy - 0.5 * v_npy_p1))
        br = br.at[:, k_nm1, :].set(xt_n2 - v_npy_m1)
        bl = bl.at[:, k_n, :].set(xt_n2 - v_npy)

        # pert_ppm(iv=1) at j=2 and j=npy-2 (standard PPM constraint).
        bl_2 = bl[:, 2, :]
        br_2 = br[:, 2, :]
        bl_2_new, br_2_new = _pert_ppm(bl_2, br_2)
        bl = bl.at[:, 2, :].set(bl_2_new)
        br = br.at[:, 2, :].set(br_2_new)
        bl_nm2 = bl[:, k_nm2, :]
        br_nm2 = br[:, k_nm2, :]
        bl_nm2_new, br_nm2_new = _pert_ppm(bl_nm2, br_nm2)
        bl = bl.at[:, k_nm2, :].set(bl_nm2_new)
        br = br.at[:, k_nm2, :].set(br_nm2_new)

    # --- Flux evaluation (FV3 sw_core.F90:3339-3349) ---
    # cfl = c * rdy[j-1] (positive) or c * rdy[j] (negative)
    # Pad rdelta to get rdy at interface-adjacent cells
    rd_pad = jnp.pad(rd, [(0, 0), (1, 1), (0, 0)], mode='edge')  # (6, N+2, M)
    # Interface j: upwind cell j-1 → rd_pad[:, j, :], downwind cell j → rd_pad[:, j+1, :]
    rdy_pos = rd_pad[:, :nn+1, :]   # rdy[j-1] for j=0..N
    rdy_neg = rd_pad[:, 1:nn+2, :]  # rdy[j] for j=0..N

    v_pos = vp[:, h3-1:h3-1+nn+1, :]  # v[j-1] for j=0..N
    v_neg = vp[:, h3:h3+nn+1, :]      # v[j] for j=0..N
    bl_pos = bl[:, :nn+1, :]           # bl[j-1]
    br_pos = br[:, :nn+1, :]           # br[j-1]
    bl_neg = bl[:, 1:nn+2, :]          # bl[j]
    br_neg = br[:, 1:nn+2, :]          # br[j]

    # Fortran ytp_v (sw_core.F90:3342): cfl = c * rdy (no clamping)
    cfl_pos = jnp.abs(c) * rdy_pos
    cfl_neg = jnp.abs(c) * rdy_neg

    flux_pos = v_pos + (1.0 - cfl_pos) * (br_pos - cfl_pos * (bl_pos + br_pos))
    flux_neg = v_neg + (1.0 - cfl_neg) * (bl_neg - cfl_neg * (bl_neg + br_neg))

    flux = jnp.where(c > 0, flux_pos, flux_neg)

    # Transpose back
    if axis == 2:
        flux = jnp.swapaxes(flux, 1, 2)

    return flux


def _bgrid_ke_transport(u_d, v_d, uc, vc, cdgrid, dt):
    """FV3 d_sw3: B-grid KE transport at D-grid corners.

    Computes KE at corners via operator-split 1D PPM transport of
    D-grid winds using B-grid contravariant Courant numbers derived
    from C-grid covariant velocities.  Matches sw_core.F90:1201-1388
    (duogrid/bounded_domain branch).

    Parameters
    ----------
    u_d : (6, n, n+1) D-grid x-velocity
    v_d : (6, n+1, n) D-grid y-velocity
    uc : (6, n+1, n) C-grid covariant u (updated by c_sw + p_grad_c)
    vc : (6, n, n+1) C-grid covariant v
    cdgrid : CubedSphereCDGrid
    dt : float

    Returns
    -------
    ke_corner : (6, n+1, n+1) kinetic energy at D-grid corners
    """
    n = cdgrid.n
    dt5 = 0.5 * dt
    cosa = cdgrid.cosa_corner    # (6, n+1, n+1)
    rsina = cdgrid.rsin2_corner  # (6, n+1, n+1) = 1/sin²

    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2

    # --- Step 1: B-grid contravariant v-velocity (Courant number) ---
    # vb(i,j) = dt/2 * (vc(i-1,j) + vc(i,j) - (uc(i,j-1) + uc(i,j)) * cosa) * rsina
    # vc: (6, n, n+1) → need vc(i-1,j) and vc(i,j) at corner (i,j)
    # Corner i ranges 0..n; vc row ranges 0..n-1.  Need vc at i=-1 (halo).
    # uc: (6, n+1, n) → need uc(i,j-1) and uc(i,j) at corner (i,j)
    # Corner j ranges 0..n; uc col ranges 0..n-1.  Need uc at j=-1 (halo).
    #
    # Iter-946 (Fortran-fidelity fix): on the duogrid path with ng>=3,
    # source the vc/uc halo cells from the OLD u_d, v_d via
    # `_pad_halo_uc_vc_via_d2a2c` (4th-order d2a2c machinery) instead
    # of `mode='edge'`.  HALO ROWS only are replaced; the interior of
    # vc, uc is preserved exactly so the corner Courant numbers at
    # interior corners are bit-identical to pre-iter-946.  Caveat
    # (same as in `_d_sw1_recompute_ut_vt`): derived halo lacks the
    # c_sw + p_grad_c increment.
    # iter-947 (Fortran-fidelity fix, post-iter-946 negative result):
    # use `_pad_halo_uc_vc_new_via_old_delta` to source NEW-corrected
    # cross-face halo for vc/uc.  The helper combines the iter-946
    # d2a2c-derived OLD halo with the NEW interior boundary cell to
    # estimate NEW halo = NEW_boundary + (OLD_halo - OLD_boundary).
    # This fixes the iter-946 OLD/NEW discontinuity by anchoring the
    # halo on the NEW boundary value while still carrying the OLD
    # cross-face geometric delta.  Falls back to `mode='edge'` for
    # non-duogrid or duogrid with ng<3 (same gate as iter-946).
    if use_duogrid and dg.ng >= 3:
        uc_pad, vc_pad = _pad_halo_uc_vc_new_via_old_delta(
            uc, vc, u_d, v_d, cdgrid)
    else:
        vc_pad = jnp.pad(vc, [(0, 0), (1, 1), (0, 0)], mode='edge')
        uc_pad = jnp.pad(uc, [(0, 0), (0, 0), (1, 1)], mode='edge')
    vc_sum = vc_pad[:, :-1, :] + vc_pad[:, 1:, :]  # (6, n+1, n+1)
    uc_sum = uc_pad[:, :, :-1] + uc_pad[:, :, 1:]  # (6, n+1, n+1)

    vb = dt5 * (vc_sum - uc_sum * cosa) * rsina  # (6, n+1, n+1)

    # Iter-945 (Fortran-fidelity fix): on the duogrid path, populate
    # cross-face halo cells of (u_d, v_d) along the PPM sweep axis via
    # `_pad_halo_dgrid_for_ppm` (`ext_vector_dgrid` pipeline — same
    # mechanism as `_d2a2c_vect_duogrid`'s halo).  PPM hord=9 then
    # reads cross-face data at the four cube-face boundaries instead of
    # `mode='edge'` same-face replicas.  Fortran sources this halo from
    # `mpp_update_domains(u_d, v_d, gridtype=DGRID_NE)` upstream of
    # d_sw3.  Non-duogrid path is unchanged: PPM falls back to
    # `mode='edge'` internally (external_halo=0).
    # Iter-950 (NEGATIVE-RESULT): extending the iter-945 D-grid PPM
    # halo from h_dg=2 to h_dg=3 (when duogrid ng>=3) reduced the
    # mode='edge' outer-halo gap from 2 cells to 1 cell.  Result on
    # duogrid C36 W2 1-day:
    #     iter-947 baseline (h_dg=2):  |u|=76.91, |v|=75.38, v_ll=55.6
    #     iter-950 (h_dg=3):           |u|=77.76, |v|=70.64, v_ll=58.0
    # |v_max| improved 6% but v_ll_Linf REGRESSED — the deeper halo
    # spreads cube-vertex artifacts further into the panel.  Reverted
    # to h_dg=2; the v_ll_Linf metric (the actual W2 acceptance gate)
    # is the right north star.
    if use_duogrid:
        h_dg = 2
        u_d_ihalo, v_d_jhalo = _pad_halo_dgrid_for_ppm(
            u_d, v_d, cdgrid, halo=h_dg)
    else:
        h_dg = 0
        u_d_ihalo = u_d
        v_d_jhalo = v_d

    # --- Step 2: transport v_d in y-direction using vb (PPM hord=9) ---
    # FV3 sw_core.F90:1315 calls ytp_v with hord_mt=9 (default).
    # Iter-967: enable d_sw3 cube-edge boundary fix.  Fortran's
    # call passes ``bounded_domain=.false.`` HARDCODED at line 1316,
    # so the boundary fix at sw_core.F90:3239-3316 fires regardless
    # of duogrid status.
    # Iter-967 (NEGATIVE-RESULT): tested enabling Fortran's d_sw3
    # cube-edge boundary fix (sw_core.F90:3239-3316 ytp_v, similar
    # for xtp_u).  Fortran's d_sw3 hardcodes ``bounded_domain=.false.``
    # in the calls (sw_core.F90:1316/1374), so the boundary fix
    # always fires.  Adding it to our Python `_ppm_transport_1d`
    # via the new `apply_d_sw3_boundary_fix` kwarg WORSENED W2
    # v_ll_Linf 55.6 → 119.8 m/s.
    #
    # Likely reason: with iter-945's `_pad_halo_dgrid_for_ppm` we
    # already provide proper cross-face halo for u_d, v_d at depth
    # h_dg=2.  Fortran's boundary fix assumes mode='edge'-style halo
    # (no cross-face data) and applies a corrective extrapolation
    # using s11/s14/s15 coefficients.  Applying that correction ON
    # TOP of correct cross-face halo over-corrects.  The boundary
    # fix is NOT compatible with iter-945's halo strategy — they're
    # alternative paths.
    #
    # Reverted; the kwarg is preserved on `_ppm_transport_1d` for
    # potential future use (e.g., a non-duogrid path that doesn't
    # have iter-945's halo).
    rdy = 1.0 / jnp.maximum(cdgrid.dy_edge_x, _EPS)  # (6, n+1, n)
    transported_y = _ppm_transport_1d(
        v_d_jhalo, vb, rdy, axis=2, external_halo=h_dg)

    # --- Step 3: B-grid contravariant u-velocity (Courant number) ---
    ub = dt5 * (uc_sum - vc_sum * cosa) * rsina  # (6, n+1, n+1)

    # --- Step 4: transport u_d in x-direction using ub (PPM hord=9) ---
    rdx = 1.0 / jnp.maximum(cdgrid.dx_edge_y, _EPS)  # (6, n, n+1)
    transported_x = _ppm_transport_1d(
        u_d_ihalo, ub, rdx, axis=1, external_halo=h_dg)

    # --- Step 5: Fortran-faithful BGRID_NE component sync (dyn_core.F90:968-1019) ---
    # Fortran syncs ubb (x-component) and vbbtemp (y-component) via
    # `mpp_get_boundary(..., gridtype=BGRID_NE)` BEFORE computing KE so that
    # shared cube-face corners agree on the transported vector values.
    #
    # Python realises the same sync via `synchronize_bgrid_ne_corner_geo`
    # (iter-102), which routes the vector average through the geographic
    # frame — this avoids having to encode per-seam rotation tables for the
    # 8 reversed seams, 4 cross-axis non-reversed seams, and the 8 cube
    # vertices.  On duogrid-enabled grids this replaces the previous
    # scalar-KE sync (Fortran commented-out alternative, dyn_core.F90:1029-1055).
    # Name the intermediates to match Fortran convention:
    #   ubbtemp = ytp_v output = transported_y
    #   vbbtemp = y-Courant scalar = vb
    #   ubb     = x-Courant scalar = ub
    #   vbb     = xtp_u output = transported_x
    ubbtemp = transported_y
    vbbtemp = vb
    ubb = ub
    vbb = transported_x
    # iter-944b: GATE BGRID_NE corner sync on duogrid (Fortran-faithful).
    #
    # iter-941 mistakenly removed the `if use_duogrid:` gate based on the
    # erroneous belief that Fortran's `mpp_get_boundary(... gridtype=
    # BGRID_NE)` "fires regardless of duogrid".  iter-944b's review of
    # the reference source `dyn_core.F90:968-1011` shows the call is
    # explicitly inside an `if (duogrid)` block — non-duogrid Fortran
    # runs do NOT sync (ubb, vbbtemp) here, relying instead on the
    # standard halo from `mpp_update_domains(uc, vc, gridtype=CGRID_NE)`
    # at dyn_core.F90:633/689/702/1291 to keep the C-grid winds consistent
    # across face boundaries (which feeds through the (ubb, vbbtemp)
    # construction at d_sw3 to keep cube-vertex KE consistent).
    #
    # The gate restores Fortran-faithful behaviour:
    #   - duogrid runs: BGRID_NE sync fires (matches Fortran).
    #   - non-duogrid runs: NO sync (matches Fortran).  FB chain step
    #     survival on C36 W2 reverts to ~41 steps for non-duogrid;
    #     duogrid is the supported FV3 W2 path.
    #
    # Production `fv3_sw_tendencies` (FV3EdgeShallowWaterModel default)
    # does NOT call `_d_sw_native`, so production sentinels are
    # unchanged either way.  (`dg` and `use_duogrid` are already in
    # scope from the iter-945 PPM-halo gate above.)
    if use_duogrid:
        cac = cdgrid.cos_angle_corner
        sac = cdgrid.sin_angle_corner
        ubb, vbbtemp = synchronize_bgrid_ne_corner_geo(
            ubb, vbbtemp, cac, sac, n)

    # --- Step 6: KE at corners (Lin-Rood average of two sweeps) ---
    # FV3 dyn_core.F90:1013-1020:
    #   kee = 0.5*(ubbtemp*vbbtemp + ubb*vbb)
    ke_corner = 0.5 * (ubbtemp * vbbtemp + ubb * vbb)

    return ke_corner


def _d_sw_native(h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid, dt, g,
                 div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
                 damp_v=0.0, nord_v=0,
                 apply_legacy_d_sw4_corner_ke_fix=False,
                 apply_legacy_d_sw5_corner_corrections=False,
                 apply_fortran_xppm_boundary=False):
    """D-grid full-step (FV3 d_sw1..d_sw6).

    Matches the FV3 dyn_core.F90 d_sw sequence:
    - d_sw1: transport velocity recomputation + PPM mass/tracer transport
    - d_sw3: B-grid KE transport at corners
    - d_sw5: corner divergence damping added to KE + vorticity transport
    - d_sw6: D-grid wind replacement formula + vorticity damping

    Parameters
    ----------
    h : (6, n, n) — ORIGINAL height (for PPM mass transport)
    u_d : (6, n, n+1) — OLD D-grid x-velocity
    v_d : (6, n+1, n) — OLD D-grid y-velocity
    uc : (6, n+1, n) — UPDATED covariant C-grid u (from c_sw + p_grad_c)
    vc : (6, n, n+1) — UPDATED covariant C-grid v (from c_sw + p_grad_c)
    ua, va : (6, n, n) — A-grid contravariant (from c_sw's d2a2c_vect)
    cdgrid : CubedSphereCDGrid
    dt : float — full time step
    g : float
    div_damp : float — LEGACY, UNUSED in this function.  The d_sw5
        corner divergence damping at step (5) uses `d2_bg / dddmp /
        d4_bg / nord` exclusively.  `div_damp` is neither read nor
        validated in the body of `_d_sw_native`; it is kept in the
        signature only for backward compatibility with the FB
        chain callers (`fv3_forward_backward_step`, `fv3_fb_sw_step`)
        that forward it.  NOTE: three SEPARATE paths DO use
        `div_damp` and share `CDGridShallowWaterConfig.div_damp`:
        `cdgrid_momentum_tendencies` (in operators_cdgrid),
        `fv3_sw_tendencies` (in operators_cdgrid — the A-L RK3
        PRODUCTION path), and `fv3_csw_tendencies` (in this module —
        experimental CSW).  The config field therefore remains live
        across three paths; only the FB chain forwards-and-ignores
        it.  Future consolidation should drop `div_damp` from the FB
        entry points only, not from the config or the three live-
        use paths.  (Function-name refs only; in-code line numbers
        were removed in iter-180 because they shifted on every
        edit.)
    d2_bg : float — FV3 d_sw5 background del-2 coefficient (default 0.0)
    dddmp : float — FV3 d_sw5 adaptive Smagorinsky coefficient (default 0.0)
    d4_bg : float — FV3 d_sw5 background del-4+ coefficient (default 0.16)
    nord : int — damping order: 0=del-2, 1=del-4, 2=del-6 (default 1)
    damp_v : float — vorticity damping coefficient (FV3 vtdm4, default 0.0 = off)
    nord_v : int — vorticity damping order (default 0 = del-2)

    Returns
    -------
    h_new, u_d_new, v_d_new
    """
    n = cdgrid.n

    # === 1. Contravariant transport velocity from updated C-grid ===
    # Use FV3 d_sw1 boundary handling (adjacent strips + corner 2×2 solve)
    # for cross-face consistent transport at panel boundaries.
    # Iter-947: forward OLD u_d, v_d so the duogrid path can compute
    # NEW-corrected cross-face vc/uc halo via
    # `_pad_halo_uc_vc_new_via_old_delta`.
    ut, vt = _d_sw1_recompute_ut_vt(
        uc, vc, cdgrid, dt, u_d_old=u_d, v_d_old=v_d)
    # iter-944b: REVERTED iter-944's CGRID_NE sync of (ut, vt) here.
    # Fortran reference dyn_core.F90:855-900 syncs only the MASS flux
    # (`fxx_delp/fyy_delp`) via `mpp_get_boundary(... gridtype=CGRID_NE)`,
    # gated on `if (duogrid)`, AFTER the d_sw2 transport call.  The
    # transport velocities ut, vt themselves are NOT separately synced —
    # they are inputs to d_sw2/d_sw3 and Fortran relies on the standard
    # halo from `mpp_update_domains(uc, vc, gridtype=CGRID_NE)` at
    # dyn_core.F90:633/689/702/1291 to keep them consistent across face
    # boundaries.  iter-944's standalone sync of (ut, vt) was Python-
    # only smoothing not present in Fortran; removed per iter-944b's
    # Fortran-fidelity audit.

    # === 2. PPM mass transport using ORIGINAL h ===
    # Fortran d_sw1 (sw_core.F90:886-887) UNCONDITIONALLY passes
    # ``nord=nord_v, damp_c=damp_v`` into the delp transport.  The
    # actual damping gate lives inside ``fv_tp_2d`` at
    # tp_core.F90:217-219 (``if ( damp_c > 1.E-4 ) then``), which
    # our Python ``fv_tp_2d`` mirrors at
    # ``src/legoesm/core/fv_tp_2d.py:528``.  Iter-727 first-pass had
    # an outer Python guard ``if damp_v > 1e-5`` here to bypass the
    # damped branch for damp_v=0 — that threshold matched the
    # step-(9) ``_del6_vt_flux`` branch (sw_core.F90:1948-1950) but
    # NOT the d_sw1 mass branch which has no outer threshold in
    # Fortran.  Codex adversarial review flagged this as a non-
    # faithful accretion.  Iter-728 removes the guard: always
    # forward ``nord=nord_v, damp_c=damp_v`` and let the internal
    # tp_core.F90:217 gate handle damp_v=0.  Fortran-exact; no
    # behavioural change for any damp_v because both the Python and
    # the Fortran internal gates are ``> 1e-4`` (so 0 <= damp_v
    # <= 1e-4 is a no-op both ways, and damp_v > 1e-4 invokes the
    # del-n smoother both ways).
    #
    # NOTE on the Fortran param naming (dyn_core.F90:762-770):
    #   damp_t = damp_v = damp_vt(k) = flagstruct%vtdm4
    #   nord_t = nord_v(k)
    # So mass damping (sw_core.F90:886-887) and vorticity damping
    # (sw_core.F90:1948) share one coefficient pair.  ``q_con``
    # transport at sw_core.F90:942-943 uses ``damp_t`` / ``nord_t``
    # — same numeric value per the dyn_core assignment, but the
    # code-path names are distinct.  We forward damp_v/nord_v here
    # because the delp call at sw_core.F90:886-887 names them
    # literally.
    # Iter-888b (Codex iter-888 stop-time fix): forward
    # `apply_fortran_xppm_boundary` so the FB-chain mass-transport call
    # is the entry point for the Fortran s11/s14/s15 boundary formula
    # (tp_core.F90:614-628, :632-647).  Default False — pre-iter-888b
    # the kwarg was a leaf-level addition unreachable from any caller.
    h_new = transport_step(h, ut, vt, dt, cdgrid,
                           nord=nord_v, damp_c=damp_v,
                           apply_fortran_xppm_boundary=(
                               apply_fortran_xppm_boundary))

    # === 3. Cell-centre vorticity from D-grid circulation ===
    dx_u = cdgrid.dx_edge_y  # (6, n, n+1) — edge length for u_d
    dy_v = cdgrid.dy_edge_x  # (6, n+1, n) — edge length for v_d

    vt_circ = u_d * dx_u  # (6, n, n+1) — u circulation
    ut_circ = v_d * dy_v  # (6, n+1, n) — v circulation

    rarea = 1.0 / cdgrid.base.area  # (6, n, n)
    # CCW circulation: bottom - top + right - left
    zeta = rarea * (vt_circ[:, :, :-1] - vt_circ[:, :, 1:]
                    + ut_circ[:, 1:, :] - ut_circ[:, :-1, :])
    zeta_abs = zeta + cdgrid.base.f  # (6, n, n)

    # === 4. B-grid KE transport (FV3 d_sw3, sw_core.F90:1201-1388) ===
    ke_corner = _bgrid_ke_transport(u_d, v_d, uc, vc, cdgrid, dt)

    # Iter-869: optional Fortran d_sw4 cube-vertex KE fix
    # (sw_core.F90:1442-1465).  Default-off opt-in matching iter-862's
    # d_sw5 corner-corrections pattern.  When enabled AND in the
    # legacy non-bounded-domain global cubed sphere, overrides ke at
    # the four cube-vertex corners with Fortran's `dt/6 * (...)`
    # formula combining ut, vt, u_d, v_d.  Halo-input gap (mode='edge'
    # same-face extension) carried forward; cross-face halo deferred
    # to iter-870+.  Production (`fv3_sw_tendencies`) does NOT invoke
    # this branch — _d_sw_native is FB-chain only.
    if apply_legacy_d_sw4_corner_ke_fix:
        ke_corner = _apply_legacy_d_sw4_corner_ke_fix(
            ke_corner, ut, vt, u_d, v_d, dt,
            bounded_domain=cdgrid.base.bounded_domain)

    # === 5. Corner divergence damping added to KE (FV3 d_sw5) ===
    # Fortran d_sw5 (sw_core.F90:1641-1821) computes divergence at D-grid
    # corners and adds damp*delpc to ke BEFORE the wind update.  This is
    # the standard FV3 path with nord=1, d4_bg=0.16 as defaults.
    use_d_sw5_damping = (d2_bg > 1e-10 or dddmp > 1e-10 or d4_bg > 1e-10)
    if use_d_sw5_damping:
        # Iter-871b: forward iter-862's `apply_legacy_corner_corrections`
        # opt-in flag through the FB-chain wrapper so a caller of
        # `_d_sw_native` can opt into Check 3 corner corrections.
        # iter-862 originally added the flag only to
        # `_d_sw5_corner_divergence`; the wrapper-level wiring was
        # missing, making the flag unreachable from the FB chain.
        # Codex iter-871 stop-time review caught this asymmetry vs
        # iter-869b's wrapper-plumbed flag.  Default-off preserved.
        ke_damping = _d_sw5_corner_divergence(
            u_d, v_d, ua, va, cdgrid, dt,
            d2_bg=d2_bg, dddmp=dddmp, d4_bg=d4_bg, nord=nord,
            apply_legacy_corner_corrections=(
                apply_legacy_d_sw5_corner_corrections))
        ke_corner = ke_corner + ke_damping

    # iter-944b: REVERTED iter-942's `synchronize_corner_scalar` on
    # `ke_corner`.  Fortran reference dyn_core.F90:1029-1055 AND
    # :1180-1207 contain the corresponding ke_corner corner sync but
    # it is COMMENTED OUT (between `! if(duogrid)` and `!endif`), with
    # adjacent commentary noting it "seems to reduce noise a lot for
    # few timesteps only".  Even within the duogrid branch the sync is
    # disabled in the reference source.  iter-942's unconditional sync
    # was Python-only smoothing not present in Fortran; removed per
    # iter-944b's Fortran-fidelity audit.

    # === 6. KE gradient at D-grid edge positions ===
    # FV3 d_sw6 (sw_core.F90:1935-1944):
    #   u(i,j) = vt(i,j) + ke(i,j) - ke(i+1,j) + fy(i,j)
    #   v(i,j) = ut(i,j) + ke(i,j) - ke(i,j+1) - fx(i,j)
    ke_diff_u_scaled = ke_corner[:, :-1, :] - ke_corner[:, 1:, :]  # (6, n, n+1)
    ke_diff_v_scaled = ke_corner[:, :, :-1] - ke_corner[:, :, 1:]  # (6, n+1, n)

    # === 7. Vorticity transport to D-grid edges (FV3 d_sw5 fv_tp_2d) ===
    # Iter-864: pass `apply_cgrid_flux_sync=False` so this call matches
    # Fortran's commented-out vorticity-flux averaging block in
    # dyn_core.F90:1124-1207.  Fortran computes vortfluxx/vortfluxy at
    # sw_core.F90:1861 inside d_sw5 and then EXPLICITLY DOES NOT
    # `mpp_get_boundary`-sync them before d_sw6 (the sync block exists
    # in source but is commented out).  Our `fv_tp_2d` previously
    # always applied the iter-808 CGRID_NE sync when duogrid was on,
    # which silently over-synced the vorticity flux relative to
    # Fortran.  Following the Fortran oracle exactly: skip the sync
    # here.  The mass-flux call inside `transport_step` (step 2 above)
    # still defaults to sync=True, matching Fortran's ACTIVE
    # `mpp_get_boundary` averaging block at dyn_core.F90:850-900.
    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
        ut, vt, dt, cdgrid)
    fx_vort, fy_vort = fv_tp_2d(
        zeta_abs, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
        apply_cgrid_flux_sync=False,
        apply_fortran_xppm_boundary=apply_fortran_xppm_boundary)
    # iter-944b: REVERTED iter-944's explicit CGRID_NE sync of
    # (fx_vort, fy_vort).  Fortran reference dyn_core.F90:1124-1207
    # has the corresponding vorticity-flux sync entirely COMMENTED OUT
    # (between `! ... ! endif`), and the surrounding comment block reads
    # "Revisit the vorticity flux averaging / should be applied to have
    # a consistent logic".  Even within the `if (duogrid)` branch the
    # sync is disabled in the reference source.  iter-944's force-sync
    # was Python-only smoothing not present in Fortran; removed per
    # iter-944b's Fortran-fidelity audit.

    # === 8. D-grid wind update (FV3 d_sw6, sw_core.F90:1935-1944) ===
    # Incremental form equivalent to Fortran replacement formula:
    #   u_new*dx = u_old*dx + ke_diff + fy_vort
    rdx_u = 1.0 / jnp.maximum(dx_u, _EPS)  # (6, n, n+1)
    rdy_v = 1.0 / jnp.maximum(dy_v, _EPS)  # (6, n+1, n)

    u_d_new = u_d + (ke_diff_u_scaled + fy_vort) * rdx_u
    v_d_new = v_d + (ke_diff_v_scaled - fx_vort) * rdy_v

    # === 9. Vorticity damping (FV3 d_sw6, sw_core.F90:1948-2000) ===
    # Fortran: if (damp_v > 1e-5) then
    #   damp4 = (damp_v * da_min_c)**(nord_v+1)
    #   call del6_vt_flux(nord_v, ..., damp4, wk, ...)
    #   u = u + vt   (vt = fy2 from del6_vt_flux)
    #   v = v - ut   (ut = fx2 from del6_vt_flux)
    if damp_v > 1e-5:
        da_min_c = jnp.min(cdgrid.area_corner)
        damp4 = (damp_v * da_min_c) ** (nord_v + 1)
        dg = cdgrid.base.duogrid
        _use_dg = dg is not None and dg.ng >= 2
        fx2, fy2 = _del6_vt_flux(nord_v, damp4, zeta, cdgrid,
                                  use_duogrid=_use_dg)
        u_d_new = u_d_new + fy2 * rdx_u
        v_d_new = v_d_new - fx2 * rdy_v

    return h_new, u_d_new, v_d_new


def fv3_fb_sw_step(h, u_d, v_d, h_s, cdgrid, dt, g=constants.g,
                   div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
                   damp_v=0.0, nord_v=0,
                   apply_legacy_d_sw4_corner_ke_fix=False,
                   apply_legacy_d_sw5_corner_corrections=False,
                   apply_fortran_xppm_boundary=False):
    """EXPERIMENTAL: Complete FV3 forward-backward shallow water time step.

    Known unstable at C16 (halo quality limitation).
    Use ``fv3_sw_tendencies`` with RK3 for production work.

    Three phases matching FV3 dyn_core.F90:
    1. c_sw (forward, dt/2): d2a2c_vect + mass transport + KE/vorticity
    2. p_grad_c (backward, dt/2): pressure gradient at C-grid
    3. d_sw (full dt): d_sw1-d_sw6 chain (PPM transport + B-grid KE +
       corner divergence damping + vorticity transport + vorticity damping
       + wind update)

    Parameters
    ----------
    h, u_d, v_d, h_s : state arrays
    cdgrid : CubedSphereCDGrid
    dt, g : float
    div_damp : float — LEGACY, UNUSED in the FB chain.  Forwarded to
        `_d_sw_native` which ignores it; see `_d_sw_native` docstring
        for the split (this FB chain unused; three separate paths —
        `cdgrid_momentum_tendencies`, `fv3_sw_tendencies` A-L RK3
        production, `fv3_csw_tendencies` experimental CSW — all use
        `div_damp` and share `CDGridShallowWaterConfig.div_damp`).
        The FB chain's divergence damping comes from `d2_bg/dddmp/
        d4_bg/nord` (d_sw5 coefficients).
    d2_bg : float — FV3 background del-2 coefficient (default 0.0)
    dddmp : float — FV3 adaptive Smagorinsky coefficient (default 0.0)
    d4_bg : float — FV3 background del-4+ coefficient (default 0.16)
    nord : int — damping order (default 1 = del-4)
    damp_v : float — vorticity damping coefficient (FV3 vtdm4, default 0.0 = off)
    nord_v : int — vorticity damping order (default 0 = del-2)
    """
    dt2 = 0.5 * dt

    # Phase 1: c_sw — forward half-step at C-grid
    h_star, uc_new, vc_new, ua, va = _c_sw(
        h, u_d, v_d, h_s, cdgrid, dt, g)

    # Phase 2: p_grad_c — backward pressure gradient at C-grid
    dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt2, g)
    uc_new = uc_new + dp_x
    vc_new = vc_new + dp_y

    # Phase 3: d_sw — full-step D-grid update
    # Iter-871c: forward iter-869b/iter-871b opt-in flags through this
    # FB-chain entry point so callers of `fv3_fb_sw_step` (the
    # canonical FB step function used by `FV3FBShallowWaterModel`)
    # can opt into the legacy corner fixes.  Default-off preserves
    # bit-identical behaviour for existing callers.
    h_new, u_d_new, v_d_new = _d_sw_native(
        h, u_d, v_d, h_s, uc_new, vc_new, ua, va, cdgrid, dt, g,
        div_damp=div_damp, d2_bg=d2_bg, dddmp=dddmp, d4_bg=d4_bg, nord=nord,
        damp_v=damp_v, nord_v=nord_v,
        apply_legacy_d_sw4_corner_ke_fix=apply_legacy_d_sw4_corner_ke_fix,
        apply_legacy_d_sw5_corner_corrections=(
            apply_legacy_d_sw5_corner_corrections),
        apply_fortran_xppm_boundary=apply_fortran_xppm_boundary)

    return h_new, u_d_new, v_d_new
