"""FV3-faithful 2D finite-volume transport on the cubed sphere.

Implements Putman & Lin (2007) Lin-Rood transport with monotone PPM
(hord=8) and Courant-number flux integration.  Face-boundary edges
use position-aware weighted averages derived from the known halo
interpolation offsets, replacing the equal-spacing assumption.

References
----------
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
- GFDL tp_core.F90
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo

_R3 = 1.0 / 3.0


def _pert_ppm(bl, br):
    """FV3 pert_ppm iv=1: standard PPM constraint (tp_core.F90:1193-1212).

    Prevents new extrema in the reconstruction.  When bl and br have
    opposite signs (parabola crosses cell value), clips the overshoot.
    When both have the same sign (cell value is already an extremum),
    zeros both to flatten the reconstruction.
    """
    is_ext = bl * br >= 0.0
    da1 = bl - br
    da2 = da1 ** 2
    a6da = 3.0 * (bl + br) * da1
    # Fortran: if a6da < -da2: ar = -2*al → br = -2*bl
    #          if a6da >  da2: al = -2*ar → bl = -2*br
    br_out = jnp.where(a6da < -da2, -2.0 * bl, br)
    bl_out = jnp.where(a6da > da2, -2.0 * br, bl)
    bl_out = jnp.where(is_ext, 0.0, bl_out)
    br_out = jnp.where(is_ext, 0.0, br_out)
    return bl_out, br_out


def _pert_ppm_iv0(q, bl, br):
    """FV3 pert_ppm iv=0: positive definite constraint (tp_core.F90:1169-1192).

    Ensures the PPM parabola does not produce negative values when the
    cell mean ``q`` is positive.  When ``q <= 0``, zeroes the reconstruction.
    When ``q > 0`` and the parabola minimum is negative, clips bl/br.

    This is the limiter used by hord=9 (FV3 default for mass, vorticity,
    and momentum transport).
    """
    r12 = 1.0 / 12.0
    zero = jnp.zeros_like(bl)

    # Parabola coefficients: a4 = -3*(bl+br), da1 = br-bl
    a4 = -3.0 * (br + bl)
    da1 = br - bl

    # Condition: parabola has an extremum in [0,1] ↔ |da1| < -a4
    has_extremum = jnp.abs(da1) < -a4

    # Minimum of parabola: q + 0.25/a4 * da1² + a4/12
    # Guard against a4=0 (flat parabola — no extremum anyway)
    a4_safe = jnp.where(jnp.abs(a4) < 1e-30, -1e-30, a4)
    fmin = q + 0.25 / a4_safe * da1 ** 2 + a4_safe * r12
    is_negative = fmin < 0.0

    # When both conditions met and q>0: apply fix
    needs_fix = has_extremum & is_negative & (q > 0.0)
    both_positive = (br > 0.0) & (bl > 0.0)
    da1_positive = da1 > 0.0

    bl_fix = jnp.where(both_positive, zero,
                       jnp.where(da1_positive, bl, -2.0 * br))
    br_fix = jnp.where(both_positive, zero,
                       jnp.where(da1_positive, -2.0 * bl, br))

    bl_out = jnp.where(q <= 0.0, zero, jnp.where(needs_fix, bl_fix, bl))
    br_out = jnp.where(q <= 0.0, zero, jnp.where(needs_fix, br_fix, br))

    return bl_out, br_out


def _ppm_1d(q, n, off_left=None, off_right=None,
            off_left_d1=None, off_right_d1=None,
            use_duogrid=False,
            apply_fortran_xppm_boundary=False,
            bounded_domain=False):
    """PPM bl/br along axis=1 with hord=9 + position-aware boundaries.

    Parameters
    ----------
    q : (6, n+4, M)  field with halo=2 in sweep direction
    n : int           number of interior cells
    off_left : (6, M) or None — halo interp offset at left boundary (depth=0)
    off_right : (6, M) or None — halo interp offset at right boundary (depth=0)
    off_left_d1 : (6, M) or None — depth=1 offset (outer halo)
    off_right_d1 : (6, M) or None — depth=1 offset (outer halo)
    use_duogrid : bool
        If True:
          * skip the ``pert_ppm(iv=1)`` face-boundary monotonicity
            constraint (matches Fortran ``.not. (bounded_domain .or.
            duogrid)`` gate at tp_core.F90:612); and
          * skip the offset-based dm rescaling and position-aware ``al``
            edge corrections — when the duogrid kinked-to-extended remap
            is in effect, halo cells already sit at the correct physical
            positions so the standard uniform-spacing PPM formula is
            Fortran-faithful.  Fortran keeps the standard dm formula at
            all cells and instead handles any residual non-uniformity via
            explicit ``bl/br`` rewrites (which we also skip for duogrid,
            matching the Fortran gate).
    bounded_domain : bool, default False
        Iter-890 (Codex iter-889b stop-time follow-up): match Fortran's
        full gate ``.not. (bounded_domain .or. duogrid)`` at
        `tp_core.F90:612` and `:333/357`.  Pre-iter-890 ``_ppm_1d``
        gated only on ``use_duogrid`` (duogrid-only); regional/nested
        bounded-domain panels (where `bounded_domain = (regional .or.
        nested .or. duogrid)` per `fv_arrays.F90:1512`) would
        incorrectly take the legacy global-face boundary overrides
        and the iv=1 limiter.  iter-890 closes that gap by adding the
        explicit `bounded_domain` kwarg; every existing
        ``not use_duogrid`` gate inside this function becomes
        ``not (use_duogrid or bounded_domain)``.  Default False
        preserves prior global-cubed-sphere behaviour bit-for-bit.
    apply_fortran_xppm_boundary : bool, default False
        Iter-888: when True AND ``not use_duogrid``, overwrite the bl/br
        values at the 6 face-boundary cells (indices 0, 1, 2 and -3, -2,
        -1) with Fortran's s11/s14/s15 + 4-point boundary formulas from
        ``tp_core.F90:614-628`` (left) and ``:632-647`` (right).
        Constants: ``s11 = 11/14, s14 = 4/7, s15 = 3/14`` (tp_core.F90:58).
        The 4-point xt formula uses the UNIFORM-GRID simplification
        ``xt = 0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))``, which collapses
        the Fortran ``dxa``-weighted formula at lines 616-617/638-639
        when cell widths are uniform.  Cubed-sphere boundary cells have
        non-uniform ``dxa`` near corners so this is a partial Fortran-
        fidelity fix; the full ``dxa``-weighted formula is deferred until
        ``dxa`` plumbing through ``_xppm/_yppm/fv_tp_2d`` is added.  The
        overrides apply BEFORE ``pert_ppm(iv=1)`` so the limiter sees
        Fortran-faithful boundary bl/br (matches Fortran's lines 614-629
        + 632-648 ordering).  Default False preserves prior behaviour.

    Returns
    -------
    bl, br   : (6, n+2, M)
    q_cells  : (6, n+2, M)
    """
    qe = jnp.pad(q, [(0, 0), (1, 1), (0, 0)], mode='edge')  # (6, n+6, M)

    # Iter-890: Fortran's full gate at tp_core.F90:333/357/612 is
    # `.not. (bounded_domain .or. duogrid) .and. grid_type<3`.  In our
    # convention `bounded_domain = (regional .or. nested .or. duogrid)`
    # so `bounded_domain` already covers the duogrid case, but we keep
    # `use_duogrid` as a separate parameter for backward compatibility
    # with existing callers and to distinguish "duogrid kinked-to-
    # extended remap is active" from "regional/nested panel BC is
    # active".  The gates below all check that NEITHER flag is set
    # (i.e., the legacy global-cubed-sphere face is in play).
    fortran_legacy_face = (not use_duogrid) and (not bounded_domain)

    # Monotone slopes
    xt = 0.25 * (qe[:, 2:, :] - qe[:, :-2, :])
    qm = qe[:, 1:-1, :]
    q_hi = jnp.maximum(jnp.maximum(qe[:, :-2, :], qm), qe[:, 2:, :])
    q_lo = jnp.minimum(jnp.minimum(qe[:, :-2, :], qm), qe[:, 2:, :])
    dm = jnp.sign(xt) * jnp.minimum(
        jnp.abs(xt), jnp.minimum(q_hi - qm, qm - q_lo))
    # dm[:, k, :] = slope at qe cell (k+1)

    # Correct dm at boundary cells for non-uniform halo spacing.
    # Standard dm = 0.25*(q[i+1]-q[i-1]) assumes span=2.
    # Actual span at boundary cells differs by halo offsets.
    def _correct_dm(dm_arr, idx, q_hi_arr, q_lo_arr, qm_arr, scale):
        """Correct dm at index idx by scale factor, re-apply monotone limit."""
        dm_scaled = dm_arr[:, idx, :] * scale
        pmp = q_hi_arr[:, idx, :] - qm_arr[:, idx, :]
        pmm = qm_arr[:, idx, :] - q_lo_arr[:, idx, :]
        dm_lim = jnp.sign(dm_scaled) * jnp.minimum(
            jnp.abs(dm_scaled), jnp.minimum(pmp, pmm))
        return dm_arr.at[:, idx, :].set(dm_lim)

    # Only apply dm-rescaling for non-uniform halo spacing when the halo is
    # delivered via interp_offsets (non-duogrid).  In duogrid mode the
    # kinked-to-extended remap already places halo cells at their correct
    # positions, so the standard monotone dm is Fortran-faithful
    # (tp_core.F90:539-545 uses a single uniform-spacing formula for all
    # cells and relies on uniform halo spacing from MPI).
    if fortran_legacy_face and off_left is not None:
        # dm at halo cell -1 (dm index 1): spans from halo(-2) to interior(0)
        if off_left_d1 is not None:
            span_halo = 2.0 - off_left_d1 + off_left
            dm = _correct_dm(dm, 1, q_hi, q_lo, qm,
                             2.0 / jnp.maximum(span_halo, 0.5))
        # dm at first interior cell 0 (dm index 2): spans from halo(-1) to interior(1)
        # halo(-1) is at position (-1+off0), interior(1) at position 1 → span = 2-off0
        span_int0 = 2.0 - off_left
        dm = _correct_dm(dm, 2, q_hi, q_lo, qm,
                         2.0 / jnp.maximum(span_int0, 0.5))

    if fortran_legacy_face and off_right is not None:
        # dm at halo cell n (dm index n+2): spans from interior(n-1) to halo(n+1)
        if off_right_d1 is not None:
            span_halo_r = 2.0 + off_right - off_right_d1
            dm = _correct_dm(dm, n + 2, q_hi, q_lo, qm,
                             2.0 / jnp.maximum(span_halo_r, 0.5))
        # dm at last interior cell n-1 (dm index n+1): spans from interior(n-2) to halo(n)
        # halo(n) at position (n+off0), interior(n-2) at n-2 → span = 2+off0
        span_int_nm1 = 2.0 + off_right
        dm = _correct_dm(dm, n + 1, q_hi, q_lo, qm,
                         2.0 / jnp.maximum(span_int_nm1, 0.5))

    # Edge values al (dm-corrected)
    al = (0.5 * (qe[:, 1:-2, :] + qe[:, 2:-1, :])
          + _R3 * (dm[:, :-1, :] - dm[:, 1:, :]))

    # Position-aware correction at face boundary EDGES:
    # The halo cell is at position (-1 + offset), not -1.
    # The face boundary edge is at position -0.5.
    # Correct edge value using actual distances to the boundary.
    # Skipped for duogrid (kinked-extended remap already aligns halo cells).
    #
    # Iter-887 (Fortran-fidelity gap, documented but not yet fixed).
    # Fortran tp_core.F90:614-628 (left) and 632-647 (right) implements
    # a richer boundary procedure when ``.not. (bounded_domain .or.
    # duogrid) .and. grid_type<3``:
    #   1. Sets bl(0)/br(npx) via ``s14*dm(-1) + s11*(q1(-1)-q1(0))``
    #      using constants ``s11 = 11/14, s14 = 4/7, s15 = 3/14``
    #      (tp_core.F90:58).
    #   2. Computes a 4-point dxa-weighted boundary edge value
    #      ``xt = 0.5 * (left_avg + right_avg)`` where each *_avg is a
    #      ``((2*dxa(i)+dxa(i±1))*q1(i) - dxa(i)*q1(i±1))/(dxa(i±1)+
    #      dxa(i))`` weighted ratio (tp_core.F90:616-617, 638-639).
    #      For uniform grid this simplifies to
    #      ``0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))``, which is a
    #      4-point cubic-style stencil — different from our 2-point
    #      ``(0.5*q_hm1 + h_L*q_i0)/(h_L+0.5)`` average below.
    #   3. Clips xt to within ``min/max(q1(-1..2))`` (tp_core.F90:619-
    #      620, 641-642).
    #   4. Then sets br(0)/bl(1) and br(1)/bl(2) (left side) or
    #      br(npx-2)/bl(npx-1) and br(npx-1)/bl(npx) (right side)
    #      from ``xt - q1(...)``.
    #
    # Our Python below implements only a 2-point position-aware
    # average for items 2-4 above and SKIPS items 1 and the explicit
    # bl(0..2)/br(0..2) overrides.  This is a Fortran-fidelity gap on
    # the legacy non-duogrid path only — duogrid runs are unaffected
    # because Fortran's ``.not. (bounded_domain .or. duogrid)`` gate
    # bypasses the entire block.  Production CDGrid uses
    # ``_ppm_reconstruct_1d`` (operators_cdgrid.py) which has its own
    # boundary handling, so this gap currently only affects FB-chain
    # ``fv_tp_2d`` consumers.  Implementing the full s11/s14/s15
    # formula requires plumbing dxa through the call site and is
    # deferred to a future iter alongside FB-chain stabilisation.
    if fortran_legacy_face and off_left is not None:
        # Left face-boundary edge: al[:, 1, :] between halo(-1) and interior(0)
        q_hm1 = qe[:, 2, :]   # halo -1 at position (-1 + off0)
        q_i0 = qe[:, 3, :]    # interior 0 at position 0
        off0 = off_left
        h_L = jnp.maximum(0.5 - off0, 0.01)
        al_L0 = (0.5 * q_hm1 + h_L * q_i0) / (h_L + 0.5)
        al_L0 = jnp.clip(al_L0, jnp.minimum(q_hm1, q_i0),
                          jnp.maximum(q_hm1, q_i0))
        al = al.at[:, 1, :].set(al_L0)

        # Second edge: al[:, 0, :] between halo(-2) and halo(-1)
        if off_left_d1 is not None:
            q_hm2 = qe[:, 1, :]   # halo -2 at position (-2 + off1)
            off1 = off_left_d1
            h_outer = jnp.maximum(0.5 - off1, 0.01)  # dist from halo-2 to midpoint
            h_inner = jnp.maximum(0.5 + off0, 0.01)   # dist from midpoint to halo-1
            al_L1 = (h_inner * q_hm2 + h_outer * q_hm1) / (h_outer + h_inner)
            al_L1 = jnp.clip(al_L1, jnp.minimum(q_hm2, q_hm1),
                              jnp.maximum(q_hm2, q_hm1))
            al = al.at[:, 0, :].set(al_L1)

    if fortran_legacy_face and off_right is not None:
        # Right face-boundary edge: al[:, n+1, :] between interior(n-1) and halo(n)
        q_inm1 = qe[:, n + 2, :]  # interior n-1
        q_hn = qe[:, n + 3, :]    # halo n at position (n + off0)
        off0r = off_right
        h_R = jnp.maximum(0.5 + off0r, 0.01)
        al_R0 = (h_R * q_inm1 + 0.5 * q_hn) / (0.5 + h_R)
        al_R0 = jnp.clip(al_R0, jnp.minimum(q_inm1, q_hn),
                          jnp.maximum(q_inm1, q_hn))
        al = al.at[:, n + 1, :].set(al_R0)

        # Second edge: al[:, n+2, :] between halo(n) and halo(n+1)
        if off_right_d1 is not None:
            q_hnp1 = qe[:, n + 4, :]
            off1r = off_right_d1
            h_inner = jnp.maximum(0.5 - off0r, 0.01)
            h_outer = jnp.maximum(0.5 + off1r, 0.01)
            al_R1 = (h_outer * q_hn + h_inner * q_hnp1) / (h_inner + h_outer)
            al_R1 = jnp.clip(al_R1, jnp.minimum(q_hn, q_hnp1),
                              jnp.maximum(q_hn, q_hnp1))
            al = al.at[:, n + 2, :].set(al_R1)

    # hord=9: FV3 default (fv_arrays.F90:339,343: hord_dp=9, hord_vt=9).
    # Simple PPM reconstruction (bl = al - q, br = al - q) with positive-
    # definite constraint via pert_ppm(iv=0).  This is LESS restrictive
    # than hord=8 (2*dm monotone) or hord=10 (pmp/lac), preserving more
    # sub-grid structure.  FV3 tp_core.F90 xppm lines 603-610.
    q_c = qe[:, 2:-2, :]       # (6, n+2, M) — cells at padded indices 2..n+3
    al_L = al[:, :-1, :]        # al at left edge of each cell
    al_R = al[:, 1:, :]         # al at right edge of each cell

    bl = al_L - q_c
    br = al_R - q_c

    # pert_ppm(iv=0): positive definite constraint (tp_core.F90:610)
    bl, br = _pert_ppm_iv0(q_c, bl, br)

    # Iter-888: optional Fortran-faithful boundary `bl/br` overrides
    # via the s11/s14/s15 + 4-point xt formulas (tp_core.F90:614-628
    # left, :632-647 right).  Constants: s11 = 11/14, s14 = 4/7,
    # s15 = 3/14 (tp_core.F90:58).  Only fires for the legacy non-
    # duogrid path AND when `apply_fortran_xppm_boundary=True`.
    # Default False preserves pre-iter-888 behaviour (current
    # production runs do NOT pass this flag).
    #
    # Index map (qe has halo=3 padding so qe[k+1] is the cell at our
    # q-array index k):
    #   Fortran q1(-1) = qe[1] (halo depth-1)
    #   Fortran q1(0)  = qe[2] (halo depth-0) = q_c[0]
    #   Fortran q1(1)  = qe[3] (interior 0)   = q_c[1]
    #   Fortran q1(2)  = qe[4] (interior 1)   = q_c[2]
    #   Fortran q1(npx-2) = qe[n+1] (interior n-2) = q_c[n-1]
    #   Fortran q1(npx-1) = qe[n+2] (interior n-1) = q_c[n]
    #   Fortran q1(npx)   = qe[n+3] (halo depth-0) = q_c[n+1]
    #   Fortran q1(npx+1) = qe[n+4] (halo depth-1)
    # And for dm (shape (n+4), dm[k] = slope at qe[k+1]):
    #   Fortran dm(-1) = dm[0]
    #   Fortran dm(0)  = dm[1]
    #   Fortran dm(2)  = dm[3]
    #   Fortran dm(npx-2) = dm[n]
    #   Fortran dm(npx+1) = dm[n+3]
    if apply_fortran_xppm_boundary and fortran_legacy_face:
        s11 = 11.0 / 14.0
        s14 = 4.0 / 7.0
        s15 = 3.0 / 14.0

        # --- LEFT boundary (tp_core.F90:614-628) ---
        # Line 614: bl(0) = s14*dm(-1) + s11*(q1(-1)-q1(0))
        bl_0_L = s14 * dm[:, 0, :] + s11 * (qe[:, 1, :] - qe[:, 2, :])

        # Lines 616-617: 4-point xt (uniform-grid simplification of the
        # dxa-weighted formula).  For uniform dxa:
        #   xt = 0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))
        xt_L = (0.75 * (qe[:, 2, :] + qe[:, 3, :])
                - 0.25 * (qe[:, 1, :] + qe[:, 4, :]))
        # Lines 619-620: clip xt to within range of q1(-1..2)
        q_lo_L = jnp.minimum(jnp.minimum(qe[:, 1, :], qe[:, 2, :]),
                             jnp.minimum(qe[:, 3, :], qe[:, 4, :]))
        q_hi_L = jnp.maximum(jnp.maximum(qe[:, 1, :], qe[:, 2, :]),
                             jnp.maximum(qe[:, 3, :], qe[:, 4, :]))
        xt_L = jnp.clip(xt_L, q_lo_L, q_hi_L)

        # Line 622: br(0) = xt - q1(0)
        br_0_L = xt_L - qe[:, 2, :]
        # Line 623: bl(1) = xt - q1(1)
        bl_1_L = xt_L - qe[:, 3, :]

        # Line 624: xt2 = s15*q1(1) + s11*q1(2) - s14*dm(2)
        xt2_L = (s15 * qe[:, 3, :] + s11 * qe[:, 4, :]
                 - s14 * dm[:, 3, :])
        # Line 625: br(1) = xt2 - q1(1)
        br_1_L = xt2_L - qe[:, 3, :]
        # Line 626: bl(2) = xt2 - q1(2)
        bl_2_L = xt2_L - qe[:, 4, :]

        # Line 628: br(2) = al(3) - q1(2) — UNCHANGED from standard
        # (already computed by `br = al_R - q_c` above; al_R[2] = al[3]
        # and q_c[2] = qe[4] so br[2] = al[3] - qe[4] which equals
        # Fortran's br(2)).  No override needed.

        bl = bl.at[:, 0, :].set(bl_0_L)
        br = br.at[:, 0, :].set(br_0_L)
        bl = bl.at[:, 1, :].set(bl_1_L)
        br = br.at[:, 1, :].set(br_1_L)
        bl = bl.at[:, 2, :].set(bl_2_L)
        # br[2] left as-is (Fortran-faithful by construction).

        # --- RIGHT boundary (tp_core.F90:632-647) ---
        # Line 632: bl(npx-2) = al(npx-2) - q1(npx-2) — UNCHANGED
        # (al[n-1] - q_c[n-1] = al[n-1] - qe[n+1] which is what bl[n-1]
        # already holds.)

        # Line 634: xt = s15*q1(npx-1) + s11*q1(npx-2) + s14*dm(npx-2)
        xt_R = (s15 * qe[:, n + 2, :] + s11 * qe[:, n + 1, :]
                + s14 * dm[:, n, :])
        # Line 635: br(npx-2) = xt - q1(npx-2)
        br_nm2_R = xt_R - qe[:, n + 1, :]
        # Line 636: bl(npx-1) = xt - q1(npx-1)
        bl_nm1_R = xt_R - qe[:, n + 2, :]

        # Lines 638-639: 4-point xt (uniform-grid simplification).
        # For uniform dxa: xt = 0.75*(q1(npx-1)+q1(npx))
        #                       - 0.25*(q1(npx-2)+q1(npx+1))
        xt2_R = (0.75 * (qe[:, n + 2, :] + qe[:, n + 3, :])
                 - 0.25 * (qe[:, n + 1, :] + qe[:, n + 4, :]))
        # Lines 641-642: clip
        q_lo_R = jnp.minimum(jnp.minimum(qe[:, n + 1, :], qe[:, n + 2, :]),
                             jnp.minimum(qe[:, n + 3, :], qe[:, n + 4, :]))
        q_hi_R = jnp.maximum(jnp.maximum(qe[:, n + 1, :], qe[:, n + 2, :]),
                             jnp.maximum(qe[:, n + 3, :], qe[:, n + 4, :]))
        xt2_R = jnp.clip(xt2_R, q_lo_R, q_hi_R)

        # Line 644: br(npx-1) = xt - q1(npx-1)
        br_nm1_R = xt2_R - qe[:, n + 2, :]
        # Line 645: bl(npx) = xt - q1(npx)
        bl_n_R = xt2_R - qe[:, n + 3, :]

        # Line 647: br(npx) = s11*(q1(npx+1)-q1(npx)) - s14*dm(npx+1)
        br_n_R = (s11 * (qe[:, n + 4, :] - qe[:, n + 3, :])
                  - s14 * dm[:, n + 3, :])

        # Right-side overrides (positive indices for clarity):
        # bl/br shape is (n+2); index n-1 = -3, n = -2, n+1 = -1.
        # bl[n-1] (left as-is, line 632)
        br = br.at[:, n - 1, :].set(br_nm2_R)
        bl = bl.at[:, n, :].set(bl_nm1_R)
        br = br.at[:, n, :].set(br_nm1_R)
        bl = bl.at[:, n + 1, :].set(bl_n_R)
        br = br.at[:, n + 1, :].set(br_n_R)

    # pert_ppm(iv=1) at face-boundary cells: extra monotonicity for
    # the three cells whose PPM stencil crosses a face boundary.
    # Fortran tp_core.F90:629 calls pert_ppm iv=1 at indices 0,1,2 on
    # the left side and tp_core.F90:648 at npx-2,npx-1,npx on the right.
    # Fortran gates on (.not. (bounded_domain .or. duogrid)) at line 612:
    # duogrid provides real cross-face halo data so the extra iv=1
    # limiter is unnecessary.
    #
    # q_c shape is (n+2) with q_c[0] = halo-(-1) (Fortran q1(0)),
    # q_c[1..n] = interior 0..n-1 (Fortran q1(1..npx-1)),
    # q_c[n+1] = halo n (Fortran q1(npx)).
    #
    # Iter-884 (Fortran-fidelity off-by-one fix): Fortran's pert_ppm
    # call at line 629 uses bl/br indices 0,1,2 — which correspond to
    # cells q1(0), q1(1), q1(2) in Fortran indexing → q_c[0,1,2] in
    # our Python.  Pre-iter-884 we used [1, 2, 3] (shifted INWARD by
    # one cell) and the comment incorrectly labelled these "Fortran
    # interior 0,1,2".  Similarly on the right, Fortran uses indices
    # npx-2,npx-1,npx → q_c[-3,-2,-1] in our Python; pre-iter-884 we
    # used [-4,-3,-2] (also shifted INWARD).  iter-884 corrects to
    # [0, 1, 2, -3, -2, -1] to match Fortran's exact index range.
    #
    # Behavioural impact: the iv=1 constraint now applies to the
    # FACE-BOUNDARY-ADJACENT halo cell (q_c[0] / q_c[-1]) plus the
    # two adjacent interior cells, instead of three interior cells
    # one cell away from the boundary.  W2 sentinel uses
    # `FV3EdgeShallowWaterModel` which doesn't route through this
    # `_ppm_1d` (production W2 uses `cgrid_mass_flux_divergence` →
    # `_ppm_reconstruct_1d` in operators_cdgrid.py); the FB chain
    # transport path (`fv_tp_2d` → `_xppm`/`_yppm` → `_ppm_1d`) is
    # the affected code.
    if fortran_legacy_face:
        for k in [0, 1, 2, -3, -2, -1]:
            bl_k, br_k = _pert_ppm(bl[:, k, :], br[:, k, :])
            bl = bl.at[:, k, :].set(bl_k)
            br = br.at[:, k, :].set(br_k)

    return bl, br, q_c


def _xppm(q_h2, crx, n, off_left=None, off_right=None,
          off_left_d1=None, off_right_d1=None, use_duogrid=False,
          apply_fortran_xppm_boundary=False,
          bounded_domain=False):
    """PPM in x with hord=9 Courant-number integration.

    FV3 tp_core.F90 xppm lines 670-677: uses raw Courant number ``crx``
    in the standard PPM flux formula.  Boundary non-uniformity is handled
    entirely through bl/br corrections in ``_ppm_1d``, NOT by scaling crx
    (the Fortran does not adjust the Courant number at face boundaries).

    Iter-888b (Codex iter-888 stop-time fix): forwards the
    ``apply_fortran_xppm_boundary`` kwarg to ``_ppm_1d``.  Pre-iter-888b
    the kwarg was added to ``_ppm_1d`` only and was unreachable from
    every existing caller — Codex stop-time review correctly flagged
    this as dead code.  Default False preserves behaviour.

    Iter-890 (Codex iter-889b stop-time follow-up): forwards
    ``bounded_domain`` so ``_ppm_1d``'s gates match Fortran's full
    ``.not. (bounded_domain .or. duogrid)`` semantics on regional /
    nested panels.  Default False preserves global-cubed-sphere
    behaviour bit-for-bit.
    """
    bl, br, q_c = _ppm_1d(q_h2, n, off_left, off_right,
                           off_left_d1, off_right_d1,
                           use_duogrid=use_duogrid,
                           apply_fortran_xppm_boundary=(
                               apply_fortran_xppm_boundary),
                           bounded_domain=bounded_domain)
    bl_L, br_L, q_L = bl[:, :n+1, :], br[:, :n+1, :], q_c[:, :n+1, :]
    bl_R, br_R, q_R = bl[:, 1:n+2, :], br[:, 1:n+2, :], q_c[:, 1:n+2, :]

    fx_pos = q_L + (1.0 - crx) * (br_L - crx * (bl_L + br_L))
    fx_neg = q_R + (1.0 + crx) * (bl_R + crx * (bl_R + br_R))
    return jnp.where(crx > 0, fx_pos, fx_neg)


def _yppm(q_h2, cry, n, off_left=None, off_right=None,
          off_left_d1=None, off_right_d1=None, use_duogrid=False,
          apply_fortran_xppm_boundary=False,
          bounded_domain=False):
    """PPM in y with hord=9 Courant-number integration.

    Iter-888b: see ``_xppm`` docstring for the
    ``apply_fortran_xppm_boundary`` plumbing rationale.
    Iter-890: ``bounded_domain`` plumbing — see ``_xppm`` docstring.
    """
    q_t = jnp.swapaxes(q_h2, 1, 2)
    c_t = jnp.swapaxes(cry, 1, 2)
    bl, br, q_c = _ppm_1d(q_t, n, off_left, off_right,
                           off_left_d1, off_right_d1,
                           use_duogrid=use_duogrid,
                           apply_fortran_xppm_boundary=(
                               apply_fortran_xppm_boundary),
                           bounded_domain=bounded_domain)
    bl_L, br_L, q_L = bl[:, :n+1, :], br[:, :n+1, :], q_c[:, :n+1, :]
    bl_R, br_R, q_R = bl[:, 1:n+2, :], br[:, 1:n+2, :], q_c[:, 1:n+2, :]
    fy_pos = q_L + (1.0 - c_t) * (br_L - c_t * (bl_L + br_L))
    fy_neg = q_R + (1.0 + c_t) * (bl_R + c_t * (bl_R + br_R))
    return jnp.swapaxes(jnp.where(c_t > 0, fy_pos, fy_neg), 1, 2)


# =========================================================================
# Transport quantities
# =========================================================================

def compute_transport_quantities(ut, vt, dt, cdgrid):
    """Compute Courant numbers, area fluxes, swept areas.

    Follows FV3 sw_core.F90:830-862:
    - xfx = dt * ut * dy * sin_sg(upwind)
    - crx = xfx * rdxa(upwind_cell)  (NOT rdxc at face!)
    The Fortran uses upwind-selected cell-centre rdxa for the Courant
    number, which gives the CFL as a fraction of the upwind CELL WIDTH.
    """
    n = cdgrid.n
    grid = cdgrid.base
    sin_sg = cdgrid.sin_sg

    dy = cdgrid.dy_edge_x
    dx = cdgrid.dx_edge_y

    # Route halo exchanges through the duogrid kinked-to-extended remap
    # when duogrid is active, so that sin_sg / rdxa / rdya at cross-face
    # halo cells have duogrid quality (matches the Fortran `bounded_domain`
    # path in sw_core.F90:830-862 which skips `copy_corners`).
    dg = grid.duogrid
    _use_dg = dg is not None and dg.ng >= 2
    _offs = None if _use_dg else grid.halo_interp_offsets
    _dg = dg if _use_dg else None

    # --- Transport distance (Fortran: xfx_adv = dt*ut before dy*sin scaling) ---
    xfx_raw = dt * ut   # (6, n+1, n) distance in contravariant coords
    yfx_raw = dt * vt   # (6, n, n+1)

    # --- x-direction Courant number (FV3 sw_core.F90:849-853) ---
    # crx = (dt*ut) * rdxa(upwind_cell)  where rdxa = 1/cell_width.
    # FV3 computes rdxa from exact face-to-face distance (fv_grid_tools.F90).
    rdxa_pad = pad_halo(cdgrid.rdxa, interp_offsets=_offs, duogrid=_dg)
    rdxa_upwind = jnp.where(ut > 0,
                            rdxa_pad[:, :n+1, 1:-1],    # cell i-1
                            rdxa_pad[:, 1:n+2, 1:-1])   # cell i
    crx = xfx_raw * rdxa_upwind

    # --- x-direction area flux (xfx = dt*ut*dy*sin_sg_upwind) ---
    sin_east = sin_sg[:, :, :, 2]
    sin_west = sin_sg[:, :, :, 0]
    se_pad = pad_halo(sin_east, interp_offsets=_offs, duogrid=_dg)
    sw_pad = pad_halo(sin_west, interp_offsets=_offs, duogrid=_dg)
    sin_x = jnp.where(ut > 0, se_pad[:, :n+1, 1:-1], sw_pad[:, 1:n+2, 1:-1])
    xfx = xfx_raw * dy * sin_x

    # --- y-direction Courant number ---
    rdya_pad = pad_halo(cdgrid.rdya, interp_offsets=_offs, duogrid=_dg)
    rdya_upwind = jnp.where(vt > 0,
                            rdya_pad[:, 1:-1, :n+1],
                            rdya_pad[:, 1:-1, 1:n+2])
    cry = yfx_raw * rdya_upwind

    # --- y-direction area flux ---
    sin_north = sin_sg[:, :, :, 3]
    sin_south = sin_sg[:, :, :, 1]
    sn_pad = pad_halo(sin_north, interp_offsets=_offs, duogrid=_dg)
    ss_pad = pad_halo(sin_south, interp_offsets=_offs, duogrid=_dg)
    sin_y = jnp.where(vt > 0, sn_pad[:, 1:-1, :n+1], ss_pad[:, 1:-1, 1:n+2])
    yfx = yfx_raw * dx * sin_y

    area = grid.area
    ra_x = area + xfx[:, :-1, :] - xfx[:, 1:, :]
    ra_y = area + yfx[:, :, :-1] - yfx[:, :, 1:]
    return crx, cry, xfx, yfx, ra_x, ra_y


# =========================================================================
# fv_tp_2d: Lin-Rood 2D transport
# =========================================================================

def _deln_flux(nord, damp, q, fx, fy, cdgrid, mass=None):
    """FV3 deln_flux: del-n damping for cell-mean values (tp_core.F90:1217-1365).

    Adds diffusive fluxes to the transport fluxes ``fx``/``fy`` to provide
    del-2 (nord=0), del-4 (nord=1), or del-6 (nord=2) damping.

    Currently implements nord=0 (del-2).  Higher orders require iterative
    Laplacian application with intermediate halo exchanges.

    Parameters
    ----------
    nord : int — damping order (0=del-2, 1=del-4, ...)
    damp : float — damping coefficient (pre-scaled: (damp_c * da_min)^(nord+1))
    q : (6, n, n) — transported field (cell-mean values)
    fx : (6, n+1, n) — x-direction transport flux (modified in-place)
    fy : (6, n, n+1) — y-direction transport flux (modified in-place)
    cdgrid : CubedSphereCDGrid
    mass : (6, n, n) or None — mass field for mass-weighted damping

    Returns
    -------
    fx, fy : modified fluxes with diffusive contribution added
    """
    n = cdgrid.n
    grid = cdgrid.base
    sg = cdgrid.sin_sg
    dy = cdgrid.dy_edge_x   # (6, n+1, n) — edge length at u-faces
    dx = cdgrid.dx_edge_y   # (6, n, n+1) — edge length at v-faces
    rdxc = cdgrid.rdxc       # (6, n+1, n)
    rdyc = cdgrid.rdyc       # (6, n, n+1)
    rarea = 1.0 / grid.area  # (6, n, n)

    # Route halo exchanges through the duogrid kinked-to-extended remap
    # when duogrid is active, so that face-boundary values used by the
    # Laplacian stencil have duogrid quality (matches the bounded_domain
    # path in the Fortran `deln_flux`, which skips `copy_corners` and
    # relies on duogrid MPI halo exchange).
    dg = grid.duogrid
    _use_dg = dg is not None and dg.ng >= 2
    _offs = None if _use_dg else grid.halo_interp_offsets
    _dg_arg = dg if _use_dg else None

    # Step 1: initialize d2 (tp_core.F90:1253-1265)
    if mass is None:
        d2 = damp * q
    else:
        d2 = q

    # Step 2: Laplacian diffusive fluxes (tp_core.F90:1270-1290, USE_SG path)
    # fx2 = 0.5*(sin_sg(i-1,j,E)+sin_sg(i,j,W)) * dy * (d2[i-1]-d2[i]) * rdxc
    d2_pad = pad_halo(d2, interp_offsets=_offs, duogrid=_dg_arg)
    sin_E = sg[:, :, :, 2]   # E-edge
    sin_W = sg[:, :, :, 0]   # W-edge
    sin_E_pad = pad_halo(sin_E, interp_offsets=_offs, duogrid=_dg_arg)
    sin_W_pad = pad_halo(sin_W, interp_offsets=_offs, duogrid=_dg_arg)
    sin_uv_x = 0.5 * (sin_E_pad[:, :n+1, 1:-1] + sin_W_pad[:, 1:n+2, 1:-1])
    fx2 = sin_uv_x * dy * (d2_pad[:, :-1, 1:-1] - d2_pad[:, 1:, 1:-1]) * rdxc

    sin_N = sg[:, :, :, 3]   # N-edge
    sin_S = sg[:, :, :, 1]   # S-edge
    sin_N_pad = pad_halo(sin_N, interp_offsets=_offs, duogrid=_dg_arg)
    sin_S_pad = pad_halo(sin_S, interp_offsets=_offs, duogrid=_dg_arg)
    sin_uv_y = 0.5 * (sin_N_pad[:, 1:-1, :n+1] + sin_S_pad[:, 1:-1, 1:n+2])
    fy2 = sin_uv_y * dx * (d2_pad[:, 1:-1, :-1] - d2_pad[:, 1:-1, 1:]) * rdyc

    # Step 3: Higher-order iteration (nord > 0, tp_core.F90:1298-1331)
    for _it in range(nord):
        # Compute divergence of diffusive fluxes
        d2 = (fx2[:, :-1, :] - fx2[:, 1:, :] + fy2[:, :, :-1] - fy2[:, :, 1:]) * rarea
        # Re-exchange and recompute fluxes with sign flip (d2[i]-d2[i-1])
        d2_pad = pad_halo(d2, interp_offsets=_offs, duogrid=_dg_arg)
        fx2 = sin_uv_x * dy * (d2_pad[:, 1:, 1:-1] - d2_pad[:, :-1, 1:-1]) * rdxc
        fy2 = sin_uv_y * dx * (d2_pad[:, 1:-1, 1:] - d2_pad[:, 1:-1, :-1]) * rdyc

    # Step 4: Add diffusive fluxes to transport fluxes (tp_core.F90:1339-1363)
    if mass is not None:
        mass_pad = pad_halo(mass, interp_offsets=_offs, duogrid=_dg_arg)
        mass_u = 0.5 * (mass_pad[:, :-1, 1:-1] + mass_pad[:, 1:, 1:-1])  # (6, n+1, n)
        mass_v = 0.5 * (mass_pad[:, 1:-1, :-1] + mass_pad[:, 1:-1, 1:])  # (6, n, n+1)
        fx = fx + 0.5 * damp * mass_u * fx2
        fy = fy + 0.5 * damp * mass_v * fy2
    else:
        fx = fx + fx2
        fy = fy + fy2

    return fx, fy


def fv_tp_2d(q, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
             nord=None, damp_c=None, mass=None,
             apply_cgrid_flux_sync=True,
             apply_fortran_xppm_boundary=False):
    """Lin-Rood operator-split 2D transport (Putman & Lin 2007).

    Parameters
    ----------
    apply_cgrid_flux_sync : bool, default True
        Iter-864: when ``True`` (default), the duogrid CGRID_NE flux
        synchronization (`synchronize_cgrid_fluxes`) is applied at the
        end so neighbour-face flux pairs agree post-iter-808 sign-flip.
        Default ``True`` matches Fortran's ``mpp_get_boundary``
        averaging block in dyn_core.F90:850-900 around the d_sw1 mass
        flux call.

        Set ``False`` to MATCH Fortran's commented-out vorticity-flux
        averaging block in dyn_core.F90:1124-1207 (the d_sw5 → d_sw6
        sync that Fortran has explicitly disabled).  The FB chain's
        vorticity-flux call (`_d_sw_native` step 7) passes ``False``
        so the post-d_sw5 vortfluxx/vortfluxy fields match Fortran's
        un-synchronised behaviour exactly.

        Production `fv3_sw_tendencies` does not call this helper, so
        it is unaffected.  `transport_step` and direct callers default
        to ``True``, preserving prior behaviour for the mass-flux call.
    apply_fortran_xppm_boundary : bool, default False
        Iter-888b (Codex iter-888 stop-time fix): forwards through to
        ``_xppm`` / ``_yppm`` / ``_ppm_1d`` so the Fortran s11/s14/s15
        boundary formula at ``tp_core.F90:614-628`` (left) and
        ``:632-647`` (right) becomes reachable from FB-chain transport.
        Pre-iter-888b the kwarg was added at the ``_ppm_1d`` leaf only
        and was dead code from every transport caller.  Default False
        preserves prior behaviour bit-for-bit.  See iter-888 doc entry
        for scope and the uniform-grid simplification rationale.
    """
    n = cdgrid.n
    grid = cdgrid.base
    area = grid.area
    offsets_h2 = grid.halo_interp_offsets_h2

    # Matches Fortran bounded_domain = (regional .or. nested .or. duogrid)
    # (fv_arrays.F90:1512).  The flag gates face-boundary specials in
    # tp_core.F90 and sw_core.F90 away from duogrid/bounded-domain paths.
    #
    # When duogrid is active we also switch `pad_halo` from interp_offsets
    # mode to full duogrid mode so the halo quality that justifies the iv=1
    # gate is actually delivered (mirrors the _pad_halo_auto_h2 pattern in
    # operators_cdgrid.py).  pad_halo rejects both kwargs simultaneously.
    dg = grid.duogrid
    use_duogrid = dg is not None and dg.ng >= 2
    halo_offsets = None if use_duogrid else offsets_h2
    halo_dg = dg if use_duogrid else None

    # Iter-890 (Codex iter-889b stop-time follow-up): the FB-chain
    # `_ppm_1d` legacy face-boundary specials must also be bypassed on
    # regional / nested bounded-domain panels (where `bounded_domain`
    # is True but `use_duogrid` is False).  Forward `bounded_domain`
    # to `_xppm` / `_yppm` so they can hand it through to `_ppm_1d`.
    bounded_domain = bool(grid.bounded_domain)

    # Iter-890c (Codex iter-890b stop-time fix).  Pre-iter-890b the
    # offset extraction below would crash on regional / nested panels
    # because `cubed_sphere.py:783` sets `halo_interp_offsets_h2=None`
    # for single-face panels (`_pad_halo_wall` is the wall-BC path
    # that makes the offsets unnecessary in the first place).
    # iter-890b added a `NotImplementedError` guard, which Codex
    # correctly noted converted a silent TypeError into an explicit
    # crash on a path that should ACTUALLY work — the iter-890 gate
    # already ensures the legacy boundary specials are bypassed for
    # `bounded_domain=True`, so the offsets are unused on the regional
    # path anyway.  iter-890c replaces the guard with conditional
    # extraction: when `offsets_h2 is None` we set every offset to
    # ``None`` and rely on `_ppm_1d`'s `fortran_legacy_face` gate
    # (which is False for `bounded_domain=True`) to skip the offset-
    # consuming code path entirely.  `pad_halo` already dispatches to
    # `_pad_halo_wall` for single-face inputs regardless of
    # `interp_offsets`, so the halo padding is correct on regional
    # grids without further changes.

    # Extract boundary offsets for sweep directions when available.
    # offsets_h2: (6, 4, 2, n) — [face, edge, depth, cell_along_edge]
    # WEST=0, EAST=1, SOUTH=2, NORTH=3; depth 0 = adjacent to interior
    if offsets_h2 is not None:
        ox_L0 = offsets_h2[:, 0, 0, :]   # WEST depth=0
        ox_R0 = offsets_h2[:, 1, 0, :]   # EAST depth=0
        oy_L0 = offsets_h2[:, 2, 0, :]   # SOUTH depth=0
        oy_R0 = offsets_h2[:, 3, 0, :]   # NORTH depth=0
        ox_L1 = offsets_h2[:, 0, 1, :]   # WEST depth=1
        ox_R1 = offsets_h2[:, 1, 1, :]   # EAST depth=1
        oy_L1 = offsets_h2[:, 2, 1, :]   # SOUTH depth=1
        oy_R1 = offsets_h2[:, 3, 1, :]   # NORTH depth=1
    else:
        # Regional / nested panel: offsets are unused because the
        # iter-890 `bounded_domain=True` gate inside `_ppm_1d` short-
        # circuits the offset-consuming code path.  Set all offsets to
        # None so any accidental use (which would indicate a gate
        # regression) raises a clear AttributeError.
        ox_L0 = ox_R0 = oy_L0 = oy_R0 = None
        ox_L1 = ox_R1 = oy_L1 = oy_R1 = None

    q_full = pad_halo(q, halo=2, interp_offsets=halo_offsets, duogrid=halo_dg)

    # Pass 1: Y-sweep on q, X-sweep on cross-corrected q_i
    fy2 = _yppm(q_full[:, 2:-2, :], cry, n, oy_L0, oy_R0, oy_L1, oy_R1,
                use_duogrid=use_duogrid,
                apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
                bounded_domain=bounded_domain)
    fyy = yfx * fy2
    q_i = (q * area + fyy[:, :, :-1] - fyy[:, :, 1:]) / ra_y

    # Proper halo exchange for q_i (required for mass conservation)
    q_i_pad = pad_halo(q_i, halo=2, interp_offsets=halo_offsets, duogrid=halo_dg)
    fx1 = _xppm(q_i_pad[:, :, 2:-2], crx, n, ox_L0, ox_R0, ox_L1, ox_R1,
                use_duogrid=use_duogrid,
                apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
                bounded_domain=bounded_domain)

    # Pass 2: X-sweep on q, Y-sweep on cross-corrected q_j
    fx2 = _xppm(q_full[:, :, 2:-2], crx, n, ox_L0, ox_R0, ox_L1, ox_R1,
                use_duogrid=use_duogrid,
                apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
                bounded_domain=bounded_domain)
    fxx = xfx * fx2
    q_j = (q * area + fxx[:, :-1, :] - fxx[:, 1:, :]) / ra_x

    # Proper halo exchange for q_j (required for mass conservation)
    q_j_pad = pad_halo(q_j, halo=2, interp_offsets=halo_offsets, duogrid=halo_dg)
    fy1 = _yppm(q_j_pad[:, 2:-2, :], cry, n, oy_L0, oy_R0, oy_L1, oy_R1,
                use_duogrid=use_duogrid,
                apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
                bounded_domain=bounded_domain)

    if mass is not None:
        # With mass: fx = 0.5*(fx1+fx2)*mfx, fy = 0.5*(fy1+fy2)*mfy
        # (tp_core.F90:188-196).  Here mfx/mfy are the mass fluxes = xfx/yfx.
        fx = 0.5 * (fx1 + fx2) * xfx
        fy = 0.5 * (fy1 + fy2) * yfx
    else:
        # Without mass: fx = 0.5*(fx1+fx2)*xfx, fy = 0.5*(fy1+fy2)*yfx
        # (tp_core.F90:207-216)
        fx = 0.5 * (fx1 + fx2) * xfx
        fy = 0.5 * (fy1 + fy2) * yfx

    # Del-n damping (tp_core.F90:197-201 and 217-222)
    if nord is not None and damp_c is not None and damp_c > 1e-4:
        damp = (damp_c * jnp.min(cdgrid.base.area)) ** (nord + 1)
        fx, fy = _deln_flux(nord, damp, q, fx, fy, cdgrid, mass=mass)

    # Duogrid flux synchronization (see cgrid_mass_flux_divergence for rationale).
    # Iter-864: gated by `apply_cgrid_flux_sync` so the FB chain's d_sw5
    # vorticity-flux call can opt out and match Fortran's commented-out
    # `mpp_get_boundary(... gridtype=CGRID_NE)` averaging at
    # dyn_core.F90:1124-1207.  Default True keeps the iter-808 sync
    # active for the mass-flux callers (`transport_step`).
    dg = cdgrid.base.duogrid
    if apply_cgrid_flux_sync and dg is not None and dg.ng >= 2:
        from legoesm.grids.halo import synchronize_cgrid_fluxes
        fx, fy = synchronize_cgrid_fluxes(fx, fy, n)

    return fx, fy


def transport_step(h, ut, vt, dt, cdgrid, mass_target=None,
                   nord=None, damp_c=None,
                   apply_fortran_xppm_boundary=False, **_kwargs):
    """Single FV3-style transport step with mass conservation.

    Parameters
    ----------
    h : (6, n, n) height field
    ut, vt : contravariant velocities
    dt : timestep
    cdgrid : CubedSphereCDGrid
    mass_target : float or None — if provided, enforce exact mass conservation
    nord, damp_c : optional del-n damping parameters forwarded to
        ``fv_tp_2d``.  Matches Fortran sw_core.F90:886-887 which calls
        ``fv_tp_2d(delp, ..., nord=nord_v, damp_c=damp_v)`` inside
        ``d_sw1`` so that mass transport picks up the same 4th-order
        smoother the Fortran FB chain applies.  Both default ``None``
        (no damping) to preserve legacy behaviour for existing callers
        that do not pass these kwargs (iter-727).
    apply_fortran_xppm_boundary : bool, default False
        Iter-888b (Codex iter-888 stop-time fix): forwards through to
        ``fv_tp_2d`` so FB-chain transport callers (e.g. ``_d_sw_native``)
        can opt into Fortran's s11/s14/s15 boundary formulas
        (tp_core.F90:614-628, :632-647).  Default False preserves
        prior behaviour bit-for-bit; the kwarg is reachable from
        ``_d_sw_native`` via this plumbing.
    """
    area = cdgrid.base.area
    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
        ut, vt, dt, cdgrid)
    fx, fy = fv_tp_2d(h, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
                      nord=nord, damp_c=damp_c,
                      apply_fortran_xppm_boundary=(
                          apply_fortran_xppm_boundary))
    h_new = h + (fx[:, :-1, :] - fx[:, 1:, :]
                 + fy[:, :, :-1] - fy[:, :, 1:]) / area

    # Mass conservation fixer: clip negative values and rescale
    # positive values to conserve total mass.  This compensates for
    # flux mismatches at face boundaries while maintaining non-negativity.
    if mass_target is not None:
        # Step 1: clip negatives to zero
        h_pos = jnp.maximum(h_new, 0.0)
        mass_pos = jnp.sum(h_pos * area)
        # Step 2: scale positive values to match target mass
        scale = mass_target / jnp.maximum(mass_pos, 1.0)
        h_new = h_pos * scale

    return h_new
