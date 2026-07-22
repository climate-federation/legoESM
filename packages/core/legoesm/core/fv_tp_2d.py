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

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.grids.halo import (
    pad_halo,
    pad_halo_4d,
    pad_halo_pair_h2,
    synchronize_cgrid_fluxes,
)

_R3 = 1.0 / 3.0


def pert_ppm(bl, br):
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


def apply_hord8_limiter(bl, br, dm):
    """FV3_3D iter 585: FV3 iord=8 Lin (1996) monotonicity limiter
    (tp_core.F90:548-553).

    Alternative to ``pert_ppm`` (iord=9).  iord=8 bounds bl, br by
    ±2·|dm| where dm is the cell-center monotone slope:

        xt = 2·dm
        bl = -sign(min(|xt|, |al - q|), xt)
        br =  sign(min(|xt|, |al' - q|), xt)

    Since legoESM's bl = al - q and br = al' - q are passed directly,
    we re-express as:

        bl_out = clip(bl, -2|dm|, 2|dm|) with sign-preserved
        br_out = clip(br, -2|dm|, 2|dm|) with sign-preserved

    But the FV3 form is more subtle: it forces the bl/br sign to match
    xt's sign (i.e., the monotone slope sign).

    Parameters
    ----------
    bl, br : jax.Array
        Standard PPM left/right cell-edge perturbations (al-q, al'-q).
    dm : jax.Array
        Cell-center monotone slope.

    Returns
    -------
    bl_out, br_out : jax.Array
        iord=8 limited values.

    Notes
    -----
    NOT yet wired into the default transport path (which uses iord=9
    via pert_ppm).  Exposed as a utility for users who want the iord=8
    variant for tracers (e.g., as FV3 namelist sets hord_tr=8 in some
    configs).
    """
    xt = 2.0 * dm
    bl_out = -jnp.sign(xt) * jnp.minimum(jnp.abs(xt), jnp.abs(bl))
    br_out = jnp.sign(xt) * jnp.minimum(jnp.abs(xt), jnp.abs(br))
    # Fortran flips sign on bl: bl uses -sign(min(|xt|,|al-q|), xt).
    # al-q corresponds to bl here.  Note xt = 2*dm; sign(xt)=sign(dm).
    # The -sign accounts for bl pointing in the OPPOSITE direction of dm
    # (al at i-1/2 is one cell to the left → bl = al - q < 0 typically
    #  when dm > 0).
    return bl_out, br_out


def apply_hord10_limiter(bl, br, dm, q):
    """FV3_3D iter 593: FV3 iord=10 Lin+Rood (1996) limiter with
    pmp/lac extra constraints (tp_core.F90:554-572).

    The most subtle of FV3's hord variants.  Uses one-sided differences
    ``dq[i] = 2·(q[i+1] - q[i])`` to build pmp (positive max) and lac
    (lower asymmetric constraint) bounds that prevent new extrema while
    allowing tighter convergence than iord=9.

    Algorithm:
        dq[i] = 2·(q[i+1] - q[i])
        # near-flat region: zero bl, br
        if |dm[i-1]| + |dm[i]| + |dm[i+1]| < near_zero:
            bl, br = 0, 0
        # new extremum: apply pmp/lac bounds
        elif |3·(bl+br)| > |bl-br|:
            pmp_2 = dq[i-1]; lac_2 = pmp_2 - 0.75·dq[i-2]
            br = min(max(0, pmp_2, lac_2),
                     max(br, min(0, pmp_2, lac_2)))
            pmp_1 = -dq[i]; lac_1 = pmp_1 + 0.75·dq[i+1]
            bl = min(max(0, pmp_1, lac_1),
                     max(bl, min(0, pmp_1, lac_1)))

    Parameters
    ----------
    bl, br, dm, q : jax.Array, shape (..., N)
        Cell-center perturbations, monotone slope, and cell values
        along the transport axis (last dimension).  N must be ≥ 5
        (need 2 ghosts on each side for dq stencil).

    Returns
    -------
    bl_out, br_out : jax.Array, same shape as inputs.

    Notes
    -----
    NOT yet wired into the default transport path.  Exposed as a
    utility.  Interior cells [2:-2] are limited; boundary cells
    keep their original bl, br (caller's responsibility to handle
    halos / pad if needed).
    """
    near_zero = 1e-30
    # dq[i] = 2·(q[i+1] - q[i]) — needs N+1 q values; truncate.
    # Build dq along last axis.
    dq = 2.0 * (q[..., 1:] - q[..., :-1])  # shape (..., N-1)

    # For interior cells i ∈ [2, N-3], we need:
    #   dm[i-1], dm[i], dm[i+1]
    #   dq[i-2], dq[i-1], dq[i], dq[i+1]
    # Build aligned slices for interior region.
    bl_i = bl[..., 2:-2]
    br_i = br[..., 2:-2]
    dm_im1 = dm[..., 1:-3]
    dm_i = dm[..., 2:-2]
    dm_ip1 = dm[..., 3:-1]
    dq_im2 = dq[..., :-3]
    dq_im1 = dq[..., 1:-2]
    dq_i = dq[..., 2:-1]
    dq_ip1 = dq[..., 3:]

    sum_dm = jnp.abs(dm_im1) + jnp.abs(dm_i) + jnp.abs(dm_ip1)
    is_flat = sum_dm < near_zero
    has_new_extremum = jnp.abs(3.0 * (bl_i + br_i)) > jnp.abs(bl_i - br_i)

    pmp_2 = dq_im1
    lac_2 = pmp_2 - 0.75 * dq_im2
    br_clipped = jnp.minimum(
        jnp.maximum(jnp.maximum(0.0, pmp_2), lac_2),
        jnp.maximum(br_i, jnp.minimum(jnp.minimum(0.0, pmp_2), lac_2)),
    )
    pmp_1 = -dq_i
    lac_1 = pmp_1 + 0.75 * dq_ip1
    bl_clipped = jnp.minimum(
        jnp.maximum(jnp.maximum(0.0, pmp_1), lac_1),
        jnp.maximum(bl_i, jnp.minimum(jnp.minimum(0.0, pmp_1), lac_1)),
    )

    bl_new = jnp.where(
        is_flat, 0.0,
        jnp.where(has_new_extremum, bl_clipped, bl_i),
    )
    br_new = jnp.where(
        is_flat, 0.0,
        jnp.where(has_new_extremum, br_clipped, br_i),
    )

    # Re-assemble: keep boundary cells unchanged, update interior.
    bl_out = bl.at[..., 2:-2].set(bl_new)
    br_out = br.at[..., 2:-2].set(br_new)
    return bl_out, br_out


def apply_hord11_limiter(bl, br, dm, ppm_fac: float = 1.5):
    """FV3_3D iter 592: FV3 iord=11 limiter (tp_core.F90:573-579).

    Same form as iord=8 (iter 585) but with a configurable factor
    ``ppm_fac`` instead of fixed 2.0.  FV3 default ``ppm_fac = 1.5``
    (tp_core.F90:35).  Called "emulation of 2nd van Leer scheme using
    PPM codes".

    Formula:
        xt = ppm_fac · dm
        bl = -sign(min(|xt|, |al - q|), xt)
        br =  sign(min(|xt|, |al' - q|), xt)

    Parameters
    ----------
    bl, br : jax.Array
        Standard PPM left/right cell-edge perturbations.
    dm : jax.Array
        Cell-center monotone slope.
    ppm_fac : float, default 1.5
        FV3's ppm_fac parameter, "nonlinear scheme limiter:
        between 1 and 2" (tp_core.F90:35 comment).

    Returns
    -------
    bl_out, br_out : jax.Array
        iord=11 limited values.

    Notes
    -----
    NOT yet wired into the default transport path (still iord=9).
    Exposed as a utility for users who want iord=11 for tracers.
    With ``ppm_fac=2.0`` this is exactly iord=8 (iter 585).
    """
    xt = ppm_fac * dm
    bl_out = -jnp.sign(xt) * jnp.minimum(jnp.abs(xt), jnp.abs(bl))
    br_out = jnp.sign(xt) * jnp.minimum(jnp.abs(xt), jnp.abs(br))
    return bl_out, br_out


def _ppm_1d(q, n, off_left=None, off_right=None,
            off_left_d1=None, off_right_d1=None,
            use_duogrid=False,
            apply_fortran_xppm_boundary=False,
            bounded_domain=False,
            hord: int = 12):
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

    # FV3_3D iter 595: hord-dispatched limiter (default hord=12 = iv=0
    # positive-def, preserving pre-iter-595 behavior).
    # dm has shape (6, n+4, M); q_c is qe[:, 2:-2, :] → cells at padded
    # indices 2..n+3 → dm at those positions is dm[:, 1:n+3, :].
    if hord == 8:
        dm_c = dm[:, 1:n + 3, :]
        bl, br = apply_hord8_limiter(bl, br, dm_c)
    elif hord == 9:
        # FV3 iord=9 → pert_ppm(iv=0) for the SCALAR/mass/vorticity transport
        # (tp_core.F90:610: `if(iord==9 .or. iord==13) call pert_ppm(...,0)`).
        # iv=1 (`pert_ppm`) is FV3's BOUNDARY-only limiter (tp_core.F90:629,
        # 648) + the MOMENTUM ytp_v/xtp_u path (handled separately in
        # fv3_sw_core `ppm_transport_1d`).  This `_ppm_1d` is the scalar
        # path, so hord=9 must use iv=0 — matching the `_pert_ppm_iv0`
        # docstring ("the limiter used by hord=9") and the hord=12 default.
        # (Was `pert_ppm` (iv=1): a latent mislabel; unexercised because the
        # live scalar callers use the hord=12 default — codex/oracle iter62.)
        bl, br = _pert_ppm_iv0(q_c, bl, br)
    elif hord == 10:
        dm_c = dm[:, 1:n + 3, :]
        bl, br = apply_hord10_limiter(bl, br, dm_c, q_c)
    elif hord == 11:
        dm_c = dm[:, 1:n + 3, :]
        bl, br = apply_hord11_limiter(bl, br, dm_c)
    elif hord == 12:
        # pert_ppm(iv=0): positive definite constraint (tp_core.F90:610)
        bl, br = _pert_ppm_iv0(q_c, bl, br)
    else:
        raise ValueError(
            f"hord must be one of {{8, 9, 10, 11, 12}}; got {hord}"
        )

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
            bl_k, br_k = pert_ppm(bl[:, k, :], br[:, k, :])
            bl = bl.at[:, k, :].set(bl_k)
            br = br.at[:, k, :].set(br_k)

    return bl, br, q_c


def _xppm(q_h2, crx, n, off_left=None, off_right=None,
          off_left_d1=None, off_right_d1=None, use_duogrid=False,
          apply_fortran_xppm_boundary=False,
          bounded_domain=False,
          hord: int = 12):
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
                           bounded_domain=bounded_domain,
                           hord=hord)
    bl_L, br_L, q_L = bl[:, :n+1, :], br[:, :n+1, :], q_c[:, :n+1, :]
    bl_R, br_R, q_R = bl[:, 1:n+2, :], br[:, 1:n+2, :], q_c[:, 1:n+2, :]

    fx_pos = q_L + (1.0 - crx) * (br_L - crx * (bl_L + br_L))
    fx_neg = q_R + (1.0 + crx) * (bl_R + crx * (bl_R + br_R))
    return jnp.where(crx > 0, fx_pos, fx_neg)


def _yppm(q_h2, cry, n, off_left=None, off_right=None,
          off_left_d1=None, off_right_d1=None, use_duogrid=False,
          apply_fortran_xppm_boundary=False,
          bounded_domain=False,
          hord: int = 12):
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
                           bounded_domain=bounded_domain,
                           hord=hord)
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

    # Step 1: initialize d2 (tp_core.F90:1253-1265).
    #
    # Iter-934 (FB-chain low-res NaN fix): factor `damp` out of the
    # iteration loop and apply it at Step 4 instead of Step 1.  All
    # operations between Step 1 and Step 4 are LINEAR in `d2`, so the
    # overall result is mathematically identical, but the intermediate
    # `d2`/`fx2`/`fy2` arrays no longer carry the huge `damp` factor
    # (which scales as `area^(nord+1)`).  At low resolution this is
    # critical: at C8, `damp = (damp_c*area)^(nord+1) ≈ 4e32`; the
    # product `damp * q * dy ≈ 1.5e42` overflows float32 (max 3.4e38)
    # even though the final per-step damped flux is small.  Factoring
    # damp out keeps every intermediate within float32 range.
    #
    # Pre-iter-934 the FB chain produced NaN in `h` at C8/C12/C16
    # (iter-933 stability scan).  iter-934 fix: defer damp to Step 4.
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

    # Step 4: Add diffusive fluxes to transport fluxes (tp_core.F90:1339-1363).
    # Apply `damp` here rather than at Step 1 (iter-934 float32-overflow fix).
    if mass is not None:
        mass_pad = pad_halo(mass, interp_offsets=_offs, duogrid=_dg_arg)
        # ``mass_u``/``mass_v`` ARE the 0.5 face-average — the SAME ``0.5`` the
        # Fortran carries inside ``damp*0.5*(mass(i-1,j)+mass(i,j))*fx2``
        # (tp_core.F90:1339-1363).  So the increment is ``damp * mass_avg *
        # fx2`` — an extra ``0.5`` here double-counted the average and applied
        # HALF the certified del-n damping on the mass-weighted path (#1255).
        mass_u = 0.5 * (mass_pad[:, :-1, 1:-1] + mass_pad[:, 1:, 1:-1])  # (6, n+1, n)
        mass_v = 0.5 * (mass_pad[:, 1:-1, :-1] + mass_pad[:, 1:-1, 1:])  # (6, n, n+1)
        fx = fx + damp * mass_u * fx2
        fy = fy + damp * mass_v * fy2
    else:
        fx = fx + damp * fx2
        fy = fy + damp * fy2

    return fx, fy


class _TpSetup(NamedTuple):
    """Level-INDEPENDENT ``fv_tp_2d`` setup (grid geometry + PPM boundary offsets).

    Extracted so the PURE-LOCAL PPM sweeps (:func:`_fv_tp_2d_sweep1` /
    :func:`_fv_tp_2d_sweep2`) can be shared bit-for-bit between the 2D
    :func:`fv_tp_2d` (one ``pad_halo`` per exchange) and the 4D-halo transport
    (one ``pad_halo_4d`` for ALL levels — #811).  All fields are grid constants,
    so under ``vmap(level)`` they are captured (unbatched) and vmap-safe.
    """
    n: int
    area: jax.Array
    use_duogrid: bool
    halo_offsets: jax.Array | None
    halo_dg: object
    bounded_domain: bool
    ox_L0: jax.Array | None
    ox_R0: jax.Array | None
    oy_L0: jax.Array | None
    oy_R0: jax.Array | None
    ox_L1: jax.Array | None
    ox_R1: jax.Array | None
    oy_L1: jax.Array | None
    oy_R1: jax.Array | None


def _fv_tp_2d_setup(cdgrid) -> _TpSetup:
    """Grid-derived, level-independent ``fv_tp_2d`` config (see :class:`_TpSetup`).

    Byte-identical to the inline setup that used to live at the top of
    ``fv_tp_2d`` (the duogrid gate, ``bounded_domain`` flag, and the
    ``halo_interp_offsets_h2`` extraction)."""
    n = cdgrid.n
    grid = cdgrid.base
    area = grid.area
    offsets_h2 = grid.halo_interp_offsets_h2

    # bounded_domain = (regional | nested | duogrid); duogrid also switches
    # pad_halo from interp_offsets mode to full duogrid mode.
    dg = grid.duogrid
    use_duogrid = dg is not None and dg.ng >= 2
    halo_offsets = None if use_duogrid else offsets_h2
    halo_dg = dg if use_duogrid else None
    bounded_domain = bool(grid.bounded_domain)

    # #811 MPI face-scatter: ``halo_interp_offsets_h2`` is KEPT FULL ``(6, ...)``
    # by the scatter (it is GLOBAL-face-indexed — ``pad_halo_mpi`` /
    # ``pad_halo_mpi_4d`` index it by global face id inside their per-owned-face
    # loop, so ``halo_offsets`` above MUST stay full).  But the PPM boundary
    # offsets ``ox_*``/``oy_*`` below feed the PURE-LOCAL, vectorised-over-all-
    # faces ``_ppm_1d`` / ``_correct_dm``, which multiply SAME-FACE PPM slopes
    # (sliced to the rank's OWNED faces under scatter) by these offsets — so they
    # must carry the owned faces ONLY, else a ``(n_owned, n) x (6, n)`` broadcast
    # error (the #811 blocker: the flux-form moisture substep is the first
    # transport run under face-scatter on the non-duogrid grid).  Detect scatter
    # by the grid's own (already-sliced) face count and slice the offsets to the
    # owned GLOBAL faces; a no-op on single-rank / replicated (face count == 6),
    # preserving bit-identity there.
    ppm_offsets_h2 = offsets_h2
    if offsets_h2 is not None and area.shape[0] != offsets_h2.shape[0]:
        from legoesm.grids.halo import get_mpi_topology
        topo = get_mpi_topology()
        owned = (getattr(topo, "local_face_ids", None)
                 if topo is not None else None)
        if owned is None:
            raise RuntimeError(
                "fv_tp_2d setup: grid is face-sliced "
                f"(faces={area.shape[0]}) but halo_interp_offsets_h2 is full "
                f"(faces={offsets_h2.shape[0]}) with no active MPI topology to "
                "identify the owned faces — cannot align the PPM boundary "
                "offsets to the transported data (#811).")
        ppm_offsets_h2 = offsets_h2[jnp.asarray(list(owned), dtype=jnp.int32)]

    if ppm_offsets_h2 is not None:
        # offsets_h2: (n_faces, 4, 2, n) — [face, edge, depth, cell]; WEST=0
        # EAST=1 SOUTH=2 NORTH=3; depth 0 = adjacent to interior.  n_faces is the
        # rank's owned-face count under scatter (== 6 single-rank/replicated).
        ox_L0 = ppm_offsets_h2[:, 0, 0, :]
        ox_R0 = ppm_offsets_h2[:, 1, 0, :]
        oy_L0 = ppm_offsets_h2[:, 2, 0, :]
        oy_R0 = ppm_offsets_h2[:, 3, 0, :]
        ox_L1 = ppm_offsets_h2[:, 0, 1, :]
        ox_R1 = ppm_offsets_h2[:, 1, 1, :]
        oy_L1 = ppm_offsets_h2[:, 2, 1, :]
        oy_R1 = ppm_offsets_h2[:, 3, 1, :]
    else:
        # Regional/nested: offsets unused (the bounded_domain gate in _ppm_1d
        # short-circuits the offset path); None so any accidental use raises.
        ox_L0 = ox_R0 = oy_L0 = oy_R0 = None
        ox_L1 = ox_R1 = oy_L1 = oy_R1 = None

    return _TpSetup(n, area, use_duogrid, halo_offsets, halo_dg, bounded_domain,
                    ox_L0, ox_R0, oy_L0, oy_R0, ox_L1, ox_R1, oy_L1, oy_R1)


def _fv_tp_2d_sweep1(q_full, q, crx, cry, xfx, yfx, ra_x, ra_y, s: _TpSetup,
                     hord, apply_fortran_xppm_boundary):
    """PURE-LOCAL first pass of ``fv_tp_2d``: the Y- and X-sweeps on the
    PRE-HALOED ``q_full`` → the intermediate ``q_i``, ``q_j`` (+ the ``fx2``,
    ``fy2`` reused by :func:`_fv_tp_2d_sweep2`).  No ``pad_halo`` — safe to
    ``vmap`` over levels on a pre-4D-haloed ``q_full`` (#811)."""
    fy2 = _yppm(q_full[:, 2:-2, :], cry, s.n, s.oy_L0, s.oy_R0, s.oy_L1, s.oy_R1,
                use_duogrid=s.use_duogrid,
                apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
                bounded_domain=s.bounded_domain, hord=hord)
    fyy = yfx * fy2
    q_i = (q * s.area + fyy[:, :, :-1] - fyy[:, :, 1:]) / ra_y

    fx2 = _xppm(q_full[:, :, 2:-2], crx, s.n, s.ox_L0, s.ox_R0, s.ox_L1, s.ox_R1,
                use_duogrid=s.use_duogrid,
                apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
                bounded_domain=s.bounded_domain, hord=hord)
    fxx = xfx * fx2
    q_j = (q * s.area + fxx[:, :-1, :] - fxx[:, 1:, :]) / ra_x
    return q_i, q_j, fx2, fy2


def _fv_tp_2d_sweep2(q_i_pad, q_j_pad, fx2, fy2, crx, cry, xfx, yfx, s: _TpSetup,
                     hord, apply_fortran_xppm_boundary):
    """PURE-LOCAL second pass of ``fv_tp_2d``: the cross-sweeps on the PRE-HALOED
    ``q_i_pad`` / ``q_j_pad`` and the flux combine ``0.5*(fx1+fx2)*xfx``.  The
    ``mass`` branch of the original was identical to the ``else`` (both
    ``*xfx``/``*yfx``), so it collapses here.  No ``pad_halo`` — vmap-safe."""
    fx1 = _xppm(q_i_pad[:, :, 2:-2], crx, s.n, s.ox_L0, s.ox_R0, s.ox_L1, s.ox_R1,
                use_duogrid=s.use_duogrid,
                apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
                bounded_domain=s.bounded_domain, hord=hord)
    fy1 = _yppm(q_j_pad[:, 2:-2, :], cry, s.n, s.oy_L0, s.oy_R0, s.oy_L1, s.oy_R1,
                use_duogrid=s.use_duogrid,
                apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
                bounded_domain=s.bounded_domain, hord=hord)
    fx = 0.5 * (fx1 + fx2) * xfx
    fy = 0.5 * (fy1 + fy2) * yfx
    return fx, fy


def fv_tp_2d(q, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
             nord=None, damp_c=None, mass=None,
             apply_cgrid_flux_sync=True,
             apply_fortran_xppm_boundary=False,
             hord: int = 12):
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
    # Level-independent setup (grid geometry + PPM boundary offsets) and the two
    # PURE-LOCAL sweeps are factored into shared helpers so the 4D-halo transport
    # (#811) can reuse them bit-for-bit with ONE pad_halo_4d per exchange.
    s = _fv_tp_2d_setup(cdgrid)

    # Exchange 1: the transported field.
    q_full = pad_halo(q, halo=2, interp_offsets=s.halo_offsets, duogrid=s.halo_dg)
    q_i, q_j, fx2, fy2 = _fv_tp_2d_sweep1(
        q_full, q, crx, cry, xfx, yfx, ra_x, ra_y, s,
        hord, apply_fortran_xppm_boundary)

    # Exchange 2: pack q_i and q_j into a single halo=2 exchange (mass
    # conservation requires both halos filled before the cross-sweeps).
    q_i_pad, q_j_pad = pad_halo_pair_h2(
        q_i, q_j, interp_offsets=s.halo_offsets, duogrid=s.halo_dg,
    )
    fx, fy = _fv_tp_2d_sweep2(
        q_i_pad, q_j_pad, fx2, fy2, crx, cry, xfx, yfx, s,
        hord, apply_fortran_xppm_boundary)

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
        fx, fy = synchronize_cgrid_fluxes(fx, fy, s.n)

    return fx, fy


def transport_step(h, ut, vt, dt, cdgrid, mass_target=None,
                   nord=None, damp_c=None,
                   apply_fortran_xppm_boundary=False,
                   hord: int = 12, **_kwargs):
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
                          apply_fortran_xppm_boundary),
                      hord=hord)
    return _finalize_transport(h, fx, fy, area, mass_target)


def _finalize_transport(h, fx, fy, area, mass_target):
    """fp64 flux closure + optional mass-conserving rescale (shared).

    new_test_dycores iter-3: fp64 flux differencing.  Without this promotion
    the cube transport accrues ~4.66e-10 cancellation noise per step from the
    fp32 ``fx[:-1] - fx[1:]`` subtraction across ~6·N² cells.  Promotion
    preserves bit-clean flux closure; output cast back to input dtype.
    """
    from legoesm.core.conservation import conservation_accumulator
    _acc = conservation_accumulator()
    h64 = h.astype(_acc)
    fx64 = fx.astype(_acc)
    fy64 = fy.astype(_acc)
    area64 = area.astype(_acc)
    h_new = (h64 + (fx64[:, :-1, :] - fx64[:, 1:, :]
                    + fy64[:, :, :-1] - fy64[:, :, 1:]) / area64
             ).astype(h.dtype)

    # Mass conservation fixer: clip negatives and rescale positives to conserve
    # total mass.  iter-6: fp64 budget accumulator (matches SW model fixer).
    if mass_target is not None:
        h_pos = jnp.maximum(h_new, 0.0)
        mass_pos = jnp.sum(h_pos.astype(_acc) * area.astype(_acc))
        scale = mass_target / jnp.maximum(mass_pos, 1.0)
        h_new = h_pos * scale.astype(h_pos.dtype)

    return h_new


def transport_step_4d(h_4d, ut_4d, vt_4d, dt, cdgrid, hord: int = 12):
    """4D-halo (all-levels-one-message) MPI-aware ``transport_step`` (#811).

    The per-level ``jax.vmap(transport_step)`` used by the flux-form moisture
    substep makes ``fv_tp_2d``'s internal ``pad_halo`` a *vmapped* cross-face
    ``sendrecv`` — a ``batch_axes`` failure under MPI face-scatter (CLAUDE.md:
    never ``vmap(pad_halo)``).  This routes the TWO field halos through
    ``pad_halo_4d`` (ONE message for all levels each) and ``vmap``s only the
    PURE-LOCAL PPM sweeps (``_fv_tp_2d_sweep1``/``_fv_tp_2d_sweep2`` — the SAME
    helpers ``fv_tp_2d`` uses, no numeric duplication).

    ``compute_transport_quantities`` is ``vmap``ped UNCHANGED and is MPI-safe:
    its only ``pad_halo`` calls are on ``cdgrid`` grid constants (``rdxa`` /
    ``sin_sg``), which under ``vmap`` are captured (batch-dim ``not_mapped``) →
    a SINGLE unbatched ``pad_halo`` per constant, never vmapped.

    Raw closure only (``mass_target=None``): the moisture substep does its mass
    conservation in the caller's batched, allreduce-aware ``_conserving_rescale``
    (#811 first half).  BIT-IDENTICAL to
    ``jax.vmap(transport_step(..., mass_target=None))`` on single-rank
    (``pad_halo_4d`` local == per-level ``pad_halo`` stacked; a halo is a pure
    index gather, no reduction → no fp-associativity change) — locked by
    ``test_transport_step_4d_matches_vmap``.

    NON-duogrid only: ``fv_tp_2d``'s duogrid ``synchronize_cgrid_fluxes`` /
    del-n damping are not on the flux-form path and are not 4D-ified — a duogrid
    grid raises here.

    Parameters
    ----------
    h_4d : (6, n, n, nlev) — transported field (level trailing).
    ut_4d, vt_4d : (6, n+1, n, nlev) / (6, n, n+1, nlev) — contravariant winds.

    Returns
    -------
    (6, n, n, nlev) — the raw flux-form transported field.
    """
    s = _fv_tp_2d_setup(cdgrid)
    if s.use_duogrid:
        raise NotImplementedError(
            "transport_step_4d does not support duogrid grids (the cgrid "
            "flux-sync + duogrid PPM boundary path is not 4D-ified); the "
            "flux-form moisture substep runs on the non-duogrid grid (#811).")

    area = cdgrid.base.area

    # Per-level Courant numbers / area fluxes.  vmap is MPI-safe (grid-constant
    # pads are captured => a single unbatched pad_halo each).
    crx, cry, xfx, yfx, ra_x, ra_y = jax.vmap(
        lambda utk, vtk: compute_transport_quantities(utk, vtk, dt, cdgrid),
        in_axes=(-1, -1), out_axes=-1)(ut_4d, vt_4d)

    # Exchange 1 (field) — one message for all levels — then the pure-local
    # first sweeps per level.
    q_full = pad_halo_4d(h_4d, halo=2, interp_offsets=s.halo_offsets,
                         duogrid=s.halo_dg)
    q_i, q_j, fx2, fy2 = jax.vmap(
        lambda qf, qk, cx, cy, xf, yf, rx, ry: _fv_tp_2d_sweep1(
            qf, qk, cx, cy, xf, yf, rx, ry, s, hord, False),
        in_axes=(-1, -1, -1, -1, -1, -1, -1, -1), out_axes=-1)(
        q_full, h_4d, crx, cry, xfx, yfx, ra_x, ra_y)

    # Exchange 2 (the q_i / q_j pair — two pad_halo_4d == pad_halo_pair_h2's two
    # sequential calls with identical arithmetic) — then the cross-sweeps.
    q_i_pad = pad_halo_4d(q_i, halo=2, interp_offsets=s.halo_offsets,
                          duogrid=s.halo_dg)
    q_j_pad = pad_halo_4d(q_j, halo=2, interp_offsets=s.halo_offsets,
                          duogrid=s.halo_dg)
    fx, fy = jax.vmap(
        lambda qip, qjp, f2, g2, cx, cy, xf, yf: _fv_tp_2d_sweep2(
            qip, qjp, f2, g2, cx, cy, xf, yf, s, hord, False),
        in_axes=(-1, -1, -1, -1, -1, -1, -1, -1), out_axes=-1)(
        q_i_pad, q_j_pad, fx2, fy2, crx, cry, xfx, yfx)

    # Raw fp64 flux closure (mass_target=None), per level — reuses the shared
    # _finalize_transport (no numeric duplication).
    return jax.vmap(
        lambda hk, fxk, fyk: _finalize_transport(hk, fxk, fyk, area, None),
        in_axes=(-1, -1, -1), out_axes=-1)(h_4d, fx, fy)


def streamfunction_mass_fluxes(cdgrid, psi_corner, dt):
    """Discretely divergence-free C-grid mass fluxes from a corner
    streamfunction (FV3 ``test_cases.F90`` wind_field=0 construction).

    The transport mass flux is the discrete curl of ``psi`` evaluated at cube
    corners::

        xfx[i,j] = dt*(psi[i,j] - psi[i,j+1])     # u-face flux  (6,n+1,n)
        yfx[i,j] = dt*(psi[i+1,j] - psi[i,j])     # v-face flux  (6,n,n+1)

    so the discrete divergence ``xfx[i]-xfx[i+1] + yfx[j]-yfx[j+1]``
    telescopes to ZERO for ANY ``psi`` — i.e. free-stream is preserved to
    machine precision.  This bypasses the contravariant ``d2a2c`` flux
    reconstruction whose face-boundary seam injects spurious area-flux
    divergence (issue 504: that seam fragments the corner-crossing bell).

    Parameters
    ----------
    cdgrid : CubedSphereCDGrid
    psi_corner : (6, n+1, n+1) — velocity streamfunction [m^2/s] at cube corners
    dt : float — transport timestep [s]

    Returns
    -------
    crx, cry, xfx, yfx, ra_x, ra_y : ready for :func:`fv_tp_2d`.
    """
    # Sign: the FV update is h += (xfx[i]-xfx[i+1])/area, so xfx is the flux
    # F_x with F[i]-F[i+1] = inflow.  For a streamfunction the +x mass flux is
    # F_x = -dpsi/dy ~ psi[i,j]-psi[i,j+1] (and F_y = +dpsi/dx ~
    # psi[i+1,j]-psi[i,j]).  Negating BOTH preserves the telescoping (free-
    # stream) and orients the flow to match the physical wind / the exact
    # solution (the opposite sign advects the bell BACKWARDS — caught by the
    # 1-day cross-grid L2, invisible to the full-revolution metric).
    area = cdgrid.base.area
    xfx = dt * (psi_corner[:, :, :-1] - psi_corner[:, :, 1:])   # (6,n+1,n)
    yfx = dt * (psi_corner[:, 1:, :] - psi_corner[:, :-1, :])   # (6,n,n+1)
    # Courant numbers ~ swept-area / local cell area (sign carries direction).
    ap = pad_halo(area, halo=1)                                 # (6,n+2,n+2)
    area_u = 0.5 * (ap[:, :-1, 1:-1] + ap[:, 1:, 1:-1])         # (6,n+1,n)
    area_v = 0.5 * (ap[:, 1:-1, :-1] + ap[:, 1:-1, 1:])         # (6,n,n+1)
    crx = xfx / area_u
    cry = yfx / area_v
    ra_x = area + xfx[:, :-1, :] - xfx[:, 1:, :]
    ra_y = area + yfx[:, :, :-1] - yfx[:, :, 1:]
    return crx, cry, xfx, yfx, ra_x, ra_y


def streamfunction_transport_step(h, fluxes, cdgrid, mass_target=None,
                                  hord: int = 10,
                                  apply_fortran_xppm_boundary: bool = True):
    """One free-stream-preserving transport step from prescribed divergence-
    free mass ``fluxes`` (see :func:`streamfunction_mass_fluxes`).

    Reuses the SAME :func:`fv_tp_2d` PPM operator and flux closure as
    :func:`transport_step`; only the flux *source* differs.
    """
    crx, cry, xfx, yfx, ra_x, ra_y = fluxes
    fx, fy = fv_tp_2d(h, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid, hord=hord,
                      apply_fortran_xppm_boundary=apply_fortran_xppm_boundary)
    return _finalize_transport(h, fx, fy, cdgrid.base.area, mass_target)
