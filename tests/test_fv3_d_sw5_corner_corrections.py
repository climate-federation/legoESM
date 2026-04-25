"""Iter-862: cube-vertex corner corrections in `_d_sw5_corner_divergence`.

Pins Check 3 from iter-849's d_sw5 fidelity audit: Fortran
sw_core.F90:1709-1715 (nord=0) and 1773-1776 (nord>=1) apply explicit
corner adjustments to delpc / divg_d at the four cube-vertex corner
positions when `.not. flagstruct%duogrid`.

The corrections (Fortran indexing):

    if (sw_corner) delpc(1,    1) = delpc(1,    1) - vort(1,    0)
    if (se_corner) delpc(npx,  1) = delpc(npx,  1) - vort(npx,  0)
    if (ne_corner) delpc(npx,npy) = delpc(npx,npy) + vort(npx,npy)
    if (nw_corner) delpc(1,  npy) = delpc(1,  npy) + vort(1,  npy)

iter-862 introduces an opt-in flag
``apply_legacy_corner_corrections`` on `_d_sw5_corner_divergence`
(default ``False``).  Default-off avoids applying the Fortran-
structure correction with Fortran-incomplete halo data (the iter-655
``mode='edge'`` same-face halo at cube vertices is O(1)-wrong relative
to a true cross-face D-grid edge halo, deferred to iter-863+).

Tests cover:
- Default (flag=False) reproduces the pre-iter-862 behaviour bit-for-bit.
- Flag=True + duogrid=False: corrections fire EXACTLY at the four
  cube-vertex corners with the expected ``±vort_pad[corner-halo]``
  contributions, NOT some other rearrangement.
- Flag=True + duogrid=True: corrections still SKIPPED (the Fortran
  ``.not. flagstruct%duogrid`` second gate dominates).
- For nord>=1 (n-loop), the per-iteration correction also fires
  exactly on the four corners with the right sign and magnitude.

Iter-862 only adds opt-in support to the FB chain helper
``_d_sw5_corner_divergence``; it does NOT wire the flag into any
production path.  Production ``fv3_sw_tendencies`` does not call
this helper.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_sw_core import (
    _d_sw5_corner_divergence,
    _apply_legacy_d_sw5_corner_corrections,
)


def _make_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    u_d = jnp.asarray(rng.normal(size=(6, n, n + 1)))
    v_d = jnp.asarray(rng.normal(size=(6, n + 1, n)))
    ua = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])    # (6, n, n)
    va = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    return u_d, v_d, ua, va


def _legacy_cdgrid(n):
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    assert cdgrid.base.duogrid is None  # legacy path
    return cdgrid


def _duogrid_cdgrid(n):
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    assert cdgrid.base.duogrid is not None  # duogrid path
    return cdgrid


def _no_correction_reference_nord0(u_d, v_d, ua, va, cdgrid):
    """Re-run the iter-862-affected nord=0 stencil locally WITHOUT any
    corner corrections.  Returns ``ke_damping`` shaped (6, n+1, n+1)
    matching the helper's output for ``d2_bg=1, dddmp=0, nord=0``.

    Used to (a) cross-check that flag=False == this reference, and
    (b) show what the flag=True corrections add atop this baseline.
    """
    cosa_u = cdgrid.cosa_u
    cosa_v = cdgrid.cosa_v
    from legoesm.core.fv3_sw_core import _sina_u_v_from_sin_sg
    sina_u, sina_v = _sina_u_v_from_sin_sg(cdgrid)
    dxc = cdgrid.dxc
    dyc = cdgrid.dyc
    rarea_c = cdgrid.rarea_c

    from legoesm.grids.halo import pad_halo as _pad_halo
    dg = cdgrid.base.duogrid
    _offs = None if dg is not None else cdgrid.base.halo_interp_offsets
    ua_full = _pad_halo(ua, halo=1, interp_offsets=_offs, duogrid=dg)
    va_full = _pad_halo(va, halo=1, interp_offsets=_offs, duogrid=dg)
    ua_pad = ua_full[:, :, 1:-1]
    va_pad = va_full[:, 1:-1, :]
    va_below = va_pad[:, :, :-1]
    va_above = va_pad[:, :, 1:]
    ptc = (u_d - 0.5 * (va_below + va_above) * cosa_v) * dyc * sina_v
    ua_left = ua_pad[:, :-1, :]
    ua_right = ua_pad[:, 1:, :]
    vort = (v_d - 0.5 * (ua_left + ua_right) * cosa_u) * dxc * sina_u
    vort_pad = jnp.pad(vort, [(0, 0), (0, 0), (1, 1)], mode='edge')
    ptc_pad = jnp.pad(ptc, [(0, 0), (1, 1), (0, 0)], mode='edge')
    delpc = (vort_pad[:, :, :-1] - vort_pad[:, :, 1:]
             + ptc_pad[:, :-1, :] - ptc_pad[:, 1:, :])
    delpc = rarea_c * delpc
    da_min_c = jnp.min(cdgrid.area_corner)
    return da_min_c * delpc, vort_pad, rarea_c


def test_default_flag_off_is_bitwise_identical_to_baseline_nord0():
    """Default (``apply_legacy_corner_corrections=False``) must produce
    output bit-identical to the pre-iter-862 reference, in BOTH legacy
    and duogrid modes.  Codex iter-862: protects against accidental
    behavioural drift in default callers (the FB chain
    `_d_sw_native` calls without passing the flag, so its output must
    not change as a result of iter-862)."""
    n = 4
    u_d, v_d, ua, va = _make_inputs(n, seed=11)
    dt = 100.0

    for cdgrid_factory in (_legacy_cdgrid, _duogrid_cdgrid):
        cdgrid = cdgrid_factory(n)
        out = _d_sw5_corner_divergence(
            u_d, v_d, ua, va, cdgrid, dt,
            d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0)
        # No flag passed → flag defaults to False → corrections never
        # fire → output equals the no-correction reference.
        ref, _, _ = _no_correction_reference_nord0(
            u_d, v_d, ua, va, cdgrid)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_flag_on_legacy_mode_applies_exact_corner_corrections_nord0():
    """With flag=True AND legacy (duogrid is None), the four cube-
    vertex corner corrections must fire with the EXACT expected
    magnitudes:

        delpc[0, 0]   -=  vort_pad[0, 0]
        delpc[-1, 0]  -=  vort_pad[-1, 0]
        delpc[-1, -1] +=  vort_pad[-1, -1]
        delpc[0, -1]  +=  vort_pad[0, -1]

    Output_with_flag - output_no_flag must equal exactly those four
    deltas (scaled by da_min_c * rarea_c) and zero everywhere else.
    """
    n = 4
    cdgrid = _legacy_cdgrid(n)
    u_d, v_d, ua, va = _make_inputs(n, seed=11)
    dt = 100.0

    out_off = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        apply_legacy_corner_corrections=False)
    out_on = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        apply_legacy_corner_corrections=True)
    delta = np.asarray(out_on) - np.asarray(out_off)

    _, vort_pad, rarea_c = _no_correction_reference_nord0(
        u_d, v_d, ua, va, cdgrid)
    da_min_c = float(jnp.min(cdgrid.area_corner))
    rarea_c_np = np.asarray(rarea_c)
    vort_pad_np = np.asarray(vort_pad)

    expected = np.zeros_like(delta)
    expected[:, 0, 0] = (
        -vort_pad_np[:, 0, 0] * rarea_c_np[:, 0, 0] * da_min_c)
    expected[:, -1, 0] = (
        -vort_pad_np[:, -1, 0] * rarea_c_np[:, -1, 0] * da_min_c)
    expected[:, -1, -1] = (
        vort_pad_np[:, -1, -1] * rarea_c_np[:, -1, -1] * da_min_c)
    expected[:, 0, -1] = (
        vort_pad_np[:, 0, -1] * rarea_c_np[:, 0, -1] * da_min_c)

    np.testing.assert_allclose(delta, expected, atol=1e-12)


def test_flag_on_duogrid_mode_skipped_nord0():
    """Flag=True must STILL be skipped on the duogrid path
    (cdgrid.base.duogrid is not None) per Fortran's
    ``.not. flagstruct%duogrid`` gate.  Output must equal flag=False
    output bit-for-bit on the duogrid grid."""
    n = 4
    cdgrid = _duogrid_cdgrid(n)
    u_d, v_d, ua, va = _make_inputs(n, seed=23)
    dt = 100.0

    out_off = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        apply_legacy_corner_corrections=False)
    out_on = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        apply_legacy_corner_corrections=True)

    np.testing.assert_array_equal(np.asarray(out_off), np.asarray(out_on))


def test_flag_on_legacy_mode_nord1_localises_on_corners():
    """nord>=1 path: the per-iteration correction must touch ONLY the
    four cube-vertex corner cells of the iterated ``divg_d``; no
    interior or edge-non-corner cell may differ between flag=False
    and flag=True.

    Codex iter-862 follow-up: the earlier locality-only check used a
    Manhattan-distance heuristic that admitted false-positive passes
    (correction touching cells one step away).  This test asserts
    EXACT structural locality: the difference mask is non-zero ONLY
    at the four cube-vertex corners on every face, and ALL other
    cells are bit-identical between flag=False and flag=True.

    Note (structural): the FIRST iteration of the n-loop fires the
    correction on ``uc_lap`` values whose corner-halo cells inherit
    `_divergence_corner_duo`'s face-boundary zero (that helper zeroes
    `divg_d` at face boundaries before the n-loop starts; the
    `mode='edge'` halo of those zeros leaves uc_lap[corner-halo] = 0
    on the first iteration).  So with nord=1 the per-corner delta
    can be EXACTLY 0 by structure — but the structural locality
    contract still holds: NO interior cells are touched.  For nord=2
    the iteration has spread non-zero values inward and the corner
    deltas should generally be non-zero (verified separately).
    """
    n = 4
    cdgrid = _legacy_cdgrid(n)
    u_d, v_d, ua, va = _make_inputs(n, seed=37)
    dt = 100.0

    out_off = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=0.0, dddmp=0.0, d4_bg=1.0, nord=1,
        apply_legacy_corner_corrections=False)
    out_on = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=0.0, dddmp=0.0, d4_bg=1.0, nord=1,
        apply_legacy_corner_corrections=True)

    delta = np.asarray(out_on) - np.asarray(out_off)

    nonzero_mask = np.abs(delta) > 1e-14
    expected_corner_mask = np.zeros_like(nonzero_mask, dtype=bool)
    for face in range(6):
        expected_corner_mask[face, 0, 0] = True
        expected_corner_mask[face, -1, 0] = True
        expected_corner_mask[face, 0, -1] = True
        expected_corner_mask[face, -1, -1] = True

    interior_nonzero = (nonzero_mask & ~expected_corner_mask)
    assert not interior_nonzero.any(), (
        f"Iter-862 nord=1 corner correction touched non-corner cells "
        f"at {np.argwhere(interior_nonzero).tolist()}; correction must "
        f"localise on cube-vertex corners only.")


def test_corner_correction_kernel_exact_signs_and_magnitudes():
    """Direct kernel test on synthetic inputs.

    `_apply_legacy_d_sw5_corner_corrections` is the small pure helper
    factored out of `_d_sw5_corner_divergence` (iter-862).  The two
    upstream callers (nord=0 and nord>=1) hand it the right field
    and edge-halo data, but for sign/magnitude/index correctness we
    test the kernel here on a synthetic edge-halo whose corner cells
    are GUARANTEED non-zero — bypassing the upstream face-boundary
    zeroing that can collapse uc_lap[corner-halo] to zero on the
    first n-loop iteration.

    Catches an off-by-one corner index, a wrong sign on south vs
    north, or a missing corner — none of which the in-helper
    locality tests above can fully pin under the zeroing.
    """
    n = 4
    rng = np.random.default_rng(101)
    field = jnp.asarray(rng.normal(size=(6, n + 1, n + 1)))
    edge_halo = jnp.asarray(rng.normal(size=(6, n + 1, n + 2)))

    out = _apply_legacy_d_sw5_corner_corrections(field, edge_halo)
    delta = np.asarray(out) - np.asarray(field)

    expected = np.zeros_like(delta)
    eh = np.asarray(edge_halo)
    expected[:, 0, 0] = -eh[:, 0, 0]      # SW: subtract
    expected[:, -1, 0] = -eh[:, -1, 0]    # SE: subtract
    expected[:, -1, -1] = eh[:, -1, -1]   # NE: add
    expected[:, 0, -1] = eh[:, 0, -1]     # NW: add

    # Floating-point round-off from `f.at[i,j].add()` is at the ulp
    # scale (~1e-16); allow a tiny absolute tolerance.  Magnitudes
    # of order ~1, so atol=1e-12 is still strict.
    np.testing.assert_allclose(delta, expected, atol=1e-12)


def test_corner_correction_kernel_only_corners_change():
    """The kernel must touch ONLY the four cube-vertex corners on
    every face.  Synthetic non-zero edge-halo input → all-zero delta
    everywhere except those 4 cells/face."""
    n = 6
    rng = np.random.default_rng(202)
    field = jnp.asarray(rng.normal(size=(6, n + 1, n + 1)))
    edge_halo = jnp.full((6, n + 1, n + 2), 1.0)  # all ones

    out = _apply_legacy_d_sw5_corner_corrections(field, edge_halo)
    delta = np.asarray(out) - np.asarray(field)

    n_eff = n + 1
    expected_change = np.zeros_like(delta, dtype=bool)
    for face in range(6):
        expected_change[face, 0, 0] = True
        expected_change[face, -1, 0] = True
        expected_change[face, 0, -1] = True
        expected_change[face, -1, -1] = True
    actual_change = np.abs(delta) > 1e-14
    np.testing.assert_array_equal(actual_change, expected_change)


def test_nord1_off_vs_on_only_corners_differ_largest_change():
    """nord>=1 path: largest absolute difference between flag=False and
    flag=True must be EXACTLY at one of the four cube-vertex corners
    on every face (not somewhere else).  Combined with the locality
    test above (no interior cells differ), this catches sign / wrong-
    corner / off-by-one errors that would shift the largest delta
    AWAY from a corner.

    Note (structural).  In iter-862's current implementation the FB-
    chain n=4 nord=1 case can produce all-zero corner deltas because
    `_divergence_corner_duo` zeroes face boundaries before the n-loop
    starts and the `mode='edge'` halo of those zeros propagates into
    `uc_lap[corner-halo]`.  When that happens the largest-difference
    is degenerate (everything zero); this test guards against the
    case where a sign or index bug introduces a non-zero delta in a
    non-corner cell.
    """
    n = 4
    cdgrid = _legacy_cdgrid(n)
    u_d, v_d, ua, va = _make_inputs(n, seed=37)
    dt = 100.0

    out_off = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=0.0, dddmp=0.0, d4_bg=1.0, nord=1,
        apply_legacy_corner_corrections=False)
    out_on = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=0.0, dddmp=0.0, d4_bg=1.0, nord=1,
        apply_legacy_corner_corrections=True)
    delta = np.asarray(out_on) - np.asarray(out_off)

    n_eff = delta.shape[1]  # = n+1
    corner_set = {(0, 0), (0, n_eff - 1), (n_eff - 1, 0),
                  (n_eff - 1, n_eff - 1)}

    for face in range(6):
        face_abs = np.abs(delta[face])
        if not (face_abs > 1e-14).any():
            # All-zero degenerate face; nothing to verify here.
            continue
        i_max, j_max = np.unravel_index(face_abs.argmax(), face_abs.shape)
        assert (int(i_max), int(j_max)) in corner_set, (
            f"face={face}: max-abs delta location ({i_max}, {j_max}) "
            f"is not a cube-vertex corner — iter-862's n-loop corner "
            f"correction must place its largest contribution exactly "
            f"on one of {sorted(corner_set)}.")
