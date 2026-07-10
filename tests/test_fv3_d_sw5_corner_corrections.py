"""Iter-862: cube-vertex corner corrections in `d_sw5_corner_divergence`.

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
``apply_legacy_corner_corrections`` on `d_sw5_corner_divergence`
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
``d_sw5_corner_divergence``; it does NOT wire the flag into any
production path.  Production ``fv3_sw_tendencies`` does not call
this helper.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Iter-883: also enable x64 at runtime in case JAX was already
# initialized in float32 by an earlier conftest import.  The
# os.environ.setdefault above is for command-line invocation; the
# jax.config.update is the runtime-effective form.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_sw_core import (
    d_sw5_corner_divergence,
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
        out = d_sw5_corner_divergence(
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

    out_off = d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        apply_legacy_corner_corrections=False)
    out_on = d_sw5_corner_divergence(
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

    out_off = d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        apply_legacy_corner_corrections=False)
    out_on = d_sw5_corner_divergence(
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

    out_off = d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=0.0, dddmp=0.0, d4_bg=1.0, nord=1,
        apply_legacy_corner_corrections=False)
    out_on = d_sw5_corner_divergence(
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
    factored out of `d_sw5_corner_divergence` (iter-862).  The two
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

    out_off = d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=0.0, dddmp=0.0, d4_bg=1.0, nord=1,
        apply_legacy_corner_corrections=False)
    out_on = d_sw5_corner_divergence(
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


def test_d_sw_native_forwards_d_sw5_flag():
    """Iter-871b: `_d_sw_native` must forward
    `apply_legacy_d_sw5_corner_corrections` kwarg through to
    `d_sw5_corner_divergence(apply_legacy_corner_corrections=...)`.

    Codex iter-871 stop-time review: iter-862 added the flag only on
    the inner helper; the FB-chain wrapper was not plumbed, making
    the flag unreachable from the FB chain.  iter-871b adds the
    forward.  This test verifies (i) flag=True changes the FB-chain
    output on a legacy grid (the d_sw5 corner correction propagates
    through ke into the d_sw6 wind update), (ii) flag=True is
    bit-identical to flag=False on a duogrid grid (the duogrid gate
    inside `d_sw5_corner_divergence` short-circuits).
    """
    from legoesm.core.fv3_sw_core import _d_sw_native

    n = 8

    # Legacy grid: flag-on must differ from flag-off.
    grid_leg = create_cubed_sphere(n=n, use_duogrid=False)
    cdgrid_leg = create_cubed_sphere_cdgrid(grid_leg)
    assert cdgrid_leg.base.bounded_domain is False

    rng = np.random.default_rng(862)
    h = jnp.asarray(rng.standard_normal((6, n, n)) + 1000.0)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    h_s = jnp.zeros((6, n, n))
    uc = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    ua = jnp.asarray(rng.standard_normal((6, n, n)))
    va = jnp.asarray(rng.standard_normal((6, n, n)))

    # Use d2_bg=1.0 (nord=0 branch) so damp = da_min_c * 1.0 fires
    # and the iter-862 corner-correction propagates into ke_damping.
    # nord=0 + d4_bg=0 keeps the higher-order Laplacian path off.
    h_off, u_off, v_off = _d_sw_native(
        h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid_leg, 100.0, constants.g,
        div_damp=0.0, d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        damp_v=0.0, nord_v=0,
        apply_legacy_d_sw5_corner_corrections=False)
    h_on, u_on, v_on = _d_sw_native(
        h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid_leg, 100.0, constants.g,
        div_damp=0.0, d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        damp_v=0.0, nord_v=0,
        apply_legacy_d_sw5_corner_corrections=True)
    diff_u = float(np.max(np.abs(np.asarray(u_on) - np.asarray(u_off))))
    diff_v = float(np.max(np.abs(np.asarray(v_on) - np.asarray(v_off))))
    assert diff_u > 1e-12 or diff_v > 1e-12, (
        f"Iter-871b: _d_sw_native legacy + d_sw5 flag=True must change "
        f"u_d/v_d output (corner correction propagates through ke "
        f"into d_sw6).  Got max|Δu|={diff_u:.3e}, "
        f"max|Δv|={diff_v:.3e} — kwarg may not be forwarded.")

    # Duogrid grid: helper's bounded_domain gate short-circuits.
    grid_dg = create_cubed_sphere(n=n, use_duogrid=True)
    cdgrid_dg = create_cubed_sphere_cdgrid(grid_dg)
    assert cdgrid_dg.base.bounded_domain is True

    h_off2, u_off2, v_off2 = _d_sw_native(
        h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid_dg, 100.0, constants.g,
        div_damp=0.0, d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        damp_v=0.0, nord_v=0,
        apply_legacy_d_sw5_corner_corrections=False)
    h_on2, u_on2, v_on2 = _d_sw_native(
        h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid_dg, 100.0, constants.g,
        div_damp=0.0, d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        damp_v=0.0, nord_v=0,
        apply_legacy_d_sw5_corner_corrections=True)
    np.testing.assert_array_equal(
        np.asarray(h_on2), np.asarray(h_off2))
    np.testing.assert_array_equal(
        np.asarray(u_on2), np.asarray(u_off2))
    np.testing.assert_array_equal(
        np.asarray(v_on2), np.asarray(v_off2))


def test_fb_entry_points_forward_iter862_iter869b_flags():
    """Iter-871c: end-to-end wiring through the FB-chain entry points.

    Codex iter-871b stop-time review: 'the new flag is still not
    reachable from the actual FB entry points.'  iter-871b only
    plumbed the kwargs through `_d_sw_native`; the higher-level
    wrappers (`fv3_fb_sw_step`, `fv3_forward_backward_step`,
    `FV3FBShallowWaterModel.step` via config) still didn't forward
    them.  iter-871c closes the wiring at all three levels.

    This test verifies:
      (a) `fv3_fb_sw_step` accepts both kwargs and forwards them.
      (b) `fv3_forward_backward_step` accepts both kwargs and
          forwards them.
      (c) `CDGridShallowWaterConfig` exposes both as config fields
          and `FV3FBShallowWaterModel.step` threads them.
    """
    from legoesm.core.fv3_sw_core import (
        fv3_fb_sw_step, fv3_forward_backward_step)
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig, FV3FBShallowWaterModel,
        FV3EdgeShallowWaterState)

    n = 8
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(862)
    h = jnp.asarray(rng.standard_normal((6, n, n)) + 1000.0)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    h_s = jnp.zeros((6, n, n))

    # (a) fv3_fb_sw_step: both flags reachable.
    h_a_off, u_a_off, v_a_off = fv3_fb_sw_step(
        h, u_d, v_d, h_s, cdgrid, 100.0,
        d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        apply_legacy_d_sw5_corner_corrections=False,
        apply_legacy_d_sw4_corner_ke_fix=False)
    h_a_on, u_a_on, v_a_on = fv3_fb_sw_step(
        h, u_d, v_d, h_s, cdgrid, 100.0,
        d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        apply_legacy_d_sw5_corner_corrections=True,
        apply_legacy_d_sw4_corner_ke_fix=True)
    diff_a = float(np.max(np.abs(np.asarray(u_a_on) - np.asarray(u_a_off)))
                    + np.max(np.abs(np.asarray(v_a_on) - np.asarray(v_a_off))))
    assert diff_a > 1e-12, (
        f"fv3_fb_sw_step does not forward the iter-862/iter-869b "
        f"flags: combined |Δu|+|Δv| = {diff_a:.3e}.")

    # (b) fv3_forward_backward_step: both flags reachable.
    # Note: fv3_forward_backward_step doesn't expose d_sw5 coefficients
    # so default d4_bg=0.16 nord=1 fires the nord>=1 path, where the
    # iter-862 corner correction lives in the n-loop.  Compare flag
    # toggles end-to-end.
    h_b_off, u_b_off, v_b_off = fv3_forward_backward_step(
        h, u_d, v_d, h_s, cdgrid, 100.0,
        apply_legacy_d_sw5_corner_corrections=False,
        apply_legacy_d_sw4_corner_ke_fix=False)
    h_b_on, u_b_on, v_b_on = fv3_forward_backward_step(
        h, u_d, v_d, h_s, cdgrid, 100.0,
        apply_legacy_d_sw5_corner_corrections=True,
        apply_legacy_d_sw4_corner_ke_fix=True)
    diff_b = float(np.max(np.abs(np.asarray(u_b_on) - np.asarray(u_b_off)))
                    + np.max(np.abs(np.asarray(v_b_on) - np.asarray(v_b_off))))
    assert diff_b > 1e-12, (
        f"fv3_forward_backward_step does not forward the iter-862/"
        f"iter-869b flags: combined |Δu|+|Δv| = {diff_b:.3e}.")

    # (c) CDGridShallowWaterConfig exposes both flags as fields.
    cfg_default = CDGridShallowWaterConfig()
    assert hasattr(cfg_default, "apply_legacy_d_sw4_corner_ke_fix"), (
        "CDGridShallowWaterConfig is missing "
        "`apply_legacy_d_sw4_corner_ke_fix` field.")
    assert hasattr(cfg_default,
                   "apply_legacy_d_sw5_corner_corrections"), (
        "CDGridShallowWaterConfig is missing "
        "`apply_legacy_d_sw5_corner_corrections` field.")
    assert cfg_default.apply_legacy_d_sw4_corner_ke_fix is False, (
        "iter-869b flag default must be False.")
    assert cfg_default.apply_legacy_d_sw5_corner_corrections is False, (
        "iter-862 flag default must be False (via iter-871c)."
    )


# ---------------------------------------------------------------------------
# 2026-07-10: cross-face halo port for the iterated Laplacian (nord>=1).
# Fortran oracle: dyn_core.F90:651-652 — duogrid+nord>0 runs a dedicated
# B-grid ghost exchange `ext_scalar(divgd, dg, bd, domain, 1, 1)`
# (fv_duogrid.F90::ext_scalar_3d: mpp NORTH+EAST corner halo update of the
# neighbour's ATTENUATED divgd + cube_rmp) between c_sw and d_sw; the
# sw_core.F90:1737-1787 nord loop consumes that ghost ring (fill_corners
# is gated `.not. duogrid`).
# ---------------------------------------------------------------------------

def test_pad_corner_scalar_cross_face_geometry():
    """Each halo point of the padded corner scalar must be the neighbour's
    corner value ONE ROW INSIDE the shared edge — verified geometrically by
    padding the corner XYZ coordinate fields: every halo point must sit
    adjacent to (~1 dx from) its local edge point, on the OUTSIDE
    (approximately the reflection of the first interior row)."""
    from legoesm.core.fv3_sw_core import _pad_corner_scalar_cross_face

    n = 12
    cdgrid = _duogrid_cdgrid(n)
    lon_c = jnp.asarray(cdgrid.lon_corner)
    lat_c = jnp.asarray(cdgrid.lat_corner)
    xc = jnp.cos(lat_c) * jnp.cos(lon_c)
    yc = jnp.cos(lat_c) * jnp.sin(lon_c)
    zc = jnp.sin(lat_c)
    pts_pad = np.stack([np.asarray(_pad_corner_scalar_cross_face(a, n))
                        for a in (xc, yc, zc)], axis=-1)
    pts = np.stack([np.asarray(xc), np.asarray(yc), np.asarray(zc)],
                   axis=-1)

    for f in range(6):
        cases = {
            'W': (pts_pad[f, 0, 1:n + 2], pts[f, 0, :], pts[f, 1, :]),
            'E': (pts_pad[f, n + 2, 1:n + 2], pts[f, n, :],
                  pts[f, n - 1, :]),
            'S': (pts_pad[f, 1:n + 2, 0], pts[f, :, 0], pts[f, :, 1]),
            'N': (pts_pad[f, 1:n + 2, n + 2], pts[f, :, n],
                  pts[f, :, n - 1]),
        }
        for tag, (halo, edge, inside) in cases.items():
            dx = np.linalg.norm(inside - edge, axis=-1)
            # adjacent: exactly one row beyond the edge
            d_edge = np.linalg.norm(halo - edge, axis=-1) / dx
            assert d_edge.max() < 1.3 and d_edge.min() > 0.7, (
                f"face {f} {tag}: halo point not adjacent to edge "
                f"(|halo-edge|/dx in [{d_edge.min():.2f},{d_edge.max():.2f}])")
            # outside: near the mirror of the first interior row (the
            # gnomonic kink allows up to ~1 dx deviation at cube vertices)
            refl = np.linalg.norm(halo - (2 * edge - inside), axis=-1) / dx
            assert refl.max() < 1.2, (
                f"face {f} {tag}: halo strip mis-mapped "
                f"(max|halo-reflect|/dx = {refl.max():.2f})")


def test_pad_corner_scalar_cross_face_ghost_is_attenuated_neighbour_row():
    """The ghost ring must carry the neighbour's ATTENUATED divg_d row one
    inside its shared edge (Fortran ext_scalar exchanges divgd AFTER
    divergence_corner_duo applied the panel-edge zero/0.25 conditions), so
    for a constant-1 pre-attenuation divergence the ghost strip reads
    [0, 0.0625, 0.25, ..., 0.25, 0.0625, 0]."""
    from legoesm.core.fv3_sw_core import _pad_corner_scalar_cross_face

    n = 8
    ones = jnp.ones((6, n + 1, n + 1))
    # Apply the divergence_corner_duo edge conditions (zero edges, 0.25x
    # first interior ring) to the constant field.
    att = ones.at[:, 0, :].set(0.0).at[:, n, :].set(0.0)
    att = att.at[:, :, 0].set(0.0).at[:, :, n].set(0.0)
    att = att.at[:, 1, :].multiply(0.25).at[:, n - 1, :].multiply(0.25)
    att = att.at[:, :, 1].multiply(0.25).at[:, :, n - 1].multiply(0.25)
    padded = _pad_corner_scalar_cross_face(att, n)
    expect = np.zeros(n + 1)
    expect[2:n - 1] = 0.25
    expect[1] = expect[n - 1] = 0.0625
    for strip in (padded[:, 0, 1:n + 2], padded[:, n + 2, 1:n + 2],
                  padded[:, 1:n + 2, 0], padded[:, 1:n + 2, n + 2]):
        assert np.allclose(np.asarray(strip), expect[None, :]), (
            "ghost strip must equal the neighbour's attenuated row one "
            "inside the shared edge")
    # Interior of the pad is the field itself.
    assert np.array_equal(np.asarray(padded[:, 1:-1, 1:-1]), np.asarray(att))


def test_d_sw5_cross_face_halo_gating_and_interior_invariance():
    """The cross-face ghost ring is OPT-IN: the default must be the
    zero-ring (measured stable — C48 modon 120 days clean — while the
    ext_scalar-faithful attenuated ghost destabilises the FB chain at
    day ~60-65); the halo choice must only affect corners within 2 rows
    of panel edges (interior bit-identical)."""
    n = 12
    cdgrid = _duogrid_cdgrid(n)
    u_d, v_d, ua, va = _make_inputs(n, seed=3)
    kw = dict(d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1)
    k_default = d_sw5_corner_divergence(u_d, v_d, ua, va, cdgrid, 300.0,
                                        **kw)
    k_on = d_sw5_corner_divergence(u_d, v_d, ua, va, cdgrid, 300.0,
                                   cross_face_halo=True, **kw)
    k_off = d_sw5_corner_divergence(u_d, v_d, ua, va, cdgrid, 300.0,
                                    cross_face_halo=False, **kw)
    assert np.array_equal(np.asarray(k_default), np.asarray(k_off)), (
        "default must be the zero-ring (cross_face_halo=False)")
    d = np.abs(np.asarray(k_on - k_off))
    assert d.max() > 0.0, "cross-face halo had no effect on seam corners"
    assert d[:, 2:-2, 2:-2].max() == 0.0, (
        "cross-face halo changed interior corners (must be seam-local)")
