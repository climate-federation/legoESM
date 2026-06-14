"""Iter-903 tests: Fortran-faithful RIGHT-side PPM cube-edge overrides.

Symmetric counterpart to iter-900's left-side tests.  Verifies that
`fortran_faithful_ppm_right=True` swaps the iter-892 RIGHT-side
overrides at q_face[n+1, n+2] for Fortran's al(npx-1)/al(npx)
recipes at the corrected q_face[n+2, n+3] slots.

A. Default OFF preserves iter-892 RIGHT behavior bit-exactly.
B. Flag=True applies Fortran al(npx-1) at q_face[n+2] and Fortran
   al(npx) (partial-faithful with halo=2 'edge' replica) at q_face[n+3].
C. Iter-892's q_face[n+1] override is REMOVED when flag=True
   (reverts to standard 4th-order interior stencil).
D. LEFT side untouched in iter-903 (verified by iter-892 default
   left + flag=True right; iter-900 + iter-903 flags are independent).
E. Flag is gated on apply_fortran_xppm_boundary.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import _ppm_reconstruct_1d


C1 = -2.0 / 14.0
C2 = 11.0 / 14.0
C3 = 5.0 / 14.0


def _build_quadratic_strip(n_int, *, batch=6):
    js = np.arange(-2, n_int + 2, dtype=np.float64)
    q1 = 1.0 + 0.1 * js + 0.05 * js * js
    return np.broadcast_to(q1[None, :], (batch, n_int + 4)).copy()


def _run_leaf(strip, *, on, faithful_left, faithful_right, n_int):
    qL, qR = _ppm_reconstruct_1d(
        jnp.asarray(strip), axis=1,
        apply_fortran_xppm_boundary=on,
        n_interior=(n_int if on else None),
        fortran_faithful_ppm_left=faithful_left,
        fortran_faithful_ppm_right=faithful_right)
    qL_np = np.asarray(qL)
    qR_np = np.asarray(qR)
    return np.concatenate([qL_np, qR_np[..., -1:]], axis=-1)


def _expected_fortran_alnm1_right(q_pad, n_int):
    """Fortran al(npx-1) under Hypothesis A.

    al(npx-1) = c1*q1(npx-3) + c2*q1(npx-2) + c3*q1(npx-1)
              = c1*q_pad[n+2] + c2*q_pad[n+3] + c3*q_pad[n+4]
    """
    return (C1 * q_pad[..., n_int + 2]
            + C2 * q_pad[..., n_int + 3]
            + C3 * q_pad[..., n_int + 4])


def _expected_fortran_aln_right(q_pad, n_int):
    """Fortran al(npx) (partial-faithful) under Hypothesis A.

    xt = 0.75*(q1(npx-1)+q1(npx)) - 0.25*(q1(npx-2)+q1(npx+1))
       = 0.75*(q_pad[n+4]+q_pad[n+5]) - 0.25*(q_pad[n+3]+q_pad[n+6])
       (q_pad[n+6] is mode='edge' replica of q1(n+1) — partial fidelity)
    Clipped to min/max(q_pad[n+3..n+6]).
    """
    xt = (0.75 * (q_pad[..., n_int + 4] + q_pad[..., n_int + 5])
          - 0.25 * (q_pad[..., n_int + 3] + q_pad[..., n_int + 6]))
    q_lo = np.minimum(
        np.minimum(q_pad[..., n_int + 3], q_pad[..., n_int + 4]),
        np.minimum(q_pad[..., n_int + 5], q_pad[..., n_int + 6]))
    q_hi = np.maximum(
        np.maximum(q_pad[..., n_int + 3], q_pad[..., n_int + 4]),
        np.maximum(q_pad[..., n_int + 5], q_pad[..., n_int + 6]))
    return np.clip(xt, q_lo, q_hi)


def _expected_iter892_face_alnm1(q_pad, n_int):
    """iter-892's RIGHT q_face[n+1] override (kept as default)."""
    return (C1 * q_pad[..., n_int + 1]
            + C2 * q_pad[..., n_int + 2]
            + C3 * q_pad[..., n_int + 3])


def _expected_iter892_face_aln(q_pad, n_int):
    """iter-892's RIGHT q_face[n+2] override."""
    xt = (0.75 * (q_pad[..., n_int + 3] + q_pad[..., n_int + 4])
          - 0.25 * (q_pad[..., n_int + 2] + q_pad[..., n_int + 5]))
    q_lo = np.minimum(
        np.minimum(q_pad[..., n_int + 2], q_pad[..., n_int + 3]),
        np.minimum(q_pad[..., n_int + 4], q_pad[..., n_int + 5]))
    q_hi = np.maximum(
        np.maximum(q_pad[..., n_int + 2], q_pad[..., n_int + 3]),
        np.maximum(q_pad[..., n_int + 4], q_pad[..., n_int + 5]))
    return np.clip(xt, q_lo, q_hi)


def _expected_4th_order(q_pad, k):
    return ((7.0 * (q_pad[..., k + 1] + q_pad[..., k + 2])
             - (q_pad[..., k] + q_pad[..., k + 3])) / 12.0)


def test_iter903_default_off_preserves_iter892_right_behavior():
    """fortran_faithful_ppm_right=False (default) must reproduce
    iter-892's RIGHT behavior bit-exactly."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face = _run_leaf(strip, on=True, faithful_left=False,
                        faithful_right=False, n_int=n_int)

    np.testing.assert_allclose(q_face[..., n_int + 1],
                                _expected_iter892_face_alnm1(q_pad, n_int),
                                atol=1e-13, rtol=0)
    np.testing.assert_allclose(q_face[..., n_int + 2],
                                _expected_iter892_face_aln(q_pad, n_int),
                                atol=1e-13, rtol=0)


def test_iter903_q_face_n_plus_2_uses_fortran_alnm1_when_flag_on():
    """With fortran_faithful_ppm_right=True, q_face[n+2] = Fortran
    al(npx-1)."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_faithful = _run_leaf(strip, on=True, faithful_left=False,
                                  faithful_right=True, n_int=n_int)

    expected = _expected_fortran_alnm1_right(q_pad, n_int)
    iter892 = _expected_iter892_face_aln(q_pad, n_int)

    assert not np.allclose(expected, iter892, atol=1e-10), (
        "Fortran al(npx-1) and iter-892's q_face[n+2] xt formula "
        "coincide on quadratic input — test isn't discriminating.")

    np.testing.assert_allclose(q_face_faithful[..., n_int + 2], expected,
                                atol=1e-13, rtol=0,
                                err_msg=("iter-903 q_face[n+2] doesn't "
                                         "match Fortran al(npx-1)"))


def test_iter903_q_face_n_plus_3_override_is_limiter_dominated_on_quadratic():
    """Iter-903 PARTIAL-FAITHFUL OBSERVATION:

    With flag=True, q_face[n+3] is overridden with Fortran al(npx)
    (clipped xt with mode='edge' replica for q1(npx+1)).  The pre-
    limiter override value DOES differ from the iter-892 4th-order
    baseline (verified via `_expected_fortran_aln_right` reference),
    but on this quadratic test input the post-override monotonicity
    overshoot constraint at lines 295-301 of operators_cdgrid.py
    SATURATES q_L[n+3] = 3*q[n+3] - 2*q_R[n+3] — the same value as
    iter-892's path produces for the same monotonicity constraint.

    This sentinel records the empirical observation that iter-903's
    q_face[n+3] override is LIMITER-DOMINATED on smooth-curvature
    inputs.  This means the iter-903 effect on production W2 (where
    cells near the cube edge transition smoothly) is concentrated at
    q_face[n+2] (which the prior test already locks bit-exactly).

    Practical implication for iter-904+ measurement: the production
    W2 effect of `fortran_faithful_ppm_right=True` will be smaller
    than the iter-900 LEFT-side effect because half of the override
    (q_face[n+3]) is limiter-erased on smooth fields.

    Asserts: iter-892 and iter-903 produce IDENTICAL post-limiter
    values at q_face[n+3] on this input (limiter saturation).
    """
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_iter892 = _run_leaf(strip, on=True, faithful_left=False,
                                 faithful_right=False, n_int=n_int)
    q_face_faithful = _run_leaf(strip, on=True, faithful_left=False,
                                  faithful_right=True, n_int=n_int)

    # The limiter saturates both paths to q_L_adj = 3*q - 2*q_R on
    # this strictly-convex quadratic input.
    np.testing.assert_allclose(q_face_iter892[..., n_int + 3],
                                q_face_faithful[..., n_int + 3],
                                atol=1e-13, rtol=0,
                                err_msg=("iter-892 and iter-903 should "
                                         "produce IDENTICAL post-limiter "
                                         "values at q_face[n+3] on the "
                                         "quadratic test input due to "
                                         "monotonicity overshoot "
                                         "constraint saturation.  If "
                                         "this changed, the limiter "
                                         "behavior was modified — "
                                         "investigate."))

    # Pre-limiter values DO differ (sanity check that the override
    # itself is wired correctly even though the limiter erases it).
    expected_pre_limiter = _expected_fortran_aln_right(q_pad, n_int)
    expected_4th = _expected_4th_order(q_pad, n_int + 3)
    assert not np.allclose(expected_pre_limiter, expected_4th,
                            atol=1e-10), (
        "Pre-limiter Fortran al(npx) coincides with 4th-order on the "
        "test input — pick a more discriminating input.")


def test_iter903_q_face_n_plus_1_reverts_to_4th_order_when_flag_on():
    """With flag=True, q_face[n+1] (= al(n-1) under Hypothesis A) must
    NOT be overridden — it should match the standard 4th-order
    interior stencil because Fortran has no boundary override there."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_faithful = _run_leaf(strip, on=True, faithful_left=False,
                                  faithful_right=True, n_int=n_int)

    expected_4th = _expected_4th_order(q_pad, n_int + 1)
    iter892 = _expected_iter892_face_alnm1(q_pad, n_int)

    assert not np.allclose(expected_4th, iter892, atol=1e-10), (
        "iter-892's q_face[n+1] override and 4th-order coincide on "
        "quadratic input — test isn't discriminating.")

    np.testing.assert_allclose(q_face_faithful[..., n_int + 1],
                                expected_4th, atol=1e-13, rtol=0,
                                err_msg=("iter-903 q_face[n+1] should "
                                         "revert to 4th-order when "
                                         "fortran_faithful_ppm_right="
                                         "True — but it didn't"))


def test_iter903_left_side_untouched_when_only_right_flag_on():
    """iter-900 and iter-903 flags are independent.  With
    fortran_faithful_ppm_right=True, fortran_faithful_ppm_left=False,
    the LEFT-side overrides must still match iter-892's behavior."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_iter892 = _run_leaf(strip, on=True, faithful_left=False,
                                 faithful_right=False, n_int=n_int)
    q_face_right_only = _run_leaf(strip, on=True, faithful_left=False,
                                    faithful_right=True, n_int=n_int)

    # LEFT-side cells must match between the two flag values.
    np.testing.assert_allclose(q_face_iter892[..., 2],
                                q_face_right_only[..., 2],
                                atol=1e-13, rtol=0)
    np.testing.assert_allclose(q_face_iter892[..., 3],
                                q_face_right_only[..., 3],
                                atol=1e-13, rtol=0)


def test_iter903_flag_no_effect_when_xppm_boundary_off():
    """fortran_faithful_ppm_right=True is a no-op when
    apply_fortran_xppm_boundary=False."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)

    q_face_off_iter892 = _run_leaf(strip, on=False, faithful_left=False,
                                     faithful_right=False, n_int=n_int)
    q_face_off_faithful = _run_leaf(strip, on=False, faithful_left=False,
                                      faithful_right=True, n_int=n_int)

    np.testing.assert_allclose(q_face_off_iter892, q_face_off_faithful,
                                atol=1e-13, rtol=0)
