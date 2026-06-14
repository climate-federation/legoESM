"""Iter-900 tests: Fortran-faithful LEFT-side PPM cube-edge overrides.

iter-899 surfaced a 1-cell shift bug in iter-892's overrides: under
production strip layout (Hypothesis A: q[k]=q1(k-2)) iter-892's xt
formula at q_face[2] computes a non-Fortran value at the al(0) slot,
and its c3/c2/c1 formula at q_face[3] computes a non-Fortran value at
the al(1) slot.

iter-900 adds a feature flag `fortran_faithful_ppm_left` (default
False) that swaps the LEFT-side overrides to Fortran's actual recipes
at the corrected indices:

  q_face[2] = al(0) = c1*q1(-2) + c2*q1(-1) + c3*q1(0)
            = c1*q_pad[2] + c2*q_pad[3] + c3*q_pad[4]
  q_face[3] = al(1) = xt clipped using q1(-1..2) = q_pad[3..6]
                  xt = 0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))
  q_face[4] = al(2) = c3*q1(1) + c2*q1(2) + c1*q1(3)
            = c3*q_pad[5] + c2*q_pad[6] + c1*q_pad[7]

This adds a NEW q_face[4] override (iter-892 had no override at
q_face[4]) and CHANGES the values at q_face[2] and q_face[3].

These tests verify:
A. Default OFF preserves iter-892 behavior (so iter-893's W2 baseline
   is untouched until measurement confirms iter-900 is at least as good).
B. With flag=True, q_face[2,3,4] use the Fortran-faithful formulas.
C. With flag=True, RIGHT-side q_face[n+1, n+2] still use iter-892
   (no right-side change in iter-900; deferred to iter-901+).
D. With flag=True but apply_fortran_xppm_boundary=False, the override
   does NOT fire (gated correctly).
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


def _build_quadratic_strip(n_int: int, *, batch: int = 6) -> np.ndarray:
    """Quadratic q1(j) = 1.0 + 0.1*j + 0.05*j^2 -> production layout
    strip with q[k] = q1(k-2) for k=0..n_int+3."""
    js = np.arange(-2, n_int + 2, dtype=np.float64)
    q1 = 1.0 + 0.1 * js + 0.05 * js * js
    return np.broadcast_to(q1[None, :], (batch, n_int + 4)).copy()


def _run_leaf(strip, *, on, faithful_left, n_int):
    qL, qR = _ppm_reconstruct_1d(
        jnp.asarray(strip), axis=1,
        apply_fortran_xppm_boundary=on,
        n_interior=(n_int if on else None),
        fortran_faithful_ppm_left=faithful_left)
    qL_np = np.asarray(qL)
    qR_np = np.asarray(qR)
    return np.concatenate([qL_np, qR_np[..., -1:]], axis=-1)


def _expected_fortran_al0(q_pad):
    return C1 * q_pad[..., 2] + C2 * q_pad[..., 3] + C3 * q_pad[..., 4]


def _expected_fortran_al1(q_pad):
    xt = (0.75 * (q_pad[..., 4] + q_pad[..., 5])
          - 0.25 * (q_pad[..., 3] + q_pad[..., 6]))
    q_lo = np.minimum(np.minimum(q_pad[..., 3], q_pad[..., 4]),
                       np.minimum(q_pad[..., 5], q_pad[..., 6]))
    q_hi = np.maximum(np.maximum(q_pad[..., 3], q_pad[..., 4]),
                       np.maximum(q_pad[..., 5], q_pad[..., 6]))
    return np.clip(xt, q_lo, q_hi)


def _expected_fortran_al2(q_pad):
    return C3 * q_pad[..., 5] + C2 * q_pad[..., 6] + C1 * q_pad[..., 7]


def _expected_iter892_xt_L(q_pad):
    """iter-892's q_face[2] formula (kept as default for backward compat)."""
    xt = (0.75 * (q_pad[..., 3] + q_pad[..., 4])
          - 0.25 * (q_pad[..., 2] + q_pad[..., 5]))
    q_lo = np.minimum(np.minimum(q_pad[..., 2], q_pad[..., 3]),
                       np.minimum(q_pad[..., 4], q_pad[..., 5]))
    q_hi = np.maximum(np.maximum(q_pad[..., 2], q_pad[..., 3]),
                       np.maximum(q_pad[..., 4], q_pad[..., 5]))
    return np.clip(xt, q_lo, q_hi)


def _expected_iter892_face_al2(q_pad):
    """iter-892's q_face[3] formula."""
    return C3 * q_pad[..., 4] + C2 * q_pad[..., 5] + C1 * q_pad[..., 6]


# ============================================================
# A. Default-OFF preserves iter-892 behavior
# ============================================================

def test_iter900_default_off_preserves_iter892_behavior():
    """fortran_faithful_ppm_left=False (default) must reproduce iter-892
    behavior bit-exactly, so iter-893's production W2 baseline (0.132)
    is unaffected until measurement validates a default flip."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_iter892 = _run_leaf(strip, on=True, faithful_left=False,
                                 n_int=n_int)

    expected_q_face_2 = _expected_iter892_xt_L(q_pad_np)
    expected_q_face_3 = _expected_iter892_face_al2(q_pad_np)

    np.testing.assert_allclose(q_face_iter892[..., 2], expected_q_face_2,
                                atol=1e-13, rtol=0)
    np.testing.assert_allclose(q_face_iter892[..., 3], expected_q_face_3,
                                atol=1e-13, rtol=0)


# ============================================================
# B. Flag=True applies Fortran-faithful overrides at correct slots
# ============================================================

def test_iter900_q_face_2_uses_fortran_al0_when_flag_on():
    """With fortran_faithful_ppm_left=True, q_face[2] must equal Fortran
    al(0) = c1*q1(-2) + c2*q1(-1) + c3*q1(0)."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_faithful = _run_leaf(strip, on=True, faithful_left=True,
                                  n_int=n_int)

    expected = _expected_fortran_al0(q_pad_np)
    iter892 = _expected_iter892_xt_L(q_pad_np)

    # Sanity invariant: Fortran al(0) and iter-892 xt_L must DIFFER on
    # quadratic input — otherwise iter-900's flag wouldn't change
    # anything observable.
    assert not np.allclose(expected, iter892, atol=1e-10), (
        "Fortran al(0) and iter-892's xt_L coincide on quadratic input - "
        "test isn't discriminating.")

    np.testing.assert_allclose(q_face_faithful[..., 2], expected,
                                atol=1e-13, rtol=0,
                                err_msg=("iter-900 q_face[2] doesn't "
                                         "match Fortran al(0)"))


def test_iter900_q_face_3_uses_fortran_al1_when_flag_on():
    """With flag=True, q_face[3] must equal Fortran al(1) = xt clipped
    using q1(-1..2)."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_faithful = _run_leaf(strip, on=True, faithful_left=True,
                                  n_int=n_int)

    expected = _expected_fortran_al1(q_pad_np)
    iter892 = _expected_iter892_face_al2(q_pad_np)

    assert not np.allclose(expected, iter892, atol=1e-10), (
        "Fortran al(1) and iter-892's c3/c2/c1 mirror coincide on quadratic - "
        "test isn't discriminating.")

    np.testing.assert_allclose(q_face_faithful[..., 3], expected,
                                atol=1e-13, rtol=0,
                                err_msg=("iter-900 q_face[3] doesn't "
                                         "match Fortran al(1)"))


def test_iter900_q_face_4_adds_new_fortran_al2_override_when_flag_on():
    """With flag=True, q_face[4] must equal Fortran al(2) = c3*q1(1) +
    c2*q1(2) + c1*q1(3).  iter-892 had NO override at q_face[4]
    (used the standard 4th-order edge stencil)."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_faithful = _run_leaf(strip, on=True, faithful_left=True,
                                  n_int=n_int)

    expected_fortran = _expected_fortran_al2(q_pad_np)
    expected_4th = ((7.0 * (q_pad_np[..., 5] + q_pad_np[..., 6])
                     - (q_pad_np[..., 4] + q_pad_np[..., 7])) / 12.0)

    assert not np.allclose(expected_fortran, expected_4th, atol=1e-10), (
        "Fortran al(2) and 4th-order coincide on quadratic - test isn't "
        "discriminating.")

    np.testing.assert_allclose(q_face_faithful[..., 4], expected_fortran,
                                atol=1e-13, rtol=0,
                                err_msg=("iter-900 q_face[4] doesn't "
                                         "match Fortran al(2)"))


# ============================================================
# C. Right-side untouched in iter-900
# ============================================================

def test_iter900_right_side_unchanged_when_flag_on():
    """iter-900's flag only changes the LEFT side.  q_face[n+1] and
    q_face[n+2] (right side) must still use iter-892's formulas."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_iter892 = _run_leaf(strip, on=True, faithful_left=False,
                                 n_int=n_int)
    q_face_faithful = _run_leaf(strip, on=True, faithful_left=True,
                                  n_int=n_int)

    # RIGHT-side cells must match between the two flag values.
    np.testing.assert_allclose(q_face_iter892[..., n_int + 1],
                                q_face_faithful[..., n_int + 1],
                                atol=1e-13, rtol=0,
                                err_msg=("iter-900 unexpectedly changed "
                                         "RIGHT-side q_face[n+1]"))
    np.testing.assert_allclose(q_face_iter892[..., n_int + 2],
                                q_face_faithful[..., n_int + 2],
                                atol=1e-13, rtol=0,
                                err_msg=("iter-900 unexpectedly changed "
                                         "RIGHT-side q_face[n+2]"))


# ============================================================
# D. Flag is gated on apply_fortran_xppm_boundary
# ============================================================

def test_iter900_flag_no_effect_when_xppm_boundary_off():
    """fortran_faithful_ppm_left=True is a no-op when
    apply_fortran_xppm_boundary=False — the gate at the top of the
    override block must apply to BOTH iter-892 and iter-900 paths."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_off_iter892 = _run_leaf(strip, on=False, faithful_left=False,
                                     n_int=n_int)
    q_face_off_faithful = _run_leaf(strip, on=False, faithful_left=True,
                                      n_int=n_int)

    np.testing.assert_allclose(q_face_off_iter892, q_face_off_faithful,
                                atol=1e-13, rtol=0,
                                err_msg=("iter-900 flag had effect when "
                                         "apply_fortran_xppm_boundary=False"))
