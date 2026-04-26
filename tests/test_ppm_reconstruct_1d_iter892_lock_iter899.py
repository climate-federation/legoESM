"""Iter-899 regression sentinel: pin the iter-892 PPM cube-edge
boundary override BEHAVIOR at production strip layout.

iter-899 investigation revealed that iter-892's overrides at q_face
indices [2, 3, n+1, n+2] were designed under the assumption q[k]=q1(k-1)
(Hypothesis B) but production actually feeds q[k]=q1(k-2) (Hypothesis A,
verified empirically by `scripts/diag_iter899_ppm_strip_layout.py`).

This is a 1-cell index-map shift bug.  Despite the shift, iter-893
landed the iter-892 overrides on the production W2 path and measured a
17 % v_ll_Linf reduction (0.1593 -> 0.1319 m/s).

This test PINS the iter-892 override BEHAVIOR at the production strip
layout so that any future iter-900+ "fix" of the index map (or of the
formulas themselves) MUST EXPLICITLY change the pinned values — i.e.,
no silent semantic drift.  The pinned values are computed at runtime
inside the test from the iter-892 formulas applied to a fixed synthetic
input strip in the production layout (q[k]=q1(k-2)).

Pinning strategy: rather than hard-coding numerical thresholds, the
test computes the iter-892 overrides DIRECTLY (via the formulas in
operators_cdgrid.py:217-264) and asserts the leaf produces
bit-identical output (rtol=0 atol=1e-15).  This converts the iter-892
override formulas into a behavioral lock — their outputs at the 4
overridden q_face indices must match the documented formulas
verbatim.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import _ppm_reconstruct_1d


def _build_production_strip(n_int: int, *, batch: int = 6) -> np.ndarray:
    """Build a halo=2-per-side strip in the production layout.

    q[k] = q1(k-2) for k=0..n_int+3.  q1 values: linear ramp so the
    iter-892 xt clipping never fires, enabling clean comparison.

    Returns shape (batch, n_int+4) — batch axis stands in for face.
    """
    q1_full = np.array(
        [float(k * 0.1 + 1.0) for k in range(-2, n_int + 2)],
        dtype=np.float64)
    # q[k] = q1(k-2) so q has the SAME values as q1_full slice:
    strip = q1_full.copy()  # length n_int + 4
    return np.broadcast_to(strip[None, :], (batch, n_int + 4)).copy()


def _iter892_xt_L(q_pad: np.ndarray) -> np.ndarray:
    """Replicate iter-892's xt_L formula at q_face[2] for sanity-check."""
    xt = (0.75 * (q_pad[..., 3] + q_pad[..., 4])
          - 0.25 * (q_pad[..., 2] + q_pad[..., 5]))
    q_lo = np.minimum(np.minimum(q_pad[..., 2], q_pad[..., 3]),
                       np.minimum(q_pad[..., 4], q_pad[..., 5]))
    q_hi = np.maximum(np.maximum(q_pad[..., 2], q_pad[..., 3]),
                       np.maximum(q_pad[..., 4], q_pad[..., 5]))
    return np.clip(xt, q_lo, q_hi)


def _iter892_face_al2(q_pad: np.ndarray) -> np.ndarray:
    """Replicate iter-892's c3/c2/c1 mirror at q_face[3]."""
    c1 = -2.0/14.0; c2 = 11.0/14.0; c3 = 5.0/14.0
    return c3 * q_pad[..., 4] + c2 * q_pad[..., 5] + c1 * q_pad[..., 6]


def test_iter892_override_at_q_face_2_matches_documented_formula():
    """iter-892's q_face[2] override must equal the documented xt_L
    formula on a known input strip in the production layout.

    Iter-899 sentinel: documents the actual current behavior at the
    cube-edge LEFT slot.  Under the iter-892 docstring this is
    "al(1) per Fortran"; under the actual production layout
    (Hypothesis A, q[k]=q1(k-2)) this index is al(0) — but the
    formula is xt-style not Fortran's c1/c2/c3.
    """
    n_int = 12
    strip = _build_production_strip(n_int)
    # Mimic _ppm_reconstruct_1d's pad(2,2, edge):
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    expected_at_q_face_2 = _iter892_xt_L(q_pad_np)

    qL, qR = _ppm_reconstruct_1d(
        jnp.asarray(strip), axis=1,
        apply_fortran_xppm_boundary=True, n_interior=n_int)

    # Reconstruct q_face from qL/qR: q_face[k] = qL[k] for k=0..n+3,
    # plus q_face[n+4] = qR[n+3].
    qL_np = np.asarray(qL)
    actual_at_q_face_2 = qL_np[:, 2]   # qL[k]=q_face[k]

    # Linear ramp q1 -> xt formulas land at the linearly-interpolated
    # face value, with no clipping triggered.  iter-892's xt_L at
    # q_face[2] should be 0.75*(0.9+1.0) - 0.25*(0.8+1.1) = 0.95 m/s.
    np.testing.assert_allclose(actual_at_q_face_2, expected_at_q_face_2,
                                atol=1e-13, rtol=0)

    # On a linear field, the actual value should be 0.95 (q1(-1)+q1(0))/2
    # bit-arithmetic.  Pin numerically:
    np.testing.assert_allclose(actual_at_q_face_2[0], 0.95, atol=1e-13)


def test_iter892_override_at_q_face_3_matches_documented_formula():
    """iter-892's q_face[3] override must equal the documented
    c3/c2/c1 formula on a known input strip in the production layout.
    """
    n_int = 12
    strip = _build_production_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    expected_at_q_face_3 = _iter892_face_al2(q_pad_np)

    qL, qR = _ppm_reconstruct_1d(
        jnp.asarray(strip), axis=1,
        apply_fortran_xppm_boundary=True, n_interior=n_int)
    qL_np = np.asarray(qL)
    actual_at_q_face_3 = qL_np[:, 3]

    # Note the monotonicity limiter (lines 273-301 of operators_cdgrid)
    # is applied AFTER the override.  On a strictly linear field with
    # no extrema, no limiting fires.  Verify bit-equality:
    np.testing.assert_allclose(actual_at_q_face_3, expected_at_q_face_3,
                                atol=1e-13, rtol=0)

    # Numerical pin: c3*q1(0) + c2*q1(1) + c1*q1(2)
    #              = (5/14)*1.0 + (11/14)*1.1 + (-2/14)*1.2
    #              = (5 + 12.1 - 2.4)/14 = 14.7/14 = 1.05.
    np.testing.assert_allclose(actual_at_q_face_3[0], 1.05, atol=1e-13)


def test_iter892_off_default_preserves_pre_iter892_4th_order():
    """iter-892 default (apply_fortran_xppm_boundary=False) must NOT
    apply any cube-edge overrides — the leaf produces the standard
    4th-order edge stencil with mode='edge' replicas at the boundary.
    Sentinel against accidentally flipping the default.
    """
    n_int = 12
    strip = _build_production_strip(n_int)

    qL, qR = _ppm_reconstruct_1d(
        jnp.asarray(strip), axis=1,
        apply_fortran_xppm_boundary=False)
    qL_np = np.asarray(qL)

    # 4th-order at q_face[2] = (7*(q_pad[3]+q_pad[4]) - (q_pad[2]+q_pad[5]))/12
    # On linear ramp, this reduces to the linear-interpolated face value
    # (q1(-1)+q1(0))/2 = 0.95.  Same as iter-892's xt_L for a linear
    # field — the test that distinguishes the two paths needs a curved
    # input, but the linear-field ON/OFF coincidence is itself a
    # nontrivial property worth pinning here.
    np.testing.assert_allclose(qL_np[0, 2], 0.95, atol=1e-13)


def test_iter892_xppm_off_n_interior_none_is_legal():
    """When apply_fortran_xppm_boundary=False, n_interior=None is fine
    (the kwarg is ignored).  Sentinel against accidentally requiring
    the kwarg in the OFF default path.
    """
    n_int = 8
    strip = _build_production_strip(n_int)
    qL, qR = _ppm_reconstruct_1d(
        jnp.asarray(strip), axis=1,
        apply_fortran_xppm_boundary=False, n_interior=None)
    assert qL.shape == strip.shape
    assert qR.shape == strip.shape
