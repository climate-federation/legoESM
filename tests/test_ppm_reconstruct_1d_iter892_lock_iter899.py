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
no silent semantic drift.

Iter-899b fix (Codex stop-time review of iter-899): the original
sentinel used a LINEAR input where 4th-order, iter-892's xt-clipped,
and iter-892's c3/c2/c1 formulas all collapse to the same value at
the cube-edge slots, so the test did NOT distinguish iter-892 from
the OFF path.  Iter-899b uses a quadratic test field where:
  - 4th-order at q_face[k] = (7*(q_pad[k+1]+q_pad[k+2]) -
                              (q_pad[k]+q_pad[k+3]))/12.
  - iter-892's xt-clipped uses different coefficient weights.
  - iter-892's c3/c2/c1 mirror uses {-2/14, 11/14, 5/14} on
    {q_pad[k+1], q_pad[k+2], q_pad[k+3]} (or similar).
On a quadratic input these three formulas give DIFFERENT numbers,
making the test discriminating.  Iter-899b also adds explicit
RIGHT-edge assertions at q_face[n+1] and q_face[n+2].
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_cdgrid import _ppm_reconstruct_1d


def _build_quadratic_strip(n_int: int, *, batch: int = 6) -> np.ndarray:
    """Build a halo=2 strip with a STRICTLY QUADRATIC q1 field.

    q1(j) = 1.0 + 0.1*j + 0.05*j^2 (smooth, no extrema -> monotonicity
    limiter doesn't fire on cube-edge cells, so the iter-892 override
    values pass through to the q_face output without modification).

    Production strip layout: q[k] = q1(k-2) for k=0..n_int+3.

    On a non-linear field, 4th-order, xt-clipped, and c3/c2/c1
    produce DIFFERENT values -> the test discriminates the iter-892
    formulas from alternatives.

    Returns shape (batch, n_int+4) so the leaf treats batch as
    the leading non-axis dimension.
    """
    js = np.arange(-2, n_int + 2, dtype=np.float64)
    q1 = 1.0 + 0.1 * js + 0.05 * js * js
    return np.broadcast_to(q1[None, :], (batch, n_int + 4)).copy()


# === Constants from tp_core.F90:63-65 ===
C1 = -2.0 / 14.0
C2 = 11.0 / 14.0
C3 = 5.0 / 14.0


def _expected_xt_L_at_q_face_2(q_pad: np.ndarray) -> np.ndarray:
    """iter-892's xt_L formula at q_face[2].

    From operators_cdgrid.py:229-235:
      xt_L = 0.75*(q_pad[3] + q_pad[4]) - 0.25*(q_pad[2] + q_pad[5])
      clipped to min/max(q_pad[2..5]).
    """
    xt = (0.75 * (q_pad[..., 3] + q_pad[..., 4])
          - 0.25 * (q_pad[..., 2] + q_pad[..., 5]))
    q_lo = np.minimum(np.minimum(q_pad[..., 2], q_pad[..., 3]),
                       np.minimum(q_pad[..., 4], q_pad[..., 5]))
    q_hi = np.maximum(np.maximum(q_pad[..., 2], q_pad[..., 3]),
                       np.maximum(q_pad[..., 4], q_pad[..., 5]))
    return np.clip(xt, q_lo, q_hi)


def _expected_face_al2_at_q_face_3(q_pad: np.ndarray) -> np.ndarray:
    """iter-892's c3/c2/c1 mirror at q_face[3].

    From operators_cdgrid.py:238-239:
      face_al2 = c3 * q_pad[4] + c2 * q_pad[5] + c1 * q_pad[6]
    """
    return C3 * q_pad[..., 4] + C2 * q_pad[..., 5] + C1 * q_pad[..., 6]


def _expected_face_alnm1_at_q_face_n_plus_1(q_pad: np.ndarray, n_int: int):
    """iter-892's c1/c2/c3 at q_face[n_int+1].

    From operators_cdgrid.py:250-252:
      face_alnm1 = c1*q_pad[n+1] + c2*q_pad[n+2] + c3*q_pad[n+3]
    """
    return (C1 * q_pad[..., n_int + 1]
            + C2 * q_pad[..., n_int + 2]
            + C3 * q_pad[..., n_int + 3])


def _expected_xt_R_at_q_face_n_plus_2(q_pad: np.ndarray, n_int: int):
    """iter-892's xt_R clipped formula at q_face[n_int+2].

    From operators_cdgrid.py:256-264:
      xt_R = 0.75*(q_pad[n+3]+q_pad[n+4]) - 0.25*(q_pad[n+2]+q_pad[n+5])
      clipped to min/max(q_pad[n+2..n+5]).
    """
    xt = (0.75 * (q_pad[..., n_int + 3] + q_pad[..., n_int + 4])
          - 0.25 * (q_pad[..., n_int + 2] + q_pad[..., n_int + 5]))
    q_lo = np.minimum(
        np.minimum(q_pad[..., n_int + 2], q_pad[..., n_int + 3]),
        np.minimum(q_pad[..., n_int + 4], q_pad[..., n_int + 5]))
    q_hi = np.maximum(
        np.maximum(q_pad[..., n_int + 2], q_pad[..., n_int + 3]),
        np.maximum(q_pad[..., n_int + 4], q_pad[..., n_int + 5]))
    return np.clip(xt, q_lo, q_hi)


def _expected_4th_order_at_q_face_k(q_pad: np.ndarray, k: int):
    """Standard 4th-order edge stencil at q_face[k]."""
    return ((7.0 * (q_pad[..., k + 1] + q_pad[..., k + 2])
             - (q_pad[..., k] + q_pad[..., k + 3])) / 12.0)


def _run_leaf(strip: np.ndarray, *, on: bool, n_int: int):
    """Run _ppm_reconstruct_1d, return q_face = concat(qL, [qR[-1]])."""
    qL, qR = _ppm_reconstruct_1d(
        jnp.asarray(strip), axis=1,
        apply_fortran_xppm_boundary=on,
        n_interior=(n_int if on else None))
    qL_np = np.asarray(qL)
    qR_np = np.asarray(qR)
    last_face = qR_np[..., -1:]
    q_face = np.concatenate([qL_np, last_face], axis=-1)
    return q_face


# ============================================================
# LEFT-edge sentinels
# ============================================================

def test_iter892_q_face_2_uses_xt_clipped_not_4th_order_curved_input():
    """On a CURVED (quadratic) input, iter-892's xt_L at q_face[2] must
    DIFFER from the standard 4th-order edge stencil — proving the
    override is actually firing."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_on = _run_leaf(strip, on=True, n_int=n_int)
    q_face_off = _run_leaf(strip, on=False, n_int=n_int)

    expected_iter892 = _expected_xt_L_at_q_face_2(q_pad_np)
    expected_4th = _expected_4th_order_at_q_face_k(q_pad_np, 2)

    # The two formulas must differ on this curved input — otherwise the
    # test isn't discriminating anything.
    assert not np.allclose(expected_iter892, expected_4th, atol=1e-10), (
        f"Test invariant failed: xt_L and 4th-order coincide on quadratic "
        f"input.  Use a more curved q1 to make them distinguishable.")

    # On = iter-892's xt_L (post-monotonicity-limiting; quadratic chosen
    # so limiter doesn't fire on q_face[2]).
    np.testing.assert_allclose(q_face_on[..., 2], expected_iter892,
                                atol=1e-13, rtol=0,
                                err_msg=("iter-892's q_face[2] override "
                                         "no longer matches xt-clipped"))
    # Off = standard 4th-order edge stencil.
    np.testing.assert_allclose(q_face_off[..., 2], expected_4th,
                                atol=1e-13, rtol=0,
                                err_msg=("OFF q_face[2] no longer matches "
                                         "standard 4th-order"))


def test_iter892_q_face_3_uses_c3_c2_c1_mirror_curved_input():
    """On a curved input, iter-892's c3/c2/c1 mirror at q_face[3] must
    DIFFER from the standard 4th-order edge stencil at the same slot."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_on = _run_leaf(strip, on=True, n_int=n_int)
    q_face_off = _run_leaf(strip, on=False, n_int=n_int)

    expected_iter892 = _expected_face_al2_at_q_face_3(q_pad_np)
    expected_4th = _expected_4th_order_at_q_face_k(q_pad_np, 3)

    assert not np.allclose(expected_iter892, expected_4th, atol=1e-10), (
        f"Test invariant failed: c3/c2/c1 mirror and 4th-order coincide "
        f"on quadratic input.")

    np.testing.assert_allclose(q_face_on[..., 3], expected_iter892,
                                atol=1e-13, rtol=0,
                                err_msg=("iter-892's q_face[3] override "
                                         "no longer matches c3/c2/c1 mirror"))
    np.testing.assert_allclose(q_face_off[..., 3], expected_4th,
                                atol=1e-13, rtol=0)


# ============================================================
# RIGHT-edge sentinels (added in iter-899b per Codex stop-time)
# ============================================================

def test_iter892_q_face_n_plus_1_uses_c1_c2_c3_curved_input():
    """RIGHT-edge sentinel: iter-892's c1/c2/c3 at q_face[n+1] must
    DIFFER from the standard 4th-order edge stencil on a curved input."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_on = _run_leaf(strip, on=True, n_int=n_int)
    q_face_off = _run_leaf(strip, on=False, n_int=n_int)

    expected_iter892 = _expected_face_alnm1_at_q_face_n_plus_1(
        q_pad_np, n_int)
    expected_4th = _expected_4th_order_at_q_face_k(q_pad_np, n_int + 1)

    assert not np.allclose(expected_iter892, expected_4th, atol=1e-10), (
        f"Test invariant failed: c1/c2/c3 and 4th-order coincide on "
        f"quadratic input at q_face[n+1].")

    np.testing.assert_allclose(q_face_on[..., n_int + 1], expected_iter892,
                                atol=1e-13, rtol=0,
                                err_msg=("iter-892's q_face[n+1] override "
                                         "no longer matches c1/c2/c3"))
    np.testing.assert_allclose(q_face_off[..., n_int + 1], expected_4th,
                                atol=1e-13, rtol=0)


def test_iter892_q_face_n_plus_2_uses_xt_clipped_curved_input():
    """RIGHT-edge sentinel: iter-892's xt_R at q_face[n+2] must DIFFER
    from the standard 4th-order edge stencil on a curved input."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_on = _run_leaf(strip, on=True, n_int=n_int)
    q_face_off = _run_leaf(strip, on=False, n_int=n_int)

    expected_iter892 = _expected_xt_R_at_q_face_n_plus_2(q_pad_np, n_int)
    expected_4th = _expected_4th_order_at_q_face_k(q_pad_np, n_int + 2)

    assert not np.allclose(expected_iter892, expected_4th, atol=1e-10), (
        f"Test invariant failed: xt_R and 4th-order coincide on "
        f"quadratic input at q_face[n+2].")

    np.testing.assert_allclose(q_face_on[..., n_int + 2], expected_iter892,
                                atol=1e-13, rtol=0,
                                err_msg=("iter-892's q_face[n+2] override "
                                         "no longer matches xt-clipped"))
    np.testing.assert_allclose(q_face_off[..., n_int + 2], expected_4th,
                                atol=1e-13, rtol=0)


# ============================================================
# Default-OFF + signature sentinels (carried from iter-899)
# ============================================================

def test_iter892_off_default_does_not_apply_overrides():
    """Default OFF must keep all q_face values on the standard 4th-order
    edge stencil at q_face[2,3,n+1,n+2] — no override of any kind."""
    n_int = 12
    strip = _build_quadratic_strip(n_int)
    q_pad_np = np.pad(strip, [(0, 0), (2, 2)], mode='edge')

    q_face_off = _run_leaf(strip, on=False, n_int=n_int)

    for k in (2, 3, n_int + 1, n_int + 2):
        expected = _expected_4th_order_at_q_face_k(q_pad_np, k)
        np.testing.assert_allclose(q_face_off[..., k], expected,
                                    atol=1e-13, rtol=0,
                                    err_msg=(f"OFF q_face[{k}] doesn't "
                                             f"match standard 4th-order — "
                                             f"silently activated overrides?"))


def test_iter892_xppm_off_n_interior_none_is_legal():
    """When apply_fortran_xppm_boundary=False, n_interior=None is fine."""
    n_int = 8
    strip = _build_quadratic_strip(n_int)
    qL, qR = _ppm_reconstruct_1d(
        jnp.asarray(strip), axis=1,
        apply_fortran_xppm_boundary=False, n_interior=None)
    assert qL.shape == strip.shape
    assert qR.shape == strip.shape
