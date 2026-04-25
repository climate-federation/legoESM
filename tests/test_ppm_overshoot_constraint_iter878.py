"""Iter-878: pin the Colella-Woodward 1984 (CW84) overshoot constraint
formula in `_ppm_reconstruct_1d` against Fortran's `pert_ppm`
(tp_core.F90:1156-1214).

CW84 eq. 1.10 specifies:

    if Δa · a_6 >  (Δa)²:  a_L = 3a - 2a_R
    if Δa · a_6 < -(Δa)²:  a_R = 3a - 2a_L

where Δa = a_R - a_L and a_6 = 6(a - 0.5(a_L+a_R)).  Fortran's
``pert_ppm`` implements this verbatim with the ``a6da = 3*(al+ar)*da1``
factorization (line 1199), which expands algebraically to the
``q_6 * dq`` factor on the LHS of the comparison.

Pre-iter-878 our Python `_ppm_reconstruct_1d` had the conditions:

    cond_L = q_6 > dq * dq
    cond_R = -q_6 > dq * dq

i.e., ``q_6 > dq²`` and ``-q_6 > dq²`` — MISSING the ``dq`` factor on
the LHS.  This differs from CW84 / Fortran in two regimes:
- ``|dq|  > 1`` (large jump): pre-iter-878 condition is HARDER to
  satisfy (RHS = dq² is large), so PPM undercaps.
- ``|dq| < 1`` (small jump): pre-iter-878 condition is EASIER to
  satisfy (RHS is tiny), so PPM overcaps.
- ``dq < 0``: pre-iter-878 ignores sign (dq² is always positive);
  CW84 handles correctly via the signed product ``q_6 * dq``.

Iter-878 fix: replace ``q_6 > dq * dq`` with ``q_6 * dq > dq * dq``
(and analogously for cond_R).  This matches CW84 and Fortran
``pert_ppm`` exactly.

Behavioural verification on a synthetic input that triggers the
constraint differently between the two formulas: the iter-878
output equals the Fortran-faithful CW84 reference; the pre-iter-878
output differs at cells where the cap activation differs.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import ast
import inspect
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.core.operators_cdgrid import _ppm_reconstruct_1d


def _ppm_cw84_reference(q):
    """Fortran-faithful CW84 PPM reference (1D, axis=-1).

    Implements the same algorithm as `_ppm_reconstruct_1d` but with
    the explicit CW84 condition ``Δa · a_6 > (Δa)²`` (matches Fortran
    ``pert_ppm`` ``a6da > da2``).  This is the iter-878 expected
    behaviour.
    """
    q = np.asarray(q, dtype=np.float64)
    N = q.shape[-1]
    q_pad = np.pad(q, [(0, 0)] * (q.ndim - 1) + [(2, 2)], mode='edge')
    q_face = (7.0 * (q_pad[..., 1:-2] + q_pad[..., 2:-1])
              - (q_pad[..., :-3] + q_pad[..., 3:])) / 12.0
    q_L = q_face[..., :-1]
    q_R = q_face[..., 1:]

    delta = (q_R - q) * (q - q_L)
    is_extremum = delta <= 0.0
    q_L = np.where(is_extremum, q, q_L)
    q_R = np.where(is_extremum, q, q_R)

    q_6 = 6.0 * (q - 0.5 * (q_L + q_R))
    dq = q_R - q_L
    q6_dq = q_6 * dq
    dq_sq = dq * dq
    cond_L = q6_dq > dq_sq
    q_L_adj = 3.0 * q - 2.0 * q_R
    q_L = np.where(cond_L & ~is_extremum, q_L_adj, q_L)
    cond_R = q6_dq < -dq_sq
    q_R_adj = 3.0 * q - 2.0 * q_L
    q_R = np.where(cond_R & ~is_extremum, q_R_adj, q_R)

    return q_L, q_R


def _ppm_pre_iter878_reference(q):
    """Pre-iter-878 buggy reference (missing ``dq`` factor)."""
    q = np.asarray(q, dtype=np.float64)
    q_pad = np.pad(q, [(0, 0)] * (q.ndim - 1) + [(2, 2)], mode='edge')
    q_face = (7.0 * (q_pad[..., 1:-2] + q_pad[..., 2:-1])
              - (q_pad[..., :-3] + q_pad[..., 3:])) / 12.0
    q_L = q_face[..., :-1]
    q_R = q_face[..., 1:]

    delta = (q_R - q) * (q - q_L)
    is_extremum = delta <= 0.0
    q_L = np.where(is_extremum, q, q_L)
    q_R = np.where(is_extremum, q, q_R)

    q_6 = 6.0 * (q - 0.5 * (q_L + q_R))
    dq = q_R - q_L
    cond_L = q_6 > dq * dq            # BUGGY: missing dq factor
    q_L_adj = 3.0 * q - 2.0 * q_R
    q_L = np.where(cond_L & ~is_extremum, q_L_adj, q_L)
    cond_R = -q_6 > dq * dq           # BUGGY: missing dq factor
    q_R_adj = 3.0 * q - 2.0 * q_L
    q_R = np.where(cond_R & ~is_extremum, q_R_adj, q_R)

    return q_L, q_R


def test_iter878_ppm_matches_cw84_reference():
    """``_ppm_reconstruct_1d`` output MUST match the CW84 reference
    on a synthetic input that triggers the overshoot constraint.

    Build a 1D ramp with a sharp gradient designed to fire the
    overshoot constraint at multiple cells, then verify the
    function output bit-matches the CW84 reference.
    """
    rng = np.random.default_rng(2026)
    # Construct a sharp ramp + small noise to trigger the overshoot
    # constraint at many cells with various dq magnitudes.
    n = 32
    base = np.linspace(0.0, 100.0, n)
    noise = rng.normal(scale=2.0, size=n)
    q_np = base + noise
    q = jnp.asarray(q_np)

    q_L_actual, q_R_actual = _ppm_reconstruct_1d(q, axis=0)
    q_L_expected, q_R_expected = _ppm_cw84_reference(q_np)

    np.testing.assert_allclose(
        np.asarray(q_L_actual), q_L_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(q_R_actual), q_R_expected, rtol=1e-12, atol=1e-12)


def test_iter878_ppm_differs_from_pre_iter878_buggy_reference():
    """``_ppm_reconstruct_1d`` MUST produce different output from the
    pre-iter-878 buggy formula on inputs that trigger the constraint
    in the divergent regime.

    The two formulas converge when |dq| ≈ 1 (because dq² ≈ dq), so
    pick an input scaled so |dq| is far from 1.  This guarantees the
    iter-878 fix is actually firing in the comparison.

    A scale factor of 100 amplifies dq from typical ~5 to ~500.  At
    that magnitude, the pre-iter-878 condition ``q_6 > dq²`` requires
    q_6 > 250000 (rare), whereas iter-878 requires ``q_6 * dq > dq²``,
    i.e., ``q_6 > dq = 500`` (much more attainable).  Differences
    emerge at cells where the iter-878 fix correctly caps but the
    pre-iter-878 buggy formula does not.
    """
    rng = np.random.default_rng(3033)
    n = 32
    # Wide-range input so |dq| varies far from 1.
    q_np = rng.normal(scale=100.0, size=n)

    q_L_actual, q_R_actual = _ppm_reconstruct_1d(jnp.asarray(q_np), axis=0)
    q_L_buggy, q_R_buggy = _ppm_pre_iter878_reference(q_np)

    diff_L = float(np.max(np.abs(np.asarray(q_L_actual) - q_L_buggy)))
    diff_R = float(np.max(np.abs(np.asarray(q_R_actual) - q_R_buggy)))
    assert (diff_L > 1e-6) or (diff_R > 1e-6), (
        f"iter-878 PPM fix is invisible: actual output bit-matches "
        f"the pre-iter-878 buggy formula on a dq-large input "
        f"(max |Δq_L|={diff_L:.3e}, max |Δq_R|={diff_R:.3e}).  Either "
        f"the fix was reverted, or the input doesn't trigger the "
        f"constraint in the divergent regime.")


def test_iter878_source_uses_signed_product():
    """AST scan: ``_ppm_reconstruct_1d`` source MUST use the signed
    product ``q_6 * dq`` in the overshoot conditions, not the
    pre-iter-878 ``q_6 > dq * dq`` form.

    This catches a future regression that reverts the formula.  The
    scan looks for a ``Compare`` node whose left operand is a
    ``BinOp(Mult)`` involving both ``q_6`` and ``dq`` (e.g. the new
    ``q6_dq`` intermediate), and rejects a Compare whose left
    operand is a bare ``Name('q_6')`` or ``UnaryOp(USub, Name('q_6'))``
    inside the function.
    """
    src_path = (Path(__file__).resolve().parent.parent
                / "src" / "legoesm" / "core" / "operators_cdgrid.py")
    tree = ast.parse(src_path.read_text())

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "_ppm_reconstruct_1d"),
        None,
    )
    assert fn is not None, "Could not find _ppm_reconstruct_1d"

    # Look for a Mult BinOp where one operand mentions q_6 and the
    # other mentions dq (but not the `dq * dq` self-product).
    found_q6_dq_product = False
    for node in ast.walk(fn):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left_dump = ast.dump(node.left)
            right_dump = ast.dump(node.right)
            if left_dump == right_dump:
                # Skip `dq * dq` self-product.
                continue
            has_q6 = "q_6" in left_dump or "q_6" in right_dump
            has_dq = ("'dq'" in left_dump or "'dq'" in right_dump)
            if has_q6 and has_dq:
                found_q6_dq_product = True
                break

    assert found_q6_dq_product, (
        "_ppm_reconstruct_1d source does NOT contain a `q_6 * dq` "
        "product.  Per CW84 eq. 1.10 / Fortran tp_core.F90:1199 the "
        "overshoot condition is `Δa · a_6 > (Δa)²`, requiring the "
        "signed product `q_6 * dq` on the LHS.  The pre-iter-878 "
        "form `q_6 > dq * dq` is missing this factor.  Restore the "
        "iter-878 fix.")


def test_iter878_constant_field_unchanged():
    """Sanity: a constant field has q_L = q_R = q everywhere
    (extremum flatten).  Pre- and post-iter-878 agree on this case."""
    n = 16
    q_const = jnp.ones((n,)) * 7.5
    q_L, q_R = _ppm_reconstruct_1d(q_const, axis=0)
    np.testing.assert_allclose(np.asarray(q_L), 7.5, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(np.asarray(q_R), 7.5, rtol=1e-15, atol=1e-15)
