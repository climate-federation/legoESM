"""Iter-879: pin consistency between the two PPM limiters in the
codebase to prevent future divergence.

The repo has TWO PPM monotonicity limiter implementations:

1. ``_ppm_reconstruct_1d`` in ``src/legoesm/core/operators_cdgrid.py``
   (used by ``cgrid_mass_flux_divergence`` — production W2 path).
2. ``_ppm_limit`` in ``src/legoesm/core/operators_fv.py`` (used by
   ``fv_flux_divergence`` — alternative FV transport path).

Both implement the Colella-Woodward 1984 (CW84) monotonicity
constraint per eq. 1.10:

    if Δa · q_6 >  (Δa)²:  q_L = 3q - 2q_R
    if Δa · q_6 < -(Δa)²:  q_R = 3q - 2q_L

Pre-iter-878, ``_ppm_reconstruct_1d`` had the buggy form
``q_6 > Δa²`` (missing the ``Δa`` factor) while ``_ppm_limit`` had
the correct ``Δa · q_6 > Δa²``.  iter-878 fixed
``_ppm_reconstruct_1d`` to match.  This iter-879 sentinel verifies
the two implementations agree on identical inputs — catches a
future regression that re-introduces a divergence.

The two functions have different signatures:
- ``_ppm_reconstruct_1d(q, axis)``: computes face values via 4th-
  order interpolation THEN applies CW84 limiter, returns
  ``(q_L, q_R)``.
- ``_ppm_limit(q_bar, q_L, q_R)``: takes pre-computed face values
  and applies CW84 limiter, returns ``(q_L_lim, q_R_lim)``.

To compare apples-to-apples we manually pre-compute the same 4th-
order face values that ``_ppm_reconstruct_1d`` uses, then feed them
to both limiters.  The output must match.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.core.operators_cdgrid import _ppm_reconstruct_1d
from legoesm.core.operators_fv import _ppm_limit


def _compute_face_values_4thorder(q_pad):
    """Same 4th-order face-value formula as `_ppm_reconstruct_1d`."""
    q_face = (7.0 * (q_pad[..., 1:-2] + q_pad[..., 2:-1])
              - (q_pad[..., :-3] + q_pad[..., 3:])) / 12.0
    q_L = q_face[..., :-1]
    q_R = q_face[..., 1:]
    return q_L, q_R


@pytest.mark.parametrize("seed", [11, 23, 31])
def test_iter879_ppm_reconstruct_1d_matches_ppm_limit(seed):
    """Both PPM limiters MUST produce identical output on the same
    pre-limited (q_L, q_R) inputs.

    This pins the iter-878 fix that brought ``_ppm_reconstruct_1d``
    into agreement with ``_ppm_limit``.  If a future iter reverts
    or modifies either implementation in a way that reintroduces
    divergence, this test fails loudly.
    """
    rng = np.random.default_rng(seed)
    n = 32
    # Wide range to trigger the constraint at multiple cells.
    q_np = rng.normal(scale=10.0, size=n)
    q = jnp.asarray(q_np)

    # _ppm_reconstruct_1d output.
    q_L_a, q_R_a = _ppm_reconstruct_1d(q, axis=0)

    # Manually compute the same 4th-order face values, then run the
    # _ppm_limit limiter.
    q_pad = np.pad(q_np, (2, 2), mode='edge')
    q_L_unlimited, q_R_unlimited = _compute_face_values_4thorder(q_pad)
    q_L_b, q_R_b = _ppm_limit(
        jnp.asarray(q_np),
        jnp.asarray(q_L_unlimited),
        jnp.asarray(q_R_unlimited))

    np.testing.assert_allclose(
        np.asarray(q_L_a), np.asarray(q_L_b), rtol=1e-12, atol=1e-12,
        err_msg=(
            f"`_ppm_reconstruct_1d` and `_ppm_limit` produce "
            f"DIFFERENT q_L on seed={seed}.  This means one of the "
            f"two PPM limiter implementations diverged from CW84 "
            f"eq. 1.10 / Fortran pert_ppm.  Audit both and bring "
            f"them back into agreement (iter-878 made "
            f"`_ppm_reconstruct_1d` match `_ppm_limit`)."))
    np.testing.assert_allclose(
        np.asarray(q_R_a), np.asarray(q_R_b), rtol=1e-12, atol=1e-12,
        err_msg=(
            f"`_ppm_reconstruct_1d` and `_ppm_limit` produce "
            f"DIFFERENT q_R on seed={seed}."))


def test_iter879_both_limiters_use_signed_product():
    """AST scan: both limiter implementations MUST use the signed
    product ``q_6 * dq`` (or equivalent ``dm * d6``) in the
    overshoot conditions.  Catches a future regression that
    re-introduces the missing-``dq`` form.
    """
    import ast
    from pathlib import Path

    paths = [
        ("src/legoesm/core/operators_cdgrid.py", "_ppm_reconstruct_1d"),
        ("src/legoesm/core/operators_fv.py", "_ppm_limit"),
    ]

    repo_root = Path(__file__).resolve().parent.parent

    for rel_path, fn_name in paths:
        src = (repo_root / rel_path).read_text()
        tree = ast.parse(src)
        fn = next(
            (n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == fn_name),
            None,
        )
        assert fn is not None, (
            f"Could not find `{fn_name}` in {rel_path}.")

        # Look for a signed-product BinOp where one operand mentions
        # the parabolic term (q_6/d6) and the other mentions the
        # span (dq/dm).  Reject cases where left==right (self-product).
        found = False
        for node in ast.walk(fn):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                left = ast.dump(node.left)
                right = ast.dump(node.right)
                if left == right:
                    continue
                # Check for parabolic-term × span-term combo.
                left_has_parabolic = (
                    "q_6" in left or "'d6'" in left or "'q6_dq'" in left)
                left_has_span = ("'dq'" in left or "'dm'" in left)
                right_has_parabolic = (
                    "q_6" in right or "'d6'" in right or "'q6_dq'" in right)
                right_has_span = ("'dq'" in right or "'dm'" in right)
                if ((left_has_parabolic and right_has_span)
                        or (right_has_parabolic and left_has_span)):
                    found = True
                    break
        assert found, (
            f"Function `{fn_name}` in {rel_path} does NOT contain a "
            f"signed `q_6*dq` (or `d6*dm`) product.  Per CW84 eq. "
            f"1.10 the overshoot constraint is `Δa · a_6 > (Δa)²`, "
            f"requiring this product on the LHS.  iter-878 fixed "
            f"the missing-product bug in `_ppm_reconstruct_1d`; "
            f"this test catches a regression in either limiter.")
