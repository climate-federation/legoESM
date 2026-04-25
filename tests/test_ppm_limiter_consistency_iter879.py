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


def test_iter879_both_limiters_use_signed_product_in_overshoot_compare():
    """AST scan: both limiter implementations MUST express the
    overshoot constraint as a Compare whose LHS is a signed product
    of the parabolic term (``q_6`` / ``d6``) and the span (``dq`` /
    ``dm``), and whose RHS is the squared span (``dq*dq`` / ``dm**2``).

    Iter-879 take-2 (Codex stop-time): the original AST scan only
    checked for the EXISTENCE of a ``q_6 * dq`` product anywhere in
    the function, which can false-pass if a function contains the
    product as an intermediate but uses the WRONG form in the
    actual constraint Compare.  This stricter scan walks every
    Compare node in the function body, identifies the ones that
    look like overshoot constraints (RHS or comparator involves
    ``dq*dq`` / ``dm*dm`` / ``dm**2`` / their negation), and
    requires the LHS to be a signed product or a Name bound to
    such a product.

    A regression to ``q_6 > dq * dq`` (LHS is bare ``q_6``, no
    span factor) would fail this scan because the LHS Name doesn't
    match the signed-product structural pattern.
    """
    import ast
    from pathlib import Path

    paths = [
        ("src/legoesm/core/operators_cdgrid.py", "_ppm_reconstruct_1d"),
        ("src/legoesm/core/operators_fv.py", "_ppm_limit"),
    ]

    repo_root = Path(__file__).resolve().parent.parent

    parabolic_names = {"q_6", "d6"}
    span_names = {"dq", "dm"}

    def _is_signed_product(node, intermediates):
        """Return True if `node` is a BinOp(Mult) of parabolic*span,
        OR a Name bound earlier to such a BinOp.
        """
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            l = node.left
            r = node.right
            l_para = (isinstance(l, ast.Name) and l.id in parabolic_names)
            l_span = (isinstance(l, ast.Name) and l.id in span_names)
            r_para = (isinstance(r, ast.Name) and r.id in parabolic_names)
            r_span = (isinstance(r, ast.Name) and r.id in span_names)
            return (l_para and r_span) or (r_para and l_span)
        if isinstance(node, ast.Name):
            return node.id in intermediates
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return _is_signed_product(node.operand, intermediates)
        return False

    def _is_squared_span(node, squared_span_aliases):
        """Return True if `node` is `dq*dq`, `dm*dm`, `dq**2`,
        `dm**2`, a USub thereof, or a Name bound to such a value.
        """
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return _is_squared_span(node.operand, squared_span_aliases)
        if isinstance(node, ast.Name) and node.id in squared_span_aliases:
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            l, r = node.left, node.right
            if (isinstance(l, ast.Name) and isinstance(r, ast.Name)
                    and l.id == r.id and l.id in span_names):
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            l, r = node.left, node.right
            if (isinstance(l, ast.Name) and l.id in span_names
                    and isinstance(r, ast.Constant) and r.value == 2):
                return True
        return False

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

        # Collect intermediate Names bound to a signed product
        # (e.g. `q6_dq = q_6 * dq`).
        intermediates = set()
        # Collect intermediate Names bound to a squared span
        # (e.g. `dq_sq = dq * dq`).
        squared_span_aliases = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    val = node.value
                    if (isinstance(val, ast.BinOp)
                            and isinstance(val.op, ast.Mult)
                            and _is_signed_product(val, set())):
                        intermediates.add(target.id)
                    if isinstance(val, ast.BinOp):
                        if isinstance(val.op, ast.Mult):
                            l, r = val.left, val.right
                            if (isinstance(l, ast.Name)
                                    and isinstance(r, ast.Name)
                                    and l.id == r.id
                                    and l.id in span_names):
                                squared_span_aliases.add(target.id)
                        elif isinstance(val.op, ast.Pow):
                            l, r = val.left, val.right
                            if (isinstance(l, ast.Name)
                                    and l.id in span_names
                                    and isinstance(r, ast.Constant)
                                    and r.value == 2):
                                squared_span_aliases.add(target.id)

        # Iter-879d (Codex iter-879c stop-time): also catch reverse
        # comparisons (`squared_span < q_6` instead of
        # `q_6 > squared_span`).  Walk every Compare and consider
        # BOTH directions: if EITHER the LHS is squared-span and
        # RHS is parabolic, OR vice versa, treat it as a candidate
        # constraint Compare.  The "parabolic-side" operand must be
        # a signed product; the "squared-span-side" operand is
        # already checked against the squared-span pattern.
        def _references_parabolic_or_signed_product(node):
            """Return True if `node` mentions q_6, d6, or a Name in
            ``intermediates`` (i.e., a signed-product alias).  Used
            to identify the 'parabolic side' of a Compare."""
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return _references_parabolic_or_signed_product(node.operand)
            if isinstance(node, ast.Name):
                return (node.id in parabolic_names
                        or node.id in intermediates)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                # Includes the bare `q_6 * dq` BinOp form.
                return _is_signed_product(node, set())
            return False

        constraint_compares = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare):
                continue
            lhs = node.left
            for comparator in node.comparators:
                # Direction A: LHS=parabolic-side, RHS=squared-span.
                if _is_squared_span(comparator, squared_span_aliases):
                    if _references_parabolic_or_signed_product(lhs):
                        constraint_compares.append((lhs, comparator))
                # Direction B: LHS=squared-span, RHS=parabolic-side.
                elif _is_squared_span(lhs, squared_span_aliases):
                    if _references_parabolic_or_signed_product(comparator):
                        # Swap so the "parabolic side" is always
                        # checked uniformly below.
                        constraint_compares.append((comparator, lhs))

        assert constraint_compares, (
            f"Function `{fn_name}` in {rel_path} has no Compare "
            f"with a squared-span RHS (`dq*dq`, `dm*dm`, `dq**2`, "
            f"`dm**2`, or negated).  The overshoot constraint per "
            f"CW84 eq. 1.10 must be `signed_product > squared_span` "
            f"and `signed_product < -squared_span`.  Either the "
            f"limiter was rewritten without these constraints, or "
            f"the variable names diverged from the iter-879 sentinel "
            f"vocabulary.  Update both source and sentinel together "
            f"if this is intentional.")

        # Iter-879c (Codex iter-879b stop-time): EVERY Compare with a
        # squared-span RHS MUST have a signed-product LHS.  iter-879b
        # required only "at least one" which false-passed if a
        # function had both a buggy and a correct Compare.  The
        # CW84-compliant pattern is structural: every overshoot
        # constraint Compare must have the signed product, no
        # exceptions.  A bare ``q_6`` LHS in any squared-span Compare
        # is the iter-878 regression and must fail loudly.
        bad_compares = [
            (lhs, rhs) for lhs, rhs in constraint_compares
            if not _is_signed_product(lhs, intermediates)
        ]
        assert not bad_compares, (
            f"Function `{fn_name}` in {rel_path} has Compare(s) with "
            f"a squared-span RHS but a non-signed-product LHS — the "
            f"iter-878 missing-product regression.  Per CW84 eq. "
            f"1.10 the LHS must be `q_6 * dq` (or `d6 * dm`); "
            f"pre-iter-878 the LHS was bare `q_6` (no span factor).  "
            f"Iter-879c+d require ALL squared-span compares (both "
            f"directions) to have signed-product on the parabolic "
            f"side so a co-existing 'good' Compare cannot mask a "
            f"buggy one.\n"
            f"Bad compares ({len(bad_compares)} of "
            f"{len(constraint_compares)}):\n"
            + "\n".join(f"  parabolic-side={ast.unparse(lhs)} "
                       f"(against squared-span={ast.unparse(rhs)})"
                       for lhs, rhs in bad_compares))


# Iter-879d (Codex iter-879c stop-time): the AST sentinel above
# catches the iter-878 regression class for the canonical syntactic
# forms (`q_6 > dq_sq`, `dq_sq < q_6`, with various operator+sign
# variants).  It does NOT catch obfuscated forms such as:
#
# - `jax.lax.gt(q_6, dq_sq)` (function call instead of Compare)
# - `(q_6 - dq_sq) > 0` (algebraic rewrite that hides the structure)
# - Renaming `q_6` / `dq` / `q6_dq` / `dq_sq` to non-vocabulary names
#   (the scan only knows the documented variable names)
#
# These are out-of-scope for a structural sentinel — catching them
# would require partial evaluation or theorem proving.  For
# behavioural coverage of regressions in any form, the
# `test_iter879_ppm_reconstruct_1d_matches_ppm_limit` test above
# compares actual function output against the cross-implementation
# reference (`_ppm_limit`) on random inputs — a regression that
# changes ANY of the limiter semantics will fail the bit-match
# check at 1e-12 rtol, regardless of how the source is written.
