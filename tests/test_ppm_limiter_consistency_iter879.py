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
``_ppm_reconstruct_1d`` to match.

Iter-879f (Codex iter-879a..e stop-time chain): the iter-879
chain went through five rounds of progressive AST-sentinel
tightening trying to catch the regression syntactically.  Each
round closed a known false-pass class (existence-only → "any" →
"all" → reverse-direction → unfiltered both-directions), and each
round Codex stop-time review identified yet another false-pass
path.  After five rounds it became clear that an AST sentinel can
NEVER be 100% complete: regressions can be obfuscated indefinitely
via function calls (`jax.lax.gt`), algebraic rewrites
(`(q_6 - dq_sq) > 0`), variable renames outside the vocabulary,
chained comparisons (`a < b < c`), Subscript LHS, etc.

iter-879f accepts this and removes the AST sentinel entirely.  The
canonical regression sentinel is the BEHAVIOURAL bit-match test
below: a regression that changes ANY of the limiter semantics in
EITHER source file will fail the cross-implementation bit-match at
1e-12 rtol, regardless of how the source is written.  This is the
strongest possible structural guarantee: equivalent semantics in
both implementations, end of story.

For PER-FUNCTION existence verification of the iter-878 fix in
``_ppm_reconstruct_1d`` specifically, see
``tests/test_ppm_overshoot_constraint_iter878.py``'s
``test_iter878_source_uses_signed_product`` AST scan, which is
narrower in scope (single function, single existence check) and
appropriately matches its purpose.
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

    This is the CANONICAL regression sentinel for the iter-878 fix
    that brought ``_ppm_reconstruct_1d`` into agreement with
    ``_ppm_limit``.  A regression in EITHER source file (any form
    — Compare, function call, algebraic rewrite, rename, etc.)
    that changes the limiter semantics will fail this bit-match
    check at 1e-12 rtol.

    iter-879f scope clarification: this behavioural test
    SUPERSEDES the AST sentinel iterations (iter-879a..e) that
    tried to catch the regression syntactically.  Five rounds of
    Codex stop-time review found progressive false-pass paths in
    the AST scan (existence-only → "any" → "all" → reverse-
    direction → unfiltered).  iter-879f accepts that AST scans
    can never be 100% complete and relies on this end-to-end
    bit-match instead.
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


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_iter879_ppm_consistency_across_scales(scale):
    """Bit-match consistency MUST hold across multiple input scales.

    iter-879f addition: parametrize over very small (constraint
    rarely fires), unit (mixed), and very large (constraint fires
    often) input scales.  This widens the behavioural coverage
    beyond the seed-only parametrization above.  A regression that
    only manifests in one scale regime (e.g., changing the cap
    threshold) would fail at least one of these scale points.
    """
    rng = np.random.default_rng(42)
    n = 32
    q_np = rng.normal(scale=scale, size=n)
    q = jnp.asarray(q_np)

    q_L_a, q_R_a = _ppm_reconstruct_1d(q, axis=0)
    q_pad = np.pad(q_np, (2, 2), mode='edge')
    q_L_unlimited, q_R_unlimited = _compute_face_values_4thorder(q_pad)
    q_L_b, q_R_b = _ppm_limit(
        jnp.asarray(q_np),
        jnp.asarray(q_L_unlimited),
        jnp.asarray(q_R_unlimited))

    # Use scale-aware atol (1e-12 * scale) to account for the
    # natural magnitude of values at this scale.  rtol stays at
    # 1e-12 (relative tolerance is scale-invariant).
    atol = max(1e-12, 1e-12 * scale)
    np.testing.assert_allclose(
        np.asarray(q_L_a), np.asarray(q_L_b), rtol=1e-12, atol=atol,
        err_msg=(
            f"PPM limiter divergence at scale={scale} q_L.  "
            f"iter-879f cross-scale consistency check failed; "
            f"audit both `_ppm_reconstruct_1d` and `_ppm_limit`."))
    np.testing.assert_allclose(
        np.asarray(q_R_a), np.asarray(q_R_b), rtol=1e-12, atol=atol)
