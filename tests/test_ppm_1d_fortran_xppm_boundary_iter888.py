"""Iter-888: pin the Fortran s11/s14/s15 + 4-point boundary formula
implementation in `_ppm_1d` behind the new
`apply_fortran_xppm_boundary` kwarg (default False).

iter-887 (commit 4f04e76) documented the Fortran-fidelity gap in
``_ppm_1d``: Fortran ``tp_core.F90:614-628`` (left) and ``:632-647``
(right) implement a richer boundary procedure for the legacy non-
duogrid path using:

1. ``s11/s14/s15`` constants from ``tp_core.F90:58``:
   - ``s11 = 11/14, s14 = 4/7, s15 = 3/14``
2. A 4-point ``dxa``-weighted boundary edge value ``xt``
   (``tp_core.F90:616-617, 638-639``).  For uniform dxa this collapses
   to ``0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))``.
3. Clip ``xt`` to within ``min/max(q1(-1..2))``.
4. Set bl/br at indices 0/1/2 (left) and npx-2/npx-1/npx (right) from
   ``s11/s14/s15`` overrides + ``xt`` differences.

iter-888 implements the UNIFORM-GRID simplification (skips the
``dxa`` plumbing — that's a future iter) behind the new kwarg
``apply_fortran_xppm_boundary``.  Default False preserves the pre-
iter-888 behaviour bit-for-bit.

This test pins three properties:

A. **Default-OFF preserves prior behaviour.**  ``_ppm_1d`` called
   without the kwarg (or with kwarg=False) produces the SAME output
   as before iter-888 — the pre-iter-888 source semantics are
   preserved.  We capture the OFF baseline at runtime by computing
   the standard non-overridden bl/br via the same path with
   kwarg=False.  The kwarg=False path bit-equals the no-kwarg path.

B. **ON path applies the Fortran-faithful overrides.**  With kwarg=
   True, the bl/br at the 6 boundary cells (indices 0/1/2 and
   -3/-2/-1) DIFFER from the OFF baseline, and the differences match
   the Fortran formula prediction within numerical precision.

C. **ON path is no-op for duogrid.**  When ``use_duogrid=True``, the
   kwarg has no effect (Fortran's ``.not. (bounded_domain .or.
   duogrid)`` gate at line 612).

This guards against:
- A future regression that flips the default to True without an
  explicit ABI change (would break callers that rely on the
  pre-iter-888 numerics).
- A regression in the s11/s14/s15 formulas (e.g., constant typo,
  index shift).
- A regression that fires the override in the duogrid path.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.core.fv_tp_2d import _ppm_1d


def _make_test_input(n=8, M=4, seed=888):
    """Create a generic q with halo=2 padding."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(6, n + 4, M)))


def test_iter888_default_off_preserves_prior_behaviour():
    """Default-OFF (kwarg unset) and explicit kwarg=False must
    produce the SAME output bit-for-bit.  This guards against an
    accidental default-flip in a future commit.
    """
    n = 12
    M = 3
    q = _make_test_input(n=n, M=M, seed=8881)

    # No kwarg
    bl_default, br_default, qc_default = _ppm_1d(
        q, n, off_left=jnp.zeros((6, M)), off_right=jnp.zeros((6, M)),
        use_duogrid=False)

    # kwarg=False
    bl_off, br_off, qc_off = _ppm_1d(
        q, n, off_left=jnp.zeros((6, M)), off_right=jnp.zeros((6, M)),
        use_duogrid=False,
        apply_fortran_xppm_boundary=False)

    np.testing.assert_array_equal(np.asarray(bl_default), np.asarray(bl_off))
    np.testing.assert_array_equal(np.asarray(br_default), np.asarray(br_off))
    np.testing.assert_array_equal(np.asarray(qc_default), np.asarray(qc_off))


def test_iter888_on_path_applies_fortran_formula_at_left_boundary():
    """With kwarg=True (and not use_duogrid), the bl/br at LEFT
    boundary indices [0, 1, 2] must reflect the Fortran s11/s14/s15
    + 4-point xt formula on a uniform-spacing input.

    Test strategy: use a SPECIFIC input designed to make the override
    arithmetic predictable.  Use a smooth quadratic q so the iv=1
    extremum-flatten doesn't overwrite our overrides.
    """
    n = 12
    M = 1
    # Smooth quadratic q on extended domain — guarantees no extrema
    # so iv=1 doesn't flatten bl/br to zero, leaving the s11/s14/s15
    # overrides visible in the output.
    x = np.arange(-2, n + 2, dtype=np.float64)  # halo=2 indices
    q_1d = 1.0 + 0.1 * x + 0.01 * x ** 2  # monotone increasing
    q_np = np.broadcast_to(q_1d[None, :, None], (6, n + 4, M)).copy()
    q = jnp.asarray(q_np)

    bl_off, br_off, _ = _ppm_1d(
        q, n,
        use_duogrid=False,
        apply_fortran_xppm_boundary=False)
    bl_on, br_on, _ = _ppm_1d(
        q, n,
        use_duogrid=False,
        apply_fortran_xppm_boundary=True)

    # ON path must DIFFER from OFF path at the 6 boundary indices,
    # bit-equal at all interior indices.
    diff_bl = np.abs(np.asarray(bl_on) - np.asarray(bl_off))
    diff_br = np.abs(np.asarray(br_on) - np.asarray(br_off))

    # Maximum diff at boundary cells should be > 1e-10 (formula
    # genuinely changes the values).
    boundary_diff = max(
        float(diff_bl[:, 0, :].max()),
        float(diff_br[:, 0, :].max()),
        float(diff_bl[:, 1, :].max()),
        float(diff_br[:, 1, :].max()),
        float(diff_bl[:, 2, :].max()),
        # br[2] is unchanged by Fortran (line 628: br(2) = al(3) -
        # q1(2)), so we don't check it here.
    )
    assert boundary_diff > 1e-10, (
        f"iter-888 ON path produced output bit-identical to OFF at "
        f"left boundary (max diff = {boundary_diff:.3e}).  The "
        f"s11/s14/s15 override is invisible — either the kwarg is "
        f"silently ignored or the formula coincidentally matches "
        f"the pre-iter-888 output on this input.")

    # Far-interior cells (index 4..n-3) should be IDENTICAL.
    np.testing.assert_array_equal(
        np.asarray(bl_on)[:, 4:n - 3, :],
        np.asarray(bl_off)[:, 4:n - 3, :])
    np.testing.assert_array_equal(
        np.asarray(br_on)[:, 4:n - 3, :],
        np.asarray(br_off)[:, 4:n - 3, :])


def test_iter888_on_path_matches_fortran_formula_predictions():
    """With kwarg=True on a smooth quadratic input designed to NOT
    trigger the iv=1 limiter (no extrema, no opposite-sign bl/br),
    the s11/s14/s15 formula predictions must match the actual
    output.

    Uses Fortran tp_core.F90:614-628 explicit formulas (left side
    only — symmetric on the right).
    """
    n = 16
    M = 1
    # Smooth quadratic on extended domain.
    x = np.arange(-2, n + 2, dtype=np.float64)
    q_1d = 100.0 + 1.0 * x + 0.01 * x ** 2
    q_np = np.broadcast_to(q_1d[None, :, None], (6, n + 4, M)).copy()
    q = jnp.asarray(q_np)

    # Compute Fortran-formula predictions manually.
    # qe = pad(q, halo=1) → shape (6, n+6, M)
    qe = np.pad(q_np, [(0, 0), (1, 1), (0, 0)], mode='edge')
    # Compute dm via the same monotone limiter as _ppm_1d.
    xt_dm = 0.25 * (qe[:, 2:, :] - qe[:, :-2, :])
    qm = qe[:, 1:-1, :]
    q_hi = np.maximum(np.maximum(qe[:, :-2, :], qm), qe[:, 2:, :])
    q_lo = np.minimum(np.minimum(qe[:, :-2, :], qm), qe[:, 2:, :])
    dm = np.sign(xt_dm) * np.minimum(
        np.abs(xt_dm), np.minimum(q_hi - qm, qm - q_lo))

    s11 = 11.0 / 14.0
    s14 = 4.0 / 7.0
    s15 = 3.0 / 14.0

    # LEFT predictions
    expected_bl_0 = s14 * dm[:, 0, :] + s11 * (qe[:, 1, :] - qe[:, 2, :])
    xt_L = (0.75 * (qe[:, 2, :] + qe[:, 3, :])
            - 0.25 * (qe[:, 1, :] + qe[:, 4, :]))
    q_lo_L = np.minimum(np.minimum(qe[:, 1, :], qe[:, 2, :]),
                         np.minimum(qe[:, 3, :], qe[:, 4, :]))
    q_hi_L = np.maximum(np.maximum(qe[:, 1, :], qe[:, 2, :]),
                         np.maximum(qe[:, 3, :], qe[:, 4, :]))
    xt_L = np.clip(xt_L, q_lo_L, q_hi_L)
    expected_br_0 = xt_L - qe[:, 2, :]
    expected_bl_1 = xt_L - qe[:, 3, :]
    xt2_L = (s15 * qe[:, 3, :] + s11 * qe[:, 4, :]
             - s14 * dm[:, 3, :])
    expected_br_1 = xt2_L - qe[:, 3, :]
    expected_bl_2 = xt2_L - qe[:, 4, :]

    bl, br, _ = _ppm_1d(q, n, use_duogrid=False,
                        apply_fortran_xppm_boundary=True)
    bl_np = np.asarray(bl)
    br_np = np.asarray(br)

    # On a smooth monotone-increasing quadratic, bl is negative-or-
    # zero and br is positive-or-zero → bl*br <= 0 → iv=1's
    # `is_ext = bl*br >= 0` triggers ONLY at the equality case.  In
    # general iv=1 may zero some cells.  To make the test robust we
    # compare cell-by-cell ONLY where iv=1 did NOT flatten (i.e.,
    # bl[k]*br[k] strictly < 0).  iv=1's `a6da` test then either
    # passes through bl/br or applies br=-2*bl / bl=-2*br rewrites.
    # To avoid getting tangled in those branches, use a TRACE of the
    # formula: re-apply iv=1 on the predicted bl/br and compare
    # against the actual output.
    from legoesm.core.fv_tp_2d import _pert_ppm

    # Predicted iv=1-applied bl/br at index 0:
    pred_bl_0_iv1, pred_br_0_iv1 = _pert_ppm(
        jnp.asarray(expected_bl_0), jnp.asarray(expected_br_0))
    # Predicted iv=1-applied bl/br at index 1:
    pred_bl_1_iv1, pred_br_1_iv1 = _pert_ppm(
        jnp.asarray(expected_bl_1), jnp.asarray(expected_br_1))
    # Predicted iv=1-applied bl/br at index 2: br[2] is UNCHANGED
    # from the standard al-based computation, so we don't have a
    # clean Fortran-formula prediction without re-running the
    # function.  Skip this index.

    np.testing.assert_allclose(
        bl_np[:, 0, :], np.asarray(pred_bl_0_iv1), rtol=1e-12, atol=1e-12,
        err_msg="bl[0] does not match Fortran s11/s14/s15 + iv=1 prediction")
    np.testing.assert_allclose(
        br_np[:, 0, :], np.asarray(pred_br_0_iv1), rtol=1e-12, atol=1e-12,
        err_msg="br[0] does not match Fortran 4-point xt + iv=1 prediction")
    np.testing.assert_allclose(
        bl_np[:, 1, :], np.asarray(pred_bl_1_iv1), rtol=1e-12, atol=1e-12,
        err_msg="bl[1] does not match Fortran 4-point xt + iv=1 prediction")
    np.testing.assert_allclose(
        br_np[:, 1, :], np.asarray(pred_br_1_iv1), rtol=1e-12, atol=1e-12,
        err_msg="br[1] does not match Fortran s11/s14/s15 (xt2) + iv=1 prediction")


def test_iter888_duogrid_path_unaffected():
    """When ``use_duogrid=True``, the kwarg has no effect (Fortran's
    ``.not. (bounded_domain .or. duogrid)`` gate at line 612).
    """
    n = 12
    M = 2
    q = _make_test_input(n=n, M=M, seed=8884)

    bl_default, br_default, _ = _ppm_1d(
        q, n, use_duogrid=True,
        apply_fortran_xppm_boundary=False)
    bl_on, br_on, _ = _ppm_1d(
        q, n, use_duogrid=True,
        apply_fortran_xppm_boundary=True)

    np.testing.assert_array_equal(np.asarray(bl_default), np.asarray(bl_on))
    np.testing.assert_array_equal(np.asarray(br_default), np.asarray(br_on))


def test_iter888_constants_match_fortran():
    """AST scan: the iter-888 source must contain the literal Fortran
    constants ``11.0 / 14.0`` (s11), ``4.0 / 7.0`` (s14), and
    ``3.0 / 14.0`` (s15) inside ``_ppm_1d``.

    A regression that drifts the constants (e.g., ``11.0/16.0``)
    silently breaks Fortran fidelity without affecting the formula
    structure.  This sentinel catches such drift.
    """
    import ast
    from pathlib import Path

    src_path = (Path(__file__).resolve().parent.parent
                / "src" / "legoesm" / "core" / "fv_tp_2d.py")
    tree = ast.parse(src_path.read_text())

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "_ppm_1d"),
        None,
    )
    assert fn is not None, "Could not find _ppm_1d in fv_tp_2d.py"

    # Walk the AST and collect all literal float pairs from BinOp(/)
    # — `11.0 / 14.0` parses as BinOp(left=Constant(11.0), op=Div,
    # right=Constant(14.0)).
    found_pairs = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                and isinstance(node.left, ast.Constant)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.left.value, (int, float))
                and isinstance(node.right.value, (int, float))):
            found_pairs.add(
                (float(node.left.value), float(node.right.value)))

    expected_constants = {
        (11.0, 14.0): "s11",
        (4.0, 7.0): "s14",
        (3.0, 14.0): "s15",
    }
    missing = []
    for pair, name in expected_constants.items():
        if pair not in found_pairs:
            missing.append(f"{name} ({pair[0]}/{pair[1]})")
    assert not missing, (
        f"iter-888 constant regression: ``_ppm_1d`` source does not "
        f"contain BinOp(Div) literal pair(s) {missing}.  The Fortran "
        f"reference constants are s11=11/14, s14=4/7, s15=3/14 "
        f"(tp_core.F90:58).  A regression that changed the literal "
        f"form (e.g., to ``s11 = 0.7857142857142857``) would also "
        f"trip this sentinel — that's intentional, since changing "
        f"from rational form to decimal increases the risk of "
        f"silent precision loss vs the Fortran exact constants.")
