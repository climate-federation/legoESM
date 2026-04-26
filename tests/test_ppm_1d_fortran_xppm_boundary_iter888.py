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


# ----------------------------------------------------------------------
# Iter-888b (Codex iter-888 stop-time fix): plumbing-reachability tests.
#
# Codex iter-888 stop-time review correctly flagged that the new kwarg
# was added to ``_ppm_1d`` only and was unreachable from any caller —
# i.e., it was dead code from the FB-chain transport perspective.
# iter-888b plumbs the kwarg through ``_xppm`` / ``_yppm`` / ``fv_tp_2d``
# / ``transport_step`` / ``_d_sw_native`` / ``fv3_forward_backward_step``
# / ``fv3_fb_sw_step`` so FB-chain callers can opt in.  These tests
# assert the plumbing is intact at every level.
# ----------------------------------------------------------------------


def test_iter888b_xppm_yppm_forward_kwarg():
    """``_xppm`` and ``_yppm`` accept and forward
    ``apply_fortran_xppm_boundary`` to ``_ppm_1d``.  Verified by
    constructing a simple input that triggers the override and
    comparing against ``_ppm_1d`` direct invocation.
    """
    from legoesm.core.fv_tp_2d import _xppm, _yppm

    n = 12
    M = 1
    rng = np.random.default_rng(8888)
    q = jnp.asarray(rng.normal(size=(6, n + 4, M)))
    crx = jnp.asarray(rng.normal(size=(6, n + 1, M)) * 0.3)

    # _xppm with kwarg
    fx_off = _xppm(q, crx, n, use_duogrid=False,
                   apply_fortran_xppm_boundary=False)
    fx_on = _xppm(q, crx, n, use_duogrid=False,
                  apply_fortran_xppm_boundary=True)
    diff_x = float(np.max(np.abs(np.asarray(fx_on) - np.asarray(fx_off))))
    assert diff_x > 1e-12, (
        f"_xppm kwarg is unreachable: fx_on equals fx_off "
        f"(max diff = {diff_x:.3e}).  The kwarg is silently ignored "
        f"or not forwarded to _ppm_1d.")

    # _yppm with kwarg
    cry = jnp.asarray(rng.normal(size=(6, n, n + 1)) * 0.3)
    q_y = jnp.asarray(rng.normal(size=(6, n, n + 4)))
    fy_off = _yppm(q_y, cry, n, use_duogrid=False,
                   apply_fortran_xppm_boundary=False)
    fy_on = _yppm(q_y, cry, n, use_duogrid=False,
                  apply_fortran_xppm_boundary=True)
    diff_y = float(np.max(np.abs(np.asarray(fy_on) - np.asarray(fy_off))))
    assert diff_y > 1e-12, (
        f"_yppm kwarg is unreachable: fy_on equals fy_off "
        f"(max diff = {diff_y:.3e}).  The kwarg is silently ignored "
        f"or not forwarded to _ppm_1d.")


def test_iter888b_fv_tp_2d_forwards_kwarg():
    """``fv_tp_2d`` accepts and forwards ``apply_fortran_xppm_boundary``
    to ``_xppm`` / ``_yppm``.  We can't easily verify the bit-equality
    of fv_tp_2d output without a grid, so this test checks the
    function signature and AST-asserts the keyword is forwarded.
    """
    import ast
    import inspect
    from pathlib import Path

    from legoesm.core.fv_tp_2d import fv_tp_2d

    sig = inspect.signature(fv_tp_2d)
    assert "apply_fortran_xppm_boundary" in sig.parameters, (
        f"fv_tp_2d signature missing apply_fortran_xppm_boundary kwarg.  "
        f"Got: {list(sig.parameters.keys())}.  Iter-888b expects "
        f"the kwarg to be plumbed through.")
    assert sig.parameters["apply_fortran_xppm_boundary"].default is False, (
        f"fv_tp_2d apply_fortran_xppm_boundary default is "
        f"{sig.parameters['apply_fortran_xppm_boundary'].default!r}, "
        f"expected False.")

    # AST: every _xppm/_yppm call inside fv_tp_2d's body must include
    # apply_fortran_xppm_boundary as a kwarg.
    src_path = (Path(__file__).resolve().parent.parent
                / "src" / "legoesm" / "core" / "fv_tp_2d.py")
    tree = ast.parse(src_path.read_text())
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "fv_tp_2d"),
        None,
    )
    assert fn is not None
    bad_calls = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("_xppm", "_yppm")):
            kw_names = {kw.arg for kw in node.keywords}
            if "apply_fortran_xppm_boundary" not in kw_names:
                bad_calls.append(
                    f"line {node.lineno}: {node.func.id}() missing "
                    f"apply_fortran_xppm_boundary kwarg; got "
                    f"{sorted(kw_names)}")
    assert not bad_calls, (
        "fv_tp_2d has _xppm/_yppm calls that do NOT forward the "
        "apply_fortran_xppm_boundary kwarg:\n  "
        + "\n  ".join(bad_calls))


def test_iter888b_transport_step_signature():
    """``transport_step`` accepts ``apply_fortran_xppm_boundary``
    and forwards it to ``fv_tp_2d``.
    """
    import inspect
    from legoesm.core.fv_tp_2d import transport_step

    sig = inspect.signature(transport_step)
    assert "apply_fortran_xppm_boundary" in sig.parameters, (
        f"transport_step signature missing apply_fortran_xppm_boundary; "
        f"got {list(sig.parameters.keys())}")


def test_iter888b_d_sw_native_signature():
    """``_d_sw_native`` accepts ``apply_fortran_xppm_boundary``."""
    import inspect
    from legoesm.core.fv3_sw_core import _d_sw_native

    sig = inspect.signature(_d_sw_native)
    assert "apply_fortran_xppm_boundary" in sig.parameters, (
        f"_d_sw_native signature missing apply_fortran_xppm_boundary; "
        f"got {list(sig.parameters.keys())}")


def test_iter888b_fb_chain_entry_points_signatures():
    """The FB-chain top entry points ``fv3_forward_backward_step`` and
    ``fv3_fb_sw_step`` accept ``apply_fortran_xppm_boundary``.
    """
    import inspect
    from legoesm.core.fv3_sw_core import (
        fv3_forward_backward_step, fv3_fb_sw_step)

    for fn in (fv3_forward_backward_step, fv3_fb_sw_step):
        sig = inspect.signature(fn)
        assert "apply_fortran_xppm_boundary" in sig.parameters, (
            f"{fn.__name__} signature missing apply_fortran_xppm_boundary; "
            f"got {list(sig.parameters.keys())}")


def test_iter888b_fb_chain_end_to_end_kwarg_changes_output():
    """End-to-end behavioural check: invoking the FB chain
    ``fv3_fb_sw_step`` with ``apply_fortran_xppm_boundary=True`` must
    produce DIFFERENT output from the default-OFF call.  This catches
    a regression where the kwarg is propagated through every signature
    but silently dropped before reaching ``_ppm_1d``.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.core.fv3_sw_core import fv3_fb_sw_step
    import sys
    sys.path.insert(0, "tests")
    from test_cases.williamson import williamson_test2

    n = 12  # small grid for fast test
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    dt = 60.0  # small timestep for stability
    h_off, ud_off, vd_off = fv3_fb_sw_step(
        sw.h.data, u_d, v_d, sw.h_s.data, cdgrid, dt,
        apply_fortran_xppm_boundary=False)
    h_on, ud_on, vd_on = fv3_fb_sw_step(
        sw.h.data, u_d, v_d, sw.h_s.data, cdgrid, dt,
        apply_fortran_xppm_boundary=True)

    diff_h = float(np.max(np.abs(np.asarray(h_on) - np.asarray(h_off))))
    diff_u = float(np.max(np.abs(np.asarray(ud_on) - np.asarray(ud_off))))
    diff_v = float(np.max(np.abs(np.asarray(vd_on) - np.asarray(vd_off))))
    max_diff = max(diff_h, diff_u, diff_v)

    assert max_diff > 1e-12, (
        f"fv3_fb_sw_step end-to-end output bit-identical for "
        f"apply_fortran_xppm_boundary in (False, True): "
        f"max diff h={diff_h:.3e}, u_d={diff_u:.3e}, v_d={diff_v:.3e}.  "
        f"The kwarg is plumbed through signatures but silently dropped "
        f"somewhere — check that every transport_step / fv_tp_2d / "
        f"_xppm / _yppm / _ppm_1d call site forwards the kwarg.")


# ----------------------------------------------------------------------
# Iter-888c (Codex iter-888b stop-time fix): canonical FB MODEL config
# reachability.  Codex iter-888b correctly flagged that even with
# function-level plumbing through fv3_fb_sw_step, the canonical FB
# MODEL class (`FV3FBShallowWaterModel`) did NOT forward the kwarg
# from its config.  iter-888c surfaces `apply_fortran_xppm_boundary`
# as a `CDGridShallowWaterConfig` field and forwards it from
# `FV3FBShallowWaterModel.step` to `fv3_fb_sw_step`.
# ----------------------------------------------------------------------


def test_iter888c_config_field_default_off():
    """`CDGridShallowWaterConfig` exposes ``apply_fortran_xppm_boundary``
    as a field with default False.  Locked by the iter-873 sentinel
    via the inventory list (see test_fortran_fidelity_default_flags_iter873.py).
    """
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig)

    cfg = CDGridShallowWaterConfig()
    assert hasattr(cfg, "apply_fortran_xppm_boundary"), (
        "CDGridShallowWaterConfig is missing apply_fortran_xppm_boundary.")
    assert cfg.apply_fortran_xppm_boundary is False, (
        f"apply_fortran_xppm_boundary default is "
        f"{cfg.apply_fortran_xppm_boundary!r}; expected False.")


def test_iter888c_fb_model_step_forwards_config_field():
    """`FV3FBShallowWaterModel.step` forwards the
    ``apply_fortran_xppm_boundary`` config field to ``fv3_fb_sw_step``.
    Verified by an end-to-end behavioural diff: a model with config
    flag True must produce different output from a model with the
    config flag False on the same input.  This catches a regression
    where the field is added to config but not threaded through the
    model class.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        FV3FBShallowWaterModel,
        FV3EdgeShallowWaterState,
    )
    import sys
    sys.path.insert(0, "tests")
    from test_cases.williamson import williamson_test2

    n = 12
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

    def _step_with_flag(flag):
        cfg = CDGridShallowWaterConfig(
            apply_fortran_xppm_boundary=flag)
        model = FV3FBShallowWaterModel(grid, config=cfg)
        cdgrid = model.cdgrid
        u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
        v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
        state = FV3EdgeShallowWaterState(
            h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
        model.set_initial_mass(state)
        return model.step(state, 60.0)

    state_off = _step_with_flag(False)
    state_on = _step_with_flag(True)

    diff_h = float(np.max(np.abs(np.asarray(state_on.h)
                                  - np.asarray(state_off.h))))
    diff_u = float(np.max(np.abs(np.asarray(state_on.u_d)
                                  - np.asarray(state_off.u_d))))
    diff_v = float(np.max(np.abs(np.asarray(state_on.v_d)
                                  - np.asarray(state_off.v_d))))
    max_diff = max(diff_h, diff_u, diff_v)

    assert max_diff > 1e-12, (
        f"FV3FBShallowWaterModel.step does NOT forward "
        f"config.apply_fortran_xppm_boundary to fv3_fb_sw_step.  "
        f"Output identical with config flag False vs True: "
        f"max diff h={diff_h:.3e}, u_d={diff_u:.3e}, v_d={diff_v:.3e}.  "
        f"Verify the model class threads the field through the "
        f"fv3_fb_sw_step call.")


def test_iter889_production_fv3edge_model_responds_to_flag():
    """Iter-889 (supersedes iter-888c "production inert" contract):
    `FV3EdgeShallowWaterModel` (default `use_experimental_csw=False`)
    now routes `apply_fortran_xppm_boundary` through `fv3_sw_tendencies`
    → `cgrid_mass_flux_divergence` → `_ppm_reconstruct_1d`, where
    iter-889 added the Fortran iord<7 cube-edge boundary overrides
    (tp_core.F90:357-369).  Setting the config flag True must produce
    DIFFERENT output from False on the same Williamson-2 IC.

    Pre-iter-889 this test asserted the OPPOSITE (production unaffected
    by the flag), reflecting iter-888c's leaf-only production scope.
    iter-889 implements production-path support, so the contract
    flips: ON-path must change output; OFF-path preserves prior
    behaviour bit-for-bit (locked separately by the
    test_iter889_production_off_path_bit_identical_to_pre_iter889
    test below).
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        FV3EdgeShallowWaterModel,
        FV3EdgeShallowWaterState,
    )
    import sys
    sys.path.insert(0, "tests")
    from test_cases.williamson import williamson_test2

    n = 12
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

    def _step_with_flag(flag):
        cfg = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            div_damp=8.0 * 1.5e7 * (48.0 / n) ** 2,
            boundary_fix=True,
            damp_v=0.06,
            nord_v=2,
            apply_fortran_xppm_boundary=flag)
        model = FV3EdgeShallowWaterModel(grid, config=cfg)
        cdgrid = model.cdgrid
        u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
        v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
        state = FV3EdgeShallowWaterState(
            h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
        model.set_initial_mass(state)
        return model.step(state, 60.0)

    state_off = _step_with_flag(False)
    state_on = _step_with_flag(True)

    diff_h = float(np.max(np.abs(np.asarray(state_on.h)
                                  - np.asarray(state_off.h))))
    diff_u = float(np.max(np.abs(np.asarray(state_on.u_d)
                                  - np.asarray(state_off.u_d))))
    diff_v = float(np.max(np.abs(np.asarray(state_on.v_d)
                                  - np.asarray(state_off.v_d))))
    max_diff = max(diff_h, diff_u, diff_v)
    assert max_diff > 1e-12, (
        f"FV3EdgeShallowWaterModel (production CDGrid path) is NOT "
        f"reachable via apply_fortran_xppm_boundary: max diff "
        f"h={diff_h:.3e}, u_d={diff_u:.3e}, v_d={diff_v:.3e}.  "
        f"iter-889 plumbed the kwarg through fv3_sw_tendencies → "
        f"cgrid_mass_flux_divergence → _ppm_reconstruct_1d.  Either "
        f"the kwarg is silently dropped or the Fortran-faithful "
        f"boundary overrides are not activating.")


def test_iter889_w2_legacy_is_known_worse_on_flag():
    """Iter-889 known-worse sentinel (mirrors iter-766's
    `test_fortran_a2b_corner_avg_is_known_worse` pattern).

    Setting `apply_fortran_xppm_boundary=True` on the canonical W2
    LEGACY matrix config makes W2 v_north Linf at C36 1-day SUBSTANTIALLY
    WORSE than the default-OFF empirical baseline.

    iter-889 measurement at C36 1-day:
      OFF (default 4-pt centred PPM): v_north Linf = 0.189 m/s
      ON  (Fortran iord<7 boundary): v_north Linf = 0.756 m/s
    Ratio ON/OFF ≈ 4.0x worse.

    Interpretation.  Fortran's iord<7 cube-edge boundary formulas
    (s11/s14/s15 + uniform-grid 4-point xt at the cube-face edge,
    tp_core.F90:357-369) are STRICTLY MORE Fortran-faithful than our
    default centred 4-point PPM at boundary faces.  But our hybrid
    A-L+RK3+boundary_fix production stack does NOT match Fortran's
    full numerical environment; replacing only the PPM boundary
    formula amplifies a tension between operator-split and Fortran-
    faithful reconstruction at cube vertices.  Same pattern as
    iter-766's a2b corner average and iter-769's skip_corners.

    This is a Fortran-fidelity vs empirical-W2 tension iter-889
    documents but does NOT resolve.  Default-OFF preserves the
    empirically-better current behaviour while making the Fortran-
    faithful path REACHABLE for callers (e.g., FB-chain stabilisation
    tests, future iters that land complementary Fortran-faithful
    changes).

    This sentinel pins the known-worse outcome so:
    - A future change that enables the flag by default in the matrix
      runner shifts the production W2 baseline 4x and would fail
      iter-873's "matrix runner does not activate flags" test.
    - A future repair that closes the ON/OFF gap (i.e., the Fortran-
      faithful path becomes the empirical winner) would fail this
      test, prompting an audit + potentially re-baselining iter-873.
    - A future regression that silently disables the kwarg threading
      would also fail this test (since ON would equal OFF).
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        FV3EdgeShallowWaterModel,
        FV3EdgeShallowWaterState,
    )
    import sys
    sys.path.insert(0, "tests")
    from test_cases.williamson import williamson_test2

    n = 36
    dt = 300.0
    n_steps = int(86400 / dt)  # 1 day
    div_damp = 8.0 * 1.5e7 * (48.0 / n) ** 2
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

    def _run_and_measure(flag):
        cfg = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            div_damp=div_damp,
            boundary_fix=True,
            damp_v=0.06,
            nord_v=2,
            apply_fortran_xppm_boundary=flag)
        model = FV3EdgeShallowWaterModel(grid, config=cfg)
        cdgrid = model.cdgrid
        u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
        v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
        state = FV3EdgeShallowWaterState(
            h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
        model.set_initial_mass(state)
        for _ in range(n_steps):
            state = model.step(state, dt)

        from legoesm.grids.cubed_sphere_cdgrid import (
            cell_centre_angles_from_4edge)
        ca, sa = cell_centre_angles_from_4edge(cdgrid)
        u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                      + np.asarray(state.u_d)[:, :, 1:])
        v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                      + np.asarray(state.v_d)[:, 1:, :])
        v_north = (np.asarray(sa) * u_cc + np.asarray(ca) * v_cc)
        return float(np.max(np.abs(v_north)))

    v_off = _run_and_measure(flag=False)
    v_on = _run_and_measure(flag=True)

    # Pin OFF baseline near iter-888's measurement (~0.19 m/s direct
    # cell-centre measurement; sentinel test gives ~0.16 m/s through
    # latlon regridding).  0.30 ceiling allows for resolution drift.
    assert v_off < 0.30, (
        f"OFF (default) baseline v_north Linf = {v_off:.3e} m/s "
        f"drifted above 0.30 m/s — the iter-888 measurement (0.189) "
        f"no longer applies.  Re-baseline the iter-889 known-worse "
        f"sentinel.")

    # Pin ON path to be materially worse than OFF (>= 2x gap).
    # iter-889 measured ratio ~4.0x.  A smaller ratio means either
    # the boundary formula was changed or the OFF baseline drifted up.
    ratio = v_on / v_off
    assert ratio > 2.0, (
        f"apply_fortran_xppm_boundary=True v_north Linf={v_on:.3e} "
        f"produced ratio {ratio:.2f}x over OFF baseline ({v_off:.3e}) "
        f"— UNEXPECTEDLY SMALL gap.  iter-889 falsified this path at "
        f"~4.0x blowup.  A new smaller ratio means either:\n"
        f"  (a) the iord<7 Fortran boundary path has been repaired — "
        f"re-examine whether it now reduces W2 mode A and could "
        f"replace the default centred 4-point reconstruction, OR\n"
        f"  (b) the opt-in was silently disabled — restore the kwarg "
        f"threading in _ppm_reconstruct_1d / cgrid_mass_flux_divergence "
        f"/ fv3_sw_tendencies / FV3EdgeShallowWaterModel.step per "
        f"iter-889.")
