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
    from legoesm.core.fv_tp_2d import pert_ppm

    # Predicted iv=1-applied bl/br at index 0:
    pred_bl_0_iv1, pred_br_0_iv1 = pert_ppm(
        jnp.asarray(expected_bl_0), jnp.asarray(expected_br_0))
    # Predicted iv=1-applied bl/br at index 1:
    pred_bl_1_iv1, pred_br_1_iv1 = pert_ppm(
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
    from tests.legoesm_paths import legoesm_source_path

    src_path = legoesm_source_path("core/fv_tp_2d.py")
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
    from tests.legoesm_paths import legoesm_source_path

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
    src_path = legoesm_source_path("core/fv_tp_2d.py")
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
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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
    """Iter-892 (Codex iter-891b stop-time follow-up): RENAMED from
    "known-worse" to "known-IMPROVED".  The pre-iter-892 measurement
    of 4× degradation (`apply_fortran_xppm_boundary=True` makes W2
    v_north Linf 4× worse) was an artifact of an OFF-BY-ONE BUG in
    iter-889's left-side boundary override (used q_pad cells shifted
    +1 from Fortran).  iter-892 corrected the off-by-one; the
    Fortran-faithful boundary formula now PRODUCES BETTER W2
    v_north Linf than the default centred 4-point PPM.

    iter-892 measurement at C36 1-day W2 LEGACY canonical config:
      OFF (default 4-pt centred PPM):    v_north Linf = 0.189 m/s
      ON  (Fortran iord<7 corrected):    v_north Linf = 0.152 m/s
    Ratio ON/OFF ≈ 0.80x — ON is **better** by 19.6% on Linf and
    10.1% on L2.

    Interpretation.  Fortran's iord<7 cube-edge boundary formulas
    (`tp_core.F90:357-369`: c1/c2/c3 mirror + 4-point xt clipped) are
    BOTH more Fortran-faithful AND empirically reduce W2 v_north
    Linf when correctly placed at q_face[2, 3, n+1, n+2] = al(1, 2,
    npx-1, npx).  The pre-iter-892 off-by-one placed each formula
    one cell shifted, producing a non-Fortran formula that happened
    to amplify W2 mode-A 4×.

    This sentinel locks the iter-892 IMPROVEMENT so:
    - A future regression that re-introduces the off-by-one would
      drift the ratio back toward 4×, failing the upper bound here.
    - A future code change that silently disables the kwarg threading
      would make ON equal OFF (ratio = 1.0), also failing.
    - A future improvement that closes the gap further (ratio < 0.5)
      would prompt a re-baseline.

    Iter-892 keeps the flag default-OFF on the production W2 matrix
    config.  Enabling it by default is iter-893+ candidate work
    (would shift the W2 sentinel baseline + several iter-873 inputs).
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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
    # cell-centre measurement).  0.30 ceiling allows for resolution drift.
    assert v_off < 0.30, (
        f"OFF (default) baseline v_north Linf = {v_off:.3e} m/s "
        f"drifted above 0.30 m/s — the iter-888 measurement (0.189) "
        f"no longer applies.  Re-baseline the iter-892 known-improved "
        f"sentinel.")

    # iter-892 known-improved: ON path is 19.6% BETTER than OFF
    # (ratio ≈ 0.80).  Pin between 0.5 and 0.95.  Below 0.5 means
    # an unexpectedly-large improvement (audit needed); above 0.95
    # means the improvement collapsed (off-by-one regression or
    # kwarg threading silently disabled).
    ratio = v_on / v_off
    assert 0.5 < ratio < 0.95, (
        f"apply_fortran_xppm_boundary=True v_north Linf={v_on:.3e} "
        f"produced ratio {ratio:.3f}x over OFF baseline ({v_off:.3e}) "
        f"— OUTSIDE the iter-892 expected range [0.5, 0.95].  "
        f"Possible causes:\n"
        f"  (a) ratio >= 0.95: the iter-892 off-by-one fix was reverted, "
        f"or the kwarg threading was silently disabled.  Verify the "
        f"override block in `_ppm_reconstruct_1d` places formulas at "
        f"q_face[2, 3, n+1, n+2] (NOT q_face[1, 2, 3, n+1, n+2, n+3]).\n"
        f"  (b) ratio <= 0.5: an unexpectedly-large W2 improvement.  "
        f"Audit whether other Fortran-fidelity changes co-landed with "
        f"this measurement.")


def test_iter889b_duogrid_bypasses_boundary_override():
    """Iter-889b (Codex iter-889 stop-time fix): on duogrid /
    bounded-domain grids the iter-889 production-path boundary
    override must be BYPASSED, matching Fortran's gate at
    `tp_core.F90:333` / `:357`:
        if ( .not. (bounded_domain .or. duogrid) .and. grid_type<3 )

    Pre-iter-889b `cgrid_mass_flux_divergence` forwarded
    `apply_fortran_xppm_boundary` unconditionally to
    `_ppm_reconstruct_1d`.  When a user enabled the flag on a
    duogrid grid, the legacy non-duogrid boundary formula would fire
    incorrectly (the cross-face halo from `pad_halo_vector` already
    delivers Fortran-faithful neighbour-face values, so the legacy
    s11/s14/s15-style override is wrong on duogrid).

    iter-889b adds an `effective_xppm_boundary = apply_fortran_xppm_boundary
    AND not cdgrid.base.bounded_domain` gate inside
    `cgrid_mass_flux_divergence`.

    This test verifies: setting `apply_fortran_xppm_boundary=True` on
    a DUOGRID grid produces output bit-identical to setting it False
    (i.e., the gate bypasses the override exactly as Fortran does).
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        FV3EdgeShallowWaterModel,
        FV3EdgeShallowWaterState,
    )
    import sys
    sys.path.insert(0, "tests")
    from test_cases.williamson import williamson_test2

    n = 12
    # Use duogrid mode so bounded_domain is True
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    assert grid.bounded_domain, (
        "Test setup error: duogrid grid did not produce "
        "bounded_domain=True; iter-889b gate test is invalid.")

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

    def _step_with_flag(flag):
        cfg = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            div_damp=0.0,
            boundary_fix=False,  # boundary_fix is duogrid-disabled
            damp_v=0.0,
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

    # Duogrid output MUST be bit-identical: the gate bypasses the
    # override regardless of the flag value.
    np.testing.assert_array_equal(
        np.asarray(state_off.h), np.asarray(state_on.h),
        err_msg=(
            "Duogrid + apply_fortran_xppm_boundary=True produced "
            "different h than =False.  iter-889b gate "
            "(`not cdgrid.base.bounded_domain`) is broken — the "
            "Fortran-non-duogrid boundary override fires on duogrid, "
            "which is exactly the bug Codex flagged in iter-889 "
            "stop-time review."))
    np.testing.assert_array_equal(
        np.asarray(state_off.u_d), np.asarray(state_on.u_d))
    np.testing.assert_array_equal(
        np.asarray(state_off.v_d), np.asarray(state_on.v_d))


def test_iter889b_legacy_non_duogrid_still_responds_to_flag():
    """Sanity: on the legacy global cubed sphere (NOT bounded_domain),
    the iter-889b gate does NOT bypass — the flag still produces
    different output from the default.  Mirrors the iter-889 production
    behavioural test but explicitly contrasts with the iter-889b
    duogrid-bypass case above.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        FV3EdgeShallowWaterModel,
        FV3EdgeShallowWaterState,
    )
    import sys
    sys.path.insert(0, "tests")
    from test_cases.williamson import williamson_test2

    n = 12
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    assert not grid.bounded_domain, (
        "Test setup error: legacy non-duogrid grid produced "
        "bounded_domain=True; iter-889b gate test is invalid.")

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

    # Legacy (non-bounded-domain): output MUST differ
    diff_h = float(np.max(np.abs(np.asarray(state_on.h)
                                  - np.asarray(state_off.h))))
    diff_u = float(np.max(np.abs(np.asarray(state_on.u_d)
                                  - np.asarray(state_off.u_d))))
    diff_v = float(np.max(np.abs(np.asarray(state_on.v_d)
                                  - np.asarray(state_off.v_d))))
    max_diff = max(diff_h, diff_u, diff_v)
    assert max_diff > 1e-12, (
        f"Legacy non-bounded-domain + apply_fortran_xppm_boundary=True "
        f"produced output bit-identical to =False: "
        f"h={diff_h:.3e}, u_d={diff_u:.3e}, v_d={diff_v:.3e}.  "
        f"iter-889b gate is over-zealous — it should ONLY bypass on "
        f"duogrid/bounded-domain, NOT on the legacy global cubed "
        f"sphere where the override is the entire purpose of the flag.")


# ----------------------------------------------------------------------
# Iter-890 (Codex iter-889b stop-time follow-up): close the FB-chain
# `_ppm_1d` `bounded_domain` gap.  Pre-iter-890 the FB-chain `_ppm_1d`
# gated on `not use_duogrid` only — regional/nested bounded-domain
# panels would incorrectly take the legacy global-cubed-sphere face
# overrides and the iv=1 limiter.  iter-890 adds a `bounded_domain`
# kwarg + a `fortran_legacy_face = (not use_duogrid) and (not
# bounded_domain)` gate inside `_ppm_1d`, plumbed via `_xppm`/`_yppm`
# from `fv_tp_2d` (which reads `cdgrid.base.bounded_domain`).
# ----------------------------------------------------------------------


def test_iter890_ppm_1d_bounded_domain_gates_legacy_overrides():
    """Direct unit test on `_ppm_1d`: when `bounded_domain=True` (or
    `use_duogrid=True`), the legacy global-face boundary overrides
    AND the iv=1 face-boundary limiter MUST be bypassed.  When both
    flags are False, the overrides fire (legacy global cubed sphere
    is in play).

    Strategy: monkey-patch `pert_ppm` to count invocations.  The
    legacy code path calls `pert_ppm` 6 times at boundary indices.
    bounded_domain=True or use_duogrid=True must skip those 6 calls.
    """
    from unittest import mock
    from legoesm.core import fv_tp_2d as fv_tp_2d_mod
    from legoesm.core.fv_tp_2d import _ppm_1d

    n = 8
    M = 4
    rng = np.random.default_rng(890)
    q = jnp.asarray(rng.normal(size=(6, n + 4, M)))

    n_calls = {"n": 0}

    def counting_pert_ppm(bl_slice, br_slice):
        n_calls["n"] += 1
        return bl_slice, br_slice

    with mock.patch.object(fv_tp_2d_mod, "pert_ppm", counting_pert_ppm):
        # Case A: legacy global cubed sphere (both flags False) — 6 calls
        n_calls["n"] = 0
        _ppm_1d(q, n, use_duogrid=False, bounded_domain=False)
        n_legacy = n_calls["n"]

        # Case B: bounded_domain=True (regional/nested) — 0 calls
        n_calls["n"] = 0
        _ppm_1d(q, n, use_duogrid=False, bounded_domain=True)
        n_bounded = n_calls["n"]

        # Case C: use_duogrid=True — 0 calls (existing iter-884 contract)
        n_calls["n"] = 0
        _ppm_1d(q, n, use_duogrid=True, bounded_domain=False)
        n_duogrid = n_calls["n"]

        # Case D: both True — 0 calls (also bypassed)
        n_calls["n"] = 0
        _ppm_1d(q, n, use_duogrid=True, bounded_domain=True)
        n_both = n_calls["n"]

    assert n_legacy == 6, (
        f"Legacy global cubed sphere case: expected 6 pert_ppm calls "
        f"at boundary indices, got {n_legacy}.")
    assert n_bounded == 0, (
        f"bounded_domain=True case: expected 0 pert_ppm calls "
        f"(Fortran tp_core.F90:612 gate `.not. (bounded_domain .or. "
        f"duogrid)` must bypass the iv=1 limiter), got {n_bounded}.")
    assert n_duogrid == 0, (
        f"use_duogrid=True case: expected 0 pert_ppm calls (legacy "
        f"contract from iter-884), got {n_duogrid}.")
    assert n_both == 0, (
        f"Both flags True case: expected 0 pert_ppm calls, got {n_both}.")


def test_iter890_ppm_1d_bounded_domain_skips_position_aware_corrections():
    """When `bounded_domain=True`, the position-aware boundary `dm` and
    `al` corrections (lines 192-285 of `_ppm_1d`, gated on
    `fortran_legacy_face`) MUST also be bypassed.  This catches a
    regression where the iv=1 gate was widened but the dm/al gates
    were missed.
    """
    from legoesm.core.fv_tp_2d import _ppm_1d

    n = 8
    M = 1
    rng = np.random.default_rng(8902)
    q = jnp.asarray(rng.normal(size=(6, n + 4, M)))
    # Provide non-trivial offsets so the position-aware corrections
    # would fire if the gate were missing.
    off_left = jnp.full((6, M), 0.3)
    off_right = jnp.full((6, M), 0.4)
    off_left_d1 = jnp.full((6, M), 0.6)
    off_right_d1 = jnp.full((6, M), 0.7)

    # Legacy: position-aware corrections fire
    bl_legacy, br_legacy, _ = _ppm_1d(
        q, n, off_left, off_right, off_left_d1, off_right_d1,
        use_duogrid=False, bounded_domain=False)
    # bounded_domain=True: same call signature, but corrections
    # must be bypassed
    bl_bounded, br_bounded, _ = _ppm_1d(
        q, n, off_left, off_right, off_left_d1, off_right_d1,
        use_duogrid=False, bounded_domain=True)

    # The two outputs MUST differ at boundary cells (since the
    # legacy corrections are bypassed in case B but applied in case A).
    diff_bl = float(np.max(np.abs(np.asarray(bl_legacy)
                                   - np.asarray(bl_bounded))))
    diff_br = float(np.max(np.abs(np.asarray(br_legacy)
                                   - np.asarray(br_bounded))))
    assert max(diff_bl, diff_br) > 1e-10, (
        f"`_ppm_1d(bounded_domain=True)` produced output bit-identical "
        f"to `bounded_domain=False` (max bl diff={diff_bl:.3e}, max "
        f"br diff={diff_br:.3e}).  iter-890 expected the position-aware "
        f"dm/al corrections to be gated by `fortran_legacy_face`, but "
        f"they appear to fire regardless of `bounded_domain`.")

    # And: bounded_domain=True output MUST equal use_duogrid=True output
    # (both bypass all legacy face logic; offsets are unused either way).
    bl_dg, br_dg, _ = _ppm_1d(
        q, n, off_left, off_right, off_left_d1, off_right_d1,
        use_duogrid=True, bounded_domain=False)
    np.testing.assert_array_equal(
        np.asarray(bl_bounded), np.asarray(bl_dg),
        err_msg="bounded_domain=True != use_duogrid=True bl mismatch")
    np.testing.assert_array_equal(
        np.asarray(br_bounded), np.asarray(br_dg),
        err_msg="bounded_domain=True != use_duogrid=True br mismatch")


def test_iter890_fv_tp_2d_forwards_bounded_domain_from_grid():
    """`fv_tp_2d` MUST read `bounded_domain = grid.bounded_domain`
    and forward it through `_xppm` / `_yppm` to `_ppm_1d`.

    AST scan: every `_xppm`/`_yppm` call inside `fv_tp_2d` must
    forward `bounded_domain` as a kwarg.  Catches a regression where
    the kwarg is added to the leaf but a future refactor drops it
    from the call-site forwarding.
    """
    import ast
    from tests.legoesm_paths import legoesm_source_path

    src_path = legoesm_source_path("core/fv_tp_2d.py")
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
            if "bounded_domain" not in kw_names:
                bad_calls.append(
                    f"line {node.lineno}: {node.func.id}() missing "
                    f"bounded_domain kwarg; got {sorted(kw_names)}")
    assert not bad_calls, (
        "fv_tp_2d has _xppm/_yppm calls that do NOT forward the "
        "bounded_domain kwarg:\n  " + "\n  ".join(bad_calls))


def test_iter890_fb_chain_duogrid_no_op_with_or_without_flag():
    """End-to-end FB-chain regression: the iter-890 `bounded_domain`
    plumbing must make `FV3FBShallowWaterModel` on a DUOGRID grid
    bypass the legacy global-face boundary specials regardless of
    whether `apply_fortran_xppm_boundary` is True or False.

    Mirrors iter-889b's `test_iter889b_duogrid_bypasses_boundary_override`
    but for the FB chain (FV3FBShallowWaterModel.step) instead of
    production (FV3EdgeShallowWaterModel.step).
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        FV3FBShallowWaterModel,
        FV3EdgeShallowWaterState,
    )
    import sys
    sys.path.insert(0, "tests")
    from test_cases.williamson import williamson_test2

    n = 12
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    assert grid.bounded_domain
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

    np.testing.assert_array_equal(
        np.asarray(state_off.h), np.asarray(state_on.h),
        err_msg=("FB chain (FV3FBShallowWaterModel) duogrid + flag=True "
                 "produced different h than flag=False.  iter-890 gate "
                 "(`bounded_domain=True` short-circuits the legacy "
                 "boundary overrides) is broken or unreachable through "
                 "the FB-chain plumbing."))
    np.testing.assert_array_equal(
        np.asarray(state_off.u_d), np.asarray(state_on.u_d))
    np.testing.assert_array_equal(
        np.asarray(state_off.v_d), np.asarray(state_on.v_d))


def test_iter890c_fv_tp_2d_no_crash_with_panel_offsets():
    """Iter-890c (Codex iter-890b stop-time fix): `fv_tp_2d` MUST
    handle `cdgrid` instances where `halo_interp_offsets_h2 is None`
    without crashing on the offset extraction.

    The full regional FB-chain transport pipeline has additional
    incompleteness layers that iter-890c does NOT fix — most notably
    `create_cubed_sphere_cdgrid(panel)` produces global-cubed-sphere
    shaped metrics (`rdxa.shape == (6, n, n)`) on a 1-face panel
    (`area.shape == (1, n, n)`), so `compute_transport_quantities`
    crashes upstream of `fv_tp_2d` with a broadcasting error.  iter-
    890c scope: `fv_tp_2d` itself no longer crashes when given
    pre-computed transport quantities AND `offsets_h2 is None`.

    This test exercises iter-890c's contract narrowly: directly
    invoke `_ppm_1d`-via-`_xppm` with `offsets_h2 is None`-equivalent
    inputs (off_left=None etc., bounded_domain=True) and assert no
    exception is raised.  The end-to-end panel pipeline crash from
    `compute_transport_quantities` is a separate, deeper deferred
    item.
    """
    from legoesm.core.fv_tp_2d import _xppm

    n = 12
    M = 1
    rng = np.random.default_rng(890)
    q = jnp.asarray(rng.normal(size=(6, n + 4, M)))
    crx = jnp.asarray(rng.normal(size=(6, n + 1, M)) * 0.3)

    # Simulate the iter-890c pathway: offsets are None (regional),
    # bounded_domain=True (Fortran-faithful gate inside _ppm_1d
    # bypasses the legacy face-boundary specials).
    fx = _xppm(q, crx, n,
               off_left=None, off_right=None,
               off_left_d1=None, off_right_d1=None,
               use_duogrid=False,
               bounded_domain=True,
               apply_fortran_xppm_boundary=False)

    assert jnp.all(jnp.isfinite(fx)), (
        "_xppm on regional-equivalent inputs (offsets=None, "
        "bounded_domain=True) produced non-finite fx — iter-890c "
        "conditional offset handling is broken.")
    assert fx.shape == (6, n + 1, M), (
        f"Expected fx shape (6, {n+1}, {M}); got {fx.shape}")


# ----------------------------------------------------------------------
# Iter-891 — parallel `ppm_edge_values` Fortran iord<7 boundary
# overrides (tp_core.F90:357-369).  Mirrors iter-889's pattern but for
# `operators_fv.py:ppm_edge_values`, which is used by the lat-lon /
# 3D / non-cubed-sphere PPM transport paths and (via
# `_ppm_reconstruct_x` / `_y` / `fv_flux_divergence`) the cubed-sphere
# `fv_flux_divergence` path.  Production W2 (`fv3_sw_tendencies`) and
# FB chain (`_d_sw_native`) do NOT use this code; the Fortran-fidelity
# gain is locked behind a default-OFF kwarg and reachable only by
# direct callers.
# ----------------------------------------------------------------------


def test_iter891_default_off_preserves_prior_behaviour():
    """Iter-891: `ppm_edge_values` called without the new kwarg or
    with `apply_fortran_xppm_boundary=False` must produce output
    bit-identical to pre-iter-891.  Default-OFF preservation.
    """
    from legoesm.core.operators_fv import ppm_edge_values

    n = 12
    K = 3
    rng = np.random.default_rng(891)
    q = jnp.asarray(rng.normal(size=(6, n + 4, K)))

    q_hat_default = ppm_edge_values(q)
    q_hat_off = ppm_edge_values(q, apply_fortran_xppm_boundary=False)

    np.testing.assert_array_equal(
        np.asarray(q_hat_default), np.asarray(q_hat_off))


def test_iter891_on_path_overrides_4_boundary_indices():
    """Iter-891b (Codex iter-891 stop-time fix): `ppm_edge_values`
    with kwarg=True AND `n_interior` provided MUST overwrite the
    4 cube-edge `q_hat` indices [1, 2, n_interior, n_interior+1]
    corresponding to Fortran al(1), al(2), al(npx-1), al(npx).

    Iter-891b correction: pre-iter-891b iter-891 used indices
    [1, 2, 3, n+1, n+2] which is OFF-BY-ONE — the al(i) → q_hat[i]
    mapping (via q_hat[k] = face between q_1d[k] and q_1d[k+1] =
    face between q1(k-1) and q1(k) = al(k)).  iter-891 inflicted
    each formula at a position one cell to the right of where
    Fortran places it; iter-891b corrects.

    Fortran al(0) (would be at q_hat[0]) and al(npx+1) (would be at
    q_hat[n_int+2]) are NOT overridden — they require q1(-2) and
    q1(npx+2) respectively, which are halo depth 2 cells unavailable
    with our halo=2 input.  Those slots remain at their default
    boundary 0.5*(q_1d[outer halo]+q_1d[inner halo]) values.

    Indices outside the cube-edge override (the standard 4th-order
    interior) MUST be identical.
    """
    from legoesm.core.operators_fv import ppm_edge_values

    n = 16
    K = 1
    rng = np.random.default_rng(8911)
    q = jnp.asarray(rng.normal(size=(6, n + 4, K)))

    q_hat_off = ppm_edge_values(q, apply_fortran_xppm_boundary=False)
    q_hat_on = ppm_edge_values(q, apply_fortran_xppm_boundary=True,
                                 n_interior=n)

    # Indices that MUST differ (al(1), al(2), al(npx-1), al(npx))
    overridden_indices = [1, 2, n, n + 1]
    for k in overridden_indices:
        diff = float(np.max(np.abs(np.asarray(q_hat_on)[:, k, :]
                                    - np.asarray(q_hat_off)[:, k, :])))
        assert diff > 1e-12, (
            f"q_hat[{k}] (Fortran-faithful boundary override slot) "
            f"matches the default 4th-order value (diff={diff:.3e}); "
            f"the iter-891b override is invisible at this index.")

    # Indices that MUST be identical (deep interior + the al(0)/al(npx+1)
    # boundary halos that we cannot Fortran-faithfully override with
    # halo=2).
    untouched_indices = [0] + list(range(3, n)) + [n + 2]
    for k in untouched_indices:
        np.testing.assert_array_equal(
            np.asarray(q_hat_on)[:, k, :],
            np.asarray(q_hat_off)[:, k, :],
            err_msg=(
                f"q_hat[{k}] drifted under iter-891b — either the "
                f"iter-891b shift broke or the override leaked into "
                f"a slot it shouldn't.  Untouched slots: q_hat[0] "
                f"(al(0), needs halo=3), q_hat[3..n-1] (deep "
                f"interior), q_hat[n+2] (al(npx+1), needs halo=3)."))


def test_iter891_on_matches_fortran_formula_predictions():
    """Iter-891b: on a smooth quadratic input, the actual ON-path
    output bit-equals the Fortran iord<7 formula prediction at the
    4 corrected override indices [1, 2, n, n+1] = al(1, 2, npx-1, npx).

    Iter-891b verification: q_hat[k] = face between q_1d[k] and
    q_1d[k+1] = face between q1(k-1) and q1(k) = Fortran al(k).
    So al(1) → q_hat[1], al(2) → q_hat[2], al(npx-1) → q_hat[n],
    al(npx) → q_hat[n+1].

    On the LEFT, al(1) uses q_1d[0..3] = q1(-1..2):
        xt = 0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))
        clipped to min/max(q1(-1..2)).
    al(2) uses q_1d[2..4] = q1(1..3):
        c3*q1(1) + c2*q1(2) + c1*q1(3).

    On the RIGHT, al(npx-1) uses q_1d[n-2..n] = q1(npx-3..npx-1):
        c1*q1(npx-3) + c2*q1(npx-2) + c3*q1(npx-1).
    al(npx) uses q_1d[n-1..n+2] = q1(npx-2..npx+1):
        xt = 0.75*(q1(npx-1)+q1(npx)) - 0.25*(q1(npx-2)+q1(npx+1))
        clipped to min/max(q1(npx-2..npx+1)).
    """
    from legoesm.core.operators_fv import ppm_edge_values

    n = 16
    K = 1
    x = np.arange(-2, n + 2, dtype=np.float64)
    q_1d_arr = 100.0 + 1.0 * x + 0.01 * x ** 2
    q_np = np.broadcast_to(q_1d_arr[None, :, None], (6, n + 4, K)).copy()
    q = jnp.asarray(q_np)

    c1 = -2.0 / 14.0
    c2 = 11.0 / 14.0
    c3 = 5.0 / 14.0

    # LEFT predictions
    # al(1): xt clipped, uses q_1d[0..3]
    xt_L = (
        0.75 * (q_np[..., 1, :] + q_np[..., 2, :])
        - 0.25 * (q_np[..., 0, :] + q_np[..., 3, :]))
    q_lo_L = np.minimum(np.minimum(q_np[..., 0, :], q_np[..., 1, :]),
                        np.minimum(q_np[..., 2, :], q_np[..., 3, :]))
    q_hi_L = np.maximum(np.maximum(q_np[..., 0, :], q_np[..., 1, :]),
                        np.maximum(q_np[..., 2, :], q_np[..., 3, :]))
    expected_al1 = np.clip(xt_L, q_lo_L, q_hi_L)
    # al(2): c3*q1(1) + c2*q1(2) + c1*q1(3) — uses q_1d[2, 3, 4]
    expected_al2 = (
        c3 * q_np[..., 2, :] + c2 * q_np[..., 3, :]
        + c1 * q_np[..., 4, :])

    # RIGHT predictions
    # al(npx-1): c1*q1(npx-3) + c2*q1(npx-2) + c3*q1(npx-1) — uses
    # q_1d[n-1, n, n+1] (since q1(j) = q_1d[j+1] and q1(npx-3) = q1(n-2)
    # = q_1d[n-1], etc.)
    expected_alnm1 = (
        c1 * q_np[..., n - 1, :] + c2 * q_np[..., n, :]
        + c3 * q_np[..., n + 1, :])
    # al(npx): xt clipped, uses q_1d[n, n+1, n+2, n+3] (q1(npx-2..npx+1))
    xt_R = (
        0.75 * (q_np[..., n + 1, :] + q_np[..., n + 2, :])
        - 0.25 * (q_np[..., n, :] + q_np[..., n + 3, :]))
    q_lo_R = np.minimum(
        np.minimum(q_np[..., n, :], q_np[..., n + 1, :]),
        np.minimum(q_np[..., n + 2, :], q_np[..., n + 3, :]))
    q_hi_R = np.maximum(
        np.maximum(q_np[..., n, :], q_np[..., n + 1, :]),
        np.maximum(q_np[..., n + 2, :], q_np[..., n + 3, :]))
    expected_aln = np.clip(xt_R, q_lo_R, q_hi_R)

    q_hat_on = ppm_edge_values(q, apply_fortran_xppm_boundary=True,
                                 n_interior=n)
    q_hat_np = np.asarray(q_hat_on)

    np.testing.assert_allclose(
        q_hat_np[..., 1, :], expected_al1,
        rtol=1e-12, atol=1e-12,
        err_msg="q_hat[1] != Fortran al(1) (4-point xt clipped)")
    np.testing.assert_allclose(
        q_hat_np[..., 2, :], expected_al2,
        rtol=1e-12, atol=1e-12,
        err_msg="q_hat[2] != Fortran al(2) (c3/c2/c1 formula)")
    np.testing.assert_allclose(
        q_hat_np[..., n, :], expected_alnm1,
        rtol=1e-12, atol=1e-12,
        err_msg="q_hat[n] != Fortran al(npx-1) (c1/c2/c3 formula)")
    np.testing.assert_allclose(
        q_hat_np[..., n + 1, :], expected_aln,
        rtol=1e-12, atol=1e-12,
        err_msg="q_hat[n+1] != Fortran al(npx) (4-point xt clipped)")


def test_iter891_constants_match_fortran():
    """AST scan: `ppm_edge_values` source contains BinOp(Div) literal
    pairs (-2.0, 14.0), (11.0, 14.0), (5.0, 14.0) for c1/c2/c3.
    """
    import ast
    from tests.legoesm_paths import legoesm_source_path

    src_path = legoesm_source_path("core/operators_fv.py")
    tree = ast.parse(src_path.read_text())
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "ppm_edge_values"),
        None,
    )
    assert fn is not None
    found_pairs = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                and isinstance(node.left, ast.UnaryOp)
                and isinstance(node.left.op, ast.USub)
                and isinstance(node.left.operand, ast.Constant)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.left.operand.value, (int, float))
                and isinstance(node.right.value, (int, float))):
            found_pairs.add(
                (-float(node.left.operand.value), float(node.right.value)))
        elif (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                and isinstance(node.left, ast.Constant)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.left.value, (int, float))
                and isinstance(node.right.value, (int, float))):
            found_pairs.add(
                (float(node.left.value), float(node.right.value)))

    expected = {(-2.0, 14.0): "c1", (11.0, 14.0): "c2", (5.0, 14.0): "c3"}
    missing = [n for p, n in expected.items() if p not in found_pairs]
    assert not missing, (
        f"`ppm_edge_values` is missing iter-891 constants {missing}.  "
        f"Fortran reference: tp_core.F90:63-65 c1=-2/14, c2=11/14, "
        f"c3=5/14.  Found pairs: {sorted(found_pairs)}")


def test_iter891_fv_flux_divergence_responds_to_flag_on_global_cubed_sphere():
    """`fv_flux_divergence` on a non-bounded-domain global cubed sphere
    MUST produce different output for kwarg ON vs OFF.  Validates the
    plumbing through `_ppm_reconstruct_x` / `_y` / `ppm_edge_values`.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.core.operators_fv import fv_flux_divergence

    n = 12
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    assert not grid.bounded_domain  # sanity: gate must be active

    rng = np.random.default_rng(8915)
    q = jnp.asarray(rng.normal(size=(6, n, n)))
    u = jnp.asarray(rng.normal(size=(6, n, n)) * 5.0)
    v = jnp.asarray(rng.normal(size=(6, n, n)) * 5.0)

    div_off = fv_flux_divergence(
        q, u, v, grid, apply_fortran_xppm_boundary=False)
    div_on = fv_flux_divergence(
        q, u, v, grid, apply_fortran_xppm_boundary=True)

    diff = float(np.max(np.abs(np.asarray(div_on) - np.asarray(div_off))))
    assert diff > 1e-12, (
        f"fv_flux_divergence kwarg unreachable: ON path bit-equals OFF "
        f"(max diff = {diff:.3e}).  Either the kwarg threading is "
        f"broken or the iter-891b bounded_domain gate is misfiring.")


def test_iter891b_fv_flux_divergence_duogrid_bypasses_override():
    """`fv_flux_divergence` on a duogrid (bounded_domain=True) grid
    MUST produce bit-identical output for kwarg ON vs OFF — Fortran's
    iord<7 boundary block is gated on `not (bounded_domain or
    duogrid)` (`tp_core.F90:333/357`).  iter-891 mirrors iter-889b's
    `effective_xppm_boundary = flag and not grid.bounded_domain` gate.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.core.operators_fv import fv_flux_divergence

    n = 12
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    assert grid.bounded_domain

    rng = np.random.default_rng(8916)
    q = jnp.asarray(rng.normal(size=(6, n, n)))
    u = jnp.asarray(rng.normal(size=(6, n, n)) * 5.0)
    v = jnp.asarray(rng.normal(size=(6, n, n)) * 5.0)

    div_off = fv_flux_divergence(
        q, u, v, grid, apply_fortran_xppm_boundary=False)
    div_on = fv_flux_divergence(
        q, u, v, grid, apply_fortran_xppm_boundary=True)

    np.testing.assert_array_equal(
        np.asarray(div_on), np.asarray(div_off),
        err_msg="fv_flux_divergence on duogrid responds to flag — gate broken")
