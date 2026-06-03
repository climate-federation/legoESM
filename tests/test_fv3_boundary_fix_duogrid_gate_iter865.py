"""Iter-865: production `boundary_fix` block must be bypassed in duogrid mode.

Per CLAUDE.md duogrid constraint #2 ("Legacy edge handling must be
disabled in duogrid mode via bounded_domain = .true."), the
non-FV3 `boundary_fix` smoothing in `fv3_sw_tendencies` (which
averages face-boundary tendency cells with adjacent-interior cells)
must be bypassed when `cdgrid.base.duogrid is not None`.  In duogrid
mode the cross-face halo placed by `pad_halo_vector` on
`du_cc/dv_cc` provides Fortran-faithful neighbour-face values at
face boundaries; the same-face smoothing is the legacy non-FV3 hack
that should not run.

Iter-865 adds the additional gate `cdgrid.base.duogrid is None` to
the `if boundary_fix and ... and n > 2:` test in operators_cdgrid.py.

Tests:
- LEGACY mode (use_duogrid=False) + boundary_fix=True: tendency at
  face-boundary cells differs from boundary_fix=False (smoothing
  fires).  Iter-511 documented this as load-bearing for W2 L2.
- DUOGRID mode (use_duogrid=True) + boundary_fix=True: tendency
  matches boundary_fix=False bit-for-bit (smoothing bypassed by
  iter-865 gate).

Production W2/W5/cosine bell sentinels use LEGACY mode, so these
remain unaffected by iter-865.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Iter-883: also enable x64 at runtime in case JAX was already
# initialized in float32 by an earlier conftest import.  The
# os.environ.setdefault above is for command-line invocation; the
# jax.config.update is the runtime-effective form.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.operators_cdgrid import fv3_sw_tendencies


def _build_state(n, seed):
    rng = np.random.default_rng(seed)
    h = jnp.asarray(rng.standard_normal((6, n, n)) + 1000.0)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    h_s = jnp.zeros((6, n, n))
    return h, u_d, v_d, h_s


def test_boundary_fix_fires_in_legacy_mode():
    """LEGACY (use_duogrid=False) + boundary_fix=True: face-boundary
    tendency cells differ from boundary_fix=False because the smoothing
    fires.  This test verifies that iter-865's gate did NOT silently
    disable the LEGACY-mode smoothing."""
    n = 8
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    h, u_d, v_d, h_s = _build_state(n, seed=11)

    div_damp = 1.5e7 * (48.0 / n) ** 2
    dh_off, du_off, dv_off = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, g=constants.g,
        div_damp=div_damp, hyperdiff_coeff=0.0,
        boundary_fix=False)
    dh_on, du_on, dv_on = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, g=constants.g,
        div_damp=div_damp, hyperdiff_coeff=0.0,
        boundary_fix=True)

    # The h-tendency `dh` is computed before boundary_fix and should
    # be unaffected.  But du/dv are smoothed at face-boundary cells —
    # they MUST differ between flag=True and flag=False in legacy mode.
    diff_u = float(np.max(np.abs(np.asarray(du_on) - np.asarray(du_off))))
    diff_v = float(np.max(np.abs(np.asarray(dv_on) - np.asarray(dv_off))))
    assert diff_u > 1e-10 or diff_v > 1e-10, (
        f"LEGACY mode boundary_fix=True must change du/dv at face-"
        f"boundary cells, but max|Δdu|={diff_u:.3e}, max|Δdv|={diff_v:.3e} "
        f"are both ≤ 1e-10.  iter-865 must not have silently disabled the "
        f"legacy-mode smoothing.")


def test_boundary_fix_bypassed_in_duogrid_mode():
    """DUOGRID (use_duogrid=True) + boundary_fix=True: face-boundary
    tendency cells match boundary_fix=False bit-for-bit because
    iter-865's gate bypasses the smoothing.  Per CLAUDE.md duogrid
    constraint #2, the legacy edge-handling hack must not fire in
    duogrid mode."""
    n = 8
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    assert cdgrid.base.duogrid is not None
    h, u_d, v_d, h_s = _build_state(n, seed=11)

    div_damp = 1.5e7 * (48.0 / n) ** 2
    dh_off, du_off, dv_off = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, g=constants.g,
        div_damp=div_damp, hyperdiff_coeff=0.0,
        boundary_fix=False)
    dh_on, du_on, dv_on = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, g=constants.g,
        div_damp=div_damp, hyperdiff_coeff=0.0,
        boundary_fix=True)

    # iter-865 gate: in duogrid mode boundary_fix=True must produce
    # the SAME output as boundary_fix=False (smoothing skipped).
    np.testing.assert_array_equal(
        np.asarray(dh_on), np.asarray(dh_off))
    np.testing.assert_array_equal(
        np.asarray(du_on), np.asarray(du_off))
    np.testing.assert_array_equal(
        np.asarray(dv_on), np.asarray(dv_off))


def test_fortran_vector_corner_fill_bypassed_in_duogrid_mode():
    """Iter-865 (Codex iter-865 second-pass finding): the public
    `fortran_vector_corner_fill` flag in `fv3_sw_tendencies` is a
    NON-DUOGRID legacy cube-vertex corner formula
    (`fill_corners_agrid_r8`).  It must be bypassed when duogrid is
    active — otherwise enabling the public flag on a duogrid grid
    would overwrite the duogrid halo's correct cube-vertex values.

    DUOGRID + fortran_vector_corner_fill=True must produce the SAME
    output as fortran_vector_corner_fill=False on a duogrid grid.
    """
    n = 8
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    h, u_d, v_d, h_s = _build_state(n, seed=29)

    div_damp = 1.5e7 * (48.0 / n) ** 2
    dh_off, du_off, dv_off = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, g=constants.g,
        div_damp=div_damp, hyperdiff_coeff=0.0,
        boundary_fix=False,
        fortran_vector_corner_fill=False)
    dh_on, du_on, dv_on = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, g=constants.g,
        div_damp=div_damp, hyperdiff_coeff=0.0,
        boundary_fix=False,
        fortran_vector_corner_fill=True)

    np.testing.assert_array_equal(
        np.asarray(dh_on), np.asarray(dh_off))
    np.testing.assert_array_equal(
        np.asarray(du_on), np.asarray(du_off))
    np.testing.assert_array_equal(
        np.asarray(dv_on), np.asarray(dv_off))


def test_fortran_vector_corner_fill_active_in_legacy_mode():
    """In LEGACY mode, fortran_vector_corner_fill=True changes the
    output (the corner formula fires).  This guards that iter-865's
    duogrid gate did not silently disable the legacy mode behaviour."""
    n = 8
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    h, u_d, v_d, h_s = _build_state(n, seed=41)

    div_damp = 1.5e7 * (48.0 / n) ** 2
    dh_off, du_off, dv_off = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, g=constants.g,
        div_damp=div_damp, hyperdiff_coeff=0.0,
        boundary_fix=False,
        fortran_vector_corner_fill=False)
    dh_on, du_on, dv_on = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid, g=constants.g,
        div_damp=div_damp, hyperdiff_coeff=0.0,
        boundary_fix=False,
        fortran_vector_corner_fill=True)

    diff_u = float(np.max(np.abs(np.asarray(du_on) - np.asarray(du_off))))
    diff_v = float(np.max(np.abs(np.asarray(dv_on) - np.asarray(dv_off))))
    assert diff_u > 1e-12 or diff_v > 1e-12, (
        f"LEGACY mode + fortran_vector_corner_fill=True should change "
        f"output but max|Δdu|={diff_u:.3e}, max|Δdv|={diff_v:.3e} are "
        f"both ≤ 1e-12.  iter-865 must not have silently disabled the "
        f"legacy-mode corner-fill behaviour.")


def test_bounded_domain_property_recognises_panel_and_duogrid():
    """Iter-865b: the new `bounded_domain` property on
    `CubedSphereGrid` must return True for both bounded-domain cases:
    - duogrid (full 6-face cubed sphere with cross-face halo).
    - regional / nested single-face panel (`lat.shape[0] == 1`).
    And False for a plain non-duogrid global cubed sphere."""
    n = 8
    legacy_grid = create_cubed_sphere(n, use_duogrid=False)
    duogrid_grid = create_cubed_sphere(n, use_duogrid=True)

    # Regional panel from create_cubed_sphere_panel.
    from legoesm.grids.cubed_sphere import create_cubed_sphere_panel
    panel = create_cubed_sphere_panel(n)

    assert legacy_grid.bounded_domain is False, (
        f"non-duogrid global cubed sphere should report "
        f"bounded_domain=False; got {legacy_grid.bounded_domain}")
    assert duogrid_grid.bounded_domain is True, (
        f"duogrid global cubed sphere should report "
        f"bounded_domain=True; got {duogrid_grid.bounded_domain}")
    assert panel.bounded_domain is True, (
        f"single-face regional panel should report "
        f"bounded_domain=True; got {panel.bounded_domain}")


def test_iter865_gate_visible_in_source():
    """Iter-865 / iter-865b source-scan: the production
    `boundary_fix` block in `fv3_sw_tendencies` must be guarded by
    ALL THREE conditions: `boundary_fix`, `not cdgrid.base.bounded_domain`
    (or equivalently `cdgrid.base.duogrid is None` in legacy
    iter-865 form), AND `n > 2`.

    Iter-865b: original iter-865 hardcoded `cdgrid.base.duogrid is
    None`; Codex stop-time review pointed out this leaves regional/
    nested single-face panels on the legacy path.  iter-865b uses
    the proper Fortran-faithful `bounded_domain` flag.  This scan
    accepts BOTH forms (`not cdgrid.base.bounded_domain` and
    `cdgrid.base.duogrid is None`) so a future revert to the
    duogrid-only form would not silently pass.

    Codex iter-865 second-pass tightening: the prior version checked
    only that some `if boundary_fix ... .duogrid is None` block existed
    anywhere under the function — it didn't (a) target the actual
    smoothing assignments, (b) verify the `cdgrid.base.duogrid`
    attribute chain, or (c) verify the `n > 2` size guard.

    The scan finds every direct-body assignment of the form
    ``du_cc.at[:, ...].set(...)`` or ``dv_cc.at[:, ...].set(...)`` in
    `fv3_sw_tendencies`, walks up to the dominating `If` block, and
    requires that `If` test contains all three guards via an AND chain.
    """
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "src" / "legoesm" / "core" / "operators_cdgrid.py")
    tree = ast.parse(src.read_text())

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "fv3_sw_tendencies"),
        None,
    )
    assert fn is not None, "fv3_sw_tendencies function not found."

    def _is_du_dv_at_set(call):
        """True if `call` is `du_cc.at[...].set(...)` or
        `dv_cc.at[...].set(...)` (the smoothing assignment pattern)."""
        if not (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in ("set", "add")):
            return False
        outer = call.func.value
        if (not isinstance(outer, ast.Subscript)
                or not isinstance(outer.value, ast.Attribute)
                or outer.value.attr != "at"):
            return False
        base = outer.value.value
        return (isinstance(base, ast.Name)
                and base.id in ("du_cc", "dv_cc"))

    def _check_test_has_all_three_guards(test_expr):
        """True iff `test_expr` is an AND chain (or single Compare/
        BoolOp tree) that includes:
          - a Name `boundary_fix`
          - a `cdgrid.base.duogrid is None` Compare
          - an `n > 2` Compare
        """
        # Flatten AND-chain operands.
        operands = []

        def flatten(node):
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                for v in node.values:
                    flatten(v)
            else:
                operands.append(node)

        flatten(test_expr)

        has_boundary_fix = any(
            isinstance(op, ast.Name) and op.id == "boundary_fix"
            for op in operands
        )

        def is_duogrid_is_none(op):
            return (isinstance(op, ast.Compare)
                    and len(op.ops) == 1
                    and isinstance(op.ops[0], ast.Is)
                    and len(op.comparators) == 1
                    and isinstance(op.comparators[0], ast.Constant)
                    and op.comparators[0].value is None
                    and isinstance(op.left, ast.Attribute)
                    and op.left.attr == "duogrid"
                    and isinstance(op.left.value, ast.Attribute)
                    and op.left.value.attr == "base")

        def is_not_bounded_domain(op):
            """Iter-865b: ``not cdgrid.base.bounded_domain`` form."""
            if not (isinstance(op, ast.UnaryOp)
                    and isinstance(op.op, ast.Not)):
                return False
            inner = op.operand
            return (isinstance(inner, ast.Attribute)
                    and inner.attr == "bounded_domain"
                    and isinstance(inner.value, ast.Attribute)
                    and inner.value.attr == "base")

        has_duogrid = any(
            is_duogrid_is_none(op) or is_not_bounded_domain(op)
            for op in operands)

        def is_n_gt_2(op):
            return (isinstance(op, ast.Compare)
                    and len(op.ops) == 1
                    and isinstance(op.ops[0], ast.Gt)
                    and isinstance(op.left, ast.Name)
                    and op.left.id == "n"
                    and len(op.comparators) == 1
                    and isinstance(op.comparators[0], ast.Constant)
                    and op.comparators[0].value == 2)

        has_n_guard = any(is_n_gt_2(op) for op in operands)

        return has_boundary_fix, has_duogrid, has_n_guard

    # Walk the direct body for the dominating If of any
    # du_cc/dv_cc.at[].set(...) call.
    def find_dominating_if(stmts, target_call):
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                # Check body
                for sub in ast.walk(stmt):
                    if sub is target_call:
                        return stmt
        return None

    # Collect smoothing calls and their dominating ifs.
    smoothing_calls = []
    for sub in ast.walk(fn):
        if _is_du_dv_at_set(sub):
            smoothing_calls.append(sub)

    assert smoothing_calls, (
        "No `du_cc/dv_cc.at[...].set(...)` smoothing assignments "
        "found in fv3_sw_tendencies — has the boundary_fix block "
        "been removed?  Iter-865 expects this block to exist and be "
        "duogrid-gated.")

    # For each smoothing call, walk up to find the enclosing top-level
    # If block whose body contains it.
    failures = []
    for call in smoothing_calls:
        # Find the if statement in the function body whose subtree
        # contains this call.
        dominating_if = None
        for stmt in fn.body:
            if isinstance(stmt, ast.If):
                if any(call is sub for sub in ast.walk(stmt)):
                    dominating_if = stmt
                    break
        if dominating_if is None:
            failures.append(
                f"line {call.lineno}: not inside any top-level If")
            continue
        has_bf, has_dg, has_n = _check_test_has_all_three_guards(
            dominating_if.test)
        missing = []
        if not has_bf:
            missing.append("boundary_fix")
        if not has_dg:
            missing.append("cdgrid.base.duogrid is None")
        if not has_n:
            missing.append("n > 2")
        if missing:
            failures.append(
                f"line {call.lineno}: dominating If at line "
                f"{dominating_if.lineno} missing guards: {missing}")

    assert not failures, (
        f"Iter-865 boundary_fix gate is incomplete:\n  "
        + "\n  ".join(failures)
        + "\n\nThe top-level If guarding du_cc/dv_cc smoothing must "
          "AND together: (a) `boundary_fix`, (b) `not "
          "cdgrid.base.bounded_domain` (iter-865b) OR `cdgrid.base."
          "duogrid is None` (legacy iter-865), (c) `n > 2`.")
