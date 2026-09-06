"""Round 32: the stage-3 correction operator, its placement, and its gate.

Every test here has a synthetic-violation arm, because a test that cannot fail
proves nothing.  The three things under test are:

* ``rk3_stage_barotropic_correction`` -- the operator this round moved, now a
  module-level function so a fidelity gate can drive it with NEMO's operands;
* WHERE the production step calls it, which is the actual change and has no
  numerical expression, so it is checked structurally on the real source with
  a mutated copy proving the check bites;
* the round-32 gate's own arithmetic -- its tridiagonal solver, its layout
  helper and its verdict ladder.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from legoesm.ocean.dynamics.barotropic_common import (
    rk3_stage_barotropic_correction,
)

REPO = Path(__file__).resolve().parents[3]
GATES = REPO / "scripts/validate/ocean_fidelity/testcases"
STEP_MODULE = (REPO / "packages/ocean/legoesm/ocean/dynamics"
               / "ocean_model_latlon_cgrid.py")


def _load_gate():
    """Import the round-32 gate the way it runs, with its siblings importable."""
    if str(GATES) not in sys.path:
        sys.path.insert(0, str(GATES))
    spec = importlib.util.spec_from_file_location(
        "nemo_testcase_l2_gyre_round32_ordering",
        GATES / "nemo_testcase_l2_gyre_round32_ordering.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------

def _column_case(seed: int = 3):
    rng = np.random.default_rng(seed)
    nlev = 6
    field = rng.normal(size=(4, 5, nlev))
    h = np.abs(rng.normal(size=(4, 5, nlev))) + 1.0
    mask3 = np.ones((4, 5, nlev))
    mask3[0, 0, 4:] = 0.0          # a shallower column
    mask3[1, 1, :] = 0.0           # a land column
    h = h * mask3
    depth = np.maximum(h.sum(axis=-1), 1e-10)
    wet2d = (mask3 > 0).any(axis=-1).astype(float)
    target = rng.normal(size=(4, 5)) * wet2d
    return field, target, h, depth, wet2d, mask3


def test_correction_installs_the_target_mean():
    """Weighting-free invariant: the corrected column mean IS the target."""
    field, target, h, depth, wet2d, mask3 = _column_case()
    out = np.asarray(rk3_stage_barotropic_correction(
        field, target, h, depth, wet2d, mask3))
    got = (out * h).sum(axis=-1) / depth
    wet = wet2d > 0
    assert np.allclose(got[wet], target[wet], rtol=0, atol=1e-14)


def test_correction_would_fail_without_removing_the_old_mean():
    """Non-vacuity: the same assertion rejects an add-only 'correction'."""
    field, target, h, depth, wet2d, mask3 = _column_case()
    bad = (field + target[..., None]) * mask3
    got = (bad * h).sum(axis=-1) / depth
    wet = wet2d > 0
    assert not np.allclose(got[wet], target[wet], rtol=0, atol=1e-14)


def test_correction_masks_land_and_is_idempotent():
    field, target, h, depth, wet2d, mask3 = _column_case()
    out = np.asarray(rk3_stage_barotropic_correction(
        field, target, h, depth, wet2d, mask3))
    assert np.all(out[mask3 == 0.0] == 0.0)
    twice = np.asarray(rk3_stage_barotropic_correction(
        out, target, h, depth, wet2d, mask3))
    assert np.allclose(twice, out, rtol=0, atol=1e-14)


def test_correction_is_the_identity_when_the_mean_already_matches():
    field, _target, h, depth, wet2d, mask3 = _column_case()
    own = (field * h).sum(axis=-1) / depth * wet2d
    out = np.asarray(rk3_stage_barotropic_correction(
        field, own, h, depth, wet2d, mask3))
    assert np.allclose(out, field * mask3, rtol=0, atol=1e-14)


# --------------------------------------------------------------------------
# WHERE the production step calls it
# --------------------------------------------------------------------------

def stage3_placement_defects(source: str) -> list[str]:
    """Structural check that the stage-3 correction is DEFERRED past the solve.

    The change round 32 landed is an ORDER, and an order has no numerical
    signature a unit test can assert without running the whole step.  So this
    reads the real module and requires three things at once, each of which a
    revert would break:

    * no call passing ``u3_raw`` to ``_replace_stage_mean`` (that call site is
      exactly the pre-solve application this round removed);
    * ``_ws_stage3_correction`` is bound to a tuple naming
      ``_replace_stage_mean``;
    * that name is later unpacked and CALLED, so binding it without using it
      does not pass.
    """
    tree = ast.parse(source)
    defects = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_replace_stage_mean"
                and any(isinstance(a, ast.Name) and a.id in {"u3_raw", "v3_raw"}
                        for a in node.args)):
            defects.append(
                "the stage-3 explicit update is corrected BEFORE the solve")
    bound = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name)
                        and t.id == "_ws_stage3_correction" for t in node.targets)
                and isinstance(node.value, ast.Tuple)
                and any(isinstance(e, ast.Name) and e.id == "_replace_stage_mean"
                        for e in node.value.elts)):
            bound = True
    if not bound:
        defects.append(
            "_ws_stage3_correction is never bound to the correction closure")
    called = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_stage3_corr" for node in ast.walk(tree))
    if not called:
        defects.append("the deferred correction is never applied")
    return defects


def test_the_production_step_defers_the_stage3_correction():
    assert stage3_placement_defects(STEP_MODULE.read_text()) == []


def test_the_placement_check_rejects_the_pre_solve_ordering():
    """Non-vacuity: restore the removed call site and the check must bite."""
    reverted = (
        "u3_corr, v3_corr = _replace_stage_mean(u3_raw, v3_raw, tu, tv)\n"
        "_ws_stage3_correction = (_replace_stage_mean, tu, tv)\n"
        "a, b = _stage3_corr(x, y, tu, tv)\n")
    defects = stage3_placement_defects(reverted)
    assert any("BEFORE the solve" in d for d in defects)


def test_the_placement_check_rejects_a_bound_but_unused_correction():
    unused = "_ws_stage3_correction = (_replace_stage_mean, tu, tv)\n"
    assert stage3_placement_defects(unused) == [
        "the deferred correction is never applied"]


def test_the_production_step_calls_the_shared_operator():
    """The closure must delegate, or the gate would score a stale copy."""
    source = STEP_MODULE.read_text()
    assert "rk3_stage_barotropic_correction" in source
    tree = ast.parse(source)
    inside = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_replace_stage_mean"]
    assert len(inside) == 1
    calls = {n.func.id for n in ast.walk(inside[0])
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "rk3_stage_barotropic_correction" in calls


# --------------------------------------------------------------------------
# the gate's own arithmetic
# --------------------------------------------------------------------------

def test_thomas_matches_a_dense_solve():
    gate = _load_gate()
    rng = np.random.default_rng(11)
    nk = 7
    sub = rng.normal(size=(3, nk)) * 0.1
    sup = rng.normal(size=(3, nk)) * 0.1
    diag = 2.0 + np.abs(rng.normal(size=(3, nk)))
    rhs = rng.normal(size=(3, nk))
    got = gate._thomas(sub, diag, sup, rhs)
    for row in range(3):
        dense = np.diag(diag[row])
        for k in range(1, nk):
            dense[k, k - 1] = sub[row, k]
        for k in range(nk - 1):
            dense[k, k + 1] = sup[row, k]
        assert np.allclose(got[row], np.linalg.solve(dense, rhs[row]),
                           rtol=0, atol=1e-12)


def test_thomas_is_sensitive_to_its_coefficients():
    """Non-vacuity: perturbing one off-diagonal must change the answer."""
    gate = _load_gate()
    nk = 5
    sub = np.full((1, nk), 0.2)
    sup = np.full((1, nk), 0.3)
    diag = np.full((1, nk), 2.0)
    rhs = np.arange(nk, dtype=float)[None, :]
    base = gate._thomas(sub, diag, sup, rhs)
    bumped = sub.copy()
    bumped[0, 2] += 0.5
    assert not np.allclose(base, gate._thomas(bumped, diag, sup, rhs))


def test_scored_2d_layout_matches_the_gates_3d_helper():
    """``_to_scored_2d`` must strip and transpose exactly like ``_xyz``."""
    gate = _load_gate()
    from nemo_testcase_l2_gyre_phase3_gate import DIMS, _xyz

    nx, ny, nz = DIMS
    flat = np.arange(nx * ny, dtype=float)
    two_d = flat.reshape((nx, ny), order="F")
    volume = np.repeat(flat, nz)
    got_3d = _xyz(
        np.asarray(two_d[..., None].repeat(nz, axis=-1)).ravel(order="F"),
        nx, ny, nz)
    assert np.array_equal(gate._to_scored_2d(two_d), got_3d[..., 0])
    assert volume.size == nx * ny * nz  # the fixture is the shape it claims


def test_scored_2d_layout_rejects_a_transposed_copy():
    gate = _load_gate()
    from nemo_testcase_l2_gyre_phase3_gate import DIMS

    nx, ny, _ = DIMS
    two_d = np.arange(nx * ny, dtype=float).reshape((nx, ny), order="F")
    assert not np.array_equal(gate._to_scored_2d(two_d),
                              two_d[2:-2, 2:-2])


def test_slope_verdict_ladder_runs_at_both_ends():
    gate = _load_gate()
    assert gate.slope_verdict([0.0, 0.01]) == "SOLVE RESPONSE CONFIRMED"
    assert gate.slope_verdict([0.5, 0.9]) == "SECOND RESIDUAL"
    assert gate.slope_verdict([0.0, 0.9]) == "NO VERDICT"
    assert gate.slope_verdict([0.13, 0.13]) == "NO VERDICT"
    with pytest.raises(ValueError):
        gate.slope_verdict([])


def test_slope_verdict_boundaries_are_where_the_preregistration_put_them():
    gate = _load_gate()
    assert gate.CONFIRM_FRACTION == 0.05
    assert gate.REFUTE_FRACTION == 0.20
    assert gate.slope_verdict([gate.CONFIRM_FRACTION]) == (
        "SOLVE RESPONSE CONFIRMED")
    assert gate.slope_verdict([gate.REFUTE_FRACTION]) == "NO VERDICT"
