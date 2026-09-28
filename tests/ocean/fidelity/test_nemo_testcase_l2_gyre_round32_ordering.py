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
    nemo_reference_depth_reciprocal,
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
    # ROUND 36: NEMO's own reciprocal, domain.f90:213, which carries the
    # dry-column zero the operator no longer takes as a separate mask.
    r1_depth = np.asarray(nemo_reference_depth_reciprocal(depth, wet2d))
    target = rng.normal(size=(4, 5)) * wet2d
    return field, target, h, r1_depth, wet2d, mask3


def test_correction_installs_the_target_mean():
    """Weighting-free invariant: the corrected column mean IS the target."""
    field, target, h, r1_depth, wet2d, mask3 = _column_case()
    out = np.asarray(rk3_stage_barotropic_correction(
        field, target, h, r1_depth, mask3))
    got = (out * h).sum(axis=-1) * r1_depth
    wet = wet2d > 0
    assert np.allclose(got[wet], target[wet], rtol=0, atol=1e-14)


def test_correction_would_fail_without_removing_the_old_mean():
    """Non-vacuity: the same assertion rejects an add-only 'correction'."""
    field, target, h, r1_depth, wet2d, mask3 = _column_case()
    bad = (field + target[..., None]) * mask3
    got = (bad * h).sum(axis=-1) * r1_depth
    wet = wet2d > 0
    assert not np.allclose(got[wet], target[wet], rtol=0, atol=1e-14)


def test_correction_masks_land_and_is_idempotent():
    field, target, h, r1_depth, wet2d, mask3 = _column_case()
    out = np.asarray(rk3_stage_barotropic_correction(
        field, target, h, r1_depth, mask3))
    assert np.all(out[mask3 == 0.0] == 0.0)
    twice = np.asarray(rk3_stage_barotropic_correction(
        out, target, h, r1_depth, mask3))
    assert np.allclose(twice, out, rtol=0, atol=1e-14)


def test_correction_is_the_identity_when_the_mean_already_matches():
    field, _target, h, r1_depth, wet2d, mask3 = _column_case()
    own = (field * h).sum(axis=-1) * r1_depth
    out = np.asarray(rk3_stage_barotropic_correction(
        field, own, h, r1_depth, mask3))
    assert np.allclose(out, field * mask3, rtol=0, atol=1e-14)


# --------------------------------------------------------------------------
# WHERE the production step calls it
# --------------------------------------------------------------------------

APPLY_ARGS = ("state_new.u.data", "state_new.v.data", "_stage3_tu",
              "_stage3_tv")


def stage3_placement_defects(source: str) -> list[str]:
    """Structural check that the stage-3 correction is DEFERRED past the solve.

    The change round 32 landed is an ORDER, and an order has no numerical
    signature a unit test can assert without running the whole step.  So this
    reads the real module and requires five things at once.

    An earlier revision of this checker looked only for the deleted call
    STRING, and an independent review broke it in four different ways that all
    passed.  Each of those four is now a test below, so the checker is pinned
    to what it must catch rather than to what it happened to catch.
    """
    tree = ast.parse(source)
    defects = []
    built = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            names = []
            if isinstance(target, ast.Name):
                names = [target.id]
            elif isinstance(target, ast.Tuple):
                names = [e.id for e in target.elts if isinstance(e, ast.Name)]
            for name in names:
                if name in {"u3_corr", "v3_corr"}:
                    built.setdefault(name, []).append(node.value)
    # 1. the stage-3 pair IS the masked explicit update, nothing else
    for name in ("u3_corr", "v3_corr"):
        values = built.get(name)
        if not values:
            defects.append(f"{name} is never built")
            continue
        raw = "u3_raw" if name == "u3_corr" else "v3_raw"
        mask = "_ws_stage_u_mask" if name == "u3_corr" else "_ws_stage_v_mask"
        first = ast.unparse(values[0])
        if first != f"{raw} * {mask}":
            defects.append(
                f"{name} is built as {first!r}, not the masked explicit "
                f"update {raw} * {mask}")
    # 2. no correction call may take the stage-3 vectors, under any spelling
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_replace_stage_mean"
                and {ast.unparse(a) for a in node.args}
                & {"u3_raw", "v3_raw", "u3_corr", "v3_corr"}):
            defects.append(
                "the stage-3 velocity is corrected before the solve")
    # 3. the deferred binding carries the closure and NEMO's uu_b targets
    bound = [n for n in ast.walk(tree)
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name)
                     and t.id == "_ws_stage3_correction" for t in n.targets)
             and isinstance(n.value, ast.Tuple)]
    if not bound:
        defects.append("_ws_stage3_correction is never bound to the closure")
    else:
        elts = [ast.unparse(e) for e in bound[0].value.elts]
        if elts != ["_replace_stage_mean", "target_u", "target_v"]:
            defects.append(
                f"_ws_stage3_correction binds {elts!r}, not the closure with "
                "its uu_b/vv_b targets")
    # 4. the apply site is guarded on the binding ALONE.  Any extra condition
    #    can delete the correction on a card that resolves it False, which is
    #    exactly the hole a review caught before this round measured anything.
    applied = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != "_ws_stage3_correction is not None":
            continue
        calls = [n for n in ast.walk(node)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "_stage3_corr"]
        if calls:
            applied = calls[0]
    if applied is None:
        defects.append(
            "the deferred correction is not applied under a guard testing "
            "_ws_stage3_correction alone")
    else:
        args = tuple(ast.unparse(a) for a in applied.args)
        if args != APPLY_ARGS:
            defects.append(
                f"the deferred correction is applied as {args!r}, not "
                f"{APPLY_ARGS!r}")
    # 5. the barotropic removal reads the PROGNOSTIC uu_b (dynzdf.F90:150-151)
    prefers = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_u_bt_mean"
                for t in n.targets)
        and "_uu_b_zdf" in ast.unparse(n.value)
        for n in ast.walk(tree))
    if not prefers:
        defects.append(
            "the barotropic removal does not read the prognostic uu_b")
    return defects


def _mutate(old: str, new: str) -> str:
    source = STEP_MODULE.read_text()
    assert old in source, "the mutation anchor moved; the test is stale"
    mutated = source.replace(old, new, 1)
    assert mutated != source
    return mutated


def test_the_production_step_defers_the_stage3_correction():
    assert stage3_placement_defects(STEP_MODULE.read_text()) == []


def test_the_check_rejects_re_gating_the_apply_site():
    """The exact hole a review caught: an extra condition deletes it."""
    mutated = _mutate(
        "if _ws_stage3_correction is not None:",
        "if _ws_stage3_correction is not None and _impose_mean:")
    assert any("guard testing" in d
               for d in stage3_placement_defects(mutated))


def test_the_check_rejects_a_renamed_pre_solve_restore():
    mutated = _mutate(
        "            u3_corr = u3_raw * _ws_stage_u_mask\n"
        "            v3_corr = v3_raw * _ws_stage_v_mask\n",
        "            u3_corr, v3_corr = _replace_stage_mean(\n"
        "                u3_raw, v3_raw, target_u, target_v)\n")
    defects = stage3_placement_defects(mutated)
    assert any("before the solve" in d for d in defects)
    assert any("not the masked explicit update" in d for d in defects)


def test_the_check_rejects_swapped_correction_targets():
    mutated = _mutate(
        "state_new.u.data, state_new.v.data, _stage3_tu, _stage3_tv",
        "state_new.u.data, state_new.v.data, _stage3_tv, _stage3_tu")
    assert any("is applied as" in d for d in stage3_placement_defects(mutated))


def test_the_check_rejects_reverting_the_uu_b_operand():
    mutated = _mutate(
        "_u_bt_mean = _uu_b_zdf.data[..., jnp.newaxis].astype(",
        "_u_bt_mean_dead = _uu_b_zdf.data[..., jnp.newaxis].astype(")
    assert stage3_placement_defects(mutated) == [
        "the barotropic removal does not read the prognostic uu_b"]


def test_the_check_rejects_a_bound_but_unused_correction():
    unused = "_ws_stage3_correction = (_replace_stage_mean, target_u, target_v)\n"
    assert "guard testing" in " ".join(stage3_placement_defects(unused))


def test_the_production_step_calls_the_shared_operator():
    """The closure must delegate, or the gate would score a stale copy."""
    source = STEP_MODULE.read_text()
    tree = ast.parse(source)
    inside = [node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name == "_replace_stage_mean"]
    assert len(inside) == 1
    calls = {n.func.id for n in ast.walk(inside[0])
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "rk3_stage_barotropic_correction" in calls


def test_the_delegation_check_rejects_an_inlined_copy():
    inlined = ("def _replace_stage_mean(u_in, v_in, tu, tv):\n"
               "    return u_in + tu, v_in + tv\n")
    tree = ast.parse(inlined)
    inside = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    calls = {n.func.id for n in ast.walk(inside[0])
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "rk3_stage_barotropic_correction" not in calls


def test_the_compensated_pair_passes_the_solve_untouched():
    """Why the operand swap is safe, as an invariant rather than an argument.

    NEMO removes a column constant AND subtracts the matching bottom-stress
    term built from the SAME coefficient the diagonal carries
    (dynzdf.F90:150-151 with :156-159, against :296).  Because ``M.1 = 1 +
    c.e_bot`` exactly under zero-flux boundaries, that pair passes the solve as
    the constant itself -- so WHICH constant is removed cannot change the
    baroclinic answer, only the barotropic one the correction then resets.
    """
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        implicit_vertical_diffusion_nemo_momentum,
    )

    rng = np.random.default_rng(19)
    nlev = 8
    field = rng.normal(size=(2, 3, nlev))
    dz = np.full((2, 3, nlev), 50.0)
    e3w = np.full((2, 3, nlev - 1), 50.0)
    avm = np.abs(rng.normal(size=(2, 3, nlev - 1))) * 1e-3
    wet = np.ones((2, 3, nlev))
    drag = np.zeros((2, 3, nlev))
    drag[..., -1] = 2.4e-3
    kw = dict(extra_diag=drag)
    dt = 2400.0
    kappa = rng.normal(size=(2, 3, 1))
    base = np.asarray(implicit_vertical_diffusion_nemo_momentum(
        field, avm, dz, e3w, dt, wet, **kw))
    # the production sign: SUBTRACT the constant, and subtract the matching
    # bottom term built from the same coefficient the diagonal carries.
    shifted_in = field - kappa - drag * kappa
    shifted = np.asarray(implicit_vertical_diffusion_nemo_momentum(
        shifted_in, avm, dz, e3w, dt, wet, **kw))
    assert np.allclose(shifted + kappa, base, rtol=0, atol=1e-13)
    # non-vacuity: drop the bottom term and the constant no longer passes
    broken = np.asarray(implicit_vertical_diffusion_nemo_momentum(
        field - kappa, avm, dz, e3w, dt, wet, **kw))
    assert not np.allclose(broken + kappa, base, rtol=0, atol=1e-13)


def test_correction_refuses_a_broadcastable_but_wrong_target():
    field, target, h, r1_depth, wet2d, mask3 = _column_case()
    with pytest.raises(ValueError):
        rk3_stage_barotropic_correction(
            field, target[..., None], h, r1_depth, mask3)
    with pytest.raises(ValueError):
        rk3_stage_barotropic_correction(
            field, target, h[..., :-1], r1_depth, mask3)


def test_correction_refuses_a_wrong_shaped_mask():
    """Round-33 correction: the two MASKS were exempt from the shape check.

    Round 32 closed the shape hole for ``target_mean``, ``h_face_ref`` and
    the depth and left ``face_mask`` and ``stage_mask`` open.  Round 36
    REMOVED ``face_mask`` -- NEMO's ``SUM`` carries no mask and the
    dry-column zero lives inside ``r1_hu_0`` -- so that arm now pins the
    reciprocal's own shape refusal instead.  What the
    new check buys was MEASURED before it was written, and it is smaller than
    "a silent hole": a level axis on ``face_mask`` already raised a
    ``ValueError`` from JAX's broadcasting, and a short ``stage_mask`` raised
    a ``TypeError`` from inside ``lax.mul`` naming neither operand.  The gain
    is a NAMED refusal at the boundary of a function the Rule-12 gates drive
    with oracle arrays.

    The ``stage_mask`` arm is the one that goes red without the check (it
    raised ``TypeError``, not ``ValueError``); the reciprocal arm passes
    either way and is kept only to pin the message's operand.
    """
    field, target, h, r1_depth, wet2d, mask3 = _column_case()
    with pytest.raises(ValueError, match="r1_depth_ref"):
        rk3_stage_barotropic_correction(
            field, target, h, r1_depth[..., None], mask3)
    with pytest.raises(ValueError, match="stage_mask"):
        rk3_stage_barotropic_correction(
            field, target, h, r1_depth, mask3[..., :-1])
    # ... and the level-BROADCAST stage mask the model's own
    # ``legacy_2d_stage_face_mask`` arm passes is still accepted.
    out = np.asarray(rk3_stage_barotropic_correction(
        field, target, h, r1_depth, wet2d[..., None]))
    assert np.all(np.isfinite(out))


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


def test_the_reciprocal_refuses_a_broadcastable_but_shorter_mask():
    """The guard that left with ``face_mask``, put back where it belongs.

    Round 36 removed ``rk3_stage_barotropic_correction``'s ``face_mask``
    argument -- NEMO has no mask inside its SUM -- and with it went the check
    that the mask was the column shape.  An independent diff review then
    showed a shorter mask BROADCASTS through the reciprocal, the operator
    accepts the result, and the correction reaches dry columns: 15 of 15 dry
    cells wrong by 0.227.  The check now lives on the reciprocal.
    """
    field, target, h, r1_depth, wet2d, mask3 = _column_case()
    with pytest.raises(ValueError, match="face_mask"):
        nemo_reference_depth_reciprocal(
            np.maximum((h * mask3).sum(axis=-1), 1e-10), wet2d[0])
    # and the correctly shaped call still works, and is zero on land
    good = np.asarray(nemo_reference_depth_reciprocal(
        np.maximum((h * mask3).sum(axis=-1), 1e-10), wet2d))
    assert good.shape == field.shape[:-1]
    assert np.all(good[wet2d == 0.0] == 0.0)
    assert np.all(good[wet2d > 0.0] > 0.0)
