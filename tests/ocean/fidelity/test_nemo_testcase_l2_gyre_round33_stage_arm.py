"""Round 33: the RK3 momentum stage update, both arms, one implementation.

NEMO selects the momentum stage update with ONE predicate,
``IF( ln_dynadv_vec .OR. lk_linssh )``, and it does so at every stage: at
``stprk3_stg.F90:365`` for stages 1 and 2, and at ``dynzdf.F90:119`` for stage
3, whose time step lives inside ``dyn_zdf`` because ``stprk3_stg.F90:395``
CASE(3) does no time stepping at all.  legoESM honoured that at stages 1 and 2
and not at stage 3.

Every test here has a synthetic-violation arm.  Three things are under test:

* ``rk3_stage_velocity_update`` -- both arms, against an INDEPENDENT
  transcription of NEMO's two statements written here rather than imported,
  driven with a NON-ZERO before-velocity.  That last part is the round-33
  claim review's finding: GYRE's own record has ``uu(Kbb)`` identically zero,
  so the Rule-12 discharge exercises only half the expression and a wrong
  ``velocity_before`` operand would pass it bit-exact;
* WHERE the production step calls it -- all three stages, checked structurally
  on the real source with mutated copies proving the check bites;
* the card-level facts the Rule-12 disposition rests on: GYRE takes the vector
  arm and both tanks take the thickness-weighted one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    rk3_stage_velocity_update,
)

REPO = Path(__file__).resolve().parents[3]
STEP_MODULE = (REPO / "packages/ocean/legoesm/ocean/dynamics"
               / "ocean_model_latlon_cgrid.py")


def _case(seed: int = 7):
    """A non-zero before-velocity, a non-trivial mask and three distinct ratios."""
    rng = np.random.default_rng(seed)
    shape = (4, 5, 6)
    before = rng.normal(size=shape)                  # NON-zero, deliberately
    rhs = rng.normal(size=shape) * 1e-6
    mask = (rng.random(shape) > 0.25).astype(float)
    ratios = tuple(1.0 + 1e-3 * rng.normal(size=(4, 5, 1)) for _ in range(3))
    return before, rhs, 1200.0, mask, ratios


# --------------------------------------------------------------------------
# the operator, against an independent transcription of NEMO's statements
# --------------------------------------------------------------------------

def nemo_vector_arm(before, rhs, dt, mask):
    """``dynzdf.F90:121`` / ``stprk3_stg.F90:367``, written here from the source.

    ``puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kbb) + rDt * puu(ji,jj,jk,Krhs) )
    * umask(ji,jj,jk)``
    """
    return (before + dt * rhs) * mask


def nemo_qco_arm(before, rhs, dt, q_bb, q_mm, q_aa):
    """``dynzdf.F90:127-129`` / ``stprk3_stg.F90:373-375``, key_qco form.

    ``puu(Kaa) = ( (1+r3u(Kbb))*puu(Kbb) + rDt*(1+r3u(Kmm))*puu(Krhs) )
    / (1+r3u(Kaa)) * umask``.  legoESM applies the mask after the barotropic
    correction instead, so this transcription stops before it -- the point
    under test is the arithmetic, and the masking difference is measured
    separately by the trajectory gates.
    """
    return (q_bb * before + dt * q_mm * rhs) / q_aa


def test_vector_arm_matches_nemos_statement_with_a_nonzero_before_velocity():
    before, rhs, dt, mask, ratios = _case()
    got = np.asarray(rk3_stage_velocity_update(
        before, rhs, dt, mask, vector_form=True,
        qco_before=ratios[0], qco_now=ratios[1], qco_after=ratios[2]))
    assert np.array_equal(got, nemo_vector_arm(before, rhs, dt, mask))
    # the before-velocity is genuinely load-bearing here
    assert np.max(np.abs(before)) > 0.1


def test_vector_arm_is_sensitive_to_the_before_velocity():
    """The arm the record cannot exercise, because NEMO's uu(Kbb) is all zeros."""
    before, rhs, dt, mask, _ = _case()
    base = np.asarray(rk3_stage_velocity_update(
        before, rhs, dt, mask, vector_form=True))
    bumped = before.copy()
    wet = np.argwhere(mask > 0)[0]
    bumped[tuple(wet)] = np.nextafter(bumped[tuple(wet)], np.inf)
    moved = np.asarray(rk3_stage_velocity_update(
        bumped, rhs, dt, mask, vector_form=True))
    assert not np.array_equal(base, moved)
    # and dropping the term entirely must not go unnoticed either
    dropped = np.asarray(rk3_stage_velocity_update(
        np.zeros_like(before), rhs, dt, mask, vector_form=True))
    assert not np.allclose(dropped, base, rtol=0, atol=0)


def test_qco_arm_matches_nemos_statement():
    before, rhs, dt, mask, (q_bb, q_mm, q_aa) = _case()
    got = np.asarray(rk3_stage_velocity_update(
        before, rhs, dt, mask, vector_form=False,
        qco_before=q_bb, qco_now=q_mm, qco_after=q_aa))
    assert np.array_equal(got, nemo_qco_arm(before, rhs, dt, q_bb, q_mm, q_aa))


def test_qco_arm_distinguishes_its_three_time_levels():
    """Swapping Kmm and Kaa must change the answer, or the ratios are decorative."""
    before, rhs, dt, mask, (q_bb, q_mm, q_aa) = _case()
    base = np.asarray(rk3_stage_velocity_update(
        before, rhs, dt, mask, vector_form=False,
        qco_before=q_bb, qco_now=q_mm, qco_after=q_aa))
    swapped = np.asarray(rk3_stage_velocity_update(
        before, rhs, dt, mask, vector_form=False,
        qco_before=q_bb, qco_now=q_aa, qco_after=q_mm))
    assert not np.array_equal(base, swapped)


def test_the_two_arms_are_not_the_same_function():
    """A refactor that silently collapses the selector would pass everything else."""
    before, rhs, dt, mask, (q_bb, q_mm, q_aa) = _case()
    vector = np.asarray(rk3_stage_velocity_update(
        before, rhs, dt, mask, vector_form=True))
    qco = np.asarray(rk3_stage_velocity_update(
        before, rhs, dt, mask, vector_form=False,
        qco_before=q_bb, qco_now=q_mm, qco_after=q_aa))
    assert not np.allclose(vector, qco, rtol=0, atol=1e-18)


def test_the_qco_arm_refuses_a_missing_ratio():
    """Dispatch hardening: no silent degradation to the other arm."""
    before, rhs, dt, mask, (q_bb, q_mm, _) = _case()
    with pytest.raises(ValueError, match="three"):
        rk3_stage_velocity_update(before, rhs, dt, mask, vector_form=False,
                                  qco_before=q_bb, qco_now=q_mm)


def test_the_operator_refuses_mismatched_shapes():
    before, rhs, dt, mask, _ = _case()
    with pytest.raises(ValueError, match="rhs"):
        rk3_stage_velocity_update(before, rhs[..., :-1], dt, mask,
                                  vector_form=True)
    with pytest.raises(ValueError, match="face_mask"):
        rk3_stage_velocity_update(before, rhs, dt, mask[..., :-1],
                                  vector_form=True)
    # the level-broadcast mask stays accepted: the model's own
    # legacy_2d_stage_face_mask arm passes exactly that
    out = np.asarray(rk3_stage_velocity_update(
        before, rhs, dt, mask[..., :1], vector_form=True))
    assert out.shape == before.shape


# --------------------------------------------------------------------------
# WHERE the production step calls it
# --------------------------------------------------------------------------

STAGE_CALLS = {
    "u1_raw": ("u0", "_u1_rhs", "dt_mom / 3.0", "_ws_stage_u_mask"),
    "v1_raw": ("v0", "_v1_rhs", "dt_mom / 3.0", "_ws_stage_v_mask"),
    "u2_raw": ("u0", "p1u_corr", "dt_mom / 2.0", "_ws_stage_u_mask"),
    "v2_raw": ("v0", "p1v_corr", "dt_mom / 2.0", "_ws_stage_v_mask"),
    "u3_raw": ("u0", "p2u_corr", "dt_mom", "_ws_stage_u_mask"),
    "v3_raw": ("v0", "p2v_corr", "dt_mom", "_ws_stage_v_mask"),
}
STAGE_RATIOS = {
    "u1_raw": ("_qu_b", "_qu_b", "_qu_13"),
    "v1_raw": ("_qv_b", "_qv_b", "_qv_13"),
    "u2_raw": ("_qu_b", "_qu_13", "_qu_12"),
    "v2_raw": ("_qv_b", "_qv_13", "_qv_12"),
    "u3_raw": ("_qu_b", "_qu_12", "_qu_aa"),
    "v3_raw": ("_qv_b", "_qv_12", "_qv_aa"),
}


def stage_arm_defects(source: str) -> list[str]:
    """Every stage's raw update must BE the shared helper, with NEMO's operands.

    Structural, because an arm selection has no numerical signature a unit test
    can assert without running the whole step.  Three things at once: each of
    the six raw fields is assigned exactly one call to the helper; the call
    carries that stage's own dt, RHS and mask; and every call passes the SAME
    ``vector_form=_vector_velocity_stage_update`` selector, so stage 3 cannot
    drift from stages 1 and 2 again.
    """
    tree = ast.parse(source)
    defects = []
    seen = {}
    selector_bindings = 0
    helper_rebindings = 0
    entry_bindings = []
    for node in ast.walk(tree):
        # AugAssign too.  The round-33 DIFF review defeated the first version
        # of this checker with ``u3_raw *= _qu_b / _qu_aa`` on the line AFTER
        # the call, which is an ast.AugAssign and was invisible here.
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id in STAGE_CALLS:
                defects.append(
                    f"{node.target.id} is modified in place after it is built")
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "rk3_stage_velocity_update":
                helper_rebindings += 1
            continue
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            names = ([target] if isinstance(target, ast.Name)
                     else [e for e in getattr(target, "elts", [])
                           if isinstance(e, ast.Name)])
            for name in names:
                if name.id in STAGE_CALLS:
                    seen.setdefault(name.id, []).append(node.value)
                if name.id == "_vector_velocity_stage_update":
                    selector_bindings += 1
                if name.id == "rk3_stage_velocity_update":
                    helper_rebindings += 1
                if name.id in ("u0", "v0"):
                    entry_bindings.append((name.id, ast.unparse(node.value)))
    # The selector must be computed ONCE.  One extra line,
    # ``_vector_velocity_stage_update = False``, replants the exact defect
    # round 33 fixed and left the first version of this checker clean.
    if selector_bindings != 1:
        defects.append(
            f"_vector_velocity_stage_update is bound {selector_bindings} "
            "times; it must be computed exactly once")
    # ... and the helper must be the module-level one, never shadowed.
    if helper_rebindings != 1:
        defects.append(
            f"rk3_stage_velocity_update is bound {helper_rebindings} times; "
            "a local shadow would make every call below meaningless")
    # ... and the BEFORE-level operands must be the state's own fields, so a
    # ``u0, v0 = v0, u0`` line cannot swap the faces under the calls.
    for name, value in entry_bindings:
        want = "state.u.data" if name == "u0" else "state.v.data"
        if value != want:
            defects.append(
                f"{name} is bound to {value!r}, not the state's own {want}")
    for name, want in STAGE_CALLS.items():
        values = seen.get(name)
        if not values:
            defects.append(f"{name} is never built")
            continue
        if len(values) > 1:
            defects.append(f"{name} is built {len(values)} times")
        value = values[0]
        if not (isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "rk3_stage_velocity_update"):
            defects.append(
                f"{name} is built as {ast.unparse(value)!r}, not a call to "
                "the shared rk3_stage_velocity_update")
            continue
        args = tuple(ast.unparse(a) for a in value.args)
        if args != want:
            defects.append(f"{name} is called with {args!r}, not {want!r}")
        kwargs = {k.arg: ast.unparse(k.value) for k in value.keywords}
        if kwargs.get("vector_form") != "_vector_velocity_stage_update":
            defects.append(
                f"{name} selects its arm with "
                f"{kwargs.get('vector_form')!r}, not the shared selector")
        ratios = tuple(kwargs.get(k) for k in
                       ("qco_before", "qco_now", "qco_after"))
        if ratios != STAGE_RATIOS[name]:
            defects.append(
                f"{name} passes ratios {ratios!r}, not {STAGE_RATIOS[name]!r}")
    # Finally, the corrected stage-3 pair must DERIVE from the raw pair a call
    # just built.  Without this the call can stay in place and be dead -- the
    # review's "keep the call but build u3_corr from the old inline form".
    corrected = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ("u3_corr",
                                                              "v3_corr"):
                corrected.setdefault(target.id, []).append(node.value)
    for name in ("u3_corr", "v3_corr"):
        values = corrected.get(name)
        raw = "u3_raw" if name == "u3_corr" else "v3_raw"
        if not values:
            defects.append(f"{name} is never built")
        elif raw not in ast.unparse(values[0]):
            defects.append(
                f"{name} is built from {ast.unparse(values[0])!r}, which does "
                f"not consume {raw}, so the stage call is dead")
    return defects


def _mutate(old: str, new: str) -> str:
    source = STEP_MODULE.read_text()
    assert old in source, "the mutation anchor moved; the test is stale"
    mutated = source.replace(old, new, 1)
    assert mutated != source
    return mutated


def test_all_three_stages_route_through_the_shared_helper():
    assert stage_arm_defects(STEP_MODULE.read_text()) == []


def test_the_check_rejects_stage3_keeping_the_qco_form():
    """The exact defect round 33 fixed, replanted."""
    mutated = _mutate(
        "            u3_raw = rk3_stage_velocity_update(\n"
        "                u0, p2u_corr, dt_mom, _ws_stage_u_mask,\n"
        "                vector_form=_vector_velocity_stage_update,\n"
        "                qco_before=_qu_b, qco_now=_qu_12, qco_after=_qu_aa)\n",
        "            u3_raw = (_qu_b * u0 + dt_mom * _qu_12 * p2u_corr) "
        "/ _qu_aa\n")
    assert any("not a call to the shared" in d
               for d in stage_arm_defects(mutated))


def test_the_check_rejects_a_stage_pinned_to_one_arm():
    mutated = _mutate(
        "                u0, p2u_corr, dt_mom, _ws_stage_u_mask,\n"
        "                vector_form=_vector_velocity_stage_update,",
        "                u0, p2u_corr, dt_mom, _ws_stage_u_mask,\n"
        "                vector_form=True,")
    assert any("not the shared selector" in d
               for d in stage_arm_defects(mutated))


def test_the_check_rejects_a_swapped_stage_time_level():
    mutated = _mutate(
        "qco_before=_qu_b, qco_now=_qu_12, qco_after=_qu_aa)",
        "qco_before=_qu_b, qco_now=_qu_aa, qco_after=_qu_12)")
    assert any("passes ratios" in d for d in stage_arm_defects(mutated))


def test_the_check_rejects_a_stage_taking_the_wrong_timestep():
    mutated = _mutate(
        "                u0, p2u_corr, dt_mom, _ws_stage_u_mask,",
        "                u0, p2u_corr, dt_mom / 2.0, _ws_stage_u_mask,")
    assert any("is called with" in d for d in stage_arm_defects(mutated))


def test_the_check_rejects_a_stage_taking_the_wrong_rhs():
    mutated = _mutate(
        "                u0, p2u_corr, dt_mom, _ws_stage_u_mask,",
        "                u0, p1u_corr, dt_mom, _ws_stage_u_mask,")
    assert any("is called with" in d for d in stage_arm_defects(mutated))


def test_the_check_rejects_the_selector_rebound_to_a_constant():
    """DIFF review M3: one inserted line replants the whole defect."""
    mutated = _mutate(
        "            # ARM.  The stage-3 time step is dyn_zdf's",
        "            _vector_velocity_stage_update = False\n"
        "            # ARM.  The stage-3 time step is dyn_zdf's")
    assert any("bound 2 times" in d for d in stage_arm_defects(mutated))


def test_the_check_rejects_an_in_place_reweighting_after_the_call():
    """DIFF review M1: an AugAssign is not an Assign."""
    mutated = _mutate(
        "            u3_corr = u3_raw * _ws_stage_u_mask",
        "            u3_raw *= _qu_b / _qu_aa\n"
        "            u3_corr = u3_raw * _ws_stage_u_mask")
    assert any("modified in place" in d for d in stage_arm_defects(mutated))


def test_the_check_rejects_a_local_shadow_of_the_helper():
    """DIFF review M2: rebinding the name makes every call below a no-op."""
    mutated = _mutate(
        "            # ARM.  The stage-3 time step is dyn_zdf's",
        "            def rk3_stage_velocity_update(*a, **k):\n"
        "                return a[0]\n"
        "            # ARM.  The stage-3 time step is dyn_zdf's")
    assert any("bound 2 times" in d for d in stage_arm_defects(mutated))


def test_the_check_rejects_a_dead_stage3_call():
    """DIFF review M6: keep the call, build the answer the old way."""
    mutated = _mutate(
        "            u3_corr = u3_raw * _ws_stage_u_mask",
        "            u3_corr = ((_qu_b * u0 + dt_mom * _qu_12 * p2u_corr)\n"
        "                       / _qu_aa) * _ws_stage_u_mask")
    assert any("does not consume u3_raw" in d
               for d in stage_arm_defects(mutated))


def test_the_check_rejects_swapped_entry_velocities():
    """DIFF review M5: swap u0 and v0 above the calls and every arm passes."""
    mutated = _mutate(
        "            u0 = state.u.data",
        "            u0 = state.v.data")
    assert any("not the state's own state.u.data" in d
               for d in stage_arm_defects(mutated))


def test_the_stale_belief_is_gone_from_the_source():
    """The comment that produced the defect is a claim, and it was false.

    ``dynzdf.F90:119`` selects exactly as ``stprk3_stg.F90:365`` does, so
    "stage 3 still enters dyn_zdf's key_qco solve in either momentum program"
    was wrong.  WHAT THIS DOES NOT TEST, said out loud because a DIFF review
    pointed it out: it would still pass with the CODE fully reverted, as long
    as the comment stayed corrected.  The code is pinned by
    :func:`stage_arm_defects`; this only stops the false sentence coming back.
    """
    source = STEP_MODULE.read_text()
    assert "key_qco solve in either momentum program" not in source
    assert "dynzdf.F90:119 carries the" in source


# --------------------------------------------------------------------------
# the card-level facts the Rule-12 disposition rests on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case,scheme,vector", [
    ("GYRE-zco", "vector_invariant", True),
    ("LOCK_EXCHANGE-zco", "flux_form", False),
    ("OVERFLOW-zps", "flux_form", False),
])
def test_each_card_resolves_the_arm_nemos_deck_resolves(case, scheme, vector):
    """Printed from the card, never inferred (Rule 10).

    NEMO's side, read from each run's own ocean.output: ``ln_dynadv_vec`` is T
    on GYRE (``round19_oracle_v2_external/ocean.output:798``) and F on both
    tanks (``lock_kt1_10/ocean.output:705``,
    ``overflow_kt1_10/ocean.output:822``), and ``lk_linssh`` is ``.FALSE.`` in
    all three builds' own ``BLD/ppsrc/nemo/dom_oce.f90`` because none of them
    compiles ``key_linssh``.  So GYRE takes the vector arm and the tanks take
    the thickness-weighted one, and legoESM must agree card for card.
    """
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    config = build_nemo_testcase_card(case).recipe.model_config
    assert config.momentum_advection == scheme
