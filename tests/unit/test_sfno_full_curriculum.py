"""Pure-Python unit tests for the sfno_full rollout-curriculum epoch plan.

Covers :func:`legoesm.training.neural_gcm_spectral.build_sfno_curriculum_epoch_plan`
— the NeuralGCM-style curriculum epoch-plan builder wired into
``_train_sfno_full_loop`` (mirror of the dycore-mode ``_train_spectral_loop``
curriculum). The function is pure python/math (no jax), so these tests exercise:

  * correct per-phase ``n_sfno_steps`` = round(lead*3600/dt_sfno) and
    ``k_target`` = multi_step_hours.index(lead), one plan entry per phase epoch;
  * a ``ValueError`` when a curriculum lead has no loaded target
    (lead not in ``multi_step_hours``), and when ``multi_step_hours`` is empty;
  * ``rollout_curriculum=None`` -> the ``(None, None, None)`` fallback plan of
    length ``n_epochs_fallback`` (the non-curriculum path is unchanged).

The helper is AST-extracted and exec'd in an isolated namespace so the test runs
WITHOUT importing the parent module (which imports jax) — login-node / jax-free
safe. Under normal CI (jax available) the same function object is exercised.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "packages" / "ml" / "legoesm" / "training" / "neural_gcm_spectral.py"
)
_FUNC_NAME = "build_sfno_curriculum_epoch_plan"


def _load_pure_helper():
    """Extract + exec ONLY ``build_sfno_curriculum_epoch_plan`` (no jax import)."""
    src = _MODULE_PATH.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == _FUNC_NAME:
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(mod, str(_MODULE_PATH), "exec"), ns)  # noqa: S102
            return ns[_FUNC_NAME]
    raise AssertionError(f"{_FUNC_NAME} not found in {_MODULE_PATH}")


build = _load_pure_helper()


def test_per_phase_nsteps_and_ktarget():
    """n_sfno_steps + k_target correct per phase; one plan entry per phase epoch."""
    dt_sfno = 21600.0  # 6 h macro step
    multi_step_hours = [6, 12, 24, 48, 120]
    curriculum = [[6, 4], [12, 4], [24, 6], [48, 6], [120, 8]]
    plan = build(curriculum, multi_step_hours, dt_sfno, n_epochs_fallback=99)

    # Total epochs == sum of phase epochs (overrides the fallback).
    assert len(plan) == 4 + 4 + 6 + 6 + 8 == 28

    # Expected (lead, k_target, n_sfno_steps): steps = round(lead*3600/dt_sfno).
    expected = {
        6: (6, 0, 1),
        12: (12, 1, 2),
        24: (24, 2, 4),
        48: (48, 3, 8),
        120: (120, 4, 20),
    }
    # Each phase contributes exactly ``ep`` identical specs, contiguously.
    idx = 0
    for lead, ep in curriculum:
        for _ in range(ep):
            assert plan[idx] == expected[lead], (idx, plan[idx], expected[lead])
            idx += 1
    assert idx == len(plan)


def test_ktarget_follows_multi_step_hours_order():
    """k_target is the INDEX into multi_step_hours, not the phase order."""
    # multi_step_hours deliberately not equal to the curriculum order.
    multi_step_hours = [6, 12, 24, 48, 120]
    # Curriculum visits 24 h before 12 h: k_target must still be the msh index.
    plan = build([[24, 1], [12, 1]], multi_step_hours, 21600.0, n_epochs_fallback=5)
    assert plan[0] == (24, 2, 4)   # 24h -> msh index 2, 4 macro steps
    assert plan[1] == (12, 1, 2)   # 12h -> msh index 1, 2 macro steps


def test_exact_multiple_leads():
    """n_sfno_steps = lead*3600/dt_sfno when the lead is an exact multiple."""
    # dt_sfno = 3 h; 9 h -> 3 steps, 12 h -> 4 steps (both exact multiples).
    plan = build([[9, 1], [12, 1]], [9, 12], 10800.0, n_epochs_fallback=1)
    assert plan[0] == (9, 0, 3)
    assert plan[1] == (12, 1, 4)


def test_lead_not_multiple_of_dt_sfno_raises():
    """A curriculum lead that is not an exact positive multiple of dt_sfno is a
    hard ValueError (round() would silently supervise the wrong horizon or give a
    zero-gradient phase). dt_sfno=4 h, lead=6 h -> 1.5 macro steps -> reject."""
    with pytest.raises(ValueError, match=r"not a positive exact multiple"):
        build([[6, 1]], [6], 14400.0, n_epochs_fallback=1)
    # dt_sfno=12 h, lead=6 h -> 0.5 steps -> reject (would be 0 steps = no rollout).
    with pytest.raises(ValueError, match=r"not a positive exact multiple"):
        build([[6, 1]], [6], 43200.0, n_epochs_fallback=1)


def test_lead_not_in_multi_step_hours_raises():
    """A curriculum lead with no loaded target is a hard ValueError."""
    with pytest.raises(ValueError, match=r"Curriculum lead 72h has no loaded target"):
        build([[6, 2], [72, 2]], [6, 12, 24], 21600.0, n_epochs_fallback=4)


def test_empty_multi_step_hours_with_curriculum_raises():
    """A curriculum with no loaded targets at all is a hard ValueError."""
    with pytest.raises(ValueError, match=r"needs loss_config.multi_step_hours"):
        build([[6, 2]], [], 21600.0, n_epochs_fallback=4)
    with pytest.raises(ValueError, match=r"needs loss_config.multi_step_hours"):
        build([[6, 2]], None, 21600.0, n_epochs_fallback=4)


def test_none_curriculum_returns_fallback_plan():
    """rollout_curriculum=None -> a no-op plan of length n_epochs_fallback."""
    plan = build(None, [6, 12], 21600.0, n_epochs_fallback=7)
    assert plan == [(None, None, None)] * 7
    # Empty tuple is also 'no curriculum'.
    plan2 = build((), [6, 12], 21600.0, n_epochs_fallback=3)
    assert plan2 == [(None, None, None)] * 3


def test_tuple_input_accepted():
    """Accepts tuple-of-tuple curricula (the NamedTuple config form)."""
    plan = build(((6, 1), (24, 2)), (6, 24), 21600.0, n_epochs_fallback=1)
    assert plan == [(6, 0, 1), (24, 1, 4), (24, 1, 4)]


# --- Config-artifact guard: the GraphCast-scale suite parses + carries the two
# levers (rollout curriculum + many-year windows). yaml-only, no jax. ---

_SUITE_PATH = (
    Path(__file__).resolve().parents[2]
    / "config" / "aimip" / "wbcompare" / "suite_sfno_full_scale.yaml"
)


def test_scale_suite_parses_and_has_curriculum_and_windows():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(_SUITE_PATH.read_text())
    assert cfg["variants"] == ["sfno_full"]
    co = cfg["cfg_overrides"]

    # (1) rollout curriculum present, short->long, leads all have loaded targets.
    curriculum = co["aimip_rollout_curriculum"]
    assert curriculum and all(len(p) == 2 for p in curriculum)
    leads = [int(h) for h, _ in curriculum]
    assert leads == sorted(leads), "curriculum leads must be short-lead-first"
    msh = [int(h) for h in co["loss"]["multi_step_hours"]]
    assert all(lead in msh for lead in leads), (leads, msh)

    # (2) many-year seasonal data windows present (GraphCast-scale lever).
    windows = co["train_windows"]
    assert len(windows) >= 100, "expected many-year x seasonal coverage"
    years = {int(w[0]) for w in windows}
    assert len(years) >= 20, "expected multi-decade year coverage"

    # Plan built from the suite's own leads is consistent with the helper.
    dt_sfno = float(co["dt_sfno"])
    plan = build(
        [(int(h), int(ep)) for h, ep in curriculum], msh, dt_sfno,
        n_epochs_fallback=int(co["aimip_n_epochs"]),
    )
    assert len(plan) == sum(int(ep) for _, ep in curriculum)
    for (lead, _ep), spec in zip(curriculum, _first_of_each_phase(plan, curriculum)):
        assert spec[0] == int(lead)
        assert spec[2] == round(int(lead) * 3600.0 / dt_sfno)


def _first_of_each_phase(plan, curriculum):
    """Yield the first plan spec of each curriculum phase (for per-lead checks)."""
    idx = 0
    for _lead, ep in curriculum:
        yield plan[idx]
        idx += int(ep)
