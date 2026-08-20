"""Unit tests for the shared rollout-curriculum module (jax-free)."""

import pytest

from legoesm.training.curriculum import (
    CurriculumStage,
    build_curriculum_epoch_plan,
    parse_curriculum,
    stage_boundaries,
    stage_for_epoch,
    total_epochs,
)

# The T106 scale-campaign curriculum (config/aimip/scale/base_t106_allyears.yaml).
LEGACY_T106 = ((12, 2), (24, 2), (72, 2), (120, 1))


def test_parse_legacy_tuples():
    stages = parse_curriculum(LEGACY_T106)
    assert stages == (
        CurriculumStage(12.0, 2),
        CurriculumStage(24.0, 2),
        CurriculumStage(72.0, 2),
        CurriculumStage(120.0, 1),
    )
    assert all(s.lr_scale == 1.0 for s in stages)
    assert all(s.pushforward_no_grad_steps == 0 for s in stages)


def test_parse_dict_form():
    stages = parse_curriculum(
        {
            "stages": [
                {"rollout_hours": 6, "n_epochs": 3},
                {
                    "rollout_hours": 24,
                    "n_epochs": 2,
                    "lr_scale": 0.1,
                    "pushforward_no_grad_steps": 2,
                },
            ]
        }
    )
    assert stages[0] == CurriculumStage(6.0, 3)
    assert stages[1] == CurriculumStage(24.0, 2, 0.1, 2)


def test_parse_empty_and_none():
    assert parse_curriculum(None) == ()
    assert parse_curriculum(()) == ()
    assert parse_curriculum([]) == ()


@pytest.mark.parametrize(
    "bad",
    [
        [(0, 2)],  # non-positive hours
        [(6, -1)],  # negative epochs
        [{"rollout_hours": 6, "n_epochs": 1, "lr_scale": 0.0}],
        [{"rollout_hours": 6, "n_epochs": 1, "pushforward_no_grad_steps": -1}],
        [{"rollout_hours": 6}],  # missing n_epochs
        [{"rollout_hours": 6, "n_epochs": 1, "nope": 1}],  # unknown key
        [(6,)],  # malformed pair
        {"phases": []},  # wrong wrapper key
    ],
)
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_curriculum(bad)


def test_parse_skips_zero_epoch_phase():
    # legacy inline behaviour: a zero-epoch phase contributes nothing and is
    # dropped, not an error (matches the pre-extraction epoch-plan build).
    stages = parse_curriculum([(6, 0), (12, 3)])
    assert stages == (CurriculumStage(12.0, 3),)


def test_parse_rejects_decreasing_ladder():
    with pytest.raises(ValueError, match="ladder decreases"):
        parse_curriculum([(24, 1), (6, 1)])
    # escape hatch + equal-hours repeats always fine
    assert len(parse_curriculum([(24, 1), (6, 1)], allow_non_monotonic=True)) == 2
    assert len(parse_curriculum([(6, 1), (6, 2)])) == 2


def test_stage_arithmetic():
    stages = parse_curriculum(LEGACY_T106)
    assert total_epochs(stages) == 7
    assert stage_boundaries(stages) == [0, 2, 4, 6]
    assert stage_for_epoch(stages, 0) == (0, stages[0])
    assert stage_for_epoch(stages, 1) == (0, stages[0])
    assert stage_for_epoch(stages, 2) == (1, stages[1])
    assert stage_for_epoch(stages, 6) == (3, stages[3])
    with pytest.raises(ValueError):
        stage_for_epoch(stages, 7)
    with pytest.raises(ValueError):
        stage_for_epoch(stages, -1)


def test_epoch_plan_matches_legacy_inline_semantics():
    # Byte-for-byte the plan the pre-extraction inline builder produced for
    # the T106 curriculum at dycore dt=1800 s with targets at every lead.
    leads = (12, 24, 72, 120)
    plan = build_curriculum_epoch_plan(LEGACY_T106, leads, 1800.0, 99)
    expected = (
        [(12, 0, 24)] * 2
        + [(24, 1, 48)] * 2
        + [(72, 2, 144)] * 2
        + [(120, 3, 240)] * 1
    )
    assert plan == expected


def test_epoch_plan_fallback_when_no_curriculum():
    assert build_curriculum_epoch_plan(None, (6,), 1800.0, 3) == [
        (None, None, None)
    ] * 3


def test_epoch_plan_requires_loaded_targets():
    with pytest.raises(ValueError, match="multi_step_hours"):
        build_curriculum_epoch_plan(LEGACY_T106, (), 1800.0, 1)
    with pytest.raises(ValueError, match="no loaded target"):
        build_curriculum_epoch_plan(((12, 1),), (6, 24), 1800.0, 1)


def test_epoch_plan_exact_multiple_gate():
    # 6 h at dt_sfno=4 h is 1.5 macro steps: strict mode rejects, permissive
    # mode rounds (the dycore micro-step behaviour).
    with pytest.raises(ValueError, match="exact multiple"):
        build_curriculum_epoch_plan(
            ((6, 1),), (6,), 4 * 3600.0, 1, require_exact=True, dt_name="dt_sfno"
        )
    plan = build_curriculum_epoch_plan(((6, 1),), (6,), 4 * 3600.0, 1)
    assert plan == [(6, 0, 2)]  # round(1.5) banker's-rounds to 2


def test_epoch_plan_is_order_permissive():
    # Plan-build keeps the legacy permissive order (k_target indexes
    # multi_step_hours, not the phase order); ladder strictness lives only
    # in parse_curriculum's default (suite validation).
    plan = build_curriculum_epoch_plan([(24, 1), (12, 1)], (6, 12, 24), 1800.0, 5)
    assert plan == [(24, 2, 48), (12, 1, 24)]


def test_epoch_plan_accepts_parsed_stages():
    stages = parse_curriculum(LEGACY_T106)
    assert build_curriculum_epoch_plan(
        stages, (12, 24, 72, 120), 1800.0, 1
    ) == build_curriculum_epoch_plan(LEGACY_T106, (12, 24, 72, 120), 1800.0, 1)


def test_epoch_plan_accepts_dict_stage_form():
    # {"stages": [...]} must reach the epoch plan, not KeyError on the
    # index probe (codex HIGH regression guard).
    spec = {"stages": [
        {"rollout_hours": 12, "n_epochs": 2},
        {"rollout_hours": 24, "n_epochs": 1},
    ]}
    plan = build_curriculum_epoch_plan(spec, (12, 24), 1800.0, 9)
    assert plan == [(12, 0, 24), (12, 0, 24), (24, 1, 48)]


def test_epoch_plan_generator_keeps_first_phase():
    # A generator of CurriculumStages must not lose its first item to the
    # type probe (codex LOW).
    gen = (s for s in parse_curriculum(LEGACY_T106))
    plan = build_curriculum_epoch_plan(gen, (12, 24, 72, 120), 1800.0, 1)
    assert plan[0] == (12, 0, 24)
    assert len(plan) == 7


def test_epoch_plan_pushforward_bounds():
    with pytest.raises(ValueError, match="pushforward"):
        build_curriculum_epoch_plan(
            [{"rollout_hours": 6, "n_epochs": 1, "pushforward_no_grad_steps": 12}],
            (6,),
            1800.0,
            1,
        )
    # valid: 12 steps at dt=1800, 2 no-grad prefix steps
    plan = build_curriculum_epoch_plan(
        [{"rollout_hours": 6, "n_epochs": 1, "pushforward_no_grad_steps": 2}],
        (6,),
        1800.0,
        1,
    )
    assert plan == [(6, 0, 12)]
