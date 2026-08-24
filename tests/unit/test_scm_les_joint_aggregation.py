"""The joint aggregation decides what the multi-case fit actually minimizes.

``joint_score``'s plain mean weights every regime's ABSOLUTE spread-normalized
error equally, so the regime every scheme fits worst absorbs the gradient. On
the shipped five-case run that put 43% of the loss on ekman and 3.1% on bomex.
``--joint-aggregation default_relative`` divides each case by its own DEFAULT
score so every regime enters with equal improvement headroom instead.

These tests pin the arithmetic of both, and the floor that stops an
already-well-fitted case from having its residual noise promoted to the
dominant gradient.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "run" / "run_scm_les_turbulence_tuning.py")


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("_scm_les_tuner", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_scm_les_tuner"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_relative_joint_is_one_at_the_default_parameters(drv):
    """s_i/s_i^0 == 1 for every case when nothing has moved."""
    default = {"bomex": 0.10, "ekman": 3.40, "gabls1": 0.90}
    assert drv.relative_joint(dict(default), default, 0.0) == pytest.approx(1.0)


def test_relative_joint_rewards_headroom_not_absolute_size(drv):
    """The whole point: a 10% gain on any case counts the same.

    bomex and ekman differ by 34x in absolute score. Improving either by 10%
    must move the relative joint by the same amount.
    """
    default = {"bomex": 0.10, "ekman": 3.40}
    better_bomex = {"bomex": 0.09, "ekman": 3.40}
    better_ekman = {"bomex": 0.10, "ekman": 3.06}
    a = drv.relative_joint(better_bomex, default, 0.0)
    b = drv.relative_joint(better_ekman, default, 0.0)
    assert a == pytest.approx(b)
    assert a < 1.0


def test_plain_mean_is_dominated_by_the_largest_case(drv):
    """The behaviour that motivated the alternative, stated as a test.

    The same 10% improvement is worth 34x more to the plain mean when it lands
    on the case with the larger absolute score.
    """
    default = {"bomex": 0.10, "ekman": 3.40}
    d_bomex = sum(default.values()) / 2 - (0.09 + 3.40) / 2
    d_ekman = sum(default.values()) / 2 - (0.10 + 3.06) / 2
    assert d_ekman / d_bomex == pytest.approx(34.0, rel=1e-9)


def test_floor_caps_the_amplification_on_an_already_fitted_case(drv):
    """Without a floor, a case fitted to ~0 divides by ~0."""
    default = {"tiny": 1.0e-6, "normal": 1.0}
    scores = {"tiny": 2.0e-6, "normal": 1.0}
    unfloored = drv.relative_joint(scores, default, 0.0)
    floored = drv.relative_joint(scores, default, 0.05)
    # unfloored: (2.0 + 1.0)/2 = 1.5 -- the noise doubled and dominates
    assert unfloored == pytest.approx(1.5)
    # floored: the tiny case contributes ~0 instead of 2.0
    assert floored < 0.51
    assert floored == pytest.approx((2.0e-6 / 0.05 + 1.0) / 2.0)


def test_relative_joint_is_none_without_normalizers(drv):
    """A run that never captured default scores reports nothing, not a wrong
    number."""
    assert drv.relative_joint({"a": 1.0}, None, 0.05) is None
    assert drv.relative_joint({"a": 1.0}, {}, 0.05) is None


def test_joint_score_mean_matches_hand_arithmetic(drv, monkeypatch):
    """joint_score with case_norm=None is the plain mean of the arm scores."""
    arms = [SimpleNamespace(name=n) for n in ("a", "b", "c")]
    scores = {"a": 0.2, "b": 1.0, "c": 3.0}

    def fake_arm_score(scheme, arm, args, params=None, base_cfg=None):
        return ({}, None, {"theta": scores[arm.name]}, scores[arm.name], False)

    monkeypatch.setattr(drv, "_arm_score", fake_arm_score)
    args = SimpleNamespace(relative_norm_floor=0.05)
    joint, per_case, _ = drv.joint_score("louis", arms, args)
    assert joint == pytest.approx((0.2 + 1.0 + 3.0) / 3.0)
    assert per_case == pytest.approx(scores)


def test_joint_score_relative_divides_by_the_defaults(drv, monkeypatch):
    """case_norm switches the objective but NOT what per_case reports."""
    arms = [SimpleNamespace(name=n) for n in ("a", "b")]
    scores = {"a": 0.1, "b": 2.0}
    norms = {"a": 0.2, "b": 4.0}

    def fake_arm_score(scheme, arm, args, params=None, base_cfg=None):
        return ({}, None, {"theta": scores[arm.name]}, scores[arm.name], False)

    monkeypatch.setattr(drv, "_arm_score", fake_arm_score)
    args = SimpleNamespace(relative_norm_floor=0.0)
    joint, per_case, _ = drv.joint_score("louis", arms, args, case_norm=norms)
    assert joint == pytest.approx(0.5)          # both halved
    # per_case must stay RAW so the per-case table means the same thing in
    # both modes.
    assert per_case == pytest.approx(scores)


def test_cli_defaults_to_the_mean_and_accepts_the_alternative(drv):
    """Default stays 'mean' so an existing command line keeps its meaning."""
    base = ["--cases", "bomex:/nonexistent"]
    args = drv.parse_args(base)
    assert args.joint_aggregation == "mean"
    assert args.relative_norm_floor == pytest.approx(0.05)

    args = drv.parse_args(base + ["--joint-aggregation", "default_relative",
                                  "--relative-norm-floor", "0.01"])
    assert args.joint_aggregation == "default_relative"
    assert args.relative_norm_floor == pytest.approx(0.01)


def test_cli_rejects_an_unknown_aggregation(drv):
    """Dispatch hardening: a typo must not silently select the default."""
    with pytest.raises(SystemExit):
        drv.parse_args(["--cases", "bomex:/nonexistent",
                        "--joint-aggregation", "meen"])
