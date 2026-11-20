"""Direct C4 runner/scorer controls, including restart and non-vacuity plants."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean import bulk_flux_omip as bulk
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round345_decision188_gate as decision188,
)
from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
    c4_year_runner as runner,
)
from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
    c4_year_scorer as scorer,
)


def _carry(seed: float, kt: int) -> bulk.NemoORCA2SI3Carry:
    values = [jnp.asarray(seed + index, dtype=jnp.float64) for index in range(16)]
    return bulk.NemoORCA2SI3Carry(*values, jnp.asarray(kt, dtype=jnp.int64))


def _pair(carry: bulk.NemoORCA2SI3Carry) -> bulk.NemoORCA2SI3Carry:
    updates = {
        name: getattr(carry, name) + jnp.asarray(index + 1, dtype=jnp.float64)
        for index, name in enumerate(carry.__slots__[:-1])
    }
    updates["kt"] = carry.kt + jnp.asarray(2, dtype=jnp.int64)
    return carry._replace(**updates)


def _assert_tree_bits_equal(left, right) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for got, want in zip(left_leaves, right_leaves, strict=True):
        got_array = np.asarray(got)
        want_array = np.asarray(want)
        assert got_array.dtype == want_array.dtype
        assert got_array.shape == want_array.shape
        assert got_array.tobytes() == want_array.tobytes()


def _empty_month(month: int) -> dict:
    window = runner.year_metrics.calendar_month_day_ranges()[month - 1]
    return {
        **window,
        "format": runner.C4_METRIC_FORMAT,
        "status": "EVALUATED_COMPLETE_MONTH",
        "metrics": {},
    }


def _synthetic_reports():
    admission = decision188.minimal_envelope_admission()
    candidate = scorer._add_synthetic_ice(
        decision188.minimal_year_report(sst=0.0, ke=0.0), 0.0
    )
    envelope = scorer._add_synthetic_ice(
        decision188.minimal_year_report(sst=0.0, ke=0.001), 0.001
    )
    return candidate, envelope, admission


def test_restart_identity_compiled_pairs_save_restore_bitwise(tmp_path: Path) -> None:
    """N pairs straight equals k pairs + checkpoint + restore + N-k pairs."""

    executable = jax.jit(_pair).lower(_carry(1.0, 1)).compile()
    straight = _carry(1.0, 1)
    for _ in range(4):
        straight = executable(straight)
    jax.block_until_ready(straight)

    split = _carry(1.0, 1)
    for _ in range(2):
        split = executable(split)
    jax.block_until_ready(split)
    checkpoint = tmp_path / runner.checkpoint_name(1, 4)
    runner.write_checkpoint(
        checkpoint,
        split,
        month=1,
        completed_kt=4,
        reference_root=tmp_path / "reference",
        month_report=_empty_month(1),
    )
    restored, metadata, _report = runner.restore_checkpoint(
        checkpoint, _carry(1.0, 1), tmp_path / "reference", validate_calendar=False
    )
    assert metadata["next_kt"] == 5
    for _ in range(2):
        restored = executable(restored)
    jax.block_until_ready(restored)
    _assert_tree_bits_equal(restored, straight)


def test_checkpoint_carries_every_nemo_orca2_si3_leaf(tmp_path: Path) -> None:
    carry = _carry(2.0, 3)
    path = tmp_path / runner.checkpoint_name(1, 2)
    written = runner.write_checkpoint(
        path,
        carry,
        month=1,
        completed_kt=2,
        reference_root=tmp_path / "reference",
        month_report=_empty_month(1),
    )
    leaves, _ = jax.tree_util.tree_flatten(carry)
    assert written["leaf_count"] == len(leaves) == len(carry.__slots__)
    restored, _, _ = runner.restore_checkpoint(
        path, _carry(2.0, 1), tmp_path / "reference", validate_calendar=False
    )
    _assert_tree_bits_equal(restored, carry)


def test_scorer_refuses_checkpoint_set_with_missing_month(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    for month in range(1, 13):
        completed = _empty_month(month)["end_day"] * runner.year_metrics.STEPS_PER_DAY
        runner.write_checkpoint(
            tmp_path / runner.checkpoint_name(month, completed),
            _carry(float(month), completed + 1),
            month=month,
            completed_kt=completed,
            reference_root=reference,
            month_report=_empty_month(month),
        )
    june_kt = _empty_month(6)["end_day"] * runner.year_metrics.STEPS_PER_DAY
    (tmp_path / runner.checkpoint_name(6, june_kt)).unlink()
    with pytest.raises(scorer.C4ScoreError, match="exactly 12 months, found 11"):
        scorer.load_checkpoint_report(tmp_path)


def test_month_checkpoint_validator_refuses_wrong_carry_clock(tmp_path: Path) -> None:
    report = _empty_month(1)
    expected = report["end_day"] * runner.year_metrics.STEPS_PER_DAY
    path = tmp_path / runner.checkpoint_name(1, expected)
    metadata = {"month": 1, "completed_kt": expected - 2, "next_kt": expected - 1}
    with pytest.raises(RuntimeError, match="carry clock is not the end"):
        runner.validate_month_checkpoint(path, metadata, report)


def test_scorer_assembles_all_twelve_checkpoint_months(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    for month in range(1, 13):
        completed = _empty_month(month)["end_day"] * runner.year_metrics.STEPS_PER_DAY
        runner.write_checkpoint(
            tmp_path / runner.checkpoint_name(month, completed),
            _carry(float(month), completed + 1),
            month=month,
            completed_kt=completed,
            reference_root=reference,
            month_report=_empty_month(month),
        )
    report = scorer.load_checkpoint_report(tmp_path)
    assert report["nemo_root"] == str(reference.resolve())
    assert [month["month"] for month in report["months"]] == list(range(1, 13))


def test_c4_control_is_pass_12_of_12_with_per_quantity_verdicts() -> None:
    candidate, envelope, admission = _synthetic_reports()
    result = scorer.score_c4(candidate, envelope, admission)
    assert result["status"] == "PASS_C4_12_OF_12"
    assert result["months_passed"] == 12
    assert all(
        row["pass"]
        for month in result["months"]
        for row in month["quantities"].values()
    )
    assert set(runner.ICE_KEYS) <= set(result["months"][0]["quantities"])


def test_one_month_one_ocean_quantity_plant_flips_pass_to_fail() -> None:
    candidate, envelope, admission = _synthetic_reports()
    candidate["months"][2]["metrics"]["SST"]["rms"] = math.nextafter(0.01, math.inf)
    result = scorer.score_c4(candidate, envelope, admission)
    assert result["status"] == "FAIL_C4_M03_Mar_SST"
    assert result["months_passed"] == 11
    assert result["months"][2]["quantities"]["SST"]["pass"] is False


def test_one_month_ice_area_plant_flips_pass_to_fail() -> None:
    candidate, envelope, admission = _synthetic_reports()
    planted = copy.deepcopy(candidate)
    metric = planted["months"][5]["metrics"]["ICE_AREA_SH"]
    metric["candidate_monthly_mean"] = 10.2
    metric["absolute_difference"] = 0.2
    metric["relative_abs_difference"] = 0.02
    result = scorer.score_c4(planted, envelope, admission)
    assert result["status"] == "FAIL_C4_M06_Jun_ICE_AREA_SH"
    assert result["months_passed"] == 11
    assert result["months"][5]["quantities"]["ICE_AREA_SH"]["pass"] is False


def test_zero_ice_envelope_is_refused_before_verdict() -> None:
    candidate, envelope, admission = _synthetic_reports()
    metric = envelope["months"][0]["metrics"]["ICE_VOLUME_NH"]
    metric["candidate_monthly_mean"] = metric["nemo_monthly_mean"]
    metric["absolute_difference"] = 0.0
    metric["relative_abs_difference"] = 0.0
    with pytest.raises(scorer.C4ScoreError, match="envelope positive"):
        scorer.score_c4(candidate, envelope, admission)


def test_builtin_plants_name_ocean_and_ice_failures() -> None:
    plants = scorer.run_plants()
    assert plants["control"] == "PASS_C4_12_OF_12"
    assert plants["one_month_ocean_quantity_plant"].startswith("FAIL_C4_M04_Apr_SST")
    assert plants["one_month_ice_area_plant"].startswith("FAIL_C4_M08_Aug_ICE_AREA_NH")
