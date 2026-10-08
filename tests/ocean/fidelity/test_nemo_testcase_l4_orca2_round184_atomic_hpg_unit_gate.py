from __future__ import annotations

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round184_atomic_hpg_unit_gate as gate,
)


def test_decision96_excludes_score_equal_rows_from_vote():
    census = {"toward": 26, "away": 23, "equal": 34}
    assert gate._strict_score_moved_majority(census)
    assert not gate._strict_score_moved_majority(
        census, include_equal=True)


def test_decision96_requires_strict_score_moved_majority():
    assert not gate._strict_score_moved_majority(
        {"toward": 23, "away": 23, "equal": 154})
    assert not gate._strict_score_moved_majority(
        {"toward": 22, "away": 23, "equal": 155})
