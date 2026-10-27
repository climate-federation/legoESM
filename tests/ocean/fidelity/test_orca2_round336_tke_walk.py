"""Focused controls for the round-336 production TKE boundary walk."""

from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round336_tke_walk as walk,
)


def test_score_is_wet_bitwise_and_reports_maximum() -> None:
    expected = np.array([0.0, 1.0, 2.0])
    actual = expected.copy()
    actual[0] = -0.0
    actual[1] = np.nextafter(actual[1], np.inf)
    row = walk._score(actual, expected, np.array([False, True, True]))
    assert row["wet_words"] == 2
    assert row["unequal_wet_words"] == 1
    assert row["max_abs"] == row["max_abs_unequal"]


def test_walk_uses_two_actual_card_steps_and_orders_post_solve_first() -> None:
    source = Path(walk.__file__).read_text(encoding="utf-8")
    assert "for kt in (1, 2):" in source
    assert "model.step(" in source
    assert source.index('"solve_back", "after_solve"') < source.index(
        '"after_solve": (trace.en_post_sweep')
    assert '"final_avt"' in source

