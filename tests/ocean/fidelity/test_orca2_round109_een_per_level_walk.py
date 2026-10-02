import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round109_een_per_level_walk as gate,
)


def test_score_separates_signed_zero_from_magnitude():
    candidate = np.array([[[0.0, 2.0]]], dtype=np.float64)
    reference = np.array([[[-0.0, 3.0]]], dtype=np.float64)
    row = gate._score(candidate, reference)
    assert row["bit_unequal"] == 2
    assert row["signed_zero_only"] == 1
    assert row["magnitude_unequal"] == 1


def test_source_order_starts_at_compiled_zpvo_assignment():
    assert gate.SOURCE_ORDER == (
        "zpvo_nw", "e3u_live", "e3v_live", "neighbor_mask", "term_nw")
