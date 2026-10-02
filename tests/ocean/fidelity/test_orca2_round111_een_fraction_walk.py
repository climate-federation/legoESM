import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round111_een_fraction_walk as gate,
)


def test_source_order_matches_compiled_three_fraction_expression():
    assert gate.SOURCE_ORDER == (
        "west_ff", "west_e3f0", "west_r3f", "west_mask", "west_denom", "frac_west",
        "center_ff", "center_e3f0", "center_r3f", "center_mask", "center_denom", "frac_center",
        "south_ff", "south_e3f0", "south_r3f", "south_mask", "south_denom", "frac_south",
        "sum_west_center", "sum_all",
    )


def test_round109_score_reports_magnitude_and_signed_zero_separately():
    candidate = np.array([0.0, 2.0], dtype=np.float64)
    reference = np.array([-0.0, 3.0], dtype=np.float64)
    row = gate.r109._score(candidate, reference)
    assert row["bit_unequal"] == 2
    assert row["magnitude_unequal"] == 1
    assert row["signed_zero_only"] == 1
