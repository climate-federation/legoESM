"""Frozen outcome values for ORCA2 round 33."""

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round33_outcome_gate as gate,
)


def test_registered_kt10_values_are_the_measured_mask_landing():
    assert gate.EXPECTED_KT10 == {
        "u": 0.33860877536933787,
        "v": 0.5353439568885162,
    }
