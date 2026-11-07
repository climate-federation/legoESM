"""Binding tests for the ORCA2 round-43 tracer-handoff gate."""

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round43_tracer_handoff_gate as gate,
)


def test_bit_score_is_bitwise_and_one_ulp_sensitive():
    expected = np.zeros((2, 2), dtype=np.float64)
    mask = np.asarray([[True, False], [True, True]])
    exact = gate.bit_score(expected, expected, mask)
    assert exact["bit_exact"]
    actual = expected.copy()
    actual[1, 1] = np.nextafter(actual[1, 1], np.inf)
    row = gate.bit_score(actual, expected, mask)
    assert not row["bit_exact"]
    assert row["unequal"] == 1
    assert row["count"] == 3


def test_first_non_bit_follows_compiled_boundary_order():
    rows = {
        name: {"bit_exact": name not in ("after_sbc_T", "stage1_T")}
        for name in gate.ORDER
    }
    assert gate.first_non_bit(rows) == "after_sbc_T"
