"""Binding tests for the ORCA2 round-42 stage-1 HPG walk."""

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round42_stage1_hpg_walk_gate as gate,
)


def _row(exact: bool) -> dict:
    return {
        "bit_exact": exact,
        "differing_cells": 0 if exact else 1,
        "count": 8,
        "absolute_max": 0.0 if exact else 1.0,
    }


def test_first_non_bit_uses_registered_input_then_statement_order():
    rows = {name: _row(True) for name in gate.INPUT_ORDER + gate.STATEMENT_ORDER}
    rows["zuap_u"] = _row(False)
    rows["e3w"] = _row(False)
    assert gate.first_non_bit(rows)["boundary"] == "e3w"


def test_owned_assembly_closes_two_rank_column_census():
    records = []
    for rank in range(2):
        field = np.zeros((152, 94, 2), dtype=np.float64)
        field[2:-2, 2:-2] = rank + 1
        records.append({"fields": {"rhd": field}})
    result = gate._owned(records, "rhd")
    assert result.shape == (148, 180, 2)
    np.testing.assert_array_equal(result[:, :90], 1.0)
    np.testing.assert_array_equal(result[:, 90:], 2.0)


def test_support_pair_wraps_last_to_first_cell():
    field = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    pair = gate._support_pair(field, np.asarray([1]))
    np.testing.assert_array_equal(pair[0, 0], field[1, -1])
    np.testing.assert_array_equal(pair[0, 1], field[1, 0])
