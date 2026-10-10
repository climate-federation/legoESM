from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round232_v_transport_operand_gate as gate,
)


def _rows(first: str | None):
    return {
        name: {"unequal": int(first == name)} for name in gate.OPERANDS
    }


def test_first_unequal_follows_compiled_statement_order() -> None:
    for expected in (*gate.OPERANDS, None):
        assert gate._first_unequal(_rows(expected)) == expected
    with pytest.raises(gate.GateError, match="compiled operand order moved"):
        gate._first_unequal(_rows("zvb"), tuple(reversed(gate.OPERANDS)))


def test_rank0_mapping_and_ratio_signature() -> None:
    values = np.arange(149 * 180, dtype=np.float64).reshape(149, 180)
    mapped = gate._rank0(values)
    assert mapped.shape == (148, 90)
    assert np.array_equal(mapped, values[1:, :90])
    native = np.arange(148 * 180 * 2, dtype=np.float64).reshape(148, 180, 2)
    assert np.array_equal(gate._rank0_native(native), native[:, :90])
    oracle = np.ones((2, 2), np.float64)
    candidate = np.array([[1.0, -2.0], [3.0, 4.0]])
    signature = gate._ratio_sign(candidate, oracle)
    assert signature["unequal"] == 3
    assert signature["sign_mismatch"] == 1
    assert signature["ratio_median"] == 3.0


def test_each_operand_bit_control_is_nonvacuous() -> None:
    baseline = np.ones((2, 2), np.float64)
    for name in gate.OPERANDS:
        control = baseline.copy()
        control[0, 0] = np.nextafter(control[0, 0], np.inf)
        assert gate._exact(control, baseline)["unequal"] == 1, name


def test_masked_score_excludes_canonical_support_zero() -> None:
    candidate = np.array([2.0, 7.0])
    oracle = np.array([2.0, 0.0])
    row = gate._exact_masked(candidate, oracle, np.array([True, False]))
    assert row["unequal"] == 0
    assert row["argmax"] == [0]
    with pytest.raises(gate.GateError, match="support is empty"):
        gate._exact_masked(candidate, oracle, np.zeros(2, dtype=bool))
