from __future__ import annotations

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round129_substep2_walk as gate,
)


def test_first_nonbit_preserves_source_order():
    rows = [
        {"boundary": "entry", "bit_exact": True},
        {"boundary": "midpoint", "bit_exact": False},
        {"boundary": "coriolis", "bit_exact": False},
    ]
    assert gate.first_nonbit(rows) is rows[1]


def test_first_nonbit_accepts_an_exact_walk():
    assert gate.first_nonbit([{"bit_exact": True}]) is None


def test_first_operand_nonzero_ignores_signed_zero_only_rows():
    rows = [
        {"boundary": "entry", "operand_absolute_max": 0.0},
        {"boundary": "transport", "operand_absolute_max": 1.0},
    ]
    assert gate.first_operand_nonzero(rows) is rows[1]
