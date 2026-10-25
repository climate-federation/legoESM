"""Controls for round 333's kt=2 tracer acquisition admission."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round333_tracer_gate as gate,
)


def test_inherited_stage3_schema_and_field_order() -> None:
    result = gate.self_test()
    assert result["status"] == "PASS_R333_TRACER_PARSER_CONTROL"


@pytest.mark.parametrize("plant", ("header", "truncation"))
def test_parser_plants_fire(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.self_test(plant=plant)


def test_comparison_counts_signed_zero_bits() -> None:
    reference = np.array([0.0, 2.0], dtype=np.float64)
    candidate = np.array([-0.0, np.nextafter(2.0, np.inf)], dtype=np.float64)
    row = gate._comparison(reference, candidate)
    assert row["unequal"] == 2
    assert row["max_abs"] > 0.0
