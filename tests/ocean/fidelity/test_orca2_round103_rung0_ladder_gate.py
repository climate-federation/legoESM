"""Controls for the round-103 ORCA2 rung-0 ten-step gate."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as gate,
)


def _fields(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "T": np.full((2, 3, 1), value, dtype=np.float64),
        "S": np.full((2, 3, 1), value, dtype=np.float64),
        "u": np.full((2, 3, 1), value, dtype=np.float64),
        "v": np.full((2, 3, 1), value, dtype=np.float64),
        "ssh": np.full((2, 3), value, dtype=np.float64),
    }


def test_checkpoint_requires_bit_exact_for_at_bar() -> None:
    actual = _fields()
    expected = _fields()
    row = gate._checkpoint(1, "entry", actual, expected)
    assert all(item["status"] == "AT_BAR_BIT_EXACT"
               for item in row["rows"].values())
    expected["T"].flat[0] = np.nextafter(0.0, np.inf)
    planted = gate._checkpoint(1, "entry", actual, expected)
    assert planted["rows"]["T"]["status"] == "DEBT"
    assert planted["rows"]["T"]["unequal"] == 1


def test_checkpoint_refuses_nonfinite_candidate() -> None:
    actual = _fields()
    actual["S"].flat[0] = np.nan
    with pytest.raises(gate.GateError, match="candidate is non-finite"):
        gate._checkpoint(4, "stage2", actual, _fields())


def test_source_statement_is_the_measured_round94_boundary() -> None:
    statement = gate.FIRST_SOURCE_STATEMENT
    assert statement["nemo_source"].endswith("stp2d.f90:206-219")
    assert statement["status"] == "HELD_BY_GYRE_2ULP_GATE"


def test_materialization_arm_requires_unmasked_transport() -> None:
    source = gate.Path(gate.__file__).read_text()
    assert "not materialize_v_transport or unmasked_v_transport" in source
