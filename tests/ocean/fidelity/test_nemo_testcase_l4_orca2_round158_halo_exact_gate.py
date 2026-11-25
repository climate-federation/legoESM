"""Controls for the ORCA2 round-158 exact halo operand gate."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round158_halo_exact_gate as gate,
)


def test_two_rank_halo_registry_is_compiled_order() -> None:
    assert tuple(row[0] for row in gate.U_HALO_PAIRS) == (
        "rank0-west-outer", "rank0-west-inner",
        "rank0-east-inner", "rank0-east-outer",
        "rank1-west-outer", "rank1-west-inner",
        "rank1-east-inner", "rank1-east-outer",
    )


def test_frame_registry_is_closed() -> None:
    assert gate._frame_name(1, "u") == "j001_ua_new"
    assert gate._frame_name(65, "v") == "j065_va_new"
    with pytest.raises(gate.GateError, match="outside"):
        gate._frame_name(66, "u")
    with pytest.raises(gate.GateError, match="unknown face"):
        gate._frame_name(1, "t")


def test_selector_refuses_unknown_value() -> None:
    for value in ("", "u_cyclic", "v_fold"):
        gate.validate_selector(value)
    with pytest.raises(gate.GateError, match="unknown"):
        gate.validate_selector("plausible")


def test_one_bit_plant_is_nonvacuous() -> None:
    source = np.array([0.0, 2.0], dtype=np.float64)
    planted = gate._plant_one(source)
    row = gate._pair(planted, source)
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1
