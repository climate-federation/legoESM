"""Unit controls for the round-336 resume-19 Decision-122, parent-provenance,
wet-value, and cross-tree caller output comparator gate.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for _pkg in (REPO, REPO / "packages/core", REPO / "packages/ocean"):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round336_d122_gate as gate,
)


def test_d122_self_test_passes_and_plants_fire(tmp_path: Path) -> None:
    res = gate.self_test(plant="none", tmp_dir=tmp_path)
    assert res["status"] == "PASS_R336_D122_SELF_TEST"
    assert res["census"]["control_commit"] == gate.PARENT_COMMIT
    assert res["census"]["registered_census"] == {"toward": 1, "away": 0, "equal": 2}
    assert res["census"]["first_over_bar"]["old_target"] == [2, "stage3", "T"]
    assert res["census"]["first_over_bar"]["new_first_over_bar"] == [3, "stage3", "T"]
    assert res["caller_audit"]["status"] == "PASS_R336_CALLER_AUDIT"
    assert res["caller_audit"]["parent_commit"] == gate.PARENT_COMMIT
    assert res["caller_audit"]["orca2_unequal_words"] == 3

    for plant_name in ("exact-loss", "away-row", "parent-commit", "caller-unequal"):
        with pytest.raises(gate.GateError):
            gate.self_test(plant=plant_name, tmp_dir=tmp_path)


def test_wet_row_detail_separates_dry_signed_zero_from_wet_value_diff() -> None:
    masks = {"v": np.array([[[False, True]]], dtype=bool)}
    expected = np.array([[[0.0, 2.0]]], dtype=np.float64)
    actual_dry_sz = np.array([[[-0.0, 2.0]]], dtype=np.float64)
    d0 = gate._compute_wet_row_detail("v", actual_dry_sz, expected, masks)
    assert d0["wet_bit_identical"] is True
    assert d0["unequal_wet"] == 0
    assert d0["unequal_dry"] == 1
    assert d0["wet_classification"] == "BIT_EXACT"

    actual_wet_diff = np.array([[[-0.0, 2.0 + 5.0e-12]]], dtype=np.float64)
    d1 = gate._compute_wet_row_detail("v", actual_wet_diff, expected, masks)
    assert d1["wet_bit_identical"] is False
    assert d1["unequal_wet"] == 1
    assert d1["unequal_dry"] == 1
    assert d1["wet_classification"] == "OVER_BAR"
    assert d1["_max_index"] == (0, 0, 1)
