"""Unit controls for the round-341 current-tip re-score, 3-D wet-mask audit, and
offline source-order walk gate.
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
    nemo_testcase_l4_orca2_round341_rescore_walk_gate as gate,
)


def test_round341_self_test_passes_and_plants_fire() -> None:
    res = gate.self_test(plant="none")
    assert res["status"] == "PASS_R341_SELFTEST"
    assert res["e3t_stored_hex"] == "0x40234d4bd6440687"
    assert res["e3t_rebuilt_hex"] == "0x40234d4bd6440686"
    assert res["T_stage2_stored_hex"] == "0xbff663da73f853c6"
    assert res["T_stage2_rebuilt_hex"] == "0xbff663da73f853c7"

    for plant_name in (
        "signed-zero",
        "wet-3d-mask",
        "stage2-thickness",
        "walk-closure",
        "exact-loss",
        "away-row",
        "parent-commit",
        "caller-unequal",
    ):
        with pytest.raises((gate.GateError, StopIteration)):
            gate.self_test(plant=plant_name)


def test_compute_row_wet_detail_separates_3d_subbottom_dry_and_wet_value() -> None:
    masks_2d = {"v": np.array([[[True, True, False]]], dtype=bool)}
    masks_3d = {
        "v": np.array([[[True, False, False]]], dtype=bool),
        "ss_v": np.array([[True]], dtype=bool),
    }
    expected = np.array([[[0.25, 0.0, 0.0]]], dtype=np.float64)
    actual_dry_sz = np.array([[[0.25, -0.0, 0.0]]], dtype=np.float64)
    d0 = gate._compute_row_wet_detail(
        "v", actual_dry_sz, expected, masks_2d, masks_3d
    )
    assert d0["wet_bit_unequal"] == 1
    assert d0["wet_3d_bit_identical"] is True
    assert d0["wet_3d_bit_unequal"] == 0
    assert d0["unequal_3d_wet"] == 0
    assert d0["unequal_3d_dry"] == 1
    assert d0["subbottom_dry_bit_unequal"] == 1
    assert d0["surface_dry_bit_unequal"] == 0
    assert d0["wet_3d_classification"] == "BIT_EXACT"

    actual_wet_ulp = np.array(
        [[[0.25 + np.spacing(0.25), -0.0, 0.0]]], dtype=np.float64
    )
    d1 = gate._compute_row_wet_detail(
        "v", actual_wet_ulp, expected, masks_2d, masks_3d
    )
    assert d1["wet_3d_bit_identical"] is False
    assert d1["wet_3d_bit_unequal"] == 1
    assert d1["unequal_3d_wet"] == 1
    assert d1["unequal_3d_dry"] == 1
    assert d1["wet_3d_classification"] == "AT_BAR_NOT_EXACT"
    assert d1["_max_index"] == (0, 0, 0)


def test_walk_source_order_and_plants_on_admitted_records() -> None:
    r5_root = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round5/acquisition/orca1ice_surface_entry_every_step_a_np2"
    )
    r326_root = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round326/acquisition/orca1ice_tke_hmi_10step_np2"
    )
    r332_root = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round332/acquisition/orca1ice_raw_postfold_v_kt1_np2"
    )
    if not (r5_root.exists() and r326_root.exists()):
        pytest.skip("admitted round-5 / round-326 records not present on host")

    res = gate.walk_source_order(
        r5_root,
        r326_root,
        r332_root if r332_root.exists() else None,
        plant="none",
    )
    assert res["status"] == "PASS_R341_SOURCE_ORDER_WALK"
    assert res["cross_record_r5_vs_r326_stage12_unequal"] == 0
    w_raw = res["first_raw_non_bit_row_walk"]
    assert w_raw["verdict"] == "CONFIRMED"
    assert w_raw["wet_3d_unequal"] == 0
    assert w_raw["dry_3d_unequal"] == 7534
    assert w_raw["subbottom_dry_unequal"] == 5550
    assert w_raw["surface_dry_unequal"] == 1984
    assert w_raw["subbottom_dry_ssvmask_unequal"] == 5884
    assert w_raw["surface_dry_ssvmask_unequal"] == 1650
    assert w_raw["north_fold_subbottom_dry_faces"] == 334
    assert w_raw["legoesm_dry_word_hex"] == "0x8000000000000000"
    assert w_raw["raw_nemo_dry_word_hex"] == "0x8000000000000000"
    assert w_raw["canonicalized_oracle_dump_dry_word_hex"] == "0x0000000000000000"

    w_wet = res["first_wet_3d_non_bit_row_walk"]
    assert w_wet["verdict"] == "CONFIRMED"
    assert w_wet["total_interior_wet_T_current_unequal_kt1_10"] == 1
    assert w_wet["total_interior_wet_T_proposed_unequal_kt1_10"] == 0
    assert w_wet["total_interior_wet_S_current_unequal_kt1_10"] == 0
    assert w_wet["total_interior_wet_S_proposed_unequal_kt1_10"] == 0

    for plant_name in ("stage2-thickness", "walk-closure"):
        with pytest.raises(gate.GateError):
            gate.walk_source_order(r5_root, r326_root, plant=plant_name)
