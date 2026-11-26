from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round246_atomic_pair_gate as r246,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round247_resolved_stage_mask_gate as gate,
)
from tests.ocean.fidelity import (
    test_nemo_testcase_l4_orca2_round246_atomic_pair_gate as fixtures,
)


def _report():
    report = fixtures._report()
    for operands in report["operands"].values():
        operands["correction_input_rows"] = {
            name: fixtures._difference()
            for name in ("vn_adv", "r1_hv", "vv_b")
        }
        operands["zFv_signature"] = fixtures._difference()
    return report


def test_classifier_accepts_only_the_exact_resolved_mask_signature():
    result = gate.classify(_report())
    assert result["status"] == "PASS_R247_RESOLVED_STAGE_MASK_ATOMIC_KT10"
    assert result["predictions"]["R247-P1"].endswith("BOTH_LABELS")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_round247_plant_refuses(plant):
    with pytest.raises((gate.GateError, r246.GateError)):
        gate.classify(copy.deepcopy(_report()), plant=plant)


def test_live_w_boundary_is_reconstructed_from_completed_state():
    field = lambda value: SimpleNamespace(data=np.asarray(value, np.float64))
    state = SimpleNamespace(
        eta=field([[np.nan, 0.0]]),
        H_bathy=field([[100.0, 200.0]]),
        T=field([[[1.0, 1.0], [1.0, 1.0]]]),
        S=field([[[35.0, 35.0], [35.0, 35.0]]]),
        u=field([[[0.0, 0.0], [0.0, 0.0]]]),
        v=field([[[0.0, 0.0], [0.0, 0.0]]]),
    )
    card = SimpleNamespace(recipe=SimpleNamespace(z_coord=SimpleNamespace(
        nemo_e3w_0=np.ones((1, 2, 2), np.float64))))
    row = gate._live_w_row(card, state)
    assert row["invalid_e3w_count"] == 1
    assert row["first_invalid_jik"] == [0, 0, 0]
    assert row["field_nonfinite_counts"]["eta"] == 1
    assert np.isnan(row["first_invalid"]["ssh_m"])


def _held_report():
    qualified = _report()
    clean = {
        "invalid_e3w_count": 0,
        "first_invalid_jik": None,
        "field_nonfinite_counts": {
            "S": 0, "T": 0, "eta": 0, "u": 0, "v": 0,
        },
    }
    bad = {
        "invalid_e3w_count": 476557,
        "first_invalid_jik": [1, 49, 0],
        "field_nonfinite_counts": {
            "S": 430552, "T": 430552, "eta": 26640,
            "u": 803640, "v": 799200,
        },
        "first_invalid": {
            "ht0_m": 729.0, "raw_e3w0_m": 10.00035061113249,
            "ssh_m": float("nan"), "live_e3w_m": float("nan"),
        },
    }
    boundaries = {}
    logs = {}
    for label in gate.LABELS:
        boundaries[label] = {
            "status": "PASS_R247_LIVE_W_BOUNDARY_LOCATED",
            "label": label,
            "observer": "offline completed-stage reconstruction",
            "candidate_kt4": {
                name: copy.deepcopy(clean)
                for name in ("entry", "stage1", "stage2", "stage3")
            },
            "nemo_kt4": {
                name: copy.deepcopy(clean)
                for name in ("entry", "stage1", "stage2", "stage3")
            },
            "candidate_kt5": {
                "entry": copy.deepcopy(clean), "stage1": copy.deepcopy(bad),
            },
            "nemo_kt5": {
                name: copy.deepcopy(clean)
                for name in ("entry", "stage1", "stage2")
            },
            "first_bad_kt5_completed_boundary": "stage1",
        }
        logs[label] = "\n".join([
            *(f"PROGRESS round241 {label} kt={kt}" for kt in range(1, 5)),
            "raw-mesh e3w_int must contain only finite values > 0",
            "STATUS REFUSE none:",
        ])
    return {
        "direct": qualified["direct"], "atomic_logs": logs,
        "operands": qualified["operands"], "boundaries": boundaries,
    }


def test_held_classifier_names_the_stage1_downstream_boundary():
    result = gate.classify_held(_held_report())
    assert result["status"] == "HELD_R247_EXACT_ZFV_KT5_STAGE1_BLOWUP"
    assert result["predictions"]["R247-P3"].startswith("REFUTED")


@pytest.mark.parametrize("plant", gate.HELD_PLANTS[1:])
def test_every_round247_held_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify_held(_held_report(), plant=plant)
