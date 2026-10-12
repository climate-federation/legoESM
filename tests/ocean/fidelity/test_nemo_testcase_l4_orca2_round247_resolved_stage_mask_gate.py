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
