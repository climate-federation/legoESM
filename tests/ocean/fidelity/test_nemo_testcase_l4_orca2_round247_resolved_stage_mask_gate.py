from __future__ import annotations

import copy

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
