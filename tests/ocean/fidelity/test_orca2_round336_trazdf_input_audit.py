"""Controls for the round-336 admitted-input classifier."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round336_trazdf_input_audit as audit,
)


def _report():
    zero = {
        "support": 18,
        "bit_unequal": 0,
        "value_unequal": 0,
        "max_abs": 0.0,
        "rms": 0.0,
    }
    return {
        "execution": "production-jit-cpu-fp64-x64-libm",
        "row_order": list(audit.ROW_ORDER),
        "selectors": copy.deepcopy(audit.SELECTORS),
        "observer_passivity": {"bit_identical": True, "changed_leaves": 0},
        "target_count_T": 18,
        "recorded_input_scores": {
            "e3w_Kmm_target_columns": copy.deepcopy(zero),
            "e3t_Kaa_targets": copy.deepcopy(zero),
            "content_rhs_T_targets": copy.deepcopy(zero),
        },
        "missing_discriminating_rows": list(audit.MISSING_ROWS),
        "first_differing_operand_or_statement": None,
        "claim_label": "PLAUSIBLE",
    }


def test_clean_report_stops_for_record():
    result = audit.classify(_report())
    assert result["status"] == "PASS_R336_TRAZDF_INPUT_AUDIT_ACQUISITION_NEEDED"
    assert result["predictions"]["R336-P3"] == "TRIGGERED"


@pytest.mark.parametrize("plant", audit.PLANTS)
def test_every_classifier_plant_fires(plant):
    with pytest.raises(audit.GateError):
        audit.classify(_report(), plant=plant)
