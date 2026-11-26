"""Known-answer controls for the ORCA2 round-195 operand split."""

from __future__ import annotations

import copy
import inspect

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round195_transport_operands as gate,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
)


def _row(bit_exact: bool, cells: int = 0) -> dict[str, object]:
    return {
        "bit_exact": bit_exact,
        "differing_cells": cells,
        "first_unequal_index": None if bit_exact else [0, 0],
        "maximum_absolute": 0.0 if bit_exact else 1.0,
    }


def _report() -> dict[str, object]:
    exact = _row(True)
    debt = _row(False, 4)
    return {
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": {
            "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
        },
        "record_fields": {
            "j003_va_ext": True,
            "j003_hvp2_e": True,
            "j003_zhV": True,
        },
        "operand_order": ["e1v", "va_e", "zhvp2_e"],
        "passivity": {"ssh": True, "u": True},
        "prerequisite_transport_exact": [True, True, False],
        "split": {
            "operand_rows": {
                "e1v": copy.deepcopy(exact),
                "va_e": copy.deepcopy(debt),
                "zhvp2_e": copy.deepcopy(exact),
            },
            "rows": {
                "first_product": copy.deepcopy(debt),
                "unmasked_transport_v": copy.deepcopy(debt),
            },
            "single_substitution_transport_v": {
                "e1v": copy.deepcopy(debt),
                "va_e": copy.deepcopy(exact),
                "zhvp2_e": copy.deepcopy(debt),
            },
            "cumulative_substitution_transport_v": {
                "e1v": copy.deepcopy(debt),
                "va_e": copy.deepcopy(exact),
                "zhvp2_e": copy.deepcopy(exact),
            },
        },
    }


def test_classifier_names_first_operand_and_closing_substitution():
    result = gate.classify(_report())

    assert result["first_nonbit_boundary"] == "va_e"
    assert result["closing_single_substitution"] == "va_e"
    assert result["prediction_ledger"]["R195-P3"] == "CONFIRMED_VA_E_FIRST"
    assert result["status"] == "PASS_R195_SUBSTEP3_OPERAND_SPLIT"


def test_measurement_uses_only_current_barotropic_hook_api():
    source = inspect.getsource(gate.measure)
    signature = inspect.signature(barotropic_substeps_latlon_cgrid)
    requested = {
        "_nemo_substep_trace_test_hook",
        "_nemo_reference_face_depth_test_override",
        "_nemo_unmasked_v_transport_test_override",
        "_nemo_materialize_v_transport_test_override",
        "_nemo_external_mode_association_test_override",
    }

    assert requested <= set(signature.parameters)
    assert "_nemo_unmasked_v_reciprocal_test_override" not in source


@pytest.mark.parametrize(
    ("plant", "message"),
    (
        ("registry-order", "registry reordered"),
        ("operand-bit", "static e1v"),
        ("missing-stream", "stream is absent"),
    ),
)
def test_each_known_answer_plant_refuses(plant: str, message: str):
    with pytest.raises(gate.GateError, match=message):
        gate.classify(_report(), plant)


def test_exact_midpoint_velocity_refutes_prediction_without_invalidating_gate():
    report = _report()
    report["split"]["operand_rows"]["va_e"] = _row(True)
    report["split"]["rows"]["first_product"] = _row(True)
    report["split"]["operand_rows"]["zhvp2_e"] = _row(False, 2)

    result = gate.classify(report)

    assert result["first_nonbit_boundary"] == "zhvp2_e"
    assert result["prediction_ledger"]["R195-P3"] == "REFUTED_FIRST_ZHVP2_E"
