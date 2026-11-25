import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round240_hpg_fold_split as gate,
)


def _row(*, at_floor=True, bit_exact=True):
    return {
        "at_floor": at_floor, "bit_exact": bit_exact,
        "differing_cells": 0 if bit_exact else 68,
        "compared_cells": 1319, "absolute_max": 0.0 if at_floor else 1.0,
        "rms": 0.0 if at_floor else 1.0, "argmax": [147, 0, 0],
        "candidate_at_argmax": 0.0, "oracle_at_argmax": 0.0 if at_floor else 1.0,
    }


def _report():
    return {
        "claim_label": "independent hierarchy rung 0",
        "admission": {
            "fold": "PASS_R181_HPG_FOLD_ADMISSION",
            "component": "PASS_R180_HPG1_ADMISSION",
        },
        "component_order": list(gate.COMPONENT_ORDER),
        "primitive_order": list(gate.PRIMITIVE_ORDER),
        "target": {"cells": 68, "row": 147, "wet_levels": 1319},
        "record_component_identity": _row(),
        "component_rows": {name: _row() for name in gate.COMPONENT_ORDER},
        "first_component": None,
        "primitive_rows": {name: _row() for name in gate.PRIMITIVE_ORDER},
        "first_primitive": None,
        "resolved_branch": {"name": "hpg_sco", "zps_hde_executed": False},
    }


def test_classifier_exonerates_an_exact_hpg():
    result = gate.classify(_report())
    assert result["status"] == "HELD_R240_HPG_EXONERATED_REATTRIBUTE_R239"
    assert result["predictions"]["R240-P2"] == "REFUTED"


def test_classifier_names_zhpj_and_its_first_primitive():
    report = _report()
    report["component_rows"]["zhpj"] = _row(at_floor=False, bit_exact=False)
    report["component_rows"]["pvv"] = _row(at_floor=False, bit_exact=False)
    report["primitive_rows"]["north_rhd"] = _row(at_floor=False, bit_exact=False)
    report["first_component"] = {
        "boundary": "zhpj", **report["component_rows"]["zhpj"]}
    report["first_primitive"] = {
        "boundary": "north_rhd", **report["primitive_rows"]["north_rhd"]}
    assert gate.classify(report)["status"] == "HELD_R240_FIRST_HPG_COMPONENT_ZHPJ"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)
