"""Known-answer controls for the round-298 source-order classifier."""

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round298_stage7_writer as gate,
)


def _regions(value):
    row = {"max_abs": value, "unequal": int(value != 0.0)}
    return {"complete": dict(row), "interior": dict(row), "fold_band": dict(row)}


def _label(name):
    rows = {field: _regions(0.0) for boundary in gate.SOURCE_ORDER
            for field in ((boundary + "_u", boundary + "_v")
                          if boundary in ("rhs", "pre", "cor") else ())}
    rows.update({name_: _regions(0.0) for name_ in (
        "trp_u", "trp_v", "trp_w", "adv_t", "adv_s", "sbc_t", "sbc_s",
        "qco_t", "qco_s", "final_u", "final_v", "final_t", "final_s")})
    rows["qco_s"]["fold_band"]["max_abs"] = 40.0
    return {
        "status": "PASS_R298_LABEL_MEASUREMENT", "label": name,
        "passivity": {"T": True, "S": True}, "rows": rows,
        "entry": {"T": _regions(1.0), "S": _regions(2.0)},
        "candidate_replay": {"T": _regions(0.0), "S": _regions(0.0)},
        "oracle_transport_qco_arm": {"T": _regions(0.0), "S": _regions(30.0)},
        "oracle_adv_qco_arm": {"T": _regions(0.0), "S": _regions(2.0)},
        "unforced_checks": {"oracle_adv_t_equals_sbc_t": True,
                            "oracle_adv_s_equals_sbc_s": True,
                            "candidate_adv_t_equals_sbc_t": True,
                            "candidate_adv_s_equals_sbc_s": True},
        "one_bit_control": {"unequal": 1},
    }


def _report():
    return {
        "admission": {"status": "PASS_R297_OMT4_KT7_STAGE1_ADMISSION"},
        "source_order": list(gate.SOURCE_ORDER),
        "labels": [_label(name) for name in gate.LABELS],
    }


def test_classifier_names_advection_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["classification"]["first_growth_carrier"] == "adv"
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(copy.deepcopy(_report()), plant)


def test_subnormal_flush_is_counted_as_matching():
    tiny = __import__("numpy").nextafter(0.0, 1.0)
    row = gate._score([0.0], [tiny], [True])
    assert row["raw_unequal"] == 1
    assert row["subnormal_matches"] == 1
    assert row["unequal"] == 0
