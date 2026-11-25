from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round244_inloop_fold_audit as gate,
)


def _diff(unequal: int = 0) -> dict:
    return {
        "support": 1,
        "unequal": unequal,
        "signed_zero_only": 0,
        "max_abs": 0.0 if unequal == 0 else 1.0,
        "argmax": [0],
        "first_nonfinite": None,
    }


def _regional(unequal: int = 0) -> dict:
    return {name: _diff(unequal) for name in ("complete", "interior", "fold_band")}


def _report(label: str) -> dict:
    rows = []
    for substep in gate.SUBSTEPS:
        exits = {}
        for name in gate.EXIT_ORDER:
            fold = _diff(1 if (substep, name) == (2, "eta_exit") else 0)
            fold["sign"] = gate.SIGN[name]
            exits[name] = {
                "candidate_fold_identity": fold,
                "candidate_vs_nemo": _regional(),
            }
        pair = {name: _regional() for name in gate.PAIR_ORDER}
        if substep == 6:
            pair["v_mid"] = _regional(129)
            pair["face_depth_v_mid"] = _regional(120)
        rows.append({"substep": substep, "exits": exits, "pair": pair})
    return {
        "status": "PASS_R244_INLOOP_FOLD_AUDIT",
        "label": label,
        "source": {"lines": "721-739"},
        "substeps": list(gate.SUBSTEPS),
        "exit_order": list(gate.EXIT_ORDER),
        "pair_order": list(gate.PAIR_ORDER),
        "passivity": {"eta": True},
        "rows": rows,
        "one_bit_control": _diff(1),
    }


def test_regional_support_excludes_canonical_fill():
    candidate = np.zeros((148, 90), np.float64)
    oracle = candidate.copy()
    candidate[0, 0] = 1.0
    support = np.ones_like(candidate, dtype=bool)
    support[0, 0] = False
    result = gate._regional(candidate, oracle, support)
    assert result["complete"]["unequal"] == 0


def test_classification_requires_both_labels_and_every_control():
    reports = [_report("independent"), _report("given_nemo_entry")]
    result = gate.classify(reports)
    assert result["first_fold_violation"]["field"] == "eta_exit"
    assert result["first_fold_violation"]["substep"] == 2
    for plant in gate.CLASSIFY_PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(copy.deepcopy(reports), plant=plant)


def test_candidate_must_close_fold_and_both_pair_operands():
    baseline = [_report("independent"), _report("given_nemo_entry")]
    candidate = copy.deepcopy(baseline)
    for report in candidate:
        for row in report["rows"]:
            for values in row["exits"].values():
                values["candidate_fold_identity"]["unequal"] = 0
        for values in report["rows"][5]["pair"].values():
            values["complete"]["unequal"] = 0
    result = gate.classify_candidate(baseline, candidate)
    assert result["status"] == "PASS_R244_COMPLETE_INLOOP_ASSOCIATION"

    candidate[0]["rows"][5]["pair"]["v_mid"]["complete"]["unequal"] = 1
    with pytest.raises(gate.GateError):
        gate.classify_candidate(baseline, candidate)


def test_one_bit_control_is_nonvacuous():
    result = gate._one_bit_control()
    assert result["unequal"] == 1
    assert result["max_abs"] > 0.0
