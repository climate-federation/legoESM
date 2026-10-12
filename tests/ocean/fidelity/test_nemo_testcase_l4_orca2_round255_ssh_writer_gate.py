from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round255_ssh_writer_gate as gate,
)


def _score(unequal: int = 0) -> dict[str, object]:
    return {"unequal": unequal, "max_abs": float(bool(unequal))}


def _report(label: str) -> dict[str, object]:
    boundaries = {name: _score(1) for name in gate.BOUNDARIES}
    operands = {name: _score(1) for name in gate.OPERANDS}
    return {
        "status": "PASS_R255_SSH_WRITER_SPLIT",
        "label": label,
        "round254_dependency": "HELD_R254_SSHN_E_FIRST_NONBIT",
        "operand_route": "production-d114-resolved",
        "boundary_order": list(gate.BOUNDARIES),
        "operand_order": list(gate.OPERANDS),
        "boundary_rows": boundaries,
        "operand_rows": operands,
        "boundary_identity": {
            "literal_to_candidate_clamp": _score(1),
            "clamp_to_association": _score(),
            "association_to_rotation": _score(),
        },
        "first_nonbit_boundary": "literal_continuity_output",
        "first_nonbit_operand": "sshn_e",
        "candidate_divergence_replay": _score(),
        "candidate_output_replay": _score(),
        "nemo_output_replay": _score(),
        "one_variable_substitutions": {
            name: _score(1) for name in gate.OPERANDS if name != "zhdiv"
        },
        "one_bit_control": _score(1),
    }


def test_literal_continuity_keeps_written_association():
    inputs = {
        "sshn_e": np.asarray([[9.0, 10.0]]),
        "zhU": np.asarray([[2.0, 5.0]]),
        "zhV": np.asarray([[7.0, 11.0]]),
        "r1_e1e2t": np.asarray([[0.5, 0.25]]),
        "ssh_frc": np.asarray([[1.0, -1.0]]),
        "rDt_e": np.asarray(2.0),
        "ssmask": np.asarray([[1.0, 0.0]]),
    }
    terms = gate._continuity_terms(inputs)
    np.testing.assert_array_equal(terms["du"], [[-3.0, 3.0]])
    np.testing.assert_array_equal(terms["dv"], [[7.0, 11.0]])
    np.testing.assert_array_equal(terms["zhdiv"], [[2.0, 3.5]])
    np.testing.assert_array_equal(terms["output"], [[3.0, 0.0]])


def test_classifier_names_incoming_ssh_without_blaming_rotation():
    result = gate.classify([_report(label) for label in gate.LABELS])
    assert result["status"] == "HELD_R255_INCOMING_SSH_OWNS_CONTINUITY"
    assert result["first_nonbit_boundary"] == "literal_continuity_output"
    assert result["first_nonbit_operand"] == "sshn_e"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_classifier_plants_fail_closed(plant):
    reports = copy.deepcopy([_report(label) for label in gate.LABELS])
    with pytest.raises(gate.GateError):
        gate.classify(reports, plant=plant)
