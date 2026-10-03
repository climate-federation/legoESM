"""Controls for the round-65 ORCA2 QCO-boundary composition gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_round65_qco_scope_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round65_qco_scope", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _scope():
    return {
        "status": "PASS",
        "disagreements": [],
        "card_scope": {
            "ORCA2-zps": ["vector_invariant", "upwind"],
            "GYRE-zco": ["vector_invariant", "upwind"],
            "OVERFLOW-zps": ["flux_form", "nemo_up3"],
            "LOCK-zco": ["flux_form", "nemo_up3"],
        },
    }


def _qco():
    rows = {}
    for tracer, (unequal, count) in gate.EXPECTED.items():
        rows[tracer] = {
            "production_stage1": {"unequal": unequal, "count": count},
            "numpy_literal": {"bit_exact": True, "unequal": 0,
                              "count": count},
            "jax_source_ordered": {"bit_exact": True, "unequal": 0,
                                   "count": count},
            "jax_fused": {"unequal": unequal, "count": count},
        }
    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "claim_label": "GIVEN_NEMO_ENTRY_DECISION52",
        "execution": {"dtype": "float64"},
        "replay_rows": rows,
    }


def test_classifies_the_executing_orca2_boundary():
    report = gate.classify(_scope(), _qco())
    assert report["status"] == "PASS_QCO_SCOPE"
    disposition = report["round60_boundary_disposition"]
    assert disposition["vector_velocity_update"] == "EXECUTED_BY_ORCA2"
    assert (disposition["thickness_weighted_momentum_update"]
            == "NOT_EXECUTED_BY_ORCA2")
    assert (report["first_actual_orca2_nonbit_statement"]["statement"]
            == "tracer_qco_rk_assignment")


@pytest.mark.parametrize("plant", ["scope", "qco-count"])
def test_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_scope(), _qco(), plant=plant)


def test_compiled_source_guard_is_bound(tmp_path):
    source = gate.SOURCE.read_text().replace(
        "IF( ln_dynadv_vec .OR. lk_linssh ) THEN",
        "IF( lk_linssh ) THEN",
        1,
    )
    path = tmp_path / "stprk3_plant.f90"
    path.write_text(source)
    with pytest.raises(gate.GateError, match="predicate changed"):
        gate.classify(_scope(), _qco(), source=path)
