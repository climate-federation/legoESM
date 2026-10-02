"""Controls for the ORCA2 NEMO-side hierarchy completion gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round19_gate.py"
)
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round19_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_complete_hierarchy_is_admitted_at_authorized_background_boundary():
    report = gate.evaluate()
    assert report["status"] == "PASS_ORCA2_NEMO_HIERARCHY_RUNGS_1_10"
    assert [row["rung"] for row in report["rungs"]] == list(range(1, 11))
    assert [row["nn_havtb"] for row in report["rungs"]] == [0] * 6 + [1] * 4
    assert {row["frames"] for row in report["rungs"]} == {480}


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_completion_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.evaluate(plant=plant)
