"""Controls for the ORCA2 hierarchy rung-1/rung-0 handoff gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round12_gate.py"
)
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round12_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_boundary_is_fully_classified_and_has_one_active_extra():
    report = gate.evaluate()
    assert report["status"] == "PASS_RUNG1_RECORD_ACTIVE_RUNG0_DIFFERENCE"
    assert report["classified_assignments"]["physics_active_extra"] == [
        "namzdf.nn_havtb"
    ]
    assert report["resolved_nn_havtb"] == {"rung1": "1", "main_rung0": "0"}
    assert report["nn_havtb_signal"]["affected_surface_wet_cells_rank0"] > 0


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.evaluate(plant=plant)
