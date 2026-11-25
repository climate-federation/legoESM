import json
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round217_atomic_unit_gate as gate,
)


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds")


def _inputs():
    round217 = ROOT / "round217"
    return (
        json.loads((round217 / "omt1_independent_decision96.json").read_text()),
        json.loads((round217 / "omt1_given_decision96.json").read_text()),
        (round217 / "rung0.log").read_text(),
    )


def test_round217_holds_at_rung0_live_thickness() -> None:
    report = gate.classify(*_inputs())
    assert report["status"] == "HELD_R217_RUNG0_LIVE_THICKNESS"
    assert report["predictions"]["R217-P2"] == "CONFIRMED_BOTH_LABELS"
    assert report["predictions"]["R217-P3"].startswith("REFUTED")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round217_plants_fire(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(*_inputs(), plant=plant)
