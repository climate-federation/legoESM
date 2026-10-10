import json
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round216_downstream_gate as gate,
)


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds")
PAIR = ROOT / "round215/vector_pair_replay.json"
ASSOCIATION = ROOT / "round216/association_remeasure.json"


def _inputs():
    return json.loads(PAIR.read_text()), json.loads(ASSOCIATION.read_text())


def test_round216_classifies_first_downstream_statement() -> None:
    report = gate.classify(*_inputs())
    assert report["status"] == "HELD_R216_DOWNSTREAM_TRANSPORT"
    assert report["post_association_v"]["bit_exact"]
    assert report["first_downstream_non_bit"]["boundary"] == "continuity_dv"
    assert report["transport_v"]["production"]["differing_cells"] == 68


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round216_plants_fire(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(*_inputs(), plant=plant)
