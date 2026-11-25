from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round233_geometry_unit_gate as gate,
)


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round233")


def _reports():
    return [
        json.loads((ROOT / "independent_boundary.json").read_text()),
        json.loads((ROOT / "given_boundary.json").read_text()),
    ]


def test_round233_refutes_geometry_as_the_root_owner() -> None:
    result = gate.classify(_reports())
    assert result["status"] == "HELD_R233_INDEPENDENT_EXTERNAL_MODE_DEBT"
    assert result["predictions"]["R233-P2"] == "REFUTED_VN_ADV_8589_OF_8589"
    assert result["rows"]["independent"]["e3v_unequal"] == 0
    assert result["rows"]["independent"]["vmask_unequal"] == 0


@pytest.mark.parametrize("plant", gate.CLASSIFY_PLANTS[1:])
def test_round233_classification_plants_fire(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_reports(), plant=plant)
