from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round234_complete_association_gate as gate,
)


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds")


def _inputs():
    association = json.loads(
        (ROOT / "round234" / "association.json").read_text(encoding="utf-8"))
    reports = [json.loads(
        (ROOT / "round234" / name).read_text(encoding="utf-8"))
        for name in ("independent_boundary.json", "given_boundary.json")]
    return association, reports


@pytest.mark.parametrize(
    "plant", ("association-depth-bit", "association-ssh-bit",
              "label-coverage", "false-owner", "endpoint"))
def test_round234_plants_fire(plant: str) -> None:
    association, reports = _inputs()
    with pytest.raises(gate.GateError):
        gate.classify(association, reports, plant=plant)


def test_round234_real_evidence_classifies() -> None:
    association, reports = _inputs()
    result = gate.classify(association, reports)
    assert result["status"].startswith(("HELD_R234_", "QUALIFIED_R234_"))
