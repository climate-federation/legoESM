from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round215_vector_pair_gate as gate,
)


REPORT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round215/"
    "vector_pair_replay.json"
)


@pytest.mark.skipif(not REPORT.is_file(), reason="round-215 measurement not present")
def test_round215_report_classifies() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert gate.classify(report)["status"] == "PASS_R215_OMT1_VECTOR_PAIR_REPLAY"


@pytest.mark.skipif(not REPORT.is_file(), reason="round-215 measurement not present")
@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round215_plants_fire(plant: str) -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    with pytest.raises(gate.GateError):
        gate.classify(report, plant)
