from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round238_geometry_landing_gate as gate,
)


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round238")


def _report():
    return json.loads((ROOT / "independent_boundary_raw.json").read_text())


def test_round238_production_geometry_is_exact() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_R238_PRODUCTION_GEOMETRY"
    assert result["live_e3v"]["unequal"] == 0
    assert result["vmask"]["unequal"] == 0


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round238_geometry_plants_fire(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)
