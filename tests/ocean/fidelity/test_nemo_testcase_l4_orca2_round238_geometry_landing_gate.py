from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round238_geometry_landing_gate as gate,
)


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round238")


def _result():
    return json.loads((ROOT / "independent_boundary_raw.json").read_text())


def test_round238_production_geometry_is_exact() -> None:
    result = _result()
    assert result["status"] == "PASS_R238_PRODUCTION_GEOMETRY"
    assert result["live_e3v"]["unequal"] == 0
    assert result["vmask"]["unequal"] == 0


def _raw_report() -> dict[str, object]:
    """Minimal inherited report: plants test classification, not file shape."""
    result = _result()
    zero = {"unequal": 0}
    return {
        "label": result["label"],
        "production_geometry": True,
        "status": "PASS_R233_GEOMETRY_UNIT_BOUNDARY",
        "passivity": {"ordinary_vs_traced": True},
        "geometry_registry": ["fmask", "tmask", "umask", "vmask"],
        "geometry_rows": {
            name: dict(zero) for name in ("fmask", "tmask", "umask", "vmask")
        },
        "operand_rows": {
            "e3v": result["live_e3v"],
            "vmask": result["vmask"],
            "vn_adv": result["vn_adv"],
            "zFv": result["zFv"],
        },
        "fold_band": result["fold_band"],
        "worktree": result["worktree"],
    }


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round238_geometry_plants_fire(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_raw_report(), plant=plant)
