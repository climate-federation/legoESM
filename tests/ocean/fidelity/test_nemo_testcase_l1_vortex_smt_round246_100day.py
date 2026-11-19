"""Controls for the round-246 SMT-4 100-day measurement gate."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


TOOLS = (Path(__file__).parents[3] / "scripts" / "validate" /
         "ocean_fidelity" / "testcases")
sys.path.insert(0, str(TOOLS))

import nemo_testcase_l1_vortex_smt_round246_100day_gate as gate  # noqa: E402
import nemo_testcase_l1_vortex_round210_100day_comparison as scorer  # noqa: E402


def _score() -> dict:
    return json.loads((gate.ROOT / "round210_scores.json").read_text())


def _smt3() -> dict:
    return json.loads((gate.SMT3_ROOT / "round210_scores.json").read_text())


def test_real_measurement_passes_and_registers_predictions():
    report = gate.validate(_score(), _smt3())
    assert report["status"] == "PASS"
    assert report["admission"]["status"] == "ADMITTED"
    assert report["admission"]["restart_byte_identical"] is True
    assert set(report["admission"]["plant_status"].values()) == {"REFUSED"}
    assert report["nemo_daily_restarts"] == 100
    assert report["lego_daily_snapshots"] == 100
    assert set(report["checkpoints"]) == {str(x) for x in gate.CHECKPOINTS}
    predictions = report["predictions"]
    assert predictions["day100_T_reproduces_round238"]["status"] == "CONFIRMED"
    assert predictions["day100_T_above_smt3"]["status"] == "CONFIRMED"
    assert report["visuals"]["gif_frames"] == 100


@pytest.mark.parametrize(
    ("kind", "message"),
    (("checkpoint", "daily row registry is incomplete"),
     ("reproduction", "reproduction prediction changed to REFUTED"),
     ("ordering", "SMT-3 ordering prediction changed to REFUTED")),
)
def test_plants_fire(kind: str, message: str):
    with pytest.raises(gate.GateError, match=message):
        gate.plant(copy.deepcopy(_score()), _smt3(), kind)


def test_card_oracle_and_current_ladder_are_the_preregistered_inputs():
    score = _score()
    assert score["cards"]["smt4"]["case"] == "VORTEX_SMT4_VEC-zps"
    assert Path(score["cards"]["smt4"]["nemo_restart_dir"]) == gate.ORACLE
    assert scorer.CERTIFIED_LADDER["smt4"] == Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round238/"
        "smt4_ladder.json")
