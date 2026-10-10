"""Controls for the round-244 SMT-2 100-day measurement gate."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


TOOLS = (Path(__file__).parents[3] / "scripts" / "validate" /
         "ocean_fidelity" / "testcases")
sys.path.insert(0, str(TOOLS))

import nemo_testcase_l1_vortex_smt_round244_100day_gate as gate  # noqa: E402


def _score() -> dict:
    return json.loads((gate.ROOT / "round210_scores.json").read_text())


def _smt1() -> dict:
    return json.loads((gate.SMT1_ROOT / "round210_scores.json").read_text())


def test_real_measurement_passes_and_registers_inherited_predictions():
    report = gate.validate(_score(), _smt1())
    assert report["status"] == "PASS"
    assert report["nemo_daily_restarts"] == 100
    assert report["lego_daily_snapshots"] == 100
    assert set(report["checkpoints"]) == {str(x) for x in gate.CHECKPOINTS}
    predictions = report["predictions"]
    assert predictions["day100_T_within_2x_smt1"]["status"] == "CONFIRMED"
    assert predictions["nemo_day100_u_max_below_smt1"]["status"] == "REFUTED"
    assert report["visuals"]["gif_frames"] == 100


@pytest.mark.parametrize(
    ("kind", "message"),
    (("checkpoint", "daily row registry is incomplete"),
     ("bound", "day-100 T prediction changed to REFUTED"),
     ("nemo-u", "NEMO U prediction changed to REFUTED")),
)
def test_plants_fire(kind: str, message: str):
    with pytest.raises(gate.GateError, match=message):
        gate.plant(copy.deepcopy(_score()), _smt1(), kind)


def test_card_and_oracle_are_the_preregistered_pair():
    score = _score()
    assert score["cards"]["smt2"]["case"] == "VORTEX_SMT2_VEC-zps"
    assert Path(score["cards"]["smt2"]["nemo_restart_dir"]) == gate.ORACLE
