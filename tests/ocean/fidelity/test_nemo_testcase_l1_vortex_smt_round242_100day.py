"""Controls for the round-242 SMT-1 100-day measurement gate."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


TOOLS = (Path(__file__).parents[3] / "scripts" / "validate" /
         "ocean_fidelity" / "testcases")
sys.path.insert(0, str(TOOLS))

import nemo_testcase_l1_vortex_smt_round242_100day_gate as gate  # noqa: E402


def _score() -> dict:
    return json.loads((gate.ROOT / "round210_scores.json").read_text())


def _flat() -> dict:
    return json.loads(gate.FLAT_REFERENCE.read_text())


def test_real_measurement_passes_and_registers_every_checkpoint():
    report = gate.validate(_score(), _flat())
    assert report["status"] == "PASS"
    assert report["nemo_daily_restarts"] == 100
    assert report["lego_daily_snapshots"] == 100
    assert set(report["checkpoints"]) == {str(x) for x in gate.CHECKPOINTS}
    assert report["day100_T_rms_K"] < gate.DAY100_T_BOUND_K
    assert report["visuals"]["gif_frames"] == 100


def test_checkpoint_plant_fires_on_a_missing_daily_row():
    score = copy.deepcopy(_score())
    with pytest.raises(gate.GateError, match="daily row registry is incomplete"):
        gate.plant(score, _flat(), "checkpoint")


def test_bound_plant_fires_at_the_frozen_limit():
    score = copy.deepcopy(_score())
    with pytest.raises(gate.GateError, match="exceeds frozen bound"):
        gate.plant(score, _flat(), "bound")


def test_card_and_oracle_are_the_preregistered_pair():
    score = _score()
    assert score["cards"]["smt1"]["case"] == "VORTEX_SMT1_VEC-zps"
    assert Path(score["cards"]["smt1"]["nemo_restart_dir"]) == gate.ORACLE
