from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round199_omt0_record_gate as omt0_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round203_omt0_frame_record_gate as gate,
)


def test_preflight_pins_additions_only_writer_and_retraction() -> None:
    report = gate.preflight()
    assert report["status"] == "PASS_R203_OMT0_FRAME_PREFLIGHT"
    assert report["expected_frames_per_twin"] == 80
    assert report["writer_preflight"]["removed_source_lines"] == 0
    assert report["retracted_protocol"] == "nn_stock=1 with nn_fsbc=2"


def test_frame_inventory_is_rank_step_stage_complete(tmp_path: Path) -> None:
    for name in gate._wanted_frames():
        (tmp_path / name).touch()
    assert len(gate._validate_inventory(tmp_path, "none")) == 80


def test_missing_frame_plant_is_nonvacuous(tmp_path: Path) -> None:
    for name in gate._wanted_frames():
        (tmp_path / name).touch()
    with pytest.raises(gate.GateError, match="frame inventory mismatch"):
        gate._validate_inventory(tmp_path, "missing-frame")


def test_round202_frequency_protocol_stays_retracted() -> None:
    with pytest.raises(omt0_gate.GateError, match="nn_stock=1.*nn_fsbc=2"):
        omt0_gate.render_frequency_run_deck("", itend=10)


def test_binary_pins_are_distinct() -> None:
    assert gate.BASE_BINARY_SHA256 != gate.INSTRUMENT_BINARY_SHA256
