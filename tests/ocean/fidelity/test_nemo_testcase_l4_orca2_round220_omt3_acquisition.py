from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round220_omt3_deck_gate as deck_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round220_omt3_frame_record_gate as record_gate,
)


SOURCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round218/acquisition/omt2_namelist_cfg"
)
RUNNER = Path(
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round220_omt3_frames_acquisition/run.sh"
)


def test_omt3_deck_is_exactly_the_momentum_ldf_module_edge(
        tmp_path: Path) -> None:
    candidate = tmp_path / "namelist_cfg"
    candidate.write_text(deck_gate.render_omt3(SOURCE.read_text()))

    report = deck_gate.validate_deck(SOURCE, candidate)

    assert report["changed_assignments"] == ["namdyn_ldf.ln_dynldf_lap"]
    assert report["removed_assignments"] == ["namdyn_ldf.ln_dynldf_off"]
    assert report["resolved_from_namelist_ref"] == {
        "namdyn_ldf.ln_dynldf_off": ".false.",
        "namdyn_ldf.ln_dynldf_lev": ".true.",
        "namdyn_ldf.nn_ahm_ijk_t": "-30",
    }


@pytest.mark.parametrize("plant", deck_gate.PLANTS[1:])
def test_deck_plants_fire(tmp_path: Path, plant: str) -> None:
    candidate = tmp_path / "namelist_cfg"
    candidate.write_text(deck_gate.render_omt3(SOURCE.read_text()))
    with pytest.raises(deck_gate.GateError):
        deck_gate.validate_deck(SOURCE, candidate, plant)


def test_record_preflight_reuses_admitted_writer_for_ten_steps() -> None:
    report = record_gate.preflight()
    assert report["status"] == "PASS_R220_OMT3_FRAME_PREFLIGHT"
    assert report["steps"] == list(range(1, 11))
    assert report["expected_frames_per_twin"] == 80
    assert report["writer_preflight"]["removed_source_lines"] == 0


def test_missing_frame_plant_is_nonvacuous(tmp_path: Path) -> None:
    for name in record_gate._wanted_frames():
        (tmp_path / name).touch()
    with pytest.raises(record_gate.GateError, match="frame inventory mismatch"):
        record_gate._validate_inventory(tmp_path, "missing-frame")


def test_runner_reuses_binaries_and_smokes_before_records() -> None:
    text = RUNNER.read_text()
    assert "makenemo" not in text
    assert "ORCA2_OMIP_L4_R210OMT1_P3/BLD/bin/nemo.exe" in text
    assert "CALL dynldf_lev_lap" in text
    smoke = text.index('stage_run "$smoke" "$base/nemo" 2 2 2')
    calibration = text.index('stage_run "$calibration" "$base/nemo" 10 10 10')
    twin = text.index('stage_run "$twin_a" "$instrument_binary" 10 10 10')
    month = text.index('stage_run "$month" "$base/nemo" 96 96')
    assert smoke < calibration < twin < month


def test_runner_month_protocol_obeys_compiled_limits() -> None:
    text = RUNNER.read_text()
    assert "10,20,30,40,50,60,70,80,90,95" in text
    assert '[[ "$last_step" -gt 10 ]]' in text
    assert "/usr/bin/time" not in text
