from __future__ import annotations

from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round209_omt1_frame_record_gate as record_gate,
)


RUNNER = Path(
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round211_omt1_frames_acquisition/run.sh"
)


def test_recovery_reuses_record_binary_without_building() -> None:
    text = RUNNER.read_text()

    assert "ORCA2_OMIP_L4_R210OMT1_P3/BLD/bin/nemo.exe" in text
    assert "instrument_binary_sha=5b82a3254c40" in text
    assert "makenemo" not in text


def test_recovery_uses_exact_eight_step_protocol() -> None:
    text = RUNNER.read_text()

    assert "--itend 8 --stock 8" in text
    assert "--restart-steps 2,4,6,8" in text
    assert "orca2_omt1_frames_8step_a_np2" in text
    assert "orca2_omt1_frames_8step_b_np2" in text


def test_recovery_admits_preserved_step_nine_boundary() -> None:
    text = RUNNER.read_text()

    assert "round210/acquisition/orca2_omt1_uninstrumented_10step_np2" in text
    assert "--boundary-only" in text
    assert "--boundary \"$boundary\"" in text
    assert "wrong-boundary" in text


def test_recovery_contract_is_eight_steps_and_sixty_four_frames() -> None:
    report = record_gate.preflight()

    assert report["steps"] == list(range(1, 9))
    assert report["expected_frames_per_twin"] == 64
