"""Controls for the round-89 exact debug-target resume."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / (
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round89_rung0_frame_debug_resume/run.sh"
)
PREREG = ROOT / "docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round89.md"


def test_launcher_is_parseable_and_uses_exact_resume_target() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUN)], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    text = RUN.read_text()
    assert "TARGET_CFG=ORCA2_OMIP_L4_R88FRAMEDEBUG" in text
    assert "TARGET_RUN=$EVIDENCE/orca2_rung0_frame_debug2_resume_np2" in text
    assert "./makenemo" not in text
    assert "/usr/bin/time" not in text


def test_launcher_pins_partial_checkpoint_before_first_target_write() -> None:
    text = RUN.read_text()
    fingerprint_check = text.index('actual_fingerprint=$(checkpoint_fingerprint "$TARGET_ROOT")')
    first_target_write = text.index('cp "$expected_bld" "$TARGET_ROOT/BLD/bld.cfg"')
    assert fingerprint_check < first_target_write
    assert "source-change checkpoint plant stayed green" in text
    assert "advanced-target checkpoint plant stayed green" in text
    assert text.count("copied checkpoint control does not reproduce baseline") == 1
    assert text.count("copied advanced-target control does not reproduce baseline") == 1
    assert '[[ ! -e "$TARGET_RUN" ]]' in text
    assert re.search(r"readonly PARTIAL_FINGERPRINT=[0-9a-f]{64}", text)


def test_launcher_supplies_only_missing_build_configuration() -> None:
    text = RUN.read_text()
    assert 'sed "s|$SOURCE_ROOT|$TARGET_ROOT|g" "$TEMPLATE_BLD"' in text
    assert 'pin "$TARGET_BLD_SHA" "$expected_bld"' in text
    assert "fcm build --ignore-lock -v 1 -j 1" in text
    assert 'pin "$COMPILED_TRAADV_SHA"' in text
    assert 'pin "$COMPILED_STP_SHA"' in text
    assert "traadv\\.f90:[0-9]+" in text
    assert "ORCA2_ROUND89_RUNG0_FRAME_DEBUG_REPRODUCED" in text


def test_preregistration_precedes_launcher_commit() -> None:
    assert PREREG.is_file()
    text = PREREG.read_text()
    assert "Base: `5ca47b129`" in text
    assert "Debug values remain" in text
