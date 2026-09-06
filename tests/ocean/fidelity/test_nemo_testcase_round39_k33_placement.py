"""Round 39: the isoneutral fold's placement, and the acquisition's admission rule.

Every test here is written so that it FAILS when the thing it checks is
removed; the two that guard a shipped shell rule EXECUTE that rule rather than
matching its text, so a rewording cannot make them vacuous.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
GATES = REPO / "scripts/validate/ocean_fidelity/testcases"
RUN_SH = GATES / "nemo_testcase_l2_gyre_round38_trazdf_kt2/run.sh"
sys.path.insert(0, str(GATES))


# --------------------------------------------------------------------------
# The acquisition's twin rule: raw identity for ONE record, consumed-field
# identity for the rest.
# --------------------------------------------------------------------------
def _raw_guard() -> str:
    """The shipped guard, extracted from run.sh and run as-is.

    Matching run.sh's TEXT would pass on a rewording that inverted the rule.
    This lifts the block between its own two markers and executes it, so the
    test fails whenever the shipped bytes stop implementing the rule.
    """
    text = RUN_SH.read_text()
    start = text.index("# THE ONE RECORD THAT MUST BE RAW-IDENTICAL")
    end = text.index("# EVERY OTHER round-37 record")
    block = text[start:end]
    assert "exit 71" in block, "the extracted block carries no refusal"
    return block


def _run_guard(tmp_path: Path, log_lines: list[str]) -> int:
    (tmp_path / "round38_raw_twin_cmp.log").write_text(
        "\n".join(log_lines) + "\n")
    script = (
        "set -uo pipefail\n"
        f'TARGET_RUN="{tmp_path}"\n'
        "KT1_RECORD=oracle_trazdf_matrix_kt00000001.bin\n"
        + _raw_guard()
        + "\nexit 0\n"
    )
    return subprocess.run(["bash", "-c", script], capture_output=True).returncode


def test_a_halo_only_difference_in_another_record_no_longer_refuses(tmp_path):
    """The defect this round fixed: a good acquisition was refused.

    Round 38's rule refused whenever ANY round-37 record differed raw.  Eight
    of that acquisition's 52 records differ in halo or undefined-slot bytes --
    uninitialised memory, which differs between two runs of the same binary --
    so the rule refused a run whose consumed fields were all identical.
    """
    assert _run_guard(tmp_path, [
        "RAW_IDENTICAL oracle_trazdf_matrix_kt00000001.bin",
        "RAW_DIFFERS   oracle_zdf_matrix_kt00000001.bin (falls through)",
    ]) == 0


def test_a_moved_kt1_trazdf_record_still_refuses(tmp_path):
    """The rule that must SURVIVE: the arm's own record is bit-pinned."""
    assert _run_guard(tmp_path, [
        "RAW_DIFFERS   oracle_trazdf_matrix_kt00000001.bin (falls through)",
        "RAW_IDENTICAL oracle_zdf_matrix_kt00000001.bin",
    ]) == 71


def test_a_missing_kt1_line_refuses(tmp_path):
    """A log that never mentions the record is a refusal, not a pass."""
    assert _run_guard(tmp_path, ["RAW_IDENTICAL oracle_rhs_kt00000001.bin"]) == 71


def test_run_sh_delegates_the_other_records_to_the_admission_gate():
    """The relaxed records must be decided by SOMETHING, and it is named."""
    text = RUN_SH.read_text()
    block = text[text.index("# EVERY OTHER round-37 record"):]
    head = block[:block.index("test -s")]
    assert '--baseline "$SOURCE_RUN"' in head, (
        "the source-run consumed-field admission is gone; the other records "
        "would then be checked by nothing")
    assert "--plant-consumed" in head, "the source admission has no plant"
    assert head.count("exit 7") >= 2, "the admission or its plant cannot refuse"
