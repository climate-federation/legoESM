"""Unit guards for the ORCA2 inventory citation gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TESTCASES = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_orca2_parallel_receipt_gate",
    TESTCASES / "nemo_testcase_l2_orca2_parallel_receipt_gate.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def test_extract_deduplicates_file_line_citations():
    text = "`one.json:2-4` `not a cite` `one.json:2-4` `two.py:7`"
    assert gate.extract(text) == ("one.json:2-4", "two.py:7")


def test_line_number_parser_and_shift_are_fail_closed():
    assert gate._line_numbers("2-4,8") == (2, 3, 4, 8)
    assert gate._shift_spec("2-4,8", 1) == "3-5,9"
    with pytest.raises(gate.CitationError, match="reversed"):
        gate._line_numbers("4-2")
