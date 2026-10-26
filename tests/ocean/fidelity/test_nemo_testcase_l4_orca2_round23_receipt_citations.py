"""Round-23 receipt citations are rendered, live, and shift-sensitive."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RECEIPT = (REPO / "docs/ocean/fidelity/testcases"
           / "nemo_testcases_l4_orca2_round23_second_solve_receipt.md")
GATE = (REPO / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_receipt_citation_gate.py")
PREFIX = "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"


def _gate():
    spec = importlib.util.spec_from_file_location("_r23_citation_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rendered() -> list[str]:
    return sorted(set(re.findall(
        r"`(" + re.escape(PREFIX) + r"[^`]+)`", RECEIPT.read_text())))


def test_receipt_renders_exactly_three_compiled_citations():
    assert len(_rendered()) == 3, _rendered()


@pytest.mark.parametrize("citation", _rendered())
def test_each_rendered_citation_is_mapped_and_live(citation):
    gate = _gate()
    assert citation in gate.CITATION_MAP
    assert gate.check(citation, gate.CITATION_MAP[citation])["status"] == "OK"


@pytest.mark.parametrize("citation", _rendered())
def test_a_two_line_shift_makes_each_citation_fail(citation):
    gate = _gate()
    row = gate.check(citation, gate.CITATION_MAP[citation], shift=2)
    assert row["status"] != "OK", row
