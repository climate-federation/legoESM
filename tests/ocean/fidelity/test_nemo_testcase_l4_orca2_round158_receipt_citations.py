"""Round-158 receipt citations are rendered, live, and shift-sensitive."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RECEIPT = (
    REPO / "docs/ocean/fidelity/testcases"
    / "nemo_testcases_l4_orca2_round158_halo_exact_receipt.md"
)
GATE = (
    REPO / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_receipt_citation_gate.py"
)
PREFIX = "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/"


def _gate():
    spec = importlib.util.spec_from_file_location("_r158_citation_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rendered() -> list[str]:
    return sorted(set(re.findall(
        r"`(" + re.escape(PREFIX) + r"[^`]+)`", RECEIPT.read_text())))


def test_receipt_renders_exactly_eight_compiled_citations():
    assert _rendered() == [
        PREFIX + "lbclnk.f90:1961-1979",
        PREFIX + "lbclnk.f90:2060-2073",
        PREFIX + "lbclnk.f90:2105-2113",
        PREFIX + "lbcnfd.f90:1561-1577",
        PREFIX + "lbcnfd.f90:1629-1669",
        PREFIX + "lbcnfd.f90:1712-1739",
        PREFIX + "lbcnfd.f90:1747-1766",
        PREFIX + "mppini.f90:1412-1440",
    ]


def test_rendered_citations_are_mapped_and_live():
    gate = _gate()
    for citation in _rendered():
        assert citation in gate.CITATION_MAP
        assert gate.check(citation, gate.CITATION_MAP[citation])["status"] == "OK"


def test_a_two_line_shift_makes_each_citation_fail():
    gate = _gate()
    for citation in _rendered():
        row = gate.check(citation, gate.CITATION_MAP[citation], shift=2)
        assert row["status"] != "OK", row
