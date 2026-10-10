"""Round-214 compiled citations are mapped, live, and shift-sensitive."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RECEIPT = (
    REPO / "docs/ocean/fidelity/testcases"
    / "nemo_testcases_l4_orca2_round214_vector_header_recovery_receipt.md"
)
GATE = (
    REPO / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_receipt_citation_gate.py"
)
PREFIX = "ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/"


def _gate():
    spec = importlib.util.spec_from_file_location("_r214_citation_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rendered() -> list[str]:
    return sorted(set(re.findall(
        r"`(" + re.escape(PREFIX) + r"[^`]+)`", RECEIPT.read_text())))


def test_round214_receipt_renders_the_three_compiled_citations():
    assert _rendered() == [
        PREFIX + "dynspg_ts.f90:714-726",
        PREFIX + "stp2d.f90:303-303",
        PREFIX + "stprk3.f90:204-215",
    ]


def test_round214_citations_are_mapped_live_and_shift_sensitive():
    gate = _gate()
    for citation in _rendered():
        assert citation in gate.CITATION_MAP
        assert gate.check(citation, gate.CITATION_MAP[citation])["status"] == "OK"
        assert gate.check(
            citation, gate.CITATION_MAP[citation], shift=2)["status"] != "OK"
