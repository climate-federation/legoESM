"""Round-228 compiled citations are mapped, live, and shift-sensitive."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RECEIPT = (
    REPO / "docs/ocean/fidelity/testcases"
    / "nemo_testcases_l4_orca2_round228_fold_invariant_audit_receipt.md"
)
GATE = (
    REPO / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_receipt_citation_gate.py"
)
PREFIXES = (
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/",
    "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/",
    "barotropic_latlon_cgrid.py:",
)


def _gate():
    spec = importlib.util.spec_from_file_location("_r228_citation_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rendered() -> list[str]:
    text = RECEIPT.read_text()
    candidates = re.findall(r"`([^`]+)`", text)
    return sorted(set(
        citation for citation in candidates
        if citation.startswith(PREFIXES)
    ))


def test_round228_citations_are_mapped_live_and_shift_sensitive() -> None:
    gate = _gate()
    assert len(_rendered()) == 4
    for citation in _rendered():
        assert citation in gate.CITATION_MAP
        assert gate.check(citation, gate.CITATION_MAP[citation])["status"] == "OK"
        assert gate.check(
            citation, gate.CITATION_MAP[citation], shift=2)["status"] != "OK"
