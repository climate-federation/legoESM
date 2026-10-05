"""Round-16 receipt citations are rendered, live, and rigid."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
CITATION_GATE = (REPO / "scripts/validate/ocean_fidelity/testcases"
                 / "nemo_testcase_receipt_citation_gate.py")
ROUND_GATE = (REPO / "scripts/validate/ocean_fidelity/orca2_l4"
              / "nemo_testcase_l4_orca2_round16_slow_forcing_gate.py")
RECEIPT = (REPO / "docs/ocean/fidelity/testcases"
           / "nemo_testcases_l4_orca2_round16_slow_forcing_receipt.md")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _citations() -> list[str]:
    return sorted(set(_load(ROUND_GATE, "_r16_gate").CITATIONS.values()))


def test_receipt_renders_every_stamped_compiled_citation():
    text = RECEIPT.read_text()
    citations = _citations()
    assert len(citations) == 2
    assert all(citation in text for citation in citations)
    assert "dynspg_ts.f90:550-558" in text


@pytest.mark.parametrize("citation", _citations())
def test_each_stamped_citation_is_mapped_and_live(citation):
    gate = _load(CITATION_GATE, "_citation_gate")
    assert citation in gate.CITATION_MAP
    assert gate.check(citation, gate.CITATION_MAP[citation])["status"] == "OK"


@pytest.mark.parametrize("citation", _citations())
def test_a_rigid_two_line_shift_makes_each_citation_fail(citation):
    gate = _load(CITATION_GATE, "_citation_gate_shift")
    row = gate.check(citation, gate.CITATION_MAP[citation], shift=2)
    assert row["status"] != "OK", row
