"""Every compiled citation emitted by the round-15 gate is live and rigid."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
CITATION_GATE = (REPO / "scripts/validate/ocean_fidelity/testcases"
                 / "nemo_testcase_receipt_citation_gate.py")
SOLVER_GATE = (REPO / "scripts/validate/ocean_fidelity/orca2_l4"
               / "nemo_testcase_l4_orca2_round15_barotropic_solver_gate.py")
PREFIX = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _citations() -> list[str]:
    return sorted(set(_load(SOLVER_GATE, "_r15_gate").CITATIONS.values()))


def test_gate_stamps_only_the_record_build_and_all_source_groups():
    citations = _citations()
    assert len(citations) == 10, citations
    assert all(citation.startswith(PREFIX) for citation in citations)


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
