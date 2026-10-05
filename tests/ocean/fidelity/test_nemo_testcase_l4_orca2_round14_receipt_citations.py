"""Every compiled citation the round-14 ORCA2 receipt renders is mapped.

Round 13 opened this for its own receipt; the same cover is owed by every
later ORCA2 receipt, because ``audit_map`` checks that a map KEY resolves and
not that the RECEIPT renders that key.  Without this a receipt could cite
``:196-200`` while the map pinned ``:196-199`` and nothing would notice.  So
the receipt's compiled citations and the map's keys must be the same strings,
each must still identify its line in the record's own compiled branch, and
each must FAIL under a two-line shift.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RECEIPT = (REPO / "docs/ocean/fidelity/testcases"
           / "nemo_testcases_l4_orca2_round14_barotropic_owner_receipt.md")
GATE = (REPO / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_receipt_citation_gate.py")
PREFIX = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/"


def _gate():
    spec = importlib.util.spec_from_file_location("_citation_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rendered() -> list[str]:
    text = RECEIPT.read_text()
    return sorted(set(re.findall(r"`(" + re.escape(PREFIX) + r"[^`]+)`", text)))


def test_the_receipt_renders_the_compiled_citations_it_claims():
    """Non-vacuity: if the receipt stops citing them this goes red."""
    assert len(_rendered()) == 10, _rendered()


@pytest.mark.parametrize("citation", _rendered())
def test_each_rendered_citation_is_mapped_and_still_identifies_its_line(
        citation):
    gate = _gate()
    assert citation in gate.CITATION_MAP, (
        f"{citation} is rendered in the round-14 receipt but is in no map "
        "entry, so nothing checks it")
    row = gate.check(citation, gate.CITATION_MAP[citation])
    assert row["status"] == "OK", row


@pytest.mark.parametrize("citation", _rendered())
def test_a_two_line_shift_makes_each_one_fail(citation):
    """The check is not vacuous for any one of them."""
    gate = _gate()
    row = gate.check(citation, gate.CITATION_MAP[citation], shift=2)
    assert row["status"] != "OK", row


# --- the gate script's OWN citation dict -----------------------------------
# It is copied verbatim into every evidence JSON the gate writes, and nothing
# checked it: this test used to read only the receipt's rendered strings, and
# the dict carried a range that pointed at a comment banner instead of the
# statement it named.  A citation nobody checks is exactly what this campaign's
# gate exists to prevent, so the dict is checked the same way the receipt is.
OWNER_GATE = (REPO / "scripts/validate/ocean_fidelity/orca2_l4"
              / "nemo_testcase_l4_orca2_round14_barotropic_owner_gate.py")


def _gate_citations() -> list[str]:
    spec = importlib.util.spec_from_file_location("_r14_gate", OWNER_GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return sorted(set(module.CITATIONS.values()))


def test_the_gate_stamps_only_compiled_citations_from_the_record_branch():
    """Non-vacuity: an entry from another build would go red here."""
    citations = _gate_citations()
    assert len(citations) == 11, citations
    assert all(c.startswith(PREFIX) for c in citations), citations


@pytest.mark.parametrize("citation", _gate_citations())
def test_each_gate_citation_is_mapped_and_still_identifies_its_line(citation):
    gate = _gate()
    assert citation in gate.CITATION_MAP, (
        f"{citation} is stamped into this gate's evidence JSON but is in no "
        "map entry, so nothing checks it")
    assert gate.check(citation, gate.CITATION_MAP[citation])["status"] == "OK"


@pytest.mark.parametrize("citation", _gate_citations())
def test_a_two_line_shift_makes_each_gate_citation_fail(citation):
    gate = _gate()
    row = gate.check(citation, gate.CITATION_MAP[citation], shift=2)
    assert row["status"] != "OK", row
