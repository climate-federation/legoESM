"""Every compiled citation the round-13 ORCA2 receipt renders is mapped.

The campaign's citation gate walks the GYRE receipt, so until this round the
ORCA2 receipts' own compiled citations were in no map and nothing checked them.
``audit_map`` now verifies the seven this round added -- but only that each MAP
KEY resolves, not that the RECEIPT renders that key.  This closes the other
half: the receipt's compiled citations and the map's keys must be the same
strings, and each must still identify its line in the record's own compiled
branch.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RECEIPT = (REPO / "docs/ocean/fidelity/testcases"
           / "nemo_testcases_l4_orca2_round13_runoff_water_receipt.md")
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
    assert len(_rendered()) == 7, _rendered()


@pytest.mark.parametrize("citation", _rendered())
def test_each_rendered_citation_is_mapped_and_still_identifies_its_line(
        citation):
    gate = _gate()
    assert citation in gate.CITATION_MAP, (
        f"{citation} is rendered in the round-13 receipt but is in no map "
        "entry, so nothing checks it")
    row = gate.check(citation, gate.CITATION_MAP[citation])
    assert row["status"] == "OK", row


@pytest.mark.parametrize("citation", _rendered())
def test_a_two_line_shift_makes_each_one_fail(citation):
    """The check is not vacuous for any one of them."""
    gate = _gate()
    row = gate.check(citation, gate.CITATION_MAP[citation], shift=2)
    assert row["status"] != "OK", row
