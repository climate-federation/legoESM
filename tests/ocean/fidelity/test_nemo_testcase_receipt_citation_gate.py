"""Guards for the receipt citation gate.

The gate's job is to make a wrong ``file:line`` in the receipt fail a check
instead of a reader's attention -- three rounds in a row shipped one.  What is
guarded here is the part that decides WHICH citation is checked: the
continuation binder.  A bare ``:NNN`` span inherits the last named file, and
binding it to the wrong file was itself the shape of the round-26 defect (the
``stp2d.F90`` continuations were read against ``stprk3_stg.F90``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TESTCASES = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
)
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_receipt_citation_gate",
    TESTCASES / "nemo_testcase_receipt_citation_gate.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_bare_line_spans_bind_to_the_last_named_file():
    text = ("`stprk3_stg.F90:324` then, in `stp2d.F90` order, `:128` and "
            "`:131`; back to `dynhpg.F90:359` and `:383`.")
    assert gate.extract(text) == [
        "stprk3_stg.F90:324", "stp2d.F90:128", "stp2d.F90:131",
        "dynhpg.F90:359", "dynhpg.F90:383",
    ]


def test_a_leading_bare_span_is_reported_unbound_not_silently_dropped():
    assert gate.extract("`:128` before any file is named") == [
        "<UNBOUND>:128"]


def test_fenced_blocks_are_not_scanned_for_citations():
    text = "```text\n`dynhpg.F90:1`\n```\nreal: `dynhpg.F90:359`"
    assert gate.extract(text) == ["dynhpg.F90:359"]


def test_line_numbers_expands_commas_and_ranges():
    assert gate.line_numbers("139,145") == [139, 145]
    assert gate.line_numbers("103-105") == [103, 104, 105]
    assert gate.line_numbers("1,3-5") == [1, 3, 4, 5]


@pytest.mark.parametrize("citation,symbol,status", [
    ("stp2d.F90:128", "CALL dyn_hpg", "OK"),
    ("stp2d.F90:126", "CALL dyn_hpg", "SYMBOL-NOT-AT-LINE"),
    ("stp2d.F90:99999999", "CALL dyn_hpg", "OUT-OF-RANGE"),
    ("traqsr.F90:1", "anything", "UNRESOLVED"),
])
def test_check_classifies_the_real_source(citation, symbol, status):
    """The round-26 defect itself: :126 is a comment, dyn_hpg is at :128."""
    if not gate.FILES["stp2d.F90"].is_file():
        pytest.skip("NEMO oracle source tree not present")
    assert gate.check(citation, symbol)["status"] == status
