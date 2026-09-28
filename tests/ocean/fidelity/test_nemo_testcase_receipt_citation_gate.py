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


@pytest.mark.parametrize("spec", ["309-300", ""])
def test_a_reversed_or_empty_range_raises_instead_of_crashing_later(spec):
    """It used to return ``[]`` and crash in ``check`` with an IndexError.

    A gate that raises an IndexError on a malformed citation has no verdict
    for it, which is the one thing a fail-closed gate may not do.
    """
    with pytest.raises(ValueError):
        gate.line_numbers(spec)


@pytest.mark.parametrize("citation,symbol,status", [
    ("stp2d.F90:128", "CALL dyn_hpg( kt, Kbb", "OK"),
    ("stp2d.F90:126", "CALL dyn_hpg( kt, Kbb", "SYMBOL-NOT-AT-LINE"),
    ("stp2d.F90:99999999", "CALL dyn_hpg( kt, Kbb", "OUT-OF-RANGE"),
    ("traqsr.F90:1", "anything", "UNRESOLVED"),
])
def test_check_classifies_the_real_source(citation, symbol, status):
    """The round-26 defect itself: :126 is a comment, dyn_hpg is at :128."""
    if not gate.FILES["stp2d.F90"].is_file():
        pytest.skip("NEMO oracle source tree not present")
    assert gate.check(citation, symbol)["status"] == status


def test_a_range_pins_both_endpoints_not_just_some_line_inside_it():
    """The gate's first draft joined the range and asked ``symbol in text``.

    A 146-line range then passed whenever ANY line held the symbol, so a
    wrong line number sailed through -- 36 of 78 mapped citations survived a
    shift of up to six lines.  Both endpoints are pinned now.
    """
    if not gate.FILES["stprk3_stg.F90"].is_file():
        pytest.skip("NEMO oracle source tree not present")
    citation = "stprk3_stg.F90:453-598"
    good = gate.CITATION_MAP[citation]
    assert gate.check(citation, good)["status"] == "OK"
    # a symbol that merely lives SOMEWHERE inside the range is now rejected
    assert gate.check(citation, ["tra_zdf", "tra_zdf", 146])["status"] == (
        "SYMBOL-NOT-AT-LINE")
    # and the correct symbols do not survive a wrong line number
    assert gate.check(citation, good, shift=2)["status"] != "OK"


def test_every_map_entry_is_shift_sensitive():
    """The gate's own non-vacuity control, asserted as a test as well.

    An entry whose anchors cannot tell its line from a neighbour has stopped
    checking anything; the audit must keep the map empty of those.  Round 28
    made the audit shift each endpoint INDEPENDENTLY, because moving both
    together never tests a changed range EXTENT.
    """
    if not gate.FILES["stp2d.F90"].is_file():
        pytest.skip("NEMO oracle source tree not present")
    assert gate.audit_map() == []


def test_the_audit_itself_can_fail():
    """Synthetic violation: a deliberately generic symbol must be caught."""
    if not gate.FILES["stp2d.F90"].is_file():
        pytest.skip("NEMO oracle source tree not present")
    original = dict(gate.CITATION_MAP)
    try:
        gate.CITATION_MAP["stp2d.F90:128"] = "CALL"
        blind = gate.audit_map()
        assert [row["citation"] for row in blind] == ["stp2d.F90:128"]
    finally:
        gate.CITATION_MAP.clear()
        gate.CITATION_MAP.update(original)


def test_a_widened_range_extent_fails_even_when_both_ends_match():
    """The defeat that beat round 27's gate.

    ``stprk3_stg.F90:309-334`` still passed as ``:309-533`` because line 334
    and line 533 are both ``ENDIF`` and the audit only shifted every line
    together.  Two independent guards close it: the map pins the range LENGTH,
    and a recurring terminal token is refused as a bare anchor.
    """
    if not gate.FILES["stprk3_stg.F90"].is_file():
        pytest.skip("NEMO oracle source tree not present")
    value = gate.CITATION_MAP["stprk3_stg.F90:309-334"]
    assert gate.check("stprk3_stg.F90:309-334", value)["status"] == "OK"
    assert gate.check("stprk3_stg.F90:309-533", value)["status"] == (
        "EXTENT-MISMATCH")
    # widen the pinned extent too, and the occurrence-pinned end still fails
    widened = [value[0], value[1], 225]
    assert gate.check("stprk3_stg.F90:309-533", widened)["status"] == (
        "SYMBOL-NOT-AT-LINE")
    # and a BARE recurring terminal token is refused outright
    assert gate.check(
        "stprk3_stg.F90:309-334",
        ["SELECT CASE( kstg )", "ENDIF", 26])["status"] == "AMBIGUOUS-ANCHOR"


def test_the_gate_runs_clean_on_the_real_receipt():
    """CI must see an unmapped or wrong-extent citation, not a reader.

    Round 27 shipped the gate but nothing called ``run`` on the receipt in the
    test suite, so a bad citation only failed if somebody remembered to run
    the script.  This is that check.
    """
    if not gate.FILES["stp2d.F90"].is_file():
        pytest.skip("NEMO oracle source tree not present")
    report = gate.run(gate.DEFAULT_RECEIPT, gate.DEFAULT_HEADING)
    assert report["unmapped_citations"] == []
    assert report["failures"] == []
    assert report["map_entries_failing_audit"] == []
    assert all(row["fired"] for row in report["self_test"])
    assert report["status"] == "PASS"


def test_the_real_receipt_run_can_fail():
    """Non-vacuity for the test above: a planted shift must turn it red."""
    if not gate.FILES["stp2d.F90"].is_file():
        pytest.skip("NEMO oracle source tree not present")
    report = gate.run(gate.DEFAULT_RECEIPT, gate.DEFAULT_HEADING,
                      plant="stp2d.F90:128")
    assert report["status"] == "FAIL"
    assert any(row["citation"] == "stp2d.F90:128"
               for row in report["failures"])
