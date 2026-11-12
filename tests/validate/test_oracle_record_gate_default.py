"""The oracle-record gate must be ARMED BY DEFAULT, not opt-in.

``--expect-day`` existed and worked, but only 5 of the 105 cards that score
against NEMO passed it, so for the other 100 the gate was inert -- including
the card whose inherited ``--nemo-time-idx`` default scored day 30 against
NEMO's day 90 and reported the result as an MLD regression (2026-09-19).

A gate that 95% of callers forget to switch on is a defect report, not a
feature. The snapshots already carry their own day, so the scorer can derive
the expected day itself and nothing needs passing.

These tests pin the arithmetic the gate rests on and the fact that the default
path is wired, since the wiring is what was missing rather than the check.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "compare_three_way_nemo",
    _ROOT / "scripts/validate/ocean_fidelity/compare_three_way_nemo.py")
m = importlib.util.module_from_spec(_SPEC)
sys.modules["compare_three_way_nemo"] = m
_SPEC.loader.exec_module(m)


def test_the_record_ending_on_a_day_is_that_day_over_five_minus_one():
    """Day D maps to record D/5 - 1 for 5-day means."""
    # Day 30 -> record 5. Passing record 5 must be accepted.
    m.check_oracle_record(30.0, 5, n_time=18)
    # Day 5 -> record 0.
    m.check_oracle_record(5.0, 0, n_time=18)


def test_a_mismatched_record_is_fatal():
    """The exact 2026-09-19 defect: day 30 scored against NEMO's day 90."""
    with pytest.raises(SystemExit):
        m.check_oracle_record(30.0, 17, n_time=18)


def test_a_day_that_is_not_a_record_boundary_is_fatal():
    with pytest.raises(SystemExit):
        m.check_oracle_record(27.0, 5, n_time=18)


def test_no_expected_day_means_no_check():
    """Preserved so a caller that genuinely cannot state a day still runs."""
    m.check_oracle_record(None, 17, n_time=18)


def test_a_month_selection_is_exempt():
    """--nemo-month selects a climatological month, not a 5-day record."""
    m.check_oracle_record(30.0, 17, n_time=18, nemo_month=3)


def test_the_gate_is_armed_from_the_snapshot_day_when_no_flag_is_given():
    """The wiring, which is the part that was missing.

    Scoped to the call site so the test fails if the derivation is removed,
    and reads code rather than comments.
    """
    src = (_ROOT / "scripts/validate/ocean_fidelity/"
           "compare_three_way_nemo.py").read_text()
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "_expect = a.expect_day" in code
    assert "if _expect is None and len(known) == 1:" in code
    assert "check_oracle_record(_expect," in code, (
        "the gate must be called with the DERIVED day, not the raw flag")
