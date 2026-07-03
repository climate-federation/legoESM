"""Unit tests for the truth-tier precedence gate (#388 Ask#4).

The doctrine (oracle_recipe_strategy.md §2/§3-E, CLAUDE.md): a tier-≥3 oracle
match is trusted ONLY for a block that also clears truth tiers 0–2. These tests
lock that rule — including the synthetic-violation self-test (a tier-2 failure
must LOCK tier-3) that proves the gate is non-vacuous.
"""
from __future__ import annotations

from legoesm.ocean.fidelity.diff import PassFailRecord
import pytest

from legoesm.ocean.fidelity.precedence import (
    ORACLE_TIER_FLOOR,
    TRUTH_TIERS,
    _coerce_passed,
    aggregate_tiers,
    evaluate_precedence,
)


def _rec(tier: int, passed: bool, name: str = "c") -> dict:
    return {"tier": tier, "passed": passed, "case_name": name}


def _pf(tier: int, passed: bool, name: str = "c") -> PassFailRecord:
    return PassFailRecord(
        case_name=name, tier=tier, metric="m", measured=0.0,
        tolerance_window=(0.0, 1.0), passed=passed, triage_hint=None,
    )


class TestAggregateTiers:
    def test_counts_from_dicts_and_objects(self):
        for mk in (_rec, _pf):
            tiers = aggregate_tiers([mk(0, True), mk(0, False), mk(2, True)])
            assert tiers[0].n_pass == 1 and tiers[0].n_fail == 1
            assert tiers[2].n_pass == 1 and tiers[2].n_fail == 0
            assert tiers[0].ran and not tiers[0].green
            assert tiers[2].green


class TestPrecedence:
    def test_all_truth_green_trusts_oracle(self):
        v = evaluate_precedence(
            [_rec(0, True), _rec(1, True), _rec(2, True), _rec(3, True)])
        assert v.status == "trusted"
        assert v.oracle_trusted is True
        assert v.oracle_locked is False
        assert v.truth_failed == ()
        assert v.truth_missing == ()
        assert v.trusted_oracle_tiers() == (3,)
        assert v.locked_oracle_tiers() == ()
        assert "trustworthy" in v.reason

    def test_truth_failure_locks_oracle(self):
        # SYNTHETIC-VIOLATION SELF-TEST: tier-2 fails, tier-3 passes -> LOCKED.
        v = evaluate_precedence(
            [_rec(0, True), _rec(1, True), _rec(2, False), _rec(3, True)])
        assert v.status == "locked"
        assert v.oracle_locked is True
        assert v.oracle_trusted is False
        assert v.truth_failed == (2,)
        assert v.locked_oracle_tiers() == (3,)
        assert v.trusted_oracle_tiers() == ()
        assert "LOCKED" in v.reason

    def test_oracle_failure_with_truth_green_is_mismatch_not_trusted(self):
        # Codex round-2 regression: a FAILING oracle tier must NOT read TRUSTED.
        v = evaluate_precedence(
            [_rec(0, True), _rec(1, True), _rec(2, True), _rec(3, False)])
        assert v.status == "mismatch"
        assert v.oracle_trusted is False
        assert v.oracle_locked is False     # truth is green -> not a precedence lock
        assert v.oracle_failed == (3,)
        assert v.trusted_oracle_tiers() == ()
        assert "discrepancy" in v.reason

    def test_tier0_failure_also_locks(self):
        v = evaluate_precedence([_rec(0, False), _rec(3, True), _rec(4, True)])
        assert v.oracle_locked is True
        assert set(v.locked_oracle_tiers()) == {3, 4}

    def test_missing_truth_tier_is_incomplete_not_locked(self):
        # only tier 2 + tier 3 present, both green: NOT a hard violation, but
        # NOT fully trusted either (truth coverage incomplete).
        v = evaluate_precedence([_rec(2, True), _rec(3, True)])
        assert v.status == "incomplete"
        assert v.oracle_locked is False
        assert v.oracle_trusted is False
        assert set(v.truth_missing) == {0, 1}
        assert v.locked_oracle_tiers() == ()
        assert v.trusted_oracle_tiers() == ()
        assert "incomplete" in v.reason

    def test_no_oracle_tiers_gate_not_engaged(self):
        v = evaluate_precedence([_rec(0, True), _rec(2, False)])
        assert v.oracle_tiers() == ()
        assert v.status == "not_engaged"
        assert v.oracle_locked is False  # no oracle to lock
        assert "not engaged" in v.reason

    def test_string_passed_false_counts_as_failure(self):
        # HIGH-1 regression: a JSON "false" must NOT be a truthy pass.
        v = evaluate_precedence(
            [_rec(2, "false"), _rec(3, "true")])
        assert v.truth_failed == (2,)
        assert v.oracle_locked is True

    def test_coerce_passed_strict(self):
        assert _coerce_passed(True) is True
        assert _coerce_passed("PASS") is True
        assert _coerce_passed("false") is False
        assert _coerce_passed("FAIL") is False
        for bad in ("yes", "1", 1, None, [], "maybe"):
            with pytest.raises(ValueError):
                _coerce_passed(bad)

    def test_empty_records(self):
        v = evaluate_precedence([])
        assert v.tiers == {}
        assert v.oracle_tiers() == ()
        assert v.status == "not_engaged"

    def test_constants(self):
        assert TRUTH_TIERS == (0, 1, 2)
        assert ORACLE_TIER_FLOOR == 3
