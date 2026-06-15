"""Truth-tier precedence gate for the oracle-recipe fidelity ladder (#388 Ask#4).

The verification ladder (``docs/ocean_fidelity/oracle_recipe_strategy.md`` §2/§3
rule E, summarised in ``CLAUDE.md``) splits into:

* **truth tiers** 0–2 — conservation, equivariance, analytic / manufactured
  solutions; the reference is *truth* (an invariant or closed form), so a green
  truth tier verifies *correctness*;
* **oracle tiers** ≥ 3 — tendency / trajectory / statistics match against an
  external model (Veros); a green oracle tier verifies *imitation*.

The doctrine: **a tier-≥3 oracle match is only trusted for a block that also
clears the truth tiers 0–2** — the truth tiers catch bugs inherited *from* the
oracle, the oracle catches wiring/convention bugs the truth tiers miss; a bug
survives only in the intersection.  This was documented but never enforced in
code.  This module turns it into a machine-checked verdict: any *failing* truth
tier LOCKS every oracle tier (their match no longer "counts"), with a clear
reason — so a tier-3 green can never be reported as trustworthy while tier-2 is
red.

Pure / dependency-free: operates on any iterable of records exposing ``tier``
(int) and ``passed`` (bool) — e.g. :class:`legoesm.ocean.fidelity.diff.PassFailRecord`
or plain dicts loaded from a results JSON.
"""
from __future__ import annotations

from dataclasses import dataclass

# Tiers whose reference is TRUTH (invariant / closed form), not the oracle.
# Everything at or above ORACLE_TIER_FLOOR is oracle-matching.
TRUTH_TIERS: tuple[int, ...] = (0, 1, 2)
ORACLE_TIER_FLOOR: int = 3


@dataclass(frozen=True)
class TierStatus:
    """Aggregated pass/fail for one tier."""

    tier: int
    n_pass: int
    n_fail: int

    @property
    def n_total(self) -> int:
        return self.n_pass + self.n_fail

    @property
    def ran(self) -> bool:
        return self.n_total > 0

    @property
    def green(self) -> bool:
        """A tier is green only if it ran and had zero failures."""
        return self.ran and self.n_fail == 0


@dataclass(frozen=True)
class PrecedenceVerdict:
    """Result of applying truth-tier precedence to a set of fidelity records.

    States when oracle tiers are present (most-severe first):

    * ``status="locked"``   — a truth tier (0–2) FAILED; the oracle-match cannot
      be trusted regardless of its own result (``oracle_locked=True``, the hard
      precedence violation).
    * ``status="mismatch"`` — truth tiers all green, but an oracle tier itself
      FAILED (the model disagrees with the oracle); precedence is cleared so
      this is a real, actionable discrepancy — NOT trusted.
    * ``status="incomplete"`` — no failures anywhere, but not every truth tier
      was evaluated; the oracle-match is NOT fully cleared (advisory).
    * ``status="trusted"``  — every truth tier ran AND is green AND every oracle
      tier present is green; the oracle-match fully clears the rule
      (``oracle_trusted=True``).

    With no oracle tiers present the gate is ``status="not_engaged"``.
    """

    tiers: dict[int, TierStatus]
    truth_failed: tuple[int, ...]     # truth tiers (0-2) that ran AND failed
    truth_missing: tuple[int, ...]    # truth tiers with no cases in this set
    oracle_failed: tuple[int, ...]    # oracle tiers (>=3) that ran AND failed
    oracle_present: bool
    oracle_locked: bool               # hard violation: a truth tier failed
    oracle_trusted: bool              # truth complete+green AND oracle green
    status: str                       # locked|mismatch|incomplete|trusted|not_engaged
    reason: str

    def oracle_tiers(self) -> tuple[int, ...]:
        return tuple(sorted(t for t in self.tiers if t >= ORACLE_TIER_FLOOR))

    def trusted_oracle_tiers(self) -> tuple[int, ...]:
        return self.oracle_tiers() if self.oracle_trusted else ()

    def locked_oracle_tiers(self) -> tuple[int, ...]:
        return self.oracle_tiers() if self.oracle_locked else ()


def _coerce_passed(value) -> bool:
    """Strictly interpret a record's ``passed`` field as a boolean.

    Accepts real bools and the known string encodings (case-insensitive)
    ``true/false``, ``pass/fail``, ``passed/failed``, ``error``.  Anything else
    (e.g. an arbitrary truthy string, an int) RAISES — a silent ``bool("false")
    == True`` would turn a real truth-tier failure into a false PASS and defeat
    the gate.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "pass", "passed"):
            return True
        if v in ("false", "fail", "failed", "error"):
            return False
    raise ValueError(
        f"cannot interpret passed={value!r} as a boolean (use a bool or one of "
        "true/false/pass/fail)"
    )


def aggregate_tiers(records) -> dict[int, TierStatus]:
    """Group records (each with ``tier`` + ``passed``) into per-tier pass/fail.

    Accepts objects with ``.tier`` / ``.passed`` attributes (PassFailRecord) or
    mappings with ``"tier"`` / ``"passed"`` keys.  ``passed`` is coerced
    strictly (see :func:`_coerce_passed`).
    """
    counts: dict[int, list[int]] = {}
    for r in records:
        if hasattr(r, "tier"):
            tier, passed = int(r.tier), _coerce_passed(r.passed)
        else:
            tier, passed = int(r["tier"]), _coerce_passed(r["passed"])
        pf = counts.setdefault(tier, [0, 0])
        if passed:
            pf[0] += 1
        else:
            pf[1] += 1
    return {t: TierStatus(tier=t, n_pass=pf[0], n_fail=pf[1])
            for t, pf in sorted(counts.items())}


def evaluate_precedence(records) -> PrecedenceVerdict:
    """Apply truth-tier precedence to a set of fidelity pass/fail records.

    A truth-tier (0–2) FAILURE LOCKS the oracle tiers (≥ :data:`ORACLE_TIER_FLOOR`)
    — the hard violation.  Oracle tiers are *fully trusted* only when every
    truth tier ran and is green.  If no truth tier failed but some truth tier
    was not evaluated, the verdict is ``incomplete`` (advisory, not a hard
    violation), so a deliberately-scoped run is not blocked yet is not reported
    as a fully-cleared oracle match either.
    """
    tiers = aggregate_tiers(records)
    truth_failed = tuple(
        t for t in TRUTH_TIERS if t in tiers and not tiers[t].green
    )
    truth_missing = tuple(t for t in TRUTH_TIERS if t not in tiers)
    oracle_failed = tuple(
        t for t in tiers if t >= ORACLE_TIER_FLOOR and not tiers[t].green
    )
    oracle_present = any(t >= ORACLE_TIER_FLOOR for t in tiers)
    oracle_locked = oracle_present and bool(truth_failed)
    # Fully trusted only when the precedence is cleared (all truth tiers ran and
    # are green) AND the oracle tiers themselves passed.
    oracle_trusted = (
        oracle_present and not truth_failed and not truth_missing
        and not oracle_failed
    )

    if not oracle_present:
        status = "not_engaged"
        reason = "no oracle tiers (>=3) present; precedence gate not engaged"
    elif truth_failed:
        status = "locked"
        reason = (
            f"oracle tiers LOCKED: truth tier(s) {list(truth_failed)} have "
            "failures, so the oracle-match cannot be trusted (a bug may be "
            "inherited from the oracle)."
        )
    elif oracle_failed:
        status = "mismatch"
        _truth_note = ("" if not truth_missing
                       else f" (truth tier(s) {list(truth_missing)} not evaluated)")
        reason = (
            f"no evaluated truth tier failed{_truth_note}, but oracle tier(s) "
            f"{list(oracle_failed)} FAILED — a genuine model-vs-oracle "
            "discrepancy, not an inherited-truth bug."
        )
    elif truth_missing:
        status = "incomplete"
        reason = (
            f"oracle tiers NOT fully cleared: truth tier(s) {list(truth_missing)} "
            "were not evaluated — truth coverage is incomplete (advisory, not a "
            "hard violation)."
        )
    else:
        status = "trusted"
        reason = "all truth tiers (0-2) green; oracle-match is trustworthy."

    return PrecedenceVerdict(
        tiers=tiers,
        truth_failed=truth_failed,
        truth_missing=truth_missing,
        oracle_failed=oracle_failed,
        oracle_present=oracle_present,
        oracle_locked=oracle_locked,
        oracle_trusted=oracle_trusted,
        status=status,
        reason=reason,
    )
