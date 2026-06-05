"""Deterministic CI guard for the cross-oracle concept registry (dedup doctrine).

Two jobs (the deterministic half of the duplication auditor; the semantic half
is lego-modularity-tester dimension 10):

1. **Registry is internally consistent** — it can't itself become a tangle of
   contradictory names.
2. **Ratcheted aliases don't spread** — an alias marked ``ratchet=True`` in
   ``concept_registry`` may only appear in files already in the committed
   baseline. A new file using it fails the gate ("use the canonical name, or —
   with review — extend the baseline"). Debt only shrinks, mirroring the
   LOC_ALLOW_LIST pattern in ``test_clarity_guards.py``.

The detector is proven non-vacuous against a synthetic source.
"""

from __future__ import annotations

import pathlib
import re

from legoesm.ocean.fidelity.concept_registry import (
    CONCEPTS,
    KNOWN_ORACLES,
    oracle_to_canonical,
    ratcheted_aliases,
)

_STATUSES = {"canonical", "unit-hazard", "entrenched", "fidelity-scoped"}

OCEAN_SRC = (
    pathlib.Path(__file__).resolve().parents[3] / "src" / "legoesm" / "ocean"
)

# Files (relative to src/legoesm) currently allowed to contain each ratcheted
# alias. BASELINE ONLY — must shrink as call sites migrate to the canonical
# name. Growing it requires a reviewed edit here. Seeded from the 2026-05-29
# survey (bottom_drag_coeff is an experiment-layer config field that maps to
# bottom_drag_r at model-config build time).
ALIAS_BASELINE: dict[str, frozenset[str]] = {
    "bottom_drag_coeff": frozenset({
        "ocean/experiments/held_larichev.py",
        "ocean/experiments/munk_gyre.py",
        "ocean/experiments/global_barotropic_wind.py",
        "ocean/experiments/global_overturning.py",
        "ocean/experiments/acc_channel.py",
        "ocean/experiments/baroclinic_gyre.py",
        "ocean/experiments/regional_gyre.py",
        "ocean/experiments/eady_instability.py",
        "ocean/experiments/eady_uniform.py",
    }),
}


def _files_containing(alias: str) -> set[str]:
    """relpaths (under src/legoesm) of ocean source files using ``alias`` as a
    whole-word identifier. Excludes the registry module + this test's siblings
    (tests live outside src/, so only src is scanned)."""
    pat = re.compile(rf"\b{re.escape(alias)}\b")
    found: set[str] = set()
    for p in OCEAN_SRC.rglob("*.py"):
        if p.name == "concept_registry.py":
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        if pat.search(text):
            rel = "ocean/" + str(p.relative_to(OCEAN_SRC))
            found.add(rel)
    return found


# ---------------------------------------------------------------------------
# 1. Internal consistency
# ---------------------------------------------------------------------------


def test_registry_internally_consistent():
    canonicals = [c.canonical for c in CONCEPTS]
    assert len(canonicals) == len(set(canonicals)), "duplicate canonical names"

    all_aliases: dict[str, str] = {}
    for c in CONCEPTS:
        assert c.status in _STATUSES, f"{c.canonical}: bad status {c.status!r}"
        for orc, _name in c.oracle_aliases:
            assert orc in KNOWN_ORACLES, f"{c.canonical}: unknown oracle {orc!r}"
        for a in c.aliases:
            # An alias must not also be a canonical name of another concept.
            assert a not in canonicals, (
                f"{a!r} is both an alias (of {c.canonical}) and a canonical name"
            )
            # An alias must not be claimed by two concepts.
            assert a not in all_aliases, (
                f"alias {a!r} claimed by both {all_aliases[a]} and {c.canonical}"
            )
            all_aliases[a] = c.canonical
        if c.ratchet:
            assert c.aliases, f"{c.canonical}: ratchet=True but no aliases"


def test_oracle_to_canonical_maps_are_reachable():
    """Every oracle mapping resolves to a canonical name, and the helper builds
    a usable {oracle_name: canonical} dict (what a bridge would consume)."""
    canonicals = {c.canonical for c in CONCEPTS}
    veros = oracle_to_canonical("veros")
    assert veros, "expected at least one Veros mapping"
    for oracle_name, canon in veros.items():
        assert canon in canonicals, f"{oracle_name}->{canon} not a canonical"
    # Spot-check a known mapping from the survey.
    assert veros.get("r_bot") == "bottom_drag_r"


# ---------------------------------------------------------------------------
# 2. Ratchet: aliases must not spread beyond the baseline
# ---------------------------------------------------------------------------


def test_ratcheted_aliases_confined_to_baseline():
    """No ratcheted alias may appear in a source file outside its baseline."""
    violations = []
    for alias in ratcheted_aliases():
        baseline = ALIAS_BASELINE.get(alias, frozenset())
        used_in = _files_containing(alias)
        strayed = used_in - baseline
        for f in sorted(strayed):
            violations.append((alias, f))
    assert not violations, (
        "Ratcheted concept alias used outside its baseline — use the canonical "
        "name (see concept_registry), or extend ALIAS_BASELINE with review:\n"
        + "\n".join(f"  {a}  ->  {f}" for a, f in violations)
    )


def test_ratchet_baseline_has_no_stale_entries():
    """A baseline file that no longer uses the alias should be dropped (the
    baseline only shrinks)."""
    stale = []
    for alias, baseline in ALIAS_BASELINE.items():
        used_in = _files_containing(alias)
        for f in sorted(baseline - used_in):
            stale.append((alias, f))
    assert not stale, (
        "ALIAS_BASELINE names files that no longer use the alias (debt shrank — "
        f"drop them): {stale}"
    )


def test_every_ratcheted_alias_has_a_baseline():
    """Adding a ratcheted alias without a baseline would make the gate vacuous
    (or spuriously red) — force the author to declare the current footprint."""
    for alias in ratcheted_aliases():
        assert alias in ALIAS_BASELINE, (
            f"ratcheted alias {alias!r} has no ALIAS_BASELINE entry"
        )


def test_ratchet_detector_non_vacuous(tmp_path):
    """The whole-word scan must actually flag a stray alias (and not match
    substrings). Proven on synthetic files so the gate is not vacuous."""
    pat = re.compile(r"\bbottom_drag_coeff\b")
    hit = tmp_path / "hit.py"
    hit.write_text("cfg = dict(bottom_drag_coeff=1e-3)\n")
    miss = tmp_path / "miss.py"
    miss.write_text("x = my_bottom_drag_coeff_helper()  # substring, not the id\n")
    assert pat.search(hit.read_text()) is not None
    assert pat.search(miss.read_text()) is None
