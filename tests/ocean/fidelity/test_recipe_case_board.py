"""Ratchet for the unified ocean case board.

Forces completeness: every registered experiment (``AVAILABLE_EXPERIMENTS``) and
every oracle comparison driver (``scripts/validate/ocean_fidelity/compare_*.py``,
minus the non-case diagnostic harnesses) MUST appear in the board — so a new case
cannot be added without classifying its cell. Also checks valid statuses, no
duplicate cases, and that the committed Markdown render is fresh.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from legoesm.ocean.experiments import AVAILABLE_EXPERIMENTS
from legoesm.ocean.fidelity.recipe_case_board import (
    CASES, VALID_STATUSES, VALID_REF_KINDS, NO_REF)

_REPO = Path(__file__).resolve().parents[3]

# compare_*.py drivers that are single-step / cross-grid / closure DIAGNOSTICS, not
# experiment cases — they don't get a board row. Adding a new compare_* driver forces
# a conscious choice: give it a board row, or add it here (no silent omission).
_NON_CASE_COMPARE = frozenset({
    "oceananigans_tendency",        # single-step du/dt tendency match
    "oceananigans_internal_tide_w", # single-step w/continuity match (internal_tide row covers the case)
    "legoesm_cube_vs_latlon",       # cross-grid self-comparison, no oracle
    "eke_kappa_veros",              # GM closure diagnostic
    "eke_len_veros",                # EKE length-scale diagnostic
    # Four drivers landed 2026-07-27..2026-08-12 without a disposition, which
    # is the omission this ratchet exists to force.  Each reason below is
    # quoted from the driver's OWN docstring, not inferred.
    "tendencies_nemo",              # "per process, at a matched state -- no
                                    # time integration"; the oceananigans_tendency class
    "advection_nemo",               # one operator vs NEMO's ttrd_totad, no
                                    # time integration; same class
    "grids_tripole_mpas",           # "It is NOT a NEMO-fidelity statement";
                                    # the legoesm_cube_vs_latlon class
    "three_way_nemo",               # re-scores the SAME global_omip arms on
                                    # one common grid; adds no case
})


def _all_board_names() -> set[str]:
    names: set[str] = set()
    for c in CASES:
        names.add(c["case"])
        names.update(c.get("aliases", ()))
    return names


def test_status_vocabulary_valid():
    for c in CASES:
        for r in c["results"]:
            assert r["status"] in VALID_STATUSES, (
                f"{c['case']}/{r['recipe']}: bad status {r['status']!r}")


def test_no_duplicate_case_names():
    names = [c["case"] for c in CASES]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate case rows: {dupes}"


def test_required_fields_present():
    req = {"case", "tests", "ref_kind", "ref", "results"}
    res_req = {"grid", "recipe", "status", "note"}
    for c in CASES:
        assert not (req - c.keys()), f"{c['case']}: missing fields {req - c.keys()}"
        assert c["results"], f"{c['case']}: must have at least one run result"
        for r in c["results"]:
            assert not (res_req - r.keys()), f"{c['case']}: run missing {res_req - r.keys()}"


def test_reference_kind_valid():
    """Every case declares a valid reference kind; non-'none' kinds name the truth.
    (oracle vs analytic vs published is an editorial judgment — a runnable model is an
    oracle, a paper with no runnable data is 'published', not oracle-level.)"""
    for c in CASES:
        assert c["ref_kind"] in VALID_REF_KINDS, f"{c['case']}: bad ref_kind {c['ref_kind']!r}"
        if c["ref_kind"] != NO_REF:
            assert c["ref"], f"{c['case']}: ref_kind {c['ref_kind']} must name the reference"


def test_no_duplicate_grid_recipe_per_case():
    """Within a case, each (grid, recipe) run appears at most once (unique matrix cell)."""
    for c in CASES:
        keys = [(r["grid"], r["recipe"]) for r in c["results"]]
        dupes = {k for k in keys if keys.count(k) > 1}
        assert not dupes, f"{c['case']}: duplicate (grid, recipe) run(s) {dupes}"


def test_every_registered_experiment_has_a_row():
    """A new AVAILABLE_EXPERIMENTS entry must be classified on the board."""
    board = _all_board_names()
    missing = set(AVAILABLE_EXPERIMENTS) - board
    assert not missing, (
        f"experiments missing from the case board (add a row or alias): {sorted(missing)}")


def test_every_oracle_comparison_has_a_row():
    """A new compare_*.py oracle driver must be a board row or an acknowledged diagnostic."""
    compare_dir = _REPO / "scripts" / "validate" / "ocean_fidelity"
    stems = {p.stem[len("compare_"):] for p in compare_dir.glob("compare_*.py")}
    assert stems, "no compare_*.py drivers found — wrong path?"
    board = _all_board_names()
    missing = {s for s in stems if s not in board and s not in _NON_CASE_COMPARE}
    assert not missing, (
        f"oracle comparison drivers missing from the case board (add a row/alias, or "
        f"add to _NON_CASE_COMPARE if it is a diagnostic): {sorted(missing)}")


def test_blocked_cells_cite_a_reference():
    """A `blocked` result must point at an issue/PR explaining the limitation."""
    for c in CASES:
        for r in c["results"]:
            if r["status"] == "blocked":
                assert "#" in r["note"], (
                    f"{c['case']}/{r['recipe']}: blocked cell must cite an issue (#NNN)")


def test_committed_markdown_is_fresh():
    """The committed render must match the generator (Tier-0 freshness gate)."""
    spec = importlib.util.spec_from_file_location(
        "_build_board",
        _REPO / "scripts" / "validate" / "ocean_fidelity" / "build_recipe_case_board.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = _REPO / "docs" / "ocean" / "fidelity" / "recipe_case_board.md"
    assert out.exists(), "render the board: build_recipe_case_board.py"
    assert out.read_text() == mod.render(), (
        "docs/ocean/fidelity/recipe_case_board.md is STALE — "
        "run build_recipe_case_board.py and commit.")
