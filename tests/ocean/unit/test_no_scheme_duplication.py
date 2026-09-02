"""Enforcement test for ocean dynamics scheme consolidation (#214).

Verifies that the shared helpers introduced under
``legoesm/ocean/dynamics/`` are actually used by the grid-specific
``ocean_pe_*`` and ``barotropic_*`` files instead of being silently
duplicated again the next time someone fixes a bug.

The checks are intentionally narrow: they look for the *pattern* that
was previously copied across files and verify it now lives in the
common module, not for full structural equivalence (which is impossible
because the grid-specific stencils legitimately differ).

If a future change re-introduces a copy of one of the consolidated
patterns into a grid-specific file, this test will fail and point at
the offending file so the duplication can be folded back into
``ocean_tendency_common`` or ``barotropic_common``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests.legoesm_paths import legoesm_source_path
from tests.ocean.unit._nemo_branch_isomorphism_baseline import (
    ARTIFICIAL_BRANCH_BASELINE,
    Impl,
    ROUTINE_REGISTRY,
    RoutineRow,
)

DYN = legoesm_source_path("ocean/dynamics/ocean_tendency_common.py").parent

# Files that previously held duplicated logic and now should delegate
# to the common modules.
OCEAN_PE_FILES = [
    DYN / "ocean_pe_cdgrid.py",
    DYN / "ocean_pe_latlon_cgrid.py",
    DYN / "ocean_pe_mpas.py",
]
BAROTROPIC_FILES_WITH_FILTER = [
    DYN / "barotropic_latlon_cgrid.py",
    DYN / "barotropic_mpas.py",
]
BAROTROPIC_FILES_WITH_BOTTOM_DRAG = [
    DYN / "barotropic_latlon_cgrid.py",
    DYN / "barotropic_mpas.py",
]


def _read(path: pathlib.Path) -> str:
    return path.read_text()


# ---------------------------------------------------------------------
# Common modules exist
# ---------------------------------------------------------------------

def test_ocean_tendency_common_module_present():
    """``ocean_tendency_common.py`` must exist and export the helpers."""
    mod = DYN / "ocean_tendency_common.py"
    assert mod.is_file(), f"{mod} should exist (introduced by #214)"
    text = _read(mod)
    for name in (
        "iterate_eos_and_pressure_anomaly",
        "apply_sponge_tracer_relaxation",
        "apply_freshwater_virtual_salt_top",
        "implicit_bottom_drag_factor",
    ):
        assert f"def {name}" in text, (
            f"ocean_tendency_common.py is missing helper {name}; the "
            "ocean PE files will silently regress to inline copies."
        )


def test_barotropic_common_module_present():
    """``barotropic_common.py`` must exist and export the helpers."""
    mod = DYN / "barotropic_common.py"
    assert mod.is_file(), f"{mod} should exist (introduced by #214)"
    text = _read(mod)
    for name in ("compute_filter_weights", "bebt_blend", "maxvel_clip"):
        assert f"def {name}" in text, (
            f"barotropic_common.py is missing helper {name}; the "
            "barotropic solvers will silently regress to inline copies."
        )


# ---------------------------------------------------------------------
# Phase 1: baroclinic tendency consolidation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("path", OCEAN_PE_FILES, ids=lambda p: p.name)
def test_ocean_pe_uses_eos_helper(path: pathlib.Path):
    """Each ocean_pe_*.py must call the shared EOS-iteration helper."""
    text = _read(path)
    assert "iterate_eos_and_pressure_anomaly" in text, (
        f"{path.name} no longer routes the EOS / pressure-anomaly "
        "iteration through ocean_tendency_common — please refactor to "
        "call iterate_eos_and_pressure_anomaly instead of duplicating "
        "the 2-pass loop inline."
    )


@pytest.mark.parametrize("path", OCEAN_PE_FILES, ids=lambda p: p.name)
def test_ocean_pe_no_inline_eos_loop(path: pathlib.Path):
    """No grid-specific file may keep an inline 2-pass EOS loop."""
    text = _read(path)
    # The exact pattern previously duplicated across all three files.
    assert "for _ in range(2):" not in text, (
        f"{path.name} contains an inline ``for _ in range(2):`` — use "
        "ocean_tendency_common.iterate_eos_and_pressure_anomaly instead."
    )
    # ``compute_hydrostatic_pressure`` should also be routed through the
    # helper.  None of the three files needs to import it directly after
    # the refactor.
    assert "compute_hydrostatic_pressure" not in text, (
        f"{path.name} directly references compute_hydrostatic_pressure — "
        "route through iterate_eos_and_pressure_anomaly so the EOS / "
        "reference-Jacobian convention stays in one place."
    )


def test_latlon_cgrid_pe_uses_sponge_helper():
    """Lat-lon C-grid PE delegates tracer sponge to the helper."""
    text = _read(DYN / "ocean_pe_latlon_cgrid.py")
    assert "apply_sponge_tracer_relaxation" in text, (
        "ocean_pe_latlon_cgrid.py should reuse "
        "apply_sponge_tracer_relaxation instead of re-implementing the "
        "tracer sponge inline."
    )


def test_mpas_pe_uses_sponge_helper():
    """MPAS PE delegates tracer sponge + virtual-salt to the helpers."""
    text = _read(DYN / "ocean_pe_mpas.py")
    assert "apply_sponge_tracer_relaxation" in text
    assert "apply_freshwater_virtual_salt_top" in text


# ---------------------------------------------------------------------
# Phase 2: barotropic substep consolidation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("path", BAROTROPIC_FILES_WITH_FILTER,
                         ids=lambda p: p.name)
def test_barotropic_uses_filter_helper(path: pathlib.Path):
    """Cosine/box filter weights come from the shared helper."""
    text = _read(path)
    assert "compute_filter_weights" in text, (
        f"{path.name} should call barotropic_common.compute_filter_weights "
        "instead of inlining the cosine bell formula."
    )
    # Reject the previous inline cosine formulation.
    assert "1.0 + jnp.cos(" not in text, (
        f"{path.name} still contains an inline ``1.0 + jnp.cos(...)`` "
        "filter — please use compute_filter_weights."
    )


@pytest.mark.parametrize("path", BAROTROPIC_FILES_WITH_FILTER,
                         ids=lambda p: p.name)
def test_barotropic_uses_bebt_helper(path: pathlib.Path):
    text = _read(path)
    assert "bebt_blend" in text, (
        f"{path.name} should call barotropic_common.bebt_blend for the "
        "semi-implicit eta blending."
    )


@pytest.mark.parametrize("path", BAROTROPIC_FILES_WITH_FILTER,
                         ids=lambda p: p.name)
def test_barotropic_uses_maxvel_helper(path: pathlib.Path):
    text = _read(path)
    # Either the helper is used or MAXVEL is not present at all.
    assert "maxvel_clip" in text or "_maxvel" not in text


@pytest.mark.parametrize("path", BAROTROPIC_FILES_WITH_BOTTOM_DRAG,
                         ids=lambda p: p.name)
def test_barotropic_uses_bottom_drag_helper(path: pathlib.Path):
    text = _read(path)
    assert "implicit_bottom_drag_factor" in text, (
        f"{path.name} should call ocean_tendency_common."
        "implicit_bottom_drag_factor for the per-substep bottom drag."
    )


# ---------------------------------------------------------------------
# Common modules are import-safe
# ---------------------------------------------------------------------

def test_common_modules_import_cleanly():
    """The new common modules must import without grid-specific deps."""
    import importlib

    mod_oc = importlib.import_module(
        "legoesm.ocean.dynamics.ocean_tendency_common")
    mod_bt = importlib.import_module(
        "legoesm.ocean.dynamics.barotropic_common")

    # Public helpers are exposed.
    for name in (
        "iterate_eos_and_pressure_anomaly",
        "apply_sponge_tracer_relaxation",
        "apply_freshwater_virtual_salt_top",
        "implicit_bottom_drag_factor",
    ):
        assert hasattr(mod_oc, name), (
            f"ocean_tendency_common does not expose {name}")
    for name in ("compute_filter_weights", "bebt_blend", "maxvel_clip"):
        assert hasattr(mod_bt, name), (
            f"barotropic_common does not expose {name}")


# ---------------------------------------------------------------------
# NEMO branch-isomorphism ratchet (fidelity audit, 2026-09).
#
# USER PRINCIPLE: legoESM's branch structure must be isomorphic to NEMO's —
# the only legitimate branch points are NEMO's own namelist/cpp switches. A
# second legoESM implementation of a NEMO routine that NEMO treats as ONE arm
# (a scheme-local twin, a '_ws'/'_nemo_kmm' copy of a shared helper, a
# per-case selector NEMO does not have) is an artificial branch even when
# both copies are individually correct, because a fix landed in one does not
# reach the other.
#
# See ``docs/ocean/fidelity/nemo_branch_isomorphism_map.md`` for the audit
# and ``_nemo_branch_isomorphism_baseline.py`` for the registry this checks.
# ---------------------------------------------------------------------

def _ast_symbol_exists(rel_path: str, symbol: str) -> bool:
    """True if a ``def``/``class`` named ``symbol`` is AST-resolvable
    anywhere in the file at ``rel_path`` (top-level or nested — a helper
    nested inside another function is a legitimate implementation symbol)."""
    path = legoesm_source_path(rel_path)
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text())
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == symbol
        for node in ast.walk(tree)
    )


def find_isomorphism_violations(
    registry: tuple[RoutineRow, ...], baseline: dict[str, str],
) -> list[str]:
    """Pure checker: ``registry`` + ``baseline`` -> violation strings (empty
    = clean). Factored out so the production gate below and the synthetic
    self-checks exercise the exact same logic.

    Rules:
      1. Every ``Impl`` named in the registry must AST-resolve (a rename or
         deletion cannot silently drop coverage).
      2. A row with ``disposition == "ARTIFICIAL_BRANCH"`` and >=2 *distinct*
         impls is a NEW undocumented artificial branch unless its
         ``routine_id`` is a key in ``baseline`` — that is the ratchet:
         legoESM may not grow a second implementation of a shared NEMO
         routine without a human writing down why.
      3. Shrink-only: a ``baseline`` key naming a routine_id that is missing,
         or whose row no longer qualifies (disposition changed away from
         ``ARTIFICIAL_BRANCH``, or collapsed to <=1 distinct impl), is stale
         and must be deleted — a collapse PR is not allowed to leave the old
         allowance sitting around.
    """
    errors: list[str] = []
    seen_ids: set[str] = set()
    for row in registry:
        if row.routine_id in seen_ids:
            errors.append(f"duplicate routine_id in registry: {row.routine_id}")
        seen_ids.add(row.routine_id)

        for impl in row.impls:
            if not _ast_symbol_exists(impl.rel_path, impl.symbol):
                errors.append(
                    f"{row.routine_id} ({row.nemo_routine}): registry symbol "
                    f"{impl.symbol!r} no longer resolves in {impl.rel_path} "
                    "(renamed or removed? update the registry, or restore "
                    "the symbol)")

        distinct = set(row.impls)
        if row.disposition == "ARTIFICIAL_BRANCH" and len(distinct) >= 2:
            if row.routine_id not in baseline:
                errors.append(
                    f"{row.routine_id} ({row.nemo_routine}): {len(distinct)} "
                    "distinct legoESM implementations of one NEMO routine "
                    "with no NEMO switch backing the split, and NOT in "
                    "ARTIFICIAL_BRANCH_BASELINE — this is a NEW artificial "
                    "branch point (see nemo_branch_isomorphism_map.md); "
                    "either delete the duplicate implementation or add a "
                    "baseline entry naming the reason.")

    registry_by_id = {r.routine_id: r for r in registry}
    for rid, reason in baseline.items():
        row = registry_by_id.get(rid)
        if row is None:
            errors.append(
                f"ARTIFICIAL_BRANCH_BASELINE has a stale entry {rid!r} "
                f"({reason!r}): no such routine_id in the registry — remove "
                "it (shrink-only).")
            continue
        distinct = set(row.impls)
        if not (row.disposition == "ARTIFICIAL_BRANCH" and len(distinct) >= 2):
            errors.append(
                f"ARTIFICIAL_BRANCH_BASELINE has a stale entry {rid!r}: the "
                f"branch was collapsed (disposition={row.disposition!r}, "
                f"{len(distinct)} distinct impl(s) remain) — remove the "
                "baseline entry (shrink-only).")
    return errors


def test_nemo_branch_isomorphism_registry_is_clean():
    """Production gate: every registered symbol still exists, every
    undocumented 2+-impl artificial branch is baselined with a reason, and
    the baseline carries no stale entries."""
    errors = find_isomorphism_violations(ROUTINE_REGISTRY, ARTIFICIAL_BRANCH_BASELINE)
    assert not errors, "\n".join(errors)


def test_nemo_branch_isomorphism_registry_nonvacuous():
    """Anti-vacuity: the registry and baseline actually have rows (an empty
    registry would make the gate above pass trivially)."""
    assert len(ROUTINE_REGISTRY) >= 20, (
        "NEMO branch-isomorphism registry looks empty/truncated")
    assert len(ARTIFICIAL_BRANCH_BASELINE) >= 5, (
        "artificial-branch baseline looks empty/truncated")


# --- synthetic-violation self-checks: prove the checker can go red -----

def test_nemo_branch_isomorphism_flags_new_undocumented_duplicate():
    """Plant a NEW artificial branch — two real, distinct, existing symbols
    under one ``ARTIFICIAL_BRANCH`` row that is in no baseline — and confirm
    the checker rejects it. Proves rule 2 (">=2 impls needs a baseline
    entry") actually fires rather than always passing."""
    planted = RoutineRow(
        "FAKE-DUP", "planted duplicate (self-check only)", "ARTIFICIAL_BRANCH",
        "none", (
            Impl("ocean/eos.py", "compute_buoyancy_frequency"),
            Impl("ocean/eos.py", "compute_buoyancy_frequency_adiabatic"),
        ))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-DUP" in e and "NOT in ARTIFICIAL_BRANCH_BASELINE" in e
        for e in errors
    ), f"planted duplicate was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_stale_baseline_entry():
    """Plant a baseline entry for a row that no longer duplicates (only one
    distinct impl remains) and confirm the checker demands its removal.
    Proves rule 3 (shrink-only) actually fires."""
    planted = RoutineRow(
        "FAKE-COLLAPSED", "planted collapsed branch (self-check only)",
        "SHARED", "none", (Impl("ocean/eos.py", "compute_buoyancy_frequency"),))
    errors = find_isomorphism_violations(
        (planted,), {"FAKE-COLLAPSED": "stale reason from a finished collapse"})
    assert any(
        "FAKE-COLLAPSED" in e and "collapsed" in e for e in errors
    ), f"stale baseline entry was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_missing_symbol():
    """Plant a registry row naming a symbol that does not exist and confirm
    the checker rejects it. Proves a rename/deletion cannot silently drop
    coverage."""
    planted = RoutineRow(
        "FAKE-MISSING", "planted missing-symbol row (self-check only)",
        "SHARED", "none",
        (Impl("ocean/eos.py", "this_symbol_does_not_exist_anywhere_zzz"),))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-MISSING" in e and "no longer resolves" in e for e in errors
    ), f"missing symbol was not flagged: {errors}"
