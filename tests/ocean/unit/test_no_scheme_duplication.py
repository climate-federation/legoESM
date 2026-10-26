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
import re

import pytest

from tests.legoesm_paths import legoesm_source_path
from tests.ocean.unit._nemo_branch_isomorphism_baseline import (
    ARTIFICIAL_BRANCH_BASELINE,
    EXTRA_NON_NEMO_RECIPE_FILES,
    KNOWN_CARDS,
    ROUTINE_REGISTRY,
    VALID_BASELINE_KINDS,
    VALID_REFERENCE_MODELS,
    BaselineEntry,
    Impl,
    Reference,
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

_DOC_PATH = pathlib.Path(__file__).resolve().parents[3] / (
    "docs/ocean/fidelity/nemo_branch_isomorphism_map.md")

# A doc-row id, at the start of a markdown table row: "| S-03 | ..." or
# "| M-01 | ...". Matches only the audit's "Routine -> implementation map"
# table (section 1) and the reclassification table appended in section 6 —
# no other table in the doc uses this first-column format.
_DOC_ROW_ID_RE = re.compile(r"^\|\s*([SM]-\d+)\s*\|")

# A registry routine_id may cover more than one doc row id (e.g. "S-28_29"
# covers both S-28 and S-29) or fewer than a full id (e.g. "S-07b" is a named
# sub-branch of doc row "S-07"). Ids that don't match either shape (e.g.
# "CARD-ASSEMBLY") are legoESM-internal rows with no doc-row counterpart.
_COMBINED_ID_RE = re.compile(r"^([SM])-(\d+)_(\d+)$")
_SUFFIXED_ID_RE = re.compile(r"^([SM])-(\d+)[a-z]$")


def _doc_row_ids() -> set[str]:
    """The set of S-xx/M-xx row ids the audit doc actually enumerates,
    parsed straight from its own markdown table(s) — the doc is the source
    of truth for "how many routines were audited", not a hardcoded count."""
    text = _DOC_PATH.read_text()
    ids: set[str] = set()
    for line in text.splitlines():
        m = _DOC_ROW_ID_RE.match(line)
        if m:
            ids.add(m.group(1))
    return ids


def _covered_doc_ids(routine_id: str) -> set[str]:
    """Doc row id(s) a given registry ``routine_id`` counts as covering."""
    if re.match(r"^[SM]-\d+$", routine_id):
        return {routine_id}
    m = _COMBINED_ID_RE.match(routine_id)
    if m:
        letter, n1, n2 = m.groups()
        return {f"{letter}-{n1}", f"{letter}-{n2}"}
    m = _SUFFIXED_ID_RE.match(routine_id)
    if m:
        letter, n = m.groups()
        return {f"{letter}-{n}"}
    return set()


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


def _reference_groups(row: RoutineRow) -> dict[tuple[str, str], list[Impl]]:
    """Group ``row``'s impls by ``(reference.model, reference.arm)``. Two
    impls landing in the same group means two legoESM symbols both claim to
    be the SAME reference target — the literal definition of a duplicate,
    independent of whatever ``disposition`` the row carries."""
    groups: dict[tuple[str, str], list[Impl]] = {}
    for impl in row.impls:
        key = (impl.reference.model, impl.reference.arm)
        groups.setdefault(key, []).append(impl)
    return groups


def _non_nemo_recipe_names() -> frozenset[str]:
    """Names that DO legitimize a non-``nemo``/non-``paper`` impl's
    ``selected_by``: real entries from the recipe catalog
    (``legoesm.ocean.recipes.list_recipes()``) plus the small closed set of
    additional non-NEMO fidelity recipe FILES the audit doc names
    (``EXTRA_NON_NEMO_RECIPE_FILES`` in the registry) that aren't catalog
    entries. Deliberately excludes ``KNOWN_CARDS`` (DINO/GYRE/LOCK/OVERFLOW/
    ORCA1) — a NEMO-fidelity card landing on a non-NEMO-cited arm is exactly
    the S-16/S-18/S-19 defect, not a distinct recipe's deliberate choice."""
    from legoesm.ocean.recipes import list_recipes
    return frozenset(list_recipes()) | frozenset(EXTRA_NON_NEMO_RECIPE_FILES)


def _known_selector_names() -> frozenset[str]:
    """Every name a ``selected_by`` tuple may verifiably contain (2026-09-02
    second-review fix, item 3): real recipes plus the closed card/driver list.
    Used only for the "is this name real, not a typo/fabrication" check — see
    ``_non_nemo_recipe_names`` above for which of these actually legitimize an
    impl."""
    return _non_nemo_recipe_names() | frozenset(KNOWN_CARDS)


def _is_legitimately_referenced(impl: Impl, non_nemo_recipe_names: frozenset[str]) -> bool:
    """True if ``impl`` needs no further justification. ``nemo`` needs none
    (it IS the reference this whole ratchet measures against) and ``paper``
    needs none (an external peer-reviewed FORMULA citation — Wright 1997,
    Visbeck et al. 1997 — stands on its own, unlike a claim of "this
    transcribes model X's recipe" which needs a live consumer to back it up).
    Every other model (``legoesm_legacy``, ``unclassified``, and even
    ``veros``/``mitgcm``/``oceananigans`` — S-18's MIN convention cites a real
    paper but is run only by NEMO-fidelity cards, not a distinct recipe, which
    is exactly why it is a genuine duplicate) needs ``selected_by`` to name at
    least one real, non-NEMO-card recipe."""
    if impl.reference.model in ("nemo", "paper"):
        return True
    return any(name in non_nemo_recipe_names for name in impl.reference.selected_by)


def _unreferenced_arms(row: RoutineRow, non_nemo_recipe_names: frozenset[str]) -> list[Impl]:
    """The impls in ``row`` that are not legitimately referenced (see above)."""
    return [i for i in row.impls if not _is_legitimately_referenced(i, non_nemo_recipe_names)]


def _row_has_nemo_plus_unreferenced_arm(row: RoutineRow, non_nemo_recipe_names: frozenset[str]) -> bool:
    """True if ``row`` mixes a ``nemo``-model impl with >=1 unreferenced arm
    (2026-09-02 second-review fix, closes the "common duplicate shape"
    laundering demonstrated on S-16/S-18/S-19): flipping a genuine
    nemo_duplicate row's disposition away from ``ARTIFICIAL_BRANCH`` and
    deleting its baseline entry used to be enough to escape detection the
    instant the two impls' ``arm`` strings merely LOOKED different — this
    check is disposition-independent and does not care what string the
    unreferenced arm chose for itself."""
    if not any(i.reference.model == "nemo" for i in row.impls):
        return False
    return bool(_unreferenced_arms(row, non_nemo_recipe_names))


def _row_needs_baseline_entry(row: RoutineRow) -> bool:
    """True if ``row``, as it stands right now, has a condition that
    legitimately requires an ``ARTIFICIAL_BRANCH_BASELINE`` entry: an
    ``ARTIFICIAL_BRANCH`` row with >=2 distinct impls (the original rule), a
    reference-duplicate group (>=2 impls sharing one (model, arm)), any impl
    whose reference is still untriaged (``model="unclassified"``), or (2026-
    09-02 second-review fix) a ``nemo`` impl paired with >=1 unreferenced arm
    regardless of the arm's own label. Used both to REQUIRE an entry (below)
    and, negated, to detect a STALE one — one predicate, so the two checks
    can never drift apart."""
    if row.disposition == "ARTIFICIAL_BRANCH" and len(set(row.impls)) >= 2:
        return True
    if any(len(set(g)) >= 2 for g in _reference_groups(row).values()):
        return True
    if any(impl.reference.model == "unclassified" for impl in row.impls):
        return True
    if _row_has_nemo_plus_unreferenced_arm(row, _non_nemo_recipe_names()):
        return True
    return False


def find_isomorphism_violations(
    registry: tuple[RoutineRow, ...], baseline: dict[str, BaselineEntry],
) -> list[str]:
    """Pure checker: ``registry`` + ``baseline`` -> violation strings (empty
    = clean). Factored out so the production gate below and the synthetic
    self-checks exercise the exact same logic.

    Rules:
      1. Every ``Impl`` named in the registry must AST-resolve (a rename or
         deletion cannot silently drop coverage).
      2. Every impl's ``reference.model`` must be one of
         ``VALID_REFERENCE_MODELS``, every ``reference.citation`` must be
         non-empty, and a ``model="nemo"`` impl's ``arm`` may not be
         ``""``/``"none"`` (a NEMO reference must name which arm it is).
      3. DISPOSITION-INDEPENDENT duplicate check (closes the 2026-09-02
         BLOCKING finding: a row's ``disposition`` string used to be the only
         thing gating enforcement, so relabelling a row silently escaped the
         ratchet). For EVERY row, regardless of ``disposition``: implementations
         are grouped by ``(reference.model, reference.arm)``; a group of >=2
         needs ``row.routine_id`` in ``baseline`` with ``kind`` ``nemo_duplicate``
         or ``unclassified`` — a NEW artificial branch point (see
         nemo_branch_isomorphism_map.md) otherwise.
      4. Any impl whose ``reference.model == "unclassified"`` requires
         ``row.routine_id`` in ``baseline`` (any kind) — an untriaged
         reference cannot just be written down and forgotten; it must be
         either cited for real or flagged for someone to triage.
      5. An ``OTHER_RECIPE`` row (once it has >=2 impls to compare — a row
         with 0/1 AST-checkable impl is not diversity-enforced, same carve-out
         as the inline-branch rows) must resolve to >=2 distinct
         ``(model, arm)`` groups, and at most one of its impls may cite
         ``model="nemo"`` — two NEMO arms behind "OTHER_RECIPE" is
         NEMO_SWITCH/ARTIFICIAL_BRANCH territory, not a genuine cross-recipe
         fork (this is exactly the reviewer's demonstrated laundering: flip a
         real nemo_duplicate's disposition and delete its baseline entry).
      6. (original rule, kept) A row with ``disposition == "ARTIFICIAL_BRANCH"``
         and >=2 *distinct* impls is a NEW undocumented artificial branch
         unless its ``routine_id`` is a key in ``baseline``.
      7. Shrink-only: a ``baseline`` key naming a routine_id that is missing,
         or whose row no longer needs one (``_row_needs_baseline_entry`` is
         now False), is stale and must be deleted — a collapse PR (or a
         reference fully triaged out of ``unclassified``) is not allowed to
         leave the old allowance sitting around.
      8. Every baseline entry's ``kind`` must be one of ``VALID_BASELINE_KINDS``
         (legoESM hosts several recipes, so a 2nd implementation of one NEMO
         routine is only a defect if both claim the SAME reference arm — a
         separate reclassification pass assigns the real kind by editing
         this field, and a typo/free-text value here would silently defeat
         that data-only workflow).
      9. (2026-09-02 THIRD-review fix, closes the "common duplicate shape"
         laundering that rules 3/5 still missed) Every name in any impl's
         ``reference.selected_by`` must be a real, verifiable identifier — in
         ``legoesm.ocean.recipes.list_recipes()``, in
         ``EXTRA_NON_NEMO_RECIPE_FILES``, or in ``KNOWN_CARDS`` — else a
         made-up/misspelled name would silently launder an impl. AND,
         disposition-independent: a row with >=1 ``nemo`` impl and >=1
         UNREFERENCED arm (any non-``nemo``/non-``paper`` impl whose
         ``selected_by`` names no genuine non-NEMO recipe — see
         ``_is_legitimately_referenced``) needs ``row.routine_id`` in
         ``baseline`` with ``kind`` ``nemo_duplicate`` or ``unclassified``.
         This is what actually closes the gap rules 3/5 left open: S-16,
         S-18 and S-19 each pair a ``nemo`` impl with a differently-named
         non-``nemo`` impl, so rule 3's (model, arm)-matching never fires;
         and once disposition is flipped to ``OTHER_RECIPE``, rule 5's
         diversity check is satisfied too, because 2 DIFFERENTLY-CITED impls
         (even an unreferenced one) already count as 2 distinct groups. Rule
         9 does not care what the unreferenced arm's own label looks like —
         only whether anything real actually selects it.

    Delegated rows (``disposition`` literally ``"see S-XX"``) share their
    underlying impls with the row they point at and are exempt from rules
    3/5/6/9 under their OWN routine_id — their duplication is already fully
    enforced under the referenced row's id; re-requiring a second baseline
    entry for the alias would just be bookkeeping noise, not a new defect.
    """
    errors: list[str] = []
    seen_ids: set[str] = set()
    known_names = _known_selector_names()
    non_nemo_recipe_names = _non_nemo_recipe_names()
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
            ref = impl.reference
            if ref.model not in VALID_REFERENCE_MODELS:
                errors.append(
                    f"{row.routine_id} ({row.nemo_routine}): impl {impl.symbol!r} "
                    f"reference.model={ref.model!r} is not one of "
                    f"{sorted(VALID_REFERENCE_MODELS)} — fix the model field.")
            if not ref.citation:
                errors.append(
                    f"{row.routine_id} ({row.nemo_routine}): impl {impl.symbol!r} "
                    "has an empty reference.citation — every reference needs one "
                    "(a NEMO file:line, a source path, a DOI/paper, or "
                    "'legoESM legacy pre-existing: <commit or module>').")
            if ref.model == "nemo" and ref.arm in ("", "none"):
                errors.append(
                    f"{row.routine_id} ({row.nemo_routine}): impl {impl.symbol!r} "
                    f"has reference.model='nemo' with arm={ref.arm!r} — a NEMO "
                    "reference must name its actual namelist/cpp arm, not "
                    "'none', or it can't be told apart from any other NEMO impl "
                    "in this row.")
            for name in ref.selected_by:
                if name not in known_names:
                    errors.append(
                        f"{row.routine_id} ({row.nemo_routine}): impl {impl.symbol!r} "
                        f"has selected_by name {name!r} that is not in "
                        "list_recipes() or the closed KNOWN_CARDS/"
                        "EXTRA_NON_NEMO_RECIPE_FILES lists — fix the typo, or add "
                        "the identifier to the closed list with its file:line if "
                        "it is real.")

        delegated = row.disposition.startswith("see ")

        if not delegated:
            for key, impls_in_group in _reference_groups(row).items():
                if len(set(impls_in_group)) >= 2:
                    entry = baseline.get(row.routine_id)
                    if entry is None or entry.kind not in ("nemo_duplicate", "unclassified"):
                        errors.append(
                            f"{row.routine_id} ({row.nemo_routine}): "
                            f"{len(set(impls_in_group))} implementations share "
                            f"reference model={key[0]!r} arm={key[1]!r} with no "
                            "ARTIFICIAL_BRANCH_BASELINE entry whose kind is "
                            "'nemo_duplicate' or 'unclassified' — this is a NEW "
                            "artificial branch point regardless of the row's "
                            "disposition label.")

            if any(impl.reference.model == "unclassified" for impl in row.impls):
                if row.routine_id not in baseline:
                    errors.append(
                        f"{row.routine_id} ({row.nemo_routine}): has an impl "
                        "with reference.model='unclassified' and no "
                        "ARTIFICIAL_BRANCH_BASELINE entry — classify its "
                        "reference for real, or add a baseline entry so it "
                        "can't hide.")

            unreferenced = _unreferenced_arms(row, non_nemo_recipe_names)
            if any(i.reference.model == "nemo" for i in row.impls) and unreferenced:
                entry = baseline.get(row.routine_id)
                if entry is None or entry.kind not in ("nemo_duplicate", "unclassified"):
                    names = ", ".join(sorted(i.symbol for i in unreferenced))
                    errors.append(
                        f"{row.routine_id} ({row.nemo_routine}): pairs a nemo "
                        f"impl with unreferenced arm(s) [{names}] (no recipe in "
                        "selected_by names them, and their model is not "
                        "'paper') with no ARTIFICIAL_BRANCH_BASELINE entry "
                        "whose kind is 'nemo_duplicate' or 'unclassified' — "
                        "this is a duplicate regardless of the row's "
                        "disposition label or the unreferenced arm's own "
                        "arm-string (the S-16/S-18/S-19 laundering shape).")

            if row.disposition == "OTHER_RECIPE" and len(row.impls) >= 2:
                distinct_groups = _reference_groups(row)
                nemo_count = sum(1 for i in row.impls if i.reference.model == "nemo")
                if len(distinct_groups) < 2:
                    errors.append(
                        f"{row.routine_id} ({row.nemo_routine}): disposition="
                        f"OTHER_RECIPE but its {len(row.impls)} impl(s) collapse "
                        f"to {len(distinct_groups)} distinct (model, arm) "
                        "group(s) — not a genuine multi-recipe fork.")
                if nemo_count > 1:
                    errors.append(
                        f"{row.routine_id} ({row.nemo_routine}): disposition="
                        f"OTHER_RECIPE but {nemo_count} implementations cite "
                        "reference.model='nemo' — two NEMO arms is "
                        "NEMO_SWITCH/ARTIFICIAL_BRANCH territory, not a "
                        "cross-recipe fork.")

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
    for rid, entry in baseline.items():
        if entry.kind not in VALID_BASELINE_KINDS:
            errors.append(
                f"ARTIFICIAL_BRANCH_BASELINE[{rid!r}].kind={entry.kind!r} is "
                f"not one of {sorted(VALID_BASELINE_KINDS)} — fix the kind "
                "field (data-only edit, no code change needed).")
        row = registry_by_id.get(rid)
        if row is None:
            errors.append(
                f"ARTIFICIAL_BRANCH_BASELINE has a stale entry {rid!r} "
                f"({entry.reason!r}): no such routine_id in the registry — "
                "remove it (shrink-only).")
            continue
        if not _row_needs_baseline_entry(row):
            errors.append(
                f"ARTIFICIAL_BRANCH_BASELINE has a stale entry {rid!r}: the "
                f"branch was collapsed (disposition={row.disposition!r}) or "
                "every impl's reference is now fully triaged — remove the "
                "baseline entry (shrink-only).")
    return errors


def test_nemo_branch_isomorphism_registry_is_clean():
    """Production gate: every registered symbol still exists, every
    undocumented 2+-impl artificial branch is baselined with a reason, and
    the baseline carries no stale entries."""
    errors = find_isomorphism_violations(ROUTINE_REGISTRY, ARTIFICIAL_BRANCH_BASELINE)
    assert not errors, "\n".join(errors)


def test_nemo_branch_isomorphism_registry_covers_all_doc_rows():
    """Coverage gate (independent-review Finding 1, 2026-09-02): every S-xx/
    M-xx row the audit doc enumerates must appear in ROUTINE_REGISTRY — fully
    enforced (AST-resolvable impls) or as an explicit placeholder row (empty
    or single-symbol impls with a one-line reason). A doc row silently
    missing from the registry is exactly how S-11, S-12, S-15, S-22, S-30,
    S-31, S-36, S-39, S-43, S-44, M-02, M-03, M-06 went unenforced despite the
    module docstring's "full map coverage" claim."""
    doc_ids = _doc_row_ids()
    covered: set[str] = set()
    for row in ROUTINE_REGISTRY:
        covered |= _covered_doc_ids(row.routine_id)
    missing = sorted(doc_ids - covered)
    assert not missing, (
        f"{len(missing)} doc row(s) missing from ROUTINE_REGISTRY: {missing} "
        "-- add a RoutineRow (see nemo_branch_isomorphism_map.md for the "
        "citation), even if only as an unenforced placeholder with a reason.")


def test_nemo_branch_isomorphism_registry_nonvacuous():
    """Anti-vacuity: the registry and baseline actually have rows (an empty
    registry would make the gate above pass trivially). The registry-size
    check is tightened to EQUALITY with the doc's own row count (not a loose
    ">=20") so a future truncation of either side is caught immediately."""
    doc_row_count = len(_doc_row_ids())
    assert doc_row_count >= 20, (
        "doc row parser found too few rows -- broken regex, or "
        "nemo_branch_isomorphism_map.md itself looks empty/truncated "
        f"({doc_row_count} rows)")
    assert len(ROUTINE_REGISTRY) == doc_row_count, (
        f"ROUTINE_REGISTRY has {len(ROUTINE_REGISTRY)} rows, doc has "
        f"{doc_row_count} -- see test_nemo_branch_isomorphism_registry_"
        "covers_all_doc_rows for which one(s)")
    assert len(ARTIFICIAL_BRANCH_BASELINE) >= 5, (
        "artificial-branch baseline looks empty/truncated")


# --- synthetic-violation self-checks: prove the checker can go red -----

# A real, valid reference reused by self-checks that don't care about its
# content — only that construction succeeds and it doesn't itself trip rule 2
# (closed-set model / non-empty citation / nemo-arm-not-none).
_OK_REF = Reference("nemo", "some_arm", "some_file.F90:1 (self-check placeholder)")


def test_nemo_branch_isomorphism_flags_new_undocumented_duplicate():
    """Plant a NEW artificial branch — two real, distinct, existing symbols
    both citing the SAME reference, under one ``ARTIFICIAL_BRANCH`` row that
    is in no baseline — and confirm the checker rejects it. Proves the
    original rule (">=2 impls needs a baseline entry") actually fires rather
    than always passing."""
    planted = RoutineRow(
        "FAKE-DUP", "planted duplicate (self-check only)", "ARTIFICIAL_BRANCH",
        "none", (
            Impl("ocean/eos.py", "compute_buoyancy_frequency", _OK_REF),
            Impl("ocean/eos.py", "compute_buoyancy_frequency_adiabatic", _OK_REF),
        ))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-DUP" in e and "NOT in ARTIFICIAL_BRANCH_BASELINE" in e
        for e in errors
    ), f"planted duplicate was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_stale_baseline_entry():
    """Plant a baseline entry for a row that no longer duplicates (only one
    distinct impl remains, fully triaged) and confirm the checker demands its
    removal. Proves the shrink-only stale-entry rule actually fires."""
    planted = RoutineRow(
        "FAKE-COLLAPSED", "planted collapsed branch (self-check only)",
        "SHARED", "none", (Impl("ocean/eos.py", "compute_buoyancy_frequency", _OK_REF),))
    errors = find_isomorphism_violations(
        (planted,),
        {"FAKE-COLLAPSED": BaselineEntry(
            "stale reason from a finished collapse", "unclassified")})
    assert any(
        "FAKE-COLLAPSED" in e and "collapsed" in e for e in errors
    ), f"stale baseline entry was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_invalid_kind():
    """Plant a baseline entry whose ``kind`` is not one of the four allowed
    strings and confirm the checker rejects it. Proves kind is a closed set,
    so the reclassification pass can only ever land on a recognized value."""
    planted = RoutineRow(
        "FAKE-KIND", "planted bad-kind row (self-check only)",
        "ARTIFICIAL_BRANCH", "none", (
            Impl("ocean/eos.py", "compute_buoyancy_frequency",
                 Reference("nemo", "arm_a", "some_file.F90:1")),
            Impl("ocean/eos.py", "compute_buoyancy_frequency_adiabatic",
                 Reference("nemo", "arm_b", "some_file.F90:2")),
        ))
    errors = find_isomorphism_violations(
        (planted,),
        {"FAKE-KIND": BaselineEntry("some reason", "not_a_real_kind")})
    assert any(
        "FAKE-KIND" in e and "not one of" in e for e in errors
    ), f"invalid kind was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_missing_symbol():
    """Plant a registry row naming a symbol that does not exist and confirm
    the checker rejects it. Proves a rename/deletion cannot silently drop
    coverage."""
    planted = RoutineRow(
        "FAKE-MISSING", "planted missing-symbol row (self-check only)",
        "SHARED", "none",
        (Impl("ocean/eos.py", "this_symbol_does_not_exist_anywhere_zzz", _OK_REF),))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-MISSING" in e and "no longer resolves" in e for e in errors
    ), f"missing symbol was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_invalid_reference_model():
    """Plant an impl whose ``reference.model`` is not in the closed set and
    confirm the checker rejects it. Proves the model field can't silently
    hold a typo/free-text value."""
    planted = RoutineRow(
        "FAKE-BAD-MODEL", "planted bad reference.model (self-check only)",
        "SHARED", "none",
        (Impl("ocean/eos.py", "compute_buoyancy_frequency",
              Reference("not_a_real_model", "some_arm", "some_file.F90:1")),))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-BAD-MODEL" in e and "not one of" in e and "reference.model" in e
        for e in errors
    ), f"invalid reference.model was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_empty_citation():
    """Plant an impl with an empty ``reference.citation`` and confirm the
    checker rejects it. Proves every reference is forced to actually cite
    something, not just carry a model/arm label."""
    planted = RoutineRow(
        "FAKE-EMPTY-CITATION", "planted empty citation (self-check only)",
        "SHARED", "none",
        (Impl("ocean/eos.py", "compute_buoyancy_frequency",
              Reference("nemo", "some_arm", "")),))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-EMPTY-CITATION" in e and "empty reference.citation" in e
        for e in errors
    ), f"empty citation was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_nemo_arm_none():
    """Plant an impl citing ``model='nemo'`` with ``arm='none'`` and confirm
    the checker rejects it (independent-review item 4). Proves a NEMO
    reference can't hide behind a placeholder arm that can never collide
    with — or be told apart from — any other NEMO impl in the same row."""
    planted = RoutineRow(
        "FAKE-NEMO-ARM-NONE", "planted nemo arm='none' (self-check only)",
        "SHARED", "none",
        (Impl("ocean/eos.py", "compute_buoyancy_frequency",
              Reference("nemo", "none", "some_file.F90:1")),))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-NEMO-ARM-NONE" in e and "arm='none'" in e for e in errors
    ), f"nemo arm='none' was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_unclassified_without_baseline():
    """Reproduce the reviewer's exact laundering (independent-review item 5,
    self-check 1): a row shaped exactly like the real S-16 defect — one impl
    genuinely citing NEMO, the other citing no non-NEMO reference at all
    (``model="unclassified"``) — with its disposition flipped to
    ``OTHER_RECIPE`` and its baseline entry removed. Confirms the checker
    still goes red: an untriaged reference can no longer be laundered by
    relabelling the row's disposition and deleting the baseline entry, because
    the unclassified-needs-baseline rule is disposition-independent."""
    planted = RoutineRow(
        "FAKE-S16-LAUNDER", "planted S-16-style laundering (self-check only)",
        "OTHER_RECIPE", "none", (
            Impl("ocean/dynamics/barotropic_latlon_cgrid.py",
                 "nemo_literal_continuity_divergence",
                 Reference("nemo", "dyn_spg_ts_continuity", "dynspg_ts.F90:640-700")),
            Impl("ocean/dynamics/barotropic_latlon_cgrid.py", "_run_substep_loop",
                 Reference("unclassified", "dyn_spg_ts_continuity_generic",
                           "no reference named in audit")),
        ))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-S16-LAUNDER" in e and "unclassified" in e and "baseline" in e
        for e in errors
    ), f"unclassified-without-baseline laundering was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_matching_reference_group_without_baseline():
    """Plant two impls that cite the identical (model, arm) reference under a
    row whose disposition is NOT ``ARTIFICIAL_BRANCH`` (e.g. ``SHARED``) and
    confirm the checker still rejects it. Proves the disposition-independent
    duplicate-group rule (independent-review item 2) fires even when the
    original ARTIFICIAL_BRANCH-only rule would stay silent."""
    planted = RoutineRow(
        "FAKE-SAME-REF-GROUP", "planted matching-reference group (self-check only)",
        "SHARED", "none", (
            Impl("ocean/eos.py", "compute_buoyancy_frequency",
                 Reference("nemo", "same_arm", "some_file.F90:1")),
            Impl("ocean/eos.py", "compute_buoyancy_frequency_adiabatic",
                 Reference("nemo", "same_arm", "some_file.F90:2")),
        ))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-SAME-REF-GROUP" in e and "share reference" in e for e in errors
    ), f"matching reference group was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_other_recipe_without_diversity():
    """Plant an ``OTHER_RECIPE`` row with only one impl (so it can never
    resolve to >=2 distinct (model, arm) groups) — using an ``ARTIFICIAL_
    BRANCH``-shaped disposition would be a different check, so this isolates
    the diversity requirement (independent-review item 3, first clause)."""
    planted = RoutineRow(
        "FAKE-OTHER-RECIPE-NO-DIVERSITY",
        "planted OTHER_RECIPE with no diversity (self-check only)",
        "OTHER_RECIPE", "none", (
            Impl("ocean/eos.py", "compute_buoyancy_frequency", _OK_REF),
            Impl("ocean/eos.py", "compute_buoyancy_frequency_adiabatic", _OK_REF),
        ))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-OTHER-RECIPE-NO-DIVERSITY" in e and "distinct (model, arm)" in e
        for e in errors
    ), f"OTHER_RECIPE diversity violation was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_legacy_arm_unreferenced_laundering():
    """Reproduce the THIRD-review "common duplicate shape" finding exactly:
    a ``nemo`` impl paired with a ``legoesm_legacy`` impl whose ``arm`` string
    is simply DIFFERENT (not a matching (model, arm) pair — rule 3 stays
    silent) and whose ``selected_by`` is EMPTY (no recipe actually runs it),
    under ``disposition="OTHER_RECIPE"`` with no baseline entry (rule 5's
    diversity check is satisfied: 2 distinct groups, only 1 nemo). This is
    the literal S-16/S-18/S-19 shape after being laundered exactly as the
    reviewer demonstrated. Confirms rule 9 still rejects it."""
    planted = RoutineRow(
        "FAKE-LEGACY-LAUNDER", "planted legacy-arm laundering (self-check only)",
        "OTHER_RECIPE", "none", (
            Impl("ocean/eos.py", "compute_buoyancy_frequency_nemo_bn2",
                 Reference("nemo", "nemo_own_arm", "some_file.F90:1")),
            Impl("ocean/eos.py", "compute_buoyancy_frequency",
                 Reference("legoesm_legacy", "totally_different_arm_name", "legoESM legacy pre-existing: x")),
        ))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-LEGACY-LAUNDER" in e and "unreferenced arm" in e for e in errors
    ), f"legacy-arm unreferenced laundering was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_fake_selected_by_name():
    """Plant an impl whose ``selected_by`` names a recipe that does not exist
    and confirm the checker rejects it (item 3): a made-up/misspelled recipe
    name must not silently legitimize an unreferenced arm."""
    planted = RoutineRow(
        "FAKE-BAD-SELECTED-BY", "planted fake selected_by name (self-check only)",
        "OTHER_RECIPE", "none", (
            Impl("ocean/eos.py", "compute_buoyancy_frequency_nemo_bn2",
                 Reference("nemo", "nemo_own_arm", "some_file.F90:1")),
            Impl("ocean/eos.py", "compute_buoyancy_frequency",
                 Reference("legoesm_legacy", "some_arm", "legoESM legacy pre-existing: x",
                            selected_by=("this_recipe_does_not_exist_zzz",))),
        ))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-BAD-SELECTED-BY" in e and "this_recipe_does_not_exist_zzz" in e
        for e in errors
    ), f"fake selected_by name was not flagged: {errors}"


def test_nemo_branch_isomorphism_flags_other_recipe_both_nemo():
    """Plant an ``OTHER_RECIPE`` row whose two impls have DIFFERENT arms (so
    they pass the diversity check) but BOTH cite ``model='nemo'`` (self-check
    2 of independent-review item 5). Confirms the checker rejects it: two
    NEMO arms is not a cross-recipe fork, no matter how diverse the arm
    labels look."""
    planted = RoutineRow(
        "FAKE-OTHER-RECIPE-BOTH-NEMO",
        "planted OTHER_RECIPE row whose 2 arms both cite NEMO (self-check only)",
        "OTHER_RECIPE", "none", (
            Impl("ocean/eos.py", "compute_buoyancy_frequency_nemo_bn2",
                 Reference("nemo", "ln_dynvor_ene", "dynvor.F90:100")),
            Impl("ocean/eos.py", "compute_buoyancy_frequency",
                 Reference("nemo", "ln_dynvor_een", "dynvor.F90:200")),
        ))
    errors = find_isomorphism_violations((planted,), {})
    assert any(
        "FAKE-OTHER-RECIPE-BOTH-NEMO" in e and "model='nemo'" in e
        for e in errors
    ), f"OTHER_RECIPE both-nemo laundering was not flagged: {errors}"
