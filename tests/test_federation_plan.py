"""The federation carve plan (FEDERATION.md) must cover every subpackage (Stage D).

Pins the member -> subpackage map to the actual tree so the blueprint cannot drift:
add a new top-level subpackage and this fails until it is assigned to a member.
The four Earth-system component members mirror the import-linter independence
contract, so the carve stays consistent with the proven boundaries.
"""

from __future__ import annotations

from tests.legoesm_paths import legoesm_loose_modules, legoesm_subpackages

#: The carve blueprint: federation member -> the src/legoesm subpackages it bundles.
#: Mirrors the table in FEDERATION.md (and the import-linter layer DAG).
FEDERATION_MEMBERS: dict[str, tuple[str, ...]] = {
    "legoesm-core": ("core", "grids", "runtime", "parallel", "io",
                     "timestepping", "components"),
    "legoesm-atmosphere": ("atmosphere",),
    "legoesm-ocean": ("ocean",),
    "legoesm-land": ("land",),
    "legoesm-ice": ("ice",),
    "legoesm-coupler": ("coupler", "driver"),
    "legoesm-ml": ("ml", "training", "da"),
    "legoesm-tools": ("forcing", "diagnostics", "experiments", "visualization"),
}

#: The loose top-level modules (``legoesm.<mod>``, not subpackages) -> member.
#: These are not directories with an ``__init__.py``, so they fall outside the
#: subpackage map above and would silently escape coverage; pinning them here
#: closes that gap (add a new top-level module and the test fails until assigned).
#:
#: Placement follows the import DAG, NOT just "it lives in src/legoesm today":
#:   * ``surface_albedo`` (+ ``constants``/``thermo``/``registry``) ship with the
#:     SUBSTRATE (legoesm-core) — land/ice/coupler import ``legoesm.surface_albedo``
#:     at top level, so a standalone ``pip install legoesm-land`` must get it from
#:     core, not from the meta package above it;
#:   * ``tuning`` ships with legoesm-ml — its only importer is ml/training, and it
#:     imports forcing/driver, so it cannot live in the meta layer above ml;
#:   * ``cli``/``config``/``dycore_factory``/``supported_matrix`` are the
#:     meta orchestration layer (the root ``legoesm`` member) — nothing below them
#:     imports them, and they pull the whole stack.
FEDERATION_LOOSE_MODULES: dict[str, tuple[str, ...]] = {
    "legoesm-core": ("constants", "thermo", "registry", "surface_albedo"),
    "legoesm-ml": ("tuning",),
    "legoesm": ("cli", "config", "dycore_factory", "experiment_registry",
                "scaling_preflight", "supported_matrix"),
}

#: The four mutually-independent Earth-system components (import-linter contract #2).
COMPONENT_MEMBERS = ("legoesm-atmosphere", "legoesm-ocean",
                     "legoesm-land", "legoesm-ice")


def _actual_subpackages() -> set[str]:
    # Namespace-aware: spans every legoesm.__path__ root, so the substrate
    # packages stay visible after the carve relocates them out of src/legoesm.
    return legoesm_subpackages()


def test_every_subpackage_is_assigned_to_exactly_one_member() -> None:
    assigned: dict[str, str] = {}
    for member, pkgs in FEDERATION_MEMBERS.items():
        for pkg in pkgs:
            assert pkg not in assigned, (
                f"{pkg} assigned to both {assigned[pkg]} and {member}")
            assigned[pkg] = member

    actual = _actual_subpackages()
    orphans = actual - set(assigned)
    phantom = set(assigned) - actual
    assert not orphans, f"subpackages not assigned to a federation member: {orphans}"
    assert not phantom, f"federation member references non-existent subpackages: {phantom}"


def test_every_loose_module_is_assigned_to_exactly_one_member() -> None:
    assigned: dict[str, str] = {}
    for member, mods in FEDERATION_LOOSE_MODULES.items():
        for mod in mods:
            assert mod not in assigned, (
                f"{mod} assigned to both {assigned[mod]} and {member}")
            assigned[mod] = member

    actual = legoesm_loose_modules()
    orphans = actual - set(assigned)
    phantom = set(assigned) - actual
    assert not orphans, f"loose modules not assigned to a federation member: {orphans}"
    assert not phantom, f"federation plan references non-existent loose modules: {phantom}"


def test_loose_module_members_are_known_federation_members() -> None:
    # Every member that ships a loose module must be a real federation member:
    # the eight package members, or the root ``legoesm`` meta-member.
    known = set(FEDERATION_MEMBERS) | {"legoesm"}
    assert set(FEDERATION_LOOSE_MODULES) <= known, (
        f"unknown members in loose-module map: {set(FEDERATION_LOOSE_MODULES) - known}")


def test_component_members_are_the_independent_earth_system_components() -> None:
    component_pkgs = {p for m in COMPONENT_MEMBERS for p in FEDERATION_MEMBERS[m]}
    assert component_pkgs == {"atmosphere", "ocean", "land", "ice"}
    # each component member bundles exactly one subpackage (so they carve apart)
    for m in COMPONENT_MEMBERS:
        assert len(FEDERATION_MEMBERS[m]) == 1


def test_supported_matrix_lists_every_resolvable_atmosphere_solver() -> None:
    """PR D: ``supported_matrix`` is the documented single source of truth for
    distinct atmosphere dycore+discretization+grid implementations, so it must
    list every solver resolvable via ``dynamics._SOLVER_TO_CLASS``.
    ``ucast_primitive_equations`` was a live solver missing from the matrix;
    this pins the two together so the matrix can never silently drop a
    resolvable solver again."""
    from legoesm.supported_matrix import canonical_solver_names
    from legoesm.atmosphere.dynamics import _SOLVER_TO_CLASS

    assert set(canonical_solver_names("atmosphere")) == set(_SOLVER_TO_CLASS), (
        "supported_matrix ATMOSPHERE_MATRIX is out of sync with the resolvable "
        "solvers in dynamics._SOLVER_TO_CLASS — add/remove the SolverEntry."
    )
