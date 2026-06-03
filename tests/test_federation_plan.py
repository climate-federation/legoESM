"""The federation carve plan (FEDERATION.md) must cover every subpackage (Stage D).

Pins the member -> subpackage map to the actual tree so the blueprint cannot drift:
add a new top-level subpackage and this fails until it is assigned to a member.
The four Earth-system component members mirror the import-linter independence
contract, so the carve stays consistent with the proven boundaries.
"""

from __future__ import annotations

from tests.legoesm_paths import legoesm_subpackages

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


def test_component_members_are_the_independent_earth_system_components() -> None:
    component_pkgs = {p for m in COMPONENT_MEMBERS for p in FEDERATION_MEMBERS[m]}
    assert component_pkgs == {"atmosphere", "ocean", "land", "ice"}
    # each component member bundles exactly one subpackage (so they carve apart)
    for m in COMPONENT_MEMBERS:
        assert len(FEDERATION_MEMBERS[m]) == 1
