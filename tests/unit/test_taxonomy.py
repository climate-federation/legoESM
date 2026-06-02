"""The capability taxonomy must stay consistent with the canonical factories."""

from __future__ import annotations

import pytest

from legoesm import taxonomy
from legoesm.atmosphere.dynamics import DISCRETIZATION_OPTIONS, DYNAMICS_OPTIONS
from legoesm.grids.factory import GLOBAL_GRID_TYPES, REGIONAL_GRID_TYPES


def test_grid_global_flags_match_the_factory() -> None:
    declared_global = {c.name for c in taxonomy.GRID_CAPABILITIES if c.global_}
    assert declared_global == set(GLOBAL_GRID_TYPES)


def test_grid_regional_flags_match_the_factory() -> None:
    declared_regional = {c.name for c in taxonomy.GRID_CAPABILITIES if c.regional}
    assert declared_regional == set(REGIONAL_GRID_TYPES)


def test_every_grid_family_is_global_or_regional() -> None:
    # the union of the two factory lists is exactly the taxonomy's grid families
    assert {c.name for c in taxonomy.GRID_CAPABILITIES} == (
        set(GLOBAL_GRID_TYPES) | set(REGIONAL_GRID_TYPES))


def test_variable_resolution_grids_carry_a_refinement_method() -> None:
    for c in taxonomy.GRID_CAPABILITIES:
        # variable-resolution iff a refinement mechanism is named (no silent flag)
        assert c.variable_resolution == bool(c.refinement)
    var_res = {c.name for c in taxonomy.GRID_CAPABILITIES if c.variable_resolution}
    assert var_res == {"cubed_sphere", "mpas"}  # the two refinement-capable families


def test_complexity_ladders_cover_every_component() -> None:
    assert set(taxonomy.COMPLEXITY_LADDERS) == set(taxonomy.COMPONENT_KINDS)


def test_capability_report_is_well_formed_and_matches_canonical_sources() -> None:
    rep = taxonomy.capability_report()
    assert rep["atmosphere_dynamics"]["model_types"] == list(DYNAMICS_OPTIONS)
    assert rep["atmosphere_dynamics"]["discretizations"] == list(DISCRETIZATION_OPTIONS)
    assert set(rep["axes"]["grids"]) == {c.name for c in taxonomy.GRID_CAPABILITIES}
    assert rep["axes"]["extent"] == list(taxonomy.EXTENT_MODES)
    assert set(rep["axes"]["complexity"]) == set(taxonomy.COMPONENT_KINDS)


def test_grid_capability_lookup_raises_on_unknown() -> None:
    assert taxonomy.grid_capability("mpas").variable_resolution is True
    with pytest.raises(ValueError, match="unknown grid family"):
        taxonomy.grid_capability("nope")
