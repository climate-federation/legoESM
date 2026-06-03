"""The capability taxonomy must stay consistent with the canonical factories AND
with the refinement APIs it advertises."""

from __future__ import annotations

import inspect

import pytest

from legoesm import taxonomy
from legoesm.atmosphere.dynamics import DISCRETIZATION_OPTIONS, DYNAMICS_OPTIONS
from legoesm.components.complexity import ModelComplexity
from legoesm.grids.factory import GLOBAL_GRID_TYPES, REGIONAL_GRID_TYPES


def test_grid_flags_are_derived_from_the_factory() -> None:
    # global_/regional are computed from the factory lists (not retyped), so they
    # match by construction — assert it to guard the derivation.
    assert {c.name for c in taxonomy.GRID_CAPABILITIES if c.global_} == set(
        GLOBAL_GRID_TYPES)
    assert {c.name for c in taxonomy.GRID_CAPABILITIES if c.regional} == set(
        REGIONAL_GRID_TYPES)
    assert {c.name for c in taxonomy.GRID_CAPABILITIES} == (
        set(GLOBAL_GRID_TYPES) | set(REGIONAL_GRID_TYPES))


def test_variable_resolution_apis_actually_exist() -> None:
    """API-drift guard: a grid declared variable-resolution must still expose the
    refinement kwargs the taxonomy names (not just a synced boolean/string)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.voronoi import create_voronoi_mesh

    var_res = {c.name for c in taxonomy.GRID_CAPABILITIES if c.variable_resolution}
    assert var_res == {"cubed_sphere", "mpas"}
    for c in taxonomy.GRID_CAPABILITIES:
        assert c.variable_resolution == bool(c.refinement)  # no orphan flag/string

    cube_params = inspect.signature(create_cubed_sphere).parameters
    assert {"stretch_fac", "target_lat", "target_lon"} <= set(cube_params)
    assert "density_fn" in inspect.signature(create_voronoi_mesh).parameters


def test_complexity_ladders_cover_every_component() -> None:
    assert set(taxonomy.COMPLEXITY_LADDERS) == set(taxonomy.COMPONENT_KINDS)


def test_model_dial_resolves_every_level_to_all_components() -> None:
    dial = taxonomy.model_dial()
    assert set(dial) == {level.value for level in ModelComplexity}
    for level, rungs in dial.items():
        assert set(rungs) == set(taxonomy.COMPONENT_KINDS), level
        assert all(isinstance(v, str) and v for v in rungs.values())


def test_capability_report_grounded_in_canonical_sources() -> None:
    rep = taxonomy.capability_report()
    # dynamics grounded in the atmosphere.dynamics canonical lists
    assert rep["atmosphere_dynamics"]["model_types"] == list(DYNAMICS_OPTIONS)
    assert rep["atmosphere_dynamics"]["discretizations"] == list(DISCRETIZATION_OPTIONS)
    # grids grounded in the factory lists (not in taxonomy's own constant)
    assert set(rep["axes"]["grids"]) == set(GLOBAL_GRID_TYPES) | set(REGIONAL_GRID_TYPES)
    # the resolved model dial is present and complete
    assert set(rep["model_complexity"]) == {level.value for level in ModelComplexity}


def test_grid_capability_lookup_raises_on_unknown() -> None:
    assert taxonomy.grid_capability("mpas").variable_resolution is True
    with pytest.raises(ValueError, match="unknown grid family"):
        taxonomy.grid_capability("nope")
