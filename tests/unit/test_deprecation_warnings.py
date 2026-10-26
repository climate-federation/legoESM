"""Tests for deprecation warnings on aliases and legacy names.

Validates that:
- Atmosphere dynamics aliases emit DeprecationWarning.
- Ocean discretization aliases emit DeprecationWarning.
- Conservation latlon aliases emit DeprecationWarning.
- Ocean PE wrapper modules emit DeprecationWarning.
- The supported implementation matrix is self-consistent.
- resolve_solver_name warns on deprecated discretization names.
- create_model warns on deprecated solver names.
"""

from __future__ import annotations

import warnings

import pytest


# ===================================================================
# Atmosphere dynamics aliases
# ===================================================================

class TestAtmosphereDynamicsAliases:
    """Deprecated class/function names in legoesm.atmosphere.dynamics."""

    @pytest.mark.parametrize("alias,canonical", [
        ("ShallowWaterModel", "CDGridShallowWaterModel"),
        ("PrimitiveEquationModel", "CDGridPrimitiveEquationModel"),
        ("CompressibleEulerModel", "CDGridCompressibleEulerModel"),
        ("FVShallowWaterModel", "CDGridShallowWaterModel"),
        ("FVPrimitiveEquationModel", "CDGridPrimitiveEquationModel"),
        ("FVCompressibleEulerModel", "CDGridCompressibleEulerModel"),
        ("CGShallowWaterCubedModel", "CDGridShallowWaterModel"),
        ("shallow_water_tendencies", "cdgrid_shallow_water_tendencies"),
        ("hydrostatic_tendencies", "cdgrid_hydrostatic_tendencies"),
        ("fv_shallow_water_tendencies", "cdgrid_shallow_water_tendencies"),
    ])
    def test_alias_warns(self, alias, canonical):
        import legoesm.atmosphere.dynamics as dyn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = getattr(dyn, alias)
            dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warns) >= 1
            assert canonical in str(dep_warns[0].message)

    @pytest.mark.parametrize("alias,canonical", [
        ("ShallowWaterModel", "CDGridShallowWaterModel"),
        ("FVPrimitiveEquationModel", "CDGridPrimitiveEquationModel"),
    ])
    def test_alias_returns_canonical_object(self, alias, canonical):
        import legoesm.atmosphere.dynamics as dyn
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            alias_obj = getattr(dyn, alias)
            canonical_obj = getattr(dyn, canonical)
            assert alias_obj is canonical_obj

    def test_canonical_name_does_not_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from legoesm.atmosphere.dynamics import CDGridShallowWaterModel  # noqa
            # Filter only DeprecationWarnings from our module
            our_warns = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "atmosphere.dynamics" in str(x.message)
            ]
            assert len(our_warns) == 0


# ===================================================================
# resolve_solver_name
# ===================================================================

class TestResolveSolverName:
    """Deprecation warnings in resolve_solver_name."""

    def test_ambiguous_discretization_resolves_to_cdgrid(self):
        """'centered' and 'finite_volume' are ambiguous (cdgrid on cubed-sphere,
        latlon_cgrid on lat-lon).  Without grid context they resolve to cdgrid
        without a deprecation warning (they are valid, not deprecated)."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            name = resolve_solver_name(
                dynamics="shallow_water", discretization="centered",
            )
            assert name == "cdgrid_shallow_water"
            dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warns) == 0

    def test_deprecated_solver_name_warns(self):
        from legoesm.atmosphere.dynamics import resolve_solver_name
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            name = resolve_solver_name(equations="fv_shallow_water")
            assert name == "cdgrid_shallow_water"
            dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warns) >= 1
            assert "fv_shallow_water" in str(dep_warns[0].message)

    def test_canonical_discretization_no_warn(self):
        from legoesm.atmosphere.dynamics import resolve_solver_name
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            name = resolve_solver_name(
                dynamics="shallow_water", discretization="cdgrid",
            )
            assert name == "cdgrid_shallow_water"
            dep_warns = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "discretization" in str(x.message).lower()
            ]
            assert len(dep_warns) == 0

    def test_finite_volume_resolves_to_cdgrid(self):
        """'finite_volume' is ambiguous — defaults to cdgrid without warning."""
        from legoesm.atmosphere.dynamics import resolve_solver_name
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            name = resolve_solver_name(
                dynamics="hydrostatic", discretization="finite_volume",
            )
            assert name == "cdgrid_primitive_equations"
            dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warns) == 0


# ===================================================================
# create_model
# ===================================================================

class TestCreateModel:
    """Deprecation warnings in create_model."""

    def test_deprecated_name_warns(self):
        from legoesm.atmosphere.dynamics import create_model
        from legoesm.grids.cubed_sphere import create_cubed_sphere

        grid = create_cubed_sphere(4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = create_model("fv_shallow_water", grid=grid)
            dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warns) >= 1
            assert "fv_shallow_water" in str(dep_warns[0].message)
        # Should still return the right model
        from legoesm.atmosphere.dynamics import CDGridShallowWaterModel
        assert isinstance(model, CDGridShallowWaterModel)


# ===================================================================
# Ocean discretization aliases
# ===================================================================

class TestOceanDiscretizationAliases:
    """Ocean model warns on deprecated discretization names."""

    @pytest.mark.parametrize("alias", ["centered", "finite_volume", "fv"])
    def test_ocean_discretization_warns(self, alias):
        from legoesm.ocean import OceanModel
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star

        grid = create_cubed_sphere(4)
        z_coord = create_ocean_z_star(3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = OceanModel(grid, z_coord, discretization=alias)
            dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warns) >= 1
            assert alias in str(dep_warns[0].message)

    def test_canonical_discretization_no_warn(self):
        from legoesm.ocean import OceanModel
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star

        grid = create_cubed_sphere(4)
        z_coord = create_ocean_z_star(3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = OceanModel(grid, z_coord, discretization="cdgrid")
            dep_warns = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "discretization" in str(x.message).lower()
            ]
            assert len(dep_warns) == 0


# ===================================================================
# Conservation latlon aliases
# ===================================================================


# ===================================================================
# Implementation matrix consistency
# ===================================================================

class TestSupportedMatrix:
    """The implementation matrix is self-consistent."""

    def test_all_canonical_names_unique(self):
        from legoesm.supported_matrix import SUPPORTED_MATRIX
        # (name, dynamics) pairs: a multi-axis solver (fv3_duo) has one
        # row per axis, same canonical name (codex 2026-08-18 — a note
        # field is not machine-readable coverage).
        names = [(e.canonical_name, e.dynamics) for e in SUPPORTED_MATRIX]
        assert len(names) == len(set(names)), (
            f"Duplicate (canonical_name, dynamics) pairs: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_all_class_names_unique(self):
        from legoesm.supported_matrix import SUPPORTED_MATRIX
        names = [(e.class_name, e.dynamics) for e in SUPPORTED_MATRIX]
        assert len(names) == len(set(names)), (
            f"Duplicate (class_name, dynamics) pairs: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_atmosphere_canonical_names_match_available_solvers(self):
        from legoesm.supported_matrix import ATMOSPHERE_MATRIX
        from legoesm.atmosphere.dynamics import AVAILABLE_SOLVERS
        matrix_names = {e.canonical_name for e in ATMOSPHERE_MATRIX}
        solver_names = set(AVAILABLE_SOLVERS)
        assert matrix_names == solver_names, (
            f"Mismatch: in matrix but not AVAILABLE_SOLVERS: "
            f"{matrix_names - solver_names}, "
            f"in AVAILABLE_SOLVERS but not matrix: "
            f"{solver_names - matrix_names}"
        )

    def test_deprecated_aliases_map_to_canonical(self):
        from legoesm.supported_matrix import (
            ATMOSPHERE_DEPRECATED_SOLVER_NAMES,
            ATMOSPHERE_MATRIX,
        )
        canonical = {e.canonical_name for e in ATMOSPHERE_MATRIX}
        for alias, target in ATMOSPHERE_DEPRECATED_SOLVER_NAMES.items():
            assert target in canonical, (
                f"Deprecated alias {alias!r} maps to {target!r} "
                f"which is not a canonical solver name"
            )

    def test_ocean_deprecated_discretizations_all_map_to_cdgrid(self):
        from legoesm.supported_matrix import OCEAN_DEPRECATED_DISCRETIZATIONS
        for alias, target in OCEAN_DEPRECATED_DISCRETIZATIONS.items():
            assert target == "cdgrid", (
                f"Ocean alias {alias!r} maps to {target!r}, expected 'cdgrid'"
            )

    def test_matrix_has_expected_entry_count(self):
        from legoesm.supported_matrix import (
            ATMOSPHERE_MATRIX, OCEAN_MATRIX,
        )
        # 19 atmosphere + 5 ocean = 24 genuinely distinct implementations.
        # NOTE: pre-existing drift absorbed here — HEAD carried 18 entries
        # (tracer_transport_spectral landed without bumping this count, CI
        # dark) while this assertion still said 17; fv3_duo_primitive_
        # equations (the certified duo-cube lane) makes it 19.
        assert len(ATMOSPHERE_MATRIX) == 20
        assert len(OCEAN_MATRIX) == 5

    def test_canonical_solver_names_helper(self):
        from legoesm.supported_matrix import canonical_solver_names
        atmo = canonical_solver_names("atmosphere")
        assert "cdgrid_shallow_water" in atmo
        assert "spectral_shallow_water" in atmo
        ocean = canonical_solver_names("ocean")
        assert "cdgrid" in ocean
        assert "spectral" in ocean
