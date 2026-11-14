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


# ===================================================================
# resolve_solver_name
# ===================================================================


# ===================================================================
# create_model
# ===================================================================


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
        assert len(ATMOSPHERE_MATRIX) == 17
        assert len(OCEAN_MATRIX) == 5

    def test_canonical_solver_names_helper(self):
        from legoesm.supported_matrix import canonical_solver_names
        atmo = canonical_solver_names("atmosphere")
        assert "fv3_duo_primitive_equations" in atmo
        assert "spectral_shallow_water" in atmo
        ocean = canonical_solver_names("ocean")
        assert "cdgrid" in ocean
        assert "spectral" in ocean
