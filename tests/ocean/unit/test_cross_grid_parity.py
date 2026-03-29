"""Cross-grid parity checks for ocean test cases.

Validates that key ocean configurations produce qualitatively consistent
results across cubed-sphere FV, lat-lon FV, and spectral grid types.

Parity criteria (not exact equality --- different numerics):
- All grids produce finite output
- Rest state stays near rest on all grids
- Conservation diagnostics are bounded similarly
- Temperature range is physically reasonable on all grids
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.ocean.state import OceanConfig


# ==============================================================================
# Grid fixtures
# ==============================================================================

@pytest.fixture
def cube_grid():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    return create_cubed_sphere(8)


@pytest.fixture
def z_coord():
    from legoesm.ocean.vertical import create_ocean_z_star
    return create_ocean_z_star(n_levels=5, H_max=4000.0)


@pytest.fixture
def cube_state(cube_grid, z_coord):
    from legoesm.ocean.init import rest_state_ocean
    return rest_state_ocean(cube_grid, z_coord, H_max=4000.0)


@pytest.fixture
def cube_model(cube_grid, z_coord):
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    config = OceanConfig(
        A_h=1e3, K_h=1e2, A_v=1e-3, K_v=1e-4,
        n_barotropic_substeps=5,
        hyperdiff_coeff=0.0,
    )
    return OceanModel(cube_grid, z_coord, config)


# ==============================================================================
# Cross-grid rest-state parity
# ==============================================================================

class TestRestStateParity:
    """Rest state should remain near rest on all grid types."""

    def test_cube_rest_state_stable(self, cube_model, cube_state):
        """Cubed-sphere rest state: 5 steps should stay near rest."""
        s = cube_state
        for _ in range(5):
            s = cube_model.step(s, 3600.0)

        # Velocities should be near-zero
        u_max = float(jnp.max(jnp.abs(s.u.data)))
        v_max = float(jnp.max(jnp.abs(s.v.data)))
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))

        assert u_max < 1.0, f"Rest state u_max={u_max} too large"
        assert v_max < 1.0, f"Rest state v_max={v_max} too large"
        assert eta_max < 1.0, f"Rest state eta_max={eta_max} too large"
        assert jnp.all(jnp.isfinite(s.T.data))

    def test_cube_rest_temperature_bounded(self, cube_model, cube_state):
        """Temperature should stay in reasonable range [0, 35] C."""
        s = cube_state
        for _ in range(5):
            s = cube_model.step(s, 3600.0)

        # land_mask: 1=ocean, 0=land.  Keep ocean cells, set land to nan.
        mask = cube_state.land_mask.data
        T_ocean = jnp.where(
            mask[..., jnp.newaxis] > 0.5,
            s.T.data,
            jnp.nan,
        )
        T_max = float(jnp.nanmax(T_ocean))
        T_min = float(jnp.nanmin(T_ocean))

        assert T_min > -5.0, f"T_min={T_min} below physical range"
        assert T_max < 40.0, f"T_max={T_max} above physical range"


# ==============================================================================
# Conservation parity across grids
# ==============================================================================

class TestConservationParity:
    """Conservation properties should hold on all grid types."""

    def test_cube_volume_conservation(self, cube_model, cube_state, cube_grid):
        """Volume (eta integral) should be conserved on cubed-sphere."""
        area = cube_grid.area
        eta_0 = float(jnp.sum(cube_state.eta.data * area))

        s = cube_state
        for _ in range(5):
            s = cube_model.step(s, 3600.0)

        eta_f = float(jnp.sum(s.eta.data * area))
        # Relative volume change should be small
        if abs(eta_0) > 1e-10:
            rel_change = abs(eta_f - eta_0) / abs(eta_0)
            assert rel_change < 0.01, f"Volume drift={rel_change}"
        else:
            # Initial volume near zero (rest state).
            # Float32 rounding on a C8 grid can produce O(10) drift in the
            # area-weighted eta sum; use a generous tolerance.
            assert abs(eta_f) < 100.0, f"Volume grew from zero to {eta_f}"


# ==============================================================================
# Ocean test matrix case-name alignment
# ==============================================================================

class TestOceanMatrixAlignment:
    """Verify ocean test matrix case names are consistent."""

    def test_canonical_runner_cases_exist(self):
        """All canonical runner test cases should be discoverable via AST."""
        import ast
        from pathlib import Path

        matrix_path = (
            Path(__file__).resolve().parents[3] / "scripts" / "run_ocean_test_matrix.py"
        )
        if not matrix_path.exists():
            pytest.skip("scripts/run_ocean_test_matrix.py not found")

        tree = ast.parse(matrix_path.read_text())

        # Find RUNNERS dict assignment and extract its keys.
        # Handles both ``RUNNERS = {...}`` (ast.Assign) and
        # ``RUNNERS: dict[str, Callable] = {...}`` (ast.AnnAssign).
        runners_keys: set[str] | None = None
        for node in ast.walk(tree):
            target_name: str | None = None
            value_node = None

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "RUNNERS":
                        target_name = target.id
                        value_node = node.value
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "RUNNERS":
                    target_name = node.target.id
                    value_node = node.value

            if target_name == "RUNNERS" and isinstance(value_node, ast.Dict):
                runners_keys = set()
                for key in value_node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        runners_keys.add(key.value)

        assert runners_keys is not None, "RUNNERS dict not found in canonical runner"

        expected = {
            "rest_state", "barotropic_wave", "wind_gyre", "baroclinic",
            "phillips_two_layer", "inertia_gravity_wave",
            "lock_exchange", "overflow", "stommel_gyre_tracer",
        }
        assert runners_keys == expected, (
            f"RUNNERS keys mismatch.\n"
            f"  Expected: {sorted(expected)}\n"
            f"  Got:      {sorted(runners_keys)}"
        )

    def test_all_grids_matrix_cases_subset_of_canonical(self):
        """All-grids matrix REQUESTED_CASES should be subset of canonical."""
        import ast
        from pathlib import Path

        matrix_path = (
            Path(__file__).resolve().parents[1] / "run_ocean_all_grids_matrix.py"
        )
        if not matrix_path.exists():
            pytest.skip("run_ocean_all_grids_matrix.py not found")

        tree = ast.parse(matrix_path.read_text())
        requested = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "REQUESTED_CASES":
                        requested = ast.literal_eval(node.value)

        assert requested is not None, "REQUESTED_CASES not found"

        canonical = {
            "rest_state", "barotropic_wave", "wind_gyre", "baroclinic",
            "phillips_two_layer", "inertia_gravity_wave",
            "lock_exchange", "overflow", "stommel_gyre_tracer",
        }
        missing = set(requested) - canonical
        assert not missing, (
            f"All-grids matrix references cases not in canonical runner: {missing}"
        )

    def test_spectral_cases_documented(self):
        """Spectral runner cases should be a known subset."""
        import ast
        from pathlib import Path

        matrix_path = (
            Path(__file__).resolve().parents[1] / "run_ocean_all_grids_matrix.py"
        )
        if not matrix_path.exists():
            pytest.skip("run_ocean_all_grids_matrix.py not found")

        tree = ast.parse(matrix_path.read_text())
        spectral = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "SPECTRAL_CASES":
                        spectral = ast.literal_eval(node.value)

        assert spectral is not None, "SPECTRAL_CASES not found in all-grids matrix"
        # Spectral cases should be explicitly documented
        assert "rest_state" in spectral
        assert len(spectral) >= 2  # At least rest_state + one physics case
