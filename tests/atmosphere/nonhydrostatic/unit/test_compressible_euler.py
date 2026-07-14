"""Unit tests for the non-hydrostatic compressible Euler dynamical core.

Tests cover:
- HeightCoordinate creation and reference state hydrostatic balance
- TerrainMetric for flat and mountain terrain
- NonHydrostaticState pytree compatibility
- Compressible Euler tendency computation
- Split-explicit time stepping
- Exner function perturbation
- DCMIP-2025 test case initialization
- Kessler microphysics
- JAX differentiability
"""

import pytest
import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    HeightCoordinate,
    TerrainMetric,
    create_height_coordinate,
    compute_terrain_metric,
    compute_reference_state,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
    compute_exner_perturbation,
)
from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel as CompressibleEulerModel,
    CDGridCompressibleEulerConfig,
    cdgrid_compressible_euler_slow_tendencies,
)
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm import constants


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def grid():
    """Small cubed-sphere grid for testing."""
    return create_cubed_sphere(8)


@pytest.fixture
def height_coord():
    """20-level height coordinate with 30km top."""
    return create_height_coordinate(20, 30000.0)


@pytest.fixture
def terrain_metric(grid, height_coord):
    """Flat terrain metric."""
    z_s = jnp.zeros((6, grid.n, grid.n))
    return compute_terrain_metric(z_s, height_coord)


@pytest.fixture
def cdgrid(grid):
    """C-D grid metrics for testing."""
    return create_cubed_sphere_cdgrid(grid)


def _make_nh_state(grid, height_coord, n_tracers=0):
    """Create a rest-state NonHydrostaticState for testing."""
    nlev = height_coord.n_levels
    n = grid.n
    shape_3d = (6, n, n, nlev)
    shape_w = (6, n, n, nlev + 1)
    shape_2d = (6, n, n)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    return NonHydrostaticState(
        u=Field(data=jnp.zeros(shape_3d), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros(shape_3d), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros(shape_w), name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(
            data=jnp.zeros(shape_3d), name="theta_prime", dims=dims_3d, units="K",
        ),
        rho_prime=Field(
            data=jnp.zeros(shape_3d), name="rho_prime", dims=dims_3d, units="kg/m^3",
        ),
        phis=Field(
            data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2",
        ),
        tracers=Field(
            data=jnp.zeros((*shape_3d, n_tracers)),
            name="tracers",
            dims=("face", "x", "y", "level", "tracer"),
            units="kg/kg",
        ),
    )


# ==============================================================================
# HeightCoordinate tests
# ==============================================================================

class TestHeightCoordinate:
    """Tests for the height-based vertical coordinate."""

    def test_create_height_coordinate(self, height_coord):
        """HeightCoordinate has correct dimensions."""
        hc = height_coord
        assert hc.n_levels == 20
        assert hc.z_full.shape == (20,)
        assert hc.z_half.shape == (21,)
        assert hc.dz.shape == (20,)
        assert hc.dz_half.shape == (19,)

    def test_z_half_boundaries(self, height_coord):
        """Model top and surface boundaries are correct."""
        hc = height_coord
        assert float(hc.z_half[0]) == pytest.approx(hc.H, rel=1e-5)
        assert float(hc.z_half[-1]) == pytest.approx(0.0, abs=1e-5)

    def test_dz_sums_to_H(self, height_coord):
        """Layer thicknesses sum to model top height."""
        hc = height_coord
        total = float(jnp.sum(hc.dz))
        assert total == pytest.approx(hc.H, rel=1e-5)

    def test_z_full_between_half(self, height_coord):
        """Full levels are between adjacent half levels."""
        hc = height_coord
        for k in range(hc.n_levels):
            assert float(hc.z_half[k]) > float(hc.z_full[k])
            assert float(hc.z_full[k]) > float(hc.z_half[k + 1])

    def test_reference_state_positive(self, height_coord):
        """Reference state profiles are positive."""
        hc = height_coord
        assert jnp.all(hc.rho_ref > 0)
        assert jnp.all(hc.theta_ref > 0)
        assert jnp.all(hc.exner_ref > 0)

    def test_reference_state_rho_decreases_with_height(self, height_coord):
        """Reference density decreases with altitude."""
        hc = height_coord
        # z_full is top-to-bottom, so rho should increase with index
        for k in range(hc.n_levels - 1):
            assert float(hc.rho_ref[k]) < float(hc.rho_ref[k + 1])

    def test_reference_state_hydrostatic_balance(self):
        """Reference state satisfies hydrostatic balance d(pi)/dz = -g/(c_p*theta)."""
        hc = create_height_coordinate(100, 30000.0)  # High res for accuracy
        g = constants.g
        c_p = constants.c_pd

        # Check d(pi)/dz at interior points
        dpi_dz = jnp.diff(hc.exner_ref) / jnp.diff(hc.z_full)
        theta_avg = 0.5 * (hc.theta_ref[:-1] + hc.theta_ref[1:])
        expected = -g / (c_p * theta_avg)

        # Should be close (trapezoidal integration error)
        rel_error = jnp.abs((dpi_dz - expected) / expected)
        assert float(jnp.max(rel_error)) < 0.05


# ==============================================================================
# TerrainMetric tests
# ==============================================================================

class TestTerrainMetric:
    """Tests for terrain-following coordinate metrics."""

    def test_flat_terrain_jacobian(self, grid, height_coord):
        """Flat terrain has Jacobian = 1."""
        z_s = jnp.zeros((6, grid.n, grid.n))
        tm = compute_terrain_metric(z_s, height_coord)
        assert jnp.allclose(tm.jacobian, 1.0)

    def test_mountain_terrain_jacobian(self, grid, height_coord):
        """Mountain terrain has Jacobian < 1."""
        z_s = jnp.ones((6, grid.n, grid.n)) * 1000.0  # 1km everywhere
        tm = compute_terrain_metric(z_s, height_coord)
        expected_J = (height_coord.H - 1000.0) / height_coord.H
        assert jnp.allclose(tm.jacobian, expected_J, rtol=1e-5)

    def test_terrain_metric_z_at_surface(self, grid, height_coord):
        """Physical z at the surface matches z_s."""
        z_s = jnp.ones((6, grid.n, grid.n)) * 500.0
        tm = compute_terrain_metric(z_s, height_coord)
        # z_half[-1] = 0 (surface in z*), so z_physical = z_s + 0 * J = z_s
        z_surface = tm.z_half_3d[..., -1]
        assert jnp.allclose(z_surface, 500.0, atol=1.0)


# ==============================================================================
# NonHydrostaticState tests
# ==============================================================================

class TestNonHydrostaticState:
    """Tests for the NH state container."""

    def test_state_is_named_tuple(self, grid, height_coord):
        """State is a NamedTuple."""
        state = _make_nh_state(grid, height_coord)
        assert hasattr(state, '_fields')
        assert 'u' in state._fields
        assert 'w' in state._fields
        assert 'theta_prime' in state._fields

    def test_state_jit_compatible(self, grid, height_coord):
        """State works with jax.jit."""
        state = _make_nh_state(grid, height_coord)

        @jax.jit
        def identity(s):
            return s

        result = identity(state)
        assert jnp.allclose(result.u.data, state.u.data)

    def test_state_tree_map(self, grid, height_coord):
        """State works with jax.tree.map."""
        state = _make_nh_state(grid, height_coord)
        doubled = jax.tree.map(lambda x: x * 2, state)
        assert jnp.allclose(doubled.u.data, 0.0)  # 2 * 0 = 0


# ==============================================================================
# Compressible Euler tendency tests
# ==============================================================================

class TestCompressibleEulerTendencies:
    """Tests for C-D grid tendency computation."""

    def test_rest_state_tendencies_small(self, grid, height_coord, terrain_metric, cdgrid):
        """Rest state should produce near-zero tendencies."""
        state = _make_nh_state(grid, height_coord)
        config = CDGridCompressibleEulerConfig(hyperdiff_coeff=0.0, sponge_coeff=0.0)
        tend = cdgrid_compressible_euler_slow_tendencies(
            state, grid, height_coord, terrain_metric, cdgrid, config,
        )
        # u, v tendencies should be zero for rest state
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1e-6
        assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1e-6

    def test_tendencies_finite(self, grid, height_coord, terrain_metric, cdgrid):
        """All tendencies should be finite."""
        state = _make_nh_state(grid, height_coord)
        # Add some non-zero wind
        state = state._replace(
            u=state.u.replace(data=jnp.ones_like(state.u.data) * 10.0),
        )
        config = CDGridCompressibleEulerConfig()
        tend = cdgrid_compressible_euler_slow_tendencies(
            state, grid, height_coord, terrain_metric, cdgrid, config,
        )
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dw_dt.data))
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
        assert jnp.all(jnp.isfinite(tend.drho_prime_dt.data))

    def test_tendency_shapes(self, grid, height_coord, terrain_metric, cdgrid):
        """Tendency shapes match state shapes."""
        state = _make_nh_state(grid, height_coord)
        config = CDGridCompressibleEulerConfig()
        tend = cdgrid_compressible_euler_slow_tendencies(
            state, grid, height_coord, terrain_metric, cdgrid, config,
        )
        assert tend.du_dt.data.shape == state.u.data.shape
        assert tend.dv_dt.data.shape == state.v.data.shape
        assert tend.dw_dt.data.shape == state.w.data.shape
        assert tend.dtheta_prime_dt.data.shape == state.theta_prime.data.shape


# ==============================================================================
# Exner function tests
# ==============================================================================

class TestExnerPerturbation:
    """Tests for the Exner function computation."""

    def test_zero_perturbation(self, height_coord, grid):
        """Zero perturbations give zero Exner perturbation."""
        n = 8
        nlev = height_coord.n_levels
        rho_p = jnp.zeros((6, n, n, nlev))
        theta_p = jnp.zeros((6, n, n, nlev))
        pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)
        assert jnp.allclose(pi_p, 0.0, atol=1e-3)

    def test_positive_theta_perturbation(self, height_coord, grid):
        """Positive theta' gives positive Exner perturbation."""
        n = 8
        nlev = height_coord.n_levels
        rho_p = jnp.zeros((6, n, n, nlev))
        theta_p = jnp.ones((6, n, n, nlev)) * 1.0  # +1K
        pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)
        # Positive theta should increase Exner function
        assert float(jnp.mean(pi_p)) > 0

    def test_large_negative_perturbations_remain_finite(self, height_coord, grid):
        """Guarded Exner computation should stay finite for stressed states."""
        n = 8
        nlev = height_coord.n_levels
        rho_p = jnp.ones((6, n, n, nlev)) * (-0.999 * height_coord.rho_ref)
        theta_p = jnp.ones((6, n, n, nlev)) * (-0.999 * height_coord.theta_ref)
        pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)
        assert jnp.all(jnp.isfinite(pi_p))


# ==============================================================================
# CompressibleEulerModel tests
# ==============================================================================

class TestCompressibleEulerModel:
    """Tests for the model class."""

    def test_single_step(self, grid, height_coord, terrain_metric):
        """Model can take a single time step."""
        config = CDGridCompressibleEulerConfig(
            hyperdiff_coeff=0.0,
            n_acoustic_substeps=2,
        )
        model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)
        state = _make_nh_state(grid, height_coord)
        new_state = model.step(state, dt=1.0)
        assert jnp.all(jnp.isfinite(new_state.u.data))
        assert jnp.all(jnp.isfinite(new_state.w.data))

    def test_multi_step_stability(self, grid, height_coord, terrain_metric):
        """Model is stable for a few time steps from rest."""
        config = CDGridCompressibleEulerConfig(
            hyperdiff_coeff=0.0,
            n_acoustic_substeps=2,
            sponge_coeff=0.0,
        )
        model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)
        state = _make_nh_state(grid, height_coord)
        for _ in range(5):
            state = model.step(state, dt=1.0)
        assert jnp.all(jnp.isfinite(state.u.data))
        assert jnp.all(jnp.isfinite(state.theta_prime.data))


# ==============================================================================
# DCMIP-2025 Test Case initialization tests
# ==============================================================================

class TestDCMIP2025:
    """Tests for DCMIP-2025 test case initialization."""

    def test_tc1_init(self, grid):
        """Test Case 1 initializes without errors."""
        from tests.test_cases.dcmip2025.test_case_1 import dcmip25_tc1_init
        state, hc, tm = dcmip25_tc1_init(grid, n_levels=20)
        assert jnp.all(jnp.isfinite(state.u.data))
        assert jnp.all(jnp.isfinite(state.theta_prime.data))
        assert hc.H == 40000.0

    def test_tc1_topography(self, grid):
        """TC1 mountain has reasonable height."""
        from tests.test_cases.dcmip2025.test_case_1 import dcmip25_tc1_topography
        z_s = dcmip25_tc1_topography(grid)
        assert float(jnp.max(z_s)) <= 2001.0  # peak ~2000m
        assert float(jnp.min(z_s)) >= 0.0

    def test_tc2_init(self, grid):
        """Test Case 2 initializes without errors."""
        from tests.test_cases.dcmip2025.test_case_2 import dcmip25_tc2_init
        state, hc, tm, small_grid = dcmip25_tc2_init(grid, n_levels=20, subcase="a")
        assert jnp.all(jnp.isfinite(state.u.data))
        # Check small-Earth scaling
        assert small_grid.radius < grid.radius

    def test_tc3_init(self, grid):
        """Test Case 3 initializes without errors."""
        from tests.test_cases.dcmip2025.test_case_3 import dcmip25_tc3_init
        state, hc, tm, small_grid = dcmip25_tc3_init(grid, n_levels=20)
        assert jnp.all(jnp.isfinite(state.u.data))
        assert state.tracers.data.shape[-1] == 3  # vapor, cloud, rain
        assert jnp.all(state.tracers.data[..., 0] >= 0)  # vapor >= 0


# ==============================================================================
# Kessler microphysics tests
# ==============================================================================

class TestKesslerMicrophysics:
    """Tests for Kessler warm-rain microphysics."""

    def test_dry_air_no_tendency(self, grid, height_coord, terrain_metric):
        """Dry air (no moisture) produces zero microphysics tendencies."""
        from legoesm.atmosphere.physics.microphysics.config import KesslerConfig, MicrophysicsConfig
        from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
        state = _make_nh_state(grid, height_coord, n_tracers=3)
        config = KesslerConfig()
        micro_config = MicrophysicsConfig(scheme="kessler", kessler=config)
        physics_fn = make_microphysics_physics(micro_config, model_type="nonhydrostatic", dt=1.0)
        tend = physics_fn(state, grid, height_coord, terrain_metric)
        # With zero moisture, tendencies should be near zero
        assert float(jnp.max(jnp.abs(tend.dtracers_dt.data))) < 1e-2

    def test_saturation_mixing_ratio(self):
        """Saturation mixing ratio increases with temperature."""
        from legoesm.thermo import saturation_mixing_ratio
        T_cold = jnp.array(260.0)
        T_warm = jnp.array(300.0)
        p = jnp.array(1e5)
        q_cold = saturation_mixing_ratio(T_cold, p)
        q_warm = saturation_mixing_ratio(T_warm, p)
        assert float(q_warm) > float(q_cold)
        assert float(q_cold) > 0

    def test_kessler_tendencies_finite(self, grid, height_coord, terrain_metric):
        """Kessler tendencies are finite with moisture."""
        from legoesm.atmosphere.physics.microphysics.config import KesslerConfig, MicrophysicsConfig
        from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
        state = _make_nh_state(grid, height_coord, n_tracers=3)
        # Add some moisture
        tracers = state.tracers.data.at[..., 0].set(0.01)  # q_vapor = 10 g/kg
        state = state._replace(tracers=state.tracers.replace(data=tracers))
        config = KesslerConfig()
        micro_config = MicrophysicsConfig(scheme="kessler", kessler=config)
        physics_fn = make_microphysics_physics(micro_config, model_type="nonhydrostatic", dt=1.0)
        tend = physics_fn(state, grid, height_coord, terrain_metric)
        assert jnp.all(jnp.isfinite(tend.dtracers_dt.data))
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))


# ==============================================================================
# Differentiability tests
# ==============================================================================

class TestDifferentiability:
    """Tests that the NH model is differentiable through JAX."""

    def test_grad_through_tendencies(self, grid, height_coord, terrain_metric, cdgrid):
        """jax.grad works through C-D grid tendency computation."""
        config = CDGridCompressibleEulerConfig(hyperdiff_coeff=0.0, sponge_coeff=0.0)
        state = _make_nh_state(grid, height_coord)

        def loss_fn(theta_p_data):
            s = state._replace(
                theta_prime=state.theta_prime.replace(data=theta_p_data),
            )
            tend = cdgrid_compressible_euler_slow_tendencies(
                s, grid, height_coord, terrain_metric, cdgrid, config,
            )
            return jnp.sum(tend.drho_prime_dt.data ** 2)

        grad = jax.grad(loss_fn)(state.theta_prime.data)
        assert jnp.all(jnp.isfinite(grad))


# ==============================================================================
# Small-Earth scaling tests
# ==============================================================================

class TestSmallEarthFactor:
    """Tests for small_earth_factor wiring."""

    def test_small_earth_factor_scales_grid(self, grid, height_coord, terrain_metric):
        """small_earth_factor=10 should scale grid radius and Coriolis."""
        config = CDGridCompressibleEulerConfig(small_earth_factor=10.0)
        model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)
        expected_radius = constants.R_earth / 10.0
        assert abs(model.grid.radius - expected_radius) / expected_radius < 1e-6
        # Coriolis should be scaled by factor
        assert float(jnp.max(jnp.abs(model.grid.f))) > float(jnp.max(jnp.abs(grid.f))) * 9.0

    def test_small_earth_factor_1_no_change(self, grid, height_coord, terrain_metric):
        """Default small_earth_factor=1.0 should preserve original grid."""
        config = CDGridCompressibleEulerConfig(small_earth_factor=1.0)
        model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)
        assert model.grid.radius == grid.radius
        assert jnp.allclose(model.grid.f, grid.f)


class TestMassConservation:
    """Tests for mass conservation options in CE model."""

    def test_nh_mass_diagnostic(self, grid, height_coord, terrain_metric):
        """NH dry mass diagnostic should be positive and finite."""
        from legoesm.core.conservation import compute_nh_dry_mass
        state = _make_nh_state(grid, height_coord)
        mass = compute_nh_dry_mass(
            state.rho_prime.data, height_coord, terrain_metric, grid,
        )
        assert jnp.isfinite(mass)
        assert float(mass) > 0

    def test_fix_mass_config(self):
        """fix_mass and anchor_mass_to_initial config fields exist."""
        config = CompressibleEulerConfig()
        assert config.fix_mass == False
        assert config.anchor_mass_to_initial == False
