"""Unit tests for the ocean dynamical core.

Tests cover:
- Wright (1997) EOS
- Ocean z-star vertical coordinate
- State containers and pytree compatibility
- Baroclinic tendency computation
- Barotropic substeps
- Split-explicit model stepping
- Land masking
- Conservation fixers
- JAX differentiability
- Spectral ocean variant
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.eos import (
    wright_eos,
    density_perturbation,
    compute_hydrostatic_pressure,
    compute_buoyancy_frequency,
)
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    create_ocean_z_star,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.state import OceanState, OceanConfig
from legoesm.ocean.init import rest_state_ocean, idealized_bathymetry
from legoesm.ocean.dynamics.ocean_pe import (
    ocean_baroclinic_tendencies,
    _vertical_advection_ocean,
)
from legoesm.ocean.dynamics.ocean_model import OceanModel
from legoesm.ocean.conservation import (
    fix_volume_ocean,
    fix_heat_ocean,
    fix_salt_ocean,
    ocean_conservation_fixer,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def ocean_grid():
    """Small C8 cubed-sphere grid for fast tests."""
    return create_cubed_sphere(8)


@pytest.fixture
def ocean_z_coord():
    """Small 10-level vertical coordinate for fast tests."""
    return create_ocean_z_star(n_levels=10, H_max=4000.0)


@pytest.fixture
def ocean_state(ocean_grid, ocean_z_coord):
    """Rest-state ocean initial condition."""
    return rest_state_ocean(
        ocean_grid, ocean_z_coord,
        T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )


@pytest.fixture
def ocean_config():
    """Test configuration with moderate mixing."""
    return OceanConfig(
        A_h=1e3, K_h=1e2, A_v=1e-3, K_v=1e-4,
        n_barotropic_substeps=10,
        hyperdiff_coeff=0.0,
    )


# ==============================================================================
# EOS Tests
# ==============================================================================

class TestWrightEOS:
    """Tests for the Wright (1997) equation of state."""

    def test_reference_value(self):
        """EOS should give reasonable density for standard seawater."""
        rho = wright_eos(
            jnp.array(10.0),  # T=10C
            jnp.array(35.0),  # S=35 PSU
            jnp.array(0.0),   # surface
        )
        # Seawater at T=10, S=35, p=0 should be ~1026-1027 kg/m^3
        assert 1020.0 < float(rho) < 1030.0

    def test_warm_water_lighter(self):
        """Warmer water should be less dense at same salinity."""
        rho_warm = wright_eos(jnp.array(25.0), jnp.array(35.0), jnp.array(0.0))
        rho_cold = wright_eos(jnp.array(5.0), jnp.array(35.0), jnp.array(0.0))
        assert float(rho_warm) < float(rho_cold)

    def test_salty_water_heavier(self):
        """Saltier water should be more dense at same temperature."""
        rho_salty = wright_eos(jnp.array(15.0), jnp.array(38.0), jnp.array(0.0))
        rho_fresh = wright_eos(jnp.array(15.0), jnp.array(32.0), jnp.array(0.0))
        assert float(rho_salty) > float(rho_fresh)

    def test_pressure_increases_density(self):
        """Higher pressure should increase density."""
        rho_surface = wright_eos(jnp.array(10.0), jnp.array(35.0), jnp.array(0.0))
        rho_deep = wright_eos(jnp.array(10.0), jnp.array(35.0), jnp.array(5e7))  # 500 bar
        assert float(rho_deep) > float(rho_surface)

    def test_vectorized(self):
        """EOS should work with array inputs."""
        T = jnp.array([5.0, 10.0, 20.0, 25.0])
        S = jnp.full(4, 35.0)
        p = jnp.zeros(4)
        rho = wright_eos(T, S, p)
        assert rho.shape == (4,)
        # Density should decrease with temperature
        assert jnp.all(jnp.diff(rho) < 0)

    def test_density_perturbation(self):
        """density_perturbation should subtract reference density."""
        rho_p = density_perturbation(
            jnp.array(10.0), jnp.array(35.0), jnp.array(0.0), rho_ref=1025.0,
        )
        rho = wright_eos(jnp.array(10.0), jnp.array(35.0), jnp.array(0.0))
        assert float(rho_p) == pytest.approx(float(rho) - 1025.0, abs=1e-6)

    def test_jit_compatible(self):
        """EOS should work under jax.jit."""
        f = jax.jit(wright_eos)
        rho = f(jnp.array(10.0), jnp.array(35.0), jnp.array(0.0))
        assert jnp.isfinite(rho)


# ==============================================================================
# Vertical Coordinate Tests
# ==============================================================================

class TestOceanZStar:
    """Tests for the ocean z-star vertical coordinate."""

    def test_creation(self):
        """Coordinate should have correct shape and properties."""
        z = create_ocean_z_star(n_levels=50, H_max=5500.0)
        assert z.n_levels == 50
        assert z.H_max == 5500.0
        assert z.z_full_ref.shape == (50,)
        assert z.z_half_ref.shape == (51,)
        assert z.dz_ref.shape == (50,)
        assert z.dz_half_ref.shape == (49,)

    def test_monotonic_depth(self):
        """Levels should monotonically decrease (go deeper)."""
        z = create_ocean_z_star(n_levels=50, H_max=5500.0)
        # z_full_ref should be monotonically decreasing (more negative)
        for k in range(49):
            assert float(z.z_full_ref[k]) > float(z.z_full_ref[k + 1])

    def test_boundary_values(self):
        """Surface at 0, bottom at -H_max."""
        z = create_ocean_z_star(n_levels=50, H_max=5500.0)
        assert float(z.z_half_ref[0]) == pytest.approx(0.0, abs=1e-3)
        assert float(z.z_half_ref[-1]) == pytest.approx(-5500.0, abs=1e-3)

    def test_positive_thickness(self):
        """All layer thicknesses should be positive."""
        z = create_ocean_z_star(n_levels=50, H_max=5500.0)
        assert jnp.all(z.dz_ref > 0)

    def test_stretched_grid(self):
        """Surface layers should be thinner than deep layers."""
        z = create_ocean_z_star(n_levels=50, H_max=5500.0, dz_surface=10.0, dz_deep=200.0)
        assert float(z.dz_ref[0]) < float(z.dz_ref[-1])

    def test_layer_thickness_with_eta(self, ocean_grid, ocean_z_coord):
        """Layer thickness should change with sea surface height."""
        H_bathy = jnp.full((6, 8, 8), 4000.0, dtype=jnp.float32)
        eta_zero = jnp.zeros((6, 8, 8), dtype=jnp.float32)
        eta_pos = jnp.full((6, 8, 8), 1.0, dtype=jnp.float32)

        h_zero = compute_layer_thickness(eta_zero, H_bathy, ocean_z_coord)
        h_pos = compute_layer_thickness(eta_pos, H_bathy, ocean_z_coord)

        # Positive eta should make layers slightly thicker
        assert jnp.all(h_pos > h_zero)

    def test_jacobian(self, ocean_z_coord):
        """Jacobian should be 1 when eta=0 and H_bathy=H_max."""
        H_max = ocean_z_coord.H_max
        eta = jnp.array(0.0)
        H_bathy = jnp.array(H_max)
        J = compute_ocean_jacobian(eta, H_bathy, ocean_z_coord)
        assert float(J) == pytest.approx(1.0, abs=1e-6)

    def test_input_validation(self):
        """Coordinate constructor should reject invalid values."""
        with pytest.raises(ValueError):
            create_ocean_z_star(n_levels=1)
        with pytest.raises(ValueError):
            create_ocean_z_star(H_max=0.0)
        with pytest.raises(ValueError):
            create_ocean_z_star(dz_surface=0.0)
        with pytest.raises(ValueError):
            create_ocean_z_star(dz_deep=0.0)


# ==============================================================================
# State Tests
# ==============================================================================

class TestOceanState:
    """Tests for ocean state containers."""

    def test_pytree_compatible(self, ocean_state):
        """State should be a valid JAX pytree."""
        leaves = jax.tree_util.tree_leaves(ocean_state)
        assert len(leaves) == 7  # u, v, T, S, eta, H_bathy, land_mask

    def test_tree_map(self, ocean_state):
        """tree_map should work on ocean state."""
        doubled = jax.tree.map(lambda x: x * 2, ocean_state)
        assert jnp.allclose(doubled.u.data, ocean_state.u.data * 2)

    def test_shapes(self, ocean_state, ocean_grid, ocean_z_coord):
        """State fields should have correct shapes."""
        n = ocean_grid.n
        nlev = ocean_z_coord.n_levels
        assert ocean_state.u.shape == (6, n, n, nlev)
        assert ocean_state.v.shape == (6, n, n, nlev)
        assert ocean_state.T.shape == (6, n, n, nlev)
        assert ocean_state.S.shape == (6, n, n, nlev)
        assert ocean_state.eta.shape == (6, n, n)
        assert ocean_state.H_bathy.shape == (6, n, n)
        assert ocean_state.land_mask.shape == (6, n, n)

    def test_bathymetry_input_validation(self, ocean_grid):
        """Bathymetry helper should reject invalid depth/latitude parameters."""
        with pytest.raises(ValueError):
            idealized_bathymetry(ocean_grid, H_max=0.0)
        with pytest.raises(ValueError):
            idealized_bathymetry(ocean_grid, H_max=1000.0, land_lat_threshold=-1.0)
        with pytest.raises(ValueError):
            idealized_bathymetry(ocean_grid, H_max=1000.0, land_lat_threshold=91.0)


# ==============================================================================
# Tendency Tests
# ==============================================================================

class TestOceanTendencies:
    """Tests for ocean baroclinic tendency computation."""

    def test_rest_state_small_tendencies(self, ocean_state, ocean_grid, ocean_z_coord, ocean_config):
        """Rest state should produce near-zero tendencies."""
        tend = ocean_baroclinic_tendencies(
            ocean_state, ocean_grid, ocean_z_coord, ocean_config,
        )
        # Velocity tendencies should be small (Coriolis on zero velocity = 0)
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1e-3
        assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1e-3
        # Eta tendency should be small
        assert float(jnp.max(jnp.abs(tend.deta_dt.data))) < 1e-3

    def test_tendencies_finite(self, ocean_state, ocean_grid, ocean_z_coord, ocean_config):
        """All tendencies should be finite."""
        tend = ocean_baroclinic_tendencies(
            ocean_state, ocean_grid, ocean_z_coord, ocean_config,
        )
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
        assert jnp.all(jnp.isfinite(tend.deta_dt.data))

    def test_static_fields_zero_tendency(self, ocean_state, ocean_grid, ocean_z_coord, ocean_config):
        """H_bathy and land_mask tendencies should be exactly zero."""
        tend = ocean_baroclinic_tendencies(
            ocean_state, ocean_grid, ocean_z_coord, ocean_config,
        )
        assert float(jnp.max(jnp.abs(tend.dH_bathy_dt.data))) == 0.0
        assert float(jnp.max(jnp.abs(tend.dland_mask_dt.data))) == 0.0

    def test_land_masking(self, ocean_grid, ocean_z_coord, ocean_config):
        """Tendencies should be exactly zero on land cells."""
        state = rest_state_ocean(
            ocean_grid, ocean_z_coord, H_max=4000.0, land_lat_threshold=60.0,
        )
        tend = ocean_baroclinic_tendencies(
            state, ocean_grid, ocean_z_coord, ocean_config,
        )
        land = state.land_mask.data < 0.5
        # On land cells, all tendencies should be zero
        assert float(jnp.max(jnp.abs(tend.du_dt.data[land]))) == 0.0
        assert float(jnp.max(jnp.abs(tend.dv_dt.data[land]))) == 0.0
        assert float(jnp.max(jnp.abs(tend.deta_dt.data[land]))) == 0.0


class TestVerticalAdvection:
    """Tests for vertical upwind advection."""

    def test_upward_flow_uses_deeper_donor(self, ocean_z_coord):
        """Upward flow should use deeper donor and zero-gradient at bottom."""
        field = ocean_z_coord.z_full_ref[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        jac = jnp.ones((1, 1, 1))
        w_half = jnp.ones((1, 1, 1, ocean_z_coord.n_levels + 1))

        adv = _vertical_advection_ocean(field, w_half, ocean_z_coord, jac)

        assert jnp.allclose(adv[..., :-1], -1.0, atol=1e-6)
        assert float(adv[0, 0, 0, -1]) == pytest.approx(0.0, abs=1e-8)

    def test_downward_flow_uses_shallower_donor(self, ocean_z_coord):
        """Downward flow should use shallower donor and zero-gradient at surface."""
        field = ocean_z_coord.z_full_ref[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        jac = jnp.ones((1, 1, 1))
        w_half = -jnp.ones((1, 1, 1, ocean_z_coord.n_levels + 1))

        adv = _vertical_advection_ocean(field, w_half, ocean_z_coord, jac)

        assert float(adv[0, 0, 0, 0]) == pytest.approx(0.0, abs=1e-8)
        assert jnp.allclose(adv[..., 1:], 1.0, atol=1e-6)


# ==============================================================================
# Model Tests
# ==============================================================================

class TestOceanModel:
    """Tests for the ocean model class."""

    def test_single_step(self, ocean_grid, ocean_z_coord, ocean_config, ocean_state):
        """A single model step should run without error."""
        model = OceanModel(ocean_grid, ocean_z_coord, ocean_config)
        state_new = model.step(ocean_state, 3600.0)
        assert jnp.all(jnp.isfinite(state_new.u.data))
        assert jnp.all(jnp.isfinite(state_new.eta.data))

    def test_shapes_preserved(self, ocean_grid, ocean_z_coord, ocean_config, ocean_state):
        """Output shapes should match input shapes."""
        model = OceanModel(ocean_grid, ocean_z_coord, ocean_config)
        state_new = model.step(ocean_state, 3600.0)
        assert state_new.u.shape == ocean_state.u.shape
        assert state_new.eta.shape == ocean_state.eta.shape
        assert state_new.T.shape == ocean_state.T.shape

    def test_static_fields_unchanged(self, ocean_grid, ocean_z_coord, ocean_config, ocean_state):
        """H_bathy and land_mask should not change."""
        model = OceanModel(ocean_grid, ocean_z_coord, ocean_config)
        state_new = model.step(ocean_state, 3600.0)
        assert jnp.allclose(state_new.H_bathy.data, ocean_state.H_bathy.data)
        assert jnp.allclose(state_new.land_mask.data, ocean_state.land_mask.data)

    def test_multi_step_stability(self, ocean_grid, ocean_z_coord, ocean_state):
        """10-step integration should remain bounded."""
        config = OceanConfig(
            A_h=1e4, K_h=1e3, A_v=1e-3, K_v=1e-4,
            n_barotropic_substeps=10,
            use_conservation_fixer=False,
        )
        model = OceanModel(ocean_grid, ocean_z_coord, config)
        state = ocean_state
        for _ in range(10):
            state = model.step(state, 3600.0)
        mask = state.land_mask.data[..., jnp.newaxis]
        T_ocean = state.T.data * mask
        # Temperature should stay in reasonable bounds
        T_max = float(jnp.max(jnp.where(mask > 0.5, T_ocean, -999)))
        T_min = float(jnp.min(jnp.where(mask > 0.5, T_ocean, 999)))
        assert T_min > -5.0, f"T_min = {T_min}"
        assert T_max < 40.0, f"T_max = {T_max}"


# ==============================================================================
# Conservation Tests
# ==============================================================================

class TestOceanConservation:
    """Tests for ocean conservation fixers."""

    def test_volume_conservation(self, ocean_grid, ocean_z_coord, ocean_state):
        """Volume fixer should restore eta integral."""
        # Perturb eta
        state_perturbed = ocean_state._replace(
            eta=ocean_state.eta.replace(
                data=ocean_state.eta.data + 0.01 * jnp.ones_like(ocean_state.eta.data),
            ),
        )
        state_fixed = fix_volume_ocean(state_perturbed, ocean_state, ocean_grid)

        mask = ocean_state.land_mask.data
        ocean_area = float(jnp.sum(mask * ocean_grid.area))
        vol_old = float(jnp.sum(ocean_state.eta.data * mask * ocean_grid.area))
        vol_fixed = float(jnp.sum(state_fixed.eta.data * mask * ocean_grid.area))
        # Check mean eta error is within float32 precision
        mean_eta_err = abs(vol_fixed - vol_old) / max(ocean_area, 1.0)
        assert mean_eta_err < 1e-6


# ==============================================================================
# Differentiability Tests
# ==============================================================================

class TestOceanDifferentiability:
    """Tests for JAX differentiability through ocean model."""

    def test_grad_through_tendencies(self, ocean_state, ocean_grid, ocean_z_coord, ocean_config):
        """jax.grad should work through tendency computation."""
        def loss_fn(eta_data):
            state = ocean_state._replace(
                eta=ocean_state.eta.replace(data=eta_data),
            )
            tend = ocean_baroclinic_tendencies(
                state, ocean_grid, ocean_z_coord, ocean_config,
            )
            return jnp.sum(tend.du_dt.data**2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(ocean_state.eta.data)
        assert jnp.all(jnp.isfinite(grad))


# ==============================================================================
# Hydrostatic Pressure Tests
# ==============================================================================

class TestHydrostaticPressure:
    """Tests for hydrostatic pressure computation."""

    def test_pressure_increases_with_depth(self, ocean_z_coord):
        """Pressure should increase with depth."""
        nlev = ocean_z_coord.n_levels
        rho = jnp.full((1, 1, 1, nlev), 1025.0)
        eta = jnp.zeros((1, 1, 1))
        J = jnp.ones((1, 1, 1))
        p = compute_hydrostatic_pressure(
            rho, eta, ocean_z_coord.dz_ref, J,
        )
        # Pressure should increase monotonically with depth (increasing k)
        for k in range(nlev - 1):
            assert float(p[0, 0, 0, k]) < float(p[0, 0, 0, k + 1])

    def test_surface_pressure_from_eta(self, ocean_z_coord):
        """Positive eta should increase all pressures."""
        nlev = ocean_z_coord.n_levels
        rho = jnp.full((1, 1, 1, nlev), 1025.0)
        J = jnp.ones((1, 1, 1))
        p_zero = compute_hydrostatic_pressure(
            rho, jnp.zeros((1, 1, 1)), ocean_z_coord.dz_ref, J,
        )
        p_pos = compute_hydrostatic_pressure(
            rho, jnp.ones((1, 1, 1)), ocean_z_coord.dz_ref, J,
        )
        assert jnp.all(p_pos > p_zero)


# ==============================================================================
# Buoyancy Frequency Tests
# ==============================================================================

class TestBuoyancyFrequency:
    """Tests for Brunt-Vaisala frequency."""

    def test_stable_stratification_positive_N2(self, ocean_z_coord):
        """Stable stratification (lighter on top) should give positive N^2."""
        nlev = ocean_z_coord.n_levels
        # Density increasing with depth (stable)
        rho = jnp.linspace(1023.0, 1028.0, nlev)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        J = jnp.ones((1, 1, 1))
        N2 = compute_buoyancy_frequency(
            rho, ocean_z_coord.dz_ref, J,
        )
        assert jnp.all(N2 > 0)
