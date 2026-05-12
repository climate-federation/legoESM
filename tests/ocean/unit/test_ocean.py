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

import warnings

import pytest
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.eos import (
    wright_eos,
    linear_eos,
    make_eos_fn,
    LinearEOSConfig,
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
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
    ocean_baroclinic_tendencies_cdgrid,
    _vertical_advection_ocean,
)
from legoesm.ocean.physics.mixing import vertical_diffusion
from legoesm.ocean.dynamics.barotropic import barotropic_substeps
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
def ocean_cdgrid(ocean_grid):
    """CDGrid for tendency computation."""
    return create_cubed_sphere_cdgrid(ocean_grid)


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
            jnp.array(10.0), jnp.array(35.0), jnp.array(0.0), rho_ref=constants.rho_ocean,
        )
        rho = wright_eos(jnp.array(10.0), jnp.array(35.0), jnp.array(0.0))
        assert float(rho_p) == pytest.approx(float(rho) - constants.rho_ocean, abs=1e-6)

    def test_jit_compatible(self):
        """EOS should work under jax.jit."""
        f = jax.jit(wright_eos)
        rho = f(jnp.array(10.0), jnp.array(35.0), jnp.array(0.0))
        assert jnp.isfinite(rho)

    def test_grad_interior_nonzero(self):
        """Gradient of EOS w.r.t. T should be nonzero inside valid range."""
        def rho_of_T(T):
            return wright_eos(T, jnp.array(35.0), jnp.array(0.0))
        g = jax.grad(rho_of_T)(jnp.array(10.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_grad_finite_nonzero_outside_valid_range(self):
        """Gradient must be finite and nonzero outside [-2, 40] degC.

        Wright (1997) extrapolates smoothly outside its nominal validity
        box. The EOS must not clip inputs — silent clipping zeros grads at
        the boundary and masks upstream state bugs. See issue #165.
        """
        def rho_of_T(T):
            return wright_eos(T, jnp.array(35.0), jnp.array(0.0))
        # Well above the nominal valid range
        g_hot = jax.grad(rho_of_T)(jnp.array(45.0))
        assert jnp.isfinite(g_hot)
        assert float(g_hot) != 0.0, f"Expected nonzero grad at T=45C, got {float(g_hot)}"
        # Well below the nominal valid range
        g_cold = jax.grad(rho_of_T)(jnp.array(-5.0))
        assert jnp.isfinite(g_cold)
        assert float(g_cold) != 0.0, f"Expected nonzero grad at T=-5C, got {float(g_cold)}"

        # And same story for S outside [0, 42] PSU.
        def rho_of_S(S):
            return wright_eos(jnp.array(10.0), S, jnp.array(0.0))
        g_fresh = jax.grad(rho_of_S)(jnp.array(-1.0))
        assert jnp.isfinite(g_fresh) and float(g_fresh) != 0.0
        g_brine = jax.grad(rho_of_S)(jnp.array(45.0))
        assert jnp.isfinite(g_brine) and float(g_brine) != 0.0

    def test_density_finite_outside_valid_range(self):
        """Density itself must also be finite outside the nominal box."""
        # Mild overshoot (advection / diffusion style)
        rho_mild = wright_eos(jnp.array(-3.0), jnp.array(43.0), jnp.array(0.0))
        assert jnp.isfinite(rho_mild)
        assert 900.0 < float(rho_mild) < 1100.0
        # Larger excursion — should still be finite, may be unphysical.
        rho_wild = wright_eos(jnp.array(50.0), jnp.array(-2.0), jnp.array(0.0))
        assert jnp.isfinite(rho_wild)


class TestLinearEOS:
    """Tests for the linear equation of state."""

    def test_reference_density(self):
        """At T=T_ref, S=S_ref, density should be rho_ref."""
        rho = linear_eos(
            jnp.array(10.0), jnp.array(35.0), jnp.array(0.0),
            rho_ref=constants.rho_ocean, T_ref=10.0, S_ref=35.0,
        )
        assert float(rho) == pytest.approx(constants.rho_ocean)

    def test_warm_water_lighter(self):
        """Warmer water should be less dense (positive alpha_T)."""
        rho_warm = linear_eos(jnp.array(25.0), jnp.array(35.0), jnp.array(0.0))
        rho_cold = linear_eos(jnp.array(5.0), jnp.array(35.0), jnp.array(0.0))
        assert float(rho_warm) < float(rho_cold)

    def test_salty_water_heavier(self):
        """Saltier water should be more dense (positive beta_S)."""
        rho_salty = linear_eos(jnp.array(10.0), jnp.array(38.0), jnp.array(0.0))
        rho_fresh = linear_eos(jnp.array(10.0), jnp.array(32.0), jnp.array(0.0))
        assert float(rho_salty) > float(rho_fresh)

    def test_pressure_independent(self):
        """Linear EOS should not depend on pressure."""
        rho_sfc = linear_eos(jnp.array(10.0), jnp.array(35.0), jnp.array(0.0))
        rho_deep = linear_eos(jnp.array(10.0), jnp.array(35.0), jnp.array(5e7))
        assert float(rho_sfc) == pytest.approx(float(rho_deep))

    def test_analytical_value(self):
        """Check against hand-computed value."""
        # rho = 1025 * (1 - 2e-4*(20-10) + 7.4e-4*(36-35))
        #     = 1025 * (1 - 0.002 + 0.00074) = 1025 * 0.99874 = 1023.7085
        rho = linear_eos(
            jnp.array(20.0), jnp.array(36.0), jnp.array(0.0),
            rho_ref=constants.rho_ocean, alpha_T=2e-4, beta_S=7.4e-4,
            T_ref=10.0, S_ref=35.0,
        )
        assert float(rho) == pytest.approx(constants.rho_ocean * 0.99874, rel=1e-6)

    def test_vectorized(self):
        """Linear EOS should work with array inputs."""
        T = jnp.array([5.0, 10.0, 20.0, 25.0])
        S = jnp.full(4, 35.0)
        p = jnp.zeros(4)
        rho = linear_eos(T, S, p)
        assert rho.shape == (4,)
        assert jnp.all(jnp.diff(rho) < 0)

    def test_jit_and_grad(self):
        """Linear EOS should be JIT-able and differentiable."""
        f = jax.jit(lambda T: linear_eos(T, jnp.array(35.0), jnp.array(0.0)))
        rho = f(jnp.array(10.0))
        assert jnp.isfinite(rho)

        g = jax.grad(lambda T: linear_eos(T, jnp.array(35.0), jnp.array(0.0)))
        drho_dT = g(jnp.array(10.0))
        # drho/dT = -rho_ref * alpha_T = -1025 * 2e-4 = -0.205
        assert float(drho_dT) == pytest.approx(-constants.rho_ocean * 2e-4, rel=1e-6)


class TestMakeEosFn:
    """Tests for the EOS dispatcher."""

    def test_wright_returns_wright(self):
        """make_eos_fn('wright') should return wright_eos."""
        fn = make_eos_fn("wright")
        assert fn is wright_eos

    def test_linear_returns_callable(self):
        """make_eos_fn('linear') should return a callable."""
        fn = make_eos_fn("linear")
        rho = fn(jnp.array(10.0), jnp.array(35.0), jnp.array(0.0))
        assert jnp.isfinite(rho)

    def test_linear_with_config(self):
        """make_eos_fn('linear', cfg) should use config values."""
        cfg = LinearEOSConfig(rho_ref=1000.0, alpha_T=1e-4, T_ref=0.0)
        fn = make_eos_fn("linear", cfg)
        # rho = 1000 * (1 - 1e-4 * (10 - 0)) = 1000 * 0.999 = 999
        rho = fn(jnp.array(10.0), jnp.array(35.0), jnp.array(0.0))
        assert float(rho) == pytest.approx(999.0 + 1000.0 * 7.4e-4 * (35.0 - 35.0), rel=1e-6)

    def test_unknown_raises(self):
        """Unknown EOS should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown EOS"):
            make_eos_fn("cubic")


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

    def test_jacobian_respects_min_water_column(self, ocean_z_coord):
        """Jacobian helper should clip collapsed columns when requested."""
        H_bathy = jnp.array(4000.0, dtype=jnp.float32)
        eta = jnp.array(-3999.9, dtype=jnp.float32)
        J = compute_ocean_jacobian(
            eta,
            H_bathy,
            ocean_z_coord,
            min_water_column_m=0.5,
        )
        assert float(J) == pytest.approx(0.5 / ocean_z_coord.H_max, rel=1e-6)

    def test_layer_thickness_respects_min_water_column(self, ocean_z_coord):
        """Layer thickness helper should not return non-positive layers."""
        H_bathy = jnp.full((1, 1, 1), 4000.0, dtype=jnp.float32)
        eta = jnp.full((1, 1, 1), -3999.9, dtype=jnp.float32)
        h_k = compute_layer_thickness(
            eta,
            H_bathy,
            ocean_z_coord,
            min_water_column_m=0.5,
        )
        assert float(jnp.min(h_k)) > 0.0

    def test_input_validation(self):
        """Coordinate constructor should reject invalid values."""
        with pytest.raises(ValueError):
            create_ocean_z_star(n_levels=0)
        with pytest.raises(ValueError):
            create_ocean_z_star(H_max=0.0)
        with pytest.raises(ValueError):
            create_ocean_z_star(dz_surface=0.0)
        with pytest.raises(ValueError):
            create_ocean_z_star(dz_deep=0.0)

    def test_nlev1_creation(self):
        """nlev=1 should produce valid coordinate for barotropic experiments."""
        z = create_ocean_z_star(n_levels=1, H_max=5000.0)
        assert z.n_levels == 1
        assert z.dz_ref.shape == (1,)
        assert float(z.dz_ref[0]) == pytest.approx(5000.0)
        assert z.dz_half_ref.shape == (0,)
        assert z.z_full_ref.shape == (1,)
        assert z.z_half_ref.shape == (2,)
        assert float(z.z_half_ref[0]) == pytest.approx(0.0)
        assert float(z.z_half_ref[1]) == pytest.approx(-5000.0)

    def test_nlev1_model_step(self):
        """Ocean model should be able to step with nlev=1 (barotropic)."""
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.state import OceanConfig
        from legoesm.ocean.init import rest_state_ocean

        grid = create_cubed_sphere(4)
        z = create_ocean_z_star(n_levels=1, H_max=5000.0)
        config = OceanConfig(
            A_v=0.0, K_v=0.0, n_barotropic_substeps=5,
        )
        model = OceanModel(grid, z, config)
        state = rest_state_ocean(grid, z, H_max=5000.0)
        state_new = model.step(state, 60.0)
        assert jnp.all(jnp.isfinite(state_new.u.data))
        assert jnp.all(jnp.isfinite(state_new.T.data))
        assert jnp.all(jnp.isfinite(state_new.eta.data))
        assert state_new.u.data.shape[-1] == 1


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

    def test_rest_state_small_tendencies(self, ocean_state, ocean_grid, ocean_cdgrid, ocean_z_coord, ocean_config):
        """Rest state should produce near-zero tendencies."""
        tend = ocean_baroclinic_tendencies_cdgrid(
            ocean_state, ocean_grid, ocean_z_coord, ocean_cdgrid, ocean_config,
        )
        # Velocity tendencies should be small (Coriolis on zero velocity = 0)
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1e-3
        assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1e-3
        # Eta tendency should be small
        assert float(jnp.max(jnp.abs(tend.deta_dt.data))) < 1e-3

    def test_tendencies_finite(self, ocean_state, ocean_grid, ocean_cdgrid, ocean_z_coord, ocean_config):
        """All tendencies should be finite."""
        tend = ocean_baroclinic_tendencies_cdgrid(
            ocean_state, ocean_grid, ocean_z_coord, ocean_cdgrid, ocean_config,
        )
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
        assert jnp.all(jnp.isfinite(tend.deta_dt.data))

    def test_tendencies_finite_for_thin_columns(self, ocean_state, ocean_grid, ocean_cdgrid, ocean_z_coord):
        """Tendency path should remain finite when eta approaches dry columns."""
        state_thin = ocean_state._replace(
            eta=ocean_state.eta.replace(
                data=-ocean_state.H_bathy.data + 0.1,
            ),
        )
        config = OceanConfig(
            A_h=1e3,
            K_h=1e2,
            A_v=1e-3,
            K_v=1e-4,
            n_barotropic_substeps=10,
            hyperdiff_coeff=0.0,
            min_water_column_m=0.5,
        )
        tend = ocean_baroclinic_tendencies_cdgrid(
            state_thin, ocean_grid, ocean_z_coord, ocean_cdgrid, config,
        )
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
        assert jnp.all(jnp.isfinite(tend.deta_dt.data))

    def test_static_fields_zero_tendency(self, ocean_state, ocean_grid, ocean_cdgrid, ocean_z_coord, ocean_config):
        """H_bathy and land_mask tendencies should be exactly zero."""
        tend = ocean_baroclinic_tendencies_cdgrid(
            ocean_state, ocean_grid, ocean_z_coord, ocean_cdgrid, ocean_config,
        )
        assert float(jnp.max(jnp.abs(tend.dH_bathy_dt.data))) == 0.0
        assert float(jnp.max(jnp.abs(tend.dland_mask_dt.data))) == 0.0

    def test_land_masking(self, ocean_grid, ocean_cdgrid, ocean_z_coord, ocean_config):
        """Tendencies should be exactly zero on land cells."""
        state = rest_state_ocean(
            ocean_grid, ocean_z_coord, H_max=4000.0, land_lat_threshold=60.0,
        )
        tend = ocean_baroclinic_tendencies_cdgrid(
            state, ocean_grid, ocean_z_coord, ocean_cdgrid, ocean_config,
        )
        land = state.land_mask.data < 0.5
        # On land cells, all tendencies should be zero
        assert float(jnp.max(jnp.abs(tend.du_dt.data[land]))) == 0.0
        assert float(jnp.max(jnp.abs(tend.dv_dt.data[land]))) == 0.0
        assert float(jnp.max(jnp.abs(tend.deta_dt.data[land]))) == 0.0

    def test_coriolis_applies_to_baroclinic_shear(self, ocean_grid, ocean_cdgrid, ocean_z_coord):
        """FV tendencies should include planetary Coriolis on shear flow."""
        state = rest_state_ocean(
            ocean_grid, ocean_z_coord, H_max=4000.0, land_lat_threshold=90.0,
            T_surface=15.0, T_deep=15.0,
        )
        nlev = ocean_z_coord.n_levels
        shear_profile = jnp.linspace(-1.0, 1.0, nlev, dtype=jnp.float32)
        v_shear = jnp.broadcast_to(
            shear_profile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
            state.v.data.shape,
        )
        state = state._replace(
            u=state.u.replace(data=jnp.zeros_like(state.u.data)),
            v=state.v.replace(data=v_shear),
        )
        config = OceanConfig(
            A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0,
            hyperdiff_coeff=0.0,
        )
        tend = ocean_baroclinic_tendencies_cdgrid(state, ocean_grid, ocean_z_coord, ocean_cdgrid, config)
        # Exclude equatorial points where f=0 and bottom level where upwind BC can zero tendency.
        off_equator = jnp.abs(ocean_grid.f) > 1.0e-8
        shear_not_zero = jnp.max(jnp.abs(v_shear), axis=-1) > 1.0e-6
        active = off_equator & shear_not_zero
        assert float(jnp.max(jnp.abs(tend.du_dt.data[active]))) > 0.0


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

    def test_vertical_advection_jacobian_scaling(self, ocean_z_coord):
        """Advection magnitude should use physical w and physical dz."""
        J = 2.0
        jac = jnp.full((1, 1, 1), J)
        # Field linear in physical depth z = J * z*.
        field = (J * ocean_z_coord.z_full_ref)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        w_half = jnp.ones((1, 1, 1, ocean_z_coord.n_levels + 1))

        adv = _vertical_advection_ocean(field, w_half, ocean_z_coord, jac)
        assert jnp.allclose(adv[..., :-1], -1.0, atol=1e-6)
        assert float(adv[0, 0, 0, -1]) == pytest.approx(0.0, abs=1e-8)


class TestFluxFormVerticalMomentumAdvection:
    """Tests for ``flux_form_vertical_momentum_advection`` (issue #171)."""

    def _helper(self):
        from legoesm.ocean.vertical import (
            flux_form_vertical_momentum_advection,
            flux_form_vertical_tracer_advection,
        )
        return flux_form_vertical_momentum_advection, flux_form_vertical_tracer_advection

    def test_rest_state_zero(self):
        """u=0 gives zero tendency trivially."""
        helper, _ = self._helper()
        nlev = 5
        u = jnp.zeros(nlev)
        w_half = jnp.concatenate([jnp.array([0.0]), jnp.array([0.1, -0.2, 0.05, -0.1]), jnp.array([0.0])])
        h_u = jnp.full((nlev,), 100.0)

        tendency = helper(u, w_half, h_u)
        assert jnp.allclose(tendency, 0.0, atol=1e-15)

    def test_zero_w_zero_tendency(self):
        """w_half = 0 gives zero tendency for any u."""
        helper, _ = self._helper()
        nlev = 5
        u = jnp.array([1.0, 2.0, -0.5, 0.3, -1.2])
        w_half = jnp.zeros(nlev + 1)
        h_u = jnp.full((nlev,), 50.0)

        tendency = helper(u, w_half, h_u)
        assert jnp.allclose(tendency, 0.0, atol=1e-15)

    def test_column_momentum_conserved_in_closed_column(self):
        """h-weighted column integral of tendency is zero to round-off.

        For any u and any w_half with w_half[0] = w_half[nlev] = 0
        (closed column), the flux-form tendency must satisfy
            sum_k(tendency[k] * h_u[k]) == 0
        exactly. This is the column momentum flux balance property
        that the old cell-upwind form did not have.
        """
        helper, _ = self._helper()
        nlev = 6
        u = jnp.array([0.5, 1.2, -0.3, 0.8, -0.4, 0.1])
        # Non-trivial w profile with zero boundary values.
        w_half = jnp.array([0.0, 0.15, -0.05, 0.1, -0.08, 0.02, 0.0])
        h_u = jnp.array([10.0, 20.0, 30.0, 40.0, 30.0, 20.0])

        tendency = helper(u, w_half, h_u)
        column_integral = float(jnp.sum(tendency * h_u))
        assert abs(column_integral) < 1e-12, (
            f"Column momentum flux balance violated: sum(tend*h) = {column_integral}"
        )

    def test_matches_tracer_flux_form_divided_by_h(self):
        """Identity: tendency = -flux_form_tracer(u, w) / h_u."""
        helper, tracer_helper = self._helper()
        nlev = 5
        u = jnp.array([1.0, 2.0, 3.0, 2.5, 1.5])
        w_half = jnp.array([0.0, 0.1, -0.05, 0.08, -0.02, 0.0])
        h_u = jnp.array([15.0, 25.0, 35.0, 25.0, 15.0])

        tendency = helper(u, w_half, h_u)
        tracer_flux_div = tracer_helper(u, w_half)
        expected = -tracer_flux_div / h_u
        assert jnp.allclose(tendency, expected, atol=1e-14)

    def test_interface_upwind_upward_picks_below(self):
        """Upward flow at an interior interface picks the below cell."""
        helper, _ = self._helper()
        # 3 levels: u = [10, 20, 30]. All boundaries closed except
        # interface 1 is upward (w_half[1] = +1 m/s). h_u = 1 everywhere.
        u = jnp.array([10.0, 20.0, 30.0])
        w_half = jnp.array([0.0, 1.0, 0.0, 0.0])
        h_u = jnp.array([1.0, 1.0, 1.0])

        tendency = helper(u, w_half, h_u)
        # F[0]=0, F[1] = w[1] * u_below = 1 * 20 = 20 (upward → from below = u[1])
        # F[2] = 0, F[3] = 0
        # vert_flux_div = [F[0]-F[1], F[1]-F[2], F[2]-F[3]] = [-20, 20, 0]
        # tendency = -vert_flux_div / h_u = [20, -20, 0]
        assert float(tendency[0]) == pytest.approx(20.0, abs=1e-10)
        assert float(tendency[1]) == pytest.approx(-20.0, abs=1e-10)
        assert float(tendency[2]) == pytest.approx(0.0, abs=1e-10)

    def test_interface_upwind_downward_picks_above(self):
        """Downward flow at an interior interface picks the above cell."""
        helper, _ = self._helper()
        # 3 levels: u = [10, 20, 30]. Interface 2 is downward (w_half[2] = -1).
        u = jnp.array([10.0, 20.0, 30.0])
        w_half = jnp.array([0.0, 0.0, -1.0, 0.0])
        h_u = jnp.array([1.0, 1.0, 1.0])

        tendency = helper(u, w_half, h_u)
        # F[0]=0, F[1]=0, F[2] = w[2] * u_above = -1 * 20 = -20
        # F[3]=0. vert_flux_div = [0, -(-20), -20] = [0, 20, -20]
        # tendency = [0, -20, 20]
        assert float(tendency[0]) == pytest.approx(0.0, abs=1e-10)
        assert float(tendency[1]) == pytest.approx(-20.0, abs=1e-10)
        assert float(tendency[2]) == pytest.approx(20.0, abs=1e-10)

    def test_constant_u_column_integral_zero(self):
        """For spatially-constant u, column momentum flux balance is zero
        even though per-layer tendencies are not.

        Flux-form under dynamic h preserves ``sum(h*u)`` exactly when the
        boundary fluxes vanish — per-layer u values can shift, but the
        vertically integrated momentum is conserved.
        """
        helper, _ = self._helper()
        nlev = 5
        u = jnp.full((nlev,), 1.5)  # constant
        w_half = jnp.array([0.0, 0.3, -0.1, 0.2, -0.05, 0.0])
        h_u = jnp.array([10.0, 20.0, 30.0, 20.0, 10.0])

        tendency = helper(u, w_half, h_u)
        column_integral = float(jnp.sum(tendency * h_u))
        assert abs(column_integral) < 1e-12

    def test_boundary_not_hard_zeroed(self):
        """Top and bottom levels receive nonzero tendency from adjacent
        interface flux, unlike the cell-upwind gradient form which
        forced ``grad = 0`` at k=0 and k=nlev-1.

        With interior upward w at interface 1 only, the surface level
        (k=0) and level k=1 both see the flux; other levels don't.
        """
        helper, _ = self._helper()
        u = jnp.array([1.0, 2.0, 3.0, 4.0])
        w_half = jnp.array([0.0, 0.5, 0.0, 0.0, 0.0])
        h_u = jnp.full((4,), 1.0)

        tendency = helper(u, w_half, h_u)
        # F[1] = 0.5 * u[1] = 1 (upward → below). All other F = 0.
        # vert_flux_div = [0-1, 1-0, 0-0, 0-0] = [-1, 1, 0, 0]
        # tendency = [1, -1, 0, 0]
        assert float(tendency[0]) == pytest.approx(1.0, abs=1e-10)
        assert float(tendency[1]) == pytest.approx(-1.0, abs=1e-10)
        # The surface cell's tendency is NOT forced to zero.
        assert float(tendency[0]) != 0.0


class TestVerticalMixing:
    """Tests for vertical diffusion operator."""

    def test_vertical_diffusion_no_scatter_dtype_warning(self, ocean_z_coord):
        """vertical_diffusion should avoid mixed-dtype scatter updates."""
        nlev = ocean_z_coord.n_levels
        field = jnp.linspace(0.0, 1.0, nlev, dtype=jnp.float32)[
            jnp.newaxis, jnp.newaxis, jnp.newaxis, :
        ]
        jac_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        jac = jnp.ones((1, 1, 1), dtype=jac_dtype)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=".*scatter inputs have incompatible types.*",
                category=FutureWarning,
            )
            tendency = vertical_diffusion(field, ocean_z_coord, jac, coeff=1.0e-4)

        assert tendency.dtype == field.dtype
        assert jnp.all(jnp.isfinite(tendency))


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

    def test_barotropic_substeps_enforce_min_water_column(
        self, ocean_grid, ocean_z_coord, ocean_state,
    ):
        """Barotropic mode should floor eta so wet columns stay positive."""
        config = OceanConfig(
            n_barotropic_substeps=2,
            min_water_column_m=0.5,
            use_conservation_fixer=False,
        )
        eta_bad = -ocean_state.H_bathy.data + 0.1
        state_bad = ocean_state._replace(
            eta=ocean_state.eta.replace(data=eta_bad),
        )
        state_new = barotropic_substeps(
            state_bad,
            dt_s=60.0,
            n_substeps=config.n_barotropic_substeps,
            grid=ocean_grid,
            z_coord=ocean_z_coord,
            config=config,
        )
        wet = state_new.land_mask.data > 0.5
        water_col = state_new.eta.data + state_new.H_bathy.data
        min_wet = float(jnp.min(jnp.where(wet, water_col, jnp.inf)))
        assert min_wet >= config.min_water_column_m - 1.0e-6

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"A_h": -1.0}, "A_h"),
            ({"K_h": -1.0}, "K_h"),
            ({"A_v": -1.0}, "A_v"),
            ({"K_v": -1.0}, "K_v"),
            ({"hyperdiff_coeff": -1.0}, "hyperdiff_coeff"),
            ({"barotropic_diffusion_alpha": -1.0}, "barotropic_diffusion_alpha"),
            ({"n_barotropic_substeps": 0}, "n_barotropic_substeps"),
            ({"barotropic_diffusion_dt_ref": 0.0}, "barotropic_diffusion_dt_ref"),
            ({"min_water_column_m": 0.0}, "min_water_column_m"),
            ({"max_abs_eta_m": 0.0}, "max_abs_eta_m"),
            (
                {"temperature_min_c": 10.0, "temperature_max_c": 0.0},
                "temperature_min_c",
            ),
            (
                {"salinity_min_psu": 40.0, "salinity_max_psu": 30.0},
                "salinity_min_psu",
            ),
        ],
    )
    def test_invalid_config_rejected(self, ocean_grid, ocean_z_coord, kwargs, match):
        """OceanModel should fail fast on invalid config values."""
        with pytest.raises(ValueError, match=match):
            OceanModel(ocean_grid, ocean_z_coord, OceanConfig(**kwargs))

    def test_step_checked_with_runtime_checks(self, ocean_grid, ocean_z_coord, ocean_state):
        """Runtime-checked step should run and return finite state."""
        config = OceanConfig(
            n_barotropic_substeps=10,
            enable_runtime_checks=True,
            max_abs_eta_m=1.0e5,
            temperature_min_c=-10.0,
            temperature_max_c=50.0,
            salinity_min_psu=-1.0,
            salinity_max_psu=60.0,
        )
        model = OceanModel(ocean_grid, ocean_z_coord, config)
        state_new = model.step_checked(ocean_state, 3600.0)
        assert jnp.all(jnp.isfinite(state_new.u.data))
        assert jnp.all(jnp.isfinite(state_new.eta.data))

    def test_runtime_check_rejects_too_thin_water_column(self, ocean_grid, ocean_z_coord, ocean_state):
        """Runtime checks should reject eta that collapses water column."""
        config = OceanConfig(enable_runtime_checks=True, min_water_column_m=0.5)
        model = OceanModel(ocean_grid, ocean_z_coord, config)
        eta_bad = -ocean_state.H_bathy.data + 0.1
        state_bad = ocean_state._replace(
            eta=ocean_state.eta.replace(data=eta_bad),
        )
        with pytest.raises(ValueError, match="water column"):
            model._assert_runtime_invariants(state_bad)

    def test_integrate_scan_rejects_runtime_checks(self, ocean_grid, ocean_z_coord, ocean_state):
        """integrate_scan should reject host runtime checks."""
        config = OceanConfig(enable_runtime_checks=True)
        model = OceanModel(ocean_grid, ocean_z_coord, config)
        with pytest.raises(ValueError, match="integrate_scan"):
            model.integrate_scan(ocean_state, n_steps=1, dt=3600.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"duration": -1.0, "dt": 3600.0, "save_every": 1}, "duration"),
            ({"duration": 3600.0, "dt": 0.0, "save_every": 1}, "dt"),
            ({"duration": 3600.0, "dt": 3600.0, "save_every": 0}, "save_every"),
            ({"duration": 1.0, "dt": 3600.0, "save_every": 1}, "zero steps"),
        ],
    )
    def test_integrate_validates_inputs(
        self, ocean_grid, ocean_z_coord, ocean_state, kwargs, match,
    ):
        """integrate should reject invalid host-side control parameters."""
        model = OceanModel(ocean_grid, ocean_z_coord, OceanConfig())
        with pytest.raises(ValueError, match=match):
            model.integrate(ocean_state, **kwargs)

    @pytest.mark.parametrize(
        ("n_steps", "dt", "match"),
        [
            (-1, 3600.0, "n_steps"),
            (1, 0.0, "dt"),
        ],
    )
    def test_integrate_scan_validates_inputs(
        self, ocean_grid, ocean_z_coord, ocean_state, n_steps, dt, match,
    ):
        """integrate_scan should reject invalid host-side control parameters."""
        model = OceanModel(ocean_grid, ocean_z_coord, OceanConfig())
        with pytest.raises(ValueError, match=match):
            model.integrate_scan(ocean_state, n_steps=n_steps, dt=dt)


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

    def test_volume_fixer_respects_min_water_column(self, ocean_grid, ocean_state):
        """Volume fixer should enforce optional wet-column lower bound."""
        state_thin = ocean_state._replace(
            eta=ocean_state.eta.replace(
                data=-ocean_state.H_bathy.data + 0.1,
            ),
        )
        state_fixed = fix_volume_ocean(
            state_thin,
            ocean_state,
            ocean_grid,
            min_water_column_m=0.5,
        )
        wet = state_fixed.land_mask.data > 0.5
        water_col = state_fixed.eta.data + state_fixed.H_bathy.data
        min_wet = float(jnp.min(jnp.where(wet, water_col, jnp.inf)))
        assert min_wet >= 0.5 - 1.0e-6

    def test_heat_salt_fixers_finite_for_thin_columns(
        self, ocean_grid, ocean_z_coord, ocean_state,
    ):
        """Heat/salt fixers should remain finite with thin-column clipping."""
        state_thin = ocean_state._replace(
            eta=ocean_state.eta.replace(
                data=-ocean_state.H_bathy.data + 0.1,
            ),
        )
        state_heat = fix_heat_ocean(
            state_thin,
            ocean_state,
            ocean_grid,
            ocean_z_coord,
            min_water_column_m=0.5,
        )
        state_salt = fix_salt_ocean(
            state_heat,
            ocean_state,
            ocean_grid,
            ocean_z_coord,
            min_water_column_m=0.5,
        )
        assert jnp.all(jnp.isfinite(state_heat.T.data))
        assert jnp.all(jnp.isfinite(state_salt.S.data))


# ==============================================================================
# Differentiability Tests
# ==============================================================================

class TestOceanDifferentiability:
    """Tests for JAX differentiability through ocean model."""

    def test_grad_through_tendencies(self, ocean_state, ocean_grid, ocean_cdgrid, ocean_z_coord, ocean_config):
        """jax.grad should work through tendency computation."""
        def loss_fn(eta_data):
            state = ocean_state._replace(
                eta=ocean_state.eta.replace(data=eta_data),
            )
            tend = ocean_baroclinic_tendencies_cdgrid(
                state, ocean_grid, ocean_z_coord, ocean_cdgrid, ocean_config,
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
        rho = jnp.full((1, 1, 1, nlev), constants.rho_ocean)
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
        rho = jnp.full((1, 1, 1, nlev), constants.rho_ocean)
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


# ==============================================================================
# Spectral Ocean Tests
# ==============================================================================

@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="Spectral ocean path requires JAX x64 support.",
)
class TestSpectralOcean:
    """Tests for spectral ocean model wiring and conservation."""

    @staticmethod
    def _cell_area(grid):
        dlon = 2.0 * jnp.pi / grid.n_lon
        return (grid.radius ** 2) * grid.weights[:, jnp.newaxis] * dlon

    def _invariants(self, state, grid, z_coord, min_water_column_m):
        from legoesm.grids.gaussian import sh_synthesis, sh_synthesis_3d

        mask = state.land_mask_grid.data
        area = self._cell_area(grid)
        weighted_area = area * mask

        H_bathy = sh_synthesis(grid, state.H_bathy_hat.data).real
        eta = sh_synthesis(grid, state.eta_hat.data).real * mask
        T = sh_synthesis_3d(grid, state.T_hat.data).real
        S = sh_synthesis_3d(grid, state.S_hat.data).real

        h_k = compute_layer_thickness(
            eta, H_bathy, z_coord, min_water_column_m=min_water_column_m,
        )
        ocean_area = jnp.sum(weighted_area)
        ocean_volume = jnp.sum(jnp.sum(h_k, axis=-1) * weighted_area)
        vol = jnp.sum(eta * weighted_area)
        heat = jnp.sum(jnp.sum(T * h_k, axis=-1) * weighted_area)
        salt = jnp.sum(jnp.sum(S * h_k, axis=-1) * weighted_area)
        return ocean_area, ocean_volume, vol, heat, salt

    def test_conservation_fixer_restores_volume_heat_salt(self):
        from legoesm.grids.gaussian import (
            create_gaussian_grid,
            sh_analysis,
            sh_analysis_3d,
            sh_synthesis,
            sh_synthesis_3d,
        )
        from legoesm.ocean.dynamics.spectral_ocean_pe import (
            SpectralOceanConfig,
            _spectral_conservation_fixer,
            rest_state_spectral_ocean,
        )

        grid = create_gaussian_grid(8, allow_unsupported_backend=True)
        z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
        state = rest_state_spectral_ocean(
            grid, z_coord, H_max=4000.0, land_lat_threshold=70.0,
        )
        cfg = SpectralOceanConfig(use_conservation_fixer=True, min_water_column_m=0.5)

        mask = state.land_mask_grid.data
        mask_3d = mask[..., jnp.newaxis]
        eta_grid = sh_synthesis(grid, state.eta_hat.data).real
        T_grid = sh_synthesis_3d(grid, state.T_hat.data).real
        S_grid = sh_synthesis_3d(grid, state.S_hat.data).real

        perturbed = state._replace(
            eta_hat=state.eta_hat.replace(data=sh_analysis(grid, eta_grid + 0.15 * mask)),
            T_hat=state.T_hat.replace(data=sh_analysis_3d(grid, T_grid + 1.2 * mask_3d)),
            S_hat=state.S_hat.replace(data=sh_analysis_3d(grid, S_grid - 0.35 * mask_3d)),
        )
        fixed = _spectral_conservation_fixer(
            perturbed, state, grid, z_coord, cfg,
        )

        area0, vol0, eta_int0, heat0, salt0 = self._invariants(
            state, grid, z_coord, cfg.min_water_column_m,
        )
        _, volp, eta_intp, heatp, saltp = self._invariants(
            perturbed, grid, z_coord, cfg.min_water_column_m,
        )
        area1, vol1, eta_int1, heat1, salt1 = self._invariants(
            fixed, grid, z_coord, cfg.min_water_column_m,
        )

        mean_eta_drift_before = float(jnp.abs(eta_intp - eta_int0) / jnp.maximum(area0, 1.0))
        mean_eta_drift = float(jnp.abs(eta_int1 - eta_int0) / jnp.maximum(area0, 1.0))
        heat_rel_before = float(jnp.abs(heatp - heat0) / jnp.maximum(jnp.abs(heat0), 1.0))
        heat_rel = float(jnp.abs(heat1 - heat0) / jnp.maximum(jnp.abs(heat0), 1.0))
        salt_rel_before = float(jnp.abs(saltp - salt0) / jnp.maximum(jnp.abs(salt0), 1.0))
        salt_rel = float(jnp.abs(salt1 - salt0) / jnp.maximum(jnp.abs(salt0), 1.0))
        vol_rel = float(jnp.abs(vol1 - vol0) / jnp.maximum(jnp.abs(vol0), 1.0))

        # The fixer should substantially reduce all global drifts (>95%).
        assert mean_eta_drift < mean_eta_drift_before * 0.05
        assert heat_rel < heat_rel_before * 0.05
        assert salt_rel < salt_rel_before * 0.05
        assert mean_eta_drift < 1e-2
        assert heat_rel < 5e-3
        assert salt_rel < 1e-4
        assert vol_rel < 5e-6
        assert vol1 < volp
        assert area1 > 0.0

    def test_integrate_validates_inputs(self):
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.ocean.dynamics.spectral_ocean_pe import (
            SpectralOceanConfig,
            SpectralOceanModel,
            rest_state_spectral_ocean,
        )

        grid = create_gaussian_grid(8, allow_unsupported_backend=True)
        z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
        state = rest_state_spectral_ocean(
            grid, z_coord, H_max=4000.0, land_lat_threshold=70.0,
        )
        model = SpectralOceanModel(
            grid, z_coord, SpectralOceanConfig(use_conservation_fixer=False),
            allow_unsupported_backend=True,
        )

        with pytest.raises(ValueError, match="dt"):
            model.integrate(state, duration=3600.0, dt=0.0)
        with pytest.raises(ValueError, match="duration"):
            model.integrate(state, duration=-1.0, dt=600.0)
        with pytest.raises(ValueError, match="save_every"):
            model.integrate(state, duration=3600.0, dt=600.0, save_every=0)
        with pytest.raises(ValueError, match="zero steps"):
            model.integrate(state, duration=1.0, dt=600.0)

    def test_rest_state_keeps_land_tracer_extension_smooth(self):
        from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis_3d
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean

        grid = create_gaussian_grid(8, allow_unsupported_backend=True)
        z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
        state = rest_state_spectral_ocean(
            grid, z_coord, H_max=4000.0, land_lat_threshold=60.0,
        )
        land = state.land_mask_grid.data < 0.5
        T_grid = sh_synthesis_3d(grid, state.T_hat.data).real
        S_grid = sh_synthesis_3d(grid, state.S_hat.data).real

        assert bool(jnp.any(land))
        assert float(jnp.max(jnp.abs(T_grid[land]))) > 0.0
        assert float(jnp.max(jnp.abs(S_grid[land]))) > 0.0

    def test_rest_state_validates_inputs(self):
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean

        grid = create_gaussian_grid(8, allow_unsupported_backend=True)
        z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)

        with pytest.raises(ValueError, match="H_max"):
            rest_state_spectral_ocean(grid, z_coord, H_max=0.0)
        with pytest.raises(ValueError, match="land_lat_threshold"):
            rest_state_spectral_ocean(grid, z_coord, land_lat_threshold=-1.0)
        with pytest.raises(ValueError, match="land_lat_threshold"):
            rest_state_spectral_ocean(grid, z_coord, land_lat_threshold=91.0)
        with pytest.raises(ValueError, match="T_surface"):
            rest_state_spectral_ocean(grid, z_coord, T_surface=float("inf"))

    def test_model_warns_for_ignored_barotropic_substeps(self):
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.ocean.dynamics.spectral_ocean_pe import (
            SpectralOceanConfig,
            SpectralOceanModel,
        )

        grid = create_gaussian_grid(8, allow_unsupported_backend=True)
        z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)

        with pytest.warns(RuntimeWarning, match="n_barotropic_substeps.*ignored"):
            SpectralOceanModel(
                grid,
                z_coord,
                SpectralOceanConfig(n_barotropic_substeps=4),
                allow_unsupported_backend=True,
            )

    def test_tendencies_finite_for_all_ocean_mask(self):
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.ocean.dynamics.spectral_ocean_pe import (
            SpectralOceanConfig,
            rest_state_spectral_ocean,
            spectral_ocean_tendencies,
        )

        grid = create_gaussian_grid(8, allow_unsupported_backend=True)
        z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
        state = rest_state_spectral_ocean(
            grid,
            z_coord,
            H_max=4000.0,
            land_lat_threshold=90.0,
        )
        config = SpectralOceanConfig(
            A_h=0.0,
            K_h=0.0,
            A_v=0.0,
            K_v=0.0,
            hyperdiff_coeff=0.0,
        )
        tend = spectral_ocean_tendencies(state, grid, z_coord, config)

        assert float(jnp.mean(state.land_mask_grid.data)) == pytest.approx(1.0, abs=1e-12)
        for leaf in jax.tree.leaves(tend):
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
                assert jnp.all(jnp.isfinite(leaf))


# ==============================================================================
# Long-Run Conservation Tests
# ==============================================================================

class TestLongRunConservation:
    """Tests for mass, heat, and salt conservation over many timesteps.

    These verify that the z-star transport velocity correctly reduces
    spurious drift, and that the conservation fixer restores invariants.
    """

    @staticmethod
    def _ocean_invariants(state, grid, z_coord, config):
        """Compute volume, heat, and salt global integrals."""
        mask = state.land_mask.data
        area = grid.area
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord,
            min_water_column_m=config.min_water_column_m,
        )
        weighted_area = mask * area
        ocean_area = float(jnp.sum(weighted_area))
        # Use mean eta (not total eta*area) for volume to avoid
        # zero-reference issues when eta starts at 0.
        mean_eta = float(jnp.sum(state.eta.data * weighted_area)) / max(ocean_area, 1.0)
        heat = float(jnp.sum(jnp.sum(state.T.data * h_k, axis=-1) * weighted_area))
        salt = float(jnp.sum(jnp.sum(state.S.data * h_k, axis=-1) * weighted_area))
        return mean_eta, heat, salt, ocean_area

    def test_longrun_conservation_with_fixer(self):
        """50-step integration with conservation fixer: drift should be small."""
        grid = create_cubed_sphere(8)
        z_coord = create_ocean_z_star(n_levels=10, H_max=4000.0)
        state = rest_state_ocean(
            grid, z_coord, T_surface=20.0, T_deep=2.0,
            S_uniform=35.0, H_max=4000.0,
        )
        config = OceanConfig(
            A_h=1e4, K_h=1e3, A_v=1e-3, K_v=1e-4,
            n_barotropic_substeps=10,
            use_conservation_fixer=True,
        )
        model = OceanModel(grid, z_coord, config)

        eta0, heat0, salt0, _ = self._ocean_invariants(state, grid, z_coord, config)

        for _ in range(50):
            state = model.step(state, 3600.0)

        eta1, heat1, salt1, _ = self._ocean_invariants(state, grid, z_coord, config)

        # With the conservation fixer, mean eta drift should be very small.
        assert abs(eta1 - eta0) < 1e-8, \
            f"Mean eta drift: {abs(eta1 - eta0):.2e}"
        # Heat and salt: relative errors should be bounded by the
        # cumulative fp32 storage-precision floor.  Heat ~ 1e19 J,
        # fp32 ULP ~ heat * 1.2e-7, so per-step quantization is ~ 1e12;
        # over 50 steps the drift accumulates to ~5e13 = ~5e-6 relative.
        # 1e-5 is the operational ceiling for an fp32-storage Boussinesq
        # ocean with the conservation fixer (was 1e-8 — only achievable
        # with bit-exact fp64 storage, which the default precision
        # policy no longer provides).
        assert abs(heat1 - heat0) / max(abs(heat0), 1.0) < 1e-5, \
            f"Heat drift: {abs(heat1 - heat0) / max(abs(heat0), 1.0):.2e}"
        assert abs(salt1 - salt0) / max(abs(salt0), 1.0) < 1e-5, \
            f"Salt drift: {abs(salt1 - salt0) / max(abs(salt0), 1.0):.2e}"

        # Check fields remain finite and bounded.
        mask_3d = state.land_mask.data[..., jnp.newaxis]
        assert jnp.all(jnp.isfinite(state.T.data))
        assert jnp.all(jnp.isfinite(state.S.data))
        assert jnp.all(jnp.isfinite(state.u.data))
        assert jnp.all(jnp.isfinite(state.v.data))
        assert jnp.all(jnp.isfinite(state.eta.data))
        T_ocean = jnp.where(mask_3d > 0.5, state.T.data, jnp.nan)
        assert float(jnp.nanmin(T_ocean)) > -5.0
        assert float(jnp.nanmax(T_ocean)) < 40.0

    def test_longrun_stability_without_fixer(self):
        """50-step integration WITHOUT fixer: state should remain finite and bounded."""
        grid = create_cubed_sphere(8)
        z_coord = create_ocean_z_star(n_levels=10, H_max=4000.0)
        state = rest_state_ocean(
            grid, z_coord, T_surface=20.0, T_deep=2.0,
            S_uniform=35.0, H_max=4000.0,
        )
        config = OceanConfig(
            A_h=1e4, K_h=1e3, A_v=1e-3, K_v=1e-4,
            n_barotropic_substeps=10,
            use_conservation_fixer=False,
        )
        model = OceanModel(grid, z_coord, config)

        for _ in range(50):
            state = model.step(state, 3600.0)

        # Without fixer, check stability (finite and bounded).
        assert jnp.all(jnp.isfinite(state.T.data))
        assert jnp.all(jnp.isfinite(state.S.data))
        assert jnp.all(jnp.isfinite(state.eta.data))
        mask_3d = state.land_mask.data[..., jnp.newaxis]
        T_ocean = jnp.where(mask_3d > 0.5, state.T.data, jnp.nan)
        assert float(jnp.nanmin(T_ocean)) > -5.0
        assert float(jnp.nanmax(T_ocean)) < 40.0

    def test_zstar_transport_velocity_boundary_conditions(self):
        """Z-star transport velocity should be zero at surface and bottom."""
        from legoesm.ocean.dynamics.ocean_pe_cdgrid import _diagnose_w_from_flux_div

        z_coord = create_ocean_z_star(n_levels=10, H_max=4000.0)
        # Arbitrary flux divergence (not identically zero).
        flux_div = jnp.linspace(-0.01, 0.01, 10).reshape(1, 1, 1, 10)

        w = _diagnose_w_from_flux_div(flux_div, z_coord)

        # Surface (k=0) and bottom (k=nlev) should be zero.
        assert float(jnp.max(jnp.abs(w[..., 0]))) < 1e-15, \
            f"Surface w = {float(w[0, 0, 0, 0]):.2e}"
        assert float(jnp.max(jnp.abs(w[..., -1]))) < 1e-15, \
            f"Bottom w = {float(w[0, 0, 0, -1]):.2e}"

        # Interior should be non-zero for non-trivial divergence.
        assert float(jnp.max(jnp.abs(w[..., 1:-1]))) > 0.0

    def test_advective_form_zero_for_uniform_tracer(self):
        """Advective form -ẇ·∂T/∂z should give zero for uniform T."""
        from legoesm.ocean.dynamics.ocean_pe_cdgrid import (
            _diagnose_w_from_flux_div,
            _vertical_advection_ocean,
        )

        z_coord = create_ocean_z_star(n_levels=10, H_max=4000.0)
        J = jnp.ones((1, 1, 1))
        T_uniform = jnp.full((1, 1, 1, 10), 15.0)

        # Non-trivial flux divergence → non-trivial ẇ.
        flux_div = jnp.linspace(-0.01, 0.01, 10).reshape(1, 1, 1, 10)
        w = _diagnose_w_from_flux_div(flux_div, z_coord)

        tendency = _vertical_advection_ocean(T_uniform, w, z_coord, J)
        assert float(jnp.max(jnp.abs(tendency))) < 1e-14, \
            f"Uniform tracer tendency: {float(jnp.max(jnp.abs(tendency))):.2e}"

    def test_zstar_eulerian_comparison(self):
        """Z-star velocity should differ from Eulerian when deta/dt != 0."""
        from legoesm.ocean.dynamics.ocean_pe_cdgrid import _diagnose_w_from_flux_div

        z_coord = create_ocean_z_star(n_levels=10, H_max=4000.0)
        # Asymmetric flux divergence so sum != 0 → Eulerian w[0] != 0.
        flux_div = jnp.linspace(0.0, 0.02, 10).reshape(1, 1, 1, 10)

        w_euler = _diagnose_w_from_flux_div(flux_div, None)
        w_zstar = _diagnose_w_from_flux_div(flux_div, z_coord)

        # Eulerian: w[0] = -sum(flux_div) != 0, w[nlev] = 0
        assert float(jnp.abs(w_euler[0, 0, 0, 0])) > 1e-5
        assert float(jnp.abs(w_euler[0, 0, 0, -1])) < 1e-15

        # Z-star: w[0] = 0, w[nlev] = 0
        assert float(jnp.abs(w_zstar[0, 0, 0, 0])) < 1e-15
        assert float(jnp.abs(w_zstar[0, 0, 0, -1])) < 1e-15

        # Interior values should differ.
        diff = float(jnp.max(jnp.abs(w_zstar - w_euler)))
        assert diff > 0.0
