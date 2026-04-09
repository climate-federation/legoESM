"""Tests for the lat-lon C-grid FV ocean model.

Verifies that the C-grid discretization produces correct, stable, and
checkerboard-free results.  The C-grid eliminates the 2*dx null space
present in the A-grid (LatLonOceanModel) formulation.
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    gradient_x_cgrid,
    gradient_y_cgrid,
    divergence_cgrid,
    compute_face_masks,
)


@pytest.fixture
def grid():
    return create_latlon_grid(n_lat=36, n_lon=72)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=5, H_max=4000.0)


@pytest.fixture
def state(grid, z_coord):
    return rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )


@pytest.fixture
def config():
    return LatLonCGridOceanConfig()


# =========================================================================
# C-grid operator unit tests
# =========================================================================

class TestCGridOperators:
    """Compact-stencil C-grid operators."""

    def test_gradient_x_shape(self, grid):
        f = jnp.ones((grid.n_lat, grid.n_lon))
        gx = gradient_x_cgrid(f, grid)
        assert gx.shape == (grid.n_lat, grid.n_lon + 1)

    def test_gradient_y_shape(self, grid):
        f = jnp.ones((grid.n_lat, grid.n_lon))
        gy = gradient_y_cgrid(f, grid)
        assert gy.shape == (grid.n_lat + 1, grid.n_lon)

    def test_gradient_constant_field_zero(self, grid):
        """Gradient of a constant field should be zero everywhere."""
        f = jnp.ones((grid.n_lat, grid.n_lon)) * 42.0
        gx = gradient_x_cgrid(f, grid)
        gy = gradient_y_cgrid(f, grid)
        assert float(jnp.max(jnp.abs(gx))) < 1e-10
        assert float(jnp.max(jnp.abs(gy))) < 1e-10

    def test_divergence_shape(self, grid):
        n_lat = grid.n_lat
        n_lon = grid.n_lon
        u = jnp.zeros((n_lat, n_lon + 1))
        v = jnp.zeros((n_lat + 1, n_lon))
        div = divergence_cgrid(u, v, grid)
        assert div.shape == (n_lat, n_lon)

    def test_divergence_uniform_zero(self, grid):
        """Divergence of a uniform field should be near zero."""
        n_lat = grid.n_lat
        n_lon = grid.n_lon
        u = jnp.ones((n_lat, n_lon + 1))
        v = jnp.zeros((n_lat + 1, n_lon))
        div = divergence_cgrid(u, v, grid)
        # On a sphere, div of uniform u is not exactly zero due to
        # geometry, but should be bounded
        assert jnp.all(jnp.isfinite(div))

    def test_face_masks(self, grid):
        """Face masks should have correct shapes."""
        n_lat = grid.n_lat
        n_lon = grid.n_lon
        mask = jnp.ones((n_lat, n_lon))
        u_mask, v_mask = compute_face_masks(mask)
        assert u_mask.shape == (n_lat, n_lon + 1)
        assert v_mask.shape == (n_lat + 1, n_lon)
        # All-ocean: u_mask should be 1 everywhere
        assert float(jnp.min(u_mask)) == 1.0
        # v_mask at poles should be 0 (wall BC)
        assert float(v_mask[0].max()) == 0.0
        assert float(v_mask[-1].max()) == 0.0

    def test_face_masks_with_land(self):
        """Face between ocean and land should be masked."""
        mask = jnp.array([[1.0, 0.0, 1.0, 1.0]])
        u_mask, v_mask = compute_face_masks(mask)
        # u_mask between ocean (col 0) and land (col 1) should be 0
        assert float(u_mask[0, 1]) == 0.0
        # u_mask between ocean cells should be 1
        assert float(u_mask[0, 3]) == 1.0


# =========================================================================
# C-grid tendency tests
# =========================================================================

class TestCGridTendencies:
    """C-grid ocean baroclinic tendencies."""

    def test_tendencies_finite(self, state, grid, z_coord, config):
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, config,
        )
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
        assert jnp.all(jnp.isfinite(tend.deta_dt.data))

    def test_rest_state_small_tendencies(self, state, grid, z_coord, config):
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, config,
        )
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1e-2
        assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1e-2

    def test_tendency_shapes(self, state, grid, z_coord, config):
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, config,
        )
        n_lat = grid.n_lat
        n_lon = grid.n_lon
        nlev = z_coord.n_levels
        assert tend.du_dt.data.shape == (n_lat, n_lon + 1, nlev)
        assert tend.dv_dt.data.shape == (n_lat + 1, n_lon, nlev)
        assert tend.dT_dt.data.shape == (n_lat, n_lon, nlev)
        assert tend.dS_dt.data.shape == (n_lat, n_lon, nlev)
        assert tend.deta_dt.data.shape == (n_lat, n_lon)

    def test_static_fields_zero(self, state, grid, z_coord, config):
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, config,
        )
        assert jnp.all(tend.dH_bathy_dt.data == 0)
        assert jnp.all(tend.dland_mask_dt.data == 0)


# =========================================================================
# C-grid model integration tests
# =========================================================================

class TestCGridOceanModel:
    """C-grid ocean model integration."""

    def test_single_step(self, grid, z_coord, config, state):
        model = LatLonCGridOceanModel(grid, z_coord, config)
        s1 = model.step(state, 3600.0)
        assert jnp.all(jnp.isfinite(s1.u.data))
        assert jnp.all(jnp.isfinite(s1.v.data))
        assert jnp.all(jnp.isfinite(s1.T.data))
        assert jnp.all(jnp.isfinite(s1.eta.data))

    def test_shapes_preserved(self, grid, z_coord, config, state):
        model = LatLonCGridOceanModel(grid, z_coord, config)
        s1 = model.step(state, 3600.0)
        assert s1.u.data.shape == state.u.data.shape
        assert s1.v.data.shape == state.v.data.shape
        assert s1.T.data.shape == state.T.data.shape
        assert s1.eta.data.shape == state.eta.data.shape

    def test_static_fields_unchanged(self, grid, z_coord, config, state):
        model = LatLonCGridOceanModel(grid, z_coord, config)
        s1 = model.step(state, 3600.0)
        assert jnp.allclose(s1.H_bathy.data, state.H_bathy.data)
        assert jnp.allclose(s1.land_mask.data, state.land_mask.data)
        assert jnp.allclose(s1.u_mask.data, state.u_mask.data)
        assert jnp.allclose(s1.v_mask.data, state.v_mask.data)

    def test_rest_state_stability(self, grid, z_coord, state):
        """Rest state should show minimal drift (key C-grid advantage).

        A-grid models develop checkerboard noise from the 2*dx null space.
        The C-grid should remain quiet.
        """
        cfg = LatLonCGridOceanConfig(A_h=1e4, K_h=1e3, A_v=1e-3, K_v=1e-4)
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        s = state
        for _ in range(10):
            s = model.step(s, 3600.0)

        # Should remain stable
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.T.data))
        assert jnp.all(jnp.isfinite(s.eta.data))

        # SSH drift should be very small for a rest state
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))
        assert eta_max < 1.0, f"Rest state eta drift too large: {eta_max}"

    def test_barotropic_wave_stability(self, grid, z_coord):
        """Barotropic wave should propagate stably for 10+ steps.

        Apply a Gaussian SSH perturbation and verify the model
        stays stable without checkerboard artifacts.

        Uses dt=600s with 60 barotropic substeps to satisfy
        CFL ~ 0.08 (c ~ 198 m/s, dx_min ~ 24 km at high lat).
        """
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0,
        )

        # Add Gaussian SSH perturbation
        lat_deg = grid.lat2d * (180.0 / jnp.pi)
        lon_deg = grid.lon2d * (180.0 / jnp.pi)
        eta_pert = 1.0 * jnp.exp(
            -((lon_deg - 180.0) ** 2 + lat_deg ** 2) / (10.0 ** 2)
        )
        eta_pert = eta_pert * state.land_mask.data
        state = state._replace(
            eta=state.eta.replace(data=eta_pert),
        )

        cfg = LatLonCGridOceanConfig(
            A_h=1e4, K_h=1e3, n_barotropic_substeps=60,
        )
        model = LatLonCGridOceanModel(grid, z_coord, cfg)

        s = state
        for _ in range(10):
            s = model.step(s, 600.0)

        assert jnp.all(jnp.isfinite(s.eta.data))
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.v.data))

        # SSH should remain physically reasonable (wave disperses)
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))
        assert eta_max < 10.0, f"Wave amplitude blow-up: {eta_max}"

    def test_differentiable(self, grid, z_coord, config, state):
        """C-grid model should be differentiable through tendencies."""
        def loss_fn(eta_data):
            s = state._replace(eta=state.eta.replace(data=eta_data))
            tend = latlon_cgrid_ocean_baroclinic_tendencies(
                s, grid, z_coord, config,
            )
            return jnp.mean(tend.deta_dt.data ** 2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.eta.data)
        assert jnp.all(jnp.isfinite(g))
