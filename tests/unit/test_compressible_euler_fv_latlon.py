"""Unit tests for the FV non-hydrostatic compressible Euler model on lat-lon grid."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_height_coordinate, compute_terrain_metric
from legoesm.atmosphere.dynamics.compressible_euler_fv_latlon import (
    FVCompressibleEulerLatLonConfig,
    FVCompressibleEulerLatLonModel,
    fv_compressible_euler_latlon_slow_tendencies,
)


pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="Requires JAX_ENABLE_X64=1",
)


@pytest.fixture
def grid():
    return create_latlon_grid(8, 16)


@pytest.fixture
def height_coord():
    return create_height_coordinate(10, 30000.0)


@pytest.fixture
def terrain_metric(grid, height_coord):
    z_s = jnp.zeros((grid.n_lat, grid.n_lon))
    return compute_terrain_metric(z_s, height_coord)


def _make_nh_state(grid, height_coord, n_tracers=0):
    """Create a rest-state NonHydrostaticState on lat-lon grid."""
    nlev = height_coord.n_levels
    shape_3d = (grid.n_lat, grid.n_lon, nlev)
    shape_w = (grid.n_lat, grid.n_lon, nlev + 1)
    shape_2d = (grid.n_lat, grid.n_lon)

    dims_3d = ("lat", "lon", "level")
    dims_w = ("lat", "lon", "level_half")
    dims_2d = ("lat", "lon")

    return NonHydrostaticState(
        u=Field(data=jnp.zeros(shape_3d), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros(shape_3d), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros(shape_w), name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros(shape_3d), name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros(shape_3d), name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d,
                   units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((*shape_3d, n_tracers)), name="tracers",
                      dims=("lat", "lon", "level", "tracer"), units="kg/kg"),
    )


class TestFVCELatLonTendencies:

    def test_rest_state_small_tendencies(self, grid, height_coord, terrain_metric):
        """Rest state should produce near-zero tendencies."""
        state = _make_nh_state(grid, height_coord)
        config = FVCompressibleEulerLatLonConfig(
            hyperdiff_coeff=0.0, sponge_coeff=0.0,
            use_polar_filter=False,
        )

        tend = fv_compressible_euler_latlon_slow_tendencies(
            state, grid, height_coord, terrain_metric, config,
        )

        assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1e-6, \
            f"du_dt max: {float(jnp.max(jnp.abs(tend.du_dt.data)))}"
        assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1e-6, \
            f"dv_dt max: {float(jnp.max(jnp.abs(tend.dv_dt.data)))}"

    def test_tendencies_finite(self, grid, height_coord, terrain_metric):
        """Tendencies should be finite for a non-trivial state."""
        state = _make_nh_state(grid, height_coord)
        # Add some perturbation
        state = state._replace(
            u=state.u.replace(data=state.u.data + 5.0),
            theta_prime=state.theta_prime.replace(data=state.theta_prime.data + 1.0),
        )
        config = FVCompressibleEulerLatLonConfig(
            hyperdiff_coeff=0.0, use_polar_filter=False,
        )

        tend = fv_compressible_euler_latlon_slow_tendencies(
            state, grid, height_coord, terrain_metric, config,
        )

        assert jnp.all(jnp.isfinite(tend.du_dt.data)), "du_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dv_dt.data)), "dv_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dw_dt.data)), "dw_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)), "dtheta_prime not finite"
        assert jnp.all(jnp.isfinite(tend.drho_prime_dt.data)), "drho_prime not finite"

    def test_tendencies_with_polar_filter(self, grid, height_coord, terrain_metric):
        """Tendencies with polar filter should be finite."""
        state = _make_nh_state(grid, height_coord)
        state = state._replace(
            u=state.u.replace(data=state.u.data + 5.0),
        )
        config = FVCompressibleEulerLatLonConfig(
            hyperdiff_coeff=0.0, use_polar_filter=True,
        )
        from legoesm.grids.polar_filter import compute_polar_filter_mask
        mask = compute_polar_filter_mask(grid, dt=1.0)

        tend = fv_compressible_euler_latlon_slow_tendencies(
            state, grid, height_coord, terrain_metric, config,
            polar_mask=mask,
        )

        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))


class TestFVCELatLonModel:

    def test_single_step(self, grid, height_coord, terrain_metric):
        """Model can take a single time step."""
        config = FVCompressibleEulerLatLonConfig(
            hyperdiff_coeff=0.0, n_acoustic_substeps=2,
            use_polar_filter=False,
        )
        model = FVCompressibleEulerLatLonModel(
            grid, height_coord, terrain_metric, config, dt=1.0,
        )
        state = _make_nh_state(grid, height_coord)
        new_state = model.step(state, 1.0)

        assert jnp.all(jnp.isfinite(new_state.u.data)), "u not finite"
        assert jnp.all(jnp.isfinite(new_state.v.data)), "v not finite"
        assert jnp.all(jnp.isfinite(new_state.w.data)), "w not finite"
        assert jnp.all(jnp.isfinite(new_state.theta_prime.data)), "theta' not finite"
        assert jnp.all(jnp.isfinite(new_state.rho_prime.data)), "rho' not finite"

    def test_multi_step_stability(self, grid, height_coord, terrain_metric):
        """Model should be stable for a few time steps from rest."""
        config = FVCompressibleEulerLatLonConfig(
            hyperdiff_coeff=0.0, n_acoustic_substeps=2,
            use_polar_filter=False,
        )
        model = FVCompressibleEulerLatLonModel(
            grid, height_coord, terrain_metric, config, dt=1.0,
        )
        state = _make_nh_state(grid, height_coord)

        for _ in range(5):
            state = model.step(state, 1.0)

        assert jnp.all(jnp.isfinite(state.u.data)), "u not finite after 5 steps"
        assert jnp.all(jnp.isfinite(state.theta_prime.data)), "theta' not finite"

    def test_single_step_with_polar_filter(self, grid, height_coord, terrain_metric):
        """Model with polar filter can take a single step."""
        config = FVCompressibleEulerLatLonConfig(
            hyperdiff_coeff=0.0, n_acoustic_substeps=2,
            use_polar_filter=True,
        )
        model = FVCompressibleEulerLatLonModel(
            grid, height_coord, terrain_metric, config, dt=1.0,
        )
        state = _make_nh_state(grid, height_coord)
        new_state = model.step(state, 1.0)

        assert jnp.all(jnp.isfinite(new_state.u.data)), "u not finite with polar filter"

    def test_phis_static(self, grid, height_coord, terrain_metric):
        """Surface geopotential should not change."""
        config = FVCompressibleEulerLatLonConfig(
            n_acoustic_substeps=2, use_polar_filter=False,
        )
        model = FVCompressibleEulerLatLonModel(
            grid, height_coord, terrain_metric, config, dt=1.0,
        )
        state = _make_nh_state(grid, height_coord)
        phis_init = state.phis.data

        for _ in range(3):
            state = model.step(state, 1.0)

        assert jnp.allclose(state.phis.data, phis_init, atol=1e-12)

    def test_grad_through_tendencies(self, grid, height_coord, terrain_metric):
        """jax.grad should work through tendency computation."""
        state = _make_nh_state(grid, height_coord)
        config = FVCompressibleEulerLatLonConfig(
            hyperdiff_coeff=0.0, sponge_coeff=0.0,
            use_polar_filter=False,
        )

        def loss(u_data):
            s = state._replace(u=state.u.replace(data=u_data))
            tend = fv_compressible_euler_latlon_slow_tendencies(
                s, grid, height_coord, terrain_metric, config,
            )
            return jnp.mean(tend.du_dt.data**2)

        grads = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grads)), "Gradient contains NaN/Inf"
