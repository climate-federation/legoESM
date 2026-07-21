"""Integration tests for the SFNO Shallow Water model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
    SpectralSWState,
    SpectralSWConfig,
    williamson_test2_spectral,
)
from legoesm.atmosphere.dynamics.neural.sfno_sw import (
    SFNOShallowWaterModel,
    SFNOShallowWaterConfig,
)
from legoesm.atmosphere.dynamics import (
    resolve_solver_name,
    create_model,
    AVAILABLE_SOLVERS,
    DISCRETIZATION_OPTIONS,
)
from legoesm.ml.sfno import SFNOConfig
from legoesm.ml.channel_packing import pack_sw_state, unpack_sw_output
from legoesm.core.field import Field

# Enable float64
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def grid_t10():
    """Small T10 grid for fast tests."""
    return create_gaussian_grid(n_max=10)


@pytest.fixture(scope="module")
def sw_state(grid_t10):
    """Williamson test 2 initial condition."""
    return williamson_test2_spectral(grid_t10)


@pytest.fixture(scope="module")
def sfno_config():
    """Small SFNO config for testing."""
    return SFNOConfig(
        in_channels=4, out_channels=4,
        embed_dim=16, n_blocks=2, mlp_expansion=2,
    )


# =============================================================================
# Factory registration tests
# =============================================================================

class TestFactoryRegistration:

    def test_sfno_in_available_solvers(self):
        """SFNO solver names should be in AVAILABLE_SOLVERS."""
        assert "sfno_shallow_water" in AVAILABLE_SOLVERS
        assert "sfno_primitive_equations" in AVAILABLE_SOLVERS

    def test_sfno_in_discretization_options(self):
        """'sfno' should be a valid discretization option."""
        assert "sfno" in DISCRETIZATION_OPTIONS

    def test_resolve_sfno_sw(self):
        """resolve_solver_name should resolve SFNO shallow water."""
        name = resolve_solver_name(
            dynamics="shallow_water", discretization="sfno"
        )
        assert name == "sfno_shallow_water"

    def test_resolve_sfno_pe(self):
        """resolve_solver_name should resolve SFNO primitive equations."""
        name = resolve_solver_name(
            dynamics="hydrostatic", discretization="sfno"
        )
        assert name == "sfno_primitive_equations"

    def test_create_sfno_sw_model(self, grid_t10):
        """create_model should instantiate SFNOShallowWaterModel."""
        model = create_model("sfno_shallow_water", grid=grid_t10)
        assert isinstance(model, SFNOShallowWaterModel)


# =============================================================================
# Channel packing tests
# =============================================================================

class TestChannelPacking:

    def test_pack_shape(self, sw_state, grid_t10):
        """Packed state should have shape (n_lat, n_lon, 4)."""
        packed = pack_sw_state(sw_state, grid_t10)
        assert packed.shape == (grid_t10.n_lat, grid_t10.n_lon, 4)

    def test_pack_no_nans(self, sw_state, grid_t10):
        """Packed state should be finite."""
        packed = pack_sw_state(sw_state, grid_t10)
        assert jnp.all(jnp.isfinite(packed))

    def test_pack_unpack_roundtrip(self, sw_state, grid_t10):
        """Pack then unpack should approximately recover the original."""
        packed = pack_sw_state(sw_state, grid_t10)
        recovered = unpack_sw_output(
            packed, sw_state, grid_t10, mode="state_update"
        )
        # Check that spectral coefficients are close
        np.testing.assert_allclose(
            np.abs(recovered.vor_hat.data),
            np.abs(sw_state.vor_hat.data),
            atol=1e-8,
        )


# =============================================================================
# SFNO Shallow Water Model tests
# =============================================================================

class TestSFNOShallowWaterModel:

    def test_state_update_mode(self, grid_t10, sw_state, sfno_config):
        """State update mode should produce valid output."""
        config = SFNOShallowWaterConfig(
            sfno_config=sfno_config,
            mode="state_update",
        )
        model = SFNOShallowWaterModel(
            grid=grid_t10, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(sw_state, dt=3600.0)
        assert new_state.vor_hat.data.shape == sw_state.vor_hat.data.shape
        assert not jnp.any(jnp.isnan(new_state.vor_hat.data))
        assert not jnp.any(jnp.isnan(new_state.phi_hat.data))

    def test_hybrid_mode(self, grid_t10, sw_state, sfno_config):
        """Hybrid tendency mode should produce valid output."""
        config = SFNOShallowWaterConfig(
            sfno_config=sfno_config,
            mode="hybrid_tendencies",
        )
        model = SFNOShallowWaterModel(
            grid=grid_t10, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(sw_state, dt=600.0)
        assert new_state.vor_hat.data.shape == sw_state.vor_hat.data.shape
        assert not jnp.any(jnp.isnan(new_state.vor_hat.data))

    def test_multi_step_integration(self, grid_t10, sw_state, sfno_config):
        """Multi-step integration should not produce NaNs."""
        config = SFNOShallowWaterConfig(
            sfno_config=sfno_config,
            mode="state_update",
        )
        model = SFNOShallowWaterModel(
            grid=grid_t10, config=config,
            key=jax.random.PRNGKey(0),
        )
        # Run 5 steps
        final_state, trajectory = model.integrate(
            sw_state, duration=5 * 3600.0, dt=3600.0, save_every=1,
        )
        assert len(trajectory) == 6  # initial + 5 saved
        assert not jnp.any(jnp.isnan(final_state.vor_hat.data))
        assert not jnp.any(jnp.isnan(final_state.phi_hat.data))

    def test_output_shapes_preserved(self, grid_t10, sw_state, sfno_config):
        """Output state should have same shapes as input state."""
        config = SFNOShallowWaterConfig(
            sfno_config=sfno_config,
            mode="state_update",
        )
        model = SFNOShallowWaterModel(
            grid=grid_t10, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(sw_state, dt=3600.0)
        for old_field, new_field in zip(sw_state, new_state):
            assert old_field.data.shape == new_field.data.shape
            assert old_field.data.dtype == new_field.data.dtype
