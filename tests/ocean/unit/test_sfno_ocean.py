"""Integration tests for the SFNO Ocean model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.state import SpectralOceanState, SpectralOceanConfig
from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
from legoesm.ocean.dynamics.sfno_ocean import (
    SFNOOceanModel,
    SFNOOceanConfig,
)
from legoesm.ocean import SFNOOceanModel as SFNOOceanModelFromInit
from legoesm.ml.sfno import SFNOConfig
from legoesm.ocean.dynamics.channel_packing import (
    OceanChannelSpec,
    pack_ocean_state,
    unpack_ocean_output,
)
from legoesm.ml.conservation import (
    correct_ocean_volume,
    correct_ocean_tracer,
)
from legoesm.core.field import Field

# Enable float64
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def grid_t10():
    """Small T10 grid for fast tests."""
    return create_gaussian_grid(n_max=10)


@pytest.fixture(scope="module")
def z_coord():
    """Small ocean vertical coordinate (4 levels)."""
    return create_ocean_z_star(n_levels=4, H_max=5500.0)


@pytest.fixture(scope="module")
def ocean_state(grid_t10, z_coord):
    """Rest-state spectral ocean initial condition."""
    return rest_state_spectral_ocean(grid_t10, z_coord)


@pytest.fixture(scope="module")
def sfno_config(z_coord):
    """Small SFNO config for testing (4 ocean levels)."""
    nlev = z_coord.n_levels
    n_channels = 4 * nlev + 2  # u, v, T, S at each level + eta, H_bathy
    return SFNOConfig(
        in_channels=n_channels,
        out_channels=n_channels,
        embed_dim=16,
        n_blocks=2,
        mlp_expansion=2,
    )


# =============================================================================
# Module export tests
# =============================================================================

class TestModuleExports:

    def test_sfno_ocean_in_init(self):
        """SFNOOceanModel should be importable from ocean.__init__."""
        assert SFNOOceanModelFromInit is SFNOOceanModel


# =============================================================================
# Ocean Channel Spec tests
# =============================================================================

class TestOceanChannelSpec:

    def test_channel_count(self):
        """OceanChannelSpec should compute correct channel count."""
        spec = OceanChannelSpec(nlev=10)
        assert spec.n_channels == 4 * 10 + 2  # u,v,T,S + eta,H_bathy

    def test_slices(self):
        """Channel slices should be non-overlapping and contiguous."""
        spec = OceanChannelSpec(nlev=4)
        assert spec.u_slice == slice(0, 4)
        assert spec.v_slice == slice(4, 8)
        assert spec.T_slice == slice(8, 12)
        assert spec.S_slice == slice(12, 16)
        assert spec.eta_idx == 16
        assert spec.H_bathy_idx == 17


# =============================================================================
# Ocean Channel Packing tests
# =============================================================================

class TestOceanChannelPacking:

    def test_pack_shape(self, ocean_state, grid_t10, z_coord):
        """Packed ocean state should have correct shape."""
        packed = pack_ocean_state(ocean_state, grid_t10)
        nlev = z_coord.n_levels
        expected_channels = 4 * nlev + 2
        assert packed.shape == (grid_t10.n_lat, grid_t10.n_lon, expected_channels)

    def test_pack_no_nans(self, ocean_state, grid_t10):
        """Packed ocean state should be finite."""
        packed = pack_ocean_state(ocean_state, grid_t10)
        assert jnp.all(jnp.isfinite(packed))

    def test_pack_unpack_roundtrip(self, ocean_state, grid_t10):
        """Pack then unpack should approximately recover the original."""
        packed = pack_ocean_state(ocean_state, grid_t10)
        recovered = unpack_ocean_output(
            packed, ocean_state, grid_t10, mode="state_update"
        )
        # Check that T spectral coefficients are close
        np.testing.assert_allclose(
            np.abs(recovered.T_hat.data),
            np.abs(ocean_state.T_hat.data),
            atol=1e-6,
        )
        # Check that S spectral coefficients are close
        np.testing.assert_allclose(
            np.abs(recovered.S_hat.data),
            np.abs(ocean_state.S_hat.data),
            atol=1e-6,
        )

    def test_land_mask_preserved(self, ocean_state, grid_t10):
        """Land mask should be preserved through pack/unpack."""
        packed = pack_ocean_state(ocean_state, grid_t10)
        recovered = unpack_ocean_output(
            packed, ocean_state, grid_t10, mode="state_update"
        )
        np.testing.assert_array_equal(
            recovered.land_mask_grid.data,
            ocean_state.land_mask_grid.data,
        )

    def test_tendency_mode_zeros_static(self, ocean_state, grid_t10):
        """In tendency mode, static fields should be zeroed."""
        packed = pack_ocean_state(ocean_state, grid_t10)
        tendencies = unpack_ocean_output(
            packed, ocean_state, grid_t10, mode="tendencies"
        )
        np.testing.assert_array_equal(
            tendencies.H_bathy_hat.data,
            jnp.zeros_like(ocean_state.H_bathy_hat.data),
        )
        np.testing.assert_array_equal(
            tendencies.land_mask_grid.data,
            jnp.zeros_like(ocean_state.land_mask_grid.data),
        )


# =============================================================================
# Ocean Conservation tests
# =============================================================================

class TestOceanConservation:

    def test_volume_correction(self, grid_t10):
        """Volume correction should preserve global eta integral."""
        mask = jnp.ones((grid_t10.n_lat, grid_t10.n_lon))
        eta_old = jnp.zeros((grid_t10.n_lat, grid_t10.n_lon))
        eta_new = jnp.ones((grid_t10.n_lat, grid_t10.n_lon)) * 0.5

        eta_fixed = correct_ocean_volume(eta_new, eta_old, grid_t10, mask)

        # Global integral should match old
        w = grid_t10.weights[:, None]
        dlon = 2.0 * jnp.pi / grid_t10.n_lon
        area = (grid_t10.radius ** 2) * w * dlon

        vol_old = jnp.sum(eta_old * mask * area)
        vol_fixed = jnp.sum(eta_fixed * mask * area)
        np.testing.assert_allclose(vol_fixed, vol_old, atol=1e-4)

    def test_heat_correction(self, grid_t10):
        """Heat correction should preserve volume-integrated T."""
        mask = jnp.ones((grid_t10.n_lat, grid_t10.n_lon))
        nlev = 4
        h_k = jnp.ones((grid_t10.n_lat, grid_t10.n_lon, nlev)) * 100.0

        T_old = jnp.ones((grid_t10.n_lat, grid_t10.n_lon, nlev)) * 15.0
        T_new = jnp.ones((grid_t10.n_lat, grid_t10.n_lon, nlev)) * 16.0

        T_fixed = correct_ocean_tracer(T_new, T_old, h_k, h_k, grid_t10, mask)

        w = grid_t10.weights[:, None]
        dlon = 2.0 * jnp.pi / grid_t10.n_lon
        area = (grid_t10.radius ** 2) * w * dlon

        heat_old = jnp.sum(jnp.sum(T_old * h_k, axis=-1) * mask * area)
        heat_fixed = jnp.sum(jnp.sum(T_fixed * h_k, axis=-1) * mask * area)
        np.testing.assert_allclose(float(heat_fixed), float(heat_old), rtol=1e-8)


# =============================================================================
# SFNO Ocean Model tests
# =============================================================================

class TestSFNOOceanModel:

    def test_state_update_mode(self, grid_t10, z_coord, ocean_state, sfno_config):
        """State update mode should produce valid output."""
        config = SFNOOceanConfig(
            sfno_config=sfno_config,
            mode="state_update",
            correct_volume=False,
            correct_heat=False,
            correct_salt=False,
        )
        model = SFNOOceanModel(
            grid=grid_t10, z_coord=z_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(ocean_state, dt=3600.0)

        # Check shapes match
        assert new_state.vor_hat.data.shape == ocean_state.vor_hat.data.shape
        assert new_state.T_hat.data.shape == ocean_state.T_hat.data.shape
        assert new_state.eta_hat.data.shape == ocean_state.eta_hat.data.shape
        # Check no NaNs
        assert not jnp.any(jnp.isnan(new_state.vor_hat.data))
        assert not jnp.any(jnp.isnan(new_state.T_hat.data))
        assert not jnp.any(jnp.isnan(new_state.eta_hat.data))

    def test_hybrid_mode(self, grid_t10, z_coord, ocean_state, sfno_config):
        """Hybrid tendency mode should produce valid output."""
        config = SFNOOceanConfig(
            sfno_config=sfno_config,
            mode="hybrid_tendencies",
            correct_volume=False,
            correct_heat=False,
            correct_salt=False,
        )
        model = SFNOOceanModel(
            grid=grid_t10, z_coord=z_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(ocean_state, dt=600.0)
        assert new_state.T_hat.data.shape == ocean_state.T_hat.data.shape
        assert not jnp.any(jnp.isnan(new_state.T_hat.data))

    def test_with_conservation(self, grid_t10, z_coord, ocean_state, sfno_config):
        """Conservation corrections should not introduce NaNs."""
        config = SFNOOceanConfig(
            sfno_config=sfno_config,
            mode="state_update",
            correct_volume=True,
            correct_heat=True,
            correct_salt=True,
        )
        model = SFNOOceanModel(
            grid=grid_t10, z_coord=z_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(ocean_state, dt=3600.0)
        assert not jnp.any(jnp.isnan(new_state.T_hat.data))
        assert not jnp.any(jnp.isnan(new_state.S_hat.data))
        assert not jnp.any(jnp.isnan(new_state.eta_hat.data))

    def test_multi_step_integration(self, grid_t10, z_coord, ocean_state, sfno_config):
        """Multi-step integration should not produce NaNs."""
        config = SFNOOceanConfig(
            sfno_config=sfno_config,
            mode="state_update",
            correct_volume=False,
            correct_heat=False,
            correct_salt=False,
        )
        model = SFNOOceanModel(
            grid=grid_t10, z_coord=z_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        final_state, trajectory = model.integrate(
            ocean_state, duration=3 * 3600.0, dt=3600.0, save_every=1,
        )
        assert len(trajectory) == 4  # initial + 3 saved
        assert not jnp.any(jnp.isnan(final_state.T_hat.data))
        assert not jnp.any(jnp.isnan(final_state.S_hat.data))

    def test_output_shapes_preserved(self, grid_t10, z_coord, ocean_state, sfno_config):
        """Output state should have same shapes as input state."""
        config = SFNOOceanConfig(
            sfno_config=sfno_config,
            mode="state_update",
            correct_volume=False,
            correct_heat=False,
            correct_salt=False,
        )
        model = SFNOOceanModel(
            grid=grid_t10, z_coord=z_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(ocean_state, dt=3600.0)
        for old_field, new_field in zip(ocean_state, new_state):
            assert old_field.data.shape == new_field.data.shape
            assert old_field.data.dtype == new_field.data.dtype

    def test_static_fields_preserved(self, grid_t10, z_coord, ocean_state, sfno_config):
        """Static fields (H_bathy, land_mask) should be preserved in state_update."""
        config = SFNOOceanConfig(
            sfno_config=sfno_config,
            mode="state_update",
            correct_volume=False,
            correct_heat=False,
            correct_salt=False,
        )
        model = SFNOOceanModel(
            grid=grid_t10, z_coord=z_coord, config=config,
            key=jax.random.PRNGKey(0),
        )
        new_state = model.step(ocean_state, dt=3600.0)
        # H_bathy and land_mask should be identical
        np.testing.assert_array_equal(
            new_state.H_bathy_hat.data,
            ocean_state.H_bathy_hat.data,
        )
        np.testing.assert_array_equal(
            new_state.land_mask_grid.data,
            ocean_state.land_mask_grid.data,
        )
