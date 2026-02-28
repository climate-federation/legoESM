"""Tests for backend compatibility guards (Metal / float64).

Verifies that the spectral solver is blocked on unsupported backends
(e.g., Apple Metal) and passes on supported ones (CPU, GPU, TPU).
"""

import warnings
from unittest.mock import patch, MagicMock

import pytest

from legoesm.core.hardware import check_spectral_backend, get_backend


# =============================================================================
# check_spectral_backend — core guard function
# =============================================================================

class TestCheckSpectralBackend:
    """Tests for check_spectral_backend()."""

    def test_cpu_backend_passes(self):
        """CPU backend should pass without error."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="cpu"):
            check_spectral_backend()  # Should not raise

    def test_gpu_backend_passes(self):
        """CUDA GPU backend should pass without error."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="gpu"):
            check_spectral_backend()  # Should not raise

    def test_tpu_backend_passes(self):
        """TPU backend should pass without error."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="tpu"):
            check_spectral_backend()  # Should not raise

    def test_metal_backend_raises(self):
        """Metal backend should raise ValueError."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            with pytest.raises(ValueError, match="spectral solver requires float64"):
                check_spectral_backend()

    def test_metal_case_insensitive(self):
        """Metal detection should work regardless of case."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="metal"):
            with pytest.raises(ValueError, match="spectral solver requires float64"):
                check_spectral_backend()

    def test_metal_allow_unsupported_warns(self):
        """Metal with allow_unsupported=True should warn, not raise."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_spectral_backend(allow_unsupported=True)
                guard_warnings = [
                    x for x in w
                    if issubclass(x.category, RuntimeWarning)
                    and "unsupported backend" in str(x.message).lower()
                ]
                assert len(guard_warnings) == 1

    def test_error_message_has_remediation(self):
        """Error message should include JAX_PLATFORMS hint."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            with pytest.raises(ValueError, match="JAX_PLATFORMS=cpu"):
                check_spectral_backend()

    def test_error_message_suggests_alternative(self):
        """Error message should suggest the finite-volume alternative."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            with pytest.raises(ValueError, match="finite-volume"):
                check_spectral_backend()

    def test_error_message_mentions_config_override(self):
        """Error message should mention the config override path."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            with pytest.raises(ValueError, match="allow_unsupported"):
                check_spectral_backend()


# =============================================================================
# get_backend
# =============================================================================

class TestGetBackend:
    """Tests for get_backend()."""

    def test_returns_uppercase(self):
        """get_backend() should always return uppercase."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="cpu"):
            assert get_backend() == "CPU"

    def test_returns_uppercase_metal(self):
        with patch("legoesm.core.hardware.jax.default_backend", return_value="metal"):
            assert get_backend() == "METAL"


# =============================================================================
# Integration: create_gaussian_grid guard
# =============================================================================

class TestGaussianGridGuard:
    """Test that create_gaussian_grid calls the backend guard."""

    def test_metal_blocks_grid_creation(self):
        """Creating a Gaussian grid on Metal should raise."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            from legoesm.grids.gaussian import create_gaussian_grid
            with pytest.raises(ValueError, match="spectral solver requires float64"):
                create_gaussian_grid(n_max=21)

    def test_metal_grid_creation_with_override_warns(self):
        """Creating a Gaussian grid on Metal with override should warn."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            from legoesm.grids.gaussian import create_gaussian_grid
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # The guard should warn (not raise), but float64 allocation
                # may still fail. We only test the guard behavior here.
                try:
                    create_gaussian_grid(n_max=21, allow_unsupported_backend=True)
                except Exception:
                    pass  # Expected: float64 may fail on actual Metal
                guard_warnings = [
                    x for x in w
                    if issubclass(x.category, RuntimeWarning)
                    and "unsupported backend" in str(x.message).lower()
                ]
                assert len(guard_warnings) >= 1


# =============================================================================
# Integration: SpectralShallowWaterModel guard
# =============================================================================

class TestSpectralSWModelGuard:
    """Test that SpectralShallowWaterModel.__init__ calls the backend guard."""

    def test_metal_auto_routes_to_cpu(self):
        """On Metal, SpectralShallowWaterModel auto-routes to CPU."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            from legoesm.atmosphere.dynamics.spectral_sw import (
                SpectralShallowWaterModel,
            )
            # Use a real tiny grid instead of MagicMock so device_put works.
            import jax
            import jax.numpy as jnp
            from legoesm.grids.gaussian import GaussianGrid
            n_sh = 6
            dummy_grid = GaussianGrid(
                n_lat=4, n_lon=8, n_max=2, radius=1.0,
                lat=jnp.zeros(4), lon=jnp.zeros(8),
                lat2d=jnp.zeros((4, 8)), lon2d=jnp.zeros((4, 8)),
                cos_lat=jnp.ones(4), sin_lat=jnp.zeros(4),
                f=jnp.zeros((4, 8)), weights=jnp.ones(4),
                Pnm=jnp.zeros((4, n_sh)), Hnm=jnp.zeros((4, n_sh)),
                Pnm_oc2=jnp.zeros((4, n_sh)), Dnm=jnp.zeros((4, n_sh)),
                n_sh=n_sh,
                ls=jnp.zeros(n_sh, dtype=jnp.int32),
                ms=jnp.zeros(n_sh, dtype=jnp.int32),
                lap=jnp.zeros(n_sh), ilap=jnp.zeros(n_sh),
            )
            # On Metal, model should construct without raising
            # (auto-routing spectral to CPU).
            model = SpectralShallowWaterModel(dummy_grid)
            assert model._use_cpu_for_spectral is True

    def test_cpu_allows_model_creation(self):
        """Constructing SpectralShallowWaterModel on CPU should not raise."""
        with patch("legoesm.core.hardware.jax.default_backend", return_value="cpu"):
            from legoesm.atmosphere.dynamics.spectral_sw import (
                SpectralShallowWaterModel,
            )
            mock_grid = MagicMock()
            # Should not raise (the model stores grid/config, no float64 needed yet)
            SpectralShallowWaterModel(mock_grid)
