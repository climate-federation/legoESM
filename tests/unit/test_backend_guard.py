"""Tests for backend compatibility guards (Metal / float64)."""

import warnings
from unittest.mock import MagicMock, patch

import jax
import pytest

from legoesm.core.hardware import check_spectral_backend, get_backend


def _require_x64() -> None:
    if not jax.config.jax_enable_x64:
        pytest.skip("requires JAX_ENABLE_X64=True")


def _require_x32() -> None:
    if jax.config.jax_enable_x64:
        pytest.skip("requires JAX_ENABLE_X64=False")


class TestCheckSpectralBackend:
    """Tests for check_spectral_backend()."""

    def test_cpu_backend_passes(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="cpu"):
            check_spectral_backend()  # Should not raise

    def test_gpu_backend_passes(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="gpu"):
            check_spectral_backend()  # Should not raise

    def test_tpu_backend_passes(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="tpu"):
            check_spectral_backend()  # Should not raise

    def test_metal_backend_raises(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            with pytest.raises(ValueError, match="spectral solver requires float64"):
                check_spectral_backend()

    def test_metal_case_insensitive(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="metal"):
            with pytest.raises(ValueError, match="spectral solver requires float64"):
                check_spectral_backend()

    def test_metal_allow_unsupported_warns(self):
        _require_x64()
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
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            with pytest.raises(ValueError, match="JAX_PLATFORMS=cpu"):
                check_spectral_backend()

    def test_error_message_suggests_alternative(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            with pytest.raises(ValueError, match="finite-volume"):
                check_spectral_backend()

    def test_error_message_mentions_config_override(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            with pytest.raises(ValueError, match="allow_unsupported"):
                check_spectral_backend()


class TestGetBackend:
    """Tests for get_backend()."""

    def test_returns_uppercase(self):
        with patch("legoesm.core.hardware.jax.default_backend", return_value="cpu"):
            assert get_backend() == "CPU"

    def test_returns_uppercase_metal(self):
        with patch("legoesm.core.hardware.jax.default_backend", return_value="metal"):
            assert get_backend() == "METAL"


class TestGaussianGridGuard:
    """Test that create_gaussian_grid handles Metal guard/fallback paths."""

    def test_metal_without_x64_raises(self):
        _require_x32()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            from legoesm.grids.gaussian import create_gaussian_grid
            with pytest.raises(ValueError, match="requires JAX_ENABLE_X64=True"):
                create_gaussian_grid(n_max=21)

    def test_metal_auto_fallback_creates_cpu_grid(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            from legoesm.grids.gaussian import create_gaussian_grid
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                grid = create_gaussian_grid(n_max=21)
            assert "CPU" in str(grid.lat.device).upper()
            warn_text = "\n".join(str(x.message) for x in w)
            assert "fallback" in warn_text.lower()
            assert "unsupported backend" in warn_text.lower()

    def test_metal_grid_creation_with_override_warns(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            from legoesm.grids.gaussian import create_gaussian_grid
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                create_gaussian_grid(n_max=21, allow_unsupported_backend=True)
                guard_warnings = [
                    x for x in w
                    if issubclass(x.category, RuntimeWarning)
                    and "unsupported backend" in str(x.message).lower()
                ]
                assert len(guard_warnings) >= 1


class TestSpectralSWModelGuard:
    """Test that SpectralShallowWaterModel.__init__ applies backend guard."""

    def test_metal_auto_routes_to_cpu(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="METAL"):
            from legoesm.atmosphere.dynamics.spectral_sw import (
                SpectralShallowWaterModel,
            )
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
            model = SpectralShallowWaterModel(dummy_grid)
            assert model._use_cpu_for_spectral is True

    def test_cpu_allows_model_creation(self):
        _require_x64()
        with patch("legoesm.core.hardware.jax.default_backend", return_value="cpu"):
            from legoesm.atmosphere.dynamics.spectral_sw import (
                SpectralShallowWaterModel,
            )
            mock_grid = MagicMock()
            SpectralShallowWaterModel(mock_grid)
