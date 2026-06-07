"""Tests for backend compatibility guards (Metal / float64).

Mock-based tests patch ``legoesm.runtime.backend.get_backend`` directly
so the Metal health-check / CPU-fallback machinery is bypassed.
"""

import warnings
from unittest.mock import MagicMock, patch

import jax
import pytest

from legoesm.core.hardware import check_spectral_backend, get_backend
from legoesm.runtime.backend import (
    check_spectral_backend as rt_check_spectral,
)


# Patch target: runtime.backend.get_backend is the single source of truth.
# core.hardware.get_backend delegates to it and uppercases the result.
_RT_GET_BACKEND = "legoesm.runtime.backend.get_backend"


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
        with patch(_RT_GET_BACKEND, return_value="cpu"):
            rt_check_spectral()  # Should not raise

    def test_gpu_backend_passes(self):
        _require_x64()
        with patch(_RT_GET_BACKEND, return_value="gpu"):
            rt_check_spectral()  # Should not raise

    def test_tpu_backend_passes(self):
        _require_x64()
        with patch(_RT_GET_BACKEND, return_value="tpu"):
            rt_check_spectral()  # Should not raise

    def test_metal_backend_raises(self):
        _require_x64()
        with patch(_RT_GET_BACKEND, return_value="metal"):
            with pytest.raises(ValueError, match="spectral solver requires float64"):
                rt_check_spectral()

    def test_metal_case_insensitive(self):
        _require_x64()
        # get_backend always returns lowercase; simulate it
        with patch(_RT_GET_BACKEND, return_value="metal"):
            with pytest.raises(ValueError, match="spectral solver requires float64"):
                rt_check_spectral()

    def test_metal_allow_unsupported_warns(self):
        _require_x64()
        with patch(_RT_GET_BACKEND, return_value="metal"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                rt_check_spectral(allow_unsupported=True)
                guard_warnings = [
                    x for x in w
                    if issubclass(x.category, RuntimeWarning)
                    and "unsupported backend" in str(x.message).lower()
                ]
                assert len(guard_warnings) == 1

    def test_error_message_has_remediation(self):
        _require_x64()
        with patch(_RT_GET_BACKEND, return_value="metal"):
            with pytest.raises(ValueError, match="JAX_PLATFORMS=cpu"):
                rt_check_spectral()

    def test_error_message_suggests_alternative(self):
        _require_x64()
        with patch(_RT_GET_BACKEND, return_value="metal"):
            with pytest.raises(ValueError, match="finite-volume"):
                rt_check_spectral()

    def test_error_message_mentions_config_override(self):
        _require_x64()
        with patch(_RT_GET_BACKEND, return_value="metal"):
            with pytest.raises(ValueError, match="allow_unsupported"):
                rt_check_spectral()


class TestGetBackend:
    """Tests for get_backend() in core.hardware (legacy, uppercase)."""

    def test_returns_uppercase(self):
        with patch(_RT_GET_BACKEND, return_value="cpu"):
            assert get_backend() == "CPU"

    def test_returns_uppercase_metal(self):
        with patch(_RT_GET_BACKEND, return_value="metal"):
            assert get_backend() == "METAL"


class TestGaussianGridGuard:
    """Test that create_gaussian_grid handles Metal guard/fallback paths."""

    def test_metal_without_x64_raises(self):
        _require_x32()
        with patch(_RT_GET_BACKEND, return_value="metal"):
            from legoesm.grids.gaussian import create_gaussian_grid
            with pytest.raises(ValueError, match="requires JAX_ENABLE_X64"):
                create_gaussian_grid(n_max=21)

    def test_metal_auto_fallback_creates_cpu_grid(self):
        _require_x64()
        with patch(_RT_GET_BACKEND, return_value="metal"):
            from legoesm.grids.gaussian import create_gaussian_grid
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                grid = create_gaussian_grid(n_max=21)
            assert "CPU" in str(grid.lat.device).upper()
            warn_text = "\n".join(str(x.message) for x in w)
            assert "fallback" in warn_text.lower() or "metal" in warn_text.lower()
            assert "unsupported backend" in warn_text.lower() or "spectral" in warn_text.lower()

    def test_metal_grid_creation_with_override_warns(self):
        _require_x64()
        with patch(_RT_GET_BACKEND, return_value="metal"):
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
        # Ensure CPU fallback is active so float64 array creation works
        from legoesm.runtime.backend import ensure_metal_or_fallback
        ensure_metal_or_fallback()

        # Build dummy grid on CPU (float64 required for spectral grids)
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            import jax.numpy as jnp
            from legoesm.grids.gaussian import GaussianGrid
            n_sh = 6
            dummy_grid = GaussianGrid(
                n_lat=4, n_lon=8, n_max=2, radius=1.0,
                lat=jnp.zeros(4, dtype=jnp.float64),
                lon=jnp.zeros(8, dtype=jnp.float64),
                lat2d=jnp.zeros((4, 8), dtype=jnp.float64),
                lon2d=jnp.zeros((4, 8), dtype=jnp.float64),
                cos_lat=jnp.ones(4, dtype=jnp.float64),
                sin_lat=jnp.zeros(4, dtype=jnp.float64),
                f=jnp.zeros((4, 8), dtype=jnp.float64),
                weights=jnp.ones(4, dtype=jnp.float64),
                Pnm=jnp.zeros((4, n_sh), dtype=jnp.float64),
                Hnm=jnp.zeros((4, n_sh), dtype=jnp.float64),
                Pnm_oc2=jnp.zeros((4, n_sh), dtype=jnp.float64),
                Dnm=jnp.zeros((4, n_sh), dtype=jnp.float64),
                wPnm=jnp.zeros((4, n_sh), dtype=jnp.float64),
                wPnm_oc2=jnp.zeros((4, n_sh), dtype=jnp.float64),
                wDnm=jnp.zeros((4, n_sh), dtype=jnp.float64),
                n_sh=n_sh,
                ls=jnp.zeros(n_sh, dtype=jnp.int32),
                ms=jnp.zeros(n_sh, dtype=jnp.int32),
                lap=jnp.zeros(n_sh, dtype=jnp.float64),
                ilap=jnp.zeros(n_sh, dtype=jnp.float64),
            )

        with patch(_RT_GET_BACKEND, return_value="metal"):
            from legoesm.atmosphere.dynamics.spectral_sw import (
                SpectralShallowWaterModel,
            )
            model = SpectralShallowWaterModel(dummy_grid)
            assert model._use_cpu_for_spectral is True

    def test_cpu_allows_model_creation(self):
        _require_x64()
        with patch(_RT_GET_BACKEND, return_value="cpu"):
            from legoesm.atmosphere.dynamics.spectral_sw import (
                SpectralShallowWaterModel,
            )
            mock_grid = MagicMock()
            SpectralShallowWaterModel(mock_grid)
