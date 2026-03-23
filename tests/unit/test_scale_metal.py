"""Category 9: Apple Metal compatibility.

Tests hardware detection, CPU/GPU routing, and spectral CPU fallback.
Most tests work on any backend by checking the routing logic.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.parallel.metal import (
    get_metal_config,
    to_cpu,
    is_metal_backend,
    route_to_cpu,
    route_to_default,
    ensure_spectral_on_cpu,
)
from legoesm.parallel.device_config import detect_devices, HardwareConfig


IS_METAL = is_metal_backend()


# =========================================================================
# 9a) Hardware detection
# =========================================================================

class TestHardwareDetection:
    """detect_devices should return valid config."""

    def test_backend_detected(self):
        config = detect_devices()
        assert config.backend.lower() in ("cpu", "gpu", "tpu", "metal")

    def test_device_count_positive(self):
        config = detect_devices()
        assert config.device_count >= 1


# =========================================================================
# 9b) Metal float64 detection
# =========================================================================

class TestMetalFloat64:
    """Metal should not support float64."""

    @pytest.mark.skipif(not IS_METAL, reason="Metal only")
    def test_no_float64_on_metal(self):
        config = detect_devices()
        assert not config.supports_float64

    @pytest.mark.skipif(IS_METAL, reason="Non-Metal only")
    def test_float64_on_non_metal(self):
        config = detect_devices()
        # CPU and CUDA GPU support float64 when x64 enabled
        if jax.config.jax_enable_x64:
            assert config.supports_float64


# =========================================================================
# 9c) CPU/GPU routing
# =========================================================================

class TestDeviceRouting:
    """to_cpu and route_to_cpu should preserve values."""

    def test_to_cpu_preserves_values(self):
        x = jnp.array([1.0, 2.0, 3.0])
        x_cpu = to_cpu(x)
        np.testing.assert_allclose(np.array(x_cpu), [1.0, 2.0, 3.0])

    def test_route_to_cpu_pytree(self):
        state = {"a": jnp.ones(5), "b": jnp.zeros(3)}
        state_cpu = route_to_cpu(state)
        np.testing.assert_allclose(np.array(state_cpu["a"]), 1.0)
        np.testing.assert_allclose(np.array(state_cpu["b"]), 0.0)

    def test_route_roundtrip(self):
        """route_to_cpu then route_to_default should preserve values."""
        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        state = {"x": x}
        state_cpu = route_to_cpu(state)
        state_back = route_to_default(state_cpu)
        np.testing.assert_allclose(np.array(state_back["x"]), np.array(x))


# =========================================================================
# 9d) Spectral CPU fallback
# =========================================================================

class TestSpectralCPUFallback:
    """ensure_spectral_on_cpu decorator should route computation correctly."""

    def test_decorator_preserves_result(self):
        """Decorated function should produce same result as undecorated."""

        def compute(x):
            return x ** 2 + 1.0

        decorated = ensure_spectral_on_cpu(compute)

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        result_direct = compute(x)
        result_decorated = decorated(x)

        np.testing.assert_allclose(
            np.array(result_decorated), np.array(result_direct), atol=1e-12
        )

    def test_decorator_with_kwargs(self):
        """Decorator should handle kwargs correctly."""

        def compute(x, scale=2.0):
            return x * scale

        decorated = ensure_spectral_on_cpu(compute)
        x = jnp.array([1.0, 2.0], dtype=jnp.float64)
        result = decorated(x, scale=3.0)
        np.testing.assert_allclose(np.array(result), [3.0, 6.0])


# =========================================================================
# 9e) MetalConfig
# =========================================================================

class TestMetalConfig:
    """get_metal_config should return valid config."""

    def test_config_has_cpu_device(self):
        config = get_metal_config()
        assert config.cpu_device is not None

    def test_is_metal_flag(self):
        config = get_metal_config()
        if IS_METAL:
            assert config.is_metal
        else:
            assert not config.is_metal
