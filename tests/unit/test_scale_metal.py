"""Category 9: Apple Metal compatibility tests.

Tests hardware detection, Metal float64 limitations, CPU fallback,
and spectral routing on the Metal backend.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.parallel.metal import (
    MetalConfig,
    get_metal_config,
    to_cpu,
    to_metal,
    route_to_cpu,
    route_to_default,
    is_metal_backend,
    ensure_spectral_on_cpu,
)
from legoesm.core.hardware import (
    get_backend,
    detect_devices,
    _UNSUPPORTED_F64_BACKENDS,
)


# ---------------------------------------------------------------------------
# MetalConfig detection
# ---------------------------------------------------------------------------

class TestMetalConfig:
    def test_config_is_named_tuple(self):
        config = get_metal_config()
        assert isinstance(config, MetalConfig)
        assert hasattr(config, 'metal_device')
        assert hasattr(config, 'cpu_device')
        assert hasattr(config, 'is_metal')

    def test_cpu_device_always_available(self):
        config = get_metal_config()
        assert config.cpu_device is not None

    def test_is_metal_flag_matches_backend(self):
        config = get_metal_config()
        backend = get_backend()
        if backend == "METAL":
            assert config.is_metal is True
            assert config.metal_device is not None
        else:
            assert config.is_metal is False
            assert config.metal_device is None


# ---------------------------------------------------------------------------
# to_cpu / to_metal device transfer
# ---------------------------------------------------------------------------

class TestDeviceTransfer:
    def test_to_cpu_moves_to_cpu(self):
        x = jnp.ones(5)
        x_cpu = to_cpu(x)
        # Should be on CPU device
        assert x_cpu.devices().pop().platform == "cpu"

    def test_to_cpu_preserves_values(self):
        x = jnp.array([1.0, 2.0, 3.0])
        x_cpu = to_cpu(x)
        np.testing.assert_array_equal(x_cpu, x)

    def test_to_cpu_preserves_dtype(self):
        x = jnp.ones(5, dtype=jnp.float32)
        x_cpu = to_cpu(x)
        assert x_cpu.dtype == jnp.float32

    def test_to_metal_on_non_metal_is_identity(self):
        """On non-Metal backends, to_metal should return input unchanged."""
        backend = get_backend()
        if backend == "METAL":
            pytest.skip("Running on Metal — to_metal is not identity")
        x = jnp.ones(5)
        x_out = to_metal(x)
        np.testing.assert_array_equal(x_out, x)


# ---------------------------------------------------------------------------
# route_to_cpu / route_to_default (pytree transfer)
# ---------------------------------------------------------------------------

class TestRouteTransfer:
    def test_route_to_cpu_pytree(self):
        tree = {"a": jnp.ones(3), "b": jnp.zeros(2)}
        out = route_to_cpu(tree)
        for key in tree:
            assert out[key].devices().pop().platform == "cpu"
            np.testing.assert_array_equal(out[key], tree[key])

    def test_route_to_default_pytree(self):
        tree = {"a": jnp.ones(3)}
        out = route_to_default(tree)
        default_dev = jax.devices()[0]
        for key in tree:
            assert out[key].devices().pop() == default_dev


# ---------------------------------------------------------------------------
# is_metal_backend
# ---------------------------------------------------------------------------

class TestIsMetalBackend:
    def test_returns_bool(self):
        result = is_metal_backend()
        assert isinstance(result, bool)

    def test_matches_get_backend(self):
        assert is_metal_backend() == (get_backend() == "METAL")


# ---------------------------------------------------------------------------
# ensure_spectral_on_cpu
# ---------------------------------------------------------------------------

class TestEnsureSpectralOnCpu:
    def test_non_metal_is_noop(self):
        """On non-Metal, ensure_spectral_on_cpu should return fn unchanged."""
        backend = get_backend()
        if backend == "METAL":
            pytest.skip("Running on Metal")

        def f(x):
            return x * 2.0

        wrapped = ensure_spectral_on_cpu(f)
        # On non-Metal, should be the same function
        assert wrapped is f

    def test_wrapped_fn_correct_result(self):
        """The wrapped function should produce the correct result."""
        def f(x):
            return x * 2.0 + 1.0

        wrapped = ensure_spectral_on_cpu(f)
        x = jnp.array([1.0, 2.0, 3.0])
        result = wrapped(x)
        expected = x * 2.0 + 1.0
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

class TestHardwareDetection:
    def test_detect_devices_returns_dict(self):
        info = detect_devices()
        assert isinstance(info, dict)
        assert "backend" in info
        assert "n_devices" in info
        assert "devices" in info
        assert "supports_f64" in info
        assert "distributed" in info

    def test_n_devices_positive(self):
        info = detect_devices()
        assert info["n_devices"] >= 1

    def test_backend_is_string(self):
        assert isinstance(get_backend(), str)

    def test_unsupported_f64_backends_is_frozenset(self):
        assert isinstance(_UNSUPPORTED_F64_BACKENDS, frozenset)
        assert "METAL" in _UNSUPPORTED_F64_BACKENDS


# ---------------------------------------------------------------------------
# Metal float64 limitations
# ---------------------------------------------------------------------------

class TestMetalFloat64:
    def test_cpu_supports_float64(self):
        """CPU should always support float64."""
        cpu = jax.devices("cpu")[0]
        x = jax.device_put(jnp.array(1.0, dtype=jnp.float64), cpu)
        assert x.dtype == jnp.float64

    def test_cpu_supports_complex128(self):
        """CPU should support complex128 for spectral transforms."""
        cpu = jax.devices("cpu")[0]
        x = jax.device_put(jnp.array(1.0 + 2.0j, dtype=jnp.complex128), cpu)
        assert x.dtype == jnp.complex128
