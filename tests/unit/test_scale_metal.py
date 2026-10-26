"""Category 9: Apple GPU (Metal / jax-mps) compatibility tests.

Tests hardware detection, the Apple GPU (``mps``) float64 limitation,
and spectral routing.  The ``mps`` backend (jax-mps / MLX) is float32-only,
so spectral transforms must run on CPU — these tests verify that path.
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
from legoesm.runtime import get_backend, supports_float64


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
        if backend == "mps":
            # On the Apple GPU (mps) backend
            assert config.is_metal is True
            assert config.metal_device is not None
        else:
            # Not on the Apple GPU (mps) backend
            assert config.is_metal is False
            assert config.metal_device is None


# ---------------------------------------------------------------------------
# to_cpu / to_metal device transfer
# ---------------------------------------------------------------------------

class TestDeviceTransfer:
    def test_to_cpu_moves_to_cpu(self):
        # Explicit float32 so this works on all backends including mps
        x = jnp.ones(5, dtype=jnp.float32)
        x_cpu = to_cpu(x)
        # Should be on CPU device
        assert x_cpu.devices().pop().platform == "cpu"

    def test_to_cpu_preserves_values(self):
        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        x_cpu = to_cpu(x)
        np.testing.assert_array_equal(x_cpu, x)

    def test_to_cpu_preserves_dtype(self):
        x = jnp.ones(5, dtype=jnp.float32)
        x_cpu = to_cpu(x)
        assert x_cpu.dtype == jnp.float32

    def test_to_metal_on_non_metal_is_identity(self):
        """On non-mps backends, to_metal should return input unchanged."""
        backend = get_backend()
        if backend == "mps":
            pytest.skip("Running on mps — to_metal is not identity")
        x = jnp.ones(5, dtype=jnp.float32)
        x_out = to_metal(x)
        np.testing.assert_array_equal(x_out, x)


# ---------------------------------------------------------------------------
# route_to_cpu / route_to_default (pytree transfer)
# ---------------------------------------------------------------------------

class TestRouteTransfer:
    def test_route_to_cpu_pytree(self):
        tree = {"a": jnp.ones(3, dtype=jnp.float32),
                "b": jnp.zeros(2, dtype=jnp.float32)}
        out = route_to_cpu(tree)
        for key in tree:
            assert out[key].devices().pop().platform == "cpu"
            np.testing.assert_array_equal(out[key], tree[key])

    def test_route_to_default_pytree(self):
        tree = {"a": jnp.ones(3, dtype=jnp.float32)}
        out = route_to_default(tree)
        # Routes to the backend's default device
        for key in tree:
            assert out[key].shape == tree[key].shape
            np.testing.assert_array_equal(out[key], tree[key])


# ---------------------------------------------------------------------------
# is_metal_backend
# ---------------------------------------------------------------------------

class TestIsMetalBackend:
    def test_returns_bool(self):
        result = is_metal_backend()
        assert isinstance(result, bool)

    def test_matches_get_backend(self):
        # Both use runtime.backend.get_backend() (lowercase) under the hood.
        from legoesm.runtime.backend import get_backend as rt_get_backend
        assert is_metal_backend() == (rt_get_backend() == "mps")


# ---------------------------------------------------------------------------
# ensure_spectral_on_cpu
# ---------------------------------------------------------------------------

class TestEnsureSpectralOnCpu:
    def test_non_metal_is_noop(self):
        """On non-mps backends, ensure_spectral_on_cpu is a no-op."""
        if is_metal_backend():
            pytest.skip("Running on the Apple GPU (mps) backend")

        def f(x):
            return x * 2.0

        wrapped = ensure_spectral_on_cpu(f)
        # On non-mps backends, should be the same function
        assert wrapped is f

    def test_wrapped_fn_correct_result(self):
        """The wrapped function should produce the correct result."""
        def f(x):
            return x * 2.0 + 1.0

        wrapped = ensure_spectral_on_cpu(f)
        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        result = wrapped(x)
        expected = x * 2.0 + 1.0
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

class TestHardwareDetection:
    def test_backend_is_string(self):
        assert isinstance(get_backend(), str)

    def test_mps_has_no_float64(self):
        assert not supports_float64("mps")


# ---------------------------------------------------------------------------
# Metal float64 limitations
# ---------------------------------------------------------------------------

class TestMetalFloat64:
    def test_cpu_supports_float64(self):
        """CPU should always support float64."""
        if not jax.config.jax_enable_x64:
            pytest.skip("JAX_ENABLE_X64 not set")
        cpu = jax.devices("cpu")[0]
        # Create via numpy to avoid the default device (may be mps, no f64)
        import numpy as _np
        x = jax.device_put(_np.float64(1.0), cpu)
        assert x.dtype == jnp.float64

    def test_cpu_supports_complex128(self):
        """CPU should support complex128 for spectral transforms."""
        if not jax.config.jax_enable_x64:
            pytest.skip("JAX_ENABLE_X64 not set")
        cpu = jax.devices("cpu")[0]
        import numpy as _np
        x = jax.device_put(_np.complex128(1.0 + 2.0j), cpu)
        assert x.dtype == jnp.complex128
