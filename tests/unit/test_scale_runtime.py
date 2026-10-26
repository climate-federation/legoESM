"""Category 14: Runtime configuration & hardware detection.

Tests ParallelRuntime, HardwareConfig, and XLA configuration.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.parallel.runtime import (
    ParallelRuntime,
    validate_device_count,
    HaloBackend,
    ReductionBackend,
)
from legoesm.parallel.device_config import (
    detect_devices,
    HardwareConfig,
    mixed_precision_policy,
    get_optimal_dtype,
    MixedPrecisionPolicy,
)


# =========================================================================
# 14a-c) ParallelRuntime
# =========================================================================

class TestParallelRuntime:
    """ParallelRuntime factory and modes."""

    def test_serial_creation(self):
        """Serial runtime should have rank=0, world_size=1."""
        rt = ParallelRuntime.create(grid_type="cubed_sphere", n_devices=1)
        assert rt.rank == 0
        assert rt.world_size == 1
        desc = rt.describe()
        assert desc["mode"] in ("serial", "multi_device")

    def test_describe_has_required_keys(self):
        """describe() should return a dict with expected keys."""
        rt = ParallelRuntime.create(n_devices=1)
        desc = rt.describe()
        for key in ["mode", "rank", "world_size", "local_device_count"]:
            assert key in desc, f"Missing key: {key}"


# =========================================================================
# 14d) HardwareConfig detection
# =========================================================================

class TestHardwareDetection:
    """detect_devices should return valid HardwareConfig."""

    def test_returns_hardware_config(self):
        config = detect_devices()
        assert isinstance(config, HardwareConfig)

    def test_device_count_positive(self):
        config = detect_devices()
        assert config.device_count >= 1

    def test_backend_string(self):
        config = detect_devices()
        assert config.backend.lower() in ("cpu", "gpu", "tpu", "mps")

    def test_memory_positive(self):
        config = detect_devices()
        assert config.memory_per_device_gb > 0


# =========================================================================
# 14g) Mixed precision policy
# =========================================================================

class TestMixedPrecision:
    """mixed_precision_policy should return valid dtypes."""

    def test_returns_policy(self):
        config = detect_devices()
        policy = mixed_precision_policy(config)
        assert isinstance(policy, MixedPrecisionPolicy)

    def test_valid_dtypes(self):
        config = detect_devices()
        policy = mixed_precision_policy(config)
        valid_dtypes = {jnp.float16, jnp.float32, jnp.float64, jnp.bfloat16}
        for dt in [policy.compute_dtype, policy.param_dtype, policy.output_dtype]:
            assert dt in valid_dtypes, f"Unexpected dtype: {dt}"

    def test_get_optimal_dtype(self):
        config = detect_devices()
        dt = get_optimal_dtype(config, precision="single")
        assert dt in (jnp.float32, jnp.float64, jnp.bfloat16, jnp.float16)


# =========================================================================
# Backend constants
# =========================================================================

class TestBackendConstants:
    """HaloBackend and ReductionBackend have expected values."""

    def test_halo_backends(self):
        assert HaloBackend.LOCAL == "local"
        assert HaloBackend.MPI == "mpi"

    def test_reduction_backends(self):
        assert ReductionBackend.LOCAL == "local"
        assert ReductionBackend.MPI == "mpi"
