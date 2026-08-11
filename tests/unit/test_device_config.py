"""Unit tests for the device configuration and hardware-aware optimization module.

Tests auto-detection, XLA flag configuration, mixed-precision policies,
dtype selection, and optimal mesh construction.  All tests run on CPU
without requiring GPU/TPU hardware.
"""

import os
import jax
import jax.numpy as jnp
import pytest
from unittest.mock import patch, MagicMock

from legoesm.parallel.device_config import (
    HardwareConfig,
    MixedPrecisionPolicy,
    detect_devices,
    configure_jax_for_device,
    get_optimal_dtype,
    mixed_precision_policy,
    cast_for_device,
    get_optimal_mesh,
    _estimate_device_memory,
    _recommended_batch_size,
)
# The XLA-flag helpers live in runtime.backend (device_config imports them from
# there); import from their actual home, not the module they moved out of.
from legoesm.runtime.backend import (
    set_xla_flags,
    TPU_XLA_FLAGS,
    NVIDIA_GPU_XLA_FLAGS,
    AMD_GPU_XLA_FLAGS,
)


def _docstring_line_numbers(src: str) -> set[int]:
    """1-based line numbers spanned by module/class/function docstrings.

    The reintroduction guards below scan source text for forbidden XLA
    flag / config names.  The fix that removed those names left
    explanatory docstrings that *cite* them by name (so a future reader
    understands why they are gone); a docstring mention is NOT an
    executable emission and must not trip the guard.  ``#`` comments are
    already handled by ``line.split('#')`` at the call site — this only
    covers triple-quoted docstrings, which that split cannot see.
    """
    import ast as _ast

    lines: set[int] = set()
    for node in _ast.walk(_ast.parse(src)):
        if not isinstance(
            node,
            (_ast.Module, _ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef),
        ):
            continue
        if _ast.get_docstring(node, clean=False) is None:
            continue
        doc = node.body[0].value  # the docstring Constant node
        lines.update(range(doc.lineno, (doc.end_lineno or doc.lineno) + 1))
    return lines


# ============================================================================
# Hardware detection
# ============================================================================

class TestDetectDevices:
    """Tests for detect_devices() auto-detection."""

    def test_returns_hardware_config(self):
        """detect_devices returns a HardwareConfig NamedTuple."""
        config = detect_devices()
        assert isinstance(config, HardwareConfig)

    def test_backend_is_lowercase_string(self):
        """Backend name is a lowercase string."""
        config = detect_devices()
        assert isinstance(config.backend, str)
        assert config.backend == config.backend.lower()

    def test_device_count_positive(self):
        """At least one device is detected."""
        config = detect_devices()
        assert config.device_count >= 1

    def test_devices_per_host_positive(self):
        """At least one device per host."""
        config = detect_devices()
        assert config.devices_per_host >= 1
        assert config.devices_per_host <= config.device_count

    def test_num_hosts_positive(self):
        """At least one host."""
        config = detect_devices()
        assert config.num_hosts >= 1

    def test_cpu_supports_float64(self):
        """CPU backend supports float64."""
        config = detect_devices()
        if config.backend == "cpu":
            assert config.supports_float64 is True
            assert config.supports_complex128 is True

    def test_memory_positive(self):
        """Memory estimate is positive."""
        config = detect_devices()
        assert config.memory_per_device_gb > 0

    def test_recommended_batch_size_positive(self):
        """Recommended batch size is at least 1."""
        config = detect_devices()
        assert config.recommended_batch_size >= 1

    def test_tpu_detection_mock(self):
        """Simulated TPU detection returns correct backend."""
        with patch("legoesm.parallel.device_config.jax.default_backend",
                   return_value="tpu"):
            with patch("legoesm.parallel.device_config.jax.devices",
                       return_value=[MagicMock(device_kind="TPU v4")]):
                with patch("legoesm.parallel.device_config.jax.local_devices",
                           return_value=[MagicMock(device_kind="TPU v4")]):
                    with patch("legoesm.parallel.device_config.jax.process_count",
                               return_value=1):
                        config = detect_devices()
        assert config.backend == "tpu"
        assert config.supports_float64 is True

    def test_metal_detection_mock(self):
        """Simulated Apple GPU (mps) detection reports no float64."""
        with patch("legoesm.parallel.device_config.jax.default_backend",
                   return_value="mps"):
            with patch("legoesm.parallel.device_config.jax.devices",
                       return_value=[MagicMock(device_kind="Apple M2")]):
                with patch("legoesm.parallel.device_config.jax.local_devices",
                           return_value=[MagicMock(device_kind="Apple M2")]):
                    with patch("legoesm.parallel.device_config.jax.process_count",
                               return_value=1):
                        config = detect_devices()
        assert config.backend == "mps"
        assert config.supports_float64 is False
        assert config.supports_complex128 is False


# ============================================================================
# Memory estimation
# ============================================================================

class TestMemoryEstimation:
    """Tests for _estimate_device_memory heuristics."""

    def test_tpu_v4_memory(self):
        """TPU v4 devices should report 32 GB."""
        devices = [MagicMock(device_kind="TPU v4")]
        mem = _estimate_device_memory("tpu", devices)
        assert mem == 32.0

    def test_tpu_v3_memory(self):
        """TPU v3 devices should report 16 GB."""
        devices = [MagicMock(device_kind="TPU v3")]
        mem = _estimate_device_memory("tpu", devices)
        assert mem == 16.0

    def test_tpu_v5p_memory(self):
        """TPU v5p devices should report 95 GB."""
        devices = [MagicMock(device_kind="TPU v5p")]
        mem = _estimate_device_memory("tpu", devices)
        assert mem == 95.0

    def test_unknown_tpu_default(self):
        """Unknown TPU generation defaults to 16 GB."""
        devices = [MagicMock(device_kind="TPU v99")]
        mem = _estimate_device_memory("tpu", devices)
        assert mem == 16.0

    def test_gpu_a100_memory(self):
        """A100 should report 40 GB."""
        devices = [MagicMock(device_kind="NVIDIA A100")]
        mem = _estimate_device_memory("gpu", devices)
        assert mem == 40.0

    def test_gpu_h100_memory(self):
        """H100 should report 80 GB."""
        devices = [MagicMock(device_kind="NVIDIA H100")]
        mem = _estimate_device_memory("gpu", devices)
        assert mem == 80.0

    def test_unknown_gpu_default(self):
        """Unknown GPU defaults to 16 GB."""
        devices = [MagicMock(device_kind="Unknown GPU")]
        mem = _estimate_device_memory("gpu", devices)
        assert mem == 16.0

    def test_cpu_memory(self):
        """CPU reports 64 GB."""
        mem = _estimate_device_memory("cpu", [])
        assert mem == 64.0

    def test_gpu_mi250x_memory(self):
        """AMD MI250X should report 128 GB."""
        devices = [MagicMock(device_kind="AMD Instinct MI250X OAM")]
        mem = _estimate_device_memory("gpu", devices)
        assert mem == 128.0

    def test_gpu_mi300x_memory(self):
        """AMD MI300X should report 192 GB."""
        devices = [MagicMock(device_kind="AMD Instinct MI300X")]
        mem = _estimate_device_memory("gpu", devices)
        assert mem == 192.0

    def test_gpu_mi300a_memory(self):
        """AMD MI300A should report 128 GB."""
        devices = [MagicMock(device_kind="AMD Instinct MI300A")]
        mem = _estimate_device_memory("gpu", devices)
        assert mem == 128.0

    def test_metal_memory(self):
        """Apple GPU (mps) reports 8 GB."""
        mem = _estimate_device_memory("mps", [])
        assert mem == 8.0


# ============================================================================
# Batch size heuristics
# ============================================================================

class TestBatchSize:
    """Tests for _recommended_batch_size."""

    def test_high_memory(self):
        """>=80 GB should recommend 6 faces."""
        assert _recommended_batch_size(80.0) == 6
        assert _recommended_batch_size(95.0) == 6

    def test_medium_memory(self):
        """32-79 GB should recommend 4 faces."""
        assert _recommended_batch_size(32.0) == 4
        assert _recommended_batch_size(64.0) == 4

    def test_low_memory(self):
        """16-31 GB should recommend 2 faces."""
        assert _recommended_batch_size(16.0) == 2
        assert _recommended_batch_size(24.0) == 2

    def test_very_low_memory(self):
        """<16 GB should recommend 1 face."""
        assert _recommended_batch_size(8.0) == 1
        assert _recommended_batch_size(4.0) == 1


# ============================================================================
# XLA flag configuration
# ============================================================================

class TestXLAFlags:
    """Tests for set_xla_flags and configure_jax_for_device."""

    def test_set_xla_flags_adds_to_env(self):
        """set_xla_flags appends flags to XLA_FLAGS."""
        old = os.environ.get("XLA_FLAGS", "")
        try:
            os.environ["XLA_FLAGS"] = ""
            set_xla_flags({"test_flag_abc": "true"})
            assert "test_flag_abc=true" in os.environ["XLA_FLAGS"]
        finally:
            if old:
                os.environ["XLA_FLAGS"] = old
            else:
                os.environ.pop("XLA_FLAGS", None)

    def test_set_xla_flags_does_not_duplicate(self):
        """set_xla_flags does not duplicate existing flags."""
        old = os.environ.get("XLA_FLAGS", "")
        try:
            os.environ["XLA_FLAGS"] = "--test_flag_xyz=false"
            set_xla_flags({"test_flag_xyz": "true"})
            # Should not add because key already present
            assert os.environ["XLA_FLAGS"].count("test_flag_xyz") == 1
        finally:
            if old:
                os.environ["XLA_FLAGS"] = old
            else:
                os.environ.pop("XLA_FLAGS", None)

    def test_set_xla_flags_merges_with_existing(self):
        """set_xla_flags preserves existing flags while adding new ones."""
        old = os.environ.get("XLA_FLAGS", "")
        try:
            os.environ["XLA_FLAGS"] = "--existing_flag=1"
            set_xla_flags({"new_flag_qrs": "2"})
            flags = os.environ["XLA_FLAGS"]
            assert "existing_flag=1" in flags
            assert "new_flag_qrs=2" in flags
        finally:
            if old:
                os.environ["XLA_FLAGS"] = old
            else:
                os.environ.pop("XLA_FLAGS", None)

    def test_configure_cpu_does_not_crash(self):
        """configure_jax_for_device on CPU completes without error."""
        config = detect_devices()
        configure_jax_for_device(config)

    def test_cpu_backend_does_not_set_invalid_intra_op_flag(self):
        """Iter-154 regression: the legacy
        `intra_op_parallelism_threads=N` XLA_FLAGS entry is not
        recognized by current XLA and crashes JAX at first use with:

            F parse_flags_from_env.cc:234]
              Unknown flag in XLA_FLAGS:
              --intra_op_parallelism_threads=N

        `runtime.backend.configure_backend('cpu')` previously wrote
        this flag unconditionally (when XLA_FLAGS was unset), making
        the canonical CPU bootstrap crash at first `jnp.zeros(...)`.

        This test line-scans `runtime/backend.py` and the sibling
        `parallel/device_config.py` for executable (non-comment)
        emission of the `intra_op_parallelism_threads` XLA flag so
        a future refactor can't silently re-introduce the crash.
        """
        from tests.legoesm_paths import legoesm_source_path
        for rel_path in (
            "runtime/backend.py",
            "parallel/device_config.py",
        ):
            src = legoesm_source_path(rel_path).read_text()
            docstring_lines = _docstring_line_numbers(src)
            for lineno, line in enumerate(src.splitlines(), 1):
                if lineno in docstring_lines:
                    continue
                code_part = line.split("#", 1)[0]
                if "intra_op_parallelism_threads" in code_part:
                    raise AssertionError(
                        f"{rel_path}:{lineno} reintroduced the XLA "
                        f"flag `intra_op_parallelism_threads`, which "
                        f"current XLA rejects with an `Unknown flag` "
                        f"fatal error at first JAX use.  Remove it; "
                        f"XLA auto-scales CPU thread-pool size from "
                        f"`os.cpu_count()`."
                    )

    def test_configure_tpu_multihost_does_not_crash_on_jax_0_9(self):
        """Iter-149/150 regression: the canonical TPU bootstrap must NOT
        call `jax.config.update("jax_spmd_mode", "allow_all")`.

        Context: the legacy `jax_spmd_mode='allow_all'` toggle was
        removed in JAX 0.9 with the unified-sharding migration.
        Calling `jax.config.update("jax_spmd_mode", ...)` now raises
        `AttributeError: Unrecognized config option: jax_spmd_mode`.

        Historical state had two separate call sites that made this
        update in multi-host mode:
          1. `parallel.device_config._configure_tpu` (iter-149)
          2. `runtime.backend.configure_backend('tpu')` (iter-150)

        Both bootstrap paths must work without the update.  This test
        exercises both by mocking `jax.process_count() → 2` (simulating
        a 2-host TPU pod) and asserting neither entry point raises
        AttributeError on the removed config option.
        """
        from legoesm.parallel.device_config import (
            HardwareConfig, configure_jax_for_device,
        )
        from legoesm.runtime.backend import configure_backend

        # Path 1: configure_backend('tpu') — canonical runtime entry
        # point.  It queries `jax.process_count()` internally to decide
        # whether to apply the multi-host branch; mock to force the
        # >1 branch.
        with patch.object(jax, "process_count", return_value=2):
            # Must NOT raise AttributeError("jax_spmd_mode") on JAX 0.9+
            resolved = configure_backend("tpu")
            assert resolved == "tpu"

        # Path 2: configure_jax_for_device with multi-host HardwareConfig.
        # It reads num_hosts from the config directly (no jax.process_count
        # mock needed), but mock anyway to keep the two cases symmetric.
        cfg = HardwareConfig(
            backend="tpu",
            device_count=16,
            devices_per_host=8,
            num_hosts=2,
            supports_float64=False,
            supports_complex128=False,
            memory_per_device_gb=16.0,
            recommended_batch_size=6,
        )
        with patch.object(jax, "process_count", return_value=2):
            # Must NOT raise AttributeError on JAX 0.9+
            configure_jax_for_device(cfg)

        # Additional invariant: NEITHER source file may reference
        # `jax_spmd_mode` in an EXECUTABLE `jax.config.update(...)`
        # call (i.e. not inside a `#` comment).  This catches
        # re-introduction via copy-paste while letting the explanatory
        # NOTE comments the fix left behind remain.
        from tests.legoesm_paths import legoesm_source_path
        for rel_path in (
            "parallel/device_config.py",
            "runtime/backend.py",
        ):
            src = legoesm_source_path(rel_path).read_text()
            docstring_lines = _docstring_line_numbers(src)
            for lineno, line in enumerate(src.splitlines(), 1):
                if lineno in docstring_lines:
                    continue
                # Strip Python line comment before inspecting for the
                # forbidden pattern — NOTE comments mention the API by
                # name, and that is fine.
                code_part = line.split("#", 1)[0]
                if ("jax.config.update" in code_part
                        and "jax_spmd_mode" in code_part):
                    raise AssertionError(
                        f"{rel_path}:{lineno} reintroduced "
                        f"`jax.config.update(..., 'jax_spmd_mode', ...)`.  "
                        f"That call raises AttributeError on JAX 0.9+ and "
                        f"crashes the canonical multi-host TPU bootstrap."
                    )


# ============================================================================
# Optimal dtype selection
# ============================================================================

class TestGetOptimalDtype:
    """Tests for get_optimal_dtype."""

    def test_single_always_float32(self):
        """Single precision returns float32 on any backend."""
        for backend in ("cpu", "gpu", "tpu", "mps"):
            config = HardwareConfig(
                backend=backend,
                device_count=1, devices_per_host=1, num_hosts=1,
                supports_float64=backend not in ("mps",),
                supports_complex128=backend not in ("mps",),
                memory_per_device_gb=16.0,
                recommended_batch_size=2,
            )
            assert get_optimal_dtype(config, "single") == jnp.float32

    def test_half_on_tpu_is_bfloat16(self):
        """Half precision on TPU returns bfloat16."""
        config = HardwareConfig(
            backend="tpu", device_count=4, devices_per_host=4, num_hosts=1,
            supports_float64=True, supports_complex128=True,
            memory_per_device_gb=32.0, recommended_batch_size=4,
        )
        assert get_optimal_dtype(config, "half") == jnp.bfloat16

    def test_half_on_gpu_is_float16(self):
        """Half precision on GPU returns float16."""
        config = HardwareConfig(
            backend="gpu", device_count=1, devices_per_host=1, num_hosts=1,
            supports_float64=True, supports_complex128=True,
            memory_per_device_gb=16.0, recommended_batch_size=2,
        )
        assert get_optimal_dtype(config, "half") == jnp.float16

    def test_half_on_cpu_is_float32(self):
        """Half precision on CPU falls back to float32."""
        config = HardwareConfig(
            backend="cpu", device_count=1, devices_per_host=1, num_hosts=1,
            supports_float64=True, supports_complex128=True,
            memory_per_device_gb=64.0, recommended_batch_size=4,
        )
        assert get_optimal_dtype(config, "half") == jnp.float32

    def test_half_on_metal_is_float32(self):
        """Half precision on Apple GPU (mps) falls back to float32."""
        config = HardwareConfig(
            backend="mps", device_count=1, devices_per_host=1, num_hosts=1,
            supports_float64=False, supports_complex128=False,
            memory_per_device_gb=8.0, recommended_batch_size=1,
        )
        assert get_optimal_dtype(config, "half") == jnp.float32

    def test_double_on_supported(self):
        """Double precision returns float64 when supported."""
        config = HardwareConfig(
            backend="gpu", device_count=1, devices_per_host=1, num_hosts=1,
            supports_float64=True, supports_complex128=True,
            memory_per_device_gb=16.0, recommended_batch_size=2,
        )
        assert get_optimal_dtype(config, "double") == jnp.float64

    def test_double_on_metal_falls_back(self):
        """Double precision on Apple GPU (mps) falls back to float32."""
        config = HardwareConfig(
            backend="mps", device_count=1, devices_per_host=1, num_hosts=1,
            supports_float64=False, supports_complex128=False,
            memory_per_device_gb=8.0, recommended_batch_size=1,
        )
        assert get_optimal_dtype(config, "double") == jnp.float32


# ============================================================================
# Mixed-precision policy
# ============================================================================

class TestMixedPrecisionPolicy:
    """Tests for mixed_precision_policy and cast_for_device."""

    def test_tpu_policy(self):
        """TPU policy uses bfloat16 for compute."""
        config = HardwareConfig(
            backend="tpu", device_count=4, devices_per_host=4, num_hosts=1,
            supports_float64=True, supports_complex128=True,
            memory_per_device_gb=32.0, recommended_batch_size=4,
        )
        policy = mixed_precision_policy(config)
        assert isinstance(policy, MixedPrecisionPolicy)
        assert policy.compute_dtype == jnp.bfloat16
        assert policy.param_dtype == jnp.float32
        assert policy.output_dtype == jnp.float32

    def test_gpu_policy(self):
        """GPU policy uses float32 for everything."""
        config = HardwareConfig(
            backend="gpu", device_count=1, devices_per_host=1, num_hosts=1,
            supports_float64=True, supports_complex128=True,
            memory_per_device_gb=16.0, recommended_batch_size=2,
        )
        policy = mixed_precision_policy(config)
        assert policy.compute_dtype == jnp.float32
        assert policy.param_dtype == jnp.float32
        assert policy.output_dtype == jnp.float32

    def test_cpu_policy(self):
        """CPU policy uses float32 for everything."""
        config = detect_devices()
        if config.backend == "cpu":
            policy = mixed_precision_policy(config)
            assert policy.compute_dtype == jnp.float32
            assert policy.param_dtype == jnp.float32

    def test_cast_for_device_compute(self):
        """cast_for_device with role='compute' applies compute dtype."""
        config = detect_devices()
        x = jnp.ones(5, dtype=jnp.float32)
        y = cast_for_device(x, config, role="compute")
        assert y.shape == x.shape
        assert jnp.allclose(y, x)

    def test_cast_for_device_param(self):
        """cast_for_device with role='param' keeps float32."""
        config = detect_devices()
        x = jnp.ones(5, dtype=jnp.float64)
        y = cast_for_device(x, config, role="param")
        assert y.dtype == jnp.float32

    def test_cast_for_device_output(self):
        """cast_for_device with role='output' returns float32."""
        config = detect_devices()
        x = jnp.ones(5, dtype=jnp.float64)
        y = cast_for_device(x, config, role="output")
        assert y.dtype == jnp.float32

    def test_cast_noop_when_already_correct(self):
        """cast_for_device is a no-op when dtype already matches."""
        config = detect_devices()
        x = jnp.ones(5, dtype=jnp.float32)
        y = cast_for_device(x, config, role="param")
        # Should be the exact same object (no copy).
        assert y is x

    def test_cast_invalid_role_raises(self):
        """cast_for_device with unknown role raises ValueError."""
        config = detect_devices()
        x = jnp.ones(5)
        with pytest.raises(ValueError, match="Unknown role"):
            cast_for_device(x, config, role="unknown")

    def test_tpu_cast_to_bfloat16(self):
        """On TPU, compute cast should produce bfloat16."""
        config = HardwareConfig(
            backend="tpu", device_count=4, devices_per_host=4, num_hosts=1,
            supports_float64=True, supports_complex128=True,
            memory_per_device_gb=32.0, recommended_batch_size=4,
        )
        x = jnp.ones(5, dtype=jnp.float32)
        y = cast_for_device(x, config, role="compute")
        assert y.dtype == jnp.bfloat16


# ============================================================================
# Optimal mesh construction
# ============================================================================

class TestGetOptimalMesh:
    """Tests for get_optimal_mesh."""

    def test_cubed_sphere_single_device(self):
        """Single-device cubed-sphere returns valid DeviceConfig."""
        from legoesm.parallel.mesh import DeviceConfig as MeshDeviceConfig
        config = detect_devices()
        mesh_cfg = get_optimal_mesh(config, grid_type="cubed_sphere")
        assert isinstance(mesh_cfg, MeshDeviceConfig)
        assert mesh_cfg.grid_type == "cubed_sphere"

    def test_spectral_single_device(self):
        """Single-device spectral returns valid DeviceConfig."""
        from legoesm.parallel.mesh import DeviceConfig as MeshDeviceConfig
        config = detect_devices()
        mesh_cfg = get_optimal_mesh(config, grid_type="spectral", nlev=40)
        assert isinstance(mesh_cfg, MeshDeviceConfig)
        assert mesh_cfg.grid_type == "spectral"

    def test_latlon_single_device(self):
        """Single-device lat-lon returns valid DeviceConfig."""
        from legoesm.parallel.mesh import DeviceConfig as MeshDeviceConfig
        config = detect_devices()
        mesh_cfg = get_optimal_mesh(config, grid_type="latlon")
        assert isinstance(mesh_cfg, MeshDeviceConfig)
        assert mesh_cfg.grid_type == "latlon"

    def test_unknown_grid_type_raises(self):
        """Unknown grid type raises ValueError."""
        config = detect_devices()
        with pytest.raises(ValueError, match="Unknown grid_type"):
            get_optimal_mesh(config, grid_type="hexagonal")


# ============================================================================
# Re-export from parallel package
# ============================================================================

class TestReExports:
    """Verify that public API is accessible via legoesm.parallel."""

    def test_hardware_config_importable(self):
        from legoesm.parallel import HardwareConfig
        assert HardwareConfig is not None

    def test_detect_hardware_importable(self):
        from legoesm.parallel import detect_hardware
        config = detect_hardware()
        assert isinstance(config, HardwareConfig)

    def test_configure_jax_for_device_importable(self):
        from legoesm.parallel import configure_jax_for_device
        assert callable(configure_jax_for_device)

    def test_get_optimal_dtype_importable(self):
        from legoesm.parallel import get_optimal_dtype
        assert callable(get_optimal_dtype)

    def test_mixed_precision_policy_importable(self):
        from legoesm.parallel import MixedPrecisionPolicy, mixed_precision_policy
        assert callable(mixed_precision_policy)

    def test_cast_for_device_importable(self):
        from legoesm.parallel import cast_for_device
        assert callable(cast_for_device)

    def test_get_optimal_mesh_importable(self):
        from legoesm.parallel import get_optimal_mesh
        assert callable(get_optimal_mesh)


# ============================================================================
# GPU vendor detection
# ============================================================================

class TestGPUVendorDetection:
    """Tests for GPU vendor detection in backend.py."""

    def test_nvidia_detected_from_device_kind(self):
        """NVIDIA GPU detected from device_kind string."""
        from legoesm.runtime.backend import _is_nvidia_gpu, _is_amd_gpu, gpu_vendor
        with patch("jax.devices",
                   return_value=[MagicMock(device_kind="NVIDIA A100-SXM4-40GB")]):
            assert _is_nvidia_gpu()
            assert not _is_amd_gpu()
            assert gpu_vendor() == "nvidia"

    def test_amd_detected_from_device_kind(self):
        """AMD GPU detected from device_kind string."""
        from legoesm.runtime.backend import _is_nvidia_gpu, _is_amd_gpu, gpu_vendor
        with patch("jax.devices",
                   return_value=[MagicMock(device_kind="AMD Instinct MI250X OAM")]):
            assert not _is_nvidia_gpu()
            assert _is_amd_gpu()
            assert gpu_vendor() == "amd"

    def test_unknown_vendor_on_cpu(self):
        """CPU backend returns 'unknown' vendor."""
        from legoesm.runtime.backend import gpu_vendor
        with patch("jax.devices",
                   return_value=[MagicMock(device_kind="cpu")]):
            assert gpu_vendor() == "unknown"

    def test_xla_flags_nvidia_only(self):
        """NVIDIA XLA flags contain cuDNN; AMD flags do not."""
        assert "xla_gpu_cudnn_gemm_fusion_level" in NVIDIA_GPU_XLA_FLAGS
        assert "xla_gpu_cudnn_gemm_fusion_level" not in AMD_GPU_XLA_FLAGS

    @pytest.mark.parametrize("env_val,expect_collectives", [
        (None, False),   # default: COLLECTIVES excluded
        ("1", True),     # explicit opt-in restores capture
    ])
    def test_command_buffer_collectives_optin(self, env_val,
                                              expect_collectives):
        """COLLECTIVES capture is opt-in: measured +25% step time on the
        MPAS shard_map lane (job 26873637) when collectives are frozen
        into command buffers (kills dynamic compute/comm overlap).

        Fresh interpreter per case with the env var explicitly absent /
        set (import-time resolution; ambient env must not leak in —
        codex review of the first version showed an env-dependent
        assert is not a reversion guard)."""
        import subprocess, sys
        setup = ("import os; os.environ.pop("
                 "'LEGOESM_XLA_CMDBUF_COLLECTIVES', None); "
                 if env_val is None else
                 f"import os; os.environ['LEGOESM_XLA_CMDBUF_COLLECTIVES']"
                 f"='{env_val}'; ")
        check = ("'COLLECTIVES' in v" if expect_collectives
                 else "'COLLECTIVES' not in v and 'FUSION' in v "
                      "and 'CUSTOM_CALL' in v")
        code = (setup +
                "from legoesm.runtime.backend import NVIDIA_GPU_XLA_FLAGS; "
                "v=NVIDIA_GPU_XLA_FLAGS['xla_gpu_enable_command_buffer']; "
                f"assert {check}, v; print('OK')")
        env = {k: v for k, v in os.environ.items()
               if k != "LEGOESM_XLA_CMDBUF_COLLECTIVES"}
        out = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True)
        assert out.returncode == 0 and "OK" in out.stdout, out.stderr
