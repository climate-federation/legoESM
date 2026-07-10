"""Tests for the canonical legoesm.runtime bootstrap layer.

Covers:
- Backend detection and query helpers
- X64 enabling / disabling
- Precision policy resolution and application
- Distributed bootstrap guardrails
- Legacy wrapper backward compatibility
- No circular imports
"""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest


# =========================================================================
# 1. Backend selection (runtime.backend)
# =========================================================================

class TestBackendDetection:
    """Tests for runtime.backend helpers."""

    def test_get_backend_returns_lowercase(self):
        from legoesm.runtime.backend import get_backend
        backend = get_backend()
        assert backend == backend.lower()
        assert backend in ("cpu", "gpu", "tpu", "mps")

    def test_supports_float64_cpu(self):
        from legoesm.runtime.backend import supports_float64
        # CPU always supports f64.
        assert supports_float64("cpu") is True

    def test_supports_float64_mps(self):
        from legoesm.runtime.backend import supports_float64
        assert supports_float64("mps") is False

    def test_supports_float64_gpu(self):
        from legoesm.runtime.backend import supports_float64
        assert supports_float64("gpu") is True

    def test_supports_float64_default(self):
        from legoesm.runtime.backend import supports_float64
        # Default uses current backend; should not raise.
        result = supports_float64()
        assert isinstance(result, bool)

    def test_configure_backend_returns_lowercase(self):
        from legoesm.runtime.backend import configure_backend
        backend = configure_backend()
        assert backend == backend.lower()

    def test_configure_backend_with_explicit(self):
        from legoesm.runtime.backend import configure_backend
        # Passing "CPU" should normalise to "cpu".
        result = configure_backend("CPU")
        assert result == "cpu"


# =========================================================================
# 2. X64 policy (runtime.backend)
# =========================================================================

class TestX64Policy:
    """Tests for X64 enabling."""

    def test_enable_x64_idempotent(self):
        from legoesm.runtime.backend import enable_x64, is_x64_enabled
        enable_x64(quiet=True)
        assert is_x64_enabled() is True
        # Calling again is fine.
        enable_x64(quiet=True)
        assert is_x64_enabled() is True

    def test_is_x64_enabled_returns_bool(self):
        from legoesm.runtime.backend import is_x64_enabled
        result = is_x64_enabled()
        assert isinstance(result, bool)


# =========================================================================
# 3. Precision policy resolution (runtime.precision)
# =========================================================================

class TestPrecisionResolution:
    """Tests for precision mode resolution and application."""

    def test_resolve_fp32(self):
        from legoesm.runtime.precision import resolve_precision, PrecisionPolicy
        policy = resolve_precision("fp32")
        assert policy.storage == jnp.float32
        assert policy.compute == jnp.float32
        assert policy.accumulate == jnp.float32
        assert policy.control == jnp.float32

    def test_resolve_fp64(self):
        from legoesm.runtime.precision import resolve_precision
        policy = resolve_precision("fp64")
        assert policy.storage == jnp.float64
        assert policy.compute == jnp.float64
        assert policy.accumulate == jnp.float64
        assert policy.control == jnp.float64

    def test_resolve_mixed(self):
        from legoesm.runtime.precision import resolve_precision
        policy = resolve_precision("mixed")
        assert policy.storage == jnp.float32
        assert policy.compute == jnp.float32
        assert policy.accumulate == jnp.float64
        assert policy.control == jnp.float64

    def test_resolve_float32_alias(self):
        from legoesm.runtime.precision import resolve_precision
        p1 = resolve_precision("fp32")
        p2 = resolve_precision("float32")
        assert p1 == p2

    def test_resolve_unknown_raises(self):
        from legoesm.runtime.precision import resolve_precision
        with pytest.raises(ValueError, match="Unknown precision mode"):
            resolve_precision("int8")

    def test_apply_precision_sets_global_policy(self):
        from legoesm.runtime.precision import apply_precision, get_policy
        policy = apply_precision("fp32")
        assert get_policy() == policy

    def test_apply_precision_mixed_sets_overrides(self):
        from legoesm.runtime.precision import (
            apply_precision,
            get_module_overrides,
            clear_module_overrides,
        )
        clear_module_overrides()
        apply_precision("mixed")
        overrides = get_module_overrides()
        # Mixed mode should set recommended overrides for key modules.
        assert len(overrides) > 0


# =========================================================================
# 4. Full bootstrap (runtime.config)
# =========================================================================

class TestBootstrap:
    """Tests for the one-shot bootstrap function."""

    def test_bootstrap_returns_runtime_config(self):
        from legoesm.runtime.config import bootstrap, RuntimeConfig
        rc = bootstrap(precision="fp32")
        assert isinstance(rc, RuntimeConfig)
        assert rc.backend == rc.backend.lower()
        assert isinstance(rc.x64, bool)
        assert isinstance(rc.distributed, bool)

    def test_bootstrap_fp32_defaults(self):
        from legoesm.runtime.config import bootstrap
        rc = bootstrap(precision="fp32")
        assert rc.precision.compute == jnp.float32
        assert rc.precision.storage == jnp.float32

    def test_bootstrap_fp64_enables_x64(self):
        from legoesm.runtime.config import bootstrap
        rc = bootstrap(precision="fp64")
        assert rc.x64 is True
        assert rc.precision.compute == jnp.float64

    def test_bootstrap_mixed_enables_x64(self):
        from legoesm.runtime.config import bootstrap
        rc = bootstrap(precision="mixed")
        assert rc.x64 is True
        assert rc.precision.accumulate == jnp.float64
        assert rc.precision.compute == jnp.float32

    def test_bootstrap_sets_singleton(self):
        from legoesm.runtime.config import bootstrap, get_runtime_config
        rc = bootstrap(precision="fp32")
        assert get_runtime_config() is rc

    def test_bootstrap_explicit_x64_false_no_enable_call(self):
        """When x64=False, bootstrap must not call enable_x64."""
        from legoesm.runtime.config import bootstrap
        with patch("legoesm.runtime.backend.enable_x64") as mock_enable:
            rc = bootstrap(precision="fp32", x64=False)
            mock_enable.assert_not_called()

    def test_bootstrap_with_backend_override(self):
        from legoesm.runtime.config import bootstrap
        rc = bootstrap(precision="fp32", backend="cpu")
        assert rc.backend == "cpu"


# =========================================================================
# 5. Distributed bootstrap guardrails
# =========================================================================

class TestDistributedGuardrails:
    """Tests that distributed mode correctly gates MPI init."""

    def test_bootstrap_distributed_false_no_mpi(self):
        """Non-distributed mode should not call initialize_distributed."""
        from legoesm.runtime.config import bootstrap
        with patch("legoesm.parallel.distributed.initialize_distributed") as mock_init:
            rc = bootstrap(precision="fp32", distributed=False)
            mock_init.assert_not_called()
            assert rc.distributed is False

    def test_bootstrap_distributed_true_calls_mpi(self):
        """Distributed mode should attempt MPI initialization."""
        mock_dev_config = MagicMock()
        mock_dev_config.n_devices = 1
        with patch(
            "legoesm.parallel.distributed.initialize_distributed",
            return_value=mock_dev_config,
        ) as mock_init:
            from legoesm.runtime.config import bootstrap
            rc = bootstrap(precision="fp32", distributed=True)
            mock_init.assert_called_once()
            assert rc.distributed is True

    def test_distributed_mpi_coordinator_setup_multinode(self, monkeypatch):
        """Multi-node MPI should call jax.distributed.initialize with coordinator.

        The multi-node branch delegates to
        ``initialize_jax_distributed_multiprocess`` (#749), whose mpi4py hook
        is ``_require_mpi4py`` — mocked here alongside ``require_mpi_stack``.
        Launcher-size env vars are cleared so the helper probes the mocked
        COMM_WORLD instead of short-circuiting to single-process, and the
        scheduler-jobid vars so ``resolve_coordinator_port`` (env override /
        job-id-derived / legacy 1234) pins the legacy port the assertion
        expects.
        """
        for var in ("LEGOESM_COORDINATOR_PORT", "SLURM_JOB_ID", "PBS_JOBID",
                    "OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
                    "SLURM_STEP_NUM_TASKS", "SLURM_NTASKS", "SLURM_LOCALID"):
            monkeypatch.delenv(var, raising=False)
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 2
        # Simulate multi-node: ranks on different hosts
        mock_comm.allgather.return_value = ["node0", "node1"]

        mock_MPI = MagicMock()
        mock_MPI.COMM_WORLD = mock_comm

        mock_mpi4jax = MagicMock()

        with patch(
            "legoesm.parallel.distributed.require_mpi_stack",
            return_value=(mock_mpi4jax, mock_MPI),
        ), patch(
            "legoesm.parallel.distributed._require_mpi4py",
            return_value=mock_MPI,
        ), patch(
            "legoesm.parallel.distributed.jax.distributed.initialize",
        ) as mock_jax_init, patch(
            "legoesm.parallel.distributed.jax.distributed.is_initialized",
            return_value=False,
        ), patch(
            "legoesm.parallel.distributed.jax.local_devices",
            return_value=[MagicMock()],
        ), patch(
            "legoesm.parallel.distributed.create_device_mesh",
            return_value=MagicMock(
                mesh=None, face_sharding=None, replicated_sharding=None,
                n_devices=1, backend="cpu",
            ),
        ), patch(
            "legoesm.parallel.distributed.set_active_config",
        ), patch(
            "legoesm.parallel.distributed.build_comm_topology",
            return_value=MagicMock(tiling=None),
        ), patch(
            "legoesm.grids.halo.set_halo_backend",
        ), patch(
            "legoesm.parallel.distributed.jax.process_index",
            return_value=0,
        ), patch(
            # ``initialize_distributed`` now skips ``jax.distributed.initialize``
            # when JAX is already federated (``process_count() != 1``), i.e. when
            # ``maybe_init_jax_distributed`` ran in early_init.  Model the
            # not-yet-federated precondition (count==1) so legoESM performs the
            # federation itself and the coordinator call fires.
            "legoesm.parallel.distributed.jax.process_count",
            return_value=1,
        ):
            import legoesm.parallel.distributed as dist_mod
            dist_mod._active_topology = None
            dist_mod._active_layout = None

            try:
                dist_mod.initialize_distributed()
                mock_jax_init.assert_called_once_with(
                    coordinator_address="node0:1234",
                    num_processes=2,
                    process_id=0,
                )
            finally:
                dist_mod._active_topology = None
                dist_mod._active_layout = None

    def test_distributed_single_node_skips_grpc(self):
        """Single-node MPI should skip jax.distributed.initialize."""
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 2
        # Simulate single-node: all ranks on same host
        mock_comm.allgather.return_value = ["localhost", "localhost"]

        mock_MPI = MagicMock()
        mock_MPI.COMM_WORLD = mock_comm

        mock_mpi4jax = MagicMock()

        with patch(
            "legoesm.parallel.distributed.require_mpi_stack",
            return_value=(mock_mpi4jax, mock_MPI),
        ), patch(
            "legoesm.parallel.distributed.jax.distributed.initialize",
        ) as mock_jax_init, patch(
            "legoesm.parallel.distributed.jax.local_devices",
            return_value=[MagicMock()],
        ), patch(
            "legoesm.parallel.distributed.create_device_mesh",
            return_value=MagicMock(
                mesh=None, face_sharding=None, replicated_sharding=None,
                n_devices=1, backend="cpu",
            ),
        ), patch(
            "legoesm.parallel.distributed.set_active_config",
        ), patch(
            "legoesm.parallel.distributed.build_comm_topology",
            return_value=MagicMock(tiling=None),
        ), patch(
            "legoesm.grids.halo.set_halo_backend",
        ):
            import legoesm.parallel.distributed as dist_mod
            dist_mod._active_topology = None
            dist_mod._active_layout = None

            try:
                dist_mod.initialize_distributed()
                # gRPC should NOT be called for single-node
                mock_jax_init.assert_not_called()
            finally:
                dist_mod._active_topology = None
                dist_mod._active_layout = None


# =========================================================================
# 6. YAML config bootstrap
# =========================================================================

class TestYamlBootstrap:
    """Tests for bootstrap_from_yaml_config."""

    def _make_config(self, overrides=None):
        """Create a mock Config with .get() method."""
        data = {
            "hardware.precision.dynamics": None,
            "hardware.precision.ml": None,
            "hardware.precision.conservation": None,
            "hardware.parallelism.n_devices": "auto",
            "hardware.parallelism.backend": None,
            "hardware.parallelism.distributed": False,
            "hardware.devices": "auto",
        }
        if overrides:
            data.update(overrides)

        config = MagicMock()
        config.get = lambda key, default=None: data.get(key, default)
        return config

    def test_yaml_default_is_fp32(self):
        from legoesm.runtime.config import bootstrap_from_yaml_config
        config = self._make_config()
        rc = bootstrap_from_yaml_config(config)
        assert rc.precision.compute == jnp.float32

    def test_yaml_fp64_dynamics(self):
        from legoesm.runtime.config import bootstrap_from_yaml_config
        config = self._make_config({"hardware.precision.dynamics": "float64"})
        rc = bootstrap_from_yaml_config(config)
        assert rc.precision.compute == jnp.float64
        assert rc.x64 is True

    def test_yaml_mixed_via_conservation(self):
        from legoesm.runtime.config import bootstrap_from_yaml_config
        config = self._make_config({
            "hardware.precision.dynamics": "float32",
            "hardware.precision.conservation": "float64",
        })
        rc = bootstrap_from_yaml_config(config)
        assert rc.precision.accumulate == jnp.float64
        assert rc.precision.compute == jnp.float32


# =========================================================================
# 7. Legacy backward compatibility
# =========================================================================

class TestLegacyCompat:
    """Tests that legacy core.hardware imports still work."""

    def test_legacy_get_backend_uppercase(self):
        from legoesm.core.hardware import get_backend
        backend = get_backend()
        assert backend == backend.upper()

    def test_legacy_check_spectral_backend(self):
        """check_spectral_backend should delegate to runtime."""
        from legoesm.core.hardware import check_spectral_backend
        # On CPU with x64 enabled, should not raise.
        jax.config.update("jax_enable_x64", True)
        check_spectral_backend()

    def test_legacy_unsupported_f64_constant(self):
        from legoesm.core.hardware import _UNSUPPORTED_F64_BACKENDS
        # The canonical constant stores backend names in lowercase
        # (matching `runtime.backend._NO_F64_BACKENDS`).  Call sites
        # always compare with `backend.lower()` before membership
        # (see hardware.py detect_devices).  The Apple GPU backend
        # (jax-mps / MLX) is the float32-only entry.
        assert "mps" in _UNSUPPORTED_F64_BACKENDS

    def test_legacy_parse_precision_dtype(self):
        from legoesm.core.hardware import _parse_precision_dtype
        assert _parse_precision_dtype("fp32", field_name="test") == jnp.float32
        assert _parse_precision_dtype("float64", field_name="test") == jnp.float64

    def test_legacy_precision_policy_roundtrip(self):
        from legoesm.core.hardware import (
            set_runtime_precision_policy,
            get_runtime_precision_policy,
            get_runtime_precision_dtype,
        )
        set_runtime_precision_policy(dynamics="float32", conservation=None)
        policy = get_runtime_precision_policy()
        assert policy["dynamics"] == jnp.float32
        assert policy["conservation"] is None
        assert get_runtime_precision_dtype("dynamics") == jnp.float32

    def test_legacy_core_init_exports(self):
        from legoesm.core import check_spectral_backend, get_backend
        assert callable(check_spectral_backend)
        assert callable(get_backend)

    def test_legacy_apply_hardware_config_delegates(self):
        """apply_hardware_config should delegate to runtime.bootstrap."""
        from legoesm.core.hardware import apply_hardware_config
        config = MagicMock()
        data = {
            "hardware.precision.dynamics": "float32",
            "hardware.precision.ml": None,
            "hardware.precision.conservation": None,
            "hardware.parallelism.n_devices": "auto",
            "hardware.parallelism.backend": None,
            "hardware.parallelism.distributed": False,
            "hardware.devices": "auto",
        }
        config.get = lambda key, default=None: data.get(key, default)
        result = apply_hardware_config(config)
        assert "precision" in result
        assert "distributed" in result
        assert "device_config" in result


# =========================================================================
# 8. Circular import validation
# =========================================================================

class TestNoCircularImports:
    """Ensure the runtime package can be imported cleanly."""

    def test_import_runtime(self):
        """Importing legoesm.runtime must not raise."""
        import legoesm.runtime
        assert hasattr(legoesm.runtime, "bootstrap")
        assert hasattr(legoesm.runtime, "get_backend")
        assert hasattr(legoesm.runtime, "PrecisionPolicy")

    def test_import_runtime_backend(self):
        import legoesm.runtime.backend
        assert callable(legoesm.runtime.backend.get_backend)

    def test_import_runtime_precision(self):
        import legoesm.runtime.precision
        assert callable(legoesm.runtime.precision.resolve_precision)

    def test_import_runtime_devices(self):
        import legoesm.runtime.devices
        assert callable(legoesm.runtime.devices.setup_devices)

    def test_import_runtime_config(self):
        import legoesm.runtime.config
        assert callable(legoesm.runtime.config.bootstrap)

    def test_import_core_hardware_after_runtime(self):
        """Importing core.hardware after runtime must not raise."""
        import legoesm.runtime
        import legoesm.core.hardware
        assert callable(legoesm.core.hardware.get_backend)

    def test_import_runtime_after_core_hardware(self):
        """Importing runtime after core.hardware must not raise."""
        import legoesm.core.hardware
        import legoesm.runtime
        assert callable(legoesm.runtime.get_backend)


# =========================================================================
# 9. Spectral backend guard
# =========================================================================

class TestSpectralBackendGuard:
    """Tests for check_spectral_backend via runtime."""

    def test_cpu_with_x64_passes(self):
        from legoesm.runtime.backend import check_spectral_backend
        jax.config.update("jax_enable_x64", True)
        # Should not raise on CPU.
        check_spectral_backend()

    def test_no_x64_raises(self):
        from legoesm.runtime.backend import check_spectral_backend
        # Temporarily disable x64 for this test.
        original = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", False)
            with pytest.raises(ValueError, match="jax_enable_x64"):
                check_spectral_backend()
        finally:
            jax.config.update("jax_enable_x64", original)

    def test_no_x64_warn_mode(self):
        from legoesm.runtime.backend import check_spectral_backend
        original = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", False)
            with pytest.warns(RuntimeWarning, match="jax_enable_x64"):
                check_spectral_backend(allow_unsupported=True)
        finally:
            jax.config.update("jax_enable_x64", original)


class TestCubedSphereLevelFallbackBootstrap:
    """Codex adversarial review 019e544b (issue #273): the
    ``allow_level_fallback`` flag added in ``d0deac3d`` was unreachable
    from the canonical ``runtime.bootstrap`` path.  These tests pin the
    end-to-end opt-in contract so the 4-GPU unblock is actually
    routable from production startup."""

    def test_default_bootstrap_keeps_face_path(self):
        """Default bootstrap on a cubed-sphere grid lands on the
        face-sharding path (``grid_type='cubed_sphere'``), not the
        level fallback.  On a single-device host this is the trivial
        ``n_devices=1`` path; the assertion still proves the default
        does not silently switch to the level mesh."""
        from legoesm.runtime.config import bootstrap
        rc = bootstrap(precision="fp32", grid_type="cubed_sphere")
        assert rc.device_config.grid_type == "cubed_sphere"

    def test_explicit_4_devices_with_fallback_takes_level_path(self):
        """Bootstrap with ``allow_level_fallback=True`` and an explicit
        ``n_devices=4`` must route to ``cubed_sphere_level``.  Skipped
        unless the test host has ≥4 emulated devices.  To exercise
        locally:

            XLA_FLAGS="--xla_force_host_platform_device_count=4" \\
              JAX_PLATFORMS=cpu pytest tests/unit/test_runtime_bootstrap.py
        """
        if len(jax.devices()) < 4:
            pytest.skip("needs ≥4 emulated devices")
        from legoesm.runtime.config import bootstrap
        rc = bootstrap(
            precision="fp32",
            grid_type="cubed_sphere",
            n_devices=4,
            allow_level_fallback=True,
        )
        assert rc.device_config.grid_type == "cubed_sphere_level"
        assert rc.device_config.n_devices == 4

    def test_explicit_4_devices_without_fallback_raises(self):
        """Without the opt-in, ``n_devices=4`` on cubed-sphere must
        raise — proves the face-divisibility constraint is still
        enforced on the legacy path and the new fallback flag is the
        only escape hatch."""
        if len(jax.devices()) < 4:
            pytest.skip("needs ≥4 emulated devices")
        from legoesm.runtime.config import bootstrap
        with pytest.raises((ValueError, RuntimeError)):
            bootstrap(
                precision="fp32",
                grid_type="cubed_sphere",
                n_devices=4,
                allow_level_fallback=False,
            )
