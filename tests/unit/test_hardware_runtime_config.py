"""Tests for runtime consumption of hardware.* configuration options."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from legoesm.config import Config
from legoesm.atmosphere.dynamics import create_model
from legoesm.core.conservation import _accumulation_dtype
from legoesm.runtime.config import bootstrap_from_yaml_config
from legoesm.core.precision import PrecisionPolicy, set_policy


@pytest.fixture(autouse=True)
def _reset_precision_policy():
    """Reset global precision policy before and after each test."""
    set_policy(PrecisionPolicy.fp32())
    yield
    set_policy(PrecisionPolicy.fp32())


def _dummy_cfg():
    return {
        "mesh": None,
        "face_sharding": None,
        "replicated_sharding": None,
        "n_devices": 1,
        "backend": "CPU",
        "is_distributed": False,
        "tiling": (1, 1),
        "grid_type": "cubed_sphere",
    }


class TestApplyHardwareConfig:
    """Runtime setup should consume precision/devices/parallelism settings."""

    def test_legacy_hardware_devices_is_consumed(self):
        cfg = Config.from_dict(
            {
                "hardware": {
                    "devices": 3,
                    "parallelism": {
                        "n_devices": "auto",
                        "backend": "cpu",
                        "distributed": False,
                    },
                }
            }
        )
        with patch(
            "legoesm.parallel.mesh.create_device_mesh",
            return_value=_dummy_cfg(),
        ) as create_mesh:
            bootstrap_from_yaml_config(cfg)
        create_mesh.assert_called_once_with(
            n_devices=3, backend="cpu", allow_level_fallback=False
        )

    def test_parallelism_n_devices_overrides_legacy_devices(self):
        cfg = Config.from_dict(
            {
                "hardware": {
                    "devices": 2,
                    "parallelism": {
                        "n_devices": 6,
                        "backend": "cpu",
                        "distributed": False,
                    },
                }
            }
        )
        with patch(
            "legoesm.parallel.mesh.create_device_mesh",
            return_value=_dummy_cfg(),
        ) as create_mesh:
            bootstrap_from_yaml_config(cfg)
        create_mesh.assert_called_once_with(
            n_devices=6, backend="cpu", allow_level_fallback=False
        )

    def test_distributed_mode_uses_initialize_distributed(self):
        cfg = Config.from_dict(
            {
                "hardware": {
                    "parallelism": {
                        "distributed": True,
                        "n_devices": 4,
                        "backend": "gpu",
                    }
                }
            }
        )
        with patch(
            "legoesm.parallel.distributed.initialize_distributed",
            return_value=_dummy_cfg(),
        ) as init_dist, patch(
            "legoesm.parallel.mesh.create_device_mesh",
        ) as create_mesh:
            bootstrap_from_yaml_config(cfg)
        init_dist.assert_called_once()
        create_mesh.assert_not_called()

    def test_create_model_consumes_hardware_config(self):
        cfg = Config.from_dict(
            {
                "hardware": {
                    "parallelism": {"distributed": False, "n_devices": 1},
                }
            }
        )
        # Clear the lazy cache so the mock is picked up by _resolve_lazy.
        import legoesm.atmosphere.dynamics as _dyn_mod
        _dyn_mod._LAZY_CACHE.pop("CDGridShallowWaterModel", None)
        try:
            with patch(
                "legoesm.runtime.config.bootstrap_from_yaml_config"
            ) as bootstrap_fn, patch(
                "legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid.CDGridShallowWaterModel",
                return_value=object(),
            ):
                bootstrap_fn.return_value = None  # prevent actual bootstrap
                create_model("shallow_water", legoesm_config=cfg, grid=object())
            bootstrap_fn.assert_called_once_with(cfg)
        finally:
            # Clear cache again so the mock doesn't leak into subsequent tests
            _dyn_mod._LAZY_CACHE.pop("CDGridShallowWaterModel", None)


class TestConservationPrecisionHook:
    """Conservation accumulators should honor runtime precision policy."""

    def test_accumulation_dtype_uses_configured_float32(self):
        set_policy(PrecisionPolicy.fp32())
        # fp32 policy: accumulate=float32 → always float32, even when
        # JAX x64 is enabled.  Conservation now honors the policy.
        assert _accumulation_dtype() == jnp.float32

    def test_accumulation_dtype_uses_configured_float64(self):
        set_policy(PrecisionPolicy.mixed())
        # mixed policy: accumulate=float64 → float64 when backend
        # supports it; float32 otherwise (e.g. Metal).
        from legoesm.runtime.backend import supports_float64, is_x64_enabled
        if supports_float64() and is_x64_enabled():
            assert _accumulation_dtype() == jnp.float64
        else:
            assert _accumulation_dtype() == jnp.float32

    def test_accumulation_dtype_falls_back_when_float64_unavailable(self):
        set_policy(PrecisionPolicy.mixed())
        with patch("legoesm.runtime.backend.get_backend", return_value="mps"):
            assert _accumulation_dtype() == jnp.float32
