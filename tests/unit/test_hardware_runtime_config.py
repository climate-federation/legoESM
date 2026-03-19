"""Tests for runtime consumption of hardware.* configuration options."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from legoesm.config import Config
from legoesm.atmosphere.dynamics import create_model
from legoesm.core.conservation import _accumulation_dtype
from legoesm.core.hardware import (
    apply_hardware_config,
    get_runtime_precision_dtype,
    set_runtime_precision_policy,
)


@pytest.fixture(autouse=True)
def _reset_precision_policy():
    """Reset global precision policy before and after each test."""
    set_runtime_precision_policy(
        dynamics="float32",
        ml="bfloat16",
        conservation=None,
    )
    yield
    set_runtime_precision_policy(
        dynamics="float32",
        ml="bfloat16",
        conservation=None,
    )


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
            apply_hardware_config(cfg)
        create_mesh.assert_called_once_with(n_devices=3, backend="cpu")

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
            apply_hardware_config(cfg)
        create_mesh.assert_called_once_with(n_devices=6, backend="cpu")

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
            return_value=(_dummy_cfg(), {"rank": 0}),
        ) as init_dist, patch(
            "legoesm.parallel.mesh.create_device_mesh",
        ) as create_mesh:
            with pytest.warns(RuntimeWarning, match="distributed=true ignores"):
                apply_hardware_config(cfg)
        init_dist.assert_called_once_with(return_topology=True)
        create_mesh.assert_not_called()

    def test_precision_policy_is_applied(self):
        cfg = Config.from_dict(
            {
                "hardware": {
                    "precision": {
                        "dynamics": "float64",
                        "ml": "float32",
                        "conservation": "float32",
                    },
                    "parallelism": {
                        "distributed": False,
                        "n_devices": 1,
                        "backend": "cpu",
                    },
                }
            }
        )
        should_enable_x64 = not jax.config.jax_enable_x64
        with patch(
            "legoesm.parallel.mesh.create_device_mesh",
            return_value=_dummy_cfg(),
        ), patch("legoesm.core.hardware.jax.config.update") as update_cfg:
            apply_hardware_config(cfg)

        assert get_runtime_precision_dtype("dynamics") == jnp.float64
        assert get_runtime_precision_dtype("ml") == jnp.float32
        assert get_runtime_precision_dtype("conservation") == jnp.float32
        if should_enable_x64:
            update_cfg.assert_called_with("jax_enable_x64", True)
        else:
            update_cfg.assert_not_called()

    def test_create_model_consumes_hardware_config(self):
        cfg = Config.from_dict(
            {
                "hardware": {
                    "parallelism": {"distributed": False, "n_devices": 1},
                }
            }
        )
        with patch("legoesm.core.hardware.apply_hardware_config") as apply_hw, patch(
            "legoesm.atmosphere.dynamics.CDGridShallowWaterModel",
            return_value=object(),
        ):
            create_model("shallow_water", legoesm_config=cfg, grid=object())
        apply_hw.assert_called_once_with(cfg)


class TestConservationPrecisionHook:
    """Conservation accumulators should honor runtime precision policy."""

    def test_accumulation_dtype_uses_configured_float32(self):
        set_runtime_precision_policy(conservation="float32")
        assert _accumulation_dtype() == jnp.float32

    def test_accumulation_dtype_falls_back_when_float64_unavailable(self):
        set_runtime_precision_policy(conservation="float64")
        with patch("legoesm.core.hardware.get_backend", return_value="METAL"):
            assert _accumulation_dtype() == jnp.float32
