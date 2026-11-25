"""The Orbax -> NPZ key/transpose mapping used by the one-shot converter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = (Path(__file__).resolve().parents[3]
           / "scripts/data/convert_neuralswm_checkpoint.py")
_spec = importlib.util.spec_from_file_location("convert_neuralswm_checkpoint", _SCRIPT)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _linear(seed: float):
    return {
        "kernel": seed + np.arange(12, dtype=np.float32).reshape(4, 3),
        "bias": seed * 10 + np.arange(4, dtype=np.float32),
    }


def _synthetic_state():
    mlp = {f"linear{index}": _linear(100.0 * (index + 1)) for index in range(4)}
    for index in range(3):
        mlp[f"norm{index}"] = {
            "scale": 1000.0 * (index + 1) + np.arange(3, dtype=np.float32),
            "bias": np.zeros(3, dtype=np.float32),
        }
    theta_layers = {
        "0": _linear(1.0),
        "1": {"scale": np.arange(2, dtype=np.float32) + 3.0,
              "bias": np.arange(2, dtype=np.float32) + 4.0},
        "2": _linear(2.0),
    }
    k_layers = {
        "0": _linear(5.0),
        "1": {"scale": np.arange(2, dtype=np.float32) + 7.0,
              "bias": np.arange(2, dtype=np.float32) + 8.0},
        "2": _linear(9.0),
    }
    return {
        "parameterization": {"mlp": mlp},
        "theta_transport_network": {"layers": theta_layers},
        "k_transport_network": {"layers": k_layers},
    }


def test_flatten_model_state_names_and_transposes():
    flat = _module.flatten_model_state(_synthetic_state())

    # Every linear layer must keep its own kernel (transposed) and bias; a
    # misrouted layer cannot pass with distinct source values.
    for index in range(4):
        expected = _linear(100.0 * (index + 1))
        np.testing.assert_array_equal(
            flat[f"parameterization.mlp.linear{index}.weight"],
            expected["kernel"].T)
        np.testing.assert_array_equal(
            flat[f"parameterization.mlp.linear{index}.bias"], expected["bias"])
    for index in range(3):
        np.testing.assert_array_equal(
            flat[f"parameterization.mlp.norm{index}.weight"],
            1000.0 * (index + 1) + np.arange(3, dtype=np.float32))

    np.testing.assert_array_equal(
        flat["theta_transport_network.layers.0.weight"],
        _linear(1.0)["kernel"].T)
    np.testing.assert_array_equal(
        flat["theta_transport_network.layers.1.weight"],
        np.arange(2, dtype=np.float32) + 3.0)
    np.testing.assert_array_equal(
        flat["k_transport_network.layers.2.bias"],
        _linear(9.0)["bias"])
    # The two networks are distinct inputs, not one shared dict.
    assert not np.array_equal(
        flat["theta_transport_network.layers.0.weight"],
        flat["k_transport_network.layers.0.weight"])

    # 4 linear (weight+bias) + 3 norms (weight+bias) + 2 networks * 3 layers * 2
    assert len(flat) == 4 * 2 + 3 * 2 + 2 * 3 * 2
