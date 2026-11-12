"""Weight loading: path mapping plus hard failures on missing/mismatched data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.wave.checkpoint import _path_name, load_wave_weights
from legoesm.wave.model import WaveModel


def _weights_dict(model):
    flat, _ = jax.tree_util.tree_flatten_with_path(model)
    return {_path_name(path): np.asarray(leaf) for path, leaf in flat}


def _leaves_dict(model):
    flat, _ = jax.tree_util.tree_flatten_with_path(model)
    return {_path_name(path): np.asarray(leaf) for path, leaf in flat}


def _perturbed(weights):
    """Offset every array so an identity loader cannot match the file."""
    return {name: (value + np.float32(1.0)).astype(value.dtype)
            for name, value in weights.items()}


def test_round_trip_reloads_every_leaf(tiny_config, tmp_path):
    model = WaveModel(tiny_config, key=jax.random.key(0))
    perturbed = _perturbed(_weights_dict(model))
    path = tmp_path / "weights.npz"
    np.savez(path, **perturbed)

    reloaded = load_wave_weights(model, path)
    after = _leaves_dict(reloaded)
    assert set(after) == set(perturbed)
    for name, value in perturbed.items():
        np.testing.assert_array_equal(after[name], value)


def test_missing_array_raises(tiny_config, tmp_path):
    model = WaveModel(tiny_config, key=jax.random.key(0))
    weights = _weights_dict(model)
    weights.pop(next(iter(weights)))
    path = tmp_path / "weights.npz"
    np.savez(path, **weights)
    with pytest.raises(KeyError, match="checkpoint is missing"):
        load_wave_weights(model, path)


def test_shape_mismatch_raises(tiny_config, tmp_path):
    model = WaveModel(tiny_config, key=jax.random.key(0))
    weights = _weights_dict(model)
    key = next(iter(weights))
    weights[key] = np.zeros((1, 1), dtype=np.float32)
    path = tmp_path / "weights.npz"
    np.savez(path, **weights)
    with pytest.raises(ValueError, match="shape"):
        load_wave_weights(model, path)


def test_dtype_mismatch_raises(tiny_config, tmp_path):
    model = WaveModel(tiny_config, key=jax.random.key(0))
    weights = _weights_dict(model)
    key = next(iter(weights))
    weights[key] = np.asarray(weights[key], dtype=np.float16)
    path = tmp_path / "weights.npz"
    np.savez(path, **weights)
    with pytest.raises(ValueError, match="dtype"):
        load_wave_weights(model, path)


def test_partial_load_warns_and_keeps_initial_leaves(tiny_config, tmp_path):
    model = WaveModel(tiny_config, key=jax.random.key(0))
    before = _leaves_dict(model)
    supplied = _perturbed(_weights_dict(model))
    missing = next(iter(supplied))
    supplied.pop(missing)
    path = tmp_path / "weights.npz"
    np.savez(path, **supplied)

    with pytest.warns(RuntimeWarning, match="missing"):
        reloaded = load_wave_weights(model, path, allow_partial=True)
    after = _leaves_dict(reloaded)

    # The absent array kept the freshly initialised value; every supplied
    # array came from the file (the +1 offsets prove the latter).
    np.testing.assert_array_equal(after[missing], before[missing])
    for name, value in supplied.items():
        np.testing.assert_array_equal(after[name], value)


def test_unused_array_raises(tiny_config, tmp_path):
    model = WaveModel(tiny_config, key=jax.random.key(0))
    weights = _weights_dict(model)
    weights["not.a.real.leaf"] = np.zeros((1,), dtype=np.float32)
    path = tmp_path / "weights.npz"
    np.savez(path, **weights)
    with pytest.raises(KeyError, match="does not use"):
        load_wave_weights(model, path)
