"""Load converted NeuralSWM weights into :class:`WaveModel`.

The flat ``.npz`` is produced by
``scripts/data/convert_neuralswm_checkpoint.py``.  Keys are Equinox
attribute paths; a missing key or a shape mismatch is a hard error, so a
silently half-initialised network cannot enter a run.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.wave.model import WaveModel


def _path_name(path) -> str:
    parts: list[str] = []
    for entry in path:
        if isinstance(entry, jax.tree_util.GetAttrKey):
            parts.append(entry.name)
        elif isinstance(entry, jax.tree_util.SequenceKey):
            parts.append(str(entry.idx))
        elif isinstance(entry, jax.tree_util.DictKey):
            parts.append(str(entry.key))
        else:  # pragma: no cover - defensive for an unexpected key type
            raise TypeError(f"unsupported pytree key {entry!r}")
    return ".".join(parts)


def load_wave_weights(
    model: WaveModel,
    path: str | Path,
    *,
    allow_partial: bool = False,
) -> WaveModel:
    """Return ``model`` with every weight replaced from ``path``.

    ``allow_partial`` exists for tests that only supply the source-term
    network; production runs must load every array.
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(model)
    new_leaves = []
    used: set[str] = set()
    missing: list[str] = []

    with np.load(Path(path)) as data:
        files = set(data.files)
        for key_path, leaf in flat:
            name = _path_name(key_path)
            if name not in files:
                if not allow_partial:
                    raise KeyError(f"checkpoint is missing {name!r}")
                missing.append(name)
                new_leaves.append(leaf)
                continue
            value = jnp.asarray(data[name])
            if value.shape != leaf.shape:
                raise ValueError(
                    f"checkpoint {name!r} has shape {value.shape}, "
                    f"model expects {leaf.shape}")
            if value.dtype != leaf.dtype:
                raise ValueError(
                    f"checkpoint {name!r} has dtype {value.dtype}, "
                    f"model expects {leaf.dtype}")
            new_leaves.append(value)
            used.add(name)
        unused = sorted(files - used)

    if missing:
        warnings.warn(
            f"wave checkpoint is missing {len(missing)} arrays; those leaves "
            "keep their initial values", RuntimeWarning, stacklevel=2)
    if unused and not allow_partial:
        raise KeyError(f"checkpoint has arrays the model does not use: {unused}")

    return jax.tree_util.tree_unflatten(treedef, new_leaves)


__all__ = ["load_wave_weights"]
