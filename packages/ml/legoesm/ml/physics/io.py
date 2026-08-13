"""Checkpoint and normalization I/O for the joint ML parameterization."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from legoesm.ml.normalization import NormalizationStats


class PhysicsNormalizationBundle(NamedTuple):
    """Input/output normalization stats for the joint ML model."""

    input_stats: NormalizationStats
    output_stats: NormalizationStats


def save_physics_checkpoint(model, path: str | Path) -> None:
    """Serialize an Equinox joint-physics model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(path), model)


def load_physics_checkpoint(model_template, path: str | Path):
    """Load an Equinox joint-physics model from disk."""
    from legoesm.ml.checkpoint_io import load_checkpoint_or_fail
    return load_checkpoint_or_fail(str(path), model_template)


def save_physics_stats(
    stats_bundle: PhysicsNormalizationBundle,
    path: str | Path,
) -> None:
    """Save joint normalization stats as a NumPy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        input_mean=np.asarray(stats_bundle.input_stats.mean),
        input_std=np.asarray(stats_bundle.input_stats.std),
        output_mean=np.asarray(stats_bundle.output_stats.mean),
        output_std=np.asarray(stats_bundle.output_stats.std),
    )


def load_physics_stats(path: str | Path) -> PhysicsNormalizationBundle:
    """Load joint normalization stats from a NumPy archive."""
    with np.load(Path(path)) as bundle:
        return PhysicsNormalizationBundle(
            input_stats=NormalizationStats(
                mean=jnp.asarray(bundle["input_mean"]),
                std=jnp.asarray(bundle["input_std"]),
            ),
            output_stats=NormalizationStats(
                mean=jnp.asarray(bundle["output_mean"]),
                std=jnp.asarray(bundle["output_std"]),
            ),
        )
