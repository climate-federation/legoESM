"""FV3_3D iter 581: ``compute_edge_artifact_metric`` helper test.

Adds a public-facing diagnostic helper that computes the
edge_std / interior_std / ratio for a 4D field.  Used
throughout iter 466-580 internally; now exposed as a stable
API.

Tests
-----

1. ``test_compute_edge_artifact_metric_basic``.
2. ``test_returns_correct_dict_keys``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.halo import compute_edge_artifact_metric


def test_compute_edge_artifact_metric_basic():
    """Compute on a deterministic field."""
    n_face, n, n_lev = 6, 8, 5
    # Random field with extra noise on the edges (10x amplitude)
    rng = np.random.default_rng(seed=581)
    arr = 0.01 * rng.uniform(-1, 1, size=(n_face, n, n, n_lev))
    arr[:, 0, :, :] += 0.1 * rng.uniform(-1, 1, size=(n_face, n, n_lev))
    arr[:, -1, :, :] += 0.1 * rng.uniform(-1, 1, size=(n_face, n, n_lev))
    arr[:, :, 0, :] += 0.1 * rng.uniform(-1, 1, size=(n_face, n, n_lev))
    arr[:, :, -1, :] += 0.1 * rng.uniform(-1, 1, size=(n_face, n, n_lev))
    metrics = compute_edge_artifact_metric(arr)
    assert metrics["interior_std"] > 0.0
    assert metrics["edge_std"] > 0.0
    # Edge should have higher std than interior
    assert metrics["ratio"] > 1.0, (
        f"edge ratio should be > 1 with edge-loaded noise; "
        f"got {metrics['ratio']:.2f}"
    )


def test_returns_correct_dict_keys():
    arr = np.random.default_rng(seed=581).uniform(
        -1, 1, size=(6, 8, 8, 5),
    )
    metrics = compute_edge_artifact_metric(arr)
    assert set(metrics.keys()) == {"edge_std", "interior_std", "ratio"}
    assert np.isfinite(metrics["edge_std"])
    assert np.isfinite(metrics["interior_std"])
    assert np.isfinite(metrics["ratio"])
