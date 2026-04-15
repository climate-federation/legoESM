"""Unit tests for the joint ML physics parameterization package."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

from legoesm.ml.physics import (
    PhysicsColumnBatch,
    PhysicsParameterizationModel,
    load_physics_checkpoint,
    load_physics_stats,
    pack_physics_parameterization_features,
    pack_physics_parameterization_targets,
    save_physics_checkpoint,
    save_physics_stats,
    unpack_physics_parameterization_targets,
)
from legoesm.ml.physics.io import PhysicsNormalizationBundle
from legoesm.ml.normalization import NormalizationStats


def test_feature_vector_shape():
    nlev = 4
    feature = pack_physics_parameterization_features(
        T=jnp.ones((nlev,)),
        u=jnp.ones((nlev,)),
        v=jnp.ones((nlev,)),
        q_v=jnp.ones((nlev,)) * 1e-3,
        p_full=jnp.linspace(2e4, 1e5, nlev),
        z_full=jnp.linspace(100.0, 10000.0, nlev),
        p_s=jnp.asarray(1e5),
        T_sfc=jnp.asarray(300.0),
        q_sfc=jnp.asarray(0.01),
        lat=jnp.asarray(0.4),
        cape=jnp.asarray(500.0),
        M_c=jnp.asarray(1e-3),
        dt=jnp.asarray(300.0),
    )
    assert feature.shape == (6 * nlev + 8,)


def test_target_vector_shape_and_unpack():
    nlev = 4
    packed = pack_physics_parameterization_targets(
        Km=jnp.ones((2, nlev)),
        Kh=jnp.ones((2, nlev)) * 2.0,
        M_eq=jnp.ones((2,)) * 3.0,
    )
    assert packed.shape == (2, 2 * nlev + 1)
    unpacked = unpack_physics_parameterization_targets(packed, nlev)
    assert unpacked["Km"].shape == (2, nlev)
    assert unpacked["Kh"].shape == (2, nlev)
    assert unpacked["M_eq"].shape == (2,)


def test_checkpoint_and_stats_roundtrip():
    nlev = 4
    model = PhysicsParameterizationModel(
        nlev=nlev,
        hidden_dim=16,
        n_layers=2,
        key=jax.random.PRNGKey(0),
    )
    stats = PhysicsNormalizationBundle(
        input_stats=NormalizationStats(
            mean=jnp.zeros((6 * nlev + 8,)),
            std=jnp.ones((6 * nlev + 8,)),
        ),
        output_stats=NormalizationStats(
            mean=jnp.zeros((2 * nlev + 1,)),
            std=jnp.ones((2 * nlev + 1,)),
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "physics.eqx"
        stats_path = Path(tmpdir) / "physics_stats.npz"
        save_physics_checkpoint(model, ckpt)
        save_physics_stats(stats, stats_path)
        restored_model = load_physics_checkpoint(model, ckpt)
        restored_stats = load_physics_stats(stats_path)

    x = jnp.ones((6 * nlev + 8,))
    assert jnp.allclose(model(x), restored_model(x))
    assert jnp.allclose(stats.input_stats.mean, restored_stats.input_stats.mean)
    assert jnp.allclose(stats.output_stats.std, restored_stats.output_stats.std)
