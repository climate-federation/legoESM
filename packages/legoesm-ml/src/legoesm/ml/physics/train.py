"""Training utilities for the joint ML physics parameterization."""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from legoesm.ml.normalization import compute_normalization_stats, normalize
from legoesm.ml.physics.evaluate import (
    PhysicsEvaluationMetrics,
    evaluate_physics_parameterization,
)
from legoesm.ml.physics.io import PhysicsNormalizationBundle
from legoesm.ml.physics.model import (
    PhysicsParameterizationModel,
    pack_physics_parameterization_features,
    pack_physics_parameterization_targets,
)


class PhysicsModelConfig(NamedTuple):
    """Architecture config for the joint ML model."""

    hidden_dim: int = 128
    n_layers: int = 3
    seed: int = 0


class PhysicsTrainingConfig(NamedTuple):
    """Optimization config for the joint ML model."""

    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 32
    epochs: int = 60
    patience: int = 10
    grad_clip_norm: float = 1.0
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    seed: int = 0


class PhysicsTrainingResult(NamedTuple):
    """Output of a joint ML training run."""

    model: PhysicsParameterizationModel
    stats_bundle: PhysicsNormalizationBundle
    metrics: PhysicsEvaluationMetrics
    best_train_loss: float
    best_val_loss: float
    test_loss: float
    epochs_ran: int
    best_epoch: int
    train_loss_history: tuple[float, ...]
    val_loss_history: tuple[float, ...]


def split_dataset_indices(
    n_samples: int,
    config: PhysicsTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset indices into train/val/test subsets."""
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(n_samples)
    n_train = int(n_samples * config.train_fraction)
    n_val = int(n_samples * config.val_fraction)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def build_physics_training_features(
    columns,
    *,
    microphysics_scheme: str = "none",
) -> jax.Array:
    """Build per-column input features for joint training."""
    return jax.vmap(
        lambda T, u, v, q_v, q_c, q_r, p_full, z_full, p_s, T_sfc, q_sfc, lat, cape, M_c, dt: (
            pack_physics_parameterization_features(
                T=T,
                u=u,
                v=v,
                q_v=q_v,
                q_c=q_c,
                q_r=q_r,
                p_full=p_full,
                z_full=z_full,
                p_s=p_s,
                T_sfc=T_sfc,
                q_sfc=q_sfc,
                lat=lat,
                cape=cape,
                M_c=M_c,
                dt=dt,
                microphysics_scheme=microphysics_scheme,
            )
        ),
    )(
        columns.T,
        columns.u,
        columns.v,
        columns.q_v,
        columns.q_c,
        columns.q_r,
        columns.p_full,
        columns.z_full,
        columns.p_s,
        columns.T_sfc,
        columns.q_sfc,
        columns.lat,
        columns.cape,
        columns.M_c,
        columns.dt,
    )


def build_physics_training_targets(dataset) -> jax.Array:
    """Build flat direct teacher targets for joint training."""
    return pack_physics_parameterization_targets(
        dataset.Km,
        dataset.Kh,
        dataset.M_eq,
        rain_survival_fraction=dataset.rain_survival_fraction,
        dq_v_dt_micro=dataset.dq_v_dt_micro,
        dq_c_dt_micro=dataset.dq_c_dt_micro,
        dq_r_dt_micro=dataset.dq_r_dt_micro,
        precip_micro=dataset.precip_micro,
        microphysics_scheme=dataset.microphysics_scheme,
    )


def compute_stats_bundle(
    features: jax.Array,
    targets: jax.Array,
) -> PhysicsNormalizationBundle:
    """Compute z-score stats on the training split."""
    return PhysicsNormalizationBundle(
        input_stats=compute_normalization_stats(features),
        output_stats=compute_normalization_stats(targets),
    )


def _mse_loss(model, features: jax.Array, targets: jax.Array) -> jax.Array:
    """Compute mean-squared error for one normalized training batch."""
    pred = jax.vmap(model)(features)
    return jnp.mean((pred - targets) ** 2)


def train_physics_parameterization(
    dataset,
    model_config: PhysicsModelConfig | None = None,
    training_config: PhysicsTrainingConfig | None = None,
) -> PhysicsTrainingResult:
    """Train the joint ML model against direct ``Km/Kh/M_eq`` labels."""
    model_config = model_config or PhysicsModelConfig()
    training_config = training_config or PhysicsTrainingConfig()

    features = build_physics_training_features(
        dataset.columns,
        microphysics_scheme=dataset.microphysics_scheme,
    )
    targets = build_physics_training_targets(dataset)

    train_idx, val_idx, test_idx = split_dataset_indices(
        features.shape[0], training_config,
    )
    train_x = features[train_idx]
    val_x = features[val_idx]
    test_x = features[test_idx]
    train_y = targets[train_idx]
    val_y = targets[val_idx]
    test_y = targets[test_idx]

    stats_bundle = compute_stats_bundle(train_x, train_y)
    train_x = normalize(train_x, stats_bundle.input_stats)
    val_x = normalize(val_x, stats_bundle.input_stats)
    test_x = normalize(test_x, stats_bundle.input_stats)
    train_y = normalize(train_y, stats_bundle.output_stats)
    val_y = normalize(val_y, stats_bundle.output_stats)
    test_y = normalize(test_y, stats_bundle.output_stats)

    model = PhysicsParameterizationModel(
        nlev=dataset.columns.T.shape[1],
        hidden_dim=model_config.hidden_dim,
        n_layers=model_config.n_layers,
        microphysics_scheme=dataset.microphysics_scheme,
        key=jax.random.PRNGKey(model_config.seed),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(training_config.grad_clip_norm),
        optax.adamw(training_config.lr, weight_decay=training_config.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def _train_step(model, opt_state, batch_x, batch_y):
        loss, grads = eqx.filter_value_and_grad(_mse_loss)(model, batch_x, batch_y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    eval_loss = eqx.filter_jit(_mse_loss)
    best_model = model
    best_train = float("inf")
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    rng = np.random.default_rng(training_config.seed)

    for epoch in range(training_config.epochs):
        permutation = rng.permutation(train_x.shape[0])
        for start in range(0, train_x.shape[0], training_config.batch_size):
            batch_idx = permutation[start:start + training_config.batch_size]
            model, opt_state, _ = _train_step(
                model, opt_state, train_x[batch_idx], train_y[batch_idx],
            )

        train_loss = float(eval_loss(model, train_x, train_y))
        val_loss = float(eval_loss(model, val_x, val_y))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        if val_loss < best_val:
            best_model = model
            best_train = train_loss
            best_val = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.patience:
                break

    test_loss = float(eval_loss(best_model, test_x, test_y))
    test_dataset = dataset._replace(
        columns=jax.tree_util.tree_map(lambda x: x[test_idx], dataset.columns),
        Km=dataset.Km[test_idx],
        Kh=dataset.Kh[test_idx],
        M_eq=dataset.M_eq[test_idx],
        dq_v_dt_micro=dataset.dq_v_dt_micro[test_idx],
        dq_c_dt_micro=dataset.dq_c_dt_micro[test_idx],
        dq_r_dt_micro=dataset.dq_r_dt_micro[test_idx],
        precip_micro=dataset.precip_micro[test_idx],
        rain_survival_fraction=dataset.rain_survival_fraction[test_idx],
    )
    metrics = evaluate_physics_parameterization(best_model, stats_bundle, test_dataset)
    return PhysicsTrainingResult(
        model=best_model,
        stats_bundle=stats_bundle,
        metrics=metrics,
        best_train_loss=best_train,
        best_val_loss=best_val,
        test_loss=test_loss,
        epochs_ran=len(train_loss_history),
        best_epoch=best_epoch,
        train_loss_history=tuple(train_loss_history),
        val_loss_history=tuple(val_loss_history),
    )
