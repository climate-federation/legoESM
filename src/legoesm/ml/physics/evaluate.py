"""Evaluation helpers for the joint ML parameterization."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.ml.normalization import denormalize, normalize
from legoesm.ml.physics.model import (
    pack_physics_parameterization_features,
    unpack_physics_parameterization_targets,
)


class PhysicsEvaluationMetrics(NamedTuple):
    """Held-out evaluation metrics for the joint ML model."""

    rmse_Km: float
    rmse_Kh: float
    rmse_M_eq: float


def predict_physics_parameterization_targets(
    model,
    stats_bundle,
    columns,
) -> dict[str, jax.Array]:
    """Predict denormalized ``Km``, ``Kh``, and ``M_eq`` for a dataset."""
    features = jax.vmap(pack_physics_parameterization_features)(
        columns.T,
        columns.u,
        columns.v,
        columns.q_v,
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
    features_norm = normalize(features, stats_bundle.input_stats)
    targets_norm = jax.vmap(model)(features_norm)
    targets = denormalize(targets_norm, stats_bundle.output_stats)
    return unpack_physics_parameterization_targets(targets, columns.T.shape[1])


def evaluate_physics_parameterization(
    model,
    stats_bundle,
    dataset,
) -> PhysicsEvaluationMetrics:
    """Evaluate held-out RMSE on direct teacher targets."""
    predicted = predict_physics_parameterization_targets(
        model=model,
        stats_bundle=stats_bundle,
        columns=dataset.columns,
    )
    return PhysicsEvaluationMetrics(
        rmse_Km=float(jnp.sqrt(jnp.mean((predicted["Km"] - dataset.Km) ** 2))),
        rmse_Kh=float(jnp.sqrt(jnp.mean((predicted["Kh"] - dataset.Kh) ** 2))),
        rmse_M_eq=float(jnp.sqrt(jnp.mean((predicted["M_eq"] - dataset.M_eq) ** 2))),
    )
