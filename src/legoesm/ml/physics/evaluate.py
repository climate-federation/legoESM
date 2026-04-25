"""Evaluation helpers for the joint ML parameterization."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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
    rmse_rain_survival_fraction: float = 0.0
    rmse_dq_v_dt_micro: float = 0.0
    rmse_dq_c_dt_micro: float = 0.0
    rmse_dq_r_dt_micro: float = 0.0
    rmse_precip_micro: float = 0.0


def predict_physics_parameterization_targets(
    model,
    stats_bundle,
    columns,
) -> dict[str, jax.Array]:
    """Predict denormalized ``Km``, ``Kh``, and ``M_eq`` for a dataset."""
    microphysics_scheme = getattr(model, "microphysics_scheme", "none")
    features = jax.vmap(
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
    features_norm = normalize(features, stats_bundle.input_stats)
    targets_norm = jax.vmap(model)(features_norm)
    targets = denormalize(targets_norm, stats_bundle.output_stats)
    return unpack_physics_parameterization_targets(
        targets,
        columns.T.shape[1],
        microphysics_scheme=microphysics_scheme,
    )


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
    # Fuse the three RMSE reductions into one host transfer.
    _km = predicted["Km"] - dataset.Km
    _kh = predicted["Kh"] - dataset.Kh
    _meq = predicted["M_eq"] - dataset.M_eq
    _h = np.asarray(jnp.stack([
        jnp.sqrt(jnp.mean(_km ** 2)),
        jnp.sqrt(jnp.mean(_kh ** 2)),
        jnp.sqrt(jnp.mean(_meq ** 2)),
    ]))
    metrics = PhysicsEvaluationMetrics(
        rmse_Km=float(_h[0]),
        rmse_Kh=float(_h[1]),
        rmse_M_eq=float(_h[2]),
    )
    if dataset.microphysics_scheme == "kessler":
        metrics = metrics._replace(
            rmse_dq_v_dt_micro=float(
                jnp.sqrt(jnp.mean((predicted["dq_v_dt_micro"] - dataset.dq_v_dt_micro) ** 2))
            ),
            rmse_dq_c_dt_micro=float(
                jnp.sqrt(jnp.mean((predicted["dq_c_dt_micro"] - dataset.dq_c_dt_micro) ** 2))
            ),
            rmse_dq_r_dt_micro=float(
                jnp.sqrt(jnp.mean((predicted["dq_r_dt_micro"] - dataset.dq_r_dt_micro) ** 2))
            ),
            rmse_precip_micro=float(
                jnp.sqrt(jnp.mean((predicted["precip_micro"] - dataset.precip_micro) ** 2))
            ),
        )
    elif dataset.microphysics_scheme == "sundqvist":
        metrics = metrics._replace(
            rmse_rain_survival_fraction=float(
                jnp.sqrt(
                    jnp.mean(
                        (
                            predicted["rain_survival_fraction"]
                            - dataset.rain_survival_fraction
                        ) ** 2
                    )
                )
            ),
        )
    return metrics
