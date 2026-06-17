"""Assemble a spatially-varying GCM parameter field from LES diagnoses.

Stage 7 of ``docs/COMPARE_REANALYSIS.md`` (gap #7, **assembly** half): take the
closure coefficients diagnosed per worst-performing column
(:mod:`legoesm.atmosphere.dynamics.les_closure_diagnosis`) and turn them into a
full-grid parameter field that the next AMIP/CMIP iteration consumes.

Two generalization strategies (the §6 open question), both pure-JAX and
differentiable so the field is a clean pytree leaf flowing through
``jax.grad``/``jit`` without retrace (per CLAUDE.md):

* :func:`scatter_column_field` — **static (lat, lon)**: a background field with
  the diagnosed value placed at each worst column.  Sparse; the simplest option.
* :func:`environment_kernel_field` — **regress onto environmental predictors**:
  Gaussian-kernel (Nadaraya–Watson) regression of the diagnosed values onto a
  normalized environment space (SST / CAPE / shear from the manifest tags), so
  *every* grid column gets a value generalized from environmentally-similar
  diagnosed columns.  Spatially coherent; columns with no nearby diagnosis fall
  back to the background.

The *application* of the field to a scheme ``*Config`` (extending a target field
with a ``shape`` key + ``__param_spec__`` and routing through
``param_collector.apply_param_overrides``) is the companion piece — the
``shape_key`` infra already exists in ``param_collector``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

# Total-kernel-weight floor below which a grid column has no environmentally
# similar diagnosed sample and falls back to the background.  Set to the weight
# of a SINGLE sample at 3 normalized sigma (exp(-½·3²) ≈ 0.011): a column needs
# at least that much accumulated similarity or it is not extrapolated to.
_MIN_TOTAL_WEIGHT = math.exp(-0.5 * 3.0**2)
# Positive floor on per-predictor length scales (guards 0/0 in the kernel).
_LENGTH_SCALE_FLOOR = 1.0e-30


def scatter_column_field(
    grid_shape: tuple[int, ...],
    flat_indices: jax.Array,
    values: jax.Array,
    *,
    background: float = 0.0,
    valid: jax.Array | None = None,
) -> jax.Array:
    """Static ``(lat, lon)`` field: ``background`` everywhere, ``values`` scattered.

    ``flat_indices`` are flat column indices into ``grid_shape`` (e.g. from
    :func:`legoesm.training.column_era5_metrics.rank_worst_columns`); ``values``
    is the diagnosed coefficient per column.  ``valid`` (same length) drops
    flagged-invalid diagnoses (they keep the background).  Differentiable w.r.t.
    ``values``; ``flat_indices`` are static.

    Duplicate indices: the **last** write wins (``.set`` semantics) — dedupe
    upstream if several diagnoses map to one column.
    """
    values = jnp.asarray(values)
    flat_indices = jnp.asarray(flat_indices)
    n = 1
    for d in grid_shape:
        n *= int(d)
    field = jnp.full((n,), jnp.asarray(background, dtype=values.dtype))
    if valid is not None:
        valid = jnp.asarray(valid, dtype=bool)
        # Invalid samples write the existing background at their index (no-op
        # value) so the scatter stays a fixed-shape, traceable operation.
        bg = jnp.asarray(background, dtype=values.dtype)
        values = jnp.where(valid, values, bg)
    field = field.at[flat_indices].set(values)
    return field.reshape(grid_shape)


def environment_kernel_field(
    grid_env: jax.Array,
    sample_env: jax.Array,
    sample_values: jax.Array,
    *,
    length_scales: jax.Array,
    background: float = 0.0,
    valid: jax.Array | None = None,
    min_total_weight: float = _MIN_TOTAL_WEIGHT,
) -> jax.Array:
    """Generalize sampled diagnoses to every grid column by kernel regression.

    Nadaraya–Watson regression in a *normalized* environment space:

    ``field_i = Σ_j w_ij v_j / Σ_j w_ij``,
    ``w_ij = exp(-½ Σ_p ((e_ip − s_jp) / L_p)²)``

    where ``grid_env`` ``(ncol, npred)`` are the environment predictors at every
    grid column, ``sample_env`` ``(nsamp, npred)`` / ``sample_values``
    ``(nsamp,)`` are the worst-column predictors + diagnosed coefficients, and
    ``length_scales`` ``(npred,)`` set the per-predictor similarity scale (so
    SST [K], CAPE [J/kg], shear [m/s] — with very different magnitudes — are
    weighted comparably).  Columns whose total weight is below
    ``min_total_weight`` (no environmentally-similar diagnosis) fall back to
    ``background``.  ``valid`` drops flagged-invalid samples.

    Pure-JAX and differentiable w.r.t. ``sample_values`` / ``sample_env`` /
    ``grid_env`` (AD-safe: the normalizer is masked before division).  Returns
    ``(ncol,)``.
    """
    dtype = jnp.result_type(grid_env, sample_env, sample_values, jnp.float32)
    grid_env = jnp.asarray(grid_env, dtype=dtype)
    sample_env = jnp.asarray(sample_env, dtype=dtype)
    sample_values = jnp.asarray(sample_values, dtype=dtype)
    length_scales = jnp.maximum(
        jnp.asarray(length_scales, dtype=dtype),
        jnp.asarray(_LENGTH_SCALE_FLOOR, dtype=dtype),
    )

    if valid is not None:
        # Sanitize invalid samples to a FINITE dummy BEFORE any arithmetic so a
        # NaN in an invalid sample can neither contaminate the masked sum
        # (0·NaN=NaN) nor leak a NaN adjoint through the inactive where-branch.
        valid = jnp.asarray(valid, dtype=bool)
        sample_env = jnp.where(valid[:, None], sample_env, jnp.zeros_like(sample_env))
        sample_values = jnp.where(valid, sample_values, jnp.zeros_like(sample_values))

    # (ncol, nsamp, npred) normalized differences.
    diff = (grid_env[:, None, :] - sample_env[None, :, :]) / length_scales[None, None, :]
    dist_sq = jnp.sum(diff**2, axis=-1)          # (ncol, nsamp)
    weights = jnp.exp(-0.5 * dist_sq)            # (ncol, nsamp)
    if valid is not None:
        weights = jnp.where(valid[None, :], weights, jnp.zeros_like(weights))

    total = jnp.sum(weights, axis=-1)            # (ncol,)
    weighted = jnp.sum(weights * sample_values[None, :], axis=-1)  # (ncol,)
    has_neighbor = total >= jnp.asarray(min_total_weight, dtype=dtype)
    denom = jnp.where(has_neighbor, total, jnp.ones_like(total))
    bg = jnp.asarray(background, dtype=dtype)
    return jnp.where(has_neighbor, weighted / denom, bg)
