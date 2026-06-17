"""Reduce per-column LES diagnoses to a feedback parameter field.

Stage 7 glue of ``docs/COMPARE_REANALYSIS.md``: take the per-worst-column closure
diagnoses produced by the column-LES driver (``scripts/run/run_column_les.py``,
iter 15) and assemble them into the spatially-varying parameter field that
:func:`legoesm.training.feedback.apply_column_parameter_field` splices into the
next AMIP/CMIP config — closing the loop:

  worst-column manifest record  +  LES diagnosis (K profile / w_e)
    → per-column scalar coefficient   (:func:`reduce_column_diagnosis`)
    → grid field at the worst columns  (:func:`assemble_feedback_field`,
                                        via :func:`build_parameter_field`).

The reduction turns a height-resolved eddy-diffusivity ``K`` profile (or an
entrainment-velocity scalar) into the single representative value the GCM scheme
field consumes; invalid columns (no valid closure level / no capping inversion)
keep the background.  Pure-JAX, differentiable w.r.t. the diagnosed values.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp

from legoesm.training.feedback import build_parameter_field

_METHODS = ("eddy_diffusivity", "entrainment")


def reduce_column_diagnosis(
    diagnosis: Any, method: str
) -> tuple[jax.Array, jax.Array]:
    """Reduce one column's LES diagnosis to ``(value, valid)`` scalars.

    * ``"eddy_diffusivity"`` — the valid-level (unweighted) mean of the ``K``
      profile, a representative column eddy diffusivity; ``valid`` is true iff
      any level is valid.  (A height/mass-weighted mean is a natural refinement
      if the target scheme is sensitive to the BL-depth weighting.)
    * ``"entrainment"`` — the scalar ``w_entrainment`` and its ``valid`` flag.

    Raises on an unknown ``method`` (dispatch hardening).  Differentiable w.r.t.
    the diagnosed arrays.
    """
    if method == "eddy_diffusivity":
        K = jnp.asarray(diagnosis.K)
        valid = jnp.asarray(diagnosis.valid, dtype=bool)
        n = jnp.sum(valid)
        value = jnp.where(
            n > 0,
            jnp.sum(jnp.where(valid, K, jnp.zeros_like(K))) / jnp.maximum(n, 1),
            jnp.asarray(0.0, dtype=K.dtype),
        )
        return value, n > 0
    if method == "entrainment":
        return (
            jnp.asarray(diagnosis.w_entrainment),
            jnp.asarray(diagnosis.valid, dtype=bool),
        )
    raise ValueError(
        f"Unknown diagnosis method {method!r}; choose from {_METHODS}."
    )


def assemble_feedback_field(
    records: Sequence[Any],
    diagnoses: Sequence[Any],
    grid_shape: tuple[int, ...],
    *,
    method: str = "eddy_diffusivity",
    background: float = 0.0,
) -> jax.Array:
    """Assemble the static ``(lat, lon)`` feedback field from the LES diagnoses.

    ``records`` are the manifest :class:`~legoesm.training.column_manifest.ColumnRecord`
    s (one per worst column, in the same order as ``diagnoses``); each diagnosis
    is reduced to a scalar and scattered at its column's ``flat_index`` into a
    ``grid_shape`` field (``background`` elsewhere).  Columns whose diagnosis is
    invalid keep the background.  Builds via :func:`build_parameter_field`
    (``"static"`` strategy), so it is differentiable w.r.t. the diagnosed values.

    The environment-kernel generalization (``strategy="environment"``) is the
    natural extension: pass the manifest env tags as ``sample_env`` — kept out of
    this helper since it needs the full-grid environment from the caller.
    """
    if method not in _METHODS:
        # Validate up front so an unknown method raises even with empty records
        # (the empty branch would otherwise never reach reduce_column_diagnosis).
        raise ValueError(
            f"Unknown diagnosis method {method!r}; choose from {_METHODS}."
        )
    if len(records) != len(diagnoses):
        raise ValueError(
            f"records ({len(records)}) and diagnoses ({len(diagnoses)}) must "
            f"have the same length (one diagnosis per worst column)."
        )
    if not records:
        # No worst columns → the background field unchanged (scalar → uniform,
        # array → the accumulated field reshaped to the grid).
        bg = jnp.asarray(background)
        return (
            jnp.full(grid_shape, bg) if bg.ndim == 0 else bg.reshape(grid_shape)
        )

    flat_indices = jnp.asarray([int(r.flat_index) for r in records])
    reduced = [reduce_column_diagnosis(d, method) for d in diagnoses]
    values = jnp.stack([v for v, _ in reduced])
    valid = jnp.stack([ok for _, ok in reduced])
    return build_parameter_field(
        "static",
        grid_shape=grid_shape,
        flat_indices=flat_indices,
        values=values,
        valid=valid,
        background=background,
    )
