"""Reduce per-column LES diagnoses to a feedback parameter field.

Stage 7 glue of ``docs/COMPARE_REANALYSIS.md``: take the per-worst-column closure
diagnoses produced by the column-LES library
(``legoesm.atmosphere.dynamics.les.column_les``, iter 15/39) and assemble them into
the spatially-varying parameter field that
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

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from legoesm.training.feedback import build_parameter_field

_METHODS = (
    "eddy_diffusivity", "entrainment", "clubb_coefficient", "prandtl_number",
    "c_eps",
)


def _valid_profile_mean(values: Any) -> tuple[jax.Array, jax.Array]:
    """Unweighted mean of a ``(profile, valid)`` over its valid levels.

    Returns ``(value, valid)``; ``valid`` is true iff ≥1 level is valid AND the
    reduced value is finite (a non-finite reduced coefficient — a degenerate /
    blown-up LES column — is marked invalid so it keeps the background instead of
    injecting NaN/inf).  (A height/mass-weighted mean within the diagnosed BL
    depth is a natural refinement if the target scheme is BL-depth sensitive.)
    """
    profile, valid = values
    profile = jnp.asarray(profile)
    valid = jnp.asarray(valid, dtype=bool)
    n = jnp.sum(valid)
    value = jnp.where(
        n > 0,
        jnp.sum(jnp.where(valid, profile, jnp.zeros_like(profile)))
        / jnp.maximum(n, 1),
        jnp.asarray(0.0, dtype=profile.dtype),
    )
    return value, (n > 0) & jnp.isfinite(value)


def reduce_column_diagnosis(
    diagnosis: Any, method: str
) -> tuple[jax.Array, jax.Array]:
    """Reduce one column's LES diagnosis to ``(value, valid)`` scalars.

    * ``"eddy_diffusivity"`` — the valid-level mean of the DIMENSIONAL ``K``
      [m²/s] profile (a representative column eddy diffusivity). LEGACY: a
      dimensional ``K`` CANNOT be injected as the dimensionless ``clubb_lite_C_K``
      (it would just saturate the bounds-clamp); use ``"clubb_coefficient"`` for
      the ``clubb_lite_C_K`` promotion.
    * ``"clubb_coefficient"`` — the valid-level mean of the DIMENSIONLESS CLUBB
      ``C_K`` profile (the actual GCM coefficient, ``K_m = C_K·ℓ·√wp2``); the
      dimensionally-correct target for the ``clubb_lite_C_K`` promotion.
    * ``"prandtl_number"`` — the valid-level mean of the DIMENSIONLESS turbulent
      Prandtl number ``Pr_t = K_m/K_h`` (the ``clubb_lite_Pr_t`` target).
    * ``"c_eps"`` — the valid-level mean of the DIMENSIONLESS wp2-dissipation
      coefficient ``C_eps`` (the ``clubb_lite_C_eps`` target).
    * ``"entrainment"`` — the scalar ``w_entrainment`` and its ``valid`` flag.

    Raises on an unknown ``method`` (dispatch hardening).  Differentiable w.r.t.
    the diagnosed arrays.
    """
    if method == "eddy_diffusivity":
        return _valid_profile_mean((diagnosis.K, diagnosis.valid))
    if method == "clubb_coefficient":
        return _valid_profile_mean((diagnosis.C_K, diagnosis.valid))
    if method == "prandtl_number":
        return _valid_profile_mean((diagnosis.Pr_t, diagnosis.valid))
    if method == "c_eps":
        return _valid_profile_mean((diagnosis.C_eps, diagnosis.valid))
    if method == "entrainment":
        w_e = jnp.asarray(diagnosis.w_entrainment)
        return (
            w_e,
            jnp.asarray(diagnosis.valid, dtype=bool) & jnp.isfinite(w_e),
        )
    raise ValueError(
        f"Unknown diagnosis method {method!r}; choose from {_METHODS}."
    )


def count_valid_diagnoses(diagnoses: Sequence[Any], method: str) -> int:
    """How many columns have a VALID single-method LES diagnosis (≥1 valid level AND
    a finite reduced value) — the EXACT per-column validity
    :func:`assemble_feedback_field` uses, so this is precisely the set of columns
    that receive a non-background correction.  ``n == 0`` with corrected columns
    means EVERY LES spin-off was rejected by the realism gate (e.g. too short to
    develop turbulence) → the round makes no correction by design.  Host-side
    (``bool`` on a concrete reduced flag); call once per iteration, not in a scan.
    """
    return int(sum(bool(reduce_column_diagnosis(d, method)[1]) for d in diagnoses))


def count_valid_multi_diagnoses(diagnoses: Sequence[Any], methods) -> int:
    """How many columns are valid for AT LEAST ONE of the REQUESTED coefficient
    ``methods`` — each column's diagnosis is a ``{method: diagnosis}`` dict (the
    SIMULTANEOUS multi-coefficient path), and a column receives some correction iff
    any one SPEC method's diagnosis is valid.  Only ``methods`` are checked: a
    diagnose_fn may emit EXTRA methods that no spec corrects, which must NOT mark a
    column valid.  Same per-column validity as :func:`count_valid_diagnoses`.
    """
    return int(sum(
        any(bool(reduce_column_diagnosis(d[m], m)[1]) for m in methods)
        for d in diagnoses
    ))


def assemble_feedback_field(
    records: Sequence[Any],
    diagnoses: Sequence[Any],
    grid_shape: tuple[int, ...],
    *,
    method: str = "eddy_diffusivity",
    background: float = 0.0,
    strategy: str = "static",
    grid_env: Any = None,
    length_scales: Any = None,
) -> jax.Array:
    """Assemble the ``(lat, lon)`` feedback field from the LES diagnoses.

    ``records`` are the manifest :class:`~legoesm.training.column_manifest.ColumnRecord`
    s (one per worst column, in the same order as ``diagnoses``); each diagnosis
    is reduced to a scalar (``method``).  Columns whose diagnosis is invalid keep
    the background.  Differentiable w.r.t. the diagnosed values.

    ``strategy="static"`` (default) scatters each reduced value at its column's
    ``flat_index`` (``background`` elsewhere) — only the flagged columns change.
    ``strategy="environment"`` GENERALISES: it Nadaraya–Watson-regresses the
    reduced values onto the **environment** (the manifest's SST/CAPE/shear tags as
    ``sample_env``) over the FULL grid, so every column environmentally similar to
    a diagnosed one gets a value — the §6 "regress onto env predictors" option
    that lets a few LES (one per cluster, §7) correct many columns.  It needs the
    full-grid env predictors ``grid_env`` ``(ncol, 3)`` + per-predictor
    ``length_scales`` ``(3,)`` from the caller (see :func:`column_environment_grid`).
    An unknown strategy raises (dispatch hardening).
    """
    if method not in _METHODS:
        # Validate up front so an unknown method raises even with empty records
        # (the empty branch would otherwise never reach reduce_column_diagnosis).
        raise ValueError(
            f"Unknown diagnosis method {method!r}; choose from {_METHODS}."
        )
    if strategy not in ("static", "environment"):
        raise ValueError(
            f"Unknown strategy {strategy!r}; choose 'static' or 'environment'."
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

    reduced = [reduce_column_diagnosis(d, method) for d in diagnoses]
    values = jnp.stack([v for v, _ in reduced])
    valid = jnp.stack([ok for _, ok in reduced])

    if strategy == "static":
        return build_parameter_field(
            "static",
            grid_shape=grid_shape,
            flat_indices=jnp.asarray([int(r.flat_index) for r in records]),
            values=values,
            valid=valid,
            background=background,
        )

    # strategy == "environment"
    if grid_env is None or length_scales is None:
        raise ValueError(
            "strategy='environment' requires grid_env (ncol, 3) + length_scales "
            "(3,) — the full-grid environment predictors (see "
            "column_environment_grid)."
        )
    sample_env = jnp.asarray([
        [float(r.environment.sst_K), float(r.environment.cape_J_kg),
         float(r.environment.bulk_shear_m_s)]
        for r in records
    ])
    # Fail LOUD on a non-finite env tag in a VALID sample (e.g. a NaN SST over land):
    # unlike the cluster_columns_by_environment guard (which raises), the NW kernel
    # would let ONE NaN tag poison every grid column's total weight (exp(-½·NaN)=NaN
    # summed into each column) — has_neighbor=(NaN≥floor)=False everywhere — silently
    # collapsing the regression to an all-background NO-OP correction. Host-side here
    # (sample_env are concrete manifest tags); invalid-diagnosis samples are exempt
    # (build_parameter_field zeros them before the kernel).
    finite_valid = jnp.where(valid[:, None], jnp.isfinite(sample_env), True)
    if not bool(jnp.all(finite_valid)):
        raise ValueError(
            "assemble_feedback_field(strategy='environment'): non-finite environment "
            "tag (SST/CAPE/shear) in a VALID worst-column sample — a NaN/inf would "
            "silently collapse the kernel regression to an all-background no-op "
            "(matches the cluster_columns_by_environment fail-loud guard)."
        )
    return build_parameter_field(
        "environment",
        grid_shape=grid_shape,
        grid_env=jnp.asarray(grid_env),
        sample_env=sample_env,
        sample_values=values,
        length_scales=jnp.asarray(length_scales),
        valid=valid,
        background=background,
    )


def column_environment_grid(
    model: Any,
    sigma: Any,
    *,
    env_config: Any = None,
    p_full: Any = None,
    p_half: Any = None,
) -> tuple[jax.Array, jax.Array]:
    """Full-grid environment predictors + per-predictor length scales from a
    :class:`~legoesm.training.compare_reanalysis.ColumnState`, for the
    ``strategy="environment"`` feedback generalization.

    Computes ``(ncol, 3)`` ``[SST, CAPE, bulk-shear]`` via the SAME
    :func:`~legoesm.training.column_manifest.compute_column_environment` the
    manifest used, and per-predictor length scales = the std over the grid
    (floored) so the Nadaraya–Watson kernel weights the three
    very-different-magnitude axes comparably.

    **Consistency** (Codex iter-42): ``grid_env`` and the records' ``sample_env``
    MUST be computed identically or the kernel distances are meaningless, so this
    accepts the SAME knobs the comparison can use — ``env_config`` (default
    :class:`EnvironmentConfig`, e.g. the shear reference levels) and explicit
    ``p_full``/``p_half`` (both-or-neither; default = the coordinate's pressures
    ``sigma.pressure_at_full/half(p_s)`` — hybrid-correct, ``== σ·p_s`` for pure-sigma).
    Pass the same ``env_config``/pressure the manifest (``make_compare_fn``) used
    for a hybrid-coordinate or custom-config run.  ``sst`` falls back to the
    surface-level air temperature when the state carries no SST (WARNS once per
    session — iter 281 — so the operator learns the env-kernel grid tags are
    approximate, symmetric with :func:`compare_reanalysis.compare_state_to_reference`).
    """
    from legoesm.training.column_manifest import (
        EnvironmentConfig,
        compute_column_environment,
    )

    p_s = jnp.asarray(model.p_s)
    sigma_full = jnp.asarray(sigma.sigma_full, dtype=p_s.dtype)
    if (p_full is None) != (p_half is None):
        raise ValueError(
            "column_environment_grid: pass both p_full and p_half, or neither "
            "(neither ⇒ the coordinate's pressures via sigma.pressure_at_full/half)."
        )
    if p_full is None:
        # Use the coordinate's TRUE layer pressures (hybrid-correct; iter 342) so the env
        # predictors (CAPE/shear) are on the model's actual levels — the same pressures the
        # compare uses (iter 338/340).  Pure-sigma: pressure_at_full == σ·p_s (byte-identical).
        p_full = jnp.asarray(sigma.pressure_at_full(p_s), dtype=p_s.dtype)
        p_half = jnp.asarray(sigma.pressure_at_half(p_s), dtype=p_s.dtype)
    if getattr(model, "sst_K", None) is not None:
        sst = model.sst_K
    else:
        # The SECOND SST-fallback path (the env-kernel GRID env, iter 128) — warn for
        # symmetry with compare_state_to_reference (iter 280): a silent air-temp tag here
        # degrades the cross-resolution deploy's similarity matching. Structural branch
        # (sst_K None-ness is not data-dependent), so this fires once per trace/call and
        # Python dedups it to once per session — no per-column noise (iter 281).
        import warnings

        warnings.warn(
            "column_environment_grid: model.sst_K is None — using the lowest-level air "
            "temperature as the SST environment tag for the env-kernel grid (the "
            "cross-resolution deploy's env-similarity matching is APPROXIMATE). Supply "
            "the prescribed/coupled SST for accurate tags.", stacklevel=2)
        sst = jnp.asarray(model.T)[..., -1]
    fields = compute_column_environment(
        T=jnp.asarray(model.T), q_v=jnp.asarray(model.q_v),
        u=jnp.asarray(model.u), v=jnp.asarray(model.v),
        p_full=jnp.asarray(p_full), p_half=jnp.asarray(p_half),
        sst=jnp.asarray(sst), sigma_full=sigma_full,
        config=env_config if env_config is not None else EnvironmentConfig(),
    )
    grid_env = jnp.stack(
        [fields.sst_K.reshape(-1), fields.cape_J_kg.reshape(-1),
         fields.bulk_shear_m_s.reshape(-1)], axis=-1)            # (ncol, 3)
    length_scales = jnp.maximum(jnp.std(grid_env, axis=0), 1.0e-6)
    return grid_env, length_scales
