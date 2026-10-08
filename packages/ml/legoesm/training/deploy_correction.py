"""Deploy a saved correction-campaign output as a production turbulence override.

Closes the practical loop of ``docs/COMPARE_REANALYSIS.md`` ("updating these
parameters in the AMIP/CMIP simulations"): a campaign job
(``scripts/run/run_correction_campaign.py``) writes the corrected per-column
``clubb_lite`` coefficient field(s) to JSON, and a fresh PRODUCTION run loads them
as an :class:`~legoesm.atmosphere.physics.turbulence.config.TurbulenceConfig` for
``ExperimentConfig.turbulence_override`` — so the bias-reducing parameters are
applied to a new simulation.

The per-column override is RUNTIME-ONLY (the standard config serializers null it,
iter 35), so this JSON + loader is the deploy vehicle.  Usage::

    from legoesm.training.deploy_correction import corrected_turbulence_override
    override = corrected_turbulence_override("campaign_out.json")
    cfg = ExperimentConfig(..., turbulence="clubb_lite", turbulence_override=override)

The base config's ``turbulence`` MUST be ``"clubb_lite"`` (``validate_strict``
requires the override scheme to match).
"""

from __future__ import annotations

import json
import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# Below this ``fraction_covered``, MOST deploy-grid columns fell back to background
# (the campaign sampled too few environments to match this grid's climate), so the
# env-kernel override is a (near-)NO-OP correction. Auto-WARNED at this floor even when
# the opt-in ``min_fraction_covered`` fail-loud is unset — a campaign-control judgment
# (cf. ``campaign_summary._MIN_FRACTIONAL_REDUCTION``), not physics.
_LOW_COVERAGE_WARN = 0.5

# CLUBB-lite promotion key → CLUBBLiteConfig field name (the campaign output's
# multi "fields" dict is keyed by promotion_key; the single output by field name).
_CLUBB_PROMOTION_TO_FIELD = {
    "clubb_lite_C_K": "C_K",
    "clubb_lite_Pr_t": "Pr_t",
    "clubb_lite_C_eps": "C_eps",
}
_CLUBB_FIELDS = ("C_K", "Pr_t", "C_eps")


def corrected_clubb_config(data: dict):
    """Build a per-column :class:`CLUBBLiteConfig` from a campaign-output dict.

    Handles both campaign output shapes:

    * MULTI (``--coefficients``): a ``"fields"`` dict ``{promotion_key: [...]}`` —
      each clubb promotion key maps to its config field.
    * SINGLE (``--diagnosis-method``): a top-level ``"C_K"`` / ``"Pr_t"`` /
      ``"C_eps"`` array (exactly one).

    Raises on an empty / unrecognized / non-CLUBB output (dispatch hardening).
    """
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

    if not isinstance(data, dict):
        raise ValueError(
            f"campaign output must be a JSON object; got {type(data).__name__}.")
    defaults = CLUBBLiteConfig()
    overrides: dict[str, Any] = {}
    if "fields" in data:
        fields = data["fields"]
        if not isinstance(fields, dict):
            raise ValueError(
                "multi-coefficient campaign output 'fields' must be a dict keyed "
                f"by promotion_key; got {type(fields).__name__}.")
        for key, vals in fields.items():
            field = _CLUBB_PROMOTION_TO_FIELD.get(key)
            if field is None:
                raise ValueError(
                    f"deploy loader is CLUBB-lite-specific; campaign output has an "
                    f"unsupported promotion_key {key!r} "
                    f"(expected one of {tuple(_CLUBB_PROMOTION_TO_FIELD)}).")
            overrides[field] = _clean_field_array(field, vals, defaults)
    else:
        for field in _CLUBB_FIELDS:
            if field in data:
                overrides[field] = _clean_field_array(field, data[field], defaults)
        if len(overrides) != 1:
            raise ValueError(
                "single-coefficient campaign output must carry exactly one of "
                f"{_CLUBB_FIELDS}; got {sorted(overrides)}.")
    if not overrides:
        raise ValueError(
            "no corrected CLUBB-lite coefficient field found in the campaign "
            "output (expected a 'fields' dict or a C_K/Pr_t/C_eps array).")
    # Multi-field internal consistency, independent of the optional grid check (which is
    # skipped when grid=None): a corrupted output mixing column counts must fail loud at
    # BUILD time, not silently mis-broadcast in production.
    _assert_consistent_ncol(overrides)
    return CLUBBLiteConfig(**overrides)


def _clean_field_array(field: str, vals: Any, defaults):
    """Validate + dtype-normalize a corrected per-column coefficient array.

    Casts to the scheme field's default (float) dtype so a hand-edited / integer
    JSON array can never enter ``CLUBBLiteConfig`` as a non-differentiable int
    array, and rejects an empty / non-1-D / non-finite field LOUDLY at deploy
    time rather than silently feeding NaN turbulence (or a rank-2 broadcast
    crash) into a production run.
    """
    import jax.numpy as jnp

    dtype = jnp.asarray(getattr(defaults, field)).dtype
    arr = jnp.asarray(vals, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(
            f"corrected '{field}' field must be a 1-D per-column array; "
            f"got ndim={arr.ndim} (shape {tuple(arr.shape)}).")
    if arr.size == 0:
        raise ValueError(f"corrected '{field}' field is empty.")
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise ValueError(f"corrected '{field}' field has non-finite values.")
    return arr


def _assert_consistent_ncol(per_column: dict[str, Any]) -> None:
    """Every per-column corrected field must span the SAME column count (one grid).

    A mismatch — a hand-edited / corrupted MULTI output with e.g. ``C_K`` length 128
    but ``Pr_t`` length 64 — would otherwise build a ``CLUBBLiteConfig`` whose fields
    index DIFFERENT columns, a silent ``jnp.take`` OOB-fill in the physics.  Shared by
    the deploy BUILD (:func:`corrected_clubb_config`, which previously only validated
    each field individually) and the distributed SLICE (:func:`_per_column_clubb_fields`)
    so the consistency rule lives in ONE place.
    """
    import jax.numpy as jnp

    lengths = {int(jnp.asarray(a).shape[0]) for a in per_column.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"corrected per-column fields have inconsistent column counts {lengths}; "
            "every corrected field must span the same grid (one ncol).")


def corrected_turbulence_override(
    source: dict | str | Any,
    *,
    grid: Any = None,
    allow_unverified_grid: bool = False,
):
    """Load a campaign output into a deployable ``TurbulenceConfig``.

    ``source`` is the campaign-output dict, a JSON path (str/``os.PathLike``), or
    an open file object.  Returns ``TurbulenceConfig(scheme="clubb_lite",
    clubb_lite=<per-column CLUBBLiteConfig>)`` for ``ExperimentConfig.turbulence_
    override`` (the base config's ``turbulence`` must be ``"clubb_lite"``).

    If ``grid`` (the PRODUCTION grid object) is supplied, the per-column
    coefficients are verified to belong on it via :func:`assert_deploy_compatible`
    — a per-column field learned on a DIFFERENT grid would otherwise silently land
    on the wrong cells.  ``allow_unverified_grid=True`` deploys on the array-length
    check alone when the output predates grid provenance (UNSAFE if grids differ).
    """
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    data = _load_campaign_output(source)
    cfg = corrected_clubb_config(data)
    if grid is not None:
        _assert_compatible(cfg, data, grid, allow_unverified_grid)
    return TurbulenceConfig(scheme="clubb_lite", clubb_lite=cfg)


def _load_campaign_output(source: dict | str | Any) -> dict:
    """Resolve a campaign-output dict / JSON path / open file to a dict."""
    if isinstance(source, dict):
        return source
    if hasattr(source, "read"):
        return json.load(source)
    with open(source) as f:
        return json.load(f)


# Round flattened coordinates (radians) before hashing: absorbs cross-machine ULP
# jitter for the grid AS BUILT. The guarantee is one-sided by design — the SAME
# grid construction (same config/resolution/dtype, the normal campaign->deploy
# case) yields an identical fingerprint, and the coordinate hash makes a collision
# between two genuinely-different grids essentially impossible. A grid rebuilt in a
# DIFFERENT precision may hash differently → a SAFE false-positive block (fails
# loud, resolvable with allow_unverified_grid), never a silent wrong-cell deploy.
_COORD_HASH_DECIMALS = 9


def grid_fingerprint(grid: Any) -> dict:
    """Adapter-order structural + coordinate fingerprint of a grid.

    Beyond ``(ncol, shape_2d)`` (reshape compatibility) it hashes the flattened
    lat/lon in the EXACT order :class:`~legoesm.core.grid_adapters.ColumnAdapter`
    consumes them, so two grids with equal shape but different cell identity /
    flatten order (Gaussian vs equiangular lat, a cubed-sphere panel permutation,
    a transposed axis) are DISTINGUISHED — equal shape alone does not prove a
    per-column coefficient lands on the same physical cell.

    Returns ``{"ncol", "shape_2d", "coord_sha256"}`` (the campaign records this as
    the output's ``"grid"`` provenance block; the deploy guard compares against it).
    """
    import hashlib

    import numpy as np
    from legoesm.core.grid_adapters import make_adapter

    adapter = make_adapter(grid)
    ncol = int(adapter.ncol)
    shape_2d = [int(s) for s in adapter.shape_2d]
    lat = np.asarray(grid.grid_lat).reshape(-1)
    lon = np.asarray(grid.grid_lon).reshape(-1)
    if lat.size != ncol or lon.size != ncol:
        raise ValueError(
            f"grid_fingerprint: flattened lat/lon size ({lat.size}/{lon.size}) "
            f"!= ncol ({ncol}); the grid's coordinate field is not in column "
            "order, so a per-column deploy cannot be fingerprinted on it.")
    # round + (+0.0) normalizes negative zero; the FIXED little-endian float64
    # dtype ('<f8') makes the byte stream — and thus the hash — machine-endianness
    # independent (a big-endian HPC node hashes identically).
    le = np.dtype("<f8")
    lat_b = (np.round(lat.astype(np.float64), _COORD_HASH_DECIMALS) + 0.0).astype(le)
    lon_b = (np.round(lon.astype(np.float64), _COORD_HASH_DECIMALS) + 0.0).astype(le)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(lat_b).tobytes())
    h.update(np.ascontiguousarray(lon_b).tobytes())
    return {"ncol": ncol, "shape_2d": shape_2d, "coord_sha256": h.hexdigest()}


def assert_deploy_compatible(
    source: dict | str | Any,
    grid: Any,
    *,
    allow_unverified_grid: bool = False,
) -> None:
    """Raise if the campaign output's per-column coefficients do not belong on ``grid``.

    Two layers: (a) ALWAYS — every corrected per-column array's length must equal
    the target grid's column count; (b) if the output carries a ``"grid"``
    provenance block, its ``ncol``/``shape_2d``/``coord_sha256`` must match the
    target grid's :func:`grid_fingerprint` EXACTLY — this is what catches the
    silent same-``ncol`` different-grid case (coefficients on the wrong cells).

    A missing provenance block (an output predating this guard) is an EXPLICIT
    error when ``grid`` is supplied — the layout cannot be verified — unless
    ``allow_unverified_grid=True`` (deploys on the length check alone; UNSAFE if
    the deploy grid differs from the campaign grid).  Provenance is GLOBAL
    (whole-grid) column order; validate the global grid before any MPI rank-local
    slicing.
    """
    data = _load_campaign_output(source)
    _assert_compatible(
        corrected_clubb_config(data), data, grid, allow_unverified_grid)


def _assert_compatible(cfg, data: dict, grid: Any, allow_unverified_grid: bool) -> None:
    """Shared check over an already-built CLUBBLiteConfig (no double build)."""
    import jax.numpy as jnp

    target = grid_fingerprint(grid)
    target_ncol = target["ncol"]

    # (a) length: every corrected per-column array spans the target columns.
    for field in _CLUBB_FIELDS:
        arr = jnp.asarray(getattr(cfg, field))
        if arr.ndim == 1 and int(arr.shape[0]) != target_ncol:
            raise ValueError(
                f"deploy mismatch: corrected '{field}' spans {int(arr.shape[0])} "
                f"columns but the target grid has {target_ncol} (the campaign was "
                "run on a different grid).")

    # (b) grid identity (coordinate fingerprint) — catches same-ncol wrong layout.
    # A present-but-null coord_sha256 is treated as missing provenance (not a
    # silent bypass): a missing / null / non-dict block, or a falsy coord_sha256,
    # all route to the unverified path.
    _raw = data.get("grid")
    rec = _raw if isinstance(_raw, dict) else {}
    if not rec.get("coord_sha256"):
        if allow_unverified_grid:
            return
        raise ValueError(
            "deploy: campaign output lacks a 'grid' coordinate-provenance block "
            "(or its coord_sha256 is null), so the per-column coefficients' grid "
            "layout cannot be verified against the target grid — a same-ncol "
            "different-grid deploy would silently apply coefficients to the WRONG "
            "cells. Re-run the campaign to record provenance, or pass "
            "allow_unverified_grid=True to deploy on the length check alone "
            "(UNSAFE if the grids differ).")
    for key in ("ncol", "shape_2d", "coord_sha256"):
        rv = rec.get(key)
        tv = target[key]
        if key == "shape_2d" and rv is not None:
            rv = list(rv)
        if rv is not None and rv != tv:
            raise ValueError(
                f"deploy grid mismatch on {key!r}: campaign output recorded "
                f"{rv!r} but the target grid has {tv!r}. The per-column "
                "coefficients were learned on a different grid and would land on "
                "the wrong cells; deploy on the SAME grid the campaign used.")


# --------------------------------------------------------------------------- #
# Distributed deploy: slice a GLOBAL per-column override to a rank's LOCAL columns.
# Under MPI spatial decomposition each rank owns a tile (a subset of columns), so
# the physics runs on rank-local (ncol_local, nlev) shapes — a global (ncol,)
# override must be sliced to match. The campaign + grid guard operate on the GLOBAL
# grid; these helpers produce the rank-local override AFTER that global validation.
# --------------------------------------------------------------------------- #
def _per_column_clubb_fields(override):
    """Validate a clubb_lite override + return (clubb_cfg, {field: 1-D array}).

    Only ndim==1 per-column fields are sliceable; ndim>1 is rejected (the physics
    `broadcast_column_param` accepts scalar or 1-D only). Scalar defaults are left
    out (they broadcast on every rank unchanged).
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    if not isinstance(override, TurbulenceConfig) or override.scheme != "clubb_lite":
        raise ValueError(
            "override must be a clubb_lite TurbulenceConfig; got "
            f"{getattr(override, 'scheme', type(override).__name__)!r}.")
    clubb = override.clubb_lite
    per_column = {}
    for field in _CLUBB_FIELDS:
        arr = jnp.asarray(getattr(clubb, field))
        if arr.ndim == 0:
            continue                       # scalar default → broadcasts per rank
        if arr.ndim != 1:
            raise ValueError(
                f"per-column override field '{field}' must be 1-D (ncol,); got "
                f"ndim={arr.ndim} (shape {tuple(arr.shape)}).")
        per_column[field] = arr
    # All per-column fields share ONE column count (the global ncol) — else an
    # index valid for one field would be OOB for another (silent jnp.take fill).
    _assert_consistent_ncol(per_column)
    return clubb, per_column


def slice_override_columns(override, local_column_indices):
    """Gather a GLOBAL per-column clubb override to a rank's LOCAL columns (grid-AGNOSTIC).

    ``local_column_indices`` are the rank's columns' GLOBAL flat indices in local
    order (the caller derives them from its decomposition's canonical scatter — for
    lat-lon use :func:`slice_override_latlon_2d`, which needs no indices).  Each
    corrected per-column field is gathered at those indices (AD-safe ``jnp.take`` —
    grads flow to the global leaf, so a TRAINED override stays differentiable);
    scalar default fields pass through.  Indices are bounds-checked host-side at
    deploy time (NOT traced), so an out-of-range index fails LOUDLY rather than
    silently filling/clamping under JIT.
    """
    import jax.numpy as jnp

    clubb, per_column = _per_column_clubb_fields(override)
    idx = jnp.asarray(local_column_indices).reshape(-1)
    if not jnp.issubdtype(idx.dtype, jnp.integer):
        raise ValueError(
            f"local_column_indices must be integer; got dtype {idx.dtype}.")
    if per_column and idx.shape[0] > 0:
        # Bounds-check only a NON-empty index set: a zero-column rank (more ranks than
        # cells, or a custom/unstructured decomposition) is valid — jnp.take with a
        # size-0 index yields a clean size-0 field. min()/max() on an empty array would
        # otherwise raise an OPAQUE "zero-size reduction" instead of slicing cleanly
        # (codex-review iter 404).
        ncol = int(next(iter(per_column.values())).shape[0])
        lo, hi = int(idx.min()), int(idx.max())
        if lo < 0 or hi >= ncol:
            raise ValueError(
                f"local_column_indices out of range [0, {ncol}) for the global "
                f"override: got [{lo}, {hi}].")
    sliced = {f: jnp.take(arr, idx, axis=0) for f, arr in per_column.items()}
    # _replace, not a rebuild: the override's other fields (liquid_partition,
    # update_interval_steps) must survive slicing.
    return override._replace(clubb_lite=clubb._replace(**sliced))


def slice_override_latlon_2d(override, layout):
    """Slice a GLOBAL per-column clubb override to a rank's LOCAL lat-lon tile (STRICT).

    The deploy-context (strict) wrapper around the shared slicer
    :func:`legoesm.atmosphere.physics.turbulence.override_sharding.localize_turbulence_override`:
    it validates the override is a clubb_lite config whose per-column fields are 1-D,
    consistent, and span the layout's GLOBAL grid (a wrong-grid deploy fails LOUDLY
    here), then delegates the reshape→``[lat_start:lat_end, lon_start:lon_end]``
    cell-centered slice→row-major-flatten — landing the local override on exactly the
    columns the rank's physics consumes.  ``make_latlon_2d_layout(0, 1, 1, ...)`` is
    the single-rank identity.  lat-lon cell-centered only; non-lat-lon decompositions
    supply their own indices to :func:`slice_override_columns`.
    """
    from legoesm.atmosphere.physics.turbulence.override_sharding import (
        localize_turbulence_override,
    )

    _clubb, per_column = _per_column_clubb_fields(override)
    n_lat, n_lon = int(layout.n_lat_global), int(layout.n_lon_global)
    ncol = n_lat * n_lon
    for field, arr in per_column.items():        # strict: must span the global grid
        if int(arr.shape[0]) != ncol:
            raise ValueError(
                f"per-column override field '{field}' has {int(arr.shape[0])} "
                f"columns but the layout's global grid is {n_lat}x{n_lon}={ncol}.")
    return localize_turbulence_override(override, layout)


# --------------------------------------------------------------------------- #
# Cross-resolution deploy via the environment kernel (iter 69).
# The per-column array deploy above is GRID-LOCKED (the iter-58 guard rejects a
# different grid). The environment kernel is GRID-AGNOSTIC: the diagnosed
# (environment, coefficient) SAMPLES + length-scales evaluate on ANY grid's
# environment by environmental similarity — so a cheap LOW-res campaign deploys on
# the expensive HIGH-res production run (the practical way to scale).
# --------------------------------------------------------------------------- #
#: Canonical predictor ORDER for the environment kernel — the SINGLE source of truth
#: shared by the producer (:func:`...feedback_assembly.column_environment_grid` +
#: :func:`build_env_kernel`, both stacking ``[sst_K, cape_J_kg, bulk_shear_m_s]``) and
#: the deploy. ``new_grid_env`` columns passed to :func:`apply_env_kernel_override` MUST
#: be in THIS order; a kernel records it (``EnvKernel.predictor_names``) so the artifact
#: self-documents the contract a wrong-order deploy would otherwise silently violate.
ENV_PREDICTOR_NAMES: tuple[str, ...] = ("sst_K", "cape_J_kg", "bulk_shear_m_s")


class EnvKernel(NamedTuple):
    """A serializable, grid-agnostic environment→coefficient kernel.

    ``sample_env`` ``(nsamp, 3)`` [SST, CAPE, bulk-shear] + ``sample_values``
    ``(nsamp,)`` are the campaign's diagnosed columns' environments + reduced
    coefficients; ``length_scales`` ``(3,)`` set the per-predictor similarity scale.
    ``env_lo``/``env_hi`` ``(3,)`` are the sampled env range — provenance for
    detecting a deploy whose grid lies OUTSIDE the sampled hull (domain shift).
    ``predictor_names`` records the predictor ORDER (``ENV_PREDICTOR_NAMES``) so the
    serialized artifact self-documents the column contract of ``sample_env`` /
    ``new_grid_env`` (a deploy whose env is in a DIFFERENT order would silently produce
    miscorrelated coefficients — codex-review iter 405).
    """

    sample_env: Any            # (nsamp, 3)
    sample_values: Any         # (nsamp,)
    valid: Any                 # (nsamp,) bool
    length_scales: Any         # (3,)
    field: str                 # "C_K" / "Pr_t" / "C_eps"
    background: float
    env_lo: Any                # (3,) sampled-env min (domain-shift provenance)
    env_hi: Any                # (3,) sampled-env max
    predictor_names: tuple[str, ...] = ENV_PREDICTOR_NAMES   # column contract (iter 405)


def build_env_kernel(records, diagnoses, method, length_scales, *,
                     field: str = "C_K", background: float = 0.0) -> EnvKernel:
    """Build a grid-agnostic :class:`EnvKernel` from a campaign's worst-column
    diagnoses (the producer side of the cross-resolution deploy).

    ``records`` carry each worst column's ``ColumnEnvironment`` (the SAME [SST, CAPE,
    bulk-shear] tags the loop's env strategy uses); ``diagnoses`` are reduced to a
    scalar per column via the SHARED :func:`legoesm.training.feedback_assembly.
    reduce_column_diagnosis` (so the kernel samples MATCH what the static strategy
    would scatter — no re-derivation).  ``length_scales`` come from the campaign's
    :func:`column_environment_grid`.
    """
    import jax.numpy as jnp
    from legoesm.training.feedback_assembly import reduce_column_diagnosis

    if field not in _CLUBB_FIELDS:
        raise ValueError(
            f"field {field!r} must be one of {_CLUBB_FIELDS}.")
    records = list(records)
    diagnoses = list(diagnoses)
    if len(records) != len(diagnoses):
        raise ValueError(
            f"records ({len(records)}) and diagnoses ({len(diagnoses)}) must align.")
    if not records:
        raise ValueError("need at least one diagnosed column to build a kernel.")
    sample_env = jnp.asarray([
        [float(r.environment.sst_K), float(r.environment.cape_J_kg),
         float(r.environment.bulk_shear_m_s)] for r in records])
    reduced = [reduce_column_diagnosis(d, method) for d in diagnoses]
    sample_values = jnp.stack([v for v, _ in reduced])
    valid = jnp.stack([jnp.asarray(ok, bool) for _, ok in reduced])
    return _validate_env_kernel(EnvKernel(
        sample_env=sample_env, sample_values=sample_values, valid=valid,
        length_scales=jnp.asarray(length_scales).reshape(-1), field=field,
        background=float(background),
        env_lo=jnp.min(sample_env, axis=0), env_hi=jnp.max(sample_env, axis=0)))


def _validate_env_kernel(kernel: EnvKernel) -> EnvKernel:
    """Enforce the EnvKernel shape invariants (shared by build + deserialize), so a
    malformed / version-skewed kernel fails LOUDLY rather than silently broadcasting
    (e.g. a length-1 ``env_lo`` corrupting the hull check). Returns ``kernel``."""
    import jax.numpy as jnp

    if kernel.field not in _CLUBB_FIELDS:
        raise ValueError(f"kernel.field {kernel.field!r} not in {_CLUBB_FIELDS}.")
    se = jnp.asarray(kernel.sample_env)
    if se.ndim != 2:
        raise ValueError(
            f"sample_env must be (nsamp, npred); got {tuple(se.shape)}.")
    nsamp, npred = int(se.shape[0]), int(se.shape[1])
    if len(kernel.predictor_names) != npred:
        raise ValueError(
            f"env kernel predictor_names {kernel.predictor_names} has "
            f"{len(kernel.predictor_names)} names but sample_env has {npred} predictors "
            "— the column contract is inconsistent (a corrupted/version-skewed kernel).")
    for name, arr, n in (("sample_values", kernel.sample_values, nsamp),
                         ("valid", kernel.valid, nsamp),
                         ("length_scales", kernel.length_scales, npred),
                         ("env_lo", kernel.env_lo, npred),
                         ("env_hi", kernel.env_hi, npred)):
        if int(jnp.asarray(arr).reshape(-1).shape[0]) != n:
            raise ValueError(
                f"env kernel '{name}' length "
                f"{int(jnp.asarray(arr).reshape(-1).shape[0])} != {n}.")
    if nsamp == 0 or not bool(jnp.any(jnp.asarray(kernel.valid, bool))):
        raise ValueError("env kernel has no VALID samples — it would be empty.")
    # Every numeric leaf must be FINITE: a non-finite sample_env (e.g. a worst column
    # with a NaN SST tag over land — the iter-179 fail-loud convention the clustering +
    # feedback paths share) or a corrupted deserialize would otherwise (a) propagate a
    # NaN through ``env_lo``/``env_hi`` = min/max(sample_env) and the similarity kernel,
    # and (b) write a non-standard ``NaN``/``Infinity`` token into ``<out>.env_kernel.json``
    # ⇒ unparseable by the cross-grid deploy reader (iter 247). Mask land/invalid columns
    # from the comparison (``valid_mask``) before building the kernel.
    for name, arr in (("sample_env", kernel.sample_env),
                      ("sample_values", kernel.sample_values),
                      ("length_scales", kernel.length_scales),
                      ("env_lo", kernel.env_lo), ("env_hi", kernel.env_hi)):
        if not bool(jnp.all(jnp.isfinite(jnp.asarray(arr, dtype=float)))):
            raise ValueError(
                f"env kernel '{name}' has a non-finite (NaN/inf) value — a worst column's "
                "environment tag is undefined (e.g. a NaN SST over land). Mask such columns "
                "from the comparison (valid_mask) before building the env kernel.")
    # length_scales are per-predictor SIMILARITY SCALES (divisors in the kernel
    # distance ``((env-sample)/ls)²``); they MUST be strictly positive. The campaign
    # already floors them at the per-predictor std (≥1e-6, ``feedback_assembly.
    # _env_grid_predictors``), so a non-positive scale only reaches here from a directly-
    # built or hand-edited/deserialized kernel — a misconfiguration. ``environment_
    # kernel_field`` floors any scale at a tiny ``_LENGTH_SCALE_FLOOR=1e-30`` (so the
    # eval never DIVIDES by zero — no NaN), but that means a ZERO or NEGATIVE scale is
    # silently floored to ~0, collapsing that predictor to an EXACT-MATCH-ONLY kernel:
    # every grid column that does not (near-)exactly match a sample in that predictor
    # falls back to ``background``, i.e. a SILENT near-all-background NO-OP deploy
    # (empirically: ls=0 and ls<0 give the IDENTICAL exact-match-only field). Reject it
    # at construction so the misconfiguration fails LOUD instead of deploying nothing.
    ls = jnp.asarray(kernel.length_scales, dtype=float)
    if not bool(jnp.all(ls > 0.0)):
        raise ValueError(
            "env kernel 'length_scales' must be strictly POSITIVE (each is a per-predictor "
            "similarity scale; the campaign sets them from the per-predictor std). A "
            "non-positive scale is floored to ~0 by the evaluator, collapsing that "
            "predictor to an exact-match-only kernel ⇒ most grid columns fall back to "
            "background (a silent near-no-op deploy). Got "
            f"{jnp.asarray(kernel.length_scales).reshape(-1).tolist()}.")
    return kernel


def apply_env_kernel_override(kernel: EnvKernel, new_grid_env, *,
                              min_total_weight: float | None = None,
                              min_fraction_covered: float | None = None):
    """Evaluate an :class:`EnvKernel` on a NEW grid's environment → an override.

    ``new_grid_env`` ``(ncol_new, 3)`` from :func:`column_environment_grid` on the
    production run's state (ANY resolution).  Its columns MUST be in the kernel's
    ``predictor_names`` order (``ENV_PREDICTOR_NAMES`` = ``[sst_K, cape_J_kg,
    bulk_shear_m_s]``); a different order would silently pair each predictor with the
    WRONG length-scale and produce miscorrelated coefficients — use
    :func:`column_environment_grid` (which emits exactly that order) rather than
    hand-building the array.  Returns ``(TurbulenceConfig, coverage)``
    where ``coverage`` reports the fraction of columns that found an environmentally
    similar diagnosis vs fell back to ``background``, and the fraction WITHIN the
    sampled env hull — a LOW ``fraction_covered`` warns the deploy grid's climate
    lies outside the campaign's sampled environments (Codex iter-69).

    ``min_fraction_covered`` (opt-in, default ``None`` = the original return-and-let-
    the-caller-decide behaviour) FAILS LOUD when ``fraction_covered`` falls below it:
    a deploy grid whose columns (near-)all lie outside the sampled environments yields
    an (near-)all-``background`` override — a silent NO-OP correction — so a production
    user can assert a minimum coverage and catch a misconfigured cross-grid deploy
    BEFORE a multi-day run rather than discover it did nothing afterwards.  The OSSE
    go/no-go (:mod:`perfect_model_osse`) gates the SAME way externally.  (Passing
    ``0.0`` is accepted but never fires — ``fraction_covered`` is always ≥ 0 — so use
    a strictly-positive threshold for real protection.)
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.turbulence.config import (
        CLUBBLiteConfig,
        TurbulenceConfig,
    )
    from legoesm.training.parameter_field import environment_kernel_field

    if kernel.field not in _CLUBB_FIELDS:
        raise ValueError(f"kernel.field {kernel.field!r} not in {_CLUBB_FIELDS}.")
    # Range-guard the min_total_weight floor (min_fraction_covered is already range-
    # checked where it is applied below). A NEGATIVE weight floor silently disables the
    # background fallback: ``total >= a_negative_floor`` is ALWAYS True, so EVERY column
    # reads as having an environmentally-similar neighbour and none falls back to
    # ``background`` — the opposite of the intended protection. Fail loud at the call.
    if min_total_weight is not None and min_total_weight < 0.0:
        raise ValueError(
            f"min_total_weight must be >= 0 (a similarity-weight floor); got "
            f"{min_total_weight}. A negative floor makes every column a 'neighbor' "
            "(no background fallback) — the opposite of the intended protection.")
    grid_env = jnp.asarray(new_grid_env)
    npred = jnp.asarray(kernel.sample_env).shape[1]
    if grid_env.ndim != 2 or grid_env.shape[1] != npred:
        raise ValueError(
            f"new_grid_env must be (ncol, {npred}); got {tuple(grid_env.shape)}.")
    kw = {} if min_total_weight is None else {"min_total_weight": min_total_weight}
    field, has_neighbor = environment_kernel_field(
        grid_env, kernel.sample_env, kernel.sample_values,
        length_scales=kernel.length_scales, background=kernel.background,
        valid=kernel.valid, return_coverage=True, **kw)
    field = jnp.asarray(field).reshape(-1)
    if not bool(jnp.all(jnp.isfinite(field))):
        raise ValueError("kernel evaluation produced non-finite coefficients.")
    in_hull = jnp.all(
        (grid_env >= kernel.env_lo) & (grid_env <= kernel.env_hi), axis=1)
    coverage = {
        "n_columns": int(field.shape[0]),
        "fraction_covered": float(jnp.mean(has_neighbor)),
        "fraction_in_hull": float(jnp.mean(in_hull)),
    }
    # Auto-WARN on a (near-)no-op deploy even when the opt-in min_fraction_covered
    # fail-loud is unset: a careless production deploy must not SILENTLY land a
    # mostly-background correction (the run would be largely uncorrected and the
    # operator left wondering why the bias did not improve).
    if coverage["fraction_covered"] < _LOW_COVERAGE_WARN:
        logger.warning(
            "env-kernel deploy: fraction_covered=%.3g < %.3g — MOST deploy-grid columns "
            "fall back to background (a near-no-op correction); the production run will be "
            "largely UNCORRECTED. Sample more diverse environments in the campaign, deploy "
            "on a closer grid/climate, or set min_fraction_covered to fail loud.",
            coverage["fraction_covered"], _LOW_COVERAGE_WARN)
    if min_fraction_covered is not None:
        if not 0.0 <= min_fraction_covered <= 1.0:
            raise ValueError(
                f"min_fraction_covered must be in [0, 1]; got {min_fraction_covered}.")
        if coverage["fraction_covered"] < min_fraction_covered:
            raise ValueError(
                f"env-kernel deploy coverage fraction_covered="
                f"{coverage['fraction_covered']:.3g} is below the required "
                f"min_fraction_covered={min_fraction_covered:.3g}: most deploy-grid "
                "columns lie OUTSIDE the campaign's sampled environments, so the "
                "override is (near-)all background — a NO-OP correction. Sample more "
                "diverse environments in the campaign, deploy on a closer grid/climate, "
                "or lower min_fraction_covered to proceed knowingly.")
    override = TurbulenceConfig(
        scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(**{kernel.field: field}))
    return override, coverage


#: Schema tag stamped on a serialized kernel — labels the artifact the RAW
#: env→coefficient regression (full-step, unclipped, scalar out-of-hull fallback),
#: NOT a campaign's accepted/line-searched/accumulated feedback field. A deploy
#: re-applies it fresh on the target grid and runs its OWN line search + gate.
_ENV_KERNEL_ARTIFACT = "raw_environment_kernel"


def env_kernel_to_dict(kernel: EnvKernel) -> dict:
    """Serialize an :class:`EnvKernel` to a JSON-friendly dict (lists).

    Stamps ``"artifact": "raw_environment_kernel"`` so a consumer cannot mistake
    this for a campaign's accepted feedback field (see :data:`_ENV_KERNEL_ARTIFACT`).
    """
    import numpy as np

    return {
        "artifact": _ENV_KERNEL_ARTIFACT,
        "sample_env": np.asarray(kernel.sample_env).tolist(),
        "sample_values": np.asarray(kernel.sample_values).reshape(-1).tolist(),
        "valid": [bool(v) for v in np.asarray(kernel.valid).reshape(-1)],
        "length_scales": np.asarray(kernel.length_scales).reshape(-1).tolist(),
        "field": kernel.field,
        "background": float(kernel.background),
        "env_lo": np.asarray(kernel.env_lo).reshape(-1).tolist(),
        "env_hi": np.asarray(kernel.env_hi).reshape(-1).tolist(),
        "predictor_names": list(kernel.predictor_names),   # the column contract (iter 405)
    }


def env_kernel_from_dict(data: dict) -> EnvKernel:
    """Deserialize an :class:`EnvKernel` (inverse of :func:`env_kernel_to_dict`)."""
    import jax.numpy as jnp

    if not isinstance(data, dict):
        raise ValueError(
            f"env kernel must be a JSON object; got {type(data).__name__}.")
    # Reject a mislabelled artifact LOUDLY (a campaign accepted-field JSON is NOT a
    # kernel); a missing tag is tolerated for back-compat with pre-tag kernels.
    artifact = data.get("artifact")
    if artifact is not None and artifact != _ENV_KERNEL_ARTIFACT:
        raise ValueError(
            f"expected a {_ENV_KERNEL_ARTIFACT!r} artifact, got {artifact!r}.")
    # A truncated / corrupted kernel JSON missing a required key would otherwise raise a
    # bare KeyError; name the missing key(s) so the operator knows the artifact is bad
    # (parallel to the iter-106 ERA5-loader required-key hardening). 'artifact' is the only
    # optional key (back-compat with pre-tag kernels, handled above).
    required = ("sample_env", "sample_values", "valid", "length_scales", "field",
                "background", "env_lo", "env_hi")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"env kernel JSON is missing required key(s) {missing} — the artifact is "
            "truncated or corrupted; re-export it from the campaign "
            "(env_kernel_to_dict).")
    # ``_validate_env_kernel`` enforces field membership + every shape invariant
    # (a length-skewed ``env_lo`` would otherwise broadcast silently in the hull
    # check). Default float dtype (no forced float64) honours the x64 flag and
    # matches what ``build_env_kernel`` produces.
    return _validate_env_kernel(EnvKernel(
        sample_env=jnp.asarray(data["sample_env"]),
        sample_values=jnp.asarray(data["sample_values"]),
        valid=jnp.asarray(data["valid"], dtype=bool),
        length_scales=jnp.asarray(data["length_scales"]),
        field=data.get("field"), background=float(data["background"]),
        env_lo=jnp.asarray(data["env_lo"]),
        env_hi=jnp.asarray(data["env_hi"]),
        # Optional for back-compat with pre-iter-405 kernels (default the canonical
        # order); a present block round-trips exactly.
        predictor_names=tuple(data.get("predictor_names", ENV_PREDICTOR_NAMES))))
