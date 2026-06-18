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
from typing import Any

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
    lengths = {int(a.shape[0]) for a in per_column.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"per-column override fields have inconsistent column counts {lengths}; "
            "every corrected field must span the same global grid.")
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
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    clubb, per_column = _per_column_clubb_fields(override)
    idx = jnp.asarray(local_column_indices).reshape(-1)
    if not jnp.issubdtype(idx.dtype, jnp.integer):
        raise ValueError(
            f"local_column_indices must be integer; got dtype {idx.dtype}.")
    if per_column:
        ncol = int(next(iter(per_column.values())).shape[0])
        lo, hi = int(idx.min()), int(idx.max())
        if lo < 0 or hi >= ncol:
            raise ValueError(
                f"local_column_indices out of range [0, {ncol}) for the global "
                f"override: got [{lo}, {hi}].")
    sliced = {f: jnp.take(arr, idx, axis=0) for f, arr in per_column.items()}
    return TurbulenceConfig(scheme="clubb_lite", clubb_lite=clubb._replace(**sliced))


def slice_override_latlon_2d(override, layout):
    """Slice a GLOBAL per-column clubb override to a rank's LOCAL lat-lon tile.

    The safe lat-lon path: each corrected ``(ncol,)`` field is reshaped to the
    global ``(n_lat, n_lon)`` and sliced with the EXACT same partition the model
    scatters state with (:func:`legoesm.parallel.latlon_mpi.scatter_field_latlon_2d`
    — the rank's ``[lat_start:lat_end, lon_start:lon_end]`` cell-centered block),
    then row-major-flattened to ``(ncol_local,)``.  Because it reuses the model's
    own cell-centered scatter, the local override lands on exactly the columns the
    rank's physics consumes (the row-major ColumnAdapter flatten).  ``layout`` is a
    ``LatLon2DLayout`` (``make_latlon_2d_layout(0, 1, 1, ...)`` is the single-rank
    identity).  lat-lon cell-centered only; non-lat-lon decompositions supply their
    own indices to :func:`slice_override_columns`.
    """

    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.parallel.latlon_mpi import scatter_field_latlon_2d

    clubb, per_column = _per_column_clubb_fields(override)
    n_lat, n_lon = int(layout.n_lat_global), int(layout.n_lon_global)
    ncol = n_lat * n_lon
    sliced = {}
    for field, arr in per_column.items():
        if int(arr.shape[0]) != ncol:
            raise ValueError(
                f"per-column override field '{field}' has {int(arr.shape[0])} "
                f"columns but the layout's global grid is {n_lat}x{n_lon}={ncol}.")
        local = scatter_field_latlon_2d(arr.reshape(n_lat, n_lon), layout)
        sliced[field] = local.reshape(-1)
    return TurbulenceConfig(scheme="clubb_lite", clubb_lite=clubb._replace(**sliced))
