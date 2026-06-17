"""Worst-column manifest + environment tagging for LES spin-off.

Stage 3 of ``docs/COMPARE_REANALYSIS.md`` (gap #2): turn the per-column
model-vs-ERA5 scores from :mod:`legoesm.training.column_era5_metrics` into a
small, serializable manifest of the ``top-N`` worst columns.  Each record
carries the grid index + coordinates + time, the component error breakdown, and
**environment tags** (SST, CAPE, bulk wind shear) used later to cluster columns
and to choose an LES regime (shallow vs deep) per column.

Design (CLAUDE.md):

* **Reuse, don't re-derive.**  CAPE comes from the canonical
  :func:`legoesm.atmosphere.physics.thermodynamics.parcel_profile_and_cape`
  (the same surface-parcel recipe every convection scheme uses); ranking comes
  from :func:`legoesm.training.column_era5_metrics.rank_worst_columns`.  The one
  genuinely new diagnostic is a documented **bulk wind shear** (no canonical
  column-shear helper exists).
* **Pure-JAX environment fields.**  :func:`compute_column_environment` is a pure
  array function (vmap/jit-friendly over arbitrary leading column dims).  Manifest
  *assembly* (:func:`build_worst_column_manifest`) is intentionally host-side: it
  produces a Python list of records for serialization, not a traced array.
* **No hardcoded tunables.**  The shear reference sigma levels live in
  :class:`EnvironmentConfig` with documented defaults.

The full large-scale column *state* (profiles) for LES forcing is **not** stored
in the manifest — it is re-extracted from the saved AMIP state at the recorded
``(time_index, grid_index)`` in the stage-4 forcing extractor.  The manifest is
the lightweight index that drives that stage.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.thermodynamics import parcel_profile_and_cape
from legoesm.training.column_era5_metrics import ColumnErrorFields, rank_worst_columns

# --- bulk-shear reference levels (sigma) ------------------------------------
# Lower ~ near top of boundary layer, upper ~ upper troposphere.  These select
# *which* model levels define the bulk shear; they are diagnostic reference
# levels, not tunable physics.
_SHEAR_LOWER_SIGMA = 0.85
_SHEAR_UPPER_SIGMA = 0.25


class EnvironmentConfig(NamedTuple):
    """Reference sigma levels for the bulk wind-shear environment tag."""

    shear_lower_sigma: float = _SHEAR_LOWER_SIGMA
    shear_upper_sigma: float = _SHEAR_UPPER_SIGMA


class ColumnEnvironmentFields(NamedTuple):
    """Per-column environment predictors (physical units), column-shaped."""

    sst_K: jax.Array
    cape_J_kg: jax.Array
    bulk_shear_m_s: jax.Array


class ColumnEnvironment(NamedTuple):
    """Scalar environment tags for one manifest record."""

    sst_K: float
    cape_J_kg: float
    bulk_shear_m_s: float


class ColumnRecord(NamedTuple):
    """One worst-performing column: identity, errors, and environment tags."""

    flat_index: int
    grid_index: tuple[int, ...]
    lat_deg: float
    lon_deg: float
    time_index: int
    combined_score: float
    T_rmse_K: float
    qv_rmse_kg_kg: float
    wind_rmse_m_s: float
    precip_err_mm_day: float
    environment: ColumnEnvironment


def _nearest_level(sigma_full: jax.Array, target: float) -> jax.Array:
    """Index of the model level nearest ``target`` sigma.

    JAX-native (``jnp.argmin``) so it composes under ``jax.jit`` even when
    ``sigma_full`` is a traced argument; the returned index is used with
    :func:`jax.numpy.take` (dynamic gather) rather than a static slice.
    """
    sigma = jnp.asarray(sigma_full)
    return jnp.argmin(jnp.abs(sigma - jnp.asarray(target, dtype=sigma.dtype)))


def compute_bulk_shear(
    u: jax.Array,
    v: jax.Array,
    sigma_full: jax.Array,
    config: EnvironmentConfig = EnvironmentConfig(),
) -> jax.Array:
    """Per-column bulk wind shear ``|V(upper) - V(lower)|`` [m/s].

    ``u``/``v`` are ``[..., nlev]`` (surface-last) on sigma levels
    ``sigma_full`` ``[nlev]``.  The lower/upper reference levels are the model
    levels nearest ``config.shear_lower_sigma`` / ``shear_upper_sigma``.  The
    shear is the magnitude of the *vector* wind difference, so a turning wind
    counts even at constant speed.
    """
    u = jnp.asarray(u)
    v = jnp.asarray(v, dtype=u.dtype)
    k_lo = _nearest_level(sigma_full, config.shear_lower_sigma)
    k_hi = _nearest_level(sigma_full, config.shear_upper_sigma)
    du = jnp.take(u, k_hi, axis=-1) - jnp.take(u, k_lo, axis=-1)
    dv = jnp.take(v, k_hi, axis=-1) - jnp.take(v, k_lo, axis=-1)
    return jnp.sqrt(du * du + dv * dv)


def compute_column_environment(
    *,
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    sst: jax.Array,
    sigma_full: jax.Array,
    config: EnvironmentConfig = EnvironmentConfig(),
) -> ColumnEnvironmentFields:
    """Compute per-column environment tags (SST, CAPE, bulk shear).

    Profiles ``T``/``q_v``/``u``/``v``/``p_full`` are ``[..., nlev]``
    (surface-last); ``p_half`` is ``[..., nlev+1]``; ``sst`` is ``[...]``;
    ``sigma_full`` is ``[nlev]``.  CAPE reuses the canonical surface-parcel
    recipe :func:`parcel_profile_and_cape` (virtual-temperature CAPE with the
    surface humidity launch), reshaping the leading column dims to ``(ncol,
    nlev)`` for it and back.
    """
    T = jnp.asarray(T)
    q_v = jnp.asarray(q_v, dtype=T.dtype)
    p_full = jnp.asarray(p_full, dtype=T.dtype)
    p_half = jnp.asarray(p_half, dtype=T.dtype)

    lead = T.shape[:-1]
    nlev = T.shape[-1]
    T2 = T.reshape(-1, nlev)
    q2 = q_v.reshape(-1, nlev)
    pf2 = p_full.reshape(-1, nlev)
    ph2 = p_half.reshape(-1, nlev + 1)
    _, cape_flat = parcel_profile_and_cape(T2, pf2, ph2, q_v=q2)
    cape = cape_flat.reshape(lead)

    shear = compute_bulk_shear(u, v, sigma_full, config)
    return ColumnEnvironmentFields(
        sst_K=jnp.asarray(sst, dtype=T.dtype),
        cape_J_kg=cape,
        bulk_shear_m_s=shear,
    )


def build_worst_column_manifest(
    *,
    error_fields: ColumnErrorFields,
    environment: ColumnEnvironmentFields,
    lat_deg: jax.Array,
    lon_deg: jax.Array,
    time_index: int,
    n: int,
    valid_mask: jax.Array | None = None,
) -> list[ColumnRecord]:
    """Assemble the manifest of the ``n`` worst columns (host-side).

    ``error_fields`` / ``environment`` are column-shaped (e.g. ``(n_lat,
    n_lon)`` or ``(6, n, n)``); ``lat_deg`` / ``lon_deg`` broadcast to that
    column shape (a lat-lon grid passes 1-D vectors that are meshed here).
    Columns flagged invalid by :func:`rank_worst_columns` (when ``n`` exceeds
    the valid-column count) are dropped, so the returned list has length
    ``min(n, n_valid_columns)``.
    """
    shape = error_fields.combined_score.shape
    idx, scores, valid = rank_worst_columns(
        error_fields.combined_score, n, valid_mask=valid_mask
    )
    idx = np.asarray(idx)
    valid = np.asarray(valid)

    combined = np.asarray(error_fields.combined_score).reshape(-1)
    T_rmse = np.asarray(error_fields.T_rmse_K).reshape(-1)
    qv_rmse = np.asarray(error_fields.qv_rmse_kg_kg).reshape(-1)
    wind_rmse = np.asarray(error_fields.wind_rmse_m_s).reshape(-1)
    precip_err = np.asarray(error_fields.precip_err_mm_day).reshape(-1)
    sst = np.asarray(environment.sst_K).reshape(-1)
    cape = np.asarray(environment.cape_J_kg).reshape(-1)
    shear = np.asarray(environment.bulk_shear_m_s).reshape(-1)

    lat = np.asarray(lat_deg)
    lon = np.asarray(lon_deg)
    lat_b, lon_b = _broadcast_coords(lat, lon, shape)
    lat_flat = lat_b.reshape(-1)
    lon_flat = lon_b.reshape(-1)

    records: list[ColumnRecord] = []
    for flat_i, is_valid in zip(idx.tolist(), valid.tolist()):
        if not is_valid:
            continue
        grid_index = tuple(int(c) for c in np.unravel_index(flat_i, shape))
        records.append(
            ColumnRecord(
                flat_index=int(flat_i),
                grid_index=grid_index,
                lat_deg=float(lat_flat[flat_i]),
                lon_deg=float(lon_flat[flat_i]),
                time_index=int(time_index),
                combined_score=float(combined[flat_i]),
                T_rmse_K=float(T_rmse[flat_i]),
                qv_rmse_kg_kg=float(qv_rmse[flat_i]),
                wind_rmse_m_s=float(wind_rmse[flat_i]),
                precip_err_mm_day=float(precip_err[flat_i]),
                environment=ColumnEnvironment(
                    sst_K=float(sst[flat_i]),
                    cape_J_kg=float(cape[flat_i]),
                    bulk_shear_m_s=float(shear[flat_i]),
                ),
            )
        )
    return records


def _broadcast_coords(
    lat: np.ndarray, lon: np.ndarray, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast lat/lon to the full column ``shape``.

    Accepts coordinates already shaped like the grid (returned as-is) or, for a
    rectilinear lat-lon grid, 1-D ``lat[n_lat]`` + ``lon[n_lon]`` vectors that
    are meshed to ``(n_lat, n_lon)``.  Raises on anything else so a mismatched
    grid fails loudly instead of mis-indexing.
    """
    if lat.shape == shape and lon.shape == shape:
        return lat, lon
    if (
        len(shape) == 2
        and lat.ndim == 1
        and lon.ndim == 1
        and lat.shape[0] == shape[0]
        and lon.shape[0] == shape[1]
    ):
        lat_b, lon_b = np.meshgrid(lat, lon, indexing="ij")
        return lat_b, lon_b
    raise ValueError(
        f"lat/lon shapes {lat.shape}/{lon.shape} are not broadcastable to the "
        f"column grid shape {shape}; pass grid-shaped coordinates or 1-D "
        f"lat[n_lat]+lon[n_lon] for a rectilinear grid."
    )


def manifest_to_dicts(records: list[ColumnRecord]) -> list[dict]:
    """Convert manifest records to plain JSON-serializable dicts."""
    out: list[dict] = []
    for r in records:
        d = r._asdict()
        d["grid_index"] = list(r.grid_index)
        d["environment"] = r.environment._asdict()
        out.append(d)
    return out


def dicts_to_manifest(dicts: list[dict]) -> list[ColumnRecord]:
    """Inverse of :func:`manifest_to_dicts`."""
    records: list[ColumnRecord] = []
    for d in dicts:
        env = ColumnEnvironment(**d["environment"])
        records.append(
            ColumnRecord(
                flat_index=int(d["flat_index"]),
                grid_index=tuple(int(c) for c in d["grid_index"]),
                lat_deg=float(d["lat_deg"]),
                lon_deg=float(d["lon_deg"]),
                time_index=int(d["time_index"]),
                combined_score=float(d["combined_score"]),
                T_rmse_K=float(d["T_rmse_K"]),
                qv_rmse_kg_kg=float(d["qv_rmse_kg_kg"]),
                wind_rmse_m_s=float(d["wind_rmse_m_s"]),
                precip_err_mm_day=float(d["precip_err_mm_day"]),
                environment=env,
            )
        )
    return records


def write_manifest(records: list[ColumnRecord], path: str) -> None:
    """Write the manifest to ``path`` as JSON."""
    with open(path, "w") as f:
        json.dump(manifest_to_dicts(records), f, indent=2)


def read_manifest(path: str) -> list[ColumnRecord]:
    """Read a manifest written by :func:`write_manifest`."""
    with open(path) as f:
        return dicts_to_manifest(json.load(f))
