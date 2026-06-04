"""Field-comparison utilities (model vs reference) for ocean fidelity.

Pure numpy. Handles masked (e.g. land) cells and optional per-cell weights
(typically ``cos(lat) · dz``). Returns a :class:`ComparisonMetrics` named
tuple ready to feed pass/fail diffs.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

_EPS = 1e-30


class ComparisonMetrics(NamedTuple):
    """Bundle of standard comparison scores between model and reference."""

    rmse: float
    bias: float
    nrmse: float
    pattern_corr: float
    l2_area_weighted: float
    n_valid_cells: int


def _valid_mask(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    base = np.isfinite(arr)
    if mask is None:
        return base
    return base & (np.asarray(mask) > 0)


def compare_field(
    model_field: np.ndarray,
    reference_field: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> ComparisonMetrics:
    """Compare two fields of identical shape with optional mask + weights.

    Parameters
    ----------
    model_field, reference_field : np.ndarray
        Same shape. Comparison is element-wise.
    mask : np.ndarray, optional
        ``1 = valid``, ``0 = ignore`` (e.g. land). Same shape as the fields.
    weights : np.ndarray, optional
        Per-cell weights (e.g. ``cos(lat) · dz``). Default = uniform.

    Returns
    -------
    ComparisonMetrics
    """
    m = np.asarray(model_field, dtype=float)
    r = np.asarray(reference_field, dtype=float)
    if m.shape != r.shape:
        raise ValueError(
            f"shape mismatch: model {m.shape} vs reference {r.shape}"
        )
    if mask is not None and mask.shape != m.shape:
        raise ValueError(f"mask shape {mask.shape} must match field {m.shape}")
    if weights is not None and weights.shape != m.shape:
        raise ValueError(
            f"weights shape {weights.shape} must match field {m.shape}"
        )

    valid = _valid_mask(m, mask) & np.isfinite(r)
    w = np.ones_like(m) if weights is None else np.asarray(weights, dtype=float)
    w_valid = w * valid
    w_sum = float(w_valid.sum())
    n_valid = int(valid.sum())
    if w_sum <= _EPS or n_valid == 0:
        nan = float("nan")
        return ComparisonMetrics(nan, nan, nan, nan, nan, 0)

    # Replace masked / non-finite entries with 0 so that arithmetic does not
    # poison the reductions via ``0 * NaN = NaN``.
    m_safe = np.where(valid, m, 0.0)
    r_safe = np.where(valid, r, 0.0)
    diff = m_safe - r_safe
    bias = float((w_valid * diff).sum() / w_sum)
    mse = float((w_valid * diff * diff).sum() / w_sum)
    rmse = math.sqrt(mse)
    r_valid_vals = r_safe[valid]
    r_range = float(r_valid_vals.max() - r_valid_vals.min())
    nrmse = rmse / max(r_range, _EPS)

    m_mean = float((w_valid * m_safe).sum() / w_sum)
    r_mean = float((w_valid * r_safe).sum() / w_sum)
    m_anom = (m_safe - m_mean) * valid
    r_anom = (r_safe - r_mean) * valid
    num = float((w_valid * m_anom * r_anom).sum())
    den_m = float((w_valid * m_anom * m_anom).sum())
    den_r = float((w_valid * r_anom * r_anom).sum())
    denom = math.sqrt(max(den_m, _EPS) * max(den_r, _EPS))
    pattern_corr = num / denom if denom > 0 else float("nan")

    l2_area_weighted = math.sqrt(float((w_valid * diff * diff).sum()))

    return ComparisonMetrics(
        rmse=rmse,
        bias=bias,
        nrmse=nrmse,
        pattern_corr=pattern_corr,
        l2_area_weighted=l2_area_weighted,
        n_valid_cells=n_valid,
    )


def compare_zonal_mean(
    model_field: np.ndarray,
    reference_field: np.ndarray,
    *,
    lat_rad: np.ndarray,
    mask: np.ndarray | None = None,
) -> ComparisonMetrics:
    """Compare zonal means of two 2-D ``(n_lat, n_lon)`` fields.

    Reduction over the longitude axis respects ``mask`` (1 = valid). The
    resulting 1-D profiles are then compared with ``cos(lat)`` weights.
    """
    m = np.asarray(model_field, dtype=float)
    r = np.asarray(reference_field, dtype=float)
    if m.ndim != 2 or r.shape != m.shape:
        raise ValueError(
            f"expected matching 2-D arrays, got {m.shape} and {r.shape}"
        )
    lat = np.asarray(lat_rad, dtype=float)
    if lat.shape != (m.shape[0],):
        raise ValueError(
            f"lat_rad must have shape ({m.shape[0]},), got {lat.shape}"
        )
    if mask is None:
        m_zm = m.mean(axis=-1)
        r_zm = r.mean(axis=-1)
    else:
        if mask.shape != m.shape:
            raise ValueError("mask shape must match field")
        denom = np.maximum(np.asarray(mask, dtype=float).sum(axis=-1), 1.0)
        m_zm = (m * mask).sum(axis=-1) / denom
        r_zm = (r * mask).sum(axis=-1) / denom
    return compare_field(m_zm, r_zm, weights=np.cos(lat))


def compare_moc(
    model_psi: np.ndarray,
    reference_psi: np.ndarray,
    *,
    lat_rad: np.ndarray,
    depth: np.ndarray,
) -> ComparisonMetrics:
    """Compare meridional-overturning streamfunctions ``ψ(lat, depth)``.

    Both inputs are 2-D with the latitude axis first and the depth axis
    second. Weights are ``cos(lat)`` along the latitude axis, broadcast
    across depth.
    """
    m = np.asarray(model_psi, dtype=float)
    r = np.asarray(reference_psi, dtype=float)
    if m.ndim != 2 or r.shape != m.shape:
        raise ValueError(
            f"expected matching 2-D arrays, got {m.shape} and {r.shape}"
        )
    lat = np.asarray(lat_rad, dtype=float)
    z = np.asarray(depth, dtype=float)
    if lat.shape != (m.shape[0],):
        raise ValueError(
            f"lat_rad must have shape ({m.shape[0]},), got {lat.shape}"
        )
    if z.shape != (m.shape[1],):
        raise ValueError(
            f"depth must have shape ({m.shape[1]},), got {z.shape}"
        )
    weights = np.broadcast_to(np.cos(lat)[:, None], m.shape).copy()
    return compare_field(m, r, weights=weights)
