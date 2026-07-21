"""Shared SCM-RCE profile, precipitation, and realism metrics.

The SCM campaign and the gradient trainer both compare single-column RCE
profiles to CRM truth with the same vertically mass-weighted,
standard-deviation-normalized RMSE.  Keep the arithmetic here so the
derivative-free and AD paths cannot drift.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat

# Shared, AD-safe profile primitives live in the low-level core module so the LES
# suite and the SCM-RCE metrics cannot drift. Re-exported here for the existing
# call sites that import them from this module.
from legoesm.core.profile_metrics import safe_sqrt, weighted_rmse, weighted_std


PRECIP_NORMALIZATION_MM_DAY = 3.0
PRECIP_SCORE_WEIGHT = 1.0


MADIAB_MEAN_TOL_K = 8.0
MADIAB_MAX_TOL_K = 30.0
TROP_MIN_Z_KM = 12.0
TROP_MAX_Z_KM = 25.0
COLD_POINT_MIN_K = 175.0
COLD_POINT_MAX_K = 210.0
MIN_FREE_TROP_LEVELS = 3


def score_profiles_jax(
    ref: Any,
    T_profile: jax.Array,
    qv_profile: jax.Array,
    qcond_profile: jax.Array,
    *,
    profile_floor: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return component and combined normalized SCM-vs-CRM profile scores."""
    T_profile = jnp.asarray(T_profile)
    qv_profile = jnp.asarray(qv_profile, dtype=T_profile.dtype)
    qcond_profile = jnp.asarray(qcond_profile, dtype=T_profile.dtype)
    dtype = T_profile.dtype
    weights = jnp.asarray(ref.mass_weights, dtype=dtype)
    T_ref = jnp.asarray(ref.T_ref, dtype=dtype)
    qv_ref = jnp.asarray(ref.qv_ref, dtype=dtype)
    qcond_ref = jnp.asarray(ref.qcond_ref, dtype=dtype)
    floor = jnp.asarray(profile_floor, dtype=dtype)

    T_std = jnp.maximum(weighted_std(T_ref, weights), floor)
    qv_std = jnp.maximum(weighted_std(qv_ref, weights), floor)
    qcond_std = jnp.maximum(weighted_std(qcond_ref, weights), floor)
    T_rmse = weighted_rmse((T_profile - T_ref) / T_std, weights)
    qv_rmse = weighted_rmse((qv_profile - qv_ref) / qv_std, weights)
    cloud_rmse = weighted_rmse((qcond_profile - qcond_ref) / qcond_std, weights)
    combined = safe_sqrt((T_rmse**2 + qv_rmse**2 + cloud_rmse**2) / 3.0)
    return T_rmse, qv_rmse, cloud_rmse, combined


def precip_score_jax(
    precip_mm_day: jax.Array | float,
    precip_ref_mm_day: jax.Array | float,
    *,
    normalization_mm_day: float = PRECIP_NORMALIZATION_MM_DAY,
) -> jax.Array:
    """Return normalized absolute surface-precipitation error.

    The SCM and CRM diagnostics use liquid-water-equivalent ``mm/day``.
    A 3 mm/day normalizer makes an order-1 score correspond to a large
    tropical-RCE precipitation error without letting precipitation dominate
    the profile RMSE terms.
    """
    precip = jnp.asarray(precip_mm_day)
    dtype = precip.dtype
    ref = jnp.asarray(precip_ref_mm_day, dtype=dtype)
    precip = jnp.where(jnp.isfinite(precip), precip, 0.0)
    ref = jnp.where(jnp.isfinite(ref), ref, 0.0)
    norm = jnp.maximum(
        jnp.asarray(normalization_mm_day, dtype=dtype),
        jnp.asarray(1.0e-12, dtype=dtype),
    )
    return jnp.abs(precip - ref) / norm


def score_profiles_precip_jax(
    ref: Any,
    T_profile: jax.Array,
    qv_profile: jax.Array,
    qcond_profile: jax.Array,
    precip_mm_day: jax.Array | float,
    *,
    profile_floor: float,
    precip_weight: float = PRECIP_SCORE_WEIGHT,
    precip_normalization_mm_day: float = PRECIP_NORMALIZATION_MM_DAY,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return profile component scores, precipitation score, and combined score."""
    T_rmse, qv_rmse, cloud_rmse, _old_combined = score_profiles_jax(
        ref,
        T_profile,
        qv_profile,
        qcond_profile,
        profile_floor=profile_floor,
    )
    precip_ref = getattr(ref, "precip_ref_mm_day", 0.0)
    precip_rmse = precip_score_jax(
        precip_mm_day,
        precip_ref,
        normalization_mm_day=precip_normalization_mm_day,
    )
    weight = jnp.asarray(precip_weight, dtype=T_rmse.dtype)
    combined = safe_sqrt(
        (T_rmse**2 + qv_rmse**2 + cloud_rmse**2 + weight * precip_rmse**2)
        / (3.0 + weight)
    )
    return T_rmse, qv_rmse, cloud_rmse, precip_rmse, combined


def moist_adiabat_diagnostics_jax(
    *,
    T_profile: jax.Array,
    qv_profile: jax.Array,
    p_full: jax.Array,
    sigma_full: jax.Array,
    z_m: jax.Array,
) -> dict[str, jax.Array]:
    """Diagnose surface-anchored moist-adiabat and cold-point realism.

    Arrays are top-to-bottom with the surface at ``[-1]``.  The free
    troposphere mask follows the existing RCE validation sweep: above the
    mixed layer (``sigma < 0.85``) and at/below the cold point.  The helper
    intentionally reuses :func:`compute_moist_adiabat` instead of duplicating
    saturation or lapse-rate formulae.
    """
    T = jnp.asarray(T_profile)
    qv = jnp.asarray(qv_profile, dtype=T.dtype)
    p = jnp.asarray(p_full, dtype=T.dtype)
    sigma = jnp.asarray(sigma_full, dtype=T.dtype)
    z = jnp.asarray(z_m, dtype=T.dtype)
    z_km = z / 1000.0

    parcel = compute_moist_adiabat(
        T[-1:],
        p[None, :],
        q_v_base=jnp.maximum(qv[-1:], jnp.asarray(0.0, dtype=T.dtype)),
    )[0]

    idx = jnp.arange(T.shape[0])
    cold_layer = (z_km >= TROP_MIN_Z_KM) & (z_km <= TROP_MAX_Z_KM)
    cold_idx = jnp.argmin(jnp.where(cold_layer, T, jnp.inf))
    cold_T = T[cold_idx]
    cold_z_km = z_km[cold_idx]
    free_trop = (sigma < 0.85) & (z_km <= cold_z_km)
    n_ft = jnp.sum(free_trop)
    dev = T - parcel
    abs_dev = jnp.abs(dev)
    mean_abs = jnp.where(
        n_ft >= MIN_FREE_TROP_LEVELS,
        jnp.sum(jnp.where(free_trop, abs_dev, 0.0)) / jnp.maximum(n_ft, 1),
        jnp.nan,
    )
    max_abs = jnp.where(
        n_ft >= MIN_FREE_TROP_LEVELS,
        jnp.max(jnp.where(free_trop, abs_dev, 0.0)),
        jnp.nan,
    )
    bias = jnp.where(
        n_ft >= MIN_FREE_TROP_LEVELS,
        jnp.sum(jnp.where(free_trop, dev, 0.0)) / jnp.maximum(n_ft, 1),
        jnp.nan,
    )
    return {
        "moist_adiabat": parcel,
        "mean_abs_K": mean_abs,
        "max_abs_K": max_abs,
        "bias_K": bias,
        "n_free_trop_levels": n_ft,
        "cold_point_T_K": cold_T,
        "cold_point_z_km": cold_z_km,
    }


def realism_reasons_from_diagnostics(
    diag: dict[str, float],
    *,
    mean_tol_K: float = MADIAB_MEAN_TOL_K,
    max_tol_K: float = MADIAB_MAX_TOL_K,
    cold_point_min_K: float = COLD_POINT_MIN_K,
    cold_point_max_K: float = COLD_POINT_MAX_K,
    trop_min_z_km: float = TROP_MIN_Z_KM,
    trop_max_z_km: float = TROP_MAX_Z_KM,
    min_free_trop_levels: int = MIN_FREE_TROP_LEVELS,
) -> list[str]:
    """Return human-readable realism-gate failures from scalar diagnostics."""
    reasons: list[str] = []
    # Fail CLOSED on a missing/NaN level count, consistent with the float fields below
    # (a NaN must REJECT the LES, never crash `int(nan)` and take the campaign down with it).
    n_ft_raw = float(diag.get("n_free_trop_levels", 0))
    n_ft = int(n_ft_raw) if math.isfinite(n_ft_raw) else 0
    if n_ft < min_free_trop_levels:
        reasons.append(f"too few free-troposphere levels ({n_ft})")
    mean_abs = float(diag.get("mean_abs_K", float("nan")))
    max_abs = float(diag.get("max_abs_K", float("nan")))
    if not math.isfinite(mean_abs) or mean_abs > mean_tol_K:
        reasons.append(
            f"far from moist adiabat (mean|T-T_moist|={mean_abs:.3g} K)"
        )
    if not math.isfinite(max_abs) or max_abs > max_tol_K:
        reasons.append(
            f"local moist-adiabat deviation {max_abs:.3g} K"
        )
    cold_T = float(diag.get("cold_point_T_K", float("nan")))
    cold_z = float(diag.get("cold_point_z_km", float("nan")))
    if not (cold_point_min_K <= cold_T <= cold_point_max_K):
        reasons.append(
            f"cold point T={cold_T:.3g} K outside "
            f"[{cold_point_min_K}, {cold_point_max_K}] K"
        )
    if not (trop_min_z_km <= cold_z <= trop_max_z_km):
        reasons.append(
            f"cold point z={cold_z:.3g} km outside "
            f"[{trop_min_z_km}, {trop_max_z_km}] km"
        )
    return reasons
