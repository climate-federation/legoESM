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


PRECIP_NORMALIZATION_MM_DAY = 3.0
PRECIP_SCORE_WEIGHT = 1.0


def safe_sqrt(x: jax.Array) -> jax.Array:
    """``sqrt`` with a finite gradient at ``x == 0``.

    ``sqrt(0)`` is itself finite (0) but its derivative ``1/(2 sqrt(0)) = inf``,
    so a PERFECT fit (residual == 0 — the exact minimiser the trainer targets)
    produces a NaN gradient.  The double-``where`` masks the zero out of the
    branch that is differentiated, giving the EXACT value ``sqrt(0) == 0`` on
    the forward pass AND a finite (zero) gradient on the backward pass — so the
    existing "metrics are exactly 0 at a matching profile" invariant is kept.
    """
    x = jnp.asarray(x)
    safe_x = jnp.where(x > 0, x, jnp.ones_like(x))
    return jnp.where(x > 0, jnp.sqrt(safe_x), jnp.zeros_like(x))

MADIAB_MEAN_TOL_K = 8.0
MADIAB_MAX_TOL_K = 30.0
TROP_MIN_Z_KM = 12.0
TROP_MAX_Z_KM = 25.0
COLD_POINT_MIN_K = 175.0
COLD_POINT_MAX_K = 210.0
MIN_FREE_TROP_LEVELS = 3


def weighted_std(profile: jax.Array, weights: jax.Array) -> jax.Array:
    """Mass-weighted vertical standard deviation."""
    profile = jnp.asarray(profile)
    weights = jnp.asarray(weights, dtype=profile.dtype)
    mean = jnp.sum(weights * profile)
    var = jnp.sum(weights * (profile - mean) ** 2)
    # safe_sqrt keeps the gradient finite for a perfectly uniform profile
    # (var == 0) while still returning exactly 0.
    return safe_sqrt(var)


def weighted_rmse(diff: jax.Array, weights: jax.Array) -> jax.Array:
    """Mass-weighted vertical RMSE.

    Uses a floored sqrt so the gradient stays finite at a perfect fit
    (``diff == 0`` makes ``d sqrt(s)/ds = 1/(2 sqrt(s))`` blow up at ``s == 0``).
    """
    diff = jnp.asarray(diff)
    weights = jnp.asarray(weights, dtype=diff.dtype)
    return safe_sqrt(jnp.sum(weights * diff ** 2))


def normalized_profile_rmse(
    ref_profile: jax.Array,
    profile: jax.Array,
    weights: jax.Array,
    *,
    profile_floor: jax.Array | float,
) -> jax.Array:
    """Mass-weighted RMSE of ``profile`` vs ``ref_profile``, normalized by the
    reference's own mass-weighted standard deviation (floored).

    The single-variable core of every profile score in this module and in the
    LES-vs-SCM turbulence scoring: normalizing by the reference's spread makes
    variables with wildly different units (K vs kg/kg vs K m/s) commensurable,
    so they can be combined in quadrature without an arbitrary unit weight.
    """
    ref_profile = jnp.asarray(ref_profile)
    dtype = ref_profile.dtype
    profile = jnp.asarray(profile, dtype=dtype)
    weights = jnp.asarray(weights, dtype=dtype)
    floor = jnp.asarray(profile_floor, dtype=dtype)
    std = jnp.maximum(weighted_std(ref_profile, weights), floor)
    return weighted_rmse((profile - ref_profile) / std, weights)


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

    T_rmse = normalized_profile_rmse(T_ref, T_profile, weights,
                                     profile_floor=floor)
    qv_rmse = normalized_profile_rmse(qv_ref, qv_profile, weights,
                                      profile_floor=floor)
    cloud_rmse = normalized_profile_rmse(qcond_ref, qcond_profile, weights,
                                         profile_floor=floor)
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


#: Top of the layer treated as "sub-cloud" [m].  Set from the CRM reference's
#: own structure rather than convention: its condensate has a shallow-cumulus
#: maximum at 1.3 km, so 1 km is below cloud base and the layer being scored is
#: the one the surface fluxes and the downdrafts ventilate.  Configurable
#: because a different reference has a different cloud base.
DEFAULT_SUBCLOUD_TOP_M = 1000.0


def subcloud_mass_weights(
    z_m: jax.Array,
    mass_weights: jax.Array,
    *,
    top_m: float = DEFAULT_SUBCLOUD_TOP_M,
) -> jax.Array:
    """Mass weights restricted to ``z <= top_m`` and renormalized to sum to 1.

    A column-mean RMSE dilutes the sub-cloud layer to invisibility: the layer
    holds a few percent of the column mass, so an error there moves the total
    score by less than the tuner's noise.  Scoring it with its own renormalized
    weights makes it a first-class target while reusing the SAME normalized-RMSE
    arithmetic as the full-column score, so the two are directly comparable.

    Selecting no level is a hard error, never an all-zero weight vector that
    would silently score every column identically.
    """
    z_m = jnp.asarray(z_m)
    mass_weights = jnp.asarray(mass_weights, dtype=z_m.dtype)
    if z_m.shape != mass_weights.shape:
        raise ValueError(
            f"subcloud_mass_weights: z_m {z_m.shape} and mass_weights "
            f"{mass_weights.shape} must share a shape.")
    mask = z_m <= jnp.asarray(top_m, dtype=z_m.dtype)
    n_selected = int(jnp.sum(mask))
    if n_selected == 0:
        raise ValueError(
            f"subcloud_mass_weights: no level at or below {top_m} m "
            f"(lowest level is {float(jnp.min(z_m)):.1f} m).")
    selected = jnp.where(mask, mass_weights, jnp.zeros_like(mass_weights))
    total = jnp.sum(selected)
    return selected / total


def subcloud_bulk_state(
    *,
    T_air_K: jax.Array | float,
    r_air: jax.Array | float,
    p_air_Pa: jax.Array | float,
    sst_K: jax.Array | float,
    p_sfc_Pa: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Near-surface bulk-flux state: ``(relative_humidity, driver, delta_T)``.

    ``driver = r_sat(SST, p_sfc) - r_air`` [kg/kg] is the air-sea humidity
    difference the bulk evaporation is proportional to, and ``delta_T =
    SST - T_air`` [K] the one the sensible flux is proportional to.  Both are
    formed from MIXING RATIOS on both sides — the CRM reference carries SAM's
    native ``QV_avg``, a mixing ratio, and the SCM's ``q_v`` tracer is the same
    quantity, so no specific-vs-mixing conversion belongs anywhere here.

    Saturation comes from ``legoesm.thermo``; this module contains no
    saturation numerics of its own.
    """
    from legoesm.thermo import saturation_mixing_ratio

    T_air_K = jnp.asarray(T_air_K)
    dtype = T_air_K.dtype
    r_air = jnp.asarray(r_air, dtype=dtype)
    r_sat_sfc = saturation_mixing_ratio(
        jnp.asarray(sst_K, dtype=dtype), jnp.asarray(p_sfc_Pa, dtype=dtype))
    r_sat_air = saturation_mixing_ratio(
        T_air_K, jnp.asarray(p_air_Pa, dtype=dtype))
    return (r_air / r_sat_air,
            r_sat_sfc - r_air,
            jnp.asarray(sst_K, dtype=dtype) - T_air_K)


#: FIXED scales for the sub-cloud objective.  Deliberately NOT the reference's
#: own spread across the layer: over a nearly well-mixed kilometre that spread
#: is small and mostly uninformative, and it sets the T-versus-q_v weight to
#: whatever happens to fall out of the reference profile.  Measured on the
#: SAM_CRM RCE_small300 sounding, sigma_qv over the sub-cloud layer is ~0.07
#: g/kg while sigma_T is ~2.3 K, so a harmless 0.1 g/kg humidity error would
#: outweigh a 3 K temperature bias.  A declared tolerance says what "wrong by
#: one unit" means instead of discovering it.
DEFAULT_SUBCLOUD_T_SCALE_K = 1.0
DEFAULT_SUBCLOUD_QV_SCALE = 1.0e-3          # 1 g/kg
DEFAULT_SUBCLOUD_EVAP_SCALE_MM_DAY = 0.5


def _weighted_mean(profile: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.sum(weights * profile)


def score_subcloud_jax(
    ref: Any,
    T_profile: jax.Array,
    qv_profile: jax.Array,
    *,
    subcloud_weights: jax.Array,
    T_scale_K: float = DEFAULT_SUBCLOUD_T_SCALE_K,
    qv_scale: float = DEFAULT_SUBCLOUD_QV_SCALE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sub-cloud ``(T_term, qv_term, combined)`` on FIXED physical scales.

    Each variable contributes two numbers, kept separate because they are
    different defects with different fixes:

    * the layer-mean BIAS — is the sub-cloud layer too warm / too moist, which
      is the measured defect and what the surface fluxes respond to;
    * the demeaned SHAPE error — is its internal structure wrong.

    They are combined in quadrature per variable and then across variables,
    every term divided by a declared tolerance rather than by the reference's
    own spread.  Condensate is deliberately absent: this objective exists
    because the column score is condensate-dominated, and re-admitting it here
    would reproduce exactly that.
    """
    T_profile = jnp.asarray(T_profile)
    dtype = T_profile.dtype
    qv_profile = jnp.asarray(qv_profile, dtype=dtype)
    weights = jnp.asarray(subcloud_weights, dtype=dtype)
    T_ref = jnp.asarray(ref.T_ref, dtype=dtype)
    qv_ref = jnp.asarray(ref.qv_ref, dtype=dtype)

    def _term(profile, reference, scale):
        diff = (profile - reference) / jnp.asarray(scale, dtype=dtype)
        bias = _weighted_mean(diff, weights)
        shape = weighted_rmse(diff - bias, weights)
        return safe_sqrt(bias ** 2 + shape ** 2)

    T_term = _term(T_profile, T_ref, T_scale_K)
    qv_term = _term(qv_profile, qv_ref, qv_scale)
    return T_term, qv_term, safe_sqrt((T_term ** 2 + qv_term ** 2) / 2.0)


def subcloud_objective_jax(
    ref: Any,
    T_profile: jax.Array,
    qv_profile: jax.Array,
    evap_mm_day: jax.Array | float,
    *,
    subcloud_weights: jax.Array,
    T_scale_K: float = DEFAULT_SUBCLOUD_T_SCALE_K,
    qv_scale: float = DEFAULT_SUBCLOUD_QV_SCALE,
    evap_scale_mm_day: float = DEFAULT_SUBCLOUD_EVAP_SCALE_MM_DAY,
    evap_weight: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """``(T_term, qv_term, evap_term, combined)`` — the full sub-cloud target.

    The evaporation term is not decoration.  The defect that motivated this
    objective was measured as a FLUX deficit (1.2-1.9 mm/day against the
    reference's 2.73), and T and q_v at the lowest level are its downstream
    consequences.  An objective that scores only the state variables can be
    satisfied by getting them right for the wrong reason — a column can carry
    the right sub-cloud humidity while exchanging far too little water with the
    surface.  Scoring the flux alongside the state closes that.

    ``ref.evap_ref_mm_day`` is used when the reference carries a MEASURED
    surface latent heat flux; otherwise the caller is expected to have fallen
    back to the equilibrium identity E = P and to have said so.
    """
    T_term, qv_term, _profile_only = score_subcloud_jax(
        ref, T_profile, qv_profile, subcloud_weights=subcloud_weights,
        T_scale_K=T_scale_K, qv_scale=qv_scale)
    dtype = T_term.dtype
    evap_ref = jnp.asarray(
        getattr(ref, "evap_ref_mm_day", None)
        if getattr(ref, "evap_ref_mm_day", None) is not None
        else getattr(ref, "precip_ref_mm_day", 0.0), dtype=dtype)
    evap = jnp.asarray(evap_mm_day, dtype=dtype)
    evap = jnp.where(jnp.isfinite(evap), evap, evap_ref)
    evap_term = jnp.abs(evap - evap_ref) / jnp.asarray(
        max(float(evap_scale_mm_day), 1.0e-12), dtype=dtype)
    w = jnp.asarray(evap_weight, dtype=dtype)
    combined = safe_sqrt(
        (T_term ** 2 + qv_term ** 2 + w * evap_term ** 2) / (2.0 + w))
    return T_term, qv_term, evap_term, combined


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
