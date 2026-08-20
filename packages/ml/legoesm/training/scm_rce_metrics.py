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


#: FIXED tolerances for the THERMODYNAMIC objective — the target used when the
#: campaign is asked to match the CRM's temperature and humidity rather than its
#: condensate.  Declared, not discovered: normalizing by the reference's own
#: mass-weighted spread makes the T-versus-humidity trade-off whatever the
#: reference profile happens to be, and for a tropical RCE column that is
#: sigma_T ~ 30 K against sigma_qv ~ 4.5 g/kg, i.e. a 1 g/kg humidity error
#: would cost about SEVEN TIMES a 1 K temperature error without anyone having
#: decided that.  These say what "wrong by one unit" means.
DEFAULT_THERMO_T_SCALE_K = 1.0
#: 5 % absolute relative humidity.  The free-tropospheric RH spread ACROSS the
#: RCEMIP CRM ensemble is several times this, so a column inside one tolerance
#: of the reference is inside the ensemble's own disagreement.
DEFAULT_THERMO_RH_SCALE = 0.05
#: 10 % FRACTIONAL humidity error, the tolerance for the ``logq`` variable
#: (``|d ln q| = 0.1`` is a 10.5 % error).  Chosen for rough radiative parity
#: with the 1 K temperature tolerance: a 1 K free-tropospheric temperature error
#: is a few W/m^2 of OLR, and a ~10 % free-tropospheric humidity error is of the
#: same order.  Declared, like the others, rather than discovered.
DEFAULT_THERMO_LOGQ_SCALE = 0.10
#: Floor inside the log [kg/kg].  The reference's driest level is ~1e-6 kg/kg,
#: so a floor two decades below it never touches a real level; it exists so a
#: model column that dries to exactly zero yields a large finite penalty rather
#: than -inf, which would make every candidate compare False and silently
#: discard the search.
DEFAULT_THERMO_LOGQ_FLOOR = 1.0e-8

#: Which humidity variable the objective minimises.
#:
#: ``logq`` (default) scores FRACTIONAL humidity error.  It needs no saturation
#: curve, so it carries no liquid/ice phase convention and cannot double-count a
#: temperature error, and — MEASURED on the SAM_CRM reference under a uniform
#: relative perturbation — its leverage is distributed exactly like the column
#: mass (56.6 % above 5 km) instead of the 1.7 % an absolute ``q_v`` RMSE gives.
#:
#: ``rh`` is the RCEMIP-conventional diagnostic and is reported either way.  Its
#: KNOWN cost, quantified: ``RH = q_v/q_sat(T)`` and ``dlnq_sat/dT ~ 0.06 /K``
#: at 300 K rising to ~0.12 /K near the cold point, so at a 5 % RH tolerance the
#: humidity term ALSO demands ~0.8 K near the surface and ~0.4 K aloft — i.e. it
#: silently re-weights temperature by 2-4x, unevenly in the vertical, and a
#: compensating warm-and-moist bias can satisfy it exactly.
THERMO_HUMIDITY_VARIABLES = ("logq", "rh")
DEFAULT_THERMO_HUMIDITY = "logq"
#: Below this pressure the column is stratospheric: humidity there is tiny,
#: radiatively driven rather than convectively driven, and the RH ratio becomes
#: numerically fragile because q_sat over ice at ~190 K is ~1e-6 kg/kg.  A FIXED
#: pressure bound (not the diagnosed cold point) so the SCM and the CRM are
#: masked identically — a state-dependent mask would score two different domains.
DEFAULT_THERMO_MIN_P_PA = 10_000.0
#: Liquid/ice blend width for the saturation curve used on BOTH sides.
DEFAULT_THERMO_RH_BLEND_WIDTH_K = 20.0


def relative_humidity_profile(
    T_profile: jax.Array,
    qv_profile: jax.Array,
    p_profile: jax.Array,
    *,
    blend_width_K: float = DEFAULT_THERMO_RH_BLEND_WIDTH_K,
) -> jax.Array:
    """``r_v / r_sat(T, p)`` with a liquid/ice-blended saturation curve.

    MIXING RATIOS on both sides: the CRM reference carries SAM's native mixing
    ratio and the SCM's ``q_v`` tracer is the same quantity, so no
    specific-vs-mixing conversion belongs here (same argument as
    :func:`subcloud_bulk_state`).

    The blend matters: a tropical RCE column is below freezing above ~5 km, and
    scoring an ice-phase level against a liquid-only saturation curve inflates
    its RH by tens of percent — exactly the levels this objective exists to see.
    Saturation comes from ``legoesm.thermo``; nothing is re-derived here.
    """
    from legoesm.thermo import saturation_mixing_ratio_blend

    T_profile = jnp.asarray(T_profile)
    dtype = T_profile.dtype
    qv_profile = jnp.asarray(qv_profile, dtype=dtype)
    p_profile = jnp.asarray(p_profile, dtype=dtype)
    q_sat = saturation_mixing_ratio_blend(
        T_profile, p_profile, T_blend_width=float(blend_width_K))
    return qv_profile / q_sat


def log_humidity_profile(
    qv_profile: jax.Array,
    *,
    floor: float = DEFAULT_THERMO_LOGQ_FLOOR,
) -> jax.Array:
    """``ln(max(q_v, floor))`` — fractional humidity, with no saturation curve."""
    qv_profile = jnp.asarray(qv_profile)
    return jnp.log(jnp.maximum(qv_profile,
                               jnp.asarray(floor, dtype=qv_profile.dtype)))


#: Buffer kept BELOW the reference cold point when the mask is derived from it.
DEFAULT_THERMO_COLD_POINT_BUFFER_PA = 2_000.0


def reference_cold_point(
    T_ref: jax.Array,
    z_m: jax.Array,
) -> tuple[int, float, float]:
    """``(index, T [K], z [km])`` of the reference's cold point.

    Uses THIS MODULE'S EXISTING cold-point definition — the minimum of ``T``
    inside ``[TROP_MIN_Z_KM, TROP_MAX_Z_KM]`` — rather than a second one.  The
    height window is what makes it a tropopause search: an unbounded ``argmin``
    over everything above some pressure can return an upper-level inversion,
    a noise spike, or the model top on a profile that keeps cooling, and
    nothing downstream would notice.

    Raises if the window selects no level, so a reference on a different
    vertical extent fails loudly instead of yielding index 0.
    """
    T_ref = jnp.asarray(T_ref)
    z_km = jnp.asarray(z_m, dtype=T_ref.dtype) / 1000.0
    window = ((z_km >= TROP_MIN_Z_KM) & (z_km <= TROP_MAX_Z_KM))
    if int(jnp.sum(window)) == 0:
        raise ValueError(
            f"reference_cold_point: no level in [{TROP_MIN_Z_KM}, "
            f"{TROP_MAX_Z_KM}] km (z spans {float(jnp.min(z_km)):.2f}-"
            f"{float(jnp.max(z_km)):.2f} km).")
    idx = int(jnp.argmin(jnp.where(window, T_ref, jnp.inf)))
    return idx, float(T_ref[idx]), float(z_km[idx])


def tropospheric_min_pressure(
    T_ref: jax.Array,
    p_profile: jax.Array,
    z_m: jax.Array,
    *,
    floor_p_Pa: float = DEFAULT_THERMO_MIN_P_PA,
    buffer_Pa: float = DEFAULT_THERMO_COLD_POINT_BUFFER_PA,
) -> float:
    """Lowest pressure the thermodynamic objective scores, from the REFERENCE.

    A fixed 100 hPa bound is close to the RCEMIP-300 K cold point (~17 km,
    ~100 hPa) by coincidence, and the coincidence is the problem: if the cold
    point sits just BELOW the bound, the objective scores tropopause levels
    where ``dln q_sat/dT`` is ~0.15 /K and a 2 K radiatively-controlled
    temperature bias — which no convection parameter can fix — becomes a ~30 %
    humidity error carrying real weight.  Deriving the bound from the
    reference's own cold point, plus a buffer, removes the coincidence.

    It is computed from the REFERENCE ONLY, so every scheme is scored on the
    identical domain; a bound derived from each model's own cold point would
    score every scheme on a different column.
    """
    p_profile = jnp.asarray(p_profile)
    idx, _T_cold, _z_cold = reference_cold_point(T_ref, z_m)
    p_cold = float(p_profile[idx])
    return float(max(float(floor_p_Pa), p_cold + float(buffer_Pa)))


def tropospheric_mass_weights(
    p_profile: jax.Array,
    mass_weights: jax.Array,
    *,
    min_p_Pa: float = DEFAULT_THERMO_MIN_P_PA,
) -> jax.Array:
    """Mass weights restricted to ``p >= min_p_Pa`` and renormalized to sum to 1.

    Selecting no level is a hard error, never an all-zero weight vector that
    would silently score every column identically (same contract as
    :func:`subcloud_mass_weights`).
    """
    p_profile = jnp.asarray(p_profile)
    mass_weights = jnp.asarray(mass_weights, dtype=p_profile.dtype)
    if p_profile.shape != mass_weights.shape:
        raise ValueError(
            f"tropospheric_mass_weights: p_profile {p_profile.shape} and "
            f"mass_weights {mass_weights.shape} must share a shape.")
    mask = p_profile >= jnp.asarray(min_p_Pa, dtype=p_profile.dtype)
    n_selected = int(jnp.sum(mask))
    if n_selected == 0:
        raise ValueError(
            f"tropospheric_mass_weights: no level at or above {min_p_Pa} Pa "
            f"(highest pressure is {float(jnp.max(p_profile)):.1f} Pa).")
    selected = jnp.where(mask, mass_weights, jnp.zeros_like(mass_weights))
    return selected / jnp.sum(selected)


def thermo_terms_jax(
    ref: Any,
    T_profile: jax.Array,
    qv_profile: jax.Array,
    p_profile: jax.Array,
    *,
    trop_weights: jax.Array,
    T_scale_K: float = DEFAULT_THERMO_T_SCALE_K,
    rh_scale: float = DEFAULT_THERMO_RH_SCALE,
    logq_scale: float = DEFAULT_THERMO_LOGQ_SCALE,
    logq_floor: float = DEFAULT_THERMO_LOGQ_FLOOR,
    blend_width_K: float = DEFAULT_THERMO_RH_BLEND_WIDTH_K,
) -> dict[str, jax.Array]:
    """Every thermodynamic term, computed once: ``T``, ``logq`` and ``rh``.

    All three are ALWAYS returned, whichever one the objective minimises, so a
    campaign tuned on one can be read on the other without re-running it — and
    so a trade-off between them is visible rather than inferred.
    """
    T_profile = jnp.asarray(T_profile)
    dtype = T_profile.dtype
    qv_profile = jnp.asarray(qv_profile, dtype=dtype)
    p_profile = jnp.asarray(p_profile, dtype=dtype)
    weights = jnp.asarray(trop_weights, dtype=dtype)
    T_ref = jnp.asarray(ref.T_ref, dtype=dtype)
    qv_ref = jnp.asarray(ref.qv_ref, dtype=dtype)

    rh = relative_humidity_profile(
        T_profile, qv_profile, p_profile, blend_width_K=blend_width_K)
    rh_ref = relative_humidity_profile(
        T_ref, qv_ref, p_profile, blend_width_K=blend_width_K)
    lnq = log_humidity_profile(qv_profile, floor=logq_floor)
    lnq_ref = log_humidity_profile(qv_ref, floor=logq_floor)

    return {
        "T": weighted_rmse(
            (T_profile - T_ref) / jnp.asarray(T_scale_K, dtype=dtype), weights),
        "rh": weighted_rmse(
            (rh - rh_ref) / jnp.asarray(rh_scale, dtype=dtype), weights),
        "logq": weighted_rmse(
            (lnq - lnq_ref) / jnp.asarray(logq_scale, dtype=dtype), weights),
        # Physical-unit companions, on the SAME masked weights, so a caption can
        # quote a number a reader can interpret.
        "rh_rmse": weighted_rmse(rh - rh_ref, weights),
        "qv_rmse_kg_kg": weighted_rmse(qv_profile - qv_ref, weights),
    }


def score_thermo_jax(
    ref: Any,
    T_profile: jax.Array,
    qv_profile: jax.Array,
    p_profile: jax.Array,
    *,
    trop_weights: jax.Array,
    humidity: str = DEFAULT_THERMO_HUMIDITY,
    T_scale_K: float = DEFAULT_THERMO_T_SCALE_K,
    rh_scale: float = DEFAULT_THERMO_RH_SCALE,
    logq_scale: float = DEFAULT_THERMO_LOGQ_SCALE,
    logq_floor: float = DEFAULT_THERMO_LOGQ_FLOOR,
    blend_width_K: float = DEFAULT_THERMO_RH_BLEND_WIDTH_K,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(T_term, humidity_term, combined)`` — the temperature-and-humidity target.

    Two deliberate departures from :func:`score_profiles_jax`, each because the
    combined score cannot answer the question this objective is for:

    * **Condensate and precipitation are absent.**  MEASURED on the ten-scheme
      2026-08-13 arm, the condensate term carries 87-100 % of the combined
      score's sum of squares, so minimizing it is not minimizing T or humidity.
    * **Humidity is NOT scored as absolute ``q_v``.**  A mass-weighted absolute
      ``q_v`` RMSE is a boundary-layer metric: ``q_v`` falls ~3 decades between
      the surface and the upper troposphere and the mass weights add another
      low-level factor.  MEASURED on this reference under a uniform relative
      error, 1.72 % of its leverage lies above 5 km and 0.0006 % above 10 km,
      against a 56.6 % mass share — so the free-tropospheric humidity, the field
      convection schemes actually differ in, is invisible to it.

    ``humidity`` selects the variable that is MINIMISED; see
    :data:`THERMO_HUMIDITY_VARIABLES` for what each buys and costs.  A typo must
    select nothing, so this raises rather than defaulting.

    Both terms are divided by a FIXED physical tolerance and combined in
    quadrature, so the T-versus-humidity weight is a declared decision rather
    than whatever the reference profile's own spread happens to be.
    """
    if humidity not in THERMO_HUMIDITY_VARIABLES:
        raise ValueError(
            f"score_thermo_jax: unknown humidity variable {humidity!r}; "
            f"expected one of {THERMO_HUMIDITY_VARIABLES}")
    terms = thermo_terms_jax(
        ref, T_profile, qv_profile, p_profile, trop_weights=trop_weights,
        T_scale_K=T_scale_K, rh_scale=rh_scale, logq_scale=logq_scale,
        logq_floor=logq_floor, blend_width_K=blend_width_K)
    T_term = terms["T"]
    q_term = terms[humidity]
    return T_term, q_term, safe_sqrt((T_term ** 2 + q_term ** 2) / 2.0)


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
