"""LES-vs-SCM score assembly (D6 diagnostic + prognostic).

The two D6 scores turn an LES reference (:mod:`~...bridge`) and SCM output into a
single comparable number, reusing the mass-weighted, std-normalized RMSE
primitives in :mod:`legoesm.core.profile_metrics` — **no new numerics** (LES_SUITE
§5). This module never runs the SCM or the LES; it consumes arrays the driver
produces.

* **Diagnostic score** (:func:`diagnostic_flux_score`) — with the SCM mean state
  pinned to an LES snapshot, each closure returns its turbulent flux; the score is
  the normalized weighted RMSE of that flux against the LES total (resolved+SGS)
  flux. A pure test of the closure operator (no drift, no compounding).
* **Prognostic score** (:func:`prognostic_profile_score`) — the freely-integrated
  SCM state (θ_l, q_t, u, v over matched times) vs the LES mean profiles. Combined
  as an RMS across the enabled variables.

Normalization: each variable is divided by the mass-weighted std of the LES
*reference* profile (floored), so a variable with intrinsically small variability
does not dominate — identical to the SCM-RCE convention. Weights are the LES-grid
layer-thickness weights (:func:`legoesm.core.profile_metrics.layer_weights_from_heights`),
with zero weight above the case's catalog sponge base (:func:`case_score_weights`);
pass a density-scaled weight explicitly for a true mass weighting.

All functions are pure and AD-safe (finite gradient at a perfect fit) so the same
score drives both the derivative-free and the gradient tuners (D4).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.profile_metrics import (
    layer_weights_from_heights,
    safe_sqrt,
    weighted_rmse,
    weighted_std,
)

from .bridge import LESTruth
from .catalog import CATALOG_ALIASES, catalog_case

Array = jnp.ndarray

# Floor on the normalizing std so a near-constant reference profile cannot blow up
# the normalized RMSE (matches scm_rce_metrics.profile_floor usage). Applied in the
# variable's own units, so it is passed per-call from the score functions below.
PROFILE_FLOOR_THETA = 0.1        # K
PROFILE_FLOOR_QT = 1.0e-4        # kg/kg
PROFILE_FLOOR_WIND = 0.1         # m/s
PROFILE_FLOOR_FLUX = 1.0e-3      # K m/s (heat flux)
PROFILE_FLOOR_QFLUX = 1.0e-6     # (kg/kg) m/s


def _normalized_rmse(
    scm: Array, ref: Array, weights: Array, floor: float
) -> Array:
    """Mass-weighted RMSE of ``scm`` vs ``ref``, normalized by the ref std.

    ``scm``/``ref`` are ``(nz,)`` single-time profiles. The normalizing std comes
    from the *reference* profile (floored), so the score is dimensionless and
    comparable across variables.
    """
    scm = jnp.asarray(scm)
    ref = jnp.asarray(ref, dtype=scm.dtype)
    weights = jnp.asarray(weights, dtype=scm.dtype)
    std = jnp.maximum(weighted_std(ref, weights), jnp.asarray(floor, dtype=scm.dtype))
    return weighted_rmse((scm - ref) / std, weights)


def normalized_rmse_series(
    scm: Array, ref: Array, weights: Array, floor: float
) -> Array:
    """RMS-over-time of the per-snapshot normalized RMSE for a ``(nt, nz)`` series.

    The std is computed once from the *whole* reference series (a single scale per
    variable) so a variable that grows over the run is not re-normalized each step.
    """
    scm = jnp.asarray(scm)
    ref = jnp.asarray(ref, dtype=scm.dtype)
    weights = jnp.asarray(weights, dtype=scm.dtype)
    if scm.ndim == 1:
        return _normalized_rmse(scm, ref, weights, floor)
    # one scale for the whole series: std of the time-mean reference profile,
    # floored (a stable, single normalizer — avoids per-step rescaling).
    ref_mean = jnp.mean(ref, axis=0)
    std = jnp.maximum(weighted_std(ref_mean, weights), jnp.asarray(floor, dtype=scm.dtype))
    # per-snapshot mean-square (weights sum to 1 over levels), averaged over time,
    # then ONE safe_sqrt so the gradient stays finite at a perfect fit (a bare
    # intermediate sqrt of the per-time term would NaN the backward pass at 0).
    # a zero-weight level (above a sponge cutoff) contributes exactly zero even if
    # its residual is not finite, as in core.profile_metrics.weighted_rmse
    keep = weights[None, :] > 0
    scm = jnp.where(keep, scm, jnp.zeros_like(scm))
    ref = jnp.where(keep, ref, jnp.zeros_like(ref))
    per_time_ms = jnp.sum(weights[None, :] * ((scm - ref) / std) ** 2, axis=1)
    return safe_sqrt(jnp.mean(per_time_ms))


def _concrete_heights(case_name: str, heights_m) -> np.ndarray:
    if isinstance(heights_m, jax.core.Tracer):
        raise TypeError(f"{case_name}: the sponge cutoff needs concrete heights (run "
                        "metadata); close over the truth instead of tracing it")
    return np.asarray(heights_m, dtype=np.float64)


def sponge_base_for_score(case_name: str, heights_m: Array) -> float | None:
    """The catalog sponge base [m] of case ``case_name`` (or a driver's default label
    for it, :data:`~.catalog.CATALOG_ALIASES`) on a run whose cell centres are
    ``heights_m`` — the spectral LES grids are uniform, so the domain top is the top
    centre plus half its spacing — or ``None`` for a case without a sponge. A name
    that is not a catalog case raises (user 2026-10-19). ``heights_m`` must be
    concrete."""
    try:
        case = catalog_case(CATALOG_ALIASES.get(case_name, case_name))
    except KeyError:
        raise ValueError(f"{case_name!r} is not an LES catalog case (or a driver "
                         "label for one), so its sponge cutoff is unknown; pass "
                         "explicit weights for a non-catalog column") from None
    if case.setup is not None or case.sponge_base_frac is None:
        return case.sponge_base_m()          # absolute base, or no sponge
    z = _concrete_heights(case_name, heights_m)
    if z.shape[0] < 2:
        raise ValueError(f"{case_name}: the sponge cutoff needs at least two levels")
    dz = np.diff(z)
    # rounding of the stored centres, not of their spacing, sets the tolerance
    eps = np.finfo(np.asarray(heights_m).dtype).eps
    if np.max(np.abs(dz - np.mean(dz))) > 64.0 * eps * float(np.max(np.abs(z))):
        raise ValueError(f"{case_name}: the sponge cutoff infers the domain top "
                         "from a uniform grid; these heights are not uniform")
    return float(case.sponge_base_m(z[-1] + 0.5 * (z[-1] - z[-2])))


def case_score_weights(case_name: str, heights_m: Array) -> Array:
    """Layer weights for scoring case ``case_name``: the thickness weights with zero
    weight on every level above the case's catalog sponge base, renormalised to 1
    (user 2026-10-19: the damped layer is not LES truth). No cutoff for a case
    without a sponge; a name that is not a catalog case raises. The cut is decided
    in float64 on the concrete heights."""
    z_sp = sponge_base_for_score(case_name, heights_m)
    if z_sp is None:
        return layer_weights_from_heights(heights_m)
    return weights_below(case_name, heights_m, z_sp)


def weights_below(case_name: str, heights_m: Array, z_cut_m: float) -> Array:
    """Thickness weights with zero weight on every level above ``z_cut_m``,
    renormalised to 1; decided in float64 on the concrete heights."""
    w = layer_weights_from_heights(heights_m)
    keep = _concrete_heights(case_name, heights_m) <= z_cut_m
    if not keep.any():
        raise ValueError(f"{case_name}: no level lies below the cutoff {z_cut_m:g} m")
    w = jnp.where(jnp.asarray(keep), w, 0.0)
    return w / jnp.sum(w)


def _resolve_weights(
    heights_m: Array, weights: Array | None, case_name: str | None = None
) -> Array:
    """Layer weights for the score (default: :func:`case_score_weights` for
    ``case_name``, else midpoint-rule thickness).

    An explicit ``weights`` must match ``heights_m``, be non-negative, and sum to
    1 (the normalized-RMSE arithmetic in ``core.profile_metrics`` assumes a
    partition of unity). Negative weights are rejected outright — they can make the
    weighted mean-square negative, which ``safe_sqrt`` floors to 0 and would report
    a *perfect* score for a bad fit.
    """
    if weights is None:
        if case_name is None:
            return layer_weights_from_heights(heights_m)
        return case_score_weights(case_name, heights_m)
    w = jnp.asarray(weights)
    if w.shape != jnp.asarray(heights_m).shape:
        raise ValueError("weights must match heights_m shape")
    if bool(jnp.any(w < 0)):
        raise ValueError("weights must be non-negative")
    total = float(jnp.sum(w))
    if not (abs(total - 1.0) <= 1.0e-6):
        raise ValueError(
            f"weights must sum to 1 (a partition of unity), got sum={total}"
        )
    return w


@dataclass(frozen=True)
class DiagnosticScore:
    """Diagnostic (flux-operator) score for one LES snapshot.

    wtheta_rmse : normalized weighted RMSE of the heat flux [dimensionless].
    wqt_rmse : moisture-flux score, or ``None`` for a dry case.
    combined : RMS of the enabled flux scores.
    """

    case_name: str
    wtheta_rmse: Array
    wqt_rmse: Array | None
    combined: Array


def diagnostic_flux_score(
    truth: LESTruth,
    scm_wtheta: Array,
    scm_wqt: Array | None = None,
    *,
    weights: Array | None = None,
    theta_flux_floor: float = PROFILE_FLOOR_FLUX,
    qt_flux_floor: float = PROFILE_FLOOR_QFLUX,
) -> DiagnosticScore:
    """Score an SCM closure's flux against the LES total flux (D6 diagnostic).

    ``truth`` is a single-time :class:`LESTruth` (``bridge.diagnostic_truth``);
    ``scm_wtheta`` is the closure's heat flux on the same ``(nz,)`` grid. For a
    moist case pass ``scm_wqt`` too and a moist ``truth`` (``truth.wqt`` set).
    """
    theta = jnp.asarray(truth.theta)
    if theta.ndim != 1:
        raise ValueError("diagnostic_flux_score expects a single-time truth snapshot")
    w = _resolve_weights(truth.heights_m, weights, truth.case_name)
    wtheta_rmse = _normalized_rmse(scm_wtheta, truth.wtheta, w, theta_flux_floor)
    # A moist truth demands the SCM moisture flux — silently dropping it would let
    # an incomplete run out-score a complete one on the combined metric.
    if truth.wqt is not None and scm_wqt is None:
        raise ValueError(
            f"{truth.case_name}: moist truth (wqt set) requires scm_wqt; "
            "omitting it would silently reduce this to a dry score"
        )
    wqt_rmse = None
    if truth.wqt is not None:
        wqt_rmse = _normalized_rmse(scm_wqt, truth.wqt, w, qt_flux_floor)
    if wqt_rmse is None:
        combined = wtheta_rmse
    else:
        combined = safe_sqrt(0.5 * (wtheta_rmse ** 2 + wqt_rmse ** 2))
    return DiagnosticScore(
        case_name=truth.case_name,
        wtheta_rmse=wtheta_rmse,
        wqt_rmse=wqt_rmse,
        combined=combined,
    )


@dataclass(frozen=True)
class PrognosticScore:
    """Prognostic (free-running) profile score vs the LES mean profiles.

    Per-variable normalized weighted RMSEs (dimensionless) + their RMS combination.
    ``qt_rmse`` is ``None`` for a dry case.
    """

    case_name: str
    theta_rmse: Array
    u_rmse: Array
    v_rmse: Array
    qt_rmse: Array | None
    combined: Array


def prognostic_profile_score(
    truth: LESTruth,
    scm_theta: Array,
    scm_u: Array,
    scm_v: Array,
    scm_qt: Array | None = None,
    *,
    weights: Array | None = None,
    theta_floor: float = PROFILE_FLOOR_THETA,
    qt_floor: float = PROFILE_FLOOR_QT,
    wind_floor: float = PROFILE_FLOOR_WIND,
) -> PrognosticScore:
    """Score freely-integrated SCM profiles against LES truth (D6 prognostic).

    ``truth`` is a (possibly multi-time) :class:`LESTruth`
    (``bridge.prognostic_truth``); the ``scm_*`` arrays must match its θ shape
    (``(nz,)`` single time or ``(nt, nz)`` series). Combined = RMS over the enabled
    variables (θ_l, u, v, and q_t when moist).
    """
    w = _resolve_weights(truth.heights_m, weights, truth.case_name)
    theta_rmse = normalized_rmse_series(scm_theta, truth.theta, w, theta_floor)
    u_rmse = normalized_rmse_series(scm_u, truth.u, w, wind_floor)
    v_rmse = normalized_rmse_series(scm_v, truth.v, w, wind_floor)
    if truth.qt is not None and scm_qt is None:
        raise ValueError(
            f"{truth.case_name}: moist truth (qt set) requires scm_qt; "
            "omitting it would silently reduce this to a dry score"
        )
    qt_rmse = None
    if truth.qt is not None:
        qt_rmse = normalized_rmse_series(scm_qt, truth.qt, w, qt_floor)
    terms = [theta_rmse, u_rmse, v_rmse]
    if qt_rmse is not None:
        terms.append(qt_rmse)
    stacked = jnp.stack(terms)
    combined = safe_sqrt(jnp.mean(stacked ** 2))
    return PrognosticScore(
        case_name=truth.case_name,
        theta_rmse=theta_rmse,
        u_rmse=u_rmse,
        v_rmse=v_rmse,
        qt_rmse=qt_rmse,
        combined=combined,
    )
