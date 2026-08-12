"""Training loss functions for differentiable dycore rollouts.

Builds on ``ml.loss`` (area_weighted_mse, spectral_loss) with
dycore-specific additions: per-level pressure weighting and
SegmentCarry comparison.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.grids.gaussian import sh_analysis_3d
from legoesm.ml.loss import spectral_loss


class LossConfig(NamedTuple):
    """Configuration for dycore training loss.

    Per-variable normalization
    --------------------------
    Set ``normalize_by_scale=True`` to divide each variable's MSE
    contribution by its typical amplitude squared
    (``T_scale²``, ``q_scale²``, ``ps_scale²``, ``wind_scale²``).
    Without this, the raw-units MSE has variable contributions that
    differ by ~9 orders of magnitude (ps² ~ 1e10 Pa², q² ~ 1e-4
    kg²/kg², T² ~ 1e2 K², wind² ~ 1e2 m²/s²) so the moisture and
    wind branches receive negligible gradient compared to ps.
    Default: enabled, with ESM-typical anomaly scales.
    """
    # Variable weights (relative importance, applied AFTER per-variable
    # scale normalization when ``normalize_by_scale=True``)
    w_T: float = 1.0          # temperature
    w_u: float = 0.5          # zonal wind
    w_v: float = 0.5          # meridional wind
    w_q: float = 0.2          # specific humidity
    w_ps: float = 0.3         # surface pressure
    # Bias-penalty weights.  ``w_bias_<var>`` multiplies the squared
    # global-mean error ``(area_weighted_mean(pred - target))^2``,
    # normalized by the same variable scale as the MSE term.  Adds
    # an explicit penalty on constant offsets that the per-cell MSE
    # only weakly constrains -- e.g. the +1.07 K warm T bias the
    # AIMIP v5 classical variant produced at T11 L8 (residual after
    # the spatial-surface fields over-corrected).  Default 0.0
    # preserves legacy behaviour; set ``w_bias_T`` ~ 1-10 in AIMIP
    # configs to drive the bias toward zero.
    w_bias_T: float = 0.0
    w_bias_u: float = 0.0
    w_bias_v: float = 0.0
    w_bias_ps: float = 0.0
    # CRPS-style probabilistic loss weights.  For a deterministic
    # forecast (ensemble size M=1) the fair-CRPS reduces to the area-
    # weighted MAE of the per-cell error, normalised by the same
    # variable scale as the MSE term.  This is what NeuralGCM uses
    # alongside MSE for its probabilistic head; even at M=1 the MAE
    # term puts a different penalty profile on large errors than
    # squared error (less skewed toward outliers).  Default 0.0
    # preserves legacy MSE-only behaviour.  Ensemble (M>=2) fair-CRPS
    # is a separate code path tracked in the ``ensemble_size`` knob
    # below; the deterministic path emits the MAE limit.
    #
    # (That knob did NOT exist when this comment was written — it is defined
    # in the ensemble block further down now, 2026-07-30.)
    w_crps_T: float = 0.0
    w_crps_u: float = 0.0
    w_crps_v: float = 0.0
    w_crps_ps: float = 0.0
    # Specific humidity carries the heaviest tail of any prognostic
    # (convective + microphysical noise concentrates the error in
    # extreme columns).  An MAE / M=1-CRPS term down-weights those
    # outliers relative to a pure MSE objective and stabilises the
    # column-level error distribution; recommended to set this
    # higher than the corresponding MSE weight ``w_q`` so the CRPS
    # contribution dominates for q.
    w_crps_q: float = 0.0
    # Spectral-space CRPS (deterministic M=1 limit = MAE of the
    # spectral-coefficient error).  Distinct from the grid-space
    # ``w_crps_*`` family: takes ``sh_analysis_3d(pred - target)`` and
    # averages ``|.|`` over all (l, m, level) coefficients, normalised
    # by the variable scale (NOT scale^2, since |error| is in raw
    # units).  Penalises errors in the large-scale spectral pattern in
    # commensurate units with the grid losses.  Only the spectral-PE
    # path implements this (``spectral_state_vs_carry_loss``); the
    # grid-only ``carry_mse`` path silently ignores these weights.
    # Requires JAX_ENABLE_X64=True (SH transforms need float64).
    w_spec_crps_T: float = 0.0
    w_spec_crps_u: float = 0.0
    w_spec_crps_v: float = 0.0
    w_spec_crps_q: float = 0.0
    # Forecast horizons (in hours since IC) at which the loss is
    # evaluated when multi-step supervision is on.  Empty tuple
    # disables it (legacy single-target behaviour).  Set e.g.
    # ``multi_step_hours=(6, 12, 18, 24)`` to penalise the model at
    # 6 / 12 / 18 / 24 hour leads.  Each lead requires one extra
    # spectral_rollout call from the same IC, so cost scales with
    # ``len(multi_step_hours)``.  Targets are loaded once at IC time
    # and indexed by lead.
    multi_step_hours: tuple[int, ...] = ()
    # Optional per-lead weight; defaults to uniform.  Length must
    # match ``multi_step_hours`` if provided, else ignored.
    multi_step_weights: tuple[float, ...] = ()
    # Loss components
    spectral_weight: float = 0.0   # weight for spectral loss term
    level_weighting: str = "pressure"  # uniform|pressure|boundary_layer|density
    # Pressure weighting parameters
    p_ref_Pa: float = 50000.0      # reference pressure for level weighting [Pa]
    p_scale_Pa: float = 30000.0    # width of weighting function [Pa]
    # Per-variable amplitude scales used for normalization
    # (iter-69 fix; appended to keep positional construction
    # backward-compatible — older callers using
    # ``LossConfig(..., spectral_weight=...)`` keep working).
    normalize_by_scale: bool = True
    T_scale: float = 30.0          # K — typical mid-tropospheric T anomaly
    wind_scale: float = 20.0       # m/s — typical wind anomaly
    q_scale: float = 5.0e-3        # kg/kg — typical q anomaly
    ps_scale: float = 1000.0       # Pa — typical ps anomaly
    # ACE2-style RESIDUAL normalization: when True, normalize each MSE term by
    # the std of the 6-hour FIELD CHANGE (the prognostic "residual scale" of
    # Watt-Meyer et al. 2024 / ACE2) instead of the full-field std above. The
    # loss then weights the predictable TENDENCY rather than the (large) mean
    # state, the key conditioning trick behind ACE2's short-range skill. The
    # defaults are ERA5 6-hourly global tendency stds (approximate ACE's
    # data-computed scaling-residual; override per-run if needed).
    # --- ensemble / almost-fair CRPS (probabilistic, for long leads) ---------
    # Beyond ~5 days a deterministic MSE forecast is optimally a smoothed,
    # climatology-tending field (the conditional mean), so RMSE stops rewarding
    # realistic variability. A proper CRPS needs an ENSEMBLE; the M=1 limit of
    # fair-CRPS is just the MAE, which is what the ``w_crps_*`` terms above
    # emit. ``ensemble_size > 1`` switches on the real thing via
    # ``legoesm.ml.loss.area_weighted_afcrps`` (the almost-fair finite-ensemble
    # estimator, AIFS-CRPS style).
    #
    # Members are generated by perturbing the INITIAL STATE — the cheapest
    # scheme that needs no architecture change, and the standard NWP construct.
    # Cost is linear in ensemble_size: M members = M rollouts.
    ensemble_size: int = 1
    # IC perturbation amplitude, expressed as a MULTIPLE of the per-variable
    # 6 h residual scales (``*_resid_scale``) so it is dimensionless and
    # correctly scaled for T / wind / q / ps at once. 0.0 -> identical members,
    # which would make CRPS collapse to the MAE, so it must be > 0 whenever
    # ensemble_size > 1.
    ensemble_ic_noise: float = 0.0
    # Weight on the almost-fair CRPS term. 0.0 -> off (legacy MSE-only).
    w_afcrps: float = 0.0
    # Finite-ensemble adjustment; 1.0 is fair-CRPS, <1 is numerically steadier
    # for small M (see almost_fair_crps).
    afcrps_alpha: float = 0.95

    # --- persistence-normalized (skill-score) loss ---------------------------
    # When True the per-sample loss is divided by the SAME loss evaluated with
    # the INITIAL state as the forecast, i.e. L = L(pred, truth) / L(x0, truth).
    # Plain MSE is translation-invariant, so merely subtracting persistence
    # changes nothing (MSE(pred, truth) == MSE(pred - x0, truth - x0)); what
    # bites is DIVIDING by the persistence error, which
    #   * makes every lead contribute equally in SKILL terms rather than in
    #     magnitude, so a 10-day rollout cannot dominate a 6-hour one purely
    #     because its errors are bigger, and
    #   * removes any reward for reproducing the trivially-predictable part of
    #     the field (a model that just copies x0 scores 1.0, up to the floor
    #     below — exactly 1.0 only in the limit persistence_floor -> 0).
    # The denominator depends only on DATA (x0 and the target), never on the
    # model, so it is a constant per sample; it is still wrapped in
    # ``stop_gradient`` to document that and to stay safe if that ever changes.
    persistence_normalize: bool = False
    # Floor on the denominator. Guards a degenerate sample whose persistence
    # error is ~0 (target == x0), which would otherwise divide by zero.
    persistence_floor: float = 1.0e-12
    residual_normalize: bool = False
    T_resid_scale: float = 1.5     # K — ERA5 6 h |ΔT| std
    wind_resid_scale: float = 3.0  # m/s — ERA5 6 h |Δu|,|Δv| std
    q_resid_scale: float = 5.0e-4  # kg/kg — ERA5 6 h |Δq| std
    ps_resid_scale: float = 200.0  # Pa — ERA5 6 h |Δps| std
    # Radiation-flux supervision (TOA + surface) — drives cross-climate
    # generalization by constraining the radiative response, not just the
    # instantaneous state.  Reads the SegmentCarry ``held_*`` flux fields
    # (rsut = held_sw_up_toa, OLR = held_lw_up_toa, surface net SW/LW =
    # held_sw_net_sfc / held_lw_net_sfc) against ERA5-loaded targets on the
    # target carry.  ``w_flux_*`` are MSE weights; ``w_bias_flux_*`` add a
    # squared area-mean (global-energy-balance) penalty so the model's mean
    # OLR / reflected-SW / surface fluxes match ERA5 — the term that
    # actually transfers across climate states.  All default 0.0 → no-op
    # (byte-identical legacy behaviour); the term is only active when a
    # weight is >0 AND the target carry's ``held_*`` fields hold real ERA5
    # fluxes (``TrainingERA5Config.load_radiation_fluxes=True``).  rsdt
    # (held_sw_down_toa) is prescribed insolation, not predicted, so it is
    # NOT penalized; constraining rsut+OLR pins the TOA net radiation.
    w_flux_rsut: float = 0.0        # TOA reflected SW (held_sw_up_toa)
    w_flux_olr: float = 0.0         # TOA outgoing LW / OLR (held_lw_up_toa)
    w_flux_sfc_sw: float = 0.0      # surface net SW (held_sw_net_sfc)
    w_flux_sfc_lw: float = 0.0      # surface net LW (held_lw_net_sfc)
    w_bias_flux_rsut: float = 0.0
    w_bias_flux_olr: float = 0.0
    w_bias_flux_sfc_sw: float = 0.0
    w_bias_flux_sfc_lw: float = 0.0
    flux_scale: float = 20.0        # W/m² — typical radiative-flux anomaly scale


def level_weights(
    sigma_full: jax.Array,
    p_s_mean: float = 101325.0,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Compute per-level weights emphasizing mid-troposphere.

    Parameters
    ----------
    sigma_full : (nlev,) — model sigma levels
    p_s_mean : float — approximate mean surface pressure [Pa]
    config : LossConfig

    Returns
    -------
    (nlev,) — normalized weights summing to nlev
    """
    if config.level_weighting == "uniform":
        return jnp.ones_like(sigma_full)

    p_levels = sigma_full * p_s_mean  # approximate pressure at each level

    if config.level_weighting == "pressure":
        # Gaussian-like weighting centered at p_ref, width p_scale
        w = jnp.exp(-0.5 * ((p_levels - config.p_ref_Pa) / config.p_scale_Pa) ** 2)
        # Add a floor so stratosphere isn't zero
        w = w + 0.1
    elif config.level_weighting == "boundary_layer":
        # Emphasize lower troposphere (p > 700 hPa)
        w = jnp.where(p_levels > 70000.0, 2.0, 1.0)
    elif config.level_weighting == "density":
        # GraphCast / ArchesWeather upper-air weighting: "a coefficient
        # proportional to air density for upper-air variables" (Lam et al. 2023;
        # ArchesWeather arXiv:2412.12971 A.1). With rho = p / (R_d T) and T
        # varying by <~30% across the column while p varies by ~50x, rho is
        # proportional to p to leading order, so the weight is the level
        # pressure itself (normalized below). MONOTONIC toward the surface —
        # unlike "pressure", which is a Gaussian bump centred on p_ref and so
        # downweights BOTH the surface and the top.
        w = p_levels
    else:
        raise ValueError(
            f"Unknown level_weighting {config.level_weighting!r}; choose one of "
            "'uniform', 'pressure', 'boundary_layer', 'density'. (A silent "
            "uniform fallback here masked typos and ran a different objective "
            "than the config asked for.)")

    # Normalize so mean weight = 1
    return w * (sigma_full.shape[0] / jnp.sum(w))


def _lat_weighted_mean(
    field: jax.Array, lat_weights: jax.Array | None
) -> jax.Array:
    """Mean over all dims, with optional latitude weighting.

    When ``lat_weights`` is provided and ``field`` has *exactly one* axis
    of length ``n_lat = len(lat_weights)``, the mean is replaced by the
    area-weighted mean ``mean(field · lat_w) · n_lat / Σ(lat_w)``
    (resolution-independent, identical correction as iter-63
    ``ml/loss.py``).

    Disambiguation rules (size-matching is intentionally strict because
    the bias term is silent-failure-prone otherwise):
    - 0 axes match n_lat → uniform mean (e.g. cubed-sphere where the
      leading face dim differs from n_lat).
    - 1 axis matches → use it.
    - >1 axes match → raise.  Callers seeing this should pass their field
      with an unambiguous lat axis (flatten or rename the colliding axis).

    Lifted to module scope (iter-AIMIP-flux) so ``carry_mse`` and
    ``radiation_flux_loss`` share one implementation — no duplicated
    area-weighting numerics across the state and radiation-flux losses.
    """
    if lat_weights is None:
        return jnp.mean(field)
    n_lat_w = lat_weights.shape[0]
    matching_axes = [a for a, d in enumerate(field.shape) if d == n_lat_w]
    if not matching_axes:
        return jnp.mean(field)
    # >1 axis of length n_lat (e.g. a carry field (n_lat, n_lon, nlev)
    # with nlev == n_lat): the lat axis is axis 0 by convention for every
    # grid that supplies lat_weights (lat-lon / Gaussian put latitude
    # first; cubed-sphere passes lat_weights=None).  Use the FIRST match
    # (axis 0) rather than raising — adding LatLonGrid.weights newly made
    # the loss area-weighted, so a raise would crash otherwise-valid
    # n_lev==n_lat configs (codex review).
    axis = matching_axes[0]
    shape = [1] * field.ndim
    shape[axis] = n_lat_w
    w = lat_weights.reshape(shape)
    return jnp.mean(field * w) * n_lat_w / jnp.sum(lat_weights)


# Public alias: the AIMIP lat-lon eval reuses this area-weighting helper
# for held-out RMSE/bias metrics rather than re-deriving the numerics.
lat_weighted_mean = _lat_weighted_mean


def radiation_flux_loss(
    pred_carry,
    target_carry,
    lat_weights: jax.Array | None = None,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Area-weighted MSE + bias penalty on TOA + surface radiation fluxes.

    Reads the ``SegmentCarry`` held-radiation fields directly:
    ``held_sw_up_toa`` (rsut, TOA reflected SW), ``held_lw_up_toa`` (OLR /
    rlut), ``held_sw_net_sfc`` (surface net SW), ``held_lw_net_sfc``
    (surface net LW).  Each contributes a scale-normalized area-weighted
    MSE (weight ``w_flux_*``) and, when ``w_bias_flux_* > 0``, a squared
    area-mean (global-energy-balance) penalty.  Normalized by
    ``config.flux_scale²`` (MSE) / ``flux_scale`` (bias) so the terms are
    commensurate with the scale-normalized state losses.

    Active only when a flux weight is >0; otherwise returns exactly 0.0
    (the target carry's ``held_*`` are zeros unless ERA5 fluxes were
    loaded).  rsdt (``held_sw_down_toa``) is prescribed insolation, not a
    model prediction, so it is intentionally NOT penalized — constraining
    rsut + OLR already pins the net TOA radiation.

    Grid-agnostic: works on whatever 2D layout the carry uses (Gaussian /
    lat-lon ``(n_lat, n_lon)`` or cubed-sphere ``(6, n, n)`` / flat
    ``(ncol,)``) via the shared :func:`_lat_weighted_mean`.
    """
    from legoesm.core.precision import resolve_dtype
    loss = jnp.array(0.0, dtype=resolve_dtype(None, "accumulate"))

    if config.normalize_by_scale:
        f_mse_norm = config.flux_scale ** 2
        f_bias_norm = config.flux_scale ** 2
    else:
        f_mse_norm = f_bias_norm = 1.0

    # (carry field, MSE weight, bias weight)
    terms = (
        (pred_carry.held_sw_up_toa, target_carry.held_sw_up_toa,
         config.w_flux_rsut, config.w_bias_flux_rsut),
        (pred_carry.held_lw_up_toa, target_carry.held_lw_up_toa,
         config.w_flux_olr, config.w_bias_flux_olr),
        (pred_carry.held_sw_net_sfc, target_carry.held_sw_net_sfc,
         config.w_flux_sfc_sw, config.w_bias_flux_sfc_sw),
        (pred_carry.held_lw_net_sfc, target_carry.held_lw_net_sfc,
         config.w_flux_sfc_lw, config.w_bias_flux_sfc_lw),
    )
    for pred_f, tgt_f, w_mse, w_bias in terms:
        d = pred_f - tgt_f
        if w_mse > 0.0:
            loss = loss + w_mse * _lat_weighted_mean(d ** 2, lat_weights) / f_mse_norm
        if w_bias > 0.0:
            bias = _lat_weighted_mean(d, lat_weights)
            loss = loss + w_bias * bias ** 2 / f_bias_norm

    return loss


def _any_flux_weight(config: LossConfig) -> bool:
    """True when any radiation-flux loss weight is active (>0)."""
    return any(
        w > 0.0 for w in (
            config.w_flux_rsut, config.w_flux_olr,
            config.w_flux_sfc_sw, config.w_flux_sfc_lw,
            config.w_bias_flux_rsut, config.w_bias_flux_olr,
            config.w_bias_flux_sfc_sw, config.w_bias_flux_sfc_lw,
        )
    )


def carry_mse(
    pred_carry,
    target_carry,
    sigma_full: jax.Array,
    lat_weights: jax.Array | None = None,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Compute weighted MSE between two SegmentCarry states.

    Works directly on SegmentCarry fields (raw arrays on the model grid),
    avoiding regridding in the loss computation.

    Parameters
    ----------
    pred_carry, target_carry : SegmentCarry
        Predicted and target states on the model grid.
    sigma_full : (nlev,) — sigma levels for pressure weighting
    lat_weights : (n_lat,) or None — latitude weights for area weighting.
        If None, uniform weighting is used (appropriate for cubed-sphere
        where cells have approximately equal area).
    config : LossConfig

    Returns
    -------
    scalar — weighted MSE loss
    """
    lev_w = level_weights(sigma_full, config=config)

    from legoesm.core.precision import resolve_dtype
    loss = jnp.array(0.0, dtype=resolve_dtype(None, "accumulate"))

    # Area weighting uses the shared module-level ``_lat_weighted_mean``
    # (also used by ``radiation_flux_loss``) — one implementation, no
    # duplicated numerics.

    # Per-variable scale denominators.  When normalize_by_scale=True
    # each variable's MSE is divided by its typical amplitude² so the
    # different variables contribute in commensurate units.  Without
    # this, w_T·<dT²>, w_q·<dq²>, w_ps·<dps²> differ by ~9 orders of
    # magnitude (ps² ~ 1e10 dominates; q² ~ 1e-4 is invisible).  Set
    # to all-1 when normalization is off for backward compatibility.
    if getattr(config, "residual_normalize", False):
        # ACE2-style residual normalization (scale by std of the 6h field
        # change — see LossConfig). Matches _spectral_state_loss_components.
        # floor guards a misconfigured 0 residual scale (codex: no zero-div).
        T_norm = max(config.T_resid_scale ** 2, 1e-30)
        wind_norm = max(config.wind_resid_scale ** 2, 1e-30)
        q_norm = max(config.q_resid_scale ** 2, 1e-30)
        ps_norm = max(config.ps_resid_scale ** 2, 1e-30)
    elif config.normalize_by_scale:
        T_norm = config.T_scale ** 2
        wind_norm = config.wind_scale ** 2
        q_norm = config.q_scale ** 2
        ps_norm = config.ps_scale ** 2
    else:
        T_norm = wind_norm = q_norm = ps_norm = 1.0

    # Temperature: (..., nlev)
    dT = pred_carry.T - target_carry.T
    loss = loss + config.w_T * _lat_weighted_mean(dT ** 2 * lev_w, lat_weights) / T_norm

    # Winds: (..., nlev)
    du = pred_carry.u - target_carry.u
    dv = pred_carry.v - target_carry.v
    loss = loss + config.w_u * _lat_weighted_mean(du ** 2 * lev_w, lat_weights) / wind_norm
    loss = loss + config.w_v * _lat_weighted_mean(dv ** 2 * lev_w, lat_weights) / wind_norm

    # Moisture: (..., nlev)
    dq = pred_carry.q_v - target_carry.q_v
    loss = loss + config.w_q * _lat_weighted_mean(dq ** 2 * lev_w, lat_weights) / q_norm

    # Surface pressure: (...)
    dp = pred_carry.p_s - target_carry.p_s
    loss = loss + config.w_ps * _lat_weighted_mean(dp ** 2, lat_weights) / ps_norm

    # Bias-penalty terms.  Squared (level-weighted) area-mean error
    # for each variable, normalised by the same scale as the MSE
    # term.  ``_lat_weighted_mean`` returns the area-weighted mean
    # over all axes when ``lat_weights`` is provided, or the uniform
    # mean otherwise (acceptable for cubed-sphere where cells have
    # roughly equal area).  Mirrors the spectral-path bias term in
    # ``neural_gcm_spectral.spectral_state_vs_carry_loss`` so the
    # same ``w_bias_*`` weights have the same physical effect in
    # both training paths.
    if config.w_bias_T > 0.0:
        bias_T = _lat_weighted_mean(dT * lev_w, lat_weights)
        loss = loss + config.w_bias_T * bias_T ** 2 / T_norm
    if config.w_bias_u > 0.0:
        bias_u = _lat_weighted_mean(du * lev_w, lat_weights)
        loss = loss + config.w_bias_u * bias_u ** 2 / wind_norm
    if config.w_bias_v > 0.0:
        bias_v = _lat_weighted_mean(dv * lev_w, lat_weights)
        loss = loss + config.w_bias_v * bias_v ** 2 / wind_norm
    if config.w_bias_ps > 0.0:
        bias_ps = _lat_weighted_mean(dp, lat_weights)
        loss = loss + config.w_bias_ps * bias_ps ** 2 / ps_norm

    # Radiation-flux supervision (TOA + surface) — shared helper, only
    # active when a ``w_flux_*`` / ``w_bias_flux_*`` weight is >0.
    if _any_flux_weight(config):
        loss = loss + radiation_flux_loss(
            pred_carry, target_carry, lat_weights=lat_weights, config=config,
        )

    return loss


def carry_spectral_loss(
    pred_carry,
    target_carry,
    grid,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Spectral loss on temperature field (for spectral grids).

    Penalizes errors in the large-scale spectral pattern of T.
    Only meaningful for spectral (Gaussian) grids where SH analysis
    is available.

    Parameters
    ----------
    pred_carry, target_carry : SegmentCarry
    grid : GaussianGrid with sh_analysis_3d
    config : LossConfig

    Returns
    -------
    scalar — spectral L2 loss on T
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "carry_spectral_loss requires JAX_ENABLE_X64=True for "
            "float64 spectral transforms."
        )

    # Reshape for SH analysis: (..., nlev) → (n_lat, n_lon, nlev)
    pred_T = pred_carry.T.astype(jnp.float64)
    target_T = target_carry.T.astype(jnp.float64)

    pred_hat = sh_analysis_3d(grid, pred_T)
    target_hat = sh_analysis_3d(grid, target_T)

    spec_loss = spectral_loss(pred_hat, target_hat)
    # Normalize by T_scale² so spectral loss is in commensurate
    # units with the iter-69-normalized ``carry_mse``.  Without
    # this, ``spectral_weight`` user-tunings would have to
    # silently absorb a factor of T_scale² ≈ 900 K² to balance
    # against carry_mse — a calibration trap.  Iter-72 fix.
    if config.normalize_by_scale:
        spec_loss = spec_loss / (config.T_scale ** 2)
    return spec_loss


def combined_loss(
    pred_carry,
    target_carry,
    sigma_full: jax.Array,
    grid=None,
    lat_weights: jax.Array | None = None,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Combined loss: weighted MSE + optional spectral penalty.

    Parameters
    ----------
    pred_carry, target_carry : SegmentCarry
    sigma_full : (nlev,)
    grid : GaussianGrid or None (spectral loss requires grid)
    lat_weights : (n_lat,) or None
        Latitude weights forwarded to ``carry_mse``.  When ``grid`` is
        a ``GaussianGrid``, falls back to ``grid.weights`` if not
        provided — that's the right area weighting for the bias term
        when ``config.w_bias_* > 0``.  Pass explicit weights for
        cubed-sphere / lat-lon grids; ``None`` is OK on cubed-sphere
        where cells have approximately equal area.
    config : LossConfig

    Returns
    -------
    scalar — total loss
    """
    if lat_weights is None and grid is not None:
        lat_weights = getattr(grid, "weights", None)

    loss = carry_mse(
        pred_carry, target_carry, sigma_full,
        lat_weights=lat_weights, config=config,
    )

    if config.spectral_weight > 0.0 and grid is not None:
        loss = loss + config.spectral_weight * carry_spectral_loss(
            pred_carry, target_carry, grid, config=config,
        )

    return loss
