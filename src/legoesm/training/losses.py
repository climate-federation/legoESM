"""Training loss functions for differentiable dycore rollouts.

Builds on ``ml.loss`` (area_weighted_mse, spectral_loss) with
dycore-specific additions: per-level pressure weighting, multi-day
rollout loss, and SegmentCarry comparison.
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
    w_crps_T: float = 0.0
    w_crps_u: float = 0.0
    w_crps_v: float = 0.0
    w_crps_ps: float = 0.0
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
    level_weighting: str = "pressure"  # "uniform", "pressure", or "boundary_layer"
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
    else:
        w = jnp.ones_like(sigma_full)

    # Normalize so mean weight = 1
    return w * (sigma_full.shape[0] / jnp.sum(w))


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

    from legoesm.core.precision import _resolve_dtype
    loss = jnp.array(0.0, dtype=_resolve_dtype(None, "accumulate"))

    def _lat_weighted_mean(sq_err: jax.Array) -> jax.Array:
        """Mean over all dims, with optional latitude weighting.

        When ``lat_weights`` is provided and ``sq_err`` has an axis
        of length ``n_lat = len(lat_weights)``, the mean is
        replaced by the area-weighted mean
        ``mean(sq · lat_w) · n_lat / Σ(lat_w)`` (resolution-
        independent, identical correction as iter-63 ml/loss.py).
        Without lat_weights or on non-Gaussian shapes (e.g.
        cubed-sphere with leading face dim) this falls back to
        a uniform mean.
        """
        if lat_weights is None:
            return jnp.mean(sq_err)
        n_lat_w = lat_weights.shape[0]
        # Apply weight on the FIRST axis of length n_lat.
        for axis, dim in enumerate(sq_err.shape):
            if dim == n_lat_w:
                shape = [1] * sq_err.ndim
                shape[axis] = n_lat_w
                w = lat_weights.reshape(shape)
                return jnp.mean(sq_err * w) * n_lat_w / jnp.sum(lat_weights)
        # No matching axis — fall back to uniform mean.
        return jnp.mean(sq_err)

    # Per-variable scale denominators.  When normalize_by_scale=True
    # each variable's MSE is divided by its typical amplitude² so the
    # different variables contribute in commensurate units.  Without
    # this, w_T·<dT²>, w_q·<dq²>, w_ps·<dps²> differ by ~9 orders of
    # magnitude (ps² ~ 1e10 dominates; q² ~ 1e-4 is invisible).  Set
    # to all-1 when normalization is off for backward compatibility.
    if config.normalize_by_scale:
        T_norm = config.T_scale ** 2
        wind_norm = config.wind_scale ** 2
        q_norm = config.q_scale ** 2
        ps_norm = config.ps_scale ** 2
    else:
        T_norm = wind_norm = q_norm = ps_norm = 1.0

    # Temperature: (..., nlev)
    dT = pred_carry.T - target_carry.T
    loss = loss + config.w_T * _lat_weighted_mean(dT ** 2 * lev_w) / T_norm

    # Winds: (..., nlev)
    du = pred_carry.u - target_carry.u
    dv = pred_carry.v - target_carry.v
    loss = loss + config.w_u * _lat_weighted_mean(du ** 2 * lev_w) / wind_norm
    loss = loss + config.w_v * _lat_weighted_mean(dv ** 2 * lev_w) / wind_norm

    # Moisture: (..., nlev)
    dq = pred_carry.q_v - target_carry.q_v
    loss = loss + config.w_q * _lat_weighted_mean(dq ** 2 * lev_w) / q_norm

    # Surface pressure: (...)
    dp = pred_carry.p_s - target_carry.p_s
    loss = loss + config.w_ps * _lat_weighted_mean(dp ** 2) / ps_norm

    return loss


def multi_day_loss(
    pred_carries,
    target_carries,
    sigma_full: jax.Array,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Multi-day rollout loss: average carry_mse over multiple lead times.

    Parameters
    ----------
    pred_carries : pytree with leading (n_days,) dimension
        Predicted states at each day from differentiable_rollout.
    target_carries : pytree with leading (n_days,) dimension
        Target states from ERA5 at corresponding days.
    sigma_full : (nlev,)
    config : LossConfig

    Returns
    -------
    scalar — averaged loss across all lead times
    """
    # Use ``jax.vmap`` directly over the leading day axis of the carry
    # pytrees instead of a closure-over-arange + ``tree.map(lambda x: x[i])``.
    # The closure pattern forced JAX to retrace every call (the closure
    # captured ``pred_carries`` / ``target_carries`` by identity); vmap
    # with ``in_axes=0`` lets the batching machinery slice the leading
    # axis without any Python tree walk inside the inner loop.
    def _day_loss(pred_i, target_i):
        return carry_mse(pred_i, target_i, sigma_full, config=config)

    day_losses = jax.vmap(_day_loss)(pred_carries, target_carries)
    return jnp.mean(day_losses)


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
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Combined loss: weighted MSE + optional spectral penalty.

    Parameters
    ----------
    pred_carry, target_carry : SegmentCarry
    sigma_full : (nlev,)
    grid : GaussianGrid or None (spectral loss requires grid)
    config : LossConfig

    Returns
    -------
    scalar — total loss
    """
    loss = carry_mse(pred_carry, target_carry, sigma_full, config=config)

    if config.spectral_weight > 0.0 and grid is not None:
        loss = loss + config.spectral_weight * carry_spectral_loss(
            pred_carry, target_carry, grid, config=config,
        )

    return loss
