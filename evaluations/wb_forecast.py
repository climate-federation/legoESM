"""WeatherBench-2 forecast scorer for legoESM spectral dycores.

Stage 0 bridge: turn a rolled-out ``SpectralHydrostaticState`` into the WB2
headline-variable set on the model grid, so a trained dycore can be scored with
the (reused) ``evaluations.metrics`` RMSE/ACC/bias primitives.

Task 5 (this file, first piece): ``diagnose_headline_fields``.

Surface-variable caveat: the spectral DYNAMICAL state carries no skin
temperature, roughness length, or Obukhov length. Pressure-level fields
(z500, t850, q700, u/v at 850/700/500/250) and MSLP are diagnosed inside the
model column; the surface fields are documented proxies (see
``diagnose_headline_fields``). Total precipitation is not available from the
dynamical state (needs physics-output accumulation) and is omitted here.

Below-ground caveat: a target pressure level can lie BELOW the surface over
high terrain (e.g. 850 hPa over the Andes, 500 hPa over Everest/Tibet). There
the pressure interpolation clamps to the lowest model level, which is NOT a
faithful pressure-level value. Every pressure-level field therefore ships a
companion validity mask (``valid[key]`` True where the level is at/above the
surface); the scorer MUST mask those cells rather than treat clamped values as
real. Surface fields are valid everywhere.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants

from .headline_diagnostics import (
    geopotential_height_at,
    geopotential_on_levels,
    interp_to_pressure_level,
    mean_sea_level_pressure,
    wind_10m,
)

__all__ = [
    "diagnose_headline_fields",
    "diagnose_and_regrid",
    "score_forecast",
    "score_ensemble_forecast",
    "HeadlineDiagnosis",
    "HEADLINE_FIELD_KEYS",
]

# --- WB2 headline pressure levels (Pa) ---
_Z500_PA = 50000.0
_T850_PA = 85000.0
_Q700_PA = 70000.0
_WIND_LEVELS_PA = (("850", 85000.0), ("700", 70000.0), ("500", 50000.0), ("250", 25000.0))

# --- surface-proxy constants ---
_Z0M_SURFACE_PROXY = 2e-4   # open-ocean momentum roughness [m] for the neutral 10 m reduction
_L_NEUTRAL = 1e12           # Obukhov length standing in for neutral stability [m]
_Z_10M = 10.0               # WMO anemometer reference height [m]

HEADLINE_FIELD_KEYS = (
    "z500", "t850", "q700",
    "u850", "v850", "u700", "v700", "u500", "v500", "u250", "v250",
    "mslp", "t2m", "u10", "v10", "wind_speed_10m",
)

# pressure-level keys that carry a below-ground validity mask, and their target Pa
_PLEVEL_TARGET_PA = {
    "z500": _Z500_PA, "t850": _T850_PA, "q700": _Q700_PA,
    "u850": 85000.0, "v850": 85000.0, "u700": 70000.0, "v700": 70000.0,
    "u500": 50000.0, "v500": 50000.0, "u250": 25000.0, "v250": 25000.0,
}


class HeadlineDiagnosis(NamedTuple):
    """Result of :func:`diagnose_headline_fields`.

    fields : dict[str, (n_lat, n_lon) array]  — the WB2 headline fields.
    valid  : dict[str, (n_lat, n_lon) bool]   — per-field above-ground mask
             (True = usable). Pressure-level fields are invalid where the target
             level is below the surface; surface fields are valid everywhere.
    """
    fields: dict
    valid: dict


def diagnose_headline_fields(state, grid, sigma_coord) -> HeadlineDiagnosis:
    """Map a rolled-out ``SpectralHydrostaticState`` to WB2 headline fields.

    Returns a :class:`HeadlineDiagnosis` (``.fields`` keyed by
    :data:`HEADLINE_FIELD_KEYS`, ``.valid`` the matching above-ground masks).

    Pressure-level fields: ``z500`` (geopotential height, m), ``t850`` (K),
    ``q700`` (kg/kg), ``u/v`` at 850/700/500/250 hPa (m/s). These are exact
    within the model column; where the target level is below the surface the
    value is clamped and flagged ``valid=False`` (score with the mask).

    ``mslp`` (Pa): standard sea-level reduction, valid everywhere.

    Surface proxies (the spectral state has no skin-T / roughness / Obukhov L):
      * ``t2m`` := lowest-level air temperature.
      * ``u10``/``v10``/``wind_speed_10m`` := the lowest-level wind reduced to
        10 m by a NEUTRAL log law, using the lowest-level height above ground
        from the geopotential and an open-terrain roughness — a documented
        global-neutral proxy that does not resolve land/ocean roughness or
        stability, so surface-wind scores carry that caveat.
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid

    gridded = spectral_pe_to_grid(state, grid, sigma_coord)
    T, u, v = gridded["T"], gridded["u"], gridded["v"]   # (n_lat, n_lon, nlev)
    p_s, phis = gridded["p_s"], gridded["phis"]          # (n_lat, n_lon)

    # q_v tracer may be a Field (has .data) or a raw array; both are accepted by
    # SpectralHydrostaticState, so duck-type and validate the layout.
    tracers = getattr(state, "tracers", None)
    if tracers is not None and "q_v" in tracers:
        q_raw = tracers["q_v"]
        q_data = q_raw.data if hasattr(q_raw, "data") else q_raw
        q = jnp.asarray(q_data, dtype=T.dtype)
        if q.shape != T.shape:
            raise ValueError(f"q_v tracer shape {q.shape} != temperature shape {T.shape}")
    else:
        q = jnp.zeros_like(T)

    p_model = sigma_coord.pressure_at_full(p_s)          # (n_lat, n_lon, nlev), Pa
    all_valid = jnp.ones(p_s.shape, dtype=bool)

    fields, valid = {}, {}

    # --- pressure-level headline fields (masked below the surface) ---
    fields["z500"] = geopotential_height_at(T, q, p_s, phis, sigma_coord, _Z500_PA)
    fields["t850"] = interp_to_pressure_level(T, p_model, _T850_PA)
    fields["q700"] = interp_to_pressure_level(q, p_model, _Q700_PA)
    for name, pa in _WIND_LEVELS_PA:
        fields[f"u{name}"] = interp_to_pressure_level(u, p_model, pa)
        fields[f"v{name}"] = interp_to_pressure_level(v, p_model, pa)
    for key, pa in _PLEVEL_TARGET_PA.items():
        valid[key] = pa <= p_s                           # True where level is at/above surface

    # --- MSLP (defined everywhere) ---
    fields["mslp"] = mean_sea_level_pressure(p_s, T[..., -1], phis)
    valid["mslp"] = all_valid

    # --- surface proxies (spectral state carries no skin-T / z0 / L) ---
    fields["t2m"] = T[..., -1]                            # lowest-level air T (proxy)
    valid["t2m"] = all_valid
    phi = geopotential_on_levels(T, q, p_s, phis, sigma_coord)
    # lowest-level height above ground [m]; floor at the 10 m ref so the log-law
    # bracket z_ref <= z_low stays valid.
    z_low = jnp.maximum((phi[..., -1] - phis) / constants.g, _Z_10M)
    L = jnp.full_like(z_low, _L_NEUTRAL)
    z0m = jnp.full_like(z_low, _Z0M_SURFACE_PROXY)
    speed10 = wind_10m(u[..., -1], v[..., -1], z_low, L, z0m)
    speed_low = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + 1e-12)
    scale = speed10 / speed_low                           # in [0, 1]
    fields["u10"] = u[..., -1] * scale
    fields["v10"] = v[..., -1] * scale
    fields["wind_speed_10m"] = speed10
    valid["u10"] = valid["v10"] = valid["wind_speed_10m"] = all_valid

    return HeadlineDiagnosis(fields=fields, valid=valid)


def diagnose_and_regrid(state, grid, sigma_coord, *, resolution_deg=None):
    """Diagnose WB2 headline fields from a spectral state and regrid to the WB2 grid.

    Bridges :func:`diagnose_headline_fields` (Task 5) and
    ``wb_regrid.regrid_to_wb2`` (Task 6): the model Gaussian-grid headline fields
    and their above-ground masks are bilinearly regridded onto the WB2 common
    grid, so a forecast and its ERA5 verification can be scored on identical
    coordinates.

    Returns
    -------
    (fields_wb2, valid_wb2, wb2_lat_deg, wb2_lon_deg)
        Dicts keyed by :data:`HEADLINE_FIELD_KEYS` on the WB2 grid, plus its
        latitude/longitude in degrees.
    """
    import numpy as np

    from .wb_regrid import WB2_RESOLUTION_DEG, regrid_to_wb2

    res = WB2_RESOLUTION_DEG if resolution_deg is None else resolution_deg
    diag = diagnose_headline_fields(state, grid, sigma_coord)
    src_lat = np.rad2deg(np.asarray(grid.lat))       # Gaussian lat, radians S->N
    src_lon = np.rad2deg(np.asarray(grid.lon))       # radians [0, 2pi)

    fields_wb2, valid_wb2 = {}, {}
    wb2_lat = wb2_lon = None
    for key, field in diag.fields.items():
        f, wb2_lat, wb2_lon = regrid_to_wb2(
            np.asarray(field), src_lat, src_lon, resolution_deg=res)
        # CONSERVATIVE validity mask: a WB2 cell is valid only if ALL source
        # cells contributing to its bilinear interpolation were valid, so a
        # "valid" cell's field value cannot have been contaminated by a
        # below-ground/clamped source cell.
        m, _, _ = regrid_to_wb2(
            np.asarray(diag.valid[key]), src_lat, src_lon, resolution_deg=res,
            mask=True, mask_threshold=1.0 - 1e-9)
        fields_wb2[key] = f
        valid_wb2[key] = m
    return fields_wb2, valid_wb2, wb2_lat, wb2_lon


def score_forecast(pred_fields, verif_fields, clim_fields, wb2_lat_deg, *, valid=None):
    """Score WB2 headline forecast fields against verification.

    Reuses the shared ``evaluations.metrics`` primitives (no new metric math).

    Parameters
    ----------
    pred_fields, verif_fields, clim_fields : dict[str, (n_lat, n_lon) array]
        Forecast, verification (ERA5), and climatology fields on the SAME WB2
        grid, keyed identically (e.g. by :data:`HEADLINE_FIELD_KEYS`).
    wb2_lat_deg : array (n_lat,)
        WB2 grid latitudes in degrees; cosine-latitude quadrature weights are
        used (poles get zero weight).
    valid : dict[str, (n_lat, n_lon) bool], optional
        Per-field above-ground mask from :func:`diagnose_headline_fields`;
        masked (below-ground) cells are excluded from RMSE/ACC/bias.

    Returns
    -------
    dict[str, dict]
        ``{key: {"rmse": float, "acc": float, "bias": float}}``.
    """
    import warnings

    import numpy as np

    from .metrics import acc, bias, rmse

    weights = jnp.asarray(np.cos(np.deg2rad(np.asarray(wb2_lat_deg, dtype=np.float64))))
    scores = {}
    for key, pred in pred_fields.items():
        target = verif_fields[key]
        clim = clim_fields[key]
        m = None if valid is None else valid.get(key)
        if m is not None and float(jnp.sum(weights[:, None] * m)) <= 0.0:
            # An all-masked field has no valid (above-ground) cells -> the score
            # is undefined. Record NaN + warn LOUDLY rather than crashing the
            # whole scorecard: a physically-degenerate forecast (e.g. an
            # undertrained full SFNO emulator whose surface pressure is
            # unphysical, so every pressure level reads as below-ground) is a
            # legitimate input in a MODEL COMPARISON — one bad field must not
            # abort the other fields' scores. The warning still surfaces a
            # genuine below-ground-mask bug during development.
            warnings.warn(
                f"headline field {key!r} has no valid (above-ground) cells to "
                f"score; recording NaN (degenerate forecast or masking bug)",
                RuntimeWarning, stacklevel=2)
            scores[key] = {"rmse": float("nan"), "acc": float("nan"), "bias": float("nan")}
            continue
        scores[key] = {
            "rmse": float(rmse(pred, target, weights, mask=m)),
            "acc": float(acc(pred, target, clim, weights, mask=m)),
            "bias": float(bias(pred, target, weights, mask=m)),
        }
    return scores


def score_ensemble_forecast(
    member_fields, verif_fields, wb2_lat_deg, *, valid=None, alpha=0.95,
):
    """PROBABILISTIC scores for an ensemble of WB2 headline forecast fields.

    The deterministic scorecard collapses an ensemble to its mean and reports
    RMSE — which is blind to whether the spread is right, and therefore cannot
    see what a CRPS-trained model buys.  This is the missing instrument: it
    reports the score the U-Cast recipe (arXiv:2604.09041) actually optimises.

    No new metric math: the pointwise CRPS comes from
    :func:`legoesm.ml.loss.almost_fair_crps` (the same estimator the training
    loss uses, so train and eval cannot drift), and the area weighting /
    masking convention is copied from :func:`evaluations.metrics.rmse`
    (``cos(lat)`` weights times the above-ground mask) so a CRPS and an RMSE
    on the same row are integrated over exactly the same cells.

    Parameters
    ----------
    member_fields : sequence of dict[str, (n_lat, n_lon) array]
        One diagnosed+regridded field dict per ensemble member.
    verif_fields : dict[str, (n_lat, n_lon) array]
        ERA5 verification on the same WB2 grid.
    wb2_lat_deg : array (n_lat,)
        WB2 latitudes [deg].
    valid : dict[str, (n_lat, n_lon) bool], optional
        Above-ground mask, as in :func:`score_forecast`.
    alpha : float, default 0.95
        Almost-fair finite-ensemble adjustment (1.0 = fair CRPS).

    Returns
    -------
    dict[str, dict]
        ``{key: {"crps", "spread", "spread_skill"}}``.

        * ``crps`` — area-weighted almost-fair CRPS [field units]. LOWER is
          better; for a single member it degenerates to the MAE, which is why
          a 1-member call is refused.
        * ``spread`` — area-weighted RMS of the ensemble standard deviation
          (unbiased, ``ddof=1``).
        * ``spread_skill`` — ``spread / rmse(ensemble mean)`` times the
          finite-ensemble factor ``sqrt((M+1)/M)``.  ~1 is calibrated, <1 is
          UNDER-dispersed (the failure mode a deterministically-trained model
          shows), >1 over-dispersed.
    """
    import numpy as np

    from legoesm.ml.loss import almost_fair_crps

    from .metrics import rmse

    n_members = len(member_fields)
    if n_members < 2:
        raise ValueError(
            f"score_ensemble_forecast needs >= 2 members, got {n_members}; "
            "with one member the CRPS is just the MAE and the spread is 0."
        )

    weights = np.cos(np.deg2rad(np.asarray(wb2_lat_deg, dtype=np.float64)))
    # Finite-ensemble inflation of the spread-skill ratio: an M-member sample
    # standard deviation under-estimates the predictive spread that the
    # ensemble-MEAN error is compared against (Fortin et al. 2014).
    spread_factor = float(np.sqrt((n_members + 1.0) / n_members))

    scores = {}
    for key in member_fields[0]:
        ens = jnp.stack([jnp.asarray(mf[key]) for mf in member_fields])
        target = jnp.asarray(verif_fields[key])
        m = None if valid is None else valid.get(key)

        # Broadcast the (n_lat, 1) column of cos-lat weights against the full
        # (n_lat, n_lon) field BEFORE summing — exactly ``metrics.rmse``'s
        # ``jnp.sum(w * jnp.ones_like(sq_err))``. Summing the un-broadcast
        # column instead inflates every score by n_lon (caught by the
        # identical-members test, which must return the plain MAE).
        w = jnp.broadcast_to(
            jnp.asarray(weights)[:, None]
            * (1.0 if m is None else jnp.asarray(m, dtype=jnp.float64)),
            jnp.asarray(target).shape,
        )
        w_sum = float(jnp.sum(w))
        if w_sum <= 0.0:
            scores[key] = {
                "crps": float("nan"), "spread": float("nan"),
                "spread_skill": float("nan"),
            }
            continue

        crps_field = almost_fair_crps(ens, target, alpha=alpha)
        crps = float(jnp.sum(crps_field * w) / w_sum)

        # ddof=1: the ensemble is a SAMPLE from the predictive distribution.
        var = jnp.var(ens, axis=0, ddof=1)
        spread = float(jnp.sqrt(jnp.sum(var * w) / w_sum))

        ens_mean = jnp.mean(ens, axis=0)
        skill = float(rmse(ens_mean, target, jnp.asarray(weights), mask=m))
        scores[key] = {
            "crps": crps,
            "spread": spread,
            "spread_skill": (
                float(spread_factor * spread / skill) if skill > 0.0
                else float("nan")
            ),
        }
    return scores
