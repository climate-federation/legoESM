"""WeatherBench-2 forecast-eval orchestrator (Stage 0, Task 9b).

Ties the scorer pieces into the WB2 protocol: for each initialization, roll the
trained spectral dycore forward to each lead time, diagnose the headline fields,
regrid to the WB2 grid, and score against the (WB2-grid) ERA5 verification;
average the metrics over all initializations.

Data loading (ERA5 IC + verification) and checkpoint loading live in the CLI
layer (scripts/validate/run_weatherbench_eval.py); this module operates on
already-built spectral states + already-regridded verification, so it is unit
testable without network/GPU.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .wb_forecast import diagnose_and_regrid, score_forecast
from .wb_regrid import wb2_grid

__all__ = ["ForecastCase", "run_wb_forecast_eval"]

_SECONDS_PER_HOUR = 3600.0
_METRICS = ("rmse", "acc", "bias")


class ForecastCase(NamedTuple):
    """One initialization: the spectral IC and its verification at each lead.

    init_state : SpectralHydrostaticState
        Initial condition to roll forward.
    verif_by_lead : dict[int, dict]
        ``{lead_hours: {"fields": {key: (n_lat,n_lon)}, "valid": {key: bool mask}}}``
        — ERA5 verification already on the WB2 grid (see the CLI layer).
    """
    init_state: object
    verif_by_lead: dict


def run_wb_forecast_eval(physics_fn, grid, sigma_coord, pe_config, dt,
                         cases, leads_hours, clim_fields_wb2, *,
                         resolution_deg=None, rollout_fn=None):
    """Run the WB2 forecast protocol and return a lead-time scorecard.

    Parameters
    ----------
    physics_fn : callable
        ``(state, grid, sigma_coord) -> SpectralHydrostaticState`` learned-physics
        tendency (as produced by the AIMIP ``make_*_spectral_physics`` factories).
    grid, sigma_coord, pe_config, dt : dycore configuration.
    cases : iterable of ForecastCase
        Initializations + their WB2-grid verification.
    leads_hours : iterable of int
        Forecast lead times [h]. Each lead is an INDEPENDENT forecast from t0
        (the rollout is restarted per lead; cost is O(sum leads), not O(max
        lead) — snapshotting could optimize this later).
    clim_fields_wb2 : dict[str, (n_lat,n_lon)]
        Climatology on the WB2 grid for ACC (same for all cases here; a
        valid-time climatology would be supplied per case by the caller).
    rollout_fn : callable, optional
        Rollout with signature ``(state, physics_fn, grid, sigma_coord,
        pe_config, dt, n_steps) -> state``. Defaults to the real
        ``neural_gcm_spectral.spectral_rollout``; injectable for testing.

    Returns
    -------
    dict[(str, int), dict]
        ``{(headline_key, lead_hours): {"rmse","acc","bias"}}`` averaged over cases.
    """
    if rollout_fn is None:
        from legoesm.training.neural_gcm_spectral import spectral_rollout
        rollout_fn = spectral_rollout

    wb2_lat, _ = wb2_grid() if resolution_deg is None else wb2_grid(resolution_deg)
    accum: dict = {}   # (key, lead) -> list of per-case metric dicts

    for case in cases:
        for lead in leads_hours:
            n_steps = int(round(lead * _SECONDS_PER_HOUR / dt))
            rolled = rollout_fn(
                case.init_state, physics_fn, grid, sigma_coord, pe_config, dt, n_steps)
            pred_fields, pred_valid, _, _ = diagnose_and_regrid(
                rolled, grid, sigma_coord, resolution_deg=resolution_deg)
            verif = case.verif_by_lead[int(lead)]
            # a cell is scorable only where it is above ground in BOTH pred and verif
            combined_valid = {
                k: (pred_valid[k] & np.asarray(verif["valid"][k])) for k in pred_fields
            }
            case_scores = score_forecast(
                pred_fields, verif["fields"], clim_fields_wb2, wb2_lat, valid=combined_valid)
            for key, metrics in case_scores.items():
                accum.setdefault((key, int(lead)), []).append(metrics)

    scorecard: dict = {}
    for key_lead, per_case in accum.items():
        scorecard[key_lead] = {
            m: float(np.mean([d[m] for d in per_case])) for m in _METRICS
        }
    return scorecard
