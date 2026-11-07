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

from .wb_forecast import (
    diagnose_and_regrid,
    score_ensemble_forecast,
    score_forecast,
)
from .wb_regrid import wb2_grid

__all__ = ["ForecastCase", "run_wb_forecast_eval"]

_SECONDS_PER_HOUR = 3600.0


class ForecastCase(NamedTuple):
    """One initialization: the spectral IC and its verification at each lead.

    init_state : SpectralHydrostaticState
        Initial condition to roll forward.
    verif_by_lead : dict[int, dict]
        ``{lead_hours: {"fields": {key: (n_lat,n_lon)}, "valid": {key: bool mask}}}``
        — ERA5 verification already on the WB2 grid (see the CLI layer).
    forcing : dict or None
        Optional per-init prescribed surface forcing the learned-physics rollout
        consumes (``{"T_sfc", "sic", "day_of_year", "seconds_of_day"}`` — the
        ``spectral_rollout.forcing_base`` dict). When set, it is passed as
        ``forcing_base=case.forcing`` to ``rollout_fn`` (the neural_gcm / sfno
        cores need it; the classical-physics core has ``forcing=None``). LAST
        field with a default so positional construction stays ABI-safe.
    """
    init_state: object
    verif_by_lead: dict
    forcing: dict | None = None


def run_wb_forecast_eval(physics_fn, grid, sigma_coord, pe_config, dt,
                         cases, leads_hours, clim_fields_wb2, *,
                         resolution_deg=None, rollout_fn=None,
                         field_sink=None):
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
    field_sink : dict, optional
        When given, the case-mean forecast and verification FIELDS are
        accumulated into it as ``{(key, lead): {"pred", "verif", "valid"}}``,
        on the WB2 grid. This is how a map of the forecast against ERA5 is
        drawn from the SAME forecasts the scorecard scores, rather than from a
        second rollout that could differ in protocol.
    rollout_fn : callable, optional
        Rollout with signature ``(state, physics_fn, grid, sigma_coord,
        pe_config, dt, n_steps) -> state``, plus an optional
        ``forcing_base=`` keyword passed when ``case.forcing`` is set. Defaults
        to the real ``neural_gcm_spectral.spectral_rollout`` (which accepts
        ``forcing_base``); injectable for testing.

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
            steps_f = lead * _SECONDS_PER_HOUR / dt
            n_steps = int(round(steps_f))
            if abs(steps_f - n_steps) > 1e-6:
                # silent rounding would score a different valid time than requested
                raise ValueError(
                    f"lead {lead} h ({lead * _SECONDS_PER_HOUR:g}s) is not an exact "
                    f"multiple of dt={dt:g}s (ratio {steps_f}); pick leads on the dt grid"
                )
            # Learned-physics cores (neural_gcm / sfno) need the per-init
            # prescribed forcing; pass it as forcing_base ONLY when the case
            # carries it, so a classical-physics case keeps the unforced
            # rollout_fn signature (spectral_rollout's default).
            if case.forcing is not None:
                rolled = rollout_fn(
                    case.init_state, physics_fn, grid, sigma_coord, pe_config,
                    dt, n_steps, forcing_base=case.forcing)
            else:
                rolled = rollout_fn(
                    case.init_state, physics_fn, grid, sigma_coord, pe_config,
                    dt, n_steps)
            # A rollout_fn may return ONE state (deterministic) or a LIST of
            # member states (ensemble). Diagnosing each member and averaging
            # the FIELDS — rather than averaging the spectral state and
            # diagnosing once — also removes the mean-then-diagnose error that
            # z500/mslp carry (both are nonlinear functions of the state).
            #
            # The list check must NOT accept a tuple: a spectral state is a
            # NamedTuple, i.e. itself a tuple, so ``isinstance(rolled, tuple)``
            # silently unpacks one state into its FIELDS and calls them
            # members. An ensemble must be a list.
            members = rolled if isinstance(rolled, list) else [rolled]
            diagnosed = [
                diagnose_and_regrid(
                    mem, grid, sigma_coord, resolution_deg=resolution_deg)
                for mem in members
            ]
            member_fields = [d[0] for d in diagnosed]
            verif = case.verif_by_lead[int(lead)]
            # a cell is scorable only where it is above ground in BOTH pred and
            # verif — and, for an ensemble, in EVERY member (a cell one member
            # calls below-ground is not scorable for the ensemble as a whole).
            combined_valid = {}
            for k in member_fields[0]:
                v = np.asarray(verif["valid"][k])
                for d in diagnosed:
                    v = v & d[1][k]
                combined_valid[k] = v
            pred_fields = (
                member_fields[0] if len(member_fields) == 1
                else {k: np.mean([mf[k] for mf in member_fields], axis=0)
                      for k in member_fields[0]}
            )
            case_scores = score_forecast(
                pred_fields, verif["fields"], clim_fields_wb2, wb2_lat, valid=combined_valid)
            if len(member_fields) > 1:
                ens_scores = score_ensemble_forecast(
                    member_fields, verif["fields"], wb2_lat,
                    valid=combined_valid)
                for key, metrics in ens_scores.items():
                    case_scores[key].update(metrics)
            for key, metrics in case_scores.items():
                accum.setdefault((key, int(lead)), []).append(metrics)
            if field_sink is not None:
                for key in pred_fields:
                    # Sums + a count, averaged once at the end: keeping every
                    # case's field would scale with n_inits for no gain.
                    slot = field_sink.setdefault(
                        (key, int(lead)),
                        {"pred": np.zeros_like(np.asarray(pred_fields[key],
                                                          dtype=float)),
                         "verif": np.zeros_like(np.asarray(pred_fields[key],
                                                           dtype=float)),
                         "n": np.zeros_like(np.asarray(pred_fields[key],
                                                       dtype=float))})
                    v = np.asarray(verif["fields"][key], dtype=float)
                    ok = np.asarray(combined_valid[key], dtype=bool)
                    slot["pred"] += np.where(
                        ok, np.asarray(pred_fields[key], dtype=float), 0.0)
                    slot["verif"] += np.where(ok, v, 0.0)
                    slot["n"] += ok.astype(float)

    scorecard: dict = {}
    for key_lead, per_case in accum.items():
        # Average over whatever metrics the cases carry: rmse/acc/bias always,
        # plus crps/spread/spread_skill when the rollout returned an ensemble.
        # A fixed metric tuple would have silently DROPPED the probabilistic
        # columns instead of reporting them.
        #
        # Every case must carry the SAME metrics: a rollout_fn that returned an
        # ensemble for some cases and a single state for others would otherwise
        # average a probabilistic column over a subset of cases (or KeyError,
        # depending on which case came first) — an order-dependent scorecard.
        metric_keys = set(per_case[0])
        for d in per_case[1:]:
            if set(d) != metric_keys:
                raise ValueError(
                    f"inconsistent metrics across cases for {key_lead}: "
                    f"{sorted(metric_keys)} vs {sorted(d)}. The rollout must "
                    f"return an ensemble for every case or for none."
                )
        scorecard[key_lead] = {
            m: float(np.mean([d[m] for d in per_case])) for m in metric_keys
        }
    return scorecard
