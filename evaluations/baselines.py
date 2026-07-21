"""Baseline forecasts and SOTA reference scores for the WB2 headline scorecard.

- ``persistence_forecast``: persist the initial condition at every lead time.
- ``climatology_forecast``: the climatology at every lead time (lead-independent).
- ``load_sota_headline``: read a CSV of PUBLISHED WeatherBench-2 headline RMSE
  for the reference models (IFS-HRES, GraphCast, Pangu, GenCast, NeuralGCM,
  climatology, persistence).

Honesty note: the SOTA CSV is populated from the OFFICIAL WeatherBench-2 results
(see ``scripts/data/fetch_wb2_sota_headline.py``), never hand-entered
approximations. This module only parses it; it does not invent numbers.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

__all__ = ["persistence_forecast", "climatology_forecast", "load_sota_headline"]


def persistence_forecast(ic_fields, leads_hours):
    """Persistence baseline: the initial-condition fields held constant.

    Parameters
    ----------
    ic_fields : dict[str, (n_lat, n_lon) array]
        Headline fields at initialization.
    leads_hours : iterable of int
        Forecast lead times [h].

    Returns
    -------
    dict[int, dict]
        ``{lead_hours: ic_fields}`` — the same fields at every lead (skill
        degrades because the verification changes with lead while this does not).
        The field arrays are ALIASED across leads (read-only baseline for
        scoring; do not mutate forecast fields in place).
    """
    return {int(lead): dict(ic_fields) for lead in leads_hours}


def climatology_forecast(clim_fields, leads_hours):
    """Climatology baseline: lead-independent climatology fields.

    Parameters
    ----------
    clim_fields : dict[str, (n_lat, n_lon) array]
        Climatological headline fields for the initialization's valid season.
    leads_hours : iterable of int
        Forecast lead times [h].

    Returns
    -------
    dict[int, dict]
        ``{lead_hours: clim_fields}``. A season-resolved climatology would index
        by each lead's valid-time day-of-year; the orchestrator supplies the
        correct climatology per valid time (see scripts/data). Field arrays are
        ALIASED across leads (read-only baseline; do not mutate).
    """
    return {int(lead): dict(clim_fields) for lead in leads_hours}


_SOTA_COLUMNS = ("model", "variable", "level", "lead_hours", "rmse")


def load_sota_headline(csv_path):
    """Load published WB2 headline RMSE for reference models.

    CSV columns: ``model, variable, level, lead_hours, rmse``.

    Returns
    -------
    dict[str, dict[tuple, float]]
        ``{model: {(variable, level, lead_hours): rmse}}``.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"SOTA headline CSV not found: {path}")
    out: dict = {}
    seen: set = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not set(_SOTA_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(
                f"SOTA CSV must have columns {list(_SOTA_COLUMNS)}, got {reader.fieldnames}"
            )
        for lineno, row in enumerate(reader, start=2):   # data rows start at line 2
            try:
                key = (row["variable"], int(row["level"]), int(row["lead_hours"]))
                rmse_val = float(row["rmse"])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"SOTA CSV line {lineno}: bad numeric value ({exc}); row={row}"
                ) from exc
            if not math.isfinite(rmse_val):
                raise ValueError(
                    f"SOTA CSV line {lineno}: non-finite rmse {row['rmse']!r}; row={row}"
                )
            dedup_key = (row["model"], *key)
            if dedup_key in seen:                        # duplicates must not silently overwrite
                raise ValueError(f"SOTA CSV line {lineno}: duplicate entry {dedup_key}")
            seen.add(dedup_key)
            out.setdefault(row["model"], {})[key] = rmse_val
    return out
