#!/usr/bin/env python
"""Fetch PUBLISHED WeatherBench-2 headline deterministic RMSE for reference models.

Writes a CSV consumable by ``evaluations.baselines.load_sota_headline`` (columns
EXACTLY ``model,variable,level,lead_hours,rmse``) so the legoESM WB2 scorecard
plotter can overlay the official state-of-the-art curves alongside a trained
legoESM dycore.

HONESTY CONTRACT
----------------
This script NEVER fabricates numbers. Every RMSE it writes is READ from the
official WeatherBench-2 precomputed evaluation results in the public Google Cloud
bucket ``gs://weatherbench2/results/``. If the source is unreachable (offline / no
gcsfs / missing file) the script FAILS LOUD and writes NOTHING — it never emits
placeholder or guessed values. The companion ``evaluations/baselines.py`` only
parses the CSV; the numbers originate here, from the bucket.

SOURCE (canonical WeatherBench-2 published results)
---------------------------------------------------
Bucket root : ``gs://weatherbench2/results/``  (override with ``--source``)
Per-model file layout mirrors the official reproduction guide
(``docs/source/official-evaluation.md`` in google-research/weatherbench2):

    {source}/{RESOLUTION_DIR}/deterministic/{PREFIX}deterministic.nc

where ``PREFIX`` is the model's ``--output_file_prefix`` from that guide
(e.g. ``hres_vs_era_2020_``) and ``RESOLUTION_DIR`` is the evaluation grid
(default ``240x121`` = the 1.5-deg grid on which the official website scores
are computed). Each NetCDF is an xarray dataset whose DATA VARIABLES are the WB2
long variable names (``geopotential``, ``temperature``, ``2m_temperature``,
``specific_humidity``, ``u_component_of_wind``, ``10m_wind_speed``, ...) and whose
coordinates are ``metric`` (values include ``mse``), ``level`` [hPa],
``region`` (we take ``global``) and a lead-time coordinate
(``lead_time`` or ``prediction_timedelta`` [timedelta64]).

RMSE convention (ECMWF, matching the WB2 notebook): the stored metric is the
time-mean MSE; RMSE = sqrt(mean MSE). We select ``metric='mse'`` and take the
square root — we do NOT trust a pre-stored ``rmse`` slice (it may be absent).

VARIABLE / LEVEL -> legoESM headline scorecard key mapping
----------------------------------------------------------
The CSV ``variable`` column carries the legoESM headline key (matching
``evaluations.wb_forecast.HEADLINE_FIELD_KEYS`` / ``headline_diagnostics``), and
``level`` carries the pressure level in hPa (``0`` for screen/surface fields, a
convention the plotter reads back). The WB2-longname/level -> (key, level) map is:

    WB2 data var          level[hPa]   -> CSV variable   CSV level
    geopotential          500             z500           500
    temperature           850             t850           850
    specific_humidity     700             q700           700
    u_component_of_wind   850             u850           850
    u_component_of_wind   500             u500           500
    u_component_of_wind   250             u250           250
    2m_temperature        (surface)       t2m            0
    10m_wind_speed        (surface)       wind_speed_10m 0

(Only these keys are emitted; they are the subset of ``HEADLINE_FIELD_KEYS`` for
which WB2 publishes an official deterministic scorecard. z500/t850/q700/u850/
u500/u250 are exact pressure-level fields; t2m and wind_speed_10m are the
screen/anemometer fields, keyed at level 0 to match the plotter's surface
convention.)

RUN (as a DATA job, NOT on a login node)
----------------------------------------
    sbatch --account=glab --wrap "PYTHONPATH=packages/atmosphere:packages/core \
      python scripts/data/fetch_wb2_sota.py \
        --out config/wb/sota/wb2_headline_rmse.csv"

Requires ``xarray`` + ``gcsfs`` (public bucket, anonymous read). Import-light top
level (argparse/csv/math/pathlib only): the CLI/arg-parse + CSV-writer layers are
JAX- and gcsfs-free so they load and test on a login node.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import NamedTuple

# --- exact CSV column order required by evaluations.baselines.load_sota_headline ---
SOTA_CSV_COLUMNS = ("model", "variable", "level", "lead_hours", "rmse")

# --- canonical WeatherBench-2 published-results bucket (see module docstring) ---
WB2_RESULTS_BUCKET = "gs://weatherbench2/results/"
# Official website scores are computed at 1.5 deg = 240x121; that subdir holds
# the per-model deterministic scorecards.
WB2_RESOLUTION_DIR = "240x121"
WB2_REGION = "global"          # the WB2 'region' coordinate slice we score against

# WB2 result NetCDF stores time-mean MSE under this metric coordinate value;
# RMSE = sqrt(mse) (ECMWF convention, matching the WB2 evaluation notebook).
_MSE_METRIC = "mse"
# lead-time coordinate is named one of these in WB2 datasets (ECMWF conventions)
_LEAD_COORD_CANDIDATES = ("lead_time", "prediction_timedelta")

_SECONDS_PER_HOUR = 3600.0


# --- reference model -> WB2 result-file prefix (from official-evaluation.md) ---
# Each maps a legoESM/CSV model label to the WB2 ``--output_file_prefix`` used to
# name ``{prefix}deterministic.nc`` under the resolution/deterministic dir.
MODEL_PREFIXES = {
    "IFS-HRES": "hres_vs_era_2020_",
    "GraphCast": "graphcast_vs_era_2020_",
    "Pangu-Weather": "pangu_vs_era_2020_",
    "GenCast": "gencast_vs_era_2020_",
    "NeuralGCM": "neuralgcm_deterministic_vs_era_2020_",
    "ERA5-climatology": "climatology_vs_era_2020_",
}

# Default model set to fetch (order preserved in the CSV).
DEFAULT_MODELS = tuple(MODEL_PREFIXES)


class HeadlineVar(NamedTuple):
    """One WB2 headline field to extract.

    wb2_var : WB2 long data-variable name in the result NetCDF.
    level_hpa : pressure level [hPa] to select, or None for a surface field.
    csv_variable : legoESM headline scorecard key written to the CSV.
    csv_level : integer level written to the CSV (0 for surface fields).
    """
    wb2_var: str
    level_hpa: int | None
    csv_variable: str
    csv_level: int


# WB2-longname/level -> legoESM headline key. See module docstring for the table.
HEADLINE_VARS = (
    HeadlineVar("geopotential", 500, "z500", 500),
    HeadlineVar("temperature", 850, "t850", 850),
    HeadlineVar("specific_humidity", 700, "q700", 700),
    HeadlineVar("u_component_of_wind", 850, "u850", 850),
    HeadlineVar("u_component_of_wind", 500, "u500", 500),
    HeadlineVar("u_component_of_wind", 250, "u250", 250),
    HeadlineVar("2m_temperature", None, "t2m", 0),
    HeadlineVar("10m_wind_speed", None, "wind_speed_10m", 0),
)

# WB2 publishes leads on a 6/12 h grid to 15 days; the headline set is 1/3/5/10 d.
DEFAULT_LEADS_HOURS = (24, 72, 120, 240)


class FetchConfig(NamedTuple):
    source: str
    resolution_dir: str
    models: tuple
    leads_hours: tuple
    out: str


def _parse_leads(s):
    if not s:
        raise SystemExit("--leads must list at least one lead time in hours")
    leads = tuple(int(x) for x in str(s).split(",") if x != "")
    if not leads or any(lead <= 0 for lead in leads):
        raise SystemExit(f"--leads must be positive integers (hours), got {s!r}")
    return leads


def _parse_models(s):
    """Comma-separated model labels; each MUST be a known reference model.

    An unknown label is a hard SystemExit (no silent skip / default) so a typo
    can never silently drop a model from the scorecard.
    """
    if not s:
        return DEFAULT_MODELS
    models = tuple(m for m in str(s).split(",") if m != "")
    unknown = [m for m in models if m not in MODEL_PREFIXES]
    if unknown:
        raise SystemExit(
            f"--models: unknown reference model(s) {unknown}; "
            f"known models are {sorted(MODEL_PREFIXES)}")
    if not models:
        raise SystemExit("--models resolved to an empty set")
    return models


def build_fetch_config_from_args(argv=None) -> FetchConfig:
    """Parse CLI into a FetchConfig. Import-light + JAX/gcsfs-free (login-safe)."""
    p = argparse.ArgumentParser(
        description="Fetch published WeatherBench-2 headline deterministic RMSE "
                    "for reference models into a baselines-compatible CSV.")
    p.add_argument(
        "--source", default=WB2_RESULTS_BUCKET,
        help="Root of the WB2 published results (a gs:// URL or a local dir). "
             f"Default: {WB2_RESULTS_BUCKET}")
    p.add_argument(
        "--resolution-dir", default=WB2_RESOLUTION_DIR, dest="resolution_dir",
        help="Evaluation-grid subdirectory under the source "
             f"(default {WB2_RESOLUTION_DIR!r} = the 1.5-deg official grid).")
    p.add_argument(
        "--models", default="",
        help="Comma-separated reference models to fetch (default: all). "
             f"Choices: {','.join(MODEL_PREFIXES)}.")
    p.add_argument(
        "--leads", default=",".join(str(x) for x in DEFAULT_LEADS_HOURS),
        help="Comma-separated forecast leads [h] to extract (default "
             f"{','.join(str(x) for x in DEFAULT_LEADS_HOURS)} = 1/3/5/10 day).")
    p.add_argument(
        "--out", default="config/wb/sota/wb2_headline_rmse.csv",
        help="Output CSV path (columns model,variable,level,lead_hours,rmse).")
    a = p.parse_args(argv)
    return FetchConfig(
        source=a.source,
        resolution_dir=a.resolution_dir,
        models=_parse_models(a.models),
        leads_hours=_parse_leads(a.leads),
        out=a.out,
    )


def write_sota_csv(records, out_path):
    """Write ``records`` to ``out_path`` in the exact baselines-CSV layout.

    Parameters
    ----------
    records : iterable of dict
        Each with keys ``model, variable, level, lead_hours, rmse``. ``level`` and
        ``lead_hours`` coerce to int; ``rmse`` to float.
    out_path : str or Path
        Destination CSV. Parent directories are created.

    Validation (fail LOUD, write NOTHING on error — the file is written only
    after every record passes):
      * duplicate ``(model, variable, level, lead_hours)`` key -> ValueError;
      * non-finite or non-positive ``rmse`` -> ValueError;
      * empty record set -> ValueError (never emit an empty scorecard).

    Returns the number of rows written.
    """
    rows = []
    seen = set()
    for rec in records:
        try:
            model = str(rec["model"])
            variable = str(rec["variable"])
            level = int(rec["level"])
            lead_hours = int(rec["lead_hours"])
            rmse_val = float(rec["rmse"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"bad SOTA record {rec!r}: {exc}") from exc
        if not math.isfinite(rmse_val):
            raise ValueError(f"non-finite rmse in record {rec!r}")
        if rmse_val <= 0.0:
            raise ValueError(f"non-positive rmse {rmse_val!r} in record {rec!r}")
        key = (model, variable, level, lead_hours)
        if key in seen:
            raise ValueError(f"duplicate SOTA key {key}")
        seen.add(key)
        rows.append(
            {"model": model, "variable": variable, "level": level,
             "lead_hours": lead_hours, "rmse": rmse_val})

    if not rows:
        raise ValueError("no SOTA records to write (refusing to emit empty CSV)")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SOTA_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _find_lead_coord(ds):
    """Return the name of the lead-time coordinate in a WB2 result dataset."""
    for name in _LEAD_COORD_CANDIDATES:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(
        f"WB2 result has no lead-time coordinate (looked for "
        f"{_LEAD_COORD_CANDIDATES}); found coords {list(ds.coords)}")


def _lead_hours_of(td):
    """Convert a numpy timedelta64 lead value to whole hours (int)."""
    import numpy as np

    seconds = float(np.asarray(td, dtype="timedelta64[s]").astype("float64"))
    hours = seconds / _SECONDS_PER_HOUR
    return int(round(hours))


def _open_result(source, resolution_dir, prefix):
    """Open one model's WB2 deterministic result NetCDF as an xarray dataset.

    Fails loud (FileNotFoundError / OSError) if the file cannot be opened, so a
    missing/unreachable source aborts the whole fetch and writes nothing.
    """
    import xarray as xr

    root = str(source).rstrip("/")
    path = f"{root}/{resolution_dir}/deterministic/{prefix}deterministic.nc"
    if path.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem(token="anon")
        if not fs.exists(path):
            raise FileNotFoundError(
                f"WB2 result not found in bucket: {path} "
                "(no fabrication — aborting rather than guessing).")
        with fs.open(path, "rb") as fh:
            return xr.open_dataset(fh).load()
    local = Path(path)
    if not local.exists():
        raise FileNotFoundError(f"WB2 result not found: {local}")
    return xr.open_dataset(local).load()


def fetch_records(cfg):
    """Read WB2 published RMSE for the configured models/vars/leads.

    Returns a list of ``{model, variable, level, lead_hours, rmse}`` dicts. Heavy
    imports (xarray/gcsfs/numpy) are deferred to the call sites, so this only
    runs off the login node. Raises loudly on any unreachable source / missing
    variable / missing lead — never silently drops or invents a value.
    """
    records = []
    wanted_leads = set(int(x) for x in cfg.leads_hours)
    for model in cfg.models:
        if model not in MODEL_PREFIXES:                 # dispatch hardening
            raise ValueError(f"unknown reference model {model!r}")
        ds = _open_result(cfg.source, cfg.resolution_dir, MODEL_PREFIXES[model])
        lead_coord = _find_lead_coord(ds)

        # Region slice (global), if the dataset carries a region coordinate.
        if "region" in ds.coords or "region" in ds.dims:
            ds = ds.sel(region=WB2_REGION)

        for hv in HEADLINE_VARS:
            if hv.wb2_var not in ds:
                # A requested headline variable absent from THIS model's file is
                # a real gap (e.g. a model that did not report 10m wind); skip
                # only this (model, variable) rather than fabricate. Documented,
                # not silent: every skip is a genuine absence in the source.
                continue
            da = ds[hv.wb2_var].sel(metric=_MSE_METRIC)
            if hv.level_hpa is not None:
                if "level" not in da.coords and "level" not in da.dims:
                    continue
                da = da.sel(level=hv.level_hpa)
            # iterate over the model's published leads; emit only the wanted ones
            for i in range(da.sizes[lead_coord]):
                lead_val = da[lead_coord].values[i]
                lead_h = _lead_hours_of(lead_val)
                if lead_h not in wanted_leads:
                    continue
                mse = float(da.isel({lead_coord: i}).values)
                if not math.isfinite(mse) or mse < 0.0:
                    raise ValueError(
                        f"{model}/{hv.wb2_var}@{hv.level_hpa} lead {lead_h}h: "
                        f"non-finite/negative MSE {mse!r} in source (aborting).")
                rmse_val = math.sqrt(mse)     # ECMWF convention: sqrt(mean MSE)
                records.append({
                    "model": model,
                    "variable": hv.csv_variable,
                    "level": hv.csv_level,
                    "lead_hours": lead_h,
                    "rmse": rmse_val,
                })
    if not records:
        raise ValueError(
            "fetched zero WB2 records — source reachable but produced no matching "
            "(model, variable, lead) rows; refusing to write an empty CSV.")
    return records


def main(argv=None):
    """Fetch WB2 SOTA headline RMSE and write the baselines CSV."""
    cfg = build_fetch_config_from_args(argv)

    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("fetch_wb2_sota")
    log.info(
        "fetching WB2 SOTA headline RMSE: source=%s res=%s models=%s leads=%s",
        cfg.source, cfg.resolution_dir, list(cfg.models), list(cfg.leads_hours))

    records = fetch_records(cfg)          # raises loudly if source unreachable
    n = write_sota_csv(records, cfg.out)  # de-dups + validates before writing
    log.info("wrote %d rows -> %s", n, cfg.out)
    return cfg.out


if __name__ == "__main__":
    main()
