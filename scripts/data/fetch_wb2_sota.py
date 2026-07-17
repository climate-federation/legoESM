#!/usr/bin/env python
"""Fetch PUBLISHED WeatherBench-2 headline deterministic RMSE for reference models.

Writes a CSV consumable by ``evaluations.baselines.load_sota_headline`` (columns
EXACTLY ``model,variable,level,lead_hours,rmse``) so the legoESM WB2 scorecard
plotter (``scripts/plot/plot_wb_scorecard.py``) can overlay the official
state-of-the-art curves alongside a trained legoESM dycore.

HONESTY CONTRACT
----------------
This script NEVER fabricates numbers. Every RMSE it writes is READ from the
official WeatherBench-2 precomputed evaluation results in the public Google Cloud
bucket ``gs://weatherbench2/benchmark_results/``. If a source model file is
entirely unreachable AND no records at all were collected, the script FAILS LOUD
and writes NOTHING — it never emits placeholder or guessed values. A single
missing (variable, level, lead) inside an otherwise-reachable file is a
skip-with-warning (a genuine gap in the source), not a fabrication. The companion
``evaluations/baselines.py`` only parses the CSV; the numbers originate here.

SOURCE (real, probed WeatherBench-2 bucket layout)
--------------------------------------------------
Bucket dir : ``gs://weatherbench2/benchmark_results`` (override ``--source``).
The per-model scorecards are FLAT in that directory (there is NO resolution
subdirectory). Each file is named::

    {model}_vs_era5_{resolution}_{year}.nc

with ``resolution`` the evaluation-grid token (``240x121`` = the 1.5-deg official
grid; ``--resolution``), ``year`` the verification year (``--year``, default
2020), and ground truth ``era5``. Confirmed-present model files include
``graphcast_vs_era5_240x121_2020.nc``, ``gencast_...``, ``hres_...``,
``pangu_...``, ``neuralgcm_hres_...``, ``keisler_...``, ``fuxi_...`` and
``climatology_vs_era5_240x121_2020.nc``.

These NetCDFs are NetCDF-3 CLASSIC — h5netcdf FAILS on them. We read the bytes
with gcsfs and open them through xarray's ``scipy`` engine on an in-memory
buffer (see :func:`_open_result`).

INTERNAL STRUCTURE (confirmed on graphcast)
-------------------------------------------
Data variables are named ``{metric}.{variable}`` — e.g. ``rmse.geopotential``,
``rmse.temperature``, ``rmse.u_component_of_wind``, ``rmse.v_component_of_wind``,
``rmse.specific_humidity`` (also ``mse.``/``bias.``/``acc.`` variants). RMSE is
therefore PRE-STORED as ``rmse.<variable>`` — we read it DIRECTLY (no sqrt(mse),
and there is NO ``metric`` coordinate). A variable's dims are
``(region, lead_time, level)`` with coords: ``region`` (we select ``'global'``),
``lead_time`` (integer HOURS: 6, 12, 18, ..., 240) and ``level`` (pressure hPa:
500, 700, 850). These upper-air files carry NO surface fields (2m temperature,
MSLP, 10m wind), so the SOTA overlay is UPPER-AIR only.

VARIABLE / LEVEL -> CSV schema (this is the crux of the overlay lining up)
--------------------------------------------------------------------------
``evaluations.baselines.load_sota_headline`` keys each row on
``(variable, level, lead_hours)`` and the plotter
(``plot_wb_scorecard.FIELD_KEY_TO_SOTA``) matches a scorecard ``field_key`` to a
SOTA row by the WB2 ``(long variable name, hPa level)`` pair. So the CSV
``variable`` column MUST carry the WB2 LONG name and ``level`` the hPa INTEGER —
NOT the short field_key. The emitted (WB2 var, level) pairs and the scorecard
field_key each lines up with (verified against ``FIELD_KEY_TO_SOTA``) are:

    CSV variable (WB2 long)   level[hPa]   scorecard field_key
    geopotential              500          z500
    temperature               850          t850
    temperature               500          t500 (*)
    u_component_of_wind       850          u850
    u_component_of_wind       700          u700
    u_component_of_wind       500          u500
    v_component_of_wind       850          v850
    v_component_of_wind       700          v700
    v_component_of_wind       500          v500
    specific_humidity         700          q700

(*) ``t500`` is not a scorecard headline field_key; its row is harmless (unused
by the overlay unless a scorecard ever reports t500). Every OTHER pair matches a
``FIELD_KEY_TO_SOTA`` entry exactly, so the overlay is drawn rather than silently
dropped. Surface headline fields (t2m, mslp, 10m wind) are DROPPED here — they do
not exist in these upper-air files.

RUN (as a DATA job, NOT on a login node)
----------------------------------------
    sbatch --account=glab --wrap "PYTHONPATH=packages/atmosphere:packages/core \
      python scripts/data/fetch_wb2_sota.py \
        --out config/wb/sota/wb2_headline_rmse.csv"

Requires ``xarray`` + ``gcsfs`` + ``scipy`` (public bucket, anonymous read).
Import-light top level (argparse/csv/math/pathlib only): the CLI/arg-parse + CSV
writer layers are JAX/gcsfs/xarray-free so they load and test on a login node.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import NamedTuple

# --- exact CSV column order required by evaluations.baselines.load_sota_headline ---
SOTA_CSV_COLUMNS = ("model", "variable", "level", "lead_hours", "rmse")

# --- real WeatherBench-2 benchmark-results bucket (probed; see module docstring) ---
# Files are FLAT in this directory (no resolution subdir).
WB2_RESULTS_BUCKET = "gs://weatherbench2/benchmark_results"
# Evaluation-grid token in the filename; 240x121 = the 1.5-deg official grid.
WB2_RESOLUTION = "240x121"
# Verification year embedded in the filename.
WB2_YEAR = 2020
# Ground-truth token in the filename (WB2 verifies against ERA5).
_WB2_GROUND_TRUTH = "era5"
# Filename template: {model}_vs_era5_{resolution}_{year}.nc
_WB2_FILENAME_TEMPLATE = "{model}_vs_{truth}_{resolution}_{year}.nc"

WB2_REGION = "global"          # the WB2 'region' coordinate slice we score against
# RMSE is pre-stored as a data variable named 'rmse.<variable>' (NO metric coord).
_RMSE_METRIC_PREFIX = "rmse"
# lead_time in these files is an INTEGER-HOURS coordinate (6,12,...,240).
_LEAD_COORD = "lead_time"


# --- reference model -> (bucket model token, CSV display name) ---
# Key    = bucket model token used in the filename ({model}_vs_era5_...).
# Value  = clean display name written to the CSV 'model' column (and shown in the
#          scorecard legend). A dict maps bucket token -> display name.
MODEL_DISPLAY_NAMES = {
    "graphcast": "GraphCast",
    "gencast": "GenCast",
    "hres": "IFS-HRES",
    "pangu": "Pangu",
    "neuralgcm_hres": "NeuralGCM",
    "keisler": "Keisler",
    "fuxi": "FuXi",
    "climatology": "ERA5-Climatology",
}

# Default model set to fetch (order preserved in the CSV). These are the
# bucket-token keys; the CSV carries the display names above.
DEFAULT_MODELS = tuple(MODEL_DISPLAY_NAMES)


class HeadlineVar(NamedTuple):
    """One WB2 upper-air headline field to extract.

    wb2_var : WB2 long data-variable name (the '<var>' in 'rmse.<var>').
    level_hpa : pressure level [hPa] to select on the 'level' coordinate.
    csv_variable : WB2 long name written to the CSV 'variable' column
        (load_sota_headline + FIELD_KEY_TO_SOTA key on this long name).
    csv_level : hPa integer written to the CSV 'level' column.
    scorecard_field_key : the plotter field_key this row overlays (documentation
        only; the overlay match is on (csv_variable, csv_level)).
    """
    wb2_var: str
    level_hpa: int
    csv_variable: str
    csv_level: int
    scorecard_field_key: str


# WB2 long-name/level -> CSV (long-name, hPa). UPPER-AIR ONLY — surface fields
# (t2m/mslp/10m wind) are absent from these files and deliberately dropped. Each
# (csv_variable, csv_level) matches a plot_wb_scorecard.FIELD_KEY_TO_SOTA value so
# the overlay is drawn, not silently skipped (t500 is an extra, harmless row).
HEADLINE_VARS = (
    HeadlineVar("geopotential", 500, "geopotential", 500, "z500"),
    HeadlineVar("temperature", 850, "temperature", 850, "t850"),
    HeadlineVar("temperature", 500, "temperature", 500, "t500"),
    HeadlineVar("u_component_of_wind", 850, "u_component_of_wind", 850, "u850"),
    HeadlineVar("u_component_of_wind", 700, "u_component_of_wind", 700, "u700"),
    HeadlineVar("u_component_of_wind", 500, "u_component_of_wind", 500, "u500"),
    HeadlineVar("v_component_of_wind", 850, "v_component_of_wind", 850, "v850"),
    HeadlineVar("v_component_of_wind", 700, "v_component_of_wind", 700, "v700"),
    HeadlineVar("v_component_of_wind", 500, "v_component_of_wind", 500, "v500"),
    HeadlineVar("specific_humidity", 700, "specific_humidity", 700, "q700"),
)

# WB2 publishes leads on a 6 h grid to 15 days; the headline set is 1/3/5/10 d.
DEFAULT_LEADS_HOURS = (24, 72, 120, 240)


class FetchConfig(NamedTuple):
    source: str
    resolution: str
    year: int
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
    """Comma-separated model tokens; each MUST be a known reference model.

    An unknown token is a hard SystemExit (no silent skip / default) so a typo
    can never silently drop a model from the scorecard.
    """
    if not s:
        return DEFAULT_MODELS
    models = tuple(m for m in str(s).split(",") if m != "")
    unknown = [m for m in models if m not in MODEL_DISPLAY_NAMES]
    if unknown:
        raise SystemExit(
            f"--models: unknown reference model(s) {unknown}; "
            f"known models are {sorted(MODEL_DISPLAY_NAMES)}")
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
        help="Directory holding the WB2 benchmark-results files (a gs:// URL or "
             f"a local dir; files are FLAT here). Default: {WB2_RESULTS_BUCKET}")
    p.add_argument(
        "--resolution", default=WB2_RESOLUTION, dest="resolution",
        help="Evaluation-grid token in the filename "
             f"(default {WB2_RESOLUTION!r} = the 1.5-deg official grid).")
    p.add_argument(
        "--year", type=int, default=WB2_YEAR,
        help=f"Verification year in the filename (default {WB2_YEAR}).")
    p.add_argument(
        "--models", default="",
        help="Comma-separated reference model tokens to fetch (default: all). "
             f"Choices: {','.join(MODEL_DISPLAY_NAMES)}.")
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
        resolution=a.resolution,
        year=int(a.year),
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


def result_filename(model, resolution, year):
    """WB2 result filename for one model: ``{model}_vs_era5_{resolution}_{year}.nc``.

    ``model`` is the bucket token (not the display name).
    """
    return _WB2_FILENAME_TEMPLATE.format(
        model=model, truth=_WB2_GROUND_TRUTH, resolution=resolution, year=year)


def _open_result(source, filename):
    """Open one model's WB2 result NetCDF as an in-memory xarray dataset.

    The files are NetCDF-3 CLASSIC (h5netcdf fails), so we read the raw bytes and
    open them through xarray's ``scipy`` engine on a BytesIO buffer. Raises
    FileNotFoundError / OSError if the file cannot be reached — the caller decides
    whether a single unreachable model file aborts the whole fetch.
    """
    import io

    import xarray as xr

    root = str(source).rstrip("/")
    if root.startswith("gs://"):
        import gcsfs

        # Strip the gs:// scheme for gcsfs.cat (it takes bucket/path form).
        gs_path = f"{root[len('gs://'):]}/{filename}"
        fs = gcsfs.GCSFileSystem(token="anon")
        if not fs.exists(gs_path):
            raise FileNotFoundError(
                f"WB2 result not found in bucket: gs://{gs_path} "
                "(no fabrication — skipping this model rather than guessing).")
        raw = fs.cat(gs_path)
        return xr.open_dataset(io.BytesIO(raw), engine="scipy")
    local = Path(root) / filename
    if not local.exists():
        raise FileNotFoundError(f"WB2 result not found: {local}")
    with local.open("rb") as fh:
        raw = fh.read()
    return xr.open_dataset(io.BytesIO(raw), engine="scipy")


def fetch_records(cfg):
    """Read WB2 published RMSE for the configured models/vars/leads.

    Returns a list of ``{model, variable, level, lead_hours, rmse}`` dicts with
    the WB2 LONG variable name + hPa level in the 'variable'/'level' columns (so
    load_sota_headline + the plotter's FIELD_KEY_TO_SOTA line up). Heavy imports
    (xarray/gcsfs/io) are deferred to the call sites, so this only runs off the
    login node.

    Missing-data policy (never fabricate):
      * a single (var, level, lead) absent from an otherwise-reachable model file
        -> skip-with-warning (a genuine gap in the source);
      * a model FILE entirely unreachable -> warn and continue to the next model;
      * hard-fail (FileNotFoundError, no fabrication) only if NO records at all
        were collected (every model file unreachable / empty).
    """
    import logging

    log = logging.getLogger("fetch_wb2_sota")

    records = []
    wanted_leads = sorted(int(x) for x in cfg.leads_hours)
    n_unreachable = 0
    for model in cfg.models:
        if model not in MODEL_DISPLAY_NAMES:               # dispatch hardening
            raise ValueError(f"unknown reference model {model!r}")
        display = MODEL_DISPLAY_NAMES[model]
        fname = result_filename(model, cfg.resolution, cfg.year)
        try:
            ds = _open_result(cfg.source, fname)
        except (FileNotFoundError, OSError) as exc:
            n_unreachable += 1
            log.warning("model file unreachable, skipping %s: %s", model, exc)
            continue

        # Region slice (global), if the dataset carries a region coordinate.
        if "region" in ds.coords or "region" in ds.dims:
            ds = ds.sel(region=WB2_REGION)

        if _LEAD_COORD not in ds.coords and _LEAD_COORD not in ds.dims:
            log.warning(
                "model %s: no %r coordinate; skipping model", model, _LEAD_COORD)
            continue
        available_leads = set(int(v) for v in ds[_LEAD_COORD].values.tolist())

        for hv in HEADLINE_VARS:
            var_name = f"{_RMSE_METRIC_PREFIX}.{hv.wb2_var}"  # e.g. rmse.geopotential
            if var_name not in ds:
                log.warning(
                    "%s: %r absent; skipping (%s@%d)",
                    model, var_name, hv.wb2_var, hv.level_hpa)
                continue
            da = ds[var_name]
            if "level" not in da.coords and "level" not in da.dims:
                log.warning(
                    "%s: %r has no 'level' coord; skipping (%s@%d)",
                    model, var_name, hv.wb2_var, hv.level_hpa)
                continue
            level_values = set(int(v) for v in da["level"].values.tolist())
            if hv.level_hpa not in level_values:
                log.warning(
                    "%s: level %d hPa absent for %s; skipping",
                    model, hv.level_hpa, hv.wb2_var)
                continue
            da_lev = da.sel(level=hv.level_hpa)

            for lead in wanted_leads:
                if lead not in available_leads:
                    log.warning(
                        "%s/%s@%d: lead %dh absent; skipping",
                        model, hv.wb2_var, hv.level_hpa, lead)
                    continue
                rmse_val = float(da_lev.sel(**{_LEAD_COORD: lead}).item())
                if not math.isfinite(rmse_val) or rmse_val <= 0.0:
                    log.warning(
                        "%s/%s@%d lead %dh: non-finite/non-positive RMSE %r; "
                        "skipping (no fabrication)",
                        model, hv.wb2_var, hv.level_hpa, lead, rmse_val)
                    continue
                records.append({
                    "model": display,
                    "variable": hv.csv_variable,
                    "level": hv.csv_level,
                    "lead_hours": lead,
                    "rmse": rmse_val,
                })

    if not records:
        raise FileNotFoundError(
            "fetched zero WB2 records — every requested model file was "
            f"unreachable or empty ({n_unreachable}/{len(cfg.models)} files "
            "unreachable); refusing to write an empty CSV (no fabrication).")
    return records


def main(argv=None):
    """Fetch WB2 SOTA headline RMSE and write the baselines CSV."""
    cfg = build_fetch_config_from_args(argv)

    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("fetch_wb2_sota")
    log.info(
        "fetching WB2 SOTA headline RMSE: source=%s res=%s year=%d models=%s "
        "leads=%s", cfg.source, cfg.resolution, cfg.year, list(cfg.models),
        list(cfg.leads_hours))

    records = fetch_records(cfg)          # raises loudly if NOTHING reachable
    n = write_sota_csv(records, cfg.out)  # de-dups + validates before writing
    log.info("wrote %d rows -> %s", n, cfg.out)
    return cfg.out


if __name__ == "__main__":
    main()
