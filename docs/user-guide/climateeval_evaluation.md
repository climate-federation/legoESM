# Evaluating AMIP output against ERA5 (ClimateEval)

`run_amip.py --evaluate` can auto-score a completed run's CMOR output
against ERA5 (and other observational references) using
[ClimateEval](https://github.com/climate-federation/ClimateEval). This is
**optional** — legoESM runs and installs fine with no ClimateEval anywhere
on the machine; `--evaluate` only activates the bridge described below.

## ClimateEval is not a legoESM dependency

ClimateEval is a separate project (`climate-federation/ClimateEval`), not
vendored, not a git submodule, and not in `pyproject.toml`. Its stack
(iris + ESMValTool) pulls in a heavy, conda-oriented geo dependency chain
(GDAL/PROJ/cartopy) that conflicts with a pinned JAX/CUDA environment —
that's why it's kept in a fully separate Python environment even on
machines that run both. legoESM never imports `climateeval`; it only
shells out to an external interpreter.

This is a deliberate choice over the two more obvious options:

- **A git submodule** would vendor ClimateEval's *source* into this repo,
  but wouldn't solve the actual problem — you'd still need iris/ESMValTool
  installed in *some* environment to run it, and it would couple two
  projects' git histories for no benefit.
- **A pip extra** (`pip install -e ".[climateeval]"`) would only work if
  ClimateEval's dependencies could share one environment with JAX, which
  is exactly what doesn't hold in practice.

So legoESM's only job is the bridge: a config/CLI knob pointing at
*some* ClimateEval Python interpreter, a `subprocess` call across that
boundary, and a small script
(`scripts/validate/run_amip_climateeval.py`) that speaks ClimateEval's
real API on the other side.

## One-time setup (per user/machine)

There is no shared, canonical ClimateEval install or reference-data
location — set these up once for your own account:

```bash
# 1. Clone ClimateEval and build its own environment (see its own README
#    for the current recommended method — conda/pixi):
git clone https://github.com/climate-federation/ClimateEval.git
cd ClimateEval && pixi install   # or: conda env create -f environment.yml

# 2. Get access to (or build) reference-data files in the
#    {source_id}/{frequency}/{var} layout ClimateEval expects, e.g.:
#    reanalysis_ERA5/mon/tas/..., observation_GPCP/mon/pr/..., etc.
#    On a shared HPC filesystem this is usually another project's
#    directory you're given read access to — ask around before
#    re-downloading everything yourself.

# 3. Point legoESM at both (once, e.g. in your shell profile):
export LEGOESM_CLIMATEEVAL_PYTHON=/path/to/ClimateEval/.pixi/envs/default/bin/python
export LEGOESM_CLIMATEEVAL_DATA_ROOT=/path/to/climateeval_reference_data
```

Both env vars can instead be passed per-run as
`--evaluation-climateeval-python` / `--evaluation-data-root-dir` — the
env vars are just the default source for those flags.

## Running it

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip.py \
    --config config/amip/amip_production.yaml \
    --cmip-output --evaluate \
    --days 365 --output /scratch/you/amip_run
```

`--evaluate` **requires** `--cmip-output` (ClimateEval reads the CMOR
`Amon/` output tree) and a valid `climateeval_python` + `data_root_dir` —
if either points nowhere, `ExperimentConfig.validate_strict()` raises
**before the model starts**, not after an 11-hour run silently produces
no report.

On successful completion, the driver writes one
`<output_dir>/climateeval_<suite>.ddb` per suite and a single combined
`<output_dir>/climateeval_report.html` report over all of them. A failure
*during* the ClimateEval subprocess itself (e.g. missing reference-data
files for a specific variable) is logged as a warning and does **not**
fail the AMIP run — only the upfront path validation is fail-fast.

## Suites

**By default `--evaluate` runs ALL of ClimateEval's bundled suites** —
every tier — and renders them into a **single combined report**. A suite
whose reference or model data is missing/inapplicable (e.g. the ECS suite
needs abrupt-4xCO2 + piControl, the sub-daily suite needs sub-daily
output) is **skipped and reported**, not fatal, so the report always
contains whatever could actually be scored. Which suites score depends on
what the run emits:

- **`Tier1_sanity_checks`** — global-mean values vs literature ranges
  (no reference data needed).
- **`Tier1_consistency_checks`** — mass/water conservation (annual; needs
  a multi-year run to be meaningful).
- **`Tier2_atmosphere_monthly`** — spatial skill (Map, ZonalLine,
  ZonalProfile + leaderboard) against each variable's own reference.
- **`Tier2_ocean_monthly` / `Tier2_sea_ice_monthly`** — need `Omon` /
  `SImon` output (the runner loads the full `cmor/` tree so these score).
- **`Tier3_*`** (subdaily / dynamics / ecs) — usually skipped for a short
  AMIP run (need sub-daily output, long integrations, or extra experiments).

### Which reference each variable is scored against

Each variable is scored against **its own suite-designated reference**
when that reference is staged in `--data-root-dir`, and falls back to ERA5
only when it is not (`--reference-policy designated`, the default). So
`rsut`/`rlut`/`rtnt`/`swcre`/`lwcre` score against CERES-EBAF,
`hfls`/`hfss` against MERRA2, `pr` against GPCP and `tas` against HadCRUT5
if those are present; `clivi`/`clt`/`lwp` fall back to ERA5 when
ESACCI-CLOUD is not staged. A variable that neither its designated
reference nor ERA5 provides (e.g. `clwvi`) is reported as having no
reference and skipped, rather than being scored against something that
does not contain the quantity. The run prints its per-variable reference
decisions, so the log says exactly which reference produced which number.

This replaced an earlier blanket rewrite that forced **every** reference
to ERA5. Because the staged ERA5 has no `rsut`, `rlut`, `hfls`, `hfss` or
`clwvi`, that rewrite silently deleted every TOA-radiation and
surface-flux metric the suites would otherwise have produced (8 of 19
variables x 4 diagnostics in `Tier2_atmosphere_monthly` alone). Pass
`--reference-policy era5-only` to reproduce the old ERA5-only scorecard;
note that `pr` and `tas` change reference between the two policies, so
those numbers are **not** comparable across them.

`other_data` (CMIP6 multi-model intercomparison) entries are still
stripped by default — the point of `--evaluate` is model-vs-observations —
but unlike the reference rewrite this discards no observational metric.
Pass `--keep-other-data` to keep them. Restrict to one suite with e.g.
`--evaluation-suite Tier2_atmosphere_monthly`.

### Output frequency and suite selection

Model cubes are loaded per **frequency group**: ClimateEval selects a
model cube by `var_name` alone, with no frequency filter, so the runner
loads only the CMOR tables matching the frequency each diagnostic asks for
(`Amon`/`Omon`/... for `mon`, `day`/`CFday`/... for `day`) plus the fixed
fields. A suite mixing frequencies — `Tier3_dynamics` has a daily
Hovmoller and a monthly QBO — is run as two groups into the same `.ddb`.
If the run wrote no output at a requested frequency (e.g. no `1hr` tables
for `Tier3_atmosphere_subdaily`), those diagnostics are skipped **and
reported** rather than being silently served monthly means.

`climateeval check` runs before each benchmark group: it validates
presence, coordinates, units, timerange coverage and levels from metadata
only, so a reference-window mismatch (`Tier3_dynamics` pins Hovmoller to
`20000101/20021231` and QBO to `19900101/20091231`) is reported before the
expensive reference load rather than after it.

### Groups that cannot produce a metric are skipped

If that check finds **no usable model variable at all** for a group, no
metric is possible, so the group is skipped and reported instead of run.
On an atmosphere-only AMIP tree this skips `Tier2_ocean_monthly` (0/6
required variables: amoc, chl, mlotst, phcint, so, tos) and
`Tier2_sea_ice_monthly` (0/1: siconc).

This is a large saving, measured not estimated: the same evaluation of the
same run took **4 h 25 m** before and **25 m** after (10.4x), because
`Tier2_ocean_monthly` alone spent ~4 h loading ORAS5 and area-weight-
regridding its irregular grid for three `mlotst` variables the model never
wrote. A controlled re-run confirmed **zero metric rows lost** (75 before,
75 after) and every non-skipped suite's row counts unchanged.

The decision is taken from **model** data only. It never consults the
reference-availability probe, so a reference that probe cannot see —
`ORAS5` and `RAPID` read from outside `--data-root-dir` — can never cause
a scoreable group to be dropped. An empty check result (the check itself
failed, or the suite needs no model input) never skips: the rule fails
open on ignorance.

What the default costs is reference-only rows: the observed RAPID AMOC
series and the ORAS5 mixed-layer-depth annual cycle, curves with no model
to compare against (33 rows here). Pass `--run-unscoreable-groups` to keep
them, at the ~4 h price.

### Short runs and per-variable metric failures

A single-year (or otherwise short) run legitimately cannot satisfy every
diagnostic in a suite: `Tier2_atmosphere_monthly`'s `AnnualMeanTimeSeries`
diagnostic reduces both the model cube and the (multi-decade) ERA5
reference cube to one point per year, then requires the two resulting
series to be the *same length* to compute `weighted_rmse` — a 1-year run
produces a 1-point series against ERA5's multi-decade series, which is a
real, expected failure for that diagnostic+variable, not a bug.

By default the runner does **not** let that one failure take down the
whole suite: `--fail-on-metric-error` (default off, same convention as
`--fail-on-missing-data`) makes ClimateEval log the failing variable as a
warning and keep scoring every other diagnostic in the suite (`Map`,
`ZonalLine`, `ZonalProfile`, …). Passing `--fail-on-metric-error` restores
ClimateEval's own default (`fail_on_metric_error=True`), which raises on
the first such failure — since the runner's per-suite `try`/`except`
around `Suite.get_database()` then catches that exception, the entire
suite gets dropped from the report, including diagnostics that would have
scored fine. Prefer the default unless you specifically want a suite to be
all-or-nothing.

## Standalone use

The runner script doubles as a standalone CLI for scoring an *existing*
CMOR output tree without re-running the model. Point `--cmor-dir` at the
`cmor/` root (so ocean/sea-ice tables load); with no `--suite` it runs all
bundled suites and writes the per-suite `.ddb`s plus one combined
`climateeval_report.html` into `--output-dir`:

```bash
$LEGOESM_CLIMATEEVAL_PYTHON scripts/validate/run_amip_climateeval.py \
    --cmor-dir /scratch/you/amip_run/cmor \
    --data-root-dir "$LEGOESM_CLIMATEEVAL_DATA_ROOT" \
    --output-dir /scratch/you/amip_run
```

## CMOR output sampling semantics (what the monthly means actually are)

Diagnostics are collected once per `diag_days` at the segment boundary
(a fixed UTC time of day). What each CMOR field is made of:

- **True time means** (accumulated every model step inside the compiled
  segment, immune to the diagnostic cadence): `pr`, `hfss`, `hfls`,
  `rsdt`, `rsut`, `rlut`, and `tas` (segment-mean lowest-level T with the
  instantaneous MOST 2 m stability offset). Before this fix the radiation
  fields and `tas` were fixed-UTC snapshots — January `rsdt` had the
  night hemisphere at exactly 0 and a ~1400 W/m² noon peak, so per-pixel
  monthly radiation maps carried a full diurnal alias. Zonal and global
  means were unaffected (longitude sampling averages local time), which
  is why the defect passed budget checks.
- **Snapshot samples at the diagnostic cadence** (documented limitation):
  `psl`, `prw`, `ta`, `ua`, `va`, `hus`, `clt`, `clivi`, `clwvi`, `tos`,
  `siconc`, and the clear-sky fluxes. At `diag_days=5` a monthly mean is
  ~6 fixed-UTC samples — no first-order diurnal alias for most of these,
  but expect weather-sampling noise (and mild alias for the cloud
  fields). Accumulating the 3-D fields per step would grow the scan carry
  by `nlev`× per field and is deferred until a use case needs it.
- **The CMIP `day` table degenerates whenever `diag_days > 1`**: each
  written "day" is really one sample every `diag_days` days and
  `tasmin`/`tasmax` are not daily extremes. The driver logs a warning;
  set `diag_days=1` if you need day-table output.
