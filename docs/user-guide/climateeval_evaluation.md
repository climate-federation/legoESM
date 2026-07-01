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
    --evaluation-suite Tier2_atmosphere_monthly \
    --days 365 --output /scratch/you/amip_run
```

`--evaluate` **requires** `--cmip-output` (ClimateEval reads the CMOR
`Amon/` output tree) and a valid `climateeval_python` + `data_root_dir` —
if either points nowhere, `ExperimentConfig.validate_strict()` raises
**before the model starts**, not after an 11-hour run silently produces
no report.

On successful completion, the driver writes
`<output_dir>/climateeval_<suite>.ddb` (the metrics database) and
`<output_dir>/climateeval_<suite>_report.html` (a portable report). A
failure *during* the ClimateEval subprocess itself (e.g. missing
reference-data files for a specific variable) is logged as a warning and
does **not** fail the AMIP run — only the upfront path validation is
fail-fast.

## Suites

`--evaluation-suite` names one of ClimateEval's own bundled suites (e.g.
`Tier1_sanity_checks`, `Tier2_atmosphere_monthly`) rather than a
free-form variable/reference pick-list — this reuses ClimateEval's
existing suite abstraction instead of re-deriving a second one. The
runner script forces every variable in the chosen suite to reference
ERA5 specifically (stripping any other-model/other-reference entries),
since the point of `--evaluate` here is "how far off is legoESM from
ERA5," not a multi-model intercomparison.

## Standalone use

The runner script doubles as a standalone CLI for scoring an *existing*
CMOR output tree without re-running the model:

```bash
$LEGOESM_CLIMATEEVAL_PYTHON scripts/validate/run_amip_climateeval.py \
    --cmor-dir /scratch/you/amip_run/cmor/Amon \
    --suite Tier2_atmosphere_monthly \
    --data-root-dir "$LEGOESM_CLIMATEEVAL_DATA_ROOT" \
    --output-db /scratch/you/amip_run/climateeval_Tier2_atmosphere_monthly.ddb \
    --output-html /scratch/you/amip_run/climateeval_Tier2_atmosphere_monthly_report.html
```
