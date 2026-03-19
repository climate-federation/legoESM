# SFNO Slab S2S Pipeline

## Scope
This document describes the active subseasonal SFNO slab-ocean pipeline in this repo and only this pipeline.

It excludes:
- `neuralgcm_s2s`
- `ace2_s2s`
- older one-step SFNO experiments outside `src/legoesm/ml/sfno_s2s`

The canonical entrypoint is:
- [scripts/sfno_slab.py](/burg-archive/glab/users/jn2808/legoESM/scripts/sfno_slab.py)

The canonical package is:
- [src/legoesm/ml/sfno_s2s](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s)

## External References
Data and benchmark context used by this pipeline:

- ChaosBench website: `https://leap-stc.github.io/ChaosBench/`
- ChaosBench dataset card: `https://huggingface.co/datasets/LEAP/ChaosBench`
- ChaosBench paper: `https://arxiv.org/abs/2402.00712`
- ARCO ERA5 repository and bucket documentation:
  - `https://github.com/google-research/arco-era5`
  - `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- AIFS-CRPS paper used as a staging reference, not as an implementation dependency:
  - `https://www.nature.com/articles/s44387-026-00073-7`

## What The Pipeline Does
The pipeline trains a stochastic daily SFNO on ChaosBench-style atmosphere targets with SST supplied as an input-only forcing channel. At inference time it supports:

- uncoupled rollout with fixed initial SST
- slab-coupled rollout with SST updated by the simple ocean model
- ensemble evaluation with multiple stochastic members from one checkpoint
- campaign aggregation over many initialization dates
- center-level CRPS comparison against ChaosBench operational baselines

The current recommended configuration is:

- atmosphere targets: `z,q,t,u,v`
- pressure levels: `10,50,100,200,300,500,700,850,925,1000`
- ocean forcing: `sosstsst`
- extra forcing: `land_sea_mask`
- Gaussian grid truncation: `n_max=79`
- model: `embed_dim=32`, `n_blocks=4`, `mlp_expansion=4`
- stochastic conditioning: `noise_channels=1`, `noise_lat=8`, `noise_lon=16`, `time_signal=true`
- optimizer batch size: `2`

That means:

- `50` predicted atmosphere channels
- `2` input-only forcing channels in the recommended run
- one stochastic checkpoint, sampled into multiple inference members later

## Code Layout
Core package:

- [config.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/config.py)
- [data.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/data.py)
- [regrid.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/regrid.py)
- [training.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/training.py)
- [rollout.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/rollout.py)
- [coupling.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/coupling.py)
- [preparation.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/preparation.py)
- [evaluation.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/evaluation.py)
- [postprocess.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/postprocess.py)
- [cli.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/cli.py)

Operational wrappers:

- [scripts/run_sfno_campaign.py](/burg-archive/glab/users/jn2808/legoESM/scripts/run_sfno_campaign.py)

## Data Flow
### Training targets and forcings
The training loader uses the common daily date intersection across:

- `era5` for atmospheric targets
- `lra5` when land forcing variables are requested
- `oras5` or the ARCO SST cache when ocean forcing variables are requested

For the active configuration:

- atmosphere target source: `ChaosBench/era5`
- SST forcing source: ARCO SST cache prepared once into `data/arco_sst_cache/...`
- land mask source: derived from SST coverage and added as `land_sea_mask`

### Per-sample construction
For one sample index and `lead_time=1`:

1. input date is `d`
2. target dates are `d+1 ... d+n_steps`
3. the model input at `d` is:
   - normalized atmosphere at `d`
   - normalized forcing at `d`
4. the training target sequence is:
   - normalized atmosphere on future dates only
5. the future forcing sequence is:
   - normalized forcing on the same future dates

Returned tensor shapes are:

- input: `(n_lat, n_lon, n_atmos + n_forcing)`
- target: `(n_steps, n_lat, n_lon, n_atmos)`
- forcing: `(n_steps, n_lat, n_lon, n_forcing)`

### Regridding
ChaosBench files arrive on a regular lat-lon grid. The loader regrids them onto the SFNO Gaussian grid produced by `n_max=79`.

That target grid is:

- `120 x 240`
- triangular spectral truncation up to `T79`

This is handled in [regrid.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/regrid.py).

### Normalization
Normalization is channel-wise and uses the ChaosBench climatology bundle plus the ARCO SST cache statistics:

- atmosphere stats: `climatology_era5.zarr`
- land stats: `climatology_lra5.zarr`
- SST stats: ARCO SST stats cache

Ocean SST is converted into the same Celsius-like forcing units expected by ChaosBench before normalization. This matters because ARCO SST is stored in Kelvin while the training forcing convention behaves like `degC`.

## ARCO Preparation
The pipeline has two ARCO preparation paths:

1. SST cache for training
2. daily surface forcing for slab-coupled inference

### SST cache
[prepare_arco_sst_cache](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/preparation.py) reads the public ARCO ERA5 store, interpolates daily SST to the ChaosBench ORAS5 reference grid, preserves the ocean mask, converts SST units into training forcing units, and writes:

- `sosstsst`
- `ocean_mask`
- scalar mean and sigma for normalization

Current tracked defaults:

- `data/arco_sst_cache/arco_sst_daily_19790101_20231231.zarr`
- `data/arco_sst_cache/arco_sst_daily_19790101_20231231_stats.zarr`

### Coupled-inference surface forcing
[prepare_arco_surface_forcing](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/preparation.py) builds the slab-ocean surface contract on the Gaussian grid:

- `surface_pressure`
- `sea_surface_temperature`
- `sea_ice_cover`
- `sw_down`
- `lw_down`
- `initial_sea_surface_temperature`
- `initial_sea_ice_cover`

This is the surface file later consumed by `ensemble-inference --surface-source arco`.

## Training
### Model input
At each training step the SFNO input is:

- current atmosphere
- current forcing
- smooth stochastic noise channels
- optional normalized lead-time channel

The model predicts atmosphere only.

### Loss and rollout contract
Training uses:

- stochastic autoregressive rollout
- area-weighted almost-fair CRPS
- sampled ensemble members from one checkpoint

The implementation is in:

- [loss.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/loss.py)
- [training.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/training.py)

The rollout contract is:

1. start from the current atmosphere plus current forcing
2. predict next-day atmosphere
3. append the teacher-forced next-day forcing
4. continue autoregressively

### Current staged schedule
The active full run that replaced the earlier short diagnostics is:

1. rollout `1`: `20000` steps, `lr=5e-4`, `warmup=500`
2. rollout `2`: `4000` steps, `lr=5e-5`, `warmup=200`
3. rollout `3`: `1000` steps, `lr=1e-5`, `warmup=100`
4. rollout `5`: `1000` steps, `lr=1e-5`, `warmup=100`
5. rollout `7`: `1000` steps, `lr=5e-6`, `warmup=100`

This stage schedule is inspired by the staged rollout idea used in the AIFS-CRPS paper, but the actual step counts and resolutions here are local SFNO choices.

### Training entrypoint
The canonical training entrypoint is:

```bash
python scripts/sfno_slab.py train \
  --output-dir results/sfno_slab/<run_root>/stage1_rollout1 \
  --stage-name stage1_rollout1 \
  --n-steps 1 \
  --train-rollout-steps 1 \
  --total-steps 20000
```

Cluster-specific shell launchers are intentionally not part of the documented
tracked pipeline.

## Inference And Evaluation
### Single-case inference
The main user-facing entrypoint is:

```bash
python scripts/sfno_slab.py ensemble-inference \
  --checkpoint <best.eqx> \
  --sample-date YYYYMMDD \
  --n-steps 42 \
  --n-members 5 \
  --surface-source arco \
  --output-dir <case_dir>
```

This generates:

- member-level coupled rollouts
- member-level uncoupled rollouts
- representative plots and GIFs
- daily and windowed metrics

### Coupled vs uncoupled semantics
Current semantics are:

- coupled:
  - slab ocean updates SST daily
  - updated SST is fed back through the forcing channel
- uncoupled:
  - SST is fixed to the initial-condition ocean state for the full rollout
  - the input state remains finite on land by filling land points before normalization
  - saved SST diagnostics are still ocean-masked

This fixed-SST uncoupled branch is implemented in [coupling.py](/burg-archive/glab/users/jn2808/legoESM/src/legoesm/ml/sfno_s2s/coupling.py).

### Metrics
Per-case postprocessing writes:

- `daily_ensemble_metrics.csv`
- `window_ensemble_metrics.csv`
- `*_metric_summary.png`
- representative member snapshot grids
- daily GIFs

Supported metrics:

- `RMSE`
- `MAE`
- `CRPS`

The center-level comparison uses campaign-level daily CRPS only.

### Campaign evaluation
The canonical campaign wrapper is [run_sfno_campaign.py](/burg-archive/glab/users/jn2808/legoESM/scripts/run_sfno_campaign.py).

It does:

1. generate init dates, usually days `1` and `15`
2. drop invalid late-year dates that cannot support a full `42`-day horizon
3. run one `ensemble-inference` case per valid init date
4. collect all case metrics into campaign CSVs
5. generate campaign boxplots

The campaign entrypoint is:

```bash
python scripts/run_sfno_campaign.py \
  --checkpoint <best.eqx> \
  --output-dir <campaign_dir> \
  --year 2022 \
  --days 1,15
```

Current campaign settings:

- years: `2022`, `2023`
- init dates: days `1` and `15`
- horizon: `42` days
- stochastic members: `5`

### Center comparison
After both yearly campaigns finish, the center-level comparison is run with:

- or directly through `scripts/sfno_slab.py postprocess-center-crps`

It compares campaign CRPS against the ChaosBench center baselines:

- `ECMWF`
- `UKMO`
- `NCEP`
- `CMA`

Fields:

- `t-850`
- `z-500`
- `q-700`

## Current Canonical Result Root
The current training root for the working SFNO branch is:

- `results/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021`

The current fixed-SST campaign reruns are:

- `results/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/campaign_2022_days1_15_rollout7_fixedsst_v2`
- `results/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/campaign_2023_days1_15_rollout7_fixedsst_v2`
- `results/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/campaign_center_compare_2022_2023_rollout7_fixedsst_v2`

## Things Deliberately Not In The Pipeline
These are not part of the canonical SFNO slab path anymore:

- short smoke-scale staged runs used only for debugging
- the tendency-prediction branch
- standalone SFNO plotting scripts that duplicate integrated postprocessing
- dependencies from the SFNO campaign path into `neuralgcm_s2s`

## Maintenance Notes
When changing the coupled or uncoupled SST path, treat this as a regression risk:

- raw SST fields have land NaNs
- forcing tensors given to the model must remain finite everywhere
- ocean-only masking belongs in saved diagnostics, not in the model input state

This is covered by [tests/unit/test_sfno_s2s.py](/burg-archive/glab/users/jn2808/legoESM/tests/unit/test_sfno_s2s.py).
