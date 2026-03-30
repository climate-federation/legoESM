# Slab S2S Documentation

This document describes the shared subseasonal slab-ocean workflow under `legoesm.ml.s2s` and the model-specific implementations for SFNO and NeuralGCM.

## Layout

Source code now lives under:

- `src/legoesm/ml/s2s/sfno_slab`
- `src/legoesm/ml/s2s/neuralgcm_slab`
- `src/legoesm/ml/s2s/paths.py`
- `src/legoesm/ml/s2s/plotting.py`

Shared scripts now live under:

- `scripts/s2s/sfno_slab.py`
- `scripts/s2s/neuralgcm_slab.py`
- `scripts/s2s/run_sfno_campaign.py`
- `scripts/s2s/run_neuralgcm_campaign.py`
- `scripts/s2s/submit_neuralgcm_campaign.py`
- `scripts/s2s/plot_coupling_diagnostics.py`

Canonical results roots now live under:

- `results/ml/s2s/sfno_slab`
- `results/ml/s2s/neuralgcm_slab`

## Shared Utilities

`src/legoesm/ml/s2s/paths.py` defines the canonical slab S2S results roots used by both model families.

`src/legoesm/ml/s2s/plotting.py` contains shared map-plot and GIF frame helpers used by both SFNO and NeuralGCM postprocessing. The shared utilities intentionally cover the common plotting surface while leaving model-specific metric assembly and center-comparison logic inside each model package.

## SFNO Slab Workflow

The SFNO slab implementation remains the training and ensemble-rollout path built around ChaosBench-style daily subseasonal forecasting. The canonical CLI is `scripts/s2s/sfno_slab.py` and the canonical package is `legoesm.ml.s2s.sfno_slab`.

Typical outputs per case or stage live under `results/ml/s2s/sfno_slab/...` and include checkpoints, rollout member outputs, campaign metrics, center comparisons, snapshot figures, and daily GIFs.

## NeuralGCM Slab Workflow

The NeuralGCM slab implementation lives under `legoesm.ml.s2s.neuralgcm_slab` and mirrors the SFNO campaign structure where practical:

- semimonthly case scheduling for 2022 and 2023
- coupled 42-day forecasts with daily slab-updated SST and daily sea ice
- uncoupled 42-day forecasts with fixed initial SST and fixed initial sea ice forcing
- member-level forecast outputs
- case-level metrics and plots
- campaign aggregation
- center-level CRPS comparison

The canonical interactive CLI is `scripts/s2s/neuralgcm_slab.py`. Campaign automation lives in `scripts/s2s/run_neuralgcm_campaign.py` and `scripts/s2s/submit_neuralgcm_campaign.py`.

In this workflow, NeuralGCM consumes SST and sea ice as externally supplied forcing variables. The slab driver rebuilds those forcing fields once per forecast day rather than letting the lower boundary free-run for the full 42-day window. Coupled runs therefore feed back slab-updated SST plus that day's prepared sea ice, while uncoupled runs reapply the day-0 SST and day-0 sea ice every day.

Typical NeuralGCM campaign outputs now follow the same visible structure as SFNO:

- `results/ml/s2s/neuralgcm_slab/campaign_2022_days1_15_rollout42_fixedsst`
- `results/ml/s2s/neuralgcm_slab/campaign_2023_days1_15_rollout42_fixedsst`
- `results/ml/s2s/neuralgcm_slab/campaign_center_compare_2022_2023_rollout42_fixedsst`

Within each `inference_YYYYMMDD` case directory, the user-facing outputs match SFNO:

- `seed_XX/coupled.nc`
- `seed_XX/uncoupled.nc`
- `metrics/daily_ensemble_metrics.csv`
- `metrics/window_ensemble_metrics.csv`
- `plots/*`
- `surface_forcing_YYYYMMDD_42d.nc`

NeuralGCM-specific preparation artifacts are retained under hidden paths such as `_prepared/` and `_member_metadata/` so they do not clutter the visible SFNO-style case surface.

## Campaign Conventions

Both model families use semimonthly initialization dates with the 42-day horizon filter applied at the campaign stage. For 2022 and 2023, this means the last valid initialization date is `November 15`.

Campaign postprocessing is split into:

- case-level forecast evaluation and representative plots
- campaign-level aggregation across valid initialization dates
- center-level CRPS comparison using the aggregated case metrics

## Current NeuralGCM Status

The NeuralGCM slab path has been validated on a single `2022-01-01` case after the namespace refactor and slab-grid alignment fix. The corrected one-case output exists under:

- `results/ml/s2s/neuralgcm_slab` for campaign-scale runs
- single-case interactive validation outputs can also be created under ad hoc directories beneath `results/ml/s2s`

## Recommended Entry Points

- Training or rollout work for SFNO: `python scripts/s2s/sfno_slab.py ...`
- Interactive NeuralGCM slab case run: `python scripts/s2s/neuralgcm_slab.py ensemble-inference ...`
- Sequential NeuralGCM campaign run: `python scripts/s2s/run_neuralgcm_campaign.py ...`
- Slurm NeuralGCM campaign submission: `python scripts/s2s/submit_neuralgcm_campaign.py ...`
