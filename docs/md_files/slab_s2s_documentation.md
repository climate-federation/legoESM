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

Legacy `legoesm.ml.sfno_s2s` imports and the old top-level slab scripts were removed in this refactor. Downstream callers should use the `legoesm.ml.s2s.*` packages and `scripts/s2s/*` entry points directly.

Canonical results roots now live under:

- `results/ml/s2s/sfno_slab`
- `results/ml/s2s/neuralgcm_slab`

## Shared Utilities

`src/legoesm/ml/s2s/paths.py` defines the canonical slab S2S results roots used by both model families.

`src/legoesm/ml/s2s/plotting.py` contains shared map-plot and GIF frame helpers used by both SFNO and NeuralGCM postprocessing. The shared utilities intentionally cover the common plotting surface while leaving model-specific metric assembly and center-comparison logic inside each model package.

The emulator wrappers used by the dycore-coupling path are also kept compatible with the current driver-facing `PhysicsOutput` contract and with both legacy and `conv_prog`-extended `step_unified(...)` tail layouts. That compatibility layer is plumbing-only: it preserves existing traditional tendencies and zero-fills emulator-only gaps rather than changing the slab workflow semantics.

## SFNO Slab Workflow

The SFNO slab implementation remains the training and ensemble-rollout path built around ChaosBench-style daily subseasonal forecasting. The canonical CLI is `scripts/s2s/sfno_slab.py` and the canonical package is `legoesm.ml.s2s.sfno_slab`.

Typical outputs per case or stage live under `results/ml/s2s/sfno_slab/...` and include checkpoints, rollout member outputs, campaign metrics, center comparisons, snapshot figures, and daily GIFs.

The staged checkpoint family used by the slab-coupled campaign workflow lives under:

- `results/ml/s2s/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/stage1_rollout1`
- `results/ml/s2s/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/stage2_rollout2`
- `results/ml/s2s/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/stage3_rollout3`
- `results/ml/s2s/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/stage3_rollout5`
- `results/ml/s2s/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/stage3_rollout7`

The staged SFNO recipe currently used by those checkpoints is:

- training years `1979-2021`
- atmospheric targets `z,q,t,u,v` on the default 10 ChaosBench pressure levels
- ocean/auxiliary forcing `sosstsst` plus `land_sea_mask`
- ocean source `arco_sst`
- target grid `gaussian_n_max=79`
- one-day lead time with autoregressive training horizon matched to each stage's `n_steps`
- SFNO architecture `embed_dim=32`, `n_blocks=4`, `mlp_expansion=4`
- residual prediction enabled and tendency prediction disabled
- stochastic AFCRPS training with `ensemble_members=4`, `noise_channels=1`, `noise_lat=8`, `noise_lon=16`, `use_time_signal=true`, `afcrps_alpha=0.95`
- batch size `2`, weight decay `1e-5`, validation batches `512`

The warm-start chain is explicit in the checkpoint metadata:

- `stage1_rollout1`: trained from scratch with `train_rollout_steps=1`, `n_steps=1`, `total_steps=20000`, `lr=5e-4`, `warmup_steps=500`
- `stage2_rollout2`: warm-started from `stage1_rollout1/.../best.eqx` with `train_rollout_steps=2`, `n_steps=2`, `total_steps=4000`, `lr=5e-5`, `warmup_steps=200`
- `stage3_rollout3`: warm-started from `stage2_rollout2/.../best.eqx` with `train_rollout_steps=3`, `n_steps=3`, `total_steps=1000`, `lr=1e-5`, `warmup_steps=100`
- `stage3_rollout5`: warm-started from `stage3_rollout3/.../best.eqx` with `train_rollout_steps=5`, `n_steps=5`, `total_steps=1000`, `lr=1e-5`, `warmup_steps=100`
- `stage3_rollout7`: warm-started from `stage3_rollout5/.../best.eqx` with `train_rollout_steps=7`, `n_steps=7`, `total_steps=1000`, `lr=5e-6`, `warmup_steps=100`

For campaign evaluation, the canonical path is:

- generate per-init-date case outputs with `python scripts/s2s/sfno_slab.py ensemble-inference ...`
- refresh case metrics and representative plots with `python scripts/s2s/sfno_slab.py postprocess-ensemble ...`
- aggregate semimonthly campaigns with `python scripts/s2s/run_sfno_campaign.py --aggregate-only ...`
- compare against external centers with `python scripts/s2s/sfno_slab.py postprocess-center-crps ...`

The center-level comparison currently uses the aggregated `case_ensemble_daily_metrics.csv` CRPS series for the coupled and uncoupled SFNO runs, then compares those windows against the ChaosBench center baselines fetched from the LEAP/ChaosBench Hugging Face CSVs for `ECMWF`, `UKMO`, `NCEP`, and `CMA`.

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

## Refresh Workflow

Existing campaign artifacts can be refreshed under the latest slab workflow without rerunning the expensive coupled forecasts themselves.

For NeuralGCM:

- rerun case-level metrics and representative plots with `python scripts/s2s/neuralgcm_slab.py postprocess-case ...`
- rebuild campaign aggregates with `python scripts/s2s/neuralgcm_slab.py postprocess-campaign ...`
- rebuild cross-campaign CRPS comparisons with `python scripts/s2s/neuralgcm_slab.py postprocess-center-crps ...`
- regenerate proof-of-coupling figures with `python scripts/s2s/plot_coupling_diagnostics.py ...`

For SFNO:

- rerun case-level metrics and representative plots with `python scripts/s2s/sfno_slab.py postprocess-ensemble ...`
- rebuild campaign aggregates with `python scripts/s2s/run_sfno_campaign.py --aggregate-only ...`
- rebuild cross-campaign CRPS comparisons with `python scripts/s2s/sfno_slab.py postprocess-center-crps ...`

## Current NeuralGCM Status

The NeuralGCM slab path has been rerun through the full 2022 and 2023 semimonthly campaign surface under the shared `legoesm.ml.s2s` workflow. The refreshed campaign outputs live under:

- `results/ml/s2s/neuralgcm_slab/campaign_2022_days1_15_rollout42_fixedsst`
- `results/ml/s2s/neuralgcm_slab/campaign_2023_days1_15_rollout42_fixedsst`
- `results/ml/s2s/neuralgcm_slab/campaign_center_compare_2022_2023_rollout42_fixedsst`
- `results/ml/s2s/neuralgcm_slab/diagnostics`

The refreshed campaign-level metrics and center-comparison CSVs reproduce the prior artifacts exactly, and representative proof-of-coupling diagnostics are retained under `results/ml/s2s/neuralgcm_slab/diagnostics`.

## Current SFNO Status

The SFNO slab path has been rerun through the full 2022 and 2023 semimonthly case-postprocessing surface using the staged `stage3_rollout7` checkpoint family and the shared `legoesm.ml.s2s.sfno_slab` entry points. The refreshed outputs live under:

- `results/ml/s2s/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/campaign_2022_days1_15_rollout7_fixedsst`
- `results/ml/s2s/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/campaign_2023_days1_15_rollout7_fixedsst`
- `results/ml/s2s/sfno_slab/staged_afcrps_arco_lsm_state_full_1979_2021/campaign_center_compare_2022_2023_rollout7_fixedsst`

The refreshed SFNO campaign artifacts reproduce the prior results up to small floating-point roundoff introduced by the current postprocessing environment, with campaign-level deltas remaining at negligible tolerance.

## Recommended Entry Points

- Training or rollout work for SFNO: `python scripts/s2s/sfno_slab.py ...`
- SFNO campaign aggregation from existing case directories: `python scripts/s2s/run_sfno_campaign.py --aggregate-only ...`
- Interactive NeuralGCM slab case run: `python scripts/s2s/neuralgcm_slab.py ensemble-inference ...`
- NeuralGCM campaign aggregation from existing case directories: `python scripts/s2s/neuralgcm_slab.py postprocess-campaign ...`
- Sequential NeuralGCM campaign run: `python scripts/s2s/run_neuralgcm_campaign.py ...`
- Slurm NeuralGCM campaign submission: `python scripts/s2s/submit_neuralgcm_campaign.py ...`
