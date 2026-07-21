# Unified WeatherBench + AIMIP Training Driver — Design

Date: 2026-07-19 · Branch: `wb-sfno-full-stabilize` · Status: approved-by-standing-autonomy (user directive 2026-07-19)

## Goal

One shared training implementation driving both the WeatherBench forecast campaign and the
AIMIP intercomparison campaign, with SOTA long-rollout/curriculum training strategies
(ACE2, ArchesWeather, U-Cast, GraphCast, NeuralGCM, Stormer, AIMIP Phase 1), portable
across Ginsburg (SLURM), Derecho (PBS), and Levante (SLURM), covering three training
modes, and benchmarked against WeatherBench 2/X and AIMIP Phase-1 models.

## Current state (verified 2026-07-19)

Core numerics are **already single-source**:

| Layer | Module | Used by |
|---|---|---|
| Mode builders (physics / neural_gcm=column NN / sfno) | `packages/ml/legoesm/training/scale_build.py::build_mode_components` | WB + AIMIP |
| ERA5 ingestion + sampling | `training/era5_to_state.py`, `scale_build.load_era5_samples`, `_training_sample_indices` | WB + AIMIP |
| Loss | `training/losses.py::LossConfig/combined_loss` (multi-step, residual-norm, flux supervision, CRPS, bias) | WB + AIMIP |
| MPI data-parallel loop | `training/data_parallel.py` | WB + AIMIP |
| SFNO arch + packing | `ml/sfno.py`, `ml/channel_packing.py::PE3DChannelSpec` | WB + AIMIP |
| SFNO↔dycore coupling | `training/sfno_dycore_coupling.py::SFNOPhysics/make_sfno_step_unified` | WB + AIMIP |
| Optimizer (MUON/muon_partitioned, warmup+cosine+clip) | `ml/training.py::create_optimizer` | WB + AIMIP |
| WB2 eval orchestration | `evaluations/wb_orchestrator.py::run_wb_forecast_eval` | `run_weatherbench_eval.py` + `run_aimip_wb2_eval.py` |
| Scorecard plot + SOTA overlay | `scripts/plot/plot_wb_scorecard.py` + `config/wb/sota/wb2_headline_rmse.csv` | both |
| Machine profiles | `scripts/experiment/lego_detect_machine.py` + `config/machines/{default,derecho,derecho_gpu,levante-gpu,...}.yaml` | cluster scripts |

Remaining **divergence** (the actual work):

1. Two CLI/config layers: WB argv-based `ScaleConfig` (`train_weatherbench_scale.py`) vs
   AIMIP suite-YAML (`run_aimip.py`). Same builders underneath, different orchestration.
2. Curriculum split: WB = code (`multi_step_hours` fixed), AIMIP = config
   (`aimip_rollout_curriculum` phases interpreted inside `neural_gcm_spectral.py`).
   No shared curriculum module; no per-stage LR restarts; no rollout-length ladder
   abstraction usable by both cores (spectral + latlon).
3. Cloned sweep planners: `run_wb_sweep_stage1.py` is a copy of
   `run_aimip_classical_sweep_stage1.py`.
4. Missing SOTA pieces: EMA weights, ACE2 per-variable loss-weight preset,
   in-graph ACE2-style conservation corrector as a first-class training flag,
   pushforward (no-grad prefix) option for long rungs.
5. AIMIP scoring: WB2 scorecard exists; AIMIP E1–E5 battery (bias maps, global-mean
   trend time series, ENSO regression, daily variability, +2K/+4K response) only
   partially covered by `summarize_aimip_latlon.py` / fleet plots.

## Literature synthesis → adopted strategies

Distilled recipe common to every winning system: **(1) long cheap 1–2-step pretrain at
high LR → (2) short low-LR autoregressive fine-tune with growing horizon (per-stage LR
warm restart, never one monotone schedule) → (3) optional probabilistic stage → always
evaluate EMA weights → always run physical constraints inside the training graph.**

Per mode:

- **`sfno_full` (ACE2-like)**: 2-step rollout loss suffices *when hard constraints are
  in-graph* (ACE2 arXiv:2411.11268 — dry-air mass via global surface-pressure adjustment,
  column moisture via global multiplicative precip correction + residual advective flux,
  non-negativity clamps; 1000-yr stable). Per-variable weight table (×10 Z500, ×5 T850/
  ULWRFsfc, ×2 radiative fluxes, 0.25 q-mid). embed_dim 384 reference. Then horizon
  ladder 4→8→12 steps at tiny constant LR (GraphCast 3e-7; GDPS shows per-stage restarts
  at larger LR also work). EMA 0.9999 (U-Cast), evaluate EMA weights. MUON already our
  default (U-Cast validates Muon+AdamW-partitioned at 895M scale).
- **`column_nn` / `sfno_physics` (hybrid, NeuralGCM analog)**: rollout curriculum
  *through the differentiable dycore* 6h→12h→24h→72h→120h (NeuralGCM grew 6h→5d);
  conservation comes free from the dycore; sqrt-N checkpointing for long rungs
  (existing `_sqrt_checkpointed_scan`); LR warmup→constant→decay with per-stage
  restart scale.
- **`classical` (param/scheme tuning)**: physics already stable — curriculum variable is
  segment length vs gradient quality: 6–24 h segments early, lengthen for slow feedbacks;
  **radiation pinned to `rrtmgp` in every classical suite** (user requirement); scheme
  swap via existing stage-1 OAT + stage-2 Cartesian sweep, all under identical loss/eval
  protocol (controlled-comparison rule).
- **Benchmark axes are independent** (AIMIP Phase-1 finding: weather RMSE and climate
  trend fidelity uncorrelated) → both gates required: WB2 2020 protocol RMSE/ACC
  scorecard *and* AIMIP E1/E2 (bias + global-mean anomaly time series/trend) minimum.
- AIMIP protocol detail: no CO2 channel allowed; prescribed monthly SST/SIC + insolation
  only (already matches `aimip_amip_forcing.py`).

## Design

### D1. Shared campaign core (new `packages/ml/legoesm/training/campaign_driver.py`)

One suite-YAML-driven `run_campaign(suite, modes, stages)` used by BOTH entry scripts.
`run_weatherbench_campaign.py` and `run_aimip.py` become thin shells (arg parse →
`run_campaign`). WB campaign YAMLs (`config/wb/campaign/*.yaml`) and AIMIP suites
(`config/aimip/**/suite.yaml`) converge on one schema: `mode`, `training_core`
(`spectral` default | `latlon`; `cubed_sphere`/`mpas` reserved enum values, raise
`NotImplementedError` with pointer — loader `era5_to_cubedsphere_carry` exists but the
training path is future work, stated not stubbed), `curriculum`, `loss`, `data`,
`optimizer`, `radiation` (validated: classical mode **requires** `rrtmgp`).
Stages: `train`, `eval` (WB2), `aimip_eval` (E1/E2), `plot`.

### D2. Shared curriculum module (new `training/curriculum.py`)

`CurriculumStage(rollout_hours, n_epochs, lr_scale, loss_overrides)` +
`Curriculum.epoch_plan(epoch) -> stage`. Extracted from `neural_gcm_spectral.py`'s
inline phase logic; consumed by spectral AND latlon loops AND
`train_weatherbench_scale.py`. Adds per-stage LR warm restart (`lr_scale` multiplies
`create_optimizer` peak; optimizer state reset at stage boundary) and drives
`multi_step_hours` (true rollout-length ladder, not just which lead is scored).
Optional `pushforward_no_grad_steps` per stage (Brandstetter): `lax.stop_gradient`
on the first k rollout steps.

### D3. EMA (new `training/ema.py`)

`EmaState(decay=0.9999)`, `ema_update(ema, model)`, checkpoints store raw + EMA;
eval stage loads EMA weights by default (`--no-ema` escape). Unit test: EMA of
constant params is identity; decay math exact.

### D4. ACE2 constraint corrector (flag on segment build)

Reuse existing `fix_mass` / dry-mass fixer / q≥0 machinery (already in T106 v2 budget
constraints) and expose as a single suite flag `constraints: ace2` = {dry-air mass fix,
global multiplicative column-moisture/precip correction, smooth non-negativity clamps}
applied inside the segment before the loss (grads flow via `global_sum_mpi` VJP).
Sign/budget check per CLAUDE.md mandatory gate; budget-residual unit test ≈ 0.

### D5. Loss presets

`config/wb/loss_presets/ace2.yaml` (per-variable table above, residual_normalize,
2-step) and `neuralgcm.yaml` (lat-weighted MSE, level pressure weighting). Suites
reference presets; no numeric duplication (values live once in preset YAML).

### D6. Sweep planner dedup

Factor `run_wb_sweep_stage1.py` + `run_aimip_classical_sweep_stage1.py` onto one
`training/sweep_planner.py` (baseline combo, swap axes, manifest, sbatch array
emission). Both scripts become thin config wrappers. Classical sweeps pin
`radiation: rrtmgp` (validated at plan time, raise on violation).

### D7. Cluster portability

Per-site launcher pair (train + eval) for the unified driver under
`scripts/cluster/unified_training/`: `ginsburg.sbatch`, `derecho.pbs`,
`levante.slurm` — each sources site `_env.sh`, uses `lego_detect_machine.py`
profile, keeps job-chaining/resume (`CHAIN_MAX`) from the aimip_scale templates.
All heavy work via scheduler (login-node policy).

### D8. Benchmark harness

- **WB2/WBX**: existing `run_weatherbench_eval.py` scorecard (RMSE/ACC vs
  persistence/climatology + SOTA CSV overlay, eval-year 2020) for all modes;
  scorecard PNG via `plot_wb_scorecard.py`. WBX alignment = metric definitions
  (lat-weighted RMSE, 00/12z inits) already consistent; note sparse-obs truth is
  out of scope.
- **AIMIP**: new `scripts/validate/run_aimip_battery.py` — E1 area-weighted RMS bias
  (monthly climatology vs ERA5), E2 global-mean annual anomaly time series + linear
  trend (train era vs 2015+ holdout separately), vs published AIMIP Phase-1 model
  values (small committed reference CSV `config/aimip/sota/aimip_phase1_reference.csv`
  from arXiv:2605.06944). E3–E5 documented as follow-up.
- Plot: time-series panel (global-mean T2m/precip anomaly, legoESM variants vs AIMIP
  models) + scorecard.

## Error handling

Unknown mode/core/scheme → `raise ValueError`/`SystemExit` (dispatch-hardening rule,
lock in `test_dispatch_hardening.py`). Classical suite without rrtmgp → hard error at
config validation. Missing checkpoint at eval → hard error listing available epochs.
Curriculum stages with non-monotonic hours → error (explicit override flag to allow).

## Testing

Every new module gets a direct unit test: `test_curriculum.py` (epoch_plan boundaries,
lr_scale, ladder monotonic), `test_ema.py`, `test_campaign_driver.py` (suite parse,
mode dispatch raises on unknown, rrtmgp pin enforced), `test_sweep_planner.py`
(manifest reproducibility, both wrappers same planner), corrector budget-residual
test. CLI round-trip tests for changed drivers. Smoke: `campaign_smoke.sbatch`
(T21/32×64×8, 1 epoch) must pass through unified path for all 3 modes before any
scale launch. Codex adversarial review mandatory (major multi-file change).

## Sub-project order

1. Curriculum + EMA + loss presets + campaign core (D1–D5) — single PR.
2. Sweep dedup (D6) — small PR.
3. Cluster launchers (D7) — small PR + smoke jobs on Ginsburg.
4. Benchmark battery (D8) — PR + eval jobs, scorecards, time series.

## Non-goals

Cubed-sphere/MPAS training cores (enum reserved, loaders exist, not wired — needs
carry/segment support per grid); WBX sparse-obs scoring; flow-matching ensemble stage
(ArchesWeatherGen corrector — documented as future stage 3); E3–E5 AIMIP metrics.
