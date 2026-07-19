# Unified WB+AIMIP Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One shared campaign core driving WB + AIMIP training with SOTA curriculum/EMA/constraints, 3-cluster launchers, WB2 + AIMIP benchmarks.

**Architecture:** Extract curriculum/EMA into `packages/ml/legoesm/training/` leaf modules; add `campaign_driver.py` consumed by both entry scripts; loss presets as YAML referenced by suites; benchmark battery reuses `evaluations/` orchestrator.

**Tech Stack:** JAX/Equinox/optax, existing `scale_build`/`losses`/`data_parallel`, SLURM/PBS.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-19-unified-wb-aimip-training-design.md`
- Classical mode radiation pinned `rrtmgp` — validated, hard error otherwise.
- Every new `.py` gets a direct unit test (CLAUDE.md). Dispatch raises on unknown.
- No new constants/saturation re-impl; constants from `legoesm.constants`.
- Login node: tests run via `sbatch`/`srun` only; syntax checks (<5s) OK locally.
- Minimal diffs; existing sbatch fleet must keep working (entry scripts stay, become shells).
- Codex adversarial review before declaring each PR done.

---

### Task 1: `training/curriculum.py` (D2)

**Files:**
- Create: `packages/ml/legoesm/training/curriculum.py`
- Test: `tests/unit/test_curriculum.py`

**Interfaces:**
- Produces: `CurriculumStage(NamedTuple)` fields `rollout_hours: float`, `n_epochs: int`, `lr_scale: float = 1.0`, `pushforward_no_grad_steps: int = 0`, `loss_overrides: tuple[tuple[str, float], ...] = ()`.
- `Curriculum` with `.stages: tuple[CurriculumStage, ...]`, `.total_epochs`, `.stage_for_epoch(epoch:int) -> tuple[int, CurriculumStage]` (raises `ValueError` past end), `.stage_boundaries() -> list[int]` (first epoch of each stage, for optimizer restarts).
- `curriculum_from_config(spec) -> Curriculum | None`: accepts legacy `((hours, n_epochs), ...)` tuples (current `aimip_rollout_curriculum` format) AND dict form `{"stages": [{"rollout_hours":.., "n_epochs":.., "lr_scale":.., ...}]}`; `None`/empty → None. Non-monotonic `rollout_hours` → `ValueError` unless `allow_non_monotonic=True`.

**Steps:**
- [ ] Read `neural_gcm_spectral.py` inline phase logic (grep `rollout_curriculum` / `epoch_plan`) to match legacy semantics exactly.
- [ ] Write failing tests: legacy-tuple parse, dict parse, stage_for_epoch boundaries (epoch 0, last of stage, first of next), total_epochs, non-monotonic raises, empty → None, lr_scale default 1.0.
- [ ] Implement module.
- [ ] Run `tests/unit/test_curriculum.py` via srun/sbatch; PASS.
- [ ] Commit.

### Task 2: Wire curriculum into both loops (D2)

**Files:**
- Modify: `packages/ml/legoesm/training/neural_gcm_spectral.py` (replace inline phase logic with `Curriculum`).
- Modify: `scripts/run/train_weatherbench_scale.py` + `packages/ml/legoesm/training/scale_build.py::rollout_hours` (`--curriculum` / YAML `curriculum:` drives `multi_step_hours` ladder per stage; per-stage LR restart via `lr_scale` and fresh optimizer state at boundaries).
- Test: extend `tests/unit/test_curriculum.py` + CLI round-trip in existing WB CLI test (or new `tests/unit/test_train_wb_scale_cli.py`).

**Interfaces:**
- Consumes: Task 1 API. Behavior-preserving for existing suites (legacy tuples parse to identical epoch plan; assert in test).

**Steps:** read call sites → failing test (epoch plan equality vs legacy for T106 config tuple `((12,2),(24,2),(72,2),(120,1))`) → refactor → tests pass → commit.

### Task 3: `training/ema.py` (D3)

**Files:**
- Create: `packages/ml/legoesm/training/ema.py`
- Modify: training loops (`data_parallel.py` epoch hook or `training_driver.py`) to update EMA each step and save `epoch_NNNN_ema.eqx` beside raw; eval scripts (`run_weatherbench_eval.py`, `run_aimip_amip_inference.py`) load EMA when present, `--no-ema` escape.
- Test: `tests/unit/test_ema.py`

**Interfaces:**
- Produces: `init_ema(model) -> ema_model`, `ema_update(ema_model, model, decay: float) -> ema_model` (pure, inexact-array leaves only via `eqx.partition`), default `EMA_DECAY = 0.9999` (module constant, provenance comment: U-Cast/GenCast convention).

**Steps:** failing tests (identity at decay→ arbitrary: ema stays if model==ema; exact convex-combination math; static leaves untouched) → implement → wire loops + eval → tests pass → commit.

### Task 4: Loss presets (D5)

**Files:**
- Create: `config/wb/loss_presets/ace2.yaml`, `config/wb/loss_presets/neuralgcm.yaml`
- Create: `packages/ml/legoesm/training/loss_presets.py` (`load_loss_preset(name_or_path) -> dict`, merged under suite `loss:` with suite keys winning; unknown preset → `ValueError` listing available)
- Test: `tests/unit/test_loss_presets.py`

**Interfaces:**
- ace2.yaml: `residual_normalize: true`, `multi_step_hours: [6,12]`, per-var weights (w_T/w_u/w_v/w_q/w_ps flat 0.5/0.5/0.5/0.5/0.3), flux weights ×2-equivalents, doc header citing arXiv:2411.11268 table.
- Consumed by Task 6 campaign_driver (`loss_preset:` suite key).

**Steps:** failing test (load, merge precedence, unknown raises) → implement + YAMLs → pass → commit.

### Task 5: ACE2 constraint flag (D4)

**Files:**
- Modify: wherever T106 v2 "budget constraints (dry-mass fixer, q≥0 clip)" flags live (grep `fix_mass`/`dry_mass`/`budget` in `scale_build.py`, `run_aimip.py`, suite YAMLs) — unify under suite key `constraints: none|ace2`.
- Add global multiplicative column-moisture correction if missing (reuse `global_sum_mpi`-based fixer; smooth clamps).
- Test: `tests/unit/test_training_constraints.py` — budget residual ≈ 0 on synthetic column; sign-convention walk per CLAUDE.md.

**Steps:** read existing fixers first (pre-impl search mandatory) → extend only what's missing → failing budget test → implement → pass → commit.

### Task 6: `campaign_driver.py` + thin shells (D1)

**Files:**
- Create: `packages/ml/legoesm/training/campaign_driver.py`
- Modify: `scripts/run/run_weatherbench_campaign.py`, `scripts/run/run_aimip.py` → parse args, call `run_campaign`.
- Test: `tests/unit/test_campaign_driver.py`

**Interfaces:**
- `CampaignSuite` (parsed YAML): `mode(s)`, `training_core` ∈ {spectral, latlon, cubed_sphere, mpas}, `curriculum`, `loss`/`loss_preset`, `constraints`, `radiation`, `data`, `optimizer`, `output_dir`.
- `validate_suite(suite)`: classical + radiation≠rrtmgp → `ValueError`; unknown mode/core → `ValueError`; cubed_sphere/mpas → `NotImplementedError("training core reserved; loader exists (era5_to_cubedsphere_carry), segment path not wired")`.
- `run_campaign(suite, modes, stages)`: stages train/eval/aimip_eval/plot; delegates to existing `train_weatherbench_scale.main` / `neural_gcm_spectral.train_*` / eval mains (keep delegation pattern of current campaign script).
- Backward compat: existing WB campaign YAMLs and AIMIP suites parse unchanged (adapter mapping `aimip_*` keys).

**Steps:** read both entry scripts fully → failing tests (validation matrix incl. rrtmgp pin + dispatch raises; suite adapter equivalence on one real WB YAML + one real AIMIP suite) → implement → shells → CLI round-trip tests → pass → lock new raises in `tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS` → commit.

### Task 7: Sweep planner dedup (D6)

**Files:**
- Create: `packages/ml/legoesm/training/sweep_planner.py` (from common body of `run_wb_sweep_stage1.py` + `run_aimip_classical_sweep_stage1.py`: baseline combo, swap axes, `_write_combo`, manifest, sbatch array emission; `radiation="rrtmgp"` enforced at plan time).
- Modify: both scripts → thin wrappers passing their axis tables.
- Test: `tests/unit/test_sweep_planner.py` (manifest reproducibility; both wrappers produce previous outputs modulo paths; rrtmgp violation raises).

**Steps:** diff the two scripts → extract → failing tests → wrappers → pass → commit.

### Task 8: Cluster launchers (D7)

**Files:**
- Create: `scripts/cluster/unified_training/{_env_ginsburg.sh,_env_derecho.sh,_env_levante.sh}` (or source existing site `_env.sh`), `train_ginsburg.sbatch`, `train_derecho.pbs`, `train_levante.slurm`, `smoke_ginsburg.sbatch`, `eval_ginsburg.sbatch`.
- Pattern: source site env → `lego_detect_machine.py` profile → `python scripts/run/run_weatherbench_campaign.py --config $SUITE --modes $MODES --stages train` with chaining (`CHAIN_MAX=12`, resume on `epoch_*.eqx`) copied from `scripts/cluster/aimip_scale/` templates.
- Ginsburg: `--account=glab`, 72 h, glab1/burst partitions.

**Steps:** copy-adapt templates → shellcheck-style read-through → submit `smoke_ginsburg.sbatch` (T21 smoke, all 3 modes through unified path) → verify job completes + checkpoints exist → commit.

### Task 9: AIMIP benchmark battery (D8)

**Files:**
- Create: `scripts/validate/run_aimip_battery.py` (E1: monthly-climatology area-weighted RMS bias vs ERA5 via `ml/loss.py::latitude_weighted_rmse`/`latitude_weighted_bias`; E2: global-mean annual anomaly series + OLS trend, train era + 2015+ holdout separately; JSON out).
- Create: `config/aimip/sota/aimip_phase1_reference.csv` (published Phase-1 model E1/E2 values, arXiv:2605.06944, provenance header).
- Create: `scripts/plot/plot_aimip_battery.py` (time-series panels legoESM variants vs reference models + scorecard table PNG).
- Test: `tests/unit/test_aimip_battery.py` (synthetic dataset: known bias → E1 exact; linear ramp → trend exact; CSV loader schema).

**Steps:** failing tests → implement → pass → `eval_ginsburg.sbatch` on best checkpoints → collect scorecard + PNGs → send → commit.

### Task 10: Review + smoke gates

- [ ] Codex adversarial review loop (`codex exec --sandbox read-only` per memory) on each PR; fix; repeat until clean.
- [ ] `campaign_smoke.sbatch` green through unified path (3 modes) before any scale launch.
- [ ] Update `MEMORY.md` project memory with campaign state.
