# Collapse the two WeatherBench/AIMIP training loops into one

Date: 2026-08-11 · Status: PROPOSED (not started) · Follows:
`docs/superpowers/plans/2026-07-19-unified-wb-aimip-training.md`

**Goal.** One epoch loop behind both campaigns, so a WeatherBench run and an
AIMIP run differ only in their suite YAML — not in which trainer executes.

The 2026-07-19 plan unified everything *around* the loop (config parsing,
curriculum, EMA, loss presets, ERA5 ingestion, optimizer, eval, scorecard) and
explicitly stopped short of the loop itself: `campaign_driver.run_campaign`
"delegates to existing `train_weatherbench_scale.main` / `neural_gcm_spectral.
train_*`". This plan finishes that last step.

## Verified current state (2026-08-11)

Training loops that exist today:

| # | Loop | Drives | Curriculum | EMA | Multi-node data-parallel |
|---|---|---|---|---|---|
| 1 | `scripts/run/train_weatherbench_scale.py::main` (~100 lines) | WB `physics`/`neural_gcm`/`sfno` | no (one fixed horizon) | no | yes, via `mpi_data_parallel_training_loop` |
| 2 | `neural_gcm_spectral._train_spectral_loop:2727` | AIMIP `classical`/`column_nn`/`sfno_physics` | yes | yes | yes, hand-rolled on `shard_samples` + `all_reduce_grad_mean` |
| 3 | `neural_gcm_spectral._train_sfno_full_loop:4273` | AIMIP `sfno_full` | yes (`build_sfno_curriculum_epoch_plan:4205`) | yes | yes |
| 4 | `training/training_driver.py::train_physics_params` / `train_neural_gcm` | lat-lon C-grid + MPAS carry stack (`run_aimip_latlon.py`) | n/a | n/a | n/a |

Already single-source and NOT in scope to change: `scale_build.
build_mode_components` (mode builders), `scale_build.load_era5_samples`
(shard-at-build + host-resident, #1286), `training/losses.py`,
`ml/training.create_optimizer`, `training/curriculum.py`, `training/ema.py`,
`training/loss_presets.py`, `training/data_parallel.py`, `spectral_rollout`,
`evaluations/wb_orchestrator.py`, `plot_wb_scorecard.py`.

**Loop 1 owns no capability the others lack.** Its multi-node data-parallel
training is not unique — loop 2 shards and all-reduces through the same
`data_parallel.py` primitives (`neural_gcm_spectral.py:2664-2694`). Its ERA5
loader is a shared module that stays. What it uniquely has is a *simpler*
contract: one fixed rollout horizon, no curriculum, no EMA.

Two further seams, both visible in `campaign_driver.py:29-30`:

- Two names for the same three things: `WB_MODES = (physics, neural_gcm, sfno)`
  vs `AIMIP_VARIANTS = (classical, column_nn, sfno_physics, sfno_full)`.
- Two `SpectralPEConfig` builders — `scale_build._spectral_pe_config` and
  `run_aimip._build_spectral_config` — which drifted: the WB one could not
  reach the dry-mass anchor at all until 2026-08-11. This is the concrete cost
  of the split, and the reason this plan exists.

## The requirement (user, 2026-08-11)

> One unified framework for `sfno`, `column_nn` and `classical`, usable for
> either AIMIP or WeatherBench. What changes between the two campaigns is the
> **training strategy and the loss** — never the model.

Measured against that requirement, the models are ALREADY single-source on the
spectral core (`scale_build._build_mode_components_spectral:296-405` builds its
physics through AIMIP's own `make_physics_params_spectral_physics`,
`make_column_mlp_spectral_physics`, `make_sfno_spectral_physics`, and
`legoesm.ml.sfno.SFNO`). What has drifted is the *wiring around* them:

| Variant | Same code? | Same configuration? |
|---|---|---|
| classical | **NO — two different model types share the name** (below) | n/a |
| column_nn | yes — `NeuralPhysics`, 256 wide x 4 deep both sides | key name differs: WB `neural_gcm.nn_hidden` vs AIMIP `nn_hidden_dim` |
| sfno (physics) | yes — `SFNO`, zero-init decoder both sides | **no**: WB defaults embed 256 / 8 blocks, AIMIP 128 / 4, and the keys sit at different YAML depths |
| sfno_full | AIMIP only | WB cannot express it at all |

So a WB `sfno` run and an AIMIP `sfno_physics` run are the same architecture at
different sizes, silently, unless both YAMLs spell it out.

**`classical` is worse than drifted defaults — it is two model types** (codex
round-2, 2026-08-11). `scale_build`'s `physics` mode and
`neural_gcm_spectral.train_physics_params_spectral` train
`TrainablePhysicsParams` (`model_registry.py` variant `classical`), while the
AIMIP classical campaign trains `AIMIPClassicalParams`
(`run_aimip.py::_train_aimip_classical`), optionally wrapped in an
`AIMIPTrainableBundle` of spec-driven scheme parameters. Their checkpoints are
not interchangeable. Deciding which one `classical` means — or giving the two
distinct names — is a PREREQUISITE for the loop collapse and is a question for
the user, not a refactor to perform silently. That is the defect
the requirement names, and it is fixed by one model registry (Task 1), not by
touching any model code.

## Decision

**Keep loops 2+3, delete loop 1.** Loop 2's feature set is a strict superset
(curriculum through the dycore, EMA + validation-selected checkpoints, CRPS
stage, prescribed-SST forcing, four variants) and it is the loop under active
campaign use. A WB run becomes an AIMIP suite with a one-stage curriculum.

**Non-goals.** (a) Loop 4 stays: the carry-based lat-lon/MPAS stack is a
different state representation, not a duplicate — folding it is a separate
question. (b) `scale_build`'s builders stay: `run_weatherbench_eval.py` needs
them to rebuild the params skeleton for checkpoints trained by loop 1, and they
are the shared mode builders. Only the *loop* dies.

## Tasks

### Task 1 — One model registry (the requirement's core)

- Create `packages/ml/legoesm/training/model_registry.py`:
  `build_variant(variant, yml, grid, sigma, dt) -> (params, physics_fn_factory)`
  for `classical | column_nn | sfno_physics | sfno_full`, reading ONE key
  namespace with ONE set of defaults. Unknown variant -> `ValueError`.
- Both `scale_build._build_mode_components_spectral` and `run_aimip.
  _build_spectral_config`'s model construction call it. Neither keeps a private
  copy of an architecture default.
- `sfno_full` becomes expressible from a WB config for the first time.
- Test `tests/unit/test_model_registry.py`: for each variant, the pytree built
  from a WB YAML and from the equivalent AIMIP suite are structurally identical
  (`jax.tree_util.tree_structure` equal, every leaf shape equal). This is the
  mechanical form of "the model does not change between campaigns", and it goes
  red the moment a default drifts again.
- Migration note: the current WB `sfno` default (256/8) differs from AIMIP's
  (128/4). Pick one, state which in the commit, and treat every pre-existing WB
  sfno number as belonging to the OTHER size — do not compare across the change.

### Task 2 — WB suite adapter

- Create `packages/ml/legoesm/training/wb_suite_adapter.py`:
  `wb_yaml_to_suite(yml, cfg) -> dict` mapping a WB scale YAML
  (`config/wb/scale/train_07deg.yaml`) + `ScaleConfig` fields onto the AIMIP
  suite keys `run_aimip._build_spectral_config` already consumes.
- Mode -> variant: `physics->classical`, `neural_gcm->column_nn`,
  `sfno->sfno_physics`. Unknown -> `ValueError` (dispatch hardening); lock in
  `tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS`.
- `multi_step_hours` -> a one-stage curriculum `[[hours, n_epochs]]` via
  `curriculum.parse_curriculum`.
- Test `tests/unit/test_wb_suite_adapter.py`: the real `train_07deg.yaml` maps
  to a suite whose resolved `SpectralPEConfig`, `dt`, `n_max`, `n_levels` and
  `LossConfig` equal what `build_mode_components` resolves today, field by
  field. This test is the specification.

### Task 3 — Equivalence gate (do this BEFORE deleting anything)

- `scripts/validate/compare_training_loops.py`: run the same smoke config
  through loop 1 and through loop 2 (via the Task 2 adapter), same seed, same
  rank count, and compare the per-epoch loss sequence.
- Accept only if losses agree to <=1e-6 relative on epoch 0 and the trajectory
  shape matches; ANY larger gap is a finding to explain, not a tolerance to
  widen. Record the numbers in the commit.
- Expected honest outcome: they will NOT match bit-for-bit (different sample
  ordering and optimizer-state init). Then the gate is: same config trains to
  the same loss within run-to-run noise, measured across >=2 seeds.

### Task 4 — Route `run_weatherbench_campaign` through the AIMIP loop

- `run_weatherbench_campaign.py:304` currently imports
  `train_weatherbench_scale as trainer`. Replace with the adapter + the AIMIP
  trainer entry (`neural_gcm_spectral.train_*` through `run_aimip`'s
  `_train_variant`).
- Keep the WB CLI surface intact: the cluster scripts
  (`scripts/cluster/{levante/train_wb.slurm,derecho/train_wb.pbs}`) must run
  unchanged. If a flag has no AIMIP equivalent, add the suite key rather than
  dropping the flag.
- Tests: extend `tests/unit/test_train_wb_scale_cli.py` (argv round-trip still
  valid) and add a campaign-level test asserting the trainer that runs is the
  unified one.

### Task 5 — Delete loop 1

- Remove `scripts/run/train_weatherbench_scale.py`.
- Remove/retarget its tests: `tests/ml/test_train_weatherbench_scale.py`,
  `tests/unit/test_train_wb_scale_cli.py`, and the two that only borrow it as
  an import anchor (`tests/ml/test_scale_build_dt.py`,
  `tests/ml/test_training_segment_damping.py` -> point at `scale_build`).
- Per CLAUDE.md removal rule: also drop stale references in
  `docs/wb/scale_training_runbook.md` and the two 2026-07 plan/spec docs
  (mark superseded, do not rewrite history).
- `_spectral_pe_config` in `scale_build.py` keeps serving the eval skeleton;
  add a comment naming `run_aimip._build_spectral_config` as the authority so
  the two cannot drift again — or, better, have it call that builder.

### Task 6 — One vocabulary

- Collapse `WB_MODES` into `AIMIP_VARIANTS` in `campaign_driver.py` with a
  deprecation map (`physics->classical`, ...) that accepts the old names for
  one release and warns.
- Update `config/wb/campaign/*.yaml` and `config/wb/scale/*.yaml` to the
  variant names. `config/wb/sweep/stage1/*` already uses the AIMIP suite
  schema — no change.

### Task 7 — Review + smoke

- Codex adversarial review each PR (`codex exec --sandbox read-only`).
- One T21-scale smoke through the unified path for all four variants before any
  scale launch.

## Risks

1. **Silent numerics change for WB runs.** Loop 2 applies EMA and per-stage LR
   restarts that loop 1 never had. A WB config must map to a *one-stage*
   curriculum with EMA off unless the suite asks for it, or every historical WB
   number becomes incomparable. Task 2's field-by-field test is the guard.
2. **Old checkpoints.** Anything under `results/wb_*` was written by loop 1's
   skeleton. Keep `run_weatherbench_eval.py` + `scale_build` builders working;
   add a test that loads one existing WB checkpoint after the change.
3. **Scope creep into loop 4.** Explicitly out of scope. If the lat-lon carry
   trainer looks foldable, that is a separate plan.
4. **Two AIMIP loops remain (2 and 3).** This plan does not merge `sfno_full`
   into `_train_spectral_loop`; they differ in whether a dycore is in the graph.
   Name it as remaining debt rather than pretending "one loop" once loop 1 dies.

## Definition of done

`grep -rn "train_weatherbench_scale" scripts/ tests/` returns nothing; a WB
config and an AIMIP suite both reach the same loop; the Task 3 equivalence
numbers are in the commit message; the four-variant smoke is green.
