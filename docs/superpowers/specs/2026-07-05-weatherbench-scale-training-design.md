# WeatherBench Scale Training (Derecho / Levante) — Design

**Date:** 2026-07-05
**Branch:** `wb-forecast-campaign` (WeatherBench demonstration home; already on main via PR #788)
**Author:** Pierre Gentine (with Claude)

## 1. Goal

A single training script, launchable on **Derecho (PBS)** or **Levante (SLURM)**, that trains legoESM on ERA5 at scale — **multi-node, data-parallel, 0.7° lat-lon** — to be competitive on WeatherBench-2. Supports **all three training modes** (physics-param tuning, hybrid NeuralGCM, SFNO emulator). Evaluated by the WB2 scorer already built (`evaluations/wb_forecast.py`, merged in #788).

## 2. Honest competitiveness framing

The winnable lane is **NeuralGCM-tier**: differentiable physics + learned ML corrections at 0.7–1.4°, trained on ERA5 with a multi-step forecast objective, competitive with GraphCast on deterministic skill to ~5 days. Pure-ML SFNO is available (highest capacity, least physics guarantee); physics-param-only is the low-capacity Ginsburg-sweep lane. 0.7° raises the skill ceiling over 1.4° at ~4× cost/step; it still fits one A100-80GB, so **data-parallel (throughput), not model-parallel (memory)** is the scaling axis.

## 3. Decisions (user-confirmed)

| # | Decision | Choice |
|---|----------|--------|
| Machines | Which clusters | **Both** Derecho (PBS, 4× A100/node, Slingshot-11) + Levante (SLURM, 4× A100-80GB/node, IB-HDR200) |
| Modes | Training modes | **All three**, `--mode {physics,neural_gcm,sfno}` (dispatch to the existing `train_physics_params` / `train_neural_gcm` / `train_sfno` builders) |
| Resolution | Grid | **0.7° lat-lon** (~512×256), fits per-A100-80GB |
| Scaling axis | Parallelism | **Data-parallel** (replicate model, shard ERA5 batch, `pmean` grads); model-SPMD deferred (spatial `shard_map` exists but training-through-it is unwired) |
| Loss | Objective | **Multi-step forecast** (RMSE + grid-CRPS + bias); lat-lon uses truncated-BPTT (adjoint NaNs past ~6 h), spectral would be full-BPTT |

## 4. Architecture

### 4.1 Data-parallel training core — `packages/ml/legoesm/training/data_parallel.py` (NEW, the substance)

The one capability `ml/` lacks (scout-confirmed: no `pmap`/`pmean`/`Mesh` in any training code). It wraps the existing per-sample `_train_step` (`training_driver.py:111`) so N ranks (1 GPU each) train the SAME replicated model on DIFFERENT ERA5 shards, averaging gradients each step.

- `shard_samples(samples, process_id, num_processes) -> local_samples` — deterministic contiguous split of the ERA5 IC/target/forcing lists across ranks (drop-remainder so every rank has an equal count → collectives stay balanced).
- `all_reduce_grad_mean(grad_pytree) -> grad_pytree` — cross-rank gradient average. Implementation: a global 1-D `Mesh(jax.devices(), "data")` (in multi-process JAX `jax.devices()` spans all ranks' devices) + `jax.lax.pmean(grad, "data")` inside a `shard_map`; falls back to identity when `num_processes == 1`. This is the AD-safe collective idiom the repo already uses spatially (`radiation/mc3d/parallel.py:91`), applied to the gradient pytree over the data axis.
- `data_parallel_train_step(train_step, ...)` — run the local `_train_step` to get `(loss, grad)`, `all_reduce_grad_mean(grad)`, then `optimizer.update` + `eqx.apply_updates`. Because every rank applies the SAME averaged gradient, replicas stay bit-identical without broadcasting weights.
- `data_parallel_training_loop(...)` — mirrors `_training_loop` (`training_driver.py:177`) but iterates each rank's local shard, calls `data_parallel_train_step`, logs from rank 0, checkpoints from rank 0. Optional `optax.MultiSteps` gradient accumulation for a larger effective batch (`--grad-accum K` → effective batch = `N_ranks × K`).

Reuses verbatim: `create_optimizer` (`ml/training.py:72`), the mode-specific model/physics builders, the loss (`losses.py`/`neural_gcm_spectral` loss components), `era5_to_latlon_carry`, `build_segment_fn`.

### 4.2 Entry — `scripts/run/train_weatherbench_scale.py` (NEW)

1. **Line 1 of work:** `from legoesm.parallel.early_init import maybe_init_jax_distributed; maybe_init_jax_distributed()` — BEFORE any `jax.numpy` import (repo #693 XLA-ordering guard). This arms `jax.distributed` from the scheduler env (`SLURM_NTASKS`/`PMI_SIZE`/`OMPI_COMM_WORLD_SIZE`, coordinator via mpi4py allgather).
2. Parse args: `--mode`, `--resolution` (default 0.7), `--epochs`, `--multi-step-hours`, `--era5-zarr`, `--train-years`, `--eval-years`, `--optimizer`, `--lr`, `--grad-accum`, `--out`, `--resume`, `--eval-wb2` (run the WB2 scorer after training).
3. Build the config (0.7° lat-lon grid, sigma, dt sized for 512×256 CFL, ERA5 windows), the mode-specific model, the optimizer.
4. Load ERA5 IC/target/forcing over the training window; `shard_samples` per rank.
5. Run `data_parallel_training_loop`; per-epoch resumable checkpoints (rank 0).
6. If `--eval-wb2`: rank 0 runs `evaluations.wb_orchestrator.run_wb_forecast_eval` on the held-out window → WB2 scorecard.

Heavy imports (jax, legoesm) deferred inside `main()` so `--help`/arg-parse is import-light; `build_scale_config_from_args` is unit-testable without JAX.

### 4.3 Launchers (clone existing templates)

- `scripts/cluster/derecho/train_wb.pbs` — from `scaling_derecho/scaling_gpu.sh`: `#PBS -A P08010000 -q main -l select=<N>:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB`, `module load conda cuda cray-mpich craype-accel-nvidia80`, `MPICH_GPU_SUPPORT_ENABLED=1`, `MPI4JAX_USE_CUDA_MPI=0` (Slingshot GPU-direct broken → host-staged; fine for grad-allreduce), launch via `mpiexec -n <N*4> ... python scripts/run/train_weatherbench_scale.py ...`, PALS pin `CUDA_VISIBLE_DEVICES=$PALS_LOCAL_RANKID`.
- `scripts/cluster/levante/train_wb.slurm` — from `scaling_levante/gpu_moist_scaling.slurm`: `--account=bd1083_gpu --partition=gpu --nodes=<N> --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --exclusive`, source `scaling_levante/_env.sh` (UCX CUDA-aware fabric), `srun --gpu-bind=single:1 --cpu-bind=cores python scripts/run/train_weatherbench_scale.py ...`.
- Both self-chain for checkpoint/restart (pattern: `amip_ginsburg_production_chain.sbatch`) so a walltime-capped job resubmits with `--resume`.

### 4.4 Config — `config/wb/scale/train_07deg.yaml` (NEW)

0.7° lat-lon grid, `nlev`, `dt` (CFL-sized), multi-year ERA5 windows (e.g. 2015–2019 train, 2020 eval = WB2 test year), multi-step forecast loss block (RMSE + grid-CRPS + bias; spectral-CRPS off by default — the T63 compile lesson), optimizer (AdamW/MUON warmup-cosine). Per-mode overlays for `neural_gcm` (ML head dims) / `sfno` (embed/blocks).

## 5. Data flow

scheduler launches `N_nodes × 4` ranks (1/GPU) → each `maybe_init_jax_distributed()` → each loads its ERA5 shard → per-rank `(loss, grad)` → `pmean` grad over the data axis → synced update (replicas identical) → rank-0 checkpoint + log → (optional) rank-0 WB2 eval.

## 6. Testing

- **Unit (the critical one):** `all_reduce_grad_mean` + `data_parallel_train_step` on **2 CPU devices** (`--xla_force_host_platform_device_count=2`): a batch of 2 samples split 1-per-device must produce the SAME averaged gradient + parameter update as a serial 2-sample mean. This proves data-parallel correctness without a cluster (the repo's SPMD tests use exactly this CPU-multi-device trick).
- **Unit:** `shard_samples` (balanced, drop-remainder, deterministic); `build_scale_config_from_args` round-trip (arg-parse, no JAX).
- **Smoke (single-GPU):** the entry with `--mode physics --smoke` (T21/1-epoch) on Ginsburg confirms the pipeline end-to-end before a multi-node launch.
- **Integration:** the multi-node run IS the Derecho/Levante launch (2-node smoke first, then scale).

## 7. Reuse / no-duplication ledger

- Multi-node init: `parallel/early_init.py` + `parallel/distributed.py` — reused, not re-derived.
- Training step / loop / modes: `training_driver.py` (`_train_step`, `_training_loop`, `train_physics_params`/`train_neural_gcm`/`train_sfno`) — reused; data-parallel WRAPS them.
- Optimizer: `create_optimizer` — reused.
- ERA5 / rollout / loss: `era5_to_state`, `dycore_rollout`, `losses.py` — reused.
- Eval: `evaluations/` WB2 scorer (#788) — reused.
- Launcher scaffolding: `scaling_{derecho,levante}/*` templates + `_env.sh` — cloned, bench driver swapped for the train entry.
- Every new `.py` gets a direct unit test.

## 8. Risks & mitigations

- **Cross-rank grad `pmean` correctness** (main risk) → the 2-CPU-device equivalence unit test gates it before any GPU-hours.
- **0.7° lat-lon stability** (CFL at 512×256) → `dt` sized conservatively + a single-GPU smoke before multi-node.
- **lat-lon adjoint NaN past 6 h** → truncated-BPTT (existing `training_driver` behavior); keep the unroll short.
- **Levante untested / Derecho Slingshot GPU-direct broken** → host-staged fabric flags (`MPI4JAX_USE_CUDA_MPI=0`), 2-node smoke before scale; data-parallel grad-allreduce tolerates host-staged.
- **Competitiveness is not guaranteed by scale alone** → the WB2 scorecard is the honest measuring stick; the deliverable is a *capable, correct, launchable* scale trainer, not a promised leaderboard rank.

## 9. Out of scope (YAGNI)

- Model-parallel SPMD training (spatial `shard_map` through the gradient) — deferred; 0.7° fits per-GPU.
- 0.25° (needs model-SPMD + huge compute).
- Ensemble/CRPS-beyond-the-training-term, and a machine-detect→scheduler dispatcher (the two launchers hardcode their env, matching the existing `_env.sh` pattern).
