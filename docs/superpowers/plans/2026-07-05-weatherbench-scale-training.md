# WeatherBench Scale Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** A multi-node, data-parallel training script (Derecho PBS + Levante SLURM) that trains legoESM on ERA5 at 0.7° lat-lon in all three modes (physics/neural_gcm/sfno), evaluated by the WB2 scorer.

**Architecture:** Reuse the single-GPU per-sample training loop (`training_driver`) and wrap it in a NEW data-parallel layer: each of N ranks (1 GPU) trains the replicated model on a distinct ERA5 shard, gradients averaged across ranks each step via a JAX collective. `jax.distributed` init + optimizer + ERA5 ingestion + WB2 eval are reused; only the data-parallel wrapper, the entry, the config, and the two launchers are new.

**Tech Stack:** Python, JAX (multi-process via `jax.distributed`), Equinox, optax; PBS (Derecho) / SLURM (Levante).

## Global Constraints

- **Login-node policy (HARD):** all `pytest`/JAX commands run on a compute node via `srun --account=glab --pty` or `sbatch`; login node only for git/grep/edits. Multi-device CPU tests: `env JAX_ENABLE_X64=1 XLA_FLAGS=--xla_force_host_platform_device_count=2 <py> -m pytest ...` under `srun`.
- **Distributed init BEFORE jax.numpy:** the entry's first work is `maybe_init_jax_distributed()` (repo #693 XLA-ordering guard) — no `import jax.numpy` above it.
- **Reuse, do not re-derive:** `parallel/early_init.maybe_init_jax_distributed`, `parallel/distributed.initialize_jax_distributed_multiprocess`, `training_driver` (`_build_train_step`/`_training_loop`/`train_physics_params`/`train_neural_gcm`/`train_sfno`), `ml/training.create_optimizer`, `era5_to_state.era5_to_latlon_carry`, `evaluations.wb_orchestrator.run_wb_forecast_eval`.
- **Constants/thermo from `legoesm.constants`/`legoesm.thermo`.** Every new `.py` gets a direct unit test.
- **Codex adversarial review** before declaring done (`codex exec --sandbox read-only`).
- **Equivalence gate is non-negotiable:** the data-parallel gradient average must match a serial batch-mean bit-closely on 2 CPU devices before any GPU-hours.

## File structure

- `packages/ml/legoesm/training/data_parallel.py` — NEW. `shard_samples`, `pmean_grads`, `data_parallel_value_and_grad`, `data_parallel_training_loop`.
- `tests/ml/test_data_parallel.py` — NEW. shard + grad-average equivalence (2 CPU devices).
- `scripts/run/train_weatherbench_scale.py` — NEW. Entry (dist init → config → mode dispatch → data-parallel loop → checkpoint → optional WB2 eval).
- `tests/unit/test_train_wb_scale_cli.py` — NEW. Arg-parse round-trip (no JAX).
- `config/wb/scale/train_07deg.yaml` — NEW. 0.7° config + per-mode overlays.
- `scripts/cluster/derecho/train_wb.pbs`, `scripts/cluster/levante/train_wb.slurm` — NEW launchers.

---

### Task 1: Deterministic sample sharding

**Files:** Create `packages/ml/legoesm/training/data_parallel.py`; Test `tests/ml/test_data_parallel.py`.

**Produces:** `shard_samples(items, process_id, num_processes, *, drop_remainder=True) -> list` — contiguous equal split; with drop_remainder every rank gets `len(items)//num_processes` items (so collectives stay balanced).

- [ ] **Step 1: failing test**
```python
# tests/ml/test_data_parallel.py
from legoesm.training.data_parallel import shard_samples

def test_shard_samples_balanced_and_disjoint():
    items = list(range(10))
    shards = [shard_samples(items, r, 3) for r in range(3)]
    assert all(len(s) == 3 for s in shards)           # 10//3 = 3, remainder dropped
    flat = [x for s in shards for x in s]
    assert len(set(flat)) == 9 and set(flat) <= set(items)   # disjoint, no dup
    assert shard_samples(items, 0, 1) == items         # single process = identity
```
- [ ] **Step 2:** `srun --account=glab --time=0:10:00 env JAX_ENABLE_X64=1 <py> -m pytest tests/ml/test_data_parallel.py::test_shard_samples_balanced_and_disjoint -v` → FAIL (ImportError).
- [ ] **Step 3: implement**
```python
"""Data-parallel training: replicate the model across ranks, shard the ERA5
batch, average gradients each step. Wraps the single-device training_driver
step. Multi-process via jax.distributed (init in the entry BEFORE jax.numpy)."""
from __future__ import annotations

def shard_samples(items, process_id, num_processes, *, drop_remainder=True):
    """Contiguous equal split of a sample list across ranks."""
    n = len(items)
    if num_processes <= 1:
        return list(items)
    per = n // num_processes if drop_remainder else -(-n // num_processes)
    start = process_id * per
    return list(items[start:start + per])
```
- [ ] **Step 4:** rerun → PASS.
- [ ] **Step 5:** `git add packages/ml/legoesm/training/data_parallel.py tests/ml/test_data_parallel.py && git commit -m "feat(scale): deterministic ERA5 sample sharding for data-parallel training"`

---

### Task 2: Cross-device gradient average (THE equivalence gate)

**Files:** Modify `packages/ml/legoesm/training/data_parallel.py`, `tests/ml/test_data_parallel.py`.

**Consumes:** the repo collective idiom — `jax.lax.pmean(x, axis_name)` inside `jax.pmap(..., axis_name="data")` (same pattern as `radiation/mc3d/parallel.py:91`, and process-transparent under `jax.distributed`).
**Produces:** `pmean_grads(per_device_grads) -> per_device_grads` (averaged, replicated) and `data_parallel_value_and_grad(loss_fn, model, batched_args) -> (mean_loss, mean_grad)` — `batched_args` has a leading device axis of size `n_local_devices`; runs `value_and_grad` per device and `pmean`s across the `"data"` axis (all devices). Falls back to a plain serial mean when `jax.device_count() == 1`.

- [ ] **Step 1: failing equivalence test** (2 CPU devices)
```python
import numpy as np
import jax, jax.numpy as jnp
from legoesm.training.data_parallel import data_parallel_value_and_grad

def _quad_loss(w, x):            # simple differentiable stand-in for the train step
    return jnp.sum((w * x) ** 2)

def test_grad_average_matches_serial_mean():
    # 2 "samples", one per device; data-parallel mean grad must equal serial mean.
    assert jax.device_count() >= 2, "run with --xla_force_host_platform_device_count=2"
    w = jnp.array([2.0, 3.0])
    xs = jnp.stack([jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])])   # (2 devices, 2)
    mean_loss, mean_grad = data_parallel_value_and_grad(_quad_loss, w, xs)
    # serial reference
    g0 = jax.grad(_quad_loss)(w, xs[0]); g1 = jax.grad(_quad_loss)(w, xs[1])
    l0 = _quad_loss(w, xs[0]); l1 = _quad_loss(w, xs[1])
    assert np.allclose(np.asarray(mean_grad), np.asarray((g0 + g1) / 2), atol=1e-9)
    assert np.allclose(float(mean_loss), float((l0 + l1) / 2), atol=1e-9)
```
- [ ] **Step 2:** `srun --account=glab --time=0:15:00 env JAX_ENABLE_X64=1 XLA_FLAGS=--xla_force_host_platform_device_count=2 <py> -m pytest tests/ml/test_data_parallel.py::test_grad_average_matches_serial_mean -v` → FAIL.
- [ ] **Step 3: implement** (append)
```python
def data_parallel_value_and_grad(loss_fn, params, batched_args):
    """Per-device value_and_grad over the leading axis of `batched_args`, mean-
    reduced across devices. batched_args leading dim = local device count.
    Serial fallback when there is a single device."""
    import jax
    import jax.numpy as jnp

    n_dev = jax.local_device_count()
    if n_dev == 1:
        # serial mean over the batch axis
        def _single(p, x):
            return jax.value_and_grad(loss_fn)(p, x)
        losses, grads = jax.vmap(_single, in_axes=(None, 0))(params, batched_args)
        mean_grad = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads)
        return jnp.mean(losses), mean_grad

    def _step(p, x):
        loss, grad = jax.value_and_grad(loss_fn)(p, x)
        return jax.lax.pmean(loss, "data"), jax.lax.pmean(grad, "data")

    # params replicated across devices; x sharded along the device axis
    p_rep = jax.tree_util.tree_map(
        lambda a: jnp.broadcast_to(a, (n_dev,) + a.shape), params)
    losses, grads = jax.pmap(_step, axis_name="data")(p_rep, batched_args)
    # pmean makes every device identical -> take device 0
    return losses[0], jax.tree_util.tree_map(lambda g: g[0], grads)
```
- [ ] **Step 4:** rerun (2 CPU devices) → PASS (grad == serial mean).
- [ ] **Step 5:** `git commit -m "feat(scale): cross-device gradient average (pmap+pmean), 2-device equivalence test"`

Note for the implementer: the real training step is `eqx.filter_value_and_grad` over an Equinox model, not a bare array. Keep this helper array-pytree-generic (it already is via `tree_map`); the entry (Task 4) passes the model's inexact-array leaves as `params` and the ERA5 batch as `batched_args`, exactly as `training_driver._build_train_step` already partitions them. Do NOT special-case Equinox here.

---

### Task 3: Data-parallel training loop

**Files:** Modify `packages/ml/legoesm/training/data_parallel.py`, `tests/ml/test_data_parallel.py`.

**Consumes:** Task 1-2; `create_optimizer` (`ml/training.py:72`); the mode loss_fn built by the entry.
**Produces:** `data_parallel_training_loop(loss_fn, params, opt_state, optimizer, sample_batches, n_epochs, *, on_epoch=None) -> (params, opt_state, history)` — per epoch: for each batch (leading device axis), `data_parallel_value_and_grad` → `optimizer.update` → `eqx.apply_updates`; identical on all ranks (same averaged grad), so replicas stay in sync. `on_epoch(epoch, mean_loss)` hook for rank-0 logging/checkpoint.

- [ ] **Step 1: test** — a 1-epoch loop over 1 batch of 2 CPU devices with `_quad_loss` + an `optax.sgd(0.1)` reduces the loss and returns finite params; history has 1 entry.
- [ ] **Step 2:** run (2 CPU devices) → FAIL.
- [ ] **Step 3: implement** the loop (optax update + apply_updates; call `on_epoch`).
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** `git commit -m "feat(scale): data-parallel training loop (optax update on averaged grads)"`

---

### Task 4: Entry script `train_weatherbench_scale.py`

**Files:** Create `scripts/run/train_weatherbench_scale.py`; Test `tests/unit/test_train_wb_scale_cli.py`.

**Consumes:** Task 3; `maybe_init_jax_distributed`; `era5_to_latlon_carry`; the mode builders in `training_driver` (`train_physics_params`/`train_neural_gcm`/`train_sfno` factories for model + loss_fn); `run_wb_forecast_eval`.
**Produces:** `build_scale_config_from_args(argv) -> ScaleConfig` (import-light, JAX-free — testable on the login node) with `mode`, `resolution_deg`, `epochs`, `multi_step_hours`, `train_years`, `eval_years`, `grad_accum`, `lr`, `optimizer`, `out_dir`, `resume`, `eval_wb2`. `main()` (heavy imports inside) does: dist-init (already done at import top) → build 0.7° grid/sigma → build mode model+loss → load+shard ERA5 → `data_parallel_training_loop` → rank-0 resumable checkpoints → optional WB2 eval.

- [ ] **Step 1: arg-parse test** — `build_scale_config_from_args(["--mode","neural_gcm","--resolution","0.7","--epochs","20","--multi-step-hours","6,12"])` returns `mode=="neural_gcm"`, `resolution_deg==0.7`, `multi_step_hours==(6,12)`; unknown `--mode foo` → `SystemExit`.
- [ ] **Step 2:** `<py> -m pytest tests/unit/test_train_wb_scale_cli.py -v` (login-node OK — the module must NOT import jax at top; guard heavy imports inside `main`, and put `maybe_init_jax_distributed()` inside `main` before the jax imports) → FAIL.
- [ ] **Step 3: implement** the argparse + `ScaleConfig` NamedTuple + `main()` skeleton. `main()` dispatches `mode` to the existing `training_driver` builders for (model, loss_fn, params partition), then calls `data_parallel_training_loop`. Mode dispatch raises `ValueError` on unknown mode.
- [ ] **Step 4:** run → PASS (arg-parse). Full `main()` validated by the smoke (Task 7).
- [ ] **Step 5:** `git commit -m "feat(scale): train_weatherbench_scale.py entry (dist-init, mode dispatch, data-parallel)"`

---

### Task 5: 0.7° config

**Files:** Create `config/wb/scale/train_07deg.yaml`; Test `tests/unit/test_train_wb_scale_cli.py` (extend: config loads + validates).

- [ ] Base 0.7° lat-lon (`n_lat: 256, n_lon: 512` or the grid's convention), `nlev`, `dt` CFL-sized for 0.7°, ERA5 zarr + train windows 2015–2019 / eval 2020, multi-step forecast loss (RMSE + grid-CRPS + bias, spec-CRPS off), optimizer (adamw warmup-cosine). Per-mode overlay keys (`neural_gcm`/`sfno` head dims). Test: YAML loads, required keys present, `dt` positive-finite. Commit `feat(scale): 0.7deg WeatherBench scale-training config`.

---

### Task 6: Derecho PBS + Levante SLURM launchers

**Files:** Create `scripts/cluster/derecho/train_wb.pbs`, `scripts/cluster/levante/train_wb.slurm`.

- [ ] **Derecho** (clone `scaling_derecho/scaling_gpu.sh`): `#PBS -A P08010000 -q main -l select=<N>:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB -l walltime=12:00:00`; source `scaling_derecho/_env.sh`; `module load conda cuda cray-mpich craype-accel-nvidia80`; `MPICH_GPU_SUPPORT_ENABLED=1`, `MPI4JAX_USE_CUDA_MPI=0`; `mpiexec -n $((N*4)) --ppn 4 set_gpu_rank python scripts/run/train_weatherbench_scale.py <args>` (PALS pin `CUDA_VISIBLE_DEVICES=$PALS_LOCAL_RANKID`).
- [ ] **Levante** (clone `scaling_levante/gpu_moist_scaling.slurm`): `#SBATCH --account=bd1083_gpu --partition=gpu --nodes=<N> --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --exclusive --time=12:00:00`; source `scaling_levante/_env.sh` (UCX CUDA-aware); `srun --gpu-bind=single:1 --cpu-bind=cores python scripts/run/train_weatherbench_scale.py <args>`.
- [ ] Both: `--resume` self-chain (pattern `amip_ginsburg_production_chain.sbatch`). Force-add (`.sbatch`/`.pbs` may be gitignored). Commit `feat(scale): Derecho PBS + Levante SLURM training launchers`.

---

### Task 7: Single-GPU smoke + codex gate

- [ ] **Ginsburg smoke:** `srun --account=glab --gres=gpu:1` running `train_weatherbench_scale.py --mode physics --resolution 4.0 --epochs 1 --smoke` (tiny) → confirms dist-init single-process fallback, ERA5 load, one data-parallel step (n_dev=1 serial path), checkpoint. (Full 0.7° multi-node is the Derecho/Levante integration.)
- [ ] **2-device CPU integration:** the full loop on 2 CPU devices with a T21 config confirms the multi-device path end to end.
- [ ] **Codex** adversarial review of `data_parallel.py` + the entry (`codex exec --sandbox read-only`) → fix → re-review to clean.
- [ ] Commit + push.

---

## Self-Review

- **Spec coverage:** data-parallel core (spec §4.1) = Tasks 1-3; entry (§4.2) = Task 4; config (§4.4) = Task 5; launchers (§4.3) = Task 6; testing (§6) = Tasks 1-3 CPU-device tests + Task 7 smoke; reuse ledger (§7) enforced by Global Constraints. All-three-modes = Task 4 dispatch. Both machines = Task 6.
- **Placeholder scan:** Tasks 1-3 carry full code; Tasks 4-6 carry concrete structure + exact scheduler directives (real values). No TBD.
- **Type consistency:** `shard_samples`, `data_parallel_value_and_grad`, `data_parallel_training_loop`, `build_scale_config_from_args`, `ScaleConfig` used consistently. The grad helper is array-pytree-generic (works for Equinox leaves).
- **Gap flagged honestly:** the real Equinox model + multi-step ERA5 loss wiring into `data_parallel_value_and_grad` is validated by the Task-7 smoke, not a pure unit test (it needs ERA5 + a model) — the unit tests pin the *collective correctness* (the risky part), which is the right thing to gate on before GPU-hours.
