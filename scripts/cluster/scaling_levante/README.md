# Levante (DKRZ) GPU-MPI scaling — route-A, multi-node

SLURM/OpenMPI twin of `scripts/cluster/scaling_derecho/` (PBS/Cray-MPICH), for
the DKRZ **Levante** `gpu` partition (60 nodes × 4× A100, InfiniBand HDR200).

## What this is (and is not)

- **`gpu_moist_scaling.slurm`** — the **route-A** path: `run_cpu_mpi_scaling.py
  --device gpu`, one MPI rank per GPU, `mpi4jax` device-direct halos over UCX,
  domain-decomposed across GPUs **and nodes** (`--nodes>1`). Grids: `latlon`
  (latitude-band) and `icosahedral` (cell-partition) only.
- **`_env.sh`** — shared env (account, repo, conda env, modules, UCX CUDA-aware
  fabric knobs). **EDIT the marked values before first submit.**
- The **single-process multi-GPU sharding** path (cubed-sphere, gray/rrtmgp
  physics) is a *different* driver — `scripts/bench/run_levante_gpu_scaling.py`
  (+ `.sh` submitter). Do not confuse the two.

## Step 1 — build a CUDA-aware mpi4jax (once)

Route-A only device-directs halos if `mpi4jax` was built against a CUDA-aware
OpenMPI. In your `legoesm-gpu` conda env, with the **same** OpenMPI module the
job loads:

```bash
module load openmpi cuda        # EDIT to the Levante versions in _env.sh
pip install --no-binary=mpi4py mpi4py
MPI4JAX_USE_CUDA_MPI=1 pip install --no-binary=mpi4jax mpi4jax
python -c "import mpi4jax; print('cuda:', mpi4jax.has_cuda_support())"   # must print True
```

If `has_cuda_support()` is `False`, the job's **preflight fails loudly** rather
than silently host-staging (fake scaling).

## Step 2 — edit `_env.sh`

Set `LEGOESM_REPO`, `LEGOESM_CONDA_ENV`, and pin the `module load` versions to
the stack you built mpi4jax against. Confirm the `#SBATCH --account` in
`gpu_moist_scaling.slurm` matches your `*_gpu` allocation (`bd1083_gpu` for the
reference setup — the `--account` line is the source of truth).

## Step 3 — submit

```bash
# 1 node (4 GPUs), weak+strong, latlon:
sbatch scripts/cluster/scaling_levante/gpu_moist_scaling.slurm

# 4 nodes (16 GPUs), icosahedral, strong scaling, float64:
sbatch --nodes=4 --export=ALL,GRID=icosahedral,MODE=strong,PRECISION=float64 \
       scripts/cluster/scaling_levante/gpu_moist_scaling.slurm
```

Every result JSON carries the self-describing `metadata` block
(`scripts/bench/metadata.py`): `gpu_direct_active`/`host_staged_halo`,
`precision_knobs`, `decomposition`, `n_ranks`/`n_gpus`/`device_count`, so a
host-staged or CPU-fallback run is falsifiable from the record. Results land
under `$SCRATCH/legoesm_scaling/gpu_moist_levante_<stamp>/`.

## Status

Authored + shellcheck-clean + codex-reviewed on Ginsburg; **NOT yet run on
Levante** (separate cluster). First run: pin the module versions (Step 2) and
confirm the preflight passes. Cross-node GPU-direct over IB HDR is the specific
thing to validate (intra-node cuda_ipc is well-trodden; inter-node gdr_copy less
so).
