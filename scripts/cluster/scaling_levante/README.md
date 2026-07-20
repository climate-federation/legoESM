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

---

## Route-B (jax.distributed + NCCL) + ocean + CPU additions (2026-07)

Beyond the route-A `gpu_moist_scaling.slurm` above:

| File | What |
|---|---|
| `gpu_scaling.sbatch` | ATM (cube AMIP physics, single-process multi-GPU) + OCEAN (`bench_ocean_latlon_spmd_scaling.py`, 1/2/4 A100 with parity+conservation smoke) on one GPU node. Plain GPU env — no mpi4jax. |
| `gpu_multinode_scaling.sbatch` | MULTI-NODE route-B lanes over NCCL/IB (SLURM auto-detected `jax.distributed`): A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU). |
| `cpu_scaling.sbatch` | ATM + OCEAN CPU-MPI rank ladders on `compute` nodes (`legoesm-mpi` env), with fail-fast smokes. |
| `diagnosis.sbatch` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) the throughput jobs do NOT capture. Climbs the cube face-shard `1 2 3` ladder so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `sbatch --export=ALL,MODE=census` for counts only. Derecho twin: `scaling_derecho/diagnosis.pbs`. |

`_env.sh` now also carries the NCCL-over-IB defaults for the route-B lanes
(`NCCL_IB_HCA=mlx5`, `NCCL_SOCKET_IFNAME=ib0`, `NCCL_NET_GDR_LEVEL=PHB`,
`NCCL_CROSS_NIC=1`) — independent of, and coexisting with, the UCX/MPI
settings used by route-A. First multi-node run: `NCCL_DEBUG=INFO` must show
`NET/IB` (not `NET/Socket`); a wrong `NCCL_SOCKET_IFNAME` is the #1 cause of
IB-cluster bootstrap timeouts (verify with `ip addr` on a gpu node).

Account: export `SBATCH_ACCOUNT=<project>` (or pass `sbatch -A <project>`) —
SLURM directives cannot expand env vars.


## 2026-07 lane E: icosahedral/MPAS multicontroller

`gpu_multinode_scaling.sbatch` gained lane E (`RUN_MPAS=1`, default on):
icosahedral MPAS PE over `jax.distributed` + NCCL via
`scripts/bench/bench_mpas_spmd_scaling.py` (cell-partition reorder +
ppermute halos), 6 tasks = 2 nodes x 3 GPUs (`nCells = 10*4^L + 2` splits
evenly for 1/2/3/6). Subdiv-4 parity+conservation smoke gates the timed
`ICO_LEVEL` (default L7) case. Federation gate:
`tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py`.


## 2026-07 lane T: comm-tuning A/B ladder (`RUN_TUNE=1`)

Derecho-twin of the tuning ladder (full rationale + census numbers:
`docs/performance/scaling/spmd_message_census_2026-07-08.md` and the
Derecho README's lane-T section). Arms on the atm latlon (np=8), ocean
(np=8) and cube cs-spmd (np=6, `base`/`xla` only) lanes, all within ONE
allocation against a fresh `base` control arm:
`fused` (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`, bit-identical packing,
trace receipt 41→29 CPs/step), `xla`
(`--xla_gpu_collective_permute_combine_threshold_bytes=32MiB` +
`--xla_gpu_enable_pipelined_p2p=true`), `pgle`
(`JAX_ENABLE_PGLE=true`, per-run opt-in — recompiles after profiling).

```bash
sbatch --export=ALL,RUN_TUNE=1,RUN_NCCL=0,RUN_LATLON=0,RUN_OCEAN=0,RUN_MPAS=0 \
    scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch
```

Outputs under `$OUTDIR/_ab_tuning/` (aggregator-skipped by design).
