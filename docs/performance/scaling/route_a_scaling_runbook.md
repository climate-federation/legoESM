# Route-A scaling runbook (mpi4jax point-to-point)

**What this is.** The CPU-node-vs-A100 strong-scaling comparison over **Route A**:
`mpi4jax` point-to-point halo exchange (MPI `sendrecv`), one MPI rank per compute
unit, driven by `scripts/bench/run_cpu_mpi_scaling.py`. This is the "classic"
domain-decomposition path — each rank owns a subdomain and exchanges halos with
its neighbours via MPI.

For the Route-B (SPMD/collectives) campaign, see
[`route_b_scaling_runbook.md`](route_b_scaling_runbook.md). The two are summarised
side-by-side at the bottom.

---

## What Route A actually runs

| | Detail |
|---|---|
| **Transport** | `mpi4jax` — MPI `Sendrecv` halos (blocking, point-to-point). CPU: cray-mpich over the hsn NICs. GPU: CUDA-aware cray-mpich **9.0.0** (the CXI-inject fix) for cross-node GPU-direct. |
| **Driver** | `scripts/bench/run_cpu_mpi_scaling.py` (same for CPU and GPU). |
| **What it measures** | a **full dycore timestep** with a real `dt` → **SYPD** *and* Mcells/s. |
| **CPU unit** | **128 MPI ranks per node**, one rank per core, true domain decomposition. Ladder = full-node: `128, 256, 512, 1024, 2048` ranks = `1,2,4,8,16` nodes. |
| **GPU unit** | **1 rank per A100**. Ladder `1 2 4 8 16` (icosahedral auto-extends to `TOTAL_GPUS`, e.g. `… 32` on 8 nodes). |
| **latlon decomp** | **2-D pencil** (`--latlon-2d`, `proc_lat × proc_lon`) — scales past `n_lat` (both axes split). |
| **icosahedral decomp** | MPAS/Voronoi cell partition (mpi4jax `sendrecv` halo). |
| **cubed-sphere** | separate `cube_scaling_{cpu,gpu}.sh` (≤6-GPU face-scatter, single node). |
| **spectral** | no MPI path → a single 1-device point. |
| **Grids** | latlon, icosahedral, spectral via `scaling_{cpu,gpu}.sh`; cubed-sphere via `cube_scaling_*.sh`. |

**AD note.** mpi4jax `sendrecv` is wrapped in a `custom_vjp` (`halo_exchange.py`),
but it is blocking (no `Isend`/`Irecv` overlap) and historically host-staged — the
structural reason Route A does not scale to O(10³) ranks. That is *why* Route B
exists.

---

## Prerequisites

- **CPU env**: `legoesm-mpi` (CPU + MPI JAX build). `module load gcc cray-mpich`.
- **GPU env**: `legoesm-gpu` with a **CUDA-built mpi4jax** (README Step 1b). Pin
  `cray-mpich/9.0.0` (8.1.32 aborts the tiny cross-node device sendrecv on CXI).
  `MPI4JAX_USE_CUDA_MPI=1` for GPU-direct halos.

---

## How to run

One grid per job; `submit_scaling.sh` fans out the CPU job **and** the GPU job
into one outdir so a single `finalize_scaling.sh` plots them together.

```bash
# single node (CPU ranks 1..128, GPU 1..4):
PBS_ACCOUNT=<acct> \
  scripts/cluster/scaling_derecho/submit_scaling.sh $SCRATCH/scaling_ico icosahedral 6 7 8

# multi-node (NODES>1 spans nodes; latlon + icosahedral only):
NODES=16 PBS_ACCOUNT=<acct> \
  scripts/cluster/scaling_derecho/submit_scaling.sh $SCRATCH/scaling_ico icosahedral 7 8
#   CPU -> 128 ranks/node up to 2048 (16 nodes); GPU -> up to 4*NODES A100
```

Knobs (per-job, via the underlying scripts): `RESOLUTIONS`, `RANKS`/`GPU_RANKS`,
`PRECISIONS`, `MODES` (strong/weak), `PHYSICS`. Run **from the checkout you want**
(a git worktree needs `-v LEGOESM_REPO=<path>` or the job `cd`s to the main
checkout).

## Aggregate + plot
```bash
scripts/cluster/scaling_derecho/finalize_scaling.sh $SCRATCH/scaling_ico
# -> all_tidy.csv + per-grid SYPD (top) + Mcells/s (bottom) panels, x-axis = nodes / A100s
```
**Keep Route-A and Route-B results in SEPARATE outdirs** — the aggregator now
ingests both, and mixing them onto one `(grid, backend, resolution)` curve is the
usual "my curve looks broken" cause.

---

## Route A vs Route B at a glance

| | **Route A** (this doc) | **Route B** ([other doc](route_b_scaling_runbook.md)) |
|---|---|---|
| Transport | mpi4jax `sendrecv` (point-to-point, blocking) | jax.distributed SPMD `ppermute`/`psum` (collectives) |
| CPU fabric | cray-mpich (MPI over hsn) | gloo (TCP over hsn) |
| GPU fabric | CUDA-aware cray-mpich 9.0.0 (CXI) | NCCL + aws-ofi-nccl (CXI) |
| CPU unit | 128 MPI ranks/node (1/core) | 128 SPMD procs/node (1/core) |
| latlon decomp | **2-D pencil** (scales past n_lat) | **1-D lat-band** (n_devices ≤ n_lat) |
| Measures | full dycore step + real dt → **SYPD** | throughput microbench (halo+step) → Mcells/s (+SYPD) |
| Scales to | O(hundreds) ranks (blocking halos) | the path meant for O(10³) (collectives, overlap) |
| Submit | `submit_scaling.sh` (1 grid/job) | `submit_routeb.sh` (fans per-grid) |

**When to use which.** Route A = the production dycore step, real SYPD, the honest
"how fast does the model run" number at modest rank counts. Route B = the forward
path being evaluated for large-scale multi-node scaling (SCREAM-parity, issue
context) — a throughput microbenchmark of the collective transport.
