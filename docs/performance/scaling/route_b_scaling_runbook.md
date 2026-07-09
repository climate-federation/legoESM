# Route-B scaling runbook (jax.distributed SPMD / collectives)

**What this is.** The CPU-node-vs-A100 throughput comparison over **Route B**:
`jax.distributed` multi-controller SPMD, halos exchanged with `ppermute` and
reductions with `psum` (NCCL on GPU, gloo on CPU) — **no `mpi4jax`**. One process
per device; the collectives are the scalable path being evaluated for large
multi-node runs.

For the Route-A (mpi4jax point-to-point) campaign, see
[`route_a_scaling_runbook.md`](route_a_scaling_runbook.md).

---

## What Route B actually runs

| | Detail |
|---|---|
| **Transport** | `jax.distributed` SPMD — `ppermute` halos + `psum` reductions. GPU: **NCCL + aws-ofi-nccl** on the CXI fabric. CPU: **gloo** (TCP over hsn). mpi4py is bootstrap-only. |
| **Drivers** | latlon → `bench_atm_latlon_spmd_scaling.py`; icosahedral → `bench_mpas_spmd_scaling.py`; cubed-sphere → `run_cpu_mpi_scaling.py --cs-spmd`. All `--multicontroller`. |
| **What it measures** | a **throughput microbenchmark** — the sharded step (halo `ppermute` + mass-fix `psum`), median ms/step → Mcells/s (and SYPD via `tidy_throughput_fields`). NOT a full timestepped run. |
| **CPU unit** | **128 SPMD processes per node** (one CPU "device" per core, OMP=1) — full-node packing. NODE ladder `1,2,4,8,16` → DEVICE counts `128,256,512,1024,2048`; plot x-axis = nodes (`n_devices/128`). |
| **GPU unit** | **1 process per A100**. Ladder `1 2 4 8 16 …` (auto-extends to `TOTAL_GPUS`). |
| **latlon decomp** | **1-D lat-band** (`mesh("lat", N)`): `n_devices ≤ n_lat` and `n_lat % n_devices == 0`. So at 128 devices/node, **res=128 caps at 1 node, res=512 at 4, res=2048 at 16** — rungs past `n_lat` are skipped with a note. |
| **icosahedral decomp** | MPAS/Voronoi RCB cell partition; `--reorder-for` = **LCM of the device counts** pins ONE padded mesh across the whole ladder (fair strong scaling). |
| **cubed-sphere** | face partition — admits only `1,2,3,6` (or `6·kt²`) devices, so it **cannot full-node-pack**: 1 proc/node (OMP=128) on the face-divisor ladder. A ≤6-device benchmark, not a full-node curve (a structural cube limit). |
| **spectral** | no domain decomposition → **not supported** on Route B. |

**Comm-tuning (Lane T).** `gpu_multinode_scaling.pbs RUN_TUNE=1` runs same-alloc
A/B arms — base / fused-halo (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`) / XLA
(`collective_permute_combine` + `pipelined_p2p`) / PGLE — under `_ab_tuning/`
(excluded from the curves). See `spmd_message_census_2026-07-08.md`.

---

## Prerequisites

- **GPU fabric (critical)**: build **aws-ofi-nccl** (`build_nccl_ofi.sh`) and pass
  `LEGOESM_NCCL_OFI_LIB=/glade/work/$USER/nccl-ofi/<tag>/lib`. Without it NCCL
  falls back to TCP sockets (2–3× slower comm) → the curves measure the socket
  path, not the fabric. Confirm `NET/OFI` in the log.
- **CPU env**: `legoesm-mpi`. gloo over `hsn` (`GLOO_SOCKET_IFNAME`).
- **Federation on Cray PALS**: the sweeps bridge `PALS_RANKID → OMPI_COMM_WORLD_RANK`
  and pass an explicit `--coordinator` (else `jax.process_count()==1`). Handled by
  the PBS scripts; nothing to set.

---

## How to run

`submit_routeb.sh` fans out one job **per grid** (and per backend) into one outdir.

```bash
# full campaign (all grids, GPU + CPU):
LEGOESM_NCCL_OFI_LIB=/glade/work/$USER/nccl-ofi/<tag>/lib PBS_ACCOUNT=<acct> \
  scripts/cluster/scaling_derecho/submit_routeb.sh $SCRATCH/routeb_all
```
Knobs (env, NOT `-v` — the wrapper builds `-v` per job):
| Var | Effect |
|---|---|
| `GRIDS="latlon icosahedral cubed-sphere"` | which grids to fan out |
| `RUN_GPU=0` | CPU-only campaign |
| `RUN_CPU=0` | GPU-only campaign |
| `PBS_ACCOUNT` / `PBS_ACCOUNT_CPU` / `PBS_ACCOUNT_GPU` | account(s) |
| `DRYRUN=1` | print the `qsub` lines, submit nothing |

Direct qsub of one lane (this is where you write `-v`):
```bash
qsub -A <acct> -v GRID=icosahedral,NODE_RANKS=1:2:4:8:16,OUTDIR=$SCRATCH/x,\
LEGOESM_REPO=<checkout> scripts/cluster/scaling_derecho/routeb_cpu_sweep.pbs
```
Ladders are **colon-separated** in `-v` (a comma is PBS's delimiter). Run **from
the checkout you want** — a worktree needs `LEGOESM_REPO` (or the job self-derives
it now; explicit still wins).

## Aggregate + plot
```bash
scripts/cluster/scaling_derecho/finalize_scaling.sh $SCRATCH/routeb_all
# -> all_tidy.csv + per-grid Mcells/s (+ SYPD) panels
```
**Keep Route-A and Route-B in SEPARATE outdirs** — the aggregator ingests both, and
mixing them onto one `(grid, backend, resolution)` curve is the usual
"broken curve." SYPD comes from the bench (`tidy_throughput_fields`, needs a `dt`);
old pre-#894 runs have Mcells/s but no SYPD (ico backfills from a recorded `dt`;
latlon needs a re-run).

---

## Route A vs Route B at a glance

| | **Route A** ([other doc](route_a_scaling_runbook.md)) | **Route B** (this doc) |
|---|---|---|
| Transport | mpi4jax `sendrecv` (point-to-point, blocking) | jax.distributed SPMD `ppermute`/`psum` (collectives) |
| CPU fabric | cray-mpich (MPI over hsn) | gloo (TCP over hsn) |
| GPU fabric | CUDA-aware cray-mpich 9.0.0 (CXI) | NCCL + aws-ofi-nccl (CXI) |
| CPU unit | 128 MPI ranks/node (1/core) | 128 SPMD procs/node (1/core) |
| latlon decomp | **2-D pencil** (scales past n_lat) | **1-D lat-band** (n_devices ≤ n_lat) |
| cube | ≤6 face-scatter, single node | cs-spmd face divisors 1,2,3,6 |
| Measures | full dycore step + real dt → **SYPD** | throughput microbench → Mcells/s (+SYPD) |
| Scales to | O(hundreds) ranks (blocking halos) | the path meant for O(10³) (collectives, overlap) |
| Submit | `submit_scaling.sh` (1 grid/job) | `submit_routeb.sh` (fans per-grid) |

**Reading the curves.** Anti-scaling at small GPU counts (cube np6, high-device
latlon/ico at low res) is **message latency dominating tiny compute**, not a
transport bug (the census confirmed field-packing is at floor). Flat curves below
~10–30k columns/GPU are the per-device floor (matches ICON/MPAS-A). Absolute latlon
SYPD is "at the bench `--dt`" (default 60 s), so lean on the *shape* for scaling.
