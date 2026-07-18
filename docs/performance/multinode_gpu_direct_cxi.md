# Cross-node GPU-direct MPI on Derecho (CXI/Slingshot) — RESOLVED

**Status:** RESOLVED (2026-07-02). Cross-node GPU-direct halo exchange works on
**cray-mpich ≥ 9.0.0**. The earlier abort was a genuine **cray-mpich 8.1.32** bug
in the CXI inject path, fixed by NCAR's default-PE bump to 9.0.0. `scaling_gpu.sh`
now gates the multi-node transport on `CRAY_MPICH_VERSION` (≥ 9 → GPU-direct,
< 9 → host-staged fallback); single node is GPU-direct on any version. An explicit
`MPI4JAX_USE_CUDA_MPI` always overrides the gate. A second transport (route-B,
NCCL multi-controller) also works cross-node but is socket-bound today — see the
Route-B section below; **route-A on the CXI fabric is the production pick.**

**Complementary bypass (2026-07):** independent of the mpi4jax fix above, the
production cubed-sphere driver also has an mpi4jax-FREE multi-node path —
`run_amip.py --distributed --distributed-mode spmd` (multi-controller
`jax.distributed`, NCCL collectives; checkpointing supported, diagnostics
writer still off). 2-process parity is bit-exact vs single-controller (jobs
8686550/8687224; gate `scripts/validate/validate_driver_cs_spmd_parity.py`).
NCCL reaches Slingshot through `aws-ofi-nccl`, not the MPICH OFI inject path —
so the cubed-sphere driver has two independent working multi-node GPU lanes.

## Transport policy (2026-07 — the standing recommendation)

For **multi-node GPU production**, prioritize the `jax.distributed`/NCCL
SPMD (route-B) path for halos and reductions; keep **mpi4jax (route A) for
CPU MPI lanes and serial==MPI parity tests**. Rationale:

- mpi4jax cross-node GPU is host-staged today (this ticket): every halo
  round-trips through host memory, so multi-node GPU numbers on the
  mpi4jax lanes are a **lower bound**, not the achievable scaling.
- NCCL reaches Slingshot through `aws-ofi-nccl` and does not touch the
  MPICH OFI inject path that aborts here — the cubed-sphere SPMD driver
  proves this end-to-end (bit-exact 2-process parity, above), and the
  lat-lon (`bench_atm_latlon_spmd_scaling --multicontroller`), ocean
  (`bench_ocean_latlon_spmd_scaling --multicontroller`) and MPAS-atm
  (M3c native-ppermute step, `bench_mpas_spmd_scaling`) route-B lanes
  federate the same way.
- The mpi4jax lanes stay valuable where they are strong: CPU clusters
  (no host-staging penalty), laptop/CI parity gates (`mpirun -np 2`
  serial==MPI tests), and as the AD-safe reference implementation
  (`_sendrecv_vjp` via `get_sendrecv_vjp`, `allreduce(SUM)` VJP).

Per component today: cubed-sphere AMIP, lat-lon atm/ocean and MPAS-atm all
have route-B lanes; the **plane CRM/LES pencil exchange is mpi4jax-only** —
an SPMD (`shard_map`) halo variant of the plane pencil exchange is the
highest-leverage follow-up for multi-node GPU CRM scaling, and new
GPU-scaling work should target it rather than further mpi4jax transport
tuning.

## Root cause

cray-mpich **8.1.32** rejected the model's tiny cross-node device send — a
`scount=1 MPI_FLOAT` (4-byte) halo control message — through the OFI **inject**
path (`fi_tinjectdata`), which on the CXI provider does not accept a GPU device
pointer:

```
cxil_map: write error                         (xN)
MPIDI_OFI_send_normal(356): OFI tagged injectdata failed (Invalid argument) - aborting
```

Intra-node (≤ 4 GPU, 1 node) was always fine (CUDA IPC / NVLink, no OFI inject).
Only inter-node aborted. **cray-mpich 9.0.0** fixed the inject-with-device-pointer
path; the same run now completes GPU-direct.

## What was actually going on (the version trap)

The fix rode in almost invisibly, and untangling it corrected three wrong beliefs
from the 8.1.32 era:

- **`mpi4py` links the version-agnostic soname `libmpi_gnu_123.so.12`**, which the
  dynamic linker resolves from `/opt/cray/pe/lib64` — the **default PE**, now
  **9.0.0**. So `mpi4py` runs against 9.0.0 *regardless of which `cray-mpich`
  module is loaded*. `MPICH_VERSION_DISPLAY=1` printed `CRAY MPICH version
  9.0.0.113 (ANL base 4.1.2)` even while `CRAY_MPICH_VERSION=8.1.32`
  (module) — the loaded module ≠ the runtime library.
- **`craype-accel-nvidia80` no longer exists** under that name on Derecho.
  `module load craype-accel-nvidia80` errors ("unknown"), silently swallowed by
  the job scripts' `2>/dev/null`. It was **never** what enabled GPU-direct. The
  CUDA GTL (`libmpi_gtl_cuda.so.0`) is a **`DT_NEEDED` link baked into `mpi4py`
  at build time**, so it loads with no runtime accel module. (Prior notes calling
  the accel module "REQUIRED, do not drop" were wrong — it was a dead no-op.)
- The one-off `mpi4jax.sendrecv` probe that "passed" on 8.1.32 was misread as
  host-staging. With `MPI4JAX_USE_CUDA_MPI=1` + a CUDA-built `mpi4jax` there is no
  host-stage branch — the device pointer goes straight to MPI. On 8.1.32 that is
  exactly why it *aborted*; on 9.0.0 the same path *completes*.

Net: multi-node was only ever host-staging because `scaling_gpu.sh` *chose*
`MPI4JAX_USE_CUDA_MPI=0` on > 1 node — a conservative fallback for the 8.1.32 bug,
not a fabric limitation.

## The coherent stack (what the scripts now do)

Loading `cray-mpich/8.1.32` while the default-PE libmpi is 9.0.0 leaves a
**mismatched** stack (9.0.0 libmpi + 8.1.32 GTL + a misleading `CRAY_MPICH_VERSION`)
that works only by accident of path ordering. `scaling_gpu.sh` now pins the
runtime explicitly:

```bash
module load gcc cuda
module load cray-mpich/9.0.0 || module load cray-mpich   # 9.0.0 explicit; bare defaults to 8.1.32
export MPICH_GPU_SUPPORT_ENABLED=1
```

so libmpi, the GTL, and `CRAY_MPICH_VERSION` all agree on 9.0.0 and the version
gate is truthful. No rebuild of `mpi4py`/`mpi4jax` is required — the soname link
already resolves to 9.0.0 — but rebuilding against `cray-mpich/9.0.0` is the
belt-and-suspenders option if you want the build headers to match too.

## Verify (in a ≥ 2-node GPU allocation)

```bash
module load gcc cuda cray-mpich/9.0.0
conda activate legoesm-gpu
echo "CRAY_MPICH_VERSION=$CRAY_MPICH_VERSION"                       # 9.0.0
ldd $CONDA_PREFIX/lib/python3.11/site-packages/mpi4py/MPI*.so | grep -iE 'libmpi|gtl'
#   want BOTH libmpi_gnu_123.so.12 and libmpi_gtl_cuda.so.0 under .../9.0.0/...
export MPICH_GPU_SUPPORT_ENABLED=1 MPI4JAX_USE_CUDA_MPI=1 JAX_PLATFORMS=cuda

mpiexec --ppn 4 -n 8 bash -c 'export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-0}; exec "$@"' _ \
  python scripts/bench/run_cpu_mpi_scaling.py \
    --grid icosahedral --mode strong --resolution 7 \
    --physics none --precision float32 --device gpu --output-dir "$SCRATCH/gpudirect"
```

Completes (no `cxil_map`) with ranks split 4/node = cross-node GPU-direct working.

## Open follow-ups (perf + rigor, not blockers)

- **GDRCopy is not loaded** (`gdrcopy_dl_hmem_init failed!` on every rank), so the
  tiny halo messages take the slower non-gdrcopy GPU-direct path. Any GPU-direct
  SYPD is therefore a **lower bound**; loading a `gdrcopy` module is a perf
  follow-up.
- **MPI==serial numeric check** for the newly-enabled multi-node GPU-direct path
  (same resolution at `-n 1`, compare results) before trusting the scaling curve —
  separate from transport correctness.
- **Fast reproducer** (no dycore) for regression-checking the fabric — the 4-byte
  `scount=1` device `sendrecv` that was the exact 8.1.32 trigger:

```python
import os, jax, jax.numpy as jnp
from mpi4py import MPI
import mpi4jax
c = MPI.COMM_WORLD; r = c.Get_rank(); n = c.Get_size()
x = jnp.ones(1, dtype=jnp.float32) * (r + 1)
y = mpi4jax.sendrecv(x, x, source=(r - 1) % n, dest=(r + 1) % n, comm=c)  # 0.9.x: single return
y.block_until_ready()
print(f"rank {r} backend={jax.default_backend()} recv={float(y[0])}", flush=True)
```

```bash
mpiexec --ppn 4 -n 8 bash -c 'export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-0}; exec "$@"' _ \
  python probe.py
```

Passes on cray-mpich ≥ 9.0.0; aborts (`cxil_map`) on 8.1.32.

## Route-B (NCCL multi-controller) — alternative transport, socket-bound

The `jax.distributed` multi-controller path (route-B, `--multicontroller`) is a
second cross-node GPU transport that bypasses mpi4jax/CXI-inject entirely — band
`ppermute`/`psum` go over NCCL. The Derecho canary
(`scripts/cluster/scaling_derecho/mc_nccl_canary.sh`, 2 nodes × 4 A100) **passes
all stages** (2026-07-02, job 6625434): init + `psum` + `ppermute` ring + both
atm/ocean parity gates + both full-step benches. So route-B **works cross-node**.

**BUT NCCL runs on `NET/Socket` (TCP), not the fabric.** `NCCL_DEBUG=INFO` shows
`Using network Socket` — NCCL probes `NET/IB` (RoCE), can't use it, and falls back
to TCP over the hsn NICs. There is **no `aws-ofi-nccl` plugin** in the environment
(none in `module avail`, none under `/opt/cray/pe/lib64` or the apps trees), so
NCCL never reaches libfabric/CXI. Root reason: the `jax[cuda12]` wheel bundles its
own `nvidia-nccl-cu12` with **no OFI plugin**.

Consequence: route-B timings are a **lower bound** — for the latency-bound halo,
TCP sockets (~tens of µs/msg) are far slower than CXI RDMA (~1–2 µs). Reference
canary number: atm 256×512×30, 8 GPU → ~4.0 ms/step steady (10 s one-time
compile), socket-bound.

**Decision:** **route-A (cray-mpich ≥ 9.0.0 GPU-direct) is the production
multi-node GPU transport** — it is on the CXI fabric and faster cross-node than
socket-NCCL. Route-B is a validated, correct fallback but stays socket-bound until
the plugin lands.

**Follow-up to put route-B on the fabric:** obtain `aws-ofi-nccl` built against the
CXI libfabric **and matched to the wheel's bundled NCCL version** (CISL request or
self-build) → put `libnccl-net*.so` on `LD_LIBRARY_PATH` (optionally
`NCCL_NET_PLUGIN=ofi`) → re-run `mc_nccl_canary.sh` → confirm the transport line
flips to `NET/OFI`. Only then is route-B expected to beat route-A.
