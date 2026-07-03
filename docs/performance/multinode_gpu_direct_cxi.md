# Cross-node GPU-direct MPI on Derecho (CXI/Slingshot) — RESOLVED

**Status:** RESOLVED (2026-07-02). Cross-node GPU-direct halo exchange works on
**cray-mpich ≥ 9.0.0**. The earlier abort was a genuine **cray-mpich 8.1.32** bug
in the CXI inject path, fixed by NCAR's default-PE bump to 9.0.0. `scaling_gpu.sh`
now gates the multi-node transport on `CRAY_MPICH_VERSION` (≥ 9 → GPU-direct,
< 9 → host-staged fallback); single node is GPU-direct on any version. An explicit
`MPI4JAX_USE_CUDA_MPI` always overrides the gate.

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
