# Cross-node GPU-direct MPI on Derecho (CXI/Slingshot) — RESOLVED postmortem

**Status:** RESOLVED (2026-06-29). Multi-node GPU-direct halo exchange works.

## Resolution
Root cause: `scaling_gpu.sh` had stopped loading **`craype-accel-nvidia80`** at
runtime (dropped in an "env delta" cleanup). That module sets
`CRAY_ACCEL_TARGET=nvidia80` and engages cray-mpich's **GPU-aware NIC path**.
Without it, a cross-node send hands a GPU device pointer to a NIC path that
treats it as host memory → `cxil_map: write error` / OFI `injectdata Invalid
argument`. Intra-node GPU-direct worked anyway (CUDA IPC, no NIC), which masked
the regression.

**Fix:** re-added `craype-accel-nvidia80` to the `scaling_gpu.sh` module load and
defaulted `MPI4JAX_USE_CUDA_MPI=1` for all node counts. **Confirmed:** an
8-rank / 2-node mpi4jax `sendrecv` passes GPU-direct with the module loaded
(`CRAY_ACCEL_TARGET=nvidia80`, correct ring values, `rc=0`, no abort); the
identical run *without* the module aborts.

## Symptom (historical)
A multi-node GPU sweep launched and decomposed correctly (ranks placed 4/node,
`cells/rank` halving across the node boundary), then aborted in the first
cross-node halo:
```
cxil_map: write error                         (xN)
MPIDI_OFI_send_normal(356): OFI tagged injectdata failed (...:Invalid argument) - aborting
```
The failing send was a tiny `scount=1 MPI_FLOAT` device buffer; it only happened
**inter-node** (n=8 on 2 nodes), never intra-node (n≤4).

## What was ruled out before finding the cause
- **Launcher / counting** — verified correct: `TOTAL_GPUS` (the real 4-GPU cap
  was PBS exporting its own `$NGPUS`, fixed via `LEGOESM_NGPUS`), rank placement
  (`--ppn 4`, `PALS_LOCAL_RANKID` 0..3/node), and the lat-band/2-D-pencil
  decomposition (`cells/rank` 851,968 → 425,984 at 4→8 GPU).
- **JAX / dycore** — with host-staging (`MPI4JAX_USE_CUDA_MPI=0`) the same
  8-GPU/2-node run completed and was numerically correct.
- **The env knobs that did NOT fix it** — `#681`'s Slingshot CXI tuning
  (`FI_CXI_RX_MATCH_MODE=hybrid`, `FI_CXI_DEFAULT_CQ_SIZE`,
  `FI_CXI_DISABLE_HOST_REGISTER`) and `MPIR_CVAR_CH4_OFI_ENABLE_INJECT=0` each
  left the abort unchanged. They target a different failure mode (LE-pool /
  match-queue overflow); the missing piece was purely the accel module. The
  `#681` knobs are kept in `_env.sh` — still useful for the many-message halo at
  scale — but were not the fix on their own.

## Reproducer (fast, no dycore)
In a ≥2-node GPU allocation, route-A env + `module load … craype-accel-nvidia80`,
`MPICH_GPU_SUPPORT_ENABLED=1 MPI4JAX_USE_CUDA_MPI=1`:
```python
import jax, jax.numpy as jnp
from mpi4py import MPI
import mpi4jax
c = MPI.COMM_WORLD; r = c.Get_rank(); n = c.Get_size()
x = jnp.ones(4, dtype=jnp.float32) * (r + 1)
y = mpi4jax.sendrecv(x, x, source=(r-1) % n, dest=(r+1) % n, comm=c)  # FFI 0.9.x: single return
y.block_until_ready()
print(f"rank {r} backend={jax.default_backend()} recv={float(y[0])}", flush=True)
```
```bash
mpiexec --ppn 4 -n 8 bash -c 'export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-0}; exec "$@"' _ \
  python probe.py
```
Passes with `craype-accel-nvidia80` loaded; aborts (`cxil_map`) without it.
(Note: the FFI-based mpi4jax 0.9.x from #619 returns the recv array directly, not
a `(recv, token)` tuple — unpacking with `y, _ =` raises `ValueError`.)
