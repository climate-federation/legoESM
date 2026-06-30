# Known issue: cross-node GPU-direct MPI aborts on Derecho (CXI/Slingshot)

**Status:** FIX MERGED (#681), pending empirical confirmation on a multi-node
run. #681 wired the Slingshot CXI fabric knobs into `_env.sh`
(`FI_CXI_RX_MATCH_MODE=hybrid`, `FI_CXI_DEFAULT_CQ_SIZE=131072`,
`FI_CXI_DISABLE_HOST_REGISTER=1`) that target the match-queue (LE pool) overflow
and the CUDA-aware MR-cache deadlock behind the abort, and `fullnode_gpu.sh` now
defaults to GPU-direct for all node counts. **Verify** a 2-node run completes
GPU-direct (no `cxil_map` abort, no `Not using CUDA-enabled MPI` warning) before
trusting multi-node performance; if it still aborts, set `MPI4JAX_USE_CUDA_MPI=0`
to fall back to host-staging and reopen this.
**Affected:** `scripts/cluster/scaling_derecho/fullnode_gpu.sh` multi-node
(`NODES>1`) GPU sweeps for icosahedral / latlon. Single-node (≤4 GPU) GPU-direct
is fine.
**First observed:** 2026-06-29, 2-node A100 interactive run (deg0038+deg0040).

## Symptom

A multi-node GPU sweep launches and decomposes correctly (ranks placed 4/node,
`cells/rank` halves across the node boundary), then **aborts inside the first
cross-node halo exchange**:

```
cxil_map: write error                         (xN)
r2 | MPI_Sendrecv ... MPIDI_OFI_send_normal(356):
     OFI tagged injectdata failed (ofi_send.h:356:MPIDI_OFI_send_normal:Invalid argument) - aborting
MPICH ERROR [Rank 2] ... application called MPI_Abort(MPI_COMM_WORLD, ...)
```

The failing call is a tiny `MPI_Sendrecv` (`scount=1 MPI_FLOAT`) on a **device**
buffer. It only happens **inter-node** (n=8 on 2 nodes); the same sweep at
**n=4 on one node** runs clean (intra-node GPU-direct uses NVLink/IPC, no NIC).

## Root cause (working theory)

`cxil_map: write error` is the HPE **Slingshot / CXI libfabric** provider failing
to register a **GPU device buffer** for inter-node RDMA; the `injectdata ...
Invalid argument` is the downstream effect (Cray MPICH tries to *inject* a small
device-resident message and CXI rejects the device pointer). Intra-node works
because it never touches the NIC.

## What has been ruled IN / OUT

- **NOT a launcher/counting bug.** `TOTAL_GPUS`, rank placement (`--ppn 4`,
  `PALS_LOCAL_RANKID` 0..3/node, `CVD` 0..3/node), and the lat-band/2-D-pencil
  decomposition are all verified correct on 2 nodes (`cells/rank` 851,968 →
  425,984 at 4→8 GPU). See the fixes in `fullnode_gpu.sh` history (LEGOESM_NGPUS
  override, unique-host count, `--ppn`).
- **NOT a JAX/dycore bug.** With host-staging (`MPI4JAX_USE_CUDA_MPI=0`) the same
  8-GPU/2-node run completes and is numerically correct.
- **GPU memory registration IS advertised as available:** on a login node,
  `fi_info -p cxi -c FI_HMEM` returns a `cxi` provider (HMEM filter passes), so
  the libfabric build is not categorically missing GPU support. `fi_info`
  version 1.22.0 at `/opt/cray/libfabric/1.22.0`.
- **GDRCopy is not available as a module** (`module avail gdrcopy` → none), so a
  GDRCopy registration path is not an option here; dmabuf is the remaining route.

## Workaround (in effect)

`fullnode_gpu.sh` selects the mpi4jax transport by **node count**:

- **1 node → `MPI4JAX_USE_CUDA_MPI=1`** (GPU-direct, NVLink/IPC — validated).
- **>1 node → `MPI4JAX_USE_CUDA_MPI=0`** (host-staged: GPU→host→MPI→host→GPU).

An explicit `MPI4JAX_USE_CUDA_MPI` in the environment overrides this (use `=1`
to re-test GPU-direct once the fabric issue is resolved).

### Impact on results

- **Correctness: none.** Host-staging moves the same bytes; results are
  bit-identical, and the cross-node decomposition is validated.
- **Performance: real, and resolution-dependent.** Host-staging adds a fixed
  per-halo cost (two device↔host copies). At small per-rank work it dominates
  (LL256 @ 8 GPU went *slower* than 4 GPU). At production resolution (LL720+,
  ~8× more work/rank) it is a much smaller fraction. **Multi-node GPU points in
  the scaling study are a lower bound on achievable inter-node scaling** and must
  be footnoted as host-staged until GPU-direct RDMA works.

## Reproducer (fast, no dycore)

In a ≥2-node GPU allocation, with the route-A env (`source _env.sh`;
`module load gcc cray-mpich cuda`; `MPICH_GPU_SUPPORT_ENABLED=1
MPI4JAX_USE_CUDA_MPI=1`):

```python
# $SCRATCH/halo_probe.py
import jax, jax.numpy as jnp
from mpi4py import MPI
import mpi4jax
c = MPI.COMM_WORLD; r = c.Get_rank(); n = c.Get_size()
x = jnp.ones(4, dtype=jnp.float32) * (r + 1)
y, _ = mpi4jax.sendrecv(x, x, source=(r-1) % n, dest=(r+1) % n, comm=c)
y.block_until_ready()
print(f"rank {r} backend={jax.default_backend()} recv={float(y[0])}", flush=True)
```
```bash
mpiexec --ppn 4 -n 8 bash -c 'export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-0}; exec "$@"' _ \
  python $SCRATCH/halo_probe.py
```
Reproduces the `cxil_map` / `injectdata` abort with GPU-direct on; passes with
`MPI4JAX_USE_CUDA_MPI=0`.

## Next steps to recover true GPU-direct (untried — needs an interactive node)

1. With the workaround env set, re-run the probe under
   `FI_LOG_LEVEL=warn FI_LOG_PROV=cxi` and read the exact registration reject:
   ```bash
   export FI_CXI_OPTIMIZED_MRS=0 FI_MR_CACHE_MONITOR=userfaultfd
   export FI_CXI_RDZV_PROTO=alt_read FI_CXI_DISABLE_HOST_REGISTER=1
   ```
   These target the CXI memory-registration path (dmabuf, since no GDRCopy).
2. If the probe passes, bisect to the minimal var set and wire it into
   `fullnode_gpu.sh` + `_env.sh`, then flip the multi-node default back to
   `MPI4JAX_USE_CUDA_MPI=1`.
3. If it still fails with HMEM advertised, this is a **CISL/site fabric config**
   matter (dmabuf for CUDA over CXI) — open a ticket with the `halo_probe.py`
   repro + the `FI_LOG` output, not more env guessing.
