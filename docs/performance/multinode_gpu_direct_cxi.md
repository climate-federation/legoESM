# Cross-node GPU-direct MPI on Derecho (CXI/Slingshot) — OPEN

**Status:** OPEN for the mpi4jax path. Multi-node GPU scaling uses
**host-staged** halos (completes, correct); single-node GPU is GPU-direct and
unaffected. Cross-node GPU-direct still aborts for the real model.

**Strategic bypass (2026-07):** the production cubed-sphere driver now has an
mpi4jax-FREE multi-node path — `run_amip.py --distributed
--distributed-mode spmd` (multi-controller `jax.distributed`, NCCL
collectives; checkpointing supported, diagnostics writer still off).
2-process parity is bit-exact vs single-controller (jobs 8686550/8687224;
gate `scripts/validate/validate_driver_cs_spmd_parity.py`). NCCL reaches
Slingshot through `aws-ofi-nccl`, not the MPICH OFI inject path that aborts
here — so this ticket no longer blocks multi-node GPU scaling for the
cubed-sphere driver; it still bounds the mpi4jax lat-lon/ico bench lanes.

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

## Current state (2026-06-29)
A 2-node latlon canary (res256/f32) **still aborts at n=8** with `cxil_map: write
error` / `MPIDI_OFI_send_normal … injectdata Invalid argument`, even with
`craype-accel-nvidia80` loaded **and** #681's CXI knobs active. So neither is the
fix.

What we know:
- `craype-accel-nvidia80` **does** load (the MPICH stack line numbers change —
  `internal_Sendrecv` / `ofi_send.h:306` vs the non-accel `PMPI_Sendrecv` /
  `:356` — i.e. it swaps in the GPU-aware MPICH variant) but does **not** stop
  the abort.
- A trivial **eager `mpi4jax.sendrecv` probe "passed"** 8-rank/2-node — but that
  was misleading: eager mpi4jax appears to have **host-staged**, so it never
  exercised GPU-direct cross-node. The real model's halo (forced GPU-direct via
  `MPI4JAX_USE_CUDA_MPI=1`) is the true test, and it fails.
- The failing send is `scount=1 MPI_FLOAT` (4 bytes) through the OFI **inject**
  path, which rejects the GPU device pointer.

So `scaling_gpu.sh` is node-aware again: single node → GPU-direct (works),
multi-node → host-staged. Impact: multi-node GPU performance is a **lower bound**
on inter-node scaling (footnote it); correctness is unaffected.

## Still untried / next
- A **scount=1 GPU `sendrecv`** probe (match the model's exact message) to see if
  the 4-byte inject is the specific trigger.
- A **raw-mpi4py GPU sendrecv** cross-node (needs cupy/numba in the env — neither
  is currently installed) to localize: mpi4jax buffer handling vs CXI fabric.
- This is now **CISL-ticket** territory (CUDA-aware MPI over CXI inject) rather
  than more env-knob guessing.

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
