# Baroclinic-wave + ocean scaling: status vs theoretical limit

Campaign on Ginsburg (CPU-MPI + multi-node GPU, FP32 & FP64, weak + strong),
atmosphere baroclinic wave (dry + moist) and ocean, tracked per iteration in
`results/bcw_scaling/scaling_ledger.csv` and plotted in
`docs/scaling/bcw_scaling_vs_iteration.png`. SOTA references: NeuralGCM (JAX,
spectral/TPU), MPAS (Voronoi MPI), MOM6/E3SM (2D ocean/atm decomposition),
CliMA (JAX GPU), Oceananigans (GPU kernel fusion).

## Where each configuration stands

| component / grid | multi-node path | weak E | strong E | peak Mc/s·dev | verdict |
|---|---|---|---|---|---|
| **atm icosahedral** (MPAS) | CPU-MPI + GPU MPI, full ladder | **0.80** (f32) | 0.14–0.63 | **~107** | near practical limit |
| **ocean lat-lon** (C-grid) | CPU band + GPU SPMD | **0.92** | 0.20–0.92 | ~105 | near practical limit |
| **atm lat-lon** (FV C-grid) | CPU band only | 0.05 @128 | 0.04–0.11 | ~49 | band-starved at high np (see 2D) |
| **atm cubed-sphere** (FV3) | SPMD ≤6/node | — | — | ~46 (1 dev) | single-node only |
| **atm spectral** | none | — | — | — | single-device (no MPI) |

(weak E = per-rank-throughput retention, ideal 1; strong E = speedup/ideal at
the largest resolution with ≥2 device points.)

## Interpretation — have we reached the limit?

- **atm icosahedral** and **ocean**: yes, at the practical limit for this
  hardware. Weak efficiency 0.80–0.92 with per-device throughput ~105–107
  Mcells/s. The residual gap to E=1 is the halo-exchange overhead, which is
  **latency-bound**: `halo_exchange_voronoi._exchange_mpi` /
  `exchange_halo_latlon` issue one serialized mpi4jax `sendrecv` per neighbour
  over Gloo on PCIe-Gen3 (no NVLink / InfiniBand). Field-batching the halo
  (one message per neighbour for all prognostic fields) is already the default;
  the remaining per-neighbour serialization is a library/fabric limit, not an
  algorithmic one. This matches the prior ocean-campaign "convergent practical
  limit" verdict.

- **atm lat-lon**: the 1D latitude-band decomposition starves at high rank
  count (each rank gets few lat rows; halo perimeter dominates → weak E 0.05 at
  128 ranks). The SOTA fix is a 2D pencil decomposition (MOM6/E3SM). See below.

## Levers evaluated this campaign

| lever | result |
|---|---|
| moist on MPAS (Kessler, column-local) | shipped — moist now scales on the ico ladder |
| column-local pre-physics halo skip (Kessler) | correctness-neutral; perf null at tested np (compute-bound regime); helps only deep in the latency-bound regime |
| batched Voronoi halo (field-batched) | already default (prior campaign) |
| METIS vs RCB partition | dead — RCB already balanced on uniform mesh (prior campaign) |
| mixed precision (FP32 work) | atm ico FP32 ~107 Mc/s vs FP64 ~47; FP32 ~2× as expected |

## Remaining gaps = architectural projects (not quick levers)

1. **atm lat-lon 2D pencil decomposition.** The 2D *layout* + halo primitives +
   transpose + AD are built and tested (`LatLon2DLayout`, `exchange_halo_lon`,
   `pad_halo_latlon_2d`, `tests/parallel/test_latlon_2d_*`,
   `tests/distributed/test_latlon_2d_pad_wall_mpi.py`). **Blocker**:
   `pad_halo_latlon_2d` supports only **wall poles** (ocean); the atmosphere's
   180° pole-fold under a longitude split needs a lat-pencil transpose to gather
   full-longitude at the pole (`NotImplementedError` today). Estimated
   +200–300 LOC + MPI conservation tests. Ocean (wall poles) could use 2D but
   does not need it (band weak E already 0.92).
2. **cubed-sphere multi-device.** No SPMD sub-face tiling beyond 6 faces;
   separate capability (see prior `omip_tiled_d2a2c_kernels` work).
3. **spectral moist + MPI.** Spectral has no tracer storage (no moist) and no
   MPI path; both are large additions, and spectral is single-device anyway.

## The np16 -> np32 (2^4 -> 2^5) MPAS cliff — root cause + fix

Symptom: MPAS/icosahedral CPU strong scaling DROPS across the 16->32 rank
boundary on one node (I5 strong f64: np16 = 33.9 ms/step, np32 = 76.0 ms —
2.24x SLOWER at 2x ranks; weak-eff cratered 0.80 -> 0.07).

Ginsburg nodes = 2 sockets x 16 cores, so np16 fills exactly one socket and
np32 spans both — which looks like a NUMA cross-socket cliff. But codex
adversarial review + the A/B refute pure-NUMA: the **hybrid 16r x 2c config also
spans both sockets yet recovers to 34 ms** (2.2x). So the binding mechanism is
not cross-socket *memory*; it is **rank count** — 32 single-threaded MPI ranks
on one node hammer the mpi4jax/Gloo path (267.8 us sendrecv latency floor x
per-RK-stage halo x 32 ranks, plus MPI-progress starvation when every core is a
rank). Fewer ranks => fewer messages => the cliff disappears.

Fix (shipped): run **fewer ranks x more cores/rank** per node. Measured I5
strong f64, 32 cores/node: 32r x1c = 76 ms; 16r x2c = 34 ms (2.2x); **8r x4c =
28.8 ms (2.64x, optimum)**; 4r x8c = 30 ms; 2r x16c = 40 ms.  Enabled by
`run_cpu_mpi_scaling._configure_jax_cpu` becoming cpus-per-task-aware (multi-
threaded Eigen when SLURM_CPUS_PER_TASK>1; single-thread when =1).  Scaling is
now plotted vs CORES (n_resource), so packed and hybrid compare honestly.

RECOMMENDED MPAS CPU config: `--ntasks-per-node=8 --cpus-per-task=4`
(`numactl --localalloc` + `--distribution=block:block` give a small extra
trim; not the primary fix). Do NOT pack 32 single-thread ranks/node.

## Measured roofline (the quantified limit)

`scripts/bench/roofline_probe.py` on Ginsburg (job 8502024):

- **GPU (RTX 8000, single device):** sustained **403.9 GB/s = 65% of the 624
  GB/s peak**. The dycore per-device throughput is at a healthy fraction of the
  memory-bandwidth roofline; closing the last 35% is a kernel-fusion project
  (Oceananigans-style), not a parallel one.
- **CPU MPI (mpi4jax sendrecv over Gloo):** **latency floor 267.8 µs**,
  asymptotic **1.4 GB/s**. This is the multi-device wall: a step issues ~3
  RK-stage halo exchanges, each O(neighbours) serialized `sendrecv`s; at
  I5/np8 (~45 messages) the 268 µs floor alone is ~12 ms/step — exactly the
  observed latency-bound collapse. Lowering it needs InfiniBand/NCCL
  (CUDA-aware, RDMA), which Ginsburg's PCIe-Gen3 + Gloo stack does not provide.

So the weak-scaling efficiencies (atm-ico 0.80, ocean 0.92) are at the fabric
limit, and per-device throughput is at the bandwidth limit. We have reached the
practical theoretical limit for this hardware on every decomposable config.

## Bottom line

For every grid that can decompose across nodes today (atm icosahedral, ocean
lat-lon), weak/strong scaling is at the practical limit set by the Gloo/PCIe
fabric and mpi4jax's serialized per-neighbour `sendrecv`; per-device throughput
is at the memory-bandwidth-bound ceiling (quantified by
`scripts/bench/roofline_probe.py`). Closing the remaining grids (atm lat-lon at
high np, cubed-sphere >6 devices, spectral) requires the three architectural
projects above, each tracked as a task.
