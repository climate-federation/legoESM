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

## Bottom line

For every grid that can decompose across nodes today (atm icosahedral, ocean
lat-lon), weak/strong scaling is at the practical limit set by the Gloo/PCIe
fabric and mpi4jax's serialized per-neighbour `sendrecv`; per-device throughput
is at the memory-bandwidth-bound ceiling (quantified by
`scripts/bench/roofline_probe.py`). Closing the remaining grids (atm lat-lon at
high np, cubed-sphere >6 devices, spectral) requires the three architectural
projects above, each tracked as a task.
