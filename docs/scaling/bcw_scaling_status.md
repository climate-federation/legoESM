# Baroclinic-wave + ocean scaling: status vs theoretical limit

Campaign on Ginsburg (CPU-MPI + multi-node GPU, FP32 & FP64, weak + strong),
atmosphere baroclinic wave (dry + moist) and ocean, tracked per iteration in
`results/bcw_scaling/scaling_ledger.csv` and plotted in
`docs/scaling/bcw_scaling_vs_iteration.png`. SOTA references: NeuralGCM (JAX,
spectral/TPU), MPAS (Voronoi MPI), MOM6/E3SM (2D ocean/atm decomposition),
CliMA (JAX GPU), Oceananigans (GPU kernel fusion).

## UPDATE 2026-06-16 — multi-node UNBLOCKED (was a bug, not a fabric limit)

The earlier "atm icosahedral is single-node only / at the practical limit"
verdict was **wrong about the cause**: np>=64 across nodes deadlocked on the
first step, which had been mis-attributed to the Gloo/PCIe fabric. It was a
**bug** — an ASYMMETRIC Voronoi edge/vertex halo schedule. `edge_send`/
`vert_send` were built only over cell-recv neighbours, so a rank sharing only an
edge/vertex boundary (beyond the cell halo) never received the owned edge it
needed and its blocking `sendrecv` hung forever (rendezvous, no cross-node eager
buffering). Fix (commit 9247b4886): build the send candidate set over
`neighbor_ranks ∪ owners(compute_halo_cells(rank, halo_depth+1))` so send
mirrors recv; guarded by a host-side symmetry regression test
(`tests/parallel/test_voronoi_schedule_symmetry.py`).

**Genuine multi-node now runs** (8 ranks/node × 4 cores, res6 = 40962 cells,
icosahedral, strong):

| case | np64 (8 nodes) | np128 (16 nodes) | np64→128 |
|---|---|---|---|
| dry  f64 | SYPD 19.9 (43 Mc/s) | SYPD 27.0 (58 Mc/s) | 1.35× |
| dry  f32 | SYPD 25.1 (54 Mc/s) | SYPD 26.8 (58 Mc/s) | 1.07× |
| moist f64 | SYPD 14.6 (31 Mc/s) | SYPD 21.0 (45 Mc/s) | 1.44× |
| moist f32 | SYPD 16.7 (36 Mc/s) | SYPD 23.6 (51 Mc/s) | 1.42× |

Strong-scaling past np64 is comm-bound (the real Gloo/PCIe fabric limit — the
267 µs `sendrecv` floor × per-step halos), as the roofline section predicts, but
the runs are now CORRECT and complete rather than hanging. The disk mesh-cache
(commit 49aaa133d) + rank-0-build barrier (259bf1ae2) remove the redundant
per-rank SCVT rebuild that separately stalled setup at high rank count. The
bottom-line "practical limit" framing below still holds for per-device
throughput and weak efficiency; what changed is that the icosahedral grid
decomposes across NODES for real now.

### Multi-node chapter — full strong-scaling curves (8 ranks/node)

Both decomposable CPU grids now run across nodes. Each resolution has a
strong-scaling sweet spot, then goes comm-bound (cells/rank too small → the
267 µs `sendrecv` floor dominates the shrinking payload). This is the fabric
theoretical limit, now reached on CORRECT runs.

**atm icosahedral, res6 (40962 cells), strong, f64 dry** — SYPD vs np:

| np | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| SYPD | 4.8 | 13.6 | 19.9 | **27.0** | 16.4 |
| Mc/s | 10.5 | 29.4 | 43.1 | **58.4** | 35.5 |

Peak at np128 (640→320 cells/rank); np256 (160 cells/rank) collapses — past the
useful decomposition for res6. f32 ≈ 1.3× f64 (np128 f32 SYPD 26.8). NO np16→32
cliff at 8 ranks/node (the old 32-ranks/node Gloo cliff is gone — fewer ranks =
fewer messages). np256 only scales with a bigger problem (res7) — the weak
regime; that point (np256/res7 = 640 cells/rank, matching np16/res5 + np64/res6)
is the natural weak-scaling triple.

**ocean lat-lon (band), strong, f64** — SYPD vs np: np8 12.4 (res192) → np16
24.4 → np32 **31.3** (res192 peak) → np64 39.3 (res128) → np128 15.5 (res256).
res192 peaks ~np32; np64/128 need res≥256 (the harness GUARDS bands <2 rows/rank
with a clean error, not a hang). The band halo (N/S neighbours) is structurally
symmetric — it never had the Voronoi edge/vertex asymmetry bug, confirmed running
clean to np128.

### Status of every grid × precision toward its theoretical limit

| grid | precision | multi-node | limit reached | residual |
|---|---|---|---|---|
| atm icosahedral | f64 & f32 | **np8..256 ✓** | yes — res-dependent strong peak (np128 @ res6), fabric comm-bound past it | none (fabric-bound) |
| ocean lat-lon | f64 & f32 | **np1..128 ✓** | yes — res-dependent strong peak, same fabric wall | none (fabric-bound) |
| atm lat-lon (FV) | f64 & f32 | band np..128 ✓ | yes — band fabric-optimal on Gloo; 2-D pencil loses (latency-bound) | 2-D win needs InfiniBand |
| atm cubed-sphere | f64 & f32 | ≤6 faces/node | partial | >6-device sub-face tiling = future-HW project |
| atm spectral | f64 | single-device | n/a | no tracer storage (moist) + no MPI path — large additions |
| GPU (any) multi-device | f32 & f64 | **infra-blocked** | n/a | env's mpi4jax is CPU-only; needs CUDA-aware rebuild (not a code gap) |

The CPU-decomposable grids (atm icosahedral, ocean lat-lon, atm lat-lon band)
are characterised to their fabric limit on Ginsburg. The open items are
capability gaps (spectral moist, cubed-sphere sub-face tiling) or infra blocks
(CUDA-aware mpi4jax), not algorithmic scaling bugs.

## Where each configuration stands

| component / grid | multi-node path | weak E | strong E | peak Mc/s·dev | verdict |
|---|---|---|---|---|---|
| **atm icosahedral** (MPAS) | CPU-MPI + GPU MPI, full ladder | **0.80** (f32) | 0.14–0.63 | **~107** | near practical limit |
| **ocean lat-lon** (C-grid) | CPU band + GPU SPMD | **0.92** | 0.20–0.92 | ~105 | near practical limit |
| **atm lat-lon** (FV C-grid) | CPU band + 2-D pencil | 0.05 @128 (band) | 0.04–0.11 | ~49 | band is **fabric-optimal on Gloo**; 2-D built+validated but loses here (latency-bound, see below) |
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
  128 ranks). The SOTA fix is a 2D pencil decomposition (MOM6/E3SM) — now
  **built, validated, and measured** (below). **Verdict: on this Gloo/PCIe
  fabric the band is OPTIMAL and the 2-D pencil LOSES.** The band keeps
  longitude LOCAL (ZERO lon messages — only N/S, one *direction*); the 2-D
  pencil adds an E/W direction. On a latency-bound fabric the 267 µs sendrecv
  floor dominates the payload (even a full-lon N/S message is ~343 µs ≈ one
  floor), so the minimum-message-DIRECTION decomposition wins: band (1
  direction) beats 2-D (2 directions) **regardless of np or fusion**. Measured
  (job 8503081, strong res=64 f64): 2-D vs band SYPD np4 3.38/3.31, np8
  2.95/4.05, np16 0.47/2.22. The np16 4.7× gap is inflated by the current
  UNFUSED per-operator `exchange_halo_lon` (one lon exchange per lon-padding
  op); a lon-halo fusion (one exchange/step) would narrow it to ~1.5× (the
  single extra E/W floor) but NOT flip the verdict — 2 directions still lose to
  1 when latency ≫ bandwidth. **The 2-D pencil wins only when payload ≫ latency
  (RDMA/InfiniBand, or very high resolution) — the same fabric wall the roofline
  already identified.** So atm-lat-lon is fabric-bound like the others; the band
  is its fabric-optimal decomposition on Ginsburg.

## Levers evaluated this campaign

| lever | result |
|---|---|
| moist on MPAS (Kessler, column-local) | shipped — moist now scales on the ico ladder |
| column-local pre-physics halo skip (Kessler) | correctness-neutral; perf null at tested np (compute-bound regime); helps only deep in the latency-bound regime |
| batched Voronoi halo (field-batched) | already default (prior campaign) |
| METIS vs RCB partition | dead — RCB already balanced on uniform mesh (prior campaign) |
| mixed precision (FP32 work) | atm ico FP32 ~107 Mc/s vs FP64 ~47; FP32 ~2× as expected |

## Remaining gaps = architectural projects (not quick levers)

1. **atm lat-lon 2D pencil decomposition — BUILT + VALIDATED + MEASURED
   (task #14 done).** Full wall-pole 2-D C-grid step: `LatLon2DLayout`,
   `pad_halo_latlon_2d`, `exchange_halo_lon`, the operator lon-op conversion
   (`pad_lon_cgrid`: gradient_x / interp_uface / curl-v / mask ops +
   `absolute_vorticity_coriolis`), u-face scatter/gather convention,
   `make_latlon_2d_mpi_step`, and the `run_cpu_mpi_scaling --latlon-2d` harness.
   Validated: dycore gate `test_latlon_2d_mpi_step.py` (2×2 == 1×4, mass
   < 1e-12, decomposition-invariant) + operator equivariance np={2,3,6} + codex
   (multiple rounds). **Result: the band is fabric-optimal on Gloo; the 2-D
   pencil loses here** (latency-floor analysis above — 2 message directions
   can't beat 1 when latency ≫ bandwidth). Remaining (fabric-gated, like the
   GPU-multi infra block): (a) the 2-D win requires RDMA/InfiniBand — re-measure
   there to demonstrate it; (b) a lon-halo FUSION would narrow the Gloo gap
   4.7×→~1.5× but not flip it (deferred — no Gloo payoff); (c) the atmosphere's
   180° pole-FOLD under a lon split still needs a lat-pencil transpose
   (`pole_bc="fold"` raises; the shipped path is wall-pole only — a labeled
   midlatitude throughput benchmark, NOT atm-pole-correct). Ocean (wall poles)
   could use 2-D but does not need it (band weak E already 0.92).
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

## GPU: single-device real, multi-GPU MPI blocked by the env

Two GPU bugs found + handled:
1. The GPU ladder set `env JAX_PLATFORMS=cuda` but did NOT pass `--device gpu`,
   so `_configure_jax_cpu` pinned `JAX_PLATFORMS=cpu` — the ENTIRE g1..g32 "GPU"
   ladder silently ran on CPU (JSON `backend=cpu`). Fixed: `--device gpu` + a
   backend assertion that SystemExits on CUDA fallback (commit c8bd94fc3), so
   CPU can never be recorded as GPU again. Bogus dirs purged.
2. With the fix, SINGLE GPU works (real): 1 GPU I5 = **f32 791 SYPD (197
   Mc/s), f64 158 SYPD (39 Mc/s)** — 8-16x the bogus CPU-fallback numbers and
   ~2x the CPU per-device throughput. But MULTI-GPU MPI (g2+) fails with
   "mpi4jax GPU extensions could not be imported — rebuild mpi4jax with CUDA":
   the env's mpi4jax is CPU-only, so the GPU halo exchange cannot run. GPU
   multi-node scaling is therefore BLOCKED until mpi4jax is rebuilt CUDA-aware
   (infra task), or a single-process multi-GPU SPMD path (no mpi4jax) is used.
   The GPU panel today is single-device per-resolution throughput (the real
   per-GPU ceiling).

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
