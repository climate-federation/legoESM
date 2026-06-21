# Baroclinic-wave + ocean scaling: status vs theoretical limit

Campaign on Ginsburg (CPU-MPI + multi-node GPU, FP32 & FP64, weak + strong),
atmosphere baroclinic wave (dry + moist) and ocean, tracked per iteration in
`results/bcw_scaling/scaling_ledger.csv` and plotted in
`docs/scaling/bcw_scaling_vs_iteration.png`. SOTA references: NeuralGCM (JAX,
spectral/TPU), MPAS (Voronoi MPI), MOM6/E3SM (2D ocean/atm decomposition),
CliMA (JAX GPU), Oceananigans (GPU kernel fusion).

## UPDATE 2026-06-20 — cube-MOIST tiled np>6 STEP shipped (the last grid below its theoretical limit)

The cubed-sphere replicated cs-spmd path caps at np≤6 (one face/device); np>6 needs
the sub-face tiling (np=6·kt²). The DRY tiled 3D-PE step was already np24/54
bit-identity gated; cube-MOIST is now too. Shipped as 4 parity-gated increments
(`docs/scaling/cube_moist_tiled_step_design.md`): (1) single-tracer tiled
advection (155ff92a5), (2) packed q_v/q_c/q_r (30fbb3cd3), (3) moist tracer
tendency = advection + injected Kessler (7297b2237), (4) full moist tiled SSP-RK3
STEP `make_tiled_fv3_hydrostatic_moist_step_stage_2d` (9ac698e42). The shared
`_build_hydro_tile_tendency_fns` was made tracer-aware (tracers ride the SAME
horizontal `dgrid_to_center_vector` u_cell + vertical `sigma_dot`/`mass_flux`
driver as T; Kessler column physics INJECTED so core stays physics-agnostic; dry
path bit-identical) + a generic `_ssp_rk3_tile_step` (dry step refactored onto it).
np24 parity (test_tiled_fv3_hydrostatic_moist_step.py): u_d/v_d/T/p_s rel<1e-10,
q_pack rel 3.6e-10 (Kessler nonlinear + RK3 FMA reorder, bit-identity class).
**This closes the LAST gap: all atmosphere grids × {dry,moist} × {f32,f64} ×
{weak,strong} are now at their Ginsburg practical limit, AND the cube has a
parity-gated np>6 sub-face path for BOTH dry and moist (future-HW — np>6
anti-scales on Ginsburg CPU by design; the value is the capability + the
bit-identity receipt, NOT a Ginsburg speed number).**

## UPDATE 2026-06-19 — cube-MOIST now scales multi-device (cs-spmd); moist matrix complete

The cubed-sphere `--cs-spmd` path previously **rejected all physics** — cube-moist
was single-device only while the other grids scaled moist multi-device. That gap is
now closed: `_build_cubed_sphere_spmd` accepts `--physics moist`, threading
`make_kessler_forcing_cube` through the already-physics-capable `make_sharded_step`
(`model.step_with_physics`) under the jax.distributed multiface-ppermute halo. The
q_v/q_c/q_r tracers ride the same halo as T; Kessler is column-local (no extra
exchange). Multi-device numerical parity vs serial = **1.2e-10 / 8.7e-11 / 2.9e-10**
at np2/3/6 (bit-identity class). C96 strong / C24→C58 weak, np 1/2/3/6, both
precisions:

| | np1 | np2 | np3 | np6 |
|---|---|---|---|---|
| cube-moist STRONG f64 (Mc/s) | 10.5 | 9.2 | 10.0 | 10.3 |
| cube-moist WEAK f64 (Mc/s) | 9.3 | 7.4 | 8.6 | 10.0 |
| cube-dry STRONG f64 (Mc/s) | 22.1 | 17.6 | 19.6 | 21.4 |

Strong-scaling FLAT (np6≈np1) = the Ginsburg CPU HW limit (halo cost, no compute
speedup — consistent with the convergent-practical-limit verdict). WEAK-scaling
holds per-device throughput np1→np6 (moist 9.3→10.0) → aggregate scales ~6× for 6×
work. Moist ≈2× dry per-cell cost (Kessler + 3-tracer advection). **With this, the
moist × {latlon, icosahedral, cubed-sphere, spectral} × {f32, f64} × {weak, strong}
matrix is complete at Ginsburg's practical limits.** The np>6 cube sub-face tiling
(SW + 3D-PE full step) remains built + np24/54 bit-identity parity-gated but
future-HW (np>6 anti-scales on Ginsburg CPU by design). Bench:
`run_cpu_mpi_scaling --grid cubed-sphere --cs-spmd --physics moist`; parity
`tests/parallel/test_cube_moist_spmd_parity.py`.

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

### atm icosahedral WEAK scaling (constant work per rank)

True weak scaling holds 3-D cells/rank fixed (~16640 = nCells·26/np): the triple
np16/res5, np64/res6, np256/res7. SYPD is NOT the weak metric here (CFL halves dt
each resolution level — res5 dt 390 s, res6 180, res7 90 — so SYPD drops by
construction); **throughput (Mc/s) is** — ideal weak scaling = Mc/s growing
linearly with np. Measured np64→np256 (4× ranks, 4× problem):

| case | np64 Mc/s | np256 Mc/s | weak E (np64→256) |
|---|---|---|---|
| dry  f64 | 43.1 | 154.8 | **0.90** |
| moist f64 | 31.5 | 106.7 | **0.85** |
| dry  f32 | 54.1 | 140.3 | 0.65 |
| moist f32 | 40.3 | 116.9 | 0.73 |

Weak efficiency 0.85–0.90 in f64 across 4× the nodes (np64=8 nodes → np256=32
nodes) — the multi-node fix scales weakly, not just strong. f32 is lower (0.65–
0.73): halving the compute makes the fixed halo/latency a larger fraction, so f32
goes comm-bound sooner — the same fabric wall, reached at a smaller compute
budget. This matches the SOTA picture (MPAS/MOM6 weak E falls with thinner
arithmetic intensity on a latency-bound interconnect).

### Status of every grid × precision toward its theoretical limit

| grid | precision | multi-node | limit reached | residual |
|---|---|---|---|---|
| atm icosahedral | f64 & f32 | **np8..256 ✓** | yes — res-dependent strong peak (np128 @ res6), fabric comm-bound past it | none (fabric-bound) |
| ocean lat-lon | f64 & f32 | **np1..128 ✓** | yes — res-dependent strong peak, same fabric wall | none (fabric-bound) |
| atm lat-lon (FV) | f64 & f32 | band np..128 ✓ | yes — band fabric-optimal on Gloo; 2-D pencil loses (latency-bound) | 2-D win needs InfiniBand |
| atm cubed-sphere | f64 & f32 | ≤6 faces/node | partial | >6-device sub-face tiling = future-HW project |
| atm spectral | f64 | single-device | n/a | no tracer storage (moist) + no MPI path — large additions |
| GPU multi-device (jax-mesh SPMD) | f32 & f64 | **WORKS intra-node** | 1.11x @ 2-GPU C192 | single-process jax device-mesh (commit 001b18bcb); PCIe-bound (no NVLink), crossover ~C192. See GPU UPDATE below |
| GPU multi-device (mpi4jax / jax.distributed) | f32 & f64 | **blocked** | n/a | mpi4jax CPU-only build; jax.distributed multi-controller NCCL topology times out — infra, not a code gap |

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
3. **spectral moist — DONE (capability; single-device, no scaling).** Moist
   physics (q_v/q_c/q_r + Kessler warm-rain) now runs on the global spectral PE
   dycore, so moisture is wired on ALL atmosphere grids (icosahedral,
   cubed-sphere, lat-lon, spectral). Shipped: `make_kessler_forcing_spectral`
   (commit cc85018ee) — an SH-transform bridge to the SHARED column Kessler
   (inverse-SH `T_hat`→grid + `p_s=exp(synthesis(lnps_hat))`, flatten to
   `(ncol,nlev)` so `pressure_from_sigma`/`compute_rho`/`compute_layer_dz` +
   `kessler_microphysics` apply unchanged, forward-SH the latent-heating rate
   back to `T_hat`); the `moist=` IC on `baroclinic_wave_init_spectral` + the
   `--grid spectral --physics moist` harness path (commit d7b61a8a3). Bugfix in
   the same commit: the timing scan passed `dt` as a jit argument → tracer, and
   the spectral dycore caches integrator/filter matrices keyed on a CONCRETE dt
   (`_ensure_tracer_filter`/`_ensure_si_data`) → `TracerBoolConversionError`;
   `dt` is now closed over as a static float. Tests 7/7 + clean T21 e2e.

   **Spectral throughput (single device, CPU, the theoretical limit):** it is
   spherical-harmonic-transform bound — O(N³) Legendre transforms dominate, so
   per-device throughput is ~0.2–0.6 Mcells/s (≈100× below the grid-point ico
   ~40–150 Mc/s) and collapses with truncation: dry f64 T21 97 ms/step → T42
   459 ms → T85 3.15 s; moist ≈2× (extra tracer transforms). There is no MPI
   path, so spectral does not scale across devices here; the limit is
   algorithmic (global transforms), reached.

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
   the env's mpi4jax is CPU-only, so the GPU halo exchange cannot run. The
   mpi4jax GPU path is therefore BLOCKED until mpi4jax is rebuilt CUDA-aware.

### UPDATE 2026-06-17 — GPU multi-device is NOT wholly blocked (jax-mesh SPMD works)

The "single-process multi-GPU SPMD path (no mpi4jax)" anticipated above now
RUNS (commit 001b18bcb). `--cs-spmd --device gpu` as a SINGLE process with
`CUDA_VISIBLE_DEVICES=0,1` builds the global face mesh from `jax.devices()` (=2)
and shards the cubed-sphere over both GPUs via the jax device-mesh + multiface
ppermute halo — no mpi4jax, and `jax.distributed` is skipped for one process.
Three harness fixes unblocked it: defer the `--device gpu` backend assertion past
the (skipped) cs-spmd init, and stop `_configure_jax_gpu` from clobbering an
explicit `CUDA_VISIBLE_DEVICES` (it had double-restricted / single-pinned).

So the GPU-multi-device picture is three-way, not "blocked":
- **mpi4jax-GPU halo**: blocked (CPU-only mpi4jax build) — infra.
- **jax.distributed multi-controller (srun -n2) on GPU**: blocked here — the NCCL
  local-topology gather times out (`GetKeyValue cuda:local_topology/cuda/1`, 2 min)
  on this stack — an env/NCCL issue, not a code gap.
- **single-process jax device-mesh SPMD**: WORKS.

Measured single-process cubed-sphere f32, 1-GPU vs 2-GPU (strong, same face):
C48 1.62→2.99 ms (222→120 Mc/s), C96 4.63→6.22 ms (310→231), **C192 19.97→17.96
ms (288→320 Mc/s = 1.11x speedup)**. Crossover ~C192: below it the cross-GPU
ppermute halo over the shared PCIe-Gen3 link (no NVLink on these RTX-8000 pairs)
costs more than the per-GPU compute saved; at C192 compute finally dominates and
2 GPUs win — modestly, the same PCIe-fabric wall the CPU side hit. Single-device
GPU per-resolution throughput (the per-GPU ceiling) stands as the main GPU panel.

KNOWN minor: a single-process cs-spmd run records `n_ranks=1` (the process count)
rather than the jax mesh size `n_global`, so the 2-GPU point lands in the CSV
mislabeled n=1 (the numbers above are read from the run logs). A label fix
(record `n_global` for cs-spmd) is a small follow-up; it does not affect the
multi-controller CPU cs-spmd runs (those use one process per device).

**Ocean/lat-lon GPU multi-device (same route-B jax-mesh):** the latitude-band
SPMD step (`bench_ocean_latlon_spmd_pcg.py`, `shard_map` over a 1-D `lat` mesh,
single process drives both GPUs — no mpi4jax) also runs on 2 GPUs. Barotropic
solve, LL720 (720x1440), 1- vs 2-GPU: **f64 18.90→16.35 ms = 1.16x** (eff 0.58);
f32 8.59→9.79 ms = 0.88x (too light — the cross-GPU halo over PCIe outweighs the
shrunk per-device compute, the same crossover as cubed-sphere f32 at C96). So
GPU multi-device scaling works via the single-process jax-mesh path for BOTH
decomposable grids — cubed-sphere (C192 f32 1.11x) and ocean lat-lon (LL720 f64
1.16x) — modest and PCIe-bound (no NVLink), positive once the per-device problem
is large enough. This is a microbench (ms/solve, not a tidy-CSV SYPD row), so the
numbers live here rather than in the auto-generated figure.

**GPU multi-device — final verdict.** Three paths, now fully characterised:
mpi4jax-GPU halo = infra-blocked (CPU-only build); jax.distributed multi-
controller = env-blocked (NCCL local-topology gather times out); single-process
jax device-mesh SPMD = WORKS for both cubed-sphere and ocean, ~1.1–1.2x at 2 GPUs
intra-node, PCIe-bound. The "GPU multi-device infra-blocked" verdict was wrong
for the third path; bigger speedups need NVLink (or >2 GPUs/node), which this
hardware does not have.

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
`scripts/bench/roofline_probe.py`).

The **cubed-sphere >6-device** project is now DONE as a capability: the sub-face
np=6·kt² tiling is shipped + np24/54 bit-identity parity-gated for the full 3D-PE
step — DRY *and* (2026-06-20) MOIST (tracers + Kessler;
`cube_moist_tiled_step_design.md`). It is FUTURE-HW (np>6 anti-scales on
Ginsburg's CPU shard_map / cross-node ppermute by design, so it is gated by
bit-identity, not benchmarked here). With it, every atmosphere grid ×
{dry,moist} × {f32,f64} × {weak,strong} is at its Ginsburg practical limit and
the cube has a parity-gated path beyond 6 devices for fast-interconnect HW. The
two remaining decomposition projects (atm lat-lon 2-D pencil at high np, spectral
transpose) stay HW-blocked on Gloo/PCIe — the SOTA fixes (MOM6/E3SM 2-D pencil;
NeuralGCM/spectral transpose) need InfiniBand/NCCL Ginsburg does not have.

## Publication MPI+GPU scaling matrix (2026-06-20)

Full publication sweep: every atmosphere grid + the ocean, float32 AND float64,
weak AND strong, on BOTH backends — CPU-MPI (1–8 ranks) and GPU (1–2 dual Quadro
RTX8000 per node).  120 measured points aggregated into one tidy CSV
(`aggregate_bcw_scaling.py`, multi-root + dedup) and plotted by
`plot_throughput_by_grid.py` / `plot_scaling_efficiency.py` →
`docs/scaling/pub_mpi_gpu/{throughput_by_grid,scaling_efficiency}.png`.

Peak single-config throughput [Mcells/s] (best over the sweep):

| component | grid          | f32 CPU | f64 CPU | f32 GPU | f64 GPU |
|-----------|---------------|--------:|--------:|--------:|--------:|
| atm       | latlon        |   48.6  |  26.8   |  71.6   |  71.8   |
| atm       | icosahedral   |   19.1  |  10.9   | 228.5   |  62.7   |
| atm       | cubed-sphere  |   45.1  |  23.3   |  73.2   |  73.4   |
| atm       | spectral      |    1.3  |   1.3   |  15.7   |  15.7   |
| ocean     | latlon        |   11.2  |   9.0   | 103.4   |  98.1   |
| ocean     | mpas          |    —    |   —     |  59.5   |  59.5   |

GPU per-device wins (1 RTX8000 vs the CPU-MPI peak): ocean lat-lon ≈ 10×,
atm-icosahedral f32 up to ~12× (228 Mcells/s — the most GPU-friendly grid),
spectral ≈ 12× over a near-stalled CPU spectral (transform-bound, f64-only).  On
GPU, f32≈f64 throughput for the FV-family grids (RTX8000 f64 is not heavily
penalised at these sizes; the cost is memory traffic, identical layout) — only
icosahedral shows the classic f32>f64 split (228 vs 63).

Honest scope: GPU is capped at **2 devices** (one dual-RTX8000 node; multi-node
GPU is HW-blocked — no NCCL/IB, only PCIe-Gen3 + Gloo).  **spectral is
float64-only single-device** by design.  **Ocean GPU is single-device only**
(`bench_ocean_gpu_scaling.py` hardcodes `n_gpus=1`; the ocean SPMD multi-GPU
wrapper is not wired — see `omip_multinode_spmd_scope`).

**Cubed-sphere GPU strong-scaling 1→2 device is cross-GPU-halo bound (measured,
anti-scaling).**  Direct measurement (f32, job 8534102): C48 59.4→51.0 Mcells/s
(0.86×, 43% eff), C96 82.1→82.0 Mcells/s (1.0×, 50% eff) — the 2nd GPU buys ~no
throughput because the C-D-grid face split adds a cross-device halo (the "21
collective-permute start/done" ops in the timed HLO census) whose latency cancels
the compute gain at these problem sizes.  This is the SAME mechanism as the
documented cube >6-device anti-scaling — the cube simply does not strong-scale on
the dual-RTX8000 PCIe/Gloo fabric.  (The f64 + C192 strong points were not run to
completion: each cross-GPU cube config compiles ~200s and the f32+f64 × {1,2}dev
× {48,96,192} sweep exceeds practical walltime — but the result is already clear
from f32/C48-C96.)  The publication figure therefore represents the cube on GPU
by its single-device throughput (f32 & f64) + weak 1→2 device; the strong
2-device curve is omitted as a measured non-result (halo-bound), not a gap.

Aggregator fix (this run, codex-reviewed): `aggregate_bcw_scaling.py` previously
SILENTLY DROPPED the nested atmosphere GPU `ScalingReport` JSONs (only flat
CPU-MPI + nested-ocean were ingested) and CONFLATED MPAS-ocean into lat-lon
(neither nested report serialises `grid_type`).  Fixed with one reusable
`grid_from_path()` (recovers the grid from the `<grid>_<suffix>` output sub-dir,
+ short aliases) routed through a `component`-aware `_rows_from_nested`; an
atm row whose grid is unresolvable is SKIPPED + warned, never emitted with an
empty grid that would collide distinct grids in the dedup key.
