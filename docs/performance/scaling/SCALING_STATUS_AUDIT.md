# Scaling status audit — authoritative support matrix (2026-07-07)

Single source of truth for the weak/strong scaling story across atmosphere +
ocean, MPI + multi-GPU, all grids. Documentation-only audit: reconciles the
campaign docs against the benchmark harnesses as they exist in the tree today.
When another scaling doc disagrees with this file, this file wins; fix the
other doc.

## Terminology — four distinct claims, never conflated

| tag | meaning | valid evidence |
|-----|---------|----------------|
| **(a) true multi-device scaling** | weak/strong efficiency measured across ≥2 real devices or MPI ranks | speedup/efficiency vs device/rank count |
| **(b) single-device throughput-vs-size** | one GPU/CPU, problem size swept; a saturation curve, NOT device-count scaling | Mcells/s vs cells; roofline % |
| **(c) infrastructure-ready, unmeasured** | code path exists + parity/conservation-gated, no production-scale numbers yet | tests + smoke runs only |
| **(d) unsupported / N/A** | no multi-device path, by design or unbuilt | — |

`scaling_gpu.md`'s "strong/weak" plot names are (b) — its errata note 4
already says so. Any doc calling a single-GPU sweep "weak scaling" without
that qualifier is wrong.

## Atmosphere support matrix

| grid | CPU-MPI | GPU / SPMD | true weak/strong evidence (a) | known blockers | next measurement |
|------|---------|------------|-------------------------------|----------------|------------------|
| **spectral** | (d) single-rank only (global Legendre transforms; no MPI path) | (d) N/A — both multi-device schemes measured anti-scaling (`spectral_level_shard_cliff.md`); fp64-only | none possible | O(N³) global transform | none — stays N/A unless a GPU-native SHT effort (SHTns/sphericart) is explicitly launched |
| **cubed-sphere** | (a) genuine ≤6-face decomposition via `run_levante_gpu_scaling.py --cs-mpi-scatter` (bit-equal 1e-15 vs serial); default (no flag) is replicated dynamics, refused for scaling claims. Gloo/TCP multinode anti-scales (fabric, not code — `multinode_clean`) | (a)≤6 devices: face-sharded SPMD (`--cs-spmd`, single-process or multi-controller NCCL); Derecho/Levante NCCL job lanes exist (`scripts/cluster/scaling_*`) | 2-GPU strong 0.73–0.83 eff (Ginsburg, AT the PCIe roofline of that host) | >6 GPUs: (c) sub-face tiled production lanes wired (`_run_tiled_cube_spmd` blocked loop, per-segment `lax.scan` since M3b inc-1 — see lever 7) but NO >6-GPU hardware receipts; envelope = default-config dycore + Kessler / operator-split unified physics | production-size Derecho/Levante multi-node NCCL runs incl. the >6-GPU tiled lanes |
| **icosahedral / MPAS** | (a) graph-partition domain decomposition (METIS/RCB/SFC `auto`), validated vs serial ~1e-9 | (a) multi-GPU via route-A mpi4jax (opaque to XLA overlap); multi-controller NCCL SPMD not yet wired (cell-partition reorder across processes) | Derecho 2026-07-02: CPU-MPI to 128 ranks f32 ~75× (eff ~0.59); GPU 1→16 A100 @28 km ~6× (eff ~0.38), coarse grids flat (per-device floor, not a defect) | route-A mpi4jax leg is latency-bound, no comm/compute overlap | icosahedral multi-controller NCCL SPMD; re-measure 8→16 GPU leg on native ppermute |
| **lat-lon** | (a) latitude-band decomposition (`make_latlon_mpi_step`, #641; pole_bc='wall'), validated; driven by `run_levante_gpu_scaling.py --grid latlon` | (a) lat-band SPMD `bench_atm_latlon_spmd_scaling.py`: single-process multi-device + `--multicontroller` route-B (native NCCL ppermute, no mpi4jax) | Derecho: CPU-MPI 128 ranks f32 ~30× (eff ~0.23 — 1-D band perimeter cost, as designed); GPU 16 A100 @28 km 7.5× (eff ~0.47, route-A). Ginsburg 2-GPU: strong 1.10×, weak eff 0.64 (PCIe-capped) | 1-D band decomposition perimeter at high rank counts; 2-D latlon decomposition untested on a real fabric | production-size native-NCCL multicontroller runs on Derecho/Levante (job lanes C exist) |

## Ocean support matrix

| grid | CPU-MPI | GPU / SPMD | true weak/strong evidence (a) | known blockers | next measurement |
|------|---------|------------|-------------------------------|----------------|------------------|
| **lat-lon C-grid** | (a) latitude-band MPI, `bench_ocean_mpi_scaling.py` (parity + conservation gates; `--wet-balance`; distributed fixed-M PCG / single_reduce / preconditioner options) | (a) full-step SPMD `bench_ocean_latlon_spmd_scaling.py` (single-process multi-device AND `--multicontroller` NCCL; parity + conservation gates); jit-once sharded step | 2-GPU full step: strong 0.92 eff at production size, weak 0.97 @ ~590k cells/rank (Ginsburg). CPU-MPI strong np16→32 eff 0.55 (implicit-CN reduction wall; split-explicit + local clamp opt-in 1.15–1.65× at ≥2 nodes) | implicit-CN allreduce wall at high ranks; land-cell load imbalance | Derecho/Levante A100 ladders (jobs exist, unrun); wide-halo split-explicit A/B at ≥16 ranks; wet-cell-balanced partitions at scale |
| **tripole / eORCA** | (c) operator-level fold-aware band-MPI halo machinery validated (`tests/distributed/test_latlon_mpi_tripole.py`); full-model MPI step NOT validated — `bench_ocean_mpi_scaling.py --tripole` refuses loudly | (d→c) `make_sharded_ocean_step` raises `NotImplementedError` on a fold grid; single-GPU throughput measured only (b: 313 Mc/s fp32, `SCALING_SUMMARY.md`) | none | SPMD/full-model fold wiring | wire + parity-gate a full `LatLonCGridOceanModel.step` tripole MPI case, then the SPMD fold |
| **MPAS / Voronoi** | (c) bench lane EXISTS (`bench_ocean_mpas_scaling.py`: multi-rank `MPASOceanModel.step` on `layout.local_mesh`, parity + conservation gates, partition metrics, M1 fused-scan/wet-cell/residual fields, `--halo-refresh` per-step packed exchange). NOT stage-correct: owned-only reductions PASS everywhere, but no per-STAGE halo refreshes exist (only the implicit-PCG matvec exchanges); rows carry `stage_halo_correct=false`. NOTE: `make_voronoi_mpi_step` is the ATMOSPHERE step — no ocean equivalent exists. See `mpas_ocean_distributed_stage_audit.md` | (b) single-GPU throughput only (333 Mc/s fp32 impl_cn) | none | within-step halo staleness (stage-level refreshes are real infrastructure, scoped in the stage audit) | np≥2 CPU-MPI receipts on the gated lane; then per-stage refreshes (stage-audit "Deferred" list) |
| **cubed-sphere** | (c) supported in `run_omip.py`; generic distributed layout, MPI conservation tested; no dedicated scaling lane | (c) same | none | no lane | only if a science driver demands it |

## CRM / LES (plane dycore) — see `crm_les_scaling.md`

True (a) evidence exists at single-socket CPU-MPI scope: CRM domain
decomposition fast (123× eager→jit) + bit-identical (1e-15) + both precisions;
spectral-LES MPI complete (distributed FFT/projection/LASD/3-2-rule, AD-safe)
with the algorithmic FFT all-to-all ceiling. GPU numbers there are (b).
Multi-node CRM/LES is unmeasured (c).

## What the headline single-GPU docs are

- `SCALING_SUMMARY.md`, `scaling_gpu.md`, the `crm_les_scaling.md` GPU tables:
  **(b)** throughput-vs-size on one RTX 5090 laptop GPU. Ratios within a run
  reliable; absolute numbers thermally drift; no device-count scaling claim.
- `derecho_levante_sota_review_2026-07.md` §3b: **(a)** measured Derecho
  CPU-MPI + multi-GPU curves (route-A) — the current best true-scaling
  evidence at scale.
- `docs/scaling/atm_latlon_spmd_scaling.md`, ocean SPMD gates: **(a)** at
  2-GPU / virtual-device scope.

## Improvement candidates (ranked; NONE implemented here)

1. **Documentation/status unification** — this file + the stale-claim fixes
   (done in this audit).
2. **Production-size Derecho/Levante multi-GPU runs on native NCCL SPMD**
   (atm latlon/cube lanes + ocean SPMD ladder; job scripts exist, receipts
   don't). Highest-value missing *measurement*.
3. **Ocean wide-halo split-explicit barotropic** (halo width = substep count →
   one exchange/baroclinic step) — evaluate at ≥16 GPU ranks; net-negative at
   Ginsburg scale, calculus flips where per-message latency dominates.
4. **Ocean active/wet-cell compaction + wet-balanced partitioning** (~2× on
   ~40%-land grids; `--wet-balance` bands are the partial groundwork).
5. **Tripole/eORCA SPMD fold wiring** (full-model MPI parity case first).
6. **MPAS-ocean scaling lane** — lane DONE (`bench_ocean_mpas_scaling.py`,
   M1 contract fields since scaling-M3d inc-1); remaining: the stage-correct
   distributed step itself (`mpas_ocean_distributed_stage_audit.md`
   "Deferred" list — per-stage packed refreshes, wide-halo explicit
   substeps, freshwater owned-mask threading).
7. **Cubed-sphere atm sub-face tiled production step** — THE lever for >6
   GPUs.  STATUS 2026-07-13 (M3b increment 1): the blocked persistent tiled
   loop is production-dispatched (`_run_tiled_cube_spmd`, `6*kt^2` devices)
   and now advances each segment inside ONE compiled `lax.scan`
   (`tiled_step_adapter.scan_tiled_cc_steps`) — carry persistently
   tile-sharded across timesteps, HLO-gated all-gather-free, D-grid<->cell
   layout conversions only at segment boundaries
   (`tests/parallel/test_cube_tile_native_segment.py`).  Envelope:
   default-config dycore (all post-step damps zero) + Kessler column
   physics on the simple lane; unified physics on the operator-split tiled
   lane (still one jit dispatch per STEP — segment-scanning it, with
   forcing as scan xs, is the follow-up), non-default damping / duogrid
   refused loudly.  Remaining: >6-GPU hardware receipts (tier (c) until
   then), operator-split lane scan, multicontroller route-B.
8. **Message aggregation/overlap for the latency-bound GPU legs** (f64 ≈ f32
   speedup curves on Derecho ⇒ latency-, not bandwidth-bound; aggregation and
   native-NCCL overlap, not compression, are the levers).
9. **Spectral atm: keep multi-device N/A** unless a GPU-native transform
   (SHTns/sphericart) effort is explicitly launched.

## Superseded / corrected claims (with sources)

- "atm lat-lon MPI not implemented (#115)" (`SCALING_SUMMARY.md` §4, old) —
  WRONG since #641: lat-band MPI is implemented + validated in
  `run_levante_gpu_scaling.py`. Fixed there.
- "no automated ocean full-step multi-GPU harness" / "no single ocean script"
  (`REAL_HARDWARE_SCALING.md` §4, old) — superseded by
  `bench_ocean_latlon_spmd_scaling.py` (GPU/SPMD) +
  `bench_ocean_mpi_scaling.py` (CPU-MPI). Fixed there.
- "`bench_ocean_mpi_scaling --transport spmd --multicontroller`"
  (`derecho_levante_sota_review_2026-07.md` §4 item 0) — that flag does not
  exist on that script; the ocean multicontroller lane is
  `bench_ocean_latlon_spmd_scaling.py --multicontroller`. Corrected there.
- "multi-node `jax.distributed`, not yet wired"
  (`docs/scaling/atm_latlon_spmd_scaling.md`) — wired since:
  `bench_atm_latlon_spmd_scaling.py --multicontroller`. Corrected there.
