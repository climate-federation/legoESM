# Scaling-lever audit (2026-06-14) — codex adversarial, ranked by ROI

Systematic adversarial review (codex gpt-5.5, job-less `codex exec`) of the
HIGHEST-ROI weak/strong-scaling levers (MPI + multi-GPU) NOT yet exploited,
across ALL grids (atm cube SW/3D, lat-lon, spectral; ocean lat-lon C-grid,
MPAS/Voronoi).  Distinguishes BENCHABLE-on-Ginsburg (CPU nodes ≤32 ranks
Gloo-TCP no-IB; 2× RTX8000 PCIe no-NVLink) from FUTURE-HW-only.

## Ranking (achievable-on-Ginsburg first)

1. **Ocean lat-lon 2-GPU SPMD band path** — HIGH, benchable. Both RTX8000s sit
   UNUSED for ocean today (`scripts/bench/bench_ocean_gpu_scaling.py` is
   single-device, and it's MPAS not lat-lon). The lat-lon band decomp has the
   cleanest comm (N/S ppermute halos + psum barotropic reductions). Generic lat
   sharding exists (`sharded_dynamics.py:249` `P("lat",...)`) but there is NO
   ocean-safe SPMD halo/barotropic path. **Biggest "unused hardware today"
   lever → THE next major unit.**
2. **MPAS batched-halo repair** — HIGH, CPU-MPI. The 18× regression
   (`voronoi_mpi.py:76`, `halo_exchange_voronoi.py:384/420`) is likely
   pack/scatter overhead (per-field/per-neighbor gather+concat+split+scatter),
   NOT fundamental data volume. Needs compute-node profiling of
   pack/exchange/unpack; fix = persistent packed buffers / entity-group
   coalescing without flat-concat churn.
3. **MPAS METIS partition by default in scaling runs** — MED-HIGH, CPU-MPI. RCB
   is default; METIS exists (`voronoi_partition.py:204/422`). Better cut →
   fewer halo cells/neighbors/imbalance on coastal MPAS. Cheap to A/B.
4. **MPAS operator fusion / Pallas for indirect gathers** — HIGH, single-GPU
   first. MPAS is gather-heavy (`profile_mpas_ocean.py`, `ocean_pe_mpas.py:357`);
   fuse repeated edge/cell gathers.
5. **Ocean lat-lon phase-level halo coalescing** (baroclinic/tracer/GM/VMix) —
   MED, CPU-MPI. Pack same-stage N/S halo fields (barotropic reductions already
   done).
6. **Tripole full-model MPI wiring** — MED, CPU-MPI after a parity gate
   (`bench_ocean_mpi_scaling.py:2091`). Unlocks eORCA-style measurements.

## Do NOT (this HW)
- 2-D lat-lon decomposition on Gloo (measured to HURT; latency fabric — keep for
  IB/future only).
- Cube tiled-production-step assembly / more cube tiling for SPEED here: np>6
  anti-scales on Ginsburg. It is FUTURE-HW capability (TPU pod/NVLink), already
  bit-identity-validated; WIRING into `make_sharded_step` remains but yields no
  Ginsburg speedup. (`sharded_dynamics.py:749`.)
- Comm/compute overlap: blocking sendrecv only; no nonblocking mpi4jax →
  future-HW.

## Code-review note (my recent tiled ops)
Tiled stages keep face-full inputs replicated across tiles + global pre-pad
outside shard_map (correct for bit-identity; weak-scaling upside could be eaten
unless production wiring avoids repeated full-face materialization). FUTURE-HW.

## Decision
Pivot the campaign's ACHIEVABLE-scaling effort to **ocean lat-lon 2-GPU SPMD**
(lever 1) — attacks unused hardware, benchable on the RTX8000 PCIe pair. The
cube-3D tiling (future-HW capability) continues as a secondary track. MPAS
batched-halo repair (lever 2) is the next CPU-MPI lever after a profiling pass.

## MEASURED OUTCOMES (2026-06-14, lever #1 executed end-to-end + remaining-headroom re-audit)

Lever #1 (ocean lat-lon 2-GPU SPMD) executed in full via **route-A** (overlay
venv: cuda jax 0.9.1 + CUDA-built mpi4jax + mpi4py over the RTX8000 PCIe pair;
existing validated MPI harness, near-zero new code) rather than the pure-jax
shard_map (blocked: `LatLonCGridOceanModel` is not a pytree and `step()` does
host-side CFL `float()` logic). Foundation also shipped: lat-band SPMD halo +
barotropic-PCG `psum` routing (commits b51f58b9→246df03f).

**2-GPU strong (route-A, host-staged MPI = lower bound):**
- Ocean full-step: eff 0.62 (180×360 n30) → **0.92** (360×720 n60). NEAR-IDEAL
  at production scale — the 3D baroclinic+tracer compute (barotropic only
  12.4%/7.4% of the step at n30/n60, phase-profile 8486138) amortizes the
  PCIe barotropic-PCG penalty. The barotropic-PCG KERNEL alone anti-scales
  (eff 0.31) — confirming the wall is the global reduction, hidden at scale.
- Atm lat-lon FV PE: eff 0.72 (90) → **0.82** (180×360 n26). Halo-bound, no
  ocean-style barotropic reduction wall.

**2-GPU weak (ocean):** eff 0.46 (n30) → 0.57 (n60) — lower than strong; weak
exposes the fixed 2M-allreduce/step barotropic latency (the Amdahl term).

**Levers ruled OUT (measured):**
- cuda-aware MPI: env-blocked (conda mpi4py bundles its own openmpi that
  shadows the cuda-aware module + segfaults in mca_coll_cuda; needs a full
  MPI-stack rebuild — not worth it, production already 0.92-0.95).
- Reduction-cutting preconditioners (zonal_line/chebyshev M-cut, codex
  remaining-headroom audit's "last lever"): the M-cut is real (zonal_line &
  cheby8 reach jacobi-M60 accuracy at ~M/3 = 3× fewer reductions, conv
  8486241) but it does NOT win wall-time on this HW — zonal_line/M20 np2
  13.20ms vs jacobi/M60 12.68ms (job 8486251): the per-iter cyclic-Thomas
  LOCAL cost exceeds the reduction-latency saved (roofline: halos≈reductions
  on Gloo/TCP at moderate ranks). chebyshev adds halos (worse) + a
  bench-scan dtype bug.

**VERDICT: all Ginsburg-measurable scaling axes at their practical limit.**
2-GPU near-ideal at production scale; per-device at JAX-peer; CPU-multinode
reduction-latency-bound with no cheap M-cut. The only remaining THEORETICAL
lever is a **MULTIGRID barotropic preconditioner** (cut M to ~O(log n) with
cheap V-cycle restriction/prolongation halos instead of the heavy
cyclic-Thomas) — a major multi-week build, surfaced as a DECISION. Further
gains otherwise need NVLink/IB hardware.

## MULTIGRID POC — last theoretical lever RULED OUT (2026-06-14, job 8486324)

Built a recursive geometric 2-/4-grid V-cycle barotropic preconditioner POC
(scripts/tmp/_poc_multigrid_barotropic.py; weighted-jacobi smoother,
full-weighting restriction ~0.25·Pᵀ, piecewise-constant prolongation,
re-discretized coarse Helmholtz) and measured M-to-tol vs jacobi on the stiff
convergence-char operator.

  rel-residual at M (96×192, coeff 5e7):
    jacobi   M16=1.87e-2  M60=1.41e-3
    4-lvl MG M16=2.74e-3  M60=3.83e-7   (≈ identical to 2-grid: M16=2.93e-3)

MG reaches jacobi-M60 accuracy (1.4e-3) at only ~M18 — the SAME ~3× M-cut as
zonal_line/chebyshev — and adding levels (2→4) barely helped. At 25 halos per
outer iter, the cost at equal accuracy is ~450 halos+36 reductions (MG M18) vs
60 halos+120 reductions (jacobi M60) ⇒ ~2.7× SLOWER on the Gloo/TCP/PCIe
roofline (halo≈reduction).

ROOT CAUSE: lat-lon POLAR ANISOTROPY (dx→0 near the poles) defeats the
pointwise-jacobi smoother — the textbook failure mode of naive geometric
multigrid on lat-lon grids. The O(log n) iteration cut needs LINE/ZEBRA
smoothers + SEMI-COARSENING (the reason POP/MOM ship dedicated barotropic
solvers) = research-grade, and its payoff is only the non-production small/weak
comm-bound tail (production barotropic is already amortized to 7-12% of the
step). NOT worth the build on this hardware.

### FINAL VERDICT
All Ginsburg-measurable scaling axes are at their practical limit. Every
accessible barotropic preconditioner lever (jacobi/zonal_line/chebyshev/
geometric-MG) gives ≤3× M-cut, none cheap enough to win on Gloo/TCP/PCIe.
Production 2-GPU is near-ideal (ocean 0.92-0.95, atm 0.82). Further gains
require NVLink/IB hardware or a research-grade anisotropic-MG solver with
payoff only on the non-production tail.
