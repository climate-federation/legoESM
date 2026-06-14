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
