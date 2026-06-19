# Scaling campaign — resume state (2026-06-13)

Branch: `omip-faithful-nemo-comparison` · PR: **#435** (all campaign commits
pushed; this doc is the resume pointer). Full blow-by-blow is in the auto-memory
`scaling_campaign_ginsburg.md`; per-axis closeness is `distance_to_limit_2026-06-13.md`.

## Where we are: at/near the limits on the high-ROI axes

- **Atm cubed-sphere per-device** — JAX-peer limit (7.2e-9 s/cell/step); per-device fusion closed.
- **Atm 2-GPU strong** — PCIe-link roofline (eff 0.73); more needs NVLink/IB (absent on Ginsburg).
- **Atm production multinode SPMD** — SHIPPED (`distributed_mode='spmd'`); C96 np6 2.42×, scales with size.
- **Ocean barotropic reductions** — NEUTRALIZED (zonal_line + single_reduce, opt-in; regime-dependent).
- **Ocean halos** — 63 → 32 per-step N-S exchanges (−49%) across the campaign.

## Shipped this campaign (all codex-reviewed, gated, on PR #435)

| commit | what |
|---|---|
| adaca24f | P4 tiled-d2a2c in-stage strip ppermute (np>6 sub-face exchange) |
| 48d5c8af | production cubed-sphere multi-controller SPMD (`distributed_mode='spmd'`) |
| f0c27da1 | cadence-less runs no longer collapse to 1-step segments (serial 5×) |
| 65b29082 | atm F1: pack T+ln(ps) SPMD halo into one collective |
| a3b9c597 | ocean zonal_line preconditioner (cyclic-Thomas, comm-free) |
| 1f142625 | ocean vertex-mask hoist out of the traced step |
| 1a0099a0 | barotropic levers documented OPT-IN (regime crossover) |
| e39b027d | distance-to-limit assessment |
| 7a7100d6 | neumann-fill field+mask halo packed via dtype cast (census 38→32) |
| 36861ecb | lat-lon multi-GPU state shard_state grid_type fix + loud tiled-halo skip (codex P1) |
| 512ab62b | parallelization literature review |

## Codex review (2026-06-13) — all P0/P1 resolved

- P0 MPAS `barotropic_implicit_pcg_variant`: already present (mpas_config.py:261, concurrent session).
- P1 CS-atm replicated dynamics: shipped as `distributed_mode='spmd'` (48d5c8af).
- P1 lat-lon multi-GPU state sharding + P1 tiled silent-slow: FIXED (36861ecb).
- P2 single_reduce default: held OPT-IN (production-tile data: barotropic 6-10% of step, neutral).
- P2 lat-lon 2-D decomp + voronoi batched-halo, P3 spectral label: roadmap (task #7; spectral already "by design").

## In flight at handoff

- SLURM **8476361** (`ocn_fails`): full `tests/ocean/unit` under xdist to NAME the
  30 failures from the consolidation run (job 8476335: 2364 passed / 30 failed).
  Partial output already shows the cause = `mpi4py.MPI` loaded + "Fatal Python
  error: Aborted" → MPI-runtime-needing tests run SERIALLY (no `mpirun`) segfault
  a worker = the known non-serial-safe class, **NOT a campaign regression** (every
  touched file passed its targeted gate). Confirm names in
  `results/scaling_ginsburg/ocn_fails_8476361.txt` on resume; if any failure is in
  a campaign-touched file, that's the only thing to fix before merge.

## Next levers (ranked, when resuming) — all single-digit-% or hardware-blocked

1. **Ocean baroclinic neumann static-mask precompute** (deeper than the dtype-cast):
   the 3-pass fill's per-pass MASK sequence is static (land-mask-only) — precompute
   + thread to drop the mask halos (6→3/site), OR collapse 3 passes into one
   precomputed fill operator + single halo=3 (6→1/site). Needs MPI-band/pole/
   tripole-fold correctness + production-tile A/B. Ceiling bounded (baroclinic 16-21%).
2. **Ocean vmix memory-traffic** (THROUGHPUT not scaling — shards perfectly, 45-61%
   of step): fuse K-profile build into the Thomas solve, cut intermediates. (task #12)
3. **Lat-lon 2-D decomposition** (atm+ocean strong limit at high ranks): structural. (task #7)
4. **Multi-node GPU SPMD**: architecture proven (A1) but interconnect-capped on
   Ginsburg PCIe — **defer to NVLink/IB hardware**, not an engineering gap here.

## How to resume

1. `git checkout omip-faithful-nemo-comparison && git pull` (PR #435).
2. Read `scaling_campaign_ginsburg.md` (auto-memory) — it loads automatically.
3. Check 8476361's failure names (above) — confirm pre-existing.
4. NOTE: 4 tracked files are uncommitted on the branch from a CONCURRENT
   distributed-MPAS session (`barotropic_implicit_mpas.py`, `mpas_config.py`,
   `test_distributed_barotropic_pcg.py`, `ocean_faithfulness_nemo.md`) — NOT this
   campaign's; do not commit them here.
5. Recurring gates: `scripts/cluster/scaling_ginsburg/*.sbatch` (all `--account=glab`,
   compute-node only — never run heavy work on the login node).
