# Ginsburg MPI/GPU scaling campaign — plan (2026-06-10)

Goal: weak + strong scaling assessment of atmosphere (AMIP) and ocean (OMIP)
on multi-CPU (MPI) and multi-GPU, all grid types, float32 + float64; push to
near-roofline without correctness regression; table + PNGs as deliverables.

## Hardware (Columbia Ginsburg)
- CPU nodes: 32 cores/node, `short` partition 12 h, `burst` up to 14 d, account `glab`.
- GPU nodes g0xx: 2× Quadro RTX 8000 (46 GiB) per node, CUDA 12.7, JAX 0.9.1.
- MPI: `module load openmpi/gcc/64/4.1.7a1`, launch `srun --mpi=pmix`.
- Python: GPU jobs → jn2808 conda env; CPU-MPI jobs → `~/.venvs/legoesm-mpi`
  (mpi4py + mpi4jax 0.8.1.post2; jax 0.9.1 — version-warn only).
- PYTHONPATH = `src` + every `packages/<pkg>` (PEP 420 namespace merge).
- Persistent JIT cache: `LEGOESM_JAX_CACHE_DIR=/burg/glab/users/pg2328/jax_cache_scaling`.

## Existing harness (baseline, unmodified)
- Atm: `scripts/bench/run_levante_gpu_scaling.py` — grids
  {spectral, cubed-sphere, icosahedral, latlon}, `--mode both`,
  `--precision both`, physics {none, held_suarez, gray_sbm, rrtmg_full};
  emits CSV/JSON + weak/strong PNGs.
- Atm CPU MPI: `scripts/bench/run_cpu_mpi_scaling.py` — one case per mpirun
  invocation; **true multi-rank only for icosahedral**. Direct execution =
  one rank count AND one resolution (`--mode strong` does NOT sweep the
  I4/I5/I6 ladder — pass `--resolution` explicitly or use `--sweep` cases).
- Ocean: `scripts/bench/bench_ocean_gpu_scaling.py` — single-GPU only,
  grids {latlon C-grid, MPAS}; reuses TimingResult schema.
- Plots: `scripts/plot/plot_gpu_scaling.py` + per-run PNGs from atm harness.

## Known gaps (= the "improve the code" queue, each gated on codex review)
P1. Lat-lon MPI in benches (design corrected per codex review 2026-06-10):
    core `make_latlon_mpi_step(model, layout, *, halo=2)` exists (no longer
    NotImplementedError — bench call shape is stale). Wiring recipe:
    `layout = initialize_distributed_latlon(...)` →
    `slice_latlon_grid_to_band` → build rank-local model on the band grid →
    `scatter_state_latlon` → `make_latlon_mpi_step(local_model, layout)`.
    `make_latlon_mpi_step` hardcodes `physics_fn=None` (latlon_mpi.py:1673):
    add a `physics_fn` passthrough + serial==MPI parity test (held_suarez)
    BEFORE benchmarking physics tiers. Also: latlon GPU multi-device is
    blocked in the bench (`_valid_gpu_counts()` → [1]) although production
    AMIP runs SPMD — wire it.
P2. Ocean MPI scaling harness must be rank-local from construction (codex):
    `slice_cgrid_geometry_to_band` + `scatter_state_latlon_cgrid_ocean`
    BEFORE model creation — the lat-lon C-grid ocean model caches static
    structures (rigid-lid islands, ocean_model_latlon_cgrid.py:544) at
    build time. Tripole fold: `ParallelRuntime.create(grid_type="latlon")`
    does NOT plumb `global_n_lon`/`fold` — either extend it or call
    `initialize_distributed_latlon(..., fold=...)` directly. Weak + strong,
    f32/f64, MPAS ocean second.
P3. Cubed-sphere CPU-MPI: halo support exists (iter 3) but driver-side
    state scatter missing → dynamics replicated per rank. Implement
    scatter/gather so cubed-sphere CPU MPI is true domain decomposition.
P4. Multi-node GPU (2 nodes × 2 GPUs = 4 GPUs) via `jax.distributed` +
    `srun`; check NCCL/IB settings on Ginsburg.
P5. Spectral grid: transforms are global; document MPI N/A (no fake path).

## Matrix
Axes: model {atm-AMIP(gray_sbm), ocean-OMIP} × grid
{cubed-sphere, latlon(+tripole ocean), icosahedral/MPAS, spectral(atm only)}
× backend {CPU-MPI ranks 1,2,4,8,16,32; GPU 1,2,(4 multi-node)} ×
mode {weak, strong} × precision {f32, f64}.

## External reference anchors (user-supplied figures, digitized by eye)
- CliMA atmosphere (JAMES Fig. 13, moist baroclinic wave, H100/A100):
  103 km @ 1 GPU ≈ 6 SYPD; 51 km @ 1 GPU ≈ 1.5 SYPD. Our C96/C192 ladder
  aligns with their 103/51 km curves → `scripts/plot/plot_scaling_vs_clima.py`.
- Oceananigans (Silvestri et al. Fig. 10/11, A100, f64, 100 levels):
  1/4° @ 1 A100 ≈ 170 ms/step ≈ 610 Mcells/s/GPU (incl. vertical);
  weak scaling flat ~0.85 s/step from 4→1024 GPUs at fixed 1/12°-equiv
  per-GPU patch → `scripts/plot/plot_ocean_vs_oceananigans.py`.
  Ginsburg context: RTX 8000 vs A100 ≈ 19× f64 FLOPs, ≈ 2.6× BW —
  compare in f32 AND f64; per-device Mcells/s is the dt-free metric.
  Baseline 2026-06-10: legoESM latlon ocean f64 ≈ 95 Mcells/s/GPU (LL96),
  i.e. 6.4× below Oceananigans-A100 with a 19× f64-FLOP handicap →
  bandwidth-roofline check pending f32 results.

## No-regression gates
- Existing thresholds in atm harness (compile ≤ 30 s, strong eff ≥ 0.12,
  weak step growth ≤ 2.5, weak per-device throughput ratio ≥ 0.35).
- `tests/distributed/` MPI parity + differentiability suites must stay green
  (run on compute node) after each code change.
- Ocean matrix conservation drifts (heat/salt/eta ≤ 1e-3) unchanged.
- f64 vs f32: identical scaling shape; no silent dtype downcast.

## Process
1. Submit baselines on unmodified code (atm GPU, atm CPU-MPI ico, ocean GPU).
2. Codex adversarial review of this plan + each P-item design BEFORE
   implementation (user directive). Then implement → codex review diff →
   fix → re-review until clean.
3. Re-run affected scaling jobs; regenerate PNGs; iterate toward limits.
4. Final summary table (per model × grid × backend × mode × precision:
   time/step, SYPD, Mcells/s, speedup, efficiency) + weak/strong PNGs.

## Session addendum — fused halo + static-metric fold (2026-06-10 evening)

Shipped (codex-reviewed x2, gates green job 8459396):
- Fused multi-field N-S halo exchange (`pad_with_pole_bc_lat_multi` /
  `pad_ns_zero_multi` / `interp_to_v_points_multi`): one sendrecv pair
  per cut per dtype group instead of one per field.  mpi4jax sendrecvs
  are token-serialized, so every fused pad saves a full message latency.
  Env A/B: `LEGOESM_LATLON_FUSED_HALO=0` restores per-field.
- Static-metric trace-time constant folding (`_pad_static_wall_bc_mpi`):
  concrete 1-D metric pads (grid.lat / sin rows / dy) exchange ONCE at
  trace via host mpi4py and become compile-time constants — including
  the TWO pads inside the barotropic PCG `A_op` loop body (=120
  executed sendrecvs/step at M=60).  Kill-switch:
  `LEGOESM_LATLON_STATIC_METRIC_PAD=0`.
- Census probe `scripts/tmp/_probe_halo_census.py`: 63 -> ~30 traced
  exchanges/step (np=2 LL64 implicit_cn).

Measured A/B (job 8459326, f64, implicit_cn/PCG):
  strong LL128: np4 1.32x, np8 1.58x (eff 0.35->0.55), np16 1.28x
  strong LL192: np4 1.37x, np8 1.48x
  weak rows/rank=48: np8 growth x4.65 -> x2.95 (1.61x); np1 0.99x
PNGs: laws_v2.png (before) vs laws_v3.png (after); fused_halo_ab.png.

Parity-gate lesson: the np>=2 implicit_cn lane runs the fixed-M PCG
while the old serial reference ran stock CG — the gate had only ever
been validated on explicit_substep.  Fixed via
`barotropic_implicit_force_pcg` (solver-matched reference) + a
lane-aware tolerance (PARITY_TOLS_PCG_MPI, ~2.5x above the measured
allreduce-dot-order noise floor; explicit lane stays bit-exact).
Re-run the parity gate whenever a solver/dispatch changes; knob-bisect
before blaming the newest change.

atm GPU: ppermute pad fusion intentionally NOT pursued — micro-bench
8457615 shows packed >= separate on GPU (XLA overlaps collectives;
CPU-MPI differs because of mpi4jax token serialization), and strong
eff 0.73 f64 is at the PCIe-link roofline (jobs 8458801/02).

## Session addendum 2 — round-2 levers (2026-06-11)

Shipped (codex-reviewed, same-node A/B job 8460192, all gates green):
- T+S shared-factor implicit-vmix solve (`thomas_solve_shared` +
  `implicit_vertical_diffusion_ocean_pair`, default ON,
  `LEGOESM_VMIX_TSPAIR=0` kill-switch): one factorization, two RHS,
  bit-identical per-RHS.  LL192 np1 1.151x / np4 1.089x / np8 1.082x;
  LL128 np1 1.070x / np4 1.122x.
- curl sin/dx static fold: pad CONCRETE lat/cos_lat then apply sin /
  R*cos*dlon; pole rows overwritten with literal +-1.0.  Census:
  13 pads constant-folded at trace time, 34 per-step remaining.
- T+S tracer level-stack advection pair: implemented, bit-identical,
  MEASURED NET-NEGATIVE on CPU (np1 -3%, LL128-np8 -17%, no multi-rank
  win) -> default OFF, opt-in `LEGOESM_TRACER_PAIR=1` (stacking adds
  reconstruction-intermediate bandwidth; third confirmation of the
  batching lesson; trade may flip on GPU).

Cumulative ocean CPU-MPI vs pre-campaign legacy:
  LL128 strong np4 1.50x, np8 1.61x; LL192 np8 1.63x;
  weak rows/rank=48 np8 growth x4.65 -> x2.95.

Deferred (documented decisions, not regressions):
- atm GPU multi-node / NVLink: interconnect-capped on Ginsburg
  (no P2P, network < PCIe; roofline jobs 8458801/02) — revisit on
  NVLink/IB hardware.
- vmix K-profile-into-solve fusion: next per-device lever if the
  vmix share stays dominant after the pair solve (phase-split first).

PNGs: laws_v4.png (4-config same-node comparison); laws_v2/v3 = the
round-1 before/after pair.

## Session addendum 3 — all-grid-types theoretical-limit round (2026-06-11)

Measured probes (jobs 8460253/8460254/8460255/8460424/8460429):
- atm GPU command-buffer A/B: NULL (10.33 vs 10.36 ms C96) — the CS
  dycore is NOT kernel-launch-bound; per-device PID 7.2e-9 s/cell/step
  is already JAX-Fluids-class.  CUDA-graph lever closed for atm.
- ocean GPU (f64, 1 dev): graphs worth ~3%/1.5% (already default-on);
  T+S pair solve +4.6%/+9.8% (LL128/LL192) — GPU-validated too;
  PCR vmix -6..-30% (stays opt-in).  79.5 Mc/s f64 LL192; the next
  per-device GPU lever is f32-storage (RTX 8000 f64 = 1/32 FLOP rate),
  not dispatch.
- vmix internals (CPU np1): K-profile fallback 0.3-0.8 ms (0.2-0.3%) —
  the old '67% fallback artifact' caveat is CLOSED, the cost is real
  Thomas-solve bandwidth; ts_pair 1.58x isolated vs two solves; CPU
  vmix is at its bandwidth floor post-pair.
- ocean multi-node: np16 across 2 nodes LL192 = 90.2 ms vs np8 138 =
  near-ideal 8->16 (1-node np16 flattening WAS per-node DRAM
  contention); np32@16/node = 123 ms (slower than np16@8/node) =>
  ranks/node <= 8 policy at LL192; weak np32 growth x7.9 (vs x2.95
  np8) => cross-node allreduce/sendrecv latency reactivates the
  PCG reduction-count lever (pipelined-CG / batched dots) at >=2 nodes.
- post-lever ocean phase split np4 LL128 (78.7 ms): vmix 39.7%,
  baroclinic 20.1%, barotropic 9.5%, tracer 9.6%.

Shipped this round (codex-reviewed, gates job 8460429 green):
- ATM lat-lon fused entry pads: one pad_with_pole_bc_lat_multi((T, u))
  per tendency eval; padded u SHARED by curl circulation + absolute-
  vorticity 4-pt average (was padded twice).  Census 15 -> 9
  exchanges/step (+12 already constant-folded for free from the ocean
  round's static-metric work).  Serial-vs-MPI step parity bit-exact.
- MPAS implicit_cn np>1: LOUD NotImplementedError guard (was silently
  rank-local stock CG + rank-local mass projection).  Distributed
  MPAS PCG remains the documented TODO (Voronoi halo A_op +
  owned-cell-masked dots).

Per-grid distance-to-limit verdicts:
- cubed-sphere atm GPU: per-device at JAX-peer limit; 2-GPU strong at
  PCIe link roofline (0.73 f64).  CPU-MPI dynamics still REPLICATED
  (A1) — the one architectural gap; needs jax.distributed SPMD face
  mesh (multi-week, decision item).
- lat-lon atm+ocean CPU-MPI: comm engineered down 63->~30 (ocean) and
  15->9 (atm) exchanges/step; strong limited by per-node DRAM at
  >8 ranks/node; weak by PCG reduction latency at multi-node.
- icosahedral/MPAS atm GPU: 0.88 2-GPU eff = at link limit; MPAS ocean
  implicit_cn now guarded, distributed PCG = roadmap.
- spectral atm: transform = global matmul (cuBLAS-bound), single-device
  by design; multi-rank N/A documented.
- tripole ocean: inherits all lat-lon levers; fold/vector pads remain
  single-field (extend pad_with_pole_bc_lat_multi with per-field
  vector/fold flags = next mechanical item before production-OMIP
  scaling claims).

## Addendum 6 (2026-06-12): tiled-d2a2c strip stage; codex scaling-review triage; production CS SPMD design

Shipped this round (gates 8470688/8470707, codex PASS + kt=3 lane):
- Tiled d2a2c production stage module `legoesm/parallel/tiled_d2a2c.py`
  (`make_tiled_d2a2c_stage`): face-replicated `d2a2c_global_fields` +
  per-tile `d2a2c_tile_unified` under a (6,kt,kt) shard_map, strips
  applied IN-stage via 4 one-cell `lax.ppermute` halos feeding the new
  `d2a2c_tile_strips` (fv3_sw_core.py).  Dedup reassembly == d2a2c_vect
  at the fp floor (~1e-16), duplicated shared-face copies bit-identical
  (symmetric exchange).  HONEST microbench: the isolated stage on 24
  host devices is 0.06x serial (107 vs 6.7 ms, C96) — dispatch/
  collective floor at single-op size; the stage exists to unlock the
  full-tendency shard_map (np24 GSPMD-auto baseline 2253 ms), not to
  win as a lone operator.
- PCG variant evidence (job 8470723, drift-controlled A/B/A/B):
  single_reduce 3-7% faster np=2/4 (both reps), never measurably
  slower.  Default flip APPROVED but DEFERRED to a coordinated commit
  with the distributed-MPAS session (its config-contract tests pin
  "standard" on both grids).  `ocean_pcg_weak.sbatch` now runs the
  interleaved variant lanes permanently.

Codex scaling-review (2026-06-12) standing items:
1. PRODUCTION cubed-sphere driver SPMD (the A1 production wiring) —
   design pinned below.
2. MPAS pcg_variant config gap — fixed by the concurrent ocean session.
3. lat-lon 2-D decomposition — roadmap (big architectural).
4. single_reduce default — evidence complete, flip deferred (above).
5. voronoi batched-halo pack/scatter profiling — roadmap.

### Production CS driver SPMD build plan (codex-1 / A1-production)

Template: `run_cpu_mpi_scaling.py --cs-spmd` (proven: shard-local
parity 6.7e-10@5 steps job 8462928; np24 4.4e-10 job 8465445; C96
multinode 2.42x job ~e209bbb0-era).  Driver gaps, in build order:
1. Bootstrap: a `cs_spmd` parallel mode in `_bootstrap_runtime` —
   `jax.distributed.initialize()` BEFORE any JAX use (launcher-agnostic
   env detect), DeviceConfig mesh from GLOBAL `jax.devices()` (NOT
   local), cubed-sphere-only refusal, mpi4jax NEVER armed in this mode
   (mixed-stack deadlock hazard).
2. `_setup_parallel`: route the spmd-distributed case into the existing
   single-node SPMD branch (`shard_state` + `shard_pytree(tracers)` +
   `_maybe_activate_spmd_halo_backend` — gate already n-divisor-based);
   skip the mpi4jax replicated-faces branch entirely.
3. Physics: columns operate pointwise on (6,n,n,...) sharded arrays
   inside the jitted step — GSPMD splits them; forcing arrays (SST/SIC,
   ozone, aerosol) must be device_put with the SAME face sharding (a
   replicated or mis-sharded forcing triggers GSPMD resharding storms).
   ColumnAdapter stays GLOBAL-shaped (no rank-local rebuild — that
   path is mpi4jax-specific).
4. Conservation fixers: jnp-level global sums on sharded arrays are
   SPMD-global by construction (parity receipt) — audit that no fixer
   uses rank-local masks in this mode.
5. Diagnostics/IO: gather via multihost process_allgather (or
   addressable-shard assembly) with process-0-only writes; segment
   machinery (SegmentCarry/scan) traces identically on all processes.
6. Gates: 2-proc CPU short-AMIP parity vs serial (state + conservation
   diagnostics), then a GPU pair; bench hook reuses a1_cs_spmd_*.sbatch.
