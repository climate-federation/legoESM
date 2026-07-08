# Weak/strong scaling review vs SOTA + Derecho/Levante readiness (2026-07)

Scope: all grids × atmosphere + ocean × weak/strong, benchmarked against the
current SOTA dycores (Oceananigans, CliMA, MPAS-A/O, FV3/SHiELD/Pace, ICON,
SCREAM/E3SM, NICAM, IFS), and the engineering delivered this session to make
legoESM run without MPI/multi-GPU bottlenecks on **NCAR Derecho**
(Slingshot-11, 4×A100-40 SXM/node) and **DKRZ Levante** (IB HDR, 4×A100-80/node).

Prior campaign context (read first for the measured baselines):
`scaling_theoretical_limit_report_2026-06-15.md` (Ginsburg verdict),
`SCALING_SUMMARY.md` (single-node all-grids), `distance_to_limit_2026-06-13.md`,
`literature_neuralgcm_veros_mpas_2026-06.md`, `literature_parallelization_2026-06.md`.

---

## 1. Where legoESM stands today (measured)

| Axis | State (job receipts in the campaign docs) |
|---|---|
| Atm per-device (cube) | 7.2e-9 s/cell/step — JAX-peer class (JAX-Fluids-level); XLA-fusion closed; CUDA-graph A/B null |
| Atm 2-GPU strong | 0.73–0.83 eff — AT the PCIe roofline of the old test host (not the algorithm) |
| Atm multi-node CPU-MPI | clean controlled curves all grids (`docs/scaling/multinode_clean/`): ico ~0.7–0.75 eff, latlon ~0.5 @8 nodes, cube ANTI-scales on Gloo/TCP (fabric, not code) |
| Ocean 2-GPU / weak | 0.92 eff full-step at production size; WEAK 0.97 at ~590k cells/rank |
| Ocean strong (CPU-MPI) | np16→32 0.55 — implicit-CN PCG reduction wall (120 allreduce/step); split-explicit + local clamp opt-in beats it at ≥2 nodes (1.15–1.65×) |
| Spectral | single-device by design (both multi-device schemes measured anti-scaling) |

**The Ginsburg "at-limit" verdict is a fabric statement** (PCIe pairs, no
NVLink, Gloo/TCP, no IB). Everything below is about the two machines where
those constraints vanish.

## 2. SOTA comparison (2026-07 research sweep, 3 agents, primary sources)

### Ocean

| Model | Machine / devices | Config | Headline | Per-A100 fp64 throughput |
|---|---|---|---|---|
| Oceananigans (Silvestri 2025, JAMES 2024MS004465) | Perlmutter, 64 A100 | 1/12°, 100L | 10 SYPD | **~160–230 Mcell-updates/s** (derived) |
| Oceananigans | 16 A100 | 1/4° | ~75 SYPD | — |
| Oceananigans weak | 4→768 A100 | 5e7–2e8 cells/GPU | **~1.0 eff (flat)** | — |
| Oceananigans strong | 4×→16× GPUs | 1/48° | ideal→0.70 (land-imbalance, not comm) | — |
| MPAS-O (Kang 2021) | Cori-KNL 16,320 cores | EC60to30 | semi-implicit pipelined BiCGStab+RAS: barotropic 2.9×, model 1.9× vs split-explicit subcycling | — |
| Omega v0.1 (2026) | Frontier | — | device-resident packed halos + GPU-aware MPI = **4–6× lower halo time** vs host-staged | — |
| ClimaOcean (Wagner 2025) | 1 H100 | 16 km coupled | 1.5 SYPD | — |

legoESM ocean anchors: LL192 fp64 full step ~152 Mcells/s on a crippled-fp64
laptop GPU (fp64=1/64) — the A100 number is the first thing to measure with the
new `ocean_gpu_scaling` jobs; the architecture (band decomposition,
split-explicit option with local clamp, fused halos, mixed-precision opt-ins)
is aligned with what makes Oceananigans fast. Their two defining techniques we
do NOT have: **wide-halo barotropic** (halo width = substep count → ONE
exchange/baroclinic step; §5) and **active-cell (wet-point) compaction** (2×
on 42%-land global grids; their strong-scaling ceiling is the imbalance of
naive partitions — a balanced wet-cell shard would beat it).

### Atmosphere

| Model | Machine / devices | Config | Headline |
|---|---|---|---|
| SCREAM (SC23 GB winner) | Frontier 32,768 GCDs | 3.25 km | 1.26 SYPD full model; strong 512→8192 nodes ≈49% eff; >50% kernel win = <10% model (comm dominates) |
| ICON (SC25 GB winner) | JUPITER 20,480 GH200 | 1.25 km coupled | 145.7 SDPD; weak ~90% over 64×; strong degrades below ~10.8k cells/GPU; DaCe dycore beats hand-OpenACC; Levante A100 ≈ ½ GH200/device |
| ICON-A GPU (GMD 2022) | JUWELS 512 A100 | 5 km, 191L | 133 SDPD; 40,960 cols/GPU; GPU strong scaling "relatively poor" |
| MPAS-A GPU | Summit ≤4,200 V100 | 3 km fp32 | 1 V100 ≈ 130 BDW cores; weak-scaled at 40–80k cols/node |
| Pace/GT4Py FV3 (GMD 2023) | Piz Daint ≤864 P100 | ~4 km | GPU 3.5–4× node-for-node vs Fortran; perfect weak scaling at 192²×79/GPU; whole-timestep compilation is THE enabler |
| X-SHiELD (2025) | 27,648 EPYC cores | 3.25 km | 0.12 SYPD (CPU) |
| NeuralGCM | 1 TPU | 2.8° hybrid | single-device throughput class, not a scaling result |

Consistent SOTA band for km-scale halo codes: **~25–40 M gridpoint-steps/s per
A100**, strong-scaling floor at **≥10–40k columns/GPU**. Weak scaling is the
metric every SOTA model wins on; strong scaling universally dies below the
per-device floor. Our C48-class test tiles sit 4× under that floor —
production-scale claims on Derecho/Levante must use C96–C192 / LL192+ tiles
(the campaign already codified this).

### Cross-model lessons already banked in legoESM
- shard_map+ppermute = JAX-Fluids-validated architecture (0.95 weak eff @512 A100).
- XLA latency-hiding scheduler ON by default in `runtime/backend.py` (their #1 flag).
- Whole-step jit ≡ Pace's DaCe orchestration ≡ ICON's DaCe win (we get it natively).
- Halo aggregation: `pad_halo_4d` + multi-field stacking + dtype-cast fusion ≡
  Omega's packed device-resident exchanges; MPAS batched union-neighbor halo default.
- SFC/METIS Voronoi partition (`voronoi_partition.py method="auto"/"sfc"`) ≡
  ClimaCore's space-filling-curve element partition.
- Barotropic: split-explicit (halo-only) + reduction-free local clamp opt-in ≡
  the Oceananigans no-global-collectives doctrine; single_reduce (ChronGear) +
  P-CSI/Chebyshev already characterized; MPAS-O's pipelined-BiCGStab+RAS is the
  known alternative if implicit-CN must scale to high ranks.

## 3. Gaps found in THIS review and closed this session

1. **Ocean SPMD full-step retrace bug (real per-step recompile)** —
   `make_sharded_ocean_step` rebuilt an un-jitted `shard_map` EVERY call
   (the exact bug fixed on the atm twin in `fa2ce32b6`: 142 s/step → 1.1 ms).
   Fixed: jit-once cache + `dt` as traced operand
   (`packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py`).
   Equivalence gate `tests/parallel/test_latlon_ocean_spmd_step.py` passes
   unchanged; smoke: compile 1.9 s → steady 2.5 ms/step.
2. **No ocean scaling driver for multi-GPU nodes** — added the
   gates + federation hardening onto
   `scripts/bench/bench_ocean_latlon_spmd_scaling.py`: full ocean step over
   `make_sharded_ocean_step` on N local GPUs (never mpirun), with weak/strong
   geometry, parity-vs-single-device gate (re-association-floor tolerances,
   smoke-window capped), conservation gate, and the standard CSV/JSON schema.
   Tests: `tests/bench/test_bench_ocean_spmd_lane.py` (incl. end-to-end
   2-virtual-device run).
3. **Hardcoded jax.distributed coordinator port 1234** (EADDRINUSE when two
   jobs share a node) — `resolve_coordinator_port()` in
   `parallel/early_init.py`: `LEGOESM_COORDINATOR_PORT` override, else
   crc32(job-id) into the dynamic range (deterministic across ranks), else
   legacy 1234. Wired into `early_init`, `parallel/distributed.py`,
   `run_levante_gpu_scaling.py`. Tests in `tests/unit/test_early_init.py`.
4. **No PBS/PALS support for bare `jax.distributed.initialize()`** (Derecho
   `mpiexec` is not auto-detected; multi-node spmd would refuse) —
   `init_jax_distributed_with_fallback()`: bare initialize, then
   `cluster_detection_method="mpi4py"` (the ALCF Cray-EX recipe; plain-MPI
   bootstrap only, mpi4jax never armed). Wired into `runtime/config.py`
   bootstrap-spmd, `run_cpu_mpi_scaling.py --cs-spmd` (+ PMI_SIZE detection),
   `run_levante_gpu_scaling.py`.
5. **Broken-mpi4py venvs killed single-process benchmarks** (mpi4py installed
   without libmpi raises RuntimeError, guards caught only ImportError) —
   broadened in both bench drivers (fixes 2 pre-existing-red
   `tests/bench/test_scaling_moist_tier.py` cases).
6. **No NCCL-over-fabric wiring anywhere** (zero aws-ofi-nccl references;
   NCCL on Slingshot silently falls back to TCP sockets) — new
   `gpu_multinode_scaling.pbs` (Derecho): cube cs-spmd 6×A100 over 2 nodes,
   aws-ofi-nccl env block (`NCCL_NET="AWS Libfabric"`, `NCCL_CROSS_NIC=1`,
   `NCCL_NCHANNELS_PER_NET_PEER=4` for send/recv-heavy halos,
   `MPICH_GPU_SUPPORT_ENABLED=0` in the NCCL lane) + loud socket fallback,
   plus the route-A CUDA-aware cross-node lane. `_env.sh` gains
   `FI_MR_CACHE_MONITOR=userfaultfd`.
7. **No Levante job set at all** — new `scripts/cluster/scaling_levante/`
   (README + `_env.sh` with the IB/NCCL defaults + 4 sbatch jobs): single-node
   atm+ocean GPU, multi-node cube cs-spmd over NCCL/IB (SLURM auto-detect),
   route-A over CUDA-aware OpenMPI+UCX (DKRZ ICON-GPU recipe,
   `MPI4JAX_USE_CUDA_MPI=1`), CPU-MPI atm+ocean ladders.
8. **No ocean cluster jobs** — Derecho `ocean_gpu_scaling.pbs` (SPMD lane
   ladder 1/2/4 A100 with fail-fast gates) + `ocean_cpu_scaling.pbs`
   (MPI rank ladder); Levante equivalents folded into
   `gpu_scaling.sbatch`/`cpu_scaling.sbatch`.
9. **Stale multinode coordinator test** (mocked `process_count==2` against the
   `==1` init guard; red on main) — fixed to the two-phase semantics and
   pinned to the resolved port.

Auto-PGLE (`JAX_ENABLE_PGLE=true`, `JAX_PGLE_PROFILING_RUNS=3`) is deliberately
NOT defaulted in `backend.py` — it recompiles mid-job and is incompatible with
AOT; enable per-run in job scripts once the multi-node lanes are green
(documented in the Levante/Derecho READMEs' tuning order: LHS → PGLE → fabric
transport check → `NCCL_NCHANNELS_PER_NET_PEER`).

## 3b. Measured Derecho diagnostics (user-provided runs, 2026-07-02)

Strong-scaling plots for latlon + icosahedral atm, CPU (1→128 Milan ranks) and
GPU (1→16 A100 = 4 nodes), f32/f64 speedup panels (latlon res 720, ico res 8).
Readings:

1. **CPU-MPI is healthy and monotonic to 128 ranks on both grids.** At 128
   ranks: icosahedral f32 ~75× (eff ~0.59), f64 ~38×; latlon f32 ~30×
   (eff ~0.23), f64 ~20×. The ico cell-partition (METIS/RCB, low
   perimeter/area) scales far better than the latlon 1-D band (perimeter
   fixed at 2 full lon rows/rank + polar rows) — as designed; the 2-D latlon
   decomposition (fabric-blocked on Gloo) is worth an A/B **on Slingshot**
   at ≥64 ranks if latlon CPU production matters. The throughput kink above
   32 ranks (all resolutions collapsing onto one superlinear curve) is the
   per-rank tile dropping into cache — real, not an artifact.
2. **Coarse grids do NOT strong-scale on GPUs — per-device saturation floor,
   not a defect.** 111 km latlon and 112 km ico are FLAT (even anti-scaling
   for ico throughput) from 1→16 A100s: at 16 GPUs those grids are ~4k
   columns/GPU, well under the SOTA floor (ICON degrades below ~10.8k
   cells/GPU; icon4py fastest at ~33k; MPAS-A ran 15.6k/GPU). Fine grids
   scale: 28 km latlon 7.5× @16 (eff ~0.47), 28 km ico ~6× (eff ~0.38),
   still rising at 16 GPUs. **Rule for production: keep ≥ ~30k columns/GPU
   (LL512+ / ico≥56 km per 4-GPU node; 28 km-class for 16 GPUs).**
3. **f64 ≈ f32 speedup curves on GPU ⇒ the multi-GPU leg is LATENCY-bound,
   not bandwidth-bound** (halving the bytes doesn't move the curve). Consistent
   with the halo-message-count analysis from the campaign; message
   AGGREGATION and overlap, not compression, are the levers.
4. **The 8-GPU (2-node) dip** (78 km latlon throughput 600→480 then recovery
   at 16; ico 56 km flattens 8→16) marks the first Slingshot crossing. These
   multi-GPU points run route-A mpi4jax — whose sendrecv custom calls are
   OPAQUE to XLA's latency-hiding scheduler (no comm/compute overlap is
   possible on that leg, per the 2026-07 fabric research). This is exactly
   the ceiling the native-NCCL multi-controller path removes: ppermute
   collectives ARE overlappable. → elevates "multi-controller SPMD for
   latlon/ico" to the top of the gap list (below).
5. **Per-device normalization vs SOTA**: single-A100 latlon ~370 Mcells/s and
   16-GPU aggregate ~2.5e3 (28 km) ≈ 155 Mcells/s/GPU sustained under strong
   scaling — inside the SOTA per-A100 band (ICON-A ~26 Mgridpt-steps/s incl.
   physics at 191L; Oceananigans full ocean step ~160–230; exact
   apples-to-apples needs matched physics/levels, but there is no
   order-of-magnitude gap to close per device). The scaling frontier, not the
   kernel frontier, is where the remaining Derecho headroom lives.

## 4. Ranked remaining gaps (not this session; each is a deliberate project)

0. **Multi-controller NCCL SPMD for latlon atm + ocean — SHIPPED 2026-07-03**
   (icosahedral remains open, below). The existing `device_put`-based
   constructors turned out multi-controller-capable as-is (proven by a local
   2-process Gloo federation probe — the cs-spmd precedent generalized); the
   work was wiring + latent-bug surfacing:
   - `bench_atm_latlon_spmd_scaling --multicontroller/--coordinator` and
     `bench_ocean_latlon_spmd_scaling --multicontroller` federate
     1-proc-per-GPU launches into one JAX program (native ppermute band
     halos over NCCL — XLA-overlappable, unlike route-A mpi4jax). (The ocean
     multicontroller lane is on the SPMD full-step bench, NOT a
     `--transport spmd` flag of `bench_ocean_mpi_scaling`, which is the
     route-A mpi4jax CPU/PCIe-GPU phase-split bench.)
   - Latent bug fixed: `reductions.is_multi_process()` treated
     `jax.process_count()>1` as "mpi4jax-allreduce needed" — under
     federation that imports the forbidden mixed stack into the ocean
     serial-reference/invariant legs. Now mpi4jax-decomposition-only
     (spmd-backend psum branches + global-jax.Array semantics cover the
     rest); the MPAS `normalize_freshwater` refusal gained an explicit
     `process_count>1` clause.
   - Equivalence gates (2-process federated CPU):
     `tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py` (rtol 1e-6 /
     atol 1e-9 vs serial) and
     `tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py` (full bench lane,
     parity + conservation gates).
   - Job lanes C (atm latlon, np=8) + D (ocean, np=8) in BOTH
     `gpu_multinode_scaling.pbs` (Derecho, shared `nccl_env()`) and the
     Levante sbatch twin; `build_nccl_ofi.sh` builds aws-ofi-nccl on a
     Derecho login node and prints the `LEGOESM_NCCL_OFI_LIB` submit line.
   Remaining under this item: icosahedral/Voronoi multihost (cell-partition
   reorder across processes) and promoting the lanes into run_amip/run_omip
   production drivers once machine numbers land.
1. **Wide-halo split-explicit barotropic** (Oceananigans' signature): halo
   width = substep count ⇒ ONE 2-D exchange per baroclinic step. Measured
   net-negative at Ginsburg scale (barotropic ~7% of step at production tile)
   but the calculus flips at high rank counts where per-message latency
   dominates — evaluate at ≥16 GPU ranks on Levante/Derecho, as a selectable
   scheme in `barotropic_common` (stability of the averaging filter under
   stale-halo substeps must be validated first; MPAS-O's rejection was about
   unfiltered subcycling).
2. **Active-cell (wet-point) compaction** for the global ocean: 2× SOTA win at
   ~40% land; also fixes the load imbalance that caps Oceananigans' strong
   scaling — our METIS path can weight by wet cells (partial groundwork in
   `voronoi_partition.py`; lat-lon band variant = row-weighted bands).
3. **Ocean multi-node GPU SPMD** (jax.distributed multi-controller for the
   lat-band ocean, mirroring `--cs-spmd`): the single-process lane ships now;
   the multi-controller variant needs the shard-per-process state constructor.
4. **Tripole/eORCA active fold under SPMD** (`sharded_ocean_step` raises;
   MPI band path already supports the fold) — needed for eORCA GPU scaling.
5. **MPAS-ocean scaling lane** (voronoi_mpi step exists; no bench drives it).
6. **Cube np>6 sub-face production step** (P4): transport+KE stack tiled and
   parity-proven; d_sw1/d_sw5/d_sw6 assembly remains — THE lever for >6 GPUs
   on cubed-sphere, now benchable on Derecho/Levante once assembled.
7. **Pipelined-p2p XLA experiment** for scan-loop ppermutes
   (`--xla_gpu_enable_pipelined_p2p` + permute decomposer; command buffers off)
   — experimental flag, A/B only.

## 4b. Codex adversarial review

3 rounds (thread 019f25ce-1281): round 1 = 7 findings (1 rejected with
receipt — the smoke IS above the 2-row floor and ran green); fixes in
`d760e48cc` (structure-keyed jit cache ocean+atm, loud MPI failures,
PALS guard, indexed GPU pinning) and `b9c43586d` (env-routed
jax.distributed transport — no exception-text parsing; PALS per-rank-id
launcher evidence). **Round-3 verdict: CLEAN.** 124 affected tests pass.

## 5. Runbooks

- Derecho: `scripts/cluster/scaling_derecho/README.md` (envs, fabric, NCCL
  plugin, all 7 jobs).
- Levante: `scripts/cluster/scaling_levante/README.md` (envs, UCX/NCCL-IB,
  all 5 jobs).
- First-run verification: NCCL transport line (`AWS Libfabric` / `NET/IB`),
  `jax.process_count()` federation gate, per-case parity+conservation smokes
  (all fail-fast in the jobs).
- Cross-machine comparison: identical CSV schema everywhere →
  `scripts/plot/plot_scaling_efficiency.py` works unchanged.

## 6. Reconciliation note (2026-07-03)

The scaling-roadmap campaign on main (#683/#693/#710/#713/#725/#736) landed
CONCURRENTLY with this branch and independently implemented several of the
same items: route-B `--multicontroller` federation for the lat-band atm+ocean
SPMD benches (#725), the ocean full-step SPMD bench + jit-once sharded step
(#736), multi-node jax.distributed init ordering + `local_device_ids` binding
(#693/#710), and Derecho/Levante script sets. This branch was rebuilt on top
of that work, keeping only the net-new deltas: the structure-keyed shard_map
cache, the `is_multi_process` mpi4jax-only semantics fix (+ the MPAS
freshwater refusal covering `process_count>1`), job-id-derived coordinator
ports, the PALS/PMI env-routed `init_jax_distributed_with_fallback`, loud
broken-mpi4py failure under launchers, parity/conservation gates on the ocean
SPMD bench, self-spawning CI federation gates (+ port-race retry harness),
the aws-ofi-nccl build script, and the ocean/multinode cluster jobs.
