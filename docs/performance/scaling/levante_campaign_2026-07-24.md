# Levante weak/strong scaling campaign — 2026-07-24 (first hardware receipts + fixes)

One-day campaign turning the authored-but-never-run Levante job set
(`scripts/cluster/scaling_levante/`) into measured curves for every grid,
fixing what broke, and moving the worst axis (ocean strong scaling) to a
measured 2× improvement. All receipts on the post-merge tree `d3ec1ccce`+
(campaign branch `worktree-scaling-campaign`); job IDs cited throughout are
Levante SLURM jobs from 2026-07-24. Codex adversarial review: 4 rounds
(transcripts under `.physics-validator/scaling_campaign/`); every
measurement claim below carries the round-3 corrections.

Machines: Levante `gpu` partition (4× A100-80 SXM NVLink/node, IB HDR200),
`compute` (2× AMD Milan 7763). All GPU multinode = route-B
(`jax.distributed` + NCCL over IB verbs — `NET/IB mlx5` confirmed in-log;
route-A CUDA-aware mpi4jax not exercised on Levante).

## Headline results (strong scaling, f32 unless noted)

| Axis | Ladder | Result | Job(s) |
|---|---|---|---|
| Atm lat-lon LL720×1440 L26 | 4→8→16 A100 (1→4 nodes) | 7.72→5.40→3.54 ms/step, monotone; np16 = 7.6 GC/s (477 Mc/s/GPU sustained) | 26450848/26453240/26449147 |
| Atm MPAS ico L8 (28 km) L26 | 6→16 A100 | 8.66→7.08 ms/step; np16 = 2.41 GC/s — 1.6× Derecho's matched-grid 16-A100 aggregate | 26453240/26449147 |
| **Atm cube C768/L60 (same-path cs-spmd)** | 6→24 A100 | 58.35→14.09 ms/step = **4.14× = eff 1.04 (at ideal)**, 15.1 GC/s (629 Mc/s/GPU) | 26453782 |
| Atm cube C384/L60 (same-path cs-spmd) | 6→24 A100 | 15.44→8.81 ms/step = 1.75× (eff 0.44), 6.0 GC/s | 26452894 |
| Atm cube C192/L60 (same-path) | 6→24 | 6.20→6.80 ms — ANTI-scales (eff 0.23): 9.2k cols/GPU is below the ~30k-column floor | 26452979 |
| Ocean lat-lon LL576×1152 L20, production implicit | 4→8→16 | 15.6→17.2→17.4 ms — anti-scales across nodes | 26452743-45 |
| Ocean same, improved (wide-halo + vmix-f32) | 4→8→16 | 12.8→11.1→8.6 ms — monotone, **2.01× at 16 GPUs** | 26452804-06, 26453279 |
| Cube tiled >6-GPU lane (its own bench) | 24 A100, C384/L60 closed loop | 9.01 ms/step, 5.9 GC/s, 18.2 SYPD — first >6-GPU production-lane receipts | 26450938/26452632 |

Single-node GPU (job 26445836): cube C192 gray_sbm strong eff 0.84–1.07
(1→3 A100); C48/C96 latency-floored. Ocean single-node solver ladder: below.

**THE cube strong-scaling result** — read 1.04 as "at ideal", NOT "better
than ideal": efficiency slightly above 1 is expected when the BASE leg is
per-device disadvantaged (np6 holds 4x the working set per GPU of np24, so
part of the 4.14x is cache/occupancy recovery rather than parallel
efficiency). The claim is that comm does not degrade this ladder, not that
parallelism is free. (same code path, harness, config, IC;
only the tile size varies — jobs 26452979/26452894/26453782): 6→24 GPU
speedup 0.91× / 1.75× / 4.14× at 9.2k / 36.9k / 147k columns per GPU. The
"poor cube strong scaling" of the earlier receipts is ENTIRELY the
per-device saturation floor at undersized tiles, not a communication
defect: at production tiles the cube scales IDEALLY to 24 A100 across 6
nodes. Production rule confirmed: keep ≳30k columns/GPU.

Tiled-lane size sweep at fixed 24 GPUs (own bench, closed loop, CFL-scaled
dt; jobs 26450938/26452632/26453645): 1.87 / 5.89 / 14.34 GCells/s at
C192/C384/C768 = 78 / 246 / 597 Mc/s per GPU — per-device throughput still
climbing at 8.85M cells/GPU, so an A100 is not saturated even there.

## Ocean strong-scaling: bottleneck → fix (the campaign's improvement arc)

Dose-response on the implicit-CN barotropic reduction count
(LL384×768 L20 f64, solver-matched via `--force-pcg`, 4×A100 NVLink,
jobs 26447957/26449622):

| arm | reductions/step | nd1/nd2/nd4 ms | eff4 |
|---|---|---|---|
| implicit fixed-PCG standard | 123 | 29.3/21.1/15.0 | 0.49 |
| implicit PCG single_reduce | 63 | 29.6/19.8/14.0 | 0.53 |
| explicit + wide-halo | 0 (solver) | 33.4/19.1/11.5 | 0.73 |

Monotone count→efficiency mapping at every size (LL192/384/768); at LL768
all arms converge to 0.71–0.77 (tiles amortize latency). Fused-halo ≡
plain implicit (null → reductions, not pad count, dominate). NCCL_PROTO
default ≡ LL, LL128 harmful (job 26449812) — protocol lever closed.
`LEGOESM_VMIX_F32_SOLVE=1` (f32 tridiagonal vmix inside the f64 step):
icn +9.1%, wide +11.7% at nd4, conservation-gated at `--cons-rtol 1e-5`
on every arm (job 26452547). Combined best config (wide + vmix-f32):
10.32 ms at LL384 nd4 = 1.45× the production config, and the multinode
2.01× above. CLAIM SCOPE (codex round-3): the count→time slope is an
*effective time per eliminated reduction-batch in these executables*
(~14–18 µs/batch), NOT a measured allreduce latency — `single_reduce`
changes solver work/fusion too; a dependency-matched microbenchmark would
be needed for a latency claim. Both winning options are existing config
selections (`barotropic_solver="explicit_substep"` + wide-halo flags,
`LEGOESM_VMIX_F32_SOLVE`); production defaults unchanged — promotion needs
the wide-halo stability gates (`SCALING_STATUS_AUDIT` item 3) and a
science sign-off on the mixed-precision vmix.

## Atm ladders added late in the campaign

**Lat-lon WEAK at a production tile** (45 rows x 1440 lon x L26 = 64.8k
columns/GPU, f32, job 26454476): efficiency 1.00 / 0.45 / 0.41 / 0.45 /
0.44 at 1/2/4/8/16 GPUs — the cost is paid ONCE on the first cross-device
step and then FLAT to 16 GPUs across two node crossings (1.09 -> 7.66
GCells/s aggregate). Weak scaling on this grid is a fixed entry toll, not
a compounding one.

**MPAS icosahedral L8 (28 km) STRONG, identical padded mesh** (jobs
26454476 + 26454618): 19.90 / 17.09 / 6.92 / 7.10 ms at np 2/4/8/16.
Taking np2 as the base (it has the BEST per-device throughput, 430
Mc/s/GPU): 2->8 = 2.88x = eff 0.72, 2->16 = 2.80x = eff 0.35 (small-tile
floor).

OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
than np2 while np8 is 2.5x faster than np4. Evidence gathered:
- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
  17.14/17.04, 6.87/6.97).
- PLACEMENT REFUTED: np4 packed on one node (17.09 ms) == np4 spread over
  two nodes (17.12 ms), so node crossing is irrelevant.
- HALO VOLUME REFUTED: ghost-cell census on the padded mesh gives
  1540/1587/1400/1136 ghost cells per device at np 2/4/8/16 — flat to
  falling, and under 3 % of owned cells at every count.
- COLLECTIVE COUNT REFUTED (HLO census, ico L7, CPU virtual devices —
  device count is a compile-time property so the HLO matches what the GPUs
  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
  Communication cannot be what makes np4 slow. (Fusion count 136/173/240,
  bitcasts 526/582/694 — the np8 program is finer-grained.)
- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
  (geometric). Every method shows the same 2.5-3.1x jump.
- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
  command-buffers-off — every arm within 1 %, none recovers np4.
  (The multi-output-fusion arm errored on an unsupported flag name and is
  not counted.)
VERDICT: five hypotheses refuted by measurement (placement, halo volume,
collective count, partition method, codegen env knobs). The cheap-lever
space is EXHAUSTED; the remaining suspect — per-device kernel efficiency
for this shape — needs a GPU op-level profile (nsys / XLA op profile of
np4 vs np8), which is a separate instrumented project, not another timing
run. Per-GPU throughput across the ladder is non-monotone in tile size
(430 / 249 / 305 / 150 Mc/s/GPU at 327688 / 163844 / 81922 / 40961
cells/device), which is itself the clue to hand the profiler.
PRACTICAL GUIDANCE MEANWHILE: run this grid at np>=8, where per-device
throughput is 304-340 Mc/s/GPU vs 220-248 at np4.

## Ocean strong scaling vs TILE SIZE (jobs 26456334/37 vs 26452804-06)

The same improved config (wide-halo + vmix-f32, multicontroller NCCL/IB,
f32 L20) run at two tile sizes, 4 -> 16 GPUs:

| grid | cells/GPU @16 | np4 -> np16 | eff | aggregate @16 |
|---|---|---|---|---|
| LL576x1152 (13.3M) | 0.83M | 12.81 -> 8.63 ms | 0.37 | 1.53 GCells/s |
| LL1152x2304 (53.1M) | 3.3M | 43.47 -> 17.24 ms | **0.63** | **3.08 GCells/s** |

So the ocean obeys the SAME per-device-floor law the cube does: the 2.01x
multinode improvement measured at LL576 was partly a floor effect, and at
a production tile the identical code scales substantially better (0.37 ->
0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
Config is byte-identical between the two rows; only the grid changes.

## Weak scaling at production per-device size (job 26453523)

The earlier weak ladders used a 64-row base (0.17M cells/GPU — under the
latency floor). Re-run at PRODUCTION size (288 rows × 1152 lon × L20 =
6.6M cells/GPU, 1→4 A100, conservation gated): production implicit
1.00/0.72/0.70, improved wide-halo+vmix-f32 1.00/0.86/0.85.
PRECISION MATCHED (self-audit correction): BOTH ladders compared here are
**float32** — the production-tile run is f32, so it is compared against
the earlier ladder's f32 rows (eff 0.26/0.25 at nd 2/4), not its f64 rows.
An earlier revision of this file mislabelled the production-tile run f64
and cited the f64 small-base numbers; the direction and size of the effect
are unchanged, but the comparison is only valid precision-matched. So the weak collapse was a protocol artifact, and the same
config that fixes strong scaling also carries weak (+0.15 at nd4). Ideal is
flat; the improved arm holds 22.5→22.9 ms while production drifts
17.8→25.3 ms.

## "It used to be faster / did we regress?" — resolved, no regression

- Cross-machine anchor (matched bench/config/grid/physics/precision,
  job 26450081): Levante single A100-80 latlon-moist r720 f32 =
  **418.5 Mc/s** vs Derecho single-A100 ≈370. SCOPE: this establishes NO
  LARGE REGRESSION, not a precise machine ranking — the two campaigns
  differ in machine (A100-80 SXM vs A100-40), jax/tree version and date,
  so the ~13% gap is not attributable to any single factor.
- Cube "404 vs 141 Mc/s" = physics-tier confound: the 404 was RTX-5090
  Held-Suarez; gray+SBM is priced ~3× by `SCALING_SUMMARY.md`'s own tier
  table (404/3 ≈ 135 expected; 141 measured on A100).
- Ocean absolutes are memory-bandwidth-consistent across GPUs (A100 f64
  165–201 Mc/s vs 5090's compute-crippled-f64 152; f32 352 vs 400).
- Ginsburg "0.92 eff @2 GPU" reconciled: the old bench silently defaulted
  to `explicit_substep`; our explicit/wide arm reproduces that class
  (0.88 @2, LL384) — the production implicit config was never measured
  there. Protocol, not regression.
- No merge regression: nd=1 stock-CG LL192 8.90 ms pre-merge vs 8.94 post.

## CPU-MPI (compute nodes)

Single-node ladders (job 26445986, f64): atm latlon strong eff
0.87–0.92@np2 → 0.11–0.23@np32–64; ocean implicit 0.90@2 → 0.24@32; weak
collapses ≤0.14@32. CAVEATS: np=1 ocean leg was stock-CG (solver-mismatched
— fixed via `--force-pcg` in the job scripts; np≥2 slopes valid), and
single-node ladders conflate Milan DRAM contention with comm. The first
4-node pair collided into one OUTDIR (same-second stamp) and was discarded.

4-node SPREAD ladder (job 26452578, f64, ranks round-robin, solver-matched):
atm latlon np2 eff ≈1.00 (DRAM contention confirmed as the packed-ladder
confound: r128 np32 eff 0.38 spread vs 0.20 packed), decaying to 0.06–0.16
at np64–128 — the 1-D band perimeter ceiling as designed (r256/np128 = 2
rows/rank). Ocean strong spread: np2 eff 1.32 (superlinear, cache), 0.88@8,
0.35@32, wall at np64 (79 ms > np32's 74 ms). The job died in a high-rank
ocean case (one rank exit-3 → kill-on-bad-exit) before the weak tail —
np128 ocean + weak ladders and the rank-failure attribution remain open.

## Infrastructure defects found + fixed (each with a receipt)

1. py3.14 argparse eager help validation — both ocean benches crashed at
   parser construction (bare `%`).
2. Slurm step gres non-inheritance — multinode lanes saw zero CUDA devices.
3. SLURM double-pin — CVD pin + jax's `local_device_ids=[SLURM_LOCALID]`
   compose to a nonexistent ordinal; route-B lanes now leave all node GPUs
   visible (`--gpu-bind=none`) and let jax bind LOCALID-th (Derecho/PALS
   keeps its shim + `local_device_ids=[0]`; conventions must not mix).
4. cs-spmd rank identity from the federated jax runtime (mpi4py loud-guard
   killed the NCCL cube lane that needs no MPI); launcher world size now
   global-only (PALS_LOCAL_SIZE dropped), step size preferred.
5. `make_sharded_ocean_step` replicated-geometry `device_put` asserts
   bit-equality across processes while per-process XLA autotune drifts
   ULPs (size-dependent!) — process-0 broadcast + allgathered
   tolerance-compared divergence guard (quantized-equality v1
   false-positived on a rounding boundary; v2 rtol=1e-5). Gates:
   equivalence 6/6, selfspawn 2/2, live np4/8/16.
6. Multicontroller host materialization in the tiled bench finiteness gate
   → on-device global reduce.
7. OUTDIR same-second stamp collision → job-ID suffix everywhere.
8. `setup_mpi_venv.sh` could not resolve uv-workspace members with plain
   pip → pins first + `install_federation.py --all`; mpi4jax source-built
   with the system toolchain (GLIBCXX mismatch with gcc-11-built OpenMPI
   module).
9. Diagnosis tool halo/overlap phases produce garbage on the
   single-process no-mpi4jax GPU config (20-second "exchanges", 0 GB/s) —
   OPEN defect; census + scan phases are sound (cube C96/L40 census:
   0/7/14 CP/step at nd 1/2/3 + 1 all-reduce; 24-dev tiled closed loop:
   384 CP + 1 AR).

## Closed levers (nulls with receipts — do not re-run)

NCCL_PROTO forcing (default already optimal; LL128 −8–12%), fused-halo on
the implicit arm AND on the wide arm (pad aggregation is not the residual),
`xla_gpu_collective_permute_combine_threshold_bytes` alone,
`--xla_gpu_enable_pipelined_p2p` alone, `LEGOESM_BAROCLINIC_F32` (~+0.5%).
PGLE arm invalid as measured (33-step window catches its
profile+recompile; Derecho saw +8.5% with proper warmup — rerun long-window
only if revisited).

## Still open (ranked)

1. Wide-halo stability gates → promote the improved ocean config beyond
   benches.
2. ATM weak-scaling ladders at production per-device sizes (ocean done,
   job 26453523 above).
3. DONE (job 26453782/26453645): C768 same-path 6→24 eff 1.04, tiled
   closed loop 14.3 GCells/s. Next size point C1152+ only if a science
   driver needs it.
4. Diagnosis halo/overlap phase fix (defect 9) + calibrated t_bound lines
   on every plot from a dependency-matched comm microbenchmark.
5. CPU spread ladders + Milan np16 NUMA anomaly attribution.
6. Route-A CUDA-aware mpi4jax lane (`gpu_moist_scaling.slurm`) — only if a
   route-A-vs-B A/B is ever needed; route-B beat all route-A references.
