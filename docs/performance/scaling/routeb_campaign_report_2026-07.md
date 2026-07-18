# Route-B scaling campaign — throughput report (SKELETON, 2026-07)

Derecho, A100-40GB nodes (4 GPU/node) + 128-core CPU nodes. Differentiable JAX
dycore, route-B transport (`jax.distributed` multi-controller, ppermute/psum
over NCCL on the aws-ofi-nccl/Libfabric CXI fast path). Data:
`$SCRATCH/legoesm_scaling/routeb_campaign_202607/all_tidy.csv`.

> **Skeleton** — tables + framing + caveats are final; prose and plots are TODO.

## Framing — why throughput (SYPD), not strong-scaling efficiency

The competitive question a model developer asks is **"how much simulation per
wall-clock day, on how much hardware"** — i.e. **SYPD**, the ESM community's
currency (SCREAM: 1.26 SYPD @ 3.25 km). Strong-scaling *efficiency* answers a
different, internal question ("does our code parallelize a fixed problem?") and
is structurally unfair at low device counts: going 1→2 GPUs introduces halo
comms that did not exist at N=1, so efficiency drops for reasons unrelated to
the model's capability. **Efficiency is demoted to a supporting figure; SYPD and
raw throughput (Mcells/s) lead.**

Metric hierarchy:
- **SYPD** — headline. Bakes in resolution + timestep + per-step cost; rewards
  good numerics (bigger stable dt → more SYPD). Compare only at matched
  resolution.
- **Mcells/s** — resolution-normalized throughput; the fairest *cross-grid*
  number (different decompositions, different cell counts).
- **Strong-scaling efficiency** — supporting; "it parallelizes," not "it's fast."

## TABLE 1 — Matched-resolution cross-grid throughput (~40–55 km, 1 A100)

The fair cross-grid band (each grid on its own decomposition at ~comparable
physical resolution — the resolutions with full ladders):

| grid | config | res (km) | cells | 1-A100 SYPD | 1-A100 Mcells/s |
|---|---|---|---|---|---|
| lat-lon | LL512 | 39 | 13.6 M | 12.2 | 1011 |
| cubed-sphere | C192 | 52 | 5.75 M | 14.7 | 1032 |
| icosahedral | L7 | 56 | 4.26 M | 5.1 | 265 |

Reading: cube and lat-lon deliver ~1000 Mcells/s per A100 at ~50 km; ico L7 is
compute-light at 1 A100 (4.3 M cells underutilizes the card — ico shows its
throughput at higher resolution / device count, see Table 2). Mcells/s is the
fairer cross-grid metric here since the three grids differ ~3× in cell count at
matched km.

## TABLE 2 — Peak GPU throughput per grid (best point on each curve)

| grid | best config | N (A100) | peak SYPD | peak Mcells/s |
|---|---|---|---|---|
| lat-lon | LL1024 (19.5 km) | 16 | 29.8 | 9887 |
| cubed-sphere | C192 (52 km) | 6 | 20.3 | 1418 |
| icosahedral | L8 (28 km) | 16 | 10.0 | 2065 |

Lat-lon scales best (near-10 Gcells/s at 16 GPUs, 63% strong-scaling efficiency);
cube is latency-bound past 1 GPU (~46 collective-permutes/step × the ~0.11 ms
launch floor — see #1113) so it barely scales; ico is intermediate.

## TABLE 3 — HEADLINE: cube, 1 A100 vs 1 CPU node (128 cores), SAME CODE

The only grid with **both** backends complete (#1100 blocks the lat-lon/ico CPU
lanes). Same JAX dycore, same problem, GPU vs a full CPU node:

| config | res (km) | CPU-node SYPD | 1-A100 SYPD | **A100 / CPU-node** |
|---|---|---|---|---|
| C48 | 208 | 30.5 | 753.4 | **24.7×** |
| C96 | 104 | 5.10 | 130.2 | **25.5×** |
| C192 | 52 | 0.37 | 14.7 | **39.4×** |

Two findings:
1. **~1 A100 ≈ 25–40 full CPU nodes** of the same model.
2. **The advantage GROWS with resolution** (25× → 39×) — the GPU saturates with
   more work while the CPU node stays compute-bound. This is the direction that
   matters for high-resolution / cloud-resolving.
3. **Climate-viability crossover**: at C192 (52 km) 1 A100 = 14.7 SYPD (viable);
   the CPU node = 0.37 SYPD (not viable — ~months/century). The GPU makes ~50 km
   climate-viable where the same code on CPU does not.

## CAVEATS (load-bearing — do not drop from the final report)

1. **Dry-dynamics only** (`physics=none` in every row). These are **dycore
   throughput ceilings**, not full-model SYPD. RRTMGP + convection + etc. lower
   SYPD substantially. Any ESM comparison must be dynamics-vs-dynamics or add
   physics to both sides.
2. **JAX-CPU is NOT Fortran.** Table 3's CPU is XLA-on-CPU, the *same JAX code*
   — a hand-tuned Fortran ESM on that node is faster than XLA-CPU, so the 25–40×
   **overstates** the advantage over a real Fortran ESM. The JAX-GPU-vs-JAX-CPU
   claim is honest; a JAX-vs-Fortran claim needs a Fortran reference (see Gaps).
3. **Node accounting**: CPU "N=1" = one full 128-core node; GPU "N=1" = one A100
   = ¼ of a GPU node. Table 3 is per-A100-vs-per-CPU-node. A full GPU node
   (4×A100) adds ~3× for lat-lon (scales) but only ~1.4× for cube (latency-bound)
   — so state per-A100, not per-GPU-node, for cube.
4. **GDR-off lower bound**: NCCL ran host-staged (aws-ofi-nccl GDR unsupported,
   forced `NCCL_PROTO=simple`). Cross-device/-node throughput is a **lower
   bound**; GDRCopy/DMA-BUF would lift it.
5. **Internal numbers**: not vs published leaderboards; matched-resolution,
   dry-dynamics, this hardware.

## GAPS / TODO

- **The one number that completes the story: a production Fortran ESM SYPD at
  matched resolution + hardware** (dynamics-only). Turns "our GPU model is fast"
  into "it competes with Fortran ESMs." Still missing.
- **lat-lon + ico CPU lanes** (#1100) — needed for their node-vs-A100 rows;
  currently cube-only.
- **Full-physics SYPD** — add RRTMGP + physics for a real-model throughput number.
- Plots: SYPD-vs-N and Mcells/s-vs-N per grid; the Table-3 bar; the matched-band
  cross-grid bar. `finalize_scaling.sh` on the campaign root for the panels.
- ico rerun with #1175 CP-combining (merged) would improve the ico curve.

## Supporting: mechanism notes (not headline)

- Strong-scaling efficiency curves: lat-lon 63% @16 GPU (LL1024); cube 12–23% @n6
  (latency-bound); ico 34% @16 GPU (nlev=26). Fabric-limited (GDR-off) + CP-count,
  not fundamental — see #1113 (ppermute round-count × launch-floor law).
- Cube runs cross-node correctly at C96/C192 (#1177 corrected); only the tiled
  np=24 lane (#921) is a confirmed comm-init wedge.
