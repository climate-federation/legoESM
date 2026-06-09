# CRM GPU L2-cache tiling — investigation & verdict (2026-06-09)

The plane CRM (compressible-Euler) fp32 GPU throughput drops sharply once the
working set spills L2. This note quantifies the spill, tests the candidate levers,
and records the evidence-based verdict.

**Host:** 1× RTX 5090 Laptop GPU (24 GB). fp32, nlev=30, dx=2000 m, dt=2 s,
`scripts/bench/bench_crm_gpu_scaling.py`.

## The spill — measured

| nx  | cells   | step (ms) | throughput | effective HBM |
|----:|--------:|----------:|-----------:|--------------:|
| 128 | 491,520 | 1.86 | 264.8 Mc/s | **826 GB/s (113 %)** |
| 152 | 693,120 | 3.20 | 216.4 Mc/s | 675 GB/s (92 %) |
| 160 | 768,000 | 2.90 | 264.9 Mc/s | **826 GB/s (113 %)** |
| 192 |1,105,920| 4.17 | 265.3 Mc/s | **828 GB/s (113 %)** |
| 256 |1,966,080| 10.81 | 181.9 Mc/s | 568 GB/s (78 %) |

- **>100 % "HBM" = the working set is served from L2** (effective bandwidth
  exceeds the HBM peak). N≤192 (≤1.1 M cells) is L2-resident; **N256 spills to
  HBM** and throughput falls **1.45×** (265 → 182 Mc/s per cell).
- The carried acoustic-substep working set is ~5 prognostic fields × N²×nlev —
  irreducible without tiling (XLA already fuses the transient intermediates; the
  live state is the floor). So the spill is a pure working-set-vs-L2 effect.

## Levers tested

**1. XLA compiler flags (zero-code).** `--xla_gpu_enable_latency_hiding_scheduler`
+ `--xla_gpu_autotune_level=4` + `--xla_gpu_enable_while_loop_double_buffering`:
N256 → 184 Mc/s (vs 182 baseline) — **no effect**. XLA does not do stencil
temporal blocking; it cannot keep a tile L2-resident across the substep loop.

**2. Intra-kernel temporal blocking (ghost-zone tiling) — NET NEGATIVE.** To keep
a tile L2-resident across the acoustic substep loop, each tile needs a halo of
width `n_substeps` (one cell consumed per substep; ±1 horizontal stencil), so a
128² interior requires a **152² tile**. Measured cost:
- N152 already spills (92 % HBM, 3.20 ms) — the fat halo pushes the tile itself
  out of L2.
- Tiling N256 as 2×2 of 152² = 4 × 3.20 = **12.8 ms > 10.81 ms** (untiled N256).
The ghost-zone redundant compute (152²/128² ≈ 1.4×) plus the haloed tile's own
spill **exceed** the spill they were meant to avoid. Splitting the substeps into
sub-blocks (smaller halo, more passes) is worse still (doubles the field passes).
⇒ JAX-level temporal blocking does not pay here.

## Verdict & recommendation

The L2 win (1.45×) is **real but not reachable by intra-kernel JAX tiling**. Two
paths realize it:

1. **Domain decomposition (the available lever).** Keep each *device's* local tile
   **≤ ~192² (≤1.1 M cells / nlev=30)** and it stays L2-resident (265 vs 182
   Mc/s). This is exactly what the MPI / multi-GPU decomposition provides — split
   a large global domain so each rank/GPU computes an L2-sized tile. The plane CRM
   `step_halo` DD path (built + certified at np=4 2×2 this campaign) already does
   this; the L2 benefit comes **free** from decomposition. Gated on ≥2 devices
   (this host has 1 GPU; MPI here is CPU-only).
   **Practical rule: size the per-device horizontal tile to ≤ ~1.1 M cells.**

2. **Pallas on-chip-blocked acoustic kernel (deferred).** On a *single* GPU the
   only way to beat the spill is a custom kernel that blocks the 128² tile in
   shared memory / registers, loading the halo once and reusing it across all
   substeps — avoiding **both** the HBM round-trips **and** the fat-halo redundant
   compute that sinks the JAX approach. This is a large, GPU-specific, fragile
   effort (the semi-implicit vertical tridiagonal solve + EOS per substep must be
   expressed in Pallas) for a 1.45× on a throttling laptop GPU. Deferred with this
   cost/benefit recorded; revisit if single-device large-N throughput becomes a
   production bottleneck on a non-throttling datacenter GPU.

Bottom line: the CRM "L2 tiling" lever is the **domain decomposition already in
the codebase** (keep per-device tiles ≤ ~1.1 M cells); the intra-kernel tiling
that the phrase suggests is net-negative in JAX/XLA and needs Pallas to win.
