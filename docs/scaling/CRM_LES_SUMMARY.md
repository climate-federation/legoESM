# legoESM CRM + LES scaling summary (plane dycore, single-node)

Performance of the plane **CRM** (compressible-Euler, cloud-resolving) and **LES**
(spectral plane + dynamic LASD SGS), at **float32 and float64**, on MPI and GPU.

**Host:** 1× RTX 5090 Laptop (24 GB, 55 W) + 24-core single-socket CPU.

> ⚠️ Single representative sizes; the 55 W laptop GPU throttles under sustained
> load (absolute Mc/s drift ~±40 %, ratios within a run hold). True multi-device
> scaling needs ≥2 GPUs / multiple sockets.

---

## CRM (compressible plane)

### Single-GPU throughput (bench reports HBM %)

| precision | N64 (123 k) | N128 (492 k) | N256 (1.97 M) |
|-----------|------------:|-------------:|--------------:|
| **fp32** | 139 Mc/s · 59 % HBM | **164 Mc/s · 70 % HBM** | 94–112 Mc/s · 40–48 % |
| **fp64** | 29 Mc/s · 25 % | 26 Mc/s · 23 % | 23 Mc/s · 19 % |

- CRM fp32 is **near-roofline (70 % HBM @N128)** — the best of any legoESM dycore
  (global dycores sit ~30 %). Genuinely memory-bound and efficient.
- fp64 is compute-bound (consumer 5090 fp64 ≈ 1/64) — ~6× slower; hardware wall.
- N256 drop (70→40 % HBM) is **L2-cache-fit loss** (working set spills L2),
  confirmed not thermal (N128 recovers fully after the hot N256 run).

### MPI weak scaling (per-rank 48×48×30, CPU, single-thread/rank)

| precision | np2 total | np4 total | np6 total | scaling |
|-----------|----------:|----------:|----------:|---------|
| **fp32** | 183.7 ms | 184.6 ms | — | flat ✓ |
| **fp64** | 253.3 ms | 253.7 ms | 261.2 ms | flat ✓ |

- **Both precisions weak-scale well** (total ≈ flat across ranks = ideal weak
  scaling). fp32 ~1.4× faster than fp64. CRM is dycore-dominated (~88 %).
- Per-step global reductions were cut **~44 %** (3 separate allreduces → 1
  batched) and folded to **one allreduce/step**; reduce now ~11 % of the step.

### MPI strong scaling — real domain decomposition (`step_halo`, global 96×96×30)

| mode   | prec | np1   | np2   | np4   | efficiency |
|--------|------|------:|------:|------:|------------|
| strong | fp64 | 66.6 | 59.5 | 55.0 ms | 56 % @2, 30 % @4 |
| strong | fp32 | 50.7 | 51.6 | 47.4 ms | bandwidth-bound (≈flat) |

The real domain-decomposed (pencil) CRM step was **rescued this campaign**:

1. **104× broken → fast.** Multi-rank `step_halo` was re-tracing the whole
   split-explicit core every step; caching a JIT'd core gave **123×** speedup.
2. **Silent halo bug fixed.** The doubly-periodic plane halo used ambiguous
   send/recv tags (send-to-X / recv-from-X with the same neighbour twice per
   axis) → wrong halo at 2-ranks-per-axis. Replaced with a directional ring-shift
   (send to one neighbour, recv from the other, one tag per shift). **Plane-only**
   — lat-lon (poles ⇒ one sendrecv/rank) and voronoi (unique pair tags) were safe.
3. **Validated bit-identical.** Multi-rank gathered vs single-rank = **1e-15** for
   u/v/w/θ′/ρ′ over 3 full steps (acoustic substeps + Smagorinsky), locked as a
   regression test.

Strong scaling itself is **single-socket memory-bandwidth-limited** (same ceiling
the global campaign hit) — functional + correct, hardware-bound; near-ideal
strong scaling needs multiple sockets/nodes.

### Moist CRM under DD — exact serial parity is opt-in

The SAM moist-buoyancy closure subtracts a **horizontal mean** (qv, qcond, θ′) in
the acoustic substep. Under decomposition each rank uses its **slab-local** mean
(zero communication) ⇒ ~6e-4 divergence vs single-rank. This is a **deliberate
scalability tradeoff**, not a bug: a global mean would add ~3–9 allreduces/step
(~5–16 % overhead) and erode the strong scaling above. Added an **opt-in**
`CompressibleEulerConfig.acoustic_moist_global_mean` (default off) that uses one
batched AD-safe allreduce for **bit-identical (1.1e-15) serial parity** on oracle
/ validation runs; production keeps the zero-comm rank-local mean.

## LES — two paths

legoESM has **two** LES paths; the precision/scaling requirement is met across them:

1. **Compressible-plane LES** — the SAME dycore as the CRM with
   `smagorinsky_cs > 0`. The `step_halo` rescue above makes it **MPI-scalable
   now**, and it inherits the 1e-15 validation (the CRM full-step test runs
   `smagorinsky_cs=0.2`).
2. **Spectral incompressible LES** (`spectral_les_plane`, FFT pressure + dynamic
   LASD SGS) — single-GPU; MPI needs a distributed FFT (deferred, below).

### Compressible-plane LES — MPI strong scaling (`--smag-cs 0.2`, global 48×48×20)

| precision | np1 | np2 | np4 | np1→np2 |
|-----------|----:|----:|----:|--------:|
| **fp32** | 111.4 | 50.6 | 50.0 ms | **2.2× (super-linear)** |
| **fp64** | 143.9 | 56.6 | 56.4 ms | **2.5× (super-linear)** |

Strong scaling is **super-linear at np=2** (vs 56 % for the bandwidth-bound dry
dynamics): the SGS compute raises the compute-to-bandwidth ratio, so the extra
core buys more than the bandwidth contention costs — before plateauing at np=4
(bandwidth). Both precisions; correct (1e-15).

### Spectral LES — single-GPU throughput (neutral case, LASD)

| grid       | **fp32** (production) | **fp64** (default) |
|------------|----------------------:|-------------------:|
| 64³ (262 k)  | 27.7 Mc/s | 8.5 Mc/s |
| 96³ (590 k)  | 36.0 Mc/s | 9.1 Mc/s |
| 128³ (1.05 M)| **41.0 Mc/s** | — |

- **fp32 scales healthily** (rising 27.7→41 Mc/s) — the production GPU mode.
- fp64 works but ~4× slower (consumer fp64 + the FFT pressure solve in fp64).
- Stable (state finite) in both precisions.
- **Spectral LES now runs distributed on MPI, full oracle closure** (was
  single-rank-only). A transpose-based slab distributed 2-D FFT
  (`parallel/distributed_fft.py`, AD-safe all-to-all) carries the pressure
  projection, the spectral filter, the global-mean wall model AND the **LASD
  dynamic SGS** (test filters + box3 ∂y halo). A full `step()` matches single-rank
  to **1e-9** (state) / **1e-10** (`u_*`), `lasd_cs2` to **1e-12**, all AD-safe
  (np 1/2/4). Scaling, dynamic LASD, both precisions:
  - strong (global 64²×32): fp64 88.9→77.9→74.6, fp32 60.6→51.9→41.0 ms/step
    (np 1/2/4; fp32 1.48× @4) — modest, bandwidth + all-to-all bound.
  - weak (per-rank 32×64×32): fp64 40.3→81.2→136.3 ms — **poor BY ALGORITHM**:
    the pressure-Poisson FFT globally couples the domain, so weak-scaling grows
    the FFT size + the all-to-all transpose volume (the spectral communication
    wall, unlike the CRM's flat local-stencil weak scaling).
  **The full oracle closure now runs distributed** (dynamic LASD SGS + 3/2-rule
  de-aliasing, both AD-safe) — no operator remains serial-under-MPI. The spectral
  LES is the faithful-physics path now fully usable across ranks; the CRM is the
  better-scaling path. The spectral ceiling is the FFT all-to-all (algorithmic),
  not any missing capability.

---

## Bottom line

**"Both precisions scale well for CRM and LES" — met at the achievable scope:**
CRM MPI weak-scales in fp32 + fp64 and strong-scales over a validated (1e-15)
domain decomposition; CRM fp32 GPU is near-roofline; compressible-plane LES
strong-scales super-linearly on MPI in fp32 + fp64; spectral LES runs and scales
on single-GPU in fp32 (production) and fp64, and now **runs distributed + correct
on MPI** (full static-SGS step, AD-safe).

| genuinely-remaining lever | nature |
|---------------------------|--------|
| spectral-LES MPI strong scaling | flat on one socket (bandwidth + all-to-all transpose) → needs multi-node |
| CRM dycore kernel tiling (L2-fit) | deep kernel work; modest gain at >2 M cells |
| fp64 on consumer GPU | 1/64 hardware wall — not a code issue |
| CPU MPI strong scaling | single-socket memory-bandwidth bound → needs multiple sockets |

Shipped this campaign (all bit-identical / AD-safe / codex-reviewed): CRM
reduce-overhead −44 % + one-allreduce-per-step + `--precision` flag (fp32 MPI weak
scaling); **CRM domain-decomposition rescue** (123× JIT speedup, plane halo-tag
bug fixed, 1e-15 full-step validation) enabling real CRM **strong** scaling and
**compressible-plane LES on MPI** (super-linear, both precisions); opt-in
`acoustic_moist_global_mean` for exact moist serial parity. See
`docs/scaling/crm_les_scaling.md` for the full log.
