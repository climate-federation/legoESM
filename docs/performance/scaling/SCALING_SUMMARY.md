# legoESM AMIP scaling summary — atmosphere & ocean (single-node)

> **Scope tag:** this document is a **single-device throughput-vs-size sweep**
> (category (b) in `SCALING_STATUS_AUDIT.md`), NOT true multi-device weak/strong
> scaling. All §1–3 numbers are one-GPU/one-CPU saturation curves. For the
> authoritative cross-component support matrix (which grids have real
> multi-device evidence, which are infrastructure-ready, which are N/A), see
> **`SCALING_STATUS_AUDIT.md`**. §4 below covers the MPI multi-rank story.

Consolidated single-node performance for AMIP-like configs across **all grid
types**, on **MPI and GPU**, at **float32 and float64**. Held-Suarez is the
atmosphere tester; the ocean uses the production `implicit_cn` barotropic solver.

**Test host:** 1× NVIDIA RTX 5090 Laptop (24 GB, 55 W TDP) + 24-core CPU.
All timings are warm steps inside a `lax.scan` (JIT-compiled), one representative
resolution per grid (L26 atmosphere, 20-level ocean).

> ⚠️ **Read before sharing.** (1) The 55 W laptop GPU **thermally throttles**
> under sustained load, so absolute GPU Mcells/s drift ~±40 % between runs;
> *ratios measured within one run are reliable*, absolute peaks are higher when
> cool (e.g. lat-lon atm peaked at 1100 Mc/s cool vs 678 here). (2) Single sizes,
> below saturation — larger grids run faster per cell. (3) True multi-device
> (multi-GPU / multi-node) scaling is **not measurable on this 1-GPU host**; the
> infrastructure exists and is tested but needs ≥2 GPUs.

---

## 1. GPU throughput (Mcells/s = cells×levels / step-time)

### Atmosphere (Held-Suarez)

| grid          | size  | fp32 CPU | fp32 GPU | fp64 CPU | fp64 GPU |
|---------------|-------|---------:|---------:|---------:|---------:|
| icosahedral   | I5    |    17    |   298    |    8     |    93    |
| cubed-sphere  | C48   |    25    |   404    |   15     |   115    |
| lat-lon       | LL128 |    32    | **678**  |   13     |   122    |
| spectral      | T42   |  (x64)   |  (x64)   |    1     |    13    |

### Ocean (`implicit_cn` barotropic solver)

| grid           | size       | fp32 CPU | fp32 GPU | fp64 CPU | fp64 GPU |
|----------------|------------|---------:|---------:|---------:|---------:|
| lat-lon        | LL192      |    4     | **400**  |    5     |   152    |
| tripole (eORCA)| 192×384    |    —     |   313    |    —     |   144    |
| MPAS Voronoi   | I6         |    4     |   252    |    4     |   200    |

---

## 2. GPU acceleration (CPU → single-GPU speedup)

| component | grid          | **fp32** | **fp64** |
|-----------|---------------|---------:|---------:|
| atmosphere| icosahedral   |  17.6×   |  11.7×   |
| atmosphere| cubed-sphere  |  15.9×   |   7.8×   |
| atmosphere| lat-lon       |  21.1×   |   9.6×   |
| atmosphere| spectral      |  (x64)   |  10.9×   |
| ocean     | lat-lon       |  73–99×  |  30–35×  |
| ocean     | MPAS Voronoi  |  33–71×  |  25–48×  |

- Ocean accelerates far harder (33–99×) than atmosphere (16–21×): the ocean step
  is ~5× heavier on CPU (barotropic implicit solve + EOS), so the GPU wins more.
- **fp64 acceleration is lower** everywhere — the consumer 5090's fp64 is ~1/64
  of fp32, while CPU fp64 ≈ CPU fp32, so the CPU loses less going to fp64.

---

## 3. Atmosphere physics tiers (GPU, fp32, cubed-sphere)

| tier                     | rel. step cost | note |
|--------------------------|---------------:|------|
| Held-Suarez (dynamics)   | 1× (0.9 ms @C48) | Newtonian relaxation |
| gray radiation + SBM     | ~3× (2.7 ms @C48) | radiation+convection+moist tracers |
| RRTMGP full radiation    | dominant        | **6× faster after a g-point-parallelisation fix** (see below) |

RRTMGP radiative transfer was found ~6000× slower than gray (a sequential
g-point scan that starved the GPU); parallelising the g-point axis gave **6.1×**
on the radiation kernel / **5.96×** end-to-end, bit-identical, training path
untouched.

---

## 4. MPI multi-rank scaling

This host has 1 GPU, so the strong-scaling column below is CPU-MPI / small-rank
only. Real multi-device evidence at scale lives in
`derecho_levante_sota_review_2026-07.md` (§3b, measured Derecho curves) and the
`SCALING_STATUS_AUDIT.md` support matrix.

| component | grid          | MPI multi-rank        | strong scaling (this host) |
|-----------|---------------|-----------------------|----------------------------|
| atmosphere| **icosahedral** | ✅ `auto` domain decomp (METIS graph-partition when `pymetis` present, else RCB; `sfc` Hilbert option) | 1.24× @2 ranks, plateaus @4 — sync-barrier/jitter-bound on a shared node; correct vs serial to ~1e-9 |
| atmosphere| cubed-sphere  | ✅ genuine ≤6-face decomposition via `run_levante_gpu_scaling.py --cs-mpi-scatter` (bit-equal 1e-15 vs serial); default (no flag) is replicated dynamics, refused for scaling claims | face-sharded SPMD; ≥2 GPUs (Derecho/Levante lanes) |
| atmosphere| lat-lon       | ✅ latitude-band decomposition (`make_latlon_mpi_step`, #641; pole_bc='wall' + diagnostic physics), validated vs serial | Derecho CPU-MPI 128 ranks (§3b of the SOTA review) |
| atmosphere| spectral      | rank-1 (global transforms) — N/A by design | — |
| ocean     | lat-lon C-grid | ✅ latitude-band MPI (`bench_ocean_mpi_scaling.py`, parity + conservation gates) + full-step SPMD (`bench_ocean_latlon_spmd_scaling.py`) | 2-GPU full step ~0.92 strong / 0.97 weak (Ginsburg); Derecho/Levante ladders exist, unrun |
| ocean     | tripole / MPAS / cube | infrastructure-ready, no dedicated scaling lane yet (see audit) | — |

`lat-lon MPI was #115's gap; closed by #641` — the old "not implemented" line
is superseded. Icosahedral, cubed-sphere (`--cs-mpi-scatter`), and lat-lon all
domain-decompose on MPI today; ocean lat-lon has both CPU-MPI and multi-GPU
SPMD harnesses. Spectral stays single-rank by construction. Production-scale
multi-device numbers are hardware-blocked on this 1-GPU host — see the audit
and SOTA review for the Derecho/Levante evidence.

---

## 5. Headlines

- **Structured > unstructured > spectral** on GPU, both components (coalesced
  memory vs indirect gathers vs global transforms).
- **Lat-lon fastest**; **cubed-sphere best-behaved** (cleanest saturation curve);
  **spectral GPU-hostile** (fp64-only + global FFT) → keep CPU-side.
- **Tripole ≈ lat-lon − ~20 % (fp32) / −5 % (fp64)** (north-fold halo +
  rotation overhead; smaller relative cost in fp64 where arithmetic dominates).
- **GPU buys 16–21× (atm) / 33–99× (ocean) at fp32**; roughly half that at fp64
  on this consumer GPU.
- Two shipped algorithmic wins dominate the gains: **RRTMGP g-point
  parallelisation (≈6×)** and the **ocean `implicit_cn` barotropic solver
  (1.3–2.5× vs explicit 30-substep)**.
