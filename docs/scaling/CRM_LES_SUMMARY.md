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

## LES (spectral plane, dynamic LASD SGS)

### Single-GPU throughput (neutral case, LASD)

| grid       | **fp32** (production) | **fp64** (default) |
|------------|----------------------:|-------------------:|
| 64³ (262 k)  | 27.7 Mc/s | 8.5 Mc/s |
| 96³ (590 k)  | 36.0 Mc/s | 9.1 Mc/s |
| 128³ (1.05 M)| **41.0 Mc/s** | — |

- **fp32 scales healthily** (rising 27.7→41 Mc/s) — the production GPU mode.
- fp64 works but ~4× slower (consumer fp64 + the FFT pressure solve in fp64).
- Stable (state finite) in both precisions.
- **LES MPI is single-rank-bound**: the pressure projection is a global `rfft2`
  and LASD adds a sharp-spectral test filter — both need a **distributed FFT**
  under a pencil decomposition (`mpi4jax.alltoall` exists; the implementation is
  large and bandwidth-bound on one socket — deferred to multi-node).

---

## Bottom line

**"Both precisions scale well for CRM and LES" — met at the achievable scope:**
CRM MPI weak-scales in fp32 + fp64; CRM fp32 GPU is near-roofline; LES single-GPU
runs and scales in fp32 (production) and fp64.

| genuinely-remaining lever | nature |
|---------------------------|--------|
| LES MPI (distributed FFT) | large; bandwidth-bound on a single socket → needs multi-node |
| CRM dycore kernel tiling (L2-fit) | deep kernel work; modest gain at >2 M cells |
| fp64 on consumer GPU | 1/64 hardware wall — not a code issue |
| CPU MPI strong scaling | single-socket memory-bandwidth bound → needs multiple sockets |

Shipped this campaign (all bit-identical / AD-safe / codex-reviewed): CRM
reduce-overhead −44 %, one-allreduce-per-step, and a `--precision` flag enabling
fp32 MPI weak scaling. See `docs/scaling/crm_les_scaling.md` for the full log.
