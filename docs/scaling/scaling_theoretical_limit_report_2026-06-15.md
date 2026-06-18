# legoESM Scaling Campaign — Theoretical-Limit Report (2026-06-15)

> **ADDENDUM (2026-06-16) — the "at-limit" verdict below was broken through by
> SOTA-ALGORITHMIC inspiration.** The verdict was correct for the *existing
> algorithm at f64*; it was NOT a true theoretical limit. A user directive to
> draw on SOTA codes (MOM6, MPAS-Ocean, Oceananigans, NeuralGCM) found **two new
> ocean wins the within-algorithm audits had missed** — because those audits only
> tuned the *existing* schemes, never questioned the *algorithm choice* or the
> *precision*:
> 1. **Split-explicit barotropic + reduction-free local eta-floor clamp**
>    (MOM6/MPAS-O): replaces the implicit-CN PCG's ~120 allreduce/step with a
>    halo-only subcycle. Same-job head-to-head, LL192 f64 strong: explicit beats
>    implicit at ≥2 nodes, **1.15/1.23/1.65× at np16/32/64** (implicit
>    anti-scales np32→np64 on the reduction wall); weak eff 0.92 vs 0.80;
>    conservation eta_drift 1.4e-9; cold-start-stable (smoke 3.5 days). Opt-in
>    `barotropic_local_subcycle_clamp`. Jobs 8498971/8499266/8499635/8499869/8500305.
> 2. **Mixed-precision vmix** (f32 work / f64 state + f64 column-mass correction;
>    Oceananigans/NeuralGCM): **1.15× full-step GPU** at production scale (LL192
>    18.4→16.0 ms), conservation-exact. Opt-in `LEGOESM_VMIX_F32_SOLVE`. Job 8501006.
>
> **LESSON: "at-limit for the current algorithm + precision" ≠ "at theoretical
> limit."** The within-algorithm audits (preconditioner tuning, halo fusion,
> CUDA-graph) genuinely exhausted their space; the breakthrough came from a
> different *algorithm* (split-explicit) and a different *precision* (mixed). Both
> are opt-in + codex-clean; production-uptake gated by full-OMIP long-run
> validation. Remaining f32 extensions (baroclinic EOS/PGF, tracer-advection
> limiters) are large, science-risky careful-numerics. **PURSUED 2026-06-18 (see
§3 row "Mixed-precision baroclinic"): precision-VIABLE (offline experiment 8520588:
f32-anomaly PGF relRMS 8e-5 — the fp64-EOS policy rule is overcautious for the
ρ′~O(1) anomaly path) but NOT shipped** — production runs `PrecisionPolicy.fp64()`
so the EOS computes f64 regardless of input, and a *working* f32-EOS lever needs a
broad shared-EOS-compute override / refactor not justified by the marginal
(16–21 % baroclinic × ~½ EOS+PGF, GPU-mainly) upside. With this, the SOTA-extension
space the addendum opened is exhausted — the campaign is fully converged for the
current hardware.

**Verdict: the production step-kernel scaling surface is exhausted on Ginsburg
hardware.** Across ocean (lat-lon C-grid), atmosphere (cubed-sphere FV3, lat-lon
FV, icosahedral MPAS, spectral) and every parallel axis (per-device, 2-GPU,
CPU-MPI single- and multi-node), no untried lever with credible >5% step-kernel
upside remains. This conclusion is supported by **four independent
confirmations**: two codex adversarial at-limit audits, a CUDA-graph A/B measured
dead (OFF==default both precisions), and the METIS partitioning lever fully
characterized (+2.6%, opt-in). The remaining engineering frontier — sub-face
tiling of cubed-sphere faces beyond 6 devices — is **future-hardware capability**
(it anti-scales on this fabric) and is being built concurrently; it is not a
Ginsburg speedup.

This report converts the campaign into defensible knowledge per the codex
decision review (2026-06-15): hardware envelope, every harvested win and killed
lever with job IDs, the residual-matched solver evidence, the Amdahl bounds, a
lever taxonomy, and a "do not repeat" section.

---

## 1. Hardware envelope (the hard constraints)

| Resource | Spec | Consequence for scaling |
|---|---|---|
| GPU | Quadro RTX 8000 / A40, in **PCIe pairs** | no NVLink/no P2P → cross-GPU `ppermute` 70–245× slower than HBM (107 µs latency floor, 1.7 GB/s @256 KB, 5.8 GB/s asymptotic, jobs 8458801/2) |
| GPU interconnect | **no NVLink, no IB confirmed** | 2-GPU strong scaling is link-roofline-capped (~0.73 eff at production size); >2-GPU SPMD = right architecture, wrong machine |
| CPU fabric | **Gloo/TCP** collectives | `allreduce` ~111 µs flat (latency-bound; batching free), `sendrecv` ~313 µs; halos ≈ reductions at moderate ranks |
| CPU node | 8 ranks/node policy (≤8 at LL192 — DRAM-bandwidth contention above) | multinode only scales 1-proc-per-node for bandwidth-bound dynamics |
| Per-device | dynamics at 1–3 % of HBM roofline (5-field prognostic floor); 51 µs kernel-launch floor | per-device is dispatch/compute-bound, NOT bandwidth-bound — XLA-fusion territory, not a comm lever |
| Precision | RTX8000/A40 f64 = 1/32 f32 throughput | f64 production runs are FLOP-bound on GPU; f32 storage is the only per-device GPU lever (scientific choice, not taken) |

Always anchor "%-of-limit" claims to `scripts/bench/roofline_probe.py` (the
measured roofline), never to efficiency ratios alone.

---

## 2. Harvested wins (shipped, measured)

| Lever | Grid / axis | Result | Job |
|---|---|---|---|
| Multiface ppermute halo (fix fake 2-GPU) | cube 2-GPU | eff 0.46→0.64 (f32 C192), **0.73 f64** (PCIe roofline) | 8457520 |
| Fused multi-field halo + static-metric fold | ocean latlon CPU-MPI | strong np8 eff 0.35→**0.55**; weak growth 4.65→2.95; 63→39 exch/step | 8459326 |
| Halo-coalesce sweep (vmask/neumann/wind-stress) | ocean latlon | 39→**32** exch/step | 8475102/8475945 |
| A1 multinode SPMD (1 proc/node) | cube | **2.42×** np6 (C96) | 8460192 |
| Spectral SH transform → `dot_general` GEMM | atm spectral | biggest per-device atm win | (shipped) |
| T+S shared-factor vmix solve | ocean | np1 **1.15×** (LL192) — removes dup factor/traffic | 8460192 |
| CPU tridiagonal → LAPACK `gtsv` FFI | atm vmix CPU | **8.4–13.0×** (opt-in `LEGOESM_TRIDIAG=lapack`) | 8478438 |
| RK3 momentum: skip tracer-tend + reuse frozen EOS/pressure | ocean per-step | overhead 31.3→**21.7 ms** (−31 %) | 8487604 |
| Ocean full-step 2-GPU (amortization) | ocean | **0.92** eff at production scale (360×720 n60) | 8486182 |
| Atm lat-lon / icosahedral 2-GPU | atm | **0.82 / 0.83** eff (halo-bound, scales cleanly) | 8486204/8486333 |
| Ocean WEAK multinode at production tile | ocean | **0.97** near-flat (np8→16, ~590k cells/rank) | 8489559 |
| MPAS batched union-neighbor halo (default flip) | icosahedral | **+4.6 %** np8, +3.5 % np16 | 8488023 |
| `ln_ps`/`hybrid_factor`/`div_v` ride the cube-PE stage pack | cube SPMD | −3 collectives/RK3-substep (count-cut, bit-identical) | 8481480 |

---

## 3. Killed levers / dead ends (measured, do not retry on this HW)

| Lever | Why dead | Evidence |
|---|---|---|
| CUDA-graph / command-buffer (ATM MPAS) | OFF==default==aggressive (f32 11.25/11.24/11.23; f64 30.56/30.55) — zero dispatch headroom | 8490224 |
| 2-D lat-lon decomposition | 2-D halo HURTS vs 1-D band (1.73× slower np16) — Gloo latency + pole/full-lon transpose | 8477039 |
| Chebyshev barotropic preconditioner | **DIVERGES at LL192** (deg-4 resid M60=0.25) — earlier "1.23×" was a non-solve timed without residual; RETRACTED | 8489419 |
| Banded multigrid barotropic (wall-time) | converges (M12→1e-6) but **6–9× SLOWER** — zonal-line V-cycle compute ≫ jacobi diagonal; reductions weren't the bottleneck | 8488551 |
| Cube np>6 (sub-face tiling speedup) | anti-scales on Gloo/TCP+PCIe → capability only, not speedup | 8482583 |
| Spectral multi-GPU global transform | all-to-all dominated on PCIe; local GEMM is the right lever | (roofline) |
| Spectral **level-axis sharding** (multi-device) | MEASURED anti-scaling: T85 **0.74×** @4dev, T42 flat — the semi-implicit (nlev,nlev)/wavenumber solve all-gathers levels + SH transforms emit no collectives ⇒ single-device by design (`_valid_gpu_counts→[1]`). Both spectral multi-device schemes (level-shard AND transpose all-to-all) are now dead on this HW | 8520392 (`spectral_level_shard_cliff.md`, 2026-06-18) |
| CUDA-aware MPI rebuild | conda mpi4py shadows the cuda-aware libmpi → segfault; full-stack rebuild not worth it (prod already 0.92–0.95) | 8486212 |
| MPAS METIS partitioning | +2.6 % np16 (rank-growing, opt-in, pymetis dep) — real but "not a new mechanism" | 8491002 |
| Field-batched voronoi halo (the "18× regression") | was I5/f32/stale-code; batched actually WINS at I6/f64 (now default) | 8488023 |
| Mixed-precision baroclinic EOS/PGF (f32 work) | PURSUED 2026-06-18 (user ask) → precision-VIABLE but NOT shipped (v1 reverted). Offline experiment (job 8520588, thermal-front state): f32-anomaly EOS preserves the horizontal PGF to relRMS **8e-5**, spurious \|v\| ~1 mm/s/day — i.e. the precision-policy `equation_of_state`/`pressure_gradient` f64 rule is **overcautious for the baroclinic ANOMALY path** (ρ′~O(1) is f32-representable; the rule targets naive full-ρ~1025). BUT production OMIP sets `PrecisionPolicy.fp64()` (`run_omip_core2.py:2498`), so `wright_eos` (eos.py:111) computes the EOS in f64 regardless of input dtype → a *working* f32-EOS lever requires OVERRIDING that shared fp64 EOS-compute (broad: also hits non-anomaly EOS consumers the f64 rule legitimately protects, or a multi-call-site `compute_dtype` refactor through `make_eos_fn`). Marginal upside (baroclinic 16–21 % of step × ~½ EOS+PGF, GPU-mainly) does not justify the refactor. codex caught the v1 (input-cast) as a no-op under fp64. | experiment 8520588 + phase-split 8458934/8489559 |

---

## 4. Barotropic solver — residual-matched evidence (the crux)

The ocean weak/multinode wall is the fixed-M Jacobi-PCG reduction latency: M=60 ⇒
2M = **120 allreduce/step**, and at production accuracy this is near-optimal.
Every alternative was tested **with the residual checked** (job 8487973, LL192
np32 sweep; verdict `barotropic_multinode_verdict_2026-06-15.md`):

- **jacobi** M60: rel_residual ~6.6e-3 (crawls, never fully converges) but cheap
  per-iter (~0.07 ms/iter diagonal) → near-optimal at the loose production bar.
- **chebyshev** deg-4: **diverges** (M60=0.25) — cannot span the polar-anisotropic
  condition number.
- **banded multigrid** (zonal-line smoother): the ONLY true converger (M20→2e-8),
  but V-cycle compute (~6–13 ms/iter) ⇒ **wall-time-negative**.
- **single_reduce** (Chronopoulos–Gear): ~1.05× (half reductions, same
  convergence) — the one clean small win, opt-in.

Bottom line: **no cheap wall-time fix at production accuracy.** Multigrid is an
**accuracy** lever (stricter tolerance), not a Ginsburg speed lever. A fast
non-solve is not a win.

---

## 5. Amdahl / phase bounds (why strong degrades, weak does not)

- Ocean full-step STRONG (LL192, jacobi, job 8489477): np8→np16 **0.88**, np16→np32
  **0.55**. The np32 fall-off = barotropic 120-allreduce (≈24 % of step) + growing
  baroclinic halos — a fixed latency term against a shrinking per-rank tile.
- MPAS STRONG (I6, job 8489516): np32 **0.69** — scales better than ocean (no
  barotropic reduction wall; pure halo-bound dycore).
- Ocean WEAK at production tile (job 8489559): **0.97** — at ~590k cells/rank,
  compute dominates the fixed reduction latency, so weak scaling is excellent.
  (The campaign's early "weak 4.79×" was a tiny-tile artifact: rows/rank=48 is
  reduction-bound. **Weak efficiency is tile-size-dependent — measure at
  production tiles.**)
- 2-GPU: near-ideal at production size (ocean 0.92, atm 0.82–0.83); comm-bound only
  in the small/weak tail (PCIe latency).

**Realistic big-per-rank production runs scale well.** The "limit" is the
small-tile strong regime on the Gloo/TCP/PCIe fabric — a hardware property.

---

## 6. Lever taxonomy (keep these categories distinct)

- **Measured speedup (Ginsburg):** §2 — all harvested.
- **Accuracy lever (not speed):** banded multigrid barotropic (converges where
  jacobi crawls; ship if a stricter barotropic tolerance is ever required).
- **Capability / future-HW (no Ginsburg payoff):** cube >6-device sub-face tiling
  (10 shared 3D-PE ops + 1 composed vertical-transport stage np24-tiled this
  campaign; full `fv3_hydrostatic` momentum stage exists / concurrently in
  progress). Pays off only on NVLink/IB/TPU-ICI.
- **Workflow / I/O (different objective):** root-only gathered checkpoint +
  diagnostics writers (0 % step-kernel, but real total-wall at high output
  cadence) — the one open item if the metric becomes end-to-end wall clock.

---

## 7. Do not repeat (process lessons banked)

1. **Always check the residual before claiming a solver speedup.** A fast
   non-solve (chebyshev "1.23×") is not a win.
2. **Never grep bench stderr to `ms/step` only** — it hid two real errors this
   campaign. Grep BOTH the metric AND `Error|Traceback`.
3. **Weak-scaling efficiency is tile-size-dependent** — measure at production
   tiles, not tiny ones.
4. **Scaling claims need mechanical guards** (HLO tripwires, device-count asserts,
   parity gates) — every silent fallback produced months-credible fake numbers
   (the fake 2-GPU 0.500, the deadlocked "dry latlon MPI parity").
5. **A parity gate is only as good as its last matched config** — re-run it
   whenever a solver/dispatch changes; knob-bisect before blaming the newest edit.
6. **Shared checkout:** stage explicit pathspecs only; never `git add -A`; a
   concurrent agent may hold uncommitted WIP in your target file.

---

## 8. Recommendation

Conclude the Ginsburg scaling campaign. Production configurations scale to the
hardware roofline; the remaining frontier is future-HW (NVLink/IB/TPU) capability
and, separately, end-to-end workflow I/O if that becomes the metric. Reopen only
with new hardware or a new objective. Full per-row history:
`docs/scaling/scaling_indicators.csv`; barotropic detail:
`barotropic_multinode_verdict_2026-06-15.md`; lever audit:
`scaling_levers_audit_2026-06-15.md`.
