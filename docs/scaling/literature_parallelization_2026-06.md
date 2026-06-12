# Parallelization literature review — levers for legoESM scaling (2026-06-12)

Deep-research sweep (104 agents, 22 sources fetched, 25 claims adversarially
verified 3-vote → 21 confirmed) mapped to the campaign's measured walls
(`ginsburg_mpi_gpu_scaling_plan.md`). Each lever cites its evidence; "(3-0)"
= unanimous verification. The four killed claims are noted where relevant.

## Wall 2 — ocean barotropic allreduce latency (THE strongest actionable)

Our measurement: fixed-M=60 Jacobi-PCG, 2M allreduces/step × ~111 µs latency
= the weak-scaling growth term; Chronopoulos–Gear single-reduce bought only
3–7%.

1. **P-CSI (preconditioned Classical Stiefel Iteration = Chebyshev-type) —
   ZERO reductions per iteration.** Iteration parameters preset from a
   one-time Lanczos estimate of the preconditioned operator's extreme
   eigenvalues; global reductions only in occasional convergence checks.
   Production-proven: default barotropic solver in CESM 2.0; on 16,875
   cores of Yellowstone it took 0.1° POP from 6.2 → 10.5 SYPD (1.7× whole
   model), barotropic share 50% → 16%
   (Hu et al., GMD 9:4209, 2016 — 3-0 ×3 claims).
   *Transfer:* fixed trip count + no reductions = exactly
   JIT/`lax.scan`/AD-friendly. Our earlier "Chebyshev SHELVED" verdict was
   a np≤8 *single-node* phase-split (barotropic 6–7% of step); the
   reduction term is the *multi-node weak-scaling* growth term — re-open
   for the np≥16 multi-node ladder.
2. **EVP block preconditioner — communication-free iteration cutter.**
   Exact direct solve within each rank-local block; ~5× condition-number
   reduction, ~2/3 iteration-count cut at 1° and 0.1°; doubled local FLOPs
   are a net win when latency dominates (same paper, 2-0). Composable with
   our PCG *and* with P-CSI. M=60 → ~20 with zero new collectives.
3. **AMG preconditioning (Veros' production answer).** Veros/JAX replaced
   its barotropic iterative solve with PETSc GAMG via petsc4py (CPU+GPU,
   distributed), projecting AmgX for larger GPU counts
   (Häfner et al., JAMES 2021MS002717 — 3-0). Heavier dependency; ranks
   below EVP+P-CSI for us (non-JAX solver breaks the AD story unless
   wrapped like our custom-VJP solver).
4. **Single-reduce merging is a dead end beyond what we shipped.** POP data:
   ChronGear's merge achieved only ~1/3 latency reduction; reduction share
   at ~4000 cores was 74% (PCG) vs 68% (ChronGear) (3-0) — consistent with
   our 3–7%. PIPEPCG/PIPEPCR (overlap the one reduction with matvec) exist
   for POP2 (one-reduction restructuring verified 2-0) but their headline
   ~1.9–2× @10k-core speedups did NOT survive verification (abstained) —
   treat as unconfirmed.

## Wall 1/5 — halo exchange + SPMD architecture

5. **Our shard_map+ppermute architecture is the literature's winning
   pattern.** JAX-Fluids 2.0 explicitly rejects mpi4jax for JAX-native
   pmap+ppermute *to preserve differentiability*, and weak-scales ω>0.95 to
   512 A100s / 1024 TPU cores (16B cells) (arXiv:2402.05193 — 3-0 ×2). The
   architecture is not our bottleneck; the **fabric** is (their A100s have
   NVLink+IB; Ginsburg RTX8000 pairs are PCIe, CPU nodes Gloo-TCP).
6. **Staged, stencil-gated halos.** JAX-Fluids updates face→edge→vertex
   halos in ordered directional ppermutes (diagonals relay through faces —
   no corner messages) and *skips* halo regions a stencil doesn't read
   (3-0). We already relay corners; the per-operator halo-region *skipping*
   is a census-driven trimming lever for the tiled stage.
7. **GSPMD-auto can silently insert a pathological AllGather consuming 80%
   of runtime; the prescribed detection is the device-profile timeline**
   (jax-ml scaling book — 3-0). Independently confirms our np24
   GSPMD-auto 39× story + HLO-census discipline. (The JEP claim that
   shard_map is "only a surgical escape hatch" was REFUTED 1-2; the
   scaling book positions explicit collectives as first-class for
   comm-critical code.)
8. **Manual comm/compute overlap in shard_map works:** stepwise
   ppermute+partial-compute "collective matmul" removed ~77% of the
   communication overhead vs a blocking AllGather (244 µs vs 311 µs, 224 µs
   unsharded baseline) (3-0). The lighter sibling of our parked deep-halo
   idea — applicable to halo+interior-stencil overlap in the tiled stage.
   (The JEP's own transformer overlap example claim was REFUTED 0-3 — cite
   the scaling book, not the JEP, for this pattern.)
9. **CPU same-node wall is universal, not a legoESM defect.** Veros/JAX on
   8×32-core CPU nodes: within 1.4× of Fortran+MPI, the gap attributed to
   DRAM-bandwidth-bound execution + XLA's incomplete fusion; at high counts
   communication + barotropic dominate (3-0). Keeps our 1-proc/node +
   ranks/node≤8 policies evidence-backed.
10. **GPU saturation rule: ≳10⁶ grid elements per device** before weak
    scaling is meaningful; Veros 16×A100 NVLink strong scaling = 3.4×,
    weak (saturated) ≈ 0.8 efficiency (3-0). Our C48 L10 lanes ≈ 2.3×10⁵
    cells/device at np6 — 4× under-saturated by construction. Production
    scaling claims need C96–C192-class per-device tiles.
11. **mpi4jax with CUDA-aware zero-copy is Veros' transport** (3-0) — an
    *alternative* cross-node transport to Gloo-TCP. Only worth revisiting
    if profiling shows Gloo latency (not volume) dominating cross-node
    ppermute; a transport swap must never mix with jax.distributed in one
    program (our documented deadlock hazard).

## Wall 4 — per-device kernels

12. **Custom-call tridiagonal kernels are worth ~10% whole-model** (Veros:
    Cython/CUDA Thomas via custom_call) (3-0) — calibrates expectations;
    matches our vmix-batching null result. Not a priority lever.
13. **Pallas custom kernels can DEFEAT XLA's automatic async collective
    dispatch** (kernels are opaque to XLA) (3-0); Pallas async remote-DMA
    overlap is TPU-only (3-0). Do NOT Pallas-ify communication paths on
    our hardware; per-device Pallas stencil fusion remains a possible
    future GPU lever but with the opacity caveat.

## What NOT to pursue (evidence-backed)

- Further single-reduce/merge-only Krylov variants (POP: ~1/3 cap; ours 3–7%).
- Pallas communication kernels (XLA opacity; TPU-only RDMA).
- NeuralGCM's recipe for our FV3 core — its 70,000 sim-days/day is
  *single-TPU* throughput of a spectral hybrid-ML core (3-0 ×2), not a
  multi-device stencil-scaling result.
- Chasing same-node CPU multi-process strong scaling past the DRAM wall.

## Measurement discipline ("clocked" attribution)

- **Device-profile timeline first** (jax.profiler → Perfetto/TensorBoard):
  the JAX team's prescribed tool for catching collective stalls (the 80%
  AllGather class). Next concrete use here: trace the production np=2
  multinode lane to attribute its +110 ms/step (collective stall vs host
  rendezvous vs compute gap).
- **HLO dump + op_name census** (our drv_spmd_hlo_census.sbatch): static
  collective counts attributed to jax source lines — proved the production
  segment collective-clean and redirected us to the segment-collapse root
  cause.
- **Timing hygiene** (Veros): mean wall per iteration over ≥10 iters,
  discard the first 1–2 (JIT), exclude I/O; report per-phase splits. Ours
  adds: parse ACTUAL runtime params (dt clamp), measure post-JIT segments
  only, same-job same-node A/B/A/B lanes.
- Stack for this cluster: jax.profiler traces (per-process, SLURM lane) +
  XLA HLO dumps + our roofline_probe.py anchors + phase-split benches.

## Verified-claim ledger

21 confirmed / 4 killed, 9 primary sources; full claim JSON with quotes:
campaign session artifact (deep-research run wf_cf95fbce-5b0). Key sources:
arXiv:2402.05193 (JAX-Fluids 2.0); JAMES 10.1029/2021MS002717 (Veros);
GMD 9:4209 (2016) (POP P-CSI/EVP — CESM 2.0 default); Nature
s41586-024-07744-y (NeuralGCM); jax-ml scaling book; JAX Pallas
distributed docs; CCF-THPC s42514-025-00226-1 (POP2 PIPEPCG, partially
verified).
