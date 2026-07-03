# Scaling missed-opportunity audit — 2026-06-15 (codex, read-only)

Re-rank after the **banded distributed multigrid barotropic preconditioner**
(task #26) shipped + was measured (jacobi M=60 = 120 reductions/step and
unconverged vs banded-MG M=12 = 24 reductions/step to 1e-6; V-cycle adds zero
global reductions; banded==serial under mpirun -np 2/4). Supersedes the ranked
list in `scaling_levers_audit_2026-06-14.md`.

Hardware: Ginsburg — CPU-MPI nodes (≤16 ranks/node policy, Gloo/TCP, no IB) +
2× RTX8000 (PCIe, no NVLink).

## Highest ROI next
1. **MPAS batched-halo repair** (ocean+atm) — **DONE 2026-06-15.** The "18×
   regression" was I5/f32-stale; current code is FASTER batched at every size
   (I4/I5 f32 -12%/-5% job 8488057; I6 f64 np8/np16 -4.6%/-3.5% job 8488023).
   Flipped `_USE_BATCHED_HALO` default 0→1 (batched now production default,
   legacy opt-OUT). Gate 8488077 OK + codex-clean. ~4-12% on every MPAS/voronoi
   run.
2. **MPAS/Voronoi indirect-gather fusion** (ocean+atm GPU) — **MEASURED DEAD on
   this HW 2026-06-15** (job 8488104). The go/no-go was a command-buffer A/B at
   np1 (isolates per-device kernels): aggressive `FUSION,CUSTOM_CALL,CUBLAS,
   CUDNN,COLLECTIVES` == default `FUSION,CUSTOM_CALL,COLLECTIVES` to noise (f32
   11.03 vs 11.07 ms/step) ⇒ atm-ico is NOT dispatch-bound (already
   well-fused; 96.5 Mcells/s ≈ per-device limit). Fusing the gathers (Pallas/
   restructure) would not cut a dispatch wall that isn't there; the gathers are
   bandwidth-bound, where fewer kernels don't reduce traffic. Skip on Ginsburg
   (would need a much faster GPU to expose dispatch headroom).

## Ranked remaining
3. METIS/default partition audit for MPAS — **MEASURED 2026-06-15 (job 8488136),
   small opt-in win at scale.** icosahedral I6 f64 RCB-vs-METIS: np8 neutral
   (66.68 vs 67.32 ms/step, noise), np16 **METIS +1.9%** (59.93 vs 58.82) — the
   edge-cut-min win GROWS with rank (boundary/interior ratio), so >np16 likely
   approaches the +5-20% range. RCB is already well-balanced on the near-uniform
   icosahedral mesh, so the low-rank win is ~0; the gain is high-rank only. Kept
   **UPDATE 2026-06-21 — flipped to a capability-gated `method="auto"` default**
   (branch `perf/voronoi-graph-partition-sfc`): `auto` selects METIS when
   `pymetis` is importable, else falls back to geometric RCB. The win is now
   realized automatically wherever pymetis is present (the `[mesh]` extra) with NO
   hard dependency and byte-identical behavior where it is absent
   (dev/CI/shared venv), so the earlier "not worth a hard dep" objection no longer
   applies. Also added a dependency-free `method="sfc"` (Hilbert space-filling
   curve) for locality-preserving contiguous partitions, and
   `reorder_voronoi_for_sharding` now Hilbert-orders cells within each shard.
   Still install `[mesh]` for large multi-node MPAS where the high-rank edge-cut
   win (>np16, trending +5–20%) matters.
4. Root-only/gathered checkpoint + diagnostics writers — `scripts/run/run_omip.py`
   / matrix I/O; big wall-clock only at high output cadence, ~0 step-kernel gain.
5. Tripole wiring (lat-lon C-grid) — capability (eORCA scaling), not speedup.

## MG agglomeration — NOT worth at ≤8 ranks
One allreduce/V-cycle ⇒ break-even ≈ `2*(M_band - M_agglom) > M_agglom`. Our np4
`M=20` vs agglomerated `~12` is only marginal before coarse-solve overhead; np2
worse. Do it ONLY for many-rank runs where the even-alignment depth cap drives
clear M-growth.

## Missed lever — CUDA-graph ATM icosahedral: MEASURED, ~DEAD
- **CUDA-graph A/B for ATM icosahedral only** (env/bench, not model code).
  Codex audit 2026-06-15 named this as the ONE untried measurable step-kernel
  lever. Measured:
  - job 8488104 (A40, I6 np1): default-cmdbuf (FUSION,CUSTOM_CALL,COLLECTIVES)
    vs +CUBLAS,CUDNN = **noise** — f32 11.07→11.03 ms/step, f64 20.50→20.42.
    Adding CUBLAS/CUDNN command buffers does nothing.
  - job 8490224 (RTX8000, I6 np1): the missing **OFF arm** (command buffers
    fully DISABLED, `--xla_gpu_enable_command_buffer=`) vs default vs aggressive.
    **f32 RESULT: OFF=11.25, default=11.24, aggressive=11.23 ms/step — ALL EQUAL
    (noise).** Command buffers give ZERO benefit ⇒ zero dispatch headroom ⇒
    MPAS-atm GPU step is bandwidth/compute-bound, CUDA-graph lever **DEAD** on
    this HW, at-limit confirmed (f64 arms confirmatory). (Unlike MPAS-ocean fp32
    where CUDA graphs gave 2.4–2.9×; cubed-sphere measured negative earlier.)

## Codex at-limit verdict (2026-06-15)
Focused codex audit ("name a MEASURABLE Ginsburg lever NOT yet tried"): **mostly
at practical limit.** Only one honest untried step-kernel A/B (CUDA-graph ATM,
above — now being closed); the rest are workflow/I/O (root-only checkpoint = 0%
step ms) or narrow A/B extensions (MPAS METIS at np32, expected 2–5%, "not a new
lever"). Everything else = harvested, measured-dead, or PCIe/Gloo/TCP-blocked.

## HW-blocked — stop pursuing on Ginsburg
- Cube np>6 (Gloo/TCP + PCIe anti-scale) — future-HW capability only.
- 2-D lat-lon decomposition speedups (Gloo latency + pole/full-lon transpose).
- CUDA-aware MPI rebuild (stack-blocked; 2-GPU full step already near-ideal).
- Comm/compute overlap via nonblocking MPI (blocking mpi4jax/XLA schedule).
- Spectral multi-GPU global transform on PCIe (all-to-all dominated; GEMM is the
  right local lever).
