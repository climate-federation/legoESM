# Scaling missed-opportunity audit — 2026-06-15 (codex, read-only)

Re-rank after the **banded distributed multigrid barotropic preconditioner**
(task #26) shipped + was measured (jacobi M=60 = 120 reductions/step and
unconverged vs banded-MG M=12 = 24 reductions/step to 1e-6; V-cycle adds zero
global reductions; banded==serial under mpirun -np 2/4). Supersedes the ranked
list in `scaling_levers_audit_2026-06-14.md`.

Hardware: Ginsburg — CPU-MPI nodes (≤16 ranks/node policy, Gloo/TCP, no IB) +
2× RTX8000 (PCIe, no NVLink).

## Highest ROI next
1. **MPAS batched-halo repair** (ocean+atm) — `packages/core/legoesm/parallel/halo_exchange_voronoi.py:384`
   `pack_batched_sends`/`unpack_batched_recvs`. Recover the known 18× opt-in
   regression (435.9 vs 24.07 ms, I5 np8) then aim +10–30% CPU-MPI at high rank.
   The largest remaining non-HW wall outside the shipped lat-lon MG. The shared
   MPAS/Voronoi path still pays per-field/per-neighbour pack/scatter overhead;
   attacks message COUNT without needing IB.
2. **MPAS/Voronoi indirect-gather fusion** (ocean+atm GPU) —
   `primitive_eq_mpas.py:186`, `ocean_pe_mpas.py:360`, `voronoi.py:938`
   (`cellsOnEdge`/`edgesOnCell`/`edgesOnEdge`). +10–30% on RTX8000 icosahedral;
   throughput is gather-bound (uncoalesced indirect addressing → many small
   kernels), so fewer/fused gathers (or a Pallas kernel) is the next per-device
   gain, not more scan wrapping.

## Ranked remaining
3. METIS/default partition audit for MPAS — `voronoi_partition.py:204`,
   `voronoi_mpi.py:120`; +5–20% CPU-MPI if RCB cut/imbalance is bad (cheap A/B).
4. Root-only/gathered checkpoint + diagnostics writers — `scripts/run/run_omip.py`
   / matrix I/O; big wall-clock only at high output cadence, ~0 step-kernel gain.
5. Tripole wiring (lat-lon C-grid) — capability (eORCA scaling), not speedup.

## MG agglomeration — NOT worth at ≤8 ranks
One allreduce/V-cycle ⇒ break-even ≈ `2*(M_band - M_agglom) > M_agglom`. Our np4
`M=20` vs agglomerated `~12` is only marginal before coarse-solve overhead; np2
worse. Do it ONLY for many-rank runs where the even-alignment depth cap drives
clear M-growth.

## Missed lever
- **CUDA-graph A/B for ATM icosahedral only** (env/bench, not model code): 0–15%
  if atm MPAS shares the small-kernel dispatch pathology that CUDA graphs fixed
  for MPAS-ocean fp32 (2.4–2.9×); skip for cubed-sphere (measured negative).

## HW-blocked — stop pursuing on Ginsburg
- Cube np>6 (Gloo/TCP + PCIe anti-scale) — future-HW capability only.
- 2-D lat-lon decomposition speedups (Gloo latency + pole/full-lon transpose).
- CUDA-aware MPI rebuild (stack-blocked; 2-GPU full step already near-ideal).
- Comm/compute overlap via nonblocking MPI (blocking mpi4jax/XLA schedule).
- Spectral multi-GPU global transform on PCIe (all-to-all dominated; GEMM is the
  right local lever).
