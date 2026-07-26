# Evidence pack — Levante scaling campaign 2026-07-24 (post-merge tree d3ec1ccce)

## Machines
Levante gpu partition: 4x A100-80 SXM (NVLink/NVSwitch) per node, IB HDR200 across nodes.
CPU compute nodes: 2x AMD Milan 7763 (128 cores).

## Ocean lat-lon SPMD GPU (single process, solver-matched, f64 L20, 33 steps)
LL384x768 (5.9M cells): implicit-CN fixed-PCG(60 iters, jacobi): 29.3 / 21.1 / 15.0 ms at nd=1/2/4 (eff 0.69/0.49); +fused-halo: identical (null). explicit_substep: 31.1/25.0/18.4 (0.62/0.42). explicit+wide-halo: 33.4/19.1/11.5 (0.88/0.73) <- best; NOTE nd=1 wide is SLOWER than implicit nd=1 (33.4 vs 29.3).
LL192x384: icn 7.3/11.3/10.2 (0.32/0.18) anti-scales; wide 7.5/7.3/6.1 (0.51/0.31) only monotone arm.
Static census implicit arm: 123 reductions/step (60 PCG iters x 2 batches + extras), halo_messages_per_step 122 @nd=2 (analytic, barotropic scope; baroclinic 3-D pads uncounted).
Queued: single_reduce PCG variant (halves reduction count), NCCL_PROTO=LL128/LL arms (arXiv:2607.16100), wide+fused interaction.

## Ocean CPU-MPI (np=1 leg stock-CG mismatched — slope np>=2 valid)
LL192 strong: 841(np1, stock)/468/378/254/215/110 ms at 1/2/4/8/16/32. np16->np32 superlinear jump.
Weak (64 rows/rank): eff 0.96@2 -> 0.58@4 -> 0.34@8 -> 0.14@16 -> 0.125@32.

## Atm cube GPU SPMD single process (gray_sbm physics, L26, f32)
C192: eff 0.88@2 / 0.84@3 (f64: 1.07/0.91). C96: 0.47/0.36. C48: 0.26/0.19.
Census (C96/L40 dycore-only f64): collective-permutes/step 0/7/14 at nd=1/2/3, 1 all-reduce.
Cube weak (tiny ~3.5k cols/GPU tiles): eff ~0.46@2 — deep under the ~30k cols/GPU SOTA floor (protocol issue, not code).

## Atm latlon + ico CPU-MPI single node (f64, moist, L26)
latlon strong r64/128/256: eff ~0.87-0.92@2 decaying to 0.11-0.23@32-64; np16 anomaly (slower than np8, recovers np32) — Milan CCX/NUMA effect suspected.
weak (~213k cells/rank): 0.88@2 -> 0.083@64.
Single-node ladders conflate DRAM-bandwidth contention with comm (4-node spread ladder running).

## Known priors (docs)
- f64≡f32 GPU curves ⇒ latency-bound; message COUNT is the lever.
- Derecho 2026-07: latlon 28km eff 0.47@16 A100 route-A; ico 0.38@16; CPU ico 0.59@128.
- route-B (jax.distributed+NCCL ppermute, XLA-overlappable) replaced route-A's no-overlap ceiling; np8/np16 lanes queued on Levante.
- Ocean wet-cell compaction (~2x on 40%-land grids) unimplemented; tripole SPMD fold shipped but unscaled; cube >6 GPU tiled lane queued (24 GPU).
- XLA latency-hiding scheduler ON; PGLE known +8.5% on Derecho lane-T; xla CP-combine+pipelined-p2p arm measured -10% combined (needs splitting).
