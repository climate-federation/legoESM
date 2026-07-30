## 1. Cube: attack upstream skew; do not remove the reduction

A scalar all-reduce is a synchronization point, and a ~15 ms duration is compatible with early ranks spinning until the late rank arrives—not payload transfer. But it is not yet proven.

Two important caveats:

- The profile is single-shot while the fit is closed-loop, so its 15.6 ms cannot be charged to the closed-loop 11.27 ms term.
- The claimed `(1,)` psum source is not the single-shot step: that reshape is in the closed-loop communicator warm-up, while the real mass fix reduces a scalar directly. The single-shot adapter refuses the mass fixer. See [tiled_production_cdgrid.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/tiled_production_cdgrid.py:2508) and [tiled_step_adapter.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:102).

So: **do not remove, relax, or try to overlap the reduction yet.** Removing it risks changing conservation and, if it is absorbing skew, mostly moves the wait to the next dependency.

The highest-value intervention is to **find and remove the late-rank tail before that collective**—likely among the ~120 halo launches. A real implementation will probably be “fewer/fatter halo exchanges” or interior/halo overlap, but neither is honestly a ≤25-line fix without identifying the offending phase.

For the closed-loop trace, calculate per collective instance:

`arrival skew = latest NCCL-kernel start − earliest NCCL-kernel start`

If that is millisecond-scale while the last-arriving rank’s service time is microsecond-scale, the skew reading is confirmed. If starts are aligned, the all-reduce itself/scheduling is implicated instead.

Also, the **1.72× Amdahl cap is conditional**, not universal: it comes from a two-point fit and assumes the 11.27 ms term stays constant as tiles shrink and device count rises. Perimeter-like halo cost would decrease with tile size; collective latency could increase with rank count. Your 2.25× fixed-tile contrast supports only “little growth over that tested range.”

## 2. MPAS-ocean: it is a richer communication schedule, not “Voronoi”

The atmosphere and ocean lanes share a Voronoi mesh, but not a parallel algorithm. The ocean path has stage-frontier halo refreshes over more state fields, barotropic work, and conservation reductions. If this ladder used `implicit_cn` with the default PCG, it additionally has **121 serial global reduction batches/step** at 60 iterations; that is a direct device-count term. The atmosphere path is much lighter.

Your equal-5120-cells/rank contrast establishes that **cells/rank alone is insufficient** for this ocean lane. It does *not* yet distinguish global-rank cost from high-rank partition quality: average owned cells do not control halo count, neighbor count, edge cut, or worst-rank work.

Run one 2×2 experiment, keeping rank placement, dtype, solver, halo-refresh mode, and timing protocol fixed:

| | subdiv-7 / 32 | subdiv-8 / 128 |
|---|---:|---:|
| METIS low-cut partition | A | B |
| SFC/RCB higher-cut partition | C | D |

Record total time plus max owned/halo ratio, max neighbors, max halo receives, and total edge cut.

- Large C−A and D−B at fixed rank count ⇒ partition quality is material.
- A persistent B−A with matched low-cut metrics, especially concentrated in global-reduction time, ⇒ genuine device-count cost.

The best ≤25-line production fix is conditional:

- **If implicit-CN/standard PCG:** set `barotropic_implicit_pcg_variant="single_reduce"`; it halves the sequential reduction count. Validate residual and conservation.
- **If explicit-substep:** do not guess a solver fix. Pin `--partition-method metis` only if the above A/B shows a meaningful cut/time win; it is already supported.

## 3. Rising s7 speedup

It is a **base-point investigation flag**, not proof that np32 is bad. The successive speedups are 1.29, 1.43, and 1.57—still below ideal 2×. Cache/working-set effects, rank imbalance, partition shape, or CPU affinity can make relative efficiency improve as tiles shrink.

Your fixed 32 ranks/node removes the previous changing-bandwidth confound, but not those effects. Repeat each rung and retain per-rank max/median timing plus partition metrics. Do not claim either “np32 is anomalous” or “high ranks get physically better” until that is done.