Verdict: batch fields first; a deep halo is not a safe drop-in.

1. `88/4 ≠ 22` exchanges here: four rounds is face-only. The kt=2 tile pad emits 16 phases/logical scalar pad (4 edges + 4 guards + 4 diagonals + 4 corner slivers) for serial-exact offset/corner handling. [_build_tiled_pad](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/cubesphere_exchange.py:1276)  
   The RK body separately pads `dp,B,zeta,invT,lnps,T`, and the vector pad calls the scalar pad twice, ×3 RK stages. [_tile_tendency](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/tiled_production_cdgrid.py:1647)

2. Vertical batching is already done: these are 4-D pads with all L60 levels in the trailing payload. It cannot remove launches. Fields are the real structural duplication; the serial SPMD dycore already packs the analogous stage fields. [serial pack](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_cdgrid.py:404)

3. Be adversarial: GPU XLA already combines much of this (384 source permutes becomes ~101 optimized HLO ops; trace sees 88 SendRecv). Thus field packing is numerically safe, but its incremental comm win may be modest—not a justified “4×” claim. Its absolute ceiling is 6.65 ms, not the 11.27 ms fixed term.

4. A once-per-step stale halo changes RK2/RK3 boundary tendencies: reject it. Exact communication avoidance requires redundant RK evolution in an overlap of at least `3 × radius = 6` cells for a radius-2 stencil, including cube-edge interpolation/corners; current tiled transport supports only halo 1/2. That is a new exact overlap algorithm, not the ocean trick.

5. Cheapest bound: first inspect the exact optimized HLO plus per-SendRecv payload bytes; tracer-only is useless because job 26510470 is dynamics-only. Then gate one stage pack `{dp,B,zeta,invT,lnps,T}` (and separately pack geographic `u/v`) and rerun parity + the same marker trace.

Ranking by expected saved-ms / answer-risk:  
1. Stage-field + `u/v` packing — best, zero algorithmic risk; bound `[0, 6.65]` ms.  
2. Exact deep-overlap RK — potentially larger, but high implementation/correctness risk.  
3. Vertical packing — ~0 ms; already present.  
4. Stale deep halo — unacceptable: changes answers.