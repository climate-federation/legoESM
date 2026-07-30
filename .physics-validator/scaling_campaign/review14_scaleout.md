1. A alone maps the plateau; it does not identify it. Use matched contrasts: at fixed N compare C768/96 (36.9k cols/GPU) vs C1536/96 (147k) for tile-floor sensitivity; then compare C768/24 vs C1536/96 (both 147k) for N/comm sensitivity. Apply the same 4× resolution / 4× GPU pairing to LL and ico.

2. D can discriminate, but only if you report time *per PCG iteration* with fixed solver settings/iteration count. A normal weak-scaling ladder changes global problem conditioning and PCG iteration count, so it otherwise confounds “communication growth” with solver work.

3. C alone is not evidence: resolution changes dt, stencil/cache behavior, solver conditioning, and possibly PCG iterations. Use it only as half of the matched-pair design above.

4. Instrument every run: compute, halo exchange, MPI wait, and each PCG reduction separately; record PCG iterations, max/median local cells, halo length, imbalance, and peak device memory. The 95.7 µs/iter reduction path should be visible directly.

5. Hold precision, tolerances, preconditioner, rank/GPU binding, clocks, and placement fixed. For resolution comparisons, report both ms/step and cost per simulated second at matched CFL; do not treat either as a substitute for the other.

6. Check mesh partition padding explicitly—especially ico/LL at 64+ GPUs. Zero/edge tiles, different tile aspect ratios, and uneven halo ownership can mimic a tile floor.

7. Separate node count from GPUs/node: at one N (e.g. 64 GPUs), compare 16×4-GPU nodes against 64×1-GPU nodes if topology permits. Otherwise the “scale-out” result includes intra-node versus network transport.

8. Preflight f64 memory at the largest local tile; prohibit unified-memory spill. Also verify L10/C1536 does not alter algorithm selection or batching.

9. Cheapest diagnostic: two paired application runs above plus an isolated fixed-iteration PCG/all-reduce timing sweep over N. That is far cheaper and cleaner than a full D ladder.

10. Information per allocation-hour: **(1) C paired with existing/new A points**, **(2) abbreviated D with fixed-iteration normalization**, **(3) A endpoints**, **(4) B** for this plateau question. B is worthwhile CPU validation, but it is already a placement-corrected result, not evidence about GPU/ocean mechanisms.

11. Cut: the full weak-scaling ladders; ocean and MPAS 64-GPU production runs until the paired 96/64 fixed-N tests show a residual N effect; and CPU 512 ranks unless 128→256 remains clean under identical ranks-per-node/NUMA binding.