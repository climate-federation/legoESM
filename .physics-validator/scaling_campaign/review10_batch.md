Adversarial review (no files changed):

1. **np4 mega-fusion — NEEDS-QUALIFIER.** `12 × 3.6 ms × 3 / 12 = 10.8 ms/step`, consistent with “~10.9” only if the unsuffixed fusion’s 12 long launches are the third once-per-step group. Naming only `_1/2` implies 7.2 ms/step. Counts alone do not prove cadence; timestamp grouping does. The frame supports provenance at `pytree_axpy` line 10, not the recomputation mechanism.

2. **Output barrier — NEEDS-QUALIFIER.** 17.09→14.09 is **17.55% lower time** (1.213× speedup / 21.3% higher throughput). np4 repeats give 17.09±0.05 ms half-range (±0.29%); the output arm is n=1. Treating “1% agreement” as full range yields only an informal ≈±0.8 percentage-point A/B error, not a CI.

3. **Milan — NEEDS-QUALIFIER.** The ratio is correct: 186.83/87.59=2.133, and the script fixes resolution, physics, precision, levels, warmup, and timing; only `--distribution` varies. But it needs binding receipts to prove which NUMA domains ran, plus repeats/counters to call the cause DRAM saturation. “np32 spans both sockets” is unsupported: 32 one-core ranks fit within one 64-core 7763 socket.

4. **Pencils — REFUTED as a decomposition-only A/B.** The script holds resolution/dt/physics fixed, and 253.93/185.03=1.372. But `--latlon-2d` uses the separate wall-pole 2-D path, while the band path retains the atmospheric pole-fold behavior; it changes boundary semantics as well as decomposition. Report it as a regular/wall-pole throughput result.

5. **Compaction — REFUTED as a bound.** The endpoint arithmetic is right: `1/1.16=0.86` and `1/0.71=1.41`. It bounds a real step only under unshown additive-cost and representative-stencil assumptions. The microbench excludes packing/scattering, real wet topology, model stencils, and communication, so “lands between” is an end-member estimate, not a valid bound.