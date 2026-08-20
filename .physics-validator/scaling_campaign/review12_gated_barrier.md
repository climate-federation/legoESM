Verdict: **FIX-FIRST** — make the gate identify the measured workload, not just global edge rows.

- Static-safety is fine: `n_dev` is a factory-closure `int`, and `.shape[0]` is an aval-static dimension under `jit`, `lax.scan`, and `filter_value_and_grad`; no data-dependent retrace.
- The gate is over-broad: L8 has 1,966,080 **unpadded** edges; it is divisible by both 4 and 16, while L8 cells pad from 655,362 to 655,364 (np4) or 655,376 (np16).
- Thus a mesh prepared for 16 but run on 4, or another Voronoi mesh with that edge count, fires the barrier despite an unmeasured workload.
- Per-shard rows (`shape[0] // n_dev`) is not safer—it is merely a derived, equally collision-prone key.
- Key an explicitly named workload signature, e.g. `(n_dev, s.u.data.shape[0], s.T.data.shape[0], nlev)` (or captured padded `(nCells, nEdges, nVertices, nlev)`), and rename the constant accordingly.
- The comment is inaccurate: it calls the value a per-device owned+halo shape, then correctly uses global owned rows; “padded L8 edges” is false. Its -17.6%/+10.7% claims also conflict with the reported -20.7%/+1.2% noise.
- Reword the fusion explanation as observed correlation, not a proven XLA cost-model cause.