Verdict: good direction, but make the “opt-in build” single-writer safe.

- Treat a valid, version/key-matching cache hit as always admissible; for subdiv 9–10, reject a miss unless explicitly prewarming.
- `LEGOESM_ALLOW_BIG_MESH_BUILD=1` alone has a thundering-herd race: 1,024 ranks can all see a miss and all are authorized.
- Add a per-cache-key exclusive lock; one rank builds, others wait/recheck. Use a unique temp file/dir on the same `/work` filesystem and atomic `os.replace`; only expose a final manifest last. Include stale-lock recovery.
- I cannot certify that current `prewarm_voronoi_cache` already does tmp+rename without reading it; make this an explicit acceptance test, not an assumption.
- Fewer Lloyd iterations are legitimate as a distinct, clearly labelled scaling mesh; do not present them as physics receipts.
- First compare area CV/min–max, neighbor-degree distribution, partition cell imbalance, and halo/edge counts against the 50-iteration mesh. Those directly bound scaling contamination.
- `lloyd=0` is the cheapest strong candidate: an icosahedral-bisection dual can be a valid *synthetic scaling* mesh if those metrics are comparable; retain 50 for physics.
- Keep `>10` hard-refused, independent of cache/opt-in.
- Tests should cover hit bypass, miss rejection, opt-in prewarm, one-builder concurrency, atomic incomplete-cache rejection, and `>10`.
- A direct prewarm entry point under `scripts/data/` is sensible, but I can’t verify the exact `CLAUDE.md` dispatch-hardening/script requirement or existing MPAS assets: this session’s read-only shell is failing to return commands.