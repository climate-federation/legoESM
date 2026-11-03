# F4 detail votes (me / codex / GLM, majority) — 2026-11-03

| id | question | me | codex | GLM | result |
|---|---|---|---|---|---|
| V1 | perturb constants/params written to the oracle input (ZM 10-28 ruling) vs hold | a | a | a | a: perturb every passed float |
| V2 | several oracle calls per run: same seed each call vs per-call seed | a | b | b | b: seed + 20*call_index (call 0 = seed) |
| V3 | score only port-vs-oracle comparisons (no jit/gap/grad/budget tests) | a | a | a | a |
| V4 | seed runs: spy records true value, returns 0 so TOL asserts don't abort | a | a | a | a |
| V5 | harness in oracle_audit/f4 + committed copy with a direct rewriter test | a | a | a | a |
| V6 | oracle array changes shape/disappears under a seed -> UNMEASURED (not C) | a | a | a | a |
| V7 | discrete flips (zero pattern / ints) under 1 ulp: (a) family-wide no-certify, (b) per comparison | a | a | b | a: family-wide |
| V8 | MG2 prog-N_c C1 fails (fixture built Spack GCC 13.4 elsewhere): (a) UNMEASURED, (b) regenerate fixtures here | a | a | a | a |

Codex findings r1 (8) on the harness: all fixed (array-shaped neutral return, `_rel_flux` metric, argv[1]-only input file,
C3 requires non-empty equal sources, Louis `ri_pin` input pins excluded, C5 requires logged gfortran command flags via
`bash -x`, discrete-change detection for codex #6). #5 (readback is a separate parser, not the oracle's own read) kept as a
stated limitation: C1 bitwise reproduction + every float moved one step are the evidence the oracle consumed the text.
GLM review r1: proxy-vs-internal-round-off concern is the approved method (not changed); pairing divergence now reported.
- V2 implementation: a call whose float input values equal an earlier call's reuses that call's seed index (generator asserts tying two calls on the same values -- MYNN orig vs dump build, Kuo klev_max 20 vs 1 -- keep their tie).
GLM review r2: 9 points. Taken: unsafe-math denylist widened in C5 (#9). Not taken, checked: #1-3 the spy wraps the TEST's
metric, not the oracle/generator (generator self-asserts run unwrapped and stayed live); #4 every driver in these families
reads argv[1] and writes argv[2] (read in each make_fixture), and C1 + all-floats-moved confirm; #5 seed+20k cannot collide
for seeds 0..19 and default_rng(int) streams do not overlap; #7 rel <= 1e-12 = A is the user's rule (perturb.classify).
Codex r2 (2 findings: integer-valued float/bool discrete detection; C5 flags not from the executed binary) fixed; codex r3
(MG2 C5) fixed by tracing the generator's own build.sh call; codex r4: NO FINDINGS. GLM r3 on the last two changes: 6 points,
none changes a class -- every case it names errs toward UNMEASURED/C5-FAIL (conservative), the traced build REPLACES the
generator's call (not run twice) and each mode's log is a fresh directory.
