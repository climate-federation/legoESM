# ORCA2 round 251 — direct slow-V chain retains the 35-face debt

Date: 2026-10-12. Frozen base: `98e6c2b39`. Preregistration commit:
`652ea7e83`. Final measurement-tool commit: `6bf6e3140`. Status: **HELD**.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round251/`.

This measurement-only round changes no model, deck, selector, carried state,
stabiliser, sea-ice field, or `unmeasured_features` entry. Results are labelled
separately **independent** and **given NEMO's entry**. The labels agree exactly.
No acquisition was needed.

## Compiled statement and method

The compiled R230 program forms the depth average, calls the baroclinic drag
correction, then adds wind at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:194-230`. The drag
coefficient, residual, and unmasked addition are written at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1404-1463`.

The round-251 gate runs the production OMT-4 executable twice per label: once
normally and once with the already-existing passive producer trace. It refuses
unless all completed-state arrays are bit-identical. It also refuses if the
incoming, final, or drag-output slow-forcing override is present. The admitted
NEMO twins remain `EXACT_DEFINED_BYTES`.

This directly repairs round 250's instrument defect: no replay is fed back into
the executable. The candidate and oracle final Coriolis-removal statements each
self-replay bit-exactly from their own live operands.

## Source-ordered direct-chain result

Both labels give this table on the registered 35 northern-fold faces.

| output/operand | unequal | maximum absolute | verdict |
|---|---:|---:|---|
| post-drag V forcing | 35/35 | 1.6557659420864476e-6 m s^-2 | **first non-bit output** |
| wind increment | 0/35 | 0 | exact |
| post-wind / incoming copy | 35/35 | 1.6557659420864476e-6 m s^-2 | carried from post-drag |
| 2-D Coriolis operand | 0/35 | 0 | exact |
| V surface mask | 35/35 | 1 | non-bit, downstream operand |
| final slow V forcing | 35/35 | 1.6557659420864476e-6 m s^-2 | carried |

The support is fold-only: zero unequal cells occur outside the final three
rows. The direct run therefore reproduces the same magnitude that round 250
had incorrectly dismissed as belonging only to its override. Round 250's drag
*owner* remains retracted: its literal correction replay was exact, and this
round does not yet split the executing post-drag boundary into the incoming
depth mean, correction, and association. The narrower new claim is only that
the first source-matched direct-chain output is non-bit at `post_drag_v`.

An attempted diagnostic row compared legoESM's pre-drag value with NEMO's
depth mean. It was invalid because the programs order zero wind and drag
differently. Commit `6bf6e3140` removes that row from the gate, so the rejected
claim cannot be printed.

## Frozen predictions

1. **R251-P1 — CONFIRMED.** Both twins admit; all passive-state fields are
   array-identical; no slow-forcing override is active; both self-replays are
   bit-exact.
2. **R251-P2 — REFUTED.** The direct chain is not exact. Its first registered
   output is `post_drag_v`, unequal on 35/35 faces with maximum
   1.6557659420864476e-6 m s^-2 under both labels.
3. **R251-P3 — NOT REACHED.** The association/unit experiment was conditional
   on P2 confirming. It was not run.
4. **R251-P4 — CONFIRMED.** No package or configuration change lands while
   Decision 116 remains pending.
5. **R251-P5 — CONFIRMED.** All eight classifier plants fire.

Evidence hashes: `direct_independent.json`
`9be032f5bcddc9629a5327179fe28c1dce863c7389de9965affcbbc6b7ca4dae`;
`direct_given.json`
`c10e77697df51064fe5b2cad805ebdd1d58c672c59602951fa64e77816ec54fd`;
`classification.json`
`57ae3c84d72ce0efd1e12c615dc148956f93a39ffc11737ac6c6a6ddbff168cc`;
`plants.log`
`f1e365fa10e6744cbb6475971963607ad09676503fd1a5acf13be2046d585028`.

## Controls, tests, citations, and review

The focused round-251 tests pass 5/5. All eight classification plants exit 2
with `PLANT-FIRED`. The one mandatory `tests/ocean/fidelity -n 12` invocation
reached 99% with no round-251 red and then stopped reporting with one worker
outstanding. It was interrupted rather than re-run, the same registered
late-worker hang class as rounds 249-250; it printed no failing node ID to
isolate.

A separate `codex exec --sandbox read-only` review was attempted. Its
in-process app-server could not initialise on the read-only filesystem before
reading the diff. Verdict: **independent review unavailable in-sandbox**.

The receipt citation gate passes with two citations, no failures, and no
unmapped citations. The cumulative gate passes with 274 citations, no
failures, and no unmapped citations. Shifting the new `stp2d` citation by two
lines makes the gate fail, proving the control fires. No model file changed,
so the GYRE, DINO, tank, OMT, rung-0, rung-10, and month trajectories are
unchanged by construction rather than re-scored.

## OPEN

Round 252 must split the *direct*, override-free post-drag boundary. Starting
from the already-certified literal drag increment, compare the executing
pre-drag depth mean, the drag addition, and every compact/raw surface-mask
association on the 35 fold faces. Only after that boundary is exact may the
two held external-mode associations be re-scored with the private unit.
Decision 116 remains pending and is not re-emitted.
