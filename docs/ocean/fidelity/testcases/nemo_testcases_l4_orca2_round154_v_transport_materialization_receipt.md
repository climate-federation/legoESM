# ORCA2 round 154 — V transport source materialization

Date: 2026-10-05. Base `459ef0fbb`; preregistration `cf36c0076`;
measurement tip `b54ec041a`. Verdict: **HELD**.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. No configuration,
forcing, initial state, carried-state form, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Result

The private/default-off production-JIT arm confirms round 153's source-boundary
hypothesis. NEMO completes and stores the V metric transport as its own loop at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:568-570`, then reads
that array in a separate loop for the north-minus-south difference, complete
divergence, and sea-surface update at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:584-591`.

With the three previously proved prerequisites held fixed—NEMO's raw reference
face depth, omission of legoESM's extra compact V mask, and the certified
seven-array external-mode boundary association—materializing the completed
`zhV` expression closes every remaining substep-2 boundary:

| boundary | round-153 control unequal / maximum | materialized unequal / maximum |
|---|---:|---:|
| completed V metric transport | 0 / 0 | 0 / 0 |
| north-minus-south V difference | 8,786 / `2.3283064365386963e-10` | 0 / 0 |
| complete divergence | 6,761 / `1.3552527156068805e-20` | 0 / 0 |
| sea surface | 6,499 / `3.469446951953614e-18 m` | 0 / 0 |

The ordinary T/S/u/v/eta/uu_b/vv_b state is bit-identical with the private arm
off. The final causal artifact is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round154/materialization_arm.json`,
SHA-256 `9c8a4fd6f7fb547b9b26175d1ba9952e3126dc6c169bc9712fdd5c8b07286811`.
Its planted control exits nonzero with
`STATUS PLANT-FIRED transport-v-materialization`.

This exact substep closure does **not** move either stage-checkpoint ladder.
The complete four-prerequisite candidate changes 0/200 rung-0 rows and 0/200
rung-7 rows; no exact row leaves the bar and neither first-debt boundary moves.
Rung 0 remains kt=1 stage-1 T at the held round-94 statement; rung 7 remains
kt=1 stage-1 T, unattributed. The kt=10 stage-3 salinity maxima are unchanged:
`0.4156673855360964` for rung 0 and `0.28803500879531185` for rung 7. The
comparison artifact is `ladder_compare_complete.json`, SHA-256
`1f99c8c5057e68d267b83a03aea9bb24aaad6d7f4ed530091ed666c58786a625`.

The statement is therefore named and mechanically proved, but not put on the
production path this round. The retained implementation is a private,
default-false measurement hook only. A production landing must transcribe the
whole proved chain atomically and pass the complete shared landing gate.

## Retractions

Two earlier ladder comparisons from this round are **RETRACTED** as evidence
for propagation and retained only for provenance:

1. `ladder_compare.json` omitted the raw-reference-depth prerequisite.
2. `ladder_compare_final.json` added raw depth but omitted the certified
   seven-array external-mode boundary association.

Both happened to report 0/400 moved rows, but neither isolated the registered
statement chain. `ladder_compare_complete.json`, which explicitly enables all
four prerequisites in both ladders, is the only cited propagation result.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R154-P1 default-off materialization hook is passive | **CONFIRMED**: all seven ordinary/exposed state arrays are bit-identical; GYRE is also unchanged below. |
| R154-P2 materialization closes transport, V difference, divergence, and SSH | **CONFIRMED**: every named substep-2 boundary is 0 unequal cells. |
| R154-P3 both ORCA2 ladders retain exact rows and first debt | **CONFIRMED**: 0/400 rows move, no exact loss, first debt unchanged. |
| R154-P4 kt=10 stage-3 salinity does not worsen | **CONFIRMED**: both rung maxima are exactly unchanged. |
| R154-P5 shared executing cards pass if production changes | **NOT_APPLICABLE_NO_PRODUCTION_CHANGE**. |

## Shared path and review

Although only private/default-false hooks were added under `packages/`, the
required GYRE default-path checks are exact. The 70-row ten-step comparison has
zero moved cells, zero status changes, unchanged first debt kt=3, and both
residual archives have SHA-256
`7f34d4d8f42e5a23b2e4c00dcd7d35e0a778a284ed1f54306fb457618dde7af3`.
All 30 daily snapshot files are byte-identical; their normalized content
manifest is `20e05cb57c3e29d38cae784551c0b0ceb6170215e14f83f381ec118a2d3cb34e`.

The separate `codex exec --sandbox read-only` review returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`. No independent verdict is claimed.

No DINO, tank, generic-card, or production-landing gate was run because the
production path did not change and the round is HELD before landing. ASKED
choices: none. UNASKED choices: empty.

## OPEN

1. Preregister and transcribe the complete proved V-transport chain on the
   production NEMO-literal path: raw face depth, source V-mask semantics,
   certified boundary association, and materialization of completed `zhV`.
2. Re-run both ORCA2 ladders and the salinity veto; then run the full GYRE
   Decision 43/45/55/59 year gate, DINO, tanks, generic cards, citations, and
   push gate before landing.
3. If the production transcription passes, re-run the rung-0 independent month
   to its first non-finite boundary and continue the hierarchy walk.

## Verification

Verification was run CPU-only in fp64/x64. The canonical citation map was
mechanically re-anchored with `difflib.SequenceMatcher`; its real-receipt run
has zero unmapped citations, zero failures, and zero failing map entries.
This receipt's two compiled spans are live and a two-line shift makes each
fail. Focused tests and the required full fidelity battery are recorded in the
round evidence and final campaign report.
