# ORCA2 round 217 — complete vector unit held at rung-0 live thickness

Date: 2026-10-10. Frozen base: `ac1e05f26`. Preregistration commit:
`0ad7a15fd`. Candidate commit: `9c9302a64`. Restoration commit:
`0b1918b81`. Gate commit: `494ff8912`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round217/`.

The final tree changes no model file, configuration, carried state,
stabiliser, sea-ice selector, or `unmeasured_features` entry from the frozen
base. `git diff ac1e05f26..494ff8912 -- packages/` is empty. Every OMT-1
result is separately labelled **independent OMT-1** or **given NEMO's entry
OMT-1**. The rung-0 result is **independent**. Rung 10 remains **given NEMO's
entry** and is unmeasured for the candidate.

## Candidate source unit

Round 217 promoted one indivisible vector-form source unit for measurement:
the round-215 slow-V/raw-`ssvmask` pair, raw reference face depths, the complete
seven-array association, and the unmasked/materialised V transport. The
executed vector update is compiled at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:669-682`; the one
association follows at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:747-756`; and the
next substep's V transport and continuity consumer are
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:533-560`. These are
the same compiled spans admitted in rounds 215-216. No constituent was scored
as a landing by itself.

The candidate extended the already-landed literal external-mode predicate to
the vector-form RK3 branch and used NEMO's raw compact V mask at both the slow
forcing and vector update. Candidate-focused tests instantiated both the
vector and flux predicates, retained the raw northern-fold mask row, and
passed 14/14 before trajectory measurement. The line-map citation reanchor
passed the cumulative default receipt with zero unmapped citations at the
candidate commit.

## OMT-1 result — both labels qualify

Both eight-step OMT-1 ladders complete: 32 checkpoints and 160 rows per
label. Their numerical results are identical. Of 160 rows, 155 move; every
RMS-moved row moves toward NEMO, none moves away, no exact row leaves the bar,
and the first debt moves toward.

The kt=1 stage-1 SSH RMS/maximum improves from
`0.006243172975388979 / 0.13141649093684893 m` to
`0.00023211473728011986 / 0.006284710854093087 m`. The kt=8 stage-3 SSH
maximum improves from `0.7583316946619745` to
`0.04829937251182753 m`. The independent and given-entry artifacts have
SHA-256 `c6a27efe0e4f772fe04d598c9eaaf434fb5714dcbaac22fc039711180f72ae48`
and `37fc8a3ec18c6d57d2cf6c28d7808c49a7efcc6eac5cd11f30eb1479791896fe`.
Their Decision-96 reports have SHA-256
`5c816557dfedf3784eea7ab8573d09b4ab786fcf47b798f5c75e7e1051d71a2e`
and `0737053772a026067c5a1c828c2f5d8903ecc577213a24b623a6e1031d8a026c`.

## Binding independent-rung-0 refusal

The same complete candidate does not complete the ten-step rung-0 ladder.
It completes the separately exposed kt=8 stages 1-2, then the full stage-3
step refuses before publishing `PROGRESS kt=8 completed stage 3`:

`raw-mesh e3w_int must contain only finite values > 0`.

No kt=9 boundary is reached. The retained log has SHA-256
`f65124c81a7eb2b48a9290380911d94bdc85808c3e22b3596a044f16a1f4088a`.
This reproduces round 164's complete-unit failure class even after adding the
round-215 vector pair; therefore the pair is not the missing compensating
partner for rung 0.

The first observed downstream boundary is the live W-thickness construction
before completed kt=8 stage 3. NEMO consumes density and live W thickness at
the HPG boundary in compiled
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stprk3_stg.f90:383-402` and remains
finite on the admitted record. The exact first non-finite operand inside the
candidate's live `e3w` construction is **UNMEASURED_WITH_SPEC**; no source
statement is attributed from a runtime guard alone.

The rung-0 refusal is the first red landing predicate. Rung 10, GYRE, DINO and
tank candidate gates were therefore not run and are explicitly **UNMEASURED
for the candidate**, not PASS. Candidate production was restored immediately;
the final package tree is byte-identical to the frozen base.

## Preregistered predictions

| ID | disposition |
|---|---|
| R217-P1 | **CONFIRMED for the measured candidate**: the complete vector predicate instantiated, the candidate-focused tests passed, and no partial statement was offered as a landing. |
| R217-P2 | **CONFIRMED**: both OMT-1 labels have 155 toward / 0 away RMS-moved rows, no exact loss, first debt toward, and final SSH maximum toward. |
| R217-P3 | **REFUTED**: independent rung 0 refuses at kt=8 before completed stage 3 on non-positive/non-finite live W thickness. |
| R217-P4 | **UNMEASURED_PREREQUISITE_R217-P3**: rung 10 and shared-card candidate gates were not called passes. |
| R217-P5 | **CONFIRMED within sandbox limits**: all three round-217 plants fire; focused tests pass; citation gates and rigid-shift plant pass; independent review is unavailable in-sandbox. |

The committed classifier finishes `STATUS HELD_R217_RUNG0_LIVE_THICKNESS`.
Its report SHA-256 is
`2fa05b109933fb17bd9370377f94a73e88416b047cf53937fefb26c3fd430281`.

## Validation and review

The final-tree focused battery passes 16/16: the round-217 classifier and all
three plants plus the literal barotropic unit tests. The single prescribed
`tests/ocean/fidelity -n 12` battery collected 3,049 tests and reached 99%;
after no pytest process remained, the retained controller was bounded. It is
not called PASS: 3,026 passed, seven skipped, four registered pre-existing
failures emitted verdicts, and 12 remained unclassified. The four failures
are the GYRE round-129 spread-floor record stamp, allow-dirty scope,
worktree-stamp ratchet, and SI3 scalar-math provenance gate. Log SHA-256:
`57fbd8e70d95f5d5349012a584894b2005f10df8574fdf6fcace2e18a4869c35`.

Independent review was attempted with `codex exec --sandbox read-only` and
exited 1 before reading the diff: `failed to initialize in-process app-server
client: Read-only file system (os error 30)`. Independent review is unavailable
in-sandbox; this is not a PASS. Log SHA-256:
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

1. Under the same complete private unit, record the kt=8 live W thickness and
   its SSH/`r3t` operands passively. The passivity predicate is the completed
   kt=1..7 states plus kt=8 exposed stages 1-2, all array-identical to this
   round before any operand is read.
2. Name the first non-finite operand in that construction, then walk its
   producer in source order. Do not land or score any constituent of the
   vector unit separately.
3. Re-score the indivisible unit only after that downstream partner is named.
   OMT-2 waits. No acquisition or configuration decision is requested.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
