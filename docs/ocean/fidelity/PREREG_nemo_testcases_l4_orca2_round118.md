# Preregistration — ORCA2 round 118 V EEN recurrence acquisition

Date: 2026-10-03. Base: `bbf91aa1429f5a7796e5bf4d4d73ca5a929a5093`.
Scope is additions-only instrumentation on hierarchy rung 0. Every eventual
ocean number from this record is **independent** because rung 0 starts from
NEMO's own from-rest state. No model physics, card field, configuration value,
carried state, threshold, stabilizer, sea-ice selector, or
`unmeasured_features` entry may change.

## Admitted boundary

Rounds 115--117 completed the four U-grid EEN recurrences. Northwest and
northeast first differ only in exact-zero accumulator signs. Southwest and
southeast first differ in 68 southern-row neighboring V-mask magnitudes and
then in exact-zero accumulator signs; NEMO's constant-zero southern V-mask
association plus ordinary IEEE zero addition closes every recorded U field.
No production statement lands from U-only coverage.

The compiled rung-0 program next evaluates northwest, northeast, southwest,
and southeast V recurrences in that order. Each adds live V thickness times a
neighboring live U thickness, that U mask, and the source-ordered `zpvo`
coefficient to a carried accumulator
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1309-1328`).
No admitted record surrounds these four statements.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R118-P1 | Adding one write-only, rank-complete V recorder is observationally passive. | Every kt=1..10 restart and every inherited round-105, round-107, round-110, and round-116 stream is byte-identical to the admitted round-116 run. | Any moved byte rejects the instrument; reduce it and reacquire under a new target before quoting an operand. |
| R118-P2 | Each stored product is exactly replayable from its recorded source operands in the compiled written order. | Recorded V thickness, neighboring U thickness, neighboring U mask, and `zpvo` reproduce all four stored terms bitwise on every executed level. | The first failed product rejects the instrument as insufficient or perturbing; do not interpret recurrence signs. |
| R118-P3 | Each recorded recurrence is internally exact under ordinary host IEEE addition. | `acc_after == acc_before + term` bitwise for NW, NE, SW, and SE on every executed level. | A failed recurrence means separately stored operands do not describe the executed binary statement; stop and add a source-side discriminator. |
| R118-P4 | After admission, northwest V is the next source boundary; if its first difference is signed-zero-only recurrence state, the registered IEEE-zero arm closes it without moving any nonzero value. | Every earlier NW operand and product is exact; the first recurrence difference is zero-sign-only; the arm closes it with no nonzero movement. | The first earlier unequal NW row owns the walk. Any magnitude movement refutes the arm and forbids landing. |
| R118-P5 | No production statement lands before all four V streams admit and are walked in compiled order. | This round ends `STOPPED_FOR_RECORD` if the operator record is absent. | A production edit or V attribution from preflight alone is a process failure and is retracted. |

## Acquisition and admission bar

The self-describing stream records, for every V path, executed level, and both
ranks: `zpvo`, live V thickness, neighboring live U thickness, neighboring U
mask, the separately stored product, and the accumulator immediately before
and after the source recurrence. It also records `mbkv`. Arrays are
zero-initialized and writes outside `mbkv` are a hard refusal.

The checker parses names, dimensions, and payload lengths from the header. It
requires exact two-rank owned-domain coverage; rejects the dummy level and all
out-of-loop writes; replays products and recurrences bitwise; compares each
terminal recurrence against the admitted round-105 V accumulator; and requires
all twenty kt=1..10 restart shards plus every inherited stream to be
byte-identical. Header, field-name, dimensions, truncation, missing-field,
duplicate-rank, bottom-index, product, recurrence, inherited-terminal,
association, and restart-byte plants must fire.

The launcher uses a new target and run directory, an absolute pre-created
per-rank path, and SHA-256 content pins rather than a moving commit. If the
record is absent, this round reports the committed launcher as
`ACQUISITION_NEEDED`; the sandbox does not run MPI.
