# Preregistration — ORCA2 round 116 remaining U EEN recurrences

Date: 2026-10-03. Base: `f0975a020b1b8b010c9a8aab7d4e91c27dd39f5c`.
Scope is additions-only instrumentation on hierarchy rung 0, followed by a
source-ordered measurement only if the operator-produced record exists. Every
ocean number is **independent** because rung 0 starts from NEMO's own
from-rest state. No model physics, card field, configuration value, carried
state, threshold, stabilizer, sea-ice selector, or `unmeasured_features`
entry may change.

## Admitted boundary

Round 115 made every recorded northwest-U operand and stored product
bit-exact given NEMO's recorded operands. Its first non-bit statement is the
carried EEN recurrence addition: 1,618 accumulator-before and 5,197
accumulator-after values differ only in the sign of exact zero. Restoring host
IEEE zero-addition semantics closes both rows, but the result cannot land from
one of the four U recurrences alone.

The compiled rung-0 program evaluates the northeast, southwest, and southeast
U recurrences immediately after northwest U
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1263-1273`). The
admitted round-107 stream records only northwest U. This round records the
three missing recurrences before any production landing.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R116-P1 | Adding one write-only, rank-complete recorder is observationally passive. | Every kt=1..10 restart and every inherited round-105, round-107, and round-110 stream is byte-identical to the admitted round-110 run. | Any moved byte rejects the instrument; reduce and reacquire before quoting an operand. |
| R116-P2 | Each stored product is exactly replayable from its recorded source operands in the compiled written order. | Recorded U thickness, neighboring V thickness, neighboring V mask, and `zpvo` reproduce `term_ne`, `term_sw`, and `term_se` bitwise on every executed level. | The first failed product rejects the instrument as insufficient or perturbing; do not interpret recurrence signs. |
| R116-P3 | Each recorded recurrence is internally exact under ordinary host IEEE addition. | `acc_after == acc_before + term` bitwise for NE, SW, and SE on every executed level. | A failed recurrence means separately stored operands do not describe the executed binary statement; stop and add a source-side discriminator. |
| R116-P4 | After the record admits, the source walk reaches northeast U first; the round-115 IEEE-zero arm closes any zero-sign-only NE recurrence differences without moving nonzero values. | Every earlier NE operand/product row is exact; the first recurrence difference is signed-zero-only; the arm closes it and moves no nonzero value. | The first earlier unequal NE row owns the walk. Any magnitude movement refutes the arm and forbids landing. |
| R116-P5 | No model statement lands before all three new streams admit and are walked in compiled order. | The round ends `STOPPED_FOR_RECORD` if the operator record is absent. | A production edit or recurrence attribution from preflight alone is a process failure and is retracted. |

## Acquisition and admission bar

The new self-describing stream records, for each of northeast, southwest, and
southeast U and every executed level on both ranks: `zpvo`, live U thickness,
neighboring live V thickness, neighboring V mask, the stored product, and the
accumulator immediately before and after the source recurrence. It also
records `mbku`. Arrays are zero-initialized and writes outside `mbku` are a
hard refusal.

The checker parses names, dimensions, and payload lengths from the header. It
requires exact two-rank owned-domain coverage; rejects the dummy level and all
out-of-loop writes; replays products and recurrences bitwise; compares the
terminal value of each recurrence against the admitted round-105 accumulator;
and requires all twenty kt=1..10 restart shards plus all inherited streams to
be byte-identical. Header, field-name, dimensions, truncation, missing-field,
duplicate-rank, bottom-index, product, recurrence, inherited-stream,
association, and restart-byte plants must fire.

The launcher uses a new target and run directory, absolute pre-created
per-rank paths, and pins every producer input by SHA-256 content rather than a
moving commit. If the record is absent, the round reports the committed
launcher as `ACQUISITION_NEEDED`; the sandbox does not retry MPI.
