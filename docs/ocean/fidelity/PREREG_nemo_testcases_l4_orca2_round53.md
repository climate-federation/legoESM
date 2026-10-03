# ORCA2 round 53 preregistration — OVERFLOW ENS signed-zero operands

Date frozen: 2026-09-27  
Base: `1699ab33be3580867162d066d8b01af6dbb4fca8`  
ORCA2 claim label: **given NEMO's entry** (Decision 52; no ORCA2 trajectory
measurement is planned)  
OVERFLOW claim label: **given NEMO's recorded operands**

## Frozen question and scope

Round 52 established that the selected stage-2 ENS vorticity statement changes
16,135 of 16,900 active-u bit patterns from negative zero to positive zero.
The admitted round-50 boundary record does not expose the internal operands of
the producing assignment.  This round adds a WRITE-only NEMO acquisition for
the source-ordered `zwz`, `zuav`, their product, and the accumulator immediately
before and after the assignment at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:641-667`.

No configuration, selector, default, carried-state field, threshold, score
domain, stabiliser, sea-ice field, or NEMO arithmetic statement may change.
The held QCO package hunk is not restored.  The acquisition patch and every
file it reads must be committed and referenced repo-relatively.  The operator,
not this sandbox, runs `makenemo` and `mpirun` under a new target name.

## Existing implementation and instrument boundary

Repository search found the round-50 self-describing record writer/parser and
the round-52 vorticity boundary scorer.  The new instrument extends those
formats and admission checks; it does not implement a second vorticity
operator.  The writer is active only for kt=3, stage 2, `np_CME`, and records
owned arrays without feeding any value back to NEMO.

The checker must parse the record's magic, version, time/stage/branch integers,
owned origins/extents, fp width, field count, and every field's name/rank/shape
before reading its payload.  It may not predict a total byte count or a header
tuple independently of the record.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R53-P1 | An additions-only writer can expose the selected ENS assignment without moving any inherited output. | Dry diff removes zero source lines; syntax/preflight and self-describing-schema plants pass; restart, mesh, and every inherited stream are identical after the operator run. | Any removed/replaced source line, parser ambiguity, or inherited movement: refuse the acquisition. |
| R53-P2 | At the first active-u signed-zero cell from round 52, the final added product is positive zero and `-0.0 + +0.0` produces the recorded `+0.0`. | Header-derived fields locate the same cell; product is `+0.0`, accumulator-before is `-0.0`, accumulator-after is `+0.0`, and recomputing the recorded addition is bit-exact. | Any different sign/value or non-exact replay: keep the QCO arm held and name the first internal operand that differs before proposing code. |
| R53-P3 | The output comparison is non-vacuous. | A one-ULP payload plant and a producer-stamp plant both exit nonzero; after acquisition, changing one previously equal addition result adds exactly one refusal. | Any plant stays green: reject and repair the instrument before quoting a number. |
| R53-P4 | No scientific statement lands in this acquisition round. | Final `packages/` diff equals the base; disposition is `STOPPED_FOR_RECORD` with one operator-run path. | Any package change: remove it unless separately preregistered under the full shared-card landing gate. |

Failed predictions remain in the receipt as **REFUTED**; no field list, score
mask, or exact bar changes after the record is produced.

## Required verification

1. Read and cite the exact compiled `np_CME` and accumulator statements from
   the producing configuration.
2. Dry-apply the committed patch, require an additions-only diff, preprocess
   with the producing keys, and syntax-check both writer and patched routine.
3. Unit-test the header-driven parser, including malformed name, shape,
   truncation, trailing-byte, non-finite, digest, and producer-stamp refusals.
4. Run focused tests, Ruff/compile checks, default and round citation gates with
   a planted shifted citation, the shared-card battery, and the ocean-fidelity
   battery once.
5. Run separate read-only `codex exec` review and record its verdict or the
   mandated unavailable wording.

## Frozen OPEN

After the operator produces and admits the new record, walk the four operands
in compiled order and either land a source-exact signed-zero association under
all shared-card gates or keep it held at the first non-bit internal statement.
Then return to ORCA2's whole-card kt=1 stage-1 T owner, the independent
Decision-52 initial state/year, and round-20 slow forcing.  Sea ice remains at
`STOP_SELECTOR_GAP`.
