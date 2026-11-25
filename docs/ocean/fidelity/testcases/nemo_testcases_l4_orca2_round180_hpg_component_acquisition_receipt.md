# ORCA2 round 180 — independent HPG component acquisition

Date: 2026-10-08. Base `ef22a09a7`; acquisition commit `98ba96d5c`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round180/`.
Status: **STOPPED_FOR_RECORD**.

Every scientific claim remains **independent hierarchy rung 0**. No
given-NEMO-entry number is used as an independent result. No production model,
configuration, carried-state rule, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Verdict

The existing round-41 HPG component record cannot answer round 179's OPEN
item. Its admission and receipt explicitly label it **given NEMO's entry** and
its producer is the shipped ORCA1ICE deck. Round 179's debt is measured from
the corrected independent hierarchy-rung-0 entry. Mixing the records would
violate Decision 52's binding label rule.

Round 180 therefore commits a new additions-only, rank-complete writer against
the admitted round-92 rung-0 build. It records the compiled HPG boundaries in
source order: the along-surface accumulators `zhpi/zhpj`, the s-coordinate
corrections `zuap/zvap`, and their completed sums, together with `rhd`, live
`e3w`, live `gdept_z0`, and the U/V metric reciprocals. Those are the operands
and statement results written by the executing rung-0 routine
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`).

The record has not been produced. The writer's CPU preflight syntax-compiles,
and its planted missing-field layout exits 69 with `STATUS PLANT-FIRED`, but no
component payload has been read and no first component or operand is named in
this round.

## Record contract

The new stream is self-describing. Each rank file carries its magic, version,
step, RK levels, rank, local shape, global origin, owned bounds, precision and
field count; every field then carries its own name, rank and dimensions before
its payload. Admission requires both ranks to cover the 148x180 domain exactly
once, the inherited round-92 RHS streams to remain byte-identical, and all 20
ocean restarts to remain byte-identical. Plants cover the header, field name,
per-field rank, field dimensions, truncation, swapped rank, inherited RHS byte,
restart byte, and source-layout census.

The launcher pins the content hashes of its committed run script, patch,
checker and preregistration. It does not pin a producer commit that the round
itself can move past. The patch is additions-only and writes one file per rank.
It uses the new target `ORCA2_OMIP_L4_R180HPG1`; it never reuses or overwrites
an admitted run.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R180-P1 the old record is ineligible and a rung-0 record is required | **CONFIRMED** by the old admission label and different producer deck. |
| R180-P2 the new writer is passive | **UNMEASURED-with-spec** pending the rank files and restart comparisons. |
| R180-P3 `zhpj` is the first non-bit component | **UNMEASURED-with-spec**; no payload was read. |
| R180-P4 the northern-fold north-neighbour association is the first input | **UNMEASURED-with-spec**; no payload was read. |
| R180-P5 acquisition and measurement only | **CONFIRMED**. |

## Mechanical checks and review

The committed launcher passes `--preflight-only`: the exact patch applies with
zero fuzz to the pinned source, the writer layout census passes, and the
preprocessed Fortran syntax-compiles. The `--plant-layout` invocation exits 69
and prints `STATUS PLANT-FIRED layout`. The direct acquisition unit battery
passes 8/8.

The default and round-180 citation gates pass with zero unmapped citations or
audit failures. Shifting the compiled HPG citation by two lines makes the gate
exit 1 with `STATUS FAIL`.

The focused acquisition, predecessor-parser and citation-gate battery passes
34/34. The single required `tests/ocean/fidelity -n 12` invocation collected
2,836 tests and reached 99%, with 2,805 passes, 7 skips and 5 failures, then
reproduced the known nonterminal tail and was interrupted after three silent
30-second polls; 19 tests had no terminal result. Four failures are the listed
pre-existing round-35 stamp-scope ratchet, round-129 GYRE record-backed pin,
worktree-stamp ratchet and SI3 scalar-math provenance gate. The fifth is the
GYRE round-179 acquisition's layout-plant test refusing the then-untracked
round-180 receipt as a dirty tree; its final-clean-tip isolation is reported
after this receipt commit and passes 1/1.

The required separate read-only Codex review was attempted and returned
**independent review unavailable in-sandbox**:
`failed to initialize in-process app-server client: Read-only file system`.

No `packages/` file changed, so the ORCA2, GYRE, DINO, tank and trajectory
landing gates have no eligible model diff in this stopped acquisition round.

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round180_hpg1_acquisition/run.sh --run`.
   The launcher builds the new target, runs the ten-step rung-0 deck, exercises
   every admission plant and admits the record or refuses loudly.
2. After admission, compare `zhpj`, `zvap`, and `sum_v` in compiled order on
   round 179's registered 68 northern-fold faces. Stop at the first component
   above the fixed floor, then split its local and north-halo operands.
3. Analyse raw HPG, `e3v`, `vmask`, and `r1_hv0` as one cancelling unit before
   any landing. Only after that unit closes may the held V-transport/halo unit
   and the independent month's step-96 refusal be retested.

The independent 240-step month score remains **UNMEASURED-with-spec**.
ASKED choices are the source-ordered HPG component walk. UNASKED choices are
empty.
