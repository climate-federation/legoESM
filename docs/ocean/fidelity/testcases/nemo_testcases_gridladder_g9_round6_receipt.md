# Grid ladder rung G9 — round 6 receipt (gridladder_r51)

Status: **STOPPED FOR RECORD; G9 NOT CERTIFIED.**  Starting kt 2 from
NEMO's exact recorded step entry and exact recorded external/barotropic
handoff makes every owned stage-1 value bit-identical.  The kt-2 departure is
therefore amplification of kt 1's at-bar residue, not a kt-2-only statement.
Round 46 does not record NEMO's internal kt-1 FCT boundaries, so it cannot
identify the remaining statement owner and no production change is made.

Record: round 46's admitted `oracle_g9_r46`, unchanged.  A new WRITE-only
boundary record is requested below.  Preregistration commit: `f82e6a0e2`.

- codex review: operator
- independent review: operator (Claude)
- evidence: `gridladder_rounds/round51/`
- implementation: diagnostic controls and record-entry/acquisition tooling
  only; no model code changed

## 1. Registered split: carry-over confirmed

The committed ladder probe's new `--record-step-entry` route loads the
round-46 kt-2 T, S, u, v and ssh entry fields, seeds the preceding barotropic
pair, and supplies the recorded kt-2 stage-1 external handoff.  It verifies
the loaded entry before stepping and scores only ORCA's owned cells; columns
0 and -1 are cyclic overlap storage and are reported separately rather than
silently mixed into the physical score.

All seven entry-identity counts are zero.  At kt 2 stage 1, u, v, T, S, ssh,
uu_b and vv_b each have zero unequal owned cells, zero maximum difference,
and are bit-identical.  The excluded cyclic-overlap unequal counts are 0, 63,
64, 64, 0, 0 and 0 respectively in that same field order.  This confirms P1:
the independent kt-2 departure is inherited amplification.

The negative control removes only the recorded external handoff while keeping
the same exact entry.  It exits 1 and makes all seven rows DEBT.  Unequal-cell
counts for u, v, T, S, ssh, uu_b and vv_b are 15,723; 15,811; 16,297; 16,297;
16,297; 15,723; and 15,811.  Their maximum absolute differences are
`3.826e-3`, `1.055e-2`, `2.841e-5`, `9.805e-6`, `5.998e-3`, `3.826e-3` and
`1.055e-2`.  Thus the exact result is connected to the added wiring.

Evidence:

- `g9_kt2_record_entry_stage1_v5.json`
- `g9_kt2_record_entry_without_external.json` and `.log`

## 2. Round-50 control caveat closed

The literal-FCT source-order fixture now uses nonzero, nondegenerate U, V and
W transports with independently spread magnitudes.  Cell area is an explicit
traced argument to the jitted x64 call.  Its exact NumPy transcription follows
the live two-step upstream program, including NEMO's first divergence in
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-609`.

The restored test passes.  Reassociating `(du + dv) + dw` as
`du + (dv + dw)` fails on 154 of 768 values and exits 1.  Replacing the
reciprocal-area multiplication with direct division fails on 170 of 768
values and exits 1.  These are live controls: V and W are not zeros, and XLA
cannot fold the traced area reciprocal.  Transcripts are
`sum_order_plant.log` and `divide_area_plant.log`.

## 3. Statement walk and why the record cannot decide

The registered P2 prediction was that the real-record first divergence would
be the first unequal statement.  The card-side compiled trace and exact NumPy
transcription can test its arithmetic, but round 46 contains only the FCT
entry operands and the final post-advection RHS.  It contains no NEMO array at
the first divergence, midpoint, averaged-flux divergence, limiter input or
limiter output.  Comparing a card intermediate to a replay of that same card
intermediate would be a self-comparison, not the required NEMO statement
comparison.  P2 therefore remains **UNMEASURED**, not confirmed or refuted.

The recorded final boundary is already at the bar but non-exact: round 50
measured 21 T and 17 S unequal wet cells after advection.  That boundary is
the accumulated upstream result plus the limited correction written by the
final FCT loop (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:320-329`), so it
does not attribute the residue to either contributing statement.  No
reciprocal/divide, reassociation, or contraction change is justified by the
available record.  Consistent with the “fix only a cited statement” rule, no
production statement is changed.

The requested acquisition is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g9_orca2tadv/run_r51_fct_walk.sh`.
It creates one new config from the pinned round-46 record build, adds only
WRITE-only observations to `traadv_fct.F90`, and records 20 full native arrays
per tracer across entry, first/midpoint, averaged/RHS, pre-limiter,
post-limiter and final-RHS frames.  Admission requires every pre-existing
kt-1 and kt-2 record to remain byte-identical to round 46.  Read-only
preflight passes `FCT_OBSERVER_ADDITIVE_OK` and names the intended target and
evidence directory.  The author ran neither `makenemo` nor NEMO.

## 4. Certification and blast radius

P3 (both labels kt 1..10 without DEBT) and P4 (26 non-G9 jaxprs unchanged)
are not reached because no statement was changed.  A fresh score after an
unchanged model would only duplicate round 50, so it was not run.  G9 remains
uncertified, and this round certifies none of the ORCA2 fold import: the
stage-1 V transport mask, round-46 exchanges, round-47/48 fold statements and
stage pair, round-49 stage-3 RHS order, and round-50 literal FCT remain
pending the missing statement record and a subsequent clean both-label
kt 1..10 score.

## 5. Tests, gates and choices

Focused evidence before the final gate:

- hardened literal-FCT exact-value test: pass under jit, traced inputs and
  x64;
- sum-order and divide-area plants: both fail with exit 1;
- exact-entry stage-1 probe: seven of seven owned fields bit-identical;
- removed-external wiring plant: seven of seven rows DEBT, exit 1;
- acquisition preflight: pass; observer patch additive.

ASKED: preregistered kt-2 split; exact NEMO entry; statement walk; cited-only
fix discipline; conditional both-label rescore and certification verdict.

UNASKED, offered for revert: none.

UNVERIFIED: the first unequal NEMO FCT statement inside kt-1 stage 3.  The
new record is required to distinguish the first divergence, midpoint,
averaged-flux RHS, limiter and final accumulation boundaries.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r51-29306485
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r51-29306485/scripts/validate/ocean_fidelity/testcases/nemo_testcase_g9_orca2tadv/run_r51_fct_walk.sh
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — exact kt-2 entry proves carry-over, but round 46 lacks the internal NEMO FCT boundaries needed to name and fix the kt-1 statement.
