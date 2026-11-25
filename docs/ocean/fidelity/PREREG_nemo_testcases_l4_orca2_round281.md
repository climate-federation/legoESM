# ORCA2 round 281 preregistration

Date: 2026-10-15

Scope: admit round 280's rank-complete OMT-4 stage-1 record, split the
1,440/5,400-cell completed-V source-row debt between the raw vector update and
the following barotropic correction at matched pre-`lbc_lnk` program points,
then walk the first non-bit boundary as far as the admitted operands permit.
Every trajectory number is **independent OMT-4**. Decision 126 remains pending;
runoff and every sea-ice field, including `unmeasured_features`, stay unchanged.

## Frozen source order

The executing vector branch writes `vv(Kaa)` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:467-470`.
It next diagnoses the reference-thickness column correction and adds it through
the 3-D V mask at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:513-537`.
Only later does the stage program call the completed U/V/T/S boundary
association. The new record brackets the first two statements as `pre_v` and
`cor_v`; its downstream `final_v` brackets the later association.

The candidate values are read from the production OMT-4 path on the current
tree. The raw update is exposed by the existing write-only
`expose_stage1_raw_momentum` seam; the corrected value is exposed by the
existing `expose_momentum_stage=1` seam. Both seams substitute the diagnostic
return only after the ordinary step completes. No new executable observer is
permitted.

## Frozen predictions and falsifiers

**R281-P1 — admission.** The operator-produced record remains
`PASS_R280_OMT4_STAGE1_ADMISSION`: two self-describing rank records, exactly-once
148x180 coverage, 80 inherited frames and both kt=10 restarts byte-identical.
REFUTE on any admission refusal or changed inherited byte.

**R281-P2 — raw vector update.** Candidate `pre_v` is bit-exact against NEMO on
all active cells and on all 5,400 stored slots of source row `j=146`.
CONFIRM only at zero unequal; REFUTE on any unequal bit.

**R281-P3 — barotropic correction owner.** Candidate `cor_v` is the first
non-bit boundary and reproduces round 279's source-row census of 1,440/5,400
unequal with maximum `0.0162073695895 m/s`. CONFIRM only if P2 is exact and
both count and maximum reproduce; REFUTE if `pre_v` is already non-bit, if
`cor_v` stays exact, or if its census/magnitude differs.

**R281-P4 — post-stage alternative.** If both `pre_v` and `cor_v` are bit-exact
but candidate completed `final_v` retains round 279's 1,440/5,400 source-row
debt, the update/correction attribution is REFUTED and the first non-bit
statement moves to the subsequent completed-stage V association at
`stprk3_stg.f90:795-796`. CONFIRM only with exact P2/P3 boundaries and a
reproduced downstream census; REFUTE otherwise.

**R281-P5 — operand walk.** If P3 confirms, compare the correction's operands
in written order: corrected-stage input V, reference V thickness, stored
reference-depth reciprocal, external-mode V target, and 3-D V mask. The first
valued unequal operand owns the next walk only after replaying the candidate
correction from candidate operands and NEMO's correction from NEMO operands.
If an operand is absent from admitted streams, stop `STOPPED_FOR_RECORD` and
name it rather than deriving a proxy. If P4 confirms instead, no correction
operand is promoted; the association is the cited next statement.

## Controls and landing rule

The round-280 admission's eight plants must fire again. The source-split gate
adds independent raw-boundary, corrected-boundary, source-row, source-order,
matched-program-point and one-bit plants; each must fire. The instrument must
reproduce round 279's completed source-row number before any attribution is
recorded.

No package statement lands unless the first non-bit statement has a
source-exact one-variable candidate and passes the standing OMT/rung-0/rung-10,
GYRE, DINO, tank, moved-row, citation, test, and operator-review gates. A
measurement-only result is HELD with its next exact operand or record named.
