# ORCA2 round 280 preregistration

Date: 2026-10-15

Scope: split round 279's 1,440-cell stage-1 V source-row debt between
NEMO's vector-form velocity update and its following reference-depth
barotropic correction at the same pre-`lbc_lnk` program point.  All numbers
remain **independent OMT-4**.  Decision 126 is pending and the runoff card is
not changed.  Sea ice and its `unmeasured_features` tuple are untouched.

## Frozen source order

The compiled ORCA2 vector branch writes stage-1 `vv(Kaa)` from `vv(Kbb)`,
`rDt`, `vv(Krhs)`, and `vmask` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:467-470`.  It then diagnoses
`zvb = vv_b(Kaa) - SUM(e3v_3d*vv(Kaa))*r1_hv_0` and adds
`zvb*vmask` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:513-537`.  The held completed-stage V exchange
comes later and is already exonerated as the magnitude creator by round 279.

**Pre-measurement citation correction.**  The first committed preregistration
misidentified this second span as `stprk3_stg.f90:523-550`; those lines begin
inside an instrumented stage-2 writer and extend beyond the executing
correction.  The compiled correction is the range above.  No prediction,
threshold, or measurement changed.

The admitted OMT-4 record under round 222 contains the completed per-stage
state and a stage-1 transport stream, but that transport stream records
`vv(Kmm)` before the stage update, not `vv(Kaa)` before and after the
barotropic correction
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:286-310`).  Therefore it cannot split
the two source-row statements.  The round-175 rank-complete stage-1 recorder
already defines the needed `rhs_v`, `pre_v`, and `cor_v` fields; round 280
reuses that schema and writer on the OMT-4 deck.

## Frozen predictions and falsifiers

**R280-P1 — record gap.**  The admitted round-222 OMT-4 files contain no
rank-complete stage-1 `pre_v`/`cor_v` pair.  CONFIRM if the existing stream
census and compiled writer show only `Kmm` V plus the completed stage frame;
REFUTE if an admitted self-describing pair at the two required program points
already exists.

**R280-P2 — raw vector update.**  The source-row raw `pre_v` update is bit
exact against NEMO on all 5,400 row slots.  CONFIRM if unequal is 0; REFUTE if
any raw slot differs.  This prediction follows the already-landed stage-1 RHS
handoff and vector update statements, but it is a prediction, not a claim.

**R280-P3 — barotropic correction owns the row.**  `cor_v` is the first
non-bit boundary and reproduces round 279's 1,440/5,400 source-row census with
maximum `0.0162073695895 m/s`.  CONFIRM only if `pre_v` is exact and `cor_v`
has that census and maximum; REFUTE if the raw update is already non-bit, the
correction closes it, or the count/magnitude differs.

**R280-P4 — additions-only acquisition.**  The two per-rank records parse
their self-describing headers, cover the global 148x180 domain exactly once,
and the instrumented run reproduces all 80 admitted round-222 stage frames
plus both kt=10 restarts byte-for-byte.  REFUTE on any missing field, overlap,
non-finite owned value, changed inherited frame, or changed restart.

## Controls and stopping rule

The copied admission retains corrupt-header, field-name/dimension,
truncation, swapped-rank, coverage, and passivity-byte plants.  The source
split adds one raw-bit plant and one corrected-bit plant; each must fire.

No package/model statement lands before the rank-complete record admits.  If
the record is absent, this round ends `STOPPED_FOR_RECORD` with the acquisition
launcher.  Once admitted, the first non-bit statement is walked through its
recorded operands in compiled order; any landing still requires the standing
Decision-96/122 scientific gates and the operator's independent review.
