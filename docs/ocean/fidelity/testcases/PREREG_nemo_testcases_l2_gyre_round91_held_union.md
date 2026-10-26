# Round 91 preregistration: held locally exact union discriminator

Date: 2026-09-14

Frozen production commit: `ebcee36f321fcd3296e6fb0968adde6bddc38a7d`

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round91/`

## Governing question and immutable before arm

Operator note F requires the discriminating measurement before any further
single landing: apply every held candidate whose NEMO-given-input proof remains
bit-exact at this tip, measure their union, and land nothing.  The immutable
before arm is the landed Round-85 production state.  Its certified kt2
T/S/U/V maxima are `1.4210854715202004e-14`,
`2.1316282072803006e-14`, `2.7377110452773967e-12`, and
`3.284922138989399e-12`; kt3 T/S are `1.627497246303733e-4` and
`6.327735185607253e-6`; day-30 T RMS is
`1.2397011295506804e-2 K`.  No scratch toggle is a before arm.

The union answers one question: do all still-locally-exact source
materializations together move the first-over-bar boundary toward the exact
bar, or do they expose the same compensation that rejected the separate arms?
The union is a discriminator only and cannot land in this round.

## Candidate inventory and source statements

The current production state already contains three locally exact members, so
their old patches must not be applied a second time:

1. Decision-37's six absolute barotropic histories and paired window-boundary
   mean owner.  The compiled GYRE program forms its AB3/AM4 midpoint from the
   current/b/bb absolute values and rotates those histories at
   `GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-509,783-795`.
2. The tracer-cell drag coefficient, external-window entry reciprocal, and
   explicit drag association.  The compiled program stores the cell
   coefficient at
   `GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/zdfdrg.f90:229-235` and forms
   and consumes the entry reciprocal at
   `GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:363-375,703-706`.
3. The stage-specific ZAD operands.  The compiled program forms W transport,
   applies the Kmm velocity difference and live face thickness, carries the
   product downward, and updates the bottom at
   `GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90:105-137`.

The candidate delta will contain each applicable held patch once:

4. Round 60's TKE raw-mixing-length terminal update and stored
   inverse-Prandtl association, if its current-tip record replay remains exact.
   The compiled statements are
   `GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:394-412,589-701`.
5. Round 62's tracer vertical-solve coefficient expression, if the recorded
   matrix/sweep replay remains exact.  The compiled program zeros the surface
   coefficient, builds the matrix, writes the content RHS, and solves it at
   `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:450-477,523-578`.
   The separately preserved content-materialization patch is excluded because
   Round 62 refuted its live step-1 exactness; it is not a locally exact member.
6. Round 89's source-rounded Kaa/W/WZV/ZAD/assignment bundle, which is the
   strict superset of Round 88 and therefore replaces rather than accompanies
   the Round-88 patch.  The compiled program updates vector Kaa U/V and masks
   them at
   `GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:666-675`, uses
   Kaa-minus-Kbb in the W stretch at
   `GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300`, and rotates
   the next Kaa SSH scratch at
   `GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:220-226`.

Round 51's obsolete deviation-history patch is excluded because Decision 37's
absolute histories supersede it.  Round 62's refuted content patch is excluded
as stated above.  Round 88 is not separately applied because Round 89 includes
it byte-for-byte plus the assignment member.  A patch that cannot be rebased
without changing its source statement, or whose exact replay/control fails at
the current tip, is reported `LOCAL PROOF FAILED` and excluded rather than
silently repaired.

## Frozen local-proof and measurement order

1. Re-run the existing clean-stamped history, drag, ZAD/WZV/Kaa/assignment,
   TKE, and tracer-coefficient record replays.  Every admitted member requires
   zero unequal consumed cells and its established nonzero-ULP plant must exit
   nonzero.  Unit tests alone do not establish local exactness.
2. Apply only the members that pass, commit the complete scratch union, and
   run the certified GYRE kt=1..10 ladder.  Score all 954 rows against the
   immutable Round-85 artifact and register every moved row, including every
   worsening.
3. Operator note F overrides the ordinary early stop for this discriminator:
   run member 0 through days 1--30 even if the ladder rejects the union, then
   score every day against the same NEMO root and report day-30 T RMS.
4. Report tank execution statement-by-statement.  No tank result can authorize
   landing this union.  DINO shared-statement risk remains explicit.  ORCA2 is
   `UNMEASURED-WITH-SPEC`.

All scientific executions use CPU, JAX fp64/libm, production JIT, and a clean
commit stamp.  The failed Round-90 acquisition is not a union member: its run
produced a correction record, but admission stopped on the reader's incorrect
2-D shape contract before any record value was cited.

## Frozen numerical prediction and falsifiers

The dominant Round-89 superset previously produced kt2 T/S/U/V
`3.0652394795183113e-3`, `4.835710022078388e-3`,
`6.733005735178965e-7`, and `1.3183569256688065e-6`, and kt3 T/S
`1.566544749833554e-2` and `4.324733116938262e-3`.  The frozen prediction is
that adding the much smaller TKE and tracer-coefficient materializations will
not repair that compensation exposure: each of those six rows will remain
within 5 percent of the Round-89 value, first-over-bar will remain kt2
T/S/U/V rather than move toward the bar, and day-30 T RMS will exceed the
Round-85 `1.2397011295506804e-2 K` baseline.  The direction, not proximity, is
the decision-bearing claim.

The prediction is **CONFIRMED** only if every included member retains local
bit exactness, every plant exits nonzero, the complete union has first-over-bar
kt2 T/S/U/V, and day-30 T is worse than the Round-85 baseline.  It is
**REFUTED** if any local proof fails, first-over-bar moves later or loses a
field, any of the six ladder targets lies outside the registered 5-percent
band, or day-30 T is unchanged/improved.  Regardless of prediction outcome,
the decision answer is mechanical: the union moves first-over-bar toward the
bar only if its earliest DEBT step is later than kt2, or at kt2 it has a strict
subset of the baseline's `{U,V}` fields.  Equal kt2 `{U,V}` is no movement;
adding T or S is movement away.

## Rule-12 cards and prohibitions

GYRE reports the full 954-row table, kt2 U/V, kt3 T/S, first-over-bar, and the
days 1--30 table including day-30 T.  LOCK_EXCHANGE and OVERFLOW execute shared
WS-RK3/TKE statements only where their resolved configuration selects them;
otherwise the compiled namelist/source branch is cited as non-execution.  DINO
uses MLF but shares TKE and source-rounding helpers, so neutrality cannot be
inferred, especially under its 96--98 percent regional cancellation.  ORCA2
remains `UNMEASURED-WITH-SPEC`: establish its compiled integrator and mixing
branches, align native masked T/S/U/V/SSH plus every union operand/history at
kt1..10 in fp64, require zero unequal given-input rows and normalized
L-infinity at `1e-15`, and reject any AT-BAR loss or earlier first-over-bar.

No production physics lands.  No configuration, coefficient, timestep,
threshold, stabilizer, carried-state choice, restart policy, year harness,
reconciliation gate, freshwater pair, #1484 guard, NEMO source, or NEMO
executable changes are authorized.
