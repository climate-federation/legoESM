# ORCA2 round 265 preregistration — matched stage-2 U update walk

Date: 2026-10-12. Frozen base: `3ca3e4d14`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. This round integrates the
source-complete scalar V-ratio association once before both consumers inside
the existing complete private fold unit, reproduces round 264, and walks the
next matched stage-2 completed-U boundary. No deck, card selector, carried
state, stabiliser, sea-ice field, or `unmeasured_features` entry may change.

## Compiled source order and existing record

NEMO builds and associates `r3ua/r3va/r3fa` before storing the live stage
ratios at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:163-180`.
At stage 2 the vector-form momentum program then executes, in order:

1. completed U/V RHS (`eos`, HPG, VOR and vector-invariant ADV) at
   `stprk3_stg.f90:326-363`;
2. the vector velocity update at `stprk3_stg.f90:365-370`;
3. the reference-thickness depth mean and barotropic correction at
   `stprk3_stg.f90:433-447`;
4. the post-stage U/V/T/S association at `stprk3_stg.f90:779-802`.

The admitted OMT-4 record already contains Kbb, Kmm, completed Krhs, raw Kaa,
the barotropic target, diagnosed correction, corrected Kaa, and the
post-association stage frame. The gate parses that record; no NEMO acquisition
or in-executable observer is added. Every comparison uses the production D114
resolved mesh route and NEMO's matched post-association program point.

## Frozen predictions and falsifiers

1. **R265-P1 — round-264 reproduction.** With the source-complete scalar
   V-ratio association applied once before both consumers, matched stage-2
   fold-band T/S remain at or below the `2e-10` campaign floor under both
   labels, and the private prefix completes finite kt=1 and kt=2. Any larger
   T/S residual or refusal invalidates the arm before the U walk.
2. **R265-P2 — first U operand.** The stage-2 entry Kbb U is at the floor, but
   the completed stage-2 U RHS is the first valued non-bit operand in compiled
   order. Substituting NEMO's recorded RHS into the literal vector update must
   close raw Kaa U to the floor. An exact RHS refutes this prediction and moves
   the first-owner test, in order, to the literal update arithmetic, the
   reference-depth correction operands, the corrected Kaa value, and finally
   the post-stage U association.
3. **R265-P3 — cumulative closure.** Cumulative substitution through the first
   unequal source statement reduces the matched completed-U fold-band maximum
   from `0.003382014663707235` m/s to the campaign floor under both labels.
   Failure means the U row is a cancelling unit; no partial statement lands.
4. **R265-P4 — trajectory consequence.** Only if P1-P3 close does the complete
   private unit run OMT-4 kt=1..10 under both labels. Completion reaches the
   full Decision-96 OMT-1..4, rung-0, rung-10, GYRE-year, DINO-month, tank,
   moved-row, citation, review and test gates. An earlier refusal or a
   certified-row loss holds the unit and names that boundary.
5. **R265-P5 — controls.** Both-label coverage, clean-tree provenance, exact
   record identity, round-264 reproduction, matched program point, source
   order, literal self-replay, first-owner sensitivity, cumulative closure,
   prefix finiteness and one-bit sensitivity each have a planted refusal.

## Stop and landing conditions

Missing admitted operands are `STOPPED_FOR_RECORD`. A non-closing first
operand is retained as a cancelling unit and the next member is named. The
complete unit lands only after kt=1..10 and every Decision-96 gate pass; no
component lands alone. No floor, clip, damping, alternative mask, carried-state
change, configuration choice, or sea-ice change is permitted.

ASKED choices: existing complete atomic unit and standing Decision-96 gate.  
Unasked configuration choices made: none.
