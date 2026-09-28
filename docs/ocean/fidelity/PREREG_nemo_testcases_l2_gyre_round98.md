# Preregistration: NEMO-testcases L2 GYRE round 98 stage-one W walk

Date: 2026-09-16. Frozen at incoming tip
`b5ea549b04fa6bb0375fc365094ab0d7929e5a80` before any Round-98 scientific
comparison or production edit. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round98/`.

## Question, magnitude, and record boundary

Decision 41 keeps the walk in kt=1 stage 1. Round 97 made the full stage-one
momentum RHS and corrected U/V bit-exact given NEMO entry, but Rule 12 rejected
that candidate. Its paired stage table names W as the first remaining non-bit
row, at `3.5937485546815465e-8 m s-1`. The magnitude targets remain kt3 T
`1.627497246303733e-4 K` and day-30 T RMS
`1.2397011295506804e-2 K`; no W ownership of either target is assumed.

The existing stage table currently compares the model's stage transport W to
the Round-46 `ww` field. That Round-46 field is the velocity-form W computed
before the external solve and consumed by stage-one ZAD: the compiled external
program computes it and immediately calls ZAD at
`GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stp2d.f90:155-175`.
The vector stage explicitly skips that velocity-form W call at stage 1 and says
the tracer W is computed later in `tra_adv_trp` at
`GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stprk3_stg.f90:326-347`.
The direct Round-21 record is written after that later transport-form W call at
`GYRE_OMIP_L2_P3_SM_R21W/BLD/ppsrc/nemo/traadv.f90:267-295`.

The pre-implementation search found the admitted Round-21 `read_stage_ww`
reader, the Round-46/51 Decision-41 stage twin, the existing shared
`nemo_qco_wzv_operands` implementation, and the Round-87 scalar WZV trace. This
round extends those tools in place. It does not add a second stage harness or a
second numerical implementation.

## Frozen measurement contract

1. Replace only the kt=1 stage-1 W output reference in the existing stage twin
   with the admitted post-`tra_adv_trp` Round-21 `ww` record. Retain a separate,
   explicitly labelled pre-external ZAD-operand W row against the Round-46
   record. Every pre-existing non-W row must reproduce.
2. Given NEMO's recorded kt=1 stage-entry state and admitted `zFu/zFv`, walk
   the transport-form W program in compiled order: horizontal transport
   differences; area multiply and live-thickness divide; the separate
   `e3t*hdiv` materialization; Kaa-minus-Kbb QCO stretch; bracket; and the
   bottom-up carry. NEMO's transport-form divergence and materialization are
   at `GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/divhor.f90:132-154`; its
   QCO recurrence is at
   `GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/sshwzv.f90:293-300`.
3. A one-ULP change to an otherwise exact transport input must flip its named
   row and exit nonzero. A one-ULP change to an otherwise exact carry must flip
   its named row and exit nonzero. The existing commit-stamp plant must fail
   before record consumption.
4. All candidates use the one shared NEMO-identity implementation. No card
   switch, coefficient, timestep, stabilizer, or new state is eligible.

## Frozen predictions and falsifiers

1. The large `3.5937485546815465e-8` W row is predicted to be a reference-time
   mismatch and therefore **RETRACTED** as a stage-output residual. Against the
   direct post-`tra_adv_trp` record, production W is predicted AT-BAR but not
   bit-exact, consistent with the admitted Round-21 result. A DEBT result,
   bit-exact production result, or changed direct-record digest refutes this
   prediction.
2. Given NEMO transport inputs, every boundary through `hdiv` is predicted
   bit-exact. The first non-bit statement is predicted to be the separately
   stored `e3t*hdiv` product. Preserving that compiled statement with the
   repository's shared source-round primitive, and preserving the compiled
   left-associated QCO recurrence, is predicted to make final W bit-exact. Any
   earlier non-bit boundary or any remaining W unequal cell refutes this local
   candidate.
3. If and only if the W candidate is bit-exact, combine it with Round 97's held
   same-stage full-RHS member. The given-NEMO-entry kt=1 stage-1 U, V, and W
   output rows must all be BIT; every other affected stage row must be BIT or
   unchanged. Failure of any affected row holds the composition before the
   trajectory gate.
4. For an eligible composition, the frozen trajectory prediction is that kt2
   T/S remain `1.4210854715202004e-14` / `2.1316282072803006e-14`, kt2 U/V
   remain `2.7377110452773967e-12` / `3.284922138989399e-12`, kt3 T/S remain
   `1.627497246303733e-4` / `6.327735185607253e-6`, and day-30 T RMS remains
   `1.2397011295506804e-2 K`. Any movement is retained and registered. Landing
   is refused if any of 954 rows that was AT-BAR leaves the bar, first-over-bar
   moves earlier, or a moved row is absent from the full table.

## Rule-12 and testcase dispositions

| lane | frozen disposition |
|---|---|
| GYRE stage twin | Correct the W output record boundary, retain the pre-external W operand separately, require red ULP and stamp plants, and require composed U/V/W stage closure |
| GYRE kt=1--10 | Run only for a locally exact same-stage composition; compare all 954 rows against the immutable Round-85 / Round-96--97 before arm |
| GYRE days 1--30 | Run the required fresh member for any trajectory candidate and score days 1--30 against `year_owners` |
| LOCK_EXCHANGE-zco | The shared source-round statement is constructible; run focused shared-path tests and the tank gate only after GYRE eligibility |
| OVERFLOW-zps | Same shared-path requirement; preserve partial-cell construction and run the tank gate only after GYRE eligibility |
| DINO | **SHARED-STATEMENT RISK:** DINO calls the same WZV helper; no trajectory-neutrality claim, and the 96--98% regional-cancellation warning remains explicit |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record selected-integrator stage-entry transports, every transport-form W boundary and carry, native stage outputs, histories, and closure state; require bitwise tables and red plants before a claim |

No production configuration, carried-state policy, coefficient, timestep,
stabilizer, year harness, reconciliation gate, freshwater pair, #1484 guard,
NEMO source, or NEMO executable may change.
