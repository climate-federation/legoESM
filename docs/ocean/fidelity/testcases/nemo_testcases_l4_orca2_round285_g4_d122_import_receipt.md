# ORCA2 round 285 — G4 certified-statement import under Decision 122

Date: 2026-10-15

Status: **LANDED**

Labels below are explicit: `independent` means legoESM's corrected ORCA2
initial state; `given NEMO's entry` means the Decision-52 twin operand.  No
table mixes the two labels.  Sea ice and the six shipped-card ice selectors
are unchanged.  Decision 126's runoff-depth choice is still pending and was
not changed or re-raised.

## Source identity

The imported implementations are the selector-gated statements certified on
the geometry ladder's G4 rung.  G4 uses the same compiled ORCA2 routines as
the shipped card.  The executing statements are:

- the stored live-depth wind divisor at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:225-230`;
- the folded T-point stress and U/V-face interpolation at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:553-569`, followed by the top-cell
  live-face-thickness deposit at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:336-339`;
- the source-ordered stage-1/2 thickness-weighted tracer update at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:678-680`;
- NEMO's stored stage ratios at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:217-224` and
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:257`;
- the stage-3 non-solar heat, surface-only runoff, RGB shortwave and implicit
  tracer-solve consumers at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:307`,
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:279-282`,
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traqsr.f90:387-394`, and
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:231-233` plus
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:288-289`.

The package hunks were imported from the certified grid-ladder commits rather
than re-derived.  All six selectors default false.  The production ORCA2 card
selects all six; unforced rung 0 and OMT cards explicitly leave the two
surface-heat consumers off, because those programs do not supply the operands.
The G4 certification receipts are
`nemo_testcases_gridladder_g4_round2` through `round9`; their terminal gate is
bit-identical for every field at kt=1..10 under both labels.

## Frozen predictions

R285-P1 **CONFIRMED**: the imported hunks retain fail-closed false defaults and
reuse the ladder implementations.  R285-P2 **CONFIRMED**: the routines and
executing branches above are live in the shipped ORCA2 deck.  R285-P3
**CONFIRMED**: every measured card retains all exact rows and its first
over-bar row is equal or toward.  R285-P4 **CONFIRMED**: the unforced OMT-1,
OMT-2 and OMT-3 ladders are unchanged; the tracer-order statements act where
their programs expose them.  R285-P5 **CONFIRMED** by the imported G4
known-answer controls and the local selector plant.  R285-P6 **CONFIRMED**:
the shared-card gates below are unchanged.  No prediction was refuted.

Both local plants fired: retaining an unforced heat selector exits 2 with
`STATUS PLANT-FIRED`; planting an exact-row loss exits 2 with
`previously exact rows left the bar`.

## Decision-122 census

The gate result is `PASS_R278_D122_DECISION`.  Counts are RMS
`toward / away / score-equal`; score-equal rows can still have bit movement.
The direction census is registered, not a Decision-122 veto.

| card | label | toward / away / equal | early kt1-3 | first over-bar | kt10 stage-3 ssh max before -> after (m) |
|---|---|---:|---:|---|---:|
| rung 0 | independent | 89 / 80 / 0 | 14 / 15 / 0 | equal | 0.3830826119572967 -> 0.38308261195730303 |
| OMT-1 | independent | 0 / 0 / 0 | 0 / 0 / 0 | equal | 0.7469693088949136 -> identical |
| OMT-2 | independent | 0 / 0 / 0 | 0 / 0 / 0 | equal | 0.34650568911180074 -> identical |
| OMT-3 | independent | 0 / 0 / 0 | 0 / 0 / 0 | equal | 0.36359234522692674 -> identical |
| OMT-4 | independent | 110 / 57 / 2 | 14 / 14 / 1 | equal | 0.3935712593590627 -> 0.3935712593591211 |
| rung 10 | given NEMO's entry | 99 / 96 / 0 | 25 / 30 / 0 | toward | 0.2866995810292147 -> 0.28696010680741774 |
| rung 10 | independent | 96 / 99 / 0 | 21 / 34 / 0 | equal | 0.29399731938907425 -> 0.2935613576547812 |

There are zero exact-row losses on every row above.  The small maxima are
reported quantities, not substituted landing predicates.

## Shared-card and software gates

- GYRE base and tip: 70 ladder rows, zero differing or worsened rows, first
  over-bar kt=3 unchanged; residual archive SHA256
  `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
- GYRE day 30: 30/30 daily snapshots byte-identical; aggregate SHA256
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
- DINO/tank/generic-card focused battery: 61 passed.
- Imported-statement focused tests: 6 passed.
- Default citation gate: PASS after the required SequenceMatcher re-anchor;
  zero unmapped citations and zero audit failures.  The round citation plant
  fires.
- The required `tests/ocean/fidelity` battery is recorded below after its one
  permitted run.  Independent review: operator.

Evidence is in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round285/`.

## OPEN

1. Resume Decision 125's held completed-stage top-V exchange diagnosis from
   the matched stage-1 vector-form V update; it remains off on every ORCA2
   card.
2. Continue the round-282/284 vector-form fold-unit refusal walk only from a
   matched passive program point; no held selector was enabled here.
3. Decision 126 remains pending: the shipped runoff-depth setting is unchanged.
