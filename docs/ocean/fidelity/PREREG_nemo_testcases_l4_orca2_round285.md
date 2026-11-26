# Preregistration — ORCA2 round 285 G4 Decision-122 import

Date: 2026-10-15

## Scope

Import the six source units that made geometry-ladder rung G4 bit-identical
through kt 1..10 under both labels, select each applicable unit on the
production ORCA2 card, and apply Decision 122 to rung 0, OMT-1..4, and rung 10
under both labels.  This round does not change runoff depth, sea ice, any NEMO
deck/source/record, or the held completed-stage top-V exchange.  Decision 126
remains pending and is not re-raised.

The imported units are:

1. the stored barotropic wind divisor and pivot-row V-stress fold, followed by
   the surface-stress deposit through the live face thickness;
2. the source-ordered stage-1/2 tracer update;
3. the source-ordered stage-3 temperature RHS, including the materialized
   `Krhs + qsr` addition;
4. the stored stage-2 thickness used consistently by every stage-3 tracer
   consumer.

The implementation is imported from the certified grid-ladder commits rather
than re-derived.  Selectors remain false by default and are enabled explicitly
by `build_orca2_zps_card`, the production ORCA2 identity.

## Compiled statements

The shipped ORCA2 deck executes the same compiled statements as G4:

- stored wind reciprocal: `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:225-230`;
- T-stress fold and U/V face interpolation:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:553-569`;
- top-cell stress deposit:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:334-340`;
- stage-1/2 thickness-weighted tracer update:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:672-680`;
- stored stage ratio:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:215-227` and
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:257`;
- non-solar heat, runoff, RGB shortwave, and the implicit tracer solve:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:307`,
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:279-282`,
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traqsr.f90:387-394`, and
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:231-233,288-289`.

The resolved deck has `ln_rnf`, `ln_qsr_rgb`, `ln_dynspg_ts`, and
`ln_zdftke` true, so none of these is a dead branch on the shipped card.
Rung 0 and OMT-1..4 have no live surface forcing, so the three stress units
must be inert there; the tracer-order units still execute in their resolved
stage program.

## Frozen predictions and falsifiers

1. **R285-P1 — source identity.** The imported package hunks match the G4
   certified implementations.  A missing unit, a second implementation, or a
   selector defaulting true refutes this prediction.
2. **R285-P2 — executed-line identity.** Each selected unit names the same
   compiled NEMO line on G4 and the shipped ORCA2 build.  A different enclosing
   routine, scheme arm, or inactive namelist switch refutes the unit and keeps
   its production selector off.
3. **R285-P3 — Decision 122.** Singly and cumulatively, no measured card loses
   an exact row and every first-over-bar row is unchanged or moves toward NEMO.
   Any exact-row loss or first-over-bar move away refuses that unit under D122;
   toward/away/equal counts are registered but are not a veto.
4. **R285-P4 — forcing-path scope.** The surface-stress unit moves zero rows on
   rung 0 and OMT-1..4 because those cards supply no surface forcing, while it
   may move rung 10.  Movement on an unforced card refutes selector scoping.
5. **R285-P5 — G4 controls.** On the certified G4 record, deselecting each unit
   restores its registered boundary (wind/stress, 58/45 stage-1 T/S,
   489/200 stage-3 T, or the six independent kt9-10 debt rows), while the full
   selected set remains bit-identical.  A control that stays green refutes the
   corresponding import claim.
6. **R285-P6 — shared cards.** GYRE's certified ladder/year, DINO's month, and
   the tank/card gates remain unchanged because their production cards leave
   every new selector false.  Any above-floor movement is a red gate.

All failed predictions remain in the receipt.  A package landing also owes the
default citation gate plus a planted failure, focused tests, the single
permitted fidelity battery, and independent operator review.

## Evidence target

Evidence goes under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round285/`.
The receipt is
`docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round285_g4_d122_import_receipt.md`.
