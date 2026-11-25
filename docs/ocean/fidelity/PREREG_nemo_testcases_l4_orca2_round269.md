# ORCA2 round 269 preregistration — G1 statement transfer and EEN leaf

Date: 2026-10-13. Base: `770798365e09e90e82823ad0846aac5a8dc336e3`.
This file is frozen before the G1 selectors are restored on any ORCA2/OMT
card and before any round-269 numerical measurement.

## Authorized transfer

Transfer the package statements landed on the pinned grid-ladder tip
`b0fa6a3f5cc2b957900dacd1a72fcbbd70552d9c`, without re-derivation.  The
operator note calls these five G1-landed statements; mechanically the tip has
five landing groups and six compiled statements because G1 round 5 landed a
three-statement solver unit:

1. the EEN U trend reads `zwy` on the fold-image V row
   (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:764-784`);
2. the split-explicit midpoint V depth reads the wet fold-image row and its T
   halo neighbour (`dynspg_ts.f90:418-437`);
3. stage-1 sea level is two products and a sum (`stprk3_stg.f90:142-150`);
4. barotropic substep length is the true quotient (`dynspg_ts.f90:883-887`);
5. the flux-form velocity bracket is multiplied by the mask-safe inverse exit
   depth (`dynspg_ts.f90:548-568`);
6. loop-entry face depth is reference depth times one plus the live ratio
   (`dynspg_ts.f90:323-340`).

Selectors remain required, explicit and default-off.  ORCA2 hierarchy and OMT
cards select every statement their resolved NEMO branch executes.  The
flux-form-only update is selected by OMT-0 and explicitly remains off on
vector-form OMT-1..4, rung 0 and rung 10.  No deck, initial state, carried
state, stabiliser, sea-ice selector or `unmeasured_features` entry changes.

## Frozen predictions and falsifiers

- R269-P1 — faithful transfer.  The six direct G1 plants remain non-vacuous,
  and disabling any selected statement reproduces its G1 before arm.  Semantic
  drift from the pinned implementations or an option silently executing on a
  card that does not select it refutes the transfer.
- R269-P2 — Decision-96 ORCA2 census.  On OMT-1..4, independent rung 0 and
  given-NEMO-entry rung 10, no exact certified row leaves the bar; the first
  over-bar row moves toward NEMO or is unchanged; among RMS-changing rows the
  majority moves toward NEMO; rung-0 sea-surface maximum does not worsen.
  Any failed predicate holds the transfer.
- R269-P3 — stage-2 vorticity.  The fold-image pair reproduces round 268's
  strict improvement of the matched after-VOR U maximum from
  `1.7082229745349855e-6` toward `1.2863307536456779e-8 m/s2` under both
  **independent** and **given NEMO's entry** labels.  A move away refutes the
  transfer but failure to close is not a veto under Decision 96.
- R269-P4 — shared-card isolation.  GYRE's certified year, DINO's month and
  tank/generic-card gates are byte-identical or retain their frozen passing
  values because their cards do not select the new options.  Any unexplained
  moved row is red.
- R269-P5 — EEN leaf after landing.  Candidate-side replay must reproduce the
  live completed EEN accumulator before reading intermediates.  In compiled
  order, `ff_f`, relative circulation and `r1_e1e2f` are at the floor; the
  first valued operand is the live F-point thickness
  `e3f_0vor * (1 + r3f)` on ORCA2 partial cells.  A prior operand over the
  `2e-10 m/s2` floor refutes this ordering and becomes the named OPEN row.
- R269-P6 — controls.  Missing selector coverage, changed source order,
  unmatched program points, incomplete record identity, a moved certified
  row and one-bit synthetic violations each make their gate refuse.

## Measurement order

1. Restore the pinned package implementations, select them on the cards that
   execute them, and print the resolved card-by-statement execution census.
2. Run the direct G1 controls and the ORCA2 matched after-VOR measurement.
3. Run OMT-1..4, rung-0, rung-10, GYRE, DINO, tanks and moved-row gates.  Land
   the imported statements only if the Decision-96 predicates pass.
4. On the landed tree, replay the EEN statement in compiled order.  Keep all
   operands separate and name the first valued leaf; no second transcription
   lands without its own full gate.

No NEMO acquisition is requested.  Existing admitted G1 and OMT-4 records
contain the frozen oracle operands needed for this round.
