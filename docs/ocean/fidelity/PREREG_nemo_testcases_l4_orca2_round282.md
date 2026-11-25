# ORCA2 round 282 preregistration — G1b np_CRV and vector fold unit

Frozen before importing either G1b package statement or running an ORCA2
trajectory.  Claim labels remain separate: rung 0 and OMT-1..4 are
`independent`; shipped rung 10 is scored both `given NEMO's entry` and
`independent`.  Sea ice and the card's `unmeasured_features` are unchanged.

## Source-ordered question

The certified G1b rung adds vector-form momentum advection to G1 and found two
cited statements (`nemo_testcases_gridladder_g1b_round2_receipt.md`, section
7).  This round imports, measures, and if eligible selects exactly these two:

1. **A — np_CRV F-cell area.**  ORCA2 executes the combined Coriolis plus
   relative-vorticity branch (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:648-652`).
   Its relative circulation is multiplied by `r1_e1e2f`, constructed from
   `e1f*e2f` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:150-156`).  The current
   `een_total` path instead passes the generic grid `area_q` to the curl.
2. **B — literal fold unit under vector form.**  ORCA2's midpoint V depth and
   transport are written at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:509-520`
   and `:530-536`; the external velocity branch is selected later at `:666`,
   and the end-of-substep exchange is at `:721-730`.  Therefore the fold unit
   runs on the vector-form deck.  legoESM's literal atomic unit currently
   requires `nemo_flux_form_update_active`, so the ORCA2 vector card does not
   execute it.

The imported implementations are the grid-ladder commits
`bf71392b9` (A) and `a76d3a806` (B), adapted only for conflicts with this lane.
No second implementation is permitted.  Package defaults remain `False`;
only the ORCA2 production card and its derived hierarchy/OMT cards may select
a statement after it passes the frozen predicate.

## Controlled arms

On one clean committed tree, run four configurations differing only in the two
new selectors: OFF/OFF control, A only, B only, and A+B.  Run each on rung 0,
OMT-1, OMT-2, OMT-3, OMT-4, and shipped rung 10 under the labels above.  Score
the existing 20-row checkpoint table at every available step.  Report every
bit-touched row, RMS direction, max direction, exact-row losses, the first
over-bar row and its direction, and final stage-3 SSH maximum.  The classifier
must refuse mixed commits, missing arms/cards/labels, or a planted exact-row
loss.

## Frozen predictions and falsifiers

- **R282-P1 (execution, CONFIRM expected):** instantiated ORCA2/OMT cards run
  `een_total` with vector-form momentum, carry raw `e1f/e2f`, and currently
  select neither A nor B.  REFUTE if any card lacks those operands, executes a
  different NEMO branch, or already selects either statement.
- **R282-P2 (A, CONFIRM expected):** A changes the executing np_CRV term and
  moves the old matched stage-2 interior after-VOR residual at `[19,33,3]`
  toward NEMO; it qualifies under Decision 96 on every required card/label:
  strict majority of RMS-moved rows toward, first over-bar row toward/equal,
  and zero exact-row losses.  REFUTE the residual claim if the after-VOR value
  is unchanged; REFUTE landing if any D96 predicate fails.
- **R282-P3 (B, no direction assumed):** B changes at least one certified row
  on every vector-form OMT/ORCA2 card.  REFUTE if it is bit-inert.  It lands
  only if it independently passes the same D96 predicate on every card/label;
  otherwise its full census is registered and it remains off.
- **R282-P4 (pair):** A+B differs from A by exactly B's selector and from B by
  exactly A's selector.  The pair lands together only if both singles and the
  pair pass D96; a pair cannot conceal a red single.  REFUTE if the resolved
  configs differ anywhere else or if any required census is absent.
- **R282-P5 (closed cards):** GYRE, DINO, lock exchange, overflow and VORTEX do
  not select either new field; their certified trajectories/gates remain
  unchanged.  REFUTE on any moved certified row or new failure.

## Landing and stop rules

If A, B, and A+B all pass, select both on the production ORCA2 card, re-run the
same arms against the landed card, then run the full scientific gates.  If only
A passes, land A alone and leave B off; if only B passes, land B alone and
leave A off.  If neither passes, land only the default-off imported machinery
and receipt it as measured/held; do not change production selection.  No
configuration decision, stabiliser, floor, clip, sea-ice change, or runoff
change is authorised.  Decision 126 remains pending and untouched.

Package edits require GYRE's certified ten-step and year gates, DINO, tanks,
generic card census, citation gate with a firing plant, focused tests, the
single `tests/ocean/fidelity -n 12` battery, and independent operator review.

## Correction appended 2026-10-16

The frozen line 15 citation above is retained exactly as preregistered, but it
is a dead-arm citation and is retracted.  The shipped ORCA2 deck executes EEN;
its live `SUBROUTINE vor_een` np_CRV branch is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766`.  Round 284 measured and
landed statement A against that executing branch.
