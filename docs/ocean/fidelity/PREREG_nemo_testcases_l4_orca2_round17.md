# NEMO testcase Lane 4 — ORCA2 card round 17 preregistration

Date: 2026-09-25

Parent: `9f2207d45442682c47c87c7c73e58a4f977976aa`

Status: **PREREGISTERED BEFORE ANY ROUND-17 MEASUREMENT.**

Round 16 made the complete substep-1 vector update bit-exact, then found the
first scored non-bit boundary at substep-2 `continuity_du`: 64 / 8,794 wet
T cells, maximum `7.705384632572532e-07`.  This round localizes those cells
and scores both operands of the compiled subtraction before making any
attribution.

Every number is labelled **given NEMO's entry**.  Decision 52's recorded entry
sea surface is retained.  The six sea-ice selectors and the card's
`unmeasured_features` tuple are frozen.  Decisions 54, 57, and 58 remain
pending and nothing in this round acts on them.

## Compiled statement and record boundary

The executing statement is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:550-558`:
NEMO evaluates `zhU(ji,jj) - zhU(ji-1,jj)` before adding the V difference.
The admitted writer is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:755-779`.
Its `l4_canon_2d` helper is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:1533-1563`:
it initializes the record view to zero and copies owned wet cells only.  Thus
the first owned T column's left U-face operand participates in NEMO's live
subtraction but is absent from the canonical U record.

The admitted record remains
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.
Rank 0's owned 90 columns are scored; no rank-1 or full-domain claim is made.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R17-P1 | The inherited round-16 boundary reproduces exactly. | substep-2 `continuity_du` is first non-bit, 64 / 8,794 cells, maximum `7.705384632572532e-07`; substep 1 remains bit-exact. | any changed count, maximum, order, or earlier mismatch; stop for instrument drift. |
| R17-P2 | The 64 cells are a rank-boundary population, not an interior arithmetic difference. | every unequal cell has rank-local longitude `i=0`; no `i>0` cell differs. | any unequal cell at `i>0`. |
| R17-P3 | The recorded right-face operand is exact at every unequal T cell. | candidate and oracle `zhU(ji,jj)` are array-equal on all 64 cells. | any right-face operand differs there. |
| R17-P4 | The unrecorded left-face operand alone explains the subtraction. | reconstructing NEMO's left face as `recorded_right - recorded_du` and comparing it with legoESM's actual left face gives the same 64-cell support and the same maximum; replay with the reconstructed face is bit-exact. | support/magnitude differs, or replay does not reproduce either recorded subtraction. |
| R17-P5 | The operand-localization control can fail. | one ULP planted in one active candidate left-face value changes exactly its adjacent first-column `continuity_du` value and the gate refuses. | the planted subtraction remains bit-exact or moves unrelated cells. |
| R17-P6 | No production statement lands in this localization round. | no `packages/` diff; ORCA2 remains `LADDER_MEASURED` with kt=10 entry T maximum `3.9430791763114783` on 430,552 cells. | any ungated model diff, ladder refusal, earlier first statement, or unregistered moved row. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and stop rules

- Reconstruct both candidate and NEMO subtractions from their two operands and
  require exact replay before interpreting a cell census.
- Report row and column histograms for all unequal cells.  Do not attribute a
  boundary population from a maximum alone.
- The inferred NEMO left face is evidence for operand localization only; it is
  not called a directly recorded halo.  If the stream cannot distinguish a
  model defect from a missing-halo instrument boundary, write a fail-closed
  acquisition under a new target and stop without a production fix.
- The receipt citation gate must pass, and a rigid two-line shift of a rendered
  compiled citation must fail.
- No stabilizer, configuration selection, carried-state change, sea-ice change,
  or production fix is allowed in this measurement round.

## Choices

ASKED: localize substep-2 `continuity_du` on its two U-face operands before
attribution.

UNASKED: none.
