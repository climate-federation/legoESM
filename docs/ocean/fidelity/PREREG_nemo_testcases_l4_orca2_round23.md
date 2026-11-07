# NEMO testcase Lane 4 — ORCA2 card round 23 preregistration

Date: 2026-09-26

Parent: `6da9dd3fd65aea24684394c79c0c9cb08864ba62`

Status: **PREREGISTERED BEFORE ROUND-23 SCIENTIFIC SCORING.**

Round 23 executes binding Decision 58: set the ORCA2 card's already-built
second per-stage continuity solve from explicit `False` to explicit `True`,
then score the ten-step ORCA2 ladder before and after.  This is one card value,
not a new algorithm.  The GYRE card already selects the same shared statement
and must remain byte-identical.  Every ORCA2 number is labelled **independent
with Decision-52 SSH**.  The six sea-ice selectors and the card's
`unmeasured_features` tuple remain frozen.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round23/`.

## Compiled statement and controlled variable

The executing ORCA2 build takes vector-invariant momentum advection and, at
stages 2 and 3, calls `wzv` on raw Kmm velocity for momentum at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3_stg.f90:323-329`.
Tracer transport separately calls `wzv` on its corrected transport at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/traadv.f90:296-300`.
The velocity-form continuity statement is
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/divhor.f90:123-130`.

The shared implementation and its fail-closed execution predicate already
exist.  The only production change is the ORCA2 branch of
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py`:
`nemo_stage_momentum_wzv_split=False` becomes `True`.  No default, selector,
state field, sea-ice setting, or other card changes.  GYRE already sets the
same field to `True`, so its resolved configuration is unchanged.

The before arm is the clean preregistration commit with ORCA2 `False`.  The
after arm is the clean landing commit with ORCA2 `True`.  Both use the same
card, admitted NEMO record, forcing, ten steps, CPU backend, fp64 policy, and
Decision-52 sea-surface entry.  The GYRE proof runs the certified 70-row
trajectory and 30-day member at both commits.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R23-P1 | The card diff is exactly the authorized value change and executes the already-resolved two-solve program. | One resolved config leaf changes, `False` to `True`; the execution predicate changes false to true. | Any other resolved leaf moves, the program is unresolved, or the predicate does not flip. |
| R23-P2 | The first ORCA2 ladder movement is kt=1 stage-2 momentum. | Entry and stage 1 remain equal to the before arm; stage-2 u or v is the first moved row. | Any earlier row moves, or neither stage-2 momentum row moves. |
| R23-P3 | NEMO's own second solve does not move the first non-bit statement. | The first NEMO mismatch remains kt=1 stage-1 T, with the same row as before. | The first mismatch moves earlier or changes field/checkpoint. |
| R23-P4 | The controlled change is live and complete. | At least one of 200 rows moves; a one-ULP plant in an otherwise equal after document is refused. | No row moves or the plant passes. |
| R23-P5 | GYRE is byte-identical because its card already selects this shared statement. | Offline comparison has 0 differing rows, residual arrays are `np.array_equal`, and all 30 daily snapshots are byte-identical. | Any certified row, residual cell, or daily snapshot moves. |

Failed predictions remain **REFUTED** in the receipt and gate.  Directional
ORCA2 changes are reported rather than used as a landing veto: Rule 12 keeps a
cited NEMO statement even when it exposes a compensating error.  The landing
requires P1, P4, and P5; P2/P3 are scientific predictions whose failure names
the first moved or first non-bit row and does not authorize a different edit.

## Landing and stop rules

- Run and retain a fresh before and after ORCA2 kt=1..10 ladder.
- Register every moved one of the 200 rows, including whether its NEMO error
  moves toward or away by maximum absolute error.
- The first non-bit NEMO statement may not become earlier.  Any formerly
  bit-identical entry or stage-1 row that becomes non-bit is a HOLD.
- GYRE must be bit-identical at the certified 70-row ladder, residual-array,
  and 30-day snapshot levels.
- Decision 54, Decision 57, the ranked slow-forcing walk, fold-row operand
  debt, and the independent ORCA2 year remain untouched.
- No NEMO acquisition is required or permitted in this round.

## Choices

ASKED: Decision 58 changes the ORCA2 card's explicit second-continuity-solve
choice from False to True under its ten-step ladder gate.

UNASKED: none.  No configuration value besides the authorized ORCA2 card
field, carried state, stabilizer, scoring rule, or sea-ice selector changes.
